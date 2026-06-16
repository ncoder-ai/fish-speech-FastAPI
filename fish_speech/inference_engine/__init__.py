import gc
import queue
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile

    @torch.inference_mode()
    def inference(
        self, req: ServeTTSRequest, cancel_event=None
    ) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        # Load the reference audio and text based on id or hash
        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )

        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(
            req, prompt_tokens, prompt_texts, cancel_event
        )

        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # If streaming, send the header
        if req.streaming:
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        segments = []

        # Incremental-streaming decode state (action="stream"/"stream_end").
        # We re-decode the current batch's accumulated codes on each chunk and
        # emit only the newly-decoded audio, holding back a small right MARGIN of
        # frames so each emitted boundary was decoded with right-context (no
        # clicks). The margin is flushed at "stream_end". Generation is unchanged;
        # this only governs WHEN audio is decoded/emitted.
        import os

        margin_frames = int(os.environ.get("FISH_STREAM_MARGIN_FRAMES", "8"))
        batch_codes = None
        emitted = 0  # samples already emitted for the current batch

        def _emit_segment(seg: np.ndarray):
            segments.append(seg)
            if req.streaming:
                return InferenceResult(
                    code="segment", audio=(sample_rate, seg), error=None
                )
            return None

        while True:
            # Get the response from the LLAMA model
            wrapped_result: WrappedGenerateResponse = response_queue.get()
            if wrapped_result.status == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=(
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else Exception("Unknown error")
                    ),
                )
                break

            # Check the response type
            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response

            if result.action == "stream":
                # Append new codes, re-decode the batch prefix, emit the delta
                # minus a right margin (held for clean boundaries).
                batch_codes = (
                    result.codes
                    if batch_codes is None
                    else torch.cat([batch_codes, result.codes], dim=1)
                )
                audio = self._decode_codes(batch_codes)
                spf = max(1, len(audio) // max(1, batch_codes.size(1)))
                keep = len(audio) - margin_frames * spf
                if keep > emitted:
                    r = _emit_segment(audio[emitted:keep])
                    emitted = keep
                    if r is not None:
                        yield r

            elif result.action == "stream_end":
                # Flush the held margin (final decode has the true batch end).
                if batch_codes is not None:
                    audio = self._decode_codes(batch_codes)
                    if len(audio) > emitted:
                        r = _emit_segment(audio[emitted:])
                        if r is not None:
                            yield r
                batch_codes = None
                emitted = 0

            elif result.action == "next":
                break

            else:  # "sample" — per-batch decode (non-incremental path)
                segment = self.get_audio_segment(result)
                if req.streaming:
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )
                segments.append(segment)

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self,
        req: ServeTTSRequest,
        prompt_tokens: list,
        prompt_texts: list,
        cancel_event=None,
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            cancel_event=cancel_event,
            stream_chunk_tokens=req.stream_chunk_tokens,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def _decode_codes(self, codes) -> np.ndarray:
        """Decode a VQ-code tensor to a float32 numpy waveform."""
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            segment = self.decode_vq_tokens(codes=codes)
        return segment.float().cpu().numpy()

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """
        return self._decode_codes(result.codes)
