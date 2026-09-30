import threading
from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


# One codec instance is shared by every request thread. With continuous batching
# several requests decode audio at once, so serialize codec calls.
_CODEC_LOCK = threading.Lock()
_CODEC_STREAMS: dict = {}


def _codec_stream(device) -> torch.cuda.Stream:
    if device not in _CODEC_STREAMS:
        _CODEC_STREAMS[device] = torch.cuda.Stream(device=device)
    return _CODEC_STREAMS[device]


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        logger.info(f"VQ features: {codes.shape}")

        if isinstance(self.decoder_model, DAC):
            with _CODEC_LOCK:
                device = self.decoder_model.device
                if device.type != "cuda":
                    return self.decoder_model.from_indices(codes[None])[0].squeeze()
                # Run on a side stream so codec kernels overlap the LLM decode
                # steps instead of queueing behind them on the default stream.
                stream = _codec_stream(device)
                if codes.is_cuda:
                    # Codes produced on the caller's stream must be ready first.
                    stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    audio = self.decoder_model.from_indices(
                        codes.to(device, non_blocking=True)[None]
                    )[0].squeeze()
                    # Hand back a host tensor: nothing else touches side-stream memory.
                    return audio.float().cpu()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios = torch.from_numpy(reference_audio_content).to(
                self.decoder_model.device
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            # VQ Encoder
            if isinstance(self.decoder_model, DAC):
                with _CODEC_LOCK:
                    prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
