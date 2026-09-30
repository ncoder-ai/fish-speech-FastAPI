"""Continuous-batching worker for the DualAR text-to-semantic model.

The single-request worker in inference.py runs one request to completion before
starting the next. Decode is memory-bandwidth-bound (every step streams all the
weights), so running B requests in one step costs little more than running one.

This worker keeps B KV-cache slots. Each in-flight request owns one slot:

1. A new request's prompt is prefilled into a free slot (batch 1, slot-scoped
   KV writes, so the other rows are untouched).
2. Every step decodes one frame for all B rows at once, each row at its own
   position with its own sampling parameters and RAS window.
3. When a row ends a text batch, the request's `generate_long_steps` generator
   receives the tokens, emits its responses, and hands back the next prompt,
   which is prefilled into the same slot.

Requests join and leave between steps. Rows without a request compute garbage
into their own slot and are ignored; this keeps shapes static for torch.compile.
"""

import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.inference import (
    RAS_WIN_SIZE,
    GenerateRequest,
    GenerateResponse,
    PromptJob,
    WrappedGenerateResponse,
    decode_one_token_ar,
    decode_one_token_batched,
    generate_long_steps,
)
from fish_speech.tokenizer import IM_END_TOKEN

# top_k is not exposed per request; generate_long_steps defaults it to 30.
# The batched step bakes it in as a compile-time constant.
TOP_K = 30
# Attention length granularity (cache slots). Rows attend to the first
# round_up(longest active position + 1, KV_LEN_ALIGN) slots.
KV_LEN_ALIGN = 128


@dataclass
class _Job:
    steps: object  # generate_long_steps generator
    response_queue: queue.Queue
    cancel_event: object
    stream_chunk_tokens: int
    # Per text-batch state (reset on every prefill).
    prompt: Optional[torch.Tensor] = None
    tokens: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    budget: int = 0  # decode steps left for this text batch
    started: float = 0.0


class BatchScheduler:
    def __init__(self, model, batch_size: int, compile: bool):
        self.model = model
        self.B = batch_size
        cfg = model.config
        self.cd = cfg.num_codebooks + 1
        self.max_seq_len = cfg.max_seq_len
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)

        with torch.device(self.device):
            model.setup_caches(
                max_batch_size=batch_size,
                max_seq_len=cfg.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

        dev, B = self.device, batch_size
        wdtype = next(model.parameters()).dtype
        # Create step state as inference tensors, like every later update, so
        # the compiled step never recompiles over a dispatch-key mismatch.
        with torch.inference_mode():
            self.cur = torch.zeros(B, self.cd, 1, dtype=torch.int, device=dev)
            self.pos = torch.zeros(B, 1, dtype=torch.int, device=dev)
            self.temperature = torch.full((B, 1), 0.7, dtype=wdtype, device=dev)
            self.top_p = torch.full((B, 1), 0.7, dtype=wdtype, device=dev)
            self.prev = torch.zeros(
                B, self.cd, RAS_WIN_SIZE, dtype=torch.int, device=dev
            )

        self.bias = torch.full(
            (1, 1, cfg.vocab_size), float("-inf"), device=dev, dtype=wdtype
        )
        self.bias[0, 0, cfg.semantic_begin_id : cfg.semantic_end_id + 1] = 0.0
        self.bias[0, 0, self.im_end_id] = 0.0

        if os.environ.get("FISH_QUANTIZE", "").strip().lower() in ("int8", "int4"):
            # torchao weight-only kernels are fused GEMVs only at batch 1. At
            # batch >= 2 they dequantize every weight to bf16 each step, which
            # erases the batching gain (measured: B=2 int8 ~49 ms/step vs
            # B=4 bf16 ~28 ms/step on an RTX 3090).
            logger.warning(
                "Continuous batching with FISH_QUANTIZE="
                f"{os.environ['FISH_QUANTIZE']} is slow; use bf16 weights "
                "(FISH_QUANTIZE=none) with FISH_BATCH_SIZE > 1."
            )

        self.step_fn = decode_one_token_batched
        if compile:
            logger.info(f"Compiling batched decode step (batch size {B})...")
            self.step_fn = torch.compile(
                decode_one_token_batched,
                backend="inductor",
                mode="default",
                fullgraph=True,
            )

        self.slots: list[Optional[_Job]] = [None] * B
        # Host copy of each row's position, so the attention length can be
        # picked every step without a device sync.
        self.pos_host = [0] * B
        # FISH_BATCH_PROFILE=1 logs where step time goes every 100 steps.
        self._profile = os.environ.get("FISH_BATCH_PROFILE", "0") == "1"
        self._prof = [0.0, 0.0, 0.0, 0]  # launch, sync, bookkeeping, steps

    # ------------------------------------------------------------------ jobs

    def _put(self, job: _Job, response: GenerateResponse):
        job.response_queue.put(WrappedGenerateResponse(status="success", response=response))

    def _fail(self, slot: int, err: Exception):
        job = self.slots[slot]
        self.slots[slot] = None
        if job is not None:
            job.response_queue.put(WrappedGenerateResponse(status="error", response=err))

    def _advance(self, slot: int, y: Optional[torch.Tensor]):
        """Feed `y` to the job's generator until it asks for the next prompt
        (prefilled into `slot`) or finishes (slot freed)."""
        job = self.slots[slot]
        try:
            while True:
                try:
                    item = job.steps.send(y)
                except StopIteration:
                    self.slots[slot] = None
                    return
                y = None
                if isinstance(item, PromptJob):
                    self._prefill(slot, job, item)
                    return
                self._put(job, item)
        except Exception as e:
            logger.error(traceback.format_exc())
            self._fail(slot, e)

    @torch.inference_mode()
    def _prefill(self, slot: int, job: _Job, pj: PromptJob):
        T = pj.encoded.size(1)
        if T >= self.max_seq_len:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
        max_new = pj.max_new_tokens or (self.max_seq_len - T)
        max_new = min(max_new, self.max_seq_len - T)

        temp = torch.tensor(pj.temperature, device=self.device, dtype=self.dtype)
        top_p = torch.tensor(pj.top_p, device=self.device, dtype=self.dtype)
        first = decode_one_token_ar(
            self.model,
            pj.encoded.view(1, self.cd, -1),
            torch.arange(0, T, device=self.device, dtype=torch.long),
            temp,
            top_p,
            pj.top_k,
            self.bias,
            pj.audio_masks,
            pj.audio_parts,
            slot=slot,
        )  # [cd, 1]

        self.cur[slot] = first
        self.pos[slot, 0] = T
        self.pos_host[slot] = T
        self.temperature[slot, 0] = pj.temperature
        self.top_p[slot, 0] = pj.top_p
        self.prev[slot].zero_()

        job.prompt = pj.encoded
        job.tokens = [first[:, 0]]
        job.pending = [first[1:]] if job.stream_chunk_tokens > 0 else []
        job.budget = max_new - 1
        job.started = time.perf_counter()

    def _finish_batch(self, slot: int):
        job = self.slots[slot]
        if job.pending:
            self._emit(job)
        y = torch.cat(
            [job.prompt, torch.stack(job.tokens, dim=1).to(job.prompt.dtype)], dim=1
        )
        n = len(job.tokens)
        dt = time.perf_counter() - job.started
        logger.info(
            f"[batch] slot {slot}: {n} tokens in {dt:.2f}s ({n / max(dt, 1e-6):.1f} tok/s)"
        )
        job.prompt, job.tokens, job.pending = None, [], []
        self._advance(slot, y)

    def _emit(self, job: _Job):
        self._put(
            job,
            # .cpu() is nearly free here (the step just synced) and lets the
            # codec, which runs on its own CUDA stream, read the codes safely.
            GenerateResponse(action="stream", codes=torch.cat(job.pending, dim=1).cpu()),
        )
        job.pending = []

    def _admit(self, item: GenerateRequest, slot: int):
        kwargs = dict(item.request)
        sct = int(kwargs.pop("stream_chunk_tokens", 0) or 0)
        kwargs.pop("stream_emit", None)
        job = _Job(
            steps=generate_long_steps(model=self.model, incremental=sct > 0, **kwargs),
            response_queue=item.response_queue,
            cancel_event=kwargs.get("cancel_event"),
            stream_chunk_tokens=sct,
        )
        self.slots[slot] = job
        self._advance(slot, None)

    # ------------------------------------------------------------------ loop

    def _kv_len_hint(self) -> torch.Tensor:
        """Attention length for this step: longest active row, rounded up."""
        need = max(self.pos_host) + 1  # idle rows sit at position 0
        kv_len = min(self.max_seq_len, -(-need // KV_LEN_ALIGN) * KV_LEN_ALIGN)
        hint = torch.empty(kv_len, dtype=torch.uint8, device=self.device)
        torch._dynamo.maybe_mark_dynamic(hint, 0)
        return hint

    @torch.inference_mode()
    def _step(self):
        t0 = time.perf_counter()
        hint = self._kv_len_hint()
        with sdpa_kernel(SDPBackend.MATH):
            nxt = self.step_fn(
                model=self.model,
                x=self.cur,
                input_pos=self.pos,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=TOP_K,
                semantic_logit_bias=self.bias,
                previous_tokens=self.prev,
                kv_len_hint=hint,
            ).clone()  # [B, cd]

        self.cur = nxt.view(self.B, self.cd, 1)
        self.pos.add_(1)
        self.pos_host = [p + 1 for p in self.pos_host]
        self.prev = self.prev.roll(-1, dims=2)
        self.prev[:, :, -1] = nxt
        t1 = time.perf_counter()
        ends = (nxt[:, 0] == self.im_end_id).tolist()  # one host sync per step
        t2 = time.perf_counter()

        for slot, job in enumerate(self.slots):
            if job is None:
                continue
            job.tokens.append(nxt[slot])
            job.budget -= 1
            is_end = ends[slot]
            if job.stream_chunk_tokens > 0 and not is_end:
                job.pending.append(nxt[slot, 1:, None])
                if len(job.pending) >= job.stream_chunk_tokens:
                    self._emit(job)
            cancelled = job.cancel_event is not None and job.cancel_event.is_set()
            if is_end or job.budget <= 0 or cancelled:
                self._finish_batch(slot)

        # Idle rows: keep their positions in range (they write only their own row).
        for slot, job in enumerate(self.slots):
            if job is None:
                self.pos[slot, 0] = 0
                self.pos_host[slot] = 0

        if self._profile:
            p = self._prof
            p[0] += t1 - t0
            p[1] += t2 - t1
            p[2] += time.perf_counter() - t2
            p[3] += 1
            if p[3] == 100:
                active = sum(j is not None for j in self.slots)
                logger.info(
                    f"[batch] {active} active, per step: launch {p[0] * 10:.1f} ms, "
                    f"gpu wait {p[1] * 10:.1f} ms, bookkeeping {p[2] * 10:.1f} ms"
                )
                self._prof = [0.0, 0.0, 0.0, 0]

    def run(self, input_queue: queue.Queue):
        # Request threads (codec decode, audio encoding) hold the GIL while they
        # launch work; the default 5 ms switch interval then leaves the GPU idle
        # between decode steps. Hand the GIL over more often.
        sys.setswitchinterval(float(os.environ.get("FISH_GIL_SWITCH_S", "0.0005")))
        while True:
            # Admit new requests into free slots. Block only when idle.
            while None in self.slots:
                idle = all(j is None for j in self.slots)
                try:
                    item = input_queue.get(block=idle)
                except queue.Empty:
                    break
                if item is None:
                    return
                self._admit(item, self.slots.index(None))

            if all(j is None for j in self.slots):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            try:
                self._step()
            except Exception as e:
                logger.error(traceback.format_exc())
                for slot in range(self.B):
                    self._fail(slot, e)
