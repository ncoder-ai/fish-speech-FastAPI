"""Time the compiled batched decode step (no HTTP).

FISH_QUANTIZE=none BATCHES=1,4,6 PROMPT=600 [SWEEP=1] [PROFILE=1] DEV=cuda:0
"""
import os, time, torch
from fish_speech.models.text2semantic import inference as I
from fish_speech.models.text2semantic.batch_scheduler import BatchScheduler

dev = os.environ.get("DEV", "cuda:0"); T = int(os.environ.get("PROMPT", "600"))
steps = int(os.environ.get("STEPS", "200"))
model, _ = I.init_model("checkpoints/s2-pro", dev, torch.bfloat16, compile=False)
cd = model.config.num_codebooks + 1
torch._dynamo.config.cache_size_limit = 64
for B in [int(b) for b in os.environ.get("BATCHES", "1,2,4").split(",")]:
    for l in model.layers: l.attention.kv_cache = None
    model.max_batch_size = -1; model.max_seq_len = -1
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    S = BatchScheduler(model, batch_size=B, compile=True)
    S.pos.fill_(T); S.pos_host = [T] * B
    S._step_orig = S._step
    def _keep():
        S._step_orig(); S.pos.fill_(T); S.pos_host = [T] * B
    S._step = _keep
    S.cur[:, 0] = model.config.semantic_begin_id + 5
    def run(n):
        for _ in range(n): S._step()
        torch.cuda.synchronize()
    t0 = time.time(); run(3); c = time.time() - t0
    run(10); t0 = time.time(); run(steps); dt = (time.time() - t0) / steps
    print(f"@@@ B={B}: {dt*1000:.1f} ms/step  per-stream {1/dt:.1f} tok/s  "
          f"aggregate {B/dt:.1f} tok/s  ({B/dt/21.5:.1f}x realtime total)  "
          f"compile {c:.0f}s  peak {torch.cuda.max_memory_allocated(dev)/1e9:.1f}GB", flush=True)
    S.pos.fill_(T)
if os.environ.get("PROFILE"):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        run(10)
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=60))
if os.environ.get("SWEEP"):
    import torch._dynamo.utils as U
    for T2 in (300, 1000, 2000, 3500, 5000, 7000):
        S.pos.fill_(T2); S.pos_host = [T2] * B
        def _keep2():
            S._step_orig(); S.pos.fill_(T2); S.pos_host = [T2] * B
        S._step = _keep2
        run(5); t0 = time.time(); run(50); dt = (time.time() - t0) / 50
        print(f"@@@ sweep B={B} ctx={T2}: {dt*1000:.1f} ms/step  aggregate {B/dt:.0f} tok/s  "
              f"unique_graphs={U.counters['stats']['unique_graphs']}", flush=True)
