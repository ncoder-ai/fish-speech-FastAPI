"""Teacher-forced comparison: original B=1 path vs batched slots, on real prompts.

Reports max logit difference of the slow model per step and greedy token agreement.
"""
import os, torch
from fish_speech.models.text2semantic import inference as I

dev = os.environ.get("DEV", "cuda:0"); STEPS = int(os.environ.get("STEPS", "60"))
model, _ = I.init_model("checkpoints/s2-pro", dev, torch.bfloat16, compile=False)
cfg = model.config; cd = cfg.num_codebooks + 1; wd = next(model.parameters()).dtype
texts = ["<|speaker:0|>The morning market was already crowded when the two travelers arrived.",
         "<|speaker:0|>[excited] Look at those tomatoes, they are perfect for tonight, don't you think so?"]
NP = int(os.environ.get('NP', '2'))
prompts = [next(I.generate_long_steps(model=model, device=dev, text=t)).encoded.to(dev) for t in texts][:NP]
bias = torch.full((1, 1, cfg.vocab_size), float("-inf"), device=dev, dtype=wd)
bias[0, 0, cfg.semantic_begin_id:cfg.semantic_end_id + 1] = 0
bias[0, 0, model.tokenizer.get_token_id(I.IM_END_TOKEN)] = 0
temp = torch.tensor(0.7, device=dev, dtype=wd); top_p = torch.tensor(0.7, device=dev, dtype=wd)

@torch.inference_mode()
def single(p):
    """Sampled rollout with the original path; returns tokens and per-step slow logits."""
    with torch.device(dev): model.setup_caches(1, cfg.max_seq_len, wd)
    T = p.size(1); torch.manual_seed(1)
    tok = I.decode_one_token_ar(model, p.view(1, cd, -1), torch.arange(T, device=dev), temp, top_p, 30, bias, None, None)
    toks, logits = [tok], []
    pos = torch.tensor([T], device=dev)
    for _ in range(STEPS):
        logits.append(model.forward_generate(tok.view(1, cd, 1), pos).logits[0, -1].float())
        # advance with the full decode (re-writes the same KV position, same values)
        tok = I.decode_one_token_ar(model, tok.view(1, cd, 1), pos, temp, top_p, 30, bias, None, None)
        toks.append(tok); pos += 1
    return toks, logits

refs = [single(p) for p in prompts]
for l in model.layers: l.attention.kv_cache = None
model.max_batch_size = -1; model.max_seq_len = -1
from fish_speech.models.text2semantic.batch_scheduler import BatchScheduler
NB = int(os.environ.get("NB", "3")); S = BatchScheduler(model, batch_size=NB, compile=False)
with torch.inference_mode():
    for i, p in enumerate(prompts):
        I.decode_one_token_ar(model, p.view(1, cd, -1), torch.arange(p.size(1), device=dev), temp, top_p, 30, bias, None, None, slot=i)
        S.pos[i, 0] = p.size(1)
    worst = [0.0]*NP; agree = [0]*NP
    for s in range(STEPS):
        for i in range(NP): S.cur[i] = refs[i][0][s]   # teacher forcing
        lg = model.forward_generate(S.cur, S.pos).logits[:, -1].float()
        for i in range(NP):
            r = refs[i][1][s]
            m = torch.isfinite(r)
            worst[i] = max(worst[i], (lg[i][m] - r[m]).abs().max().item())
            agree[i] += int(lg[i].argmax() == r.argmax())
        S.pos.add_(1)
    for i in range(NP):
        print(f"@@@ prompt {i} (T={prompts[i].size(1)}): max |logit diff| {worst[i]:.4f}  "
              f"argmax agreement {agree[i]}/{STEPS}")
