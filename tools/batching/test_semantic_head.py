"""Check that semantic-only logits give the same sampling distribution as
full-vocabulary logits + semantic bias. Teacher-forced on a real prompt."""
import os, torch
from fish_speech.models.text2semantic import inference as I

dev = os.environ.get("DEV", "cuda:0"); STEPS = int(os.environ.get("STEPS", "40"))
model, _ = I.init_model("checkpoints/s2-pro", dev, torch.bfloat16, compile=False)
cfg = model.config; cd = cfg.num_codebooks + 1; wd = next(model.parameters()).dtype
with torch.device(dev): model.setup_caches(1, cfg.max_seq_len, wd)
im_end = model.tokenizer.get_token_id(I.IM_END_TOKEN)
bias = torch.full((cfg.vocab_size,), float("-inf"), device=dev, dtype=wd)
bias[cfg.semantic_begin_id:cfg.semantic_end_id + 1] = 0; bias[im_end] = 0
temp = torch.tensor(0.7, device=dev, dtype=wd); top_p = torch.tensor(0.7, device=dev, dtype=wd)
p = next(I.generate_long_steps(model=model, device=dev,
         text="<|speaker:0|>The morning market was already crowded when the two travelers arrived.")).encoded.to(dev)
T = p.size(1); worst = 0.0; worst_logit = 0.0
with torch.inference_mode():
    tok = I.decode_one_token_ar(model, p.view(1, cd, -1), torch.arange(T, device=dev), temp, top_p, 30, None, None, None)
    pos = torch.tensor([T], device=dev)
    for _ in range(STEPS):
        full = model.forward_generate(tok.view(1, cd, 1), pos).logits[0, -1] + bias
        sub = model.forward_generate(tok.view(1, cd, 1), pos, semantic_only=True).logits[0, -1]
        worst_logit = max(worst_logit, (full[model.semantic_ids.long()].float() - sub.float()).abs().max().item())
        pf = I.logits_to_probs(full, temp, top_p, 30)
        ps = I.logits_to_probs(sub, temp, top_p, 30)
        pf_sub = pf[model.semantic_ids.long()]
        worst = max(worst, (pf_sub - ps).abs().max().item(), pf.sum().item() - pf_sub.sum().item())
        tok = I.decode_one_token_ar(model, tok.view(1, cd, 1), pos, temp, top_p, 30, None, None, None)
        pos += 1
print(f"@@@ {STEPS} steps: max |logit diff| {worst_logit:.4f}, max |prob diff| {worst:.2e}")
