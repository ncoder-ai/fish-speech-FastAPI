"""At every text-batch boundary of a real multi-batch scene, compare the
next-token distribution from the reused-prefix prefill against a full prefill
into a spare slot. Eager, batch size 2 (slot 1 is the reference slot)."""
import os, queue, torch
from fish_speech.models.text2semantic import inference as I
from fish_speech.models.text2semantic.batch_scheduler import BatchScheduler

dev = os.environ.get("DEV", "cuda:0")
model, _ = I.init_model("checkpoints/s2-pro", dev, torch.bfloat16, compile=False)
results = []

class Checked(BatchScheduler):
    def _prefill(self, slot, job, pj):
        if job.cached is not None:
            T = pj.encoded.size(1); cached = job.cached
            n = min(cached.size(1), T - 1)
            same = (cached[:, :n] == pj.encoded[:, :n].to(cached.dtype)).all(0)
            bad = (~same).nonzero(); c = int(bad[0]) if bad.numel() else n
            with torch.inference_mode():
                lr = self.model.forward_generate(pj.encoded[:, c:].reshape(1, self.cd, -1),
                        torch.arange(c, T, device=self.device), slot=slot, semantic_only=True).logits[0, -1].float()
                lf = self.model.forward_generate(pj.encoded.reshape(1, self.cd, -1),
                        torch.arange(0, T, device=self.device), slot=1, semantic_only=True).logits[0, -1].float()
            pr, pf = torch.softmax(lr / 0.7, -1), torch.softmax(lf / 0.7, -1)
            results.append((T, c, (lr - lf).abs().max().item(), 0.5 * (pr - pf).abs().sum().item(),
                            int(lr.argmax() == lf.argmax())))
        super()._prefill(slot, job, pj)

S = Checked(model, batch_size=2, compile=False)
lines = ["The morning market was already crowded when the two travelers arrived.",
         "[excited] Look at those tomatoes, they are perfect for tonight!",
         "Let us grab a basket first, or we will be juggling everything again.",
         "She laughed and reached for one of the worn wicker baskets by the gate.",
         "Do you remember when we got lost here as kids and followed the bread smell?"]
text = "\n".join(f"<|speaker:{i % 2}|>{t}" for i, t in enumerate(lines))
rq = queue.Queue()
with torch.inference_mode():
    S._admit(I.GenerateRequest(request=dict(device=dev, max_new_tokens=0, text=text, top_p=0.7, temperature=0.7,
             chunk_length=100, prompt_tokens=None, prompt_text=None, stream_chunk_tokens=0), response_queue=rq), 0)
    while S.slots[0] is not None:
        S._step()
for T, c, dl, tv, am in results:
    print(f"@@@ batch prompt {T} tokens, reused {c} ({c / T:.0%}): max|logit diff| {dl:.3f}, "
          f"total-variation {tv:.4f}, same argmax {bool(am)}")
