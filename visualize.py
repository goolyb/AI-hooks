import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer

PROMPT = "Apple is a fruit"

model = HookedTransformer.from_pretrained("gpt2")
tokens = model.to_tokens(PROMPT)
str_tokens = model.to_str_tokens(PROMPT)

# --- capture every layer's residual-stream output via hooks ---
captured = {}


def grab(activation, hook):
    captured[hook.name] = activation.detach()
    return activation


hooks = [(n, grab) for n in model.hook_dict if n.endswith("hook_resid_post")]
with torch.no_grad():
    model.run_with_hooks(tokens, fwd_hooks=hooks)

# order layers 0..n and stack -> [layers, seq, d_model]
names = sorted(captured, key=lambda n: int(n.split(".")[1]))
resid = torch.stack([captured[n][0] for n in names])  # drop batch dim
n_layers = resid.shape[0]

# --- panel 1: per-token residual L2 norm across layers ---
norms = resid.norm(dim=-1).cpu().numpy()  # [layers, seq]

# --- panel 2: logit lens at the LAST token position ---
last = resid[:, -1, :]                       # [layers, d_model]
last = model.ln_final(last)                  # final layer norm
logits = last @ model.W_U                    # [layers, vocab]
probs = logits.softmax(dim=-1)
top_p, top_i = probs.max(dim=-1)
top_toks = [model.to_string(i) for i in top_i]

# --- plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im = ax1.imshow(norms, aspect="auto", cmap="viridis")
ax1.set_title(f'Residual-stream L2 norm  ("{PROMPT}")')
ax1.set_xlabel("token")
ax1.set_ylabel("layer")
ax1.set_xticks(range(len(str_tokens)))
ax1.set_xticklabels([repr(t) for t in str_tokens], rotation=45, ha="right")
ax1.set_yticks(range(n_layers))
fig.colorbar(im, ax=ax1, label="L2 norm")

y = np.arange(n_layers)
ax2.barh(y, top_p.cpu().numpy(), color="tab:orange")
ax2.set_title("Logit lens @ last token (top-1 prediction per layer)")
ax2.set_xlabel("probability")
ax2.set_ylabel("layer")
ax2.set_yticks(y)
ax2.invert_yaxis()
for i, (p, t) in enumerate(zip(top_p.tolist(), top_toks)):
    ax2.text(p + 0.01, i, repr(t), va="center", fontsize=9)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("activations.png", dpi=130)
print("saved activations.png")
plt.show()
