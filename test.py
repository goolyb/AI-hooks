import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")


def print_shape(activation, hook):
    print(f"{hook.name:30s} {tuple(activation.shape)}")
    return activation


hooks = [
    (name, print_shape)
    for name in model.hook_dict
    if name.endswith("hook_resid_post")
]

with torch.no_grad():
    model.run_with_hooks("Apple is a fruit", fwd_hooks=hooks)
