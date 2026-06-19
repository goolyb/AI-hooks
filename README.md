# AI Hooks — Logit Lens on Qwen2.5

Small experiment in mechanistic interpretability: peeking inside a language model
layer by layer using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## What it does

`test.py` boots `Qwen/Qwen2.5-0.5B-Instruct` via `TransformerBridge`, runs a prompt
with `run_with_cache`, then applies the **logit lens**: for every one of the 24 layers
it takes the residual stream after the layer, projects it through the final layer norm
and unembedding matrix, and prints the model's top predicted token at that depth.

This shows how the model's "draft thought" evolves from layer to layer.

## Run

```bash
pip install transformer_lens torch transformers
python test.py
```

## Files

- `test.py` — logit lens over all layers
- `visualize.py` — visualization helpers
- `index.html` — output view
