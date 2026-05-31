# Audio-Token Reproduction Notes

The official StemGen repository contains paper examples and demo assets, not
training code or released model weights.  The reproduction here is therefore
paper-guided.

## Paper Anchors

- Task: generate a target stem conditioned on context audio and target
  instrument category.
- Token representation: neural audio codec RVQ tokens.
- Training: non-autoregressive masked token modeling.
- Masking: target-stem tokens are masked; context-mix tokens remain visible.
- Token combination: sum embeddings across RVQ levels for each audio channel,
  then combine context, target, and conditioning information.
- Sampling: iterative mask-predict with causal-biased ranking.
- Paper scale: 32 kHz EnCodec, 4 tokens per frame, 50 Hz, about 250M
  Transformer parameters.

## Small-Scale E5 Target

This repo now targets a smaller but more internally aligned baseline:

- 24 kHz EnCodec wrapper
- 2 RVQ codebooks
- 10 second clips
- multi-codebook masked CE loss
- variable mask ratio training
- 32/16 iterative decoding steps for codebook 0/1 diagnostics

The key success gate is not full generation first.  The first gate is:

```text
target -> codec reconstruction
context + partially masked target tokens -> reconstructed target audio
```

Only after low-mask partial reconstruction works should full-mask generation be
treated as the main bottleneck.
