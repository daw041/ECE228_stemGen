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

## Latest Experiment Summary

As of 2026-06-04, the strongest result is the 1000-track fixed-stride cached
run:

| Run | Clips | Sampling | Best Val Loss | Best Val Acc | Best Epoch |
|---|---:|---|---:|---:|---:|
| 550 cached | 26,400 | cached sampled clips | 3.4507 | 0.521 | 56 |
| 1000 stride10 cached | 22,928 | non-overlapping 10s windows | 3.0279 | 0.572 | 19 |

The 1000-track run used fixed 10-second windows, filtered inactive target
segments, cached EnCodec tokens before training, and warm-started from the
550-track best checkpoint.  It improves validation loss by about 12.3% over the
550-track run.

The main qualitative finding is that partial-mask reconstruction can produce
audible and spectrally structured bass, while full 100% mask generation is still
unstable.  This suggests that the small reproduction learns token-level
reconstruction but still needs stronger decoding and conditioning to match the
paper-level full stem generation behavior.

See `docs/LATEST_AUDIO_TOKEN_RESULTS_CN.md` for the full Chinese experiment
summary and `presentation/` for the 3-minute highlight talk materials.
