# StemGen Reproduction: 3-Minute Highlight Slides

## Slide 1. Problem and Goal

**Context-aware bass stem generation**

- Paper target: generate a target stem while listening to the rest of the mix
- Our task: input = mixture without bass, output = bass stem
- Why it matters: the generated bass must match rhythm, harmony, and structure
- Course fit: machine learning for audio / music engineering

**Research question:** Can a small audio-token masked model reproduce the core StemGen idea under limited compute?

---

## Slide 2. Scaled-Down StemGen Pipeline

```text
context audio + masked bass tokens
-> EnCodec RVQ tokens
-> masked-token Transformer
-> iterative mask-predict decoding
-> generated bass audio
```

Implementation choices:

- 24 kHz EnCodec, 2 RVQ codebooks
- 10-second clips
- target instrument: bass
- context: mixture minus bass
- variable target mask ratio: 50% to 100%
- multi-codebook cross-entropy loss

Main engineering fixes:

- filter silent / weak-bass clips
- precompute token cache before training
- separate codec, partial-mask, and full-mask diagnostics

---

## Slide 3. Latest Results

| Run | Data | Sampling | Best Val Loss | Best Val Acc |
|---|---:|---|---:|---:|
| 550 cached | 26,400 clips | sampled cache | 3.4507 | 0.521 |
| 1000 stride10 | 22,928 clips | non-overlap windows | 3.0279 | 0.572 |

1000-track run:

- fixed 10-second stride windows
- filtered 2,587 inactive clips
- warm-started from 550 best checkpoint
- best checkpoint: epoch 19
- validation loss improved by about 12.3%

---

## Slide 4. Findings and Next Steps

What worked:

- output is no longer pure noise
- partial-mask reconstruction shows audible and spectral structure
- larger track coverage + less overlap improved validation metrics

Main limitation:

- full 100% mask generation is still unstable and can become noisy
- the model learns token reconstruction better than full context-to-stem synthesis

Next steps:

- improve full-mask decoding and mask schedule
- add classifier-free guidance / condition dropout
- standardize audio and spectrogram diagnostics
