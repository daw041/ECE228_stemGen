# StemGen Reproduction: Context-Aware Bass Stem Generation

3-minute highlight presentation

---

## Slide 1. Motivation

**Goal:** reproduce the core idea of StemGen: generate a target music stem while listening to the rest of the mix.

- Input: mixture without bass
- Condition: target instrument = bass
- Output: generated bass stem
- Course scope: ML + audio/music engineering application

**Key challenge:** generation must be musically aligned with the context, not just plausible audio.

---

## Slide 2. Method

We implement a scaled-down audio-token StemGen pipeline.

```text
context audio + masked bass tokens
-> EnCodec RVQ tokens
-> masked-token Transformer
-> iterative mask-predict decoding
-> generated bass audio
```

Settings:

- 24 kHz EnCodec
- 2 RVQ codebooks
- 10 second clips
- multi-codebook cross entropy
- variable mask ratio: 50% to 100%

---

## Slide 3. What We Changed During Reproduction

Early audio-token outputs were mostly noise.

Main fixes:

- aligned EnCodec codebook handling
- filtered silent or weak bass segments
- cached codec tokens before training
- separated diagnostics into codec reconstruction, partial-mask reconstruction, and full generation
- moved from highly overlapping clips to 10-second stride windows

This turned the pipeline from runnable into measurable.

---

## Slide 4. Results

| Run | Data | Sampling | Best Val Loss | Best Val Acc |
|---|---:|---|---:|---:|
| 550 cached | 26,400 clips | cached sampled clips | 3.4507 | 0.521 |
| 1000 stride10 | 22,928 clips | non-overlap windows | 3.0279 | 0.572 |

The 1000-track run improves validation loss by about **12.3%** over the 550-track run.

Best 1000-run checkpoint:

- epoch 19
- codebook accuracies: 0.673 / 0.471

---

## Slide 5. Qualitative Findings

The model is no longer producing only noise.

Observed behavior:

- codec reconstruction works as a sanity check
- partial-mask reconstruction shows clear learned structure
- full 100% mask generation is still unstable and can become noisy

Interpretation:

The model learns token-level reconstruction, but full context-to-stem generation remains the hard part.

---

## Slide 6. Takeaway and Next Steps

**Takeaway:** We reproduced a small but faithful audio-token version of StemGen and identified the main bottleneck.

What worked:

- audio-token masked modeling
- token caching
- silence filtering
- larger track coverage with less overlap

Next:

- improve full-mask decoding
- add classifier-free guidance / condition dropout
- compare mask schedules
- run standardized audio and spectrogram diagnostics

---

