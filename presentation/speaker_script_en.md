# 3-Minute Highlight Presentation Script

## Slide 1

Hi everyone. Our project is a reproduction of the core idea behind StemGen: context-aware stem generation. In our setting, the model is given a music mixture with the bass removed, and the target instrument condition is bass. The goal is to generate a bass stem that fits the rest of the mix.

This is different from generic music generation. The output should not only sound like audio, but also align with the rhythm, harmony, and structure of the context. So the project sits at the intersection of machine learning and audio/music engineering. Our main question is: under limited compute, can we build a small but faithful audio-token reproduction of the StemGen idea?

## Slide 2

Our implementation is a scaled-down StemGen-style pipeline. We first encode both the context audio and the target bass stem into EnCodec RVQ tokens. During training, part of the target bass tokens are masked. The model takes the context tokens, the masked target tokens, and the instrument condition, then predicts the missing target tokens with a masked-token Transformer.

Finally, we use iterative mask-predict decoding to convert the predicted tokens back into audio. Because this is a course-scale reproduction, we use a smaller setup: 24 kHz EnCodec, 2 RVQ codebooks, and 10-second clips. We also made several engineering fixes: filtering silent or weak-bass clips, precomputing token caches before training, and separating diagnostics into codec reconstruction, partial-mask reconstruction, and full-mask generation.

## Slide 3

These are our latest quantitative results. The 550-track cached run reached a best validation loss of 3.4507 and validation accuracy of 0.521. Our newer 1000-track run uses fixed 10-second stride windows, so the clips are mostly non-overlapping. It also warm-starts from the best 550-track checkpoint.

This run produced 22,928 valid clips after filtering 2,587 inactive or weak-target clips. Its best validation loss dropped to 3.0279, and validation accuracy improved to 0.572. Compared with the 550-track run, this is about a 12.3 percent reduction in validation loss. This suggests that broader track coverage, less clip overlap, target-activity filtering, and token caching all matter for making the small reproduction stable.

On the right, I also show a full mel-spectrogram diagnostic. The target bass has clear block-like harmonic structures. The codec reconstruction keeps the major structure, and the partial-mask reconstruction also follows many of those time-frequency blocks. This is why we consider the model effective beyond just the validation loss: it is not only producing random noise. At the same time, the full 100 percent mask generation is visibly less stable, which points to the remaining bottleneck.

## Slide 4

Qualitatively, the model is no longer producing only pure noise. In partial-mask reconstruction, both listening and spectrogram inspection show some meaningful bass structure. This means the model has learned part of the conditional target-token reconstruction problem.

However, full 100 percent mask generation is still unstable and can still collapse into noisy audio. So our interpretation is that the current small model learns audio-token reconstruction better than full context-to-stem synthesis. For next steps, we want to improve full-mask decoding, test better mask schedules, add classifier-free guidance or condition dropout, and standardize audio and spectrogram diagnostics for the final report.
