# Debugging Continuous Bass Generation in Minimal StemGen-Style Model

## 1. Problem Description

Current issue:

```text
The generated bass spectrogram is filled across the entire time axis.
It lacks clear rhythm, note boundaries, silence, fade-in/fade-out, and onset structure.
```

In listening terms, the model is likely generating:

```text
continuous low-frequency texture
instead of a rhythmic bass stem
```

This is a common failure mode for a small-scale codec-token generation model, especially when using:

```text
small dataset
short clips
one codec codebook
fully masked generation
small Transformer
limited training steps
```

The goal is not to blindly train longer, but to identify whether the issue comes from:

```text
data
codec reconstruction
masking setup
model architecture
generation / sampling
lack of rhythmic or silence modeling
```

---

## 2. First Debugging Checks

Before modifying the model, save and compare these three audio/spectrogram pairs:

```text
1. original target bass
2. codec reconstruction:
   target bass → codec tokens → decoded bass
3. generated bass
```

Required output files:

```text
outputs/debug/original_target_bass.wav
outputs/debug/reconstructed_target_bass.wav
outputs/debug/generated_bass.wav

outputs/debug/original_target_bass_spec.png
outputs/debug/reconstructed_target_bass_spec.png
outputs/debug/generated_bass_spec.png
```

Interpretation:

```text
If reconstructed_target_bass is also filled across time:
  the problem is likely codec / resampling / normalization / decoding.

If reconstructed_target_bass is normal but generated_bass is filled:
  the problem is likely model training or generation.
```

This check must be done before adding new losses or architectural changes.

---

## 3. StemGen Paper-Relevant Causes

The original StemGen paper avoids or reduces this issue mainly through several design choices.

### 3.1 Target-only masking

StemGen masks only the target stem tokens.

```text
context tokens: always visible / unmasked
target tokens: partially masked during training
```

The model learns:

```text
predict masked target tokens
conditioned on full context tokens and target instrument condition
```

Action item:

```text
Verify that context tokens are never masked.
Only target tokens should be masked.
```

Wrong behavior to avoid:

```text
masking both context and target tokens
```

This would weaken context conditioning and can cause generic, texture-like generation.

---

### 3.2 Partial-mask reconstruction before full generation

Full generation from fully masked target tokens is much harder than partial reconstruction.

Before testing:

```text
context + fully masked target → generated target
```

first evaluate:

```text
context + partially masked real target → reconstructed target
```

Recommended mask ratios:

```text
0.15
0.30
0.50
0.75
1.00
```

Expected debugging logic:

```text
If 0.15 / 0.30 reconstruction is bad:
  model or data pipeline is probably broken.

If 0.15 / 0.30 reconstruction is good but 1.00 generation is bad:
  the model can reconstruct but cannot generate from scratch yet.
```

Required script behavior:

```text
generate.py should support --mask_ratio
evaluate.py should save reconstruction samples for multiple mask ratios
```

---

### 3.3 Multiple RVQ codebooks

StemGen uses neural audio codec tokens with multiple RVQ codebooks.

A one-codebook prototype is acceptable for speed, but it may lose:

```text
transients
attack / decay details
amplitude texture
fine rhythmic articulation
```

Action item:

```text
Keep one-codebook as the first runnable baseline.
If generation is too smeared or always-on, test 2-4 codebooks if feasible.
```

Recommended implementation priority:

```text
Priority 1: keep token shape compatible with [batch, num_codebooks, time]
Priority 2: train first codebook only
Priority 3: optionally train 2-4 codebooks
```

Do not rewrite the whole project for full hierarchical RVQ immediately.

---

### 3.4 Causal-biased iterative decoding

StemGen notes that naive masked-token sampling can cause poor transient behavior or monotonous output.

The paper uses a causal-biased iterative decoding idea:

```text
during iterative mask-predict,
earlier time positions are biased to be sampled / fixed earlier
```

This introduces a soft temporal ordering and can improve temporal coherence.

Minimal project implementation:

```text
During iterative decoding:
  score[t] = confidence[t] + causal_bias_weight * time_bias[t]
```

Example:

```python
time_bias[t] = 1.0 - t / T
```

Then select tokens to keep based on:

```text
combined_score = confidence + causal_bias
```

Recommended config:

```yaml
generation:
  use_causal_bias: true
  causal_bias_weight: 0.1
  num_iterations: 6
```

This is optional but closer to the StemGen paper than a purely confidence-based mask-predict loop.

---

### 3.5 Classifier-free guidance over conditions

StemGen uses multi-source classifier-free guidance over:

```text
audio context
target instrument category
```

This can strengthen the dependency of the generated stem on the context.

Full CFG may be too much for the first version, but the code should optionally support condition dropout during training.

Recommended lightweight preparation:

```text
During training, with small probability:
  drop audio context condition
  or drop instrument condition
```

Example config:

```yaml
training:
  context_dropout_prob: 0.1
  instrument_dropout_prob: 0.1
```

Later inference can use CFG if implemented.

This is optional and should not block the first working demo.

---

## 4. Practical Fixes for the Small Project

The following fixes are not necessarily in the original StemGen paper, but they are practical for a small-scale reproduction.

### 4.1 Add frame-level bass activity labels

The current model may not know when bass should be silent or active.

Add a simple frame-level activity label:

```text
target bass waveform
→ frame RMS energy
→ active[t] = 1 if RMS[t] > threshold else 0
```

Then add an auxiliary prediction head:

```text
activity_head(transformer_output[t]) → active probability
```

Loss:

```text
total_loss = token_CE + lambda_activity * BCE(active_pred, active_label)
```

Recommended config:

```yaml
training:
  use_activity_head: true
  lambda_activity: 0.2
  activity_rms_threshold_db: -45
```

This can directly reduce the “bass always active” problem.

---

### 4.2 Use activity mask during inference

During inference, use predicted activity probability to suppress inactive frames.

Simple behavior:

```text
if active_prob[t] < threshold:
  suppress generated bass frame
```

Possible implementations:

```text
Option A:
  replace inactive target tokens with a learned / detected silence token

Option B:
  after decoding waveform, apply frame-level gain envelope

Option C:
  use activity prediction only for evaluation first, not generation
```

For fastest implementation, use Option B:

```text
generated waveform
× smoothed activity envelope
```

Recommended config:

```yaml
generation:
  use_activity_gating: true
  activity_threshold: 0.4
  activity_smoothing_ms: 80
```

This is a practical fix, even if not exactly from the original paper.

---

### 4.3 Log active ratio for every training clip

For each training clip, compute:

```text
active_ratio = number of active bass frames / total frames
```

Log distribution:

```text
mean active_ratio
min / max active_ratio
histogram
```

If most clips have:

```text
active_ratio > 0.9
```

then the dataset teaches the model that bass is almost always active.

Recommended filtering / balancing:

```text
keep clips with diverse active_ratio
avoid training only on always-active bass clips
include sparse rhythmic bass clips
```

Example config:

```yaml
data:
  min_active_ratio: 0.05
  max_active_ratio: 0.95
  balance_active_ratio: true
```

---

### 4.4 Increase clip length if feasible

Very short clips, such as 3-5 seconds, may not capture musical phrase structure.

If compute allows, try:

```text
clip_length_sec: 8
```

or:

```text
clip_length_sec: 10-12
```

Do this only after the 4-second pipeline works.

Recommended progression:

```text
4s clips → debug pipeline
8s clips → better rhythm/context
12s clips → optional
```

---

## 5. Recommended Priority Order

Do not implement everything at once.

### Priority 1: Sanity checks

Implement first:

```text
1. Save original target spectrogram
2. Save codec reconstruction spectrogram
3. Save generated spectrogram
4. Verify context tokens are unmasked
5. Verify only target tokens are masked
```

### Priority 2: Partial-mask reconstruction

Implement:

```text
mask_ratio = 0.15
mask_ratio = 0.30
mask_ratio = 0.50
mask_ratio = 1.00
```

Compare reconstruction quality at different mask ratios.

This is the most important diagnostic.

### Priority 3: Data activity analysis

Implement:

```text
frame RMS
activity label
active_ratio per clip
active_ratio histogram
```

Check whether training clips are biased toward continuous bass.

### Priority 4: Practical silence/activity modeling

Implement:

```text
activity head
activity BCE loss
activity gating during inference
```

This is the fastest direct fix for continuous bass.

### Priority 5: More StemGen-faithful improvements

If time allows:

```text
2-4 codec codebooks
causal-biased iterative decoding
condition dropout for future CFG
longer clips
```

---

## 6. Suggested Agent Task

Use this as the direct instruction for the code agent:

```text
The generated bass spectrogram is filled across time and lacks rhythmic gaps/onsets.
Please debug and improve the generation pipeline in the following order:

1. Save and compare original target bass, codec reconstructed target bass, and generated bass,
   including both wav files and spectrogram images.

2. Verify that context tokens are never masked and only target tokens are masked.

3. Add partial-mask reconstruction evaluation with mask ratios 0.15, 0.30, 0.50, 0.75, and 1.00.
   Save audio and spectrograms for each ratio.

4. Compute frame-level RMS activity labels for target bass.
   Log active_ratio for every clip and save an active_ratio histogram.

5. Add an optional activity prediction head with BCE loss:
   total_loss = token_CE + lambda_activity * activity_BCE.

6. Add optional activity-based inference gating to suppress continuous low-frequency output.

7. Keep these fixes compatible with the StemGen-style design:
   context unmasked, target-only masking, instrument conditioning, two-stream context/target fusion.

8. If the above works, optionally test 2-4 codec codebooks and causal-biased iterative decoding.
```

---

## 7. Important Notes

The activity head and activity gating are practical small-project fixes.

They are not the main mechanism used by the original StemGen paper.

The more paper-faithful fixes are:

```text
target-only masking
partial masked-token reconstruction
multiple RVQ codebooks
causal-biased iterative decoding
classifier-free guidance
larger data and longer clips
```

For this project, the best strategy is:

```text
first make the small model produce rhythmic and non-continuous bass,
then gradually make the implementation more StemGen-faithful.
```
