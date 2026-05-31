# MIDI-Transformer Branch for Context-Aware Bass Generation

## 1. 项目定位

本文件描述一个 **MIDI-symbolic bass generation branch**，用于补充当前 StemGen-style audio-token branch。

核心目标：

```text
context audio
→ predict bass MIDI structure
→ render bass MIDI into bass audio
→ mix with context
```

该分支不是替代原来的 audio-token branch，而是作为一个更适合 **limited compute / small model / small dataset** 的 practical baseline。

最终项目可以形成对比：

```text
Audio-token branch:
context audio → EnCodec tokens → masked-token Transformer → generated bass audio

MIDI branch:
context audio → audio features → Transformer → bass MIDI → rendered bass audio
```

核心结论方向：

```text
Audio-token generation is more general but harder under limited compute.
MIDI-symbolic generation is more stable for bass because it separates musical structure from timbre rendering.
```

---

## 2. 总体 Pipeline

推荐 pipeline：

```text
Slakh / BabySlakh
→ read mixture, bass stem, bass MIDI
→ construct context = mixture - bass
→ extract context audio features
→ convert bass MIDI to frame-level labels
→ train Transformer to predict bass activity and pitch
→ postprocess predictions into bass MIDI
→ render MIDI with bass soundfont
→ mix rendered bass with context
→ compare against target bass and audio-token branch
```

简化表示：

```text
context audio
→ mel / chroma / onset features
→ Transformer encoder
→ active head + pitch head
→ bass piano-roll / MIDI events
→ bass audio rendering
→ mixed audio
```

---

## 3. Input and Target

### 3.1 Input: Context Audio

第一阶段使用：

```text
context = mixture - bass
```

未来可以扩展到：

```text
context = selected subset of stems
```

每个 sample 是一个短 clip：

```text
clip length: 4-8 seconds
sample rate: 16kHz / 24kHz / 32kHz
```

推荐先用：

```text
clip length: 4 seconds
```

pipeline 稳定后再测试：

```text
clip length: 8 seconds
```

---

### 3.2 Input Features

不要直接把 raw waveform 输入 Transformer。先提取帧级音乐特征。

推荐特征：

```text
mel-spectrogram
chroma
onset envelope
optional: beat position
```

最小版本：

```text
mel + chroma + onset
```

特征形状：

```text
X: [T, D]
```

其中：

```text
T = number of time frames
D = feature dimension
```

示例：

```text
mel bins: 128
chroma bins: 12
onset envelope: 1

D = 141
```

batch 后：

```text
X: [B, T, D]
```

所有特征必须和 MIDI label 使用相同的 frame grid。

---

### 3.3 Target: Bass MIDI Labels

从 bass MIDI 构造 frame-level labels。

推荐 bass pitch range：

```text
MIDI pitch 28-60
approximately E1-C4
```

定义：

```text
P = number of bass pitches = 60 - 28 + 1 = 33
```

不要一开始预测完整 128 MIDI pitches。限制 bass range 可以显著降低难度。

---

## 4. Label Representation

推荐不要直接只用 piano-roll BCE，而是拆成两个 head：

```text
1. activity prediction:
   whether bass is active at this frame

2. pitch prediction:
   which bass pitch is active when bass is active
```

### 4.1 Activity Label

```text
active_label[t] = 1 if any bass note is active at frame t
active_label[t] = 0 otherwise
```

形状：

```text
active_label: [T]
```

batch 后：

```text
active_label: [B, T]
```

---

### 4.2 Pitch Label

如果当前 frame 有 bass note：

```text
pitch_label[t] = pitch_id
```

其中：

```text
pitch_id = midi_pitch - min_pitch
```

如果当前 frame 没有 bass note：

```text
pitch_label[t] = ignore_index
```

形状：

```text
pitch_label: [T]
```

batch 后：

```text
pitch_label: [B, T]
```

推荐：

```text
ignore_index = -100
```

这样 CE loss 可以忽略 silent frames。

---

### 4.3 Handling Polyphony

bass 通常接近 monophonic。

如果同一 frame 有多个 bass notes，可以简单处理：

```text
choose the lowest pitch
```

或者：

```text
choose the note with highest velocity
```

推荐第一版：

```text
choose lowest active pitch
```

原因：实现简单，而且 bass line 通常以最低音为主。

---

## 5. Model Architecture

为了和原 StemGen-style branch 保持一致，本分支使用 Transformer encoder。

### 5.1 Model Overview

```text
Input features [B, T, D]
→ Linear projection
→ positional encoding
→ Transformer encoder
→ active head
→ pitch head
```

输出：

```text
active_logits: [B, T]
pitch_logits:  [B, T, P]
```

---

### 5.2 Detailed Architecture

```text
Feature projection:
  Linear(D, d_model)

Positional encoding:
  sinusoidal or learnable position embedding

Backbone:
  TransformerEncoder

Heads:
  active_head: Linear(d_model, 1)
  pitch_head:  Linear(d_model, P)
```

Recommended config:

```yaml
model:
  type: midi_transformer
  d_model: 256
  num_layers: 4
  num_heads: 4
  dim_feedforward: 512
  dropout: 0.1
  min_pitch: 28
  max_pitch: 60
```

If training is slow or overfitting:

```yaml
model:
  d_model: 128
  num_layers: 2
  num_heads: 4
  dim_feedforward: 256
```

---

## 6. Loss Function

Use two losses:

```text
activity loss:
  BCEWithLogitsLoss(active_logits, active_label)

pitch loss:
  CrossEntropyLoss(pitch_logits, pitch_label)
  computed only on active frames
```

Total loss:

```text
total_loss = lambda_active * activity_loss
           + lambda_pitch * pitch_loss
```

Recommended:

```yaml
training:
  lambda_active: 1.0
  lambda_pitch: 1.0
```

If model predicts too much silence:

```yaml
training:
  lambda_active: 1.5
```

If model predicts active frames but wrong notes:

```yaml
training:
  lambda_pitch: 1.5
```

---

## 7. Inference

### 7.1 Predict Activity and Pitch

Given context features:

```text
active_prob = sigmoid(active_logits)
pitch_prob = softmax(pitch_logits)
```

For each frame:

```text
if active_prob[t] > activity_threshold:
    pitch_id[t] = argmax pitch_prob[t]
else:
    silence
```

Recommended threshold:

```yaml
inference:
  activity_threshold: 0.5
```

If output is too sparse:

```text
lower threshold to 0.35-0.45
```

If output is too dense / always active:

```text
raise threshold to 0.6-0.7
```

---

### 7.2 Postprocessing

Raw frame predictions are usually fragmented. Postprocessing is required.

Recommended steps:

```text
1. Convert frame-level active + pitch predictions to note segments.
2. Merge adjacent frames with the same pitch.
3. Remove very short notes.
4. Optionally smooth activity predictions.
5. Assign velocity.
6. Export MIDI.
```

Suggested defaults:

```yaml
postprocess:
  min_note_duration_ms: 80
  merge_gap_ms: 50
  default_velocity: 80
  smooth_activity: true
  smoothing_window_ms: 50
```

Monophonic bass rule:

```text
at most one active pitch per frame
```

This makes generated MIDI cleaner and more bass-like.

---

## 8. MIDI Rendering

After generating MIDI, render it into audio using a bass soundfont.

Pipeline:

```text
predicted piano-roll
→ note events
→ generated_bass.mid
→ soundfont renderer
→ generated_bass.wav
```

Possible tools:

```text
pretty_midi
fluidsynth
midi2audio
music21
```

Recommended output files:

```text
outputs/midi/generated_bass.mid
outputs/audio/generated_bass_rendered.wav
outputs/audio/context.wav
outputs/audio/mix_with_generated_bass.wav
```

Rendering is not the model's main contribution. It is a practical way to turn symbolic output into audio.

---

## 9. Evaluation

### 9.1 Symbolic Metrics

Use these first:

```text
frame-level activity accuracy
activity precision / recall / F1
pitch accuracy on active frames
note-level onset F1
note density
active ratio
```

Most important:

```text
activity F1
pitch accuracy on active frames
onset F1
```

---

### 9.2 Visualization

Save these plots:

```text
target bass piano-roll
generated bass piano-roll
context spectrogram
target bass spectrogram
rendered generated bass spectrogram
```

Recommended files:

```text
outputs/figures/target_pianoroll.png
outputs/figures/generated_pianoroll.png
outputs/figures/context_spec.png
outputs/figures/target_bass_spec.png
outputs/figures/generated_bass_rendered_spec.png
```

---

### 9.3 Audio Comparison

Save listening samples:

```text
context.wav
target_bass.wav
generated_bass_rendered.wav
mix_with_generated_bass.wav
```

Optional comparison with audio-token branch:

```text
audio_token_generated_bass.wav
midi_generated_bass.wav
```

---

## 10. Comparison with Audio-Token Branch

The final project should compare two representations.

| Branch | Representation | Strength | Weakness |
|---|---|---|---|
| Audio-token branch | EnCodec tokens | closer to StemGen, can model timbre directly | hard under limited compute, may collapse |
| MIDI branch | symbolic notes / piano-roll | stable rhythm and pitch, easier for small model | timbre depends on soundfont, less general |

Expected project insight:

```text
Under limited compute, symbolic MIDI generation is more stable for bass because it separates musical structure from audio rendering.
```

This comparison is a strong interview talking point.

---

## 11. Recommended File Structure

```text
src/
  features/
    audio_features.py

  midi/
    midi_labels.py
    pianoroll.py
    postprocess.py
    render.py

  models/
    midi_transformer.py

  training/
    train_midi.py
    eval_midi.py

scripts/
  prepare_midi_dataset.py
  train_midi_transformer.py
  generate_midi_bass.py
  render_midi_bass.py
  evaluate_midi_branch.py
  compare_audio_vs_midi.py

configs/
  midi_bass_transformer.yaml

outputs/
  midi/
  audio/
  figures/
  metrics/
```

---

## 12. Suggested Config

```yaml
data:
  dataset_name: babyslakh
  root: data/babyslakh
  target_instrument: bass
  context_mode: mixture_minus_target
  clip_length_sec: 4.0
  hop_length_sec: 0.032
  max_tracks: 20
  clips_per_track: 5

features:
  sample_rate: 32000
  n_mels: 128
  use_mel: true
  use_chroma: true
  use_onset: true

midi:
  min_pitch: 28
  max_pitch: 60
  ignore_index: -100
  polyphony_mode: lowest_pitch

model:
  type: midi_transformer
  d_model: 256
  num_layers: 4
  num_heads: 4
  dim_feedforward: 512
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 0.0003
  num_epochs: 30
  lambda_active: 1.0
  lambda_pitch: 1.0

inference:
  activity_threshold: 0.5

postprocess:
  min_note_duration_ms: 80
  merge_gap_ms: 50
  default_velocity: 80
  smooth_activity: true
  smoothing_window_ms: 50

render:
  soundfont_path: assets/soundfonts/bass.sf2
  output_sample_rate: 32000
```

---

## 13. Minimal Success Criteria

The MIDI branch is successful if it can produce:

```text
context.wav
target_bass.mid or target piano-roll
generated_bass.mid
generated_bass_rendered.wav
mix_with_generated_bass.wav
target vs generated piano-roll plot
basic symbolic metrics
```

The first success criterion is not perfect musicality. It is:

```text
the model predicts non-trivial bass activity and pitch structure
that can be rendered into recognizable bass audio.
```

---

## 14. Recommended Development Order

1. Extract context audio and bass MIDI labels.
2. Build aligned frame grid for audio features and MIDI labels.
3. Implement mel/chroma/onset feature extraction.
4. Implement MIDI-to-active/pitch label conversion.
5. Implement MIDI Transformer model.
6. Train on a tiny subset and overfit 5-10 clips.
7. Train on 50-200 clips.
8. Generate bass MIDI from context.
9. Postprocess and render to audio.
10. Save plots and listening samples.
11. Compare with audio-token branch.

---

## 15. Resume Description

Possible resume bullet:

```text
Added a symbolic MIDI generation branch that predicts bass activity and pitch from audio context using a Transformer encoder, then renders MIDI into bass audio, enabling a practical comparison against a StemGen-style EnCodec token baseline under limited compute.
```

Expanded version:

```text
Built a hybrid context-aware bass generation system comparing audio-token and symbolic MIDI representations. Implemented a Transformer-based MIDI branch that predicts frame-level bass activity and pitch from mel/chroma/onset context features and renders generated MIDI into audio, showing improved stability over fully-masked codec-token generation under small-data constraints.
```
