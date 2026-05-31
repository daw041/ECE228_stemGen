# Minimal StemGen-Style Reproduction Project

## 1. 项目定位

本项目目标是实现一个 **缩小规模但方法尽量贴近 StemGen 原文的 context-aware music stem generation system**。

重点不是训练大模型，也不是追求商业级音质，而是在有限时间内复现 StemGen 的核心问题设定和主要建模方式：

```text
given musical context + target instrument category
→ generate a compatible target stem
```

本项目应被描述为：

```text
a scaled-down StemGen-style reproduction
```

而不是：

```text
a full-scale StemGen reproduction
```

核心原则：

```text
small training scale,
but StemGen-like task formulation, conditioning, token representation, and masked-token generation.
```

---

## 2. 核心目标

项目必须优先贴近 StemGen 的核心方法，而不是只做一个普通 audio-to-audio Transformer baseline。

必须保留的核心设计：

1. 使用多轨音乐数据构造 context-target stem pairs。
2. 使用 neural audio codec 将 waveform 转成离散 codec tokens。
3. 使用 **target instrument conditioning** 控制要生成的乐器。
4. 使用 **context-conditioned target stem token generation**。
5. 训练时 **context tokens 不 mask，只 mask target stem tokens**。
6. 模型尽量采用 **non-autoregressive masked token modeling**，而不是普通 left-to-right autoregressive generation。
7. 推理时使用 iterative mask-predict generation。
8. 将生成的 target tokens decode 回 waveform。
9. 输出可播放的 generated stem 和 mixed audio。
10. 控制训练规模，保证项目能快速完成。

---

## 3. 与 StemGen 的对应关系

StemGen 的核心思想是：

```text
the model listens to existing musical context
and generates a new stem that musically fits that context
```

本项目保留以下关键部分：

| StemGen component | 本项目实现方式 |
|---|---|
| Multi-stem music context | Slakh2100 stems |
| Context-conditioned generation | context tokens condition target token prediction |
| Target instrument category | instrument embedding / condition token |
| Audio codec representation | EnCodec or compatible neural codec |
| Non-autoregressive generation | masked target token prediction |
| Target-only masking | context tokens unmasked, target tokens masked |
| Audio reconstruction | codec decoding |
| Music compatibility evaluation | token loss, spectrogram, onset / rhythm alignment, listening examples |

本项目缩小以下部分：

| Full StemGen | 本项目缩小版 |
|---|---|
| Large-scale model | small Transformer |
| Many instruments fully trained | bass first, other instruments supported by interface |
| Large dataset | small subset of Slakh2100 |
| Long audio | short clips |
| Multiple RVQ codebooks | first version may use one coarse codebook |
| Full sampling tricks | minimal iterative mask-predict first |
| High-quality generation | proof-of-concept generation |
| Full evaluation suite | lightweight objective + listening evaluation |

---

## 4. StemGen-Faithfulness Constraints

为了避免项目偏离 StemGen-style 复现，代码和设计应满足以下约束。

### 4.1 Target instrument 不能写死

模型不能写成 bass-only。

即使第一阶段实际只训练 bass，也必须保留：

```text
target_instrument: bass / drums / piano / guitar / strings / other
```

核心思想是：

```text
same model architecture
+ different target instrument condition
→ different generated stem
```

第一阶段可以只跑：

```text
target_instrument = bass
```

但模型接口应保持：

```text
context tokens
+ masked target tokens
+ target instrument embedding
→ generated target stem tokens
```

### 4.2 数据构造不能只绑定 mixture-minus-target

最小版本可以使用：

```text
target = bass
context = mixture - bass
```

但数据管线应支持更一般的 StemGen-style 构造：

```text
context = a selected subset of stems
target = one stem not included in context
```

例如：

```text
context = drums + piano
target = bass

context = drums + bass
target = guitar

context = piano + guitar
target = drums
```

为了快速完成，第一阶段可以只实际使用：

```text
context = mixture - bass
target = bass
```

但代码和配置里应保留：

```text
target_instrument
context_stem_selection
```

这样后续扩展不用重写模型。

### 4.3 只 mask target，不 mask context

训练时：

```text
context tokens: visible / unmasked
target tokens: partially masked
```

模型任务是：

```text
predict masked target tokens conditioned on visible context tokens and target instrument condition
```

不要把 context tokens 也随机 mask 掉，因为这会偏离 context-aware stem generation 的核心任务。

### 4.4 模型应接近 masked-token generation

推荐训练形式：

```text
input:
  context codec tokens
  partially masked target codec tokens
  target instrument embedding / condition token

output:
  predicted target codec tokens at masked positions
```

也就是：

```text
context tokens + masked target tokens + target instrument condition
→ reconstructed / generated target tokens
```

不要优先做普通 left-to-right autoregressive generation。

### 4.5 Context 和 target 应作为两个音频流处理

为了更接近 StemGen，context 和 target 不应只是简单拼成一个长序列。

推荐设计：

```text
context stream embedding
target stream embedding
target instrument embedding
→ fused time-step representation
→ Transformer
```

简单实现可以使用：

```text
fused_embedding[t] =
  concat_or_sum(
    context_token_embedding[t],
    target_token_embedding[t],
    instrument_embedding
  )
```

第一版不需要完全复刻原论文的所有 embedding 细节，但要避免写成完全普通的 seq2seq：

```text
[all context tokens] → [all target tokens]
```

### 4.6 Codec token 接口应支持 RVQ 扩展

第一版可以只使用一个 codec codebook：

```text
use first / coarse codebook only
```

但数据结构最好保留 codebook 维度：

```text
tokens shape:
  [num_codebooks, time]
```

或者 batch 后：

```text
[batch, num_codebooks, time]
```

这样未来可以扩展到 multiple RVQ codebooks。

更接近 StemGen 的扩展方向是：

```text
hierarchical codebook prediction
```

但这不是第一版必须实现。

### 4.7 推理使用 iterative mask-predict

最小推理方式：

```text
start from fully masked or mostly masked target tokens
→ predict masked positions
→ keep confident tokens
→ repeat for several iterations
→ final generated target tokens
```

可选更贴近原文的扩展：

```text
causal-biased iterative decoding
classifier-free guidance over audio context and instrument condition
```

这两个不是第一版必须做。

---

## 5. 数据集设定

推荐数据集：

```text
Slakh2100
```

原因：

- 有 aligned stems
- 有 mixture
- 乐器类别清晰
- 适合构造 context-target generation task
- 与 StemGen-style 任务设定接近

第一阶段实际训练任务：

```text
target = bass
context = mixture - bass
```

但配置层面应保留：

```yaml
target_instrument: bass

supported_instruments:
  - bass
  - drums
  - piano
  - guitar
  - strings
  - other

context_mode: mixture_minus_target
# future options:
# context_mode: random_stem_subset
```

clip 设置建议：

```text
clip length: 3-5 seconds
sample rate: follow codec requirement, e.g. 24kHz or 32kHz
overfit test: 5-10 clips
first training set: 50-200 clips
optional extension: 500-1000 clips
```

不要一开始做完整 Slakh2100 全量训练。

---

## 6. Audio Codec Tokenization

使用预训练 neural audio codec，例如 EnCodec 或兼容模型。

本项目不训练 codec，只使用它完成：

```text
waveform → discrete codec tokens
discrete codec tokens → waveform
```

需要支持：

```text
context.wav → context_tokens
target.wav → target_tokens
generated_target_tokens → generated_target.wav
```

第一版为了降低难度，可以：

```text
use only the first / coarse codebook
```

但实现上应保留：

```text
num_codebooks
codebook_index
```

方便后续扩展到 multiple RVQ codebooks。

---

## 7. 模型核心设计

### 7.1 输入输出

模型输入：

```text
context codec tokens
partially masked target codec tokens
target instrument condition
```

模型输出：

```text
predicted target codec token distribution
```

训练目标：

```text
cross entropy over masked target token positions only
```

### 7.2 Target instrument conditioning

必须有 instrument embedding 或 condition token：

```text
instrument_embedding = Embedding(num_instruments, dim)
```

即使当前只有 bass 训练，也不要移除这个模块。

推荐配置：

```yaml
target_instrument: bass
num_instruments: 6
instrument_vocab:
  bass: 0
  drums: 1
  piano: 2
  guitar: 3
  strings: 4
  other: 5
```

### 7.3 Two-stream token fusion

推荐结构：

```text
context token embedding
target token embedding
instrument embedding
→ fusion
→ Transformer encoder
→ token prediction head
```

简单实现可以是：

```text
x_context = Emb(context_tokens)
x_target = Emb(masked_target_tokens)
x_inst = Emb(target_instrument)

x = Linear(concat(x_context, x_target, x_inst))
```

或者：

```text
x = x_context + x_target + x_inst
```

具体实现可由 code agent 选择，但必须保留 context stream 和 target stream 的区别。

### 7.4 Transformer 配置

建议小模型：

```text
embedding dim: 256
layers: 4
heads: 4
dropout: 0.1
loss: cross entropy
```

如果训练太慢，可以降到：

```text
embedding dim: 128
layers: 2
heads: 4
```

优先保证 pipeline 跑通。

---

## 8. 训练策略

训练任务：

```text
predict masked target codec tokens
conditioned on unmasked context tokens and target instrument embedding
```

推荐训练顺序：

1. 验证 codec encode / decode。
2. 构造 Slakh context-target clips。
3. overfit 5-10 个 bass clips。
4. 在 50-200 个 bass clips 上训练。
5. 保存 loss curve 和 generated samples。
6. 如果时间允许，再加入第二个 target instrument，例如 drums。

合理验收标准：

```text
training loss decreases
masked token accuracy improves
generated tokens can be decoded
generated target audio is playable
generated target mixed with context is listenable
```

不要求短期内达到论文级音质。

---

## 9. 推理方式

最小推理流程：

```text
input context audio
→ encode context tokens
→ choose target_instrument
→ initialize target tokens as [MASK]
→ iterative mask-predict sampling
→ generated target tokens
→ codec decode
→ generated target.wav
→ mix with context
```

第一版可以使用固定迭代次数：

```text
num_iterations: 4-8
```

每一步：

```text
predict masked positions
keep high-confidence tokens
continue predicting remaining masked positions
```

可选扩展：

```text
causal-biased decoding
classifier-free guidance
temperature / top-k sampling
```

这些不属于第一版必须实现。

---

## 10. Evaluation

最小评估即可，不要做过重实验。

必须有：

```text
training loss
validation loss
masked token accuracy
generated audio samples
```

推荐有：

```text
mel-spectrogram distance
multi-resolution STFT distance
onset alignment score
spectrogram visualization
```

定性样例建议保存：

```text
context.wav
target_bass.wav
reconstructed_bass.wav
generated_bass.wav
mix_with_generated_bass.wav
```

如果后续支持其他乐器，可保存：

```text
generated_drums.wav
generated_piano.wav
```

音乐生成项目里，可听样例比复杂指标更重要。

---

## 11. Optional Extension: Beat-Aware Conditioning

如果 baseline 已经跑通，可以加入一个小改进：

```text
beat-aware / onset-aware conditioning
```

设计：

```text
context audio → onset envelope / beat feature
beat feature → embedding
add to token embedding or condition transformer
```

对比：

```text
baseline:
context tokens + masked target tokens + instrument embedding

improved:
context tokens + masked target tokens + instrument embedding + beat/onset embedding
```

这个部分是本项目的 extension，不是 StemGen 原文的必要复现部分。

---

## 12. 工程要求

项目不需要一开始做复杂系统，但需要有清晰结构。

最低应包含：

```text
README.md
requirements.txt
configs/
scripts/
src/
outputs/
```

核心脚本：

```text
prepare_data.py
encode_tokens.py
train.py
generate.py
evaluate.py
```

code agent 可以自行规划具体函数和文件拆分，但必须保持以下 pipeline：

```text
Slakh stems
→ choose target_instrument
→ build context-target pair
→ codec tokens
→ masked target token Transformer training
→ iterative generation conditioned on target_instrument
→ codec decoding
→ audio examples
```

---

## 13. 最小成功标准

项目最小成功标准：

```text
given a short music context and a target instrument label,
generate target stem codec tokens,
decode them into generated_target.wav,
and mix them with the original context.
```

第一版可以具体实现为：

```text
context = mixture - bass
target_instrument = bass
output = generated_bass.wav
```

必须能产出：

```text
context.wav
target_bass.wav
generated_bass.wav
mix_with_generated_bass.wav
loss curve
short README result summary
```

如果只完成这些，也已经是一个合理的 scaled-down StemGen-style reproduction。

---

## 14. 推荐开发优先级

优先级从高到低：

1. Codec encode / decode 正常。
2. Slakh2100 context-target pair 构造正确，并支持 `target_instrument` 配置。
3. Instrument embedding / condition token 接入模型。
4. Context stream 和 target stream 分开 embedding，再融合。
5. Masked target token training 跑通。
6. Overfit 5-10 bass clips。
7. Small bass dataset training。
8. Iterative mask-predict generation conditioned on target instrument。
9. Decode and mix generated audio。
10. Save audio examples and plots。
11. Optional: add drums / piano as second target instrument。
12. Optional: random context subset selection。
13. Optional: multiple RVQ codebooks。
14. Optional: causal-biased decoding / classifier-free guidance。
15. Optional: beat-aware conditioning。

不要优先做：

- Web UI
- Gradio demo
- 大规模训练
- 一开始完整训练所有乐器
- full multiple RVQ codebook hierarchy
- complex sampling tricks
- 复杂评估
- 过度工程化

---

## 15. Scope Control

### Must Implement

```text
EnCodec encode/decode
Slakh2100 bass clips
target_instrument config
instrument embedding
context-target pair
context unmasked
target-only masking
one-codebook masked-token Transformer
iterative mask-predict generation
generated audio samples
```

### Interface Should Support

```text
multiple target instruments
random context stem subset
multiple RVQ codebooks
different context construction modes
```

### Optional Later

```text
multi-instrument joint training
hierarchical RVQ codebook prediction
causal-biased decoding
classifier-free guidance
beat-aware conditioning
larger dataset training
```

---

## 16. 简历表述方向

完成后可以写：

```text
Reproduced a scaled-down StemGen-style music stem generation pipeline using neural audio codec tokens and a non-autoregressive masked-token Transformer conditioned on musical context and target instrument embeddings, enabling the same model architecture to generate different instrument stems by changing the target condition.
```

如果加入 beat-aware conditioning：

```text
Extended the baseline with onset-aware conditioning to improve rhythmic compatibility between generated bass stems and existing musical context.
```

---

## 17. 当前边界

本项目明确不追求：

- 完整复现 StemGen 大模型
- 论文级生成质量
- 全量 Slakh2100 训练
- 一开始训练所有乐器
- 长音乐生成
- 商业可用音质

当前最重要的是：

```text
methodologically close to StemGen,
small enough to finish quickly,
complete enough to demonstrate engineering and ML understanding.
```
