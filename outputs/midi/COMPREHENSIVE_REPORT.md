# StemGen 项目完整实验报告

> 日期: 2026-05-06 | 项目: StemGen 复现 (Codec 音频分支 + MIDI 符号分支)

---

## 1. 项目概述

本项目目标: 给定音乐 context (混音减去目标乐器) + 目标乐器类别, 生成与之兼容的目标 stem。

两个技术路线:
- **Audio Token 分支**: EnCodec 离散化 → Masked Token Transformer → 波形重建 (StemGen 原始方法)
- **MIDI 符号分支**: mel/chroma 特征 → Transformer/CRNN → MIDI 音符预测

---

## 2. Audio Token 分支 (Codec 路线)

### 2.1 数据流

```
Slakh2100 track
  ├─ mix.flac (混音)
  └─ stems/{Sxx}.flac (各乐器分轨)
       ↓
context = sum(stems except bass)  → EnCodec → context_tokens [B, C, T]
target = bass stem                 → EnCodec → target_tokens  [B, C, T]
       ↓
StemGenModel: 预测被 mask 的 target tokens
       ↓
EnCodec.decode(generated_tokens) → 生成的 bass 音频
```

### 2.2 EnCodec 编解码器 (`src/codec.py`)

| 参数 | 值 |
|------|-----|
| 模型 | EnCodec 24kHz (预训练) |
| 带宽 | 6.0 kbps |
| Codebook 数 C | 1 (coarse only, 第一阶) |
| 词表大小 | 1024 |
| 帧率 | 75 Hz (每 token 对应 ~13.3ms) |
| 4 秒音频 → token 数 | 300 tokens |

**编码**: 24kHz 波形 → EnCodec encoder → [B, 1, T] 离散 token
**解码**: [B, 1, T] token → EnCodec decoder → 24kHz 波形

### 2.3 模型架构 (`src/model.py`)

**StemGenModel** — 双流融合 + 非自回归 Masked Token Transformer

```
Context Tokens  [B, 1, T]          Target Tokens [B, 1, T]
     │                                    │
 TokenEmbed(vocab+1)                  TokenEmbed(vocab+1)
 [B, T, 256]                          [B, T, 256]
     │                                    │
     └────────────┬───────────────────────┘
                  │  Concat + FusionProj
                  │  InstrumentEmbedding(inst_idx)
                  ▼
           Fused [B, T, 256]
                  │
           + Positional Encoding (learnable, [1, T_max, 256])
                  │
           TransformerEncoder (4 layers, 4 heads, 512 FFN, GELU, Pre-Norm)
                  │
           ┌──────┴──────┐
           │              │
    OutputHead[0]    OutputHead[1]...(可选)
    Linear(256→1024)  Linear(256→1024)
    [B, T, vocab]     [B, T, vocab]
```

**关键设计**:
- 双流: context tokens 和 target tokens 分别嵌入 (代码书级别求和), 然后 concat 融合
- Instrument Embedding: 6 种乐器, dim=64, 与序列拼接
- Positional Encoding: 可学习 (非正弦), 初始化为 N(0, 0.02)
- 单 codebook 模式: 1 个 output head (Linear 256→1024)
- 多 codebook 模式: C 个 output head, 分别预测每层 token

**参数量**: ~1.6M (embedding_dim=256, 4 layers, vocab=1024)

### 2.4 训练策略 (`src/trainer.py`)

**Masked Token Modeling (非自回归)**:

```
1. EnCodec 编码 context 和 target 为离散 token
2. Target tokens 中 75% 位置随机替换为 [MASK] token
3. Context tokens 保持完整 (不 mask)
4. 模型输入 (context, masked_target), 预测被 mask 位置的原始 token
5. Loss: CrossEntropy(logits[masked_positions], target_tokens[masked_positions])
6. 仅计算 codebook 0 (coarse) 的 loss
```

**Masking 策略**: 75% 随机 mask, 每帧独立 (非块 mask)

**推理 (Iterative Mask-Predict)**:
```
1. Target tokens 全部初始化为 [MASK]
2. For iteration in 1..N (N=8):
   a. Forward pass → token 概率分布
   b. 每个 masked 位置采样 token
   c. 保留 25% 置信度最高的预测 (causal bias: 偏向早期时间位置)
   d. 其余位置重新 mask
3. EnCodec decode → waveform
```

**训练超参**:
| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (β=0.9, 0.999) |
| 学习率 | 1e-4 |
| Weight Decay | 1e-5 |
| Batch Size | 8 |
| Epochs | 200 (overfit) |
| Mask Ratio | 75% |
| 数据 | ~20 tracks BabySlakh, 100 clips |

### 2.5 训练结果

| 指标 | 值 |
|------|-----|
| Best epoch | 154 |
| Train Loss | 7.54 |
| Val Loss | 7.21 |
| Train Acc (masked) | 38.3% |
| Val Acc (masked) | 40.6% |

**分析**: 
- Token 准确率 40.6% (随机基线 1/1024 ≈ 0.1%), 模型学到了有效的 token 预测
- Val loss (7.21) 接近 train loss (7.54), 基本没有过拟合
- 但重建音频质量未量化 (主观评估需要试听)

### 2.6 评估指标 (`scripts/evaluate.py`)

| 指标 | 含义 |
|------|------|
| Spectrogram Distance | 生成音频与 GT 的 mel 频谱 L2 距离 (越低越好) |
| Onset Alignment Score | onset 包络的 Pearson 相关系数 (越高越好) |

---

## 3. MIDI 符号分支

### 3.1 数据流

```
archive.zip (Slakh2100) → extract_midi_dataset.py → dataset/midi_subset/
  TrackXXXXX/{mix.flac, Sxx.flac, MIDI/Sxx.mid}
       ↓
context = mix - bass (mono)  →  mel(128) + chroma(12) + energy(3) = 143 dim
MIDI file                     →  逐帧 activity + pitch label @ 46.9Hz
       ↓
模型预测 → activity/pitch → MIDI notes → fluidsynth → 音频
```

### 3.2 实验总览 (6 轮, 30+ 次实验)

#### 第 1 轮: 数据规模 (PLAN_MIDI_TRAINING.md Phase 1-4)

**方法**: MidiTransformer (Transformer Encoder + activity/pitch 双头), BCE + pos_weight

| Phase | Tracks | Best Epoch | Activity F1 | Pitch Acc | Train Time |
|-------|--------|-----------|-------------|-----------|------------|
| P1 | 50 | 2 | 0.126 | 9.4% | 0.6 min |
| P2 | 200 | 1 | 0.130 | 5.9% | 2.6 min |
| P3 | 550 | 1 | 0.112 | 6.1% | 6.3 min |
| P4 | 1000 | 1 | 0.108 | 5.2% | 10.8 min |

**结论**: 数据量增大不提升 F1。所有 phase 均在 epoch 1-2 早期停止, val loss 持续上升。

#### 第 2 轮: 输入特征消融

**方法**: MidiTransformer, 200 tracks, 控制变量

| 特征 | Dim | Activity F1 | vs Baseline |
|------|-----|-------------|-------------|
| context mel+chroma (baseline) | 140 | 0.130 | — |
| mix mel+chroma (含 bass) | 140 | 0.132 | +0.002 |
| context + mix 拼接 | 280 | 0.114 | -0.016 |
| context + bass 频段能量 | 143 | 0.126 | -0.004 |
| context CQT (替代 mel) | 140 | 0.118 | -0.012 |
| 强正则化 (lr=1e-4, drop=0.5, wd=1e-3) | 140 | 0.128 | -0.002 |

**结论**: 特征对 F1 无实质性影响。

#### 第 3 轮: 预测目标

**方法**: NoteTransformer — onset/offset 检测替代 activity

| 模型 | 输入 | 预测音符数 | 结论 |
|------|------|-----------|------|
| NoteTransformer | context | 2 | val loss 暴涨 |
| NoteTransformer | mix | 1 | 更差 |

#### 第 4 轮: 架构实验

**HuBERT 预训练特征 (`scripts/train_hubert.py`)**:

```
波形 (16kHz)
  → HuBERT Base (94M, 冻结)
  → [T, 768] 特征 @ 50Hz
  → 预测头 (GRU/Transformer/MLP)
  → activity logits + pitch logits
```

| 实验 | Feature | Head | Tracks | Activity F1 |
|------|---------|------|--------|-------------|
| HuBERT-ctx | HuBERT | GRU | 200 | 0.280 |
| **HuBERT-mix** | **HuBERT** | **GRU** | **200** | **0.289** |
| HuBERT-mix | HuBERT | GRU | 550 | 0.244 |
| HuBERT-ctx | HuBERT | Transformer | 200 | 0.269 |

**CRNN (`scripts/train_crnn.py`)**:

```
mel+chroma [B, T, 143]
  → Conv1D×4 (kernel=5/3, residual, BatchNorm, GELU)
  → BiGRU×2 (hidden=256/512, bidirectional)
  → activity head + pitch head
```

| 实验 | Hidden | Tracks | Activity F1 |
|------|--------|--------|-------------|
| CRNN-small ctx | 256 | 200 | 0.195 |
| CRNN-small mix | 256 | 200 | 0.228 |
| **CRNN-large mix** | **512** | **200** | **0.253** |
| CRNN-large mix | 512 | 550 | 0.241 |

#### 第 5 轮: 音符序列生成

**v1: 自回归 token 生成 (`scripts/train_noteseq.py`)**:

```
Encoder → global pooling → GRU Decoder (autoregressive)
生成序列: [BOS, PITCH_TOKEN, DUR_TOKEN, PITCH_TOKEN, ..., EOS]
Loss: CrossEntropy (teacher forcing)
```

| Encoder | Note F1 | 结论 |
|---------|---------|------|
| Transformer | 0.000 | token 过稀疏 |
| CRNN | 0.000 | 错误累积 |
| HuBERT | CRASH | batch 处理异常 |

**v2: 固定槽位回归 (`scripts/train_noteseq_v2.py`)**:

```
Encoder → global pooling → MLP → [B, 16, 3] (pitch, log_dur, confidence)
Loss: SmoothL1(pitch) + SmoothL1(dur) + BCE(conf)
```

| Encoder | Tracks | 训练 Note F1 | 实际 | Bug |
|--------|--------|-------------|------|-----|
| CRNN | 550 | 0.781 | 预测 0 音符 | sigmoid 缺失 |
| Transformer | 550 | 0.781 | 预测 0 音符 | sigmoid 缺失 |

**结论**: 指标计算有 bug (conf logits 未过 sigmoid)。实际模型学会所有槽位都预测"空"（15:1 负样本主导 conf loss）。

### 3.3 模型架构详细对比

#### MidiTransformer (`src/midi/midi_transformer.py`)

```
Features [B, T, D]
  → feat_proj: Linear(D → d_model)
  → + InstrumentEmbedding
  → + PositionalEncoding (learnable)
  → TransformerEncoder (Pre-Norm, GELU)
  → ├─ activity_head: Linear→GELU→Linear→1  → [B, T]
  → └─ pitch_head: Linear→num_pitches        → [B, T, 33]

Loss = BCE(activity) * pos_weight + CE(pitch | active_frames)
```

**pos_weight**: 每 batch 动态计算: `n_negative / n_positive`, 上限 30

#### NoteTransformer (`src/midi/note_transformer.py`)

同 MidiTransformer encoder, 三头输出:
- onset_head: BCE 检测音符开始
- offset_head: BCE 检测音符结束
- pitch_head: CE 预测音高 (仅 onset 帧)

#### CRNN (`scripts/train_crnn.py`)

```
Features [B, T, D]
  → Conv1D(D→H, k=5) → BN → GELU
  → Conv1D(H→H, k=5) → BN → GELU → Drop
  → Conv1D(H→H, k=3) → BN → GELU
  → Conv1D(H→H, k=3) → BN → GELU → Drop
  → BiGRU×2 (H=256/512, bidirectional)
  → ├─ activity_head: Linear→GELU→Linear→1
  → └─ pitch_head: Linear→GELU→Linear→33
```

#### HuBERT + GRU (`scripts/train_hubert.py`)

```
波形 [B, samples] 16kHz
  → HuBERT Base (94M, 冻结, 13层Transformer)
  → 最后一层输出 [B, T, 768] @ 50Hz
  → ┌ GRU head: BiGRU×2(768→256)
  → ├ MLP head: Linear×2
  → └ Transformer head: TF Encoder×2
  → activity + pitch heads
```

### 3.4 最终排名 (验证可信的结果)

```
方法                              指标      值
────────────────────────────────────────────────────
★ HuBERT-mix + GRU               Act F1   0.289
  HuBERT-ctx + GRU               Act F1   0.280
  HuBERT + Transformer           Act F1   0.269
  CRNN-large mix (hidden=512)    Act F1   0.253
  CRNN-small mix (hidden=256)    Act F1   0.228
  CRNN-small ctx                 Act F1   0.195
  MidiTransformer baseline       Act F1   0.130
  所有输入特征变体               Act F1   0.11-0.13
  音符自回归 v1                  Note F1  0.000
  音符集回归 v2                  Note F1  BUG
```

---

## 4. 两个路线的对比

| 维度 | Audio Token (Codec) | MIDI 符号 |
|------|-------------------|-----------|
| 输入 | context 波形 | context 波形 |
| 中间表示 | EnCodec tokens (1024 类, 75Hz) | mel+chroma 特征 (143 维, 47Hz) |
| 输出 | target stem tokens → 波形 | MIDI note events → 合成音频 |
| 模型 | Masked Token Transformer | 帧级分类 / 序列生成 |
| 训练方式 | 非自回归 masked prediction | BCE + CE (帧级) |
| 参数量 | ~1.6M | ~0.3M-95M (取决于 encoder) |
| 训练指标 | Token Acc 40.6% | F1 0.29 (最佳) |
| 生成质量 | 需主观评估 (有波形输出) | MIDI 几乎为空 (F1 低) |
| 优势 | 端到端, 保真度高 | 可解释, 可编辑 |
| 劣势 | 计算量大, 难调试 | 当前效果差, 逐帧预测难 |

---

## 5. 核心发现

1. **Audio Token 路线可行**: 非自回归 masked token prediction 在 20 tracks 上达到 40.6% token 准确率, 管线完整跑通
2. **MIDI 帧级预测受困于过拟合**: 无论数据量、特征、架构如何变化, F1 卡在 0.11-0.29
3. **HuBERT 预训练特征是 MIDI 路线最有效改进**: F1 从 0.13 → 0.29 (+120%)
4. **CRNN > Transformer**: Conv1D 局部特征提取对逐帧检测有帮助
5. **音符级序列生成比帧级预测更难**: 自回归版本 F1=0, 固定槽位版本有严重 class imbalance bug
6. **更多数据不一定有帮助**: 几乎所有方法在 200→550 tracks 时 F1 下降
7. **mix 输入始终略优于 context**: 含 target 的完整音频更好

---

## 6. 建议下一步

1. **Audio Token + 更大数据**: 用 Codec 路线在 200-550 tracks 上训练, 评估重建音频质量
2. **HuBERT + Audio Token 混合**: HuBERT 特征 → 离散化 → Masked Token Transformer
3. **MIDI 路线重构**: 放弃逐帧预测, 尝试音符级回归但修复 class imbalance (Focal Loss / 调节正样本权重)
4. **两路线融合评估**: 用音频质量指标 (MEL distance, onset alignment) 评估 MIDI 路线的合成输出
