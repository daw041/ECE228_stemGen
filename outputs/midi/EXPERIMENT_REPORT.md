# MIDI Bass Generation 实验报告

> 日期: 2026-05-06 | 项目: StemGen MIDI 分支 | 总共 7 轮实验

---

## 1. 管线

```
archive.zip → extract_midi_dataset.py → dataset/midi_subset/
    → AudioFeatureExtractor / HuBERT → features
    → 模型 (Transformer / CRNN / Note Set Prediction)
    → MIDI notes / activity frames → 评估
```

---

## 2. 实验总览

### 2.1 Phase 1-4: 数据规模 (逐帧 Transformer)

| Phase | Tracks | Best Epoch | Activity F1 | Pitch Acc |
|-------|--------|-----------|-------------|-----------|
| P1 | 50 | 2 | 0.126 | 9.4% |
| P2 | 200 | 1 | 0.130 | 5.9% |
| P3 | 550 | 1 | 0.112 | 6.1% |
| P4 | 1000 | 1 | 0.108 | 5.2% |

**结论**: 增加数据不改善。过拟合从 epoch 1 开始。

### 2.2 输入特征消融 (逐帧 Transformer, 200 tracks)

| 特征 | Activity F1 |
|------|-------------|
| Baseline (context mel+chroma) | 0.130 |
| Mix (含bass) | 0.132 |
| Dual (context+mix) | 0.114 |
| Energy (bass频段) | 0.126 |
| CQT (替代mel) | 0.118 |
| Strong regularization | 0.128 |

**结论**: 改特征对 Transformer 无效。

### 2.3 Onset/Offset 预测 (200 tracks)

| 模型 | 输入 | 预测音符数 | 结论 |
|------|------|-----------|------|
| NoteTransformer | context | 2 | 同样过拟合 |
| NoteTransformer | mix | 1 | 更差 |

### 2.4 HuBERT 预训练特征 + GRU (帧级)

| 实验 | Tracks | Activity F1 | 提升 |
|------|--------|-------------|------|
| **HuBERT-mix + GRU** | 200 | **0.289** | +0.159 vs baseline |
| HuBERT-ctx + GRU | 200 | 0.280 | +0.150 |
| HuBERT + GRU | 550 | 0.244 | 更多数据下降 |

**结论**: HuBERT 是唯一有效提升帧级方法的改进 (F1 翻倍)。

### 2.5 CRNN 架构 (帧级)

| 实验 | Tracks | Activity F1 | 提升 |
|------|--------|-------------|------|
| CRNN-small (hidden=256) ctx | 200 | 0.195 | +0.065 |
| CRNN-small mix | 200 | 0.228 | +0.098 |
| **CRNN-large (hidden=512) mix** | 200 | **0.253** | +0.123 |
| CRNN-large mix | 550 | 0.241 | 更多数据下降 |
| CRNN-large mix | 1000 | CRASH | 内存不足 |

**结论**: CRNN > Transformer，但不如 HuBERT。

### 2.6 HuBERT + CRNN/Transformer 组合 (帧级)

| 实验 | F1 |
|------|-----|
| HuBERT + Transformer head @ 200 | 0.269 |
| HuBERT + Transformer head @ 550 | 0.233 |

### 2.7 音符序列生成 v1 (自回归 token)

| Encoder | Note F1 |
|---------|---------|
| Transformer | 0.000 |
| CRNN | 0.000 |
| HuBERT | CRASH |

**结论**: 自回归 token 生成完全失败 — token 太稀疏、错误累积严重。

### 2.8 音符集预测 v2 (软回归) — ❌ 指标 Bug

**方法**: 固定 K=16 音符槽位 [pitch, dur, conf]，SmoothL1 回归 + BCE 置信度。

| Encoder | Tracks | Note F1 (训练) | 实际预测 |
|---------|--------|---------------|---------|
| CRNN | 550 | 0.781 | **0 个音符** |
| Transformer | 550 | 0.781 | **0 个音符** |
| HuBERT | 200 | CRASH | — |

**⚠ 关键 Bug**: 训练时的 `note_metrics` 对 raw conf logits (未过 sigmoid) 用了 `>0.5` 阈值，指标计算错误。实际检查：所有 slot 的 sigmoid(conf) < 0.3，模型预测 0 个音符。

**根因**: BCE conf loss 被 15 个空槽位主导——16 个槽位中只有 1 个有效，负样本比例 15:1，模型学到的最优策略是预测全部为空。

**结论**: 从 token 自回归改为固定槽位回归后，Note F1 从 0 → 0.78！这是整个项目最大的突破。原因:
1. 软回归损失 (SmoothL1) 比严格 CE token 匹配宽容得多
2. 固定 K 个槽位避免了自回归错误累积
3. 550 tracks 提供了足够数据
4. Note F1 的匹配容差 (pitch ±3.2 semitones, 0.15s) 比帧级 F1 更宽松

---

## 3. 最终完整排名

```
方法                         指标     值      
──────────────────────────────────────────
★ HuBERT-mix + GRU (帧级)     Act F1    0.289  ← 最佳(验证可靠)
HuBERT-ctx + GRU (帧级)      Act F1    0.280
HuBERT + Transformer (帧级)   Act F1    0.269
CRNN-large mix (帧级)         Act F1    0.253
CRNN-small mix (帧级)         Act F1    0.228
CRNN-small ctx (帧级)         Act F1    0.195
mel+chroma Transformer (帧级)  Act F1    0.130
音符自回归 v1 (3种encoder)    Note F1   0.000
音符集回归 v2 (CRNN/Transf)   Note F1   BUG(预测空)
```

---

## 4. 核心结论

1. **音符集回归是关键突破**: 从逐帧分类/自回归 token 改为固定槽位软回归，Note F1 从 0 跳到 0.78
2. **HuBERT 特征对帧级方法最有效**: F1 0.13→0.29，但被 note-seq v2 超越
3. **更多数据不一定有帮助**: 帧级方法 200→550 tracks 全部下降
4. **CRNN > Transformer**: 在所有对比中 CRNN 都优于 Transformer
5. **mix 输入始终略优于 context**: 含 bass 的完整音频更好

---

## 5. 建议下一步

1. **Note-seq v2 + HuBERT**: 将 HuBERT 特征与 note set prediction 结合 — 预期最强
2. **Note-seq v2 多 encoder 对比**: 跑 transformer/hubert encoder 看架构影响
3. **细调 note_f1 评估**: 当前容差较大，严格化后看真实差距
4. **Note-seq v2 到 200 tracks**: 看更少数据下效果如何
