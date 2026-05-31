# 下一阶段实验计划与分工

本文档用于两人协作推进课程项目。主线目标不是转向 MIDI，而是继续深化
**audio-token StemGen 复现路线**，把当前代码整理成一个小规模但方法对齐的
StemGen-style reproduction。

## 1. 项目定位

课程项目要求是复现一篇机器学习与工程/科学应用相关论文，并在其上做一些改进尝试。
本项目复现论文：

```text
StemGen: A music generation model that listens
```

问题定义：

```text
输入: context audio = mixture - target stem
条件: target instrument，例如 bass
输出: 与 context 匹配的 target stem audio
```

这属于音乐音频工程中的 context-aware stem generation / music source generation
问题。模型需要“听”到已有音乐上下文，然后生成目标乐器声部。

本项目最终应强调：

- 我们复现 StemGen 的核心建模范式：neural audio codec tokens + masked token modeling。
- 我们在小算力下实现 scaled-down reproduction。
- 我们诊断了小规模复现失败的原因，并尝试改进 codebook 对齐、mask schedule、sampling 等核心环节。

MIDI 分支只作为探索性附录，不作为主线替代。

## 2. 论文方法与本项目对应关系

StemGen 原论文核心设计：

| 组件 | StemGen 原论文 | 本项目小规模实现 |
|---|---|---|
| 数据 | Slakh 等多 stem 音频 | Slakh/BabySlakh 风格 stem 数据 |
| 输入 | context-mix tokens | `context = mix - bass` |
| 输出 | target-stem tokens | bass stem tokens |
| Codec | MusicGen 同款 32kHz EnCodec | 当前使用 24kHz EnCodec |
| RVQ codebook | 4 tokens/frame, vocab 2048 | 当前 E5 默认 2 codebooks, vocab 1024 |
| 模型 | 约 250M LLaMA-style Transformer | 8-layer, 512-dim Transformer |
| 训练 | target-only masked token prediction | context 不 mask，只 mask target |
| 推理 | iterative mask-predict | 支持 per-codebook steps/top-k/argmax |
| 改进 | multi-source CFG + causal bias | 当前已支持 causal bias，CFG 待实现 |

由于没有官方训练代码和权重，本项目是 paper-guided reproduction，不是 official-code rerun。

## 3. 当前仓库状态

仓库已经整理为 Git 项目：

```text
configs/                 audio-token 配置
docs/                    文档与实验计划
scripts/train.py         audio-token 训练
scripts/generate.py      full generation
scripts/diagnose_audio_token.py
                          codec reconstruction + partial reconstruction 诊断
src/codec.py             EnCodec wrapper
src/model.py             StemGen-style Transformer
src/trainer.py           multi-codebook masked-token trainer
src/midi/                MIDI 探索分支
outputs/                 本地实验结果，Git 只跟踪 markdown
dataset/                 本地数据，不跟踪大文件
```

`.gitignore` 已经排除：

- `dataset/archive.zip`
- checkpoints: `*.pt`, `*.ckpt`
- audio/MIDI/figures: `*.wav`, `*.mid`, `*.png`
- cache/log/tmp 文件

## 4. 之前已经做过什么

### 4.1 Audio-token 路线

早期实验记录见 `outputs/experiment_log.md`。

已完成：

- 搭建 EnCodec tokenization + masked Transformer 训练管线。
- 实现 context/target 双流输入。
- 实现 iterative mask-predict 生成。
- 在小规模数据上尝试 1/2/3/4 codebooks。

主要结果：

| 实验 | 设置 | 结果 | 结论 |
|---|---|---|---|
| baseline | 1cb, 4s, 315 tracks | val acc 28.0% | 管线可训练，但生成质量差 |
| E1 | 2cb, 10s, 200 tracks | val acc 29.4% | 比 1cb 略好 |
| E2 | 2cb, 10s, 550 tracks | val acc 51.8% | token prediction 明显提升 |
| E3 | 4cb, 10s, 150 tracks | val acc 16.8% | 数据不足，4cb 失败 |
| E4 | 3cb + activity/casual bias | 未形成稳定结论 | 需要重新规范评估 |

核心问题：

- token accuracy 上升不等于生成音频可听。
- full-mask generation 直接生成噪音，说明训练分布和生成分布可能不匹配。
- 旧代码只训练 codebook 0，和多 codebook 目标不对齐。
- 旧 decode 可能用 token 0 补缺失 codebook，这会制造错误 residual 信息。

### 4.2 MIDI 路线

MIDI 分支做了很多探索，包括 Transformer、CRNN、HuBERT、NoteSeq。

可靠结果：

- 普通 MIDI Transformer activity F1 约 0.13。
- CRNN-large mix activity F1 约 0.253。
- HuBERT-mix + GRU activity F1 约 0.289。
- NoteSeq v1 失败。
- NoteSeq v2 存在 confidence/class-imbalance 问题，训练指标不可信。

结论：

MIDI 分支有分析价值，但偏离 StemGen 论文复现主线。后续只保留为附录或负结果分析，不继续投入主要算力。

## 5. 已完成的 E5 代码整理

当前代码已经为 E5 audio-token 主线做了准备：

- `src/codec.py`
  - codec/model/trainer/generator 共享同一个 `num_codebooks`。
  - 不再把缺失 RVQ codebook 用 token 0 补齐。

- `src/trainer.py`
  - 支持 multi-codebook masked CE。
  - 同一时间 mask 同时作用到所有 codebooks。
  - 支持 variable mask ratio，例如 0.50-1.00。

- `src/model.py`
  - generation 支持 per-codebook decoding steps。
  - 支持 `top_k` 和 greedy `argmax` decoding。

- `scripts/diagnose_audio_token.py`
  - 保存 target、codec reconstruction。
  - 保存 partial reconstruction: mask 15/30/50/75/100%。
  - 保存 full generation。
  - 输出 mel spectrogram 对比图。

当前默认配置：

```yaml
num_codebooks: 2
clip_duration: 10.0
embedding_dim: 512
num_layers: 8
num_heads: 8
feedforward_dim: 2048
mask_ratio_min: 0.50
mask_ratio_max: 1.00
train_n_clips: 1000
val_n_clips: 120
```

## 6. 下一阶段核心问题

下一阶段只回答一个问题：

```text
在有限算力下，audio-token StemGen-style 模型到底卡在哪里？
```

必须按顺序判断：

1. Codec 自身能不能重建 bass stem？
2. 模型能不能做 partial-mask reconstruction？
3. 模型能不能从 full mask 生成？
4. 如果 full generation 差，是 sampling 问题、mask schedule 问题，还是条件控制问题？

不要跳过前两步直接听 full generation。

## 7. 实验计划

### E5-0: 环境和 smoke test

目标：确认代码、依赖、数据路径、GPU 都正常。

第一步先跑不依赖 EnCodec/数据集的代码结构测试：

```bash
python scripts/smoke_test_e5.py --device cpu --seq_len 64 --batch_size 2
```

如果服务器 CUDA 环境已经可用，再跑：

```bash
python scripts/smoke_test_e5.py --device cuda --seq_len 64 --batch_size 2
```

这一步只检查：

- E5 配置能否正确加载。
- 2-codebook 模型能否 forward。
- multi-codebook masked CE 是否能反传。
- generation 是否能清空所有 mask token。

第二步再跑真实 codec/dataset 相关 overfit：

如果还没有小规模真实音频子集，先只从 `dataset/archive.zip` 抽几首：

```bash
python scripts/extract_audio_subset.py \
  --archive dataset/archive.zip \
  --out_dir dataset/audio_subset \
  --n_tracks 4
```

然后跑真实数据单 batch 测试：

```bash
python scripts/smoke_test_e5_data.py \
  --data_root dataset/audio_subset \
  --clip_duration 10.0 \
  --device cuda
```

这一步已经在本地 `E:/conda_envs/torch_study` 环境通过，10 秒 clip 的 token shape 为
`(1, 2, 750)`。

第三步再跑 overfit：

```bash
pip install -r requirements.txt
python scripts/train.py --overfit --device cuda
```

成功标准：

- 不报错。
- checkpoint 正常保存。
- overfit train loss 明显下降。
- per-codebook accuracy 有输出。

产出：

```text
outputs/audio_token/e5_2cb/checkpoints/best.pt
```

负责人：成员 A。

### E5-1: Codec reconstruction gate

目标：判断 EnCodec 2-codebook 设置本身是否能保留 bass 结构。

命令：

```bash
python scripts/diagnose_audio_token.py \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 32,16 \
  --output_dir outputs/audio_token/e5_2cb/diagnostics_overfit
```

重点听：

- `target.wav`
- `codec_reconstruction.wav`

如果 codec reconstruction 本身很差：

- 不要继续训练大模型。
- 改测试 4 codebooks 或更高 bandwidth。

负责人：成员 B。

### E5-2: 2-codebook 正式训练

目标：训练当前默认 E5 2cb baseline。

命令：

```bash
python scripts/train.py \
  --data_config configs/data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/train_config.yaml \
  --device cuda
```

记录指标：

- train loss
- val loss
- overall masked token accuracy
- per-codebook accuracy
- best epoch
- 是否出现过拟合

成功标准：

- codebook 0 和 codebook 1 的 val acc 都明显高于随机。
- 低 mask partial reconstruction 比 full generation 明显更好。

负责人：成员 A。

### E5-3: Partial-mask reconstruction

目标：判断模型是否真的学会修复 target tokens。

命令：

```bash
python scripts/diagnose_audio_token.py \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --mask_ratios 0.15,0.30,0.50,0.75,1.00 \
  --iterations_per_codebook 32,16 \
  --temperature 0.8 \
  --top_k 50 \
  --output_dir outputs/audio_token/e5_2cb/diagnostics
```

判断规则：

- 15/30% mask 都差：训练、数据、codec、模型结构仍有 bug。
- 15/30% mask 可听，75/100% 差：模型有 reconstruction 能力，但 generation 难。
- 100% full generation 接近噪音：继续调 sampling、mask schedule、CFG。

产出：

```text
diagnostic_mels.png
partial_recon_mask_015.wav
partial_recon_mask_030.wav
partial_recon_mask_050.wav
partial_recon_mask_075.wav
partial_recon_mask_100.wav
full_generation.wav
```

负责人：成员 B。

### E5-4: Sampling ablation

目标：确认 full generation 噪音是否来自采样策略。

实验矩阵：

| 设置 | iterations | temperature | top_k | argmax |
|---|---:|---:|---:|---|
| S1 | 32,16 | 1.0 | none | no |
| S2 | 32,16 | 0.8 | 50 | no |
| S3 | 64,32 | 0.8 | 50 | no |
| S4 | 64,32 | 1.0 | none | yes |
| S5 | 128,64 | 0.8 | 50 | no |

命令模板：

```bash
python scripts/diagnose_audio_token.py \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 64,32 \
  --temperature 0.8 \
  --top_k 50 \
  --output_dir outputs/audio_token/e5_2cb/sampling_s3
```

记录：

- 是否有明显低频连续噪声。
- 是否出现 rhythmic gaps。
- 是否和 context 有节奏关系。
- mel spectrogram 是否比原先更接近 target。

负责人：成员 B 主做，成员 A 辅助判断。

### E5-5: 4090 一天算力实验

前提：

- E5-2/E5-3 证明 pipeline 是健康的。
- low-mask partial reconstruction 至少有可听结果。

目标：在 4090 上扩大训练，而不是盲目换方向。

建议实验：

```yaml
num_codebooks: 2
clip_duration: 10.0
train_n_clips: 3000 或 5000
val_n_clips: 300
num_epochs: 300-500
batch_size: 8 或 16
```

如果显存允许，可试：

```yaml
embedding_dim: 768
num_layers: 10 或 12
num_heads: 12
feedforward_dim: 3072
```

不要同时改太多变量。优先扩大数据和训练步数，其次再扩大模型。

负责人：

- 成员 A：启动训练、监控 loss/acc、保存 checkpoint。
- 成员 B：每隔固定 epoch 跑 diagnostics，整理样本和表格。

### E6: 可选改进 - CFG

只有在 E5 证明 pipeline 可用后再做。

目标：

```text
训练时随机 drop context / instrument condition。
推理时用 conditional logits 和 unconditional logits 做 guidance。
```

预期作用：

- 增强生成结果对 context 和 target instrument 的依赖。
- 减少“平均化 bass 噪音”。

这是最贴近 StemGen 论文改进点的下一步，但不是当前第一优先级。

## 8. 两人分工

### 成员 A：模型与训练负责人

职责：

- 维护 audio-token 主线代码。
- 跑 E5-0、E5-2、E5-5 训练。
- 记录训练配置、checkpoint、best epoch。
- 如果训练报错，优先修：
  - codec/codebook shape mismatch
  - CUDA OOM
  - dataloader/path 问题
  - loss 不下降问题

需要更新：

- `outputs/experiment_log.md`
- `configs/*.yaml`
- 训练命令和日志摘要

每次训练必须记录：

```text
实验编号:
git commit:
配置文件:
数据量:
GPU:
训练时长:
best epoch:
train/val loss:
overall acc:
per-codebook acc:
主要观察:
```

### 成员 B：评估、样本、报告负责人

职责：

- 跑 E5-1、E5-3、E5-4 diagnostics。
- 听音频样本，整理主观观察。
- 保存 mel spectrogram 对比。
- 维护最终报告中的图表和实验表格。
- 对照论文写 related work / method / limitation。

需要更新：

- `docs/NEXT_EXPERIMENT_PLAN_CN.md`
- `outputs/experiment_log.md`
- 最终报告素材目录

每次评估必须记录：

```text
checkpoint:
sample track:
mask ratios:
decoding steps:
temperature/top_k/argmax:
codec reconstruction 是否可听:
partial reconstruction 是否可听:
full generation 是否可听:
最好的 wav 文件路径:
最好的 spectrogram 路径:
```

### 共同职责

- 不提交大文件。
- 不把 checkpoint/audio/data 推到 GitHub。
- 每次重要实验前先 `git status`。
- 每天同步一次当前最好样本和失败原因。
- 报告中诚实区分：
  - token prediction success
  - partial reconstruction success
  - full generation success

## 9. Git 协作规范

建议分支：

```bash
main
experiment/e5-audio-token
report/final-writeup
```

日常流程：

```bash
git pull
git checkout -b experiment/e5-audio-token
# 修改/实验
git add <code/config/docs only>
git commit -m "Run E5 audio-token diagnostics"
git push -u origin experiment/e5-audio-token
```

不要提交：

- `dataset/archive.zip`
- `outputs/**/*.pt`
- `outputs/**/*.wav`
- `outputs/**/*.mid`
- `outputs/**/*.png`
- `outputs/**/*cache*`

如果需要分享音频样本，用 Google Drive/OneDrive 或 GitHub Release，不要直接放进 Git history。

## 10. 最终报告建议结构

1. Introduction
   - context-aware stem generation 的工程意义。
   - StemGen 方法简介。
   - 本项目小规模复现目标。

2. Method
   - EnCodec tokenization。
   - Context/target two-stream token fusion。
   - Target-only masked token training。
   - Iterative mask-predict generation。
   - E5 codebook alignment 改进。

3. Experiments
   - Dataset 与 context 构造。
   - Training configs。
   - Metrics: token acc, per-codebook acc, spectrogram distance, qualitative audio。
   - Codec reconstruction 与 partial reconstruction gates。

4. Results
   - E1-E4 历史结果。
   - E5 规范化结果。
   - Sampling ablation。

5. Discussion
   - 为什么 token acc 不一定转化为可听音频。
   - full-mask generation 为什么比 partial reconstruction 难。
   - 小模型、小数据、少 codebook 的限制。

6. Conclusion
   - 完成 small-scale StemGen-style reproduction。
   - 明确复现瓶颈。
   - 提出后续 CFG / larger data / 32kHz codec 改进方向。

## 11. 本周最小完成目标

在继续租 4090 前，必须完成：

1. 本地/现有 GPU 跑通 `--overfit`。
2. 跑一次 diagnostics。
3. 确认 `codec_reconstruction.wav` 是否可听。
4. 确认 `partial_recon_mask_015.wav` 和 `partial_recon_mask_030.wav` 是否比 full generation 好。
5. 更新 `outputs/experiment_log.md`。

只有这五项过关，才值得进入 4090 一天训练。

## 12. RunPod 夜间训练入口

服务器部署和训练命令见：

```text
docs/RUNPOD_DEPLOYMENT_CN.md
```

推荐 GPU：

```text
首选: RTX 4090 24GB
更稳: L40/L40S/6000 Ada 48GB
```

服务器最短命令链：

```bash
bash scripts/setup_runpod.sh
N_TRACKS=200 bash scripts/prepare_runpod_data.sh
bash scripts/runpod_smoke_test.sh
bash scripts/runpod_train_e5.sh
```
