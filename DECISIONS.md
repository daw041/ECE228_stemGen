# 关键设计决策

## 1. 音频编码

**决策**: 使用 EnCodec 24kHz 模型，Slakh 音频从 44.1kHz 重采样到 24kHz

**理由**: 
- EnCodec 24kHz checkpoint 已下载，无需额外下载 32kHz 版本
- Bass 基频 (40-400Hz) 远低于 24kHz 奈奎斯特极限 (12kHz)，不丢失信息
- torchaudio 一行代码解决重采样

## 2. 多 Codebook 策略

**决策**: 从单 codebook 开始，逐步增加到 2、4 codebooks

**理由**:
- 单 codebook (coarse): baseline，acc 上限约 44%
- 2 codebook (coarse + 1 fine): 显著改善，val acc 从 28% → 51.8%
- 4 codebook: StemGen 原论文配置，生成质量最高但数据需求大
- 逐步验证避免在错误方向浪费计算

**当前状态**: 1cb baseline done → 2cb 最佳结果 51.8% → 4cb 正在验证

## 3. Clip 时长

**决策**: 从 4s 切换到 10s

**理由**:
- StemGen 论文使用 20s clips
- 4s 只能看到 ~150 frames，bass 线条常被截断
- 10s (~750 frames) 是 GPU 内存和时序完整性的折中
- 20s 会超过 RTX 3050 4GB 显存极限

## 4. 模型规模

**决策**: 3M 参数 (256 dim, 4 layers, 8 heads)

**理由**:
- 762K 模型容量不够 (357 clips 时 train acc 仅 18%)
- 3M 模型在 2773 clips 上 train acc 50%+，val acc 51.8%
- 数据集从小到大逐步验证，避免死记硬背
- 当前数据显示模型仍在学习 (best at epoch 353/400)

## 5. 数据集策略

**决策**: 从 archive.zip 提取前 550 轨 bass stem + mix

**理由**:
- 完整 Slakh2100 压缩包 100GB，本地空间不足
- 提取 metadata 先找到 bass，然后只提取必要文件
- 每轨 ~14MB (mix.flac ~10MB + bass stem ~3-4MB)
- 550 轨 = ~7.7GB，预留 50 轨 buffer
- 之后删除 archive.zip 腾出 98GB

## 6. 训练掩码策略

**决策**: Target-only masking (context 不 mask)

**理由**:
- 与 StemGen 论文一致
- Context tokens 必须全部可见，模型才能从音乐上下文中学习
- 只在 target stem tokens 上计算 loss
- 多 codebook 时所有 codebook 级别使用相同 mask pattern

## 7. 验证集拆分

**决策**: 15% 随机 split 作为验证集，使用 early stopping (patience=50)

**理由**:
- 需要验证集检测过拟合
- 之前 bug: val set 为空导致无法 early stop → 已修复
- Patience 从 30 增加到 50 以适应大数据量时的慢收敛

## 8. 前导静音过滤

**决策**: 检测 clip 前 0.5s 的 bass 能量，<10% 则跳过

**理由**:
- Slakh 部分轨道 bass 在曲中才进入
- 前导静音 clip 对训练无意义 (模型学不到 bass 模式)
- 避免浪费计算资源

## 9. 数据集大小控制

**决策**: 添加 MAX_TRACKS 参数限制使用的轨道数

**理由**:
- 新增 codebook 数量时从小数据开始验证
- 避免重新提取数据集 (磁盘 I/O 开销)
- 便于快速迭代实验

## 10. Activity Prediction Head

**决策**: 在 Transformer output 上加一个小型 activity head（2 层 MLP: 256→64→1），预测每个 token frame 是否有 bass 活动

**理由**:
- 当前生成瓶颈：模型输出连续低频 texture，缺少节奏/休止
- 仅靠 token CE loss 不足以让模型学会「何时休止」
- 类似 StemGen 的条件 dropout + CFG 思路，但更轻量
- BCE loss: frame RMS > -45dB peak → active=1, else 0
- 总损失: total_loss = token_CE + 0.2 * activity_BCE

**当前状态**: 已实现，等待 E4 训练验证

## 11. Causal-Biased Iterative Decoding

**决策**: 在 mask-predict 迭代解码中加入时间位置偏置，前部时间点优先采样/固定

**理由**:
- Debug guide 推荐，更接近 StemGen 论文的 causal-biased decoding
- 纯 confidence 选择容易产生 "average token everywhere" 问题
- 引入前向后弱的时间偏置（time_bias = 0.1→0.0），打破对称性
- combined_score = confidence + causal_bias_weight * time_bias[t]

**当前状态**: 已实现，默认 causal_bias_weight=0.1，8 轮迭代

## 12. Activity Gating During Inference

**决策**: 用 activity head 输出概率对生成波形做帧级增益控制

**理由**:
- 即使训练好，生成时模型仍可能输出连续低频
- 用 sigmoid(activity_logits) × 生成波形，直接抑制非活跃帧
- 80ms 平滑窗口避免增益突变带来的 click artifact

**当前状态**: 已实现，保存 gated/ungated 两版 WAV 对比

## 13. Debug Evaluation Pipeline

**决策**: 每次训练后自动运行 4 层诊断

**理由**:
- P1: 保存 codec reconstruction 对比（target→encode→decode），排除 codec 本身问题
- P2: Partial-mask reconstruction 评估（mask_ratio=0.15/0.30/0.50/0.75），区分重建 vs 生成能力
- P3: 数据集 activity ratio 分布直方图，检测训练数据偏差
- P4: Activity gated/ungated 两版生成对比

**当前状态**: 已实现，集成在训练脚本 Step 6 评估阶段
