# 最新 Audio-Token 实验结果整理

更新时间：2026年6月4日

## 项目定位

本项目复现 StemGen 的核心思想：给定上下文音频，生成指定目标乐器的 stem。官方 StemGen 仓库没有提供完整训练代码或预训练权重，因此本项目采用 paper-guided scaled-down reproduction，而不是直接重跑官方代码。

当前主线是 audio-token 路线：

```text
mixture - target stem 作为 context
target stem -> EnCodec RVQ tokens
context tokens + masked target tokens + instrument condition
-> non-autoregressive masked-token Transformer
-> iterative mask-predict decoding
-> generated target stem audio
```

## 当前模型与数据设置

- 任务：bass stem generation
- 数据：Slakh 风格多轨音频子集
- context：mixture minus target
- target：bass stem
- codec：24 kHz EnCodec wrapper
- bandwidth：1.5 kbps
- RVQ codebooks：2
- clip length：10 seconds
- 模型：masked-token Transformer
- 训练目标：multi-codebook masked cross entropy
- mask ratio：0.50 到 1.00 的 variable masking
- 训练平台：RunPod H100 80GB

## 实验时间线

### 1. 早期 audio-token 结果

早期 audio-token 生成音频接近噪声，说明仅有 pipeline 能跑通还不等于复现有效。后续主要修正方向是：

- 对齐 EnCodec codebook 处理方式；
- 避免静音或弱 target 片段污染训练；
- 将随机在线采样改为 token cache，减少 GPU 时间浪费；
- 将评价重点从 full-mask generation 提前拆成 codec reconstruction、partial-mask reconstruction、full generation 三个层次。

### 2. 550-track cached run

该实验是第一个稳定可用的主结果。

| 项目 | 数值 |
|---|---:|
| tracks | 550 |
| train clips | 22,000 |
| val clips | 4,400 |
| train shards | 86 |
| val shards | 18 |
| best epoch | 56 |
| best val loss | 3.4507 |
| best val acc | 0.521 |
| best val codebook acc | [0.6228, 0.4193] |
| early stop epoch | 68 |

对应文件：

- `outputs/audio_token/cache_550_26400_e5_2cb/manifest.json`
- `outputs/audio_token/runpod_e5_2cb_550_cached_h100/checkpoints/best.pt`
- `outputs/audio_token/runpod_e5_2cb_550_cached_h100/logs/train_20260531_231446.log`

主观结果：生成音频不再是纯噪声，频谱上能看到部分结构；但在 100% mask 的 full generation 下仍容易退化，partial-mask reconstruction 更稳定。

### 3. 1000-track stride10 cached run

该实验是当前最新、最适合用于 presentation 和 final report 的结果。为了减少高度重叠片段带来的数据冗余，1000-track 实验采用固定 stride=10s，即基本非重叠的 10 秒窗口，并过滤 target 低能量片段。

| 项目 | Train | Val |
|---|---:|---:|
| candidates seen | 20,421 | 5,094 |
| inactive filtered | 2,060 | 527 |
| final clips | 18,361 | 4,567 |
| shards | 72 | 18 |

训练结果：

| 项目 | 数值 |
|---|---:|
| tracks | 1,000 |
| sampling | fixed stride, 10 seconds |
| warm start | 550-track best checkpoint |
| best epoch | 19 |
| best val loss | 3.0279 |
| best val acc | 0.572 |
| best val codebook acc | [0.6727, 0.4715] |
| early stop epoch | 31 |
| final epoch val loss | 3.2979 |
| final epoch val acc | 0.543 |

对应文件：

- `outputs/audio_token/cache_1000_stride10_e5_2cb/manifest.json`
- `outputs/audio_token/runpod_e5_2cb_1000_stride10_cached_h100/checkpoints/best.pt`
- `outputs/audio_token/runpod_e5_2cb_1000_stride10_cached_h100/logs/train_20260601_012627.log`

## 关键对比

| Run | Clips | Sampling | Best Val Loss | Best Val Acc | Best Epoch |
|---|---:|---|---:|---:|---:|
| 550 cached | 26,400 | cached sampled clips | 3.4507 | 0.521 | 56 |
| 1000 stride10 cached | 22,928 | non-overlapping stride windows | 3.0279 | 0.572 | 19 |

相对 550 run，1000 stride10 run 的 best validation loss 从 3.4507 降到 3.0279，约降低 12.3%；validation accuracy 从 0.521 提升到 0.572。

这里的重点不是“更多 clips 一定更好”，而是：

- 更大 track 覆盖提高了音乐多样性；
- stride10 降低了相邻 clip 的重复度；
- 静音过滤减少了无意义高准确率片段；
- token cache 让 H100 训练时间主要花在模型训练而不是重复 codec encode 上；
- 550 checkpoint warm start 使 1000 run 很快达到较好验证结果。

## 当前结论

1. audio-token 路线已经从“纯噪声”推进到“可听见结构、频谱部分合理”的有效 baseline。
2. partial-mask reconstruction 比 full-mask generation 稳定，说明模型学到了 token-level local reconstruction，但 unconditional-style target synthesis 仍是主要瓶颈。
3. 100% mask 生成仍容易变噪声，因此 presentation 和 report 里应诚实区分：
   - codec reconstruction：验证 tokenizer；
   - partial-mask reconstruction：验证模型能否利用已有 target 信息；
   - full-mask generation：验证真正从 context 生成 target 的能力。
4. 最新 1000-track stride10 结果是目前最强的量化结果，适合放入 3-minute highlight talk。

## 下一步建议

- 在 1000 best checkpoint 上生成一组统一诊断样本：codec reconstruction、15%/50% partial mask、100% full generation、mel spectrogram。
- 将 decoding 从单一 temperature/top-k 调参扩展到 mask schedule 和 confidence ranking 对比。
- 加入 classifier-free guidance 或 condition dropout，让模型更明确依赖 context 和 instrument condition。
- 报告里把“失败点”写成科学发现：full-mask generation 比 partial reconstruction 更难，当前小模型/低码率/2-codebook 设置下仍不足以完全复现论文级效果。
