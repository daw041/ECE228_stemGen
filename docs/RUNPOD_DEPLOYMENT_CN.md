# RunPod 服务器部署与训练手册

目标：租用 RunPod 后，尽量做到连接服务器后直接开始 E5 audio-token 训练。

## 1. 推荐 GPU 型号

### 首选：RTX 4090 24GB

推荐用于本项目当前 E5：

- 2 codebooks
- 10 秒 clips
- 28M 参数 Transformer
- AMP 混合精度
- batch size 8 起步

理由：

- 24GB VRAM 对当前 2-codebook E5 足够。
- 性价比通常最好。
- 一天训练预算比较可控。

### 备选：L40 / L40S / RTX 6000 Ada 48GB

如果价格合适，或者希望更稳地跑：

- 更大 batch size
- 4 codebooks
- 768 hidden / 10-12 layers
- 更多 diagnostics 同时保留

就租 48GB VRAM 的 L40/L40S/6000 Ada。

### 不推荐作为第一选择：A100/H100

A100/H100 当然更强，但对当前 28M 小模型不是必要。除非：

- 4090 没货；
- 需要 80GB VRAM；
- 想尝试 4 codebooks + 更大模型；
- 预算充足。

### 实用选择

```text
预算优先: RTX 4090 24GB
稳定优先: L40S 48GB
不差钱/大实验: A100 80GB
```

## 2. RunPod 创建建议

建议选择：

```text
GPU: RTX 4090 24GB
Container: RunPod PyTorch / CUDA 镜像
Container Disk: 80-120GB
Volume Disk: 120-200GB，如果要保留数据和 checkpoint
```

`dataset/archive.zip` 本地约 100GB。不要解压全量 archive。服务器只需要：

1. 上传或挂载 `dataset/archive.zip`。
2. 用 `scripts/extract_audio_subset.py` 抽取一部分 track。

建议第一晚：

```text
N_TRACKS=200
```

如果训练稳定，再扩到：

```text
N_TRACKS=550
```

## 3. 服务器目录假设

以下命令假设项目在：

```bash
/workspace/stemgen
```

如果路径不同，先进入项目根目录即可。

## 4. 第一次连接服务器后的命令

### 4.1 Clone 仓库

```bash
cd /workspace
git clone https://github.com/<your-user>/ECE228_stemGen.git stemgen
cd stemgen
```

如果已经 clone：

```bash
cd /workspace/stemgen
git pull
```

### 4.2 安装依赖并跑 synthetic smoke test

```bash
bash scripts/setup_runpod.sh
```

它会做：

- `pip install -r requirements.txt`
- 检查 `torch/torchaudio/encodec`
- 检查 CUDA/GPU 名称
- 跑 `scripts/smoke_test_e5.py`

成功输出应包含：

```text
E5 smoke test passed
```

## 5. 准备数据

先把本地 `dataset/archive.zip` 上传或挂载到服务器：

```text
/workspace/stemgen/dataset/archive.zip
```

然后只抽一部分：

```bash
N_TRACKS=200 bash scripts/prepare_runpod_data.sh
```

如果 archive 在别的位置：

```bash
ARCHIVE_PATH=/workspace/data/archive.zip N_TRACKS=200 bash scripts/prepare_runpod_data.sh
```

输出目录默认：

```text
dataset/audio_subset
```

该目录已被 `.gitignore` 忽略。

## 6. 真实数据 smoke test

```bash
bash scripts/runpod_smoke_test.sh
```

成功输出应包含：

```text
E5 real-data smoke test passed
tokens: (1, 2, 750)
```

`(1, 2, 750)` 表示：

- batch size 1
- 2 codebooks
- 10 秒音频约 750 codec frames

这一步通过后再开长训练。

如果想进一步确认完整训练循环，可以先跑 1 个 epoch：

```bash
python scripts/train.py \
  --data_config configs/runpod_data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/runpod_train_config.yaml \
  --device cuda \
  --overfit \
  --epochs 1
```

## 7. 正式训练

默认使用：

```text
configs/runpod_data_config.yaml
configs/model_config.yaml
configs/runpod_train_config.yaml
```

启动：

```bash
bash scripts/runpod_train_e5.sh
```

训练日志会写到：

```text
outputs/audio_token/runpod_e5_2cb/logs/
```

checkpoint 会写到：

```text
outputs/audio_token/runpod_e5_2cb/checkpoints/
```

默认重要参数：

```yaml
batch_size: 8
num_epochs: 300
use_amp: true
num_workers: 2
train_n_clips: 2000
val_n_clips: 240
```

如果 RTX 4090 OOM：

1. 打开 `configs/runpod_train_config.yaml`
2. 把 `batch_size: 8` 改成 `batch_size: 4`
3. 重新运行训练

## 8. 断点续训

如果 RunPod 中断，使用：

```bash
RESUME=outputs/audio_token/runpod_e5_2cb/checkpoints/epoch_100.pt \
bash scripts/runpod_train_e5.sh
```

或者从 best checkpoint 续：

```bash
RESUME=outputs/audio_token/runpod_e5_2cb/checkpoints/best.pt \
bash scripts/runpod_train_e5.sh
```

## 9. 训练后诊断

训练结束或中途想看样本：

```bash
bash scripts/runpod_diagnostics_e5.sh
```

输出：

```text
outputs/audio_token/runpod_e5_2cb/diagnostics/
```

重点看：

- `codec_reconstruction.wav`
- `partial_recon_mask_015.wav`
- `partial_recon_mask_030.wav`
- `partial_recon_mask_075.wav`
- `partial_recon_mask_100.wav`
- `full_generation.wav`
- `diagnostic_mels.png`

判断规则：

```text
codec reconstruction 差:
  codec/codebook 设置是瓶颈。

15/30% partial reconstruction 差:
  模型训练或数据管线仍有问题。

15/30% partial reconstruction 好，但 full_generation 差:
  模型有修复能力，但 full-mask generation/sampling/CFG 是瓶颈。
```

## 10. 夜间训练前 checklist

必须依次通过：

```bash
bash scripts/setup_runpod.sh
N_TRACKS=200 bash scripts/prepare_runpod_data.sh
bash scripts/runpod_smoke_test.sh
bash scripts/runpod_train_e5.sh
```

确认：

- `nvidia-smi` 能看到 GPU。
- `E5 smoke test passed`。
- `E5 real-data smoke test passed`。
- `tokens: (1, 2, 750)`。
- 训练日志开始出现 train/val loss。
- checkpoint 目录开始保存文件。

## 11. 服务器结束前要保存什么

如果使用的是临时 Pod，停止前务必下载或同步：

```text
outputs/audio_token/runpod_e5_2cb/checkpoints/best.pt
outputs/audio_token/runpod_e5_2cb/logs/
outputs/audio_token/runpod_e5_2cb/diagnostics/
```

不要依赖临时磁盘长期保存结果。
