# Reproducibility Guide

This guide summarizes how to verify the repository and reproduce the main
audio-token experiment setup used in the final report.

## 1. Environment

```bash
conda create -n stemgen python=3.10
conda activate stemgen
pip install -r requirements.txt
```

For GPU runs, install a PyTorch/torchaudio build compatible with the local CUDA
driver before installing the remaining requirements.

## 2. Fast Code Verification

This test does not require EnCodec weights or Slakh data. It verifies config
loading, model construction, multi-codebook masked loss, and iterative
generation on synthetic tokens.

```bash
python scripts/smoke_test_e5.py --device cpu
```

Use `--device cuda` if CUDA is available.

## 3. Small Real-Data Verification

Place a Slakh/BabySlakh-style archive at `dataset/archive.zip`, then extract a
small subset:

```bash
python scripts/extract_audio_subset.py \
  --archive dataset/archive.zip \
  --out_dir dataset/audio_subset \
  --n_tracks 4
```

Run the data smoke test:

```bash
python scripts/smoke_test_e5_data.py \
  --data_root dataset/audio_subset \
  --clip_duration 10.0 \
  --device cuda
```

This checks audio loading, context/target construction, EnCodec tokenization,
model forward pass, and a small diagnostic generation path.

## 4. Main Model Configuration

The final audio-token model uses:

| Component | Setting |
|---|---|
| Target instrument | bass |
| Context | mixture minus bass |
| Sample rate | 24 kHz |
| Clip length | 10 seconds |
| Codec | EnCodec 24 kHz |
| Bandwidth | 1.5 kbps |
| RVQ codebooks | 2 |
| Token vocabulary | 1024 plus mask token |
| Transformer | 8 layers, hidden size 512, 8 heads |
| Loss | multi-codebook masked cross entropy |
| Mask ratio | uniformly sampled from 0.50 to 1.00 |

Relevant configs:

- `configs/model_config.yaml`
- `configs/runpod_1000_stride10_token_cache_data_config.yaml`
- `configs/runpod_1000_stride10_cached_h100_train_config.yaml`

## 5. Token Cache Pipeline

The larger experiments train from cached EnCodec tokens instead of encoding
waveforms every epoch. The final cache used fixed 10 second windows and filtered
inactive bass clips.

```bash
python scripts/precompute_audio_token_cache.py \
  --data_config configs/runpod_1000_data_config.yaml \
  --model_config configs/model_config.yaml \
  --output_dir outputs/audio_token/cache_1000_stride10_e5_2cb \
  --sampling_mode stride \
  --stride_seconds 10.0 \
  --device cuda
```

Expected final cache statistics:

| Split | Candidates Seen | Inactive Filtered | Final Clips | Shards |
|---|---:|---:|---:|---:|
| train | 20,421 | 2,060 | 18,361 | 72 |
| val | 5,094 | 527 | 4,567 | 18 |

The generated cache is a local artifact and is not committed to git.

## 6. Final Training Run

Train from the fixed-stride token cache:

```bash
python scripts/train.py \
  --data_config configs/runpod_1000_stride10_token_cache_data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/runpod_1000_stride10_cached_h100_train_config.yaml \
  --device cuda
```

The reported final run warm-started from the 550-track cached checkpoint and ran
on a RunPod H100. The best checkpoint reached:

| Run | Best Epoch | Best Val Loss | Best Val Acc | Best Codebook Acc |
|---|---:|---:|---:|---|
| 1000-track stride10 cached | 19 | 3.0279 | 0.572 | [0.6727, 0.4715] |

The run early-stopped at epoch 31 with final validation loss 3.2979 and
validation accuracy 0.543.

## 7. Diagnostics

Run diagnostics before judging full generation:

```bash
python scripts/diagnose_audio_token.py \
  --checkpoint outputs/audio_token/runpod_e5_2cb_1000_stride10_cached_h100/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 32,16 \
  --output_dir outputs/audio_token/runpod_e5_2cb_1000_stride10_cached_h100/diagnostics
```

Interpretation:

- Good codec reconstruction means the EnCodec setting is usable.
- Good partial-mask reconstruction means the Transformer learned token-level
  conditional reconstruction.
- Weak full-mask generation means the remaining bottleneck is true
  context-to-target synthesis rather than the tokenizer alone.

## 8. Local Artifacts

The following are intentionally not committed:

- Slakh/BabySlakh datasets
- EnCodec token caches
- Checkpoints
- Generated WAV files
- TensorBoard logs
- Large local experiment bundles

The final report and selected figures are committed under `report/`.
