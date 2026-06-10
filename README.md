# StemGen Reproduction: Context-Aware Bass Stem Generation

This repository contains a course-scale reproduction of **StemGen: A music
generation model that listens**. The goal is to generate a bass stem from the
rest of the song:

```text
context audio = mixture - bass
target audio  = bass stem
```

The main system follows the audio-token route from StemGen:

```text
waveform audio
  -> EnCodec RVQ tokens
  -> masked-token Transformer
  -> iterative mask-predict generation
  -> generated bass waveform
```

The final report is in [`report/final.pdf`](report/final.pdf).

## What Is Implemented

- 24 kHz EnCodec wrapper for waveform-to-token and token-to-waveform conversion
- 2-codebook RVQ token pipeline with consistent codebook handling
- Slakh-style context/target dataset construction
- Non-autoregressive masked-token Transformer conditioned on instrument id
- Multi-codebook masked cross-entropy training
- Iterative mask-predict generation, codebook by codebook
- Token-cache precomputation for larger H100 runs
- Diagnostics for codec reconstruction, partial-mask reconstruction, and full generation
- Exploratory MIDI baselines kept as secondary experiments

## Main Result

The strongest run is the 1000-track fixed-stride cached audio-token experiment:

| Run | Clips | Sampling | Best Val Loss | Best Val Acc | Best Epoch |
|---|---:|---|---:|---:|---:|
| 550 cached | 26,400 | cached sampled clips | 3.4507 | 0.521 | 56 |
| 1000 stride10 cached | 22,928 | non-overlapping 10 s windows | 3.0279 | 0.572 | 19 |

The 1000-track run reduces best validation loss by about 12.3% compared with
the 550-track cached baseline. Qualitatively, partial-mask reconstruction shows
clearer bass structure than early runs, while full 100% mask generation remains
the main bottleneck.

## Repository Layout

```text
configs/                 Training, data, and model configs
dataset/                 Local datasets only; ignored except README
docs/                    Reproduction notes and debugging guides
outputs/                 Local experiment artifacts; ignored except selected markdown
presentation/            3-minute highlight presentation materials
report/                  Final report source, figures, and compiled PDF
scripts/                 Training, generation, cache, evaluation, and smoke-test scripts
src/                     Core Python package
  codec.py               EnCodec RVQ wrapper
  dataset.py             Slakh context-target datasets and token-cache dataset
  model.py               StemGen-style masked-token Transformer
  trainer.py             Multi-codebook masked-token trainer
  midi/                  Exploratory MIDI branch modules
```

## Setup

```bash
conda create -n stemgen python=3.10
conda activate stemgen
pip install -r requirements.txt
```

The code expects PyTorch and torchaudio with a working CPU or CUDA install. A
GPU is strongly recommended for real EnCodec/token training, but the synthetic
smoke test below can run on CPU.

## Quick Verification

Run this first to check the model, trainer, masking loss, and iterative
generation code without needing Slakh or EnCodec:

```bash
python scripts/smoke_test_e5.py --device cpu
```

Expected outcome: the script builds the model, computes a synthetic
multi-codebook masked loss, and runs a short generation pass without errors.

## Data

Large datasets are not committed. Put extracted Slakh/BabySlakh-style data under
`dataset/` or edit `configs/data_config.yaml`.

For a small local subset from an archive:

```bash
python scripts/extract_audio_subset.py \
  --archive dataset/archive.zip \
  --out_dir dataset/audio_subset \
  --n_tracks 4
```

Then run a real-data smoke test:

```bash
python scripts/smoke_test_e5_data.py \
  --data_root dataset/audio_subset \
  --clip_duration 10.0 \
  --device cuda
```

## Train the Audio-Token Model

```bash
python scripts/train.py \
  --data_config configs/data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/train_config.yaml \
  --device cuda
```

Checkpoints are written to `outputs/audio_token/e5_2cb/checkpoints/` by default.

## Token Cache and Final-Scale Runs

For larger runs, precompute EnCodec tokens first:

```bash
python scripts/precompute_audio_token_cache.py \
  --data_config configs/runpod_1000_data_config.yaml \
  --model_config configs/model_config.yaml \
  --output_dir outputs/audio_token/cache_1000_stride10_e5_2cb \
  --sampling_mode stride \
  --stride_seconds 10.0 \
  --device cuda
```

The final H100 training config is:

```bash
python scripts/train.py \
  --data_config configs/runpod_1000_stride10_token_cache_data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/runpod_1000_stride10_cached_h100_train_config.yaml \
  --device cuda
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the final experiment
configuration and expected metrics.

## Diagnostics and Generation

Before judging full generation, run diagnostics:

```bash
python scripts/diagnose_audio_token.py \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 32,16 \
  --output_dir outputs/audio_token/e5_2cb/diagnostics
```

This saves target audio, codec reconstruction, partial reconstructions,
full generation, and a mel-spectrogram figure.

Generate from a context audio file:

```bash
python scripts/generate.py \
  --context_audio path/to/context.wav \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 32,16 \
  --temperature 0.8 \
  --top_k 50
```

## Notes for Grading

The official StemGen release did not include full training code or released
weights, so this repository is a paper-guided, scaled-down reproduction rather
than a rerun of official code. Local datasets, checkpoints, generated audio, and
large experiment artifacts are intentionally ignored by git.
