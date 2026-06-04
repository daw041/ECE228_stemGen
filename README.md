# StemGen Reproduction Project

Small-scale reproduction of **StemGen: A music generation model that listens**
for a course project.  The main track is an audio-token, context-aware stem
generation pipeline:

```text
context audio = mixture - target stem
target stem audio
        -> EnCodec RVQ tokens
        -> non-autoregressive masked-token Transformer
        -> iterative mask-predict decoding
        -> generated target stem audio
```

The repository also contains exploratory MIDI experiments, but the primary
reproduction path is the audio-token branch.

## Repository Layout

```text
configs/                 Audio-token training/model/data configs
docs/                    Notes, debugging guides, reproduction references
  LATEST_AUDIO_TOKEN_RESULTS_CN.md
                          Latest Chinese summary of audio-token experiments
  NEXT_EXPERIMENT_PLAN_CN.md
                          Chinese collaboration plan for the next experiments
  RUNPOD_DEPLOYMENT_CN.md
                          RunPod setup, data prep, training, diagnostics guide
presentation/            3-minute highlight slides and speaker script
scripts/
  train.py               Train the audio-token masked Transformer
  smoke_test_e5.py       Fast config/model/trainer/generation sanity test
  smoke_test_e5_data.py  Real extracted-audio sanity test
  extract_audio_subset.py
                          Extract a tiny audio-token subset from archive.zip
  generate.py            Full-mask target-stem generation
  diagnose_audio_token.py Codec + partial-mask reconstruction diagnostics
  evaluate.py            Basic audio-token evaluation
  train_midi*.py         Exploratory MIDI branch scripts
src/
  codec.py               EnCodec wrapper with aligned RVQ codebook handling
  dataset.py             Slakh context-target dataset
  model.py               StemGen-style masked-token Transformer
  trainer.py             Multi-codebook masked-token trainer
outputs/                 Local experiment artifacts, ignored except markdown
dataset/                 Local datasets, ignored
```

## Current Audio-Token Baseline

The current default configs are set up for a faithful small-scale run:

- 24 kHz EnCodec wrapper
- 2 RVQ codebooks
- 10 second clips
- 8-layer Transformer, 512 hidden size
- variable target-mask ratio from 0.50 to 1.00
- multi-codebook cross-entropy over masked positions

This is intentionally much smaller than the paper's model, which used a
32 kHz EnCodec tokenizer with 4 codebooks and a roughly 250M parameter
LLaMA-style Transformer.

## Latest Results

The strongest current run is the 1000-track fixed-stride cached experiment:

| Run | Clips | Sampling | Best Val Loss | Best Val Acc | Best Epoch |
|---|---:|---|---:|---:|---:|
| 550 cached | 26,400 | cached sampled clips | 3.4507 | 0.521 | 56 |
| 1000 stride10 cached | 22,928 | non-overlapping 10s windows | 3.0279 | 0.572 | 19 |

Compared with the 550-track run, the 1000-track stride10 run reduces best
validation loss by about 12.3%.  Qualitatively, partial-mask reconstruction is
no longer pure noise and shows audible/spectral structure, while full 100% mask
generation remains the main bottleneck.

Detailed Chinese notes:

[docs/LATEST_AUDIO_TOKEN_RESULTS_CN.md](docs/LATEST_AUDIO_TOKEN_RESULTS_CN.md)

3-minute highlight presentation materials:

[presentation/stemgen_highlight_3min.pptx](presentation/stemgen_highlight_3min.pptx),
[presentation/highlight_slides.md](presentation/highlight_slides.md), and
[presentation/speaker_script_cn.md](presentation/speaker_script_cn.md)

## Setup

```bash
conda create -n stemgen python=3.10
conda activate stemgen
pip install -r requirements.txt
```

The dataset path is configured in `configs/data_config.yaml`.

For local testing without unpacking the full archive:

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

## Train

```bash
python scripts/train.py \
  --data_config configs/data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/train_config.yaml \
  --device cuda
```

Checkpoints are written to `outputs/audio_token/e5_2cb/checkpoints/` by default.

## RunPod

For overnight server training, see:

[docs/RUNPOD_DEPLOYMENT_CN.md](docs/RUNPOD_DEPLOYMENT_CN.md)

Quick server path:

```bash
bash scripts/setup_runpod.sh
N_TRACKS=200 bash scripts/prepare_runpod_data.sh
bash scripts/runpod_smoke_test.sh
bash scripts/runpod_train_e5.sh
```

On RunPod PyTorch images, `setup_runpod.sh` creates `.venv` with
`--system-site-packages` by default, so the image's preinstalled Torch/CUDA
stack is reused while project-only packages such as `encodec` are installed
inside the repository.

## Diagnose Before Full Generation

Run diagnostics before judging full generation quality:

```bash
python scripts/diagnose_audio_token.py \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 32,16 \
  --output_dir outputs/audio_token/e5_2cb/diagnostics
```

This saves:

- `target.wav`
- `codec_reconstruction.wav`
- `partial_recon_mask_015.wav` through `partial_recon_mask_100.wav`
- `full_generation.wav`
- `diagnostic_mels.png`

Interpretation:

- If codec reconstruction is bad, the tokenizer/codebook setting is the bottleneck.
- If low-mask partial reconstruction is bad, training or model alignment is broken.
- If partial reconstruction is good but full generation is bad, improve decoding and CFG.

## Generate

```bash
python scripts/generate.py \
  --context_audio path/to/context.wav \
  --checkpoint outputs/audio_token/e5_2cb/checkpoints/best.pt \
  --device cuda \
  --iterations_per_codebook 32,16 \
  --temperature 0.8 \
  --top_k 50
```

## Notes

The official StemGen repository currently contains examples and demo assets,
not training code or released weights.  This project is therefore a
paper-guided, scaled-down reproduction rather than an official-code rerun.
