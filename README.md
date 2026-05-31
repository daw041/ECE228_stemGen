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
scripts/
  train.py               Train the audio-token masked Transformer
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

The current default configs are set up for the next faithful small-scale run:

- 24 kHz EnCodec wrapper
- 2 RVQ codebooks
- 10 second clips
- 8-layer Transformer, 512 hidden size
- variable target-mask ratio from 0.50 to 1.00
- multi-codebook cross-entropy over masked positions

This is intentionally much smaller than the paper's model, which used a
32 kHz EnCodec tokenizer with 4 codebooks and a roughly 250M parameter
LLaMA-style Transformer.

## Setup

```bash
conda create -n stemgen python=3.10
conda activate stemgen
pip install -r requirements.txt
```

The dataset path is configured in `configs/data_config.yaml`.

## Train

```bash
python scripts/train.py \
  --data_config configs/data_config.yaml \
  --model_config configs/model_config.yaml \
  --train_config configs/train_config.yaml \
  --device cuda
```

Checkpoints are written to `outputs/audio_token/e5_2cb/checkpoints/` by default.

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
