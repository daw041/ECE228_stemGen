#!/usr/bin/env python
"""Three-level audio-domain metrics for the StemGen audio-token branch.

For each validation clip we decode three signals and score them against the
ground-truth bass waveform:

  1. codec_reconstruction : EnCodec decode of the *true* target tokens
                            (tokenizer ceiling -- no model involved).
  2. partial_mask_050     : one-pass reconstruction with 50% of the true target
                            tokens revealed (local repair task).
  3. full_generation      : iterative mask-predict from context only
                            (the real StemGen generation task).

Two metrics per level, averaged over the clips:
  - mel L2 distance  (log-mel spectrogram L2, lower is better)
  - onset score      (onset-energy correlation, higher is better, in [-1, 1])

Run on the machine that holds the checkpoint AND the raw Slakh audio (e.g. RunPod):

  python scripts/eval_audio_metrics.py \
    --checkpoint outputs/audio_token/runpod_e5_2cb_1000_stride10_cached_h100/checkpoints/best.pt \
    --data_config configs/runpod_1000_data_config.yaml \
    --model_config configs/model_config.yaml \
    --num_clips 20 --device cuda

The printed table is ready to paste into report Section 4.7.
"""
import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.codec import load_codec
from src.dataset import SlakhContextTargetDataset
from src.model import StemGenModel


def mel_spectrogram(wav: torch.Tensor, sr: int, n_mels: int = 80) -> np.ndarray:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=1024, hop_length=256
    )
    spec = transform(wav.cpu())
    return torch.log(spec + 1e-6).squeeze().numpy()


def spectrogram_distance(ref: torch.Tensor, gen: torch.Tensor, sr: int) -> float:
    """Log-mel L2 distance (lower is better)."""
    ref_mel = mel_spectrogram(ref, sr)
    gen_mel = mel_spectrogram(gen, sr)
    min_frames = min(ref_mel.shape[-1], gen_mel.shape[-1])
    diff = np.mean((ref_mel[..., :min_frames] - gen_mel[..., :min_frames]) ** 2)
    return float(diff)


def onset_alignment_score(ref: torch.Tensor, gen: torch.Tensor, sr: int) -> float:
    """Onset-energy correlation (higher is better)."""
    ref = ref.cpu()
    gen = gen.cpu()
    ref_onset = torch.diff(ref.abs().mean(dim=0), dim=0).abs()
    gen_onset = torch.diff(gen.abs().mean(dim=0), dim=0).abs()
    min_len = min(ref_onset.shape[0], gen_onset.shape[0])
    ref_norm = ref_onset[:min_len] / (ref_onset[:min_len].max() + 1e-8)
    gen_norm = gen_onset[:min_len] / (gen_onset[:min_len].max() + 1e-8)
    corr = torch.corrcoef(torch.stack([ref_norm, gen_norm]))[0, 1]
    return corr.item() if not torch.isnan(corr) else 0.0


@torch.no_grad()
def partial_reconstruct(model, context_tokens, target_tokens, instrument_idx, mask_ratio):
    """One-pass argmax reconstruction with `mask_ratio` of frames masked."""
    bsz, n_cb, seq_len = target_tokens.shape
    device = target_tokens.device
    masked = target_tokens.clone()
    time_mask = torch.rand(bsz, seq_len, device=device) < mask_ratio
    for cb in range(n_cb):
        masked[:, cb, :][time_mask] = model.mask_token_id
    inst = torch.full((bsz,), instrument_idx, dtype=torch.long, device=device)
    logits = model(context_tokens, masked, inst)
    recon = masked.clone()
    for cb in range(n_cb):
        cb_mask = recon[:, cb, :] == model.mask_token_id
        pred = logits[:, cb, :, :].argmax(dim=-1)
        recon[:, cb, :][cb_mask] = pred[cb_mask]
    return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_config", default="configs/runpod_1000_data_config.yaml")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--num_clips", type=int, default=20)
    parser.add_argument("--partial_mask_ratio", type=float, default=0.50)
    parser.add_argument("--iterations_per_codebook", default="32,16")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/audio_metrics")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        mc = yaml.safe_load(f)["model"]

    codec = load_codec(
        num_codebooks=mc.get("num_codebooks", 2),
        bandwidth=mc.get("codec_bandwidth"),
        device=device,
    )

    model = StemGenModel(
        vocab_size=mc.get("codec_vocab_size", 1024),
        embedding_dim=mc.get("embedding_dim", 512),
        num_layers=mc.get("num_layers", 8),
        num_heads=mc.get("num_heads", 8),
        feedforward_dim=mc.get("feedforward_dim", 2048),
        dropout=0.0,
        num_instruments=mc.get("num_instruments", 6),
        instrument_embed_dim=mc.get("instrument_embedding_dim", 64),
        fusion_mode=mc.get("fusion_mode", "concat"),
        num_codebooks=mc.get("num_codebooks", 2),
        max_seq_len=mc.get("max_seq_len", 1024),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) from {args.checkpoint}")

    val_ds = SlakhContextTargetDataset(
        data_root=data_cfg["data"]["data_root"],
        target_instrument=data_cfg["target_instrument"],
        context_mode=data_cfg.get("context_mode", "mixture_minus_target"),
        clip_duration=data_cfg["data"].get("clip_duration", 10.0),
        sample_rate=data_cfg["data"].get("sample_rate", codec.sample_rate),
        split="val",
        max_clips=args.num_clips,
        min_target_rms_db=data_cfg["data"].get("min_target_rms_db"),
        min_target_active_ratio=data_cfg["data"].get("min_target_active_ratio", 0.0),
        target_active_threshold=data_cfg["data"].get("target_active_threshold", 1e-4),
        max_clip_resample_attempts=data_cfg["data"].get("max_clip_resample_attempts", 30),
    )
    n = min(args.num_clips, len(val_ds))
    steps = [int(x) for x in args.iterations_per_codebook.split(",") if x.strip()]
    print(f"Scoring {n} validation clips...\n")

    levels = ["codec_reconstruction", f"partial_mask_{int(args.partial_mask_ratio*100):03d}", "full_generation"]
    acc = {lv: {"mel": [], "onset": []} for lv in levels}

    for i in range(n):
        sample = val_ds[i]
        context = sample["context"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        inst = int(sample["instrument"])

        ctx_tok = codec.encode(context).to(device)
        tgt_tok = codec.encode(target).to(device)

        # level 1: codec ceiling
        wav_codec = codec.decode(tgt_tok)
        # level 2: partial-mask repair
        recon_tok = partial_reconstruct(model, ctx_tok, tgt_tok, inst, args.partial_mask_ratio)
        wav_partial = codec.decode(recon_tok)
        # level 3: full generation from context only
        with torch.no_grad():
            gen_tok = model.generate(
                context_tokens=ctx_tok, instrument_idx=inst,
                num_iterations=steps, temperature=args.temperature, top_k=args.top_k,
            )
        wav_full = codec.decode(gen_tok)

        tgt = target.squeeze(0)
        for lv, wav in zip(levels, [wav_codec, wav_partial, wav_full]):
            w = wav.squeeze(0)
            acc[lv]["mel"].append(spectrogram_distance(tgt, w, codec.sample_rate))
            acc[lv]["onset"].append(onset_alignment_score(tgt, w, codec.sample_rate))
        print(f"  clip {i+1}/{n} done")

    print("\n=== Audio-domain evaluation (mean over {} clips) ===".format(n))
    header = f"{'Level':<22} {'Mel L2 dist (down)':>20} {'Onset score (up)':>18}"
    print(header)
    print("-" * len(header))
    summary = {}
    for lv in levels:
        mel = float(np.mean(acc[lv]["mel"]))
        ons = float(np.mean(acc[lv]["onset"]))
        summary[lv] = {"mel_l2": mel, "onset": ons,
                       "mel_l2_std": float(np.std(acc[lv]["mel"])),
                       "onset_std": float(np.std(acc[lv]["onset"]))}
        print(f"{lv:<22} {mel:>20.4f} {ons:>18.4f}")

    out = os.path.join(args.output_dir, "audio_metrics.json")
    with open(out, "w") as f:
        json.dump({"num_clips": n, "levels": summary,
                   "config": {"partial_mask_ratio": args.partial_mask_ratio,
                              "iterations_per_codebook": steps,
                              "temperature": args.temperature, "top_k": args.top_k}}, f, indent=2)
    print(f"\nSaved {out}")

    # LaTeX-ready rows
    print("\n--- paste into report/final.tex Table (Section 4.7) ---")
    pretty = {"codec_reconstruction": "Codec reconstruction (ceiling)",
              levels[1]: f"Partial-mask reconstruction ({int(args.partial_mask_ratio*100)}\\%)",
              "full_generation": "Full-mask generation"}
    for lv in levels:
        print(f"{pretty[lv]} & {summary[lv]['mel_l2']:.3f} & {summary[lv]['onset']:.3f} \\\\")


if __name__ == "__main__":
    main()
