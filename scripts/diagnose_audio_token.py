#!/usr/bin/env python
"""Codec and partial-mask diagnostics for the audio-token StemGen branch.

This script answers the first question before another full training run:
can the codec reconstruct the target, and can the model repair partially
masked target tokens when real target context is still visible?
"""
import os
import sys
import argparse
import yaml
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.codec import load_codec
from src.dataset import SlakhContextTargetDataset
from src.model import StemGenModel


def parse_ratios(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_steps(text, fallback):
    if not text:
        return fallback
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def save_mel_plot(wavs, titles, sr, output_path):
    fig, axes = plt.subplots(len(wavs), 1, figsize=(12, 2.4 * len(wavs)))
    if len(wavs) == 1:
        axes = [axes]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    for ax, wav, title in zip(axes, wavs, titles):
        with torch.no_grad():
            spec = torch.log(mel(wav.cpu()) + 1e-6).squeeze().numpy()
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_ylabel("mel")
    axes[-1].set_xlabel("frames")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


@torch.no_grad()
def reconstruct_once(model, context_tokens, target_tokens, instrument_idx, mask_ratio):
    """One-pass masked-token reconstruction for diagnostics."""
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
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--mask_ratios", default="0.15,0.30,0.50,0.75,1.00")
    parser.add_argument("--iterations_per_codebook", default="32,16")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--output_dir", default="outputs/audio_token_diagnostics")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    with open(args.data_config, "r") as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)["model"]

    codec = load_codec(
        num_codebooks=model_cfg.get("num_codebooks", 1),
        bandwidth=model_cfg.get("codec_bandwidth"),
        device=device,
    )

    model = StemGenModel(
        vocab_size=model_cfg.get("codec_vocab_size", 1024),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 4),
        feedforward_dim=model_cfg.get("feedforward_dim", 512),
        dropout=0.0,
        num_instruments=model_cfg.get("num_instruments", 6),
        instrument_embed_dim=model_cfg.get("instrument_embedding_dim", 64),
        fusion_mode=model_cfg.get("fusion_mode", "concat"),
        num_codebooks=model_cfg.get("num_codebooks", 1),
        max_seq_len=model_cfg.get("max_seq_len", 1024),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = SlakhContextTargetDataset(
        data_root=data_cfg["data"]["data_root"],
        target_instrument=data_cfg["target_instrument"],
        context_mode=data_cfg.get("context_mode", "mixture_minus_target"),
        clip_duration=data_cfg["data"].get("clip_duration", 10.0),
        sample_rate=data_cfg["data"].get("sample_rate", codec.sample_rate),
        split="val",
        max_clips=max(args.sample_index + 1, 8),
        min_target_rms_db=data_cfg["data"].get("min_target_rms_db"),
        min_target_active_ratio=data_cfg["data"].get("min_target_active_ratio", 0.0),
        target_active_threshold=data_cfg["data"].get("target_active_threshold", 1e-4),
        max_clip_resample_attempts=data_cfg["data"].get("max_clip_resample_attempts", 20),
    )
    sample = val_ds[args.sample_index]
    context = sample["context"].unsqueeze(0)
    target = sample["target"].unsqueeze(0)
    instrument_idx = int(sample["instrument"])

    context_tokens = codec.encode(context).to(device)
    target_tokens = codec.encode(target).to(device)
    codec_recon = codec.decode(target_tokens).cpu()

    outputs = [
        ("context", context.squeeze(0)),
        ("target", target.squeeze(0)),
        ("codec_reconstruction", codec_recon),
    ]

    torchaudio.save(os.path.join(args.output_dir, "context.wav"), context.squeeze(0), codec.sample_rate)
    torchaudio.save(os.path.join(args.output_dir, "target.wav"), target.squeeze(0), codec.sample_rate)
    torchaudio.save(os.path.join(args.output_dir, "codec_reconstruction.wav"), codec_recon, codec.sample_rate)

    for ratio in parse_ratios(args.mask_ratios):
        recon_tokens = reconstruct_once(model, context_tokens, target_tokens, instrument_idx, ratio)
        recon_audio = codec.decode(recon_tokens).cpu()
        name = f"partial_recon_mask_{int(ratio * 100):03d}"
        torchaudio.save(os.path.join(args.output_dir, f"{name}.wav"), recon_audio, codec.sample_rate)
        outputs.append((name, recon_audio))

    steps = parse_steps(args.iterations_per_codebook, fallback=32)
    gen_tokens = model.generate(
        context_tokens=context_tokens,
        instrument_idx=instrument_idx,
        num_iterations=steps,
        temperature=args.temperature,
        top_k=args.top_k,
        use_argmax=args.argmax,
    )
    generated = codec.decode(gen_tokens).cpu()
    torchaudio.save(os.path.join(args.output_dir, "full_generation.wav"), generated, codec.sample_rate)
    outputs.append(("full_generation", generated))

    save_mel_plot(
        [wav for _, wav in outputs],
        [name for name, _ in outputs],
        codec.sample_rate,
        os.path.join(args.output_dir, "diagnostic_mels.png"),
    )

    print(f"Saved diagnostics to {args.output_dir}")
    print(f"Codec: {codec.sample_rate}Hz, {codec.num_codebooks} codebooks, bandwidth={codec.bandwidth}kbps")
    print(f"Sample: {sample['track_id']}, target tokens={tuple(target_tokens.shape)}")


if __name__ == "__main__":
    main()
