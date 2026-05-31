#!/usr/bin/env python
"""Evaluation script for the StemGen model."""
import os
import sys
import argparse
import yaml
import torch
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import StemGenModel
from src.codec import load_codec
from src.dataset import SlakhContextTargetDataset


def mel_spectrogram(wav: torch.Tensor, sr: int, n_mels: int = 80) -> np.ndarray:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=1024, hop_length=256
    )
    spec = transform(wav)
    return torch.log(spec + 1e-6).squeeze().numpy()


def onset_alignment_score(ref: torch.Tensor, gen: torch.Tensor, sr: int) -> float:
    """Simple onset correlation score (higher is better)."""
    ref_onset = torch.diff(ref.abs().mean(dim=0), dim=0).abs()
    gen_onset = torch.diff(gen.abs().mean(dim=0), dim=0).abs()
    min_len = min(ref_onset.shape[0], gen_onset.shape[0])
    ref_norm = ref_onset[:min_len] / (ref_onset[:min_len].max() + 1e-8)
    gen_norm = gen_onset[:min_len] / (gen_onset[:min_len].max() + 1e-8)
    corr = torch.corrcoef(torch.stack([ref_norm, gen_norm]))[0, 1]
    return corr.item() if not torch.isnan(corr) else 0.0


def spectrogram_distance(ref: torch.Tensor, gen: torch.Tensor, sr: int) -> float:
    """Mel-spectrogram L2 distance (lower is better)."""
    ref_mel = mel_spectrogram(ref, sr)
    gen_mel = mel_spectrogram(gen, sr)
    min_frames = min(ref_mel.shape[1], gen_mel.shape[1])
    diff = np.mean((ref_mel[:, :min_frames] - gen_mel[:, :min_frames]) ** 2)
    return float(diff)


def plot_spectrograms(
    context: torch.Tensor,
    target: torch.Tensor,
    generated: torch.Tensor,
    sr: int,
    output_path: str,
):
    """Plot spectrograms side by side."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    for ax, wav, title in zip(
        axes,
        [context, target, generated, target - generated],
        ["Context", "Target (ground truth)", "Generated", "Difference (target - generated)"],
    ):
        mel = librosa.feature.melspectrogram(y=wav.squeeze().numpy(), sr=sr, n_mels=80)
        log_mel = np.log(mel + 1e-6)
        im = ax.imshow(log_mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax)

    axes[-1].set_xlabel("Time frames")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--output_dir", default="outputs/evaluation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        mc = yaml.safe_load(f)["model"]

    codec = load_codec(
        num_codebooks=mc.get("num_codebooks", 1),
        bandwidth=mc.get("codec_bandwidth"),
        device=device,
    )
    print("Codec loaded")

    # load model
    model = StemGenModel(
        vocab_size=mc.get("codec_vocab_size", 1024),
        embedding_dim=mc.get("embedding_dim", 256),
        num_layers=mc.get("num_layers", 4),
        num_heads=mc.get("num_heads", 4),
        feedforward_dim=mc.get("feedforward_dim", 512),
        dropout=0.0,
        num_instruments=mc.get("num_instruments", 6),
        instrument_embed_dim=mc.get("instrument_embedding_dim", 64),
        fusion_mode=mc.get("fusion_mode", "concat"),
        num_codebooks=mc.get("num_codebooks", 1),
        max_seq_len=mc.get("max_seq_len", 1024),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded: epoch {ckpt.get('epoch', '?')}")

    # evaluation dataset
    val_ds = SlakhContextTargetDataset(
        data_root=data_cfg["data"]["data_root"],
        target_instrument=data_cfg["target_instrument"],
        clip_duration=data_cfg["data"].get("clip_duration", 4.0),
        sample_rate=data_cfg["data"].get("sample_rate", 24000),
        split="val",
        max_clips=10,
    )

    results = {
        "spectrogram_distances": [],
        "onset_scores": [],
    }

    instrument_map = {
        "bass": 0, "drums": 1, "piano": 2,
        "guitar": 3, "strings": 4, "other": 5,
    }
    inst_idx = instrument_map.get(data_cfg["target_instrument"].lower(), 0)

    print(f"\nEvaluating on {len(val_ds)} clips...")
    for i in range(min(5, len(val_ds))):
        sample = val_ds[i]
        context_wav = sample["context"].unsqueeze(0)
        target_wav = sample["target"].unsqueeze(0)

        context_tokens = codec.encode(context_wav).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                context_tokens=context_tokens,
                instrument_idx=inst_idx,
                num_iterations=8,
                temperature=1.0,
            )

        gen_wav = codec.decode(gen_tokens).cpu()

        spec_dist = spectrogram_distance(target_wav.squeeze(0), gen_wav.squeeze(0), codec.sample_rate)
        onset_score = onset_alignment_score(target_wav.squeeze(0), gen_wav.squeeze(0), codec.sample_rate)

        results["spectrogram_distances"].append(spec_dist)
        results["onset_scores"].append(onset_score)

        # save audio samples for first few
        if i < 3:
            torchaudio.save(
                os.path.join(args.output_dir, f"sample_{i}_context.wav"),
                context_wav.squeeze(0), codec.sample_rate
            )
            torchaudio.save(
                os.path.join(args.output_dir, f"sample_{i}_target.wav"),
                target_wav.squeeze(0), codec.sample_rate
            )
            torchaudio.save(
                os.path.join(args.output_dir, f"sample_{i}_generated.wav"),
                gen_wav, codec.sample_rate
            )

        print(f"  Sample {i}: spec_dist={spec_dist:.4f}, onset_score={onset_score:.3f}")

    print(f"\n=== Evaluation Summary ===")
    print(f"Avg Spectrogram Distance: {np.mean(results['spectrogram_distances']):.4f}")
    print(f"Avg Onset Alignment Score: {np.mean(results['onset_scores']):.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
