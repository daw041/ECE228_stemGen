#!/usr/bin/env python
"""Real-data smoke test for E5 using one extracted audio subset batch."""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.codec import load_codec
from src.dataset import SlakhContextTargetDataset
from src.model import StemGenModel
from src.trainer import MaskedTokenTrainer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset/audio_subset")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    parser.add_argument("--clip_duration", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)["model"]
    train_cfg = load_yaml(args.train_config)["training"]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    codec = load_codec(
        num_codebooks=model_cfg.get("num_codebooks", 2),
        bandwidth=model_cfg.get("codec_bandwidth"),
        device=str(device),
    )

    dataset = SlakhContextTargetDataset(
        data_root=args.data_root,
        target_instrument="bass",
        context_mode="mixture_minus_target",
        clip_duration=args.clip_duration,
        sample_rate=codec.sample_rate,
        split="train",
        max_clips=1,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.data_root}")

    sample = dataset[0]
    context = sample["context"].unsqueeze(0)
    target = sample["target"].unsqueeze(0)
    instrument = torch.tensor([sample["instrument"]], dtype=torch.long, device=device)

    context_tokens = codec.encode(context).to(device)
    target_tokens = codec.encode(target).to(device)

    model = StemGenModel(
        vocab_size=model_cfg.get("codec_vocab_size", 1024),
        embedding_dim=model_cfg.get("embedding_dim", 512),
        num_layers=model_cfg.get("num_layers", 8),
        num_heads=model_cfg.get("num_heads", 8),
        feedforward_dim=model_cfg.get("feedforward_dim", 2048),
        dropout=model_cfg.get("dropout", 0.1),
        num_instruments=model_cfg.get("num_instruments", 6),
        instrument_embed_dim=model_cfg.get("instrument_embedding_dim", 64),
        fusion_mode=model_cfg.get("fusion_mode", "concat"),
        num_codebooks=model_cfg.get("num_codebooks", 2),
        max_seq_len=model_cfg.get("max_seq_len", 1024),
    ).to(device)

    trainer = MaskedTokenTrainer(
        model=model,
        device=str(device),
        lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        mask_ratio=train_cfg.get("mask_ratio", 0.75),
        mask_ratio_min=train_cfg.get("mask_ratio_min"),
        mask_ratio_max=train_cfg.get("mask_ratio_max"),
        codebook_weights=train_cfg.get("codebook_weights"),
    )

    loss, correct, n_tokens, cb_correct, cb_total = trainer._step(
        context_tokens, target_tokens, instrument
    )
    loss.backward()

    with torch.no_grad():
        generated = model.generate(
            context_tokens=context_tokens,
            instrument_idx=instrument,
            num_iterations=[2, 1],
            temperature=0.8,
            top_k=50,
        )
        recon_audio = codec.decode(target_tokens).cpu()

    print("E5 real-data smoke test passed")
    print(f"  device: {device}")
    print(f"  track: {sample['track_id']}")
    print(f"  context wav: {tuple(context.shape)}")
    print(f"  target wav: {tuple(target.shape)}")
    print(f"  tokens: {tuple(target_tokens.shape)}")
    print(f"  codec recon wav: {tuple(recon_audio.shape)}")
    print(f"  loss: {float(loss.detach().cpu()):.4f}")
    print(f"  masked tokens: {n_tokens}")
    print(f"  per-codebook masked tokens: {cb_total}")
    print(f"  generated tokens: {tuple(generated.shape)}")


if __name__ == "__main__":
    main()
