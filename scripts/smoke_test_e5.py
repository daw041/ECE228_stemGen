#!/usr/bin/env python
"""Fast smoke test for the E5 audio-token path.

This does not require EnCodec or Slakh data. It checks the config, model,
multi-codebook loss, and iterative generation with synthetic codec tokens.
Use it before launching an overnight server run.
"""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import StemGenModel
from src.trainer import MaskedTokenTrainer


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_cfg):
    mc = model_cfg["model"]
    return StemGenModel(
        vocab_size=mc.get("codec_vocab_size", 1024),
        embedding_dim=mc.get("embedding_dim", 512),
        num_layers=mc.get("num_layers", 8),
        num_heads=mc.get("num_heads", 8),
        feedforward_dim=mc.get("feedforward_dim", 2048),
        dropout=mc.get("dropout", 0.1),
        num_instruments=mc.get("num_instruments", 6),
        instrument_embed_dim=mc.get("instrument_embedding_dim", 64),
        fusion_mode=mc.get("fusion_mode", "concat"),
        num_codebooks=mc.get("num_codebooks", 2),
        max_seq_len=mc.get("max_seq_len", 1024),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--iterations_per_codebook", default="4,2")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    mc = model_cfg["model"]
    tc = train_cfg["training"]

    device = torch.device(args.device)
    vocab_size = int(mc.get("codec_vocab_size", 1024))
    num_codebooks = int(mc.get("num_codebooks", 2))
    seq_len = int(args.seq_len)

    if seq_len > int(mc.get("max_seq_len", 1024)):
        raise ValueError("seq_len exceeds model.max_seq_len")

    model = build_model(model_cfg).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    trainer = MaskedTokenTrainer(
        model=model,
        device=str(device),
        lr=tc.get("learning_rate", 3e-4),
        weight_decay=tc.get("weight_decay", 1e-5),
        mask_ratio=tc.get("mask_ratio", 0.75),
        mask_ratio_min=tc.get("mask_ratio_min"),
        mask_ratio_max=tc.get("mask_ratio_max"),
        codebook_weights=tc.get("codebook_weights"),
    )

    context_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, num_codebooks, seq_len),
        dtype=torch.long,
        device=device,
    )
    target_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, num_codebooks, seq_len),
        dtype=torch.long,
        device=device,
    )
    instrument_idx = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    loss, correct, n_tokens, cb_correct, cb_total = trainer._step(
        context_tokens, target_tokens, instrument_idx
    )
    if not torch.isfinite(loss):
        raise RuntimeError("loss is not finite")
    loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().norm().item())
    if grad_norm <= 0:
        raise RuntimeError("no gradients flowed through the model")

    model.eval()
    steps = [int(x.strip()) for x in args.iterations_per_codebook.split(",") if x.strip()]
    with torch.no_grad():
        generated = model.generate(
            context_tokens=context_tokens[:1],
            instrument_idx=0,
            num_iterations=steps,
            temperature=0.8,
            top_k=50,
        )

    if generated.shape != (1, num_codebooks, seq_len):
        raise RuntimeError(f"bad generated shape: {tuple(generated.shape)}")
    if int((generated == model.mask_token_id).sum().item()) != 0:
        raise RuntimeError("generation still contains mask tokens")

    print("E5 smoke test passed")
    print(f"  device: {device}")
    print(f"  params: {n_params:,}")
    print(f"  synthetic tokens: batch={args.batch_size}, codebooks={num_codebooks}, seq_len={seq_len}")
    print(f"  loss: {float(loss.detach().cpu()):.4f}")
    print(f"  masked tokens: {n_tokens}")
    print(f"  per-codebook masked tokens: {cb_total}")
    print(f"  per-codebook correct: {[int(x) for x in cb_correct]}")
    print(f"  generation shape: {tuple(generated.shape)}")


if __name__ == "__main__":
    main()
