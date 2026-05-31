#!/usr/bin/env python
"""Iterative mask-predict generation for target stem."""
import os
import sys
import argparse
import yaml
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import StemGenModel
from src.codec import load_codec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_audio", required=True, help="Path to context .wav file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--target_instrument", default="bass")
    parser.add_argument("--output_dir", default="outputs/generated")
    parser.add_argument("--num_iterations", type=int, default=8)
    parser.add_argument(
        "--iterations_per_codebook",
        default=None,
        help="Comma-separated iterative decoding steps, e.g. 32,16 for two codebooks.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--argmax", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # instrument map
    instrument_map = {
        "bass": 0, "drums": 1, "piano": 2,
        "guitar": 3, "strings": 4, "other": 5,
    }
    inst_idx = instrument_map.get(args.target_instrument.lower(), 0)

    # load model
    with open(args.model_config) as f:
        mc = yaml.safe_load(f)["model"]

    # load codec with the same RVQ depth as the model
    codec = load_codec(
        num_codebooks=mc.get("num_codebooks", 1),
        bandwidth=mc.get("codec_bandwidth"),
        device=device,
    )

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
    print(f"Model loaded from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # load and encode context audio
    context_wav, sr = torchaudio.load(args.context_audio)
    if sr != codec.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, codec.sample_rate)
        context_wav = resampler(context_wav)
    if context_wav.shape[0] > 1:
        context_wav = context_wav.mean(dim=0, keepdim=True)
    context_wav = context_wav.unsqueeze(0)

    context_tokens = codec.encode(context_wav).to(device)
    print(f"Context tokens shape: {context_tokens.shape}")

    if args.iterations_per_codebook:
        num_iterations = [int(x.strip()) for x in args.iterations_per_codebook.split(",") if x.strip()]
    else:
        num_iterations = args.num_iterations

    # generate target tokens
    generated_tokens = model.generate(
        context_tokens=context_tokens,
        instrument_idx=inst_idx,
        num_iterations=num_iterations,
        temperature=args.temperature,
        top_k=args.top_k,
        use_argmax=args.argmax,
    )
    print(f"Generated tokens shape: {generated_tokens.shape}")

    # decode to audio
    gen_audio = codec.decode(generated_tokens).cpu()

    # save outputs
    context_path = os.path.join(args.output_dir, "context.wav")
    generated_path = os.path.join(args.output_dir, f"generated_{args.target_instrument}.wav")
    mix_path = os.path.join(args.output_dir, f"mix_with_{args.target_instrument}.wav")

    torchaudio.save(context_path, context_wav.squeeze(0), codec.sample_rate)
    torchaudio.save(generated_path, gen_audio, codec.sample_rate)

    # mix
    ctx_for_mix = context_wav.squeeze(0)
    min_len = min(ctx_for_mix.shape[1], gen_audio.shape[1])
    mixed = ctx_for_mix[:, :min_len] + gen_audio[:, :min_len] * 0.5
    torchaudio.save(mix_path, mixed, codec.sample_rate)

    print(f"\nSaved to {args.output_dir}:")
    print(f"  {context_path}")
    print(f"  {generated_path}")
    print(f"  {mix_path}")


if __name__ == "__main__":
    main()
