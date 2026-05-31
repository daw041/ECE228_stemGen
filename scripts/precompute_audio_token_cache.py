#!/usr/bin/env python
"""Precompute fixed EnCodec-token shards for audio-token training."""
import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.codec import load_codec
from src.dataset import SlakhContextTargetDataset


def _build_dataset(data_cfg, split, n_clips, sample_rate):
    return SlakhContextTargetDataset(
        data_root=data_cfg["data"]["data_root"],
        target_instrument=data_cfg["target_instrument"],
        context_mode=data_cfg.get("context_mode", "mixture_minus_target"),
        clip_duration=data_cfg["data"].get("clip_duration", 4.0),
        sample_rate=data_cfg["data"].get("sample_rate", sample_rate),
        split=split,
        max_clips=n_clips,
        min_target_rms_db=data_cfg["data"].get("min_target_rms_db"),
        min_target_active_ratio=data_cfg["data"].get("min_target_active_ratio", 0.0),
        target_active_threshold=data_cfg["data"].get("target_active_threshold", 1e-4),
        max_clip_resample_attempts=data_cfg["data"].get("max_clip_resample_attempts", 20),
    )


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    return list(value)


@torch.no_grad()
def _write_split(split, dataset, codec, out_dir, batch_size, num_workers, shard_size):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(codec.device).startswith("cuda"),
    )

    pending = []
    shard_idx = 0
    n_items = 0
    for batch in tqdm(loader, desc=f"Precompute {split}"):
        context_tokens = codec.encode(batch["context"]).cpu().short()
        target_tokens = codec.encode(batch["target"]).cpu().short()
        instruments = batch["instrument"].cpu().long()
        batch_size_actual = int(context_tokens.shape[0])
        for i in range(batch_size_actual):
            pending.append({
                "context_tokens": context_tokens[i],
                "target_tokens": target_tokens[i],
                "instrument": instruments[i],
                "track_id": batch["track_id"][i],
                "start_sample": int(batch["start_sample"][i]),
                "target_rms": float(batch["target_rms"][i]),
                "target_active_ratio": float(batch["target_active_ratio"][i]),
            })
            n_items += 1
            if len(pending) >= shard_size:
                _flush_shard(split_dir, shard_idx, pending)
                shard_idx += 1
                pending = []

    if pending:
        _flush_shard(split_dir, shard_idx, pending)
        shard_idx += 1

    return {"count": n_items, "num_shards": shard_idx}


def _flush_shard(split_dir, shard_idx, samples):
    path = os.path.join(split_dir, f"shard_{shard_idx:05d}.pt")
    torch.save({
        "context_tokens": torch.stack([s["context_tokens"] for s in samples], dim=0),
        "target_tokens": torch.stack([s["target_tokens"] for s in samples], dim=0),
        "instrument": torch.stack([s["instrument"] for s in samples], dim=0),
        "track_id": [s["track_id"] for s in samples],
        "start_sample": [s["start_sample"] for s in samples],
        "target_rms": [s["target_rms"] for s in samples],
        "target_active_ratio": [s["target_active_ratio"] for s in samples],
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", default="configs/runpod_550_data_config.yaml")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_clips", type=int, default=None)
    parser.add_argument("--val_clips", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to replace it")
    if os.path.exists(args.output_dir) and args.overwrite:
        for name in ("train", "val"):
            path = os.path.join(args.output_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    mc = model_cfg["model"]
    codec = load_codec(
        num_codebooks=mc.get("num_codebooks", 1),
        bandwidth=mc.get("codec_bandwidth"),
        device=args.device,
    )
    print(
        f"Codec loaded: {codec.sample_rate}Hz, {codec.num_codebooks} codebook(s), "
        f"bandwidth={codec.bandwidth}kbps"
    )

    train_clips = args.train_clips or data_cfg.get("train_n_clips", data_cfg["data"].get("train_n_clips"))
    val_clips = args.val_clips or data_cfg.get("val_n_clips", data_cfg["data"].get("val_n_clips"))
    train_ds = _build_dataset(data_cfg, "train", train_clips, codec.sample_rate)
    val_ds = _build_dataset(data_cfg, "val", val_clips, codec.sample_rate)
    print(f"Precomputing train={len(train_ds)} clips, val={len(val_ds)} clips")

    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "data_config": args.data_config,
        "model_config": args.model_config,
        "seed": args.seed,
        "codec_sample_rate": codec.sample_rate,
        "codec_num_codebooks": codec.num_codebooks,
        "codec_bandwidth": codec.bandwidth,
        "clip_duration": data_cfg["data"].get("clip_duration", 4.0),
        "target_instrument": data_cfg["target_instrument"],
        "context_mode": data_cfg.get("context_mode", "mixture_minus_target"),
        "splits": {},
    }
    manifest["splits"]["train"] = _write_split(
        "train", train_ds, codec, args.output_dir, args.batch_size, args.num_workers, args.shard_size
    )
    manifest["splits"]["val"] = _write_split(
        "val", val_ds, codec, args.output_dir, args.batch_size, args.num_workers, args.shard_size
    )

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
