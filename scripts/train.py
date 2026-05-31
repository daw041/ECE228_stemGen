#!/usr/bin/env python
"""Train the StemGen masked-token Transformer."""
import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import StemGenModel
from src.trainer import MaskedTokenTrainer
from src.codec import load_codec
from src.dataset import SlakhContextTargetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", default="configs/data_config.yaml")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    parser.add_argument("--overfit", action="store_true", help="Overfit mode: 5-10 clips")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # load configs
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    device = args.device
    print(f"Device: {device}")

    mc = model_cfg["model"]

    # load codec with the same RVQ depth as the model
    codec = load_codec(
        num_codebooks=mc.get("num_codebooks", 1),
        bandwidth=mc.get("codec_bandwidth"),
        device=device,
    )
    print(
        f"Codec loaded: {codec.sample_rate}Hz, {codec.num_codebooks} codebook(s), "
        f"bandwidth={codec.bandwidth}kbps"
    )

    # build datasets
    n_train = (
        data_cfg.get("overfit_n_clips", data_cfg["data"].get("overfit_n_clips", 8))
        if args.overfit
        else data_cfg.get("train_n_clips", data_cfg["data"].get("train_n_clips", 100))
    )
    n_val = (
        5 if args.overfit
        else data_cfg.get("val_n_clips", data_cfg["data"].get("val_n_clips", 20))
    )

    train_ds = SlakhContextTargetDataset(
        data_root=data_cfg["data"]["data_root"],
        target_instrument=data_cfg["target_instrument"],
        context_mode=data_cfg.get("context_mode", "mixture_minus_target"),
        clip_duration=data_cfg["data"].get("clip_duration", 4.0),
        sample_rate=data_cfg["data"].get("sample_rate", codec.sample_rate),
        split="train",
        max_clips=n_train,
    )
    val_ds = SlakhContextTargetDataset(
        data_root=data_cfg["data"]["data_root"],
        target_instrument=data_cfg["target_instrument"],
        context_mode=data_cfg.get("context_mode", "mixture_minus_target"),
        clip_duration=data_cfg["data"].get("clip_duration", 4.0),
        sample_rate=data_cfg["data"].get("sample_rate", codec.sample_rate),
        split="val",
        max_clips=n_val,
    )

    print(f"Train clips: {len(train_ds)}, Val clips: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["training"].get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg["training"].get("num_workers", 0),
        pin_memory=str(device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["training"].get("batch_size", 8),
        shuffle=False,
        num_workers=train_cfg["training"].get("num_workers", 0),
        pin_memory=str(device).startswith("cuda"),
    )

    # build model
    model = StemGenModel(
        vocab_size=mc.get("codec_vocab_size", 1024),
        embedding_dim=mc.get("embedding_dim", 256),
        num_layers=mc.get("num_layers", 4),
        num_heads=mc.get("num_heads", 4),
        feedforward_dim=mc.get("feedforward_dim", 512),
        dropout=mc.get("dropout", 0.1),
        num_instruments=mc.get("num_instruments", 6),
        instrument_embed_dim=mc.get("instrument_embedding_dim", 64),
        fusion_mode=mc.get("fusion_mode", "concat"),
        num_codebooks=mc.get("num_codebooks", 1),
        max_seq_len=mc.get("max_seq_len", 1024),
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # trainer
    tc = train_cfg["training"]
    trainer = MaskedTokenTrainer(
        model=model,
        device=device,
        lr=tc.get("learning_rate", 1e-4),
        weight_decay=tc.get("weight_decay", 1e-5),
        mask_ratio=tc.get("mask_ratio", 0.75),
        mask_ratio_min=tc.get("mask_ratio_min"),
        mask_ratio_max=tc.get("mask_ratio_max"),
        codebook_weights=tc.get("codebook_weights"),
        use_amp=tc.get("use_amp", False),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
    )

    # logging
    writer = SummaryWriter(log_dir=tc.get("tensorboard_dir", "outputs/tensorboard"))
    checkpoint_dir = tc.get("checkpoint_dir", "outputs/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # training loop
    num_epochs = args.epochs or (tc.get("num_epochs", 50) if not args.overfit else 200)
    mode_str = "OVERFIT" if args.overfit else "TRAINING"
    print(f"\n{mode_str}: {num_epochs} epochs")

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    early_stop_patience = args.early_stop_patience
    if early_stop_patience is None:
        early_stop_patience = tc.get("early_stopping_patience", tc.get("early_stop_patience", 0))
    early_stop_patience = int(early_stop_patience or 0)
    early_stop_min_delta = float(tc.get("early_stopping_min_delta", tc.get("early_stop_min_delta", 0.0)))
    if early_stop_patience > 0:
        print(
            f"Early stopping: patience={early_stop_patience}, "
            f"min_delta={early_stop_min_delta}"
        )

    start_epoch = 1
    if args.resume:
        loaded_epoch, loaded_metrics = trainer.load_checkpoint(args.resume)
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_metrics.get("loss", best_val_loss)
        best_epoch = loaded_epoch
        print(f"Resumed from {args.resume} at epoch {loaded_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, codec, epoch)
        val_metrics = trainer.validate(val_loader, codec)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} | "
            f"Val CB Acc: {val_metrics['per_codebook_accuracy']}"
        )

        writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalars("Accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"],
        }, epoch)

        if epoch % tc.get("save_every_n_epochs", 10) == 0:
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
                epoch, train_metrics
            )

        improved = val_metrics["loss"] < best_val_loss - early_stop_min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve_epochs = 0
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, "best.pt"),
                epoch, val_metrics
            )
            print(f"  New best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve_epochs += 1
            if early_stop_patience > 0:
                print(
                    f"  No val improvement for {no_improve_epochs}/"
                    f"{early_stop_patience} epoch(s); best epoch={best_epoch}"
                )
                if no_improve_epochs >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    writer.close()
    print(f"\nTraining complete. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
