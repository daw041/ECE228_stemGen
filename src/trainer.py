"""Training loop for the StemGen masked-token model."""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


class MaskedTokenTrainer:
    """Trainer for masked target token prediction with single-forward-pass efficiency."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        mask_ratio: float = 0.75,
        mask_ratio_min: float = None,
        mask_ratio_max: float = None,
        codebook_weights=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.mask_ratio = mask_ratio
        self.mask_ratio = float(mask_ratio)
        self.mask_ratio_min = None if mask_ratio_min is None else float(mask_ratio_min)
        self.mask_ratio_max = None if mask_ratio_max is None else float(mask_ratio_max)
        self.mask_token_id = model.mask_token_id
        if codebook_weights is None:
            codebook_weights = [1.0] * model.num_codebooks
        if len(codebook_weights) != model.num_codebooks:
            raise ValueError(
                f"Expected {model.num_codebooks} codebook weights, got {len(codebook_weights)}."
            )
        self.codebook_weights = [float(w) for w in codebook_weights]
        self.optimizer = AdamW(
            model.parameters(), lr=float(lr), weight_decay=float(weight_decay), betas=(0.9, 0.999)
        )

    def _sample_mask_ratio(self) -> float:
        if self.mask_ratio_min is not None and self.mask_ratio_max is not None:
            return random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        return self.mask_ratio

    def _mask_target_tokens(self, target_tokens):
        bsz, n_cb, seq_len = target_tokens.shape
        mask_ratio = self._sample_mask_ratio()
        rand = torch.rand(bsz, seq_len, device=target_tokens.device)
        time_mask = rand < mask_ratio
        masked = target_tokens.clone()
        for cb in range(n_cb):
            masked[:, cb, :][time_mask] = self.mask_token_id
        return masked

    def _loss_and_accuracy(self, logits, target_tokens, masked_target):
        total_loss = torch.tensor(0.0, device=logits.device)
        total_correct = 0.0
        total_tokens = 0
        per_cb_correct = [0.0 for _ in range(self.model.num_codebooks)]
        per_cb_total = [0 for _ in range(self.model.num_codebooks)]

        for cb in range(self.model.num_codebooks):
            mask = masked_target[:, cb, :] == self.mask_token_id
            n_tok = int(mask.sum().item())
            if n_tok == 0:
                continue

            cb_logits = logits[:, cb, :, :][mask]
            cb_target = target_tokens[:, cb, :][mask]
            total_loss = total_loss + self.codebook_weights[cb] * F.cross_entropy(cb_logits, cb_target)

            preds = cb_logits.argmax(dim=-1)
            correct = (preds == cb_target).float().sum().item()
            total_correct += correct
            total_tokens += n_tok
            per_cb_correct[cb] += correct
            per_cb_total[cb] += n_tok

        if total_tokens == 0:
            total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return total_loss, total_correct, total_tokens, per_cb_correct, per_cb_total

    def _step(self, context_tokens, target_tokens, instrument_idx):
        """Single forward pass returning loss + accuracy."""
        masked_target = self._mask_target_tokens(target_tokens)
        logits = self.model(context_tokens, masked_target, instrument_idx)
        return self._loss_and_accuracy(logits, target_tokens, masked_target)

    def train_epoch(self, dataloader, codec, epoch):
        self.model.train()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        cb_correct = [0.0 for _ in range(self.model.num_codebooks)]
        cb_total = [0 for _ in range(self.model.num_codebooks)]
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            ctx = batch["context"]
            tgt = batch["target"]
            inst = batch["instrument"]
            with torch.no_grad():
                ctx_tok = codec.encode(ctx).to(self.device)
                tgt_tok = codec.encode(tgt).to(self.device)
            inst = inst.to(self.device)
            if ctx_tok.dim() == 2:
                ctx_tok = ctx_tok.unsqueeze(0)
            if tgt_tok.dim() == 2:
                tgt_tok = tgt_tok.unsqueeze(0)
            if tgt_tok.shape[0] != ctx_tok.shape[0]:
                tgt_tok = tgt_tok.expand(ctx_tok.shape[0], -1, -1)
            self.optimizer.zero_grad()
            loss, acc, n_tok, batch_cb_correct, batch_cb_total = self._step(ctx_tok, tgt_tok, inst)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            correct_tokens += acc
            total_tokens += n_tok
            for cb in range(self.model.num_codebooks):
                cb_correct[cb] += batch_cb_correct[cb]
                cb_total[cb] += batch_cb_total[cb]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc / max(1, n_tok):.3f}" if n_tok > 0 else "N/A"
            })
        per_cb_accuracy = [
            cb_correct[cb] / max(1, cb_total[cb])
            for cb in range(self.model.num_codebooks)
        ]
        return {
            "loss": total_loss / max(1, len(dataloader)),
            "accuracy": correct_tokens / max(1, total_tokens),
            "per_codebook_accuracy": per_cb_accuracy,
        }

    @torch.no_grad()
    def validate(self, dataloader, codec):
        self.model.eval()
        total_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        cb_correct = [0.0 for _ in range(self.model.num_codebooks)]
        cb_total = [0 for _ in range(self.model.num_codebooks)]
        for batch in tqdm(dataloader, desc="Validation"):
            ctx = batch["context"]
            tgt = batch["target"]
            inst = batch["instrument"]
            ctx_tok = codec.encode(ctx).to(self.device)
            tgt_tok = codec.encode(tgt).to(self.device)
            inst = inst.to(self.device)
            if ctx_tok.dim() == 2:
                ctx_tok = ctx_tok.unsqueeze(0)
            if tgt_tok.dim() == 2:
                tgt_tok = tgt_tok.unsqueeze(0)
            if tgt_tok.shape[0] != ctx_tok.shape[0]:
                tgt_tok = tgt_tok.expand(ctx_tok.shape[0], -1, -1)
            masked_target = self._mask_target_tokens(tgt_tok)
            logits = self.model(ctx_tok, masked_target, inst)
            loss, acc, n_tok, batch_cb_correct, batch_cb_total = self._loss_and_accuracy(
                logits, tgt_tok, masked_target
            )
            total_loss += loss.item()
            correct_tokens += acc
            total_tokens += n_tok
            for cb in range(self.model.num_codebooks):
                cb_correct[cb] += batch_cb_correct[cb]
                cb_total[cb] += batch_cb_total[cb]
        per_cb_accuracy = [
            cb_correct[cb] / max(1, cb_total[cb])
            for cb in range(self.model.num_codebooks)
        ]
        return {
            "loss": total_loss / max(1, len(dataloader)),
            "accuracy": correct_tokens / max(1, total_tokens),
            "per_codebook_accuracy": per_cb_accuracy,
        }

    def save_checkpoint(self, path, epoch, metrics):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["metrics"]
