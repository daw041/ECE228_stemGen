"""StemGen-style masked-token Transformer with multi-codebook RVQ support."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentEmbedding(nn.Module):
    def __init__(self, num_instruments: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_instruments, embed_dim)

    def forward(self, instrument_idx):
        return self.embedding(instrument_idx).unsqueeze(1)  # [B, 1, D]


class StemGenModel(nn.Module):
    """Non-autoregressive masked-token Transformer for multi-codebook stem generation.

    Follows StemGen's approach: sum token embeddings across codebook levels per time step,
    then use per-codebook output heads for hierarchical prediction.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        num_instruments: int = 6,
        instrument_embed_dim: int = 64,
        fusion_mode: str = "concat",
        num_codebooks: int = 4,
        max_seq_len: int = 512,
        use_activity_head: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.fusion_mode = fusion_mode
        self.num_codebooks = num_codebooks
        self.mask_token_id = vocab_size
        self.use_activity_head = use_activity_head

        # shared token embeddings across codebooks
        self.context_token_emb = nn.Embedding(vocab_size + 1, embedding_dim)
        self.target_token_emb = nn.Embedding(vocab_size + 1, embedding_dim)

        self.instrument_embedding = InstrumentEmbedding(num_instruments, instrument_embed_dim)

        fusion_input_dim = embedding_dim * 2 + instrument_embed_dim
        self.fusion_proj = nn.Linear(fusion_input_dim, embedding_dim)

        # positional encoding (learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # per-codebook output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(embedding_dim, vocab_size) for _ in range(num_codebooks)
        ])

        # Activity prediction head (frame-level bass presence)
        if use_activity_head:
            self.activity_head = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        self.dropout = nn.Dropout(dropout)

    def _embed_stream(self, tokens, emb_layer):
        """Sum token embeddings across all codebook levels per time step.

        tokens: [B, num_codebooks, T]
        returns: [B, T, embedding_dim]
        """
        emb = torch.zeros(tokens.shape[0], tokens.shape[2], self.embedding_dim,
                          device=tokens.device, dtype=torch.float)
        for cb in range(self.num_codebooks):
            emb = emb + emb_layer(tokens[:, cb, :])
        return emb

    def forward(self, context_tokens, target_tokens, instrument_idx, return_activity=False):
        """Forward pass.

        Args:
            context_tokens: [B, num_codebooks, T] unmasked context
            target_tokens:  [B, num_codebooks, T] partially masked target
            instrument_idx: [B] instrument labels
            return_activity: if True, also return activity logits

        Returns:
            logits: [B, num_codebooks, T, vocab_size]
            (activity_logits: [B, T, 1] if return_activity and use_activity_head)
        """
        bsz, n_cb, seq_len = target_tokens.shape
        if seq_len > self.pos_encoding.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.pos_encoding.shape[1]}. "
                "Increase model.max_seq_len in configs/model_config.yaml."
            )

        # sum embeddings across codebook levels
        c_emb = self._embed_stream(context_tokens, self.context_token_emb)  # [B, T, D]
        t_emb = self._embed_stream(target_tokens, self.target_token_emb)

        i_emb = self.instrument_embedding(instrument_idx).expand(-1, seq_len, -1)

        fused = torch.cat([c_emb, t_emb, i_emb], dim=-1)
        x = self.fusion_proj(fused)
        x = self.dropout(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        # per-codebook logits
        logits = torch.stack([head(x) for head in self.output_heads], dim=1)

        if return_activity and self.use_activity_head:
            activity_logits = self.activity_head(x)  # [B, T, 1]
            return logits, activity_logits
        return logits

    def compute_loss(self, context_tokens, target_tokens, masked_target_tokens, instrument_idx,
                     activity_labels=None, lambda_activity=0.2):
        """Compute cross-entropy loss over masked positions + optional activity BCE."""
        if self.use_activity_head and activity_labels is not None:
            logits, activity_logits = self.forward(
                context_tokens, masked_target_tokens, instrument_idx, return_activity=True)
        else:
            logits = self.forward(context_tokens, masked_target_tokens, instrument_idx)

        total_loss = torch.tensor(0.0, device=logits.device)
        has_mask = False

        for cb in range(self.num_codebooks):
            mask = masked_target_tokens[:, cb, :] == self.mask_token_id
            if mask.sum() > 0:
                has_mask = True
                total_loss = total_loss + F.cross_entropy(
                    logits[:, cb, :, :][mask],
                    target_tokens[:, cb, :][mask]
                )

        if not has_mask:
            total_loss = total_loss + torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Activity BCE loss
        if self.use_activity_head and activity_labels is not None:
            bce = F.binary_cross_entropy_with_logits(
                activity_logits.squeeze(-1), activity_labels
            )
            total_loss = total_loss + lambda_activity * bce

        return total_loss

    @torch.no_grad()
    def generate(self, context_tokens, instrument_idx, num_iterations=8, temperature=1.0,
                 causal_bias_weight=0.1, return_activity=False, top_k=None,
                 use_argmax=False):
        """Hierarchical iterative mask-predict generation with causal-biased decoding.

        Generates codebook 0 first (coarse), then codebook 1, 2, etc.
        Causal bias favors fixing earlier time positions first, improving temporal coherence.
        """
        self.eval()
        bsz, n_cb, seq_len = context_tokens.shape
        device = context_tokens.device
        if isinstance(instrument_idx, torch.Tensor):
            instr = instrument_idx.to(device)
        elif isinstance(instrument_idx, int):
            instr = torch.full((bsz,), instrument_idx, dtype=torch.long, device=device)
        else:
            instr = torch.tensor(instrument_idx, dtype=torch.long, device=device)
        if instr.ndim == 0:
            instr = instr.unsqueeze(0).expand(bsz)
        if instr.shape[0] != bsz:
            instr = instr.expand(bsz)
        keep_ratio = 0.25
        activity_logits_out = None
        if isinstance(num_iterations, int):
            iterations_per_codebook = [num_iterations] * n_cb
        else:
            iterations_per_codebook = list(num_iterations)
            if len(iterations_per_codebook) < n_cb:
                iterations_per_codebook.extend([iterations_per_codebook[-1]] * (n_cb - len(iterations_per_codebook)))

        # Pre-compute time position bias: earlier positions get higher bias
        time_bias = torch.linspace(causal_bias_weight, 0.0, seq_len, device=device)

        # start with all masked for all codebooks
        target_tokens = torch.full(
            (bsz, n_cb, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        # generate codebook by codebook
        for cb in range(n_cb):
            cb_target = target_tokens.clone()
            if cb > 0:
                cb_target[:, :cb, :] = target_tokens[:, :cb, :]

            cb_iterations = iterations_per_codebook[cb]
            for it in range(cb_iterations):
                if return_activity and self.use_activity_head:
                    logits, a_logits = self.forward(context_tokens, cb_target, instr, return_activity=True)
                    if it == 0 and cb == 0:
                        activity_logits_out = a_logits
                else:
                    logits = self.forward(context_tokens, cb_target, instr)

                cb_logits = logits[:, cb, :, :] / temperature
                if top_k is not None and top_k > 0:
                    kth = min(top_k, cb_logits.shape[-1])
                    values, _ = torch.topk(cb_logits, kth, dim=-1)
                    cutoff = values[..., -1, None]
                    cb_logits = cb_logits.masked_fill(cb_logits < cutoff, float("-inf"))
                probs = F.softmax(cb_logits, dim=-1)
                mask_pos = cb_target[:, cb, :] == self.mask_token_id
                num_masked = mask_pos.sum().item()
                if num_masked == 0:
                    break

                if use_argmax:
                    sampled = probs[mask_pos].argmax(dim=-1)
                else:
                    sampled = torch.multinomial(probs[mask_pos], num_samples=1).squeeze(-1)
                cb_target[:, cb, :][mask_pos] = sampled

                if it < cb_iterations - 1 and num_masked > 1:
                    conf = probs.max(dim=-1).values
                    # Causal-biased scoring: combine confidence + time position bias
                    for b in range(bsz):
                        b_mask = mask_pos[b]
                        b_masked = int(b_mask.sum().item())
                        if b_masked <= 1:
                            continue
                        combined = conf[b, b_mask] + time_bias[b_mask]
                        keep_n = max(1, int(b_masked * keep_ratio * (it + 1) / cb_iterations))
                        keep_n = min(keep_n, b_masked)
                        _, topk = torch.topk(combined, keep_n)
                        keep_within = torch.zeros(b_masked, dtype=torch.bool, device=device)
                        keep_within[topk] = True
                        all_masked_idx = torch.where(b_mask)[0]
                        remask_idx = all_masked_idx[~keep_within]
                        cb_target[b, cb, remask_idx] = self.mask_token_id

            target_tokens = cb_target.clone()

        if return_activity and self.use_activity_head:
            return target_tokens, activity_logits_out
        return target_tokens
