"""MIDI Transformer: context features → bass activity + pitch prediction.

Shares the same multi-instrument conditioning design as the audio-token branch:
instrument embedding + Transformer encoder → activity head + per-instrument pitch head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentEmbedding(nn.Module):
    """Shared with audio-token branch design."""
    def __init__(self, num_instruments, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_instruments, embed_dim)

    def forward(self, instrument_idx):
        return self.embedding(instrument_idx).unsqueeze(1)  # [B, 1, D]


class MidiTransformer(nn.Module):
    """Transformer encoder that predicts bass activity + pitch from audio context features.

    Multi-instrument: different pitch ranges per instrument via configurable pitch heads.
    """

    def __init__(
        self,
        feature_dim=141,
        d_model=256,
        num_layers=4,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        num_instruments=6,
        instrument_embed_dim=64,
        # Per-instrument pitch ranges (list of (min, max) tuples)
        instrument_pitch_ranges=None,
        max_seq_len=1024,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_instruments = num_instruments

        if instrument_pitch_ranges is None:
            instrument_pitch_ranges = [
                (28, 60),   # bass: E1-C4
                (21, 108),  # piano: full range
                (36, 96),   # guitar
                (28, 72),   # strings
                (36, 96),   # organ
                (0, 0),     # drums: no pitch, activity only
            ]
        self.instrument_pitch_ranges = instrument_pitch_ranges

        # Feature projection
        self.feat_proj = nn.Linear(feature_dim, d_model)

        # Instrument embedding
        self.instrument_embedding = InstrumentEmbedding(num_instruments, instrument_embed_dim)
        self.inst_proj = nn.Linear(instrument_embed_dim, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        # Shared activity head
        self.activity_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1),
        )
        # Default init (zero bias) — pos_weight in loss handles class imbalance

        # Per-instrument pitch heads (only for instruments with pitch)
        self.pitch_heads = nn.ModuleList()
        self.pitch_ranges = []
        for inst_idx in range(num_instruments):
            min_p, max_p = instrument_pitch_ranges[inst_idx]
            num_pitches = max_p - min_p + 1
            self.pitch_ranges.append((min_p, max_p, num_pitches))
            if num_pitches > 1:
                self.pitch_heads.append(nn.Linear(d_model, num_pitches))
            else:
                self.pitch_heads.append(None)  # drums: no pitch head

    def forward(self, features, instrument_idx):
        """Forward pass.

        Args:
            features: [B, T, D] audio features
            instrument_idx: [B] instrument labels

        Returns:
            activity_logits: [B, T]
            pitch_logits: [B, T, num_pitches] or [B, T, 1] for drums
        """
        bsz, seq_len, _ = features.shape

        x = self.feat_proj(features)
        x = self.dropout(x)

        i_emb = self.instrument_embedding(instrument_idx)  # [B, 1, inst_dim]
        i_emb = self.inst_proj(i_emb)  # [B, 1, d_model]
        x = x + i_emb

        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        # Shared activity head
        activity_logits = self.activity_head(x).squeeze(-1)  # [B, T]

        # Per-instrument pitch head (batch may have mixed instruments)
        # For training with single instrument per batch, use instrument_idx[0]
        inst = instrument_idx[0].item()
        pitch_head = self.pitch_heads[inst]
        if pitch_head is not None:
            pitch_logits = pitch_head(x)  # [B, T, num_pitches]
        else:
            pitch_logits = torch.zeros(bsz, seq_len, 1, device=x.device)

        return activity_logits, pitch_logits

    def compute_loss(self, features, instrument_idx, active_label, pitch_label,
                     lambda_active=1.0, lambda_pitch=1.0, ignore_index=-100):
        """Compute combined activity BCE + pitch CE loss.

        pitch_loss is only computed on frames where active_label == 1.
        """
        activity_logits, pitch_logits = self.forward(features, instrument_idx)

        # BCE with pos_weight for class imbalance (active frames are ~5-15%)
        n_positive = active_label.sum()
        n_total = active_label.numel()
        n_negative = n_total - n_positive
        pos_weight = min(n_negative / max(n_positive, 1.0), 30.0)
        active_loss = F.binary_cross_entropy_with_logits(
            activity_logits, active_label,
            pos_weight=torch.tensor(pos_weight, device=activity_logits.device),
        )

        # Pitch CE loss on active frames only
        active_mask = active_label > 0.5
        if active_mask.sum() > 0:
            pitch_loss = F.cross_entropy(
                pitch_logits[active_mask],
                pitch_label[active_mask],
                ignore_index=ignore_index,
            )
        else:
            pitch_loss = torch.tensor(0.0, device=activity_logits.device)

        total_loss = lambda_active * active_loss + lambda_pitch * pitch_loss
        return total_loss, active_loss.item(), pitch_loss.item()

    @torch.no_grad()
    def predict(self, features, instrument_idx, activity_threshold=0.5):
        """Predict activity and pitch from features.

        Returns:
            active_prob: [B, T] probability of bass activity
            pitch_id: [B, T] predicted pitch index, 0 when inactive
        """
        self.eval()
        activity_logits, pitch_logits = self.forward(features, instrument_idx)
        active_prob = torch.sigmoid(activity_logits)
        active_pred = active_prob > activity_threshold
        pitch_id = pitch_logits.argmax(dim=-1)  # [B, T]
        pitch_id[~active_pred] = 0
        return active_prob, pitch_id
