"""Note-level Transformer: onset/offset detection + pitch prediction.

Reformulates bass MIDI prediction as note boundary detection instead of
per-frame activity classification. Uses the same encoder architecture
as MidiTransformer but outputs onset, offset, and pitch per frame.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentEmbedding(nn.Module):
    def __init__(self, num_instruments, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_instruments, embed_dim)

    def forward(self, instrument_idx):
        return self.embedding(instrument_idx).unsqueeze(1)


class NoteTransformer(nn.Module):
    """Transformer encoder that predicts note onset, offset, and pitch.

    Compared to MidiTransformer (which predicts per-frame activity):
    - onset head: where notes START
    - offset head: where notes END
    - pitch head: which pitch (only at onset frames or active frames)
    """

    def __init__(
        self,
        feature_dim=140,
        d_model=256,
        num_layers=4,
        num_heads=4,
        dim_feedforward=512,
        dropout=0.1,
        num_instruments=6,
        instrument_embed_dim=64,
        instrument_pitch_ranges=None,
        max_seq_len=2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_instruments = num_instruments

        if instrument_pitch_ranges is None:
            instrument_pitch_ranges = [
                (28, 60), (21, 108), (36, 96), (28, 72), (36, 96), (0, 0),
            ]
        self.instrument_pitch_ranges = instrument_pitch_ranges

        self.feat_proj = nn.Linear(feature_dim, d_model)
        self.instrument_embedding = InstrumentEmbedding(num_instruments, instrument_embed_dim)
        self.inst_proj = nn.Linear(instrument_embed_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        # Onset head: predict note start locations
        self.onset_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        # Offset head: predict note end locations
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        # Activity head (keep for comparison)
        self.activity_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

        # Per-instrument pitch heads
        self.pitch_heads = nn.ModuleList()
        self.pitch_ranges = []
        for inst_idx in range(num_instruments):
            min_p, max_p = instrument_pitch_ranges[inst_idx]
            num_pitches = max_p - min_p + 1
            self.pitch_ranges.append((min_p, max_p, num_pitches))
            if num_pitches > 1:
                self.pitch_heads.append(nn.Linear(d_model, num_pitches))
            else:
                self.pitch_heads.append(None)

    def forward(self, features, instrument_idx):
        bsz, seq_len, _ = features.shape
        x = self.feat_proj(features)
        x = self.dropout(x)
        i_emb = self.instrument_embedding(instrument_idx)
        i_emb = self.inst_proj(i_emb)
        x = x + i_emb
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)

        activity_logits = self.activity_head(x).squeeze(-1)
        onset_logits = self.onset_head(x).squeeze(-1)
        offset_logits = self.offset_head(x).squeeze(-1)

        inst = instrument_idx[0].item()
        pitch_head = self.pitch_heads[inst]
        if pitch_head is not None:
            pitch_logits = pitch_head(x)
        else:
            pitch_logits = torch.zeros(bsz, seq_len, 1, device=x.device)

        return activity_logits, onset_logits, offset_logits, pitch_logits

    def compute_loss(self, features, instrument_idx,
                     onset_label, offset_label, pitch_label,
                     lambda_onset=2.0, lambda_offset=1.0, lambda_pitch=1.0,
                     ignore_index=-100):
        """Compute onset BCE + offset BCE + pitch CE loss.

        onset/offset BCE uses pos_weight for class imbalance.
        pitch CE only on frames where onset_label == 1.
        """
        _, onset_logits, offset_logits, pitch_logits = self.forward(features, instrument_idx)

        # Onset loss (very sparse — typically 1-2 onsets per second)
        n_pos = onset_label.sum()
        n_tot = onset_label.numel()
        pw_onset = min((n_tot - n_pos) / max(n_pos, 1.0), 100.0)
        onset_loss = F.binary_cross_entropy_with_logits(
            onset_logits, onset_label,
            pos_weight=torch.tensor(pw_onset, device=onset_logits.device),
        )

        # Offset loss
        n_pos_off = offset_label.sum()
        pw_offset = min((n_tot - n_pos_off) / max(n_pos_off, 1.0), 100.0)
        offset_loss = F.binary_cross_entropy_with_logits(
            offset_logits, offset_label,
            pos_weight=torch.tensor(pw_offset, device=offset_logits.device),
        )

        # Pitch loss on onset frames
        onset_mask = onset_label > 0.5
        if onset_mask.sum() > 0:
            pitch_loss = F.cross_entropy(
                pitch_logits[onset_mask], pitch_label[onset_mask],
                ignore_index=ignore_index,
            )
        else:
            pitch_loss = torch.tensor(0.0, device=onset_logits.device)

        total = lambda_onset * onset_loss + lambda_offset * offset_loss + lambda_pitch * pitch_loss
        return total, onset_loss.item(), offset_loss.item(), pitch_loss.item()

    @torch.no_grad()
    def predict(self, features, instrument_idx,
                onset_threshold=0.5, offset_threshold=0.5):
        """Predict onsets, offsets, and pitches."""
        self.eval()
        _, onset_logits, offset_logits, pitch_logits = self.forward(features, instrument_idx)
        onset_prob = torch.sigmoid(onset_logits)
        offset_prob = torch.sigmoid(offset_logits)
        pitch_id = pitch_logits.argmax(dim=-1)
        return onset_prob, offset_prob, pitch_id

    @torch.no_grad()
    def predict_notes(self, features, instrument_idx,
                       onset_threshold=0.5, offset_threshold=0.5,
                       min_note_frames=3, max_note_frames=100):
        """Convert frame predictions to note list.

        Returns list of (start_frame, end_frame, pitch_id, confidence).
        """
        onset_prob, offset_prob, pitch_id = self.predict(
            features, instrument_idx, onset_threshold, offset_threshold)

        onset_prob = onset_prob[0].cpu().numpy()
        offset_prob = offset_prob[0].cpu().numpy()
        pitch_id = pitch_id[0].cpu().numpy()

        # Find onset peaks
        onsets = []
        for t in range(1, len(onset_prob) - 1):
            if onset_prob[t] > onset_threshold and onset_prob[t] > onset_prob[t-1] and onset_prob[t] > onset_prob[t+1]:
                onsets.append((t, onset_prob[t], pitch_id[t]))

        # For each onset, find corresponding offset
        notes = []
        for onset_t, onset_conf, p_id in onsets:
            # Search forward for offset
            offset_t = onset_t + min_note_frames
            found = False
            for t in range(onset_t + min_note_frames, min(len(offset_prob), onset_t + max_note_frames)):
                if offset_prob[t] > offset_threshold:
                    offset_t = t
                    found = True
                    break
            if not found:
                offset_t = min(onset_t + max_note_frames, len(offset_prob) - 1)

            notes.append((onset_t, offset_t, p_id, onset_conf))

        return notes
