"""Note sequence generation: encoder → autoregressive decoder → [PITCH, DUR, ...].

Fundamentally different from frame-level prediction:
- Input: audio features [T, D]
- Output: note token sequence [BOS, P1, D1, P2, D2, ..., EOS]
- Loss: cross-entropy (teacher forcing)
- Metrics: note-level precision/recall/F1
"""
import os, sys, random, argparse
import torch, torchaudio, numpy as np, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pretty_midi as pm
from collections import defaultdict as dd

_flu_dir = "E:/tools/fluidsynth/bin"
if os.path.isdir(_flu_dir) and _flu_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _flu_dir + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_flu_dir)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.midi.audio_features import AudioFeatureExtractor
from src.midi.midi_labels import MidiLabelExtractor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/midi_subset")
    p.add_argument("--max-tracks", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="outputs/midi/exp_noteseq")
    p.add_argument("--clip-sec", type=float, default=4.0)
    p.add_argument("--epochs-train", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-pitch", type=int, default=28)
    p.add_argument("--max-pitch", type=int, default=60)

    # Encoder type
    p.add_argument("--encoder", type=str, default="transformer",
                   choices=["transformer", "crnn", "hubert"])
    p.add_argument("--use-mix", action="store_true")
    p.add_argument("--decoder-hidden", type=int, default=256)

    return p.parse_args()


# ─── Token vocabulary ───────────────────────────────────────────
NUM_PITCHES = 33
NUM_DUR_BINS = 16
PITCH_OFFSET = 0
DUR_OFFSET = NUM_PITCHES
BOS_TOKEN = NUM_PITCHES + NUM_DUR_BINS
EOS_TOKEN = BOS_TOKEN + 1
VOCAB_SIZE = EOS_TOKEN + 1
PAD_TOKEN = 0  # use pitch 0 as pad

# Log-spaced duration bins: 2-200 frames (40ms - 4s at 50Hz)
DUR_BINS = np.logspace(np.log10(2), np.log10(200), NUM_DUR_BINS)


class Encoder(torch.nn.Module):
    """Audio feature encoder with global pooling."""

    def __init__(self, encoder_type, feature_dim, d_model=256, dropout=0.3, device="cuda"):
        super().__init__()
        self.encoder_type = encoder_type
        self.d_model = d_model

        if encoder_type == "transformer":
            self.proj = torch.nn.Linear(feature_dim, d_model)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=512,
                dropout=dropout, batch_first=True, norm_first=True)
            self.tf = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.out_dim = d_model

        elif encoder_type == "crnn":
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv1d(feature_dim, d_model, 5, padding=2),
                torch.nn.BatchNorm1d(d_model), torch.nn.GELU())
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv1d(d_model, d_model, 5, padding=2),
                torch.nn.BatchNorm1d(d_model), torch.nn.GELU(),
                torch.nn.Dropout(dropout))
            self.conv3 = torch.nn.Sequential(
                torch.nn.Conv1d(d_model, d_model, 3, padding=1),
                torch.nn.BatchNorm1d(d_model), torch.nn.GELU())
            self.gru = torch.nn.GRU(d_model, d_model, num_layers=2,
                                    batch_first=True, dropout=dropout, bidirectional=True)
            self.out_dim = d_model * 2

        elif encoder_type == "hubert":
            bundle = torchaudio.pipelines.HUBERT_BASE
            self.hubert = bundle.get_model().to(device)
            self.hubert.eval()
            for p in self.hubert.parameters():
                p.requires_grad = False
            self.hubert_sr = 16000
            self.proj = torch.nn.Linear(768, d_model * 2)
            self.out_dim = d_model * 2

    @torch.no_grad()
    def _hubert_features(self, waveform, input_sr):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if input_sr != self.hubert_sr:
            waveform = torchaudio.functional.resample(waveform, input_sr, self.hubert_sr)
        waveform = waveform.to(next(self.hubert.parameters()).device)
        out, _ = self.hubert.extract_features(waveform)
        return out[-1].squeeze(0)  # [T, 768]

    def forward(self, features, waveform=None, input_sr=None):
        """Returns [B, out_dim] pooled representation."""
        if self.encoder_type == "hubert":
            if waveform is None:
                raise ValueError("waveform required for hubert encoder")
            # Handle batched waveforms
            feats_list = []
            for i in range(len(waveform) if waveform.dim() > 1 else 1):
                wf = waveform[i] if waveform.dim() > 1 else waveform
                sr = input_sr[i] if isinstance(input_sr, list) else input_sr
                feats_list.append(self._hubert_features(wf, sr))
            # Pad to same length
            max_t = max(f.shape[0] for f in feats_list)
            padded = []
            for f in feats_list:
                if f.shape[0] < max_t:
                    pad = torch.zeros(max_t - f.shape[0], f.shape[1], device=f.device)
                    f = torch.cat([f, pad])
                padded.append(f)
            x = torch.stack(padded)  # [B, T, 768]
            x = self.proj(x)  # [B, T, d_model*2]
        elif self.encoder_type == "transformer":
            x = self.proj(features)
            x = self.tf(x)
        elif self.encoder_type == "crnn":
            x = features.transpose(1, 2)  # [B, D, T]
            x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
            x = x.transpose(1, 2)  # [B, T, d_model]
            x, _ = self.gru(x)  # [B, T, d_model*2]

        # Global mean pooling
        return x.mean(dim=1)  # [B, out_dim]


class NoteDecoder(torch.nn.Module):
    """Autoregressive GRU decoder for note sequence generation."""

    def __init__(self, encoder_dim, hidden=256, vocab=VOCAB_SIZE, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden
        self.enc_to_hidden = torch.nn.Linear(encoder_dim, hidden * 2)  # 2 layers
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.gru = torch.nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                                dropout=dropout)
        self.output = torch.nn.Linear(hidden, vocab)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_out, target_tokens):
        """Teacher forcing training.

        Args:
            encoder_out: [B, enc_dim]
            target_tokens: [B, seq_len] (includes BOS, excludes EOS for input)

        Returns:
            logits: [B, seq_len, vocab]
        """
        B = encoder_out.shape[0]
        h0 = self.enc_to_hidden(encoder_out)  # [B, hidden*2]
        h0 = h0.view(2, B, self.hidden_dim)  # [2, B, hidden]

        emb = self.dropout(self.embedding(target_tokens[:, :-1]))  # [B, seq-1, hidden]
        out, _ = self.gru(emb, h0)
        logits = self.output(self.dropout(out))  # [B, seq-1, vocab]
        return logits

    def generate(self, encoder_out, max_len=64, temperature=1.0):
        """Autoregressive generation."""
        B = encoder_out.shape[0]
        h0 = self.enc_to_hidden(encoder_out)
        h0 = h0.view(2, B, self.hidden_dim)
        h = h0

        tokens = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=encoder_out.device)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(tokens[:, -1:])
            out, h = self.gru(emb, h)
            logits = self.output(out[:, -1]) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [B, 1]
            generated.append(next_token)
            tokens = torch.cat([tokens, next_token], dim=1)

        return torch.cat(generated, dim=1)  # [B, max_len]


def extract_note_tokens(midi_path, total_duration_sec, spf=0.02133, max_notes=64):
    """Extract note tokens from MIDI file."""
    n_frames = int(total_duration_sec / spf)
    try:
        midi = pm.PrettyMIDI(midi_path)
    except Exception:
        return None

    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            if note.pitch < 28 or note.pitch > 60:
                continue
            sf = int(note.start / spf)
            ef = int(min(note.end, total_duration_sec) / spf)
            dur = ef - sf
            if dur < 2:
                continue
            dur_bin = min(np.digitize(dur, DUR_BINS), NUM_DUR_BINS - 1)
            notes.append((sf, note.pitch - 28, dur_bin))

    notes.sort()
    tokens = [BOS_TOKEN]
    for _, pid, db in notes[:max_notes]:
        tokens.append(PITCH_OFFSET + pid)
        tokens.append(DUR_OFFSET + db)
    tokens.append(EOS_TOKEN)
    return torch.tensor(tokens, dtype=torch.long)


def build_dataset(data_root, max_tracks, clip_sec, seed, use_mix, encoder_type):
    """Build clips with note sequence targets."""
    tracks = sorted(Path(data_root).glob("Track*"))
    if max_tracks and len(tracks) > max_tracks:
        tracks = tracks[:max_tracks]

    clips = []
    for track_dir in tqdm(tracks, desc="Building clips"):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists(): continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        bass_id = next((sid for sid, info in meta.get("stems", {}).items()
                         if info.get("inst_class", "").lower() == "bass"
                         and info.get("midi_saved", False)), None)
        if not bass_id: continue
        midi_path = track_dir / "MIDI" / f"{bass_id}.mid"
        if not midi_path.exists(): continue
        for ext in [".flac", ".wav"]:
            mix_path = track_dir / f"mix{ext}"
            bass_path = track_dir / f"{bass_id}{ext}"
            if mix_path.exists() and bass_path.exists(): break
        else: continue

        info = torchaudio.info(str(mix_path))
        sr, total_frames = info.sample_rate, info.num_frames
        clip_samples = int(clip_sec * sr)
        n_clips = min(8, max(2, int(total_frames / sr / clip_sec)))

        for _ in range(n_clips):
            start = random.randint(0, max(0, total_frames - clip_samples))
            mix, _ = torchaudio.load(str(mix_path), frame_offset=start, num_frames=clip_samples)
            bass, _ = torchaudio.load(str(bass_path), frame_offset=start, num_frames=clip_samples)
            if mix.abs().max() < 0.005: continue
            mix_m = mix.mean(dim=0, keepdim=True)
            bass_m = bass.mean(dim=0, keepdim=True)
            ctx = mix_m - bass_m
            ctx = ctx / max(ctx.abs().max(), 0.01)

            tokens = extract_note_tokens(midi_path, float(clip_samples / sr),
                                         spf=0.02133)
            if tokens is None or len(tokens) < 3:
                continue

            clips.append({
                "context": ctx, "context_sr": sr, "mix": mix_m,
                "tokens": tokens, "track": track_dir.name,
            })
    return clips


def note_f1(pred_notes, target_notes, time_tol=0.1):
    """Compute note-level F1 with time tolerance (in seconds).

    pred_notes: list of (start_sec, end_sec, pitch)
    target_notes: list of (start_sec, end_sec, pitch)
    """
    def match(preds, targets):
        matched = 0
        used = set()
        for ps, pe, pp in preds:
            for i, (ts, te, tp) in enumerate(targets):
                if i in used: continue
                if pp == tp and abs(ps - ts) < time_tol and abs(pe - te) < time_tol:
                    matched += 1
                    used.add(i)
                    break
        return matched

    tp = match(pred_notes, target_notes)
    prec = tp / max(len(pred_notes), 1)
    rec = tp / max(len(target_notes), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return f1, prec, rec, tp


def tokens_to_notes(tokens, spf=0.02133, min_pitch=28):
    """Convert token sequence to note list."""
    notes = []
    i = 0
    cur_time = 0.0
    last_pitch = 0
    while i < len(tokens):
        t = tokens[i].item()
        if t >= EOS_TOKEN or t == BOS_TOKEN:
            break
        if t >= DUR_OFFSET:
            dur_bin = t - DUR_OFFSET
            dur_frames = DUR_BINS[min(dur_bin, len(DUR_BINS)-1)]
            dur_sec = dur_frames * spf
            notes.append((cur_time, cur_time + dur_sec, last_pitch + min_pitch))
            cur_time += dur_sec
        elif t >= PITCH_OFFSET:
            last_pitch = t - PITCH_OFFSET
        i += 1
    return notes


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Encoder: {args.encoder}")

    # Feature extractor for non-hubert encoders
    feature_dim = 140
    if args.encoder != "hubert":
        fe = AudioFeatureExtractor(
            sample_rate=24000, hop_length=512, use_mel=True, use_chroma=True,
            use_onset=False, use_energy=True,
            use_mix=args.use_mix, dual_channel=False)
        feature_dim = fe.feature_dim

    # Build dataset
    cache_path = os.path.join(args.output_dir, f"noteseq_{args.encoder}.pt")
    if os.path.exists(cache_path):
        print("Loading cached dataset...")
        cache = torch.load(cache_path, weights_only=False)
    else:
        clips = build_dataset(args.data_root, args.max_tracks, args.clip_sec,
                              args.seed, args.use_mix, args.encoder)
        tracks = sorted(set(c["track"] for c in clips))
        split = int(len(tracks) * 0.85)
        train_tracks = set(tracks[:split])
        print(f"Tracks: {len(tracks)}, Clips: {len(clips)}")

        cache = []
        for clip in tqdm(clips, desc="Caching features"):
            entry = {
                "context": clip["context"], "context_sr": clip["context_sr"],
                "mix": clip["mix"], "tokens": clip["tokens"],
                "is_train": clip["track"] in train_tracks, "track": clip["track"],
            }
            if args.encoder != "hubert":
                mw = clip["mix"] if args.use_mix else None
                f = fe(clip["context"], clip["context_sr"], mix_waveform=mw)
                entry["features"] = f
            cache.append(entry)
        torch.save(cache, cache_path)

    train_set = [c for c in cache if c["is_train"]]
    val_set = [c for c in cache if not c["is_train"]]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Build model
    encoder = Encoder(args.encoder, feature_dim, dropout=args.dropout, device=device).to(device)
    decoder = NoteDecoder(encoder.out_dim, hidden=args.decoder_hidden,
                          dropout=args.dropout).to(device)
    print(f"Params: encoder={sum(p.numel() for p in encoder.parameters()):,}, "
          f"decoder={sum(p.numel() for p in decoder.parameters()):,}")

    class SeqDS(torch.utils.data.Dataset):
        def __init__(self, clips):
            self.clips = [c for c in clips if len(c["tokens"]) >= 3]
        def __len__(self): return len(self.clips)
        def __getitem__(self, i):
            c = self.clips[i]
            return (c.get("features", torch.zeros(1)), c["context"], c["context_sr"],
                    c["tokens"], c.get("mix"))

    def collate(batch):
        feats, ctxs, srs, tokens, mixes = zip(*batch)
        # Pad tokens
        max_len = max(len(t) for t in tokens)
        padded_tokens = torch.zeros(len(tokens), max_len, dtype=torch.long)
        mask = torch.zeros(len(tokens), max_len, dtype=torch.bool)
        for i, t in enumerate(tokens):
            padded_tokens[i, :len(t)] = t
            mask[i, :len(t)] = True
        return (torch.stack([f for f in feats]), list(ctxs), list(srs),
                padded_tokens, mask, list(mixes))

    train_dl = torch.utils.data.DataLoader(SeqDS(train_set), batch_size=args.batch_size,
                                            shuffle=True, collate_fn=collate)
    val_dl = torch.utils.data.DataLoader(SeqDS(val_set), batch_size=args.batch_size,
                                          collate_fn=collate)

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW([p for p in params if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    best_f1 = 0
    best_ep = 1
    patience = 0

    for epoch in range(1, args.epochs_train + 1):
        encoder.train(); decoder.train()
        total_loss = 0.0
        for feats, ctxs, srs, tokens, mask, mixes in train_dl:
            feats = feats.to(device); tokens = tokens.to(device)
            opt.zero_grad()

            if args.encoder == "hubert":
                enc_out = encoder(None, waveform=ctxs, input_sr=srs)
            else:
                enc_out = encoder(feats)

            # Target: tokens[:, 1:] (shift right by 1)
            logits = decoder(enc_out, tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tokens[:, 1:logits.shape[1]+1].reshape(-1),
                ignore_index=PAD_TOKEN)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()

        # Validation (note-level F1)
        encoder.eval(); decoder.eval()
        val_loss = 0.0
        all_f1 = []
        with torch.no_grad():
            for feats, ctxs, srs, tokens, mask, mixes in val_dl:
                feats = feats.to(device); tokens = tokens.to(device)
                if args.encoder == "hubert":
                    enc_out = encoder(None, waveform=ctxs, input_sr=srs)
                else:
                    enc_out = encoder(feats)
                logits = decoder(enc_out, tokens)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    tokens[:, 1:logits.shape[1]+1].reshape(-1),
                    ignore_index=PAD_TOKEN)
                val_loss += loss.item()

                # Generate and compute note F1
                generated = decoder.generate(enc_out, max_len=64)
                for b in range(len(generated)):
                    pred_notes = tokens_to_notes(generated[b])
                    targ_notes = tokens_to_notes(tokens[b])
                    f1, _, _, _ = note_f1(pred_notes, targ_notes)
                    all_f1.append(f1)

        avg_f1 = np.mean(all_f1) if all_f1 else 0
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={total_loss/len(train_dl):.4f} "
                  f"val_loss={val_loss/len(val_dl):.4f} | note_f1={avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1; best_ep = epoch; patience = 0
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(), "f1": best_f1},
                       os.path.join(args.output_dir, "checkpoints", "best.pt"))
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}"); break

    print(f"\n{'='*50}")
    print(f"Encoder: {args.encoder}, Use-mix: {args.use_mix}")
    print(f"Best epoch: {best_ep}, Note F1: {best_f1:.4f}")

    # Comparison with frame-level baseline
    frame_f1_baseline = {"transformer": 0.130, "crnn": 0.228, "hubert": 0.289}
    base = frame_f1_baseline.get(args.encoder, 0.130)
    print(f"Frame-level baseline F1: {base:.4f}")
    print(f"Note-seq F1: {best_f1:.4f} ({best_f1 - base:+.4f} vs frame-level)")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
