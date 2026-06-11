"""HuBERT experiment: pre-trained audio features for bass MIDI prediction.

Uses frozen HuBERT Base (94M params) as feature extractor.
Replaces mel+chroma with 768-dim HuBERT features.
Compares multiple prediction head architectures.
"""
import os, sys, random, argparse
import torch, torchaudio, numpy as np, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

_flu_dir = "E:/tools/fluidsynth/bin"
if os.path.isdir(_flu_dir) and _flu_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _flu_dir + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_flu_dir)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.midi.midi_labels import MidiLabelExtractor
from src.midi.postprocess import predictions_to_midi, midi_to_audio


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/midi_subset")
    p.add_argument("--max-tracks", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="outputs/midi/exp_hubert")
    p.add_argument("--clip-sec", type=float, default=4.0)
    p.add_argument("--epochs-train", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--head-type", type=str, default="gru",
                   choices=["mlp", "gru", "transformer"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-pitch", type=int, default=28)
    p.add_argument("--max-pitch", type=int, default=60)
    return p.parse_args()


class HuBERTFeatureExtractor:
    """Frozen HuBERT Base feature extractor.

    Input: audio at 16kHz
    Output: features at 50Hz, 768-dim (last hidden layer)
    """
    def __init__(self, device="cuda"):
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.sample_rate = bundle.sample_rate  # 16000
        self.feature_dim = 768
        # HuBERT output: list of 13 layer outputs, each [B, T, 768]
        # We use the last layer
        self.device = device

    @torch.no_grad()
    def __call__(self, waveform, input_sr):
        """Extract HuBERT features. waveform: [1, samples]"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if input_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_sr, self.sample_rate)
        waveform = waveform.to(self.device)

        # HuBERT expects specific length (divisible by stride)
        # Pad or trim to reasonable length
        min_len = int(self.sample_rate * 0.05)  # 50ms minimum
        if waveform.shape[-1] < min_len:
            waveform = torch.nn.functional.pad(waveform, (0, min_len - waveform.shape[-1]))

        # Get features from last transformer layer
        output, _ = self.model.extract_features(waveform)
        # output is list of 13 layers, take the last one: [1, T, 768]
        features = output[-1].squeeze(0).cpu()  # [T, 768]
        return features


def build_prediction_head(head_type, feature_dim, num_pitches, dropout):
    """Build prediction head on top of HuBERT features."""
    if head_type == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, num_pitches + 1),  # +1 for activity
        )
    elif head_type == "gru":
        class GRUHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = torch.nn.GRU(feature_dim, 256, num_layers=2,
                                        batch_first=True, dropout=dropout, bidirectional=True)
                self.proj = torch.nn.Sequential(
                    torch.nn.Linear(512, 256),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(256, num_pitches + 1),
                )
            def forward(self, x):
                out, _ = self.gru(x)  # [B, T, 512]
                return self.proj(out)  # [B, T, num_pitches+1]
        return GRUHead()
    elif head_type == "transformer":
        class TFHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = torch.nn.Linear(feature_dim, 256)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=256, nhead=4, dim_feedforward=512,
                    dropout=dropout, batch_first=True, norm_first=True)
                self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.proj_out = torch.nn.Linear(256, num_pitches + 1)
            def forward(self, x):
                x = self.proj_in(x)
                x = self.encoder(x)
                return self.proj_out(x)
        return TFHead()


def build_clips(data_root, max_tracks, clip_sec, seed, hubert_sr=16000):
    """Build clips resampled to HuBERT sample rate."""
    tracks = sorted(Path(data_root).glob("Track*"))
    if max_tracks and len(tracks) > max_tracks:
        tracks = tracks[:max_tracks]

    all_clips = []
    for track_dir in tqdm(tracks, desc="Building clips"):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        bass_id = None
        for sid, info in meta.get("stems", {}).items():
            if info.get("inst_class", "").lower() == "bass" and info.get("midi_saved", False):
                bass_id = sid; break
        if not bass_id:
            continue
        midi_path = track_dir / "MIDI" / f"{bass_id}.mid"
        if not midi_path.exists():
            continue
        for ext in [".flac", ".wav"]:
            mix_path = track_dir / f"mix{ext}"
            bass_path = track_dir / f"{bass_id}{ext}"
            if mix_path.exists() and bass_path.exists():
                break
        else:
            continue

        info = torchaudio.info(str(mix_path))
        sr, total_frames = info.sample_rate, info.num_frames
        clip_samples_orig = int(clip_sec * sr)
        n_clips = min(8, max(2, int(total_frames / sr / clip_sec)))

        for _ in range(n_clips):
            start = random.randint(0, max(0, total_frames - clip_samples_orig))
            mix, _ = torchaudio.load(str(mix_path), frame_offset=start, num_frames=clip_samples_orig)
            bass, _ = torchaudio.load(str(bass_path), frame_offset=start, num_frames=clip_samples_orig)
            if mix.abs().max() < 0.005:
                continue
            # Resample to 16kHz for HuBERT
            mix_16k = torchaudio.functional.resample(mix, sr, hubert_sr)
            bass_16k = torchaudio.functional.resample(bass, sr, hubert_sr)
            mix_mono = mix_16k.mean(dim=0, keepdim=True)
            bass_mono = bass_16k.mean(dim=0, keepdim=True)
            ctx = mix_mono - bass_mono
            ctx = ctx / max(ctx.abs().max(), 0.01)

            all_clips.append({
                "context": ctx, "context_sr": hubert_sr, "mix": mix_mono,
                "midi_path": str(midi_path),
                "total_duration": float(ctx.shape[-1] / hubert_sr),
                "track": track_dir.name,
            })
    return all_clips


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build clips at 16kHz for HuBERT
    cache_path = os.path.join(args.output_dir, "hubert_cache.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
    else:
        clips = build_clips(args.data_root, args.max_tracks, args.clip_sec, args.seed)
        track_names = sorted(set(c["track"] for c in clips))
        split = int(len(track_names) * 0.85)
        train_tracks = set(track_names[:split])
        print(f"Tracks: {len(track_names)}, Train: {split}, Val: {len(track_names)-split}")
        print(f"Clips: {len(clips)}")

        # HuBERT feature extraction
        hubert = HuBERTFeatureExtractor(device=device)
        label_extractor = MidiLabelExtractor(
            min_pitch=args.min_pitch, max_pitch=args.max_pitch,
            hop_length=320, sample_rate=16000,  # 50Hz to match HuBERT
        )

        print(f"Extracting HuBERT features ({len(clips)} clips)...")
        cache = []
        for clip in tqdm(clips, desc="HuBERT features"):
            # Context features
            feats = hubert(clip["context"], clip["context_sr"])  # [T, 768]
            # Mix features (for comparison)
            mix_feats = hubert(clip["mix"], clip["context_sr"])
            # Labels at 50Hz
            active_label, pitch_label = label_extractor.extract(
                clip["midi_path"], clip["total_duration"])

            min_len = min(feats.shape[0], mix_feats.shape[0], len(active_label), len(pitch_label))
            cache.append({
                "ctx_features": feats[:min_len],
                "mix_features": mix_feats[:min_len],
                "active_label": active_label[:min_len],
                "pitch_label": pitch_label[:min_len],
                "is_train": clip["track"] in train_tracks,
                "track": clip["track"],
            })
        torch.save(cache, cache_path)
        print(f"Saved {len(cache)} clips to cache")

    train_set = [c for c in cache if c["is_train"]]
    val_set = [c for c in cache if not c["is_train"]]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Build prediction heads
    num_pitches = args.max_pitch - args.min_pitch + 1
    feature_dim = 768

    # We'll train two variants: context-only and mix-only
    # For this experiment, use context (to compare with baseline)
    class HuBERTDataset(torch.utils.data.Dataset):
        def __init__(self, clips, use_mix=False):
            self.clips = clips
            self.use_mix = use_mix
        def __len__(self):
            return len(self.clips)
        def __getitem__(self, i):
            c = self.clips[i]
            feats = c["mix_features"] if self.use_mix else c["ctx_features"]
            return feats, c["active_label"], c["pitch_label"]

    # Context variant
    print(f"\n=== Context features with {args.head_type} head ===")
    head = build_prediction_head(args.head_type, feature_dim, num_pitches, args.dropout).to(device)
    print(f"Head params: {sum(p.numel() for p in head.parameters()):,}")

    train_dl = torch.utils.data.DataLoader(
        HuBERTDataset(train_set, use_mix=False),
        batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        HuBERTDataset(val_set, use_mix=False),
        batch_size=args.batch_size)

    # Also train mix variant
    print(f"\n=== Mix features with {args.head_type} head ===")
    head_mix = build_prediction_head(args.head_type, feature_dim, num_pitches, args.dropout).to(device)

    train_dl_mix = torch.utils.data.DataLoader(
        HuBERTDataset(train_set, use_mix=True),
        batch_size=args.batch_size, shuffle=True)
    val_dl_mix = torch.utils.data.DataLoader(
        HuBERTDataset(val_set, use_mix=True),
        batch_size=args.batch_size)

    results = {}
    for name, mdl, tdl, vdl in [
        ("HuBERT-ctx", head, train_dl, val_dl),
        ("HuBERT-mix", head_mix, train_dl_mix, val_dl_mix),
    ]:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float("inf")
        best_ep = 1
        patience = 0
        best_f1 = 0
        best_metrics = {}

        for epoch in range(1, args.epochs_train + 1):
            mdl.train()
            train_loss = 0.0
            for feats, act, pit in tdl:
                feats, act, pit = feats.to(device), act.to(device), pit.to(device)
                opt.zero_grad()
                out = mdl(feats)  # [B, T, num_pitches+1]
                activity_logits = out[:, :, 0]
                pitch_logits = out[:, :, 1:]

                # Activity BCE with pos_weight
                n_pos = act.sum()
                n_neg = act.numel() - n_pos
                pw = min(n_neg / max(n_pos, 1.0), 30.0)
                act_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    activity_logits, act, pos_weight=torch.tensor(pw, device=device))

                # Pitch loss on active frames
                mask = act > 0.5
                if mask.sum() > 0:
                    pit_loss = torch.nn.functional.cross_entropy(
                        pitch_logits[mask], pit[mask], ignore_index=-100)
                else:
                    pit_loss = torch.tensor(0.0, device=device)

                loss = act_loss + pit_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()
                train_loss += loss.item()

            # Validate
            mdl.eval()
            val_loss = 0.0
            tp_a, fp_a, fn_a = 0, 0, 0
            correct_p, total_p = 0, 0
            with torch.no_grad():
                for feats, act, pit in vdl:
                    feats, act, pit = feats.to(device), act.to(device), pit.to(device)
                    out = mdl(feats)
                    a_logits = out[:, :, 0]
                    p_logits = out[:, :, 1:]

                    # Loss
                    n_pos = act.sum()
                    pw = min((act.numel() - n_pos) / max(n_pos, 1.0), 30.0)
                    a_l = torch.nn.functional.binary_cross_entropy_with_logits(
                        a_logits, act, pos_weight=torch.tensor(pw, device=device))
                    mask = act > 0.5
                    p_l = torch.nn.functional.cross_entropy(p_logits[mask], pit[mask], ignore_index=-100) if mask.sum() > 0 else torch.tensor(0.0, device=device)
                    val_loss += (a_l + p_l).item()

                    # F1
                    a_pred = (torch.sigmoid(a_logits) > 0.5).float()
                    tp_a += (a_pred * act).sum().item()
                    fp_a += (a_pred * (1 - act)).sum().item()
                    fn_a += ((1 - a_pred) * act).sum().item()
                    if mask.sum() > 0:
                        correct_p += (p_logits[mask].argmax(-1) == pit[mask]).sum().item()
                        total_p += mask.sum().item()

            n_batches = len(vdl)
            f1 = 2 * tp_a / max(1.0, 2 * tp_a + fp_a + fn_a)
            avg_val = val_loss / n_batches

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | train={train_loss/len(tdl):.4f} val={avg_val:.4f} | "
                      f"F1={f1:.4f} pitch_acc={correct_p/max(1,total_p):.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_val = avg_val
                best_ep = epoch
                best_metrics = {"f1": f1, "pitch_acc": correct_p/max(1,total_p),
                                "val_loss": avg_val, "epoch": epoch}
                patience = 0
                torch.save({"epoch": epoch, "head_state": mdl.state_dict(), "metrics": best_metrics},
                           os.path.join(args.output_dir, "checkpoints", f"best_{name}.pt"))
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"Early stop at epoch {epoch}")
                    break

        results[name] = best_metrics
        print(f"Best {name}: epoch={best_ep}, F1={best_f1:.4f}, pitch_acc={best_metrics['pitch_acc']:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print("HUBERT EXPERIMENT RESULTS")
    print(f"Head type: {args.head_type}")
    print(f"{'='*50}")
    print(f"{'Variant':<20} {'F1':>8} {'Pitch Acc':>10} {'Best Epoch':>10}")
    for name, m in results.items():
        print(f"{name:<20} {m['f1']:>8.4f} {m['pitch_acc']:>10.4f} {m['epoch']:>10}")

    # Compare with baseline (from previous experiments)
    print(f"\nComparison with mel+chroma baseline:")
    print(f"  Baseline (context mel+chroma): F1=0.130")
    for name, m in results.items():
        delta = m['f1'] - 0.130
        print(f"  {name}: F1={m['f1']:.4f} ({delta:+.4f})")

    print(f"\nOutput: {args.output_dir}/")


if __name__ == "__main__":
    main()
