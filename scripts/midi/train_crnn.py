"""CRNN experiment: Conv1D + BiGRU for bass MIDI prediction.

CNN extracts local time-frequency patterns from mel+chroma features.
BiGRU models temporal note dependencies.
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
from src.midi.audio_features import AudioFeatureExtractor
from src.midi.midi_labels import MidiLabelExtractor
from src.midi.postprocess import predictions_to_midi, midi_to_audio


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/midi_subset")
    p.add_argument("--max-tracks", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="outputs/midi/exp_crnn")
    p.add_argument("--clip-sec", type=float, default=4.0)
    p.add_argument("--epochs-train", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-pitch", type=int, default=28)
    p.add_argument("--max-pitch", type=int, default=60)
    p.add_argument("--use-mix", action="store_true")
    p.add_argument("--hidden", type=int, default=256, help="Hidden size for CNN/GRU")
    return p.parse_args()


class CRNN(torch.nn.Module):
    """Conv1D + BiGRU for frame-level activity + pitch prediction."""

    def __init__(self, feature_dim=140, num_pitches=33, hidden=256, dropout=0.3):
        super().__init__()

        # CNN blocks with residual connections
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(feature_dim, hidden, 5, padding=2),
            torch.nn.BatchNorm1d(hidden), torch.nn.GELU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, 5, padding=2),
            torch.nn.BatchNorm1d(hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.BatchNorm1d(hidden), torch.nn.GELU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.BatchNorm1d(hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

        # BiGRU
        self.gru = torch.nn.GRU(hidden, hidden, num_layers=2,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.gru_dropout = torch.nn.Dropout(dropout)

        # Activity head
        self.activity_head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, 128), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, 1),
        )
        # Pitch head
        self.pitch_head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, 128), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, num_pitches),
        )

    def forward(self, x):
        """x: [B, T, D] → activity: [B, T], pitch: [B, T, num_pitches]"""
        # Conv1D expects [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]

        # Residual CNN blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Back to [B, T, D]
        x = x.transpose(1, 2)  # [B, T, hidden]

        # BiGRU
        x, _ = self.gru(x)
        x = self.gru_dropout(x)

        # Heads
        activity = self.activity_head(x).squeeze(-1)  # [B, T]
        pitch = self.pitch_head(x)  # [B, T, num_pitches]

        return activity, pitch


def build_clips(data_root, max_tracks, clip_sec, seed):
    tracks = sorted(Path(data_root).glob("Track*"))
    if max_tracks and len(tracks) > max_tracks:
        tracks = tracks[:max_tracks]

    all_clips = []
    for track_dir in tqdm(tracks, desc="Building clips"):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists(): continue
        with open(meta_path) as f: meta = yaml.safe_load(f)
        bass_id = None
        for sid, info in meta.get("stems", {}).items():
            if info.get("inst_class", "").lower() == "bass" and info.get("midi_saved", False):
                bass_id = sid; break
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
            all_clips.append({
                "context": ctx, "context_sr": sr, "mix": mix_m,
                "midi_path": str(midi_path),
                "total_duration": float(clip_samples / sr),
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

    feature_extractor = AudioFeatureExtractor(
        sample_rate=24000, hop_length=512, use_mel=True, use_chroma=True,
        use_onset=False, use_energy=True,
        use_mix=args.use_mix, dual_channel=False,
    )
    label_extractor = MidiLabelExtractor(
        min_pitch=args.min_pitch, max_pitch=args.max_pitch,
        hop_length=512, sample_rate=24000,
    )
    feature_dim = feature_extractor.feature_dim
    num_pitches = args.max_pitch - args.min_pitch + 1

    cache_path = os.path.join(args.output_dir, "crnn_cache.pt")
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
    else:
        clips = build_clips(args.data_root, args.max_tracks, args.clip_sec, args.seed)
        tracks = sorted(set(c["track"] for c in clips))
        split = int(len(tracks) * 0.85)
        train_tracks = set(tracks[:split])
        print(f"Tracks: {len(tracks)}, Clips: {len(clips)}")
        print(f"Feature dim: {feature_dim}")

        cache = []
        for clip in tqdm(clips, desc="Caching features"):
            mix_wf = clip.get("mix") if args.use_mix else None
            feats = feature_extractor(clip["context"], input_sr=clip["context_sr"], mix_waveform=mix_wf)
            act, pit = label_extractor.extract(clip["midi_path"], clip["total_duration"])
            ml = min(feats.shape[0], len(act), len(pit))
            cache.append({
                "features": feats[:ml], "active_label": act[:ml], "pitch_label": pit[:ml],
                "is_train": clip["track"] in train_tracks, "track": clip["track"],
            })
        torch.save(cache, cache_path)

    train_set = [c for c in cache if c["is_train"]]
    val_set = [c for c in cache if not c["is_train"]]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    model = CRNN(feature_dim=feature_dim, num_pitches=num_pitches,
                 hidden=args.hidden, dropout=args.dropout).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    class DS(torch.utils.data.Dataset):
        def __init__(self, clips): self.clips = clips
        def __len__(self): return len(self.clips)
        def __getitem__(self, i):
            c = self.clips[i]
            return c["features"], c["active_label"], c["pitch_label"]

    train_dl = torch.utils.data.DataLoader(DS(train_set), batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(DS(val_set), batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_f1, best_ep, patience = 0, 1, 0
    best_state = None

    for epoch in range(1, args.epochs_train + 1):
        model.train(); train_loss = 0.0
        for feats, act_l, pit_l in train_dl:
            feats, act_l, pit_l = feats.to(device), act_l.to(device), pit_l.to(device)
            a_logits, p_logits = model(feats)

            n_pos = act_l.sum()
            pw = min((act_l.numel() - n_pos) / max(n_pos, 1.0), 30.0)
            a_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                a_logits, act_l, pos_weight=torch.tensor(pw, device=device))
            mask = act_l > 0.5
            p_loss = torch.nn.functional.cross_entropy(p_logits[mask], pit_l[mask], ignore_index=-100) if mask.sum() > 0 else torch.tensor(0.0, device=device)
            loss = a_loss + p_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        tp_a, fp_a, fn_a, correct_p, total_p = 0, 0, 0, 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for feats, act_l, pit_l in val_dl:
                feats, act_l, pit_l = feats.to(device), act_l.to(device), pit_l.to(device)
                a_logits, p_logits = model(feats)
                n_pos = act_l.sum()
                pw = min((act_l.numel() - n_pos) / max(n_pos, 1.0), 30.0)
                val_loss += (torch.nn.functional.binary_cross_entropy_with_logits(a_logits, act_l, pos_weight=torch.tensor(pw, device=device))).item()
                mask = act_l > 0.5
                if mask.sum() > 0:
                    val_loss += torch.nn.functional.cross_entropy(p_logits[mask], pit_l[mask], ignore_index=-100).item()
                a_pred = (torch.sigmoid(a_logits) > 0.5).float()
                tp_a += (a_pred * act_l).sum().item()
                fp_a += (a_pred * (1 - act_l)).sum().item()
                fn_a += ((1 - a_pred) * act_l).sum().item()
                if mask.sum() > 0:
                    correct_p += (p_logits[mask].argmax(-1) == pit_l[mask]).sum().item()
                    total_p += mask.sum().item()

        f1 = 2 * tp_a / max(1.0, 2 * tp_a + fp_a + fn_a)
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={train_loss/len(train_dl):.4f} val={val_loss/len(val_dl):.4f} | F1={f1:.4f} pit_acc={correct_p/max(1,total_p):.4f}")

        if f1 > best_f1:
            best_f1 = f1; best_ep = epoch; patience = 0
            best_state = {"epoch": epoch, "f1": f1, "pit_acc": correct_p/max(1,total_p), "state": model.state_dict()}
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}"); break

    if best_state:
        torch.save(best_state, os.path.join(args.output_dir, "checkpoints", "best.pt"))

    variant = "CRNN-mix" if args.use_mix else "CRNN-ctx"
    print(f"\nBest {variant}: epoch={best_ep}, F1={best_f1:.4f}, pit_acc={best_state['pit_acc']:.4f}")
    print(f"vs baseline (mel+chroma Transformer): 0.130 → {best_f1:.4f} ({best_f1-0.130:+.4f})")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
