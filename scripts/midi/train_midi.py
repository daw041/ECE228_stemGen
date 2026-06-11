"""MIDI-Transformer bass generation pipeline: features -> MIDI -> audio.

Enhanced with:
- Feature caching (pre-compute once, reuse across epochs)
- Track-level train/val split (no data leakage)
- Enhanced metrics: activity accuracy/precision/recall/F1, pitch accuracy, active ratio
- Auto-save piano-roll comparison + loss curves
"""
import os
import sys
import random
import time
import argparse
import torch
import torchaudio
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Ensure fluidsynth DLL is findable
_flu_dir = "E:/tools/fluidsynth/bin"
if os.path.isdir(_flu_dir) and _flu_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _flu_dir + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_flu_dir)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.midi.audio_features import AudioFeatureExtractor
from src.midi.midi_labels import MidiLabelExtractor
from src.midi.midi_transformer import MidiTransformer
from src.midi.postprocess import predictions_to_midi, midi_to_audio


def get_args():
    p = argparse.ArgumentParser(description="MIDI Transformer Training")
    # Data
    p.add_argument("--data-root", type=str, default="dataset/midi_subset",
                   help="Root directory for extracted tracks")
    p.add_argument("--max-tracks", type=int, default=50,
                   help="Max number of tracks to use")
    p.add_argument("--output-dir", type=str, default="outputs/midi/phase1",
                   help="Output directory for checkpoints and figures")

    # Clip config
    p.add_argument("--clip-sec", type=float, default=4.0,
                   help="Clip duration in seconds")
    p.add_argument("--max-clips-per-track", type=int, default=8,
                   help="Max clips per track")

    # Training config
    p.add_argument("--overfit-clips", type=int, default=8,
                   help="Number of clips for overfitting")
    p.add_argument("--epochs-overfit", type=int, default=50,
                   help="Max epochs for overfit phase")
    p.add_argument("--epochs-train", type=int, default=50,
                   help="Max epochs for full training")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--early-stop-patience", type=int, default=15,
                   help="Early stop patience (epochs)")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay")
    p.add_argument("--seed", type=int, default=42)

    # Feature config
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-chroma", type=int, default=12)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)

    # MIDI config
    p.add_argument("--min-pitch", type=int, default=28)
    p.add_argument("--max-pitch", type=int, default=60)

    # Model config
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)

    # Experiment flags
    p.add_argument("--use-mix", action="store_true",
                   help="Use mix audio (with bass) instead of context for features")
    p.add_argument("--dual-channel", action="store_true",
                   help="Concatenate context + mix features (doubles feature dim)")
    p.add_argument("--no-energy", action="store_true",
                   help="Disable bass-band energy features")
    p.add_argument("--use-cqt", action="store_true",
                   help="Use CQT instead of mel spectrogram")

    # Skip phases
    p.add_argument("--skip-overfit", action="store_true",
                   help="Skip overfit phase (if already done)")
    p.add_argument("--use-cache", action="store_true",
                   help="Use cached features if available")
    return p.parse_args()


def find_bass_midi(track_dir):
    """Find bass MIDI file for a track."""
    meta_path = track_dir / "metadata.yaml"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    bass_id = None
    for stem_id, info in meta.get("stems", {}).items():
        if info.get("inst_class", "").lower() == "bass" and info.get("midi_saved", False):
            bass_id = stem_id
            break
    if bass_id is None:
        return None

    for midi_subdir in ["MIDI", "midi", ""]:
        midi_path = track_dir / midi_subdir / f"{bass_id}.mid"
        if midi_path.exists():
            break
    if not midi_path.exists():
        return None

    for ext in [".flac", ".wav"]:
        mix_path = track_dir / f"mix{ext}"
        if mix_path.exists():
            bass_audio_path = None
            for loc in [track_dir / f"{bass_id}{ext}", track_dir / "stems" / f"{bass_id}{ext}"]:
                if loc.exists():
                    bass_audio_path = loc
                    break
            if bass_audio_path:
                return bass_id, midi_path, mix_path, bass_audio_path
    return None


def build_clips(data_root, max_tracks, clip_sec, max_clips_per_track, seed):
    """Build context-target pairs with MIDI labels, organized by track."""
    tracks = sorted(Path(data_root).glob("Track*"))
    if max_tracks and len(tracks) > max_tracks:
        tracks = tracks[:max_tracks]
    print(f"Scanning {len(tracks)} tracks...")

    track_clips = {}  # track_name -> list of clips
    skipped = 0

    for track_dir in tqdm(tracks, desc="Building clips"):
        result = find_bass_midi(track_dir)
        if result is None:
            skipped += 1
            continue
        bass_id, midi_path, mix_path, bass_audio_path = result

        info = torchaudio.info(str(mix_path))
        sr = info.sample_rate
        total_frames = info.num_frames
        duration_sec = total_frames / sr
        clip_samples = int(clip_sec * sr)

        n_clips = min(max_clips_per_track, max(2, int(duration_sec / clip_sec)))
        clips = []
        for _ in range(n_clips):
            if total_frames <= clip_samples:
                start = 0
            else:
                start = random.randint(0, total_frames - clip_samples)

            mix_audio, _ = torchaudio.load(str(mix_path), frame_offset=start, num_frames=clip_samples)
            if mix_audio.abs().max() < 0.005:
                continue

            mix_mono = mix_audio.mean(dim=0, keepdim=True)

            bass_audio, _ = torchaudio.load(str(bass_audio_path), frame_offset=start, num_frames=clip_samples)
            bass_mono = bass_audio.mean(dim=0, keepdim=True)

            context = mix_mono - bass_mono
            context = context / max(context.abs().max(), 0.01)

            clips.append({
                "context": context,
                "context_sr": sr,
                "mix": mix_mono,
                "midi_path": str(midi_path),
                "total_duration": float(clip_samples / sr),
            })

        if clips:
            track_clips[track_dir.name] = clips

    all_clips = []
    for tname, clips in track_clips.items():
        for c in clips:
            c["track"] = tname
            all_clips.append(c)

    print(f"  Tracks with bass MIDI: {len(track_clips)}, Total clips: {len(all_clips)}, Skipped: {skipped}")
    return all_clips, list(track_clips.keys())


def compute_features_cache(clips, feature_extractor, label_extractor, cache_path, use_mix=False, dual_channel=False):
    """Pre-compute all features and MIDI labels, save to cache file."""
    print(f"Computing feature cache ({len(clips)} clips)...")
    cache = []
    for clip in tqdm(clips, desc="Caching features"):
        mix_wf = clip.get("mix") if (use_mix or dual_channel) else None
        feats = feature_extractor(clip["context"], input_sr=clip["context_sr"],
                                  mix_waveform=mix_wf)
        active_label, pitch_label = label_extractor.extract(
            clip["midi_path"], clip["total_duration"])
        min_len = min(feats.shape[0], len(active_label), len(pitch_label))
        cache.append({
            "features": feats[:min_len],
            "active_label": active_label[:min_len],
            "pitch_label": pitch_label[:min_len],
            "instrument": torch.tensor(0, dtype=torch.long),
            "track": clip["track"],
        })
    torch.save(cache, cache_path)
    print(f"  Feature cache saved: {cache_path} ({len(cache)} clips)")
    return cache


def load_features_cache(cache_path):
    print(f"Loading feature cache from {cache_path}...")
    return torch.load(cache_path, weights_only=False)


class MidiClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        return {
            "features": clip["features"],
            "active_label": clip["active_label"],
            "pitch_label": clip["pitch_label"],
            "instrument": clip["instrument"],
        }


def train_epoch(model, dataloader, optimizer, device, lambda_active=1.0, lambda_pitch=1.0, ignore_index=-100):
    model.train()
    total_loss = 0.0
    total_active_loss = 0.0
    total_pitch_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        feats = batch["features"].to(device)
        active_lab = batch["active_label"].to(device)
        pitch_lab = batch["pitch_label"].to(device)
        inst = batch["instrument"].to(device)

        optimizer.zero_grad()
        loss, act_loss, pit_loss = model.compute_loss(
            feats, inst, active_lab, pitch_lab,
            lambda_active=lambda_active, lambda_pitch=lambda_pitch,
            ignore_index=ignore_index,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_active_loss += act_loss
        total_pitch_loss += pit_loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "act": f"{act_loss:.3f}", "pit": f"{pit_loss:.3f}"})

    n = max(1, len(dataloader))
    return total_loss / n, total_active_loss / n, total_pitch_loss / n


@torch.no_grad()
def evaluate(model, dataloader, device, activity_threshold=0.5, ignore_index=-100):
    """Enhanced evaluation with activity precision/recall/F1 and pitch accuracy."""
    model.eval()
    total_loss = 0.0

    # Activity metrics
    tp_active = 0  # true positive: pred active, label active
    fp_active = 0  # false positive: pred active, label inactive
    fn_active = 0  # false negative: pred inactive, label active
    tn_active = 0  # true negative: pred inactive, label inactive

    # Pitch accuracy on active frames
    correct_pitch = 0
    total_active = 0

    # For tracking predicted active ratio
    total_pred_active = 0
    total_frames = 0

    for batch in dataloader:
        feats = batch["features"].to(device)
        active_lab = batch["active_label"].to(device)
        pitch_lab = batch["pitch_label"].to(device)
        inst = batch["instrument"].to(device)

        loss, _, _ = model.compute_loss(
            feats, inst, active_lab, pitch_lab,
            lambda_active=1.0, lambda_pitch=1.0,
            ignore_index=ignore_index,
        )
        total_loss += loss.item()

        active_prob, pitch_pred = model.predict(feats, inst, activity_threshold=activity_threshold)
        active_pred = active_prob > activity_threshold

        # Activity confusion matrix
        lab_active = active_lab > 0.5
        tp_active += (active_pred & lab_active).float().sum().item()
        fp_active += (active_pred & ~lab_active).float().sum().item()
        fn_active += (~active_pred & lab_active).float().sum().item()
        tn_active += (~active_pred & ~lab_active).float().sum().item()

        total_pred_active += active_pred.float().sum().item()
        total_frames += active_lab.numel()

        # Pitch accuracy
        mask = lab_active
        if mask.sum() > 0:
            correct_pitch += (pitch_pred[mask] == pitch_lab[mask]).float().sum().item()
            total_active += mask.sum().item()

    n = max(1, len(dataloader))

    # Activity metrics
    activity_accuracy = (tp_active + tn_active) / max(1, tp_active + tn_active + fp_active + fn_active)
    activity_precision = tp_active / max(1, tp_active + fp_active)
    activity_recall = tp_active / max(1, tp_active + fn_active)
    activity_f1 = 2 * activity_precision * activity_recall / max(1e-8, activity_precision + activity_recall)

    # Pitch accuracy
    pitch_accuracy = correct_pitch / max(1, total_active)

    # Active ratio
    active_ratio = total_pred_active / max(1, total_frames)
    label_active_ratio = (tp_active + fn_active) / max(1, total_frames)

    return {
        "loss": total_loss / n,
        "activity_accuracy": activity_accuracy,
        "activity_precision": activity_precision,
        "activity_recall": activity_recall,
        "activity_f1": activity_f1,
        "pitch_accuracy": pitch_accuracy,
        "active_ratio": active_ratio,
        "label_active_ratio": label_active_ratio,
        "tp": tp_active, "fp": fp_active, "fn": fn_active, "tn": tn_active,
    }


def save_pianoroll_comparison(model, sample_clip, output_dir, device,
                               min_pitch, max_pitch, seconds_per_frame, activity_threshold=0.5):
    """Save target vs generated piano-roll comparison plot (uses pre-computed features from cache)."""
    num_pitches = max_pitch - min_pitch + 1
    features = sample_clip["features"].unsqueeze(0).to(device)
    inst = torch.tensor([0], device=device)

    active_prob, pitch_pred = model.predict(features, inst, activity_threshold=activity_threshold)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    T = features.shape[1]

    target_pr = np.zeros((num_pitches, T))
    for t in range(min(T, len(sample_clip["active_label"]))):
        if sample_clip["active_label"][t] > 0:
            pid = int(sample_clip["pitch_label"][t])
            if 0 <= pid < num_pitches:
                target_pr[pid, t] = 1.0
    axes[0].imshow(target_pr, aspect="auto", origin="lower", cmap="Blues",
                   extent=[0, T * seconds_per_frame, min_pitch, max_pitch])
    axes[0].set_title("Target Bass Piano-roll"); axes[0].set_ylabel("MIDI Pitch")

    gen_pr = np.zeros((num_pitches, T))
    ap = active_prob[0].cpu().numpy()
    pp = pitch_pred[0].cpu().numpy()
    for t in range(min(T, len(ap))):
        if ap[t] > activity_threshold:
            pid = pp[t]
            if 0 <= pid < num_pitches:
                gen_pr[pid, t] = 1.0
    axes[1].imshow(gen_pr, aspect="auto", origin="lower", cmap="Reds",
                   extent=[0, T * seconds_per_frame, min_pitch, max_pitch])
    axes[1].set_title("Generated Bass Piano-roll"); axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("MIDI Pitch")
    plt.tight_layout()
    path = os.path.join(output_dir, "figures", "pianoroll_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved piano-roll comparison")
    return active_prob, pitch_pred, T


def save_loss_curves(train_losses, val_losses, val_metrics, output_dir):
    """Save loss and metric curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0, 0].plot(epochs, train_losses, label="Train", color="blue")
    if val_losses:
        axes[0, 0].plot(epochs, val_losses, label="Val", color="red")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curve"); axes[0, 0].legend(); axes[0, 0].grid(True)

    # Activity accuracy
    if val_metrics:
        act_accs = [m["activity_accuracy"] for m in val_metrics]
        axes[0, 1].plot(epochs, act_accs, color="green")
        axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Activity Accuracy"); axes[0, 1].grid(True)

    # Activity F1
    if val_metrics:
        act_f1s = [m["activity_f1"] for m in val_metrics]
        axes[1, 0].plot(epochs, act_f1s, color="purple")
        axes[1, 0].axhline(y=0.8, color="gray", linestyle="--", label="Target 0.80")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("F1")
        axes[1, 0].set_title("Activity F1"); axes[1, 0].legend(); axes[1, 0].grid(True)

    # Pitch accuracy
    if val_metrics:
        pit_accs = [m["pitch_accuracy"] for m in val_metrics]
        axes[1, 1].plot(epochs, pit_accs, color="orange")
        axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_title("Pitch Accuracy (Active Frames)"); axes[1, 1].grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "figures", "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved loss curves")


def generate_midi_audio(model, sample_clip, output_dir, device,
                         min_pitch, max_pitch, seconds_per_frame, sample_rate,
                         activity_threshold=0.5):
    """Generate MIDI from model using pre-computed features from cache."""
    features = sample_clip["features"].unsqueeze(0).to(device)
    inst = torch.tensor([0], device=device)

    active_prob, pitch_pred = model.predict(features, inst, activity_threshold=activity_threshold)

    midi = predictions_to_midi(
        active_prob[0], pitch_pred[0], instrument_idx=0,
        instrument_pitch_ranges=[(min_pitch, max_pitch)],
        seconds_per_frame=seconds_per_frame,
        activity_threshold=activity_threshold,
    )
    midi_path = os.path.join(output_dir, "generated_bass.mid")
    midi.write(midi_path)
    midi_size = os.path.getsize(midi_path)
    print(f"  Saved generated MIDI: {midi_path} ({midi_size} bytes)")

    # Try to render audio from MIDI
    try:
        gen_audio = midi_to_audio(midi, sample_rate=sample_rate)
        gen_len = gen_audio.shape[-1]
        audio_rms = gen_audio.abs().mean().item()
        torchaudio.save(os.path.join(output_dir, "audio", "generated_bass.wav"), gen_audio, sample_rate)
        print(f"  Saved rendered audio (duration: {gen_len/sample_rate:.1f}s, RMS: {audio_rms:.6f})")
    except Exception as e:
        print(f"  Audio rendering skipped: {e}")

    return midi_size


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Feature & label extractors
    feature_extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft,
        hop_length=args.hop_length, n_chroma=args.n_chroma,
        use_mel=True, use_chroma=True, use_onset=False,
        use_mix=args.use_mix, dual_channel=args.dual_channel,
        use_energy=not args.no_energy, use_cqt=args.use_cqt,
    )
    label_extractor = MidiLabelExtractor(
        min_pitch=args.min_pitch, max_pitch=args.max_pitch, hop_length=args.hop_length,
        sample_rate=args.sample_rate, ignore_index=-100,
    )
    feature_dim = feature_extractor.feature_dim
    num_pitches = args.max_pitch - args.min_pitch + 1
    seconds_per_frame = args.hop_length / args.sample_rate

    print("=" * 60)
    print("MIDI-Transformer Bass Generation Pipeline")
    print(f"  Data root: {args.data_root}")
    print(f"  Max tracks: {args.max_tracks}")
    print(f"  Feature dim: {feature_dim}, Pitches: {num_pitches}")
    print(f"  Frame rate: {1/seconds_per_frame:.1f} Hz, Clip: {args.clip_sec}s")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # ── 1. Build or load clips ──────────────────────────────────────────
    cache_path = os.path.join(args.output_dir, "features_cache.pt")

    if args.use_cache and os.path.exists(cache_path):
        cache = load_features_cache(cache_path)
        print(f"  Loaded {len(cache)} cached clips")
    else:
        print("\n[1/4] Building clips with MIDI labels...")
        all_clips, track_names = build_clips(
            args.data_root, args.max_tracks, args.clip_sec,
            args.max_clips_per_track, args.seed)

        if not all_clips:
            print("  ERROR: No clips with MIDI found.")
            return

        # Track-level split: first 85% tracks for train, last 15% for val
        split_idx = int(len(track_names) * 0.85)
        train_tracks = set(track_names[:split_idx])
        val_tracks = set(track_names[split_idx:])
        print(f"  Track split: {len(train_tracks)} train, {len(val_tracks)} val")

        # Shuffle clips within their respective track sets
        train_clips_raw = [c for c in all_clips if c["track"] in train_tracks]
        val_clips_raw = [c for c in all_clips if c["track"] in val_tracks]
        random.shuffle(train_clips_raw)
        random.shuffle(val_clips_raw)

        print(f"  Train clips: {len(train_clips_raw)}, Val clips: {len(val_clips_raw)}")

        # Compute features cache
        print(f"\n[1.5/4] Computing features cache...")
        cache = compute_features_cache(all_clips, feature_extractor, label_extractor,
                                         cache_path, use_mix=args.use_mix,
                                         dual_channel=args.dual_channel)

    # Split cache by track
    all_cached = cache
    # Re-derive track split
    all_tracks = sorted(set(c["track"] for c in all_cached))
    split_idx = int(len(all_tracks) * 0.85)
    train_tracks = set(all_tracks[:split_idx])
    val_tracks = set(all_tracks[split_idx:])

    overfit_set = [c for c in all_cached if c["track"] in train_tracks][:args.overfit_clips]
    train_set = [c for c in all_cached if c["track"] in train_tracks]
    val_set = [c for c in all_cached if c["track"] in val_tracks]

    print(f"  Overfit: {len(overfit_set)}, Train: {len(train_set)}, Val: {len(val_set)}")
    print(f"  Train tracks: {len(train_tracks)}, Val tracks: {len(val_tracks)}")

    # ── 2. Model ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Building MIDI Transformer...")
    model = MidiTransformer(
        feature_dim=feature_dim, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dim_feedforward=512, dropout=args.dropout, num_instruments=6,
        instrument_embed_dim=64, max_seq_len=2048,
        instrument_pitch_ranges=[(args.min_pitch, args.max_pitch) for _ in range(6)],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params on {device}")

    # ── 3a. Overfit ──────────────────────────────────────────────────────
    best_overfit_path = os.path.join(args.output_dir, "checkpoints", "overfit_best.pt")

    if not args.skip_overfit:
        print(f"\n[3a/4] Overfitting on {len(overfit_set)} clips...")
        overfit_ds = MidiClipDataset(overfit_set)
        overfit_dl = torch.utils.data.DataLoader(overfit_ds, batch_size=min(args.batch_size, len(overfit_set)), shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_loss = float("inf")
        for epoch in range(1, args.epochs_overfit + 1):
            avg_loss, avg_act, avg_pit = train_epoch(model, overfit_dl, optimizer, device)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, best_overfit_path)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{args.epochs_overfit} | loss={avg_loss:.4f} act={avg_act:.4f} pit={avg_pit:.4f}")
                if avg_loss < 0.3:
                    print(f"  Overfit converged early at epoch {epoch}")
                    break
        print(f"  Overfit complete: best_loss={best_loss:.4f}")

    # Load best overfit checkpoint
    if os.path.exists(best_overfit_path):
        ckpt = torch.load(best_overfit_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded overfit checkpoint from epoch {ckpt['epoch']}")

    # ── 3b. Full training ──────────────────────────────────────────────
    print(f"\n[3b/4] Training on {len(train_set)} clips...")
    train_ds = MidiClipDataset(train_set)
    val_ds = MidiClipDataset(val_set) if val_set else None
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses = []
    val_losses = []
    val_metrics = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 1

    for epoch in range(1, args.epochs_train + 1):
        avg_loss, avg_act, avg_pit = train_epoch(model, train_dl, optimizer, device)
        train_losses.append(avg_loss)

        if val_dl:
            metrics = evaluate(model, val_dl, device)
            val_losses.append(metrics["loss"])
            val_metrics.append(metrics)
            print(f"  Epoch {epoch:3d}/{args.epochs_train} | "
                  f"loss={avg_loss:.4f} | vloss={metrics['loss']:.4f} | "
                  f"act_acc={metrics['activity_accuracy']:.3f} | "
                  f"act_f1={metrics['activity_f1']:.3f} | "
                  f"pit_acc={metrics['pitch_accuracy']:.3f} | "
                  f"act_ratio={metrics['active_ratio']:.3f}")

            if metrics["loss"] < best_val_loss - 0.001:
                best_val_loss = metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses, "val_losses": val_losses,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_val_loss, "best_epoch": best_epoch,
                }, os.path.join(args.output_dir, "checkpoints", "best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"  Early stop at epoch {epoch}")
                    break
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs_train} | loss={avg_loss:.4f}")

    # Load best
    best_ckpt_path = os.path.join(args.output_dir, "checkpoints", "best.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best checkpoint: epoch {ckpt['epoch']}")

    # ── 4. Generate & Evaluate ────────────────────────────────────────
    print(f"\n[4/4] Generating bass MIDI and audio...")

    # Save loss curves
    save_loss_curves(train_losses, val_losses, val_metrics, args.output_dir)

    # Piano-roll comparison (uses cached features)
    sample = val_set[0] if val_set else train_set[0]
    save_pianoroll_comparison(
        model, sample, args.output_dir, device,
        args.min_pitch, args.max_pitch, seconds_per_frame)

    # Generate MIDI and audio (uses cached features)
    midi_size = generate_midi_audio(
        model, sample, args.output_dir, device,
        args.min_pitch, args.max_pitch, seconds_per_frame, args.sample_rate)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MIDI-Transformer Training COMPLETE!")
    print(f"  Tracks: {args.max_tracks}, Clips: {len(all_cached)}")
    print(f"  Model: {n_params:,} params, Feature dim: {feature_dim}")
    if val_metrics:
        m = val_metrics[best_epoch - 1]
        print(f"  Best val (epoch {best_epoch}):")
        print(f"    Loss: {best_val_loss:.4f}")
        print(f"    Activity Accuracy: {m['activity_accuracy']:.4f}")
        print(f"    Activity F1: {m['activity_f1']:.4f}")
        print(f"    Activity Precision: {m['activity_precision']:.4f}")
        print(f"    Activity Recall: {m['activity_recall']:.4f}")
        print(f"    Pitch Accuracy (active): {m['pitch_accuracy']:.4f}")
        print(f"    Predicted Active Ratio: {m['active_ratio']:.4f} (label: {m['label_active_ratio']:.4f})")
    print(f"  Output: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
