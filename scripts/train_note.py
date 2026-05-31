"""Note-level Transformer training: onset/offset + pitch prediction.

Key difference from train_midi.py: predicts note boundaries (onset/offset)
instead of per-frame activity, enabling structured note extraction.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.midi.audio_features import AudioFeatureExtractor
from src.midi.midi_labels import MidiLabelExtractor
from src.midi.note_transformer import NoteTransformer
from src.midi.postprocess import predictions_to_midi, midi_to_audio


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/midi_subset")
    p.add_argument("--max-tracks", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="outputs/midi/note")
    p.add_argument("--clip-sec", type=float, default=4.0)
    p.add_argument("--epochs-train", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--min-pitch", type=int, default=28)
    p.add_argument("--max-pitch", type=int, default=60)
    p.add_argument("--use-mix", action="store_true")
    p.add_argument("--dual-channel", action="store_true")
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--use-cache", action="store_true")
    return p.parse_args()


def build_clips(data_root, max_tracks, clip_sec, seed):
    tracks = sorted(Path(data_root).glob("Track*"))
    if max_tracks and len(tracks) > max_tracks:
        tracks = tracks[:max_tracks]

    all_clips = []
    track_names = []
    for track_dir in tqdm(tracks, desc="Building clips"):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        bass_id = None
        for sid, info in meta.get("stems", {}).items():
            if info.get("inst_class", "").lower() == "bass" and info.get("midi_saved", False):
                bass_id = sid
                break
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
        clip_samples = int(clip_sec * sr)
        n_clips = min(8, max(2, int(total_frames / sr / clip_sec)))

        for _ in range(n_clips):
            start = random.randint(0, max(0, total_frames - clip_samples))
            mix, _ = torchaudio.load(str(mix_path), frame_offset=start, num_frames=clip_samples)
            bass, _ = torchaudio.load(str(bass_path), frame_offset=start, num_frames=clip_samples)
            if mix.abs().max() < 0.005:
                continue

            mix_mono = mix.mean(dim=0, keepdim=True)
            bass_mono = bass.mean(dim=0, keepdim=True)
            ctx = mix_mono - bass_mono
            ctx = ctx / max(ctx.abs().max(), 0.01)

            all_clips.append({
                "context": ctx, "context_sr": sr, "mix": mix_mono,
                "midi_path": str(midi_path),
                "total_duration": float(clip_samples / sr),
                "track": track_dir.name,
            })
            track_names.append(track_dir.name)

    return all_clips, sorted(set(track_names))


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = AudioFeatureExtractor(
        sample_rate=args.sample_rate, hop_length=args.hop_length,
        use_mel=True, use_chroma=True, use_onset=False,
        use_energy=not args.no_energy,
        use_mix=args.use_mix, dual_channel=args.dual_channel,
    )
    label_extractor = MidiLabelExtractor(
        min_pitch=args.min_pitch, max_pitch=args.max_pitch,
        hop_length=args.hop_length, sample_rate=args.sample_rate,
    )
    feature_dim = feature_extractor.feature_dim
    num_pitches = args.max_pitch - args.min_pitch + 1
    seconds_per_frame = args.hop_length / args.sample_rate

    print(f"Device: {device}, Feature dim: {feature_dim}, Pitches: {num_pitches}")
    print(f"Mode: use_mix={args.use_mix}, dual={args.dual_channel}")

    # Build clips
    cache_path = os.path.join(args.output_dir, "note_cache.pt")
    if args.use_cache and os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(cache)} cached clips")
    else:
        all_clips, track_names = build_clips(args.data_root, args.max_tracks, args.clip_sec, args.seed)
        print(f"Built {len(all_clips)} clips from {len(track_names)} tracks")

        # Track-level split
        split = int(len(track_names) * 0.85)
        train_tracks = set(track_names[:split])

        print("Computing onset/offset labels...")
        cache = []
        for clip in tqdm(all_clips, desc="Caching"):
            mix_wf = clip.get("mix") if (args.use_mix or args.dual_channel) else None
            feats = feature_extractor(clip["context"], input_sr=clip["context_sr"], mix_waveform=mix_wf)
            onset_lab, offset_lab, pitch_lab = label_extractor.extract_onset_offset(
                clip["midi_path"], clip["total_duration"])
            min_len = min(feats.shape[0], len(onset_lab), len(offset_lab), len(pitch_lab))
            cache.append({
                "features": feats[:min_len],
                "onset_label": onset_lab[:min_len],
                "offset_label": offset_lab[:min_len],
                "pitch_label": pitch_lab[:min_len],
                "instrument": torch.tensor(0, dtype=torch.long),
                "track": clip["track"],
                "is_train": clip["track"] in train_tracks,
            })
        torch.save(cache, cache_path)

    # Train/val split
    train_set = [c for c in cache if c["is_train"]]
    val_set = [c for c in cache if not c["is_train"]]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Build model
    model = NoteTransformer(
        feature_dim=feature_dim, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dim_feedforward=512, dropout=args.dropout, num_instruments=6,
        instrument_embed_dim=64, max_seq_len=2048,
        instrument_pitch_ranges=[(args.min_pitch, args.max_pitch) for _ in range(6)],
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Train
    class ClipDS(torch.utils.data.Dataset):
        def __init__(self, clips):
            self.clips = clips
        def __len__(self):
            return len(self.clips)
        def __getitem__(self, i):
            c = self.clips[i]
            return (c["features"], c["onset_label"], c["offset_label"],
                    c["pitch_label"], c["instrument"])

    train_dl = torch.utils.data.DataLoader(ClipDS(train_set), batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(ClipDS(val_set), batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 1
    patience = 0

    for epoch in range(1, args.epochs_train + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            feats, onset_l, offset_l, pitch_l, inst = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss, on_l, off_l, pit_l = model.compute_loss(feats, inst, onset_l, offset_l, pitch_l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                feats, onset_l, offset_l, pitch_l, inst = [b.to(device) for b in batch]
                loss, _, _, _ = model.compute_loss(feats, inst, onset_l, offset_l, pitch_l)
                val_loss += loss.item()

        avg_train = total_loss / len(train_dl)
        avg_val = val_loss / len(val_dl)
        print(f"Epoch {epoch:3d}/{args.epochs_train} | train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_val - 0.001:
            best_val = avg_val; best_epoch = epoch; patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       os.path.join(args.output_dir, "checkpoints", "best.pt"))
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}"); break

    # Load best
    ckpt = torch.load(os.path.join(args.output_dir, "checkpoints", "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Generate notes from a val sample
    sample = val_set[0] if val_set else train_set[0]
    features = sample["features"].unsqueeze(0).to(device)
    inst = torch.tensor([0], device=device)

    notes = model.predict_notes(features, inst)
    n_notes = len(notes)
    print(f"Predicted {n_notes} notes")

    # Convert to MIDI
    from src.midi.postprocess import predictions_to_midi as p2m
    import pretty_midi as pm

    midi = pm.PrettyMIDI()
    track = pm.Instrument(program=33)
    for start_f, end_f, pid, conf in notes:
        actual_pitch = pid + args.min_pitch
        start_sec = start_f * seconds_per_frame
        end_sec = end_f * seconds_per_frame
        note = pm.Note(velocity=80, pitch=int(actual_pitch), start=start_sec, end=end_sec)
        track.notes.append(note)
    midi.instruments.append(track)

    midi_path = os.path.join(args.output_dir, "generated_bass.mid")
    midi.write(midi_path)
    print(f"Saved MIDI: {midi_path} ({os.path.getsize(midi_path)} bytes, {n_notes} notes)")

    # Save onset piano-roll comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    T = min(features.shape[1], 200)
    for ax_idx, (title, label_key) in enumerate([
        ("Target Onsets", "onset_label"), ("Target Offsets", "offset_label")
    ]):
        data = sample[label_key][:T].cpu().numpy()
        axes[ax_idx].imshow(data.reshape(1, -1), aspect="auto", cmap="Reds",
                            extent=[0, T * seconds_per_frame, 0, 1])
        axes[ax_idx].set_title(title); axes[ax_idx].set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "figures", "onset_comparison.png"), dpi=150)
    plt.close()

    print(f"\nDone! Best epoch: {best_epoch}, Best val loss: {best_val:.4f}")
    print(f"Output: {args.output_dir}/")
    print(f"Predicted notes: {n_notes}")


if __name__ == "__main__":
    main()
