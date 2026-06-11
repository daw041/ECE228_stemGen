"""Note set prediction v2: soft regression + more data.

Key changes from v1:
- Continuous pitch+duration regression (MSE) replaces discrete token CE
- Fixed K_max note slots, temporally ordered, not autoregressive
- 550 tracks + relaxed token filter
- Note-level F1 with greedy matching
"""
import os, sys, random, argparse
import torch, torchaudio, numpy as np, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pretty_midi as pm

_flu_dir = "E:/tools/fluidsynth/bin"
if os.path.isdir(_flu_dir) and _flu_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _flu_dir + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_flu_dir)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.midi.audio_features import AudioFeatureExtractor

K_MAX = 16  # max notes per clip
MIN_PITCH, MAX_PITCH = 28, 60
SPF = 0.02133  # seconds per frame at 46.875Hz


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="dataset/midi_subset")
    p.add_argument("--max-tracks", type=int, default=550)
    p.add_argument("--output-dir", type=str, default="outputs/midi/noteseq_v2")
    p.add_argument("--clip-sec", type=float, default=4.0)
    p.add_argument("--epochs-train", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", type=str, default="transformer",
                   choices=["transformer", "crnn", "hubert"])
    p.add_argument("--use-mix", action="store_true")
    return p.parse_args()


def extract_notes(midi_path, total_dur_sec):
    """Extract notes as [start_time, pitch, duration] with no strict filter."""
    try:
        midi = pm.PrettyMIDI(midi_path)
    except Exception:
        return []
    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            p = note.pitch
            if p < MIN_PITCH or p > MAX_PITCH: continue
            start = max(0.0, note.start)
            end = min(note.end, total_dur_sec)
            dur = end - start
            if dur < 0.03: continue  # 30ms minimum
            notes.append((start, p, dur))
    notes.sort()
    return notes[:K_MAX]


def notes_to_target(notes):
    """Convert notes to target tensor [K_MAX, 3]: pitch_norm, log_dur, valid.

    pitch_norm: (pitch - MIN_PITCH) / (MAX_PITCH - MIN_PITCH)
    log_dur: log2(dur_sec * 10 + 0.01) / 5.0  (roughly -2 to 2 range, scaled to ~0-1)
    valid: 1.0 for real notes, 0.0 for empty slots
    """
    target = np.zeros((K_MAX, 3), dtype=np.float32)
    for i, (start, pitch, dur) in enumerate(notes):
        if i >= K_MAX: break
        target[i, 0] = (pitch - MIN_PITCH) / (MAX_PITCH - MIN_PITCH)  # pitch_norm
        target[i, 1] = np.log2(max(dur, 0.01) * 10 + 0.1) / 4.0  # log_dur scaled
        target[i, 2] = 1.0  # valid note
    return torch.from_numpy(target)


class Encoder(torch.nn.Module):
    def __init__(self, etype, feat_dim, d_model=256, dropout=0.3, device="cuda"):
        super().__init__()
        self.etype = etype
        self.d_model = d_model

        if etype == "transformer":
            self.proj = torch.nn.Linear(feat_dim, d_model)
            el = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=512,
                dropout=dropout, batch_first=True, norm_first=True)
            self.tf = torch.nn.TransformerEncoder(el, num_layers=3)
            self.out_dim = d_model

        elif etype == "crnn":
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv1d(feat_dim, d_model, 5, padding=2),
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

        elif etype == "hubert":
            bundle = torchaudio.pipelines.HUBERT_BASE
            self.hubert = bundle.get_model().to(device)
            self.hubert.eval()
            for p in self.hubert.parameters():
                p.requires_grad = False
            self.hubert_sr = 16000
            self.proj = torch.nn.Linear(768, d_model * 2)
            self.out_dim = d_model * 2

    @torch.no_grad()
    def _hubert_feats(self, wf, sr):
        if wf.dim() == 1: wf = wf.unsqueeze(0)
        if sr != self.hubert_sr:
            wf = torchaudio.functional.resample(wf, sr, self.hubert_sr)
        wf = wf.to(next(self.hubert.parameters()).device)
        out, _ = self.hubert.extract_features(wf)
        return out[-1].squeeze(0)

    def forward(self, feats, wf=None, sr=None):
        if self.etype == "hubert":
            flist = []
            for i in range(len(wf) if isinstance(wf, list) else 1):
                w = wf[i] if isinstance(wf, list) else wf
                s = sr[i] if isinstance(sr, list) else sr
                flist.append(self._hubert_feats(w, s))
            max_t = max(f.shape[0] for f in flist)
            padded = []
            for f in flist:
                if f.shape[0] < max_t:
                    pad = torch.zeros(max_t - f.shape[0], f.shape[1], device=f.device)
                    f = torch.cat([f, pad])
                padded.append(f)
            x = torch.stack(padded)
            x = self.proj(x).mean(dim=1)
        elif self.etype == "transformer":
            x = self.proj(feats)
            x = self.tf(x).mean(dim=1)
        elif self.etype == "crnn":
            x = feats.transpose(1, 2)
            x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
            x = x.transpose(1, 2)
            x, _ = self.gru(x)
            x = x.mean(dim=1)
        return x


class NoteDecoder(torch.nn.Module):
    """Predict K_MAX note slots from encoder output."""
    def __init__(self, enc_dim, hidden=256, k=K_MAX, dropout=0.3):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(enc_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, k * 3),
        )
        self.k = k

    def forward(self, enc_out):
        """Returns [B, K, 3]: pitch_norm, log_dur, confidence"""
        x = self.mlp(enc_out)  # [B, K*3]
        return x.view(-1, self.k, 3)


def note_metrics(pred, target, tol_sec=0.15, tol_pitch=0.1):
    """Compute note-level F1 with greedy matching.

    pred: [K, 3] tensor (pitch_norm, log_dur, conf)
    target: [K, 3] tensor (pitch_norm, log_dur, valid)
    """
    pn, pd = pred.cpu().numpy(), target.cpu().numpy()

    # Get valid preds (conf > 0.5) and ground truth (valid > 0.5)
    pred_notes = []
    for i in range(K_MAX):
        if pn[i, 2] > 0.5:
            pitch = pn[i, 0] * (MAX_PITCH - MIN_PITCH) + MIN_PITCH
            dur_raw = 2 ** (pn[i, 1] * 4.0) / 10
            dur = max(0.03, min(dur_raw, 4.0))
            pred_notes.append((pitch, dur))

    gt_notes = []
    for i in range(K_MAX):
        if pd[i, 2] > 0.5:
            pitch = pd[i, 0] * (MAX_PITCH - MIN_PITCH) + MIN_PITCH
            dur_raw = 2 ** (pd[i, 1] * 4.0) / 10
            dur = max(0.03, min(dur_raw, 4.0))
            gt_notes.append((pitch, dur))

    if not pred_notes and not gt_notes:
        return 1.0, 1.0, 1.0
    if not pred_notes or not gt_notes:
        return 0.0, 0.0, 0.0

    # Greedy matching by pitch distance
    matched = 0
    used = set()
    for pp, pdur in pred_notes:
        best_j, best_dist = -1, float("inf")
        for j, (gp, gdur) in enumerate(gt_notes):
            if j in used: continue
            d = abs(pp - gp)
            if d < best_dist:
                best_dist = d; best_j = j
        if best_j >= 0 and best_dist <= tol_pitch * (MAX_PITCH - MIN_PITCH):
            matched += 1
            used.add(best_j)

    prec = matched / len(pred_notes)
    rec = matched / len(gt_notes)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return f1, prec, rec


def build_clips_from_tracks(track_dirs, clip_sec, seed):
    clips = []
    for track_dir in tqdm(track_dirs, desc="Building clips", leave=False):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists(): continue
        with open(meta_path) as f: meta = yaml.safe_load(f)
        bass_id = next((sid for sid, info in meta.get("stems", {}).items()
                         if info.get("inst_class", "").lower() == "bass"
                         and info.get("midi_saved", False)), None)
        if not bass_id: continue
        midi_path = track_dir / "MIDI" / f"{bass_id}.mid"
        if not midi_path.exists(): continue
        for ext in [".flac", ".wav"]:
            mix_p = track_dir / f"mix{ext}"
            bass_p = track_dir / f"{bass_id}{ext}"
            if mix_p.exists() and bass_p.exists(): break
        else: continue

        info = torchaudio.info(str(mix_p))
        sr, total = info.sample_rate, info.num_frames
        clip_samps = int(clip_sec * sr)
        n_clips = min(8, max(2, int(total / sr / clip_sec)))

        for _ in range(n_clips):
            start = random.randint(0, max(0, total - clip_samps))
            mix, _ = torchaudio.load(str(mix_p), frame_offset=start, num_frames=clip_samps)
            bass, _ = torchaudio.load(str(bass_p), frame_offset=start, num_frames=clip_samps)
            if mix.abs().max() < 0.005: continue
            mix_m = mix.mean(dim=0, keepdim=True)
            bass_m = bass.mean(dim=0, keepdim=True)
            ctx = mix_m - bass_m
            ctx = ctx / max(ctx.abs().max(), 0.01)

            notes = extract_notes(midi_path, float(clip_samps / sr))
            target = notes_to_target(notes)

            clips.append({
                "context": ctx, "context_sr": sr, "mix": mix_m,
                "target": target, "track": track_dir.name,
            })
    return clips


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_dim = 143
    fe = None
    if args.encoder != "hubert":
        fe = AudioFeatureExtractor(
            sample_rate=24000, hop_length=512, use_mel=True, use_chroma=True,
            use_onset=False, use_energy=True,
            use_mix=args.use_mix, dual_channel=False)
        feature_dim = fe.feature_dim

    cache_path = os.path.join(args.output_dir, f"noteseq_v2_{args.encoder}.pt")
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
    else:
        # Process in chunks to avoid OOM
        tracks_all = sorted(Path(args.data_root).glob("Track*"))
        if args.max_tracks and len(tracks_all) > args.max_tracks:
            tracks_all = tracks_all[:args.max_tracks]
        split = int(len(tracks_all) * 0.85)
        train_tracks = set(t.name for t in tracks_all[:split])

        cache = []
        chunk_size = 100
        for chunk_start in range(0, len(tracks_all), chunk_size):
            chunk_tracks = tracks_all[chunk_start:chunk_start + chunk_size]
            clips = build_clips_from_tracks(chunk_tracks, args.clip_sec, args.seed)
            print(f"  Chunk {chunk_start//chunk_size + 1}: {len(clips)} clips from {len(chunk_tracks)} tracks")

            for clip in tqdm(clips, desc="Caching chunk", leave=False):
                entry = {
                    "ctx": torch.zeros(1), "ctx_sr": 0, "mix": torch.zeros(1),
                    "target": clip["target"],
                    "is_train": clip["track"] in train_tracks,
                }
                if args.encoder != "hubert":
                    mw = clip["mix"] if args.use_mix else None
                    f = fe(clip["context"], clip["context_sr"], mix_waveform=mw)
                    entry["features"] = f
                cache.append(entry)
            torch.save(cache, cache_path + ".tmp")
            del clips
        torch.save(cache, cache_path)
        print(f"Tracks: {len(tracks_all)}, Total clips: {len(cache)}")

    train_set = [c for c in cache if c["is_train"]]
    val_set = [c for c in cache if not c["is_train"]]
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    encoder = Encoder(args.encoder, feature_dim, dropout=args.dropout, device=device).to(device)
    decoder = NoteDecoder(encoder.out_dim, dropout=args.dropout).to(device)
    print(f"Params: enc={sum(p.numel() for p in encoder.parameters()):,}, "
          f"dec={sum(p.numel() for p in decoder.parameters()):,}")

    class DS(torch.utils.data.Dataset):
        def __init__(self, c): self.c = c
        def __len__(self): return len(self.c)
        def __getitem__(self, i):
            c = self.c[i]
            return (c.get("features", torch.zeros(1)), c["ctx"], c["ctx_sr"],
                    c["target"], c["mix"])

    def collate(batch):
        feats, ctxs, srs, targets, mixes = zip(*batch)
        return (torch.stack(feats), list(ctxs), list(srs),
                torch.stack(targets), list(mixes))

    train_dl = torch.utils.data.DataLoader(DS(train_set), batch_size=args.batch_size,
                                            shuffle=True, collate_fn=collate, drop_last=True)
    val_dl = torch.utils.data.DataLoader(DS(val_set), batch_size=args.batch_size,
                                          collate_fn=collate, drop_last=True)

    params = [p for p in list(encoder.parameters()) + list(decoder.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0; best_ep = 1; patience = 0

    for epoch in range(1, args.epochs_train + 1):
        encoder.train(); decoder.train()
        total_loss = 0.0
        for feats, ctxs, srs, targets, mixes in train_dl:
            feats = feats.to(device); targets = targets.to(device)
            opt.zero_grad()

            if args.encoder == "hubert":
                enc_out = encoder(None, wf=ctxs, sr=srs)
            else:
                enc_out = encoder(feats)

            pred = decoder(enc_out)  # [B, K, 3]

            # Soft regression loss
            pitch_loss = torch.nn.functional.smooth_l1_loss(
                pred[:, :, 0], targets[:, :, 0], reduction="none")
            dur_loss = torch.nn.functional.smooth_l1_loss(
                pred[:, :, 1], targets[:, :, 1], reduction="none")
            conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred[:, :, 2], targets[:, :, 2], reduction="none")

            # Weight by validity: valid notes contribute fully, invalid contribute less
            valid_mask = targets[:, :, 2]  # [B, K]
            loss = (pitch_loss + dur_loss).mean(dim=-1) * valid_mask * 2.0  # weight active more
            loss = loss + conf_loss.mean(dim=-1) * 0.5  # conf loss on all slots
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()

        # Validate
        encoder.eval(); decoder.eval()
        val_loss = 0.0
        all_f1 = []
        with torch.no_grad():
            for feats, ctxs, srs, targets, mixes in val_dl:
                feats = feats.to(device); targets = targets.to(device)
                if args.encoder == "hubert":
                    enc_out = encoder(None, wf=ctxs, sr=srs)
                else:
                    enc_out = encoder(feats)
                pred = decoder(enc_out)
                # Loss
                pl = torch.nn.functional.smooth_l1_loss(pred[:, :, 0], targets[:, :, 0], reduction="none")
                dl = torch.nn.functional.smooth_l1_loss(pred[:, :, 1], targets[:, :, 1], reduction="none")
                cl = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, :, 2], targets[:, :, 2], reduction="none")
                vm = targets[:, :, 2]
                l = (pl + dl).mean(dim=-1) * vm * 2.0 + cl.mean(dim=-1) * 0.5
                val_loss += l.mean().item()

                # Note F1
                for b in range(len(pred)):
                    f1, _, _ = note_metrics(pred[b], targets[b])
                    all_f1.append(f1)

        avg_f1 = np.mean(all_f1) if all_f1 else 0
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={total_loss/len(train_dl):.4f} "
                  f"val={val_loss/len(val_dl):.4f} | note_f1={avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1; best_ep = epoch; patience = 0
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(), "f1": best_f1},
                       os.path.join(args.output_dir, "checkpoints", "best.pt"))
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}"); break

    frame_f1_baseline = {"transformer": 0.130, "crnn": 0.228, "hubert": 0.289}
    base = frame_f1_baseline.get(args.encoder, 0.130)
    print(f"\nEncoder: {args.encoder} | Best epoch: {best_ep} | Note F1: {best_f1:.4f}")
    print(f"Frame-level F1: {base:.4f} | Note-seq v2 F1: {best_f1:.4f} ({best_f1 - base:+.4f})")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
