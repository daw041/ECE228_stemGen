"""Extract bass MIDI tracks from archive.zip for incremental data-scale experiments.

Prioritizes by split: train -> validation -> test -> omitted.
Supports incremental extraction (skips already-extracted tracks).
"""
import zipfile
import re
import os
import sys
import yaml
import argparse
from tqdm import tqdm


ARCHIVE = "E:/project/stemgen/dataset/archive.zip"
OUT_BASE = "E:/project/stemgen/dataset/midi_subset"

SPLIT_PRIORITY = ["train", "validation", "test", "omitted"]


def find_bass_midi_tracks(archive_path):
    """Scan archive for tracks with bass MIDI. Returns list of (subdir, track_id, bass_stem_id)."""
    z = zipfile.ZipFile(archive_path, "r")
    names = set(z.namelist())

    track_bass = {}
    for name in names:
        m = re.match(r"slakh2100_flac_redux/([^/]+)/Track(\d+)/", name)
        if not m:
            continue
        subdir, track_id = m.group(1), m.group(2)
        key = (subdir, track_id)
        if key in track_bass:
            continue

        meta_path = f"slakh2100_flac_redux/{subdir}/Track{track_id}/metadata.yaml"
        if meta_path not in names:
            continue

        try:
            meta = yaml.safe_load(z.read(meta_path))
        except Exception:
            continue

        for sid, info in meta.get("stems", {}).items():
            if info.get("inst_class", "").lower() != "bass":
                continue
            if not info.get("midi_saved", False):
                continue
            midi_path = f"slakh2100_flac_redux/{subdir}/Track{track_id}/MIDI/{sid}.mid"
            if midi_path not in names:
                continue
            track_bass[key] = sid
            break

    z.close()

    # Sort by split priority, then by track_id
    result = []
    for subdir in SPLIT_PRIORITY:
        for (sd, tid), bass_id in sorted(track_bass.items(), key=lambda x: int(x[0][1])):
            if sd == subdir:
                result.append((subdir, tid, bass_id))

    return result


def extract_track(z, subdir, track_id, bass_id, out_dir, dry_run=False):
    """Extract one track's files from the archive. Returns total bytes extracted."""
    prefix = f"slakh2100_flac_redux/{subdir}/Track{track_id}/"
    needed = {
        "metadata.yaml": prefix + "metadata.yaml",
        "mix.flac": prefix + "mix.flac",
        f"{bass_id}.flac": prefix + f"stems/{bass_id}.flac",
        f"{bass_id}.mid": prefix + f"MIDI/{bass_id}.mid",
    }

    names = set(z.namelist())
    for fname, apath in needed.items():
        if apath not in names:
            return 0, f"Missing: {apath}"

    track_dir = os.path.join(out_dir, f"Track{track_id}")
    if os.path.exists(track_dir):
        return 0, "Already extracted"

    if dry_run:
        sizes = {fname: z.getinfo(apath).file_size for fname, apath in needed.items()}
        total = sum(sizes.values())
        return total, f"Would extract {total/1e6:.1f}MB (dry run)"

    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(os.path.join(track_dir, "MIDI"), exist_ok=True)

    total_bytes = 0
    for fname, apath in needed.items():
        data = z.read(apath)
        total_bytes += len(data)
        if fname.endswith(".mid"):
            out_path = os.path.join(track_dir, "MIDI", fname)
        else:
            out_path = os.path.join(track_dir, fname)
        with open(out_path, "wb") as f:
            f.write(data)

    return total_bytes, "OK"


def main():
    parser = argparse.ArgumentParser(description="Extract bass MIDI tracks from archive")
    parser.add_argument("--n-tracks", type=int, required=True, help="Number of tracks to extract")
    parser.add_argument("--out-dir", type=str, default=OUT_BASE, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only count, don't extract")
    parser.add_argument("--start-from", type=int, default=0, help="Start from track index N")
    args = parser.parse_args()

    print(f"Scanning archive for bass MIDI tracks...")
    tracks = find_bass_midi_tracks(ARCHIVE)
    print(f"Found {len(tracks)} tracks with bass MIDI")

    z = zipfile.ZipFile(ARCHIVE, "r")
    os.makedirs(args.out_dir, exist_ok=True)

    extracted = 0
    skipped = 0
    total_bytes = 0

    target = min(args.n_tracks, len(tracks))
    for i in tqdm(range(args.start_from, target), desc="Extracting"):
        subdir, track_id, bass_id = tracks[i]
        n_bytes, status = extract_track(z, subdir, track_id, bass_id, args.out_dir, args.dry_run)
        if status == "OK":
            extracted += 1
            total_bytes += n_bytes
        elif "Already" in status:
            skipped += 1
        else:
            # Track is missing files - skip and try to maintain count
            pass

        if extracted >= args.n_tracks:
            break

    z.close()

    print(f"\nDone! Extracted: {extracted}, Skipped: {skipped}, Total: {total_bytes/1e6:.1f}MB")
    print(f"Output: {args.out_dir}")

    # Count what's actually on disk
    existing = [
        d for d in os.listdir(args.out_dir)
        if os.path.isdir(os.path.join(args.out_dir, d)) and d.startswith("Track")
    ]
    print(f"Tracks on disk: {len(existing)}")


if __name__ == "__main__":
    main()
