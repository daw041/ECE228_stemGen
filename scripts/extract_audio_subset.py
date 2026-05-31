#!/usr/bin/env python
"""Extract a tiny audio-token subset from Slakh archive.zip.

The script does not unpack the full 100GB archive. It scans metadata, finds
tracks with a rendered bass stem, and writes this compact structure:

dataset/audio_subset/TrackXXXXX/
  metadata.yaml
  mix.flac
  bass.flac

This is enough for `SlakhContextTargetDataset` to build
context = mix - bass for the audio-token StemGen path.
"""
import argparse
import os
import re
import zipfile

import yaml
from tqdm import tqdm


SPLIT_PRIORITY = ["train", "validation", "test", "omitted"]


def find_bass_audio_tracks(archive_path):
    with zipfile.ZipFile(archive_path, "r") as z:
        names = set(z.namelist())
        metadata_paths = [
            n for n in names
            if re.match(r"slakh2100_flac_redux/[^/]+/Track\d+/metadata.yaml$", n)
        ]

        tracks = []
        for meta_path in tqdm(metadata_paths, desc="Scanning metadata"):
            match = re.match(r"slakh2100_flac_redux/([^/]+)/(Track\d+)/metadata.yaml$", meta_path)
            if not match:
                continue
            split, track_name = match.groups()
            try:
                meta = yaml.safe_load(z.read(meta_path))
            except Exception:
                continue

            bass_id = None
            for stem_id, info in meta.get("stems", {}).items():
                if info.get("inst_class", "").lower() == "bass" and info.get("audio_rendered", False):
                    stem_path = f"slakh2100_flac_redux/{split}/{track_name}/stems/{stem_id}.flac"
                    if stem_path in names:
                        bass_id = stem_id
                        break
            if bass_id is None:
                continue

            mix_path = f"slakh2100_flac_redux/{split}/{track_name}/mix.flac"
            if mix_path not in names:
                continue
            tracks.append((split, track_name, bass_id))

    def sort_key(item):
        split, track_name, _ = item
        split_rank = SPLIT_PRIORITY.index(split) if split in SPLIT_PRIORITY else len(SPLIT_PRIORITY)
        track_num = int(track_name.replace("Track", ""))
        return split_rank, track_num

    return sorted(tracks, key=sort_key)


def extract_track(z, split, track_name, bass_id, out_dir, dry_run=False):
    prefix = f"slakh2100_flac_redux/{split}/{track_name}"
    files = {
        "metadata.yaml": f"{prefix}/metadata.yaml",
        "mix.flac": f"{prefix}/mix.flac",
        "bass.flac": f"{prefix}/stems/{bass_id}.flac",
    }
    missing = [path for path in files.values() if path not in z.namelist()]
    if missing:
        return 0, f"missing {missing[0]}"

    total = sum(z.getinfo(path).file_size for path in files.values())
    if dry_run:
        return total, "dry-run"

    track_dir = os.path.join(out_dir, track_name)
    os.makedirs(track_dir, exist_ok=True)
    for out_name, archive_name in files.items():
        with open(os.path.join(track_dir, out_name), "wb") as f:
            f.write(z.read(archive_name))
    return total, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="dataset/archive.zip")
    parser.add_argument("--out_dir", default="dataset/audio_subset")
    parser.add_argument("--n_tracks", type=int, default=8)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tracks = find_bass_audio_tracks(args.archive)
    print(f"Found {len(tracks)} tracks with rendered bass audio")
    selected = tracks[args.start_from:args.start_from + args.n_tracks]
    if not selected:
        raise SystemExit("No tracks selected")

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    with zipfile.ZipFile(args.archive, "r") as z:
        for split, track_name, bass_id in selected:
            n_bytes, status = extract_track(z, split, track_name, bass_id, args.out_dir, args.dry_run)
            total += n_bytes
            print(f"{track_name} [{split}] bass={bass_id}: {status}, {n_bytes / 1e6:.1f} MB")

    print(f"Total {'would extract' if args.dry_run else 'extracted'}: {total / 1e6:.1f} MB")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
