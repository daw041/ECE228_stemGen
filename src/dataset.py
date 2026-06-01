"""Slakh2100 context-target dataset construction."""
import os
import random
import math
import glob
import torch
from torch.utils.data import Dataset
import torchaudio
import yaml


INSTRUMENT_MAP = {
    "bass": 0, "drums": 1, "piano": 2,
    "guitar": 3, "strings": 4, "other": 5,
}


class SlakhContextTargetDataset(Dataset):
    """Build context-target stem pairs from Slakh2100.

    Phase 1: context = mixture - target, target = bass only.
    Interface supports future random_stem_subset mode.
    """

    def __init__(
        self,
        data_root: str,
        target_instrument: str = "bass",
        context_mode: str = "mixture_minus_target",
        clip_duration: float = 4.0,
        sample_rate: int = 24000,
        split: str = "train",
        max_clips: int = 100,
        min_target_rms_db: float = None,
        min_target_active_ratio: float = 0.0,
        target_active_threshold: float = 1e-4,
        max_clip_resample_attempts: int = 20,
    ):
        self.data_root = data_root
        self.target_instrument = target_instrument.lower()
        self.context_mode = context_mode
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.split = split
        self.max_clips = max_clips
        self.min_target_rms_db = min_target_rms_db
        self.min_target_active_ratio = float(min_target_active_ratio or 0.0)
        self.target_active_threshold = float(target_active_threshold)
        self.max_clip_resample_attempts = max(1, int(max_clip_resample_attempts))

        self.track_dirs = self._scan_tracks()
        self.clip_samples = int(clip_duration * sample_rate)

    def _scan_tracks(self):
        if not os.path.isdir(self.data_root):
            return []
        track_dirs = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])
        # simple train/val split: first 80% train, last 20% val
        n = len(track_dirs)
        split_idx = int(n * 0.8)
        if self.split == "train":
            return track_dirs[:split_idx]
        else:
            return track_dirs[split_idx:]

    def __len__(self):
        if not self.track_dirs:
            return 0
        return self.max_clips

    def _stem_path(self, track_dir: str, stem_type: str):
        stem_path = os.path.join(track_dir, f"{stem_type}.wav")
        if not os.path.exists(stem_path):
            stem_path = os.path.join(track_dir, f"{stem_type}.flac")
        if not os.path.exists(stem_path):
            return None
        return stem_path

    def _load_stem(self, track_dir: str, stem_type: str) -> torch.Tensor:
        stem_path = self._stem_path(track_dir, stem_type)
        if stem_path is None:
            return None
        wav, sr = torchaudio.load(stem_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    def _target_clip_score(self, target_clip: torch.Tensor):
        rms = float(torch.sqrt(torch.mean(target_clip ** 2)).item())
        active_ratio = float(
            (target_clip.abs() > self.target_active_threshold).float().mean().item()
        )
        return rms, active_ratio

    def _target_clip_is_active(self, target_clip: torch.Tensor):
        rms, active_ratio = self._target_clip_score(target_clip)
        if self.min_target_rms_db is not None:
            rms_db = 20.0 * math.log10(max(rms, 1e-12))
            if rms_db < float(self.min_target_rms_db):
                return False, rms, active_ratio
        if active_ratio < self.min_target_active_ratio:
            return False, rms, active_ratio
        return True, rms, active_ratio

    def __getitem__(self, idx: int):
        best_sample = None
        best_score = -1.0

        for attempt in range(self.max_clip_resample_attempts):
            track_idx = (idx + attempt) % len(self.track_dirs)
            track_dir = os.path.join(self.data_root, self.track_dirs[track_idx])

            stems = {}
            for inst in INSTRUMENT_MAP:
                wav = self._load_stem(track_dir, inst)
                if wav is not None:
                    stems[inst] = wav

            if self.target_instrument not in stems:
                continue

            target_wav = stems[self.target_instrument]

            # build context
            if self.context_mode == "mixture_minus_target":
                mix_wav = self._load_stem(track_dir, "mix")
                if mix_wav is not None:
                    min_len = min(mix_wav.shape[1], target_wav.shape[1])
                    context_wav = mix_wav[:, :min_len] - target_wav[:, :min_len]
                    target_wav = target_wav[:, :min_len]
                else:
                    other_stems = [w for name, w in stems.items() if name != self.target_instrument]
                    if not other_stems:
                        continue
                    context_wav = sum(other_stems)
            else:
                # future: random_stem_subset
                context_wav = sum(stems.values())

            # clip extraction
            total_samples = min(
                w.shape[1] for w in [context_wav, target_wav]
            )
            if total_samples <= self.clip_samples:
                start = 0
            else:
                start = random.randint(0, total_samples - self.clip_samples - 1)

            context_clip = context_wav[:, start:start + self.clip_samples]
            target_clip = target_wav[:, start:start + self.clip_samples]

            sample = {
                "context": context_clip,
                "target": target_clip,
                "instrument": INSTRUMENT_MAP[self.target_instrument],
                "track_id": self.track_dirs[track_idx],
                "start_sample": start,
            }
            is_active, rms, active_ratio = self._target_clip_is_active(target_clip)
            sample["target_rms"] = rms
            sample["target_active_ratio"] = active_ratio
            score = rms * max(active_ratio, 1e-6)
            if score > best_score:
                best_score = score
                best_sample = sample
            if is_active:
                return sample

        if best_sample is None:
            detail = "no target stem was available"
        else:
            detail = "all sampled target clips were below the activity threshold"
        raise RuntimeError(
            f"No usable {self.target_instrument} clip found in {self.data_root}: {detail}"
        )


class SlakhFixedWindowDataset(SlakhContextTargetDataset):
    """Deterministic fixed-stride context-target windows.

    Unlike SlakhContextTargetDataset, this dataset does not retry random starts.
    It exposes every stride window with activity metadata so cache builders can
    filter inactive target windows without hidden resampling.
    """

    def __init__(
        self,
        *args,
        stride_seconds: float = 5.0,
        max_clips_per_track: int = None,
        **kwargs,
    ):
        super().__init__(*args, max_clip_resample_attempts=1, **kwargs)
        self.stride_samples = max(1, int(float(stride_seconds) * self.sample_rate))
        self.max_clips_per_track = max_clips_per_track
        self.windows = self._build_windows()

    def _target_length_at_sample_rate(self, track_dir: str):
        stem_path = self._stem_path(track_dir, self.target_instrument)
        if stem_path is None:
            return None
        try:
            info = torchaudio.info(stem_path)
            return int(info.num_frames * self.sample_rate / info.sample_rate)
        except Exception:
            wav = self._load_stem(track_dir, self.target_instrument)
            return None if wav is None else int(wav.shape[1])

    def _select_starts(self, starts):
        if self.max_clips_per_track is None or len(starts) <= self.max_clips_per_track:
            return starts
        if self.max_clips_per_track <= 1:
            return [starts[0]]
        selected = []
        last = len(starts) - 1
        for i in range(self.max_clips_per_track):
            idx = round(i * last / (self.max_clips_per_track - 1))
            selected.append(starts[idx])
        return selected

    def _build_windows(self):
        windows = []
        for track_name in self.track_dirs:
            track_dir = os.path.join(self.data_root, track_name)
            total_samples = self._target_length_at_sample_rate(track_dir)
            if total_samples is None:
                continue
            if total_samples <= self.clip_samples:
                starts = [0]
            else:
                starts = list(range(0, total_samples - self.clip_samples + 1, self.stride_samples))
                final_start = total_samples - self.clip_samples
                if starts[-1] != final_start:
                    starts.append(final_start)
            for start in self._select_starts(starts):
                windows.append((track_name, int(start)))
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        track_name, requested_start = self.windows[idx]
        track_dir = os.path.join(self.data_root, track_name)

        stems = {}
        for inst in INSTRUMENT_MAP:
            wav = self._load_stem(track_dir, inst)
            if wav is not None:
                stems[inst] = wav

        if self.target_instrument not in stems:
            raise RuntimeError(f"Missing {self.target_instrument} stem for {track_name}")

        target_wav = stems[self.target_instrument]
        if self.context_mode == "mixture_minus_target":
            mix_wav = self._load_stem(track_dir, "mix")
            if mix_wav is not None:
                min_len = min(mix_wav.shape[1], target_wav.shape[1])
                context_wav = mix_wav[:, :min_len] - target_wav[:, :min_len]
                target_wav = target_wav[:, :min_len]
            else:
                other_stems = [w for name, w in stems.items() if name != self.target_instrument]
                if not other_stems:
                    raise RuntimeError(f"No context stems found for {track_name}")
                context_wav = sum(other_stems)
        else:
            context_wav = sum(stems.values())

        total_samples = min(w.shape[1] for w in [context_wav, target_wav])
        if total_samples <= self.clip_samples:
            start = 0
        else:
            start = min(requested_start, total_samples - self.clip_samples)

        context_clip = context_wav[:, start:start + self.clip_samples]
        target_clip = target_wav[:, start:start + self.clip_samples]
        is_active, rms, active_ratio = self._target_clip_is_active(target_clip)
        return {
            "context": context_clip,
            "target": target_clip,
            "instrument": INSTRUMENT_MAP[self.target_instrument],
            "track_id": track_name,
            "start_sample": start,
            "target_rms": rms,
            "target_active_ratio": active_ratio,
            "is_active": is_active,
        }


class CachedTokenDataset(Dataset):
    """Load precomputed context/target codec-token shards."""

    def __init__(self, cache_root: str, split: str = "train"):
        self.cache_root = cache_root
        self.split = split
        self.split_dir = os.path.join(cache_root, split)
        self.shard_paths = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No token cache shards found in {self.split_dir}")

        contexts = []
        targets = []
        instruments = []
        track_ids = []
        starts = []
        rms_values = []
        active_ratios = []
        for path in self.shard_paths:
            shard = torch.load(path, map_location="cpu")
            contexts.append(shard["context_tokens"])
            targets.append(shard["target_tokens"])
            instruments.append(shard["instrument"])
            track_ids.extend(shard.get("track_id", [""] * shard["context_tokens"].shape[0]))
            starts.extend(shard.get("start_sample", [-1] * shard["context_tokens"].shape[0]))
            rms_values.extend(shard.get("target_rms", [float("nan")] * shard["context_tokens"].shape[0]))
            active_ratios.extend(
                shard.get("target_active_ratio", [float("nan")] * shard["context_tokens"].shape[0])
            )

        self.context_tokens = torch.cat(contexts, dim=0)
        self.target_tokens = torch.cat(targets, dim=0)
        self.instruments = torch.cat(instruments, dim=0).long()
        self.track_ids = track_ids
        self.starts = starts
        self.rms_values = rms_values
        self.active_ratios = active_ratios

    def __len__(self):
        return int(self.context_tokens.shape[0])

    def __getitem__(self, idx: int):
        return {
            "context_tokens": self.context_tokens[idx].long(),
            "target_tokens": self.target_tokens[idx].long(),
            "instrument": self.instruments[idx],
            "track_id": self.track_ids[idx],
            "start_sample": self.starts[idx],
            "target_rms": self.rms_values[idx],
            "target_active_ratio": self.active_ratios[idx],
        }


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
