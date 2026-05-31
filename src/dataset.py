"""Slakh2100 context-target dataset construction."""
import os
import random
import math
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

    def _load_stem(self, track_dir: str, stem_type: str) -> torch.Tensor:
        stem_path = os.path.join(track_dir, f"{stem_type}.wav")
        if not os.path.exists(stem_path):
            stem_path = os.path.join(track_dir, f"{stem_type}.flac")
        if not os.path.exists(stem_path):
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
            }
            is_active, rms, active_ratio = self._target_clip_is_active(target_clip)
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
