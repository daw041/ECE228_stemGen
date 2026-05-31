"""Slakh2100 context-target dataset construction."""
import os
import random
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
    ):
        self.data_root = data_root
        self.target_instrument = target_instrument.lower()
        self.context_mode = context_mode
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.split = split
        self.max_clips = max_clips

        self.track_dirs = self._scan_tracks()
        self.clips_per_track = max(1, max_clips // max(1, len(self.track_dirs)))
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
        return min(self.max_clips, len(self.track_dirs) * self.clips_per_track)

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

    def __getitem__(self, idx: int):
        track_idx = idx % len(self.track_dirs)
        track_dir = os.path.join(self.data_root, self.track_dirs[track_idx])

        stems = {}
        for inst in INSTRUMENT_MAP:
            wav = self._load_stem(track_dir, inst)
            if wav is not None:
                stems[inst] = wav

        if self.target_instrument not in stems:
            return self.__getitem__((idx + 1) % len(self))

        target_wav = stems[self.target_instrument]

        # build context
        if self.context_mode == "mixture_minus_target":
            context_wav = sum(
                w for name, w in stems.items() if name != self.target_instrument
            )
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

        return {
            "context": context_clip,
            "target": target_clip,
            "instrument": INSTRUMENT_MAP[self.target_instrument],
            "track_id": self.track_dirs[track_idx],
        }


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
