"""Extract mel/chroma/onset features from context audio for MIDI Transformer input.

Supports multiple experiment variants:
- mix: extract from mix audio (with bass) instead of context (mix-minus-bass)
- dual: concatenate context + mix features
- energy: add bass-band log-energy as explicit activity signal
- cqt: use CQT instead of mel for better low-frequency resolution
"""
import torch
import torchaudio
import numpy as np


class AudioFeatureExtractor:
    def __init__(self, sample_rate=24000, n_mels=128, n_fft=2048, hop_length=512,
                 n_chroma=12, use_mel=True, use_chroma=True, use_onset=False,
                 use_mix=False, dual_channel=False, use_energy=True, use_cqt=False,
                 n_bins_per_octave=24):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.use_mel = use_mel
        self.use_chroma = use_chroma
        self.use_onset = False
        self.n_chroma = n_chroma
        self.use_mix = use_mix
        self.dual_channel = dual_channel
        self.use_energy = use_energy
        self.use_cqt = use_cqt
        self.n_bins_per_octave = n_bins_per_octave

        # Pre-compute chroma mapping
        if use_chroma:
            freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
            midi_offsets = 12 * torch.log2(freqs / 440.0) + 69
            self.chroma_bins = (midi_offsets.round().long() % 12)
            self.chroma_mask = torch.stack([
                self.chroma_bins == c for c in range(n_chroma)
            ]).float()

        self.window = torch.hann_window(n_fft)

        if use_cqt:
            try:
                self.cqt_transform = torchaudio.transforms.Spectrogram(
                    n_fft=n_fft, hop_length=hop_length, power=2.0)
            except Exception:
                self.use_cqt = False
                self.use_mel = True

        if use_mel and not use_cqt:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, power=2.0,
            )

        # Bass energy: compute band energy around bass frequencies (41-262 Hz ≈ MIDI 28-60)
        if use_energy:
            freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
            bass_mask = (freqs >= 35.0) & (freqs <= 300.0)
            self.bass_bins = bass_mask
            self.n_bass_bins = bass_mask.sum().item()

        # Compute feature dim
        self.feature_dim = 0
        if use_cqt:
            self.feature_dim += n_mels  # same output size
        elif use_mel:
            self.feature_dim += n_mels
        if use_chroma:
            self.feature_dim += n_chroma
        if use_energy:
            self.feature_dim += 3  # bass energy, total energy, bass/total ratio
        if dual_channel:
            self.feature_dim *= 2

    def _compute_chroma(self, waveform):
        x = waveform[0]
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(x.device), return_complex=True)
        mag = spec.abs() ** 2
        chroma = (self.chroma_mask.to(mag.device) @ mag)
        chroma = torch.log(chroma + 1e-6)
        return chroma.T  # [T, 12]

    def _compute_spec(self, waveform):
        x = waveform[0]
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(x.device), return_complex=True)
        return spec.abs() ** 2  # [n_fft//2+1, T]

    def _compute_energy_features(self, spec):
        """Compute bass-band and total energy features."""
        # spec: [n_freq, T]
        total_energy = spec.sum(dim=0)  # [T]
        bass_energy = spec[self.bass_bins].sum(dim=0) if self.bass_bins.any() else torch.zeros_like(total_energy)

        total_log = torch.log(total_energy + 1e-6)
        bass_log = torch.log(bass_energy + 1e-6)
        bass_ratio = bass_energy / (total_energy + 1e-6)

        return torch.stack([bass_log, total_log, bass_ratio], dim=-1)  # [T, 3]

    def __call__(self, waveform, input_sr=None, mix_waveform=None):
        """Extract features from audio waveform.

        Args:
            waveform: [1, samples] context audio
            input_sr: original sample rate
            mix_waveform: [1, samples] mix audio (for dual_channel mode)

        Returns:
            features: [T, D] frame-level features
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if input_sr is not None and input_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_sr, self.sample_rate)

        # Determine which audio to use for features
        if self.use_mix and mix_waveform is not None:
            primary = mix_waveform
            if primary.dim() == 1:
                primary = primary.unsqueeze(0)
            if input_sr is not None and input_sr != self.sample_rate:
                primary = torchaudio.functional.resample(primary, input_sr, self.sample_rate)
        else:
            primary = waveform

        feats = []
        spec = self._compute_spec(primary)

        if self.use_cqt:
            try:
                cqt = _compute_cqt(waveform if not self.use_mix else primary,
                                   self.sample_rate, self.hop_length,
                                   n_bins_per_octave=self.n_bins_per_octave,
                                   n_bins=self.feature_dim if self.dual_channel else 128)
                feats.append(cqt)
            except Exception:
                mel = self.mel_transform(primary.unsqueeze(0) if primary.dim() == 1 else primary) if hasattr(self, 'mel_transform') else torch.zeros(128, 1)
                mel = torch.log(mel + 1e-6).squeeze(0).T
                feats.append(mel)
        elif self.use_mel:
            mel = self.mel_transform(primary.unsqueeze(0) if primary.dim() == 1 else primary)
            mel = torch.log(mel + 1e-6).squeeze(0).T
            feats.append(mel)

        if self.use_chroma:
            chroma = self._compute_chroma(primary)
            feats.append(chroma)

        if self.use_energy:
            energy = self._compute_energy_features(spec)
            feats.append(energy)

        result = torch.cat(feats, dim=-1)  # [T, D]

        # Dual-channel: also extract features from context and concatenate
        if self.dual_channel and not self.use_mix:
            ctx_feats = []
            ctx_spec = self._compute_spec(waveform)

            if self.use_cqt:
                pass  # skip for dual, use mel
            elif self.use_mel:
                ctx_mel = self.mel_transform(waveform.unsqueeze(0) if waveform.dim() == 1 else waveform)
                ctx_mel = torch.log(ctx_mel + 1e-6).squeeze(0).T
                ctx_feats.append(ctx_mel)

            if self.use_chroma:
                ctx_chroma = self._compute_chroma(waveform)
                ctx_feats.append(ctx_chroma)

            if self.use_energy:
                ctx_energy = self._compute_energy_features(ctx_spec)
                ctx_feats.append(ctx_energy)

            ctx_result = torch.cat(ctx_feats, dim=-1)
            # Align lengths
            min_len = min(result.shape[0], ctx_result.shape[0])
            result = torch.cat([result[:min_len], ctx_result[:min_len]], dim=-1)

        return result


def _compute_cqt(waveform, sample_rate, hop_length, n_bins_per_octave=24, n_bins=128,
                 fmin=32.7, octaves=7):
    """Manual CQT approximation using log-spaced STFT averaging.

    Simpler than a full CQT: uses a high-resolution STFT then averages into log-spaced bins.
    """
    import librosa
    wav_np = waveform.cpu().numpy()
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=0)

    C = librosa.cqt(
        y=wav_np, sr=sample_rate, hop_length=hop_length,
        n_bins=n_bins, bins_per_octave=n_bins_per_octave,
        fmin=fmin,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return torch.from_numpy(cqt_db.T).float()  # [T, n_bins]
