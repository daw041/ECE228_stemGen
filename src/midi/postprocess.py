"""Convert frame-level predictions to MIDI note events and render to audio."""
import numpy as np


def predictions_to_midi(active_prob, pitch_id, instrument_idx=0, instrument_pitch_ranges=None,
                         seconds_per_frame=0.021333, min_note_duration_ms=80, merge_gap_ms=50,
                         default_velocity=80, activity_threshold=0.5):
    """Convert frame-level (active, pitch) predictions to a pretty_midi object."""
    import pretty_midi

    if instrument_pitch_ranges is None:
        instrument_pitch_ranges = [(28, 60)]

    min_pitch, max_pitch = instrument_pitch_ranges[instrument_idx]
    active_prob = active_prob.cpu().numpy() if hasattr(active_prob, 'cpu') else np.array(active_prob)
    pitch_id = pitch_id.cpu().numpy() if hasattr(pitch_id, 'cpu') else np.array(pitch_id)

    T = len(active_prob)
    active = active_prob > activity_threshold

    segments = []
    i = 0
    while i < T:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < T and active[j] and pitch_id[j] == pitch_id[i]:
            j += 1
        segments.append({
            "start_sec": i * seconds_per_frame,
            "end_sec": j * seconds_per_frame,
            "pitch_id": int(pitch_id[i]),
        })
        i = j

    merged = []
    for seg in segments:
        if merged and seg["pitch_id"] == merged[-1]["pitch_id"]:
            gap = seg["start_sec"] - merged[-1]["end_sec"]
            if gap <= merge_gap_ms / 1000.0:
                merged[-1]["end_sec"] = seg["end_sec"]
                continue
        merged.append(seg)

    min_duration = min_note_duration_ms / 1000.0
    merged = [s for s in merged if s["end_sec"] - s["start_sec"] >= min_duration]

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=33)

    for seg in merged:
        actual_pitch = seg["pitch_id"] + min_pitch
        actual_pitch = max(min_pitch, min(max_pitch, actual_pitch))
        note = pretty_midi.Note(
            velocity=default_velocity, pitch=actual_pitch,
            start=seg["start_sec"], end=seg["end_sec"],
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def midi_to_audio(midi, sample_rate=24000, soundfont_path=None):
    """Render MIDI to audio. Uses fluidsynth if available, falls back to sine synthesis."""
    # Try fluidsynth first
    try:
        import tempfile, os, torch
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            midi.write(f.name)
            midi_path = f.name
        try:
            audio = midi.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
            audio = torch.from_numpy(audio).float()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        finally:
            os.unlink(midi_path)
        return audio
    except (ImportError, OSError, FileNotFoundError):
        pass

    # Fallback: simple sine-wave synthesis
    return _sine_synth(midi, sample_rate)


def _sine_synth(midi, sample_rate):
    """Simple sine-wave synthesis as fallback when fluidsynth is unavailable."""
    import torch

    # Get total duration from MIDI
    total_sec = 0.0
    for inst in midi.instruments:
        for note in inst.notes:
            total_sec = max(total_sec, note.end)

    total_sec = max(total_sec, 1.0)
    n_samples = int(total_sec * sample_rate)
    audio = torch.zeros(1, n_samples, dtype=torch.float32)
    t = torch.arange(n_samples, dtype=torch.float32) / sample_rate

    for inst in midi.instruments:
        for note in inst.notes:
            start = int(note.start * sample_rate)
            end = int(note.end * sample_rate)
            if end <= start:
                continue
            # MIDI pitch to frequency
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            # Simple ADSR envelope
            seg_len = end - start
            seg_t = torch.arange(seg_len, dtype=torch.float32) / sample_rate
            attack = 0.02
            release = 0.05
            envelope = torch.ones(seg_len)
            attack_samples = int(attack * sample_rate)
            if attack_samples > 0 and attack_samples < seg_len:
                envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)
            release_samples = int(release * sample_rate)
            if release_samples > 0 and release_samples < seg_len:
                envelope[-release_samples:] = torch.linspace(1, 0, release_samples)
            waveform = torch.sin(2 * np.pi * freq * seg_t) * envelope
            audio[0, start:end] += waveform * (note.velocity / 127.0) * 0.3

    # Normalize
    peak = audio.abs().max()
    if peak > 0.99:
        audio = audio / peak * 0.9

    return audio
