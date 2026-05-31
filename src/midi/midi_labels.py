"""Convert bass MIDI files into frame-level activity/onset/offset + pitch labels."""
import torch
import numpy as np


class MidiLabelExtractor:
    def __init__(self, min_pitch=28, max_pitch=60, hop_length=512, sample_rate=24000,
                 polyphony_mode="lowest_pitch", ignore_index=-100,
                 onset_window=2, offset_window=2):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.num_pitches = max_pitch - min_pitch + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.polyphony_mode = polyphony_mode
        self.ignore_index = ignore_index
        self.seconds_per_frame = hop_length / sample_rate
        self.onset_window = onset_window
        self.offset_window = offset_window

    def extract(self, midi_path, total_duration_sec):
        """Convert MIDI to frame-level activity + pitch labels."""
        n_frames = int(total_duration_sec / self.seconds_per_frame)
        active_label = np.zeros(n_frames, dtype=np.float32)
        pitch_label = np.full(n_frames, self.ignore_index, dtype=np.int64)

        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return (
                torch.from_numpy(active_label),
                torch.from_numpy(pitch_label),
            )

        for instrument in midi.instruments:
            for note in instrument.notes:
                pitch = note.pitch
                if pitch < self.min_pitch or pitch > self.max_pitch:
                    continue

                start_frame = int(note.start / self.seconds_per_frame)
                end_frame = int(min(note.end, total_duration_sec) / self.seconds_per_frame)
                start_frame = max(0, start_frame)
                end_frame = min(n_frames, end_frame)
                if end_frame <= start_frame:
                    continue

                pitch_id = pitch - self.min_pitch
                for f in range(start_frame, end_frame):
                    active_label[f] = 1.0
                    if pitch_label[f] == self.ignore_index or pitch_id < pitch_label[f]:
                        pitch_label[f] = pitch_id

        return (
            torch.from_numpy(active_label),
            torch.from_numpy(pitch_label),
        )

    def extract_onset_offset(self, midi_path, total_duration_sec):
        """Extract onset, offset, and pitch labels for note boundary detection.

        Returns:
            onset_label: [T] — soft onset (1 in onset_window frames around note start)
            offset_label: [T] — soft offset (1 in offset_window frames around note end)
            pitch_label: [T] — pitch ID on active frames, ignore_index elsewhere
        """
        n_frames = int(total_duration_sec / self.seconds_per_frame)
        onset_label = np.zeros(n_frames, dtype=np.float32)
        offset_label = np.zeros(n_frames, dtype=np.float32)
        pitch_label = np.full(n_frames, self.ignore_index, dtype=np.int64)

        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return (
                torch.from_numpy(onset_label),
                torch.from_numpy(offset_label),
                torch.from_numpy(pitch_label),
            )

        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                pitch = note.pitch
                if pitch < self.min_pitch or pitch > self.max_pitch:
                    continue
                start_frame = int(note.start / self.seconds_per_frame)
                end_frame = int(min(note.end, total_duration_sec) / self.seconds_per_frame)
                start_frame = max(0, start_frame)
                end_frame = min(n_frames, end_frame)
                if end_frame <= start_frame:
                    continue
                notes.append((start_frame, end_frame, pitch - self.min_pitch))

        # Sort by start time
        notes.sort()

        for start_f, end_f, pid in notes:
            # Soft onset window
            for f in range(start_f, min(start_f + self.onset_window, n_frames)):
                onset_label[f] = 1.0
            # Soft offset window
            for f in range(max(0, end_f - self.offset_window), min(end_f, n_frames)):
                offset_label[f] = 1.0
            # Pitch on active frames
            for f in range(start_f, end_f):
                if pitch_label[f] == self.ignore_index or pid < pitch_label[f]:
                    pitch_label[f] = pid

        return (
            torch.from_numpy(onset_label),
            torch.from_numpy(offset_label),
            torch.from_numpy(pitch_label),
        )

    def extract_note_sequence(self, midi_path, total_duration_sec, max_notes=64):
        """Extract notes as a sequence of (pitch, duration_bin) tokens.

        Returns a flat token sequence for autoregressive training.

        Token vocabulary:
        - pitch tokens: 0..num_pitches-1
        - duration tokens: num_pitches..num_pitches+num_dur_bins-1
        - BOS: num_pitches+num_dur_bins
        - EOS: num_pitches+num_dur_bins+1
        """
        n_frames = int(total_duration_sec / self.seconds_per_frame)
        num_dur_bins = 16
        dur_bin_edges = np.logspace(np.log10(2), np.log10(200), num_dur_bins)  # 2-200 frames

        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return None, None

        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                pitch = note.pitch
                if pitch < self.min_pitch or pitch > self.max_pitch:
                    continue
                start_frame = int(note.start / self.seconds_per_frame)
                end_frame = int(min(note.end, total_duration_sec) / self.seconds_per_frame)
                dur_frames = end_frame - start_frame
                if dur_frames < 2:
                    continue
                dur_bin = np.digitize(dur_frames, dur_bin_edges)
                notes.append((start_frame, pitch - self.min_pitch, dur_bin))

        notes.sort()  # by start time

        # Build token sequence
        PITCH_OFFSET = 0
        DUR_OFFSET = self.num_pitches
        BOS = self.num_pitches + num_dur_bins
        EOS = BOS + 1

        tokens = [BOS]
        for _, pid, db in notes[:max_notes]:
            tokens.append(PITCH_OFFSET + pid)
            tokens.append(DUR_OFFSET + db)
        tokens.append(EOS)

        return torch.tensor(tokens, dtype=torch.long), n_frames
