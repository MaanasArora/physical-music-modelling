import numpy as np
import mido


def read_midi_to_events(midi_file: str, duration: float = 10.0):

    midi = mido.MidiFile(midi_file)
    ticks_per_beat = midi.ticks_per_beat
    tempo_microseconds = mido.bpm2tempo(120)  # Default 120 BPM

    events = []

    for track in midi.tracks:
        total_ticks = 0

        for msg in track:
            total_ticks += msg.time

            if msg.type == "set_tempo":
                tempo_microseconds = msg.tempo
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                # Convert ticks to seconds
                time_seconds = mido.tick2second(
                    total_ticks, ticks_per_beat, tempo_microseconds
                )

                if time_seconds >= duration:
                    break

                events.append((time_seconds, msg.note, msg.velocity))

    return sorted(events)  # Sort by time


def events_to_pianoroll(events, duration, sample_rate=24):
    total_samples = int(duration * sample_rate)
    pianoroll = np.zeros((88, total_samples))  # 88 piano keys

    for time_seconds, midi_note, velocity in events:
        sample_idx = int(time_seconds * sample_rate)

        if sample_idx < total_samples:
            amp = velocity / 127.0
            pianoroll[midi_note - 21, sample_idx] = amp

    # Remove sustained notes (onset detection)
    for t in range(1, pianoroll.shape[1]):
        for note in range(88):
            if pianoroll[note, t] > 0 and pianoroll[note, t - 1] > 0:
                pianoroll[note, t] = 0

    return pianoroll


def read_midi_to_pianoroll(
    midi_file: str,
    duration: float = 10.0,
):
    """
    Reads a MIDI file and converts it to a piano roll representation.

    Args:
        midi_file (str): Path to the MIDI file.
        sample_rate (int): Sample rate for audio rendering.
        duration (float): Duration of the audio in seconds.
        steps_per_sample (int): Number of steps per sample for rendering.

    Returns:
        np.array: Piano roll representation of the MIDI file.
    """
    notes_list = read_midi_to_events(midi_file, duration)
    pianoroll = events_to_pianoroll(notes_list, duration)
    return pianoroll
