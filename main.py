import argparse
from time import time
from pathlib import Path
from piano.piano import render_piano_from_events, PianoConfig
from midi.midi import read_midi_to_events
import soundfile as sf


def main(midi_filename):  # Example parameters for the ideal string
    sample_rate = 192000  # Hz
    duration = 4.0  # seconds

    events = read_midi_to_events(
        midi_filename,
        duration=duration,
    )

    time_start = time()
    audio, _, _, _ = render_piano_from_events(events, duration, config=PianoConfig())
    time_end = time()
    print(f"Rendered audio with shape: {audio.shape}")

    sf.write("output/piano_output.wav", audio, sample_rate)
    print(
        f"Rendered {audio.shape[0] / sample_rate} seconds of audio in {time_end - time_start} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MIDI to piano audio.")
    parser.add_argument("midi_file", type=Path, help="Path to the MIDI file.")
    args = parser.parse_args()

    main(args.midi_file)
