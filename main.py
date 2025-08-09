import argparse
from time import time
from pathlib import Path
from piano.piano import render_piano
from midi.midi import read_midi_to_pianoroll
import soundfile as sf


def main(midi_filename):  # Example parameters for the ideal string
    sample_rate = 192000  # Hz
    duration = 4.0  # seconds
    steps_per_sample = 1
    resistance_factor = 0.99998
    num_steps = int(sample_rate * duration * steps_per_sample)

    roll = read_midi_to_pianoroll(midi_filename, duration)
    print(f"Loaded piano roll with shape: {roll.shape}")

    time_start = time()
    audio = render_piano(
        roll,
        sample_rate=sample_rate,
        duration=duration,
        steps_per_sample=steps_per_sample,
        num_steps=num_steps,
        resistance_factor=resistance_factor,
    )
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
