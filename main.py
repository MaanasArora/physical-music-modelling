import argparse
from time import time
from pathlib import Path
from piano.string import SAMPLE_RATE
from piano.piano import render_piano_from_events, PianoConfig
from piano.soundboard import process_piano_sound
from midi.midi import read_midi_to_events
import soundfile as sf


def main(midi_filename, duration, impulse_response):
    events = read_midi_to_events(midi_filename, duration=duration)

    time_start = time()
    audio, _, _, _ = render_piano_from_events(events, duration, config=PianoConfig())
    time_end = time()

    print(f"Rendered {len(events)} MIDI events in {time_end - time_start:.2f} seconds.")

    ir_file = Path(impulse_response)
    ir, ir_sr = sf.read(ir_file)

    audio = process_piano_sound(
        audio,
        sample_rate=SAMPLE_RATE,
        impulse_response=ir,
        impulse_response_sample_rate=ir_sr,
    )

    output_filename = Path(midi_filename).with_suffix(".wav")
    sf.write(output_filename, audio, samplerate=SAMPLE_RATE)
    print(f"Audio saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MIDI to piano audio.")
    parser.add_argument("midi_file", type=Path, help="Path to the MIDI file.")
    parser.add_argument(
        "duration", type=float, help="Duration in seconds to render the audio."
    )
    parser.add_argument(
        "--impulse_response",
        type=Path,
        default=Path("piano/ir/s2_r1_sr.wav"),
        help="Path to the impulse response file.",
    )
    args = parser.parse_args()

    main(args.midi_file, args.duration, args.impulse_response)
