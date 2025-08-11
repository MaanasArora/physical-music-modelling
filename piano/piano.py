from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax
from scipy.signal import butter, lfilter
from tqdm import tqdm
from piano.string import (
    PianoString,
    _piano_string_config,
    piano_string_init,
    render_piano_string,
    TIMESTEP,
    SAMPLE_RATE,
    NUM_POINTS,
    STEPS_PER_SAMPLE,
)


@jax.tree_util.register_dataclass
@dataclass
class PianoConfig:
    min_length: float = 0.05
    max_length: float = 2.0
    min_b1: float = 0.5
    max_b1: float = 1.5
    min_b2: float = 1e-4
    max_b2: float = 1e-2


def piano_note_init(midi_note: int, config: PianoConfig) -> PianoString:
    frequency = 440 * (2 ** ((midi_note - 69) / 12))

    length_ratio = (127 - midi_note) / (127 - 21)
    length = (
        config.min_length + (config.max_length - config.min_length) * length_ratio**0.5
    )

    if frequency > 1600:
        damping = 1.25
    elif frequency < 800:
        damping = 0.75
    else:
        damping = 0.5

    b1 = config.min_b1 + (config.max_b1 - config.min_b1) * frequency / 440
    b2 = config.min_b2 + (config.max_b2 - config.min_b2) * frequency / 440

    speed = 2 * frequency * length

    courant = speed * TIMESTEP / (length / NUM_POINTS)
    if courant > 1:
        raise ValueError(
            f"Courant condition violated: {courant} > 1 for frequency {frequency} Hz and note {midi_note}"
        )

    return _piano_string_config(
        length=length,
        speed=speed,
        damping=damping,
        b1=b1,
        b2=b2,
        timestep=TIMESTEP,
    )


def process_note_audio(
    note: int,
    frequency: float,
    audio: jnp.ndarray,
    config: PianoConfig,
) -> jnp.ndarray:
    nyquist = SAMPLE_RATE / 2
    cutoff = np.clip(frequency * 3, 400, 900)
    norm_cutoff = cutoff / nyquist
    b, a = butter(4, norm_cutoff, btype="low")
    new_audio = lfilter(b, a, audio)

    ratio = jnp.max(jnp.abs(new_audio)) / jnp.max(jnp.abs(audio))
    if ratio > 0 and ratio < 0.8:
        new_audio = new_audio / ratio
    audio = new_audio

    min_note = 21  # A0
    max_note = 108  # C8
    pan = (note - min_note) / (max_note - min_note)
    pan = 0.3 + 0.4 * pan

    note_audio = jnp.stack(
        [
            audio * (1 - pan),  # Left channel
            audio * pan,  # Right channel
        ],
        axis=-1,
    )
    return note_audio


def render_piano_from_events(
    events: list[tuple[float, int, float]],
    duration: float,
    config: PianoConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[PianoString]]:
    min_note = 21  # A0
    max_note = 98  # C8
    num_notes = max_note - min_note + 1

    strings = [piano_note_init(note, config) for note in range(min_note, max_note + 1)]

    displacements_tuple = [piano_string_init(string) for string in strings]
    displacements_prev = jnp.stack([d[0] for d in displacements_tuple])
    displacements = jnp.stack([d[1] for d in displacements_tuple])

    chunked_events = []
    for t_second in range(int(np.ceil(duration))):
        start = t_second
        end = t_second + 1
        chunked_events.append(
            [
                (time - start, note, velocity / 127.0)
                for time, note, velocity in events
                if start <= time < end
            ]
        )

    audio = []
    pbar = tqdm(total=duration, desc="Rendering audio", unit="s", unit_scale=True)
    for t_second in range(int(np.ceil(duration))):
        len_chunk = SAMPLE_RATE * STEPS_PER_SAMPLE
        amplitude = jnp.zeros((num_notes, len_chunk), dtype=jnp.float32)
        for time, note, velocity in chunked_events[t_second]:
            if min_note <= note <= max_note and time < duration and velocity > 0:
                note_index = note - min_note
                amplitude = amplitude.at[
                    note_index, int(time * SAMPLE_RATE * STEPS_PER_SAMPLE)
                ].set(velocity)

        audio_chunk = jnp.zeros((num_notes, SAMPLE_RATE), dtype=jnp.float32)

        (active_notes,) = jnp.where(np.abs(displacements).mean(axis=1) > 1e-4)
        (amp_notes,) = jnp.where(amplitude.sum(axis=1) > 0)
        active_notes = jnp.union1d(active_notes, amp_notes)

        for note in active_notes:
            pbar.update(1 / len(active_notes))

            new_displacement, note_audio = render_piano_string(
                (displacements_prev[note], displacements[note]),
                amplitude[note],
                strings[note],
            )
            displacements_prev = displacements_prev.at[note].set(new_displacement[0])
            displacements = displacements.at[note].set(new_displacement[1])
            audio_chunk = audio_chunk.at[note].set(note_audio)

        audio.append(audio_chunk)

    processed_audio = jnp.zeros(
        (int(SAMPLE_RATE * np.ceil(duration)), 2), dtype=jnp.float32
    )
    for note in range(num_notes):
        frequency = 440 * (2 ** ((note + min_note - 69) / 12))
        audio_note = [audio_chunk[note] for audio_chunk in audio]
        audio_note = jnp.concatenate(audio_note, axis=0)
        audio_note = process_note_audio(note + min_note, frequency, audio_note, config)
        processed_audio = processed_audio.at[: len(audio_note)].add(audio_note)

    return processed_audio, displacements_prev, displacements, strings
