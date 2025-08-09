from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax, debug as jax_debug
from piano.string import get_ideal_string_params, render_ideal_string


@partial(
    jit,
    static_argnames=(
        "base_density",
        "high_length",
        "low_length",
    ),
)
def make_piano_string(
    midi_note: jnp.ndarray,
    base_density: float = 0.1,
    high_length: float = 0.4,
    low_length: float = 0.2,
):
    frequency = 440.0 * 2 ** ((midi_note - 69) / 12.0)

    density = jnp.array(base_density, dtype=jnp.float32)
    length = jnp.where(midi_note > 69, high_length, low_length)
    tension = (2 * frequency * length) ** 2 * density * 1.0

    num_points = jnp.full_like(frequency, 10, dtype=jnp.int32)
    num_points = jnp.where(frequency < 130.81, 20, num_points)
    num_points = jnp.where(frequency < 523.25, 40, num_points)

    return frequency, tension, length, density, num_points


def render_piano(
    roll: jnp.ndarray,
    sample_rate: int = 44100,
    duration: float = 10.0,
    steps_per_sample: int = 10,
    resistance_factor: float = 0.9999,
    base_density: float = 0.02,
    high_length: float = 0.2,
    low_length: float = 0.05,
    max_num_points: int = 150,
    num_steps: int = 44100 * 10 * 10,
    roll_sample_rate: int = 24,
    dist_tolerance: float = 1.0,
):
    """
    Render a piano sound based on the given notes.

    Args:
        notes: A 1D array of note frequencies.
        sample_rate: The sample rate for audio output.
        duration: The duration of the sound in seconds.
        steps_per_sample: Number of simulation steps per audio sample.
        resistance_factor: Damping factor for the string simulation.
        base_density: Base linear density of the string in kg/m.
        high_length: Length of the high-pitched string in meters.
        low_length: Length of the low-pitched string in meters.

    Returns:
        A 1D array representing the audio waveform.
    """
    notes = jnp.arange(21, 109)

    frequencies, tensions, lengths, densities, num_points = vmap(
        make_piano_string,
        in_axes=(0, None, None, None),
    )(
        notes,
        base_density,
        high_length,
        low_length,
    )

    dxs, dts, cs = vmap(
        get_ideal_string_params,
        in_axes=(0, 0, 0, None, 0, None),
    )(
        tensions,
        lengths,
        densities,
        sample_rate,
        num_points,
        steps_per_sample,
    )

    dxs = jnp.asarray(dxs, dtype=jnp.float32)
    dts = jnp.asarray(dts, dtype=jnp.float32)
    cs = jnp.asarray(cs, dtype=jnp.float32)
    num_points = jnp.asarray(num_points, dtype=jnp.int32)

    max_num_points = int(max_num_points)
    sample_rate = int(sample_rate)
    duration = float(duration)

    num_notes = roll.shape[0]
    audio = jnp.zeros((num_notes, num_steps // steps_per_sample), dtype=jnp.float32)

    dist_tolerance = int(dist_tolerance * roll_sample_rate)
    roll_sample_to_sample = steps_per_sample * sample_rate // roll_sample_rate

    for note in range(num_notes):
        print(jnp.sum(roll[note]))
        if jnp.sum(roll[note]) == 0:
            continue

        pluck_indices = jnp.where(roll[note] > 0, jnp.arange(roll.shape[1]), 0)
        pluck_indices = jnp.nonzero(pluck_indices)[0]
        pluck_indices = jnp.concatenate(
            [jnp.array([0]), pluck_indices, jnp.array([roll.shape[1] - 1])]
        )
        print(f"Pluck indices for note {note}: {pluck_indices}")
        chunked = jnp.split(
            roll[note], jnp.where(jnp.diff(pluck_indices) > dist_tolerance)[0]
        )

        chunk_time = 0
        for chunk in chunked:
            chunk_stripped = jnp.trim_zeros(chunk, trim="b")
            if chunk_stripped.size == 0:
                continue

            chunk_upsampled = jnp.zeros(
                (chunk_stripped.size * roll_sample_to_sample,),
                dtype=jnp.float32,
            )
            chunk_upsampled = chunk_upsampled.at[::roll_sample_to_sample].set(
                chunk_stripped
            )

            chunk_upsampled = jnp.pad(
                chunk_upsampled,
                (
                    0,
                    min(
                        num_steps - chunk_upsampled.size,
                        dist_tolerance * roll_sample_to_sample,
                    ),
                ),
                mode="constant",
                constant_values=0,
            )

            audio_chunk = render_ideal_string(
                chunk_upsampled,
                frequencies[note],
                dxs[note],
                dts[note],
                cs[note],
                num_points[note],
                max_num_points,
                resistance_factor,
                steps_per_sample,
                len(chunk_upsampled),
            )
            audio = audio.at[note, chunk_time : chunk_time + len(audio_chunk)].set(
                audio_chunk
            )

            chunk_time += len(chunk) * roll_sample_to_sample

    audio = jnp.sum(audio, axis=0)
    return audio
