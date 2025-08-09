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
    base_density: float = 0.02,
    high_length: float = 0.2,
    low_length: float = 0.05,
):
    frequency = 440.0 * 2 ** ((midi_note - 69) / 12.0)

    density = jnp.array(base_density, dtype=jnp.float32)
    length = jnp.where(midi_note > 69, high_length, low_length)
    tension = (2 * frequency * length) ** 2 * density * 1.0

    num_points = jnp.full_like(frequency, 50, dtype=jnp.int32)
    num_points = jnp.where(frequency < 130.81, 150, num_points)
    num_points = jnp.where(frequency < 523.25, 100, num_points)

    return frequency, tension, length, density, num_points


@partial(
    jit,
    static_argnames=(
        "resistance_factor",
        "base_density",
        "high_length",
        "low_length",
        "max_num_points",
        "sample_rate",
        "duration",
        "steps_per_sample",
        "num_steps",
    ),
)
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

    audio = vmap(
        render_ideal_string,
        in_axes=(0, 0, 0, 0, None, None, None, None, None),
    )(
        roll,
        dxs,
        dts,
        cs,
        num_points,
        max_num_points,
        resistance_factor,
        steps_per_sample,
        num_steps,
    )

    audio = jnp.sum(audio, axis=0)
    return audio
