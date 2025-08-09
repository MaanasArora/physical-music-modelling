from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax, debug as jax_debug


@partial(
    jit,
    static_argnames=("resistance_factor"),
)
def ideal_string_step(
    displacement: jnp.ndarray,
    velocity: jnp.ndarray,
    num_points: jnp.ndarray,
    resistance_factor: float,
    frequency: float,
    dx: float,
    dt: float,
    c: float,
):
    mask = jnp.arange(displacement.shape[0]) < num_points

    d2x = jnp.zeros_like(displacement, dtype=jnp.float32)
    d2x = d2x.at[1:-1].set(
        (displacement[2:] - 2 * displacement[1:-1] + displacement[:-2]) / (dx * dx)
    )
    d2x = jnp.where(mask, d2x, 0.0)

    frequency_factor = (frequency / 440.0) ** 0.3  # Adjust exponent as needed
    velocity = velocity * resistance_factor**frequency_factor
    velocity = velocity + (c * c) * d2x * dt
    displacement = displacement + velocity * dt

    displacement = displacement.at[0].set(0.0)
    velocity = velocity.at[0].set(0.0)
    displacement = displacement.at[num_points - 1].set(0.0)
    velocity = velocity.at[num_points - 1].set(0.0)

    return displacement, velocity


@partial(
    jit,
    static_argnames=("resistance_factor"),
)
def ideal_string_step_scan(
    carry,
    x,
    resistance_factor: float,
    frequency: float,
    num_points: jnp.ndarray,
    dx: float,
    dt: float,
    c: float,
):
    displacement, velocity = carry

    hammer_point = jnp.floor(num_points / 4).astype(jnp.int32)
    sound_point = jnp.floor(3 * num_points / 4).astype(jnp.int32)

    displacement = displacement.at[hammer_point].add(x)
    displacement, velocity = ideal_string_step(
        displacement,
        velocity,
        num_points,
        resistance_factor,
        frequency,
        dx,
        dt,
        c,
    )
    sound = displacement[sound_point]
    return (displacement, velocity), sound


@partial(
    jit,
    static_argnames=(
        "resistance_factor",
        "steps_per_sample",
        "max_num_points",
        "num_steps",
    ),
)
def render_ideal_string(
    plucks: jnp.ndarray,
    frequency: jnp.ndarray,
    dx: jnp.ndarray,
    dt: jnp.ndarray,
    c: jnp.ndarray,
    num_points: jnp.ndarray,
    max_num_points: int,
    resistance_factor: float,
    steps_per_sample: int = 10,
    num_steps: int = 44100 * 10 * 10,
):
    displacement = jnp.zeros(max_num_points, dtype=jnp.float32)
    velocity = jnp.zeros(max_num_points, dtype=jnp.float32)

    _, audio = lax.scan(
        partial(
            ideal_string_step_scan,
            resistance_factor=resistance_factor,
            frequency=frequency,
            dx=dx,
            dt=dt,
            c=c,
            num_points=num_points,
        ),
        (displacement, velocity),
        plucks,
        length=num_steps,
    )

    audio = audio[::steps_per_sample]

    return audio.astype(jnp.float32)


@partial(
    jit,
    static_argnames=(
        "sample_rate",
        "steps_per_sample",
    ),
)
def get_ideal_string_params(
    tension: jnp.ndarray,
    length: jnp.ndarray,
    density: jnp.ndarray,
    sample_rate: int = 44100,
    num_points: jnp.ndarray = jnp.array(40, dtype=jnp.int32),
    steps_per_sample: int = 10,
):
    c = jnp.sqrt(tension / density)
    dx = length / (num_points - 1)
    dt = 1.0 / (sample_rate * steps_per_sample)

    return dx, dt, c
