from functools import partial
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax
from jax import lax, tree_util


NUM_POINTS = 24
PLUCK_POINT = int(NUM_POINTS * 0.25)
BRIDGE_POINT = int(NUM_POINTS * 0.9)
SAMPLE_RATE = 88200  # Hz
STEPS_PER_SAMPLE = 2
TIMESTEP = 1 / (SAMPLE_RATE * STEPS_PER_SAMPLE)


@jax.tree_util.register_dataclass
@dataclass
class PianoString:
    length: float
    speed: float
    timestep: float
    spacing: float
    lamda: float
    mu: float
    damping: float
    b1: float
    b2: float


def _piano_string_config(
    length: float,
    speed: float,
    damping: float,
    b1: float,
    b2: float,
    timestep: float = 0.01,
) -> PianoString:
    spacing = length / NUM_POINTS
    lamda = speed * timestep / spacing
    mu = damping * timestep / (length * length)

    return PianoString(
        length=length,
        speed=speed,
        timestep=timestep,
        spacing=spacing,
        lamda=lamda,
        mu=mu,
        damping=damping,
        b1=b1,
        b2=b2,
    )


@jax.jit
def _piano_string_step(
    displacements: tuple[jnp.ndarray, jnp.ndarray], config: PianoString
) -> jnp.ndarray:
    lamda = config.lamda
    mu = config.mu
    damping = config.damping
    timestep = config.timestep
    b1 = config.b1
    b2 = config.b2

    denominator = 1 + b1 * timestep
    a10 = (2 - 2 * lamda**2 - 6 * mu**2 - 4 * b2 * mu / damping) / denominator
    a11 = (lamda**2 + 4 * mu**2 + 2 * b2 * mu / damping) / denominator
    a12 = -(mu**2) / denominator
    a20 = (-1 + 4 * b2 * mu / damping + b1 * timestep) / denominator
    a21 = (2 * b2 * mu / damping) / denominator

    prev_displacement, displacement = displacements

    left_ext_displacement = jnp.concatenate(
        [jnp.array([displacement[0], 0]), displacement]
    )
    right_ext_displacement = jnp.concatenate(
        [displacement, jnp.array([0, displacement[-1]])]
    )

    left_displacement = left_ext_displacement[1:-1]
    right_displacement = right_ext_displacement[1:-1]
    left_2_displacement = left_ext_displacement[:-2]
    right_2_displacement = right_ext_displacement[2:]

    prev_left_displacement = jnp.concatenate([jnp.zeros(1), prev_displacement[:-1]])
    prev_right_displacement = jnp.concatenate([prev_displacement[1:], jnp.zeros(1)])

    new_displacement = (
        a10 * displacement
        + a11 * (left_displacement + right_displacement)
        + a12 * (left_2_displacement + right_2_displacement)
        + a20 * prev_displacement
        + a21 * (prev_left_displacement + prev_right_displacement)
    )

    new_displacement = new_displacement.at[0].set(0)
    new_displacement = new_displacement.at[-1].set(0)

    return new_displacement


@jax.jit
def piano_string_init(
    config: PianoString,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    initial_displacement = jnp.zeros(NUM_POINTS)
    return initial_displacement, initial_displacement


@jax.jit
def piano_string_step(
    displacements: tuple[jnp.ndarray, jnp.ndarray],
    amplitude: jnp.ndarray | None,
    config: PianoString,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    _, displacement = displacements
    if amplitude is not None:
        displacement = displacement.at[PLUCK_POINT].add(amplitude)
    new_displacement = _piano_string_step(displacements, config)
    return (displacement, new_displacement), new_displacement[BRIDGE_POINT]


def get_num_samples(config: PianoString | None, duration: float) -> int:
    return int(SAMPLE_RATE * STEPS_PER_SAMPLE * duration)


@jax.jit
def render_piano_string(
    displacements: tuple[jnp.ndarray, jnp.ndarray],
    amplitude: jnp.ndarray | None,
    config: PianoString,
    num_samples: int = SAMPLE_RATE * STEPS_PER_SAMPLE,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    (displacements), audio = lax.scan(
        lambda c, x: piano_string_step(c, x, config),
        displacements,
        amplitude,
        length=num_samples,
    )
    return displacements, audio[::STEPS_PER_SAMPLE]
