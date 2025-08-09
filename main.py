from piano.piano import render_piano
from jax import jit
from jax import numpy as jnp
import matplotlib.pyplot as plt
import soundfile as sf


def main():  # Example parameters for the ideal string
    sample_rate = 44100
    duration = 1.0  # seconds
    steps_per_sample = 10
    resistance_factor = 0.9999
    num_steps = int(sample_rate * duration * steps_per_sample)

    roll = jnp.zeros((88, num_steps), dtype=jnp.float32)
    roll = roll.at[60, :].set(1.0)
    roll = roll.at[64, :].set(0.5)

    audio = render_piano(
        roll,
        sample_rate=sample_rate,
        duration=duration,
        steps_per_sample=steps_per_sample,
        num_steps=num_steps,
        resistance_factor=resistance_factor,
    )

    print("Audio shape:", audio.shape)

    sf.write('piano_output.wav', audio, sample_rate)

if __name__ == "__main__":
    main()
