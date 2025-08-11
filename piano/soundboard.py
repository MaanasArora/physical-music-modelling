from scipy.signal import butter, lfilter
import numpy as np
from scipy.signal import fftconvolve, resample


def lowpass_filter(cutoff, sample_rate, order=4):
    nyquist = sample_rate / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def highpass_filter(cutoff, sample_rate, order=4):
    nyquist = sample_rate / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def peaking_filter(frequency, sample_rate, Q=1.0, gain_db=6.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * frequency / sample_rate
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def convolution_reverb(
    sound: np.ndarray,
    impulse_response: np.ndarray,
    audio_sample_rate: int,
    impulse_response_sample_rate: int,
) -> np.ndarray:
    output = []
    if sound.ndim == 1:
        sound = sound[:, np.newaxis]
    if impulse_response.ndim == 1:
        impulse_response = impulse_response[:, np.newaxis]

    if sound.shape[1] != impulse_response.shape[1]:
        sound = np.tile(sound, (1, impulse_response.shape[1]))

    num_samples = int(
        len(impulse_response) * audio_sample_rate / impulse_response_sample_rate
    )
    impulse_response = resample(impulse_response, num_samples)

    for channel in range(impulse_response.shape[1]):
        convolved = fftconvolve(
            sound[:, channel], impulse_response[:, channel], mode="full"
        )
        output.append(convolved[: sound.shape[0]])
    return np.array(output).T if len(output) > 1 else output[0]


def process_piano_sound(
    sound: np.ndarray,
    sample_rate: int = 44100,
    impulse_response: np.ndarray = None,
    impulse_response_sample_rate: int = 44100,
) -> np.ndarray:
    amp = 14.0
    sound = sound * amp

    b, a = lowpass_filter(400, sample_rate, 1)
    sound = lfilter(b, a, sound, axis=0)
    b, a = lowpass_filter(4000, sample_rate, 4)
    sound = lfilter(b, a, sound, axis=0)

    resonances = [
        (20, 10),
        (40, 8),
        (55, 6),
        (100, 6),
        (125, 5),
        (176, 4),
        (220, 3),
        (453, 2),
    ]

    resonant_sound = sound.copy()
    for frequency, Q in resonances:
        b0, b1, b2, a1, a2 = peaking_filter(frequency, sample_rate, Q)
        resonant_sound = lfilter([b0, b1, b2], [1, a1, a2], resonant_sound)

    if impulse_response is not None:
        resonant_sound = convolution_reverb(
            resonant_sound, impulse_response, sample_rate, impulse_response_sample_rate
        )

    b, a = highpass_filter(10, sample_rate, 2)
    processed_sound = lfilter(b, a, resonant_sound, axis=0)

    b0, b1, b2, a1, a2 = peaking_filter(1970, sample_rate, Q=1.0, gain_db=3.0)
    processed_sound = lfilter([b0, b1, b2], [1, a1, a2], processed_sound)

    return processed_sound
