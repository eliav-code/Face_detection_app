from random import sample
import wave
import numpy as np
import sounddevice as sd
import winsound
import chime
import time
def play_siren_sound(duration : int = 10, low_freq : int = 600, high_freq : int = 1200, rate : int = 0.25, sample_rate : int = 44100):
    """
    Play a siren-like sound.

    :param duration: Total duration in seconds
    :param low_freq: Lowest frequency (Hz)
    :param high_freq: Highest frequency (Hz)
    :param rate: Number of up-down cycles per second
    :param samplerate: Audio sampling rate
    """
    # Create a linear time array from 0 to duration with samples based on the sampling rate
    time_linspace = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a sine wave oscillating between 0 and 1 at frequency 'rate' Hz to modulate frequency smoothly
    modulating_wave = (np.sin(2*np.pi*rate*time_linspace) + 1) / 2  # values range from 0 to 1, sine-distributed

    # Calculate the instantaneous frequency by interpolating between low_freq and high_freq
    freqs = low_freq + modulating_wave * (high_freq - low_freq)

    # Calculate the phase by integrating the frequency over time for continuous wave generation
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate

    # Generate the waveform as a sine wave with the calculated phase
    waveform = np.sin(phase)

    # Normalize volume to 50% of the original amplitude to avoid loudness
    waveform *= 0.5

    # Play the generated waveform using sounddevice with the given sampling rate
    sd.play(waveform, samplerate=sample_rate)

    # Wait until the sound playback is finished before returning
    sd.wait()

def play_wonderful_sound(duration=1, f0=200, sample_rate=44100):
        # if self.current_play_obj and self.current_play_obj.is_playing():
        #     return
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.3 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        waveform /= np.max(np.abs(waveform))

        sd.play(waveform, samplerate=sample_rate)

        sd.wait()