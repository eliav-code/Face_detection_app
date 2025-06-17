import wave
import numpy as np
import sounddevice as sd
import winsound
import chime
import time
def play_siren_sound(duration : int = 10, low_freq : int = 600, high_freq : int = 1200, rate : int = 0.25, samplerate : int = 44100):
    """
    Play a siren-like sound.

    :param duration: Total duration in seconds
    :param low_freq: Lowest frequency (Hz)
    :param high_freq: Highest frequency (Hz)
    :param rate: Number of up-down cycles per second
    :param samplerate: Audio sampling rate
    """
    # Create a linear time array from 0 to duration with samples based on the sampling rate
    time_linspace = np.linspace(0, duration, int(samplerate * duration), endpoint=False)

    # Create a sine wave oscillating between 0 and 1 at frequency 'rate' Hz to modulate frequency smoothly
    modulating_wave = (np.sin(2*np.pi*rate*time_linspace) + 1) / 2  # values range from 0 to 1, sine-distributed

    # Calculate the instantaneous frequency by interpolating between low_freq and high_freq
    freqs = low_freq + modulating_wave * (high_freq - low_freq)

    # Calculate the phase by integrating the frequency over time for continuous wave generation
    phase = 2 * np.pi * np.cumsum(freqs) / samplerate

    # Generate the waveform as a sine wave with the calculated phase
    waveform = np.sin(phase)

    # Normalize volume to 50% of the original amplitude to avoid loudness
    waveform *= 0.5

    # Play the generated waveform using sounddevice with the given sampling rate
    sd.play(waveform, samplerate=samplerate)

    # Wait until the sound playback is finished before returning
    sd.wait()


def play_success_sound(name : str):
    """
    Play a success notification sound using chime.
    """
    chime.theme(name=name)
    chime.success()
    time.sleep(3)

def play_error_sound(name : str):
    """
    Play a success notification sound using chime.
    """
    chime.theme(name=name)
    chime.error()
    time.sleep(3)


def main():
    # Example usage
    # play_siren_sound()
    # play_error_sound()
    name = "mario"
    play_success_sound(name=name)
    time.sleep(1)
    play_error_sound(name=name)


if __name__ == "__main__":
    main()

# ['big-sur', 'chime', 'mario', 'material', 'pokemon', 'sonic', 'zelda']]