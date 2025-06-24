from random import sample
import wave
import numpy as np
import sounddevice as sd
import winsound
import chime
import time
# import pygame
from typing import List

def play_siren_sound(duration : int = 0.2, low_freq : int = 600, high_freq : int = 1200, rate : int = 0.25, sample_rate : int = 44100):
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

def play_wonderful_sound(duration=0.2, f0=200, sample_rate=44100):
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

# def init_sound_system():
#         """Initialize pygame mixer for sound playback"""
#         try:
#             pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
#             print("Sound system initialized successfully")
#         except Exception as e:
#             print(f"Failed to initialize sound system: {e}")


# def play_beep_sequence(frequencies : List[int], duration : float):
#         """Generate and play a sequence of beeps - simplified version"""
#         try:
#             sample_rate = 22050
            
#             for freq in frequencies:
#                 # Generate sine wave
#                 samples = int(sample_rate * duration)
#                 t = np.arange(samples) / sample_rate
#                 wave = np.sin(2 * np.pi * freq * t) * 0.3
                
#                 # Convert to 16-bit integers and ensure contiguous array
#                 wave = (wave * 32767).astype(np.int16)
#                 wave = np.ascontiguousarray(wave)
                
#                 # Play the sound (mono)
#                 sound = pygame.sndarray.make_sound(wave)
#                 sound.play()
                
#                 # Wait for sound to finish
#                 time.sleep(duration + 0.1)
                
#         except Exception as e:
#             print(f"Error generating beep: {e}")
#             # Fallback: simple print beep
#             print(f"BEEP: {freq}Hz")

def play_sound_sync(sound_type : str) -> None:
    try:
        if sound_type == "known":
            # Play success/welcome sound
            play_wonderful_sound()
        elif sound_type == "unknown":
            # Play alert sound
            play_siren_sound()
    except Exception as e:
        print(f"Error playing sound: {e}")
             
    