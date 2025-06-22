import queue
import threading
import simpleaudio as sa
import numpy as np
import time

class SoundPlayer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.sound_queue = queue.Queue()
        # self.current_play_obj = None

    def run(self):
        while True:
            sound_type = self.sound_queue.get()  # מחכה לצליל חדש
            if sound_type == "wonderful":
                self.play_wonderful_sound()
            elif sound_type == "siren":
                self.play_siren_sound()

    def play_wonderful_sound(self, duration=1, f0=200, sample_rate=44100):
        # if self.current_play_obj and self.current_play_obj.is_playing():
        #     return
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = (
            1.0 * np.sin(2 * np.pi * f0 * t) +
            0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.3 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        waveform /= np.max(np.abs(waveform))
        audio = (waveform * 32767).astype(np.int16)
        sa.play_buffer(audio, 1, 2, sample_rate)

    def play_siren_sound(self, duration=1, low_freq=600, high_freq=1200, rate=0.25, sample_rate=44100):
        # if self.current_play_obj and self.current_play_obj.is_playing():
        #     return
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        modulating = (np.sin(2 * np.pi * rate * t) + 1) / 2
        freqs = low_freq + modulating * (high_freq - low_freq)
        phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
        waveform = np.sin(phase)
        waveform *= 0.5
        audio = (waveform * 32767).astype(np.int16)
        sa.play_buffer(audio, 1, 2, sample_rate)

    def queue_sound(self, sound_type):
        self.sound_queue.put(sound_type)