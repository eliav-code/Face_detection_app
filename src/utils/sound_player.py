import os
import sys
import wave
import time
import numpy as np
from typing import Optional, Dict
from config import setup_logger

logger = setup_logger(__name__)

# Native Windows Multimedia player
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------------------
# 🎵 WAV AUDIO GENERATORS (Smooth envelopes, zero clicks)
# -------------------------------------------------------------
def _create_clean_wav(filepath: str, waveform: np.ndarray, sample_rate: int = 44100) -> None:
    """Save float32 waveform (-1.0 to 1.0) into a 16-bit PCM WAV file."""
    clipped = np.clip(waveform, -0.95, 0.95)
    pcm_data = (clipped * 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)   # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data.tobytes())

# --- Known Face Sounds ---
def _gen_harmonic_chime(sample_rate: int = 44100) -> np.ndarray:
    """Modern 2-tone chime (C5 -> G5) with exponential decay."""
    duration = 0.35
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    t1 = t[t < 0.15]
    t2 = t[t >= 0.15] - 0.15
    env1 = np.exp(-t1 * 12)
    env2 = np.exp(-t2 * 8)
    wave1 = (np.sin(2 * np.pi * 523.25 * t1) + 0.3 * np.sin(2 * np.pi * 1046.5 * t1)) * env1
    wave2 = (np.sin(2 * np.pi * 783.99 * t2) + 0.3 * np.sin(2 * np.pi * 1567.98 * t2)) * env2
    return (np.concatenate([wave1, wave2]) * 0.45).astype(np.float32)

def _gen_smart_bell(sample_rate: int = 44100) -> np.ndarray:
    """High-pitched, crisp bell chime (E5 -> B5)."""
    duration = 0.30
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    t1 = t[t < 0.12]
    t2 = t[t >= 0.12] - 0.12
    env1 = np.exp(-t1 * 15)
    env2 = np.exp(-t2 * 10)
    w1 = (np.sin(2 * np.pi * 659.25 * t1) + 0.2 * np.sin(2 * np.pi * 1318.5 * t1)) * env1
    w2 = (np.sin(2 * np.pi * 987.77 * t2) + 0.2 * np.sin(2 * np.pi * 1975.5 * t2)) * env2
    return (np.concatenate([w1, w2]) * 0.45).astype(np.float32)

def _gen_marimba_triple(sample_rate: int = 44100) -> np.ndarray:
    """Warm 3-step marimba scale (C5 -> E5 -> G5)."""
    duration = 0.36
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    step = 0.12
    t1 = t[t < step]
    t2 = t[(t >= step) & (t < 2 * step)] - step
    t3 = t[t >= 2 * step] - 2 * step
    env1 = np.exp(-t1 * 18)
    env2 = np.exp(-t2 * 18)
    env3 = np.exp(-t3 * 12)
    w1 = np.sin(2 * np.pi * 523.25 * t1) * env1
    w2 = np.sin(2 * np.pi * 659.25 * t2) * env2
    w3 = np.sin(2 * np.pi * 783.99 * t3) * env3
    return (np.concatenate([w1, w2, w3]) * 0.48).astype(np.float32)

def _gen_subtle_pip(sample_rate: int = 44100) -> np.ndarray:
    """Quick, subtle modern pip (880 Hz)."""
    duration = 0.15
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    env = np.sin(np.pi * t / duration) ** 2
    return (np.sin(2 * np.pi * 880 * t) * env * 0.40).astype(np.float32)

# --- Unknown Face Sounds ---
def _gen_security_siren(sample_rate: int = 44100) -> np.ndarray:
    """Frequency-swept alert tone (600Hz -> 450Hz)."""
    duration = 0.30
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = 600 - 150 * (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    env = np.sin(np.pi * t / duration) ** 0.5
    return (np.sin(phase) * env * 0.45).astype(np.float32)

def _gen_double_buzzer(sample_rate: int = 44100) -> np.ndarray:
    """Two sharp, professional security buzzer pulses (350 Hz)."""
    duration = 0.32
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    t1 = t[t < 0.12]
    t_pause = t[(t >= 0.12) & (t < 0.18)]
    t2 = t[t >= 0.18] - 0.18
    env1 = np.sin(np.pi * t1 / 0.12) ** 2
    env2 = np.sin(np.pi * t2 / 0.14) ** 2
    w1 = (np.sin(2 * np.pi * 380 * t1) + 0.4 * np.sin(2 * np.pi * 760 * t1)) * env1
    w_pause = np.zeros_like(t_pause)
    w2 = (np.sin(2 * np.pi * 380 * t2) + 0.4 * np.sin(2 * np.pi * 760 * t2)) * env2
    return (np.concatenate([w1, w_pause, w2]) * 0.45).astype(np.float32)

def _gen_sonar_pulse(sample_rate: int = 44100) -> np.ndarray:
    """Subtle, futuristic sonar ping (700Hz -> 200Hz)."""
    duration = 0.35
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = 700 * np.exp(-t * 8) + 200
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    env = np.exp(-t * 7)
    return (np.sin(phase) * env * 0.45).astype(np.float32)

def _gen_scifi_alarm(sample_rate: int = 44100) -> np.ndarray:
    """Fast oscillating sci-fi warning tone."""
    duration = 0.32
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = 600 + 200 * np.sin(2 * np.pi * 15 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    env = np.sin(np.pi * t / duration)
    return (np.sin(phase) * env * 0.42).astype(np.float32)


# Available Sound Catalog
SOUND_PROFILES = {
    # Known Profiles
    "Harmonic Chime": ("known", os.path.join(AUDIO_DIR, "known_harmonic.wav"), _gen_harmonic_chime),
    "Smart Bell": ("known", os.path.join(AUDIO_DIR, "known_bell.wav"), _gen_smart_bell),
    "Marimba Scale": ("known", os.path.join(AUDIO_DIR, "known_marimba.wav"), _gen_marimba_triple),
    "Subtle Pip": ("known", os.path.join(AUDIO_DIR, "known_pip.wav"), _gen_subtle_pip),

    # Unknown Profiles
    "Security Siren": ("unknown", os.path.join(AUDIO_DIR, "unknown_siren.wav"), _gen_security_siren),
    "Double Buzzer": ("unknown", os.path.join(AUDIO_DIR, "unknown_buzzer.wav"), _gen_double_buzzer),
    "Sonar Pulse": ("unknown", os.path.join(AUDIO_DIR, "unknown_sonar.wav"), _gen_sonar_pulse),
    "Sci-Fi Alarm": ("unknown", os.path.join(AUDIO_DIR, "unknown_scifi.wav"), _gen_scifi_alarm),
}

KNOWN_SOUND_OPTIONS = [name for name, (category, _, _) in SOUND_PROFILES.items() if category == "known"]
UNKNOWN_SOUND_OPTIONS = [name for name, (category, _, _) in SOUND_PROFILES.items() if category == "unknown"]

def _ensure_audio_assets():
    """Generate all sound profile WAV files if they do not already exist."""
    for name, (_, path, gen_fn) in SOUND_PROFILES.items():
        if not os.path.exists(path):
            try:
                _create_clean_wav(path, gen_fn())
            except Exception as e:
                logger.error(f"Failed to generate sound asset '{name}': {e}")

_ensure_audio_assets()


def play_audio_file_async(wav_path: str) -> None:
    """
    Play a WAV file asynchronously using the native Windows Multimedia kernel.
    """
    if not os.path.exists(wav_path):
        return

    if HAS_WINSOUND:
        try:
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return
        except Exception as e:
            logger.error(f"winsound error: {e}")

    # Fallback for non-Windows systems
    if HAS_SOUNDDEVICE:
        try:
            import soundfile as sf
            data, fs = sf.read(wav_path, dtype='float32')
            sd.play(data, fs, blocking=False)
        except Exception as e:
            logger.error(f"sounddevice fallback error: {e}")

def play_sound_by_name(sound_name: str) -> None:
    """Play any registered sound by its display name."""
    if sound_name in SOUND_PROFILES:
        _, path, _ = SOUND_PROFILES[sound_name]
        play_audio_file_async(path)


class SoundService:
    """
    High-performance, glitch-free audio service with selectable sound profiles.
    """
    def __init__(
        self,
        cooldown: float = 3.0,
        volume: float = 0.5,
        enabled: bool = True,
        known_sound: str = "Harmonic Chime",
        unknown_sound: str = "Security Siren"
    ):
        self.cooldown = cooldown
        self.volume = volume
        self.enabled = enabled
        self.known_sound = known_sound if known_sound in SOUND_PROFILES else "Harmonic Chime"
        self.unknown_sound = unknown_sound if unknown_sound in SOUND_PROFILES else "Security Siren"
        self.last_detection_times = {}

    def start(self) -> None:
        """Initialize audio subsystem."""
        _ensure_audio_assets()
        logger.info(f"SoundService ready (Known: '{self.known_sound}', Unknown: '{self.unknown_sound}')")

    def stop(self) -> None:
        """Stop any active audio playback."""
        if HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def set_known_sound(self, sound_name: str) -> None:
        if sound_name in SOUND_PROFILES:
            self.known_sound = sound_name
            logger.info(f"SoundService: Known face sound updated to '{sound_name}'")

    def set_unknown_sound(self, sound_name: str) -> None:
        if sound_name in SOUND_PROFILES:
            self.unknown_sound = sound_name
            logger.info(f"SoundService: Unknown face sound updated to '{sound_name}'")

    def play_known_preview(self) -> None:
        """Preview currently selected known face sound."""
        play_sound_by_name(self.known_sound)

    def play_unknown_preview(self) -> None:
        """Preview currently selected unknown visitor sound."""
        play_sound_by_name(self.unknown_sound)

    def queue_alert(self, sound_type: str, person_name: Optional[str] = None) -> bool:
        """
        Trigger the selected audio alert if not muted and cooldown has elapsed.
        """
        if not self.enabled:
            return False

        current_time = time.time()
        key = person_name if person_name else sound_type
        last_time = self.last_detection_times.get(key, 0.0)

        if current_time - last_time < self.cooldown:
            return False

        self.last_detection_times[key] = current_time

        if sound_type == "known":
            play_sound_by_name(self.known_sound)
            return True
        elif sound_type == "unknown":
            play_sound_by_name(self.unknown_sound)
            return True

        return False

    def toggle_sound(self) -> bool:
        """Toggle mute state and return new enabled state."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop()
        return self.enabled