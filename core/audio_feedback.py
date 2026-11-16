"""
core/audio_feedback.py - Discreet Audio Feedback Engine.

Synthesizes acoustic byte-rhythms (NumPy generated sine/sawtooth pulses)
and streams them asynchronously via in-memory WAV buffers.
Avoids noisy Text-to-Speech in favor of discreet, tactile auditory signals.
"""

from typing import Dict, List, Optional, Tuple
import io
import struct
import time
import wave
import threading
import numpy as np

# Use native Windows winsound for zero-latency, zero-blocking async in-memory WAV playback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AcousticRhythmSynthesizer:
    """
    NumPy-based waveform synthesizer that encodes acoustic pulse patterns
    into 16-bit PCM WAV byte streams.
    """

    SAMPLE_RATE: int = 44100

    @classmethod
    def generate_sine_pulse(
        cls,
        frequency_hz: float,
        duration_ms: float,
        amplitude: float = 0.5,
        attack_ms: float = 5.0,
        decay_ms: float = 10.0
    ) -> np.ndarray:
        """
        Generate a single sine wave pulse with smooth attack and decay envelopes
        to prevent speaker pop/click transients.
        """
        num_samples = int(cls.SAMPLE_RATE * (duration_ms / 1000.0))
        if num_samples <= 0:
            return np.zeros(0, dtype=np.int16)

        t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
        wave_data = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)

        # Smooth linear envelope
        attack_samples = min(num_samples // 2, int(cls.SAMPLE_RATE * (attack_ms / 1000.0)))
        decay_samples = min(num_samples // 2, int(cls.SAMPLE_RATE * (decay_ms / 1000.0)))

        if attack_samples > 0:
            wave_data[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
        if decay_samples > 0:
            wave_data[-decay_samples:] *= np.linspace(1.0, 0.0, decay_samples)

        # Scale to 16-bit signed integer range
        pcm16 = np.clip(wave_data * 32767.0, -32768, 32767).astype(np.int16)
        return pcm16

    @classmethod
    def generate_silence(cls, duration_ms: float) -> np.ndarray:
        """Generate silent pause in milliseconds."""
        num_samples = int(cls.SAMPLE_RATE * (duration_ms / 1000.0))
        return np.zeros(num_samples, dtype=np.int16)

    @classmethod
    def generate_sawtooth_buzz(
        cls,
        frequency_hz: float = 180.0,
        dissonance_hz: float = 194.0,
        duration_ms: float = 220.0,
        amplitude: float = 0.6
    ) -> np.ndarray:
        """
        Generate a harsh, discordant dual-frequency buzz for Counterfeit/Suspect alert.
        """
        num_samples = int(cls.SAMPLE_RATE * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)

        # Sawtooth wave via Fourier synthesis or phase modulo
        saw1 = 2.0 * (t * frequency_hz - np.floor(0.5 + t * frequency_hz))
        # Dissonant beating wave
        saw2 = 2.0 * (t * dissonance_hz - np.floor(0.5 + t * dissonance_hz))
        mixed = amplitude * 0.5 * (saw1 + saw2)

        # Truncate clicks
        envelope = np.ones(num_samples)
        fade = min(num_samples // 4, int(cls.SAMPLE_RATE * 0.015))
        if fade > 0:
            envelope[:fade] = np.linspace(0.0, 1.0, fade)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade)

        pcm16 = np.clip(mixed * envelope * 32767.0, -32768, 32767).astype(np.int16)
        return pcm16

    @classmethod
    def pcm_to_wav_bytes(cls, pcm_array: np.ndarray) -> bytes:
        """Wrap raw int16 PCM array into canonical RIFF WAV byte format."""
        byte_io = io.BytesIO()
        with wave.open(byte_io, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(cls.SAMPLE_RATE)
            wf.writeframes(pcm_array.tobytes())
        return byte_io.getvalue()


class DiscreetAudioEngine:
    """
    Manages acoustic byte-rhythm mappings for denominations and counterfeit alarms.
    Includes debouncing, asynchronous playback, and non-blocking worker threads.
    """

    def __init__(self, debounce_interval_sec: float = 1.6, enabled: bool = True) -> None:
        self.debounce_interval_sec = debounce_interval_sec
        self.enabled = enabled
        self.last_played_time: float = 0.0
        self.last_played_tag: str = ""
        self._sound_cache: Dict[str, bytes] = {}
        self._lock = threading.Lock()

        # Pre-synthesize acoustic signatures
        self._build_rhythm_library()

    def _build_rhythm_library(self) -> None:
        """Pre-compile WAV byte rhythms for instant zero-latency playback."""
        synth = AcousticRhythmSynthesizer

        # Denomination 10: Single crisp high pulse (880 Hz, 80ms)
        p10 = synth.generate_sine_pulse(880.0, 80.0, amplitude=0.45)
        self._sound_cache["10"] = synth.pcm_to_wav_bytes(p10)

        # Denomination 20: Double pulse rhythm (880 Hz, two 45ms pulses with 40ms pause)
        p20 = np.concatenate([
            synth.generate_sine_pulse(880.0, 45.0, amplitude=0.45),
            synth.generate_silence(40.0),
            synth.generate_sine_pulse(880.0, 45.0, amplitude=0.45)
        ])
        self._sound_cache["20"] = synth.pcm_to_wav_bytes(p20)

        # Denomination 50: Rising major third interval (659 Hz -> 830 Hz)
        p50 = np.concatenate([
            synth.generate_sine_pulse(659.25, 55.0, amplitude=0.45),
            synth.generate_silence(25.0),
            synth.generate_sine_pulse(830.61, 75.0, amplitude=0.50)
        ])
        self._sound_cache["50"] = synth.pcm_to_wav_bytes(p50)

        # Denomination 100: Distinct double rising fifth chime (587 Hz -> 880 Hz)
        p100 = np.concatenate([
            synth.generate_sine_pulse(587.33, 60.0, amplitude=0.45),
            synth.generate_silence(30.0),
            synth.generate_sine_pulse(880.0, 90.0, amplitude=0.55)
        ])
        self._sound_cache["100"] = synth.pcm_to_wav_bytes(p100)

        # Denomination 200: Triple staccato pulse (784 Hz x 3)
        p200 = np.concatenate([
            synth.generate_sine_pulse(783.99, 40.0, amplitude=0.45),
            synth.generate_silence(30.0),
            synth.generate_sine_pulse(783.99, 40.0, amplitude=0.45),
            synth.generate_silence(30.0),
            synth.generate_sine_pulse(783.99, 50.0, amplitude=0.45)
        ])
        self._sound_cache["200"] = synth.pcm_to_wav_bytes(p200)

        # Denomination 500: Harmonic resonant chord triad (523 Hz + 659 Hz + 784 Hz)
        chord_triad = np.concatenate([
            synth.generate_sine_pulse(523.25, 60.0, amplitude=0.35),
            synth.generate_sine_pulse(659.25, 60.0, amplitude=0.35),
            synth.generate_sine_pulse(783.99, 120.0, amplitude=0.55)
        ])
        self._sound_cache["500"] = synth.pcm_to_wav_bytes(chord_triad)

        # Denomination 2000: Arpeggio sequence (523 Hz -> 659 Hz -> 784 Hz -> 1046 Hz)
        p2000 = np.concatenate([
            synth.generate_sine_pulse(523.25, 45.0, amplitude=0.35),
            synth.generate_silence(15.0),
            synth.generate_sine_pulse(659.25, 45.0, amplitude=0.35),
            synth.generate_silence(15.0),
            synth.generate_sine_pulse(783.99, 45.0, amplitude=0.40),
            synth.generate_silence(15.0),
            synth.generate_sine_pulse(1046.50, 90.0, amplitude=0.55)
        ])
        self._sound_cache["2000"] = synth.pcm_to_wav_bytes(p2000)

        # Counterfeit Alarm: Dissonant dual-pulse warning buzz
        alarm = np.concatenate([
            synth.generate_sawtooth_buzz(180.0, 194.0, 120.0, amplitude=0.65),
            synth.generate_silence(40.0),
            synth.generate_sawtooth_buzz(180.0, 194.0, 160.0, amplitude=0.70)
        ])
        self._sound_cache["COUNTERFEIT_ALARM"] = synth.pcm_to_wav_bytes(alarm)

        # Subtle click confirmation
        click = synth.generate_sine_pulse(1200.0, 15.0, amplitude=0.15)
        self._sound_cache["CLICK"] = synth.pcm_to_wav_bytes(click)

    def trigger_feedback(
        self,
        denomination: str,
        is_counterfeit: bool,
        force: bool = False
    ) -> bool:
        """
        Trigger asynchronous acoustic feedback based on detection results.
        Enforces debouncing to avoid acoustic fatigue.
        """
        if not self.enabled:
            return False

        current_time = time.time()
        tag = "COUNTERFEIT_ALARM" if is_counterfeit else str(denomination)

        # Suppress sound if same tag triggered within debounce window
        if not force:
            if tag == self.last_played_tag and (current_time - self.last_played_time) < self.debounce_interval_sec:
                return False
            if (current_time - self.last_played_time) < 0.45:
                # Minimum spacing between any two sounds
                return False

        wav_bytes = self._sound_cache.get(tag)
        if wav_bytes is None:
            # Fallback to generic click if denomination is unknown
            wav_bytes = self._sound_cache.get("CLICK")

        if wav_bytes is None:
            return False

        self.last_played_time = current_time
        self.last_played_tag = tag

        # Dispatch async playback without stalling main loop
        threading.Thread(target=self._play_wav_buffer, args=(wav_bytes,), daemon=True).start()
        return True

    def _play_wav_buffer(self, wav_bytes: bytes) -> None:
        """Low-level sound playback."""
        with self._lock:
            if HAS_WINSOUND:
                try:
                    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
                except Exception:
                    pass
            else:
                # Alternative fallback can be implemented if on non-Windows
                pass

    def get_wav_bytes(self, tag: str) -> Optional[bytes]:
        """Retrieve pre-compiled WAV bytes for verification/testing."""
        return self._sound_cache.get(tag)

