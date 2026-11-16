"""
tests/test_audio.py - Tests for NumPy synthesized acoustic pulse patterns.
"""

import unittest
import io
import wave
import numpy as np

from core.audio_feedback import AcousticRhythmSynthesizer, DiscreetAudioEngine


class TestAudioFeedback(unittest.TestCase):

    def setUp(self):
        self.synth = AcousticRhythmSynthesizer
        self.engine = DiscreetAudioEngine(enabled=True)

    def test_sine_pulse_generation(self):
        """Verify sine pulse waveform duration, amplitude bounds, and envelope."""
        duration_ms = 100.0
        freq = 880.0
        pulse = self.synth.generate_sine_pulse(freq, duration_ms, amplitude=0.5)

        expected_samples = int(self.synth.SAMPLE_RATE * (duration_ms / 1000.0))
        self.assertEqual(len(pulse), expected_samples)
        self.assertEqual(pulse.dtype, np.int16)

        # Pulse endpoints must fade to zero due to attack/decay windowing
        self.assertLess(abs(pulse[0]), 50)
        self.assertLess(abs(pulse[-1]), 50)

        # Peak values should be close to 0.5 * 32767
        max_val = np.max(np.abs(pulse))
        self.assertGreater(max_val, 10000)
        self.assertLessEqual(max_val, 32767)

    def test_sawtooth_buzz_generation(self):
        """Verify dissonant counterfeit alarm waveform."""
        buzz = self.synth.generate_sawtooth_buzz(frequency_hz=180.0, duration_ms=150.0)
        expected_samples = int(self.synth.SAMPLE_RATE * (150.0 / 1000.0))
        self.assertEqual(len(buzz), expected_samples)
        self.assertEqual(buzz.dtype, np.int16)

    def test_wav_riff_header_structure(self):
        """Verify that synthesized PCM wraps into a standard compliant WAV file."""
        pulse = self.synth.generate_sine_pulse(440.0, 50.0)
        wav_bytes = self.synth.pcm_to_wav_bytes(pulse)

        self.assertIsInstance(wav_bytes, bytes)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_bytes[:16])

        # Validate by parsing back with standard library wave module
        bio = io.BytesIO(wav_bytes)
        with wave.open(bio, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), self.synth.SAMPLE_RATE)
            self.assertEqual(wf.getnframes(), len(pulse))

    def test_audio_engine_library_and_debounce(self):
        """Verify denomination mappings and debouncing logic."""
        # Ensure all required denominations are pre-cached
        for denom in ["10", "20", "50", "100", "200", "500", "2000", "COUNTERFEIT_ALARM"]:
            wav = self.engine.get_wav_bytes(denom)
            self.assertIsNotNone(wav, f"Missing audio pattern for {denom}")
            self.assertGreater(len(wav), 44)  # Larger than header

        # Test debounce suppression
        triggered1 = self.engine.trigger_feedback(denomination="500", is_counterfeit=False)
        self.assertTrue(triggered1)

        # Immediate second trigger of same denomination should be debounced
        triggered2 = self.engine.trigger_feedback(denomination="500", is_counterfeit=False)
        self.assertFalse(triggered2)


if __name__ == "__main__":
    unittest.main()

