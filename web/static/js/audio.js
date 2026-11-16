/**
 * web/static/js/audio.js - Web Audio API Discreet Acoustic Byte-Rhythm Engine.
 * 
 * Synthesizes pure sine waves, musical chords, and discordant sawtooth alarms
 * directly in the browser with precise microsecond envelopes and attack/decay smoothing.
 */

class DiscreetWebAudioEngine {
    constructor() {
        this.ctx = null;
        this.enabled = true;
        this.lastTriggerTime = 0;
        this.lastTriggerTag = '';
        this.debounceWindowMs = 1600;
        this.minSpacingMs = 450;
    }

    initContext() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggleMute() {
        this.enabled = !this.enabled;
        return this.enabled;
    }

    createEnvelopePulse(frequency, durationMs, amplitude = 0.35, type = 'sine', startTimeOffset = 0) {
        if (!this.ctx || !this.enabled) return;

        const now = this.ctx.currentTime + startTimeOffset;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = type;
        osc.frequency.setValueAtTime(frequency, now);

        const durSec = durationMs / 1000.0;
        const attack = Math.min(0.015, durSec * 0.2);
        const decay = Math.min(0.025, durSec * 0.3);

        // Smooth attack & decay envelope
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(amplitude, now + attack);
        gain.gain.setValueAtTime(amplitude, now + durSec - decay);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + durSec);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(now);
        osc.stop(now + durSec + 0.05);
    }

    trigger(tag) {
        if (!this.enabled || !tag) return false;
        this.initContext();

        const now = Date.now();
        // Debounce identical denomination within window
        if (tag === this.lastTriggerTag && (now - this.lastTriggerTime) < this.debounceWindowMs) {
            return false;
        }
        if ((now - this.lastTriggerTime) < this.minSpacingMs) {
            return false;
        }

        this.lastTriggerTime = now;
        this.lastTriggerTag = tag;

        switch (String(tag)) {
            case '10':
                // 10: Single crisp high pulse (880 Hz, 80ms)
                this.createEnvelopePulse(880.0, 80, 0.35);
                break;

            case '20':
                // 20: Double pulse rhythm
                this.createEnvelopePulse(880.0, 45, 0.35, 'sine', 0);
                this.createEnvelopePulse(880.0, 45, 0.35, 'sine', 0.085);
                break;

            case '50':
                // 50: Melodic rising third (659 Hz -> 830 Hz)
                this.createEnvelopePulse(659.25, 55, 0.35, 'sine', 0);
                this.createEnvelopePulse(830.61, 75, 0.40, 'sine', 0.08);
                break;

            case '100':
                // 100: Rising fifth chime (587 Hz -> 880 Hz)
                this.createEnvelopePulse(587.33, 60, 0.35, 'sine', 0);
                this.createEnvelopePulse(880.00, 90, 0.45, 'sine', 0.09);
                break;

            case '200':
                // 200: Triple staccato pulse (784 Hz x 3)
                this.createEnvelopePulse(783.99, 40, 0.35, 'sine', 0);
                this.createEnvelopePulse(783.99, 40, 0.35, 'sine', 0.07);
                this.createEnvelopePulse(783.99, 50, 0.40, 'sine', 0.14);
                break;

            case '500':
                // 500: Resonant major triad chord (523 Hz + 659 Hz + 784 Hz)
                this.createEnvelopePulse(523.25, 140, 0.22, 'sine', 0);
                this.createEnvelopePulse(659.25, 140, 0.22, 'sine', 0);
                this.createEnvelopePulse(783.99, 160, 0.32, 'sine', 0.03);
                break;

            case '2000':
                // 2000: Ascending four-tone arpeggio
                this.createEnvelopePulse(523.25, 45, 0.25, 'sine', 0);
                this.createEnvelopePulse(659.25, 45, 0.25, 'sine', 0.055);
                this.createEnvelopePulse(783.99, 45, 0.30, 'sine', 0.11);
                this.createEnvelopePulse(1046.5, 90, 0.40, 'sine', 0.165);
                break;

            case 'COUNTERFEIT_ALARM':
                // Discordant dual sawtooth buzz (180 Hz + 194 Hz)
                this.createEnvelopePulse(180.0, 140, 0.45, 'sawtooth', 0);
                this.createEnvelopePulse(194.0, 140, 0.45, 'sawtooth', 0);
                this.createEnvelopePulse(180.0, 180, 0.50, 'sawtooth', 0.18);
                this.createEnvelopePulse(194.0, 180, 0.50, 'sawtooth', 0.18);
                break;

            case 'CLICK':
                // Subtle confirmation tick
                this.createEnvelopePulse(1200.0, 15, 0.15, 'sine', 0);
                break;

            default:
                this.createEnvelopePulse(660.0, 30, 0.15, 'sine', 0);
                break;
        }

        return true;
    }
}

// Global instance
window.audioEngine = new DiscreetWebAudioEngine();

