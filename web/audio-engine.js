/**
 * ============================================================================
 * CETACEAN ACOUSTIC SYNTHESIS ENGINE
 * High-fidelity Bioacoustic Sound Generator & Real-time Web Audio Synthesizer
 * ============================================================================
 */

class CetaceanAudioEngine {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.sourceNode = null;
    this.gainNode = null;
    this.isPlaying = false;
    this.isMuted = false;
    this.currentBuffer = null;
    this.startTime = 0;
    this.pausedAt = 0;
    this.duration = 0;
    this.animFrameId = null;

    // Callbacks
    this.onPlaybackProgress = null;
    this.onPlaybackEnd = null;
    this.onPlaybackStart = null;
  }

  /**
   * Initializes or resumes the AudioContext on user interaction
   */
  async initContext() {
    if (!this.ctx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioContextClass();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.85;

      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.setValueAtTime(0.85, this.ctx.currentTime);

      this.analyser.connect(this.gainNode);
      this.gainNode.connect(this.ctx.destination);
    }

    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }
  }

  /**
   * Evaluates pitch contour function f0(t) for a given GMM component and parameters
   * @param {number} t - Time in seconds (0 <= t <= duration)
   * @param {number} duration - Call duration in seconds
   * @param {number} componentId - GMM cluster component index (1 to 8)
   * @param {number} fCenter - Center frequency in Hz (e.g., 8500)
   * @param {number} bandwidth - Frequency span in Hz (e.g., 8000)
   * @returns {number} Frequency in Hz at time t
   */
  getFrequencyAtTime(t, duration, componentId, fCenter = 9000, bandwidth = 7000) {
    const normT = Math.min(Math.max(t / duration, 0), 1);
    const halfBw = bandwidth * 0.5;

    switch (componentId) {
      case 1:
        // Signature Whistle: Modulated multi-loop contour (characteristic individual cetacean ID)
        return fCenter + halfBw * Math.sin(normT * Math.PI * 4) * (1 - 0.2 * normT);

      case 2:
        // Ascending Upsweep: Monotonically rising whistle (common contact call)
        return (fCenter - halfBw * 0.9) + (bandwidth * 0.9) * Math.pow(normT, 0.85);

      case 3:
        // Descending Downsweep: High to low frequency sweep
        return (fCenter + halfBw * 0.9) - (bandwidth * 0.9) * Math.pow(normT, 0.85);

      case 4:
        // Convex Chirp: Inverted parabolic arc (arch whistle)
        return (fCenter - halfBw * 0.7) + (bandwidth * 0.9) * Math.sin(normT * Math.PI);

      case 5:
        // Constant Frequency Carrier with subtle vibrato (tonal harmonic whistle)
        return fCenter + (bandwidth * 0.08) * Math.sin(normT * Math.PI * 12);

      case 6:
        // Modulated Trill: Rapid sinusoidal frequency oscillation
        return fCenter + halfBw * 0.65 * Math.sin(normT * Math.PI * 16) + (bandwidth * 0.2) * normT;

      case 7:
        // Stepped Harmonic Whistle: Two distinct frequency steps connected by rapid inflection
        if (normT < 0.45) return fCenter - halfBw * 0.6;
        if (normT < 0.55) {
          const trans = (normT - 0.45) / 0.1;
          return (fCenter - halfBw * 0.6) + (bandwidth * 0.8) * trans;
        }
        return fCenter + halfBw * 0.6;

      case 8:
      default:
        // Composite Whistle-Burst: Modulated contour with acoustic inflections
        return fCenter + halfBw * 0.75 * Math.sin(normT * Math.PI * 2.5 + Math.cos(normT * Math.PI * 6) * 0.5);
    }
  }

  /**
   * Generates a realistic synthetic bioacoustic cetacean whistle audio buffer
   * Includes fundamental f0, harmonics (2f0, 3f0), hydrophone water absorption,
   * gentle ocean ambient hum, and click-train bursts for composite components.
   */
  async synthesizeWhistleBuffer({
    componentId = 1,
    duration = 1.8,
    fCenter = 9500,
    bandwidth = 8000,
    harmonicRatio = 0.35,
    sampleRate = 48000
  }) {
    await this.initContext();

    const actualSampleRate = this.ctx.sampleRate || sampleRate;
    const numSamples = Math.floor(duration * actualSampleRate);
    const audioBuffer = this.ctx.createBuffer(1, numSamples, actualSampleRate);
    const channelData = audioBuffer.getChannelData(0);

    let phase0 = 0;
    let phase1 = 0;
    let phase2 = 0;

    const twoPiOverSr = (2 * Math.PI) / actualSampleRate;

    for (let i = 0; i < numSamples; i++) {
      const t = i / actualSampleRate;
      const normT = t / duration;

      // 1. Calculate Instantaneous Fundamental Frequency
      const f0 = this.getFrequencyAtTime(t, duration, componentId, fCenter, bandwidth);
      const f1 = f0 * 2.0; // 2nd harmonic
      const f2 = f0 * 3.0; // 3rd harmonic

      // Accumulate phase accurately to prevent discontinuities
      phase0 += f0 * twoPiOverSr;
      phase1 += f1 * twoPiOverSr;
      phase2 += f2 * twoPiOverSr;

      // 2. Smooth Bioacoustic Amplitude Envelope (Attack, Sustain, Decay)
      let env = 1.0;
      const attackTime = 0.08;
      const releaseTime = 0.12;

      if (t < attackTime) {
        env = Math.sin((t / attackTime) * (Math.PI / 2));
      } else if (t > duration - releaseTime) {
        const relT = (t - (duration - releaseTime)) / releaseTime;
        env = Math.cos(relT * (Math.PI / 2));
      }

      // 3. Synthesize Fundamental and Harmonics
      const h0 = Math.sin(phase0);
      const h1 = Math.sin(phase1) * (harmonicRatio * 0.45);
      const h2 = Math.sin(phase2) * (harmonicRatio * 0.18);

      let signal = (h0 + h1 + h2) * env * 0.65;

      // 4. Composite Component 8: Inject hydrophone echolocation clicks
      if (componentId === 8 && normT > 0.25 && normT < 0.75) {
        // Intermittent broadband click pulses every 40ms
        const clickInterval = Math.floor(actualSampleRate * 0.04);
        const clickPhase = i % clickInterval;
        if (clickPhase < 8) {
          // Sharp decaying impulse
          const clickDecay = Math.exp(-clickPhase * 0.4);
          const clickNoise = (Math.random() * 2 - 1) * 0.35 * clickDecay;
          signal += clickNoise;
        }
      }

      // 5. Subtle ambient hydrophone background noise (deep ocean hum)
      const ambientHum = (Math.random() * 2 - 1) * 0.008;
      signal += ambientHum;

      // Clamp to prevent digital clipping
      channelData[i] = Math.max(-0.95, Math.min(0.95, signal));
    }

    this.currentBuffer = audioBuffer;
    this.duration = duration;
    return audioBuffer;
  }

  /**
   * A tone that follows an arbitrary pitch contour (Hz, evenly spaced over ``duration``), with the same envelope and
   * soft harmonics as synthesizeWhistleBuffer. Used for the film's hand-drawn "new whistle".
   */
  async synthesizeFromContour(freqs, duration = 1.0, harmonicRatio = 0.35) {
    await this.initContext();
    const sr = this.ctx.sampleRate, n = Math.floor(duration * sr), m = freqs.length;
    const buf = this.ctx.createBuffer(1, n, sr), x = buf.getChannelData(0);
    let p0 = 0, p1 = 0, p2 = 0;
    for (let i = 0; i < n; i++) {
      const t = i / sr, u = (i / (n - 1)) * (m - 1), k = Math.min(m - 2, Math.floor(u)), w = u - k;
      const f0 = freqs[k] * (1 - w) + freqs[k + 1] * w;
      p0 += 2 * Math.PI * f0 / sr; p1 += 4 * Math.PI * f0 / sr; p2 += 6 * Math.PI * f0 / sr;
      let env = 1;
      if (t < 0.08) env = Math.sin((t / 0.08) * (Math.PI / 2));
      else if (t > duration - 0.12) env = Math.cos(((t - (duration - 0.12)) / 0.12) * (Math.PI / 2));
      x[i] = (Math.sin(p0) + Math.sin(p1) * harmonicRatio * 0.45 + Math.sin(p2) * harmonicRatio * 0.18) * env * 0.65;
    }
    this.currentBuffer = buf;
    this.duration = duration;
    return buf;
  }

  /**
   * Plays the generated audio buffer
   */
  play(startOffset = 0) {
    if (!this.currentBuffer || !this.ctx) return;

    this.stop();

    this.sourceNode = this.ctx.createBufferSource();
    this.sourceNode.buffer = this.currentBuffer;
    this.sourceNode.connect(this.analyser);

    const offset = Math.max(0, Math.min(startOffset, this.duration));
    this.startTime = this.ctx.currentTime - offset;

    this.sourceNode.start(0, offset);
    this.isPlaying = true;

    if (this.onPlaybackStart) {
      this.onPlaybackStart(offset);
    }

    this.sourceNode.onended = () => {
      if (this.isPlaying) {
        this.isPlaying = false;
        this.pausedAt = 0;
        cancelAnimationFrame(this.animFrameId);
        if (this.onPlaybackEnd) {
          this.onPlaybackEnd();
        }
      }
    };

    this.trackProgress();
  }

  /**
   * Stops playback
   */
  stop() {
    if (this.sourceNode) {
      try {
        this.sourceNode.stop();
        this.sourceNode.disconnect();
      } catch (e) {
        // Node was already stopped
      }
      this.sourceNode = null;
    }
    this.isPlaying = false;
    cancelAnimationFrame(this.animFrameId);
  }

  /**
   * Pauses playback
   */
  pause() {
    if (!this.isPlaying) return;
    this.pausedAt = this.getCurrentTime();
    this.stop();
  }

  /**
   * Toggles Play/Pause
   */
  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play(this.pausedAt);
    }
    return this.isPlaying;
  }

  /**
   * Returns current elapsed playback time in seconds
   */
  getCurrentTime() {
    if (!this.isPlaying || !this.ctx) return this.pausedAt;
    const elapsed = this.ctx.currentTime - this.startTime;
    return Math.min(elapsed, this.duration);
  }

  /**
   * Frame-by-frame progress tracking loop
   */
  trackProgress() {
    const loop = () => {
      if (!this.isPlaying) return;
      const curr = this.getCurrentTime();

      if (this.onPlaybackProgress) {
        this.onPlaybackProgress(curr, this.duration);
      }

      if (curr < this.duration) {
        this.animFrameId = requestAnimationFrame(loop);
      }
    };
    this.animFrameId = requestAnimationFrame(loop);
  }

  /**
   * Returns current real-time waveform data from the AnalyserNode
   */
  getTimeDomainData() {
    if (!this.analyser) return new Uint8Array(512);
    const data = new Uint8Array(this.analyser.fftSize);
    this.analyser.getByteTimeDomainData(data);
    return data;
  }

  /**
   * Returns current real-time frequency spectrum data from the AnalyserNode
   */
  getFrequencyData() {
    if (!this.analyser) return new Uint8Array(256);
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    return data;
  }

  /**
   * Sets volume multiplier
   */
  setVolume(val) {
    if (this.gainNode && this.ctx) {
      this.gainNode.gain.setValueAtTime(Math.max(0, Math.min(val, 1)), this.ctx.currentTime);
    }
  }

  /**
   * Encodes audio buffer to a standard downloadable 16-bit PCM WAV Blob
   */
  exportToWav() {
    if (!this.currentBuffer) return null;

    const numChannels = this.currentBuffer.numberOfChannels;
    const sampleRate = this.currentBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    const data = this.currentBuffer.getChannelData(0);
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = data.length * bytesPerSample;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const arrayBuffer = new ArrayBuffer(totalSize);
    const view = new DataView(arrayBuffer);

    // RIFF chunk descriptor
    this.writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    this.writeString(view, 8, "WAVE");

    // "fmt " sub-chunk
    this.writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, format, true); // AudioFormat
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);

    // "data" sub-chunk
    this.writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    // Write audio samples
    let offset = 44;
    for (let i = 0; i < data.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, data[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([view], { type: "audio/wav" });
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}

window.CetaceanAudioEngine = CetaceanAudioEngine;