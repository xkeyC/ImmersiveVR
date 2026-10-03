// Plays the PC's sound: 10 ms packets of 48 kHz stereo s16 PCM arrive on the
// port and wait in a ring buffer kept short. It starts playing once TARGET is
// buffered, plays silence (and refills to TARGET) when it runs dry, and drops
// the oldest sound when more than MAX is waiting, so latency cannot grow.

const RATE = 48000;
const TARGET = Math.round(RATE * 0.04);   // 40 ms
const MAX = Math.round(RATE * 0.12);      // 120 ms
const SIZE = RATE;                        // ring capacity: 1 s of frames

class Player extends AudioWorkletProcessor {
  constructor() {
    super();
    this.left = new Float32Array(SIZE);
    this.right = new Float32Array(SIZE);
    this.read = 0;
    this.count = 0;
    this.playing = false;
    this.underruns = 0;
    this.reported = 0;
    this.port.onmessage = (event) => this.push(event.data);
  }

  push(pcm) {
    const frames = pcm.length >> 1;
    let write = (this.read + this.count) % SIZE;
    for (let i = 0; i < frames; i++) {
      this.left[write] = pcm[2 * i] / 32768;
      this.right[write] = pcm[2 * i + 1] / 32768;
      write = (write + 1) % SIZE;
    }
    this.count = Math.min(this.count + frames, SIZE);
    if (this.count > MAX) {
      const drop = this.count - TARGET;
      this.read = (this.read + drop) % SIZE;
      this.count -= drop;
    }
  }

  process(_, outputs) {
    const [left, right] = outputs[0];
    const frames = left.length;
    if (!this.playing && this.count >= TARGET) this.playing = true;
    if (this.playing && this.count < frames) {
      this.playing = false;
      this.underruns++;
    }
    if (this.playing) {
      for (let i = 0; i < frames; i++) {
        left[i] = this.left[this.read];
        if (right) right[i] = this.right[this.read];
        this.read = (this.read + 1) % SIZE;
      }
      this.count -= frames;
    }
    if (currentTime - this.reported >= 1) {
      this.reported = currentTime;
      this.port.postMessage({ bufferedMs: this.count / RATE * 1000, underruns: this.underruns });
    }
    return true;
  }
}

registerProcessor("ivr-player", Player);
