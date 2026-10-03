// The PC's sound in the headset: packets from the stream go to an
// AudioWorklet player (audio-worklet.js). An AudioContext may only start on
// a user gesture, so startAudio() runs from the 进入 VR and 预览 buttons.

let context = null;
let node = null;

/** What the player reported last (diagnostics). */
export const audio = { bufferedMs: null, underruns: 0 };

export async function startAudio() {
  try {
    if (context) {
      await context.resume();
      return;
    }
    context = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
    await context.audioWorklet.addModule("js/audio-worklet.js");
    node = new AudioWorkletNode(context, "ivr-player", { numberOfInputs: 0, outputChannelCount: [2] });
    node.port.onmessage = (event) => Object.assign(audio, event.data);
    node.connect(context.destination);
  } catch (error) {
    console.warn("[ivr] no sound:", error);
  }
}

/** One packet: interleaved stereo s16 samples. */
export function playPacket(pcm) {
  node?.port.postMessage(pcm, [pcm.buffer]);
}
