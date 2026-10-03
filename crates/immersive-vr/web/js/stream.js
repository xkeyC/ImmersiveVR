// The WebSocket stream and its decoder.
//
// The server sends its stream info as JSON text (again on every
// reconfiguration), then binary chunks:
//   [0] kind=1, [1] flags (bit0 key), [2..10) capture time, Unix us (u64 LE),
//   [10..14) divergence and [14..18) convergence the frame was rendered with (f32 LE),
//   [18..) one encoded frame (H.265 Annex B or AV1 OBUs);
// and sound: [0] kind=2, [1] 0, [2..10) capture time, [10..) 10 ms of 48 kHz stereo s16 PCM.
// Each frame holds both eyes rendered on the PC (iw3 mlbw_l2): left eye on
// top, right eye below.
// Settings go back as {"divergence","convergence","resolution","codec"};
// {"ping": ms} measures the clock offset to the PC (answered with a pong),
// {"keyframe": true} asks for a keyframe after a decoder lost its place.

import { playPacket } from "./audio.js";
import { settings, setSettings, SERVER_KEYS } from "./settings.js";
import { setStatus } from "./ui.js";

export const CODEC_LABELS = { hevc: "H.265", av1: "AV1" };

export const stream = {
  /** The current stream info (codec, layout, resolutions, ...), or null. */
  info: null,
  /** Why frames cannot be shown, if they cannot. */
  problem: "",
  /** Latest one-line summary (resolution, codec, fps, bitrate, latency). */
  summary: "",
  /** Last second's frame counts, for diagnostics. */
  received: 0,
  decoded: 0,
  /** Last second's median capture -> received / -> shown latency (ms), and the round trip; null until known. */
  latency: { received: null, shown: null, rtt: null },
};

// The PC's clock relative to ours, from ping round trips: the offset of the
// quickest recent one (least queueing, so the most exact).
const clock = { offset: null, samples: [] };
function ping() {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ ping: Date.now() }));
}
function onPong(message) {
  const now = Date.now();
  const rtt = now - message.pong;
  clock.samples.push({ rtt, offset: message.server_us / 1000 - (message.pong + now) / 2 });
  if (clock.samples.length > 20) clock.samples.shift();
  const best = clock.samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
  clock.offset = best.offset;
  stream.latency.rtt = best.rtt;
}
/** Milliseconds since a capture timestamp (us, PC clock); null before the clocks are matched. */
function sinceCapture(timestampUs) {
  return clock.offset === null ? null : Date.now() + clock.offset - timestampUs / 1000;
}
setInterval(ping, 2000);

function askKeyframe() {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ keyframe: true }));
}

const infoListeners = [];
/** Calls `listener(info)` whenever the stream is (re)configured. */
export function onStreamInfo(listener) {
  infoListeners.push(listener);
}

const summaryListeners = [];
/** Calls `listener()` once a second, after the summary is refreshed. */
export function onSummary(listener) {
  summaryListeners.push(listener);
}

let socket = null;
let decoder = null;
let waitingForKey = true;
let pendingFrame = null;   // newest decoded VideoFrame not yet taken
let pendingMeta = { divergence: 0 };
const frameMeta = new Map();   // timestamp -> what the frame was made with
const stats = { received: 0, decoded: 0, bytes: 0, since: performance.now(), receivedLatency: [], shownLatency: [] };
let lastTimestamp = -1;
let lastShown = -1;

/** The frame captured at `timestampUs` reached the screen (the texture) now. */
export function noteShown(timestampUs) {
  if (timestampUs === lastShown) return;  // a repeat of the same picture
  lastShown = timestampUs;
  const latency = sinceCapture(timestampUs);
  if (latency !== null) stats.shownLatency.push(latency);
}
// Seconds in a row that frames arrived but none came out of the decoder.
let stalledSeconds = 0;
// Codecs found undecodable since the last success (for the fallback).
const triedCodecs = new Set();

export function decoderState() {
  return decoder?.state ?? null;
}

/**
 * The newest decoded frame and the divergence it was rendered with, or
 * null; the caller closes the frame.
 */
export function takeFrame() {
  if (!pendingFrame) return null;
  const frame = { frame: pendingFrame, ...pendingMeta };
  pendingFrame = null;
  return frame;
}

let sendTimer = null;
/** Sends the server-side settings soon (changes are coalesced). */
export function scheduleSend() {
  if (sendTimer === null) sendTimer = setTimeout(sendSettings, 100);
}
function sendSettings() {
  sendTimer = null;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(Object.fromEntries(SERVER_KEYS.map((key) => [key, settings[key]]))));
  }
}

export function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket = ws;
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    sendSettings();
    clock.samples = [];
    for (let i = 0; i < 5; i++) setTimeout(ping, i * 100);
  };
  ws.onmessage = (event) => {
    if (typeof event.data !== "string") {
      const view = new DataView(event.data);
      if (view.getUint8(0) === 2) playPacket(new Int16Array(event.data.slice(10)));
      else onChunk(view, event.data);
    }
    else {
      const message = JSON.parse(event.data);
      if ("pong" in message) onPong(message);
      else configure(message);
    }
  };
  ws.onclose = () => { setStatus("连接断开，2 秒后重连"); setTimeout(connect, 2000); };
}

async function configure(info) {
  stream.info = info;
  for (const listener of infoListeners) listener(info);
  if (decoder && decoder.state !== "closed") decoder.close();
  decoder = null;
  if (!("VideoDecoder" in window)) {
    stream.problem = "WebCodecs 不可用：请用 https 地址打开（或 http://localhost）";
    setStatus(stream.problem);
    return;
  }
  const config = { codec: info.codec_string, optimizeForLatency: true, hardwareAcceleration: "prefer-hardware" };
  let support = { supported: false };
  try { support = await VideoDecoder.isConfigSupported(config); } catch (error) { console.warn("[ivr]", error); }
  if (!support.supported) {
    config.hardwareAcceleration = "no-preference";
    try { support = await VideoDecoder.isConfigSupported(config); } catch (error) { console.warn("[ivr]", error); }
  }
  if (info !== stream.info) return;  // a newer configuration arrived meanwhile
  if (!support.supported) {
    // Try the other codec once; if neither decodes, say so.
    const other = info.codec === "hevc" ? "av1" : "hevc";
    if (!triedCodecs.has(other)) {
      triedCodecs.add(info.codec);
      stream.problem = `此设备无法解码 ${CODEC_LABELS[info.codec]}，已切换到 ${CODEC_LABELS[other]}`;
      setSettings({ codec: other });
    } else {
      stream.problem = `此设备无法解码 ${info.codec_string}`;
    }
    setStatus(stream.problem);
    return;
  }
  triedCodecs.clear();
  stream.problem = "";
  decoder = new VideoDecoder({
    output: (frame) => {
      pendingFrame?.close();
      pendingFrame = frame;
      pendingMeta = frameMeta.get(frame.timestamp) ?? pendingMeta;
      frameMeta.delete(frame.timestamp);
      stats.decoded++;
    },
    error: (error) => { setStatus("解码错误: " + error.message); configure(stream.info); askKeyframe(); },
  });
  decoder.configure(config);
  waitingForKey = true;
}

function onChunk(view, buffer) {
  if (!decoder || decoder.state !== "configured" || view.getUint8(0) !== 1) return;
  const key = (view.getUint8(1) & 1) === 1;
  const timestamp = Number(view.getBigUint64(2, true));
  if (frameMeta.size > 120) frameMeta.clear();
  frameMeta.set(timestamp, { divergence: view.getFloat32(10, true) });
  stats.received++;
  stats.bytes += buffer.byteLength;
  if (timestamp !== lastTimestamp) {
    lastTimestamp = timestamp;
    const latency = sinceCapture(timestamp);
    if (latency !== null) stats.receivedLatency.push(latency);
  }
  // Too far behind: drop deltas and resume at a keyframe (asked for now).
  if (decoder.decodeQueueSize > 4 && !key) {
    if (!waitingForKey) askKeyframe();
    waitingForKey = true;
    return;
  }
  if (waitingForKey && !key) return;
  waitingForKey = false;
  decoder.decode(new EncodedVideoChunk({ type: key ? "key" : "delta", timestamp, data: new Uint8Array(buffer, 18) }));
}

setInterval(() => {
  const seconds = (performance.now() - stats.since) / 1000;
  const info = stream.info;
  stream.received = stats.received / seconds;
  stream.decoded = stats.decoded / seconds;
  // Frames arrive but the decoder gives nothing back: report it, and after
  // a few seconds start over with a fresh decoder.
  stalledSeconds = stats.received > 0 && stats.decoded === 0 && decoder ? stalledSeconds + 1 : 0;
  if (stalledSeconds >= 3 && info) {
    stalledSeconds = 0;
    console.warn("[ivr] decoder produced no frames; reconfiguring");
    configure(info);
    askKeyframe();
  }
  const median = (values) => {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[sorted.length >> 1];
  };
  // Keep the last values while nothing new arrives (a still picture).
  stream.latency.received = median(stats.receivedLatency) ?? stream.latency.received;
  stream.latency.shown = median(stats.shownLatency) ?? stream.latency.shown;
  if (info && !stream.problem) {
    const ms = (v) => (v === null ? "?" : v.toFixed(0));
    const { received, shown, rtt } = stream.latency;
    stream.summary = `${info.layout.left.width}×${info.layout.left.height} ${CODEC_LABELS[info.codec]} · ` +
      `${(stats.decoded / seconds).toFixed(0)} fps · ${(stats.bytes * 8 / seconds / 1e6).toFixed(1)} Mbps · ` +
      `延迟 收 ${ms(received)} / 显 ${ms(shown)} ms（往返 ${ms(rtt)}）`;
    setStatus(stream.summary);
  }
  Object.assign(stats, { received: 0, decoded: 0, bytes: 0, since: performance.now(), receivedLatency: [], shownLatency: [] });
  for (const listener of summaryListeners) listener();
}, 1000);
