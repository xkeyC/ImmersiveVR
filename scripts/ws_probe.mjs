// Connects to an ImmersiveVR server like the browser does and reports what
// arrives; writes each stream generation's frames for inspection.
//   node scripts/ws_probe.mjs [wss://localhost:13256/ws] [seconds=5] [out=target/probe] ['{"codec":"av1"}']
// The optional JSON is sent halfway through (a settings change). Files are
// <out>_g<generation>.<h265|obu>; set NODE_TLS_REJECT_UNAUTHORIZED=0 for
// the server's self-signed certificate.
import { writeFileSync } from "node:fs";

const [url = "wss://localhost:13256/ws", seconds = "5", out = "target/probe", update] = process.argv.slice(2);
const EXTENSION = { hevc: "h265", av1: "obu" };
const ws = new WebSocket(url);
ws.binaryType = "arraybuffer";
let info = null;
const streams = new Map();
const audio = { packets: 0, bytes: 0 };  // generation -> { info, chunks, keys, first, last, latencies, divergences }

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    info = JSON.parse(event.data);
    console.log(`info g${info.generation}: ${info.codec} ${info.codec_string} ` +
      `eye ${info.layout.left.width}x${info.layout.left.height} (frame ${info.layout.width}x${info.layout.height}) ` +
      `${info.bitrate_mbps} Mbps, resolutions ${info.resolutions}`);
    streams.set(info.generation, { info, chunks: [], keys: 0, latencies: [], divergences: new Map() });
    return;
  }
  const view = new DataView(event.data);
  if (view.getUint8(0) === 2) {
    audio.packets++;
    audio.bytes += event.data.byteLength - 10;
    return;
  }
  const stream = streams.get(info.generation);
  const key = (view.getUint8(1) & 1) === 1;
  if (stream.chunks.length === 0 && !key) throw new Error("first chunk of a stream is not a keyframe");
  stream.keys += key;
  const now = performance.now();
  if (stream.last !== undefined) (stream.gaps ??= []).push(now - stream.last);
  stream.first ??= now;
  stream.last = now;
  stream.latencies.push(Date.now() - Number(view.getBigUint64(2, true)) / 1000);
  const d = view.getFloat32(10, true).toFixed(2);
  stream.divergences.set(d, (stream.divergences.get(d) ?? 0) + 1);
  stream.chunks.push(new Uint8Array(event.data, 18));
};
ws.onerror = (e) => { console.error("websocket error", e.message ?? e); process.exit(1); };
if (update) setTimeout(() => ws.send(update), parseFloat(seconds) * 500);

setTimeout(() => {
  ws.close();
  for (const [generation, s] of streams) {
    const elapsed = Math.max((s.last - s.first) / 1000, 1e-3);
    const bytes = s.chunks.reduce((n, c) => n + c.length, 0);
    const lat = s.latencies.sort((a, b) => a - b);
    const pct = (q) => lat[Math.min(lat.length - 1, Math.floor(q * lat.length))]?.toFixed(1);
    const file = `${out}_g${generation}.${EXTENSION[s.info.codec]}`;
    writeFileSync(file, Buffer.concat(s.chunks.map((c) => Buffer.from(c))));
    console.log(`g${generation}: ${s.chunks.length} chunks = ${(s.chunks.length / elapsed).toFixed(1)} fps, ` +
      `${(bytes * 8 / elapsed / 1e6).toFixed(2)} Mbps, keyframes ${s.keys}, latency p50 ${pct(0.5)} ms p90 ${pct(0.9)} ms, ` +
      `divergence ${JSON.stringify(Object.fromEntries(s.divergences))} -> ${file}`);
    const gaps = (s.gaps ?? []).sort((a, b) => a - b);
    const gap = (q) => gaps[Math.min(gaps.length - 1, Math.floor(q * gaps.length))]?.toFixed(1);
    if (gaps.length) console.log(`g${generation} arrival gaps: p10 ${gap(0.1)} p50 ${gap(0.5)} p90 ${gap(0.9)} max ${gaps[gaps.length - 1].toFixed(1)} ms`);
  }
  console.log(`audio: ${audio.packets} packets (${(audio.packets / parseFloat(seconds)).toFixed(0)}/s), ${(audio.bytes * 8 / parseFloat(seconds) / 1e6).toFixed(2)} Mbps`);
  process.exit(streams.size ? 0 : 1);
}, (parseFloat(seconds) + 1) * 1000);
