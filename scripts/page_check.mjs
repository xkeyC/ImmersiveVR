// Drives the client page in headless Chrome over the DevTools protocol:
// runs a scenario of steps, prints the page console and state, takes
// screenshots.
//   node scripts/page_check.mjs <url> '<steps JSON>'
// Steps: {"wait": seconds} | {"state": label} | {"shot": "file.png"}
//        | {"click": "<panel action>"}  (a real mouse click on that button in the preview)
//        | {"reload": true} | {"eval": "<async function body returning a value>"}
// CHROME_LANG=en-US (or zh-CN) sets the browser's language.
// Default steps: wait 5 s, print state, screenshot to target/page.png.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url = "https://127.0.0.1:13256/?preview", stepsJson] = process.argv.slice(2);
const steps = stepsJson ? JSON.parse(stepsJson) : [{ wait: 5 }, { state: "after 5 s" }, { shot: "target/page.png" }];
const chrome = process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const port = 9333;
const browser = spawn(chrome, [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "ivr-cdp-"))}`,
  // The server's certificate is self-signed; a headset user clicks through the warning instead.
  "--ignore-certificate-errors",
  "--no-first-run", "--window-size=1600,900", "--autoplay-policy=no-user-gesture-required",
  ...(process.env.CHROME_LANG ? [`--lang=${process.env.CHROME_LANG}`, `--accept-lang=${process.env.CHROME_LANG}`] : []),
  "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page"); } catch {}
}
if (!target) { browser.kill(); throw new Error("Chrome DevTools did not come up"); }

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = ({ data }) => {
  const msg = JSON.parse(data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  else if (msg.method === "Runtime.consoleAPICalled") {
    const text = msg.params.args.map((a) => a.value ?? a.description).join(" ");
    if (!text.includes("fps")) console.log(`[console.${msg.params.type}]`, text);
  } else if (msg.method === "Runtime.exceptionThrown") {
    console.log("[exception]", msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text);
  }
};
const send = (method, params = {}) => new Promise((r) => { const n = ++id; pending.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const evaluate = async (expression) => {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  return result.result?.result?.value ?? result.result?.exceptionDetails?.exception?.description;
};

await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url });
await sleep(1500);
for (const step of steps) {
  if (step.wait) await sleep(step.wait * 1000);
  if (step.reload) { await send("Page.reload"); await sleep(2500); }
  if (step.state) console.log(`state (${step.state}):`, JSON.stringify(await evaluate("ivrState()")));
  if (step.eval) console.log("eval:", JSON.stringify(await evaluate(`(async () => { ${step.eval} })()`)));
  if (step.click) {
    const at = await evaluate(`ivrLocate(${JSON.stringify(step.click)})`);
    if (!at) { console.log(`click ${step.click}: button not found`); continue; }
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) {
      await send("Input.dispatchMouseEvent", { type, x: at.x, y: at.y, button: "left", clickCount: 1 });
    }
    console.log(`clicked ${step.click} at (${at.x.toFixed(0)}, ${at.y.toFixed(0)})`);
  }
  if (step.shot) {
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(step.shot, Buffer.from(shot.result.data, "base64"));
    console.log("wrote", step.shot);
  }
}
ws.close();
browser.kill();
process.exit(0);
