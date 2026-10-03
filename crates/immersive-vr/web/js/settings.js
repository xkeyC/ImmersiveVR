// User preferences, kept in IndexedDB so they survive reloads and restarts.

export const DEFAULTS = {
  divergence: 0.5,     // total left-right shift, % of width (iw3)
  convergence: 0.5,    // 0 = everything in front of the screen .. 1 = everything behind it
  resolution: 1440,
  codec: "hevc",
  distance: 2.5,       // m
  size: 2.4,           // screen width, m
  curvature: 0.3,      // 0 = flat .. 1 = cylinder around the viewer
  height: 0.0,         // m above eye level
  passthrough: false,
  background: 1.0,     // passthrough: how much of the room shows (1 = all, 0 = black)
  mute_pc: true,       // mute the PC's speakers while streaming (the sound comes here)
};

/**
 * Bumped when a default changes in a way saved settings should pick up:
 * 2 = stronger 3D (divergence 2 -> 3, convergence 0.65 -> 0.5);
 * 3 = divergence 3 -> 0.5 (tried on Quest 3: stronger looks wrong).
 */
const VERSION = 3;
const RESET_ON_UPGRADE = ["divergence", "convergence"];

/** Settings the server uses (sent to it on every change). */
export const SERVER_KEYS = ["divergence", "convergence", "resolution", "codec", "mute_pc"];

export const settings = { ...DEFAULTS };

const listeners = [];

/** Calls `listener(changes)` after every change. */
export function onSettingsChange(listener) {
  listeners.push(listener);
}

/** Changes settings, saves them and notifies the listeners. */
export function setSettings(changes) {
  Object.assign(settings, changes);
  save();
  for (const listener of listeners) listener(changes);
}

let db = null;
let saveTimer = null;

/** Loads the saved settings over the defaults. */
export async function loadSettings() {
  try {
    db = await new Promise((resolve, reject) => {
      const request = indexedDB.open("immersive-vr", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("settings");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const saved = await new Promise((resolve) => {
      const get = db.transaction("settings").objectStore("settings").get("prefs");
      get.onsuccess = () => resolve(get.result || {});
      get.onerror = () => resolve({});
    });
    const upgraded = saved.version !== VERSION;
    for (const key of Object.keys(DEFAULTS)) {
      if (key in saved && !(upgraded && RESET_ON_UPGRADE.includes(key))) settings[key] = saved[key];
    }
    // H.264 was removed: an old choice becomes H.265.
    if (!["hevc", "av1"].includes(settings.codec)) settings.codec = DEFAULTS.codec;
    if (upgraded) save();
  } catch (error) {
    console.warn("[ivr] settings storage unavailable", error);
  }
}

function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { db?.transaction("settings", "readwrite").objectStore("settings").put({ ...settings, version: VERSION }, "prefs"); }
    catch (error) { console.warn("[ivr] saving settings failed", error); }
  }, 300);
}
