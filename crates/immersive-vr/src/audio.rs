//! The PC's sound to the headset, and the PC's own speakers muted meanwhile.
//!
//! Capture is WASAPI process loopback excluding this process (every other
//! application's sound before it reaches the endpoint, so muting the
//! endpoint does not silence it; checked on this machine), falling back to
//! plain endpoint loopback on Windows without it. Sound goes out as 10 ms
//! packets of 48 kHz stereo 16-bit PCM (1.5 Mbit/s: nothing next to the
//! video, and no codec delay); the client plays them through a short jitter
//! buffer.
//!
//! While a client is connected and the `mute_pc` setting is on, the default
//! output device is muted; its own mute state comes back when the last
//! client leaves, the setting is turned off, or the server stops.

use crate::pipeline::Controls;
use anyhow::{anyhow, Result};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast;
use wasapi::{AudioClient, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat};
use windows::Win32::{
    Media::Audio::{eConsole, eRender, Endpoints::IAudioEndpointVolume, IMMDeviceEnumerator, MMDeviceEnumerator},
    System::Com::{CoCreateInstance, CLSCTX_ALL},
};

pub const SAMPLE_RATE: u32 = 48_000;
pub const CHANNELS: usize = 2;
/// 10 ms of sound a packet.
const PACKET_FRAMES: usize = SAMPLE_RATE as usize / 100;

/// One packet: interleaved stereo s16le, captured at `timestamp_us` (Unix).
pub struct AudioPacket {
    pub timestamp_us: u64,
    pub pcm: Vec<u8>,
}

/// Starts the capture thread; packets go to `packets` (dropped while nobody listens).
pub fn start(packets: broadcast::Sender<Arc<AudioPacket>>) -> Result<()> {
    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            if let Err(error) = capture(&packets) {
                tracing::warn!(%error, "audio capture stopped: no sound to the headset");
            }
        })?;
    Ok(())
}

fn capture(packets: &broadcast::Sender<Arc<AudioPacket>>) -> Result<()> {
    let _ = wasapi::initialize_mta();
    let format = WaveFormat::new(32, 32, &SampleType::Float, SAMPLE_RATE as usize, CHANNELS, None);
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: 0,
    };
    // Everything but this process (which plays nothing anyway).
    let client = match AudioClient::new_application_loopback_client(std::process::id(), false)
        .and_then(|mut client| client.initialize_client(&format, &Direction::Capture, &mode).map(|()| client))
    {
        Ok(client) => {
            tracing::info!("audio: process loopback (unaffected by the PC's mute)");
            client
        }
        Err(error) => {
            tracing::warn!(%error, "audio: no process loopback, capturing the output device instead");
            let device = DeviceEnumerator::new()
                .and_then(|devices| devices.get_default_device(&Direction::Render))
                .map_err(|e| anyhow!("default output device: {e}"))?;
            let mut client = device.get_iaudioclient().map_err(|e| anyhow!("audio client: {e}"))?;
            client
                .initialize_client(&format, &Direction::Capture, &mode)
                .map_err(|e| anyhow!("loopback: {e}"))?;
            client
        }
    };
    let event = client.set_get_eventhandle().map_err(|e| anyhow!("audio event: {e}"))?;
    let capture = client.get_audiocaptureclient().map_err(|e| anyhow!("audio capture: {e}"))?;
    client.start_stream().map_err(|e| anyhow!("audio start: {e}"))?;
    let frame_bytes = format.get_blockalign() as usize;
    let mut queue: VecDeque<u8> = VecDeque::new();
    loop {
        while capture
            .get_next_packet_size()
            .map_err(|e| anyhow!("audio: {e}"))?
            .unwrap_or(0)
            > 0
        {
            capture
                .read_from_device_to_deque(&mut queue)
                .map_err(|e| anyhow!("audio: {e}"))?;
        }
        while queue.len() >= PACKET_FRAMES * frame_bytes {
            // The newest sample is now; this packet ends where the queue's rest begins.
            let behind = (queue.len() / frame_bytes - PACKET_FRAMES) as u64 * 1_000_000 / SAMPLE_RATE as u64;
            let mut pcm = Vec::with_capacity(PACKET_FRAMES * CHANNELS * 2);
            for _ in 0..PACKET_FRAMES * CHANNELS {
                let bytes = [
                    queue.pop_front().unwrap_or(0),
                    queue.pop_front().unwrap_or(0),
                    queue.pop_front().unwrap_or(0),
                    queue.pop_front().unwrap_or(0),
                ];
                let sample = (f32::from_le_bytes(bytes).clamp(-1.0, 1.0) * 32767.0) as i16;
                pcm.extend_from_slice(&sample.to_le_bytes());
            }
            let _ = packets.send(Arc::new(AudioPacket {
                timestamp_us: unix_us().saturating_sub(behind),
                pcm,
            }));
        }
        // Nothing plays: no packets come, and the client hears silence.
        let _ = event.wait_for_event(100);
    }
}

/// Mutes the PC's default output while a client is connected and the
/// `mute_pc` setting is on; restores the device's own state otherwise and on
/// drop.
pub struct PcMute {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl PcMute {
    pub fn start(controls: Arc<Controls>, clients: Arc<AtomicUsize>) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = std::thread::Builder::new()
            .name("pc-mute".into())
            .spawn(move || {
                let _ = wasapi::initialize_mta();
                // The device's state before we muted it, while we have.
                let mut restore: Option<bool> = None;
                loop {
                    let quit = stopping.load(Ordering::Acquire);
                    let want = !quit && controls.get().0.mute_pc && clients.load(Ordering::Acquire) > 0;
                    if want != restore.is_some() {
                        match endpoint_volume() {
                            Ok(volume) => {
                                // SAFETY: plain COM calls on a live interface.
                                let result = unsafe {
                                    if want {
                                        volume.GetMute().map(|muted| restore = Some(muted.as_bool()))
                                            .and_then(|()| volume.SetMute(true, std::ptr::null()))
                                    } else {
                                        let muted = restore.take().unwrap_or(false);
                                        volume.SetMute(muted, std::ptr::null())
                                    }
                                };
                                match result {
                                    Ok(()) => tracing::info!(muted = want, "PC speakers"),
                                    Err(error) => tracing::warn!(%error, "muting the PC"),
                                }
                            }
                            Err(error) => tracing::warn!(%error, "PC output device"),
                        }
                    }
                    if quit {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
            })?;
        Ok(Self {
            stop,
            worker: Some(worker),
        })
    }
}

impl Drop for PcMute {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn endpoint_volume() -> windows::core::Result<IAudioEndpointVolume> {
    // SAFETY: COM is initialized on this thread; plain calls.
    unsafe {
        let devices: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        devices
            .GetDefaultAudioEndpoint(eRender, eConsole)?
            .Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None)
    }
}

fn unix_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default()
}
