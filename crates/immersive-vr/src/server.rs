//! HTTP + WebSocket server: the WebXR page at `/`, the stream at `/ws`.
//!
//! A client receives the stream info (JSON text, [`StreamInfo`]) first and
//! again whenever the stream is reconfigured, then binary chunks:
//!
//! ```text
//! byte 0       kind (1 = video chunk)
//! byte 1       flags (bit 0: keyframe)
//! bytes 2..10  when the frame's picture was captured, microseconds since the Unix epoch (u64 LE)
//! bytes 10..14 divergence the frame's stereo offsets were made with (f32 LE)
//! bytes 14..18 convergence (f32 LE)
//! bytes 18..   one encoded frame (H.265 Annex B, or AV1 OBUs)
//! ```
//!
//! and sound:
//!
//! ```text
//! byte 0       kind (2 = audio)
//! byte 1       0
//! bytes 2..10  when it was captured, Unix microseconds (u64 LE)
//! bytes 10..   10 ms of 48 kHz stereo s16le PCM, interleaved
//! ```
//!
//! Each client starts at a keyframe (one is asked for when it joins). One
//! that falls more than [`MAX_BACKLOG`] frames behind (a link slower than
//! the stream) drops frames up to the next keyframe, which is asked for at
//! once: its latency stays bounded instead of growing a queue.
//!
//! Clients send text messages:
//! * settings ([`SettingsUpdate`]), e.g. `{"divergence": 2.0, "resolution": 1440, "codec": "hevc"}`;
//! * `{"ping": <client time, ms>}`: answered with `{"pong": <the same>, "server_us": <Unix time, us>}`
//!   (the client's clock offset, for latency);
//! * `{"keyframe": true}`: the decoder lost its place.

use crate::{
    audio::AudioPacket,
    encoder::Chunk,
    pipeline::{Controls, SettingsUpdate, StreamInfo},
};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::Deserialize;
use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

/// Frames a client may have waiting before it drops to the next keyframe.
const MAX_BACKLOG: usize = 3;

#[derive(Deserialize)]
struct Ping {
    ping: f64,
}

#[derive(Deserialize)]
struct KeyframeRequest {
    keyframe: bool,
}
use tokio::sync::{
    broadcast::{self, error::RecvError},
    watch,
};

#[derive(Clone)]
struct AppState {
    stream: watch::Receiver<Option<Arc<StreamInfo>>>,
    chunks: broadcast::Sender<Arc<Chunk>>,
    audio: broadcast::Sender<Arc<AudioPacket>>,
    controls: Arc<Controls>,
    /// Connected clients (the PC is muted only while there is one).
    clients: Arc<AtomicUsize>,
}

/// HTTPS with a self-signed certificate (the default: a secure context on
/// the LAN), or plain HTTP (secure only as `http://localhost`).
pub enum Transport {
    Https(crate::tls::Certificate),
    Http,
}

pub async fn serve(
    address: SocketAddr,
    transport: Transport,
    stream: watch::Receiver<Option<Arc<StreamInfo>>>,
    chunks: broadcast::Sender<Arc<Chunk>>,
    audio: broadcast::Sender<Arc<AudioPacket>>,
    controls: Arc<Controls>,
    clients: Arc<AtomicUsize>,
) -> anyhow::Result<()> {
    let state = AppState {
        stream,
        chunks,
        audio,
        controls,
        clients,
    };
    let app = Router::new()
        .route("/ws", get(websocket))
        .fallback(get(crate::web::serve))
        .with_state(state);
    let port = address.port();
    let scheme = match transport {
        Transport::Https(_) => "https",
        Transport::Http => "http",
    };
    let hosts: Vec<String> = if address.ip().is_unspecified() {
        crate::tls::lan_addresses()
            .iter()
            .map(ToString::to_string)
            .collect()
    } else {
        vec![address.ip().to_string()]
    };
    for host in &hosts {
        tracing::info!("serving on {scheme}://{host}:{port}/");
    }
    match transport {
        Transport::Https(certificate) => {
            tracing::info!(
                cert = %certificate.path.display(),
                "self-signed certificate: the browser warns once, choose Advanced -> Proceed"
            );
            let tls = axum_server::tls_rustls::RustlsConfig::from_pem(
                certificate.cert_pem,
                certificate.key_pem,
            )
            .await?;
            axum_server::bind_rustls(address, tls)
                .serve(app.into_make_service())
                .await?;
        }
        Transport::Http => {
            tracing::info!(
                "plain HTTP: WebXR and WebCodecs work only as http://localhost:{port}/ \
                 (on the headset: `adb reverse tcp:{port} tcp:{port}`)"
            );
            use axum::serve::ListenerExt as _;
            // No Nagle delay on small messages (audio, pongs); HTTPS's acceptor does the same.
            let listener = tokio::net::TcpListener::bind(address)
                .await?
                .tap_io(|tcp| {
                    let _ = tcp.set_nodelay(true);
                });
            axum::serve(listener, app).await?;
        }
    }
    Ok(())
}

async fn websocket(upgrade: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    upgrade.on_upgrade(move |socket| async move {
        if let Err(error) = stream(socket, state).await {
            tracing::debug!(%error, "client stream ended");
        }
    })
}

async fn stream(mut socket: WebSocket, mut state: AppState) -> anyhow::Result<()> {
    tracing::info!("client connected");
    state.clients.fetch_add(1, Ordering::AcqRel);
    let result = serve_client(&mut socket, &mut state).await;
    state.clients.fetch_sub(1, Ordering::AcqRel);
    result
}

async fn serve_client(socket: &mut WebSocket, state: &mut AppState) -> anyhow::Result<()> {
    // Subscribe before reading the info, so no chunk of it is missed.
    let mut chunks = state.chunks.subscribe();
    let mut audio = state.audio.subscribe();
    let mut generation = 0;
    let mut synced = false;
    let mut sent = 0u64;
    let mut dropped = 0u64;
    let mut asked: Option<Instant> = None;
    // A keyframe soon, rather than at the next periodic one.
    let mut ask_keyframe = |controls: &Controls| {
        if asked.is_none_or(|at| at.elapsed() > Duration::from_millis(200)) {
            controls.request_keyframe();
            asked = Some(Instant::now());
        }
    };
    ask_keyframe(&state.controls);
    // The current stream info, once the pipeline has published one.
    state.stream.mark_changed();
    loop {
        tokio::select! {
            changed = state.stream.changed() => {
                if changed.is_err() {
                    break;
                }
                let info = state.stream.borrow_and_update().clone();
                if let Some(info) = info {
                    generation = info.generation;
                    synced = false;
                    let text = serde_json::to_string(info.as_ref())?;
                    if socket.send(Message::Text(text.into())).await.is_err() {
                        break;
                    }
                }
            }
            received = chunks.recv() => {
                let chunk = match received {
                    Ok(chunk) => chunk,
                    Err(RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "client fell behind; waiting for a keyframe");
                        synced = false;
                        ask_keyframe(&state.controls);
                        continue;
                    }
                    Err(RecvError::Closed) => break,
                };
                // Only the current stream, from a keyframe on.
                if chunk.generation != generation || !(synced || chunk.key) {
                    continue;
                }
                // Too many frames waiting: the link is slower than the stream.
                // Skip to a fresh keyframe instead of letting the queue grow.
                if !chunk.key && chunks.len() > MAX_BACKLOG {
                    synced = false;
                    dropped += 1;
                    ask_keyframe(&state.controls);
                    continue;
                }
                synced = true;
                let mut message = Vec::with_capacity(18 + chunk.data.len());
                message.push(1);
                message.push(chunk.key as u8);
                message.extend_from_slice(&chunk.timestamp_us.to_le_bytes());
                message.extend_from_slice(&chunk.meta.divergence.to_le_bytes());
                message.extend_from_slice(&chunk.meta.convergence.to_le_bytes());
                message.extend_from_slice(&chunk.data);
                if socket.send(Message::Binary(message.into())).await.is_err() {
                    break;
                }
                sent += 1;
            }
            packet = audio.recv() => match packet {
                Ok(packet) => {
                    let mut message = Vec::with_capacity(10 + packet.pcm.len());
                    message.push(2);
                    message.push(0);
                    message.extend_from_slice(&packet.timestamp_us.to_le_bytes());
                    message.extend_from_slice(&packet.pcm);
                    if socket.send(Message::Binary(message.into())).await.is_err() {
                        break;
                    }
                }
                // Late sound is no use: carry on with the newest.
                Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => break,
            },
            message = socket.recv() => match message {
                Some(Ok(Message::Text(text))) => {
                    if let Ok(Ping { ping }) = serde_json::from_str(&text) {
                        let pong = serde_json::json!({ "pong": ping, "server_us": unix_us() });
                        if socket.send(Message::Text(pong.to_string().into())).await.is_err() {
                            break;
                        }
                    } else if let Ok(KeyframeRequest { keyframe: true }) = serde_json::from_str(&text) {
                        ask_keyframe(&state.controls);
                    } else {
                        match serde_json::from_str::<SettingsUpdate>(&text) {
                            Ok(update) => {
                                tracing::info!(?update, "client settings");
                                state.controls.apply(update);
                            }
                            Err(error) => tracing::debug!(%error, "ignoring client message"),
                        }
                    }
                }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                Some(Ok(_)) => {}
            },
        }
    }
    tracing::info!(sent, dropped, "client disconnected");
    Ok(())
}

fn unix_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default()
}
