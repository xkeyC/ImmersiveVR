//! The browser client (`web/`), embedded in the binary at build time.

use axum::{
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};

/// `(path, content type, bytes)` of every client file.
const FILES: &[(&str, &str, &[u8])] = &[
    (
        "index.html",
        "text/html; charset=utf-8",
        include_bytes!("../web/index.html"),
    ),
    (
        "manifest.webmanifest",
        "application/manifest+json",
        include_bytes!("../web/manifest.webmanifest"),
    ),
    ("sw.js", JS, include_bytes!("../web/sw.js")),
    ("icons/icon-192.png", PNG, include_bytes!("../web/icons/icon-192.png")),
    ("icons/icon-512.png", PNG, include_bytes!("../web/icons/icon-512.png")),
    (
        "icons/maskable-512.png",
        PNG,
        include_bytes!("../web/icons/maskable-512.png"),
    ),
    (
        "style.css",
        "text/css; charset=utf-8",
        include_bytes!("../web/style.css"),
    ),
    ("js/main.js", JS, include_bytes!("../web/js/main.js")),
    ("js/audio.js", JS, include_bytes!("../web/js/audio.js")),
    (
        "js/audio-worklet.js",
        JS,
        include_bytes!("../web/js/audio-worklet.js"),
    ),
    ("js/gl.js", JS, include_bytes!("../web/js/gl.js")),
    ("js/layers.js", JS, include_bytes!("../web/js/layers.js")),
    ("js/math.js", JS, include_bytes!("../web/js/math.js")),
    ("js/panel.js", JS, include_bytes!("../web/js/panel.js")),
    ("js/preview.js", JS, include_bytes!("../web/js/preview.js")),
    ("js/screen.js", JS, include_bytes!("../web/js/screen.js")),
    (
        "js/settings.js",
        JS,
        include_bytes!("../web/js/settings.js"),
    ),
    ("js/stream.js", JS, include_bytes!("../web/js/stream.js")),
    ("js/ui.js", JS, include_bytes!("../web/js/ui.js")),
    ("js/xr.js", JS, include_bytes!("../web/js/xr.js")),
];
const JS: &str = "text/javascript; charset=utf-8";
const PNG: &str = "image/png";

/// Serves a client file by URI path (`/` is `index.html`).
pub async fn serve(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match FILES.iter().find(|(name, ..)| *name == path) {
        Some((_, content_type, bytes)) => (
            [
                (header::CONTENT_TYPE, *content_type),
                // The client changes with the binary: never serve a stale copy.
                (header::CACHE_CONTROL, "no-cache"),
            ],
            *bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_web_file_is_embedded() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web");
        let mut on_disk = Vec::new();
        for entry in walk(&root) {
            on_disk.push(
                entry
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
        on_disk.sort();
        let mut embedded: Vec<String> = FILES.iter().map(|(name, ..)| name.to_string()).collect();
        embedded.sort();
        assert_eq!(on_disk, embedded, "web/ and FILES must list the same files");
    }

    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                files.extend(walk(&path));
            } else {
                files.push(path);
            }
        }
        files
    }
}
