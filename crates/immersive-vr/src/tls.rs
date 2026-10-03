//! The self-signed certificate the page is served with.
//!
//! The headset browser exposes WebXR and WebCodecs only in a secure context
//! (`https://`, or `http://localhost`), so LAN access needs HTTPS. The
//! certificate is created at startup and kept in a directory, so the
//! browser's one-time "proceed anyway" stays valid across restarts; it is
//! replaced when the machine has an address the certificate does not name.

use anyhow::{Context as _, Result};
use std::{
    collections::BTreeSet,
    net::IpAddr,
    path::{Path, PathBuf},
};

pub struct Certificate {
    pub cert_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
    pub path: PathBuf,
}

/// IPv4 addresses of interfaces that are up, without loopback and
/// link-local ones: what a headset on the LAN would connect to.
pub fn lan_addresses() -> Vec<IpAddr> {
    let mut addresses: Vec<IpAddr> = if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter(|interface| {
            interface.is_oper_up() && !interface.is_loopback() && !interface.is_link_local()
        })
        .map(|interface| interface.ip())
        .filter(IpAddr::is_ipv4)
        .collect();
    addresses.sort();
    addresses.dedup();
    addresses
}

/// The certificate in `dir` if it names `localhost`, 127.0.0.1 and every one
/// of `addresses`; otherwise a new one for them, saved there.
pub fn load_or_create(dir: &Path, addresses: &[IpAddr]) -> Result<Certificate> {
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    let names_path = dir.join("names.txt");
    let wanted: BTreeSet<String> = ["localhost".to_string(), "127.0.0.1".to_string()]
        .into_iter()
        .chain(addresses.iter().map(IpAddr::to_string))
        .collect();
    if let (Ok(cert_pem), Ok(key_pem), Ok(names)) = (
        std::fs::read(&cert_path),
        std::fs::read(&key_path),
        std::fs::read_to_string(&names_path),
    ) {
        let covered: BTreeSet<String> = names.lines().map(str::to_string).collect();
        if wanted.is_subset(&covered) {
            return Ok(Certificate {
                cert_pem,
                key_pem,
                path: cert_path,
            });
        }
        tracing::info!("network addresses changed; issuing a new certificate");
    }
    let names: Vec<String> = wanted.into_iter().collect();
    let issued =
        rcgen::generate_simple_self_signed(names.clone()).context("generating certificate")?;
    let cert_pem = issued.cert.pem();
    let key_pem = issued.signing_key.serialize_pem();
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    std::fs::write(&cert_path, &cert_pem)?;
    std::fs::write(&key_path, &key_pem)?;
    std::fs::write(&names_path, names.join("\n"))?;
    tracing::info!(
        path = %cert_path.display(),
        names = %names.join(", "),
        "created a self-signed certificate"
    );
    Ok(Certificate {
        cert_pem: cert_pem.into_bytes(),
        key_pem: key_pem.into_bytes(),
        path: cert_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certificate_is_reused_until_an_address_is_missing() {
        let dir = std::env::temp_dir().join(format!("ivr-tls-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a: IpAddr = "192.168.1.20".parse().unwrap();
        let b: IpAddr = "10.0.0.5".parse().unwrap();
        let first = load_or_create(&dir, &[a]).unwrap();
        assert!(String::from_utf8_lossy(&first.cert_pem).contains("BEGIN CERTIFICATE"));
        assert_eq!(load_or_create(&dir, &[a]).unwrap().cert_pem, first.cert_pem);
        assert_ne!(
            load_or_create(&dir, &[a, b]).unwrap().cert_pem,
            first.cert_pem
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
