//! Video codecs: levels, WebCodecs codec strings, and telling keyframes
//! apart in each codec's bitstream (H.265 Annex B, AV1 low-overhead OBUs).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    Hevc,
    Av1,
}

/// A level chosen for a frame size and rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    /// For logs, e.g. "5.1".
    pub name: &'static str,
    /// NVENC's level value: `general_level_idc` (H.265), `seq_level_idx` (AV1).
    pub nvenc: u32,
    /// The WebCodecs codec string, e.g. "hvc1.1.6.L153.B0".
    pub codec_string: String,
}

impl Codec {
    pub fn label(self) -> &'static str {
        match self {
            Codec::Hevc => "H.265",
            Codec::Av1 => "AV1",
        }
    }

    /// The lowest level that allows `width x height` at `fps` and
    /// `bitrate_mbps` (encoders reject a level whose bitrate cap is below
    /// the requested rate).
    pub fn level(self, width: usize, height: usize, fps: u32, bitrate_mbps: u32) -> Level {
        let (w, h, rate) = (width as u64, height as u64, fps as u64);
        let samples = w * h;
        let kbps = bitrate_mbps as u64 * 1000;
        match self {
            Codec::Hevc => {
                // (name, general_level_idc, MaxLumaPs, MaxLumaSr, MaxBR kbit/s Main tier)
                const LEVELS: &[(&str, u32, u64, u64, u64)] = &[
                    ("4.1", 123, 2_228_224, 133_693_440, 20_000),
                    ("5.0", 150, 8_912_896, 267_386_880, 25_000),
                    ("5.1", 153, 8_912_896, 534_773_760, 40_000),
                    ("5.2", 156, 8_912_896, 1_069_547_520, 60_000),
                    ("6.0", 180, 35_651_584, 1_069_547_520, 60_000),
                    ("6.1", 183, 35_651_584, 2_139_095_040, 120_000),
                    ("6.2", 186, 35_651_584, 4_278_190_080, 240_000),
                ];
                let fits = |&&(_, _, max_ps, max_sr, max_br): &&(&str, u32, u64, u64, u64)| {
                    let side = ((8 * max_ps) as f64).sqrt() as u64;
                    samples <= max_ps
                        && w <= side
                        && h <= side
                        && samples * rate <= max_sr
                        && kbps <= max_br
                };
                let &(name, idc, ..) = LEVELS.iter().find(fits).unwrap_or(LEVELS.last().unwrap());
                Level {
                    name,
                    nvenc: idc,
                    codec_string: format!("hvc1.1.6.L{idc}.B0"),
                }
            }
            Codec::Av1 => {
                // (name, seq_level_idx, MaxPicSize, MaxHSize, MaxVSize, MaxDisplayRate,
                //  MaxBitrate kbit/s Main tier)
                const LEVELS: &[(&str, u32, u64, u64, u64, u64, u64)] = &[
                    ("4.0", 8, 2_228_224, 6144, 3456, 66_846_720, 12_000),
                    ("4.1", 9, 2_228_224, 6144, 3456, 133_693_440, 20_000),
                    ("5.0", 12, 8_912_896, 8192, 4352, 267_386_880, 30_000),
                    ("5.1", 13, 8_912_896, 8192, 4352, 534_773_760, 40_000),
                    ("5.2", 14, 8_912_896, 8192, 4352, 1_069_547_520, 60_000),
                    ("5.3", 15, 8_912_896, 8192, 4352, 1_069_547_520, 60_000),
                    ("6.0", 16, 35_651_584, 16_384, 8704, 1_069_547_520, 60_000),
                    ("6.1", 17, 35_651_584, 16_384, 8704, 2_139_095_040, 100_000),
                    ("6.2", 18, 35_651_584, 16_384, 8704, 4_278_190_080, 160_000),
                ];
                type Av1Level = (&'static str, u32, u64, u64, u64, u64, u64);
                let fits = |&&(_, _, max_pic, max_w, max_h, max_rate, max_br): &&Av1Level| {
                    samples <= max_pic
                        && w <= max_w
                        && h <= max_h
                        && samples * rate <= max_rate
                        && kbps <= max_br
                };
                let &(name, idx, ..) = LEVELS.iter().find(fits).unwrap_or(LEVELS.last().unwrap());
                Level {
                    name,
                    nvenc: idx,
                    codec_string: format!("av01.0.{idx:02}M.08"),
                }
            }
        }
    }

    /// Whether one encoded frame is a keyframe (a client can start there).
    pub fn is_keyframe(self, frame: &[u8]) -> bool {
        match self {
            Codec::Hevc => nal_headers(frame).any(|header| is_irap(frame[header])),
            Codec::Av1 => {
                let mut at = 0;
                while let Some((kind, header_len, total)) = parse_obu(&frame[at..]) {
                    if kind == OBU_FRAME || kind == OBU_FRAME_HEADER {
                        // uncompressed_header: show_existing_frame f(1), frame_type f(2);
                        // KEY_FRAME = 0.
                        return frame
                            .get(at + header_len)
                            .is_some_and(|first| first & 0xe0 == 0);
                    }
                    at += total;
                }
                false
            }
        }
    }
}

/// Whether an H.265 NAL header starts an IRAP picture (BLA, IDR, CRA: 16..=21).
fn is_irap(header: u8) -> bool {
    (16..=21).contains(&((header >> 1) & 0x3f))
}

/// Offsets of NAL header bytes in an Annex B buffer.
fn nal_headers(data: &[u8]) -> impl Iterator<Item = usize> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        while i + 3 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                i += 3;
                return Some(i);
            }
            i += 1;
        }
        None
    })
}

#[cfg(test)]
const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

/// `(obu_type, bytes before the payload, total bytes)` of the complete OBU
/// at the start of `data`, or `None` if it is not all there yet.
fn parse_obu(data: &[u8]) -> Option<(u8, usize, usize)> {
    let header = *data.first()?;
    let kind = (header >> 3) & 0x0f;
    let extension = (header >> 2) & 1 == 1;
    let has_size = (header >> 1) & 1 == 1;
    let mut at = 1 + extension as usize;
    if !has_size {
        // Not what NVENC writes; treat the rest as one OBU.
        return (data.len() > at).then_some((kind, at, data.len()));
    }
    let mut size: u64 = 0;
    for i in 0..8 {
        let byte = *data.get(at)?;
        at += 1;
        size |= ((byte & 0x7f) as u64) << (7 * i);
        if byte & 0x80 == 0 {
            let total = at + size as usize;
            return (data.len() >= total).then_some((kind, at, total));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_frame_size_rate_and_bitrate() {
        assert_eq!(
            Codec::Hevc.level(1920, 1408, 60, 20).codec_string,
            "hvc1.1.6.L150.B0"
        );
        // 45 Mbit/s is over level 5.1's 40: NVENC rejects 5.1 for it.
        assert_eq!(
            Codec::Hevc.level(2560, 1760, 60, 45).codec_string,
            "hvc1.1.6.L156.B0"
        );
        assert_eq!(
            Codec::Hevc.level(2560, 1760, 60, 30).codec_string,
            "hvc1.1.6.L153.B0"
        );
        // 3840x2480 is over level 5's 8.9 M luma samples; 80 Mbit/s needs 6.1.
        assert_eq!(
            Codec::Hevc.level(3840, 2480, 60, 80).codec_string,
            "hvc1.1.6.L183.B0"
        );
        assert_eq!(
            Codec::Av1.level(1920, 1408, 60, 30).codec_string,
            "av01.0.12M.08"
        );
        assert_eq!(Codec::Av1.level(3840, 2480, 60, 80).name, "6.1");
        assert_eq!(Codec::Av1.level(3840, 2480, 60, 80).nvenc, 17);
        assert_eq!(Codec::Hevc.level(2560, 1760, 60, 30).nvenc, 153);
    }

    // H.265 NAL units: the type is in bits 1..7 of the first header byte.
    const AUD: &[u8] = &[0, 0, 0, 1, 35 << 1, 1, 0x50];
    const IDR: &[u8] = &[0, 0, 1, 19 << 1, 1, 0xaa];
    const TRAIL: &[u8] = &[0, 0, 1, 1 << 1, 1, 0xbb];

    #[test]
    fn hevc_keyframes_have_an_irap_slice() {
        assert!(Codec::Hevc.is_keyframe(&[AUD, IDR].concat()));
        assert!(!Codec::Hevc.is_keyframe(&[AUD, TRAIL].concat()));
    }

    fn obu(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![(kind << 3) | 0b10, payload.len() as u8];
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn av1_keyframes_have_frame_type_zero() {
        let td = obu(OBU_TEMPORAL_DELIMITER, &[]);
        let sequence = obu(1, &[0x00, 0x01]);
        let key_frame = obu(OBU_FRAME, &[0x10, 0xaa]); // frame_type 0
        let inter = obu(OBU_FRAME, &[0x30, 0xbb]); // frame_type 1
        assert!(Codec::Av1.is_keyframe(&[td.clone(), sequence, key_frame].concat()));
        assert!(!Codec::Av1.is_keyframe(&[td, inter].concat()));
    }

    #[test]
    fn obu_sizes_use_leb128() {
        let mut long = vec![(OBU_FRAME << 3) | 0b10, 0x80 | 0x2c, 0x01];
        long.extend(std::iter::repeat_n(0u8, 172));
        assert_eq!(parse_obu(&long), Some((OBU_FRAME, 3, 175)));
        assert_eq!(parse_obu(&long[..100]), None);
    }
}
