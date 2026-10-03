//! Raw relative depth to a stable 8-bit map.
//!
//! Depth-Anything output has no fixed scale and is not temporally stable: on
//! a desktop (a page around a playing video) the whole map can swing between
//! frames. Parallax is proportional to depth, so every swing shifts the
//! picture sideways and reads as shaking. Two smoothers fight it:
//!
//! * the range (a pair of percentiles, robust to outlier pixels) follows
//!   the scene slowly;
//! * every pixel's mapped depth follows its new value exponentially, unless
//!   it changed a lot: a moving object's edge is taken at once (no trail
//!   behind it), small changes (the model's own jitter) are smoothed.
//!
//! Both are dropped on a scene cut, which the caller detects from the
//! picture ([`DepthNormalizer::reset`]); jumps in the depth itself are not
//! taken as cuts, since snapping to them is what amplifies the swings.
//! Measured on a recorded desktop session (video in a browser page), this
//! cut the mean frame-to-frame change of the map from 10.1 to 2.5 (of 255)
//! and the 99th percentile from 44 to 10, keeping the depth contrast.

use crate::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct NormalizerConfig {
    /// Percentile (0..1) of a frame's depth taken as the far end of its range.
    pub low_percentile: f32,
    /// Percentile (0..1) taken as the near end.
    pub high_percentile: f32,
    /// Weight of the newest frame in the smoothed range, in (0, 1]; 1 means
    /// no smoothing.
    pub range_smoothing: f32,
    /// Weight of the newest frame in each pixel's mapped depth, in (0, 1],
    /// for small changes. Lower is steadier (0.15 at 60 fps: a time
    /// constant of about 100 ms).
    pub pixel_smoothing: f32,
    /// Changes of a pixel's mapped depth (0..255 scale) up to this are
    /// smoothed with `pixel_smoothing`; from `motion_high` on the pixel takes
    /// its new value outright (an object moved); the weight ramps between.
    pub motion_low: f32,
    pub motion_high: f32,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            low_percentile: 0.02,
            high_percentile: 0.98,
            range_smoothing: 0.03,
            pixel_smoothing: 0.15,
            motion_low: 12.0,
            motion_high: 48.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DepthNormalizer {
    config: NormalizerConfig,
    range: Option<(f32, f32)>,
    /// Each pixel's smoothed depth, 0..=255; empty until the first frame.
    smoothed: Vec<f32>,
    /// The frame's finite values, reordered by the percentile search.
    scratch: Vec<f32>,
}

impl DepthNormalizer {
    pub fn new(config: NormalizerConfig) -> Result<Self> {
        let NormalizerConfig {
            low_percentile: low,
            high_percentile: high,
            range_smoothing,
            pixel_smoothing,
            motion_low,
            motion_high,
        } = config;
        if !(0.0..1.0).contains(&low) || low >= high || high > 1.0 || high.is_nan() {
            return Err(Error::Invalid(format!(
                "percentiles must satisfy 0 <= low < high <= 1, got {low} and {high}"
            )));
        }
        if !(0.0..motion_high).contains(&motion_low) {
            return Err(Error::Invalid(format!(
                "motion thresholds must satisfy 0 <= low < high, got {motion_low} and {motion_high}"
            )));
        }
        let unit = |v: f32| v > 0.0 && v <= 1.0;
        if !unit(range_smoothing) || !unit(pixel_smoothing) {
            return Err(Error::Invalid(format!(
                "smoothing weights must be in (0, 1], got {range_smoothing} and {pixel_smoothing}"
            )));
        }
        Ok(Self {
            config,
            range: None,
            smoothed: Vec::new(),
            scratch: Vec::new(),
        })
    }

    /// Forgets the smoothed range and depth (a scene cut): the next frame
    /// is taken as it is.
    pub fn reset(&mut self) {
        self.range = None;
        self.smoothed.clear();
    }

    /// The `(far, near)` raw depth values mapped to 0 and 255 by the last call.
    pub fn range(&self) -> Option<(f32, f32)> {
        self.range
    }

    /// The last call's smoothed depth before rounding, 0 (far) ..= 255 (near).
    pub fn smoothed(&self) -> &[f32] {
        &self.smoothed
    }

    /// Writes `depth` mapped to 0 (far) ..= 255 (near) into `out`.
    pub fn normalize(&mut self, depth: &[f32], out: &mut [u8]) -> Result<()> {
        if depth.len() != out.len() {
            return Err(Error::Invalid(format!(
                "depth has {} values but the output {}",
                depth.len(),
                out.len()
            )));
        }
        let Some((low, high)) = self.frame_range(depth) else {
            out.fill(0);
            return Ok(());
        };
        let (low, high) = match self.range {
            None => (low, high),
            Some((smooth_low, smooth_high)) => {
                let a = self.config.range_smoothing;
                (
                    smooth_low + a * (low - smooth_low),
                    smooth_high + a * (high - smooth_high),
                )
            }
        };
        self.range = Some((low, high));
        let scale = 255.0 / (high - low).max(f32::EPSILON);
        // NaN maps to 0 (the clamp keeps it NaN; the cast makes it 0).
        let map = |value: f32| ((value - low) * scale).clamp(0.0, 255.0);
        if self.smoothed.len() != depth.len() {
            self.smoothed.clear();
            self.smoothed.extend(depth.iter().map(|&value| map(value)));
        } else {
            let NormalizerConfig {
                pixel_smoothing: a,
                motion_low: low,
                motion_high: high,
                ..
            } = self.config;
            for (smoothed, &value) in self.smoothed.iter_mut().zip(depth) {
                let target = map(value);
                *smoothed = if smoothed.is_nan() {
                    target
                } else {
                    let change = target - *smoothed;
                    let motion = ((change.abs() - low) / (high - low)).clamp(0.0, 1.0);
                    *smoothed + (a + (1.0 - a) * motion) * change
                };
            }
        }
        for (pixel, &value) in out.iter_mut().zip(&self.smoothed) {
            *pixel = (value + 0.5) as u8;
        }
        Ok(())
    }

    /// This frame's `(low, high)` percentiles over its finite values (every
    /// few pixels of a large map: ~100k samples estimate them closely).
    fn frame_range(&mut self, depth: &[f32]) -> Option<(f32, f32)> {
        let stride = (depth.len() / 100_000).max(1);
        self.scratch.clear();
        self.scratch.extend(
            depth
                .iter()
                .step_by(stride)
                .copied()
                .filter(|value| value.is_finite()),
        );
        let count = self.scratch.len();
        if count == 0 {
            return None;
        }
        let rank =
            |percentile: f32| ((percentile * (count - 1) as f32).round() as usize).min(count - 1);
        let (low_rank, high_rank) = (
            rank(self.config.low_percentile),
            rank(self.config.high_percentile),
        );
        // Exact order statistics in O(n); the second search only covers the
        // values above the first.
        let (_, &mut low, above) = self
            .scratch
            .select_nth_unstable_by(low_rank, f32::total_cmp);
        let high = if high_rank > low_rank {
            *above
                .select_nth_unstable_by(high_rank - low_rank - 1, f32::total_cmp)
                .1
        } else {
            low
        };
        let floor = f32::EPSILON * low.abs().max(1.0);
        Some((low, high.max(low + floor)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize, offset: f32, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| offset + scale * i as f32 / (len - 1) as f32)
            .collect()
    }

    fn unsmoothed() -> NormalizerConfig {
        NormalizerConfig {
            range_smoothing: 1.0,
            pixel_smoothing: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn ramp_maps_percentiles_to_the_ends() {
        let mut normalizer = DepthNormalizer::new(unsmoothed()).unwrap();
        let depth = ramp(10_000, 0.0, 10.0);
        let mut out = vec![0; depth.len()];
        normalizer.normalize(&depth, &mut out).unwrap();
        let (low, high) = normalizer.range().unwrap();
        assert!((low - 0.2).abs() < 0.02, "low {low}");
        assert!((high - 9.8).abs() < 0.02, "high {high}");
        assert_eq!(out[0], 0);
        assert_eq!(out[out.len() - 1], 255);
        assert!((out[out.len() / 2] as i32 - 128).abs() <= 2);
    }

    #[test]
    fn outliers_do_not_stretch_the_range() {
        let mut normalizer = DepthNormalizer::new(unsmoothed()).unwrap();
        let mut depth = ramp(10_000, 0.0, 1.0);
        depth[17] = 1000.0;
        depth[18] = f32::NAN;
        let mut out = vec![0; depth.len()];
        normalizer.normalize(&depth, &mut out).unwrap();
        let (_, high) = normalizer.range().unwrap();
        assert!(high < 1.5, "high {high}");
        assert_eq!(out[17], 255);
        assert_eq!(out[18], 0);
    }

    #[test]
    fn range_follows_slowly_even_on_large_jumps() {
        let mut normalizer = DepthNormalizer::new(NormalizerConfig {
            low_percentile: 0.0,
            high_percentile: 1.0,
            range_smoothing: 0.5,
            pixel_smoothing: 1.0,
            ..Default::default()
        })
        .unwrap();
        let mut out = vec![0; 1000];
        normalizer
            .normalize(&ramp(1000, 0.0, 1.0), &mut out)
            .unwrap();
        normalizer
            .normalize(&ramp(1000, 0.2, 0.8), &mut out)
            .unwrap();
        let (low, _) = normalizer.range().unwrap();
        assert!((low - 0.1).abs() < 0.01, "low {low}");
        // A jump in depth alone is not a cut: still halfway.
        normalizer
            .normalize(&ramp(1000, 50.0, 1.0), &mut out)
            .unwrap();
        let (low, _) = normalizer.range().unwrap();
        assert!((low - 25.05).abs() < 0.01, "low {low}");
        // A cut reported by the caller is taken outright.
        normalizer.reset();
        normalizer
            .normalize(&ramp(1000, 50.0, 1.0), &mut out)
            .unwrap();
        let (low, _) = normalizer.range().unwrap();
        assert!((low - 50.0).abs() < 0.01, "low {low}");
    }

    #[test]
    fn pixels_follow_new_depth_exponentially() {
        let mut normalizer = DepthNormalizer::new(NormalizerConfig {
            low_percentile: 0.0,
            high_percentile: 1.0,
            range_smoothing: 1.0,
            pixel_smoothing: 0.25,
            // Every change counts as small here.
            motion_low: 1000.0,
            motion_high: 2000.0,
        })
        .unwrap();
        // Two pixels swap ends: each moves a quarter of the way per frame.
        let mut out = [0u8; 2];
        normalizer.normalize(&[0.0, 1.0], &mut out).unwrap();
        assert_eq!(out, [0, 255]);
        normalizer.normalize(&[1.0, 0.0], &mut out).unwrap();
        assert_eq!(out, [64, 191]);
        normalizer.normalize(&[1.0, 0.0], &mut out).unwrap();
        assert_eq!(out, [112, 143]);
        normalizer.reset();
        normalizer.normalize(&[1.0, 0.0], &mut out).unwrap();
        assert_eq!(out, [255, 0]);
    }

    #[test]
    fn large_changes_are_taken_at_once() {
        let mut normalizer = DepthNormalizer::new(NormalizerConfig {
            low_percentile: 0.0,
            high_percentile: 1.0,
            range_smoothing: 1.0,
            pixel_smoothing: 0.25,
            motion_low: 12.0,
            motion_high: 48.0,
        })
        .unwrap();
        // Pixels 0 and 3 pin the range to 0..=255 (1 unit = 1 step of 255).
        let mut out = [0u8; 4];
        normalizer.normalize(&[0.0, 100.0, 100.0, 255.0], &mut out).unwrap();
        // Pixel 1 jitters by 8 (smoothed: a quarter), pixel 2 jumps by 100
        // (an edge moved: taken outright).
        normalizer.normalize(&[0.0, 108.0, 200.0, 255.0], &mut out).unwrap();
        assert_eq!(out, [0, 102, 200, 255]);
        // Halfway between the thresholds: weight halfway between 0.25 and 1.
        normalizer.normalize(&[0.0, 108.0, 230.0, 255.0], &mut out).unwrap();
        assert_eq!(out[2], 219);
    }

    #[test]
    fn flat_and_empty_frames_are_handled() {
        let mut normalizer = DepthNormalizer::new(unsmoothed()).unwrap();
        let mut out = vec![7; 4];
        normalizer.normalize(&[3.0; 4], &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0 || v == 255));
        normalizer.reset();
        normalizer.normalize(&[f32::NAN; 4], &mut out).unwrap();
        assert_eq!(out, [0; 4]);
        assert!(normalizer.normalize(&[1.0; 4], &mut [0; 3]).is_err());
    }

    #[test]
    fn rejects_bad_config() {
        let bad = |config| DepthNormalizer::new(config).is_err();
        assert!(bad(NormalizerConfig {
            low_percentile: 0.9,
            high_percentile: 0.1,
            ..Default::default()
        }));
        assert!(bad(NormalizerConfig {
            range_smoothing: 0.0,
            ..Default::default()
        }));
        assert!(bad(NormalizerConfig {
            pixel_smoothing: 1.5,
            ..Default::default()
        }));
        assert!(bad(NormalizerConfig {
            motion_low: 50.0,
            motion_high: 40.0,
            ..Default::default()
        }));
    }
}
