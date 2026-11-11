//! Long video, held together (docs/LONG-MEDIA-PLAN.md, video "Next" 1 and 2). Wraps mold's chain stage renderer:
//! - **colour held to the first clip:** each later clip's pooled per-channel histogram is matched to clip 1's (one
//!   lookup table per clip, so the clip stays steady in itself) before its frame seeds the next clip. This stops the
//!   brightness and saturation drift that each clip otherwise passes on, compounding.
//! - **overlapped crossfades:** the next clip starts from the frame `overlap` frames before this clip's end, so the
//!   two clips cover the same `overlap` frames of time; the stitch crossfades them instead of cutting.
//! - **a smoothed start frame:** the carried frame is taken to half size and back, so its blocky artefacts are not
//!   passed on ("smooth_start", default on).
//!
//! Only for chains whose clips continue one scene (one prompt): a storyboard's clips may change light on purpose.

use anyhow::Result;
use image::RgbImage;
use mold_inference::chain::{ChainStageRenderer, ChainTail, StageOutcome, StageProgressEvent, StageSidecar};

type Hist = [[u64; 256]; 3];

fn hist(frames: &[RgbImage]) -> Hist {
    let mut h = [[0u64; 256]; 3];
    for f in frames {
        for p in f.pixels() {
            for c in 0..3 {
                h[c][p.0[c] as usize] += 1;
            }
        }
    }
    h
}

/// The lookup table that maps `src`'s distribution onto `reference`'s, per channel (histogram matching).
fn match_lut(src: &Hist, reference: &Hist) -> [[u8; 256]; 3] {
    let mut lut = [[0u8; 256]; 3];
    for c in 0..3 {
        let cdf = |h: &[u64; 256]| {
            let total = h.iter().sum::<u64>().max(1) as f64;
            let mut acc = 0u64;
            h.map(|n| {
                acc += n;
                acc as f64 / total
            })
        };
        let (s, r) = (cdf(&src[c]), cdf(&reference[c]));
        let mut j = 0usize;
        for i in 0..256 {
            while j < 255 && r[j] < s[i] {
                j += 1;
            }
            lut[c][i] = j as u8;
        }
    }
    lut
}

fn apply(frames: &mut [RgbImage], lut: &[[u8; 256]; 3]) {
    for f in frames {
        for p in f.pixels_mut() {
            for c in 0..3 {
                p.0[c] = lut[c][p.0[c] as usize];
            }
        }
    }
}

/// Mean brightness (0-255) of a clip, for the log.
fn mean(frames: &[RgbImage]) -> f64 {
    let h = hist(frames);
    let n: u64 = h[0].iter().sum::<u64>().max(1) * 3;
    (0..3).map(|c| h[c].iter().enumerate().map(|(v, k)| v as u64 * k).sum::<u64>()).sum::<u64>() as f64 / n as f64
}

pub struct Held<'a, R: ChainStageRenderer + ?Sized> {
    inner: &'a mut R,
    reference: Option<Hist>,
    hold_colour: bool,
    overlap: usize,
    smooth: bool,
}

impl<'a, R: ChainStageRenderer + ?Sized> Held<'a, R> {
    pub fn new(inner: &'a mut R, hold_colour: bool, overlap: usize, smooth: bool) -> Self {
        Held { inner, reference: None, hold_colour, overlap, smooth }
    }
}

impl<R: ChainStageRenderer + ?Sized> ChainStageRenderer for Held<'_, R> {
    fn render_stage(&mut self, stage_req: &mold_core::GenerateRequest, carry: Option<&ChainTail>, motion_tail_pixel_frames: u32,
                    hdr_sidecar: Option<&StageSidecar>, stage_progress: Option<&mut dyn FnMut(StageProgressEvent)>)
                    -> Result<StageOutcome> {
        let mut out = self.inner.render_stage(stage_req, carry, motion_tail_pixel_frames, hdr_sidecar, stage_progress)?;
        if self.hold_colour {
            match &self.reference {
                None => self.reference = Some(hist(&out.frames)),
                Some(r) => {
                    let before = mean(&out.frames);
                    let lut = match_lut(&hist(&out.frames), r);
                    apply(&mut out.frames, &lut);
                    eprintln!("[media] colour held to clip 1: mean {before:.1} -> {:.1}", mean(&out.frames));
                }
            }
        }
        // the frame the next clip starts from: `overlap` frames before the end (the last frame when no overlap)
        let n = out.frames.len();
        let k = self.overlap.min(n.saturating_sub(2));
        let mut seed = out.frames[n - 1 - k].clone();
        if self.smooth {
            seed = smoothed(&seed);
        }
        out.tail = ChainTail { frames: 1, tail_rgb_frames: vec![seed] };
        Ok(out)
    }
}

/// The start frame, smoothed: down to half size and back (a Catmull-Rom filter), which takes off the blocky
/// high-frequency artefacts a clip passes on and leaves the shapes and colours; the next clip draws the detail again.
pub fn smoothed(f: &RgbImage) -> RgbImage {
    use image::imageops::{resize, FilterType};
    let (w, h) = f.dimensions();
    let half = resize(f, (w / 2).max(1), (h / 2).max(1), FilterType::Triangle);
    resize(&half, w, h, FilterType::CatmullRom)
}

/// The clips as one video: each later clip's first frame (the carried one) is dropped, and its next `overlap` frames
/// are crossfaded over the earlier clip's last `overlap` frames (the same moments, drawn twice).
pub fn stitch(clips: Vec<Vec<RgbImage>>, overlap: usize) -> Vec<RgbImage> {
    let mut out: Vec<RgbImage> = Vec::new();
    for (i, clip) in clips.into_iter().enumerate() {
        if i == 0 {
            out = clip;
            continue;
        }
        let k = overlap.min(clip.len().saturating_sub(2)).min(out.len());
        let start = out.len() - k;
        for j in 0..k {
            // a rises from 1/(k+1) to k/(k+1): the earlier clip fades out as the later one fades in
            let a = (j + 1) as f32 / (k + 1) as f32;
            let (p, q) = (&mut out[start + j], &clip[1 + j]);
            for (x, y) in p.pixels_mut().zip(q.pixels()) {
                for c in 0..3 {
                    x.0[c] = (x.0[c] as f32 * (1.0 - a) + y.0[c] as f32 * a).round() as u8;
                }
            }
        }
        out.extend(clip.into_iter().skip(1 + k));
    }
    out
}

#[cfg(test)]
/// The frames a chain of `clips` clips of `n` frames gives, with this overlap.
pub fn stitched_len(clips: usize, n: usize, overlap: usize) -> usize {
    if clips == 0 { 0 } else { n + (clips - 1) * n.saturating_sub(1 + overlap.min(n.saturating_sub(2))) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(v: u8, n: usize) -> Vec<RgbImage> {
        (0..n).map(|_| RgbImage::from_pixel(4, 4, image::Rgb([v, v / 2, 255 - v]))).collect()
    }

    #[test]
    fn a_brighter_clip_is_brought_back_to_the_first() {
        // clip 1: a gradient 0..=200; clip 2: the same gradient, 40 brighter
        let grad = |off: u8| -> Vec<RgbImage> {
            vec![RgbImage::from_fn(201, 1, |x, _| { let v = x as u8 + off; image::Rgb([v, v, v]) })]
        };
        let (a, mut b) = (grad(0), grad(40));
        let lut = match_lut(&hist(&b), &hist(&a));
        apply(&mut b, &lut);
        assert!((mean(&b) - mean(&a)).abs() < 1.0, "{} vs {}", mean(&b), mean(&a));
    }

    #[test]
    fn the_start_frame_keeps_its_size_and_colour_and_loses_its_blocks() {
        // a 2-pixel checkerboard (a stand-in for block artefacts) on mid grey
        let f = RgbImage::from_fn(64, 32, |x, y| { let v = if (x / 2 + y / 2) % 2 == 0 { 150 } else { 100 }; image::Rgb([v, v, v]) });
        let g = smoothed(&f);
        assert_eq!(g.dimensions(), (64, 32));
        assert!((mean(&[g.clone()]) - mean(&[f])).abs() < 3.0);
        let spread = g.pixels().map(|p| (p.0[0] as i32 - 125).abs()).max().unwrap();
        assert!(spread < 25, "the checkerboard is smoothed: {spread}");
    }

    #[test]
    fn overlapped_clips_crossfade_and_keep_their_length() {
        let clips = vec![solid(0, 9), solid(200, 9), solid(100, 9)];
        let v = stitch(clips, 4);
        assert_eq!(v.len(), stitched_len(3, 9, 4));
        assert_eq!(v.len(), 9 + 4 + 4);
        // the fade from 0 to 200 over 4 frames rises steadily: 40, 80, 120, 160
        let r: Vec<u8> = v[5..9].iter().map(|f| f.get_pixel(0, 0).0[0]).collect();
        assert_eq!(r, vec![40, 80, 120, 160]);
        // clip 2's last 4 frames fade into clip 3 (200 to 100): 180 first
        assert_eq!(v[9].get_pixel(0, 0).0[0], 180);
        assert_eq!(v[16].get_pixel(0, 0).0[0], 100);
        // no overlap: each later clip drops only its carried frame
        assert_eq!(stitch(vec![solid(0, 9), solid(9, 9)], 0).len(), 17);
    }
}
