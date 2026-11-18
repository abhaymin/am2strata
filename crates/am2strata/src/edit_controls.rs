//! How much an image edit keeps of the original (POST /v1/images/edits): the controls a request may add, and the
//! finishing done on each result, whatever model drew it.
//!
//! - `strength` (0.05-1.0): how much the model may change: the original is the starting point (image to image) on a
//!   model that takes a source image (Z-Image, FLUX, SDXL, SD 1.5/3, Qwen Image 1); 1.0 draws anew.
//! - `mask` (an image, white = change, black = keep): on such a model the model redraws only there (inpainting); on
//!   every model the original's own pixels are put back outside it, with a soft edge (`mask_feather`, px).
//! - `keep_colors`: the result's colours are matched to the original's (mean and spread per channel).
//! - `blend` (0-1): how much of the original shows through the whole result.

use image::{imageops, GrayImage, RgbImage};
use serde_json::{json, Value};

/// Model families that start from a source image and take a mask (mold's pipelines with source_image + mask_image).
pub const RETOUCH_FAMILIES: &[&str] = &["z-image", "flux", "flux2", "sdxl", "sd15", "sd3", "qwen-image", "wuerstchen"];

#[derive(Clone, Debug, Default)]
pub struct EditControls {
    pub strength: Option<f64>,
    pub mask: Option<Vec<u8>>,
    pub feather: u32,
    pub keep_colors: bool,
    pub blend: f64,
}

impl EditControls {
    /// From a request's fields; Err names the field that is wrong.
    pub fn from_fields(v: &Value, mask: Option<Vec<u8>>) -> Result<EditControls, String> {
        let num = |k: &str| -> Result<Option<f64>, String> {
            match &v[k] {
                Value::Null => Ok(None),
                Value::Number(n) => Ok(n.as_f64()),
                Value::String(s) if s.is_empty() => Ok(None),
                Value::String(s) => s.parse::<f64>().map(Some).map_err(|_| format!("{k} must be a number")),
                _ => Err(format!("{k} must be a number")),
            }
        };
        let strength = num("strength")?;
        if strength.is_some_and(|s| !(0.05..=1.0).contains(&s)) {
            return Err("strength is how much may change, from 0.05 (almost nothing) to 1.0 (draw anew)".into());
        }
        let blend = num("blend")?.unwrap_or(0.0);
        if !(0.0..=0.95).contains(&blend) {
            return Err("blend is how much of the original shows through, from 0 to 0.95".into());
        }
        let feather = num("mask_feather")?.unwrap_or(8.0);
        if !(0.0..=128.0).contains(&feather) {
            return Err("mask_feather is the soft edge in pixels, 0 to 128".into());
        }
        let keep_colors = matches!(&v["keep_colors"], Value::Bool(true)) || v["keep_colors"].as_str().is_some_and(|s| s == "true" || s == "1");
        if let Some(m) = &mask {
            image::load_from_memory(m).map_err(|e| format!("mask: not an image ({e})"))?;
        }
        Ok(EditControls { strength, mask, feather: feather as u32, keep_colors, blend })
    }

    /// Whether anything is done after the model (a mask, colours or a blend).
    pub fn finishes(&self) -> bool {
        self.mask.is_some() || self.keep_colors || self.blend > 0.0
    }

    pub fn describe(&self) -> Value {
        json!({"strength": self.strength, "mask": self.mask.is_some(), "mask_feather": self.feather,
               "keep_colors": self.keep_colors, "blend": self.blend})
    }
}

/// The mask at `w`x`h`, softened by `feather` px: 0.0 keep the original, 1.0 take the result.
fn mask_at(mask: &[u8], w: u32, h: u32, feather: u32) -> Result<GrayImage, String> {
    let m = image::load_from_memory(mask).map_err(|e| format!("mask: {e}"))?.to_luma8();
    let mut m = imageops::resize(&m, w, h, imageops::FilterType::Triangle);
    if feather > 0 {
        m = imageops::blur(&m, feather as f32 / 2.0);
    }
    Ok(m)
}

/// Per channel: the result's mean and spread moved to the original's (Reinhard's colour transfer, in RGB).
fn match_colors(out: &mut RgbImage, orig: &RgbImage) {
    let stats = |img: &RgbImage| {
        let n = (img.width() * img.height()).max(1) as f64;
        let mut s = [0f64; 3];
        let mut q = [0f64; 3];
        for p in img.pixels() {
            for c in 0..3 {
                let v = p[c] as f64;
                s[c] += v;
                q[c] += v * v;
            }
        }
        let mean = s.map(|x| x / n);
        let sd = [0, 1, 2].map(|c| (q[c] / n - mean[c] * mean[c]).max(1e-6).sqrt());
        (mean, sd)
    };
    let (mo, so) = stats(out);
    let (mt, st) = stats(orig);
    for p in out.pixels_mut() {
        for c in 0..3 {
            let v = (p[c] as f64 - mo[c]) / so[c] * st[c] + mt[c];
            p[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// The finished result: colours matched, the original blended in, and kept outside the mask, at the larger of the two
/// images' sizes. -> a PNG.
pub fn finish(original: &[u8], result_png: &[u8], c: &EditControls) -> Result<Vec<u8>, String> {
    let out_img = image::load_from_memory(result_png).map_err(|e| format!("the result: {e}"))?.to_rgb8();
    let orig = image::load_from_memory(original).map_err(|e| format!("the original: {e}"))?.to_rgb8();
    // at the larger of the two (at most 4096 px a side): outside the mask the original keeps its own resolution
    let (rw, rh) = out_img.dimensions();
    let (ow, oh) = orig.dimensions();
    let (mut w, mut h) = if ow * oh > rw * rh { (ow, oh) } else { (rw, rh) };
    let k = (4096.0 / w.max(h) as f64).min(1.0);
    (w, h) = (((w as f64 * k) as u32).max(1), ((h as f64 * k) as u32).max(1));
    let fit = |img: RgbImage| if img.dimensions() == (w, h) { img } else { imageops::resize(&img, w, h, imageops::FilterType::Lanczos3) };
    let orig = fit(orig);
    let mut out = fit(out_img);
    if c.keep_colors {
        match_colors(&mut out, &orig);
    }
    if c.blend > 0.0 {
        for (o, p) in out.pixels_mut().zip(orig.pixels()) {
            for ch in 0..3 {
                o[ch] = (o[ch] as f64 * (1.0 - c.blend) + p[ch] as f64 * c.blend).round() as u8;
            }
        }
    }
    if let Some(m) = &c.mask {
        let m = mask_at(m, w, h, c.feather)?;
        for ((o, p), a) in out.pixels_mut().zip(orig.pixels()).zip(m.pixels()) {
            let a = a[0] as f64 / 255.0;
            for ch in 0..3 {
                o[ch] = (o[ch] as f64 * a + p[ch] as f64 * (1.0 - a)).round() as u8;
            }
        }
    }
    let mut buf = std::io::Cursor::new(Vec::new());
    out.write_to(&mut buf, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(img: &RgbImage) -> Vec<u8> {
        let mut b = std::io::Cursor::new(Vec::new());
        img.write_to(&mut b, image::ImageFormat::Png).unwrap();
        b.into_inner()
    }

    #[test]
    fn outside_the_mask_the_original_stays_exactly() {
        let orig = RgbImage::from_pixel(64, 64, image::Rgb([200, 10, 10]));
        let result = RgbImage::from_pixel(64, 64, image::Rgb([10, 10, 200]));
        // change only the right half
        let mut mask = GrayImage::new(64, 64);
        for (x, _, p) in mask.enumerate_pixels_mut() {
            p[0] = if x >= 32 { 255 } else { 0 };
        }
        let mut mb = std::io::Cursor::new(Vec::new());
        mask.write_to(&mut mb, image::ImageFormat::Png).unwrap();
        let c = EditControls::from_fields(&json!({"mask_feather": 0}), Some(mb.into_inner())).unwrap();
        let done = image::load_from_memory(&finish(&png(&orig), &png(&result), &c).unwrap()).unwrap().to_rgb8();
        assert_eq!(done.get_pixel(5, 5).0, [200, 10, 10], "kept");
        assert_eq!(done.get_pixel(60, 5).0, [10, 10, 200], "changed");
    }

    #[test]
    fn colours_and_blend_bring_the_original_back() {
        let orig = RgbImage::from_fn(32, 32, |x, _| image::Rgb([100 + x as u8, 100, 100]));
        let result = RgbImage::from_fn(32, 32, |x, _| image::Rgb([20 + x as u8 * 2, 200, 50]));
        let c = EditControls::from_fields(&json!({"keep_colors": true}), None).unwrap();
        let done = image::load_from_memory(&finish(&png(&orig), &png(&result), &c).unwrap()).unwrap().to_rgb8();
        let mean = |img: &RgbImage, ch: usize| img.pixels().map(|p| p[ch] as f64).sum::<f64>() / 1024.0;
        assert!((mean(&done, 1) - 100.0).abs() < 2.0 && (mean(&done, 0) - mean(&orig, 0)).abs() < 2.0, "colours matched");
        let c = EditControls::from_fields(&json!({"blend": 0.5}), None).unwrap();
        let done = image::load_from_memory(&finish(&png(&orig), &png(&result), &c).unwrap()).unwrap().to_rgb8();
        assert_eq!(done.get_pixel(0, 0).0, [60, 150, 75], "half of each");
    }

    #[test]
    fn wrong_controls_are_refused_by_name() {
        assert!(EditControls::from_fields(&json!({"strength": 1.5}), None).unwrap_err().contains("strength"));
        assert!(EditControls::from_fields(&json!({"blend": 1}), None).unwrap_err().contains("blend"));
        assert!(EditControls::from_fields(&json!({}), Some(b"not an image".to_vec())).unwrap_err().contains("mask"));
        assert!(!EditControls::from_fields(&json!({"strength": "0.4"}), None).unwrap().finishes());
    }
}
