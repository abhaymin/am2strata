//! Captions and title cards drawn by am2Strata itself, so compose needs no libass (Homebrew's ffmpeg has none, and
//! a new machine needs fewer tools). Text is shaped with rustybuzz (Devanagari and Malayalam need shaping: conjuncts,
//! vowel signs before the consonant) and drawn with ab_glyph into a small RGBA box picture. ffmpeg's `overlay`, which
//! every build has, then lays each box on the video for its own seconds, fading in and out over a quarter second.
//!
//! The look matches the libass style it replaces: bold white type in a dark box. A caption sits low and centred
//! (58 px type and 120 px above the bottom, at the frame's own pixel size); a title or end card sits in the middle
//! (96 px). The box is darker than before (78% instead of 41%), because a review found the old one hard to read on
//! light pictures.
//!
//! Fonts: web/fonts/captions (DejaVu Sans Bold, Noto Sans Devanagari Bold and Noto Sans Malayalam Bold, with their
//! licences), else the system's own copies of the same fonts.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Where the app's caption fonts are (set at start from the web folder).
static FONT_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn set_font_dir(web_dir: &Path) {
    let _ = FONT_DIR.set(web_dir.join("fonts").join("captions"));
}

/// The scripts a line is drawn in, each with its own font.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Script { Latin, Devanagari, Malayalam }

/// The script of a run of text: Devanagari or Malayalam when it has any of those letters, else Latin (which covers
/// punctuation, digits and the emoji-free text captions are).
pub fn script_of(text: &str) -> Script {
    if text.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c) || ('\u{A8E0}'..='\u{A8FF}').contains(&c)) {
        Script::Devanagari
    } else if text.chars().any(|c| ('\u{0D00}'..='\u{0D7F}').contains(&c)) {
        Script::Malayalam
    } else {
        Script::Latin
    }
}

fn font_file(s: Script) -> &'static str {
    match s { Script::Latin => "DejaVuSans-Bold.ttf", Script::Devanagari => "NotoSansDevanagari-Bold.ttf", Script::Malayalam => "NotoSansMalayalam-Bold.ttf" }
}

/// The font's bytes: the app's copy, else the system's (Linux, macOS, Windows places), read once.
fn font(s: Script) -> Result<&'static [u8], String> {
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<&'static str, &'static [u8]>>> = OnceLock::new();
    let name = font_file(s);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(b) = cache.lock().unwrap().get(name) {
        return Ok(b);
    }
    let mut places: Vec<PathBuf> = FONT_DIR.get().map(|d| vec![d.join(name)]).unwrap_or_default();
    // next to the program, for a server started without the web folder known yet (tests)
    places.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/fonts/captions").join(name));
    for dir in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/noto", "/usr/share/fonts/noto", "/Library/Fonts",
                "/System/Library/Fonts/Supplemental", "C:\\Windows\\Fonts"] {
        places.push(Path::new(dir).join(name));
    }
    let bytes = places.iter().find_map(|p| std::fs::read(p).ok())
        .ok_or_else(|| format!("the caption font {name} was not found (web/fonts/captions)"))?;
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    cache.lock().unwrap().insert(name, leaked);
    Ok(leaked)
}

/// One glyph placed on a line: its id and pen position in pixels.
struct Placed { id: u16, x: f32, y: f32 }

/// A line shaped at `px`: its glyphs and its width in pixels.
fn shape(text: &str, px: f32) -> Result<(Script, Vec<Placed>, f32), String> {
    let script = script_of(text);
    let bytes = font(script)?;
    let face = rustybuzz::Face::from_slice(bytes, 0).ok_or("the caption font could not be read")?;
    let scale = px / face.units_per_em() as f32;
    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(text);
    let out = rustybuzz::shape(&face, &[], buf);
    let mut x = 0.0f32;
    let mut glyphs = Vec::with_capacity(out.len());
    for (info, pos) in out.glyph_infos().iter().zip(out.glyph_positions()) {
        glyphs.push(Placed { id: info.glyph_id as u16, x: x + pos.x_offset as f32 * scale, y: -(pos.y_offset as f32) * scale });
        x += pos.x_advance as f32 * scale;
    }
    Ok((script, glyphs, x))
}

/// Words broken into lines no wider than `max` pixels (a word wider than that has a line to itself). Explicit line
/// breaks are kept.
pub fn wrap(text: &str, px: f32, max: f32) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for para in text.split('\n').map(str::trim).filter(|p| !p.is_empty()) {
        let mut cur = String::new();
        for word in para.split_whitespace() {
            let try_line = if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
            if !cur.is_empty() && shape(&try_line, px)?.2 > max {
                lines.push(std::mem::take(&mut cur));
                cur = word.to_string();
            } else {
                cur = try_line;
            }
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
    }
    Ok(lines)
}

/// The look of one kind of text.
#[derive(Clone, Copy)]
pub struct Style { pub px: f32, pub pad: f32, pub box_alpha: u8 }

pub const CAPTION: Style = Style { px: 58.0, pad: 14.0, box_alpha: 200 };
pub const CARD: Style = Style { px: 96.0, pad: 18.0, box_alpha: 200 };

/// The text as one box picture (RGBA PNG) no wider than `max_w`: each line centred on its own dark box, the lines
/// stacked. Returns the PNG and its size.
pub fn render(text: &str, style: Style, max_w: u32) -> Result<(Vec<u8>, u32, u32), String> {
    let inner = (max_w as f32 - 2.0 * style.pad).max(64.0);
    let lines = wrap(text, style.px, inner)?;
    if lines.is_empty() {
        return Err("no words to draw".into());
    }
    let line_h = (style.px * 1.45).ceil();
    let shaped: Vec<(Script, Vec<Placed>, f32)> = lines.iter().map(|l| shape(l, style.px)).collect::<Result<_, _>>()?;
    let widest = shaped.iter().map(|s| s.2).fold(0.0f32, f32::max);
    let w = (widest + 2.0 * style.pad).ceil().min(max_w as f32) as u32;
    let h = (line_h * lines.len() as f32 + 2.0 * style.pad).ceil() as u32;
    let mut img = image::RgbaImage::new(w.max(1), h.max(1));
    let (bg, fg) = ([6u8, 10, 24], [245u8, 248, 250]);
    for (k, (script, glyphs, lw)) in shaped.iter().enumerate() {
        // this line's own box (a short line gets a short box, as libass's opaque box style draws)
        let x0 = ((w as f32 - lw) / 2.0 - style.pad).max(0.0) as u32;
        let x1 = (((w as f32 + lw) / 2.0 + style.pad).ceil() as u32).min(w);
        let y0 = (k as f32 * line_h) as u32;
        let y1 = ((k as f32 + 1.0) * line_h + 2.0 * style.pad).ceil().min(h as f32) as u32;
        for y in y0..y1 { for x in x0..x1 { img.put_pixel(x, y, image::Rgba([bg[0], bg[1], bg[2], style.box_alpha])); } }
        let f = ab_glyph::FontRef::try_from_slice(font(*script)?).map_err(|e| format!("the caption font could not be read: {e}"))?;
        use ab_glyph::{Font, ScaleFont};
        // ab_glyph's scale is the font's whole height (ascent to descent); rustybuzz's advances are per em: draw at
        // the scale whose em is `px`, so the glyphs fill the advances they were shaped with
        let em = f.units_per_em().unwrap_or(1000.0);
        let scale = style.px * f.height_unscaled() / em;
        let sf = f.as_scaled(scale);
        let base = y0 as f32 + style.pad + (line_h - (sf.ascent() - sf.descent())) / 2.0 + sf.ascent();
        let left = (w as f32 - lw) / 2.0;
        for g in glyphs {
            let glyph = ab_glyph::GlyphId(g.id).with_scale_and_position(scale, ab_glyph::point(left + g.x, base + g.y));
            if let Some(o) = f.outline_glyph(glyph) {
                let b = o.px_bounds();
                o.draw(|gx, gy, c| {
                    let (x, y) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
                    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 { return; }
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    // the white type over the box: blend by coverage, keeping the more opaque alpha
                    let c = c.clamp(0.0, 1.0);
                    for i in 0..3 { p[i] = (fg[i] as f32 * c + p[i] as f32 * (1.0 - c)).round() as u8; }
                    p[3] = p[3].max((255.0 * c).round() as u8);
                });
            }
        }
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("could not write the caption picture: {e}"))?;
    Ok((png, w, h))
}

/// One line drawn straight onto a picture (a thumbnail): shaped like a caption, centred on `cx` with its baseline at
/// `baseline`, in `rgb`, with a dark outline `outline` px wide so it reads on any picture. Returns its width.
pub fn draw_line(img: &mut image::RgbaImage, text: &str, px: f32, rgb: [u8; 3], cx: f32, baseline: f32, outline: i32) -> Result<f32, String> {
    use ab_glyph::Font;
    let (script, glyphs, lw) = shape(text, px)?;
    let f = ab_glyph::FontRef::try_from_slice(font(script)?).map_err(|e| format!("the font could not be read: {e}"))?;
    let em = f.units_per_em().unwrap_or(1000.0);
    let scale = px * f.height_unscaled() / em;
    let (w, h) = (img.width() as i32, img.height() as i32);
    let left = cx - lw / 2.0;
    // coverage of the whole line first, then the outline (the coverage spread by `outline` px) and the type over it
    let mut cov = vec![0f32; (w * h) as usize];
    for g in &glyphs {
        let glyph = ab_glyph::GlyphId(g.id).with_scale_and_position(scale, ab_glyph::point(left + g.x, baseline + g.y));
        if let Some(o) = f.outline_glyph(glyph) {
            let b = o.px_bounds();
            o.draw(|gx, gy, c| {
                let (x, y) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
                if x >= 0 && y >= 0 && x < w && y < h { let k = (y * w + x) as usize; cov[k] = cov[k].max(c); }
            });
        }
    }
    if outline > 0 {
        let r = outline;
        let mut halo = vec![0f32; cov.len()];
        for y in 0..h { for x in 0..w {
            let c = cov[(y * w + x) as usize];
            if c <= 0.0 { continue; }
            for dy in -r..=r { for dx in -r..=r {
                if dx * dx + dy * dy > r * r { continue; }
                let (xx, yy) = (x + dx, y + dy + r / 2); // a little lower: the shadow's drop
                if xx >= 0 && yy >= 0 && xx < w && yy < h { let k = (yy * w + xx) as usize; halo[k] = halo[k].max(c); }
            }}
        }}
        for (k, a) in halo.iter().enumerate() {
            if *a <= 0.0 { continue; }
            let p = img.get_pixel_mut((k as i32 % w) as u32, (k as i32 / w) as u32);
            let a = a * 0.85;
            for i in 0..3 { p[i] = (p[i] as f32 * (1.0 - a)).round() as u8; }
        }
    }
    for (k, c) in cov.iter().enumerate() {
        if *c <= 0.0 { continue; }
        let p = img.get_pixel_mut((k as i32 % w) as u32, (k as i32 / w) as u32);
        for i in 0..3 { p[i] = (rgb[i] as f32 * c + p[i] as f32 * (1.0 - c)).round() as u8; }
    }
    Ok(lw)
}

/// The size at which `text` fits `max_w` pixels, no bigger than `px`.
pub fn fit_px(text: &str, px: f32, max_w: f32) -> Result<f32, String> {
    let w = shape(text, px)?.2;
    Ok(if w > max_w { px * max_w / w } else { px })
}

/// One overlay: a box picture shown from `a` to `b` seconds at (x, y).
pub struct Overlay { pub png: Vec<u8>, pub a: f64, pub b: f64, pub x: u32, pub y: u32 }

/// The captions and cards of a video as overlays, placed as the libass style placed them: captions centred 120 px
/// above the bottom (at a 1080 short side; scaled for other sizes), the title from 0.5 s to 9 s and the end card over the last 10 s, both in the middle.
pub fn overlays(w: u32, h: u32, captions: &[(f64, f64, String)], open: Option<&str>, end: Option<&str>, total: f64)
                -> Result<Vec<Overlay>, String> {
    let mut out = Vec::new();
    // the sizes are for a frame whose short side is 1080 (1920x1080 and 1080x1920 alike); others scale to it
    let k = (w.min(h) as f32 / 1080.0).max(0.1);
    let scaled = |st: Style| Style { px: st.px * k, pad: st.pad * k, box_alpha: st.box_alpha };
    let margin = (80.0 * k) as u32;
    let bottom = (120.0 * k) as u32;
    for (a, b, text) in captions {
        if *b <= *a || text.trim().is_empty() { continue; }
        let (png, bw, bh) = render(text, scaled(CAPTION), w.saturating_sub(2 * margin))?;
        out.push(Overlay { png, a: *a, b: (*b).min(total), x: (w - bw) / 2, y: h.saturating_sub(bottom + bh) });
    }
    let mut card = |t: &str, a: f64, b: f64| -> Result<(), String> {
        if t.trim().is_empty() || b <= a { return Ok(()); }
        let (png, bw, bh) = render(t, scaled(CARD), w.saturating_sub(2 * margin))?;
        out.push(Overlay { png, a, b, x: (w - bw) / 2, y: (h.saturating_sub(bh)) / 2 });
        Ok(())
    };
    if let Some(t) = open { card(t, 0.5, 9.0_f64.min(total))?; }
    if let Some(t) = end { card(t, (total - 10.0).max(0.0), total)?; }
    Ok(out)
}

/// The filtergraph that lays the overlays (inputs `first`, `first+1`, … each a looped picture of its own length) on
/// `[0:v]`, each from its start with a quarter-second fade in and out, into `[v]`; and the audio part as in
/// compose::final_filter. With no overlays the pictures pass through.
pub fn overlay_filter(ovs: &[Overlay], first: usize, played: f64, audio_from: f64) -> String {
    let mut f = String::new();
    let mut last = "[0:v]".to_string();
    for (k, o) in ovs.iter().enumerate() {
        let d = (o.b - o.a).max(0.3);
        let fade = 0.25f64.min(d / 3.0);
        f.push_str(&format!("[{}:v]format=rgba,fade=t=in:st=0:d={fade:.3}:alpha=1,fade=t=out:st={:.3}:d={fade:.3}:alpha=1,setpts=PTS+{:.3}/TB[c{k}];",
                            first + k, d - fade, o.a));
        let out = format!("[o{k}]");
        f.push_str(&format!("{last}[c{k}]overlay=x={}:y={}:eof_action=pass:enable='between(t,{:.3},{:.3})'{out};", o.x, o.y, o.a, o.b));
        last = out;
    }
    f.push_str(&format!("{last}format=yuv420p[v];[1:a]atrim=start={:.3},asetpts=N/SR/TB,afade=t=out:st={:.3}:d=4[a]",
                        audio_from.max(0.0), played.max(4.0) - 4.0));
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_line_is_drawn_in_its_scripts_font() {
        assert_eq!(script_of("Peace in our hands"), Script::Latin);
        assert_eq!(script_of("नमस्ते अस्तु भगवन्"), Script::Devanagari);
        assert_eq!(script_of("സമാധാനം"), Script::Malayalam);
    }

    #[test]
    fn a_caption_is_a_box_with_white_type_in_it() {
        let (png, w, h) = render("Peace in our hands, harmony", CAPTION, 1760).unwrap();
        let img = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!((img.width(), img.height()), (w, h));
        assert!(w > 300 && w < 1760 && h > 58, "{w}x{h}");
        // the box is dark and see-through at its corner, and some pixels are the white type
        let c = img.get_pixel(1, 1);
        assert!(c[3] > 100 && c[3] < 255 && c[0] < 40, "{c:?}");
        assert!(img.pixels().filter(|p| p[0] > 230 && p[3] > 200).count() > 500, "type was drawn");
    }

    #[test]
    fn a_long_line_wraps_and_a_devanagari_line_is_shaped() {
        let lines = wrap("Lighting up the quiet woods long after day is done and the stars come out", 58.0, 700.0).unwrap();
        assert!(lines.len() >= 2, "{lines:?}");
        // a conjunct (स्त) shapes to fewer glyphs than it has code points
        let (_, glyphs, width) = shape("अस्तु", 58.0).unwrap();
        assert!(glyphs.len() < "अस्तु".chars().count() && width > 20.0, "{} glyphs", glyphs.len());
        let (png, _, _) = render("नमस्ते अस्तु भगवन् विश्वेश्वराय महादेवाय", CAPTION, 1760).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn captions_sit_low_and_cards_in_the_middle() {
        let caps = vec![(1.0, 3.0, "hello".to_string())];
        let o = overlays(1920, 1080, &caps, Some("Title"), Some("The end"), 60.0).unwrap();
        assert_eq!(o.len(), 3);
        assert!(o[0].y > 800, "caption low: {}", o[0].y);
        assert!(o[1].y > 300 && o[1].y < 600 && o[1].a == 0.5 && o[1].b == 9.0, "title in the middle");
        assert_eq!((o[2].a, o[2].b), (50.0, 60.0), "end card over the last 10 s");
        let f = overlay_filter(&o, 2, 60.0, 0.0);
        assert!(f.contains("[2:v]format=rgba") && f.contains("[4:v]") && f.contains("[o2]format=yuv420p[v]"), "{f}");
        assert!(overlay_filter(&[], 2, 60.0, 0.0).starts_with("[0:v]format=yuv420p[v];"));
    }
}

#[cfg(test)]
mod look {
    /// Writes sample caption pictures for a person to look at: CAPTION_LOOK=<folder> cargo test look -- --ignored
    #[test]
    #[ignore = "writes pictures to look at"]
    fn write_samples() {
        let Some(dir) = std::env::var_os("CAPTION_LOOK") else { return };
        let dir = std::path::PathBuf::from(dir);
        for (name, text, st) in [("deva", "नमस्ते अस्तु भगवन् विश्वेश्वराय महादेवाय त्र्यम्बकाय", super::CAPTION),
                                 ("latin", "Fireflies, fireflies, blinking all as one", super::CAPTION),
                                 ("card", "Peace and Harmony\nacoustic evening version", super::CARD),
                                 ("mlym", "സമാധാനവും ഐക്യവും", super::CAPTION)] {
            std::fs::write(dir.join(format!("{name}.png")), super::render(text, st, 1760).unwrap().0).unwrap();
        }
    }
}
