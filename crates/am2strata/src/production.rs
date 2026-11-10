//! Productions (docs/LONG-MEDIA-PLAN.md): one request, many parts, one file. `POST /v1/media/productions`
//! {"name", "parts": [{"speak": {...}}, {"song": {...}}, {"music": {...}}], "crossfade_s"} queues one media job; each
//! part is made by its kind's own endpoint (the same checks and takes), kept on disk as it is finished (a restart goes
//! on from the last finished part), and the parts are joined here: resampled to 48 kHz stereo, their loudness evened
//! out, short crossfades between them. The job's file is one WAV; its meta holds the chapters (a part each).

use serde_json::{json, Value};

pub const RATE: u32 = 48_000;
/// The loudness each part is brought to (RMS, dBFS), before the peak limit.
const TARGET_DB: f64 = -18.0;
const PEAK: f32 = 0.98;
pub const MAX_PARTS: usize = 40;

/// A part's kind and its request to the kind's endpoint.
pub fn parts_of(plan: &Value) -> Result<Vec<(&'static str, Value)>, String> {
    let parts = plan["parts"].as_array().ok_or("parts: a list of {\"speak\"|\"song\"|\"music\": {...}}")?;
    if parts.is_empty() || parts.len() > MAX_PARTS {
        return Err(format!("parts: 1 to {MAX_PARTS} of them"));
    }
    parts.iter().enumerate().map(|(i, p)| {
        let (kind, body) = if let Some(b) = p.get("speak") { ("speak", b) }
            else if let Some(b) = p.get("song") { ("song", b) }
            else if let Some(b) = p.get("music") { ("music", b) }
            else { return Err(format!("parts[{i}]: one of speak, song, music")) };
        if !body.is_object() {
            return Err(format!("parts[{i}].{kind}: an object"));
        }
        let mut b = body.clone();
        match kind {
            "speak" if b["text"].as_str().or(b["input"].as_str()).is_none_or(|t| t.trim().is_empty()) =>
                return Err(format!("parts[{i}].speak: give the text")),
            "speak" => {
                if b.get("input").is_none() {
                    b["input"] = b["text"].clone();
                }
            }
            "song" if b["lyrics"].as_str().is_none_or(|t| t.trim().is_empty()) => return Err(format!("parts[{i}].song: give the lyrics")),
            "music" => {
                b["lyrics"] = json!("[Instrumental]");
            }
            _ => {}
        }
        if matches!(kind, "song" | "music") {
            let style = crate::music::style_for(b["genre"].as_str(), b["style"].as_str().or(if kind == "song" { Some("clear vocals") } else { None }))
                .map_err(|e| format!("parts[{i}]: {e}"))?;
            b["style"] = json!(style);
            let d = b["duration"].as_u64().unwrap_or(if kind == "song" { 120 } else { 60 });
            if !(10..=600).contains(&d) {
                return Err(format!("parts[{i}].{kind}.duration: 10 to 600 seconds (a longer song is several parts)"));
            }
            b["duration"] = json!(d);
            if kind == "song" && b.get("takes").is_none() {
                b["takes"] = json!(2);
            }
        }
        Ok((kind, b))
    }).collect()
}

/// A part's title for the chapters.
pub fn title(kind: &str, body: &Value, i: usize) -> String {
    let first = |s: &str| s.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('[')).unwrap_or("").chars().take(48).collect::<String>();
    match kind {
        "speak" => format!("{}. {}", i + 1, first(body["input"].as_str().unwrap_or(""))),
        "song" => format!("{}. Song: {}", i + 1, first(body["lyrics"].as_str().unwrap_or(""))),
        _ => format!("{}. Music: {}", i + 1, body["genre"].as_str().or(body["style"].as_str()).unwrap_or("instrumental")),
    }
}

/// PCM samples of a WAV (16-bit integer or 32-bit float), as (rate, channels, interleaved f32).
pub fn read_wav(b: &[u8]) -> Result<(u32, u16, Vec<f32>), String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut fmt, mut data) = (None, None);
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        match id {
            b"fmt " if body.len() >= 16 => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        i += 8 + len + (len & 1);
    }
    let fmt = fmt.ok_or("a WAV without its format")?;
    let data = data.ok_or("a WAV without its samples")?;
    let format = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]).max(1);
    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    let samples: Vec<f32> = match (format, bits) {
        (1, 16) => data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        (3, 32) => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        (0xFFFE, 16) => data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        _ => return Err(format!("a WAV of format {format} with {bits}-bit samples (16-bit PCM or 32-bit float only)")),
    };
    Ok((rate, channels, samples))
}

/// Interleaved samples to 48 kHz stereo [L, R, L, R...] (linear resampling: speech and songs are already band-limited).
pub fn to_stereo_48k(rate: u32, channels: u16, s: &[f32]) -> Vec<f32> {
    let ch = channels as usize;
    let frames = s.len() / ch;
    let at = |f: usize, c: usize| s[f * ch + c.min(ch - 1)];
    let n_out = ((frames as u64 * RATE as u64) / rate.max(1) as u64) as usize;
    let mut out = Vec::with_capacity(n_out * 2);
    for o in 0..n_out {
        let pos = o as f64 * rate as f64 / RATE as f64;
        let (f, frac) = (pos.floor() as usize, (pos - pos.floor()) as f32);
        for c in 0..2 {
            let a = at(f.min(frames - 1), c);
            let b = at((f + 1).min(frames - 1), c);
            out.push(a + (b - a) * frac);
        }
    }
    out
}

/// The part brought to TARGET_DB of loudness (RMS), never past the peak limit.
pub fn even_loudness(s: &mut [f32]) {
    if s.is_empty() {
        return;
    }
    let rms = (s.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / s.len() as f64).sqrt();
    if rms < 1e-5 {
        return;
    }
    let peak = s.iter().fold(0f32, |m, x| m.max(x.abs()));
    let want = 10f64.powf(TARGET_DB / 20.0) / rms;
    let gain = (want as f32).min(PEAK / peak.max(1e-6));
    for x in s.iter_mut() {
        *x *= gain;
    }
}

/// The parts (48 kHz stereo) joined with equal-power crossfades of `cross_s`; the start of each part, in seconds.
pub fn join(parts: &[Vec<f32>], cross_s: f64) -> (Vec<f32>, Vec<f64>) {
    let mut out: Vec<f32> = Vec::new();
    let mut starts = Vec::new();
    for p in parts {
        let frames_out = out.len() / 2;
        let cross = ((cross_s * RATE as f64) as usize).min(frames_out).min(p.len() / 2 / 2);
        let start = frames_out - cross;
        starts.push(start as f64 / RATE as f64);
        for f in 0..cross {
            let t = (f as f32 + 0.5) / cross as f32;
            let (a, b) = ((std::f32::consts::FRAC_PI_2 * t).cos(), (std::f32::consts::FRAC_PI_2 * t).sin());
            for c in 0..2 {
                let i = (start + f) * 2 + c;
                out[i] = out[i] * a + p[f * 2 + c] * b;
            }
        }
        out.extend_from_slice(&p[cross * 2..]);
    }
    (out, starts)
}

/// 16-bit stereo WAV at 48 kHz.
pub fn wav(s: &[f32]) -> Vec<u8> {
    let data = (s.len() * 2) as u32;
    let mut o = Vec::with_capacity(44 + s.len() * 2);
    o.extend_from_slice(b"RIFF");
    o.extend_from_slice(&(36 + data).to_le_bytes());
    o.extend_from_slice(b"WAVEfmt ");
    o.extend_from_slice(&16u32.to_le_bytes());
    o.extend_from_slice(&1u16.to_le_bytes());
    o.extend_from_slice(&2u16.to_le_bytes());
    o.extend_from_slice(&RATE.to_le_bytes());
    o.extend_from_slice(&(RATE * 4).to_le_bytes());
    o.extend_from_slice(&4u16.to_le_bytes());
    o.extend_from_slice(&16u16.to_le_bytes());
    o.extend_from_slice(b"data");
    o.extend_from_slice(&data.to_le_bytes());
    for x in s {
        o.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, ch: u16, secs: f64, amp: f32) -> Vec<u8> {
        let n = (rate as f64 * secs) as usize;
        let s: Vec<f32> = (0..n).flat_map(|i| std::iter::repeat_n(amp * (i as f32 * 0.05).sin(), ch as usize)).collect();
        let mut w = wav(&[]);
        // rewrite the header for this rate and channel count
        w.clear();
        let data = (s.len() * 2) as u32;
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&ch.to_le_bytes());
        w.extend_from_slice(&rate.to_le_bytes());
        w.extend_from_slice(&(rate * ch as u32 * 2).to_le_bytes());
        w.extend_from_slice(&(ch * 2).to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data.to_le_bytes());
        for x in &s {
            w.extend_from_slice(&((x * 32767.0) as i16).to_le_bytes());
        }
        w
    }

    #[test]
    fn parts_are_read_evened_and_joined_with_their_chapters() {
        let speech = tone(24_000, 1, 2.0, 0.05);
        let song = tone(48_000, 2, 3.0, 0.6);
        let (r1, c1, s1) = read_wav(&speech).unwrap();
        let (r2, c2, s2) = read_wav(&song).unwrap();
        assert_eq!((r1, c1, r2, c2), (24_000, 1, 48_000, 2));
        let mut a = to_stereo_48k(r1, c1, &s1);
        let mut b = to_stereo_48k(r2, c2, &s2);
        assert_eq!(a.len() / 2, 96_000, "2 s at 48 kHz");
        even_loudness(&mut a);
        even_loudness(&mut b);
        let rms = |x: &[f32]| (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        assert!((20.0 * rms(&a).log10() - TARGET_DB).abs() < 0.5 && (20.0 * rms(&b).log10() - TARGET_DB).abs() < 0.5, "loudness evened");
        let (j, starts) = join(&[a, b], 0.5);
        assert_eq!(starts.len(), 2);
        assert!((starts[1] - 1.5).abs() < 0.01, "the second part starts half a second before the first ends: {starts:?}");
        assert!(((j.len() / 2) as f64 / RATE as f64 - 4.5).abs() < 0.01, "2 + 3 - 0.5 s");
        let out = wav(&j);
        let (r, c, back) = read_wav(&out).unwrap();
        assert_eq!((r, c, back.len()), (48_000, 2, j.len()));
        assert!(parts_of(&json!({"parts": [{"song": {"lyrics": "la", "duration": 900}}]})).is_err(), "a part over 10 minutes");
        assert!(parts_of(&json!({"parts": [{"dance": {}}]})).is_err());
        let p = parts_of(&json!({"parts": [{"speak": {"text": "Welcome."}}, {"music": {"genre": "lofi"}}]})).unwrap();
        assert_eq!((p[0].0, p[1].0, p[1].1["lyrics"].as_str()), ("speak", "music", Some("[Instrumental]")));
    }
}
