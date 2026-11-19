//! Music facts of a recording — tempo, the drone's key and home note (the tonic), the notes the voice sits on (the swaras),
//! how long each phrase holds. The producer has to learn a reference chant's style before making a take that matches its pace
//! and key. This is all pure math over mono samples (std only, no model, no FFT): one-pole filters for the voice band, onset
//! strength over it for the tempo, its envelope's autocorrelation for the repeat period, YIN-style autocorrelation for the
//! pitch (fundamental, not a harmonic) and the spectrum's pitch-class histogram for the tonic. Ported from the reviewer's
//! Python in `.agent/ref/`, then fixed on real music: a constant drone below the voice band no longer hides the phrases or
//! throws the tonic onto an overtone, and the repeat period is the verse, not two of them.

use serde_json::{json, Value};

/// A 10 ms analysis frame (capgaps.py's FS), for the tempo onsets and the pitch track.
const FRAME_S: f64 = 0.01;
/// The voice-band envelope is taken at this resolution (50 ms) after removing the drone below it.
const ENV_S: f64 = 0.05;
/// The drone's band is removed below these hertz so the voice, above it, is seen on its own.
const BAND_LO_HZ: f64 = 200.0;
/// The voice-band envelope's high end (removes the higher harmonics of the drone and noise).
const BAND_HI_HZ: f64 = 3000.0;
/// A phrase ends when the voice above the drone falls under this fraction of its median level for long enough.
const PHRASE_FRAC: f64 = 0.25;
/// Two phrases are separate when the voice envelope stays that quiet for at least this many seconds (capgaps.py's GAP).
const MIN_PAUSE_S: f64 = 0.3;
/// A fragment (shorter than MIN_PHRASE_S) is joined to its neighbour when the silence between them is under this many
/// seconds; two full phrases stay separate whatever the gap.
const PHRASE_JOIN_S: f64 = 0.5;
/// A phrase shorter than this many seconds is dropped as a glip, not a phrase (the reference gave many under 1 s).
const MIN_PHRASE_S: f64 = 0.8;
/// A tempo is trusted only when its confidence reaches this; under it a reader should treat it as an unsteady pulse.
const STEADY_CONF: f64 = 0.30;
/// The tempo is searched between these beats per minute.
const BPM_MIN: f64 = 40.0;
const BPM_MAX: f64 = 200.0;
/// The repeat period (one verse of a mantra) is reported only when no shorter than this also repeats strongly.
const REPEAT_SKIP_S: f64 = 8.0;
/// The repeat period (one verse of a mantra) is searched between these seconds.
const REPEAT_MIN_S: f64 = 4.0;
const REPEAT_MAX_S: f64 = 60.0;
/// A repeat's shorter multiple is reported only when its correlation reaches this fraction of the best.
const REPEAT_HALF_FAC: f64 = 0.80;
/// The pitch is searched between these hertz (a chant's fundamental, not an overtone of the drone).
const PITCH_MIN_HZ: f64 = 70.0;
const PITCH_MAX_HZ: f64 = 500.0;
/// A frame is voiced when its autocorrelation's best peak reaches this fraction of its energy.
const VOICE_FLOOR: f64 = 0.55;
/// A section longer than this is analyzed unless a request asks for less (a cap on the default "to the end").
const MAX_SECTION_S: f64 = 600.0;
/// The pitch track is computed on a copy resampled no higher than this hertz (alias-free by first low-passing, and fast). The
/// period of the lowest fundamental still resolves across several samples here, so sub-sample parabolic interpolation pins the
/// frequency without aliasing its higher harmonics.
const PITCH_SR_HZ: f64 = 4000.0;

/// The twelve swaras above the tonic, in semitone order (Sa and its half/sharp variants as the producer expects them).
const SWARA_NAMES: &[&str] = &["Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni"];
/// Note names from a pitch class (0 == C).
const NOTE_NAMES: &[&str] = &["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

fn r2(x: f64) -> f64 { (x * 100.0).round() / 100.0 }
fn r3(x: f64) -> f64 { (x * 1000.0).round() / 1000.0 }

/// An empty JSON list, for a result field that has no entries (no voiced frames, no phrases).
fn empty_list() -> Value { Value::Array(vec![]) }

/// MIDI number of a frequency in hertz (A4 = 440).
fn midi_of(freq: f64) -> f64 { 69.0 + 12.0 * (freq / 440.0).log2() }
/// A MIDI number's frequency in hertz (A4 = 440): inverse of midi_of, so a quantized note reads back exactly.
fn freq_of_midi(midi: f64) -> f64 { 440.0 * 2.0_f64.powf((midi - 69.0) / 12.0) }
/// A frequency's pitch class as an index into NOTE_NAMES / SWARA_NAMES (0-11), robust to negative remainders.
fn pitch_class(freq: f64) -> usize { ((midi_of(freq).round() as i64 % 12 + 12) % 12) as usize }
/// A frequency's semitone above `tonic_hz`, wrapped to 0-11 (works on the raw MIDI distance so no integer types mix).
fn semitone_above(freq: f64, tonic_hz: f64) -> usize {
    (((midi_of(freq) - midi_of(tonic_hz)).round() % 12.0 + 12.0) % 12.0) as usize
}

/// A note name like "C#3" from a frequency.
fn note_name(freq: f64) -> String {
    let pc = pitch_class(freq);
    let midi = midi_of(freq).round() as i64;
    format!("{}{}", NOTE_NAMES[pc], midi / 12 - 1)
}

/// The mean of `x`, removed, and the vector itself: used before an autocorrelation so a rising or falling trend does not
/// show up as a long-period peak.
fn detrend(x: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    x.iter().map(|v| *v - mean).collect()
}

/// A one-pole low-pass of `x` at the given coefficient (a = exp(-2*pi*f_cutoff/rate); near 1 a soft, low cut, near 0 sharp).
fn lowpass(x: &[f32], a: f32) -> Vec<f32> {
    let mut y = Vec::with_capacity(x.len());
    let mut prev = 0.0f32;
    for &v in x {
        prev = a * prev + (1.0 - a) * v;
        y.push(prev);
    }
    y
}

/// A one-pole high-pass (the complement of lowpass): keeps everything above its cutoff, removes the drone's DC-ish energy.
fn highpass(x: &[f32], a: f32) -> Vec<f32> {
    let mut y = Vec::with_capacity(x.len());
    let mut prev_y = 0.0f32;
    let mut prev_x = 0.0f32;
    for &v in x {
        prev_y = a * prev_y + (v - prev_x);
        prev_x = v;
        y.push(prev_y);
    }
    y
}

/// The band between `flo` and `hi` hertz: high-pass at `flo`, then low-pass at `hi`. A drone under the voice is removed.
fn bandpass(x: &[f32], rate: f64, flo: f64, hi: f64) -> Vec<f32> {
    let a_hp = (-std::f64::consts::TAU * flo / rate).exp() as f32;
    let lp_a = (-std::f64::consts::TAU * hi / rate).exp() as f32;
    lowpass(&highpass(x, a_hp), lp_a)
}

/// A copy of the signal at ~`max_rate` hertz, aliased away by a low-pass first (simple stride resample): below this a frame's
/// pitch is found fast and its higher harmonics — which would pull it onto an overtone — are gone. Returns the buffer and its
/// effective rate.
fn resample_to_max(x: &[f32], rate: f64, max_rate: f64) -> (Vec<f32>, f64) {
    if rate <= max_rate { return (x.to_vec(), rate); }
    let a = (-std::f64::consts::TAU * (max_rate / 2.0) / rate).exp() as f32;
    let filtered = lowpass(x, a);
    let stride = ((rate / max_rate) as f64).round() as usize;
    let dec: Vec<f32> = filtered.iter().step_by(stride).copied().collect();
    (dec, rate / stride as f64)
}

/// The lag (in frames) of the strongest value and its height, over `min_lag..=max_lag` of a detrended signal's normalized
/// autocorrelation. Returns (None, 0) when the window is empty or flat, so a caller can tell "no structure" from "none here".
fn acf_best(x: &[f32], min_lag: usize, max_lag: usize) -> (Option<usize>, f64) {
    let n = x.len();
    if n < 2 || max_lag < min_lag { return (None, 0.0); }
    let xc = detrend(x);
    let acf0: f64 = xc.iter().map(|v| *v as f64 * *v as f64).sum();
    if acf0 == 0.0 { return (None, 0.0); }
    let hi = max_lag.min(n - 1);
    let mut best = -2.0f64;
    let mut bd = min_lag;
    for d in min_lag..=hi {
        let mut acc = 0.0f64;
        for i in 0..(n - d) { acc += xc[i] as f64 * xc[i + d] as f64; }
        let v = acc / acf0;
        if v > best { best = v; bd = d; }
    }
    (Some(bd), best)
}

/// The normalized autocorrelation at every lag in `min_lag..=max_lag` (index 0 == min_lag): used to compare a period with its
/// half or third, which acf_best alone cannot do.
fn acf_vals(x: &[f32], min_lag: usize, max_lag: usize) -> Option<Vec<f64>> {
    let n = x.len();
    if n < 2 || max_lag < min_lag { return None; }
    let xc = detrend(x);
    let acf0: f64 = xc.iter().map(|v| *v as f64 * *v as f64).sum();
    if acf0 == 0.0 { return None; }
    let hi = max_lag.min(n - 1);
    let mut vals = Vec::with_capacity(hi - min_lag + 1);
    for d in min_lag..=hi {
        let mut acc = 0.0f64;
        for i in 0..(n - d) { acc += xc[i] as f64 * xc[i + d] as f64; }
        vals.push(acc / acf0);
    }
    Some(vals)
}

/// The RMS in dBFS of `x` (a signal at full scale is 0 dBFS; silence is -inf, reported as -160).
fn rms_dbfs(x: &[f32]) -> f64 {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    if rms <= 0.0 { -160.0 } else { (20.0 * rms.log10()) as f64 }
}

/// The p-th percentile (0-1) of an f32 vector, sorted on entry.
fn percentile_f32(sorted: &mut [f32], p: f64) -> f32 {
    sort_f32(sorted);
    if sorted.is_empty() { return 0.0; }
    if sorted.len() == 1 { return sorted[0]; }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi { return sorted[lo]; }
    let weight = rank - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * weight as f32
}

/// The p-th percentile (0-1) of an f64 vector, sorted on entry.
fn percentile_f64(sorted: &mut [f64], p: f64) -> f64 {
    sort_f64(sorted);
    if sorted.is_empty() { return 0.0; }
    if sorted.len() == 1 { return sorted[0]; }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi { return sorted[lo]; }
    let weight = rank - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * weight
}

fn sort_f32(v: &mut [f32]) { v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); }
fn sort_f64(v: &mut [f64]) { v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); }

/// Loudness: the section's overall dBFS, and each one-second frame's in a list.
fn loudness_obj(section: &[f32], rate: f64) -> Value {
    let hop = rate.max(1.0) as usize; // one second per frame
    if hop == 0 || section.is_empty() { return json!({"dbfs": -160.0, "per_second": empty_list()}); }
    let n = section.len() / hop;
    let overall = rms_dbfs(section);
    let mut per_second = Vec::with_capacity(n);
    for i in 0..n {
        per_second.push(r2(rms_dbfs(&section[i * hop..((i + 1) * hop).min(section.len())])));
    }
    json!({"dbfs": r2(overall), "per_second": per_second})
}

/// The voice band (200-3000 Hz, the drone gone) and its envelope at ENV_S frames: the RMS of each window minus the floor
/// (its 10th percentile, the drone's steady level). Where this stays quiet the voice is not singing. Returns the band, the
/// per-frame RMS values, the floor, and the frame length in samples.
fn voice_env(section: &[f32], rate: f64) -> (Vec<f32>, Vec<f32>, f32, usize) {
    let vb = bandpass(section, rate, BAND_LO_HZ, BAND_HI_HZ);
    let hop50 = (rate * ENV_S).round() as usize;
    if hop50 == 0 || vb.is_empty() { return (vb, vec![], 0.0, 0); }
    let n = vb.len() / hop50;
    let mut rms = Vec::with_capacity(n);
    for i in 0..n {
        let w = &vb[i * hop50..((i + 1) * hop50).min(vb.len())];
        rms.push((w.iter().map(|v| v * v).sum::<f32>() / w.len().max(1) as f32).sqrt());
    }
    let mut sorted = rms.clone();
    sort_f32(&mut sorted);
    let floor = percentile_f32(&mut sorted, 0.10);
    (vb, rms, floor, hop50)
}

/// Tempo from the voice band's onsets: the positive jumps of each 10 ms frame's energy over it, then its autocorrelation over
/// 40-200 bpm. A confidence under STEADY_CONF means the beat is unsteady and should not be trusted.
fn tempo_obj(section: &[f32], rate: f64) -> Value {
    let hop = (rate * FRAME_S).round().max(1.0) as usize;
    if hop == 0 || section.len() < hop * 8 { return json!({"bpm": 0.0, "confidence": 0.0, "steady_pulse": true}); }
    let n = section.len() / hop;
    let mut energy = Vec::with_capacity(n);
    for i in 0..n {
        let e: f32 = section[i * hop..((i + 1) * hop).min(section.len())].iter().map(|v| v * v).sum();
        energy.push(e.max(0.0));
    }
    // onset strength: the positive jumps of energy (a syllable makes it rise)
    let mut onset = Vec::with_capacity(n);
    let mut prev = 0.0f32;
    for &e in &energy { onset.push((e - prev).max(0.0)); prev = e; }
    let onset: Vec<f32> = onset.into_iter().collect();
    let fs = 1.0 / FRAME_S;
    let min_lag = (60.0 / BPM_MAX * fs).round() as usize; // period of the fastest searched bpm (fewest frames)
    let max_lag = (60.0 / BPM_MIN * fs).round() as usize; // period of the slowest searched bpm
    if let Some(lag) = acf_best(&onset, min_lag.max(1), max_lag).0 {
        let period = lag as f64 * FRAME_S;
        let bpm: f64 = if period > 0.0 { 60.0 / period } else { 0.0 };
        // confidence relative to the window's mean (1 when everything else is flat)
        if let Some(vals) = acf_vals(&onset, min_lag.max(1), max_lag) {
            let peak = vals[lag - min_lag.max(1)];
            let wmean = vals.iter().sum::<f64>() / vals.len() as f64;
            let conf = ((peak - wmean) / (1.0 - wmean)).clamp(0.0, 1.0);
            return json!({"bpm": r2(bpm.min(BPM_MAX).max(BPM_MIN)), "confidence": r2(conf),
                          "steady_pulse": conf >= STEADY_CONF});
        }
    }
    json!({"bpm": 0.0, "confidence": 0.0, "steady_pulse": true})
}

/// Repeat period (one verse of a mantra) from the voice envelope's autocorrelation over 4-60 s. When a period's half (or
/// third) also repeats strongly, that shorter one is reported instead; nothing under REPEAT_SKIP_S is reported unless no
/// longer period is found. `repeats_in_section` says how many verses that makes of the section.
fn repeat_obj(rms: &[f32], hop50: usize, section_s: f64) -> Value {
    if rms.len() < 8 || hop50 == 0 { return json!({"period_s": 0.0, "confidence": 0.0, "repeats_in_section": 0.0}); }
    let env: Vec<f32> = rms.to_vec(); // envelope already has the drone's floor removed
    let max_lag = ((REPEAT_MAX_S / ENV_S).round() as usize).min(rms.len() - 1);
    // a lag in frames is a period of ENV_S seconds. The verse is >= REPEAT_SKIP_S, so search there first; if nothing repeats at
    // that scale (a fast mantra whose verse is shorter) fall back down to REPEAT_MIN_S.
    let skip_lag: usize = (REPEAT_SKIP_S / ENV_S).round() as usize;
    let mut lag: Option<usize> = acf_best(&env, skip_lag.max(1), max_lag).0;
    if lag.is_none() { lag = acf_best(&env, ((REPEAT_MIN_S / ENV_S).round() as usize).max(1), max_lag).0; }
    let mut peak = match acf_vals(&env, 1, max_lag) { Some(v) => lag.map_or(0.0, |l| v[l.saturating_sub(1).min(v.len() - 1)]), None => 0.0 };
    // a shorter multiple that repeats at least REPEAT_HALF_FAC as strongly is the verse: halve or third it, as far as it holds
    // (so two verses are not reported where one is, and a sub-8 s verse still comes through)
    while let Some(l) = lag {
        if let Some(vals) = acf_vals(&env, 1, max_lag) {
            let mut took = false;
            for k in [2usize, 3usize] {
                if l % k == 0 {
                    let d = l / k;
                    if (1..=max_lag).contains(&d) {
                        let v = vals[d.saturating_sub(1).min(vals.len() - 1)];
                        if v >= REPEAT_HALF_FAC * peak { lag = Some(d); peak = v; took = true; break; }
                    }
                }
            }
            if !took { break; }
        } else { break; }
    }
    if let Some(lag) = lag {
        let period = lag as f64 * ENV_S;
        let repeats = if period > 0.0 { section_s / period } else { 0.0 };
        return json!({"period_s": r2(period), "confidence": r2(peak.max(0.0)), "repeats_in_section": r2(repeats)});
    }
    json!({"period_s": 0.0, "confidence": 0.0, "repeats_in_section": 0.0})
}

/// The fundamental of each 10 ms frame by a YIN-style autocorrelation over PITCH_MIN_HZ..=PITCH_MAX_HZ (a drone's home note is
/// the fundamental, not an overtone). Of the peaks it keeps the strongest one: where every harmonic lines up is the period
/// itself, so this returns the true octave no matter which partial is loudest, and a neighbour side-lobe cannot read it high.
fn fundamental_frames(x: &[f32], rate: f64) -> Vec<(Option<f64>, f32)> {
    let hop = (rate * FRAME_S).round().max(1.0) as usize;
    let win = ((rate * 0.04).round() as usize).max(((rate / PITCH_MIN_HZ) * 2.0).round() as usize);
    if hop == 0 || x.len() < hop * 2 { return vec![(None, 0.0)]; }
    let dmin = (rate / PITCH_MAX_HZ).ceil() as usize;
    let dmax = ((rate / PITCH_MIN_HZ).round() as usize).min(win.saturating_sub(1));
    if dmax <= dmin { return vec![(None, 0.0); x.len() / hop]; }
    (0..x.len() / hop)
        .enumerate()
        .map(|(fi, _)| {
            let a = fi * hop;
            let b = (a + win).min(x.len());
            let w = &x[a..b];
            let len = w.len() as f64;
            if len <= dmax as f64 { return (None, 0.0); }
            let acf0: f64 = w.iter().map(|v| *v as f64 * *v as f64).sum();
            if acf0 == 0.0 { return (None, 0.0); }
            let rms = (w.iter().map(|v| v * v).sum::<f32>() / len as f32).sqrt();
            // strength of each lag's normalised autocorrelation (index 0 == dmin). The fundamental is found the YIN way: its
            // half-period trough is the first dip past dmin, and the fundamental is the FIRST peak that follows it — a later peak
            // at an octave or higher must be ignored, so this period is reported rather than a neighbour side-lobe on the peak.
            let vals: Vec<f64> = (dmin..=dmax)
                .map(|d| { let acc: f64 = (0..(w.len() - d)).map(|i| w[i] as f64 * w[i + d] as f64).sum(); acc / acf0 })
                .collect();
            let n = vals.len();
            if n < 2 { return (None, rms); }
            let mut dip = 0usize;
            for i in 1..n.saturating_sub(1) { if vals[i] < vals[i - 1] && vals[i] <= vals[i + 1] { dip = i; break; } }
            let mut peak = dip;
            while peak + 1 < n - 1 && vals[peak + 1] > vals[dip] && vals[peak + 1] >= vals[peak] { peak += 1; }
            let best = vals[peak];
            // sub-sample period: parabolic interpolation of the peak keeps the frequency accurate even at low sample rates, where
            // one integer step would otherwise read a neighbouring note off.
            let left = if peak > 0 { vals[peak - 1] } else { best };
            let right = if peak + 1 < n { vals[peak + 1] } else { best };
            let denom = left - 2.0 * best + right;
            let offset = if denom.abs() > 1e-6 { 0.5 * (left - right) / denom } else { 0.0 };
            let period = (dmin + peak) as f64 + offset;
            // a frame is voiced only when the strongest peak is itself strong (silence has none)
            if best < VOICE_FLOOR { return (None, rms); }
            (Some(rate / period), rms)
        })
        .collect()
}

#[allow(dead_code)]
/// The pitch-class histogram of the spectrum: a constant-Q filter bank over the fundamental+harmonics band (70-600 Hz). Each
/// of the twelve pitch classes accumulates the energy that passes its harmonics 1-4; the class the drone sits on is the
/// strongest there, whatever the voice does. Returns the winning pitch class and (for convenience) its lowest strong center.
fn spectral_tonic(section: &[f32], rate: f64) -> (usize, f64) {
    let (d, dr) = resample_to_max(section, rate, 8000.0);
    if d.len() < 256 { return (pitch_class(PITCH_MIN_HZ * 1.5), PITCH_MIN_HZ * 1.5); }
    let mut per_class = [0f64; 12];
    let mut best_e = 0f64;
    let mut center_hz = PITCH_MIN_HZ * 1.5;
    // log-spaced centers over the fundamental and its harmonics (a coarse constant-Q spectrum)
    let mut centers: Vec<f64> = Vec::new();
    let mut f = 70.0f64;
    while f < 650.0 { centers.push(f); f *= 1.06; }
    for &fc in &centers {
        // an octave-wide-ish band around fc (high-pass at fc/2, low-pass at 2*fc) — wide enough that the harmonics land in it
        let bp = bandpass(&d, dr, fc / 2.0, 2.0 * fc);
        let e: f64 = bp.iter().map(|v| *v as f64 * *v as f64).sum::<f64>();
        if e > best_e { best_e = e; center_hz = fc; }
        per_class[pitch_class(fc)] += e; // a harmonic of this class at any rank reinforces it
    }
    // the pitch class with the most accumulated harmonic energy; ties go to the lower number (e.g. Sa over Pa)
    let mut best_pc = 0usize;
    let mut best_pc_e = -1f64;
    for c in 0..12usize { if per_class[c] > best_pc_e { best_pc_e = per_class[c]; best_pc = c; } }
    (best_pc, center_hz)
}

/// The tone's home-note pitch class: the class the most voiced frames' fundamentals sit on. A drone is heard as its own note,
/// so the fundamentals of a frame track concentrate there whatever harmonics are loudest; ties go to the lower number.
fn tonic_class(frames: &[(Option<f64>, f32)]) -> usize {
    let mut per_class = [0usize; 12];
    let mut total = 0usize;
    for (f, _) in frames { if let Some(f) = f { per_class[pitch_class(*f)] += 1; total += 1; } }
    if total == 0 { return pitch_class(PITCH_MIN_HZ); }
    let mut best = 0usize;
    let mut best_e = usize::MIN;
    for c in 0..12usize { if per_class[c] > best_e { best_e = per_class[c]; best = c; } }
    best
}

/// The tonic hertz (the drone's home note): the median fundamental of the voiced frames whose pitch class is the most common
/// (see tonic_class), rounded to a MIDI note. Clean on an empty track (returns 0 / -1).
fn tonic_ref(frames: &[(Option<f64>, f32)], tonic_pc: usize) -> (f64, i64) {
    let mut class: Vec<f64> = frames.iter().filter_map(|(f, _)| *f)
        .filter(|f| pitch_class(*f) == tonic_pc).collect();
    if class.is_empty() { return (0.0, -1); }
    sort_f64(&mut class);
    // the median fundamental of that class: it rounds to a MIDI note (a tonic is a note), and reporting its exact frequency keeps
    // the reported hertz, the note name and every phrase's semitone in agreement (150 Hz reads back as D3, within 30 cents).
    let med = percentile_f64(&mut class, 0.5);
    let midi = midi_of(med).round() as i64;
    (freq_of_midi(midi as f64), midi)
}

/// Tonic (the drone's home note), the pitch range, and the share of voiced time on each swara above the tonic. `tonic_hz` is
/// computed once (spectral_tonic + tonic_ref) so the tonic note, its swaras and every phrase's semitones all agree. Each
/// swara carries its name and its share of voiced time, sorted by share.
fn pitch_obj(frames: &[(Option<f64>, f32)], _rate: f64, tonic_hz: f64, tonic_pc: usize) -> Value {
    let voiced_freqs: Vec<f64> = frames.iter().filter_map(|(f, _)| *f).collect();
    if voiced_freqs.is_empty() || tonic_hz <= 0.0 {
        return json!({"tonic": {"hz": 0.0, "note": "", "confidence": 0.0}, "range": {"low_hz": 0.0, "high_hz": 0.0}, "swaras": empty_list()});
    }

    // the tonic's pitch class drives its confidence (the share of voiced frames that sit on it) and the swara of each frame
    let mut counts = [0usize; 12];
    for &f in &voiced_freqs { counts[pitch_class(f)] += 1; }
    let conf = counts[tonic_pc] as f64 / voiced_freqs.len() as f64;

    // range: the 5th and 95th percentiles of every voiced frequency (the drone's note to the voice's highest)
    let mut sorted = voiced_freqs.clone();
    sort_f64(&mut sorted);
    let low = percentile_f64(&mut sorted, 0.05);
    let high = percentile_f64(&mut sorted, 0.95);

    // swaras: each voiced frame's semitone above the tonic, as a share of voiced time, sorted by share (each carries name+share)
    let mut swara_count = [0usize; 12];
    for &f in &voiced_freqs { swara_count[semitone_above(f, tonic_hz)] += 1; }
    let total: usize = swara_count.iter().sum();
    let mut swaras: Vec<Value> = (0..12)
        .filter(|&i| swara_count[i] > 0)
        .map(|i| {
            let share = if total > 0 { swara_count[i] as f64 / total as f64 } else { 0.0 };
            json!({"name": SWARA_NAMES[i], "share": r2(share)})
        })
        .collect();
    swaras.sort_by(|a, b| b["share"].as_f64().unwrap_or(0.0).partial_cmp(&a["share"].as_f64().unwrap_or(0.0)).unwrap());

    json!({"tonic": {"hz": r2(tonic_hz), "note": note_name(tonic_hz), "confidence": r2(conf)},
           "range": {"low_hz": r2(low), "high_hz": r2(high)}, "swaras": swaras})
}

/// One phrase's summary from its frame range: a median note and the swaras it moves through (repeats merged). `pf_start..pf_end`
/// is the range of fundamental-frame indices whose time falls inside the phrase.
fn one_phrase(frames: &[(Option<f64>, f32)], pf_range: std::ops::Range<usize>, tonic_midi: i64) -> Value {
    let a = pf_range.start.min(frames.len());
    let b = pf_range.end.min(frames.len());
    let start_s = a as f64 * FRAME_S;
    let end_s = b as f64 * FRAME_S;
    if b <= a {
        return json!({"start_s": r2(start_s), "end_s": r2(end_s), "median_note": "", "notes": []});
    }
    let mut freqs: Vec<f64> = frames[a..b].iter().filter_map(|(f, _)| *f).collect();
    if freqs.is_empty() {
        return json!({"start_s": r2(start_s), "end_s": r2(end_s), "median_note": "", "notes": []});
    }
    sort_f64(&mut freqs);
    let note = note_name(percentile_f64(&mut freqs, 0.5));

    // each voiced frame's semitone above the tonic (the swara it should read as), in order
    let mut semis: Vec<i64> = Vec::new();
    for &(f, _) in &frames[a..b] {
        if let Some(f) = f { semis.push(((midi_of(f).round() as i64 - tonic_midi) % 12 + 12) % 12); }
    }
    // de-glitch: replace each semitone by the median of itself and up to one neighbour; a sustained step survives, a slip does not
    let n = semis.len();
    let smoothed: Vec<i64> = (0..n).map(|i| {
        let mut w = Vec::with_capacity(3);
        if i > 0 { w.push(semis[i - 1]); }
        w.push(semis[i]);
        if i + 1 < n { w.push(semis[i + 1]); }
        w.sort_unstable();
        w[(w.len() - 1) / 2]
    }).collect();
    // swaras in order, merging consecutive repeats that read the same (de-glitched) semitone
    let mut notes = Vec::new();
    let mut last_semi: Option<i64> = None;
    for &semi in &smoothed {
        if last_semi != Some(semi) {
            notes.push(SWARA_NAMES[semi as usize].to_string());
            last_semi = Some(semi);
        }
    }

    json!({"start_s": r2(start_s), "end_s": r2(end_s), "median_note": note, "notes": notes})
}

/// Phrases over a drone: stretches of sound found from the voice's energy above the drone (the 200-3000 Hz band minus its
/// floor). A phrase ends where that falls under PHRASE_FRAC of its median for at least MIN_PAUSE_S, so a constant drone does
/// not keep everything as one phrase. Two phrases separated by less than PHRASE_JOIN_S are joined (a dip shorter than a breath
/// is one phrase), and a phrase shorter than MIN_PHRASE_S is dropped (a glip). At most 200 phrases, each with start/end and the
/// median note and swaras it moves through.
fn phrases_obj(rms: &[f32], floor: f32, hop50: usize, frames: &[(Option<f64>, f32)], tonic_midi: i64) -> Value {
    if rms.is_empty() || hop50 == 0 { return json!(Vec::<Value>::new()); }
    // the median voiced level (the drone's floor removed): a phrase ends when the envelope drops under this fraction of it
    let mut voiced_levels: Vec<f32> = rms.iter().copied().filter(|&v| v > floor.max(0.0)).collect();
    if voiced_levels.is_empty() { return json!(Vec::<Value>::new()); }
    let mut med = percentile_f32(&mut voiced_levels, 0.75);
    if med <= 0.0 { return json!(Vec::<Value>::new()); }
    // the median of the clearly-voiced frames (well above the drone's floor), so an all-drone rest does not drag it down
    if voiced_levels.is_empty() {
        let mut all: Vec<f32> = rms.iter().copied().filter(|&v| v > floor.max(0.0)).collect();
        if !all.is_empty() { med = percentile_f32(&mut all, 0.75); }
    }
    // a phrase ends when the band energy under the drone's floor falls under this fraction of the median voiced level; the
    // drone's constant harmonics alone stay well under it, so a rest where the voice drops out splits two phrases
    let thr = (PHRASE_FRAC as f32) * med;
    // a quiet gap that lasts at least this many frames separates two phrases (MIN_PAUSE_S of them)
    let min_gap_frames: usize = (MIN_PAUSE_S / ENV_S).round() as usize;

    let n = rms.len();
    let mask: Vec<bool> = rms.iter().map(|&v| v - floor > thr).collect();

    let mut ranges: Vec<(usize, usize)> = Vec::new(); // (first env frame, last env frame inclusive) per phrase
    let mut run_start = None;
    let mut k = 0usize;
    while k < n {
        if mask[k] {
            if run_start.is_none() { run_start = Some(k); }
            k += 1;
        } else {
            let gstart = k;
            while k < n && !mask[k] { k += 1; }
            let gap = k - gstart;
            if run_start.is_some() && gap >= min_gap_frames {
                ranges.push((run_start.unwrap(), gstart - 1));
                run_start = None;
            }
        }
    }
    if let Some(rs) = run_start { ranges.push((rs, n.saturating_sub(1))); }

    // join two phrases whose silence between them is under PHRASE_JOIN_S (a dip shorter than a breath keeps one phrase; the
    // ranges are in time order, so each start is compared to the previous range's end); this drops the many 0.3-0.5 s dips
    // over a drone that used to read as two phrases.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        if let Some(last) = merged.last_mut() {
            let gap = ((s as f64 - (last.1 + 1) as f64) * ENV_S).abs();
            // only a fragment is joined to its neighbour: two full phrases a breath apart stay two (joining every short gap
            // made one 80 s phrase of a chant over a drone)
            let short = |a: usize, b: usize| ((b + 1 - a) as f64 * ENV_S) < MIN_PHRASE_S;
            if gap < PHRASE_JOIN_S && (short(last.0, last.1) || short(s, e)) { last.1 = e; continue; }
        }
        merged.push((s, e));
    }

    // a phrase shorter than MIN_PHRASE_S is dropped (a glip, not a phrase), the rest kept in time order (at most 200). Each
    // env-frame span [s,e] covers the times [s*ENV_S,(e+1)*ENV_S]; fundamental frames are FRAME_S apart in time, so convert by
    // TIME, which stays right across the resample that produced these frames.
    let mut out: Vec<Value> = Vec::new();
    for (s, e) in merged {
        if out.len() >= 200 { break; }
        let pf_a = ((s as f64 * ENV_S / FRAME_S).round() as usize).min(frames.len());
        let pf_b = (((e + 1) as f64 * ENV_S / FRAME_S).round() as usize).min(frames.len());
        let p = one_phrase(frames, pf_a..pf_b, tonic_midi);
        let d = p["end_s"].as_f64().unwrap_or(0.0) - p["start_s"].as_f64().unwrap_or(0.0);
        if d >= MIN_PHRASE_S { out.push(p); }
    }
    json!(out)
}

/// Every analysis of a section: loudness, tempo, repeat period, tonic, range and phrases. `from_s` and `seconds` are seconds
/// into `pcm`; `seconds <= 0` means to the end (capped at MAX_SECTION_S). Runs clean even on a short or empty section.
pub fn run(pcm: &[f32], rate: u32, from_s: f64, seconds: f64) -> Value {
    let rate = rate as f64;
    let from_n = (from_s * rate).round() as usize;
    let span = if seconds > 0.0 { seconds.min(MAX_SECTION_S) } else { MAX_SECTION_S };
    let end_n = ((from_s + span) * rate).round() as usize;
    let end_n = end_n.min(pcm.len());
    if from_n >= end_n || pcm.is_empty() {
        return json!({"seconds": 0.0, "rate": (rate.round() as u32),
            "section": {"from_s": r2(from_s), "seconds": 0.0},
            "loudness": json!({"dbfs": -160.0, "per_second": empty_list()}),
            "tempo": json!({"bpm": 0.0, "confidence": 0.0, "steady_pulse": true}),
            "repeat": json!({"period_s": 0.0, "confidence": 0.0, "repeats_in_section": 0.0}),
            "pitch": json!({"tonic": {"hz": 0.0, "note": "", "confidence": 0.0}, "range": {"low_hz": 0.0, "high_hz": 0.0}, "swaras": empty_list()}),
            "phrases": empty_list()});
    }
    let section = &pcm[from_n..end_n];
    let sec = (end_n - from_n) as f64 / rate;

    let loudness = loudness_obj(section, rate);
    let tempo = tempo_obj(section, rate);

    // the voice band and its envelope (drone removed): drives repeat, phrases, and what a phrase is
    let (_vb, rms_env, floor, hop50) = voice_env(section, rate);
    let repeat = repeat_obj(&rms_env, hop50, sec);

    // pitch: the fundamental at 10 ms, on a low-passed copy so harmonics do not pull it onto an overtone; the spectrum's
    // strongest pitch class (the drone) decides which class is Sa, and its median fundamental is the tonic hertz
    let (xr, rate_d) = resample_to_max(section, rate, PITCH_SR_HZ);
    let frames = fundamental_frames(&xr, rate_d);
    // the tone's own home note: its fundamentals dominate the frame track (a drone is heard as that note, not an overtone)
    let tonic_pc = tonic_class(&frames);
    let (tonic_hz, tonic_midi) = tonic_ref(&frames, tonic_pc);
    let pitch = pitch_obj(&frames, rate_d, tonic_hz, tonic_pc);

    // phrases from the voice band; the fundamental frames carry each phrase's notes
    let phrases = phrases_obj(&rms_env, floor, hop50, &frames, tonic_midi);

    // the voice band is echoed back as a compact RMS trace so a caller can see where the voice (above the drone) rises and falls
    let voiced_trace: Vec<f64> = rms_env.iter().map(|&v| r3((v - floor).max(0.0) as f64)).collect();

    json!({"seconds": r2(sec), "rate": (rate.round() as u32), "section": {"from_s": r2(from_s), "seconds": r2(sec)},
           "loudness": loudness, "tempo": tempo, "repeat": repeat, "pitch": pitch, "phrases": phrases,
           "voice_band_rms": voiced_trace})
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 8_000;

    /// A sine of `freq` for `dur_s`, plus `tail` samples (silence) at the end, mono.
    fn tone(freq: f32, dur_s: f32, tail: usize) -> Vec<f32> {
        let n = (dur_s * SR as f32) as usize;
        let mut v = (0..n).map(|i| 0.8 * (2.0 * std::f32::consts::PI * freq * (i as f32 / SR as f32)).sin()).collect::<Vec<_>>();
        v.extend(std::iter::repeat(0f32).take(tail));
        v
    }

    /// The swaras a section gives, keyed by swara name -> share (for assertions).
    fn swaras_of(p: &Value) -> std::collections::BTreeMap<String, f64> {
        let mut m = std::collections::BTreeMap::new();
        if let Some(arr) = p["pitch"]["swaras"].as_array() {
            for s in arr { m.insert(s["name"].as_str().unwrap().to_string(), s["share"].as_f64().unwrap_or(0.0)); }
        }
        m
    }

    #[test]
    fn a_220_hz_tone_is_a3_with_sa_as_the_tonic() {
        // 2 s of A3 (220 Hz): the tonic's hertz, note and pitch class (Sa) are right
        let s = tone(220.0, 2.0, SR as usize / 4);
        let p = run(&s, SR, 0.0, 2.0);
        assert_eq!(p["pitch"]["tonic"]["note"], json!("A3"), "220 Hz is A3: {p}");
        assert!((p["pitch"]["tonic"]["hz"].as_f64().unwrap() - 220.0).abs() < 2.0, "tonic near 220 Hz: {}", p["pitch"]["tonic"]["hz"]);
        assert_eq!(swaras_of(&p).get("Sa"), Some(&1.0), "the whole tone is Sa over the tonic");
        assert!(p["pitch"]["tonic"]["confidence"].as_f64().unwrap() > 0.9, "a pure tone is an unambiguous tonic: {}", p["pitch"]["tonic"]["confidence"]);
    }

    #[test]
    fn clicks_every_half_second_read_as_120_bpm() {
        // one-sample clicks every 0.5 s -> a 120 bpm onset, within +/-2; plus silence so frames are not all identical
        let mut s = vec![0f32; SR as usize]; // 1 s of headroom at the start (boundaries)
        let mut t = 0.25f32;
        while ((t * SR as f32) as usize) < s.len() - 1 {
            s[(t * SR as f32) as usize] += 0.9;
            t += 0.5;
        }
        s.extend(std::iter::repeat(0f32).take(SR as usize)); // tail so the envelope has room to correlate
        let p = run(&s, SR, 0.0, 4.0);
        let bpm = p["tempo"]["bpm"].as_f64().unwrap();
        assert!((bpm - 120.0).abs() <= 2.0, "clicks every 0.5 s are 120 bpm (got {bpm})");
        assert!(p["tempo"]["confidence"].as_f64().unwrap() > 0.3, "the tempo is clear: {}", p["tempo"]["confidence"]);
    }

    #[test]
    fn three_notes_with_rests_are_three_phrases_with_the_right_notes() {
        // Sa (220 Hz) held the longest so it is unambiguously the tonic, then two other notes; each phrase is >= 0.8 s and
        // separated by a > 0.5 s rest (a gap < PHRASE_JOIN_S would join the phrases, this one stays split) -> 3 phrases in order
        let mut s = Vec::new();
        let note = |freq: f32, dur_s: f32| -> Vec<f32> {
            let n = (dur_s * SR as f32) as usize;
            (0..n).map(|i| 0.8 * (2.0 * std::f32::consts::PI * freq * (i as f32 / SR as f32)).sin()).collect()
        };
        s.extend(note(220.0, 2.0)); // Sa (tonic)
        s.extend(std::iter::repeat(0f32).take((0.7 * SR as f32) as usize)); // pause (> join and > min pause -> splits)
        s.extend(note(246.94, 1.0)); // the note a major second above (Re), long enough not to be dropped
        s.extend(std::iter::repeat(0f32).take((0.7 * SR as f32) as usize)); // pause
        s.extend(note(277.18, 1.0)); // the note a minor third above Sa (Ga), long enough not to be dropped
        s.extend(std::iter::repeat(0f32).take(SR as usize / 4));

        let p = run(&s, SR, 0.0, 5.6);
        let phrases = p["phrases"].as_array().unwrap();
        assert_eq!(phrases.len(), 3, "the rests split it into three phrases (got {})", phrases.len());
        let notes: Vec<&str> = phrases.iter().take(3).map(|f| f["median_note"].as_str().unwrap_or("")).collect();
        assert_eq!(notes[0], "A3", "first phrase's note: {notes:?}");
        // the three swaras, in order, repeats merged
        let first_three: Vec<String> = phrases.iter().take(3).filter_map(|f| f["notes"].as_array())
            .flat_map(|a| a.iter().map(|v| v.as_str().unwrap().to_string())).collect();
        assert_eq!(first_three, vec!["Sa".to_string(), "Re".to_string(), "Ga".to_string()], "swaras in order: {first_three:?}");
    }

    #[test]
    fn a_six_second_pattern_repeats_as_a_six_second_period() {
        // a loudness contour that changes second by second, repeated three times -> its autocorrelation peaks at 6 s
        let contour: [f32; 6] = [0.5, 0.7, 0.9, 0.7, 0.5, 0.3]; // seconds 0-5 of one cycle
        let mut s = Vec::new();
        for _ in 0..3 { // 18 s total: three repeats, so lag 6 (and 12) line up; the shortest such is picked
            for sec in 0..6 {
                let amp = contour[sec];
                let n = SR as usize; // one second per block
                for i in 0..n { s.push(amp * (2.0 * std::f32::consts::PI * 440.0 * ((sec * n + i) as f32 / SR as f32)).sin()); }
            }
        }
        let p = run(&s, SR, 0.0, 18.0);
        let period = p["repeat"]["period_s"].as_f64().unwrap();
        assert!((period - 6.0).abs() <= 1.0, "the loudness repeats every ~6 s (got {period})");
    }

    #[test]
    fn loudness_reports_the_overall_dbfs_and_per_second_frames() {
        // a full-amplitude tone here means amplitude 0.8 (tone()'s level), whose RMS is amp/sqrt(2) => dBFS = -4.95
        let s = tone(300.0, 1.0, 0);
        let p = run(&s, SR, 0.0, 1.0);
        let ref_dbfs = 20.0 * (0.8f32 / (2.0_f32).sqrt()).log10();
        let dbfs = p["loudness"]["dbfs"].as_f64().unwrap();
        assert!((dbfs - ref_dbfs as f64).abs() < 0.5, "a full-scale tone is ~-4.95 dBFS (got {dbfs})");
        assert!(p["loudness"]["per_second"].as_array().unwrap().len() >= 1, "one per-second frame for one second");
    }

    #[test]
    fn a_shorter_section_is_analyzed_than_the_whole_clip() {
        // from_s/seconds selects a window: the reported seconds match it, not the whole file
        let s = tone(200.0, 5.0, 0);
        let p = run(&s, SR, 1.0, 2.0);
        assert!((p["section"]["seconds"].as_f64().unwrap() - 2.0).abs() < 0.1, "the section is 2 s: {}", p["section"]["seconds"]);
    }

    #[test]
    fn the_tonic_follows_the_fundamental_and_not_an_overtone_of_the_drone() {
        // a 150 Hz drone with harmonics up to ~900 Hz plus a voice moving 150 -> 168 -> 200 Hz reads the tonic as D3 (within
        // 30 cents) and the range as about 150-200 Hz, i.e. the fundamentals, not the drone's higher overtones
        let mut s = Vec::new();
        let partial = |freq: f32, amp: f32, dur_s: f32| -> Vec<f32> {
            let n = (dur_s * SR as f32) as usize;
            (0..n).map(|i| amp * (2.0 * std::f32::consts::PI * freq * (i as f32 / SR as f32)).sin()).collect()
        };
        s.extend(partial(150.0, 0.8, 6.0));                       // the drone's fundamental = D3
        s.extend(partial(300.0, 0.4, 6.0)); s.extend(partial(450.0, 0.25, 6.0)); s.extend(partial(600.0, 0.18, 6.0));
        // a voice drifting through notes above the drone
        let mut v = Vec::new();
        v.extend(partial(150.0, 0.4, 2.0)); v.extend(partial(168.0, 0.4, 2.0)); v.extend(partial(200.0, 0.4, 2.0));
        s = [s, v].concat();
        let p = run(&s, SR, 0.0, 18.0);
        let tonic_hz = p["pitch"]["tonic"]["hz"].as_f64().unwrap();
        let midi = midi_of(tonic_hz);
        // D3 is midi 50 (146.83 Hz): within 30 cents either way
        assert!((midi - 50.0).abs() <= 0.3, "the tonic is D3 (~150 Hz), not an overtone: {tonic_hz} Hz ({p})");
        let low = p["pitch"]["range"]["low_hz"].as_f64().unwrap();
        let high = p["pitch"]["range"]["high_hz"].as_f64().unwrap();
        assert!((low - 150.0).abs() < 25.0, "the range starts near the drone's 150 Hz: {low} ({p})");
        assert!(high <= 230.0 && high >= 180.0, "the range ends near the voice's 200 Hz, not a harmonic: {high} ({p})");
    }

    #[test]
    fn phrases_over_a_drone_are_split_where_the_voice_rests() {
        // a constant drone plus a voice that sounds 3 s, rests 1 s, four times -> four phrases within +/- 0.3 s of the voices.
        // The drone's fundamental (below the voice band) and its in-band harmonics are present throughout, so a rest only removes
        // the VOICE, leaving the drone: that is what splits the phrases, not a silence in the whole signal.
        let dur_s = 24.0f32;
        let n = (dur_s * SR as f32) as usize;
        let mut s = vec![0f32; n];
        // a constant drone throughout: its fundamental (150 Hz, below the 200 Hz band) plus weak in-band harmonics that stay
        // on during rests, so the phrase envelope only rises when the voice adds to it
        for i in 0..n {
            s[i] += 0.7 * (2.0 * std::f32::consts::PI * 150.0 * (i as f32 / SR as f32)).sin();       // the drone's home note
            s[i] += 0.08 * (2.0 * std::f32::consts::PI * 450.0 * (i as f32 / SR as f32)).sin();       // an in-band harmonic
            s[i] += 0.05 * (2.0 * std::f32::consts::PI * 600.0 * (i as f32 / SR as f32)).sin();
        }
        // the voice: four voiced stretches, each 3 s, separated by 1 s rests, beginning at 0, 5, 10 and 15 s
        let voices: [(f32, f32, f32); 4] = [(0.0, 3.0, 200.0), (5.0, 3.0, 220.0), (10.0, 3.0, 240.0), (15.0, 3.0, 260.0)];
        for &(a, d, f) in &voices {
            let i0 = (a * SR as f32) as usize;
            let i1 = ((a + d) * SR as f32) as usize;
            for i in i0..i1 {
                s[i] += 0.6 * (2.0 * std::f32::consts::PI * f * (i as f32 / SR as f32)).sin();
                s[i] += 0.12 * (2.0 * std::f32::consts::PI * (f + f) * (i as f32 / SR as f32)).sin();
            }
        }
        let p = run(&s, SR, 0.0, 20.0);
        let phrases = p["phrases"].as_array().unwrap();
        assert_eq!(phrases.len(), 4, "four voiced stretches -> four phrases over the drone: {phrases:?}");
        // each phrase begins where its voice does (0, 5, 10, 15 s), within 0.3 s
        for (i, ph) in phrases.iter().enumerate() {
            let start = ph["start_s"].as_f64().unwrap();
            let expected = voices[i].0;
            assert!((start - expected as f64).abs() <= 0.3, "phrase {} starts near {expected}: {start} ({p})", i + 1);
        }
    }

    #[test]
    fn the_repeat_period_is_the_shortest_one_that_repeats_strongly() {
        // a 20 s pattern repeated six times (120 s), whose two halves are alike but not identical: it reads as 20 s, neither
        // the doubled 40 nor the halved 10
        let mut s = Vec::new();
        let block = |freq: f32, dur_s: f32| -> Vec<f32> {
            let n = (dur_s * SR as f32) as usize;
            (0..n).map(|i| 0.6 * (2.0 * std::f32::consts::PI * freq * (i as f32 / SR as f32)).sin()).collect()
        };
        // one 20 s verse: two halves that differ in pitch but repeat every 20 s
        let mut verse = Vec::new();
        verse.extend(block(160.0, 10.0));   // first half (alike to the second)
        verse.extend(block(180.0, 10.0));   // second half (not identical -> no 10 s repeat)
        for _ in 0..6 { s.extend(verse.clone()); }
        let p = run(&s, SR, 0.0, 120.0);
        let period = p["repeat"]["period_s"].as_f64().unwrap();
        assert!((period - 20.0).abs() <= 1.5, "the repeat is one verse (~20 s), not the doubled 40 or halved 10: {period} ({p})");
        let repeats = p["repeat"]["repeats_in_section"].as_f64().unwrap();
        assert!((repeats - 6.0).abs() <= 0.5, "it repeats six times in the section: {repeats}");
    }

    #[test]
    fn a_low_confidence_tempo_reports_an_unsteady_pulse_flag() {
        // noise with no beat reads under STEADY_CONF and carries steady_pulse:false so it is not trusted
        let mut s = tone(300.0, 6.0, SR as usize);
        for v in s.iter_mut() { *v += (rand() - 0.5) * 0.2; }
        let p = run(&s, SR, 0.0, 6.0);
        assert!(p["tempo"]["steady_pulse"].as_bool().unwrap_or(true) == false, "a random tempo is flagged unsteady: {p}");
    }

    #[test]
    fn swaras_each_carry_a_name_and_a_share() {
        // Sa held then Re and Ga (rests already split them): every swara entry has a name and a share, sorted by share desc
        let mut s = Vec::new();
        let note = |freq: f32, dur_s: f32| -> Vec<f32> {
            let n = (dur_s * SR as f32) as usize;
            (0..n).map(|i| 0.8 * (2.0 * std::f32::consts::PI * freq * (i as f32 / SR as f32)).sin()).collect()
        };
        s.extend(note(220.0, 1.0));
        s.extend(std::iter::repeat(0f32).take((0.35 * SR as f32) as usize));
        s.extend(note(246.94, 0.6));
        s.extend(std::iter::repeat(0f32).take((0.35 * SR as f32) as usize));
        s.extend(note(277.18, 0.6));
        s.extend(std::iter::repeat(0f32).take(SR as usize / 4));
        let p = run(&s, SR, 0.0, 2.6);
        for sw in p["pitch"]["swaras"].as_array().unwrap() {
            assert!(sw.get("name").is_some(), "each swara has a name: {sw}");
            assert!(sw.get("share").is_some(), "each swara carries its share: {sw}");
        }
    }

    /// Two bursts of a tone separated by silent gaps (0.4 or 0.6 s): the phrase split is on >= 0.3 s gaps, so both split it into
    /// two runs; a gap under PHRASE_JOIN_S then joins them into one phrase, and a single brief (< MIN_PHRASE_S) burst is dropped.
    #[test]
    fn phrases_join_over_short_gaps_and_drop_brief_ones() {
        let voiced = |parts: &[(f32, f32)]| -> Vec<f32> { // (start_s, dur_s) at absolute sample position, silence between parts; a phrase is a 440 Hz tone above the floor
            let mut s = Vec::new();
            for &(a, d) in parts {
                while (s.len() as f64) < (a as f64) * (SR as f64) { s.push(0f32); } // silence up to the absolute start of this part
                let n = (d * SR as f32) as usize;
                for i in 0..n { s.push(0.8 * (2.0 * std::f32::consts::PI * 440.0 * (i as f32 / SR as f32)).sin()); }
            }
            s.extend(std::iter::repeat(0f32).take(SR as usize / 2)); // a little tail so the end frames are not all identical
            s
        };
        // a 1 s burst and a 0.4 s fragment 0.4 s after it: the fragment joins its neighbour
        let p0 = run(&voiced(&[(0.0, 1.0), (1.4, 0.4)]), SR, 0.0, 2.4);
        assert_eq!(p0["phrases"].as_array().unwrap().len(), 1, "a fragment joins the phrase before it: {:?}", p0["phrases"]);
        // two full 1 s phrases 0.4 s apart stay two: a breath between phrases is not a join
        let p3 = run(&voiced(&[(0.0, 1.0), (1.4, 1.0)]), SR, 0.0, 3.0);
        assert_eq!(p3["phrases"].as_array().unwrap().len(), 2, "two full phrases a breath apart stay two: {:?}", p3["phrases"]);
        // two 1 s bursts 0.6 s apart: not short enough to join -> two phrases
        let p1 = run(&voiced(&[(0.0, 1.0), (1.6, 1.0)]), SR, 0.0, 3.2);
        assert_eq!(p1["phrases"].as_array().unwrap().len(), 2, "a >= 0.5 s gap keeps two phrases: {:?}", p1["phrases"]);
        // a single 0.5 s burst with silence each side is under MIN_PHRASE_S -> dropped (was one phrase before the fix)
        let p2 = run(&voiced(&[(0.0, 0.5)]), SR, 0.0, 1.5);
        assert_eq!(p2["phrases"].as_array().unwrap().len(), 0, "a phrase under 0.8 s is dropped: {:?}", p2["phrases"]);
    }

    /// A tiny deterministic noise source so the tempo test's signal is reproducible.
    fn rand() -> f32 {
        static mut N: u64 = 0x9E3779B97F4A7C15;
        unsafe { N ^= N << 13; N ^= N >> 7; N ^= N << 17; (N % 1000) as f32 / 500.0 - 1.0 }
    }
}

