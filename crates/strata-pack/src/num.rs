//! Numbers exactly as PolyStrata's Python tools produce them: numpy's float16 conversion, gguf-py's BF16 rounding,
//! numpy's pairwise float sums, and Python's float repr and json.dumps layout, so a pack written here is byte for byte
//! the one tools/iq_pack.py, mtp_pack.py and mtp_rt.py write.

use serde_json::Value;

/// IEEE half -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalize
            let mut e = 113u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE half, round to nearest even (numpy's npy_floatbits_to_halfbits; overflow gives inf).
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let f_exp = x & 0x7f80_0000;
    if f_exp >= 0x4780_0000 {
        if f_exp == 0x7f80_0000 {
            let sig = x & 0x007f_ffff;
            if sig != 0 {
                let mut ret = (0x7c00u32 + (sig >> 13)) as u16;
                if ret == 0x7c00 {
                    ret += 1;
                }
                return sign + ret;
            }
            return sign + 0x7c00;
        }
        return sign + 0x7c00;
    }
    if f_exp <= 0x3800_0000 {
        if f_exp < 0x3300_0000 {
            return sign;
        }
        let e = f_exp >> 23;
        let mut sig = 0x0080_0000 + (x & 0x007f_ffff);
        sig >>= 113 - e;
        if (sig & 0x3fff) != 0x1000 || (x & 0x7ff) != 0 {
            sig += 0x1000;
        }
        return sign + (sig >> 13) as u16;
    }
    let h_exp = ((f_exp - 0x3800_0000) >> 13) as u16;
    let mut sig = x & 0x007f_ffff;
    if (sig & 0x3fff) != 0x1000 {
        sig += 0x1000;
    }
    let h_sig = (sig >> 13) as u16;
    sign + h_sig + h_exp
}

/// f32 -> BF16 bits, round to nearest even, NaN made quiet (gguf-py BF16.quantize_blocks = ggml_compute_fp32_to_bf16).
pub fn f32_to_bf16(f: f32) -> u16 {
    let mut n = f.to_bits() as u64;
    if (n & 0x7fff_ffff) > 0x7f80_0000 {
        n = (n & 0xffff_0000) | (64 << 16);
    }
    ((n + (0x7fff + ((n >> 16) & 1))) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// numpy's pairwise sum of float32 (the add reduction of a contiguous run: 8 accumulators per 128 values).
pub fn pairwise_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f32;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_f32(&a[..n2]) + pairwise_f32(&a[n2..])
    }
}

/// The same for float64.
pub fn pairwise_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f64;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_f64(&a[..n2]) + pairwise_f64(&a[n2..])
    }
}

/// numpy's round-half-to-even (np.rint) for f32.
pub fn rint_f32(x: f32) -> f32 {
    x.round_ties_even()
}

/// Python's repr() of a float: the shortest digits that round-trip, fixed notation for exponents -4..15, else
/// scientific with a two-digit exponent (1e-05, 1e+16); integral values keep ".0".
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's {:e} prints the shortest round-trip digits ("d.ddde-N"), but breaks an exact tie upwards; Python's dtoa
    // takes, among the shortest, the closest with ties to even, which is the correctly rounded string of that length
    // ({:.Ne} rounds exactly, ties to even)
    let shortest = format!("{:e}", x.abs());
    let n = shortest.split_once('e').unwrap().0.chars().filter(|c| c.is_ascii_digit()).count();
    let even = format!("{:.*e}", n - 1, x.abs());
    let s = if even.parse::<f64>().ok() == Some(x.abs()) { even } else { shortest };
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let m = if digits.len() == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        return format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let n = digits.len() as i32;
    let body = if exp < 0 {
        format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
    } else if exp + 1 >= n {
        format!("{}{}.0", digits, "0".repeat((exp + 1 - n) as usize))
    } else {
        format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
    };
    format!("{sign}{body}")
}

/// Python's round(x, n) for a float, then as a float again.
pub fn py_round(x: f64, n: usize) -> f64 {
    format!("{:.*}", n, x).parse().unwrap()
}

/// Python json.dumps(value, indent=indent) (ensure_ascii, separators ", " / ": "; with an indent, "," at line ends).
pub fn py_json(v: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_json(&mut out, v, indent, 0);
    out
}

fn write_json(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_repr(n.as_f64().unwrap()));
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, indent, level + 1);
                write_json(out, x, indent, level + 1);
            }
            newline(out, indent, level);
            out.push(']');
        }
        Value::Object(o) => {
            if o.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, indent, level + 1);
                write_str(out, k);
                out.push_str(": ");
                write_json(out, x, indent, level + 1);
            }
            newline(out, indent, level);
            out.push('}');
        }
    }
}

fn newline(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float JSON value (kept a float even when integral, as Python's float).
pub fn jf(x: f64) -> Value {
    Value::Number(serde_json::Number::from_f64(x).unwrap_or_else(|| serde_json::Number::from(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_like_python() {
        for (x, want) in [(0.001953125, "0.001953125"), (1e-05, "1e-05"), (0.0001, "0.0001"), (1e16, "1e+16"),
                          (123456789012345678.0, "1.2345678901234568e+17"), (2.0, "2.0"), (0.1, "0.1"), (-3.5, "-3.5"),
                          (1234.5, "1234.5"), (1e15, "1000000000000000.0"), (6.103515625e-05, "6.103515625e-05"),
                          (0.30000000000000004, "0.30000000000000004")] {
            assert_eq!(py_repr(x), want, "{x}");
        }
    }

    #[test]
    fn half_round_trips() {
        for h in 0u16..=0xffff {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#x}");
        }
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "rounds to inf");
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00, "tie to even");
    }

    #[test]
    fn json_like_python() {
        let v = serde_json::json!({"a": [1, jf(0.5), "é\n"], "b": {}, "c": []});
        assert_eq!(py_json(&v, Some(1)), "{\n \"a\": [\n  1,\n  0.5,\n  \"\\u00e9\\n\"\n ],\n \"b\": {},\n \"c\": []\n}");
        assert_eq!(py_json(&v, None), "{\"a\": [1, 0.5, \"\\u00e9\\n\"], \"b\": {}, \"c\": []}");
    }
}
