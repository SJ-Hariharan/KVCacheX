//! Small dependency-free helpers: a seeded RNG, a JSON writer, percentile summaries, and
//! `--flag value` argument parsing.

use std::collections::HashMap;
use std::fmt::Write as _;

/// xoshiro256** seeded through SplitMix64. Deterministic across platforms.
#[derive(Debug, Clone)]
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Rng([next(), next(), next(), next()])
    }
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
    /// Uniform in `[lo, hi]` (inclusive).
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Exponential inter-arrival with the given mean.
    pub fn exp(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.f64()).ln()
    }
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut x = self.f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                return i;
            }
            x -= w;
        }
        weights.len() - 1
    }
}

/// Minimal JSON value for writing reports.
#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Int(i64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, 0);
        s.push('\n');
        s
    }
    fn write(&self, out: &mut String, indent: usize) {
        let pad = |n| "  ".repeat(n);
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(x) if x.is_finite() => {
                let _ = write!(out, "{x}");
            }
            Json::Num(_) => out.push_str("null"),
            Json::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Json::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        c if (c as u32) < 0x20 => {
                            let _ = write!(out, "\\u{:04x}", c as u32);
                        }
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Json::Arr(v) => {
                if v.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, x) in v.iter().enumerate() {
                    out.push_str(&pad(indent + 1));
                    x.write(out, indent + 1);
                    out.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(indent));
                out.push(']');
            }
            Json::Obj(v) => {
                if v.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (i, (k, x)) in v.iter().enumerate() {
                    out.push_str(&pad(indent + 1));
                    Json::Str(k.clone()).write(out, indent + 1);
                    out.push_str(": ");
                    x.write(out, indent + 1);
                    out.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(indent));
                out.push('}');
            }
        }
    }
}

/// Order statistics of a sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub stdev: f64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Summary {
    /// Nearest-rank percentiles. Sorts `xs` in place.
    pub fn of(xs: &mut [f64]) -> Summary {
        if xs.is_empty() {
            return Summary::default();
        }
        xs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let n = xs.len();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let pct = |p: f64| xs[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1];
        Summary {
            n,
            mean,
            stdev: var.sqrt(),
            min: xs[0],
            p50: pct(50.0),
            p95: pct(95.0),
            p99: pct(99.0),
            max: xs[n - 1],
        }
    }
    pub fn json(&self) -> Json {
        Json::obj(vec![
            ("n", Json::Int(self.n as i64)),
            ("mean", Json::Num(self.mean)),
            ("stdev", Json::Num(self.stdev)),
            ("min", Json::Num(self.min)),
            ("p50", Json::Num(self.p50)),
            ("p95", Json::Num(self.p95)),
            ("p99", Json::Num(self.p99)),
            ("max", Json::Num(self.max)),
        ])
    }
}

/// `--key value` and bare `--flag` arguments.
#[derive(Debug, Default)]
pub struct Args {
    map: HashMap<String, String>,
}

impl Args {
    pub fn parse(raw: &[String]) -> Result<Args, String> {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let k = raw[i]
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument '{}'", raw[i]))?;
            if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                map.insert(k.to_string(), raw[i + 1].clone());
                i += 2;
            } else {
                map.insert(k.to_string(), "true".to_string());
                i += 1;
            }
        }
        Ok(Args { map })
    }
    pub fn str(&self, k: &str, default: &str) -> String {
        self.map.get(k).cloned().unwrap_or_else(|| default.to_string())
    }
    pub fn opt(&self, k: &str) -> Option<&str> {
        self.map.get(k).map(String::as_str)
    }
    pub fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.map.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("--{k}: cannot parse '{v}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_nearest_rank() {
        let mut xs: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let s = Summary::of(&mut xs);
        assert_eq!((s.p50, s.p95, s.p99, s.max), (50.0, 95.0, 99.0, 100.0));
    }

    #[test]
    fn rng_is_deterministic() {
        let a: Vec<u64> = {
            let mut r = Rng::new(7);
            (0..5).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::new(7);
            (0..5).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn json_escapes() {
        assert_eq!(Json::Str("a\"b\n".into()).render(), "\"a\\\"b\\n\"\n");
    }
}
