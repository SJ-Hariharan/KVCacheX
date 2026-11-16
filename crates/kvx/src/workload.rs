//! Synthetic request workloads. Token ids are synthetic: they decide prefix identity
//! only and carry no meaning.

use crate::util::Rng;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub arrival_step: u64,
    pub namespace: u64,
    pub prompt: Vec<u32>,
    /// Tokens the request will actually generate (unknown to the scheduler).
    pub output_len: u32,
    /// `max_tokens` the client asked for: an upper bound the scheduler can see.
    pub max_output: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Chat-style short prompts and answers.
    Short,
    /// Long-document prompts.
    Long,
    /// 85% short, 15% long.
    Mixed,
    /// Few popular system prompts (1024 tokens) plus a unique user turn.
    SharedPrefix,
    /// Parallel sampling: groups of 4 requests with an identical prompt.
    Parallel,
}

impl Kind {
    pub const ALL: [Kind; 5] = [
        Kind::Short,
        Kind::Long,
        Kind::Mixed,
        Kind::SharedPrefix,
        Kind::Parallel,
    ];

    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "short" => Kind::Short,
            "long" => Kind::Long,
            "mixed" => Kind::Mixed,
            "shared-prefix" => Kind::SharedPrefix,
            "parallel" => Kind::Parallel,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Short => "short",
            Kind::Long => "long",
            Kind::Mixed => "mixed",
            Kind::SharedPrefix => "shared-prefix",
            Kind::Parallel => "parallel",
        }
    }
    pub fn describe(self) -> &'static str {
        match self {
            Kind::Short => "prompt U[32,256], output U[16,256], max_tokens 512",
            Kind::Long => "prompt U[1024,4096], output U[64,512], max_tokens 1024",
            Kind::Mixed => "85% short / 15% long",
            Kind::SharedPrefix => {
                "4 system prompts of 1024 tok (Zipf-like 8:4:2:1) + unique U[32,256], output U[32,256], max_tokens 512"
            }
            Kind::Parallel => "groups of 4 identical prompts U[256,1024], output U[32,256], max_tokens 512",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkloadSpec {
    pub kind: Kind,
    pub requests: usize,
    /// Mean requests arriving per scheduler step (Poisson). 0 = all arrive at step 0 (burst).
    pub arrival_rate: f64,
    pub seed: u64,
}

pub fn generate(spec: &WorkloadSpec) -> Vec<Request> {
    let mut rng = Rng::new(spec.seed);
    let mut next_tok: u32 = 1_000_000;
    let mut fresh = |rng: &mut Rng, n: u64| -> Vec<u32> {
        let _ = rng;
        let v: Vec<u32> = (0..n as u32).map(|i| next_tok + i).collect();
        next_tok += n as u32;
        v
    };
    let system_prompts: Vec<Vec<u32>> = (0..4u32)
        .map(|s| (0..1024).map(|i| s * 10_000 + i).collect())
        .collect();

    let mut out = Vec::with_capacity(spec.requests);
    let mut t = 0.0f64;
    let mut group: Option<(Vec<u32>, u32)> = None;
    let mut id = 0u64;
    while out.len() < spec.requests {
        if spec.arrival_rate > 0.0 {
            t += rng.exp(1.0 / spec.arrival_rate);
        }
        let arrival_step = t as u64;
        let kind = match spec.kind {
            Kind::Mixed => {
                if rng.f64() < 0.15 {
                    Kind::Long
                } else {
                    Kind::Short
                }
            }
            k => k,
        };
        let (prompt, output_len, max_output) = match kind {
            Kind::Short => {
                let n = rng.range(32, 256);
                (fresh(&mut rng, n), rng.range(16, 256) as u32, 512)
            }
            Kind::Long => {
                let n = rng.range(1024, 4096);
                (fresh(&mut rng, n), rng.range(64, 512) as u32, 1024)
            }
            Kind::SharedPrefix => {
                let s = rng.weighted(&[8.0, 4.0, 2.0, 1.0]);
                let mut p = system_prompts[s].clone();
                let n = rng.range(32, 256);
                p.extend(fresh(&mut rng, n));
                (p, rng.range(32, 256) as u32, 512)
            }
            Kind::Parallel => {
                let (p, left) = match group.take() {
                    Some((p, left)) if left > 0 => (p, left),
                    _ => {
                        let n = rng.range(256, 1024);
                        (fresh(&mut rng, n), 4)
                    }
                };
                if left > 1 {
                    group = Some((p.clone(), left - 1));
                }
                (p, rng.range(32, 256) as u32, 512)
            }
            Kind::Mixed => unreachable!(),
        };
        out.push(Request {
            id,
            arrival_step,
            namespace: 0,
            prompt,
            output_len,
            max_output,
        });
        id += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_sized() {
        let spec = WorkloadSpec {
            kind: Kind::Mixed,
            requests: 200,
            arrival_rate: 2.0,
            seed: 3,
        };
        let a = generate(&spec);
        let b = generate(&spec);
        assert_eq!(a.len(), 200);
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.prompt == y.prompt && x.arrival_step == y.arrival_step));
        assert!(a.iter().all(|r| r.output_len <= r.max_output));
        assert!(a.windows(2).all(|w| w[0].arrival_step <= w[1].arrival_step));
    }

    #[test]
    fn parallel_groups_share_prompts() {
        let spec = WorkloadSpec {
            kind: Kind::Parallel,
            requests: 8,
            arrival_rate: 0.0,
            seed: 1,
        };
        let r = generate(&spec);
        assert!(r[0..4].iter().all(|x| x.prompt == r[0].prompt));
        assert_ne!(r[0].prompt, r[4].prompt);
    }
}
