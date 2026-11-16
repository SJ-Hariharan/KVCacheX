//! Discrete-step simulation of a continuous-batching scheduler over interchangeable cache
//! backends.
//!
//! **What this measures:** how cache memory management limits batching and how much prefill
//! work prefix caching avoids, under a fixed KV memory budget. Time is counted in *scheduler
//! steps*, not seconds: one step = one decode token for every running sequence, plus prefill
//! of newly admitted prompts within a token budget. No model runs, so these are
//! not end-to-end latency or throughput numbers of a real engine.
//!
//! Scheduler (same for every backend, modeled on vLLM's FCFS policy):
//! 1. Decode: each running sequence (oldest first) grows by one token. If the backend is out
//!    of memory, the youngest running sequence is preempted: released and re-queued at the
//!    front, to be recomputed from prompt + generated tokens.
//! 2. Admit: waiting requests, FCFS, while the step's token budget and `max_seqs` allow and
//!    the backend accepts them. Admission stops at the first request that does not fit.

use std::collections::{HashMap, VecDeque};

use kvcachex::baseline::{ContiguousAllocator, Extent};
use kvcachex::{CacheConfig, CacheManager, SeqId};

use crate::util::{Json, Summary};
use crate::workload::Request;

#[derive(Debug, Clone, Copy)]
pub struct MemSample {
    /// Slots held by the backend (blocks × block size, or reserved extents).
    pub held_slots: u64,
    /// Distinct token slots holding KV.
    pub stored_tokens: u64,
    pub capacity_slots: u64,
}

pub trait Backend {
    fn name(&self) -> String;
    /// Prompt tokens the backend could serve from cache for this prompt right now.
    fn peek_cached(&self, ns: u64, prompt: &[u32]) -> usize;
    /// Admits a sequence. `max_total` is the length bound known to the scheduler.
    /// Returns cached prompt tokens, or `None` if memory is insufficient.
    fn admit(&mut self, rid: u64, ns: u64, prompt: &[u32], max_total: u64) -> Option<usize>;
    fn computed(&mut self, rid: u64, n: usize);
    fn grow(&mut self, rid: u64, token: u32) -> bool;
    fn release(&mut self, rid: u64);
    fn sample(&self) -> MemSample;
    fn check(&self) -> Result<(), String>;
    fn counters(&self) -> Vec<(&'static str, Json)>;
}

/// KVCacheX paged manager, with or without prefix caching.
#[derive(Debug)]
pub struct PagedBackend {
    m: CacheManager,
    seqs: HashMap<u64, SeqId>,
    /// Free blocks kept in reserve at admission to avoid immediately preempting (vLLM uses 1%).
    watermark: u32,
}

impl PagedBackend {
    pub fn new(cfg: CacheConfig) -> Self {
        PagedBackend {
            watermark: cfg.num_blocks / 100,
            m: CacheManager::new(cfg),
            seqs: HashMap::new(),
        }
    }
}

impl Backend for PagedBackend {
    fn name(&self) -> String {
        if self.m.config().enable_prefix_caching {
            "paged+prefix".into()
        } else {
            "paged".into()
        }
    }
    fn peek_cached(&self, ns: u64, prompt: &[u32]) -> usize {
        self.m.cached_prefix_len(ns, prompt)
    }
    fn admit(&mut self, rid: u64, ns: u64, prompt: &[u32], _max_total: u64) -> Option<usize> {
        if self.m.blocks_needed(ns, prompt) + self.watermark > self.m.num_free_blocks() {
            return None;
        }
        let a = self.m.create_sequence(ns, prompt).ok()?;
        self.seqs.insert(rid, a.seq);
        Some(a.cached_tokens)
    }
    fn computed(&mut self, rid: u64, n: usize) {
        self.m
            .mark_computed(self.seqs[&rid], n)
            .expect("valid computed count");
    }
    fn grow(&mut self, rid: u64, token: u32) -> bool {
        self.m.append_tokens(self.seqs[&rid], &[token]).is_ok()
    }
    fn release(&mut self, rid: u64) {
        let s = self.seqs.remove(&rid).expect("known request");
        self.m.release(s).expect("live sequence");
    }
    fn sample(&self) -> MemSample {
        let s = self.m.stats();
        let bs = s.block_size as u64;
        MemSample {
            held_slots: s.used_blocks as u64 * bs,
            stored_tokens: s.physical_tokens,
            capacity_slots: s.total_blocks as u64 * bs,
        }
    }
    fn check(&self) -> Result<(), String> {
        self.m.check_invariants()
    }
    fn counters(&self) -> Vec<(&'static str, Json)> {
        let c = self.m.counters();
        vec![
            ("blocks_allocated", Json::Int(c.blocks_allocated as i64)),
            ("allocation_failures", Json::Int(c.allocation_failures as i64)),
            ("evictions", Json::Int(c.evictions as i64)),
            ("prefix_query_blocks", Json::Int(c.prefix_query_blocks as i64)),
            ("prefix_hit_blocks", Json::Int(c.prefix_hit_blocks as i64)),
            ("prefix_revived_blocks", Json::Int(c.prefix_revived_blocks as i64)),
            ("prefix_block_hit_rate", Json::Num(c.prefix_hit_rate())),
            (
                "peak_used_blocks",
                Json::Int(self.m.stats().peak_used_blocks as i64),
            ),
        ]
    }
}

/// Baseline: contiguous reservation of `prompt + max_tokens` per request.
#[derive(Debug)]
pub struct ContiguousBackend {
    a: ContiguousAllocator,
    seqs: HashMap<u64, (Extent, u64)>,
}

impl ContiguousBackend {
    pub fn new(capacity_tokens: u64) -> Self {
        ContiguousBackend {
            a: ContiguousAllocator::new(capacity_tokens),
            seqs: HashMap::new(),
        }
    }
}

impl Backend for ContiguousBackend {
    fn name(&self) -> String {
        "contiguous".into()
    }
    fn peek_cached(&self, _: u64, _: &[u32]) -> usize {
        0
    }
    fn admit(&mut self, rid: u64, _ns: u64, prompt: &[u32], max_total: u64) -> Option<usize> {
        let e = self.a.reserve(max_total.max(prompt.len() as u64))?;
        self.seqs.insert(rid, (e, prompt.len() as u64));
        Some(0)
    }
    fn computed(&mut self, _: u64, _: usize) {}
    fn grow(&mut self, rid: u64, _token: u32) -> bool {
        let (e, len) = self.seqs.get_mut(&rid).expect("known request");
        if *len < e.len {
            *len += 1;
            true
        } else {
            false
        }
    }
    fn release(&mut self, rid: u64) {
        let (e, _) = self.seqs.remove(&rid).expect("known request");
        self.a.release(e);
    }
    fn sample(&self) -> MemSample {
        MemSample {
            held_slots: self.a.reserved(),
            stored_tokens: self.seqs.values().map(|(_, l)| *l).sum(),
            capacity_slots: self.a.capacity(),
        }
    }
    fn check(&self) -> Result<(), String> {
        self.a.check_invariants()
    }
    fn counters(&self) -> Vec<(&'static str, Json)> {
        vec![
            ("allocation_failures", Json::Int(self.a.failures as i64)),
            ("peak_reserved_tokens", Json::Int(self.a.peak_reserved() as i64)),
            (
                "final_external_fragmentation",
                Json::Num(self.a.external_fragmentation()),
            ),
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SchedConfig {
    pub max_batch_tokens: u64,
    pub max_seqs: usize,
    /// Run full invariant checks every N steps (0 = never).
    pub check_every: u64,
    pub max_steps: u64,
}

impl Default for SchedConfig {
    fn default() -> Self {
        SchedConfig {
            max_batch_tokens: 4096,
            max_seqs: 256,
            check_every: 0,
            max_steps: 5_000_000,
        }
    }
}

#[derive(Debug, Clone)]
struct Pending {
    req: usize,
    generated: u32,
    /// Step of the request's first generated token, kept across preemption so TTFT
    /// measures the first token ever produced, not the first after re-admission.
    first_token_step: Option<u64>,
}

#[derive(Debug, Clone)]
struct Running {
    req: usize,
    generated: u32,
    first_token_step: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SimResult {
    pub backend: String,
    pub steps: u64,
    pub completed: usize,
    pub rejected: usize,
    pub generated_tokens: u64,
    pub prompt_tokens: u64,
    pub prefill_computed_tokens: u64,
    pub prefix_cached_tokens: u64,
    pub preemptions: u64,
    pub recomputed_tokens: u64,
    pub mean_running: f64,
    pub peak_running: usize,
    pub mean_held_frac: f64,
    pub mean_stored_frac: f64,
    /// Mean, over steps holding any memory, of the share of held slots with no KV
    /// (reservation + block slack).
    pub mean_waste_frac: f64,
    /// `max_steps` was reached before every request finished; results are partial.
    pub truncated: bool,
    pub ttft_steps: Summary,
    pub e2e_steps: Summary,
    pub wall_ms: f64,
    pub counters: Vec<(&'static str, Json)>,
}

impl SimResult {
    pub fn tokens_per_step(&self) -> f64 {
        self.generated_tokens as f64 / self.steps.max(1) as f64
    }
    pub fn json(&self) -> Json {
        Json::obj(vec![
            ("backend", Json::Str(self.backend.clone())),
            ("steps", Json::Int(self.steps as i64)),
            ("completed", Json::Int(self.completed as i64)),
            ("rejected", Json::Int(self.rejected as i64)),
            ("truncated", Json::Bool(self.truncated)),
            ("generated_tokens", Json::Int(self.generated_tokens as i64)),
            ("generated_tokens_per_step", Json::Num(self.tokens_per_step())),
            ("prompt_tokens", Json::Int(self.prompt_tokens as i64)),
            (
                "prefill_computed_tokens",
                Json::Int(self.prefill_computed_tokens as i64),
            ),
            (
                "prefix_cached_tokens",
                Json::Int(self.prefix_cached_tokens as i64),
            ),
            ("preemptions", Json::Int(self.preemptions as i64)),
            ("recomputed_tokens", Json::Int(self.recomputed_tokens as i64)),
            ("mean_running_seqs", Json::Num(self.mean_running)),
            ("peak_running_seqs", Json::Int(self.peak_running as i64)),
            ("mean_held_fraction", Json::Num(self.mean_held_frac)),
            ("mean_stored_fraction", Json::Num(self.mean_stored_frac)),
            ("mean_waste_fraction", Json::Num(self.mean_waste_frac)),
            ("ttft_steps", self.ttft_steps.json()),
            ("e2e_steps", self.e2e_steps.json()),
            ("wall_ms", Json::Num(self.wall_ms)),
            (
                "backend_counters",
                Json::Obj(
                    self.counters
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.clone()))
                        .collect(),
                ),
            ),
        ])
    }
}

fn output_token(req: u64, i: u32) -> u32 {
    // Distinct from prompt token ranges; deterministic so recompute regenerates the same tokens.
    0x8000_0000 | ((req as u32).wrapping_mul(4099) ^ i)
}

fn full_prompt(r: &Request, generated: u32) -> Vec<u32> {
    let mut p = r.prompt.clone();
    p.extend((0..generated).map(|i| output_token(r.id, i)));
    p
}

pub fn run(reqs: &[Request], be: &mut dyn Backend, cfg: SchedConfig) -> SimResult {
    let t0 = std::time::Instant::now();
    let mut waiting: VecDeque<Pending> = VecDeque::new();
    let mut running: Vec<Running> = Vec::new();
    let mut next_arrival = 0usize;
    let mut step = 0u64;
    let mut r = SimResult {
        backend: be.name(),
        steps: 0,
        completed: 0,
        rejected: 0,
        generated_tokens: 0,
        prompt_tokens: reqs.iter().map(|q| q.prompt.len() as u64).sum(),
        prefill_computed_tokens: 0,
        prefix_cached_tokens: 0,
        preemptions: 0,
        recomputed_tokens: 0,
        mean_running: 0.0,
        peak_running: 0,
        mean_held_frac: 0.0,
        mean_stored_frac: 0.0,
        mean_waste_frac: 0.0,
        truncated: false,
        ttft_steps: Summary::default(),
        e2e_steps: Summary::default(),
        wall_ms: 0.0,
        counters: Vec::new(),
    };
    let mut ttft = Vec::new();
    let mut e2e = Vec::new();
    let (mut sum_run, mut sum_held, mut sum_stored, mut sum_waste) = (0.0, 0.0, 0.0, 0.0);
    let mut held_steps = 0u64;

    while (next_arrival < reqs.len() || !waiting.is_empty() || !running.is_empty()) && step < cfg.max_steps {
        while next_arrival < reqs.len() && reqs[next_arrival].arrival_step <= step {
            waiting.push_back(Pending {
                req: next_arrival,
                generated: 0,
                first_token_step: None,
            });
            next_arrival += 1;
        }
        let mut budget = cfg.max_batch_tokens;

        // 1. Decode.
        let mut i = 0;
        while i < running.len() {
            let q = &reqs[running[i].req];
            let tok = output_token(q.id, running[i].generated);
            let mut grown = false;
            loop {
                if be.grow(q.id, tok) {
                    grown = true;
                    break;
                }
                let v = running.pop().expect("non-empty");
                be.release(reqs[v.req].id);
                r.preemptions += 1;
                r.recomputed_tokens += reqs[v.req].prompt.len() as u64 + v.generated as u64;
                waiting.push_front(Pending {
                    req: v.req,
                    generated: v.generated,
                    first_token_step: v.first_token_step,
                });
                if running.len() == i {
                    break; // preempted ourselves
                }
            }
            if !grown {
                break;
            }
            let run = &mut running[i];
            run.generated += 1;
            run.first_token_step.get_or_insert(step);
            r.generated_tokens += 1;
            budget = budget.saturating_sub(1);
            let len = q.prompt.len() + run.generated as usize;
            be.computed(q.id, len);
            if run.generated >= q.output_len {
                let done = running.remove(i);
                be.release(q.id);
                r.completed += 1;
                ttft.push((done.first_token_step.unwrap_or(step) - q.arrival_step) as f64);
                e2e.push((step + 1 - q.arrival_step) as f64);
            } else {
                i += 1;
            }
        }

        // 2. Admit.
        let mut admitted_any = false;
        while let Some(p) = waiting.front() {
            if running.len() >= cfg.max_seqs {
                break;
            }
            let q = &reqs[p.req];
            let prompt = full_prompt(q, p.generated);
            let cost = (prompt.len() - be.peek_cached(q.namespace, &prompt)) as u64;
            if cost > budget && admitted_any {
                break;
            }
            let max_total = (q.prompt.len() + q.max_output as usize) as u64;
            match be.admit(q.id, q.namespace, &prompt, max_total) {
                Some(cached) => {
                    let p = waiting.pop_front().expect("front exists");
                    r.prefill_computed_tokens += (prompt.len() - cached) as u64;
                    r.prefix_cached_tokens += cached as u64;
                    budget = budget.saturating_sub((prompt.len() - cached) as u64);
                    be.computed(q.id, prompt.len());
                    running.push(Running {
                        req: p.req,
                        generated: p.generated,
                        first_token_step: p.first_token_step,
                    });
                    admitted_any = true;
                }
                None => {
                    if running.is_empty() && !admitted_any {
                        // Can never fit, even into empty memory.
                        waiting.pop_front();
                        r.rejected += 1;
                    }
                    break;
                }
            }
        }

        let s = be.sample();
        sum_run += running.len() as f64;
        r.peak_running = r.peak_running.max(running.len());
        sum_held += s.held_slots as f64 / s.capacity_slots as f64;
        sum_stored += s.stored_tokens as f64 / s.capacity_slots as f64;
        if s.held_slots > 0 {
            sum_waste += 1.0 - s.stored_tokens as f64 / s.held_slots as f64;
            held_steps += 1;
        }
        if cfg.check_every > 0 && step.is_multiple_of(cfg.check_every) {
            be.check()
                .unwrap_or_else(|e| panic!("invariant violated at step {step}: {e}"));
        }
        step += 1;
    }

    be.check()
        .unwrap_or_else(|e| panic!("invariant violated at end: {e}"));
    let n = step.max(1) as f64;
    r.steps = step;
    r.mean_running = sum_run / n;
    r.mean_held_frac = sum_held / n;
    r.mean_stored_frac = sum_stored / n;
    // Averaged over steps that hold memory, matching the documented definition.
    r.mean_waste_frac = sum_waste / held_steps.max(1) as f64;
    r.truncated = next_arrival < reqs.len() || !waiting.is_empty() || !running.is_empty();
    r.ttft_steps = Summary::of(&mut ttft);
    r.e2e_steps = Summary::of(&mut e2e);
    r.counters = be.counters();
    r.wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::{generate, Kind, Request, WorkloadSpec};

    fn cfg(blocks: u32, prefix: bool) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: blocks,
            enable_prefix_caching: prefix,
        }
    }

    #[test]
    fn all_backends_complete_every_request() {
        for kind in Kind::ALL {
            let reqs = generate(&WorkloadSpec {
                kind,
                requests: 60,
                arrival_rate: 1.0,
                seed: 11,
            });
            let sc = SchedConfig {
                check_every: 7,
                ..SchedConfig::default()
            };
            let blocks = 1200;
            let mut backends: Vec<Box<dyn Backend>> = vec![
                Box::new(ContiguousBackend::new(blocks as u64 * 16)),
                Box::new(PagedBackend::new(cfg(blocks, false))),
                Box::new(PagedBackend::new(cfg(blocks, true))),
            ];
            for be in backends.iter_mut() {
                let res = run(&reqs, be.as_mut(), sc);
                assert_eq!(
                    res.completed + res.rejected,
                    reqs.len(),
                    "{} {}",
                    kind.name(),
                    res.backend
                );
                let expected: u64 = reqs.iter().map(|q| q.output_len as u64).sum();
                assert!(res.generated_tokens >= expected, "{}", res.backend);
                let s = be.sample();
                assert_eq!(s.held_slots, 0, "{} leaked memory", res.backend);
            }
        }
    }

    #[test]
    fn prefix_caching_saves_prefill_on_shared_prompts() {
        let reqs = generate(&WorkloadSpec {
            kind: Kind::SharedPrefix,
            requests: 100,
            arrival_rate: 1.0,
            seed: 5,
        });
        let sc = SchedConfig::default();
        let off = run(&reqs, &mut PagedBackend::new(cfg(2000, false)), sc);
        let on = run(&reqs, &mut PagedBackend::new(cfg(2000, true)), sc);
        assert!(on.prefill_computed_tokens < off.prefill_computed_tokens / 2);
    }

    /// Delegates to a paged backend but fails the `fail_on`-th `grow` call once.
    struct FailOnce {
        inner: PagedBackend,
        grows: u32,
        fail_on: u32,
    }

    impl Backend for FailOnce {
        fn name(&self) -> String {
            "fail-once".into()
        }
        fn peek_cached(&self, ns: u64, p: &[u32]) -> usize {
            self.inner.peek_cached(ns, p)
        }
        fn admit(&mut self, rid: u64, ns: u64, p: &[u32], max: u64) -> Option<usize> {
            self.inner.admit(rid, ns, p, max)
        }
        fn computed(&mut self, rid: u64, n: usize) {
            self.inner.computed(rid, n)
        }
        fn grow(&mut self, rid: u64, t: u32) -> bool {
            self.grows += 1;
            self.grows != self.fail_on && self.inner.grow(rid, t)
        }
        fn release(&mut self, rid: u64) {
            self.inner.release(rid)
        }
        fn sample(&self) -> MemSample {
            self.inner.sample()
        }
        fn check(&self) -> Result<(), String> {
            self.inner.check()
        }
        fn counters(&self) -> Vec<(&'static str, Json)> {
            self.inner.counters()
        }
    }

    #[test]
    fn ttft_survives_preemption() {
        // Regression: TTFT used to restart at re-admission after a preemption.
        // One request, arrives at step 0, admitted at step 0, first token at step 1.
        // The 2nd grow (step 2) fails, so the request preempts itself and is re-admitted.
        let req = Request {
            id: 0,
            arrival_step: 0,
            namespace: 0,
            prompt: (0..40).collect(),
            output_len: 5,
            max_output: 8,
        };
        let mut be = FailOnce {
            inner: PagedBackend::new(cfg(64, true)),
            grows: 0,
            fail_on: 2,
        };
        let res = run(&[req], &mut be, SchedConfig::default());
        assert_eq!((res.completed, res.preemptions), (1, 1));
        assert!(!res.truncated);
        assert_eq!(
            res.ttft_steps.p50, 1.0,
            "TTFT must count the first token ever produced"
        );
        assert_eq!(res.generated_tokens, 5);
    }

    #[test]
    fn max_steps_truncation_is_reported() {
        let reqs = generate(&WorkloadSpec {
            kind: Kind::Short,
            requests: 50,
            arrival_rate: 0.0,
            seed: 1,
        });
        let sc = SchedConfig {
            max_steps: 5,
            ..SchedConfig::default()
        };
        let res = run(&reqs, &mut PagedBackend::new(cfg(400, false)), sc);
        assert!(res.truncated);
        assert!(res.completed < reqs.len());
    }

    #[test]
    fn tight_memory_preempts_but_finishes() {
        let reqs = generate(&WorkloadSpec {
            kind: Kind::Long,
            requests: 30,
            arrival_rate: 0.0,
            seed: 2,
        });
        let sc = SchedConfig {
            check_every: 3,
            ..SchedConfig::default()
        };
        let res = run(&reqs, &mut PagedBackend::new(cfg(400, true)), sc);
        assert_eq!(res.completed, 30);
        assert!(res.preemptions > 0);
    }
}
