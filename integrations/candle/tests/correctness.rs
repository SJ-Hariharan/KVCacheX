//! KVCacheX-backed inference vs Candle's native KV cache, on the same model and weights.
//! Synthetic weights (deterministic, no download), GQA test config, tiny blocks so every test
//! crosses many block boundaries.

use kvcachex::{CacheConfig, KvStorage};
use kvcachex_candle::weights::{synthetic, test_config};
use kvcachex_candle::{argmax, Engine, KvxEngine, NativeEngine, Prefill};

fn native() -> NativeEngine {
    NativeEngine::new(synthetic(&test_config(), 7).unwrap(), test_config()).unwrap()
}

fn kvx(block_size: u32, num_blocks: u32, prefix: bool) -> KvxEngine {
    let kv = CacheConfig {
        block_size,
        num_blocks,
        enable_prefix_caching: prefix,
    };
    KvxEngine::new(synthetic(&test_config(), 7).unwrap(), test_config(), kv).unwrap()
}

/// Greedy run: prefill, then `steps` decode steps. Returns (generated tokens, logits per step, prefill).
fn generate(
    e: &mut dyn Engine,
    id: usize,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<Vec<f32>>, Prefill) {
    let (mut logits, pre) = e.prefill(id, prompt).unwrap();
    let (mut toks, mut all) = (Vec::new(), vec![logits.clone()]);
    for _ in 0..steps {
        let t = argmax(&logits);
        toks.push(t);
        logits = e.decode(id, t).unwrap();
        all.push(logits.clone());
    }
    (toks, all, pre)
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            assert_eq!(x.len(), y.len());
            x.iter().zip(y).map(|(p, q)| (p - q).abs())
        })
        .fold(0.0, f32::max)
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 7 + seed * 13 + 1) % 64).collect()
}

#[test]
fn paged_cache_is_bit_identical_to_native_cache() {
    let p = prompt(1, 23);
    let (nt, nl, _) = generate(&mut native(), 0, &p, 40);
    let mut k = kvx(4, 64, false);
    let (kt, kl, pre) = generate(&mut k, 0, &p, 40);
    assert_eq!(
        pre,
        Prefill {
            computed_tokens: 23,
            cached_tokens: 0
        }
    );
    assert_eq!(nt, kt);
    // Same ops on the same K/V values: no tolerance needed.
    assert_eq!(max_diff(&nl, &kl), 0.0);
    k.cache().manager().check_invariants().unwrap();
}

#[test]
fn prefix_hit_skips_cached_positions_and_matches_native() {
    let a = prompt(2, 30);
    let mut b = a[..21].to_vec();
    b.extend([5, 9, 33, 2, 60]);
    let mut k = kvx(4, 64, true);
    generate(&mut k, 0, &a, 10);
    k.finish(0).unwrap();
    let (kt, kl, pre) = generate(&mut k, 1, &b, 20);
    // 21 shared tokens = 5 full blocks of 4 are reused; only positions 20..26 run.
    assert_eq!(
        pre,
        Prefill {
            computed_tokens: 6,
            cached_tokens: 20
        }
    );
    let (nt, nl, _) = generate(&mut native(), 1, &b, 20);
    assert_eq!(nt, kt);
    // Cached K/V came from a different prefill shape (30 tokens at once), so allow rounding.
    let d = max_diff(&nl, &kl);
    assert!(d < 1e-4, "max |logit diff| {d}");
    assert!(k.stats().counters.prefix_hit_tokens >= 20);
    k.cache().manager().check_invariants().unwrap();
}

#[test]
fn fully_cached_prompt_reruns_only_last_position() {
    let a = prompt(3, 32); // exactly 8 blocks
    let mut k = kvx(4, 64, true);
    let (t1, l1, _) = generate(&mut k, 0, &a, 8);
    k.finish(0).unwrap();
    let (t2, l2, pre) = generate(&mut k, 1, &a, 8);
    assert_eq!(
        pre,
        Prefill {
            computed_tokens: 1,
            cached_tokens: 32
        }
    );
    let (nt, nl, _) = generate(&mut native(), 0, &a, 8);
    assert_eq!(t1, nt);
    assert_eq!(t2, nt);
    assert_eq!(max_diff(&l1, &nl), 0.0);
    assert!(max_diff(&l2, &nl) < 1e-4);
}

#[test]
fn fork_copy_on_write_matches_native_fork() {
    let p = prompt(4, 18); // last block partially filled and shared after the fork
    let mut k = kvx(4, 64, true);
    let mut n = native();
    let mut out = Vec::new();
    for e in [&mut n as &mut dyn Engine, &mut k as &mut dyn Engine] {
        e.prefill(0, &p).unwrap();
        e.fork(0, 1).unwrap();
        // Different next tokens: the shared tail block must be copied, not overwritten.
        let mut la = vec![e.decode(0, 11).unwrap()];
        let mut lb = vec![e.decode(1, 42).unwrap()];
        for _ in 0..12 {
            let ta = argmax(la.last().unwrap());
            la.push(e.decode(0, ta).unwrap());
            let tb = argmax(lb.last().unwrap());
            lb.push(e.decode(1, tb).unwrap());
        }
        out.push((la, lb));
    }
    assert_eq!(max_diff(&out[0].0, &out[1].0), 0.0);
    assert_eq!(max_diff(&out[0].1, &out[1].1), 0.0);
    assert!(k.stats().counters.cow_copies >= 1);
    k.finish(0).unwrap();
    k.finish(1).unwrap();
    k.cache().manager().check_invariants().unwrap();
}

#[test]
fn interleaved_requests_and_eviction_match_native() {
    // A pool too small to keep every finished prefix: cached blocks get evicted and reused.
    let mut k = kvx(4, 24, true);
    let mut n = native();
    let shared = prompt(5, 12);
    let reqs: Vec<Vec<u32>> = (0..12u32)
        .map(|i| {
            let mut p = if i % 3 == 0 {
                prompt(10 + i, 9)
            } else {
                shared.clone()
            };
            p.extend(prompt(20 + i, 3 + (i as usize % 4)));
            p
        })
        .collect();
    // Two requests in flight at a time, decode steps interleaved round-robin.
    for pair in reqs.chunks(2) {
        let mut state = Vec::new();
        for (j, p) in pair.iter().enumerate() {
            let (ln, _) = n.prefill(j, p).unwrap();
            let (lk, _) = k.prefill(j, p).unwrap();
            state.push((ln, lk));
        }
        for _ in 0..8 {
            for (j, (ln, lk)) in state.iter_mut().enumerate() {
                assert_eq!(argmax(ln), argmax(lk));
                assert!(max_diff(std::slice::from_ref(ln), std::slice::from_ref(lk)) < 1e-4);
                let t = argmax(ln);
                *ln = n.decode(j, t).unwrap();
                *lk = k.decode(j, t).unwrap();
            }
        }
        for j in 0..pair.len() {
            n.finish(j).unwrap();
            k.finish(j).unwrap();
        }
        k.cache().manager().check_invariants().unwrap();
    }
    let s = k.stats();
    assert!(s.counters.evictions > 0, "pool should have evicted cached blocks");
    assert!(s.counters.prefix_hit_tokens > 0);
    assert_eq!(s.used_blocks, 0);
    assert_eq!(n.kv_bytes_in_use(), 0);
}

#[test]
fn out_of_blocks_is_an_error_not_a_crash() {
    let mut k = kvx(4, 4, false);
    assert!(k.prefill(0, &prompt(6, 17)).is_err()); // needs 5 blocks
    assert!(k.prefill(1, &prompt(6, 16)).is_ok());
    k.cache().manager().check_invariants().unwrap();
}

#[test]
fn kv_byte_accounting() {
    let cfg = test_config();
    let per_token = cfg.n_layers * 2 * (cfg.dim / cfg.n_heads) * cfg.n_kv_heads * 4;
    let p = prompt(7, 23);
    let mut n = native();
    n.prefill(0, &p).unwrap();
    assert_eq!(n.kv_bytes_in_use(), 23 * per_token);
    let mut k = kvx(4, 64, false);
    k.prefill(0, &p).unwrap();
    assert_eq!(k.kv_bytes_in_use(), 6 * 4 * per_token); // 23 tokens -> 6 blocks of 4
    assert_eq!(k.kv_bytes_allocated(), 64 * 4 * per_token);
    assert_eq!(
        k.kv_bytes_allocated(),
        k.cache().storage().layout().total_bytes().unwrap()
    );
}
