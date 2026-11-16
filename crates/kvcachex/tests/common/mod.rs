//! Shared model-based harness: random operation sequences against a storage-backed cache.
//!
//! The KV value written for position `p` is a function of the namespace and *all* tokens
//! `0..=p`, as real attention KV depends on the whole prefix. After every operation, every
//! computed position of every live sequence is read back (through `read_into`, so device
//! backends really download it) and compared. Any wrong prefix sharing, use-after-free, missed
//! copy-on-write or eviction of a pinned block shows up as a value mismatch. Manager invariants
//! are checked after every step. Used by `model_based.rs` (CPU) and `gpu.rs` (CUDA).
#![allow(dead_code)]

use std::collections::HashSet;

use kvcachex::{geometry::ModelGeometry, CacheConfig, CacheError, KvStorage, PagedKvCache, SeqId};
use proptest::prelude::*;

pub const BS: u32 = 4;

fn kv_value(ns: u64, prefix: &[u32], layer: u32) -> (f32, f32) {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ ns;
    for &t in prefix {
        h = (h ^ t as u64).wrapping_mul(0x100_0000_01b3);
    }
    h = (h ^ layer as u64).wrapping_mul(0x100_0000_01b3);
    ((h % 1_000_003) as f32, ((h >> 32) % 1_000_003) as f32)
}

#[derive(Debug, Clone)]
pub enum Op {
    Create { ns: u64, base: u8, suffix: Vec<u32> },
    Append { who: usize, n: usize },
    Compute { who: usize },
    Fork { who: usize },
    Release { who: usize },
    ClearCache,
}

pub fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (0u64..2, 0u8..3, prop::collection::vec(0u32..3, 0..10))
            .prop_map(|(ns, base, suffix)| Op::Create { ns, base, suffix }),
        3 => (any::<usize>(), 1usize..7).prop_map(|(who, n)| Op::Append { who, n }),
        4 => any::<usize>().prop_map(|who| Op::Compute { who }),
        1 => any::<usize>().prop_map(|who| Op::Fork { who }),
        2 => any::<usize>().prop_map(|who| Op::Release { who }),
        1 => Just(Op::ClearCache),
    ]
}

/// Three shared "system prompts" so prefix hits are frequent.
fn base_prompt(base: u8) -> Vec<u32> {
    let len = [8usize, 6, 12][base as usize];
    (0..len as u32).map(|i| 100 * (base as u32 + 1) + i).collect()
}

struct Model {
    seqs: Vec<(SeqId, u64)>,
    /// (namespace, full-block prefix) that has been computed at some point.
    computed_prefixes: HashSet<(u64, Vec<u32>)>,
    next_token: u32,
}

fn compute_all<S: KvStorage>(c: &mut PagedKvCache<S>, layers: u32, seq: SeqId, ns: u64, model: &mut Model) {
    let toks = c.manager().tokens(seq).unwrap().to_vec();
    let from = c.manager().computed(seq).unwrap();
    for p in from..toks.len() {
        for l in 0..layers {
            let (k, v) = kv_value(ns, &toks[..=p], l);
            let w = c.storage().token_width();
            c.write(seq, p, l, &vec![k; w], &vec![v; w]).unwrap();
        }
    }
    c.mark_computed(seq, toks.len()).unwrap();
    for b in 1..=toks.len() / BS as usize {
        model
            .computed_prefixes
            .insert((ns, toks[..b * BS as usize].to_vec()));
    }
}

fn verify<S: KvStorage>(c: &PagedKvCache<S>, layers: u32, model: &Model) {
    c.manager().check_invariants().unwrap();
    let w = c.storage().token_width();
    let (mut k, mut v) = (vec![0.0f32; w], vec![0.0f32; w]);
    for &(seq, ns) in &model.seqs {
        let toks = c.manager().tokens(seq).unwrap();
        let computed = c.manager().computed(seq).unwrap();
        for p in 0..computed {
            for l in 0..layers {
                c.read_into(seq, p, l, &mut k, &mut v).unwrap();
                let (ek, ev) = kv_value(ns, &toks[..=p], l);
                assert!(
                    k.iter().all(|&x| x == ek) && v.iter().all(|&x| x == ev),
                    "corrupt KV: {seq} pos {p} layer {l}"
                );
            }
        }
    }
}

/// Runs `ops` against a cache built by `make` (any storage backend) and checks every computed
/// position's KV after every operation.
pub fn run<S, F>(ops: Vec<Op>, num_blocks: u32, prefix: bool, make: F)
where
    S: KvStorage,
    F: FnOnce(&ModelGeometry, CacheConfig) -> PagedKvCache<S>,
{
    let geom = ModelGeometry::preset("tiny").unwrap();
    let layers = geom.num_layers;
    let mut c = make(
        &geom,
        CacheConfig {
            block_size: BS,
            num_blocks,
            enable_prefix_caching: prefix,
        },
    );
    let mut m = Model {
        seqs: Vec::new(),
        computed_prefixes: HashSet::new(),
        next_token: 1000,
    };

    for op in ops {
        let pick = |who: usize, m: &Model| (!m.seqs.is_empty()).then(|| who % m.seqs.len());
        match op {
            Op::Create { ns, base, suffix } => {
                let mut prompt = base_prompt(base);
                prompt.extend(suffix);
                let free_before = c.manager().num_free_blocks();
                let need = c.manager().blocks_needed(ns, &prompt);
                match c.create_sequence(ns, &prompt) {
                    Ok(a) => {
                        assert_eq!(a.cached_tokens % BS as usize, 0);
                        assert!(a.cached_tokens <= prompt.len());
                        assert_eq!(free_before - c.manager().num_free_blocks(), need);
                        if a.cached_tokens > 0 {
                            // Soundness: a hit must be a prefix someone actually computed.
                            assert!(
                                m.computed_prefixes
                                    .contains(&(ns, prompt[..a.cached_tokens].to_vec())),
                                "hit on never-computed prefix"
                            );
                        }
                        m.seqs.push((a.seq, ns));
                    }
                    Err(CacheError::OutOfBlocks { needed, available }) => {
                        assert!(needed > available);
                        assert_eq!(needed, need);
                        assert_eq!(c.manager().num_free_blocks(), free_before);
                    }
                    Err(e) => panic!("unexpected {e}"),
                }
            }
            Op::Append { who, n } => {
                if let Some(i) = pick(who, &m) {
                    let seq = m.seqs[i].0;
                    let len = c.manager().len(seq).unwrap();
                    let toks: Vec<u32> = (0..n as u32).map(|k| m.next_token + k).collect();
                    m.next_token += n as u32;
                    match c.append_tokens(seq, &toks) {
                        Ok(()) => assert_eq!(c.manager().len(seq).unwrap(), len + n),
                        Err(CacheError::OutOfBlocks { .. }) => {
                            assert_eq!(c.manager().len(seq).unwrap(), len)
                        }
                        Err(e) => panic!("unexpected {e}"),
                    }
                }
            }
            Op::Compute { who } => {
                if let Some(i) = pick(who, &m) {
                    let (seq, ns) = m.seqs[i];
                    compute_all(&mut c, layers, seq, ns, &mut m);
                }
            }
            Op::Fork { who } => {
                if let Some(i) = pick(who, &m) {
                    let (seq, ns) = m.seqs[i];
                    match c.fork(seq) {
                        Ok(child) => {
                            assert_eq!(
                                c.manager().tokens(child).unwrap(),
                                c.manager().tokens(seq).unwrap()
                            );
                            m.seqs.push((child, ns));
                        }
                        Err(CacheError::NotFullyComputed { .. }) => {}
                        Err(e) => panic!("unexpected {e}"),
                    }
                }
            }
            Op::Release { who } => {
                if let Some(i) = pick(who, &m) {
                    let (seq, _) = m.seqs.swap_remove(i);
                    c.release(seq).unwrap();
                    assert_eq!(c.release(seq), Err(CacheError::UnknownSequence(seq)));
                }
            }
            Op::ClearCache => {
                let cached_free = c.manager().pool().num_cached_free();
                assert_eq!(c.clear_prefix_cache(), cached_free);
                assert_eq!(c.manager().pool().num_cached_free(), 0);
                // Prefixes held only by free blocks are gone; owned ones stay findable.
                m.computed_prefixes.clear();
                for &(seq, ns) in &m.seqs {
                    let toks = c.manager().tokens(seq).unwrap();
                    let full = c.manager().computed(seq).unwrap() / BS as usize;
                    for b in 1..=full {
                        m.computed_prefixes.insert((ns, toks[..b * BS as usize].to_vec()));
                    }
                }
            }
        }
        verify(&c, layers, &m);
    }

    for (seq, _) in m.seqs.drain(..) {
        c.release(seq).unwrap();
    }
    let s = c.manager().stats();
    assert_eq!(s.used_blocks, 0, "all blocks returned after releasing everything");
    assert_eq!(s.free_blocks, num_blocks);
    c.manager().check_invariants().unwrap();
}
