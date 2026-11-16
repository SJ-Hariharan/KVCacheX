//! Multi-threaded stress test of [`SharedCacheManager`]: many threads create, compute, append,
//! fork and release sequences that share prefixes, under memory pressure. Invariants are
//! checked periodically under the lock and at the end.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use kvcachex::{CacheConfig, CacheError, SharedCacheManager};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn concurrent_lifecycle_preserves_invariants() {
    let threads = 8;
    let iters = 4_000;
    let shared = SharedCacheManager::new(CacheConfig {
        block_size: 8,
        num_blocks: 96,
        enable_prefix_caching: true,
    });
    let ooms = AtomicU64::new(0);

    thread::scope(|s| {
        for t in 0..threads {
            let shared = shared.clone();
            let ooms = &ooms;
            s.spawn(move || {
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (t as u64 + 1));
                let mut mine = Vec::new();
                for i in 0..iters {
                    match rng.below(10) {
                        0..=3 => {
                            let base = rng.below(4) as u32;
                            let mut p: Vec<u32> = (0..24).map(|k| base * 1000 + k).collect();
                            p.extend((0..rng.below(12)).map(|_| rng.below(50) as u32));
                            match shared.create_sequence(base as u64 % 2, &p) {
                                Ok(a) => {
                                    shared.mark_computed(a.seq, p.len()).unwrap();
                                    mine.push(a.seq);
                                }
                                Err(CacheError::OutOfBlocks { .. }) => {
                                    ooms.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(e) => panic!("{e}"),
                            }
                        }
                        4..=5 if !mine.is_empty() => {
                            let seq = mine[rng.below(mine.len() as u64) as usize];
                            let n = 1 + rng.below(9) as usize;
                            let toks: Vec<u32> = (0..n).map(|_| rng.below(50) as u32).collect();
                            if shared.append_tokens(seq, &toks).is_ok() {
                                let len = shared.lock().len(seq).unwrap();
                                shared.mark_computed(seq, len).unwrap();
                            }
                        }
                        6 if !mine.is_empty() => {
                            let seq = mine[rng.below(mine.len() as u64) as usize];
                            mine.push(shared.fork(seq).unwrap());
                        }
                        _ if !mine.is_empty() => {
                            let k = rng.below(mine.len() as u64) as usize;
                            shared.release(mine.swap_remove(k)).unwrap();
                        }
                        _ => {}
                    }
                    if i % 500 == 0 {
                        shared.lock().check_invariants().unwrap();
                    }
                }
                for seq in mine {
                    shared.release(seq).unwrap();
                }
            });
        }
    });

    let m = shared.lock();
    m.check_invariants().unwrap();
    let s = m.stats();
    assert_eq!(s.used_blocks, 0);
    assert_eq!(s.live_sequences, 0);
    assert!(
        s.counters.prefix_hit_blocks > 0,
        "workload should produce prefix hits"
    );
    assert!(
        ooms.load(Ordering::Relaxed) > 0,
        "workload should hit memory pressure"
    );
}
