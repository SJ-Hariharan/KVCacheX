//! Profiling driver: repeated prefix-hit admissions (used with `sample`/Instruments).
use kvcachex::{CacheConfig, CacheManager};
use std::time::Instant;

fn main() {
    let blocks: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 16);
    let mut m = CacheManager::new(CacheConfig {
        block_size: 16,
        num_blocks: blocks,
        enable_prefix_caching: true,
    });
    let sys: Vec<u32> = (0..1024).map(|i| 0x4000_0000 + i).collect();
    let a = m.create_sequence(0, &sys).unwrap();
    m.mark_computed(a.seq, 1024).unwrap();
    let mut next = 1u32;
    let iters = 2_000_000u32;
    let t = Instant::now();
    let mut lookups = 0usize;
    for _ in 0..iters {
        let mut p = sys.clone();
        p.extend(next..next + 64);
        next += 64;
        let b = m.create_sequence(0, &p).unwrap();
        lookups += b.cached_tokens;
        m.release(b.seq).unwrap();
    }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    println!("blocks={blocks} create+release with 64-block hit: {ns:.0} ns/iter ({lookups})");
}
