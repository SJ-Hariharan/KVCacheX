# ADR-0004: Single-owner manager, coarse mutex for sharing

**Status:** accepted (revisit only with profiling evidence)

**Context.** Each metadata operation takes tens to hundreds of nanoseconds. In serving engines a
single scheduler thread makes allocation decisions for a batch.

**Decision.** `CacheManager` takes `&mut self`. `SharedCacheManager` wraps it in one `std::sync::Mutex`,
and exposes `lock()` so callers can batch many operations under one acquisition. No lock-free
structures, no `unsafe` (`unsafe_code = "forbid"`).

**Measured consequence** (results/REPORT.md, M1): throughput falls as threads are added. In the
report, per-op locking goes from 4.9 M ops/s on 1 thread to 0.91 M on 8. Batching 11 ops
per acquisition reaches 1.43 M at 8 threads (about 1.6× per-op) but still does not scale. Single-thread
numbers are noisy between runs (3–6 M ops/s observed), so the per-op vs batched order at 1 thread is
not meaningful. Thread safety is real and tested;
parallel speedup is not claimed. Sharding by namespace or a lock-free free queue would be the next
step, only if a real engine shows the manager on its critical path.
