//! Property tests for the contiguous baseline allocator and the KV-memory geometry estimate.

use kvcachex::baseline::{ContiguousAllocator, Extent};
use kvcachex::geometry::{estimate, DType, ModelGeometry};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

#[derive(Debug, Clone)]
enum Op {
    Reserve(u64),
    Release(usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (1u64..=40).prop_map(Op::Reserve),
        any::<usize>().prop_map(Op::Release),
    ]
}

/// Free gaps implied by the model's held extents, as `(start, len)` in ascending order.
fn model_gaps(held: &[Extent], capacity: u64) -> Vec<(u64, u64)> {
    let mut sorted = held.to_vec();
    sorted.sort_by_key(|e| e.start);
    let mut gaps = Vec::new();
    let mut cursor = 0u64;
    for e in sorted {
        if e.start > cursor {
            gaps.push((cursor, e.start - cursor));
        }
        cursor = e.start + e.len;
    }
    if cursor < capacity {
        gaps.push((cursor, capacity - cursor));
    }
    gaps
}

fn check_state(
    a: &ContiguousAllocator,
    held: &[Extent],
    capacity: u64,
    model_failures: u64,
    model_peak: u64,
) -> Result<(), TestCaseError> {
    a.check_invariants().map_err(TestCaseError::fail)?;

    let mut sorted = held.to_vec();
    sorted.sort_by_key(|e| e.start);
    for e in &sorted {
        prop_assert!(e.len > 0, "zero-length held extent {:?}", e);
        prop_assert!(
            e.start + e.len <= capacity,
            "extent {:?} exceeds capacity {}",
            e,
            capacity
        );
    }
    for w in sorted.windows(2) {
        prop_assert!(
            w[0].start + w[0].len <= w[1].start,
            "held extents overlap: {:?} and {:?}",
            w[0],
            w[1]
        );
    }

    let sum: u64 = held.iter().map(|e| e.len).sum();
    prop_assert_eq!(a.capacity(), capacity);
    prop_assert_eq!(a.reserved(), sum);
    prop_assert_eq!(a.free_total(), capacity - sum);
    prop_assert!(
        a.largest_free() <= a.free_total(),
        "largest_free {} > free_total {}",
        a.largest_free(),
        a.free_total()
    );
    let frac = a.external_fragmentation();
    prop_assert!(
        (0.0..=1.0).contains(&frac),
        "external_fragmentation {} out of [0,1]",
        frac
    );
    prop_assert!(
        a.peak_reserved() >= a.reserved(),
        "peak {} < reserved {}",
        a.peak_reserved(),
        a.reserved()
    );
    prop_assert_eq!(a.peak_reserved(), model_peak);
    prop_assert_eq!(a.failures, model_failures);
    Ok(())
}

fn geom_with(
    num_layers: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    dtype: DType,
) -> ModelGeometry {
    ModelGeometry {
        name: "prop".to_string(),
        num_layers,
        num_query_heads,
        num_kv_heads,
        head_dim,
        dtype,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// Model-based test: the first-fit allocator must agree with a model that recomputes
    /// free gaps from the held extents, and must keep its invariants after every operation.
    #[test]
    fn contiguous_allocator_matches_model(
        capacity in 1u64..=200,
        ops in prop::collection::vec(op_strategy(), 0..200),
    ) {
        let mut a = ContiguousAllocator::new(capacity);
        let mut held: Vec<Extent> = Vec::new();
        let mut model_failures = 0u64;
        let mut model_peak = 0u64;

        for op in ops {
            match op {
                Op::Reserve(len) => {
                    // First-fit: lowest-address gap that can hold `len`.
                    let expected = model_gaps(&held, capacity)
                        .into_iter()
                        .find(|&(_, gap_len)| gap_len >= len)
                        .map(|(start, _)| start);
                    let got = a.reserve(len);
                    match (expected, got) {
                        (None, None) => model_failures += 1,
                        (Some(start), Some(e)) => {
                            prop_assert_eq!(e, Extent { start, len });
                            held.push(e);
                        }
                        (exp, got) => {
                            return Err(TestCaseError::fail(format!(
                                "reserve({len}): model expected {exp:?}, allocator returned {got:?}"
                            )));
                        }
                    }
                }
                Op::Release(i) => {
                    if !held.is_empty() {
                        let idx = i % held.len();
                        let e = held.swap_remove(idx);
                        a.release(e);
                    }
                }
            }
            let sum: u64 = held.iter().map(|e| e.len).sum();
            model_peak = model_peak.max(sum);
            check_state(&a, &held, capacity, model_failures, model_peak)?;
        }

        // Release everything; the space must coalesce back into one free extent.
        while let Some(e) = held.pop() {
            a.release(e);
            let sum: u64 = held.iter().map(|e| e.len).sum();
            check_state(&a, &held, capacity, model_failures, model_peak.max(sum))?;
        }
        prop_assert_eq!(a.reserved(), 0);
        prop_assert_eq!(a.largest_free(), capacity);
        prop_assert_eq!(a.external_fragmentation(), 0.0);
    }

    /// Paged rounding wastes less than one block per non-empty sequence, and the
    /// contiguous reservation (max length per sequence) always covers the exact bytes.
    #[test]
    fn estimate_paged_slack_is_below_one_block_per_sequence(
        seq_lens in prop::collection::vec(0u64..5000, 0..=64),
        block_size in 1u32..=64,
    ) {
        let geom = ModelGeometry::preset("llama-3-8b").expect("preset exists");
        let bpt = geom.bytes_per_token();
        let bs = block_size as u64;
        let max_len = seq_lens.iter().copied().max().unwrap_or(0);

        let e = estimate(&geom, block_size, &seq_lens, max_len);

        prop_assert_eq!(e.total_tokens, seq_lens.iter().sum::<u64>());
        prop_assert_eq!(e.exact_bytes, e.total_tokens * bpt);
        prop_assert!(e.exact_bytes <= e.paged_bytes, "exact {} > paged {}", e.exact_bytes, e.paged_bytes);
        prop_assert_eq!(e.paged_bytes, e.blocks * bs * bpt);

        let nonzero = seq_lens.iter().filter(|&&l| l > 0).count() as u64;
        let slack = e.paged_bytes - e.exact_bytes;
        if nonzero == 0 {
            prop_assert_eq!(slack, 0);
        } else {
            prop_assert!(
                slack < nonzero * bs * bpt,
                "slack {} not below {} sequences x one block ({} B)",
                slack,
                nonzero,
                bs * bpt
            );
        }

        let frac = e.paged_internal_fragmentation();
        prop_assert!((0.0..1.0).contains(&frac), "paged_internal_fragmentation {} not in [0,1)", frac);

        let n = seq_lens.len() as u64;
        prop_assert_eq!(e.reserved_bytes, n * max_len * bpt);
        prop_assert!(
            e.reserved_bytes >= e.exact_bytes,
            "reserved {} < exact {}",
            e.reserved_bytes,
            e.exact_bytes
        );
    }

    /// K and V are stored per KV head, so doubling `num_kv_heads` doubles bytes per token.
    #[test]
    fn bytes_per_token_scales_linearly_in_kv_heads(
        num_layers in 1u32..=128,
        num_kv_heads in 1u32..=4096,
        head_dim in prop::sample::select(vec![32u32, 64, 80, 96, 128, 256]),
        dtype in prop::sample::select(vec![DType::F32, DType::F16, DType::BF16, DType::Fp8]),
    ) {
        // Query heads are a multiple of KV heads in both geometries, so both are valid.
        // Query heads do not affect the cache size.
        let num_query_heads = num_kv_heads * 4;
        let base = geom_with(num_layers, num_query_heads, num_kv_heads, head_dim, dtype);
        let doubled = geom_with(num_layers, num_query_heads, num_kv_heads * 2, head_dim, dtype);
        prop_assert!(base.validate().is_ok());
        prop_assert!(doubled.validate().is_ok());

        let expected_base =
            2 * num_layers as u64 * num_kv_heads as u64 * head_dim as u64 * dtype.bytes();
        prop_assert_eq!(base.bytes_per_token(), expected_base);
        prop_assert_eq!(doubled.bytes_per_token(), 2 * base.bytes_per_token());
    }
}
