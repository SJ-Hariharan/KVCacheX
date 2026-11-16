//! Model-based property tests on the CPU backend. Harness: `tests/common/mod.rs`.

mod common;

use common::{op, run};
use kvcachex::PagedKvCache;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn random_ops_tight_memory_prefix(ops in prop::collection::vec(op(), 1..80)) {
        run(ops, 10, true, PagedKvCache::new_cpu);
    }

    #[test]
    fn random_ops_roomy_memory_prefix(ops in prop::collection::vec(op(), 1..120)) {
        run(ops, 64, true, PagedKvCache::new_cpu);
    }

    #[test]
    fn random_ops_no_prefix(ops in prop::collection::vec(op(), 1..80)) {
        run(ops, 12, false, PagedKvCache::new_cpu);
    }
}
