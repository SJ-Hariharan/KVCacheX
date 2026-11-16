# ADR-0002: Block-granular prefix cache with exact verification (not a radix tree)

**Status:** accepted

**Context.** Prefix sharing needs (a) lookup of the longest cached prefix, (b) safe sharing, and
(c) eviction. Options considered:
1. vLLM-style chained block hashes, trusting the hash.
2. SGLang-style radix tree at token granularity.
3. Chained block hashes **plus exact verification** against a never-reused parent content id.

**Decision.** Option 3. Lookups are O(1) per block, as in option 1. Correctness does not depend on
hash quality, because each hit compares namespace, parent cid and the block's tokens. This lets us
use a fast non-cryptographic hash. Sharing is at full-block granularity, which matches the paged
allocator. A radix tree adds token-granular sharing, but partial blocks cannot be shared without
copy-on-write of the tail anyway, and the tree's maintenance cost buys little at block granularity.

**Consequences.** Partial tail blocks are never shared through the prefix cache. Up to
`block_size - 1` tokens per prompt are recomputed. Tokens of registered blocks are stored in
the pool (4 B × block_size per block), negligible next to the KV block (16 × 128 KiB for Llama-3-8B).
The `degenerate-hash` test feature proves collision safety.
