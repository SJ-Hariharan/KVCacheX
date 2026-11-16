# ADR-0001: Fixed-size paged blocks instead of contiguous reservation

**Status:** accepted

**Context.** Request lengths are unknown at admission and vary widely. A contiguous allocator must
reserve `prompt + max_tokens` up front and suffers external fragmentation as requests finish in
arbitrary order (Kwon et al. 2023, §3).

**Decision.** Manage KV memory as `num_blocks` fixed-size blocks of `block_size` token slots, with a
per-sequence block table. Blocks are allocated only when the sequence reaches them. Default
`block_size = 16`, the same as vLLM.

**Consequences.** Waste is limited to under one block per sequence, and every free block is usable,
so there is no external fragmentation. The attention kernel must gather through a block table
(outside this crate's scope). `ContiguousAllocator` is kept as the measured baseline.
