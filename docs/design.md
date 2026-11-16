# KVCacheX design

## 1. Problem

In autoregressive decoding, every layer re-reads the Key/Value vectors of all previous tokens.
Engines keep them in a **KV cache** instead of recomputing them. Its size is

```
bytes/token = 2 (K,V) × layers × kv_heads × head_dim × bytes/element
```

That is 512 KiB/token for Llama-2-7B (MHA, f16) and 128 KiB/token for Llama-3-8B (GQA, 8 KV heads).
A single 4k-token request on Llama-3-8B needs 512 MiB. KV memory, not compute, usually limits how
many requests run in parallel. That makes the cache's memory manager a throughput-critical component:

| problem | consequence | KVCacheX answer |
|---|---|---|
| final length unknown at admission | contiguous allocators must reserve `max_tokens` | fixed-size blocks allocated on demand |
| variable lengths, frees in any order | external fragmentation | all blocks interchangeable; no external fragmentation |
| identical prefixes (system prompts, parallel samples) | duplicated KV, duplicated prefill | ref-counted sharing and exact prefix cache |
| memory full | requests stall | LRU eviction of unreferenced cached blocks; scheduler preemption |

This is the **attention KV cache** inside an inference engine. It is unrelated to API-level
"prompt caching" products, although those are usually built on a prefix cache like this one.

## 2. Research survey and what we took

| technique | problem solved | mechanism | trade-off | used here? |
|---|---|---|---|---|
| PagedAttention / vLLM [1] | reservation waste, fragmentation | fixed-size KV blocks + per-sequence block table; attention kernel gathers through the table | indirection in the kernel; slack ≤ 1 block/seq | **yes, core design** |
| vLLM automatic prefix caching [2] | duplicated prefixes across requests | hash of (parent hash, block tokens, extra keys); only full blocks cached; freed blocks keep their content in an LRU free queue, released in reverse order | hash table per block; partial blocks not shared | **yes, adapted** (exact verification, see §4) |
| SGLang RadixAttention [3] | prefix reuse at token granularity, multi-call programs | radix tree over token sequences, LRU leaf eviction | tree maintenance; token-granular KV slots | considered; rejected for v1 (ADR-0002) |
| H2O [4] | KV size for long contexts | evict tokens with low cumulative attention ("heavy hitters" kept) | lossy; needs attention scores from the kernel | no: changes model output, needs a real model |
| StreamingLLM [5] | infinite-length streaming | keep attention-sink tokens + recent window | lossy for mid-context recall | no: same reason; future work as a policy hook |
| KIVI [6] | KV bytes per element | 2-bit asymmetric quantization (K per-channel, V per-token) | dequant cost, accuracy | no: needs real tensors and kernels; geometry models fp8 sizing |
| GQA [7] / MQA [8] | KV size | share KV heads across query heads | model architecture choice | modeled in `geometry` |

H2O, StreamingLLM and KIVI change *what* is stored and affect model quality. Evaluating them
honestly requires a real model and attention kernels, which this CPU-only, allocator-level
project does not have. They are documented as future policy and storage extensions, not implemented.

## 3. Architecture

```
            ┌──────────────── kvx (CLI) ─────────────────┐
            │ estimate · simulate (scheduler) · bench    │
            └───────────────┬────────────────────────────┘
                            │ uses
┌───────────────────────────▼──────────────────── kvcachex (lib) ───────────────────────┐
│  PagedKvCache<S: KvStorage>  ── checked writes/reads, applies COW copies, poisoning   │
│        │                                   │                                          │
│  CacheManager (metadata)               KvStorage trait ── CpuKvStorage (reference)    │
│   · sequences: SeqId → {tokens, block table, computed, published}   └ CudaKvStorage   │
│                                                              (feature `cuda`, device) │
│   · lifecycle: create / append / mark_computed / fork / release                       │
│        │                                                                              │
│  BlockPool                                                                            │
│   · Meta[num_blocks]: ref_count, generation, filled, cache tag, LRU links             │
│   · free queue (intrusive doubly linked; clean at head, cached at tail)               │
│   · prefix index: u64 hash → bucket chain of blocks; exact verification               │
│  SharedCacheManager = Arc<Mutex<CacheManager>>                                        │
│  baseline::ContiguousAllocator (comparison only)                                      │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Metadata and tensor storage are separate on purpose. `CacheManager` never touches tensor
bytes. It returns block ids and `BlockCopy` instructions that a storage backend carries out:
host memory (`CpuKvStorage`) or CUDA device memory (`CudaKvStorage`, feature `cuda`, with
device-to-device COW copies). This is the boundary a runtime like MiniInfer would integrate against.

## 4. Key decisions

### 4.1 Block pool and ownership
* `BlockId { index, generation }`. `generation` increases on every fresh allocation, so a
  handle kept after its block was evicted and reused is detected (`is_live`, `decref` asserts).
  It wraps after 2³² reuses of one block. That is accepted and documented.
* Reference counts use `checked_add` (overflow panics). `decref` asserts against underflow
  and stale handles. These are caller bugs, and continuing would corrupt shared state.
* **One free queue holds every block with `ref_count == 0`.** Clean blocks go to the head and
  cached blocks to the tail. Allocation pops the head, so clean blocks are used before any
  cached content is evicted, and cached blocks are evicted in LRU order. A prefix hit on a free
  cached block unlinks it in O(1). Each block-level step (allocate, revive, ref/deref, evict)
  is O(1); sequence operations are O(blocks touched), and prefix matching is O(prompt blocks).
* Within a request, blocks are released tail-first (as in vLLM), so the deepest blocks of a
  chain are evicted before their ancestors.

### 4.2 Exact prefix caching (ADR-0002)
A registered block stores `(namespace, parent_cid, tokens)` plus a fresh, never reused
**content id** (`cid`). Lookup probes the bucket for `hash(namespace, parent_cid, tokens)` and
accepts a block only if all three fields match exactly. `parent_cid` names one exact
predecessor, so by induction from the root (cid 0) a match at depth *i* implies the whole
*i*-block prefix is identical. **Hash collisions cannot cause wrong sharing.** Most
implementations trust a hash of the parent chain and accept a tiny collision probability.
KVCacheX pays one `memcmp` of the block's tokens per hit instead. The `degenerate-hash`
feature forces every block into one bucket. The full library test suite passes under it locally,
and the CI workflow runs it on every push (results are on the repository's Actions page).

Compatibility: one manager serves one model and KV layout, so geometry and dtype are fixed by
construction. Everything else that changes KV for the same tokens (LoRA adapter, multimodal
input, cache salt) must be encoded in the caller's `namespace`.

### 4.3 Publish only computed blocks
`create_sequence` allocates slots, but blocks enter the prefix index only after
`mark_computed` covers them *and* they are full. A request arriving while another is still
in prefill therefore cannot hit (and read) unwritten KV. Duplicates computed concurrently stay
private, and the chain continues through the canonical copy's cid.

### 4.4 Write safety
Positions `< computed` are read-only. Uncomputed positions are always in blocks with
`ref_count == 1` that are not in the prefix index. `fork` requires a fully computed parent,
and appending to a shared partial tail triggers copy-on-write. `check_invariants` verifies all
of this, and `PagedKvCache::write` refuses writes to computed positions.

### 4.5 Failure atomicity
Every fallible operation computes its block demand first (fresh blocks + cached free blocks
it would revive) and returns `OutOfBlocks` before mutating anything. Hits are pinned before
fresh blocks are taken, so an allocation can never evict the prefix it is about to use.

### 4.6 Concurrency (ADR-0004)
`CacheManager` is `&mut self` and single-threaded. `SharedCacheManager` wraps it in one
`Mutex`, which is correct and stress-tested with 8 threads. It is **not** a parallel speedup:
throughput falls as threads are added (see the report). The intended model, as in vLLM, is one
scheduler thread owning the manager and making all decisions for a batch.

## 5. Storage layout and backends
Both backends use one allocation. Each block is `[layer][K|V][slot][kv_head × head_dim]` in `f32`,
so one block's K plane for a layer is contiguous, as a paged attention kernel would read it, and
consecutive slots of one (layer, K/V) plane move in one transfer. Elements are f32 regardless of
the modeled dtype.

The `KvStorage` contract (ADR-0005) is fallible and transfer-explicit:
* `write_slots`, `read_slots_into`, `copy_block` and `synchronize` return `StorageError`;
* reads copy into caller buffers, so there are no borrowed views of device memory;
* argument checks happen before any transfer.

`PagedKvCache` poisons itself after a fatal device error rather than serve KV that a failed
copy-on-write may have left inconsistent.

| backend | where KV lives | write | read | COW copy | validated |
|---|---|---|---|---|---|
| `CpuKvStorage` | host `Vec<f32>` | memcpy | memcpy (or zero-copy borrow) | memcpy of the copied slots | yes (all CPU suites) |
| `CudaKvStorage` (`cuda`) | one device allocation (`CudaSlice<f32>`) | H2D | D2H + sync | **D2D**, whole block | yes, Colab Tesla T4, 9/9 device tests in two runs (`results/gpu/`) |

## 6. Invariants (checked by `check_invariants`, tests, and the simulator's `--check-every`)
1. `used + free == total`; the free queue holds exactly the blocks with `ref_count == 0`, with consistent links and counters.
2. Each block's `ref_count` equals the number of block-table entries pointing at it.
3. Every block in a block table is live (generation matches, `ref_count > 0`).
4. `blocks.len() == ceil(len / block_size)`, and fill levels match sequence lengths.
5. Blocks containing uncomputed positions are exclusively owned and not cached.
6. Every cached block is indexed exactly once, its stored hash matches its tokens, and cids are unique.
7. A shared cached block's tokens equal the tokens of every sequence that holds it.

## 7. Known limitations / future work
* Unreachable cached children (their parent was evicted first) wait until LRU eviction. Cascading invalidation would free them sooner.
* No chunked prefill, swap-to-CPU offload, or quantized storage. The `KvStorage` trait and the `BlockCopy` model are the extension points.
* The CUDA backend is validated on one GPU model only (Colab Tesla T4, driver CUDA 13.0) via
  `notebooks/kvcachex_colab.ipynb`. The development machine (Apple M1) cannot run it. Metal is out of scope.
* The CUDA backend stores f32 only, uses pageable host buffers and a single stream, and copies whole blocks for COW.
* Generation counters are u32 (wrap after 4·10⁹ reuses of one block).

## References
1. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. arXiv:2309.06180.
2. vLLM docs, "Automatic Prefix Caching" (v1 design). https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html
3. L. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024. arXiv:2312.07104.
4. Z. Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023. arXiv:2306.14048.
5. G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024. arXiv:2309.17453.
6. Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," ICML 2024. arXiv:2402.02750.
7. J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023. arXiv:2305.13245.
8. N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. arXiv:1911.02150.
