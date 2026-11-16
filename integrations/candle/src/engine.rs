//! Two engines that run the same Candle llama2.c model on the same weights and differ only in
//! where K/V live:
//!
//! * [`NativeEngine`]: Candle's unmodified `llama2_c::Llama` with its native cache (one
//!   contiguous tensor per layer per sequence, grown with `Tensor::cat`);
//! * [`KvxEngine`]: [`PagedLlama`] with K/V in a KVCacheX [`PagedKvCache`] (paged blocks,
//!   optional exact prefix cache, copy-on-write forks).
//!
//! Each forward pass runs one sequence (batch size 1); concurrent sequences are interleaved by
//! the caller.

use std::collections::HashMap;

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama2_c::{Cache, Config, Llama};
use kvcachex::{CacheConfig, CacheStats, KvStorage, PagedKvCache, SeqId};

use crate::model::{device, geometry, PagedLlama};

/// What admitting a prompt cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Prefill {
    /// Prompt positions the model ran in the prefill forward pass.
    pub computed_tokens: usize,
    /// Prompt positions whose K/V came from the prefix cache (not recomputed).
    pub cached_tokens: usize,
}

pub trait Engine {
    fn label(&self) -> &'static str;
    /// Admits request `id` with `prompt` and runs prefill. Returns the logits of the last prompt
    /// position.
    fn prefill(&mut self, id: usize, prompt: &[u32]) -> Result<(Vec<f32>, Prefill)>;
    /// Appends `token` to request `id`, runs one decode step, returns the next-token logits.
    fn decode(&mut self, id: usize, token: u32) -> Result<Vec<f32>>;
    /// Creates request `child` as a copy of `parent` (same tokens and K/V so far).
    fn fork(&mut self, parent: usize, child: usize) -> Result<()>;
    /// Ends request `id` and frees (or, with prefix caching, caches) its K/V.
    fn finish(&mut self, id: usize) -> Result<()>;
    /// Bytes of K/V currently held for live requests.
    fn kv_bytes_in_use(&self) -> usize;
    /// Bytes of K/V memory allocated (KVCacheX preallocates its whole block pool).
    fn kv_bytes_allocated(&self) -> usize;
}

fn last_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let n = logits.dim(1)?;
    logits.i((0, n - 1))?.to_vec1::<f32>()
}

fn missing(id: usize) -> candle_core::Error {
    candle_core::Error::Msg(format!("unknown request {id}"))
}

fn kvx(e: kvcachex::CacheError) -> candle_core::Error {
    candle_core::Error::Msg(format!("kvcachex: {e}"))
}

/// Candle's model and native KV cache, unmodified.
#[derive(Debug)]
pub struct NativeEngine {
    model: Llama,
    template: Cache,
    seqs: HashMap<usize, (Cache, usize)>,
}

impl NativeEngine {
    pub fn new(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let template = Cache::new(true, &cfg, vb.pp("rot"))?;
        Ok(Self {
            model: Llama::load(vb, cfg)?,
            template,
            seqs: HashMap::new(),
        })
    }
}

impl Engine for NativeEngine {
    fn label(&self) -> &'static str {
        "candle-native"
    }

    fn prefill(&mut self, id: usize, prompt: &[u32]) -> Result<(Vec<f32>, Prefill)> {
        let mut cache = self.template.clone();
        let x = Tensor::new(prompt, &device())?.unsqueeze(0)?;
        let logits = self.model.forward(&x, 0, &mut cache)?;
        self.seqs.insert(id, (cache, prompt.len()));
        Ok((
            last_logits(&logits)?,
            Prefill {
                computed_tokens: prompt.len(),
                cached_tokens: 0,
            },
        ))
    }

    fn decode(&mut self, id: usize, token: u32) -> Result<Vec<f32>> {
        let (cache, pos) = self.seqs.get_mut(&id).ok_or_else(|| missing(id))?;
        let x = Tensor::new(&[token], &device())?.unsqueeze(0)?;
        let logits = self.model.forward(&x, *pos, cache)?;
        *pos += 1;
        last_logits(&logits)
    }

    fn fork(&mut self, parent: usize, child: usize) -> Result<()> {
        let entry = self.seqs.get(&parent).ok_or_else(|| missing(parent))?.clone();
        self.seqs.insert(child, entry);
        Ok(())
    }

    fn finish(&mut self, id: usize) -> Result<()> {
        self.seqs.remove(&id).map(|_| ()).ok_or_else(|| missing(id))
    }

    fn kv_bytes_in_use(&self) -> usize {
        // A fork shares the parent's tensors until its next step (Candle tensors are
        // reference-counted), so this slightly over-counts right after a fork.
        self.seqs
            .values()
            .flat_map(|(c, _)| c.kvs.iter().flatten())
            .map(|(k, v)| (k.elem_count() + v.elem_count()) * 4)
            .sum()
    }

    fn kv_bytes_allocated(&self) -> usize {
        self.kv_bytes_in_use()
    }
}

/// The same model with K/V managed by KVCacheX.
#[derive(Debug)]
pub struct KvxEngine {
    model: PagedLlama,
    cache: PagedKvCache,
    seqs: HashMap<usize, SeqId>,
    prefix_caching: bool,
}

impl KvxEngine {
    pub fn new(vb: VarBuilder, cfg: Config, kv: CacheConfig) -> Result<Self> {
        let rope = Cache::new(true, &cfg, vb.pp("rot"))?;
        let cache = PagedKvCache::new_cpu(&geometry(&cfg), kv);
        let prefix_caching = kv.enable_prefix_caching;
        Ok(Self {
            model: PagedLlama::load(vb, cfg, &rope)?,
            cache,
            seqs: HashMap::new(),
            prefix_caching,
        })
    }

    pub fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    pub fn stats(&self) -> CacheStats {
        self.cache.manager().stats()
    }

    fn block_bytes(&self) -> usize {
        self.cache.storage().layout().block_elems() * 4
    }
}

impl Engine for KvxEngine {
    fn label(&self) -> &'static str {
        if self.prefix_caching {
            "kvcachex-prefix"
        } else {
            "kvcachex"
        }
    }

    fn prefill(&mut self, id: usize, prompt: &[u32]) -> Result<(Vec<f32>, Prefill)> {
        let adm = self.cache.create_sequence(0, prompt).map_err(kvx)?;
        // A fully cached prompt still needs logits for its last position: run that token
        // again; its K/V are read from the cache, not rewritten.
        let start = adm.cached_tokens.min(prompt.len() - 1);
        let (logits, _) = self
            .model
            .forward(&prompt[start..], start, adm.seq, &mut self.cache)?;
        self.cache.mark_computed(adm.seq, prompt.len()).map_err(kvx)?;
        self.seqs.insert(id, adm.seq);
        Ok((
            last_logits(&logits)?,
            Prefill {
                computed_tokens: prompt.len() - start,
                cached_tokens: adm.cached_tokens,
            },
        ))
    }

    fn decode(&mut self, id: usize, token: u32) -> Result<Vec<f32>> {
        let seq = *self.seqs.get(&id).ok_or_else(|| missing(id))?;
        let pos = self.cache.manager().len(seq).map_err(kvx)?;
        self.cache.append_tokens(seq, &[token]).map_err(kvx)?;
        let (logits, _) = self.model.forward(&[token], pos, seq, &mut self.cache)?;
        self.cache.mark_computed(seq, pos + 1).map_err(kvx)?;
        last_logits(&logits)
    }

    fn fork(&mut self, parent: usize, child: usize) -> Result<()> {
        let seq = *self.seqs.get(&parent).ok_or_else(|| missing(parent))?;
        let forked = self.cache.fork(seq).map_err(kvx)?;
        self.seqs.insert(child, forked);
        Ok(())
    }

    fn finish(&mut self, id: usize) -> Result<()> {
        let seq = self.seqs.remove(&id).ok_or_else(|| missing(id))?;
        self.cache.release(seq).map_err(kvx)
    }

    fn kv_bytes_in_use(&self) -> usize {
        self.stats().used_blocks as usize * self.block_bytes()
    }

    fn kv_bytes_allocated(&self) -> usize {
        self.stats().total_blocks as usize * self.block_bytes()
    }
}

/// Index of the largest logit (greedy decoding; ties go to the lowest index).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in logits.iter().enumerate() {
        if x > logits[best] {
            best = i;
        }
    }
    best as u32
}
