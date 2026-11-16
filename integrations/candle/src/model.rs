//! The llama2.c transformer from `candle-transformers` 0.11.1 (`models::llama2_c`) with its KV
//! cache replaced by a KVCacheX [`PagedKvCache`].
//!
//! Why this file exists: Candle's model keeps K/V in a per-layer tensor that grows with
//! `Tensor::cat`, inside a private attention layer, and offers no hook to supply K/V from
//! elsewhere. To run the same model on KVCacheX-managed state, the forward pass is re-expressed
//! here. It loads the **same weights by the same names** from the same `VarBuilder` and uses the
//! same Candle layers and ops in the same order; only the cache read/write differs:
//!
//! * new K/V of each layer are written into the sequence's paged blocks (`write_range`);
//! * attention reads the sequence's K/V back from its block table, in position order
//!   (`read_range_into`), and runs Candle's dense attention on it.
//!
//! This is a CPU reference integration. The gather copies K/V into a contiguous buffer each step,
//! as Candle's own `Tensor::cat` cache does; there is no paged-attention kernel here.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias as linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::llama2_c::{Cache, Config};
use kvcachex::{geometry::ModelGeometry, CpuKvStorage, PagedKvCache, SeqId};

/// The KVCacheX geometry of a llama2.c config (f32 K/V, one entry per layer).
pub fn geometry(cfg: &Config) -> ModelGeometry {
    ModelGeometry {
        name: "llama2.c".into(),
        num_layers: cfg.n_layers as u32,
        num_query_heads: cfg.n_heads as u32,
        num_kv_heads: cfg.n_kv_heads as u32,
        head_dim: (cfg.dim / cfg.n_heads) as u32,
        dtype: kvcachex::DType::F32,
    }
}

fn kvx(e: kvcachex::CacheError) -> candle_core::Error {
    candle_core::Error::Msg(format!("kvcachex: {e}"))
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: Attention,
    rms_2: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// The llama2.c model whose attention reads and writes K/V through a [`PagedKvCache`].
#[derive(Debug, Clone)]
pub struct PagedLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    pub config: Config,
}

/// What one forward pass did with the cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardInfo {
    /// Positions whose K/V this pass wrote into KVCacheX.
    pub written: usize,
    /// Positions whose K/V this pass read back for attention (per layer).
    pub attended: usize,
}

impl PagedLlama {
    /// Loads the same weights as `candle_transformers::models::llama2_c::Llama::load(vb, cfg)`;
    /// `rope` is Candle's own cache, used only for its RoPE tables.
    pub fn load(vb: VarBuilder, cfg: Config, rope: &Cache) -> Result<Self> {
        let size_q = (cfg.dim / cfg.n_heads) * cfg.n_heads;
        let size_kv = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let vb = vb.pp(format!("model.layers.{i}"));
            let a = vb.pp("self_attn");
            let m = vb.pp("mlp");
            blocks.push(Block {
                rms_1: rms_norm(cfg.dim, cfg.norm_eps, vb.pp("input_layernorm"))?,
                attn: Attention {
                    q_proj: linear(cfg.dim, size_q, a.pp("q_proj"))?,
                    k_proj: linear(cfg.dim, size_kv, a.pp("k_proj"))?,
                    v_proj: linear(cfg.dim, size_kv, a.pp("v_proj"))?,
                    o_proj: linear(size_q, cfg.dim, a.pp("o_proj"))?,
                    n_head: cfg.n_heads,
                    n_kv_head: cfg.n_kv_heads,
                    head_dim: cfg.dim / cfg.n_heads,
                },
                rms_2: rms_norm(cfg.dim, cfg.norm_eps, vb.pp("post_attention_layernorm"))?,
                gate: linear(cfg.dim, cfg.hidden_dim, m.pp("gate_proj"))?,
                up: linear(cfg.dim, cfg.hidden_dim, m.pp("up_proj"))?,
                down: linear(cfg.hidden_dim, cfg.dim, m.pp("down_proj"))?,
            });
        }
        Ok(Self {
            wte: embedding(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?,
            lm_head: linear(cfg.dim, cfg.vocab_size, vb.pp("lm_head"))?,
            ln_f: rms_norm(cfg.dim, cfg.norm_eps, vb.pp("model.norm"))?,
            blocks,
            cos: rope.cos.clone(),
            sin: rope.sin.clone(),
            config: cfg,
        })
    }

    /// Runs `tokens` (positions `index_pos..index_pos + n`) of sequence `seq` and returns logits
    /// of shape `(1, n, vocab)`. Positions the cache already holds as computed (prefix-cache
    /// hits) are read, not written; every other position is written. The caller marks the
    /// positions computed afterwards (`PagedKvCache::mark_computed`).
    pub fn forward(
        &self,
        tokens: &[u32],
        index_pos: usize,
        seq: SeqId,
        cache: &mut PagedKvCache<CpuKvStorage>,
    ) -> Result<(Tensor, ForwardInfo)> {
        let device = self.cos.device().clone();
        let n = tokens.len();
        let computed = cache.manager().computed(seq).map_err(kvx)?;
        let first_write = computed.clamp(index_pos, index_pos + n) - index_pos;
        let x = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let mut x = self.wte.forward(&x)?;
        let mask = if n > 1 {
            Some(candle_transformers::utils::build_causal_mask(
                n, index_pos, &device,
            )?)
        } else {
            None
        };
        for (layer, block) in self.blocks.iter().enumerate() {
            let residual = &x;
            let h = block.rms_1.forward(&x)?;
            let h = self.attention(
                &block.attn,
                &h,
                index_pos,
                first_write,
                layer,
                seq,
                cache,
                mask.as_ref(),
            )?;
            let x1 = (h + residual)?;
            let h = block.rms_2.forward(&x1)?;
            let h = block
                .down
                .forward(&(silu(&block.gate.forward(&h)?)? * block.up.forward(&h)?)?)?;
            x = (h + &x1)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?.to_dtype(DType::F32)?;
        Ok((
            logits,
            ForwardInfo {
                written: n - first_write,
                attended: index_pos + n,
            },
        ))
    }

    fn rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        // Identical to llama2_c's CausalSelfAttention::apply_rotary_emb.
        let (b_sz, seq_len, h, n_embd) = x.dims4()?;
        let cos = self.cos.i(index_pos..index_pos + seq_len)?.unsqueeze(1)?;
        let sin = self.sin.i(index_pos..index_pos + seq_len)?.unsqueeze(1)?;
        let cos = cos.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let sin = sin.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
        let x = x.reshape((b_sz, seq_len, h, n_embd / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let dst0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let dst1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&dst0, &dst1], D::Minus1)?.reshape((b_sz, seq_len, h, n_embd))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        a: &Attention,
        x: &Tensor,
        index_pos: usize,
        first_write: usize,
        layer: usize,
        seq: SeqId,
        cache: &mut PagedKvCache<CpuKvStorage>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, n, n_embd) = x.dims3()?;
        let q = a.q_proj.forward(x)?.reshape((b_sz, n, a.n_head, a.head_dim))?;
        let k = a.k_proj.forward(x)?.reshape((b_sz, n, a.n_kv_head, a.head_dim))?;
        let v = a.v_proj.forward(x)?.reshape((b_sz, n, a.n_kv_head, a.head_dim))?;
        let q = self.rope(&q, index_pos)?;
        let k = self.rope(&k, index_pos)?;

        // Store the new positions' K/V in KVCacheX (token-major: [pos][kv_head][head_dim]).
        if first_write < n {
            let kw = k
                .narrow(1, first_write, n - first_write)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let vw = v
                .narrow(1, first_write, n - first_write)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            cache
                .write_range(seq, index_pos + first_write, layer as u32, &kw, &vw)
                .map_err(kvx)?;
        }
        // Gather every position's K/V from the block table, in order.
        let total = index_pos + n;
        let width = a.n_kv_head * a.head_dim;
        let (mut kb, mut vb) = (vec![0f32; total * width], vec![0f32; total * width]);
        cache
            .read_range_into(seq, 0, layer as u32, &mut kb, &mut vb)
            .map_err(kvx)?;
        let dev = x.device();
        let k = Tensor::from_vec(kb, (b_sz, total, a.n_kv_head, a.head_dim), dev)?;
        let v = Tensor::from_vec(vb, (b_sz, total, a.n_kv_head, a.head_dim), dev)?;

        // From here on: llama2_c's attention, unchanged.
        let k = repeat_kv(k, a.n_head / a.n_kv_head)?;
        let v = repeat_kv(v, a.n_head / a.n_kv_head)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let att = (q.matmul(&k.t()?)? / (a.head_dim as f64).sqrt())?;
        let att = match mask {
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                let neg = Tensor::new(f32::NEG_INFINITY, att.device())?.broadcast_as(mask.shape().dims())?;
                mask.where_cond(&neg, &att)?
            }
            None => att,
        };
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, n, n_embd])?;
        a.o_proj.forward(&y)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b_sz, seq_len, n_kv_head, head_dim) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b_sz, seq_len, n_kv_head, n_rep, head_dim))?
        .reshape((b_sz, seq_len, n_kv_head * n_rep, head_dim))
}

/// The device every model here runs on (Candle CPU; see the crate README for why).
pub fn device() -> Device {
    Device::Cpu
}
