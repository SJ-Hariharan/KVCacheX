//! Model weights: a real llama2.c checkpoint file, or deterministic synthetic weights for tests
//! (no download needed).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama2_c::Config;
use candle_transformers::models::llama2_c_weights::TransformerWeights;

use crate::model::device;

/// Loads a llama2.c `.bin` checkpoint (e.g. `stories15M.bin`) exactly as Candle's llama2-c
/// example does.
pub fn load_checkpoint(path: &Path) -> Result<(Config, VarBuilder<'static>)> {
    let file =
        std::fs::File::open(path).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
    let mut r = std::io::BufReader::new(file);
    let cfg = Config::from_reader(&mut r)?;
    let weights = TransformerWeights::from_reader(&mut r, &cfg, &device())?;
    let vb = weights.var_builder(&cfg, &device())?;
    Ok((cfg, vb))
}

/// A small GQA config for tests: 2 layers, 4 query heads sharing 2 K/V heads.
pub fn test_config() -> Config {
    Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 64,
        seq_len: 128,
        norm_eps: 1e-5,
    }
}

/// Deterministic pseudo-random weights with the tensor names `Llama::load` expects.
pub fn synthetic(cfg: &Config, seed: u64) -> Result<VarBuilder<'static>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut tensor = |shape: &[usize], scale: f32| -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                // xorshift64*; uniform in [-scale, scale)
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let u = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect();
        Tensor::from_vec(data, shape, &device())
    };
    let hd = cfg.dim / cfg.n_heads;
    let kv = hd * cfg.n_kv_heads;
    let mut ws = HashMap::new();
    ws.insert(
        "model.embed_tokens.weight".to_string(),
        tensor(&[cfg.vocab_size, cfg.dim], 1.0)?,
    );
    ws.insert(
        "lm_head.weight".to_string(),
        tensor(&[cfg.vocab_size, cfg.dim], 0.5)?,
    );
    ws.insert("model.norm.weight".to_string(), (tensor(&[cfg.dim], 0.1)? + 1.0)?);
    for l in 0..cfg.n_layers {
        let p = format!("model.layers.{l}");
        let s = 1.0 / (cfg.dim as f32).sqrt();
        for (name, shape) in [
            ("self_attn.q_proj", [cfg.dim, cfg.dim]),
            ("self_attn.k_proj", [kv, cfg.dim]),
            ("self_attn.v_proj", [kv, cfg.dim]),
            ("self_attn.o_proj", [cfg.dim, cfg.dim]),
            ("mlp.gate_proj", [cfg.hidden_dim, cfg.dim]),
            ("mlp.up_proj", [cfg.hidden_dim, cfg.dim]),
            ("mlp.down_proj", [cfg.dim, cfg.hidden_dim]),
        ] {
            ws.insert(format!("{p}.{name}.weight"), tensor(&shape, s * 2.0)?);
        }
        ws.insert(
            format!("{p}.input_layernorm.weight"),
            (tensor(&[cfg.dim], 0.1)? + 1.0)?,
        );
        ws.insert(
            format!("{p}.post_attention_layernorm.weight"),
            (tensor(&[cfg.dim], 0.1)? + 1.0)?,
        );
    }
    Ok(VarBuilder::from_tensors(ws, DType::F32, &device()))
}
