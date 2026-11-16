//! KV-cache memory geometry: how many bytes a token, a block, and a workload need.
//!
//! For every transformer layer, attention keeps one Key and one Value vector per KV head
//! per token. The per-token footprint is therefore
//!
//! ```text
//! bytes_per_token = 2 (K and V) * num_layers * num_kv_heads * head_dim * bytes_per_element
//! ```
//!
//! * **MHA** (multi-head attention): `num_kv_heads == num_query_heads`.
//! * **GQA** (grouped-query attention, Ainslie et al. 2023): `1 < num_kv_heads < num_query_heads`;
//!   several query heads share one KV head, shrinking the cache by `num_query_heads / num_kv_heads`.
//! * **MQA** (multi-query attention, Shazeer 2019): `num_kv_heads == 1`.
//!
//! Only `num_kv_heads` affects cache size; query heads do not. The estimate covers the
//! KV tensors only. It excludes model weights, activations, allocator slack, and per-block
//! metadata, which is listed separately by [`MemoryEstimate::metadata_bytes`].

use std::fmt;

/// Element type of stored K/V values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// 8-bit float (e.g. E4M3) or int8 KV quantization; scales are not counted.
    Fp8,
}

impl DType {
    pub const fn bytes(self) -> u64 {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Fp8 => 1,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Some(DType::F32),
            "f16" | "fp16" | "float16" | "half" => Some(DType::F16),
            "bf16" | "bfloat16" => Some(DType::BF16),
            "fp8" | "f8" | "int8" | "i8" => Some(DType::Fp8),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::Fp8 => "fp8",
        };
        f.write_str(s)
    }
}

/// Attention layout class, derived from head counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Mha,
    Gqa { group_size: u32 },
    Mqa,
}

/// The model dimensions that determine KV-cache size.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelGeometry {
    pub name: String,
    pub num_layers: u32,
    pub num_query_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
}

impl ModelGeometry {
    /// Checks that the dimensions are non-zero and that query heads divide evenly into KV groups.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_layers == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err("num_layers, num_kv_heads and head_dim must be > 0".into());
        }
        if self.num_query_heads < self.num_kv_heads || !self.num_query_heads.is_multiple_of(self.num_kv_heads)
        {
            return Err(format!(
                "num_query_heads ({}) must be a positive multiple of num_kv_heads ({})",
                self.num_query_heads, self.num_kv_heads
            ));
        }
        Ok(())
    }

    pub fn attention_kind(&self) -> AttentionKind {
        if self.num_kv_heads == 1 && self.num_query_heads > 1 {
            AttentionKind::Mqa
        } else if self.num_kv_heads == self.num_query_heads {
            AttentionKind::Mha
        } else {
            AttentionKind::Gqa {
                group_size: self.num_query_heads / self.num_kv_heads,
            }
        }
    }

    /// Number of f-elements of K (or V) for one token in one layer.
    pub const fn kv_elems_per_token_per_layer(&self) -> u64 {
        self.num_kv_heads as u64 * self.head_dim as u64
    }

    /// Bytes of K+V for one token across all layers.
    pub const fn bytes_per_token(&self) -> u64 {
        2 * self.num_layers as u64 * self.kv_elems_per_token_per_layer() * self.dtype.bytes()
    }

    /// Bytes of K+V for one block of `block_size` token slots.
    pub const fn bytes_per_block(&self, block_size: u32) -> u64 {
        self.bytes_per_token() * block_size as u64
    }

    /// How many blocks fit in `budget_bytes` (floor).
    pub const fn blocks_for_budget(&self, budget_bytes: u64, block_size: u32) -> u64 {
        budget_bytes / self.bytes_per_block(block_size)
    }

    /// Well-known public configurations, from the models' published `config.json` files.
    pub fn preset(name: &str) -> Option<Self> {
        let g = |name: &str, l, q, kv, d| ModelGeometry {
            name: name.to_string(),
            num_layers: l,
            num_query_heads: q,
            num_kv_heads: kv,
            head_dim: d,
            dtype: DType::F16,
        };
        Some(match name {
            // MHA: 32 layers, 32 heads, hidden 4096.
            "llama-2-7b" => g("llama-2-7b", 32, 32, 32, 128),
            // GQA: 32 layers, 32 query heads, 8 KV heads.
            "llama-3-8b" => g("llama-3-8b", 32, 32, 8, 128),
            // GQA: 80 layers, 64 query heads, 8 KV heads.
            "llama-3-70b" => g("llama-3-70b", 80, 64, 8, 128),
            // GQA: 32 layers, 32 query heads, 8 KV heads.
            "mistral-7b" => g("mistral-7b", 32, 32, 8, 128),
            // MQA: 32 layers, 71 query heads, 1 KV head, head_dim 64.
            "falcon-7b" => g("falcon-7b", 32, 71, 1, 64),
            // Tiny configuration for tests and CPU storage demos.
            "tiny" => g("tiny", 2, 4, 2, 8),
            _ => return None,
        })
    }

    pub const PRESETS: &'static [&'static str] = &[
        "llama-2-7b",
        "llama-3-8b",
        "llama-3-70b",
        "mistral-7b",
        "falcon-7b",
        "tiny",
    ];
}

/// Result of estimating KV memory for a batch of sequences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub total_tokens: u64,
    /// Bytes if each sequence were stored exactly, with no block rounding.
    pub exact_bytes: u64,
    /// Bytes when each sequence is rounded up to whole blocks (paged allocation).
    pub paged_bytes: u64,
    pub blocks: u64,
    /// Bytes reserved if each sequence is allocated `reserve_len` tokens up front.
    pub reserved_bytes: u64,
    /// Bookkeeping estimate: block table entries (8 B each) plus per-block metadata.
    pub metadata_bytes: u64,
}

impl MemoryEstimate {
    /// Fraction of paged memory that holds no token (last-block slack).
    pub fn paged_internal_fragmentation(&self) -> f64 {
        if self.paged_bytes == 0 {
            0.0
        } else {
            1.0 - self.exact_bytes as f64 / self.paged_bytes as f64
        }
    }

    /// Fraction of reserved memory that holds no token.
    pub fn reservation_waste(&self) -> f64 {
        if self.reserved_bytes == 0 {
            0.0
        } else {
            1.0 - self.exact_bytes as f64 / self.reserved_bytes as f64
        }
    }
}

/// Size in bytes of the per-block metadata kept by [`crate::BlockPool`]. Approximate;
/// counts fields, not allocator overhead.
pub const BLOCK_METADATA_BYTES: u64 = 64;

/// Estimates memory for sequences of the given lengths.
///
/// `reserve_len` models a contiguous allocator that reserves the maximum length per
/// sequence (each sequence reserves `max(len, reserve_len)` tokens).
pub fn estimate(geom: &ModelGeometry, block_size: u32, seq_lens: &[u64], reserve_len: u64) -> MemoryEstimate {
    assert!(block_size > 0, "block_size must be > 0");
    let bpt = geom.bytes_per_token();
    let bs = block_size as u64;
    let total_tokens: u64 = seq_lens.iter().sum();
    let blocks: u64 = seq_lens.iter().map(|&l| l.div_ceil(bs)).sum();
    let reserved_tokens: u64 = seq_lens.iter().map(|&l| l.max(reserve_len)).sum();
    MemoryEstimate {
        total_tokens,
        exact_bytes: total_tokens * bpt,
        paged_bytes: blocks * bs * bpt,
        blocks,
        reserved_bytes: reserved_tokens * bpt,
        metadata_bytes: blocks * (8 + BLOCK_METADATA_BYTES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama2_7b_is_512_kib_per_token() {
        // 2 * 32 layers * 32 heads * 128 dim * 2 bytes = 524,288 B, the widely cited ~0.5 MB/token.
        let g = ModelGeometry::preset("llama-2-7b").unwrap();
        assert_eq!(g.bytes_per_token(), 524_288);
        assert_eq!(g.attention_kind(), AttentionKind::Mha);
    }

    #[test]
    fn gqa_and_mqa_shrink_the_cache() {
        let l3 = ModelGeometry::preset("llama-3-8b").unwrap();
        assert_eq!(l3.bytes_per_token(), 131_072); // 4x smaller than MHA Llama-2-7B
        assert_eq!(l3.attention_kind(), AttentionKind::Gqa { group_size: 4 });
        let f = ModelGeometry::preset("falcon-7b").unwrap();
        assert_eq!(f.bytes_per_token(), 2 * 32 * 64 * 2);
        assert_eq!(f.attention_kind(), AttentionKind::Mqa);
    }

    #[test]
    fn all_presets_validate() {
        for p in ModelGeometry::PRESETS {
            ModelGeometry::preset(p).unwrap().validate().unwrap();
        }
    }

    #[test]
    fn validate_rejects_bad_heads() {
        let mut g = ModelGeometry::preset("tiny").unwrap();
        g.num_query_heads = 3;
        assert!(g.validate().is_err());
        g.num_kv_heads = 0;
        assert!(g.validate().is_err());
    }

    #[test]
    fn estimate_rounds_to_blocks() {
        let g = ModelGeometry::preset("llama-3-8b").unwrap();
        let e = estimate(&g, 16, &[1, 16, 17], 2048);
        assert_eq!(e.total_tokens, 34);
        assert_eq!(e.blocks, 1 + 1 + 2);
        assert_eq!(e.paged_bytes, 4 * 16 * g.bytes_per_token());
        assert_eq!(e.reserved_bytes, 3 * 2048 * g.bytes_per_token());
        assert!((e.paged_internal_fragmentation() - (1.0 - 34.0 / 64.0)).abs() < 1e-12);
    }

    #[test]
    fn dtype_parse_roundtrip() {
        for d in [DType::F32, DType::F16, DType::BF16, DType::Fp8] {
            assert_eq!(DType::parse(&d.to_string()), Some(d));
        }
        assert_eq!(DType::parse("nope"), None);
    }
}
