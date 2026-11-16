//! Reference integration of KVCacheX with the [Candle](https://github.com/huggingface/candle)
//! inference runtime: Candle's llama2.c model (TinyStories checkpoints) generating text with its
//! K/V state held in KVCacheX blocks, checked against Candle's native KV cache.
//!
//! Scope: CPU only, f32, batch size 1 per forward pass. See `README.md` in this directory for
//! what this does and does not show.

pub mod engine;
pub mod model;
pub mod tokenizer;
pub mod weights;

pub use engine::{argmax, Engine, KvxEngine, NativeEngine, Prefill};
pub use model::PagedLlama;
pub use tokenizer::Tokenizer;
