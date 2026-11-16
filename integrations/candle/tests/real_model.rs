//! Opt-in: the real TinyStories checkpoint (needs `../../scripts/fetch_tiny_models.sh`).
//! Run with `cargo test --release -- --ignored`. If the files are missing the test FAILS
//! with instructions; it never passes without running the model.

use std::path::PathBuf;

use kvcachex::CacheConfig;
use kvcachex_candle::weights::load_checkpoint;
use kvcachex_candle::{argmax, Engine, KvxEngine, NativeEngine, Tokenizer};

fn models_dir() -> PathBuf {
    std::env::var_os("KVX_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))
}

fn files(model: &str, tok: &str) -> (PathBuf, PathBuf) {
    let d = models_dir();
    let (m, t) = (d.join(model), d.join(tok));
    assert!(
        m.exists() && t.exists(),
        "missing {} or {}: run scripts/fetch_tiny_models.sh (or set KVX_MODELS_DIR)",
        m.display(),
        t.display()
    );
    (m, t)
}

fn greedy(e: &mut dyn Engine, id: usize, prompt: &[u32], n: usize) -> (Vec<u32>, Vec<f32>) {
    let (mut logits, _) = e.prefill(id, prompt).unwrap();
    let first = logits.clone();
    let mut out = Vec::new();
    for _ in 0..n {
        let t = argmax(&logits);
        out.push(t);
        logits = e.decode(id, t).unwrap();
    }
    (out, first)
}

fn check(model: &str, tok: &str) {
    let (m, t) = files(model, tok);
    let (cfg, vb) = load_checkpoint(&m).unwrap();
    let tokenizer = Tokenizer::from_reader(&mut std::fs::File::open(&t).unwrap(), cfg.vocab_size).unwrap();
    let text = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.";
    let ids = tokenizer.encode(text, true);
    assert_eq!(tokenizer.decode(&ids), text, "tokenizer round trip");

    let mut native = NativeEngine::new(vb.clone(), cfg.clone()).unwrap();
    let kv = CacheConfig {
        block_size: 16,
        num_blocks: 64,
        enable_prefix_caching: true,
    };
    let mut paged = KvxEngine::new(vb, cfg, kv).unwrap();
    let (nt, nl) = greedy(&mut native, 0, &ids, 64);
    let (kt, kl) = greedy(&mut paged, 0, &ids, 64);
    assert_eq!(nt, kt, "greedy continuation differs");
    assert_eq!(nl, kl, "prefill logits differ");
    paged.finish(0).unwrap();
    // Same prompt again: served from the prefix cache, same continuation.
    let (kt2, _) = greedy(&mut paged, 1, &ids, 64);
    assert_eq!(nt, kt2);
    assert!(paged.stats().counters.prefix_hit_tokens > 0);
    let mut all = ids.clone();
    all.extend(&nt);
    println!("{model}: {}", tokenizer.decode(&all));
}

#[test]
#[ignore = "needs downloaded model files; run with --ignored"]
fn stories15m_paged_matches_native() {
    check("stories15M.bin", "tokenizer.bin");
}

// stories260K (GQA, 4 K/V heads for 8 query heads) is not tested: Candle 0.11.1's llama2.c
// `.bin` reader assumes n_kv_heads == n_heads and fails to read that file. GQA is covered by
// the synthetic-weight tests in correctness.rs.
