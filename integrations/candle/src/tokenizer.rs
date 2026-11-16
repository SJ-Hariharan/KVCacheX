//! The llama2.c tokenizer (`tokenizer.bin` / `tok512.bin`): a sentencepiece-style BPE vocabulary
//! with merge scores and byte fallback. This is a port of `encode` / `decode` in llama2.c's
//! `run.c`, so token ids match what llama2.c itself produces for the TinyStories checkpoints.

use std::collections::HashMap;
use std::io::Read;

pub const BOS: u32 = 1;
pub const EOS: u32 = 2;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    lookup: HashMap<Vec<u8>, u32>,
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

impl Tokenizer {
    /// Reads a llama2.c tokenizer file for a model with `vocab_size` tokens.
    pub fn from_reader(r: &mut impl Read, vocab_size: usize) -> std::io::Result<Self> {
        let _max_token_length = read_i32(r)?;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            scores.push(f32::from_le_bytes(b));
            let len = read_i32(r)?;
            let len = usize::try_from(len).map_err(|_| std::io::Error::other("negative token length"))?;
            let mut piece = vec![0u8; len];
            r.read_exact(&mut piece)?;
            vocab.push(piece);
        }
        let mut lookup = HashMap::with_capacity(vocab_size);
        for (i, piece) in vocab.iter().enumerate() {
            // Like llama2.c's sorted lookup, the first id wins for duplicate pieces.
            lookup.entry(piece.clone()).or_insert(i as u32);
        }
        Ok(Self {
            vocab,
            scores,
            lookup,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encodes `text` like llama2.c: optional BOS, a dummy-prefix space, UTF-8 code points looked
    /// up directly (byte fallback `byte + 3` otherwise), then greedy merges by highest score.
    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            if let Some(&id) = self.lookup.get(b" ".as_slice()) {
                tokens.push(id);
            }
        }
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            let bytes = ch.encode_utf8(&mut buf).as_bytes();
            match self.lookup.get(bytes) {
                Some(&id) => tokens.push(id),
                None => tokens.extend(bytes.iter().map(|&b| b as u32 + 3)),
            }
        }
        let mut pair = Vec::new();
        loop {
            let mut best: Option<(f32, usize, u32)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                pair.clear();
                pair.extend_from_slice(&self.vocab[tokens[i] as usize]);
                pair.extend_from_slice(&self.vocab[tokens[i + 1] as usize]);
                if let Some(&id) = self.lookup.get(&pair) {
                    let score = self.scores[id as usize];
                    if best.is_none_or(|(s, _, _)| score > s) {
                        best = Some((score, i, id));
                    }
                }
            }
            let Some((_, i, id)) = best else { break };
            tokens[i] = id;
            tokens.remove(i + 1);
        }
        tokens
    }

    /// Decodes `token` given the previous token (llama2.c strips the leading space after BOS
    /// and turns `<0xXX>` pieces into raw bytes).
    pub fn decode_piece(&self, prev: u32, token: u32) -> Vec<u8> {
        let Some(piece) = self.vocab.get(token as usize) else {
            return Vec::new();
        };
        let piece = if prev == BOS && piece.first() == Some(&b' ') {
            &piece[1..]
        } else {
            &piece[..]
        };
        if piece.len() == 6 && piece.starts_with(b"<0x") && piece[5] == b'>' {
            if let Ok(s) = std::str::from_utf8(&piece[3..5]) {
                if let Ok(b) = u8::from_str_radix(s, 16) {
                    return vec![b];
                }
            }
        }
        piece.to_vec()
    }

    /// Decodes a token sequence (as produced by [`Tokenizer::encode`] plus generated tokens).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut out = Vec::new();
        let mut prev = BOS;
        for &t in tokens {
            if t == BOS {
                prev = t;
                continue;
            }
            out.extend(self.decode_piece(prev, t));
            prev = t;
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}
