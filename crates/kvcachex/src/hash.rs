//! Fast, non-cryptographic hashing for prefix-cache keys.
//!
//! The hash only places blocks into buckets. Every hit is checked exactly against the
//! stored namespace, parent content id and token ids (see [`crate::pool`]), so a collision
//! costs a comparison and can never cause incorrect sharing. That lets us use a cheap
//! FxHash-style mixer instead of SipHash.

use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x517c_c1b7_2722_0a95;

#[inline]
fn fx(h: u64, x: u64) -> u64 {
    (h.rotate_left(5) ^ x).wrapping_mul(K)
}

/// SplitMix64 finalizer: spreads entropy into the low bits used by hash tables.
#[inline]
pub(crate) fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Hash of one full block in a prefix chain.
#[inline]
pub(crate) fn block_hash(namespace: u64, parent_cid: u64, tokens: &[u32]) -> u64 {
    if cfg!(feature = "degenerate-hash") {
        return 0x00c0_111d_e000_0000; // force every block into one bucket
    }
    // The token hash does not depend on the parent, so when a prompt is matched block by
    // block, the CPU can overlap hashing of later blocks with the (serial) index probes.
    // Profiling showed the earlier single serial chain dominated prefix-hit admission.
    finalize(token_hash(tokens) ^ fx(fx(0x9e37_79b9_7f4a_7c15, namespace), parent_cid))
}

/// Hash of a block's token ids, four independent lanes for instruction-level parallelism.
#[inline]
fn token_hash(tokens: &[u32]) -> u64 {
    let mut lanes = [
        0x243f_6a88_85a3_08d3u64,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    ];
    let (chunks, remainder) = tokens.as_chunks::<8>();
    for c in chunks {
        for (l, lane) in lanes.iter_mut().enumerate() {
            *lane = fx(*lane, (c[2 * l] as u64) | ((c[2 * l + 1] as u64) << 32));
        }
    }
    let mut h = fx(fx(fx(lanes[0], lanes[1]), lanes[2]), lanes[3]);
    for &t in remainder {
        h = fx(h, t as u64 | (1 << 63));
    }
    fx(h, tokens.len() as u64)
}

/// Hasher for `u64` keys that are already well mixed.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PassThroughHasher(u64);

impl Hasher for PassThroughHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = fx(self.0, b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

pub(crate) type PassThroughBuild = BuildHasherDefault<PassThroughHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(
        feature = "degenerate-hash",
        ignore = "hash is constant under degenerate-hash"
    )]
    fn hash_depends_on_every_input() {
        let base = block_hash(1, 2, &[3, 4, 5]);
        assert_ne!(base, block_hash(0, 2, &[3, 4, 5]));
        assert_ne!(base, block_hash(1, 3, &[3, 4, 5]));
        assert_ne!(base, block_hash(1, 2, &[3, 4, 6]));
        assert_ne!(base, block_hash(1, 2, &[3, 4]));
        assert_ne!(block_hash(0, 0, &[0]), block_hash(0, 0, &[0, 0]));
    }
}
