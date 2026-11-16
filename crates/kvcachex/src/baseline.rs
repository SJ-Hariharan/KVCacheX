//! Baseline: contiguous per-request reservation, as in pre-PagedAttention serving systems.
//!
//! Each request reserves one contiguous region of `prompt + max_new_tokens` token slots
//! at admission, because its final length is unknown. This wastes memory in three ways,
//! following Kwon et al. (2023, §3): *reserved* slots for tokens not yet generated,
//! *internal fragmentation* when the request stops before its maximum, and *external
//! fragmentation* between regions. Free space is managed first-fit, with coalescing.
//! KVCacheX's paged manager is compared against this baseline.

use std::collections::BTreeMap;

/// A reserved contiguous region `[start, start + len)` in token slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub start: u64,
    pub len: u64,
}

/// First-fit contiguous allocator over a linear address space of token slots.
#[derive(Debug, Clone)]
pub struct ContiguousAllocator {
    capacity: u64,
    /// Free extents keyed by start; never adjacent (always coalesced).
    free: BTreeMap<u64, u64>,
    reserved: u64,
    peak_reserved: u64,
    pub failures: u64,
}

impl ContiguousAllocator {
    pub fn new(capacity_tokens: u64) -> Self {
        let mut free = BTreeMap::new();
        if capacity_tokens > 0 {
            free.insert(0, capacity_tokens);
        }
        ContiguousAllocator {
            capacity: capacity_tokens,
            free,
            reserved: 0,
            peak_reserved: 0,
            failures: 0,
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }
    pub fn reserved(&self) -> u64 {
        self.reserved
    }
    pub fn peak_reserved(&self) -> u64 {
        self.peak_reserved
    }
    pub fn free_total(&self) -> u64 {
        self.capacity - self.reserved
    }
    pub fn largest_free(&self) -> u64 {
        self.free.values().copied().max().unwrap_or(0)
    }
    /// `1 - largest_free / total_free`: 0 means all free space is one usable region.
    pub fn external_fragmentation(&self) -> f64 {
        let total = self.free_total();
        if total == 0 {
            0.0
        } else {
            1.0 - self.largest_free() as f64 / total as f64
        }
    }

    pub fn reserve(&mut self, len: u64) -> Option<Extent> {
        assert!(len > 0, "zero-length reservation");
        let (&start, &flen) = match self.free.iter().find(|(_, &l)| l >= len) {
            Some(e) => e,
            None => {
                self.failures += 1;
                return None;
            }
        };
        self.free.remove(&start);
        if flen > len {
            self.free.insert(start + len, flen - len);
        }
        self.reserved += len;
        self.peak_reserved = self.peak_reserved.max(self.reserved);
        Some(Extent { start, len })
    }

    /// Returns an extent. Panics on double free or overlap: caller bug.
    pub fn release(&mut self, e: Extent) {
        assert!(e.start + e.len <= self.capacity, "extent out of range");
        let mut start = e.start;
        let mut len = e.len;
        if let Some((&ps, &pl)) = self.free.range(..=e.start).next_back() {
            assert!(ps + pl <= e.start, "double free / overlap at {}", e.start);
            if ps + pl == e.start {
                self.free.remove(&ps);
                start = ps;
                len += pl;
            }
        }
        if let Some((&ns, &nl)) = self.free.range(e.start..).next() {
            assert!(e.start + e.len <= ns, "double free / overlap at {}", e.start);
            if e.start + e.len == ns {
                self.free.remove(&ns);
                len += nl;
            }
        }
        self.free.insert(start, len);
        self.reserved -= e.len;
    }

    pub fn check_invariants(&self) -> Result<(), String> {
        let mut prev_end: Option<u64> = None;
        let mut total = 0;
        for (&s, &l) in &self.free {
            if l == 0 || s + l > self.capacity {
                return Err(format!("bad free extent {s}+{l}"));
            }
            if let Some(pe) = prev_end {
                if s <= pe {
                    return Err(format!("free extents overlap or not coalesced at {s}"));
                }
            }
            prev_end = Some(s + l);
            total += l;
        }
        if total != self.free_total() {
            return Err(format!(
                "free total {total} != capacity - reserved {}",
                self.free_total()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_release_coalesces() {
        let mut a = ContiguousAllocator::new(100);
        let x = a.reserve(30).unwrap();
        let y = a.reserve(30).unwrap();
        let z = a.reserve(40).unwrap();
        assert!(a.reserve(1).is_none());
        a.release(y);
        assert_eq!(a.largest_free(), 30);
        assert!(a.reserve(31).is_none(), "30 free but no 31-slot hole");
        a.release(x);
        a.release(z);
        assert_eq!(a.largest_free(), 100);
        a.check_invariants().unwrap();
    }

    #[test]
    fn external_fragmentation_measured() {
        let mut a = ContiguousAllocator::new(100);
        let e: Vec<_> = (0..10).map(|_| a.reserve(10).unwrap()).collect();
        for x in e.iter().step_by(2) {
            a.release(*x);
        }
        assert_eq!(a.free_total(), 50);
        assert!((a.external_fragmentation() - 0.8).abs() < 1e-12);
        a.check_invariants().unwrap();
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn double_free_panics() {
        let mut a = ContiguousAllocator::new(100);
        let x = a.reserve(10).unwrap();
        let _y = a.reserve(10).unwrap();
        a.release(x);
        a.release(x);
    }
}
