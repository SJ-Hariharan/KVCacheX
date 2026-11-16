//! Host-memory reference backend.

use super::{BlockLayout, KvStorage, Residency, StorageError};
use crate::geometry::ModelGeometry;
use crate::manager::BlockCopy;
use crate::pool::BlockId;

/// KV storage in one contiguous host allocation, indexed by physical block.
#[derive(Debug)]
pub struct CpuKvStorage {
    data: Vec<f32>,
    layout: BlockLayout,
}

impl CpuKvStorage {
    /// Allocates zeroed host storage. Panics only if the size overflows `usize` or the
    /// allocation itself fails (standard `Vec` behaviour); use [`Self::try_new`] to get an error
    /// for overflow instead.
    pub fn new(geom: &ModelGeometry, block_size: u32, num_blocks: u32) -> Self {
        Self::try_new(geom, block_size, num_blocks).expect("CPU KV storage size overflows usize")
    }

    pub fn try_new(geom: &ModelGeometry, block_size: u32, num_blocks: u32) -> Result<Self, StorageError> {
        let layout = BlockLayout::new(geom, block_size, num_blocks);
        let len = layout.total_elems()?;
        Ok(CpuKvStorage {
            data: vec![0.0; len],
            layout,
        })
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Zero-copy view of one slot's K and V. Host-only: device backends have no equivalent.
    pub fn read(&self, block: BlockId, slot: u32, layer: u32) -> Result<(&[f32], &[f32]), StorageError> {
        let w = self.layout.width;
        let (k, v, n) = self.layout.slot_range(block, slot, layer, w, w)?;
        Ok((&self.data[k..k + n], &self.data[v..v + n]))
    }
}

impl KvStorage for CpuKvStorage {
    fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    fn residency(&self) -> Residency {
        Residency::Host
    }

    fn write_slots(
        &mut self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), StorageError> {
        let (ko, vo, n) = self.layout.slot_range(block, slot, layer, k.len(), v.len())?;
        self.data[ko..ko + n].copy_from_slice(k);
        self.data[vo..vo + n].copy_from_slice(v);
        Ok(())
    }

    fn read_slots_into(
        &self,
        block: BlockId,
        slot: u32,
        layer: u32,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), StorageError> {
        let (ko, vo, n) = self
            .layout
            .slot_range(block, slot, layer, k_out.len(), v_out.len())?;
        k_out.copy_from_slice(&self.data[ko..ko + n]);
        v_out.copy_from_slice(&self.data[vo..vo + n]);
        Ok(())
    }

    fn copy_block(&mut self, copy: BlockCopy) -> Result<(), StorageError> {
        let (s, d) = self.layout.check_copy(copy)?;
        // Copy exactly the first `slots` slots of every (layer, K/V) plane.
        let n = copy.slots as usize * self.layout.width;
        let (be, pe) = (self.layout.block_elems(), self.layout.plane_elems());
        for plane in 0..self.layout.layers * 2 {
            let so = s * be + plane * pe;
            let dst = d * be + plane * pe;
            self.data.copy_within(so..so + n, dst);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::BlockPool;

    #[test]
    fn layout_offsets_do_not_overlap() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let s = CpuKvStorage::new(&g, 4, 3);
        let pool = BlockPool::new(3, 4);
        let w = s.layout.width;
        let mut seen = std::collections::HashSet::new();
        for b in 0..3 {
            for l in 0..g.num_layers {
                for slot in 0..4 {
                    let (k, v, _) = s.layout.slot_range(pool.id_of(b), slot, l, w, w).unwrap();
                    assert!(seen.insert(k) && seen.insert(v));
                }
            }
        }
        assert_eq!(seen.len() * w, s.data.len());
    }

    #[test]
    fn copy_block_copies_only_requested_slots() {
        let g = ModelGeometry::preset("tiny").unwrap();
        let mut s = CpuKvStorage::new(&g, 4, 2);
        let pool = BlockPool::new(2, 4);
        let (a, b) = (pool.id_of(0), pool.id_of(1));
        let w = s.layout.width;
        for l in 0..g.num_layers {
            let k: Vec<f32> = (0..4 * w).map(|i| (l * 1000) as f32 + i as f32).collect();
            s.write_slots(a, 0, l, &k, &k).unwrap();
        }
        s.copy_block(BlockCopy {
            src: a,
            dst: b,
            slots: 3,
        })
        .unwrap();
        for l in 0..g.num_layers {
            for slot in 0..4 {
                let (src, _) = s.read(a, slot, l).unwrap();
                let (dst, dv) = s.read(b, slot, l).unwrap();
                if slot < 3 {
                    assert_eq!(src, dst);
                    assert_eq!(src, dv);
                } else {
                    assert!(dst.iter().all(|&x| x == 0.0));
                }
            }
        }
        assert!(s
            .copy_block(BlockCopy {
                src: a,
                dst: a,
                slots: 1
            })
            .is_err());
        assert!(s
            .copy_block(BlockCopy {
                src: a,
                dst: b,
                slots: 5
            })
            .is_err());
    }
}
