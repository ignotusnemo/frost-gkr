// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo. All rights reserved.

//! SIMD-packed field elements for binary tower fields.
//!
//! PackedBlock128 holds P independent Block128 values in a single
//! struct. P = 4 on x86-64 (AVX-512), P = 2 (AVX2), P = 1 (scalar fallback).
//! All XOR operations are lane-wise. Multiplications and squarings operate
//! per-lane but batch the lane setup/teardown amortisation.

pub mod arith;
pub mod karatsuba;
pub mod mul_table;
pub mod pow7;
pub mod simd_square_avx2;
pub mod square;

use crate::{Block128, TowerField};

/// Number of Block128 lanes per packed value.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const PACKED_LANES: usize = 4;

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
pub const PACKED_LANES: usize = 2;

#[cfg(not(all(
    target_arch = "x86_64",
    any(target_feature = "avx512f", target_feature = "avx2")
)))]
pub const PACKED_LANES: usize = 1;

/// SIMD-packed block of Block128 values.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedBlock128 {
    lanes: [u128; 4],
}

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedBlock128 {
    lanes: [u128; 2],
}

#[cfg(not(all(
    target_arch = "x86_64",
    any(target_feature = "avx512f", target_feature = "avx2")
)))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedBlock128 {
    lanes: [u128; 1],
}

impl PackedBlock128 {
    /// Zero constant.
    pub const ZERO: Self = Self {
        lanes: [0; PACKED_LANES],
    };

    /// One constant (all lanes set to Block128::ONE).
    pub const ONE: Self = Self {
        lanes: [1; PACKED_LANES],
    };

    /// Create from a slice of Block128. Panics if slice is too short.
    #[inline(always)]
    pub fn from_slice(slice: &[Block128]) -> Self {
        let mut lanes = [0u128; PACKED_LANES];
        for i in 0..PACKED_LANES {
            lanes[i] = slice[i].to_u128();
        }
        Self { lanes }
    }

    /// Create from an array of Block128.
    #[inline(always)]
    pub fn from_array(arr: [Block128; PACKED_LANES]) -> Self {
        Self {
            lanes: arr.map(|b| b.to_u128()),
        }
    }

    /// Convert to array of Block128.
    #[inline(always)]
    pub fn to_array(self) -> [Block128; PACKED_LANES] {
        self.lanes.map(Block128::from)
    }

    /// Broadcast a single Block128 value to all lanes.
    #[inline(always)]
    pub fn broadcast(val: Block128) -> Self {
        Self {
            lanes: [val.to_u128(); PACKED_LANES],
        }
    }

    /// Get lane value.
    #[inline(always)]
    pub fn get_lane(self, lane: usize) -> Block128 {
        Block128::from(self.lanes[lane])
    }

    /// Set lane value.
    #[inline(always)]
    pub fn set_lane(mut self, lane: usize, val: Block128) -> Self {
        self.lanes[lane] = val.to_u128();
        self
    }

    /// XOR (addition in GF(2^n)). The primary SIMD operation.
    #[inline(always)]
    pub fn xor(self, other: Self) -> Self {
        Self {
            lanes: std::array::from_fn(|i| self.lanes[i] ^ other.lanes[i]),
        }
    }

    /// Horizontal XOR reduction: XOR all lanes to a single Block128.
    #[inline(always)]
    pub fn reduce_xor(self) -> Block128 {
        let arr = self.to_array();
        let mut acc = Block128::ZERO;
        for v in arr {
            acc += v;
        }
        acc
    }
}

impl std::ops::BitXor for PackedBlock128 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        self.xor(rhs)
    }
}

impl std::ops::BitXorAssign for PackedBlock128 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = self.xor(rhs);
    }
}

/// Convert a slice of Block128 to PackedBlock128 slice.
/// Caller must ensure the slice length is a multiple of PACKED_LANES
/// and properly aligned.
///
/// # Safety
/// The input slice must be aligned to at least 16 bytes.
#[inline(always)]
pub fn pack_slice(slice: &[Block128]) -> &[PackedBlock128] {
    if PACKED_LANES > 1 {
        assert_eq!(slice.len() % PACKED_LANES, 0);
    }
    let ptr = slice.as_ptr() as *const PackedBlock128;
    let len = slice.len() / PACKED_LANES;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Convert a mutable slice of Block128 to mutable PackedBlock128 slice.
///
/// # Safety
#[inline(always)]
pub fn pack_slice_mut(slice: &mut [Block128]) -> &mut [PackedBlock128] {
    assert_eq!(slice.len() % PACKED_LANES, 0);
    let ptr = slice.as_mut_ptr() as *mut PackedBlock128;
    let len = slice.len() / PACKED_LANES;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Convert PackedBlock128 slice back to Block128 slice.
#[inline(always)]
pub fn unpack_slice(packed: &[PackedBlock128]) -> &[Block128] {
    let ptr = packed.as_ptr() as *const Block128;
    let len = packed.len() * PACKED_LANES;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Convert mutable PackedBlock128 slice back to mutable Block128 slice.
#[inline(always)]
pub fn unpack_slice_mut(packed: &mut [PackedBlock128]) -> &mut [Block128] {
    let ptr = packed.as_mut_ptr() as *mut Block128;
    let len = packed.len() * PACKED_LANES;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TowerField;
    use rand::Rng;

    #[test]
    fn packed_xor_matches_scalar() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a: [Block128; PACKED_LANES] =
                std::array::from_fn(|_| Block128::from(rng.gen::<u128>()));
            let b: [Block128; PACKED_LANES] =
                std::array::from_fn(|_| Block128::from(rng.gen::<u128>()));

            let pa = PackedBlock128::from_array(a);
            let pb = PackedBlock128::from_array(b);
            let packed_result = pa.xor(pb).to_array();

            for i in 0..PACKED_LANES {
                assert_eq!(packed_result[i], a[i] + b[i]);
            }
        }
    }

    #[test]
    fn packed_reduce_xor() {
        // reduce_xor must equal the lane-wise XOR reduction.
        let arr: [Block128; PACKED_LANES] = std::array::from_fn(|i| Block128::from(i as u128 + 1));
        let packed = PackedBlock128::from_array(arr);
        let expected = arr.iter().fold(Block128::ZERO, |a, &b| a + b);
        assert_eq!(packed.reduce_xor(), expected);
    }
}
