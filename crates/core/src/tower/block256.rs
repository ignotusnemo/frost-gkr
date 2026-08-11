// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! GF(2^256), represented as a quadratic extension of [`Block128`].
//!
//! An element `(lo, hi)` denotes `lo + hi * X` modulo
//! `X^2 + X + Block128::TAU`.  The type is used by the ragged GKR artifact:
//! committed Poseidon2b rows remain in GF(2^128), while challenges, sumcheck
//! messages, and terminal claims live in GF(2^256).

use crate::{
    Bit, Block128, Block16, Block32, Block64, Block8, CanonicalDeserialize, CanonicalSerialize,
    SerializationError, TowerField,
};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use zeroize::Zeroize;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Serialize, Deserialize, Zeroize)]
#[repr(C, align(16))]
pub struct Block256 {
    pub lo: Block128,
    pub hi: Block128,
}

impl Block256 {
    pub const TAU: Block128 = Block128::TAU;

    #[inline(always)]
    pub const fn new(lo: Block128, hi: Block128) -> Self {
        Self { lo, hi }
    }

    #[inline(always)]
    pub const fn from_base(value: Block128) -> Self {
        Self {
            lo: value,
            hi: Block128::ZERO,
        }
    }

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    #[inline(always)]
    pub fn scale_base(self, scalar: Block128) -> Self {
        Self {
            lo: self.lo * scalar,
            hi: self.hi * scalar,
        }
    }

    #[inline(always)]
    pub fn square(self) -> Self {
        let lo2 = self.lo.square();
        let hi2 = self.hi.square();
        Self {
            lo: lo2 + hi2 * Self::TAU,
            hi: hi2,
        }
    }

    /// Map two uniform base-field lanes into the trace-one affine challenge
    /// support.  The high coordinate is never zero, so the result never lies
    /// in the distinguished GF(2^128) subfield.
    #[inline(always)]
    pub fn from_challenge_lanes(lo: Block128, raw_hi: Block128) -> Self {
        Self {
            lo,
            hi: raw_hi.square() + raw_hi + Self::TAU,
        }
    }
}

impl TowerField for Block256 {
    const BITS: usize = 256;
    const ZERO: Self = Self::new(Block128::ZERO, Block128::ZERO);
    const ONE: Self = Self::new(Block128::ONE, Block128::ZERO);
    // This constant would define a further quadratic extension.  No such
    // extension is used by this artifact; retaining the tower convention is
    // nevertheless useful for generic MLE helpers.
    const EXTENSION_TAU: Self = Self::new(Block128::TAU, Block128::ZERO);

    fn invert(&self) -> Self {
        let hi2 = self.hi * self.hi;
        let lo2 = self.lo * self.lo;
        let hi_lo = self.hi * self.lo;
        let norm = hi2 * Self::TAU + hi_lo + lo2;
        let norm_inv = norm.invert();
        Self {
            lo: (self.hi + self.lo) * norm_inv,
            hi: self.hi * norm_inv,
        }
    }

    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        lo.copy_from_slice(&bytes[..16]);
        hi.copy_from_slice(&bytes[16..]);
        Self::new(
            Block128(u128::from_le_bytes(lo)),
            Block128(u128::from_le_bytes(hi)),
        )
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for Block256 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo + rhs.lo,
            hi: self.hi + rhs.hi,
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub for Block256 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Mul for Block256 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let v0 = self.lo * rhs.lo;
        let v1 = self.hi * rhs.hi;
        let v_sum = (self.lo + self.hi) * (rhs.lo + rhs.hi);
        Self {
            lo: v0 + v1 * Self::TAU,
            hi: v0 + v_sum,
        }
    }
}

impl AddAssign for Block256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Block256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Block256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl CanonicalSerialize for Block256 {
    fn serialized_size(&self) -> usize {
        32
    }

    fn serialize(&self, writer: &mut [u8]) -> Result<(), SerializationError> {
        if writer.len() < 32 {
            return Err(SerializationError);
        }
        writer[..16].copy_from_slice(&self.lo.0.to_le_bytes());
        writer[16..32].copy_from_slice(&self.hi.0.to_le_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for Block256 {
    fn deserialize(bytes: &[u8]) -> Result<Self, SerializationError> {
        if bytes.len() < 32 {
            return Err(SerializationError);
        }
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        lo.copy_from_slice(&bytes[..16]);
        hi.copy_from_slice(&bytes[16..32]);
        Ok(Self::new(
            Block128(u128::from_le_bytes(lo)),
            Block128(u128::from_le_bytes(hi)),
        ))
    }
}

macro_rules! from_base_integer {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for Block256 {
                fn from(value: $ty) -> Self {
                    Self::from_base(Block128::from(value))
                }
            }
        )*
    };
}

from_base_integer!(u8, u32, u64, u128);

impl From<Bit> for Block256 {
    fn from(value: Bit) -> Self {
        Self::from_base(Block128::from(value))
    }
}

impl From<Block8> for Block256 {
    fn from(value: Block8) -> Self {
        Self::from_base(Block128::from(value))
    }
}

impl From<Block16> for Block256 {
    fn from(value: Block16) -> Self {
        Self::from_base(Block128::from(value))
    }
}

impl From<Block32> for Block256 {
    fn from(value: Block32) -> Self {
        Self::from_base(Block128::from(value))
    }
}

impl From<Block64> for Block256 {
    fn from(value: Block64) -> Self {
        Self::from_base(Block128::from(value))
    }
}

impl From<Block128> for Block256 {
    fn from(value: Block128) -> Self {
        Self::from_base(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn embedded_base_field_is_a_homomorphism() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let left = Block128(rng.gen());
            let right = Block128(rng.gen());
            assert_eq!(
                Block256::from(left + right),
                Block256::from(left) + Block256::from(right)
            );
            assert_eq!(
                Block256::from(left * right),
                Block256::from(left) * Block256::from(right)
            );
        }
    }

    #[test]
    fn inversion_and_square_are_consistent() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let value = Block256::new(Block128(rng.gen()), Block128(rng.gen()));
            assert_eq!(value.square(), value * value);
            if value == Block256::ZERO {
                assert_eq!(value.invert(), Block256::ZERO);
            } else {
                assert_eq!(value * value.invert(), Block256::ONE);
            }
        }
    }

    #[test]
    fn canonical_round_trip() {
        let value = Block256::new(
            Block128(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210),
            Block128(0xf0e1_d2c3_b4a5_9687_7869_5a4b_3c2d_1e0f),
        );
        let bytes = value.to_bytes();
        assert_eq!(
            <Block256 as CanonicalDeserialize>::deserialize(&bytes).unwrap(),
            value
        );
    }

    #[test]
    fn challenge_support_excludes_the_base_subfield() {
        let mut rng = rand::thread_rng();
        for _ in 0..1_000 {
            let challenge =
                Block256::from_challenge_lanes(Block128(rng.gen()), Block128(rng.gen()));
            assert_ne!(challenge.hi, Block128::ZERO);
        }
    }
}
