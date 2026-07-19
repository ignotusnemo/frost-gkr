// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Domain separation used by the public FROST-GKR transcript.

use frost_gkr_core::Block128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainTag(pub [u8; 8]);

impl DomainTag {
    pub const fn new(label: &[u8; 8]) -> Self {
        let mut index = 0;
        while index < label.len() {
            assert!(label[index] < 128, "domain tag must be ASCII");
            index += 1;
        }
        Self(*label)
    }

    #[inline]
    pub const fn as_u64(&self) -> u64 {
        u64::from_be_bytes(self.0)
    }
}

#[inline]
pub fn capacity_iv(tag: DomainTag) -> [Block128; 2] {
    let label = tag.as_u64() as u128;
    [Block128::from(label << 64), Block128::from(label)]
}

/// Dedicated Fiat--Shamir domain for this standalone artifact.
pub const TAG_FROST_GKR: DomainTag = DomainTag::new(b"FROSTGKR");

#[cfg(test)]
mod tests {
    use super::*;
    use frost_gkr_core::TowerField;

    #[test]
    fn transcript_iv_is_nonzero_and_split() {
        let [high, low] = capacity_iv(TAG_FROST_GKR);
        assert_ne!(high, Block128::ZERO);
        assert_ne!(low, Block128::ZERO);
        assert_ne!(high, low);
    }
}
