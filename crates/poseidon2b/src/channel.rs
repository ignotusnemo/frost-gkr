// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo. All rights reserved.

//! Poseidon2b-backed Fiat-Shamir channel implementing
//! [`frost_gkr_core::transcript::FiatShamir`].
//!
//! The channel is seeded with the dedicated `FROSTGKR` capacity IV. Every
//! squeeze advances the sponge by one permutation and emits one `Block128`
//! challenge.

use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::Block128;

use crate::native::domain::{capacity_iv, TAG_FROST_GKR};
use crate::native::sponge::Poseidon2bSponge;

/// Fiat-Shamir channel backed by a Poseidon2b sponge.
#[derive(Debug, Clone)]
pub struct Poseidon2bChannel {
    sponge: Poseidon2bSponge,
    /// When we squeeze a rate block (two Block128s) we hand out the second
    /// one on the next call before advancing the sponge again.
    pending: Option<Block128>,
}

impl Default for Poseidon2bChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl Poseidon2bChannel {
    pub fn new() -> Self {
        Self {
            sponge: Poseidon2bSponge::with_iv(capacity_iv(TAG_FROST_GKR)),
            pending: None,
        }
    }
}

impl FiatShamir<Block128> for Poseidon2bChannel {
    fn absorb(&mut self, elem: Block128) {
        // Absorbing invalidates any buffered challenge — future squeezes
        // must reflect the new state.
        self.pending = None;
        self.sponge.absorb(elem);
    }

    fn squeeze(&mut self) -> Block128 {
        if let Some(b) = self.pending.take() {
            return b;
        }
        // Commit any buffered absorb bytes to state before reading.
        self.sponge.flush_to_squeeze();
        let [a, b] = self.sponge.squeeze();
        self.pending = Some(b);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_is_deterministic() {
        let mut c1 = Poseidon2bChannel::new();
        c1.absorb(Block128::from(42u128));
        let a1 = c1.squeeze();
        let b1 = c1.squeeze();

        let mut c2 = Poseidon2bChannel::new();
        c2.absorb(Block128::from(42u128));
        let a2 = c2.squeeze();
        let b2 = c2.squeeze();

        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn distinct_inputs_distinct_challenges() {
        let mut c1 = Poseidon2bChannel::new();
        c1.absorb(Block128::from(1u128));
        let a = c1.squeeze();

        let mut c2 = Poseidon2bChannel::new();
        c2.absorb(Block128::from(2u128));
        let b = c2.squeeze();

        assert_ne!(a, b);
    }

    #[test]
    fn channel_iv_diverges_from_bare_sponge() {
        // Same absorb, different IV => different squeezes.
        let mut c = Poseidon2bChannel::new();
        c.absorb(Block128::from(1u128));
        let iv_challenge = c.squeeze();

        let mut bare = Poseidon2bSponge::new();
        bare.absorb(Block128::from(1u128));
        bare.flush_to_squeeze();
        let [bare_a, _] = bare.squeeze();

        assert_ne!(iv_challenge, bare_a);
    }
}
