// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Minimal Poseidon2b sponge used by the Fiat--Shamir channel.
//!
//! Sponge parameters: t=4, rate=2, cap=2.

use super::permutation::Poseidon2bPermutation;
use frost_gkr_core::{Block128, CanonicalSerialize, TowerField};

const STATE_SIZE: usize = 4;
const RATE: usize = 2;

const PADDING_START: u8 = 0x80;
const PADDING_END: u8 = 0x01;

/// Poseidon2b sponge with t=4, rate=2, capacity=2.
#[derive(Debug, Clone)]
pub struct Poseidon2bSponge {
    state: [Block128; STATE_SIZE],
    buffer: [u8; 32],
    filled_bytes: usize,
    permutation: Poseidon2bPermutation,
}

impl Default for Poseidon2bSponge {
    fn default() -> Self {
        Self::new()
    }
}

impl Poseidon2bSponge {
    pub fn new() -> Self {
        Self {
            state: [Block128::ZERO; STATE_SIZE],
            buffer: [0u8; 32],
            filled_bytes: 0,
            permutation: Poseidon2bPermutation,
        }
    }

    /// Construct a sponge seeded with a capacity IV.
    /// `state[0]`, `state[1]` are zeroed (rate); `state[2]`, `state[3]`
    /// carry the IV.
    pub fn with_iv(iv: [Block128; 2]) -> Self {
        Self {
            state: [Block128::ZERO, Block128::ZERO, iv[0], iv[1]],
            buffer: [0u8; 32],
            filled_bytes: 0,
            permutation: Poseidon2bPermutation,
        }
    }

    /// Absorb raw bytes into the sponge.
    pub fn update(&mut self, mut data: &[u8]) {
        if self.filled_bytes != 0 {
            let to_copy = std::cmp::min(data.len(), 32 - self.filled_bytes);
            self.buffer[self.filled_bytes..self.filled_bytes + to_copy]
                .copy_from_slice(&data[..to_copy]);
            data = &data[to_copy..];
            self.filled_bytes += to_copy;

            if self.filled_bytes == 32 {
                self.permute_buffer();
                self.filled_bytes = 0;
            }
        }

        for chunk in data.chunks_exact(32) {
            self.buffer.copy_from_slice(chunk);
            self.permute_buffer();
        }

        let remaining = data.chunks_exact(32).remainder();
        if !remaining.is_empty() {
            self.buffer[..remaining.len()].copy_from_slice(remaining);
            self.filled_bytes = remaining.len();
        }
    }

    /// Absorb a single field element.
    pub fn absorb(&mut self, elem: Block128) {
        let bytes = elem.to_bytes();
        self.update(&bytes);
    }

    /// Squeeze two field elements without finalizing (for streaming).
    pub fn squeeze(&mut self) -> [Block128; 2] {
        let out = [self.state[0], self.state[1]];
        self.permutation.permute_mut(&mut self.state);
        out
    }

    /// Flush any buffered absorb bytes into the state via one padded
    /// permutation, so subsequent `squeeze()` calls are guaranteed to
    /// commit to everything absorbed so far.
    ///
    /// Idempotent when no data is pending *and* the caller has not yet
    /// squeezed. Always safe to call before switching from absorb to
    /// squeeze mode.
    pub fn flush_to_squeeze(&mut self) {
        if self.filled_bytes != 0 {
            fill_padding(&mut self.buffer[self.filled_bytes..]);
            self.permute_buffer();
            self.filled_bytes = 0;
        }
    }

    fn permute_buffer(&mut self) {
        // XOR buffer into rate portion of state
        for i in 0..RATE {
            let mut word = [0u8; 16];
            word.copy_from_slice(&self.buffer[i * 16..(i + 1) * 16]);
            let elem = Block128::from(u128::from_le_bytes(word));
            self.state[i] += elem;
        }
        self.permutation.permute_mut(&mut self.state);
    }
}

#[inline(always)]
fn fill_padding(data: &mut [u8]) {
    debug_assert!(!data.is_empty() && data.len() <= 32);
    data.fill(0);
    data[0] |= PADDING_START;
    data[data.len() - 1] |= PADDING_END;
}
