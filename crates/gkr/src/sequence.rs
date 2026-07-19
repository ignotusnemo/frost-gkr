// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Public benchmark statement: a sequence of 59 Poseidon2b permutations.
//!
//! The output state of each permutation is the input state of the next.  This
//! deliberately application-neutral relation is sufficient to compare the
//! product-chain baseline with FROST-GKR without publishing any surrounding
//! protocol.

use frost_gkr_core::Block128;
use frost_gkr_poseidon2b::native::permutation::{Poseidon2bPermutation, STATE_SIZE};

/// Number of Poseidon2b permutations used by the paper benchmark.
pub const N_PERMUTATIONS: usize = 59;

/// Public input to the fixed benchmark sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceInput {
    pub initial_state: [Block128; STATE_SIZE],
}

/// Native evaluation of the benchmark sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceWitness {
    pub states: Vec<([Block128; STATE_SIZE], [Block128; STATE_SIZE])>,
    pub final_state: [Block128; STATE_SIZE],
}

/// Evaluate the public sequence natively.
pub fn evaluate_sequence(input: &SequenceInput) -> SequenceWitness {
    let permutation = Poseidon2bPermutation;
    let mut current = input.initial_state;
    let mut states = Vec::with_capacity(N_PERMUTATIONS);

    for _ in 0..N_PERMUTATIONS {
        let state_in = current;
        permutation.permute_mut(&mut current);
        states.push((state_in, current));
    }

    SequenceWitness {
        states,
        final_state: current,
    }
}

/// Two-lane public digest used to bind the sequence to the transcript.
pub fn sequence_digest(input: &SequenceInput) -> [Block128; 2] {
    let final_state = evaluate_sequence(input).final_state;
    [final_state[0], final_state[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use frost_gkr_core::TowerField;

    fn fixture() -> SequenceInput {
        SequenceInput {
            initial_state: [
                Block128::from(1u128),
                Block128::from(2u128),
                Block128::from(3u128),
                Block128::from(4u128),
            ],
        }
    }

    #[test]
    fn sequence_has_fixed_length_and_links_adjacent_states() {
        let witness = evaluate_sequence(&fixture());
        assert_eq!(witness.states.len(), N_PERMUTATIONS);
        for pair in witness.states.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
        assert_eq!(witness.states.last().unwrap().1, witness.final_state);
    }

    #[test]
    fn digest_is_deterministic_and_input_bound() {
        let input = fixture();
        assert_eq!(sequence_digest(&input), sequence_digest(&input));

        let mut changed = input;
        changed.initial_state[0] += Block128::ONE;
        assert_ne!(sequence_digest(&input), sequence_digest(&changed));
    }
}
