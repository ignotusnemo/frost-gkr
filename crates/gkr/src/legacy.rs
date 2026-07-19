// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Product-chain baseline for the fixed 59-permutation sequence.
//!
//! Each permutation is proved independently with eight nine-variable product
//! sumchecks.  A final batch-evaluation sumcheck collapses the resulting state
//! claims to one terminal MLE evaluation.  FROST-GKR proves the same native
//! sequence and uses the same field and Fiat--Shamir channel.

use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::{Block128, TowerField};
use frost_gkr_poseidon2b::native::permutation::STATE_SIZE;
use rayon::prelude::*;

use crate::batch_eval::{
    prove_batch_eval, verify_batch_eval, BatchEvalProof, BatchEvalReduction, EvalClaim,
};
use crate::layers::evaluate_permutation;
use crate::mle_layout::{pack_sout, PermMle, N_PERM_CELLS, N_PERM_VARS};
use crate::perm_sumcheck::{prove_perm_with_mle, verify_perm, PermProof, PermStateClaim};
use crate::sequence::{evaluate_sequence, SequenceInput, N_PERMUTATIONS};

/// Smallest power of two at least `N_PERMUTATIONS`.
pub const N_PERMUTATIONS_PADDED: usize = 64;
/// Number of Boolean variables used to index the padded permutation axis.
pub const N_SEQUENCE_VARS: usize = 6;
/// Variables in the concatenated baseline boundary MLE.
pub const N_BOUNDARY_VARS: usize = N_SEQUENCE_VARS + N_PERM_VARS;
/// Cells in the concatenated baseline boundary MLE.
pub const N_BOUNDARY_CELLS: usize = 1 << N_BOUNDARY_VARS;

/// Algebraic proof emitted by the product-chain baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyProof {
    pub permutations: Vec<PermProof>,
    pub boundary: BatchEvalProof,
}

impl LegacyProof {
    pub fn byte_len(&self) -> usize {
        self.permutations
            .iter()
            .map(PermProof::byte_len)
            .sum::<usize>()
            + self.boundary.byte_len()
    }
}

/// Build the concatenated state-column MLE used by the baseline reduction.
pub fn build_boundary_mle(
    states: &[([Block128; STATE_SIZE], [Block128; STATE_SIZE])],
) -> Vec<Block128> {
    assert_eq!(states.len(), N_PERMUTATIONS);
    let mut boundary = vec![Block128::ZERO; N_BOUNDARY_CELLS];
    for (index, (state_in, _)) in states.iter().enumerate() {
        let witness = evaluate_permutation(*state_in);
        let state_mle = PermMle::from_witness(&witness).state;
        let offset = index << N_PERM_VARS;
        boundary[offset..offset + N_PERM_CELLS].copy_from_slice(&state_mle);
    }
    boundary
}

fn assemble_boundary_mle(mles: &[PermMle]) -> Vec<Block128> {
    assert_eq!(mles.len(), N_PERMUTATIONS);
    let mut boundary = vec![Block128::ZERO; N_BOUNDARY_CELLS];
    for (index, mle) in mles.iter().enumerate() {
        let offset = index << N_PERM_VARS;
        boundary[offset..offset + N_PERM_CELLS].copy_from_slice(&mle.state);
    }
    boundary
}

fn sequence_index_to_bits(index: usize) -> Vec<Block128> {
    (0..N_SEQUENCE_VARS)
        .map(|bit| {
            if (index >> bit) & 1 == 1 {
                Block128::ONE
            } else {
                Block128::ZERO
            }
        })
        .collect()
}

fn lift_claim(index: usize, claim: &PermStateClaim) -> EvalClaim {
    let mut point = Vec::with_capacity(N_BOUNDARY_VARS);
    point.extend_from_slice(&claim.point);
    point.extend_from_slice(&sequence_index_to_bits(index));
    EvalClaim {
        point,
        value: claim.value,
    }
}

fn absorb_digest<T: FiatShamir<Block128>>(channel: &mut T, digest: &[Block128; 2]) {
    channel.absorb(digest[0]);
    channel.absorb(digest[1]);
}

/// Prove the fixed sequence with the product-chain baseline.
pub fn prove_legacy<T: FiatShamir<Block128>>(
    input: &SequenceInput,
    claimed_digest: [Block128; 2],
    channel: &mut T,
) -> (LegacyProof, BatchEvalReduction) {
    let witness = evaluate_sequence(input);
    debug_assert_eq!(
        [witness.final_state[0], witness.final_state[1]],
        claimed_digest
    );
    absorb_digest(channel, &claimed_digest);

    let mles: Vec<PermMle> = witness
        .states
        .par_iter()
        .map(|(state_in, _)| PermMle::from_witness(&evaluate_permutation(*state_in)))
        .collect();

    let mut permutation_proofs = Vec::with_capacity(N_PERMUTATIONS);
    let mut claims = Vec::with_capacity(N_PERMUTATIONS * 3);
    for (index, mle) in mles.iter().enumerate() {
        let (proof, _, _, state_claims) = prove_perm_with_mle(mle, channel);
        permutation_proofs.push(proof);
        claims.extend(state_claims.iter().map(|claim| lift_claim(index, claim)));
    }

    let boundary_mle = assemble_boundary_mle(&mles);
    let (boundary, reduction) = prove_batch_eval(&boundary_mle, &claims, channel);
    (
        LegacyProof {
            permutations: permutation_proofs,
            boundary,
        },
        reduction,
    )
}

/// Verify the product-chain baseline and return its terminal MLE reduction.
pub fn verify_legacy<T: FiatShamir<Block128>>(
    proof: &LegacyProof,
    input: &SequenceInput,
    claimed_digest: [Block128; 2],
    channel: &mut T,
) -> Option<BatchEvalReduction> {
    if proof.permutations.len() != N_PERMUTATIONS {
        return None;
    }

    let witness = evaluate_sequence(input);
    if [witness.final_state[0], witness.final_state[1]] != claimed_digest {
        return None;
    }
    absorb_digest(channel, &claimed_digest);

    let sout_mles: Vec<Vec<Block128>> = witness
        .states
        .par_iter()
        .map(|(state_in, _)| pack_sout(&evaluate_permutation(*state_in)))
        .collect();

    let mut claims = Vec::with_capacity(N_PERMUTATIONS * 3);
    for (index, permutation_proof) in proof.permutations.iter().enumerate() {
        let state_claims = verify_perm(permutation_proof, channel, |point| {
            frost_gkr_core::mle::evaluate::evaluate_slice(&sout_mles[index], point)
        })?;
        claims.extend(state_claims.iter().map(|claim| lift_claim(index, claim)));
    }

    verify_batch_eval(&proof.boundary, &claims, N_BOUNDARY_VARS, channel)
}

/// Evaluate the terminal baseline reduction directly.
pub fn discharge_legacy_native(input: &SequenceInput, reduction: &BatchEvalReduction) -> bool {
    let witness = evaluate_sequence(input);
    let boundary = build_boundary_mle(&witness.states);
    frost_gkr_core::mle::evaluate::evaluate_slice(&boundary, &reduction.point) == reduction.value
}

#[cfg(test)]
mod tests {
    use super::*;
    use frost_gkr_poseidon2b::channel::Poseidon2bChannel;

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
    fn legacy_round_trip_and_native_discharge() {
        let input = fixture();
        let digest = crate::sequence::sequence_digest(&input);

        let mut prover_channel = Poseidon2bChannel::new();
        let (proof, prover_reduction) = prove_legacy(&input, digest, &mut prover_channel);

        let mut verifier_channel = Poseidon2bChannel::new();
        let verifier_reduction =
            verify_legacy(&proof, &input, digest, &mut verifier_channel).unwrap();

        assert_eq!(prover_reduction, verifier_reduction);
        assert!(discharge_legacy_native(&input, &verifier_reduction));
    }

    #[test]
    fn wrong_digest_is_rejected() {
        let input = fixture();
        let digest = crate::sequence::sequence_digest(&input);
        let mut prover_channel = Poseidon2bChannel::new();
        let (proof, _) = prove_legacy(&input, digest, &mut prover_channel);

        let mut bad = digest;
        bad[0] += Block128::ONE;
        let mut verifier_channel = Poseidon2bChannel::new();
        assert!(verify_legacy(&proof, &input, bad, &mut verifier_channel).is_none());
    }
}
