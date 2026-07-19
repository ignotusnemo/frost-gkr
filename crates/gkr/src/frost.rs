// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! FROST-GKR orchestration for the fixed 59-permutation sequence.
//!
//! The main degree-nine sumcheck proves the unified Poseidon2b relation.  The
//! degree-two shift sumcheck reduces shifted-table claims to the three original
//! witness columns.  Three batch-evaluation arguments then expose one terminal
//! claim for each column.

use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::Block128;
use frost_gkr_poseidon2b::native::permutation::STATE_SIZE;

use crate::batch_eval::{
    prove_batch_eval, verify_batch_eval, BatchEvalProof, BatchEvalReduction, EvalClaim,
};
use crate::sequence::{evaluate_sequence, SequenceInput, N_PERMUTATIONS};
use crate::unified_mle::{build_unified_mle, UnifiedMle, N_UNIFIED_VARS};
use crate::unified_sumcheck::{
    prove_shift, prove_unified_sumcheck, verify_shift, verify_unified_sumcheck, FrostCoreProof,
};

/// Algebraic proof emitted by FROST-GKR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrostProof {
    pub frost: FrostCoreProof,
    pub state_batch: BatchEvalProof,
    pub sin_batch: BatchEvalProof,
    pub sout_batch: BatchEvalProof,
}

impl FrostProof {
    pub fn byte_len(&self) -> usize {
        let main_polys = self.frost.main.round_polys.len() * 10 * 16;
        let shift_polys = self.frost.shift.round_polys.len() * 3 * 16;
        let main_finals = 12 * 16;
        let shift_finals = 3 * 16;
        main_polys
            + shift_polys
            + main_finals
            + shift_finals
            + self.state_batch.byte_len()
            + self.sin_batch.byte_len()
            + self.sout_batch.byte_len()
    }
}

/// Terminal MLE reductions emitted by the FROST-GKR verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrostReductions {
    pub state: BatchEvalReduction,
    pub sin: BatchEvalReduction,
    pub sout: BatchEvalReduction,
}

fn absorb_digest<T: FiatShamir<Block128>>(channel: &mut T, digest: &[Block128; 2]) {
    channel.absorb(digest[0]);
    channel.absorb(digest[1]);
}

/// Materialize the three-column unified witness for the public sequence.
pub fn build_unified_from_sequence(input: &SequenceInput) -> UnifiedMle {
    let witness = evaluate_sequence(input);
    assert_eq!(witness.states.len(), N_PERMUTATIONS);
    let state_inputs: Vec<[Block128; STATE_SIZE]> = witness
        .states
        .iter()
        .map(|(state_in, _)| *state_in)
        .collect();
    build_unified_mle(&state_inputs).0
}

/// Prove the fixed sequence with FROST-GKR.
pub fn prove_frost<T: FiatShamir<Block128>>(
    input: &SequenceInput,
    claimed_digest: [Block128; 2],
    channel: &mut T,
) -> (FrostProof, FrostReductions) {
    let witness = evaluate_sequence(input);
    debug_assert_eq!(
        [witness.final_state[0], witness.final_state[1]],
        claimed_digest
    );
    absorb_digest(channel, &claimed_digest);

    let mle = build_unified_from_sequence(input);
    let (main, r_prime) = prove_unified_sumcheck(&mle, channel);
    let (shift, r_double_prime) = prove_shift(&mle, &r_prime, channel);

    let state_claims = vec![
        EvalClaim {
            point: r_prime,
            value: main.state_at_r,
        },
        EvalClaim {
            point: r_double_prime.clone(),
            value: shift.state_at_r2,
        },
    ];
    let (state_batch, state_reduction) = prove_batch_eval(&mle.state, &state_claims, channel);

    let sin_claims = vec![EvalClaim {
        point: r_double_prime.clone(),
        value: shift.s_in_at_r2,
    }];
    let (sin_batch, sin_reduction) = prove_batch_eval(&mle.s_in, &sin_claims, channel);

    let sout_claims = vec![EvalClaim {
        point: r_double_prime,
        value: shift.s_out_at_r2,
    }];
    let (sout_batch, sout_reduction) = prove_batch_eval(&mle.s_out, &sout_claims, channel);

    (
        FrostProof {
            frost: FrostCoreProof { main, shift },
            state_batch,
            sin_batch,
            sout_batch,
        },
        FrostReductions {
            state: state_reduction,
            sin: sin_reduction,
            sout: sout_reduction,
        },
    )
}

/// Verify FROST-GKR and return its three terminal MLE reductions.
pub fn verify_frost<T: FiatShamir<Block128>>(
    proof: &FrostProof,
    input: &SequenceInput,
    claimed_digest: [Block128; 2],
    channel: &mut T,
) -> Option<FrostReductions> {
    let witness = evaluate_sequence(input);
    if [witness.final_state[0], witness.final_state[1]] != claimed_digest {
        return None;
    }
    absorb_digest(channel, &claimed_digest);

    let main = verify_unified_sumcheck(&proof.frost.main, channel)?;
    let shift = verify_shift(&proof.frost.shift, &main, channel)?;

    let state_claims = vec![
        EvalClaim {
            point: main.r_prime,
            value: main.state_at_r,
        },
        EvalClaim {
            point: shift.r_double_prime.clone(),
            value: shift.state_at_r2,
        },
    ];
    let state = verify_batch_eval(&proof.state_batch, &state_claims, N_UNIFIED_VARS, channel)?;

    let sin_claims = vec![EvalClaim {
        point: shift.r_double_prime.clone(),
        value: shift.s_in_at_r2,
    }];
    let sin = verify_batch_eval(&proof.sin_batch, &sin_claims, N_UNIFIED_VARS, channel)?;

    let sout_claims = vec![EvalClaim {
        point: shift.r_double_prime,
        value: shift.s_out_at_r2,
    }];
    let sout = verify_batch_eval(&proof.sout_batch, &sout_claims, N_UNIFIED_VARS, channel)?;

    Some(FrostReductions { state, sin, sout })
}

/// Evaluate all three terminal FROST-GKR reductions directly.
pub fn discharge_frost_native(input: &SequenceInput, reductions: &FrostReductions) -> bool {
    use frost_gkr_core::mle::evaluate::evaluate_slice;

    let mle = build_unified_from_sequence(input);
    evaluate_slice(&mle.state, &reductions.state.point) == reductions.state.value
        && evaluate_slice(&mle.s_in, &reductions.sin.point) == reductions.sin.value
        && evaluate_slice(&mle.s_out, &reductions.sout.point) == reductions.sout.value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::sequence_digest;
    use frost_gkr_core::TowerField;
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
    fn frost_round_trip_and_native_discharge() {
        let input = fixture();
        let digest = sequence_digest(&input);
        let mut prover_channel = Poseidon2bChannel::new();
        let (proof, prover_reductions) = prove_frost(&input, digest, &mut prover_channel);

        let mut verifier_channel = Poseidon2bChannel::new();
        let verifier_reductions =
            verify_frost(&proof, &input, digest, &mut verifier_channel).unwrap();

        assert_eq!(prover_reductions, verifier_reductions);
        assert!(discharge_frost_native(&input, &verifier_reductions));
    }

    #[test]
    fn tampered_main_claim_is_rejected() {
        let input = fixture();
        let digest = sequence_digest(&input);
        let mut prover_channel = Poseidon2bChannel::new();
        let (mut proof, _) = prove_frost(&input, digest, &mut prover_channel);
        proof.frost.main.state_at_r += Block128::ONE;

        let mut verifier_channel = Poseidon2bChannel::new();
        assert!(verify_frost(&proof, &input, digest, &mut verifier_channel).is_none());
    }

    #[test]
    fn tampered_shift_claim_is_rejected() {
        let input = fixture();
        let digest = sequence_digest(&input);
        let mut prover_channel = Poseidon2bChannel::new();
        let (mut proof, _) = prove_frost(&input, digest, &mut prover_channel);
        proof.frost.shift.s_in_at_r2 += Block128::ONE;

        let mut verifier_channel = Poseidon2bChannel::new();
        assert!(verify_frost(&proof, &input, digest, &mut verifier_channel).is_none());
    }

    #[test]
    fn tampered_batch_reduction_is_rejected() {
        let input = fixture();
        let digest = sequence_digest(&input);
        let mut prover_channel = Poseidon2bChannel::new();
        let (mut proof, _) = prove_frost(&input, digest, &mut prover_channel);
        proof.state_batch.b_final += Block128::ONE;

        let mut verifier_channel = Poseidon2bChannel::new();
        assert!(verify_frost(&proof, &input, digest, &mut verifier_channel).is_none());
    }

    #[test]
    fn wrong_digest_is_rejected() {
        let input = fixture();
        let digest = sequence_digest(&input);
        let mut prover_channel = Poseidon2bChannel::new();
        let (proof, _) = prove_frost(&input, digest, &mut prover_channel);

        let mut bad = digest;
        bad[1] += Block128::ONE;
        let mut verifier_channel = Poseidon2bChannel::new();
        assert!(verify_frost(&proof, &input, bad, &mut verifier_channel).is_none());
    }
}
