// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

use frost_gkr::{
    discharge_frost_native, discharge_legacy_native, prove_frost, prove_legacy, sequence_digest,
    verify_frost, verify_legacy, SequenceInput,
};
use frost_gkr_core::{Block128, TowerField};
use frost_gkr_poseidon2b::Poseidon2bChannel;

const LEGACY_EXPECTED_BYTES: usize = 287_712;
const FROST_EXPECTED_BYTES: usize = 5_568;

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
fn both_protocols_prove_the_same_sequence() {
    let input = fixture();
    let digest = sequence_digest(&input);

    let mut legacy_prover_channel = Poseidon2bChannel::new();
    let (legacy_proof, legacy_prover_reduction) =
        prove_legacy(&input, digest, &mut legacy_prover_channel);
    let mut legacy_verifier_channel = Poseidon2bChannel::new();
    let legacy_verifier_reduction =
        verify_legacy(&legacy_proof, &input, digest, &mut legacy_verifier_channel).unwrap();
    assert_eq!(legacy_prover_reduction, legacy_verifier_reduction);
    assert!(discharge_legacy_native(&input, &legacy_verifier_reduction));

    let mut frost_prover_channel = Poseidon2bChannel::new();
    let (frost_proof, frost_prover_reductions) =
        prove_frost(&input, digest, &mut frost_prover_channel);
    let mut frost_verifier_channel = Poseidon2bChannel::new();
    let frost_verifier_reductions =
        verify_frost(&frost_proof, &input, digest, &mut frost_verifier_channel).unwrap();
    assert_eq!(frost_prover_reductions, frost_verifier_reductions);
    assert!(discharge_frost_native(&input, &frost_verifier_reductions));

    assert_eq!(legacy_proof.byte_len(), LEGACY_EXPECTED_BYTES);
    assert_eq!(frost_proof.byte_len(), FROST_EXPECTED_BYTES);
}

#[test]
fn changing_the_public_input_invalidates_both_proofs() {
    let input = fixture();
    let digest = sequence_digest(&input);

    let mut legacy_prover_channel = Poseidon2bChannel::new();
    let (legacy_proof, _) = prove_legacy(&input, digest, &mut legacy_prover_channel);
    let mut frost_prover_channel = Poseidon2bChannel::new();
    let (frost_proof, _) = prove_frost(&input, digest, &mut frost_prover_channel);

    let mut changed = input;
    changed.initial_state[0] += Block128::ONE;

    let mut legacy_verifier_channel = Poseidon2bChannel::new();
    assert!(verify_legacy(
        &legacy_proof,
        &changed,
        digest,
        &mut legacy_verifier_channel
    )
    .is_none());

    let mut frost_verifier_channel = Poseidon2bChannel::new();
    assert!(verify_frost(&frost_proof, &changed, digest, &mut frost_verifier_channel).is_none());
}
