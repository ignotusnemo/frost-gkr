// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

#![allow(clippy::needless_range_loop)]
// Coordinate-indexed loops mirror the Boolean-hypercube equations and are
// intentionally clearer here than iterator rewrites.

//! Reference implementation of FROST-GKR and a product-chain baseline for the
//! same public sequence of 59 Poseidon2b permutations.

pub mod batch_eval;
pub mod frost;
pub mod layers;
pub mod legacy;
pub mod mle_layout;
pub mod perm_sumcheck;
pub mod product_sumcheck;
pub mod sequence;
pub mod shift;
pub mod unified_mle;
pub mod unified_sumcheck;

pub use batch_eval::{
    prove_batch_eval, verify_batch_eval, BatchEvalProof, BatchEvalReduction, BatchEvalRound,
    EvalClaim,
};
pub use frost::{
    build_unified_from_sequence, discharge_frost_native, prove_frost, verify_frost, FrostProof,
    FrostReductions,
};
pub use layers::{evaluate_permutation, round_kind, PermLayerWitness, RoundKind};
pub use legacy::{
    build_boundary_mle, discharge_legacy_native, prove_legacy, verify_legacy, LegacyProof,
    N_BOUNDARY_CELLS, N_BOUNDARY_VARS,
};
pub use mle_layout::{pack_column, PermColumn, PermMle, N_PERM_CELLS, N_PERM_VARS};
pub use perm_sumcheck::{
    active_mle, build_active_mle, build_rc_mle, prove_perm, prove_perm_with_mle, rc_mle,
    verify_perm, PermProof, PermStateClaim, N_STATE_CLAIMS_PER_SLOT,
};
pub use product_sumcheck::{
    compute_product_claim, prove_product, prove_square, verify_product, ProductProof, RoundEvals,
};
pub use sequence::{
    evaluate_sequence, sequence_digest, SequenceInput, SequenceWitness, N_PERMUTATIONS,
};
pub use shift::{
    build_mds_lane_table, build_mu_table, build_rc_table, build_sigma_table, build_u_table,
    dec_round_index, elem_of, inc_round_index, mds_coeff, mu_evaluate, pack_index, permute_by_dec,
    project_lane, rc_evaluate, round_of, sigma_evaluate, slot_of,
};
pub use unified_mle::{
    build_unified_mle, sigma_at, UnifiedMle, N_LANE_VARS, N_ROUND_VARS, N_SLOT_VARS,
    N_UNIFIED_CELLS, N_UNIFIED_VARS,
};
pub use unified_sumcheck::{
    prove_shift, prove_unified_sumcheck, verify_shift, verify_unified_sumcheck, FrostCoreProof,
    ShiftProof, ShiftReduction, UnifiedProof, UnifiedReduction, N_UNIFIED_WITNESS_CLAIMS,
    SHIFT_ROUND_DEGREE, UNIFIED_ROUND_DEGREE,
};
