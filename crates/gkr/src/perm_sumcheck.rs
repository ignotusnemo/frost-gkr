// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Chained product-sumcheck reduction for one Poseidon2b permutation.
//!
//! Given a claim `v = sout_mle(r)` at a random 9-variable point `r`,
//! this protocol reduces it — via five product sumchecks plus three
//! "sin-expansion" product sumchecks — to a set of claims on the
//! `state` column MLE that the verifier checks against the public
//! state it reconstructs natively from `state_in`.
//!
//! Why product sumchecks are enough
//! --------------------------------
//!
//! The layered S-box is `sout = x4 · x3`, `x4 = x2 · x2`,
//! `x3 = x2 · sin`, `x2 = sin · sin`. Four two-variable product layers,
//! with two consumers of `x2`. The chain is:
//!
//! ```text
//!     v₀ = sout(r₀)
//! ──► sumcheck: sout = x4·x3       at r₀   → (x4(r₁),  x3(r₁))
//! ──► sumcheck: x4   = x2·x2       at r₁   → (x2(r₂),  x2(r₂))
//! ──► sumcheck: x3   = x2·sin      at r₁   → (x2(r₃),  sin(r₃))
//! ──► sumcheck: x2   = sin·sin     at r₂   → (sin(r₄), sin(r₄))
//! ──► sumcheck: x2   = sin·sin     at r₃   → (sin(r₅), sin(r₅))
//! ```
//!
//! For each of the three resulting `sin(ρ)` claims we invoke one more
//! product sumcheck on the identity
//!
//! ```text
//!     sin(x) = active(x) · (state(x) + rc(x))
//! ```
//!
//! (which holds on every hypercube vertex: on active cells because of
//! the S-box input rule; on inactive cells because both sides are zero).
//! The sumcheck `Σ_x eq(ρ,x) · active(x) · B(x) = sin(ρ)` with
//! `B = state + rc` reduces to `(active(ρ'), B(ρ'))`. The verifier
//! recomputes `active` and `rc` at `ρ'` publicly and cross-checks
//! `B(ρ') − rc(ρ')` against the honest `state(ρ')` it reconstructs
//! from `state_in`.
//!
//! The product-chain baseline checks the resulting state claims against the
//! native permutation witness through the terminal batch-evaluation argument.

use std::sync::OnceLock;

use frost_gkr_core::mle::evaluate::evaluate_slice;
use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::{Block128, TowerField};
use frost_gkr_poseidon2b::native::permutation::{N_ROUNDS, ROUND_CONSTANTS, STATE_SIZE};

use crate::layers::{evaluate_permutation, round_kind, RoundKind};
use crate::mle_layout::{PermMle, N_PERM_CELLS, N_PERM_VARS};
use crate::product_sumcheck::{prove_product, prove_square, verify_product, ProductProof};

/// Full permutation sumcheck proof. Five S-box chain sumchecks + three
/// sin-expansion sumchecks. All over 9-variable MLEs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermProof {
    pub sout_x4x3: ProductProof,
    pub x4_x2x2: ProductProof,
    pub x3_x2sin: ProductProof,
    pub x2_at_r2_sinsin: ProductProof,
    pub x2_at_r3_sinsin: ProductProof,
    pub sin_r3_check: ProductProof,
    pub sin_r4_check: ProductProof,
    pub sin_r5_check: ProductProof,
}

impl PermProof {
    /// Sum of the eight inner `ProductProof` sizes, in raw bytes.
    pub fn byte_len(&self) -> usize {
        self.sout_x4x3.byte_len()
            + self.x4_x2x2.byte_len()
            + self.x3_x2sin.byte_len()
            + self.x2_at_r2_sinsin.byte_len()
            + self.x2_at_r3_sinsin.byte_len()
            + self.sin_r3_check.byte_len()
            + self.sin_r4_check.byte_len()
            + self.sin_r5_check.byte_len()
    }
}

/// Output of a verified permutation sumcheck. The `sout` claim is reduced to
/// three evaluations of the permutation's `state` column; the caller batches
/// those evaluations across all 59 permutations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermStateClaim {
    /// 9-variable evaluation point on the per-slot `state` MLE.
    pub point: Vec<Block128>,
    /// `state_mle(point)` as reduced by the sumcheck chain. Deriving
    /// it from the proof's last `b_final` minus `rc(point)` is the
    /// same value an honest verifier would recompute; we return it
    /// explicitly so the downstream batching is a pure algebraic
    /// reduction (no re-derivation).
    pub value: Block128,
}

/// Three `(rs, state(rs))` claims drop out of a slot, one per
/// sin-expansion sumcheck (`sin_r3_check`, `sin_r4_check`,
/// `sin_r5_check`).
pub const N_STATE_CLAIMS_PER_SLOT: usize = 3;

/// Cached public constant MLEs. Both `active` and `rc` are fully
/// determined by the round schedule and the `ROUND_CONSTANTS`, so they
/// are identical across every slot and every proof. Building them once
/// per process replaces 79 allocations per proof with a single Arc-free
/// slice reference.
static ACTIVE_MLE: OnceLock<Vec<Block128>> = OnceLock::new();
static RC_MLE: OnceLock<Vec<Block128>> = OnceLock::new();

/// Borrow the cached `active` MLE without reallocating.
#[inline]
pub fn active_mle() -> &'static [Block128] {
    ACTIVE_MLE.get_or_init(build_active_mle)
}

/// Borrow the cached `rc` MLE without reallocating.
#[inline]
pub fn rc_mle() -> &'static [Block128] {
    RC_MLE.get_or_init(build_rc_mle)
}

/// Build the `active(row, lane)` MLE: 1 on cells the S-box touches, 0
/// elsewhere. Index `(row << 2) | lane`. Deterministic, public.
pub fn build_active_mle() -> Vec<Block128> {
    let mut v = vec![Block128::ZERO; N_PERM_CELLS];
    for r in 0..N_ROUNDS {
        match round_kind(r) {
            RoundKind::Full => {
                for lane in 0..STATE_SIZE {
                    v[(r << 2) | lane] = Block128::ONE;
                }
            }
            RoundKind::Partial => {
                v[r << 2] = Block128::ONE;
            }
        }
    }
    v
}

/// Build the `rc(row, lane)` MLE: `ROUND_CONSTANTS[lane][row]` on active
/// cells, 0 elsewhere. Deterministic, public.
pub fn build_rc_mle() -> Vec<Block128> {
    let mut v = vec![Block128::ZERO; N_PERM_CELLS];
    for r in 0..N_ROUNDS {
        match round_kind(r) {
            RoundKind::Full => {
                for lane in 0..STATE_SIZE {
                    v[(r << 2) | lane] = Block128::from(ROUND_CONSTANTS[lane][r]);
                }
            }
            RoundKind::Partial => {
                v[r << 2] = Block128::from(ROUND_CONSTANTS[0][r]);
            }
        }
    }
    v
}

/// Squeeze the 9-variable claim point.
fn squeeze_claim_point<T: FiatShamir<Block128>>(channel: &mut T) -> Vec<Block128> {
    (0..N_PERM_VARS).map(|_| channel.squeeze()).collect()
}

/// Honest prover. Squeezes the claim point `r₀`, evaluates
/// `sout_mle(r₀) = v₀`, runs the chain, and returns the proof together
/// with `r₀` and `v₀` so the caller can lift `sout(r₀) = v₀` into an
/// outer boundary batch-evaluation sumcheck.
pub fn prove_perm<T: FiatShamir<Block128>>(
    state_in: [Block128; STATE_SIZE],
    channel: &mut T,
) -> (
    PermProof,
    Vec<Block128>, // r0
    Block128,      // v0 = sout_mle(r0)
    [PermStateClaim; N_STATE_CLAIMS_PER_SLOT],
) {
    let witness = evaluate_permutation(state_in);
    let mle = PermMle::from_witness(&witness);
    prove_perm_with_mle(&mle, channel)
}

/// Like [`prove_perm`] but takes a pre-computed [`PermMle`], avoiding a
/// second witness evaluation and packing pass.
pub fn prove_perm_with_mle<T: FiatShamir<Block128>>(
    mle: &PermMle,
    channel: &mut T,
) -> (
    PermProof,
    Vec<Block128>,
    Block128,
    [PermStateClaim; N_STATE_CLAIMS_PER_SLOT],
) {
    let r0 = squeeze_claim_point(channel);
    let v0 = evaluate_slice(&mle.sout, &r0);

    // S-box chain. Three of the five product sumchecks have `A == B`
    // and use the `prove_square` fast path, which folds `A·A` via
    // `Block128::square()` (≈ 20× faster than the general two-operand
    // multiplication). Wire format and per-round transcript bytes are
    // identical to `prove_product`, so the verifier is unchanged.
    let (p_sout, r1) = prove_product(&mle.x4, &mle.x3, &r0, v0, channel);
    let (p_x4, r2) = prove_square(&mle.x2, &r1, p_sout.a_final, channel);
    let (p_x3, r3) = prove_product(&mle.x2, &mle.sin, &r1, p_sout.b_final, channel);
    let (p_x2a, r4) = prove_square(&mle.sin, &r2, p_x4.a_final, channel);
    let (p_x2b, r5) = prove_square(&mle.sin, &r3, p_x3.a_final, channel);

    // Sin expansion: sin(ρ) = Σ_x eq(ρ,x) · active(x) · (state(x) + rc(x)).
    let active = active_mle();
    let rc = rc_mle();
    let state_plus_rc: Vec<Block128> = mle
        .state
        .iter()
        .zip(rc.iter())
        .map(|(s, rc_i)| *s + *rc_i)
        .collect();

    let sin_at_r3 = p_x3.b_final;
    let sin_at_r4 = p_x2a.a_final;
    let sin_at_r5 = p_x2b.a_final;

    let (p_sin_r3, rs3) = prove_product(active, &state_plus_rc, &r3, sin_at_r3, channel);
    let (p_sin_r4, rs4) = prove_product(active, &state_plus_rc, &r4, sin_at_r4, channel);
    let (p_sin_r5, rs5) = prove_product(active, &state_plus_rc, &r5, sin_at_r5, channel);

    // γ₁: emit (rs, state_mle(rs)) directly from the honest witness so
    // the downstream batching can treat them as opaque claims.
    let state_at_rs3 = evaluate_slice(&mle.state, &rs3);
    let state_at_rs4 = evaluate_slice(&mle.state, &rs4);
    let state_at_rs5 = evaluate_slice(&mle.state, &rs5);

    let proof = PermProof {
        sout_x4x3: p_sout,
        x4_x2x2: p_x4,
        x3_x2sin: p_x3,
        x2_at_r2_sinsin: p_x2a,
        x2_at_r3_sinsin: p_x2b,
        sin_r3_check: p_sin_r3,
        sin_r4_check: p_sin_r4,
        sin_r5_check: p_sin_r5,
    };
    let claims = [
        PermStateClaim {
            point: rs3,
            value: state_at_rs3,
        },
        PermStateClaim {
            point: rs4,
            value: state_at_rs4,
        },
        PermStateClaim {
            point: rs5,
            value: state_at_rs5,
        },
    ];
    (proof, r0, v0, claims)
}

/// Honest verifier. Walks the sumcheck chain with the prover. The
/// chain bottoms out in three sin-expansion sumchecks; for each one
/// the verifier cross-checks `a_final == active(rs)` against the
/// public `active` MLE and then **emits** the implied
/// `state_mle(rs) = b_final - rc(rs)` claim upward (via
/// [`PermStateClaim`]). The caller is responsible for discharging
/// those three claims against the actual boundary MLE — `verify_perm`
/// does not reconstruct the state column.
///
/// `v0 = sout_mle(r0)` is supplied by the caller from the public benchmark
/// sequence.
///
/// Returns the three state-MLE claims on accept, `None` on reject.
pub fn verify_perm<T: FiatShamir<Block128>, F: FnOnce(&[Block128]) -> Block128>(
    proof: &PermProof,
    channel: &mut T,
    v0_provider: F,
) -> Option<[PermStateClaim; N_STATE_CLAIMS_PER_SLOT]> {
    let r0 = squeeze_claim_point(channel);
    let v0 = v0_provider(&r0);

    let r1 = verify_product(&proof.sout_x4x3, &r0, v0, channel)?;
    let r2 = verify_product(&proof.x4_x2x2, &r1, proof.sout_x4x3.a_final, channel)?;
    let r3 = verify_product(&proof.x3_x2sin, &r1, proof.sout_x4x3.b_final, channel)?;

    // `x4 = x2 · x2` is the same MLE on both sides of the product; reject
    // if the sumcheck produces different final values for the two copies.
    if proof.x4_x2x2.a_final != proof.x4_x2x2.b_final {
        return None;
    }

    let r4 = verify_product(&proof.x2_at_r2_sinsin, &r2, proof.x4_x2x2.a_final, channel)?;
    let r5 = verify_product(&proof.x2_at_r3_sinsin, &r3, proof.x3_x2sin.a_final, channel)?;

    if proof.x2_at_r2_sinsin.a_final != proof.x2_at_r2_sinsin.b_final {
        return None;
    }
    if proof.x2_at_r3_sinsin.a_final != proof.x2_at_r3_sinsin.b_final {
        return None;
    }

    // Three sin claims to discharge.
    let sin_at_r3 = proof.x3_x2sin.b_final;
    let sin_at_r4 = proof.x2_at_r2_sinsin.a_final;
    let sin_at_r5 = proof.x2_at_r3_sinsin.a_final;

    let active = active_mle();
    let rc = rc_mle();

    let rs3 = verify_product(&proof.sin_r3_check, &r3, sin_at_r3, channel)?;
    let claim3 = derive_state_claim(&proof.sin_r3_check, &rs3, active, rc)?;
    let rs4 = verify_product(&proof.sin_r4_check, &r4, sin_at_r4, channel)?;
    let claim4 = derive_state_claim(&proof.sin_r4_check, &rs4, active, rc)?;
    let rs5 = verify_product(&proof.sin_r5_check, &r5, sin_at_r5, channel)?;
    let claim5 = derive_state_claim(&proof.sin_r5_check, &rs5, active, rc)?;

    Some([claim3, claim4, claim5])
}

/// Cross-check a sin-expansion sumcheck's `a_final == active(rs)` and
/// extract the implied `state_mle(rs) = b_final - rc(rs)` claim. The
/// `active`/`rc` MLEs are public constants, so this check is
/// self-contained; `state_mle(rs)` is emitted upward to be discharged
/// later by the γ₂ batching sumcheck.
fn derive_state_claim(
    proof: &ProductProof,
    rs: &[Block128],
    active_mle: &[Block128],
    rc_mle: &[Block128],
) -> Option<PermStateClaim> {
    let active_val = evaluate_slice(active_mle, rs);
    if proof.a_final != active_val {
        return None;
    }
    let rc_val = evaluate_slice(rc_mle, rs);
    let state_val = proof.b_final + rc_val;
    Some(PermStateClaim {
        point: rs.to_vec(),
        value: state_val,
    })
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn active_mle_counts_match_schedule() {
        let a = build_active_mle();
        let mut ones = 0usize;
        for v in &a {
            if *v == Block128::ONE {
                ones += 1;
            } else {
                assert_eq!(*v, Block128::ZERO);
            }
        }
        // 8 full rounds × 4 lanes + 58 partial rounds × 1 lane = 32 + 58 = 90.
        assert_eq!(ones, 8 * STATE_SIZE + 58);
    }

    #[test]
    fn rc_mle_zero_on_inactive_cells() {
        let a = build_active_mle();
        let rc = build_rc_mle();
        for i in 0..N_PERM_CELLS {
            if a[i] == Block128::ZERO {
                assert_eq!(rc[i], Block128::ZERO);
            }
        }
    }
}
