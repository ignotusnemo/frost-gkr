// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Product sumcheck primitive.
//!
//! Reduces a claim `v = Σ_x eq(r, x) · A(x) · B(x)` over the boolean
//! hypercube `{0,1}^n` to two smaller claims `a = A(r')` and
//! `b = B(r')` at a fresh point `r'` of the same length. `r'` is the
//! vector of per-round Fiat-Shamir challenges, so the verifier
//! recomputes it from the transcript.
//!
//! Protocol shape (standard Thaler / Libra product sumcheck):
//!
//! ```text
//! round i:   prover emits p_i(X) = deg-3 univariate as 4 evaluations
//!              (e0, e1, e2, e3) at X = 0, 1, 2, 3 (field elements).
//!            transcript absorbs the four coefficients.
//!            challenge r_i = channel.squeeze()
//!            update running claim to p_i(r_i)
//!            fold eq, A, B tables by r_i (highest-variable-first)
//!
//! after n rounds:
//!            prover sends (a, b) = (A(r'), B(r')).
//!            verifier accepts iff final_claim == eq(r, r') · a · b
//! ```
//!
//! `r'` is returned in **variable order** (`r'[k]` is the final
//! binding of variable `k`). Because we fold highest-var-first, the
//! variable-order point is the push-order challenge vector reversed.
//!
//! The transcript is supplied through the generic `FiatShamir<Block128>`
//! interface; this artifact uses `Poseidon2bChannel`.

use std::sync::OnceLock;

use frost_gkr_core::mle::eq::{eq_ind, eq_ind_partial_eval};
use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::{Block128, TowerField};
use rayon::prelude::*;

/// Inverse of the Lagrange denominators at the fixed evaluation points
/// `{0,1,2,3}` over GF(2^128). Cached once per process to avoid 4
/// `Block128::invert()` calls inside every per-round `evaluate`.
fn denom_inv_4() -> &'static [Block128; 4] {
    static CACHE: OnceLock<[Block128; 4]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut out = [Block128::ZERO; 4];
        for k in 0..4 {
            let xk = Block128::from(k as u128);
            let mut d = Block128::ONE;
            for j in 0..4 {
                if j == k {
                    continue;
                }
                d *= xk + Block128::from(j as u128);
            }
            out[k] = d.invert();
        }
        out
    })
}

/// Per-round work in `eval_round_at_*` is `O(half)` independent muls
/// over 9-variable MLEs (`half` ≤ 256). Switch to parallel iteration
/// only when there's enough work to amortise rayon's join overhead.
const PAR_THRESHOLD: usize = 64;

/// Round polynomial stored as its evaluations at `X = 0, 1, 2, 3`.
///
/// Sumcheck telescope check uses `evals[0] + evals[1] == running_claim`.
/// Lagrange evaluation at a challenge `r` uses the explicit
/// interpolant below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundEvals {
    pub evals: [Block128; 4],
}

impl RoundEvals {
    #[inline]
    pub fn sum_at_0_plus_1(&self) -> Block128 {
        self.evals[0] + self.evals[1]
    }

    /// Lagrange-interpolate at `r` from evaluations at `{0,1,2,3}`.
    pub fn evaluate(&self, r: Block128) -> Block128 {
        lagrange_at_0_1_2_3(&self.evals, r)
    }
}

/// Lagrange evaluation: given `e_k = p(k)` for `k ∈ {0,1,2,3}`,
/// return `p(r)`. Uses the standard Lagrange basis in GF(2^128).
/// Denominator inverses are cached in `denom_inv_4()` so the hot path
/// performs zero field inversions.
#[inline]
pub fn lagrange_at_0_1_2_3(evals: &[Block128; 4], r: Block128) -> Block128 {
    let denom_inv = denom_inv_4();
    let r0 = r + Block128::from(0u128);
    let r1 = r + Block128::from(1u128);
    let r2 = r + Block128::from(2u128);
    let r3 = r + Block128::from(3u128);
    // L_k(r) = Π_{j≠k} (r + x_j) · denom_inv[k].
    let n0 = r1 * r2 * r3;
    let n1 = r0 * r2 * r3;
    let n2 = r0 * r1 * r3;
    let n3 = r0 * r1 * r2;
    evals[0] * n0 * denom_inv[0]
        + evals[1] * n1 * denom_inv[1]
        + evals[2] * n2 * denom_inv[2]
        + evals[3] * n3 * denom_inv[3]
}

/// Full product-sumcheck proof: the per-round evaluations plus the
/// final reduced pair `(a, b)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductProof {
    pub rounds: Vec<RoundEvals>,
    pub a_final: Block128,
    pub b_final: Block128,
}

impl ProductProof {
    /// Raw field-element accounting of this proof's wire size.
    /// `rounds.len() * 4` block-128s for the round polys plus the two
    /// finals, with no serialization framing.
    pub fn byte_len(&self) -> usize {
        self.rounds.len() * 4 * 16 + 2 * 16
    }
}

/// Honest prover.
///
/// `a`, `b`: MLE tables of length `2^n`, `n = r.len()`.
/// `r`: claim point.
/// `v`: claimed value `Σ_x eq(r, x) · A(x) · B(x)`.
/// `channel`: Fiat-Shamir channel seeded and synchronized by caller.
///
/// Returns the proof and the challenge vector `r'` in the same order
/// as the sumcheck rounds (round 0 opens the highest-indexed
/// variable — matching `frost_gkr_core::mle::fold::fold_highest_var_inplace`).
pub fn prove_product<T: FiatShamir<Block128>>(
    a: &[Block128],
    b: &[Block128],
    r: &[Block128],
    v: Block128,
    channel: &mut T,
) -> (ProductProof, Vec<Block128>) {
    let n = r.len();
    assert_eq!(a.len(), 1 << n);
    assert_eq!(b.len(), 1 << n);

    debug_assert_eq!(
        compute_product_claim(a, b, r),
        v,
        "claim mismatches witness"
    );

    let mut eq_tbl = eq_ind_partial_eval(r);
    let mut a_tbl = a.to_vec();
    let mut b_tbl = b.to_vec();

    let mut rounds = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    let mut claim = v;
    let _ = claim;
    for _round in 0..n {
        let half = a_tbl.len() / 2;
        let evals = eval_round_at_0_1_2_3(&eq_tbl, &a_tbl, &b_tbl, half);
        let re = RoundEvals { evals };

        debug_assert_eq!(re.sum_at_0_plus_1(), claim);

        for e in &re.evals {
            channel.absorb(*e);
        }
        let r_i = channel.squeeze();

        claim = re.evaluate(r_i);
        fold_inplace(&mut eq_tbl, r_i);
        fold_inplace(&mut a_tbl, r_i);
        fold_inplace(&mut b_tbl, r_i);

        rounds.push(re);
        challenges.push(r_i);
    }

    debug_assert_eq!(a_tbl.len(), 1);
    debug_assert_eq!(b_tbl.len(), 1);
    debug_assert_eq!(eq_tbl.len(), 1);
    debug_assert_eq!(claim, eq_tbl[0] * a_tbl[0] * b_tbl[0]);

    challenges.reverse();

    let proof = ProductProof {
        rounds,
        a_final: a_tbl[0],
        b_final: b_tbl[0],
    };
    (proof, challenges)
}

/// Specialised version of [`prove_product`] for the case `A == B`. The
/// per-entry inner product `A(x) · B(x)` reduces to `A(x)^2`, and
/// `Block128::square()` is ~20× faster than general multiplication
/// (see `frost_gkr_core::tower::block128`). Three of the eight per-perm
/// sumchecks (`x4 = x2·x2`, two copies of `x2 = sin·sin`) are A=B in
/// every slot, so this fast path covers ~38% of perm-sumcheck arithmetic.
///
/// Wire format is identical: returns the same [`ProductProof`] /
/// `b_final == a_final` claim that the verifier already cross-checks via
/// `proof.x4_x2x2.a_final == proof.x4_x2x2.b_final` (see
/// `perm_sumcheck::verify_perm`). No protocol change.
pub fn prove_square<T: FiatShamir<Block128>>(
    a: &[Block128],
    r: &[Block128],
    v: Block128,
    channel: &mut T,
) -> (ProductProof, Vec<Block128>) {
    let n = r.len();
    assert_eq!(a.len(), 1 << n);

    debug_assert_eq!(
        compute_product_claim(a, a, r),
        v,
        "claim mismatches witness"
    );

    let mut eq_tbl = eq_ind_partial_eval(r);
    let mut a_tbl = a.to_vec();

    let mut rounds = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    let mut claim = v;
    let _ = claim;
    for _round in 0..n {
        let half = a_tbl.len() / 2;
        let evals = eval_round_at_0_1_2_3_square(&eq_tbl, &a_tbl, half);
        let re = RoundEvals { evals };

        debug_assert_eq!(re.sum_at_0_plus_1(), claim);

        for e in &re.evals {
            channel.absorb(*e);
        }
        let r_i = channel.squeeze();

        claim = re.evaluate(r_i);
        fold_inplace(&mut eq_tbl, r_i);
        fold_inplace(&mut a_tbl, r_i);

        rounds.push(re);
        challenges.push(r_i);
    }

    debug_assert_eq!(a_tbl.len(), 1);
    debug_assert_eq!(eq_tbl.len(), 1);
    debug_assert_eq!(claim, eq_tbl[0] * a_tbl[0] * a_tbl[0]);

    challenges.reverse();

    let proof = ProductProof {
        rounds,
        a_final: a_tbl[0],
        b_final: a_tbl[0],
    };
    (proof, challenges)
}

/// Verifier.
///
/// Returns `Some(r')` — the challenge vector — on success, `None` on
/// any failure. On success the caller may rely on:
///
/// - `proof.a_final == A(r')`
/// - `proof.b_final == B(r')`
///
/// where `r'` is the returned challenge vector, **provided** the
/// caller separately verifies the two final claims against
/// committed-to MLEs (or recursively against deeper sumchecks).
pub fn verify_product<T: FiatShamir<Block128>>(
    proof: &ProductProof,
    r: &[Block128],
    v: Block128,
    channel: &mut T,
) -> Option<Vec<Block128>> {
    let n = r.len();
    if proof.rounds.len() != n {
        return None;
    }

    let mut claim = v;
    let mut challenges = Vec::with_capacity(n);

    for re in &proof.rounds {
        if re.sum_at_0_plus_1() != claim {
            return None;
        }
        for e in &re.evals {
            channel.absorb(*e);
        }
        let r_i = channel.squeeze();
        claim = re.evaluate(r_i);
        challenges.push(r_i);
    }

    // Challenges were pushed in highest-var-first order; put them
    // into variable order (matching `r`) for the eq check and return.
    challenges.reverse();

    // Final identity: claim == eq(r, r') * a * b.
    let eq_rr = eq_ind(r, &challenges);
    let rhs = eq_rr * proof.a_final * proof.b_final;
    if claim != rhs {
        return None;
    }

    Some(challenges)
}

/// Compute `Σ_x eq(r, x) · A(x) · B(x)` — the honest claim from a
/// witness. Used by tests and by the prover's debug_assert.
pub fn compute_product_claim(a: &[Block128], b: &[Block128], r: &[Block128]) -> Block128 {
    let eq = eq_ind_partial_eval(r);
    debug_assert_eq!(eq.len(), a.len());
    debug_assert_eq!(eq.len(), b.len());
    let mut acc = Block128::ZERO;
    for i in 0..a.len() {
        acc += eq[i] * a[i] * b[i];
    }
    acc
}

/// Build the four eval points `(p(0), p(1), p(2), p(3))` for one
/// round of the product sumcheck. `half = current_len / 2`.
///
/// Per-entry: `t(k) = t_lo + k * (t_lo + t_hi)` (char 2).
fn eval_round_at_0_1_2_3(
    eq: &[Block128],
    a: &[Block128],
    b: &[Block128],
    half: usize,
) -> [Block128; 4] {
    let f2 = Block128::from(2u128);
    let f3 = Block128::from(3u128);

    let per_entry = |j: usize| -> [Block128; 4] {
        let eq_lo = eq[j];
        let eq_hi = eq[j + half];
        let a_lo = a[j];
        let a_hi = a[j + half];
        let b_lo = b[j];
        let b_hi = b[j + half];
        let d_eq = eq_lo + eq_hi;
        let d_a = a_lo + a_hi;
        let d_b = b_lo + b_hi;
        let e0 = eq_lo * a_lo * b_lo;
        let e1 = eq_hi * a_hi * b_hi;
        let eq_2 = eq_lo + f2 * d_eq;
        let a_2 = a_lo + f2 * d_a;
        let b_2 = b_lo + f2 * d_b;
        let eq_3 = eq_lo + f3 * d_eq;
        let a_3 = a_lo + f3 * d_a;
        let b_3 = b_lo + f3 * d_b;
        [e0, e1, eq_2 * a_2 * b_2, eq_3 * a_3 * b_3]
    };

    if half >= PAR_THRESHOLD {
        (0..half).into_par_iter().map(per_entry).reduce(
            || [Block128::ZERO; 4],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
        )
    } else {
        let mut acc = [Block128::ZERO; 4];
        for j in 0..half {
            let p = per_entry(j);
            acc[0] += p[0];
            acc[1] += p[1];
            acc[2] += p[2];
            acc[3] += p[3];
        }
        acc
    }
}

/// Round-poly evaluator for the `A == B` (square) special case. The
/// per-entry inner term `A(X)·A(X)` is one `Block128::square()` per
/// X value, ~20× faster than the general two-operand multiplication.
fn eval_round_at_0_1_2_3_square(eq: &[Block128], a: &[Block128], half: usize) -> [Block128; 4] {
    let f2 = Block128::from(2u128);
    let f3 = Block128::from(3u128);

    let per_entry = |j: usize| -> [Block128; 4] {
        let eq_lo = eq[j];
        let eq_hi = eq[j + half];
        let a_lo = a[j];
        let a_hi = a[j + half];
        let d_eq = eq_lo + eq_hi;
        let d_a = a_lo + a_hi;
        let e0 = eq_lo * a_lo.square();
        let e1 = eq_hi * a_hi.square();
        let eq_2 = eq_lo + f2 * d_eq;
        let a_2 = a_lo + f2 * d_a;
        let eq_3 = eq_lo + f3 * d_eq;
        let a_3 = a_lo + f3 * d_a;
        [e0, e1, eq_2 * a_2.square(), eq_3 * a_3.square()]
    };

    if half >= PAR_THRESHOLD {
        (0..half).into_par_iter().map(per_entry).reduce(
            || [Block128::ZERO; 4],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
        )
    } else {
        let mut acc = [Block128::ZERO; 4];
        for j in 0..half {
            let p = per_entry(j);
            acc[0] += p[0];
            acc[1] += p[1];
            acc[2] += p[2];
            acc[3] += p[3];
        }
        acc
    }
}

/// Fold the highest-indexed variable in-place by challenge `r`.
/// Matches `fold_highest_var_inplace` in `frost_gkr_core::mle::fold` but
/// kept local to avoid pulling the whole fold module surface in.
fn fold_inplace(v: &mut Vec<Block128>, r: Block128) {
    let half = v.len() / 2;
    for j in 0..half {
        let lo = v[j];
        let hi = v[j + half];
        v[j] = lo + r * (lo + hi);
    }
    v.truncate(half);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_roundtrip_through_0_1_2_3() {
        // Fix a known deg-3 polynomial p(X) = 7 + 3X + 2X^2 + X^3 and
        // verify Lagrange reconstruction at several points.
        let c0 = Block128::from(7u128);
        let c1 = Block128::from(3u128);
        let c2 = Block128::from(2u128);
        let c3 = Block128::from(1u128);
        let eval = |x: Block128| -> Block128 { c0 + c1 * x + c2 * x * x + c3 * x * x * x };

        let evals = [
            eval(Block128::from(0u128)),
            eval(Block128::from(1u128)),
            eval(Block128::from(2u128)),
            eval(Block128::from(3u128)),
        ];

        for r_i in [5u128, 17, 0x1234, u128::MAX] {
            let r = Block128::from(r_i);
            let got = lagrange_at_0_1_2_3(&evals, r);
            let want = eval(r);
            assert_eq!(got, want, "lagrange mismatch at r={r_i}");
        }
    }
}
