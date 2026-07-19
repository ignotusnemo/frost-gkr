// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Batch-evaluation sumcheck.
//!
//! Reduces `M` MLE-evaluation claims `(r_i, v_i)` on a **single**
//! multilinear polynomial `B` of length `2^n` to one point-value pair
//! `(r_B, v_B)` via a standard RLC + degree-2 sumcheck:
//!
//! ```text
//!   V := Σ_i α_i · v_i  with  α_i ← channel.squeeze()
//!   W(x) := Σ_i α_i · eq(r_i, x)
//!   H(x) := W(x) · B(x)
//!
//!   V == Σ_x H(x)   (by linearity of eq)
//!
//!   sumcheck H over n variables → (r_B, claim_B)
//!   verifier checks  claim_B == W(r_B) · v_B_claimed
//! ```
//!
//! Soundness: W(r_B) is recomputed by the verifier as
//! `Σ_i α_i · eq(r_i, r_B)`. `v_B = B(r_B)` is the reduced claim that
//! the caller discharges against the original MLE or an external opening
//! argument.
//!
//! Why a fresh primitive rather than reusing `product_sumcheck`:
//! `product_sumcheck` folds in an extra `eq(r, x)` factor (degree 3
//! per variable). Here the outer eq has been absorbed into `W`, so
//! `H = W · B` is degree 2 per variable, one less round polynomial
//! coefficient per round.

use std::sync::OnceLock;

use frost_gkr_core::mle::eq::eq_ind;
use frost_gkr_core::mle::fold::fold_highest_var_inplace;
use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::{Block128, TowerField};
use rayon::prelude::*;

/// Inverse Lagrange denominators at `{0,1,2}`. Cached so per-round
/// `evaluate` is invert-free in the hot path.
fn denom_inv_3() -> &'static [Block128; 3] {
    static CACHE: OnceLock<[Block128; 3]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut out = [Block128::ZERO; 3];
        for k in 0..3 {
            let xk = Block128::from(k as u128);
            let mut d = Block128::ONE;
            for j in 0..3 {
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

const PAR_THRESHOLD: usize = 64;

/// One round of the degree-2 batch-eval sumcheck, stored as its
/// evaluations at `X = 0, 1, 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchEvalRound {
    pub evals: [Block128; 3],
}

impl BatchEvalRound {
    #[inline]
    pub fn sum_at_0_plus_1(&self) -> Block128 {
        self.evals[0] + self.evals[1]
    }

    /// Lagrange-interpolate at `r` from evaluations at `{0,1,2}`.
    pub fn evaluate(&self, r: Block128) -> Block128 {
        lagrange_at_0_1_2(&self.evals, r)
    }
}

/// Lagrange evaluation at a single point from evals at `{0,1,2}`. Uses
/// the cached denominator inverses in `denom_inv_3()` so the hot path
/// is invert-free.
#[inline]
pub fn lagrange_at_0_1_2(evals: &[Block128; 3], r: Block128) -> Block128 {
    let denom_inv = denom_inv_3();
    let r0 = r + Block128::from(0u128);
    let r1 = r + Block128::from(1u128);
    let r2 = r + Block128::from(2u128);
    let n0 = r1 * r2;
    let n1 = r0 * r2;
    let n2 = r0 * r1;
    evals[0] * n0 * denom_inv[0] + evals[1] * n1 * denom_inv[1] + evals[2] * n2 * denom_inv[2]
}

/// One `(r, v)` MLE-evaluation claim on the shared target MLE `B`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalClaim {
    pub point: Vec<Block128>,
    pub value: Block128,
}

/// Proof object: one round poly per sumcheck variable plus the final
/// reduced `(r_B, v_B)` is derived by the verifier from the transcript
/// and the last round's final claim, so only the round polys and the
/// prover's `b_final` need to ship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEvalProof {
    pub rounds: Vec<BatchEvalRound>,
    /// `b_final = B(r_B)`. The verifier cross-checks it against
    /// `final_claim / W(r_B)`; the caller discharges the terminal claim.
    pub b_final: Block128,
}

impl BatchEvalProof {
    /// Raw field-element byte size: `rounds.len() * 3` degree-2 round
    /// evals plus `b_final`, 16 bytes each.
    pub fn byte_len(&self) -> usize {
        self.rounds.len() * 3 * 16 + 16
    }
}

/// Output of a successful verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEvalReduction {
    /// `r_B` — the sumcheck's terminal point, in variable order.
    pub point: Vec<Block128>,
    /// `v_B = B(r_B)` as claimed by the prover and telescope-consistent.
    pub value: Block128,
}

fn fold_inplace(tbl: &mut Vec<Block128>, r: Block128) {
    fold_highest_var_inplace(tbl, r);
}

/// Build `W(x) = Σ_i α_i · eq(r_i, x)` as a length-`2^n` table. We
/// unfold each claim's eq tensor directly into `w` without
/// materialising the full `eq_ind_partial_eval` as a separate buffer
/// — the hot loop otherwise allocates `M · 2^n` temporaries.
fn build_w_table(claims: &[EvalClaim], alphas: &[Block128], n: usize) -> Vec<Block128> {
    debug_assert_eq!(claims.len(), alphas.len());
    let len = 1usize << n;

    // Build each claim's `α_i · eq(r_i, ·)` tensor independently in
    // parallel, then reduce-sum. The independent-claim work is what
    // dominates setup (`M · 2^n` muls); the reduction is `M-1` linear
    // passes over `2^n` cells which parallelises trivially.
    claims
        .par_iter()
        .zip(alphas.par_iter())
        .map(|(claim, &alpha)| {
            debug_assert_eq!(claim.point.len(), n);
            let mut eq: Vec<Block128> = Vec::with_capacity(len);
            eq.push(alpha);
            for &r_i in &claim.point {
                let cur = eq.len();
                for j in 0..cur {
                    let prod = eq[j] * r_i;
                    eq[j] -= prod;
                    eq.push(prod);
                }
            }
            debug_assert_eq!(eq.len(), len);
            eq
        })
        .reduce(
            || vec![Block128::ZERO; len],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += *bi;
                }
                a
            },
        )
}

/// Evaluate `W(r) = Σ_i α_i · eq(r_i, r)` without materialising the table.
fn evaluate_w_at(claims: &[EvalClaim], alphas: &[Block128], r: &[Block128]) -> Block128 {
    debug_assert_eq!(claims.len(), alphas.len());
    let mut acc = Block128::ZERO;
    for (claim, &alpha) in claims.iter().zip(alphas.iter()) {
        debug_assert_eq!(claim.point.len(), r.len());
        acc += alpha * eq_ind(&claim.point, r);
    }
    acc
}

/// Squeeze one RLC challenge per claim.
fn squeeze_alphas<T: FiatShamir<Block128>>(channel: &mut T, m: usize) -> Vec<Block128> {
    (0..m).map(|_| channel.squeeze()).collect()
}

/// Absorb all claim points and values into the channel so `α_i` are
/// bound to the exact set of claims being batched.
fn absorb_claims<T: FiatShamir<Block128>>(channel: &mut T, claims: &[EvalClaim]) {
    for c in claims {
        for e in &c.point {
            channel.absorb(*e);
        }
        channel.absorb(c.value);
    }
}

/// Honest prover.
///
/// `b`: length-`2^n` table of the target MLE. Must be the same table
/// whose claims are being discharged.
/// `claims`: the M `(r_i, v_i)` claims.
/// `channel`: shared Fiat-Shamir channel.
///
/// Returns the proof and the `(r_B, v_B)` reduction so callers can
/// discharge it against the original MLE or pass it to an opening layer.
pub fn prove_batch_eval<T: FiatShamir<Block128>>(
    b: &[Block128],
    claims: &[EvalClaim],
    channel: &mut T,
) -> (BatchEvalProof, BatchEvalReduction) {
    let n = b.len().trailing_zeros() as usize;
    assert_eq!(b.len(), 1 << n);
    assert!(!claims.is_empty());
    for c in claims {
        assert_eq!(c.point.len(), n);
    }

    absorb_claims(channel, claims);
    let alphas = squeeze_alphas(channel, claims.len());

    let mut w_tbl = build_w_table(claims, &alphas, n);
    let mut b_tbl = b.to_vec();

    // Initial claim: V = Σ α_i · v_i.
    let mut claim = Block128::ZERO;
    for (c, &a) in claims.iter().zip(alphas.iter()) {
        claim += a * c.value;
    }

    let mut rounds = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _round in 0..n {
        let half = w_tbl.len() / 2;
        let evals = eval_round_at_0_1_2(&w_tbl, &b_tbl, half);
        let re = BatchEvalRound { evals };

        debug_assert_eq!(re.sum_at_0_plus_1(), claim);

        for e in &re.evals {
            channel.absorb(*e);
        }
        let r_i = channel.squeeze();

        claim = re.evaluate(r_i);
        fold_inplace(&mut w_tbl, r_i);
        fold_inplace(&mut b_tbl, r_i);

        rounds.push(re);
        challenges.push(r_i);
    }

    debug_assert_eq!(w_tbl.len(), 1);
    debug_assert_eq!(b_tbl.len(), 1);
    debug_assert_eq!(claim, w_tbl[0] * b_tbl[0]);

    // Variable order matches `fold_highest_var_inplace` convention used
    // elsewhere: challenges pushed highest-var-first, so reverse to get
    // variable-index order.
    challenges.reverse();

    let proof = BatchEvalProof {
        rounds,
        b_final: b_tbl[0],
    };
    let reduction = BatchEvalReduction {
        point: challenges.clone(),
        value: b_tbl[0],
    };
    (proof, reduction)
}

/// Honest verifier. Returns `Some(reduction)` on accept, `None` on
/// reject. The caller is responsible for discharging
/// `reduction.value == B(reduction.point)` against the original MLE or its
/// opening argument.
pub fn verify_batch_eval<T: FiatShamir<Block128>>(
    proof: &BatchEvalProof,
    claims: &[EvalClaim],
    n: usize,
    channel: &mut T,
) -> Option<BatchEvalReduction> {
    if claims.is_empty() {
        return None;
    }
    for c in claims {
        if c.point.len() != n {
            return None;
        }
    }
    if proof.rounds.len() != n {
        return None;
    }

    absorb_claims(channel, claims);
    let alphas = squeeze_alphas(channel, claims.len());

    let mut claim = Block128::ZERO;
    for (c, &a) in claims.iter().zip(alphas.iter()) {
        claim += a * c.value;
    }

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
    challenges.reverse();

    // Final check: claim == W(r_B) · b_final.
    let w_at = evaluate_w_at(claims, &alphas, &challenges);
    if claim != w_at * proof.b_final {
        return None;
    }

    Some(BatchEvalReduction {
        point: challenges,
        value: proof.b_final,
    })
}

/// Round-poly evaluator: `p(X) = Σ_j W(X,j) · B(X,j)` where the per-index
/// linear extensions are `t(X) = t_lo + X · (t_lo + t_hi)` (char-2).
/// Parallel over the `half` entries above `PAR_THRESHOLD` (large enough
/// to amortise rayon join overhead but covers the heavy early rounds at
/// 2^14 / 2^13 entries).
fn eval_round_at_0_1_2(w: &[Block128], b: &[Block128], half: usize) -> [Block128; 3] {
    let f2 = Block128::from(2u128);
    let per_entry = |j: usize| -> [Block128; 3] {
        let w_lo = w[j];
        let w_hi = w[j + half];
        let b_lo = b[j];
        let b_hi = b[j + half];
        let d_w = w_lo + w_hi;
        let d_b = b_lo + b_hi;
        let w2 = w_lo + f2 * d_w;
        let b2 = b_lo + f2 * d_b;
        [w_lo * b_lo, w_hi * b_hi, w2 * b2]
    };
    if half >= PAR_THRESHOLD {
        (0..half).into_par_iter().map(per_entry).reduce(
            || [Block128::ZERO; 3],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
        )
    } else {
        let mut acc = [Block128::ZERO; 3];
        for j in 0..half {
            let p = per_entry(j);
            acc[0] += p[0];
            acc[1] += p[1];
            acc[2] += p[2];
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frost_gkr_core::mle::evaluate::evaluate_slice;
    use frost_gkr_poseidon2b::channel::Poseidon2bChannel;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<Block128> {
        (0..n).map(|_| Block128::from(rng.gen::<u128>())).collect()
    }

    fn fresh_channel(seed: u64) -> Poseidon2bChannel {
        let mut ch = Poseidon2bChannel::new();
        ch.absorb(Block128::from(seed as u128));
        ch
    }

    #[test]
    fn honest_roundtrip_single_claim() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 5;
        let b = rand_vec(&mut rng, 1 << n);
        let r0 = rand_vec(&mut rng, n);
        let v0 = evaluate_slice(&b, &r0);
        let claims = vec![EvalClaim {
            point: r0,
            value: v0,
        }];

        let mut cp = fresh_channel(7);
        let (proof, red_p) = prove_batch_eval(&b, &claims, &mut cp);

        let mut cv = fresh_channel(7);
        let red_v = verify_batch_eval(&proof, &claims, n, &mut cv).unwrap();
        assert_eq!(red_p, red_v);
        assert_eq!(red_v.value, evaluate_slice(&b, &red_v.point));
    }

    #[test]
    fn honest_roundtrip_many_claims() {
        let mut rng = StdRng::seed_from_u64(2);
        let n = 6;
        let b = rand_vec(&mut rng, 1 << n);
        let m = 17;
        let claims: Vec<EvalClaim> = (0..m)
            .map(|_| {
                let r = rand_vec(&mut rng, n);
                let v = evaluate_slice(&b, &r);
                EvalClaim { point: r, value: v }
            })
            .collect();

        let mut cp = fresh_channel(13);
        let (proof, red_p) = prove_batch_eval(&b, &claims, &mut cp);

        let mut cv = fresh_channel(13);
        let red_v = verify_batch_eval(&proof, &claims, n, &mut cv).unwrap();
        assert_eq!(red_p, red_v);
        assert_eq!(red_v.value, evaluate_slice(&b, &red_v.point));
    }

    #[test]
    fn forged_verifier_claim_rejected() {
        // Prover runs honestly. Verifier is handed a claim vector with
        // one tampered `value`; its initial running claim forks, so
        // round 0's telescope `e0+e1 == claim` fails immediately.
        let mut rng = StdRng::seed_from_u64(3);
        let n = 5;
        let b = rand_vec(&mut rng, 1 << n);
        let r0 = rand_vec(&mut rng, n);
        let good = evaluate_slice(&b, &r0);
        let honest_claims = vec![EvalClaim {
            point: r0.clone(),
            value: good,
        }];

        let mut cp = fresh_channel(9);
        let (proof, _) = prove_batch_eval(&b, &honest_claims, &mut cp);

        let bad_claims = vec![EvalClaim {
            point: r0,
            value: good + Block128::from(1u128),
        }];
        let mut cv = fresh_channel(9);
        assert!(verify_batch_eval(&proof, &bad_claims, n, &mut cv).is_none());
    }

    #[test]
    fn tampered_b_final_rejected() {
        let mut rng = StdRng::seed_from_u64(4);
        let n = 5;
        let b = rand_vec(&mut rng, 1 << n);
        let r0 = rand_vec(&mut rng, n);
        let v0 = evaluate_slice(&b, &r0);
        let claims = vec![EvalClaim {
            point: r0,
            value: v0,
        }];

        let mut cp = fresh_channel(17);
        let (mut proof, _) = prove_batch_eval(&b, &claims, &mut cp);
        proof.b_final += Block128::from(1u128);

        let mut cv = fresh_channel(17);
        assert!(verify_batch_eval(&proof, &claims, n, &mut cv).is_none());
    }

    #[test]
    fn determinism() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 5;
        let b = rand_vec(&mut rng, 1 << n);
        let r0 = rand_vec(&mut rng, n);
        let v0 = evaluate_slice(&b, &r0);
        let claims = vec![EvalClaim {
            point: r0,
            value: v0,
        }];

        let mut c1 = fresh_channel(21);
        let (p1, _) = prove_batch_eval(&b, &claims, &mut c1);
        let mut c2 = fresh_channel(21);
        let (p2, _) = prove_batch_eval(&b, &claims, &mut c2);
        assert_eq!(p1, p2);
    }
}
