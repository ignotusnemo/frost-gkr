// SPDX-License-Identifier: Apache-2.0
// Portions adapted from binius64.
// Copyright (c) 2025 The Binius Developers.
// Copyright (c) 2025 Irreducible, Inc.
// Modifications Copyright (C) 2026 Ignotus Nemo.

//! Multilinear polynomial evaluation over the boolean hypercube.

use super::fold::fold_highest_var_inplace;
use crate::packed::PACKED_LANES;
use crate::{Block128, TowerField};

/// Evaluate a multilinear polynomial at a point using in-place folding.
pub fn evaluate_inplace_scalars<F: TowerField>(mut evals: Vec<F>, point: &[F]) -> F {
    let n = point.len();
    assert_eq!(
        evals.len(),
        1 << n,
        "eval length {} must equal 2^{}",
        evals.len(),
        n
    );

    for &coord in point.iter().rev() {
        fold_highest_var_inplace(&mut evals, coord);
    }

    assert_eq!(evals.len(), 1);
    evals[0]
}

/// Convenience function: evaluate without consuming the input.
pub fn evaluate_slice<F: TowerField>(evals: &[F], point: &[F]) -> F {
    evaluate_inplace_scalars(evals.to_vec(), point)
}

/// Evaluate an MLE using packed fold operations.
pub fn evaluate_packed(poly: &[Block128], point: &[Block128]) -> Block128 {
    use crate::packed::{pack_slice, PackedBlock128};

    // Too small for packed path to be worthwhile — fall back to scalar.
    if poly.len() < PACKED_LANES * 2 || !poly.len().is_multiple_of(PACKED_LANES) {
        return evaluate_slice(poly, point);
    }

    let mut evals: Vec<PackedBlock128> = pack_slice(poly).to_vec();
    let mut point_iter = point.iter().rev();

    while evals.len() > 1 {
        let Some(&r) = point_iter.next() else {
            break;
        };
        let half = evals.len() / 2;
        for i in 0..half {
            let lo = evals[i];
            let hi = evals[i + half];
            let diff = hi.xor(lo);
            let scaled = diff.scalar_mul(r);
            evals[i] = lo.xor(scaled);
        }
        evals.truncate(half);
    }

    // Unpack final element and fold any remaining variables scalar-style.
    let mut scalars: Vec<Block128> = evals.into_iter().flat_map(|p| p.to_array()).collect();
    for &r in point_iter {
        fold_highest_var_inplace(&mut scalars, r);
    }

    scalars[0]
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Block128;
    use rand::Rng;

    type F = Block128;

    #[test]
    fn test_evaluate_at_hypercube_vertex() {
        let n = 4;
        let mut rng = rand::thread_rng();
        let evals: Vec<F> = (0..(1 << n)).map(|_| F::from(rng.gen::<u128>())).collect();

        for i in 0..(1 << n) {
            let point: Vec<F> = (0..n)
                .map(|b| if (i >> b) & 1 == 1 { F::ONE } else { F::ZERO })
                .collect();
            let result = evaluate_slice(&evals, &point);
            assert_eq!(result, evals[i], "mismatch at vertex {i}");
        }
    }

    #[test]
    fn test_evaluate_linear() {
        let a = F::from(5u8);
        let b = F::from(2u8);
        let c = F::from(3u8);
        let d = F::from(7u8);

        let evals = vec![a, a + b, a + c, a + b + c + d];

        assert_eq!(evaluate_slice(&evals, &[F::ZERO, F::ZERO]), a);
        assert_eq!(evaluate_slice(&evals, &[F::ONE, F::ZERO]), a + b);
        assert_eq!(evaluate_slice(&evals, &[F::ZERO, F::ONE]), a + c);
        assert_eq!(evaluate_slice(&evals, &[F::ONE, F::ONE]), a + b + c + d);
    }
}
