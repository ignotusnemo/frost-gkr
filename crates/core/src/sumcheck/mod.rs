// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Minimal round-polynomial representation shared by the FROST-GKR prover
//! and verifier.

use crate::TowerField;

/// Univariate polynomial in coefficient form: `coeffs[i]` is the
/// coefficient of `X^i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundPolynomial<F> {
    pub coeffs: Vec<F>,
}

impl<F: TowerField> RoundPolynomial<F> {
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Horner evaluation: `c0 + x*(c1 + x*(c2 + ...))`.
    pub fn evaluate(&self, x: F) -> F {
        let mut acc = F::ZERO;
        for &coefficient in self.coeffs.iter().rev() {
            acc = acc * x + coefficient;
        }
        acc
    }

    /// Interpolate from `(0, e0), (1, e1), …, (d, e_d)`.
    pub fn from_evals(evals: &[F]) -> Self {
        let n = evals.len();
        assert!(n > 0, "need at least one evaluation");
        assert!(n <= 256, "Lagrange X-axis indexes u8");

        let mut coefficients = vec![F::ZERO; n];
        let mut scratch = vec![F::ZERO; n];
        for (k, &evaluation) in evals.iter().enumerate() {
            let x_k = F::from(k as u8);
            let mut numerator = vec![F::ZERO; n];
            numerator[0] = F::ONE;
            let mut degree = 0usize;
            let mut denominator = F::ONE;

            for j in 0..n {
                if j == k {
                    continue;
                }
                let x_j = F::from(j as u8);
                for value in scratch.iter_mut().take(degree + 2) {
                    *value = F::ZERO;
                }
                for i in 0..=degree {
                    scratch[i] += numerator[i] * x_j;
                    scratch[i + 1] += numerator[i];
                }
                numerator.copy_from_slice(&scratch);
                degree += 1;
                denominator *= x_k + x_j;
            }

            let scale = evaluation * denominator.invert();
            for i in 0..n {
                coefficients[i] += numerator[i] * scale;
            }
        }

        Self {
            coeffs: coefficients,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Block128;

    #[test]
    fn interpolation_round_trip_degree_nine() {
        let evaluations: Vec<_> = (0u8..=9).map(|x| Block128::from(x as u128 + 7)).collect();
        let polynomial = RoundPolynomial::from_evals(&evaluations);
        assert_eq!(polynomial.degree(), 9);
        for (x, expected) in evaluations.into_iter().enumerate() {
            assert_eq!(polynomial.evaluate(Block128::from(x as u128)), expected);
        }
    }
}
