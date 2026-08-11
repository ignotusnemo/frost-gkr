// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! One GKR walk over unequal-width Poseidon2b regions.
//!
//! Region `a` has Boolean width `w_a`; the aggregate sumcheck walks
//! `W = max_a w_a` coordinates.  Once a region exhausts its native
//! coordinates, its relation is multiplied by
//! `chi_a(x) = product_{j=w_a}^{W-1} (1 + x_j)`.  The selector preserves the
//! region's Boolean sum, adds only one individual degree per new coordinate,
//! and lets the prover keep the region at its native physical size.
//!
//! Committed rows remain in GF(2^128).  Claims, sumcheck messages and
//! challenges live in the quadratic extension GF(2^256).

use frost_gkr_core::hardware::{
    clmul_gcm, clmul_gcm_pair, flat_to_tower_u128, square_flat_u128, tower_to_flat_u128,
};
use frost_gkr_core::transcript::FiatShamir;
use frost_gkr_core::{Block128, Block256, TowerField};
use frost_gkr_poseidon2b::native::permutation::{
    MDS_FULL, MDS_PARTIAL, N_ROUNDS, ROUND_CONSTANTS, STATE_SIZE,
};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::sync::OnceLock;

use crate::layers::{round_kind, RoundKind};

pub const WALK_DEGREE: usize = 8;
const CHECKPOINT_SPACING: usize = 8;
const RAGGED_DOMAIN: Block128 = Block128(0x5241_4747_4544_5f47_4b52_5f56_3100_0001);

/// Quadratic extension element whose two base coordinates remain in the
/// CLMUL-friendly flat basis throughout the prover hot path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FastWide {
    lo: u128,
    hi: u128,
}

impl FastWide {
    const ZERO: Self = Self { lo: 0, hi: 0 };
    const ONE: Self = Self { lo: 1, hi: 0 };

    #[inline(always)]
    fn from_public(value: Block256) -> Self {
        Self {
            lo: tower_to_flat_u128(value.lo.0),
            hi: tower_to_flat_u128(value.hi.0),
        }
    }

    #[inline(always)]
    fn to_public(self) -> Block256 {
        Block256::new(
            Block128(flat_to_tower_u128(self.lo)),
            Block128(flat_to_tower_u128(self.hi)),
        )
    }

    #[inline(always)]
    fn from_base_flat(value: u128) -> Self {
        Self { lo: value, hi: 0 }
    }

    #[inline(always)]
    fn square(self) -> Self {
        let lo = square_flat_u128(self.lo);
        let hi = square_flat_u128(self.hi);
        Self {
            lo: lo ^ clmul_gcm(hi, flat_tau()),
            hi,
        }
    }

    #[inline(always)]
    fn scale_base(self, scalar: u128) -> Self {
        let [lo, hi] = clmul_gcm_pair(self.lo, scalar, self.hi, scalar);
        Self { lo, hi }
    }
}

impl Add for FastWide {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl AddAssign for FastWide {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for FastWide {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let [v0, v1] = clmul_gcm_pair(self.lo, rhs.lo, self.hi, rhs.hi);
        let [v1_tau, sum] = clmul_gcm_pair(v1, flat_tau(), self.lo ^ self.hi, rhs.lo ^ rhs.hi);
        Self {
            lo: v0 ^ v1_tau,
            hi: v0 ^ sum,
        }
    }
}

impl MulAssign for FastWide {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[inline(always)]
fn flat_tau() -> u128 {
    static TAU: OnceLock<u128> = OnceLock::new();
    *TAU.get_or_init(|| tower_to_flat_u128(Block128::TAU.0))
}

struct FlatSchedule {
    constants: [[u128; N_ROUNDS]; STATE_SIZE],
    full: [[u128; STATE_SIZE]; STATE_SIZE],
    partial: [[u128; STATE_SIZE]; STATE_SIZE],
}

fn flat_schedule() -> &'static FlatSchedule {
    static SCHEDULE: OnceLock<FlatSchedule> = OnceLock::new();
    SCHEDULE.get_or_init(|| FlatSchedule {
        constants: std::array::from_fn(|lane| {
            std::array::from_fn(|round| tower_to_flat_u128(ROUND_CONSTANTS[lane][round]))
        }),
        full: std::array::from_fn(|row| {
            std::array::from_fn(|column| tower_to_flat_u128(MDS_FULL[row][column]))
        }),
        partial: std::array::from_fn(|row| {
            std::array::from_fn(|column| tower_to_flat_u128(MDS_PARTIAL[row][column]))
        }),
    })
}

/// Fiat-Shamir adapter that frames GF(2^256) messages as two GF(2^128)
/// coordinates in the repository's Poseidon2b channel.
pub struct WideChannel<C> {
    inner: C,
}

impl<C> WideChannel<C>
where
    C: FiatShamir<Block128>,
{
    pub fn new(inner: C) -> Self {
        Self { inner }
    }

    #[inline]
    pub fn absorb(&mut self, value: Block256) {
        self.inner.absorb(value.lo);
        self.inner.absorb(value.hi);
    }

    pub fn absorb_slice(&mut self, values: &[Block256]) {
        for &value in values {
            self.absorb(value);
        }
    }

    #[inline]
    pub fn squeeze(&mut self) -> Block256 {
        Block256::from_challenge_lanes(self.inner.squeeze(), self.inner.squeeze())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneClaim {
    pub point: Vec<Block256>,
    pub values: [Block256; STATE_SIZE],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaggedLayerProof {
    /// Compressed degree-eight coefficients `[c_0, c_2, ..., c_8]`.
    pub round_coeffs: Vec<[Block256; WALK_DEGREE]>,
    /// Four next-layer evaluations in canonical region order.
    pub next_values: Vec<[Block256; STATE_SIZE]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaggedProof {
    /// Layer 66 first, down to layer 1.
    pub layers: Vec<RaggedLayerProof>,
}

impl RaggedProof {
    /// Raw algebraic transcript size. Shape metadata and protocol framing are
    /// public statement data and therefore are not counted as prover bytes.
    pub fn byte_len(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                32 * (WALK_DEGREE * layer.round_coeffs.len() + STATE_SIZE * layer.next_values.len())
            })
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaggedError {
    Shape,
    LayerMismatch(usize),
}

impl core::fmt::Display for RaggedError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape => write!(formatter, "ragged walk proof shape mismatch"),
            Self::LayerMismatch(layer) => {
                write!(formatter, "ragged walk layer {layer} claim mismatch")
            }
        }
    }
}

fn absorb_statement<C: FiatShamir<Block128>>(
    channel: &mut WideChannel<C>,
    widths: &[usize],
    groups: &[LaneClaim],
) {
    channel.inner.absorb(RAGGED_DOMAIN);
    channel.inner.absorb(Block128(widths.len() as u128));
    for (&width, group) in widths.iter().zip(groups) {
        channel.inner.absorb(Block128(width as u128));
        channel.inner.absorb(Block128(1));
        channel.absorb_slice(&group.point);
        channel.absorb_slice(&group.values);
    }
}

#[inline]
fn sbox7_wide(value: Block256) -> Block256 {
    let square = value.square();
    let fourth = square.square();
    value * square * fourth
}

#[inline(always)]
fn sbox7_flat(value: u128) -> u128 {
    let square = square_flat_u128(value);
    let fourth = square_flat_u128(square);
    clmul_gcm(clmul_gcm(value, square), fourth)
}

#[inline(always)]
fn sbox7_flat_pair(left: u128, right: u128) -> [u128; 2] {
    let left_square = square_flat_u128(left);
    let right_square = square_flat_u128(right);
    let [left_cube, right_cube] = clmul_gcm_pair(left_square, left, right_square, right);
    let left_fourth = square_flat_u128(left_square);
    let right_fourth = square_flat_u128(right_square);
    clmul_gcm_pair(left_fourth, left_cube, right_fourth, right_cube)
}

#[inline(always)]
#[cfg(test)]
fn apply_mds_flat(
    matrix: &[[u128; STATE_SIZE]; STATE_SIZE],
    input: [u128; STATE_SIZE],
) -> [u128; STATE_SIZE] {
    std::array::from_fn(|row| {
        (0..STATE_SIZE).fold(0, |sum, column| {
            sum ^ if matrix[row][column] == 1 {
                input[column]
            } else {
                clmul_gcm(matrix[row][column], input[column])
            }
        })
    })
}

#[inline(always)]
fn apply_full_mds_flat(input: [u128; STATE_SIZE]) -> [u128; STATE_SIZE] {
    let matrix = &flat_schedule().full;
    let [p00, p10] = clmul_gcm_pair(matrix[0][0], input[0], matrix[1][0], input[0]);
    let [p01, p11] = clmul_gcm_pair(matrix[0][1], input[1], matrix[1][1], input[1]);
    let [p03, p21] = clmul_gcm_pair(matrix[0][3], input[3], matrix[2][1], input[1]);
    let [p22, p32] = clmul_gcm_pair(matrix[2][2], input[2], matrix[3][2], input[2]);
    let [p23, p33] = clmul_gcm_pair(matrix[2][3], input[3], matrix[3][3], input[3]);
    [
        p00 ^ p01 ^ input[2] ^ p03,
        p10 ^ p11 ^ input[2] ^ input[3],
        input[0] ^ p21 ^ p22 ^ p23,
        input[0] ^ input[1] ^ p32 ^ p33,
    ]
}

#[inline(always)]
fn apply_partial_mds_flat(input: [u128; STATE_SIZE]) -> [u128; STATE_SIZE] {
    let matrix = &flat_schedule().partial;
    let sum = input[0] ^ input[1] ^ input[2] ^ input[3];
    let [p0, p1] = clmul_gcm_pair(matrix[0][0], input[0], matrix[1][1], input[1]);
    let [p2, p3] = clmul_gcm_pair(matrix[2][2], input[2], matrix[3][3], input[3]);
    [
        p0 ^ sum ^ input[0],
        p1 ^ sum ^ input[1],
        p2 ^ sum ^ input[2],
        p3 ^ sum ^ input[3],
    ]
}

#[inline(always)]
fn apply_round_flat(round: usize, mut state: [u128; STATE_SIZE]) -> [u128; STATE_SIZE] {
    let schedule = flat_schedule();
    match round_kind(round) {
        RoundKind::Full => {
            let [s0, s1] = sbox7_flat_pair(
                state[0] ^ schedule.constants[0][round],
                state[1] ^ schedule.constants[1][round],
            );
            let [s2, s3] = sbox7_flat_pair(
                state[2] ^ schedule.constants[2][round],
                state[3] ^ schedule.constants[3][round],
            );
            apply_full_mds_flat([s0, s1, s2, s3])
        }
        RoundKind::Partial => {
            state[0] = sbox7_flat(state[0] ^ schedule.constants[0][round]);
            apply_partial_mds_flat(state)
        }
    }
}

/// Apply the permutation's initial linear layer to raw input columns.  The
/// returned columns are layer S_0 consumed by the 66-round walk.
pub fn initial_layer_columns(raw: &[Vec<Block128>; STATE_SIZE]) -> [Vec<Block128>; STATE_SIZE] {
    assert!(!raw[0].is_empty() && raw[0].len().is_power_of_two());
    assert!(raw.iter().all(|column| column.len() == raw[0].len()));
    let rows = (0..raw[0].len())
        .into_par_iter()
        .map(|index| {
            let state = std::array::from_fn(|lane| tower_to_flat_u128(raw[lane][index].0));
            apply_full_mds_flat(state).map(|value| Block128(flat_to_tower_u128(value)))
        })
        .collect::<Vec<_>>();
    std::array::from_fn(|lane| rows.iter().map(|row| row[lane]).collect())
}

/// Evaluate all 66 nonlinear/linear rounds from layer S_0.
pub fn output_layer_columns(
    layer_zero: &[Vec<Block128>; STATE_SIZE],
) -> [Vec<Block128>; STATE_SIZE] {
    assert!(!layer_zero[0].is_empty() && layer_zero[0].len().is_power_of_two());
    assert!(layer_zero
        .iter()
        .all(|column| column.len() == layer_zero[0].len()));
    let rows = (0..layer_zero[0].len())
        .into_par_iter()
        .map(|index| {
            let mut state =
                std::array::from_fn(|lane| tower_to_flat_u128(layer_zero[lane][index].0));
            for round in 0..N_ROUNDS {
                state = apply_round_flat(round, state);
            }
            state.map(|value| Block128(flat_to_tower_u128(value)))
        })
        .collect::<Vec<_>>();
    std::array::from_fn(|lane| rows.iter().map(|row| row[lane]).collect())
}

fn columns_to_rows(columns: &[Vec<Block128>; STATE_SIZE]) -> Vec<[u128; STATE_SIZE]> {
    (0..columns[0].len())
        .into_par_iter()
        .map(|index| std::array::from_fn(|lane| tower_to_flat_u128(columns[lane][index].0)))
        .collect()
}

fn apply_round_span(first: usize, count: usize, rows: &mut [[u128; STATE_SIZE]]) {
    rows.par_iter_mut().for_each(|row| {
        let mut state = *row;
        for round in first..first + count {
            state = apply_round_flat(round, state);
        }
        *row = state;
    });
}

fn replay_window(
    checkpoint: Vec<[u128; STATE_SIZE]>,
    base: usize,
    length: usize,
) -> Vec<Vec<[u128; STATE_SIZE]>> {
    let mut window = Vec::with_capacity(length);
    window.push(checkpoint);
    for offset in 1..length {
        let mut next = window[offset - 1].clone();
        apply_round_span(base + offset - 1, 1, &mut next);
        window.push(next);
    }
    window
}

/// Supplies descending layer states while storing only checkpoints plus one
/// replay window. This is the same memory shape for all three benchmarked
/// constructions; only their physical row counts differ.
struct DescendingLayerStates {
    checkpoints: Vec<Vec<[u128; STATE_SIZE]>>,
    window: Vec<Vec<[u128; STATE_SIZE]>>,
    window_base: usize,
}

impl DescendingLayerStates {
    fn new(layer_zero: &[Vec<Block128>; STATE_SIZE]) -> Self {
        let mut current = columns_to_rows(layer_zero);
        let mut checkpoints = vec![current.clone()];
        let last = ((N_ROUNDS - 1) / CHECKPOINT_SPACING) * CHECKPOINT_SPACING;
        let mut round = 0;
        while round < last {
            let count = CHECKPOINT_SPACING.min(last - round);
            apply_round_span(round, count, &mut current);
            round += count;
            checkpoints.push(current.clone());
        }
        Self {
            checkpoints,
            window: Vec::new(),
            window_base: usize::MAX,
        }
    }

    fn state(&mut self, round: usize) -> Vec<[u128; STATE_SIZE]> {
        let base = (round / CHECKPOINT_SPACING) * CHECKPOINT_SPACING;
        if self.window_base != base {
            let length = CHECKPOINT_SPACING.min(N_ROUNDS - base);
            let checkpoint = core::mem::take(&mut self.checkpoints[base / CHECKPOINT_SPACING]);
            assert!(!checkpoint.is_empty(), "checkpoint requested twice");
            self.window = replay_window(checkpoint, base, length);
            self.window_base = base;
        }
        let slot = &mut self.window[round - self.window_base];
        assert!(!slot.is_empty(), "layer state requested twice");
        core::mem::take(slot)
    }
}

fn build_eq_table(point: &[Block256]) -> Vec<Block256> {
    let mut table = vec![Block256::ONE];
    for (coordinate, &challenge) in point.iter().enumerate() {
        let length = 1usize << coordinate;
        table.resize(2 * length, Block256::ZERO);
        for index in 0..length {
            let value = table[index];
            let high = value * challenge;
            table[index] = value + high;
            table[index + length] = high;
        }
    }
    table
}

fn eq_eval(left: &[Block256], right: &[Block256]) -> Block256 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(Block256::ONE, |product, (&left, &right)| {
            product * (Block256::ONE + left + right)
        })
}

/// Evaluate a base-field multilinear column at an extension-field point.
pub fn evaluate_column(column: &[Block128], point: &[Block256]) -> Block256 {
    assert_eq!(column.len(), 1usize << point.len());
    build_eq_table(point)
        .into_iter()
        .zip(column)
        .fold(Block256::ZERO, |sum, (weight, &value)| {
            sum + weight.scale_base(value)
        })
}

pub fn claims_from_outputs(
    outputs: &[[Vec<Block128>; STATE_SIZE]],
    points: &[Vec<Block256>],
) -> Vec<LaneClaim> {
    assert_eq!(outputs.len(), points.len());
    outputs
        .iter()
        .zip(points)
        .map(|(columns, point)| LaneClaim {
            point: point.clone(),
            values: std::array::from_fn(|lane| evaluate_column(&columns[lane], point)),
        })
        .collect()
}

fn lane_weights(alpha: Block256, groups: usize) -> Vec<[Block256; STATE_SIZE]> {
    let mut power = Block256::ONE;
    (0..groups)
        .map(|_| {
            std::array::from_fn(|_| {
                power *= alpha;
                power
            })
        })
        .collect()
}

fn column_weights(round: usize, lane_weights: &[Block256; STATE_SIZE]) -> [Block256; STATE_SIZE] {
    let mds = match round_kind(round) {
        RoundKind::Full => frost_gkr_poseidon2b::native::permutation::MDS_FULL,
        RoundKind::Partial => frost_gkr_poseidon2b::native::permutation::MDS_PARTIAL,
    };
    std::array::from_fn(|column| {
        (0..STATE_SIZE).fold(Block256::ZERO, |sum, lane| {
            sum + lane_weights[lane].scale_base(Block128::from(mds[lane][column]))
        })
    })
}

fn reconstruct(wire: &[Block256; WALK_DEGREE], claim: Block256) -> [Block256; WALK_DEGREE + 1] {
    let mut linear = claim;
    for &coefficient in &wire[1..] {
        linear += coefficient;
    }
    let mut full = [Block256::ZERO; WALK_DEGREE + 1];
    full[0] = wire[0];
    full[1] = linear;
    full[2..].copy_from_slice(&wire[1..]);
    full
}

#[inline]
fn horner(coefficients: &[Block256; WALK_DEGREE + 1], point: Block256) -> Block256 {
    let mut value = coefficients[WALK_DEGREE];
    for degree in (0..WALK_DEGREE).rev() {
        value = value * point + coefficients[degree];
    }
    value
}

fn layer_terms(round: usize, values: &[Block256; STATE_SIZE]) -> [Block256; STATE_SIZE] {
    match round_kind(round) {
        RoundKind::Full => std::array::from_fn(|lane| {
            sbox7_wide(values[lane] + Block256::from(Block128::from(ROUND_CONSTANTS[lane][round])))
        }),
        RoundKind::Partial => {
            let mut terms = *values;
            terms[0] =
                sbox7_wide(values[0] + Block256::from(Block128::from(ROUND_CONSTANTS[0][round])));
            terms
        }
    }
}

fn fast_build_eq_table(point: &[FastWide]) -> Vec<FastWide> {
    let mut table = vec![FastWide::ONE];
    for (coordinate, &challenge) in point.iter().enumerate() {
        let length = 1usize << coordinate;
        table.resize(2 * length, FastWide::ZERO);
        for index in 0..length {
            let value = table[index];
            let high = value * challenge;
            table[index] = value + high;
            table[index + length] = high;
        }
    }
    table
}

fn fast_eq_eval(left: &[Block256], right: &[FastWide]) -> FastWide {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(FastWide::ONE, |product, (&left, &right)| {
            product * (FastWide::ONE + FastWide::from_public(left) + right)
        })
}

fn fast_lane_weights(alpha: FastWide, groups: usize) -> Vec<[FastWide; STATE_SIZE]> {
    let mut power = FastWide::ONE;
    (0..groups)
        .map(|_| {
            std::array::from_fn(|_| {
                power *= alpha;
                power
            })
        })
        .collect()
}

fn fast_column_weights(
    round: usize,
    lane_weights: &[FastWide; STATE_SIZE],
) -> [FastWide; STATE_SIZE] {
    let matrix = match round_kind(round) {
        RoundKind::Full => &flat_schedule().full,
        RoundKind::Partial => &flat_schedule().partial,
    };
    std::array::from_fn(|column| {
        (0..STATE_SIZE).fold(FastWide::ZERO, |sum, lane| {
            sum + lane_weights[lane].scale_base(matrix[lane][column])
        })
    })
}

fn fast_sbox7_affine(a: FastWide, b: FastWide) -> [FastWide; WALK_DEGREE] {
    let a2 = a.square();
    let b2 = b.square();
    let a4 = a2.square();
    let b4 = b2.square();
    [
        a4 * a2 * a,
        a4 * a2 * b,
        a4 * b2 * a,
        a4 * b2 * b,
        b4 * a2 * a,
        b4 * a2 * b,
        b4 * b2 * a,
        b4 * b2 * b,
    ]
}

fn fast_relation_coefficients(
    round: usize,
    columns: &[FastWide; STATE_SIZE],
    state_base: &[FastWide; STATE_SIZE],
    state_delta: &[FastWide; STATE_SIZE],
) -> [FastWide; WALK_DEGREE] {
    let mut relation = [FastWide::ZERO; WALK_DEGREE];
    let full = round_kind(round) == RoundKind::Full;
    for lane in 0..STATE_SIZE {
        if full || lane == 0 {
            let coefficients = fast_sbox7_affine(
                state_base[lane] + FastWide::from_base_flat(flat_schedule().constants[lane][round]),
                state_delta[lane],
            );
            for degree in 0..WALK_DEGREE {
                relation[degree] += columns[lane] * coefficients[degree];
            }
        } else {
            relation[0] += columns[lane] * state_base[lane];
            relation[1] += columns[lane] * state_delta[lane];
        }
    }
    relation
}

#[inline]
fn fast_convolve_affine(
    relation: [FastWide; WALK_DEGREE],
    equality_base: FastWide,
    equality_delta: FastWide,
) -> [FastWide; WALK_DEGREE + 1] {
    let mut result = [FastWide::ZERO; WALK_DEGREE + 1];
    for degree in 0..WALK_DEGREE {
        result[degree] += equality_base * relation[degree];
        result[degree + 1] += equality_delta * relation[degree];
    }
    result
}

fn fast_base_round_contribution(
    round: usize,
    equality: &[FastWide],
    columns: &[FastWide; STATE_SIZE],
    states: &[[u128; STATE_SIZE]],
) -> [FastWide; WALK_DEGREE + 1] {
    debug_assert_eq!(states.len(), equality.len());
    (0..states.len() / 2)
        .into_par_iter()
        .fold(
            || [FastWide::ZERO; WALK_DEGREE + 1],
            |mut sum, pair| {
                let equality_base = equality[2 * pair];
                let equality_delta = equality_base + equality[2 * pair + 1];
                let state_base = states[2 * pair].map(FastWide::from_base_flat);
                let state_high = states[2 * pair + 1].map(FastWide::from_base_flat);
                let state_delta = std::array::from_fn(|lane| state_base[lane] + state_high[lane]);
                let contribution = fast_convolve_affine(
                    fast_relation_coefficients(round, columns, &state_base, &state_delta),
                    equality_base,
                    equality_delta,
                );
                for (sum, value) in sum.iter_mut().zip(contribution) {
                    *sum += value;
                }
                sum
            },
        )
        .reduce(
            || [FastWide::ZERO; WALK_DEGREE + 1],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
}

fn fast_wide_round_contribution(
    round: usize,
    native_coordinate: bool,
    equality: &[FastWide],
    columns: &[FastWide; STATE_SIZE],
    states: &[[FastWide; STATE_SIZE]],
) -> [FastWide; WALK_DEGREE + 1] {
    if native_coordinate {
        debug_assert_eq!(states.len(), equality.len());
        (0..states.len() / 2)
            .into_par_iter()
            .fold(
                || [FastWide::ZERO; WALK_DEGREE + 1],
                |mut sum, pair| {
                    let equality_base = equality[2 * pair];
                    let equality_delta = equality_base + equality[2 * pair + 1];
                    let state_base = states[2 * pair];
                    let state_high = states[2 * pair + 1];
                    let state_delta =
                        std::array::from_fn(|lane| state_base[lane] + state_high[lane]);
                    let contribution = fast_convolve_affine(
                        fast_relation_coefficients(round, columns, &state_base, &state_delta),
                        equality_base,
                        equality_delta,
                    );
                    for (sum, value) in sum.iter_mut().zip(contribution) {
                        *sum += value;
                    }
                    sum
                },
            )
            .reduce(
                || [FastWide::ZERO; WALK_DEGREE + 1],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            )
    } else {
        fast_convolve_affine(
            fast_relation_coefficients(round, columns, &states[0], &[FastWide::ZERO; STATE_SIZE]),
            equality[0],
            equality[0],
        )
    }
}

fn fast_fold_scalars(values: &mut Vec<FastWide>, challenge: FastWide) {
    *values = (0..values.len() / 2)
        .into_par_iter()
        .map(|pair| {
            let low = values[2 * pair];
            low + challenge * (low + values[2 * pair + 1])
        })
        .collect();
}

fn fast_fold_base_states(
    values: &[[u128; STATE_SIZE]],
    challenge: FastWide,
) -> Vec<[FastWide; STATE_SIZE]> {
    (0..values.len() / 2)
        .into_par_iter()
        .map(|pair| {
            let low = values[2 * pair];
            let high = values[2 * pair + 1];
            std::array::from_fn(|lane| {
                FastWide::from_base_flat(low[lane]) + challenge.scale_base(low[lane] ^ high[lane])
            })
        })
        .collect()
}

fn fast_fold_wide_states(values: &mut Vec<[FastWide; STATE_SIZE]>, challenge: FastWide) {
    *values = (0..values.len() / 2)
        .into_par_iter()
        .map(|pair| {
            let low = values[2 * pair];
            let high = values[2 * pair + 1];
            std::array::from_fn(|lane| low[lane] + challenge * (low[lane] + high[lane]))
        })
        .collect();
}

fn fast_compress(full: &[FastWide; WALK_DEGREE + 1]) -> [FastWide; WALK_DEGREE] {
    let mut wire = [FastWide::ZERO; WALK_DEGREE];
    wire[0] = full[0];
    wire[1..].copy_from_slice(&full[2..]);
    wire
}

#[inline]
fn fast_horner(coefficients: &[FastWide; WALK_DEGREE + 1], point: FastWide) -> FastWide {
    let mut value = coefficients[WALK_DEGREE];
    for degree in (0..WALK_DEGREE).rev() {
        value = value * point + coefficients[degree];
    }
    value
}

fn fast_layer_terms(round: usize, values: &[FastWide; STATE_SIZE]) -> [FastWide; STATE_SIZE] {
    match round_kind(round) {
        RoundKind::Full => std::array::from_fn(|lane| {
            let value =
                values[lane] + FastWide::from_base_flat(flat_schedule().constants[lane][round]);
            let square = value.square();
            value * square * square.square()
        }),
        RoundKind::Partial => {
            let mut terms = *values;
            let value = values[0] + FastWide::from_base_flat(flat_schedule().constants[0][round]);
            let square = value.square();
            terms[0] = value * square * square.square();
            terms
        }
    }
}

/// Prove one aggregate walk with one output claim per unequal-width region.
pub fn prove_ragged<C: FiatShamir<Block128>>(
    layer_zero: &[&[Vec<Block128>; STATE_SIZE]],
    output_groups: &[LaneClaim],
    channel: &mut WideChannel<C>,
) -> (RaggedProof, Vec<LaneClaim>) {
    assert!(!layer_zero.is_empty());
    assert_eq!(layer_zero.len(), output_groups.len());
    let widths = layer_zero
        .iter()
        .map(|columns| {
            let width = columns[0].len();
            assert!(width > 1 && width.is_power_of_two());
            assert!(columns.iter().all(|column| column.len() == width));
            width.trailing_zeros() as usize
        })
        .collect::<Vec<_>>();
    assert!(output_groups
        .iter()
        .zip(&widths)
        .all(|(group, &width)| group.point.len() == width));
    let max_width = *widths.iter().max().unwrap();
    absorb_statement(channel, &widths, output_groups);

    let mut servers = layer_zero
        .par_iter()
        .map(|columns| DescendingLayerStates::new(columns))
        .collect::<Vec<_>>();
    let mut groups = output_groups.to_vec();
    let mut layers = Vec::with_capacity(N_ROUNDS);

    for layer in (1..=N_ROUNDS).rev() {
        let round = layer - 1;
        let alpha = FastWide::from_public(channel.squeeze());
        let weights = fast_lane_weights(alpha, groups.len());
        let columns = weights
            .iter()
            .map(|weight| fast_column_weights(round, weight))
            .collect::<Vec<_>>();
        let mut claim =
            groups
                .iter()
                .zip(&weights)
                .fold(FastWide::ZERO, |sum, (group, weights)| {
                    sum + (0..STATE_SIZE).fold(FastWide::ZERO, |inner, lane| {
                        inner + weights[lane] * FastWide::from_public(group.values[lane])
                    })
                });
        let mut equality = groups
            .iter()
            .map(|group| {
                fast_build_eq_table(
                    &group
                        .point
                        .iter()
                        .copied()
                        .map(FastWide::from_public)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let base_states = servers
            .par_iter_mut()
            .map(|server| server.state(round))
            .collect::<Vec<_>>();
        let mut states = vec![Vec::new(); groups.len()];
        let mut point = Vec::with_capacity(max_width);
        let mut round_coeffs = Vec::with_capacity(max_width);

        for coordinate in 0..max_width {
            let contributions = (0..groups.len())
                .into_par_iter()
                .map(|region| {
                    if coordinate == 0 {
                        fast_base_round_contribution(
                            round,
                            &equality[region],
                            &columns[region],
                            &base_states[region],
                        )
                    } else {
                        fast_wide_round_contribution(
                            round,
                            coordinate < widths[region],
                            &equality[region],
                            &columns[region],
                            &states[region],
                        )
                    }
                })
                .collect::<Vec<_>>();
            let mut full = [FastWide::ZERO; WALK_DEGREE + 1];
            for contribution in contributions {
                for (coefficient, contribution) in full.iter_mut().zip(contribution) {
                    *coefficient += contribution;
                }
            }
            debug_assert_eq!(full[0] + fast_horner(&full, FastWide::ONE), claim);
            let fast_wire = fast_compress(&full);
            let wire = fast_wire.map(FastWide::to_public);
            channel.absorb_slice(&wire);
            let public_challenge = channel.squeeze();
            let challenge = FastWide::from_public(public_challenge);
            claim = fast_horner(&full, challenge);
            point.push(challenge);
            round_coeffs.push(wire);

            for region in 0..groups.len() {
                if coordinate < widths[region] {
                    fast_fold_scalars(&mut equality[region], challenge);
                    if coordinate == 0 {
                        states[region] = fast_fold_base_states(&base_states[region], challenge);
                    } else {
                        fast_fold_wide_states(&mut states[region], challenge);
                    }
                } else {
                    equality[region][0] *= FastWide::ONE + challenge;
                }
            }
        }

        let fast_next_values = states.iter().map(|state| state[0]).collect::<Vec<_>>();
        let next_values = fast_next_values
            .iter()
            .map(|values| values.map(FastWide::to_public))
            .collect::<Vec<_>>();
        for values in &next_values {
            channel.absorb_slice(values);
        }
        let mut expected = FastWide::ZERO;
        for region in 0..groups.len() {
            let mut high_gate = FastWide::ONE;
            for &coordinate in &point[widths[region]..] {
                high_gate *= FastWide::ONE + coordinate;
            }
            let aligned = fast_eq_eval(&groups[region].point, &point[..widths[region]]) * high_gate;
            let terms = fast_layer_terms(round, &fast_next_values[region]);
            let dot = (0..STATE_SIZE).fold(FastWide::ZERO, |sum, lane| {
                sum + columns[region][lane] * terms[lane]
            });
            expected += aligned * dot;
        }
        debug_assert_eq!(expected, claim, "ragged layer {layer}");
        layers.push(RaggedLayerProof {
            round_coeffs,
            next_values: next_values.clone(),
        });
        groups = next_values
            .into_iter()
            .enumerate()
            .map(|(region, values)| LaneClaim {
                point: point[..widths[region]]
                    .iter()
                    .copied()
                    .map(FastWide::to_public)
                    .collect(),
                values,
            })
            .collect();
    }

    (RaggedProof { layers }, groups)
}

pub fn verify_ragged<C: FiatShamir<Block128>>(
    widths: &[usize],
    output_groups: &[LaneClaim],
    proof: &RaggedProof,
    channel: &mut WideChannel<C>,
) -> Result<Vec<LaneClaim>, RaggedError> {
    if widths.is_empty()
        || widths.len() != output_groups.len()
        || output_groups
            .iter()
            .zip(widths)
            .any(|(group, &width)| width == 0 || group.point.len() != width)
    {
        return Err(RaggedError::Shape);
    }
    let max_width = *widths.iter().max().ok_or(RaggedError::Shape)?;
    if proof.layers.len() != N_ROUNDS
        || proof.layers.iter().any(|layer| {
            layer.round_coeffs.len() != max_width || layer.next_values.len() != output_groups.len()
        })
    {
        return Err(RaggedError::Shape);
    }

    absorb_statement(channel, widths, output_groups);
    let mut groups = output_groups.to_vec();
    for (layer_index, layer_proof) in proof.layers.iter().enumerate() {
        let layer = N_ROUNDS - layer_index;
        let round = layer - 1;
        let alpha = channel.squeeze();
        let weights = lane_weights(alpha, groups.len());
        let columns = weights
            .iter()
            .map(|weight| column_weights(round, weight))
            .collect::<Vec<_>>();
        let mut claim =
            groups
                .iter()
                .zip(&weights)
                .fold(Block256::ZERO, |sum, (group, weights)| {
                    sum + (0..STATE_SIZE).fold(Block256::ZERO, |inner, lane| {
                        inner + weights[lane] * group.values[lane]
                    })
                });
        let mut point = Vec::with_capacity(max_width);
        for wire in &layer_proof.round_coeffs {
            channel.absorb_slice(wire);
            let full = reconstruct(wire, claim);
            let challenge = channel.squeeze();
            claim = horner(&full, challenge);
            point.push(challenge);
        }
        for values in &layer_proof.next_values {
            channel.absorb_slice(values);
        }

        let mut expected = Block256::ZERO;
        for region in 0..groups.len() {
            let mut high_gate = Block256::ONE;
            for &coordinate in &point[widths[region]..] {
                high_gate *= Block256::ONE + coordinate;
            }
            let aligned = eq_eval(&groups[region].point, &point[..widths[region]]) * high_gate;
            let terms = layer_terms(round, &layer_proof.next_values[region]);
            let dot = (0..STATE_SIZE).fold(Block256::ZERO, |sum, lane| {
                sum + columns[region][lane] * terms[lane]
            });
            expected += aligned * dot;
        }
        if expected != claim {
            return Err(RaggedError::LayerMismatch(layer));
        }
        groups = layer_proof
            .next_values
            .iter()
            .enumerate()
            .map(|(region, &values)| LaneClaim {
                point: point[..widths[region]].to_vec(),
                values,
            })
            .collect();
    }
    Ok(groups)
}

/// Bind the terminal reductions to the actual layer-S_0 columns.
pub fn discharge_terminals(
    layer_zero: &[&[Vec<Block128>; STATE_SIZE]],
    terminals: &[LaneClaim],
) -> bool {
    layer_zero.len() == terminals.len()
        && layer_zero
            .par_iter()
            .zip(terminals.par_iter())
            .all(|(columns, terminal)| {
                terminal.values
                    == std::array::from_fn(|lane| evaluate_column(&columns[lane], &terminal.point))
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use frost_gkr_poseidon2b::channel::Poseidon2bChannel;
    use frost_gkr_poseidon2b::native::permutation::Poseidon2bPermutation;

    struct Rng(u64);

    impl Rng {
        fn next_u128(&mut self) -> u128 {
            fn splitmix(state: &mut u64) -> u64 {
                *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = *state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^ (value >> 31)
            }
            (splitmix(&mut self.0) as u128) | ((splitmix(&mut self.0) as u128) << 64)
        }

        fn base(&mut self) -> Block128 {
            Block128(self.next_u128())
        }

        fn wide(&mut self) -> Block256 {
            Block256::new(self.base(), self.base())
        }
    }

    fn fixture(widths: &[usize]) -> Vec<[Vec<Block128>; STATE_SIZE]> {
        widths
            .iter()
            .enumerate()
            .map(|(region, &width)| {
                let mut rng = Rng(0x7261_6767_6564_0000 + region as u64);
                let raw = std::array::from_fn(|_| {
                    (0..1usize << width).map(|_| rng.base()).collect::<Vec<_>>()
                });
                initial_layer_columns(&raw)
            })
            .collect()
    }

    fn points(widths: &[usize]) -> Vec<Vec<Block256>> {
        let mut rng = Rng(0x706f_696e_7473_0001);
        widths
            .iter()
            .map(|&width| (0..width).map(|_| rng.wide()).collect())
            .collect()
    }

    fn prove_verify(
        inputs: &[[Vec<Block128>; STATE_SIZE]],
        groups: &[LaneClaim],
    ) -> (RaggedProof, Vec<LaneClaim>) {
        let references = inputs.iter().collect::<Vec<_>>();
        let mut prover = WideChannel::new(Poseidon2bChannel::new());
        let (proof, prover_terminals) = prove_ragged(&references, groups, &mut prover);
        let widths = inputs
            .iter()
            .map(|columns| columns[0].len().trailing_zeros() as usize)
            .collect::<Vec<_>>();
        let mut verifier = WideChannel::new(Poseidon2bChannel::new());
        let verifier_terminals = verify_ragged(&widths, groups, &proof, &mut verifier).unwrap();
        assert_eq!(prover_terminals, verifier_terminals);
        assert!(discharge_terminals(&references, &verifier_terminals));
        (proof, verifier_terminals)
    }

    #[test]
    fn flat_wide_arithmetic_matches_public_tower_arithmetic() {
        let mut rng = Rng(0x666c_6174_5f77_6964);
        for _ in 0..1_000 {
            let left = rng.wide();
            let right = rng.wide();
            let fast_left = FastWide::from_public(left);
            let fast_right = FastWide::from_public(right);
            assert_eq!((fast_left + fast_right).to_public(), left + right);
            assert_eq!((fast_left * fast_right).to_public(), left * right);
            assert_eq!(fast_left.square().to_public(), left.square());
        }
    }

    #[test]
    fn specialized_flat_mds_matches_dense_matrices() {
        let mut rng = Rng(0x6d64_735f_7061_6972);
        for _ in 0..1_000 {
            let input = std::array::from_fn(|_| tower_to_flat_u128(rng.base().0));
            assert_eq!(
                apply_full_mds_flat(input),
                apply_mds_flat(&flat_schedule().full, input)
            );
            assert_eq!(
                apply_partial_mds_flat(input),
                apply_mds_flat(&flat_schedule().partial, input)
            );
        }
    }

    #[test]
    fn flat_66_layer_path_matches_native_poseidon2b() {
        let mut rng = Rng(0x706f_7365_6964_6f6e);
        let raw = std::array::from_fn(|_| (0..32).map(|_| rng.base()).collect::<Vec<_>>());
        let layer_zero = initial_layer_columns(&raw);
        let output = output_layer_columns(&layer_zero);
        let permutation = Poseidon2bPermutation;
        for index in 0..32 {
            let mut expected = std::array::from_fn(|lane| raw[lane][index]);
            permutation.permute_mut(&mut expected);
            assert_eq!(std::array::from_fn(|lane| output[lane][index]), expected);
        }
    }

    #[test]
    fn unequal_width_roundtrip_and_mutation_rejection() {
        let widths = [2, 4, 3];
        let inputs = fixture(&widths);
        let outputs = inputs.iter().map(output_layer_columns).collect::<Vec<_>>();
        let groups = claims_from_outputs(&outputs, &points(&widths));
        let (proof, _) = prove_verify(&inputs, &groups);

        let mut bad = proof.clone();
        bad.layers[7].round_coeffs[1][3] += Block256::ONE;
        let mut verifier = WideChannel::new(Poseidon2bChannel::new());
        assert!(verify_ragged(&widths, &groups, &bad, &mut verifier).is_err());
    }

    #[test]
    fn ragged_independent_and_physical_padding_prove_the_same_output_claims() {
        let widths = [2, 4, 3];
        let max_width = *widths.iter().max().unwrap();
        let inputs = fixture(&widths);
        let outputs = inputs.iter().map(output_layer_columns).collect::<Vec<_>>();
        let native_points = points(&widths);
        let native_groups = claims_from_outputs(&outputs, &native_points);
        let (ragged, _) = prove_verify(&inputs, &native_groups);

        let independent_bytes = inputs
            .iter()
            .zip(&native_groups)
            .map(|(input, group)| {
                let one_input = vec![input.clone()];
                let one_group = vec![group.clone()];
                prove_verify(&one_input, &one_group).0.byte_len()
            })
            .sum::<usize>();

        let padded_inputs = inputs
            .iter()
            .map(|columns| {
                std::array::from_fn(|lane| {
                    let mut column = columns[lane].clone();
                    column.resize(1usize << max_width, Block128::ZERO);
                    column
                })
            })
            .collect::<Vec<_>>();
        let padded_outputs = padded_inputs
            .iter()
            .map(output_layer_columns)
            .collect::<Vec<_>>();
        let padded_points = native_points
            .iter()
            .map(|point| {
                let mut point = point.clone();
                point.resize(max_width, Block256::ZERO);
                point
            })
            .collect::<Vec<_>>();
        let padded_groups = claims_from_outputs(&padded_outputs, &padded_points);
        assert_eq!(
            native_groups
                .iter()
                .map(|group| group.values)
                .collect::<Vec<_>>(),
            padded_groups
                .iter()
                .map(|group| group.values)
                .collect::<Vec<_>>()
        );
        let (padded, _) = prove_verify(&padded_inputs, &padded_groups);

        assert_eq!(ragged.byte_len(), padded.byte_len());
        assert!(ragged.byte_len() < independent_bytes);
    }
}
