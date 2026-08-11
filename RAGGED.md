# Ragged multi-instance GKR

The homogeneous FROST-GKR benchmark places repeated Poseidon2b executions in
one global trace. A recursive verifier presents a second batching problem:
its committed hash regions need not have the same Boolean width.

For regions of widths `w_a`, two direct choices lose one side of the tradeoff:

- one native GKR walk per region preserves prover work but repeats the
  transcript and verification path; or
- one aggregate walk after padding every region to `W = max_a w_a` has one
  transcript but materializes every region as `2^W` rows.

The implementation in `crates/gkr/src/ragged.rs` gives the aggregate walk an
implicit zero extension instead. In characteristic two, region `a` is embedded
with

```text
chi_a(x) = product over j = w_a .. W - 1 of (1 + x_j).
```

On the Boolean hypercube, `chi_a` is one exactly when every added coordinate is
zero. Therefore

```text
sum over {0,1}^W of f_a(x_<w_a) chi_a(x_>=w_a)
    =
sum over {0,1}^w_a of f_a(x).
```

Each added variable has individual degree one. The native Poseidon2b layer
relation already has individual degree eight after multiplication by the
equality polynomial, so the ragged embedding does not increase the sumcheck
degree or the number of transmitted coefficients per round.

During the first `w_a` rounds, the prover folds the physical table for region
`a`. During the remaining `W - w_a` rounds, it keeps the one remaining state
evaluation and folds only the selector. The dominant witness work is therefore

```text
O(L * sum_a 2^w_a),
```

plus lower-order per-region work in the added coordinates, rather than

```text
O(L * A * 2^W)
```

for physical max-width padding. Here `L` is the 66-layer Poseidon2b walk and
`A` is the number of regions.

## Instantiated comparison

The benchmark uses the nine Boolean widths from the production B25 hash
regions:

```text
[14, 15, 17, 16, 12, 15, 16, 12, 13]
```

They contain 360,448 native Poseidon rows. Padding all nine regions to width 17
would materialize 1,179,648 rows, an expansion of 3.2727 times.

All three benchmark variants prove the same output claims and the same
66-layer relation:

1. nine independent native-width walks;
2. one physically max-padded walk; and
3. one implicit ragged walk.

The published laptop run reports:

| Construction | Prover median | Prover range | Raw proof | Peak RSS |
|---|---:|---:|---:|---:|
| Independent native walks | 17.609 s | 17.553–17.902 s | 2,272,512 B | 166,912 KiB |
| Physical max padding | 43.496 s | 32.155–46.342 s | 363,264 B | 1,382,252 KiB |
| Implicit ragged walk | 10.830 s | 10.621–15.044 s | 363,264 B | 460,484 KiB |

Ragged proving was 1.626 times faster than nine independent walks at the
median while emitting a 6.256-times smaller algebraic transcript. Comparing
the fastest isolated samples, it was 3.028 times faster than physical padding;
the exact physical-row reduction is 3.2727 times. The complete report is
preserved in
[`results/2026-08-11-i7-1365u-ragged-3-sample.md`](results/2026-08-11-i7-1365u-ragged-3-sample.md).

Committed rows remain in `GF(2^128)`. Sumcheck claims, messages, and challenges
use a quadratic `GF(2^256)` extension. Fiat-Shamir challenges are sampled from
a `2^255`-element affine support outside the distinguished base subfield.

The prover keeps both extension coordinates in the CLMUL-friendly flat basis.
On x86-64 builds with `AVX2` and `VPCLMULQDQ`, one 256-bit instruction path
computes the two base-field products together. The verifier uses the public
tower-field implementation, so every benchmarked proof crosses an independent
representation boundary before acceptance.

## Correctness gates

The test and benchmark paths require all of the following before reporting a
measurement:

- the independent, padded, and ragged constructions expose identical output
  claims;
- prover and verifier derive identical terminal reductions;
- every terminal is discharged against the native layer-zero columns;
- a mutated proof is rejected;
- the flat AVX2/VPCLMUL arithmetic matches the public tower arithmetic;
- the specialized MDS kernels match dense matrix evaluation; and
- the complete flat 66-layer path matches native Poseidon2b.

## Reproduce

The expensive benchmark runs each timed proof in a new process. Variant order
rotates between sample rounds, and a cooldown separates workers so later
measurements do not inherit one continuously loaded process.

```sh
cargo test --release --locked --workspace --all-targets
cargo run --release --locked -p frost-gkr-bench --bin ragged -- \
  --warmups 0 --samples 3 --cooldown-seconds 20 --explain
```

Proof sizes are raw algebraic transcript bytes. They exclude serialization
framing and polynomial-commitment openings.

## Boundary of the result

The benchmark is Poseidon2b over binary tower fields. Its timings do not
transfer directly to Poseidon variants over other fields or to a complete
external PCS. The reusable result is the ragged embedding: heterogeneous
regions can share one GKR walk without physically padding every witness to the
largest region.
