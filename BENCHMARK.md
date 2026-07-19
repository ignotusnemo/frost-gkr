# Benchmark protocol

The benchmark compares the preserved product-chain GKR implementation with
FROST-GKR on the same batch of 59 width-four Poseidon2b permutations over
`GF(2^128)` and the same Poseidon2b Fiat–Shamir channel.

## Correctness gate

Before timing begins, the executable:

1. constructs both proofs for the same public statement;
2. verifies both proofs with independent transcripts;
3. checks that prover and verifier return identical terminal reductions;
4. evaluates every terminal MLE claim against the native witness tables; and
5. asserts exact raw proof sizes of 287,712 and 5,568 bytes.

Any failure aborts the run. Timing cannot silently continue on a divergent
statement or invalid proof.

## Sampling

The publication command is:

```sh
cargo run --release --locked -p frost-gkr-bench -- --warmups 3 --samples 20
```

Legacy and FROST-GKR operations alternate within each pair, and the first
operation alternates across samples. The report includes median, nearest-rank
p95, mean, sample standard deviation, minimum, and maximum. It also records
the CPU, enabled instruction sets, thread count, governor, turbo state, kernel,
and Rust version.

## Timing boundaries

- **Prover** is the public prover call, including native witness work, MLE
  materialization, and sumcheck generation.
- **Protocol verifier** ends at the verified terminal MLE reductions.
- **Native terminal discharge** evaluates those reductions directly against
  the preserved witness tables.
- **Verifier + native discharge** reports the two preceding operations as one
  timed call.
- **Shared sequence evaluation** is reported separately and is outside both
  prover timings.

Native discharge is a comparison harness, not a claim about a deployed
polynomial commitment scheme. Raw algebraic proof bytes exclude serialization
framing and external polynomial-commitment openings.

## Publication run

The measurements reported in the paper are preserved in
[`results/2026-07-19-i7-1365u-20-sample.md`](results/2026-07-19-i7-1365u-20-sample.md).
