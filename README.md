# FROST-GKR

**Frobenius Reduction Over Shifted Tables**

[Protocol explainer](https://lab.parano1d.org/research/frost-gkr-global-trace-protocol/) · [Paper](https://github.com/ignotusnemo/o1-lab/blob/main/papers/FROST_GKR.pdf) · [Published benchmark](results/2026-07-19-i7-1365u-20-sample.md) · [Ragged multi-instance GKR](RAGGED.md)

FROST-GKR is a research component of
[ParanO(1)d](https://parano1d.org/), a proof-native Layer 1 secured by proof of
work. This repository publishes the construction, reference
implementation, and comparative benchmark as a self-contained artifact.

FROST-GKR is a GKR arithmetization for repeated sequential computation over
binary fields. It places every slot, round, and lane of a Poseidon2b batch in
one Boolean hypercube, proves the direct degree-seven S-box relation, and
binds adjacent rounds through shifted tables.

For the public sequence of 59 permutations evaluated in the paper, this
replaces 472 product-chain constraint sumchecks with two sumchecks whose depth
is the logarithm of the padded table.

| Metric | Product-chain GKR | FROST-GKR | Reduction |
|---|---:|---:|---:|
| Constraint sumchecks | 472 | 2 | 236× |
| Constraint rounds | 4,248 | 30 | 141.60× |
| All sumcheck rounds | 4,263 | 75 | 56.84× |
| Raw algebraic proof | 287,712 B | 5,568 B | 51.67× |
| Prover median | 1,605.931 ms | 150.218 ms | 10.69× |
| Protocol verifier median | 984.269 ms | 66.499 ms | 14.80× |

The timings are medians from 20 interleaved release-mode samples on an Intel
Core i7-1365U. Proof bytes count raw algebraic field elements, including
terminal batching and excluding serialization framing and polynomial-
commitment openings. See [BENCHMARK.md](BENCHMARK.md) for the precise timing
boundaries.

## The construction in three steps

1. **Unify.** Encode `slot × round × lane` in one 15-variable Boolean
   hypercube. The witness has three columns: input to the nonlinear layer,
   output from it, and round state.
2. **Prove.** Check the direct Poseidon2b relation in one degree-nine
   sumcheck. Over `GF(2^128)`, Frobenius squaring evaluates `x^7` with two
   multiplications and two linear squarings.
3. **Shift.** Materialize round-shifted views and reduce them back to the
   original witness columns with one degree-two sumcheck. Three terminal
   batches expose the MLE claims for a commitment layer.

The repository contains both implementations of the same application-neutral
sequence, field, and Fiat–Shamir transcript. The benchmark first proves,
verifies, natively discharges every terminal claim, and asserts the exact proof
accounting. Only then does it collect timings.

## Reproduce

Rust 1.96.0 is pinned by `rust-toolchain.toml`. Release builds use the local
CPU through `-C target-cpu=native`, matching the paper artifact.

```sh
git clone https://github.com/ignotusnemo/frost-gkr.git
cd frost-gkr
cargo test --release --locked --workspace
cargo run --release --locked -p frost-gkr-bench -- --warmups 3 --samples 20
```

The benchmark prints a complete Markdown report to stdout. To record a run:

```sh
cargo run --release --locked -p frost-gkr-bench -- \
  --warmups 3 --samples 20 > results/my-machine.md
```

## Repository map

```text
benchmark/          correctness-gated comparative benchmark
crates/core/        GF(2^128), packed arithmetic, MLE and sumcheck primitives
crates/poseidon2b/  native Poseidon2b statement and Fiat–Shamir channel
crates/gkr/         product-chain, FROST-GKR, and ragged GKR implementations
results/            published benchmark reports
```

The artifact is intentionally application-neutral. Its public statement is a
fixed sequence of 59 Poseidon2b permutations; it contains no ParanO(1)d node,
wallet, transaction format, state model, consensus, networking, or deployment
integration. Terminal MLE reductions are evaluated directly for the
comparison. An integrating proof system replaces that harness step with its
own polynomial-commitment openings and end-to-end accounting.

## Citation and license

The author is **Ignotus Nemo**. Citation metadata is available in
[`CITATION.cff`](CITATION.cff). The implementation is licensed under
[Apache License 2.0](LICENSE).
