// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

//! Reproducible three-way benchmark for unequal-width Poseidon2b regions.
//!
//! The compared constructions prove the same output claims over the same
//! 66-round relation:
//! 1. one independent walk per native region;
//! 2. one aggregate walk after physically padding every region to max width;
//! 3. one aggregate ragged walk with implicit high-coordinate selectors.

use std::env;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use frost_gkr::{
    claims_from_outputs, discharge_terminals, initial_layer_columns, output_layer_columns,
    prove_ragged, verify_ragged, RaggedLaneClaim, RaggedProof, WideChannel,
};
use frost_gkr_core::{Block128, Block256, TowerField};
use frost_gkr_poseidon2b::channel::Poseidon2bChannel;

const PRODUCTION_WIDTHS: [usize; 9] = [14, 15, 17, 16, 12, 15, 16, 12, 13];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Independent,
    Padded,
    Ragged,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent native walks",
            Self::Padded => "one physically max-padded walk",
            Self::Ragged => "one implicit ragged walk",
        }
    }

    fn argument(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Padded => "padded",
            Self::Ragged => "ragged",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "independent" => Some(Self::Independent),
            "padded" => Some(Self::Padded),
            "ragged" => Some(Self::Ragged),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Config {
    warmups: usize,
    samples: usize,
    cooldown_seconds: u64,
}

#[derive(Debug)]
struct Statistics {
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl Statistics {
    fn from_samples(mut samples: Vec<f64>) -> Self {
        assert!(!samples.is_empty());
        samples.sort_by(f64::total_cmp);
        let median_ms = if samples.len().is_multiple_of(2) {
            (samples[samples.len() / 2 - 1] + samples[samples.len() / 2]) / 2.0
        } else {
            samples[samples.len() / 2]
        };
        let p95_index = ((0.95 * samples.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(samples.len() - 1);
        Self {
            median_ms,
            p95_ms: samples[p95_index],
            min_ms: samples[0],
            max_ms: samples[samples.len() - 1],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Measurement {
    prover_ms: f64,
    verifier_ms: f64,
    full_verifier_ms: f64,
    physical_rows: usize,
    layer_bytes: usize,
    proof_bytes: usize,
    peak_rss_kib: u64,
}

impl Measurement {
    fn encode(self) -> String {
        format!(
            "RESULT\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}",
            self.prover_ms,
            self.verifier_ms,
            self.full_verifier_ms,
            self.physical_rows,
            self.layer_bytes,
            self.proof_bytes,
            self.peak_rss_kib
        )
    }

    fn decode(line: &str) -> Option<Self> {
        let mut fields = line.split('\t');
        if fields.next()? != "RESULT" {
            return None;
        }
        let measurement = Self {
            prover_ms: fields.next()?.parse().ok()?,
            verifier_ms: fields.next()?.parse().ok()?,
            full_verifier_ms: fields.next()?.parse().ok()?,
            physical_rows: fields.next()?.parse().ok()?,
            layer_bytes: fields.next()?.parse().ok()?,
            proof_bytes: fields.next()?.parse().ok()?,
            peak_rss_kib: fields.next()?.parse().ok()?,
        };
        fields.next().is_none().then_some(measurement)
    }
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn base(&mut self) -> Block128 {
        Block128((self.next_u64() as u128) | ((self.next_u64() as u128) << 64))
    }

    fn wide(&mut self) -> Block256 {
        Block256::new(self.base(), self.base())
    }
}

fn parse_usize(arguments: &[String], flag: &str, default: usize) -> usize {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| {
            pair[1]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid value for {flag}"))
        })
        .unwrap_or(default)
}

fn fixture(widths: &[usize]) -> Vec<[Vec<Block128>; 4]> {
    widths
        .iter()
        .enumerate()
        .map(|(region, &width)| {
            let mut rng = Rng(0x6672_6f73_745f_0000 + region as u64);
            let raw = std::array::from_fn(|_| {
                (0..1usize << width).map(|_| rng.base()).collect::<Vec<_>>()
            });
            initial_layer_columns(&raw)
        })
        .collect()
}

fn native_points(widths: &[usize]) -> Vec<Vec<Block256>> {
    let mut rng = Rng(0x7261_6767_6564_0001);
    widths
        .iter()
        .map(|&width| (0..width).map(|_| rng.wide()).collect())
        .collect()
}

fn prepare(variant: Variant) -> (Vec<[Vec<Block128>; 4]>, Vec<RaggedLaneClaim>, Vec<usize>) {
    let native_inputs = fixture(&PRODUCTION_WIDTHS);
    let points = native_points(&PRODUCTION_WIDTHS);
    let native_outputs = native_inputs
        .iter()
        .map(output_layer_columns)
        .collect::<Vec<_>>();
    let native_groups = claims_from_outputs(&native_outputs, &points);
    if variant == Variant::Padded {
        let max_width = *PRODUCTION_WIDTHS.iter().max().unwrap();
        let inputs = native_inputs
            .iter()
            .map(|columns| {
                std::array::from_fn(|lane| {
                    let mut column = columns[lane].clone();
                    column.resize(1usize << max_width, Block128::ZERO);
                    column
                })
            })
            .collect::<Vec<_>>();
        let padded_points = points
            .iter()
            .map(|point| {
                let mut point = point.clone();
                point.resize(max_width, Block256::ZERO);
                point
            })
            .collect::<Vec<_>>();
        let groups = native_groups
            .into_iter()
            .zip(padded_points)
            .map(|(group, point)| RaggedLaneClaim {
                point,
                values: group.values,
            })
            .collect();
        (inputs, groups, vec![max_width; PRODUCTION_WIDTHS.len()])
    } else {
        (native_inputs, native_groups, PRODUCTION_WIDTHS.to_vec())
    }
}

enum Proofs {
    Independent(Vec<RaggedProof>),
    Aggregate(RaggedProof),
}

impl Proofs {
    fn byte_len(&self) -> usize {
        match self {
            Self::Independent(proofs) => proofs.iter().map(RaggedProof::byte_len).sum(),
            Self::Aggregate(proof) => proof.byte_len(),
        }
    }
}

fn prove_once(
    variant: Variant,
    inputs: &[[Vec<Block128>; 4]],
    groups: &[RaggedLaneClaim],
) -> Proofs {
    match variant {
        Variant::Independent => Proofs::Independent(
            inputs
                .iter()
                .zip(groups)
                .map(|(input, group)| {
                    let mut channel = WideChannel::new(Poseidon2bChannel::new());
                    prove_ragged(&[input], std::slice::from_ref(group), &mut channel).0
                })
                .collect(),
        ),
        Variant::Padded | Variant::Ragged => {
            let references = inputs.iter().collect::<Vec<_>>();
            let mut channel = WideChannel::new(Poseidon2bChannel::new());
            Proofs::Aggregate(prove_ragged(&references, groups, &mut channel).0)
        }
    }
}

fn verify_once(
    variant: Variant,
    inputs: &[[Vec<Block128>; 4]],
    groups: &[RaggedLaneClaim],
    widths: &[usize],
    proofs: &Proofs,
    discharge: bool,
) {
    match (variant, proofs) {
        (Variant::Independent, Proofs::Independent(proofs)) => {
            for (((input, group), &width), proof) in
                inputs.iter().zip(groups).zip(widths).zip(proofs)
            {
                let mut channel = WideChannel::new(Poseidon2bChannel::new());
                let terminal =
                    verify_ragged(&[width], std::slice::from_ref(group), proof, &mut channel)
                        .expect("independent verifier rejected an honest proof");
                if discharge {
                    assert!(discharge_terminals(&[input], &terminal));
                }
            }
        }
        (Variant::Padded | Variant::Ragged, Proofs::Aggregate(proof)) => {
            let mut channel = WideChannel::new(Poseidon2bChannel::new());
            let terminals = verify_ragged(widths, groups, proof, &mut channel)
                .expect("aggregate verifier rejected an honest proof");
            if discharge {
                let references = inputs.iter().collect::<Vec<_>>();
                assert!(discharge_terminals(&references, &terminals));
            }
        }
        _ => panic!("proof bundle does not match benchmark variant"),
    }
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

fn print_stat(label: &str, stat: &Statistics) {
    println!(
        "| {label} | {:.3} | {:.3} | {:.3} | {:.3} |",
        stat.median_ms, stat.p95_ms, stat.min_ms, stat.max_ms
    );
}

fn sample_worker(variant: Variant, mutation_check: bool) {
    let (inputs, groups, widths) = prepare(variant);
    let started = Instant::now();
    let honest_proof = black_box(prove_once(variant, black_box(&inputs), black_box(&groups)));
    let prover_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    verify_once(
        variant,
        black_box(&inputs),
        black_box(&groups),
        &widths,
        black_box(&honest_proof),
        false,
    );
    let verifier_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    verify_once(
        variant,
        black_box(&inputs),
        black_box(&groups),
        &widths,
        black_box(&honest_proof),
        true,
    );
    let full_verifier_ms = started.elapsed().as_secs_f64() * 1000.0;

    if mutation_check {
        let mut corrupted = match &honest_proof {
            Proofs::Independent(proofs) => Proofs::Independent(proofs.clone()),
            Proofs::Aggregate(proof) => Proofs::Aggregate(proof.clone()),
        };
        match &mut corrupted {
            Proofs::Independent(proofs) => {
                proofs[0].layers[3].round_coeffs[0][2] += Block256::ONE;
            }
            Proofs::Aggregate(proof) => {
                proof.layers[3].round_coeffs[0][2] += Block256::ONE;
            }
        }
        let mutation_rejected = match &corrupted {
            Proofs::Independent(proofs) => {
                let mut channel = WideChannel::new(Poseidon2bChannel::new());
                verify_ragged(
                    &[widths[0]],
                    std::slice::from_ref(&groups[0]),
                    &proofs[0],
                    &mut channel,
                )
                .is_err()
            }
            Proofs::Aggregate(proof) => {
                let mut channel = WideChannel::new(Poseidon2bChannel::new());
                verify_ragged(&widths, &groups, proof, &mut channel).is_err()
            }
        };
        assert!(mutation_rejected, "mutated proof was accepted");
    }

    let physical_rows = inputs.iter().map(|columns| columns[0].len()).sum::<usize>();
    println!(
        "{}",
        Measurement {
            prover_ms,
            verifier_ms,
            full_verifier_ms,
            physical_rows,
            layer_bytes: physical_rows * 4 * core::mem::size_of::<Block128>(),
            proof_bytes: honest_proof.byte_len(),
            peak_rss_kib: peak_rss_kib().unwrap_or(0),
        }
        .encode()
    );
}

fn run_sample(executable: &std::path::Path, variant: Variant, mutation_check: bool) -> Measurement {
    let mut command = Command::new(executable);
    command.arg("--sample-worker").arg(variant.argument());
    if mutation_check {
        command.arg("--mutation-check");
    }
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("start {} sample: {error}", variant.name()));
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("{} sample failed", variant.name());
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(Measurement::decode)
        .unwrap_or_else(|| panic!("{} sample returned no result", variant.name()))
}

fn cooldown(config: Config) {
    if config.cooldown_seconds != 0 {
        std::thread::sleep(Duration::from_secs(config.cooldown_seconds));
    }
}

fn sample_order(round: usize) -> [Variant; 3] {
    match round % 3 {
        0 => [Variant::Independent, Variant::Padded, Variant::Ragged],
        1 => [Variant::Ragged, Variant::Independent, Variant::Padded],
        _ => [Variant::Padded, Variant::Ragged, Variant::Independent],
    }
}

fn variant_index(variant: Variant) -> usize {
    match variant {
        Variant::Independent => 0,
        Variant::Padded => 1,
        Variant::Ragged => 2,
    }
}

fn arithmetic_backend() -> &'static str {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ))]
    {
        "AVX2 + VPCLMULQDQ paired GF(2^128) products"
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    )))]
    {
        "portable scalar paired-product fallback"
    }
}

fn print_measurements(variant: Variant, samples: &[Measurement]) {
    let first = samples[0];
    assert!(samples.iter().all(|sample| {
        sample.physical_rows == first.physical_rows
            && sample.layer_bytes == first.layer_bytes
            && sample.proof_bytes == first.proof_bytes
    }));
    let prover = Statistics::from_samples(samples.iter().map(|sample| sample.prover_ms).collect());
    let verifier =
        Statistics::from_samples(samples.iter().map(|sample| sample.verifier_ms).collect());
    let full_verifier = Statistics::from_samples(
        samples
            .iter()
            .map(|sample| sample.full_verifier_ms)
            .collect(),
    );
    let peak_rss = samples
        .iter()
        .map(|sample| sample.peak_rss_kib)
        .max()
        .unwrap_or(0);

    println!("## {}", variant.name());
    println!();
    println!("- Physical Poseidon rows: `{}`", first.physical_rows);
    println!("- Layer-S_0 bytes: `{}`", first.layer_bytes);
    println!("- Algebraic proof bytes: `{}`", first.proof_bytes);
    if peak_rss == 0 {
        println!("- Maximum process peak RSS: `unavailable on this platform`");
    } else {
        println!("- Maximum process peak RSS: `{peak_rss} KiB`");
    }
    println!();
    println!("| Operation | Median ms | p95 ms | Min ms | Max ms |");
    println!("|---|---:|---:|---:|---:|");
    print_stat("Prover", &prover);
    print_stat("Protocol verifier", &verifier);
    print_stat("Verifier + native terminal discharge", &full_verifier);
    println!();
}

fn coordinator(arguments: &[String], config: Config) {
    let executable = env::current_exe().expect("resolve current benchmark executable");
    println!("# Unequal-width Poseidon2b GKR benchmark");
    println!();
    println!("- Native Boolean widths: `{:?}`", PRODUCTION_WIDTHS);
    println!("- Native physical rows: `{}`", native_row_count());
    println!("- Max-padded physical rows: `{}`", padded_row_count());
    println!(
        "- Padding expansion: `{:.4}x`",
        padded_row_count() as f64 / native_row_count() as f64
    );
    println!("- Poseidon2b layers: `66`");
    println!("- Committed field: `GF(2^128)`");
    println!("- Sumcheck field: `GF(2^256)`");
    println!("- Arithmetic backend: `{}`", arithmetic_backend());
    println!("- Isolation: `one timed proof per worker process`");
    println!("- Warmups: `{}`", config.warmups);
    println!("- Samples: `{}`", config.samples);
    println!(
        "- Cooldown between workers: `{} s`",
        config.cooldown_seconds
    );
    println!();

    for round in 0..config.warmups {
        for variant in sample_order(round) {
            eprintln!(
                "warmup {}/{}: {}",
                round + 1,
                config.warmups,
                variant.name()
            );
            black_box(run_sample(&executable, variant, false));
            cooldown(config);
        }
    }

    let mut measurements = [Vec::new(), Vec::new(), Vec::new()];
    let mut mutation_checked = [false; 3];
    for round in 0..config.samples {
        for variant in sample_order(round) {
            eprintln!(
                "sample {}/{}: {}",
                round + 1,
                config.samples,
                variant.name()
            );
            let index = variant_index(variant);
            let measurement = run_sample(&executable, variant, !mutation_checked[index]);
            mutation_checked[index] = true;
            measurements[index].push(measurement);
            if round + 1 != config.samples || variant != sample_order(round)[2] {
                cooldown(config);
            }
        }
    }

    print_measurements(Variant::Independent, &measurements[0]);
    print_measurements(Variant::Padded, &measurements[1]);
    print_measurements(Variant::Ragged, &measurements[2]);

    if arguments.iter().any(|argument| argument == "--explain") {
        println!("The independent path emits nine transcripts. The padded and ragged paths emit");
        println!("the same one-walk proof shape; their difference is physical prover work only.");
    }
}

fn native_row_count() -> usize {
    PRODUCTION_WIDTHS.iter().map(|&width| 1usize << width).sum()
}

fn padded_row_count() -> usize {
    PRODUCTION_WIDTHS.len() * (1usize << PRODUCTION_WIDTHS.iter().max().unwrap())
}

fn main() {
    let arguments = env::args().collect::<Vec<_>>();
    let config = Config {
        warmups: parse_usize(&arguments, "--warmups", 0),
        samples: parse_usize(&arguments, "--samples", 3).max(1),
        cooldown_seconds: parse_usize(&arguments, "--cooldown-seconds", 20) as u64,
    };
    if let Some(index) = arguments
        .iter()
        .position(|argument| argument == "--sample-worker")
    {
        let variant = arguments
            .get(index + 1)
            .and_then(|value| Variant::parse(value))
            .expect("--sample-worker requires independent, padded, or ragged");
        sample_worker(
            variant,
            arguments
                .iter()
                .any(|argument| argument == "--mutation-check"),
        );
    } else {
        coordinator(&arguments, config);
    }
}
