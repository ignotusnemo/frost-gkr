// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo.

use std::collections::BTreeSet;
use std::env;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use frost_gkr::{
    discharge_frost_native, discharge_legacy_native, prove_frost, prove_legacy, sequence_digest,
    verify_frost, verify_legacy, SequenceInput,
};
use frost_gkr_core::Block128;
use frost_gkr_poseidon2b::channel::Poseidon2bChannel;

const ARTIFACT_URL: &str = "https://github.com/ignotusnemo/frost-gkr";
const LEGACY_CONSTRAINT_ROUNDS: usize = 4_248;
const FROST_CONSTRAINT_ROUNDS: usize = 30;
const LEGACY_TOTAL_SUMCHECK_ROUNDS: usize = 4_263;
const FROST_TOTAL_SUMCHECK_ROUNDS: usize = 75;
const LEGACY_EXPECTED_BYTES: usize = 287_712;
const FROST_EXPECTED_BYTES: usize = 5_568;

#[derive(Clone, Copy)]
struct Config {
    warmups: usize,
    samples: usize,
}

#[derive(Clone)]
struct Statistics {
    median_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    stddev_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn parse_config() -> Config {
    let mut config = Config {
        warmups: 3,
        samples: 20,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmups" => {
                config.warmups = args
                    .next()
                    .expect("--warmups requires an integer")
                    .parse()
                    .expect("invalid --warmups value");
            }
            "--samples" => {
                config.samples = args
                    .next()
                    .expect("--samples requires an integer")
                    .parse()
                    .expect("invalid --samples value");
            }
            "-h" | "--help" => {
                println!("Usage: frost-gkr-bench [--warmups N] [--samples N]");
                std::process::exit(0);
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    assert!(config.samples > 0, "--samples must be positive");
    config
}

fn fixture_input() -> SequenceInput {
    SequenceInput {
        initial_state: [
            Block128::from(1u128),
            Block128::from(2u128),
            Block128::from(3u128),
            Block128::from(4u128),
        ],
    }
}

fn timed<F, T>(f: &mut F) -> Duration
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    let output = black_box(f());
    let elapsed = start.elapsed();
    black_box(&output);
    drop(output);
    elapsed
}

fn measure_single<F, T>(warmups: usize, samples: usize, mut f: F) -> Statistics
where
    F: FnMut() -> T,
{
    for _ in 0..warmups {
        let _ = timed(&mut f);
    }
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        durations.push(timed(&mut f));
    }
    statistics(&durations)
}

fn measure_pair<FL, TL, FK, TK>(
    warmups: usize,
    samples: usize,
    mut legacy: FL,
    mut frost: FK,
) -> (Statistics, Statistics)
where
    FL: FnMut() -> TL,
    FK: FnMut() -> TK,
{
    for i in 0..warmups {
        if i % 2 == 0 {
            let _ = timed(&mut legacy);
            let _ = timed(&mut frost);
        } else {
            let _ = timed(&mut frost);
            let _ = timed(&mut legacy);
        }
    }

    let mut legacy_durations = Vec::with_capacity(samples);
    let mut frost_durations = Vec::with_capacity(samples);
    for i in 0..samples {
        if i % 2 == 0 {
            legacy_durations.push(timed(&mut legacy));
            frost_durations.push(timed(&mut frost));
        } else {
            frost_durations.push(timed(&mut frost));
            legacy_durations.push(timed(&mut legacy));
        }
    }

    (statistics(&legacy_durations), statistics(&frost_durations))
}

fn statistics(durations: &[Duration]) -> Statistics {
    let mut values: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect();
    values.sort_by(f64::total_cmp);

    let len = values.len();
    let median_ms = if len.is_multiple_of(2) {
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    } else {
        values[len / 2]
    };
    let p95_index = ((len as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(len - 1);
    let mean_ms = values.iter().sum::<f64>() / len as f64;
    let variance = if len > 1 {
        values
            .iter()
            .map(|value| {
                let delta = value - mean_ms;
                delta * delta
            })
            .sum::<f64>()
            / (len - 1) as f64
    } else {
        0.0
    };

    Statistics {
        median_ms,
        p95_ms: values[p95_index],
        mean_ms,
        stddev_ms: variance.sqrt(),
        min_ms: values[0],
        max_ms: values[len - 1],
    }
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn cpu_info_value(key: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let (candidate, value) = line.split_once(':')?;
                (candidate.trim() == key).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn enabled_isa() -> String {
    let flags = cpu_info_value("flags");
    ["pclmulqdq", "avx2", "avx512f", "gfni", "vpclmulqdq"]
        .into_iter()
        .filter(|feature| flags.split_whitespace().any(|flag| flag == *feature))
        .collect::<Vec<_>>()
        .join(", ")
}

fn physical_core_count() -> String {
    let cores = command_output("lscpu", &["--parse=CORE"])
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<BTreeSet<_>>()
        .len();
    if cores == 0 {
        "unavailable".to_owned()
    } else {
        cores.to_string()
    }
}

fn read_system_value(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|_| "unavailable".to_owned())
}

fn print_timing_row(label: &str, statistics: &Statistics) {
    println!(
        "| {label} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
        statistics.median_ms,
        statistics.p95_ms,
        statistics.mean_ms,
        statistics.stddev_ms,
        statistics.min_ms,
        statistics.max_ms,
    );
}

fn main() {
    let config = parse_config();

    let input = fixture_input();
    let claimed = sequence_digest(&input);

    // Correctness gate before measurement.
    let mut legacy_prover_channel = Poseidon2bChannel::new();
    let (legacy_proof, legacy_prover_reduction) =
        prove_legacy(&input, claimed, &mut legacy_prover_channel);
    let mut legacy_verifier_channel = Poseidon2bChannel::new();
    let legacy_verifier_reduction =
        verify_legacy(&legacy_proof, &input, claimed, &mut legacy_verifier_channel)
            .expect("legacy verifier rejected its honest proof");
    assert_eq!(legacy_prover_reduction, legacy_verifier_reduction);
    assert!(discharge_legacy_native(&input, &legacy_verifier_reduction));

    let mut frost_prover_channel = Poseidon2bChannel::new();
    let (frost_proof, frost_prover_reductions) =
        prove_frost(&input, claimed, &mut frost_prover_channel);
    let mut frost_verifier_channel = Poseidon2bChannel::new();
    let frost_verifier_reductions =
        verify_frost(&frost_proof, &input, claimed, &mut frost_verifier_channel)
            .expect("FROST-GKR verifier rejected its honest proof");
    assert_eq!(frost_prover_reductions, frost_verifier_reductions);
    assert!(discharge_frost_native(&input, &frost_verifier_reductions));

    let legacy_bytes = legacy_proof.byte_len();
    let frost_bytes = frost_proof.byte_len();
    assert_eq!(legacy_bytes, LEGACY_EXPECTED_BYTES);
    assert_eq!(frost_bytes, FROST_EXPECTED_BYTES);

    let shared_statement = measure_single(config.warmups, config.samples, || {
        sequence_digest(black_box(&input))
    });

    let (legacy_prover, frost_prover) = measure_pair(
        config.warmups,
        config.samples,
        || {
            let mut channel = Poseidon2bChannel::new();
            let (proof, reduction) = prove_legacy(&input, claimed, &mut channel);
            (proof.byte_len(), reduction.point.len())
        },
        || {
            let mut channel = Poseidon2bChannel::new();
            let (proof, reductions) = prove_frost(&input, claimed, &mut channel);
            (
                proof.byte_len(),
                reductions.state.point.len()
                    + reductions.sin.point.len()
                    + reductions.sout.point.len(),
            )
        },
    );

    let (legacy_verifier, frost_verifier) = measure_pair(
        config.warmups,
        config.samples,
        || {
            let mut channel = Poseidon2bChannel::new();
            verify_legacy(black_box(&legacy_proof), &input, claimed, &mut channel)
                .expect("legacy verifier failed during timing")
        },
        || {
            let mut channel = Poseidon2bChannel::new();
            verify_frost(black_box(&frost_proof), &input, claimed, &mut channel)
                .expect("FROST-GKR verifier failed during timing")
        },
    );

    let (legacy_discharge, frost_discharge) = measure_pair(
        config.warmups,
        config.samples,
        || {
            assert!(discharge_legacy_native(
                &input,
                black_box(&legacy_verifier_reduction)
            ));
        },
        || {
            assert!(discharge_frost_native(
                &input,
                black_box(&frost_verifier_reductions)
            ));
        },
    );

    let (legacy_full_verifier, frost_full_verifier) = measure_pair(
        config.warmups,
        config.samples,
        || {
            let mut channel = Poseidon2bChannel::new();
            let reduction = verify_legacy(black_box(&legacy_proof), &input, claimed, &mut channel)
                .expect("legacy verifier failed during combined timing");
            assert!(discharge_legacy_native(&input, &reduction));
        },
        || {
            let mut channel = Poseidon2bChannel::new();
            let reductions = verify_frost(black_box(&frost_proof), &input, claimed, &mut channel)
                .expect("FROST-GKR verifier failed during combined timing");
            assert!(discharge_frost_native(&input, &reductions));
        },
    );

    let generated = command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]);
    let logical_threads = std::thread::available_parallelism()
        .map(|count| count.get().to_string())
        .unwrap_or_else(|_| "unavailable".to_owned());
    let rayon_threads = env::var("RAYON_NUM_THREADS")
        .unwrap_or_else(|_| format!("default ({logical_threads} logical threads available)"));
    let rustflags = env::var("RUSTFLAGS").unwrap_or_else(|_| "not set".to_owned());

    println!("# FROST-GKR benchmark result");
    println!();
    println!("- Generated (UTC): `{generated}`");
    println!("- Artifact: <{ARTIFACT_URL}>");
    println!("- Profile: `cargo run --release --locked`");
    println!("- Warmups: `{}` per operation", config.warmups);
    println!("- Samples: `{}` per operation", config.samples);
    println!("- CPU: `{}`", cpu_info_value("model name"));
    println!(
        "- CPU topology: `{}` physical cores, `{logical_threads}` logical threads",
        physical_core_count()
    );
    println!("- Relevant ISA: `{}`", enabled_isa());
    println!(
        "- CPU governor: `{}`",
        read_system_value("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    );
    let no_turbo = read_system_value("/sys/devices/system/cpu/intel_pstate/no_turbo");
    let turbo = match no_turbo.as_str() {
        "0" => "enabled",
        "1" => "disabled",
        _ => "unavailable",
    };
    println!("- Turbo: `{turbo}`");
    println!("- Rayon threads: `{rayon_threads}`");
    println!("- OS/kernel: `{}`", command_output("uname", &["-srmo"]));
    println!("- Rust: `{}`", command_output("rustc", &["--version"]));
    println!("- Cargo: `{}`", command_output("cargo", &["--version"]));
    println!("- Repository rustflags: `-C target-cpu=native`");
    println!("- Additional `RUSTFLAGS`: `{rustflags}`");
    println!();
    println!("## Timings");
    println!();
    println!("All values are milliseconds. p95 uses the nearest-rank estimator; standard deviation is the sample standard deviation.");
    println!();
    println!("| Operation | Median | p95 | Mean | Std. dev. | Min | Max |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    print_timing_row("Shared sequence evaluation", &shared_statement);
    print_timing_row("Legacy prover", &legacy_prover);
    print_timing_row("FROST-GKR prover", &frost_prover);
    print_timing_row("Legacy protocol verifier", &legacy_verifier);
    print_timing_row("FROST-GKR protocol verifier", &frost_verifier);
    print_timing_row("Legacy native terminal discharge", &legacy_discharge);
    print_timing_row("FROST-GKR native terminal discharge", &frost_discharge);
    print_timing_row("Legacy verifier + native discharge", &legacy_full_verifier);
    print_timing_row(
        "FROST-GKR verifier + native discharge",
        &frost_full_verifier,
    );
    println!();
    println!(
        "Median prover speedup: `{:.2}x`",
        legacy_prover.median_ms / frost_prover.median_ms
    );
    println!(
        "Median protocol-verifier speedup: `{:.2}x`",
        legacy_verifier.median_ms / frost_verifier.median_ms
    );
    println!(
        "Median verifier-plus-discharge speedup: `{:.2}x`",
        legacy_full_verifier.median_ms / frost_full_verifier.median_ms
    );
    println!();
    println!("## Algebraic proof accounting");
    println!();
    println!("| Metric | Legacy product-chain GKR | FROST-GKR | Reduction |");
    println!("|---|---:|---:|---:|");
    println!("| Constraint sumcheck rounds | {LEGACY_CONSTRAINT_ROUNDS} | {FROST_CONSTRAINT_ROUNDS} | {:.2}x |", LEGACY_CONSTRAINT_ROUNDS as f64 / FROST_CONSTRAINT_ROUNDS as f64);
    println!("| Total sumcheck rounds, including terminal batching | {LEGACY_TOTAL_SUMCHECK_ROUNDS} | {FROST_TOTAL_SUMCHECK_ROUNDS} | {:.2}x |", LEGACY_TOTAL_SUMCHECK_ROUNDS as f64 / FROST_TOTAL_SUMCHECK_ROUNDS as f64);
    println!(
        "| Raw algebraic proof bytes | {legacy_bytes} | {frost_bytes} | {:.2}x |",
        legacy_bytes as f64 / frost_bytes as f64
    );
    println!(
        "| Raw algebraic proof KiB | {:.5} | {:.5} | {:.2}x |",
        legacy_bytes as f64 / 1024.0,
        frost_bytes as f64 / 1024.0,
        legacy_bytes as f64 / frost_bytes as f64
    );
    println!();
    println!("The proof-byte rows count raw field elements in the algebraic proof objects. They exclude serialization framing and external polynomial-commitment openings.");
}
