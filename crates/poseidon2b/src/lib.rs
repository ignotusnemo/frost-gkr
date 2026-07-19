// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ignotus Nemo. All rights reserved.

//! Poseidon2b over GF(2^128): native permutation and Fiat--Shamir channel for
//! the FROST-GKR comparison artifact.

pub mod channel;
pub mod native;

pub use channel::Poseidon2bChannel;
pub use native::*;
