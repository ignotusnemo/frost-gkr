# Security

FROST-GKR is a research artifact, not a standalone production proof system.
The repository implements and benchmarks the algebraic reduction described in
the paper. A deployed argument must additionally bind the terminal MLE claims
with a polynomial commitment scheme and account for that scheme, the
Fiat–Shamir transform, and the hash assumptions in its end-to-end analysis.

Please report suspected vulnerabilities privately to
<ignotus.nemo@proton.me>. Include a minimal reproducer when possible. Do not
open a public issue for an unpatched vulnerability.
