# Heron Exp 12 — Phase 1 Simulator Pilot — Calibration Report

**Date**: 2026-05-20
**Shots per (circuit, backend)**: 100,000
**Backends**: noiseless, marrakesh_noise, fez_noise, marrakesh_topology, fez_topology
**Status**: completed successfully (rerun on 2026-05-20 ~7:25pm VV-time)

## Per-class substrate-prediction validation

| (p, q) | Particles | Predicted (s_X, s_Z, log_Z) | noiseless | mar_noise | fez_noise | mar_topo | fez_topo | match |
|---|---|---|---|---|---|---|---|---|
| (1, 3) | p, n, Sigma+, Sigma0, Sigma- | (I,I), -1 | 1.000 | 0.926 | 0.879 | 0.685 | 0.875 | 5/5 |
| (1, 8) | mu- | (P,F3), +1 | 1.000 | 0.925 | 0.880 | 0.687 | 0.876 | 5/5 |
| (2, 1) | e- | (F2,E2), -1 | 1.000 | 0.928 | 0.883 | 0.689 | 0.879 | 5/5 |
| (2, 5) | K+ | (P,E1), +1 | 1.000 | 0.926 | 0.881 | 0.690 | 0.876 | 5/5 |
| (2, 7) | D+, J/psi | (P,F3), -1 | 1.000 | 0.927 | 0.884 | 0.689 | 0.878 | 5/5 |
| (3, 4) | tau-, Lambda, Sigma* | (E2,E2), +1 | 1.000 | 0.927 | 0.882 | 0.687 | 0.878 | 5/5 |
| (3, 5) | pi+ | (E1,E1), +1 | 1.000 | 0.927 | 0.883 | 0.688 | 0.877 | 5/5 |
| (3, 7) | D0 | (I,I), -1 | 1.000 | 0.927 | 0.880 | 0.687 | 0.875 | 5/5 |
| (4, 5) | omega | (F3,F1), -1 | 1.000 | 0.927 | 0.883 | 0.687 | 0.878 | 5/5 |
| (4, 9) | Upsilon | (P,E1), +1 | 1.000 | 0.928 | 0.881 | 0.687 | 0.877 | 5/5 |
| (5, 4) | Xi0, Xi-, Delta | (F2,F3), -1 | 1.000 | 0.928 | 0.884 | 0.685 | 0.878 | 5/5 |
| (5, 7) | rho | (E3,P), +1 | 1.000 | 0.926 | 0.881 | 0.686 | 0.877 | 5/5 |
| (6, 5) | eta | (E3,P), +1 | 1.000 | 0.927 | 0.880 | 0.686 | 0.876 | 5/5 |
| (7, 3) | pi0 | (F2,E2), -1 | 1.000 | 0.927 | 0.883 | 0.688 | 0.877 | 5/5 |
| (7, 4) | Omega- | (P,E3), +1 | 1.000 | 0.927 | 0.883 | 0.686 | 0.878 | 5/5 |
| (7, 5) | K0 | (E1,E1), +1 | 1.000 | 0.928 | 0.883 | 0.689 | 0.877 | 5/5 |

## Control circuits

| Control | Expected | noiseless | mar_noise | fez_noise | mar_topo | fez_topo |
|---|---|---|---|---|---|---|
| identity | (I,I), +1 | 1.000 | 0.928 | 0.884 | 0.688 | 0.879 |
| x7 | (I,I), -1 | 1.000 | 0.926 | 0.879 | 0.688 | 0.875 |
| z7 | (I,I), +1 | 1.000 | 0.927 | 0.883 | 0.691 | 0.877 |
| h7 | (I,I), +0 | 1.000 | 0.946 | 0.906 | 0.702 | 0.901 |

## Modal-probability statistics by backend (16 classes)

| Backend | Mean | Min | 10th pctile | Max |
|---|---|---|---|---|
| noiseless | 1.000 | 1.000 | 1.000 | 1.000 |
| marrakesh_noise | 0.927 | 0.925 | 0.926 | 0.928 |
| fez_noise | 0.882 | 0.879 | 0.880 | 0.884 |
| marrakesh_topology | 0.687 | 0.685 | 0.685 | 0.690 |
| fez_topology | 0.877 | 0.875 | 0.875 | 0.879 |

## Match summary

- **noiseless**: 16/16 classes with predicted (s_X, s_Z, log_Z) as modal outcome
- **marrakesh_noise**: 16/16 classes with predicted (s_X, s_Z, log_Z) as modal outcome
- **fez_noise**: 16/16 classes with predicted (s_X, s_Z, log_Z) as modal outcome
- **marrakesh_topology**: 16/16 classes with predicted (s_X, s_Z, log_Z) as modal outcome
- **fez_topology**: 16/16 classes with predicted (s_X, s_Z, log_Z) as modal outcome

## v4 threshold recommendation (Phase 2)

- Empirical min modal prob (worst class × worst topology backend): **0.685**
- Empirical 10th-percentile modal prob (across both topology backends): **0.685**
- **Suggested PASS threshold for v4: 0.65** (10th pctile, rounded down to nearest 0.05)
- This is well above the 1/64 ≈ 0.016 uniform-noise floor, leaving clear separation from H_null.