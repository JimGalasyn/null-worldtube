# Heron Exp 12 — Phase 1 Simulator Pilot — Calibration Report

Shots per (circuit, backend): 1000
Total runtime: 122.2s = 2.0 min
Backends: noiseless, marrakesh, fez

## Per-class modal probability under realistic Heron noise

| (p, q) | Particles | Predicted | Modal prob (noiseless) | Modal prob (marrakesh) | Modal prob (fez) | Predicted-bin prob (marrakesh) | Predicted-bin prob (fez) | Match across backends |
|---|---|---|---|---|---|---|---|---|
| (1, 3) | p, n, Sigma+, Sigma0, Sigma- | (I,I), logZ=-1 | 1.000 | 0.536 | 0.580 | 0.536 | 0.580 | 3/3 |
| (1, 8) | mu- | (P,F3), logZ=+1 | 1.000 | 0.667 | 0.549 | 0.667 | 0.549 | 3/3 |
| (2, 1) | e- | (F2,E2), logZ=-1 | 1.000 | 0.660 | 0.551 | 0.660 | 0.551 | 3/3 |
| (2, 5) | K+ | (P,E1), logZ=+1 | 1.000 | 0.584 | 0.556 | 0.584 | 0.556 | 3/3 |
| (2, 7) | D+, J/psi | (P,F3), logZ=-1 | 1.000 | 0.633 | 0.588 | 0.633 | 0.588 | 3/3 |
| (3, 4) | tau-, Lambda, Sigma* | (E2,E2), logZ=+1 | 1.000 | 0.683 | 0.544 | 0.683 | 0.544 | 3/3 |
| (3, 5) | pi+ | (E1,E1), logZ=+1 | 1.000 | 0.685 | 0.530 | 0.685 | 0.530 | 3/3 |
| (3, 7) | D0 | (I,I), logZ=-1 | 1.000 | 0.594 | 0.593 | 0.594 | 0.593 | 3/3 |
| (4, 5) | omega | (F3,F1), logZ=-1 | 1.000 | 0.600 | 0.564 | 0.600 | 0.564 | 3/3 |
| (4, 9) | Upsilon | (P,E1), logZ=+1 | 1.000 | 0.640 | 0.557 | 0.640 | 0.557 | 3/3 |
| (5, 4) | Xi0, Xi-, Delta | (F2,F3), logZ=-1 | 1.000 | 0.623 | 0.512 | 0.623 | 0.512 | 3/3 |
| (5, 7) | rho | (E3,P), logZ=+1 | 1.000 | 0.628 | 0.587 | 0.628 | 0.587 | 3/3 |
| (6, 5) | eta | (E3,P), logZ=+1 | 1.000 | 0.698 | 0.545 | 0.698 | 0.545 | 3/3 |
| (7, 3) | pi0 | (F2,E2), logZ=-1 | 1.000 | 0.560 | 0.527 | 0.560 | 0.527 | 3/3 |
| (7, 4) | Omega- | (P,E3), logZ=+1 | 1.000 | 0.573 | 0.580 | 0.573 | 0.580 | 3/3 |
| (7, 5) | K0 | (E1,E1), logZ=+1 | 1.000 | 0.616 | 0.565 | 0.616 | 0.565 | 3/3 |

## Threshold recommendation for v4 pre-reg §5.3

- Mean modal prob across 16 classes (marrakesh noise): **0.624** (range 0.536-0.698)
- Mean modal prob across 16 classes (fez noise): **0.558** (range 0.512-0.593)
- Suggested PASS threshold (10th percentile of empirical modal probs, rounded to nearest 0.05): see report data