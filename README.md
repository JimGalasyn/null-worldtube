# Null Worldtube Theory

Fundamental particles are topological vortex defects — knotted flux tubes — in a BPS condensate described by the abelian Higgs model at its self-dual point. The trefoil knot T(2,3) determines the Standard Model gauge group, coupling constants, mass spectrum, and — through its Poincaré sphere spectral geometry — the gravitational constant.

**Result:** 80 Standard Model observables at 0.1% median residual, with G predicted to 0.03%, using zero dimensionless free parameters. One mass-scale anchor (m_e). Everything else from topology.

Website: [jimgalasyn.github.io/null-worldtube](https://jimgalasyn.github.io/null-worldtube)

## Papers

| # | Title | DOI | Pages |
|---|-------|-----|-------|
| 1 | The Standard Model from a Torus Knot | [10.5281/zenodo.18891785](https://doi.org/10.5281/zenodo.18891785) | 28 |
| 2 | Three Integers and a Mass | [10.5281/zenodo.18892311](https://doi.org/10.5281/zenodo.18892311) | 15 |
| 3 | Nuclear Magic Numbers from Torus Topology | [10.5281/zenodo.19036783](https://doi.org/10.5281/zenodo.19036783) | 18 |
| 4 | From Vortex Rings to the Top Quark | [10.5281/zenodo.19051589](https://doi.org/10.5281/zenodo.19051589) | 20 |
| 5 | The Electron Mass from Phase Closure | [10.5281/zenodo.19072313](https://doi.org/10.5281/zenodo.19072313) | 14 |
| 6 | The Particle Mass Spectrum from Torus Knot Mode-Locking | [10.5281/zenodo.19225259](https://doi.org/10.5281/zenodo.19225259) | 16 |
| 7 | Charge, Leptons, and the Genus of Torus Knots | [10.5281/zenodo.19256133](https://doi.org/10.5281/zenodo.19256133) | 12 |
| 8 | The Standard Model Gauge Group from Vortex Knot Topology | [10.5281/zenodo.19334925](https://doi.org/10.5281/zenodo.19334925) | 14 |
| 9 | Coupling Constants, Mixing Matrices, and Cross Sections | [10.5281/zenodo.19335224](https://doi.org/10.5281/zenodo.19335224) | 16 |
| 10 | Vortex Strings in a Two-Component Superfluid Vacuum | [10.5281/zenodo.19516386](https://doi.org/10.5281/zenodo.19516386) | 22 |
| 11 | The Standard Model from a Gravitating Abelian Higgs Vortex | [10.5281/zenodo.19554227](https://doi.org/10.5281/zenodo.19554227) | 18 |
| 12 | Fermion Structure and Gauge Dynamics from Topological Sectors | [10.5281/zenodo.19555978](https://doi.org/10.5281/zenodo.19555978) | 14 |
| 13 | The Standard Model from NWT Topology (SM Capstone) | [10.5281/zenodo.19635239](https://doi.org/10.5281/zenodo.19635239) | 26 |
| 14 | Integers as Output | [10.5281/zenodo.19654507](https://doi.org/10.5281/zenodo.19654507) | 18 |
| 15 | One Knot, All Forces (gravitational constant) | [10.5281/zenodo.19701263](https://doi.org/10.5281/zenodo.19701263) | 14 |
| 16 | The NWT Lagrangian: A Three-Field Theory of Particles and Gravity | [10.5281/zenodo.19710846](https://doi.org/10.5281/zenodo.19710846) | 13 |
| 17 | Newton's Constant and Rest-Frame Schrödinger Evolution from K<sub>7</sub> Graph-State Information Theory | [10.5281/zenodo.19807068](https://doi.org/10.5281/zenodo.19807068) | 55 |
| 18 | Sakharov-Induced Einstein Gravity from the Null Worldtube Condensate | [10.5281/zenodo.20012352](https://doi.org/10.5281/zenodo.20012352) | 28 |

**Note on framework evolution:** Papers 1–5 used an earlier formulation (photon on a (2,1) torus with κ=3, Skilton's α formula). Papers 6–15 use the refined framework (BPS vortex on trefoil T(2,3), crossing algebra, 1/α = 25π√3+1). The later papers supersede the earlier derivations where they conflict. See the [website](https://jimgalasyn.github.io/null-worldtube) for the current summary.

## Key Results (Papers 13–15)

- **80 SM observables** at 0.1% median residual (Paper 13)
- **1/α = 25π√3 + 1 = 137.035** at 7.6 ppm (Paper 11, verified Paper 14)
- **G = (8/7)²(1+α/7)² α²¹ ℏc/m_e²** at 0.03% (Paper 15)
- **All 15 topological integers** derived from trefoil crossing algebra (Paper 14)
- **Nuclear SEMF** at 1.66% RMS, zero free parameters (Paper 16)
- **Newton's constant from Sakharov induction** matches CODATA G to **−11 ppm**, inside the ±22 ppm experimental error bar; Schwarzschild verified as a vacuum solution by symbolic computation (Paper 18)
- **Zero dimensionless free parameters** — one mass-scale anchor (m_e)

## Simulation Code

```bash
# Paper 14: integer derivation + end-to-end mass spectrum
python3 simulations/paper14_integers/integers_out.py
python3 simulations/paper14_integers/knot_selection.py
python3 simulations/paper14_integers/braid_algebra.py
python3 simulations/paper14_integers/crossings.py
python3 simulations/paper14_integers/jones.py

# Paper 15: holonomy + gravitational hierarchy
python3 simulations/level2_abelian_higgs/holonomy_from_ah.py
python3 simulations/level2_abelian_higgs/bps_equipartition.py
python3 simulations/level2_abelian_higgs/bps_necessity.py
python3 simulations/level2_abelian_higgs/crossing_angle_scan.py
python3 simulations/level2_abelian_higgs/holonomy_physical_kappa.py

# BPS profile + AB phase verification
python3 simulations/helmholtz_eigenvalue/bps_profile.py
python3 simulations/helmholtz_eigenvalue/ab_phase_verification.py

# Legacy (Papers 1–5)
python3 -m simulations.nwt --help
```

### Paper 15 reproducibility pipeline (analysis/)

The Paper 15 gravitational-hierarchy derivation is verified end-to-end by the
following scripts. Each is self-contained with numerical checks printed at run
time.

```bash
# Paper 15 b1 -- one-loop BPS pipeline (validates the Casimir machinery
# against Alonso-Izquierdo et al. 2016, Delta_mu = -0.279 in magnitude + sign)
python3 analysis/nwt_vortex_gravity_flat.py         # Stage 0: BPS profile
python3 analysis/nwt_vortex_fluctuations_b1_2.py    # 4x4 H+ operator
python3 analysis/nwt_vortex_fluctuations_b1_3.py    # FP ghost sector
python3 analysis/nwt_vortex_fluctuations_b1_4.py    # zeta-regularisation
python3 analysis/nwt_vortex_fluctuations_b1_5.py    # 2-DOF Grassmann ghost fix

# Paper 15 b2 -- Spin(7)/Cl(0,7)/2T structural chain
python3 analysis/nwt_poincare_sphere_b2_0.py        # lambda_1 = 168 on S^3/2I
python3 analysis/nwt_2T_character_7dim_b2_4.py      # 2T character table
python3 analysis/nwt_spin7_chain_b2_5.py            # Spin(7) = B_3 dims 7,8,21
python3 analysis/nwt_eulerian_amplitude_b2_7.py     # K_7 Eulerian circuit
python3 analysis/nwt_2T_spin7_clifford_b2_12.py     # Cl(0,7) embedding proof
python3 analysis/nwt_k7_so7_wilson_b2_13.py         # 21 bivectors span so(7)
python3 analysis/nwt_87_prefactor_b2_14.py          # 8/7 as Casimir ratio
python3 analysis/nwt_nlo_alpha7_b2_15.py            # (1 + alpha/7) NLO pattern

# NWT Lagrangian L1-L5 (structural decomposition of Paper 15's result
# into the minimal three-field theory L1+L2+L3; Paper 16 companion)
python3 analysis/nwt_lagrangian_L1_fields.py        # minimal field content
python3 analysis/nwt_lagrangian_L1b_uv_completion.py# SO(10) UV falsifiers
python3 analysis/nwt_lagrangian_L2_kinetic_bps.py   # Bogomolny mu = pi
python3 analysis/nwt_lagrangian_L3_skyrme_hopf.py   # Q_H = p*m quantisation
python3 analysis/nwt_lagrangian_L4_paper6_mass_spectrum.py  # 24 particles, 1.06%
python3 analysis/nwt_lagrangian_L5_gravity_hierarchy.py     # G to 0.029% NLO
```

### Paper 16 reproducibility pipeline (analysis/)

Paper 16 (*The NWT Lagrangian*) adds a six-phase one-loop Casimir
programme on the Poincaré sphere S³/2I with the BPS trefoil background.
Phases 0–5 are implemented and validated; Phase 6 (Wilson amplitude on
K₇) is deferred to a follow-up paper.

```bash
# Phase 0: heat-kernel scaffold, matches Seeley-DeWitt to 6 decimals
python3 analysis/nwt_zeta_phase0_scaffold.py

# Phase 1: free scalar zeta(s) via Jorgenson-Lang on S^3/2I
python3 analysis/nwt_zeta_phase1_free_scalar.py

# Phase 2: BPS trefoil geometry on the Clifford/Heegaard torus of S^3
python3 analysis/nwt_zeta_phase2_trefoil_bps.py

# Phase 3: tubular Casimir shift (bulk + finite-size)
python3 analysis/nwt_zeta_phase3_trefoil_casimir.py

# Phase 4: curvature corrections + 2I-orbit scheme analysis
python3 analysis/nwt_zeta_phase4_curvature_corrections.py

# Phase 5: extract 1/G from S_eff, localise the alpha^-21 suppression
python3 analysis/nwt_zeta_phase5_1overG.py
```

### Paper 17 reproducibility pipeline (analysis/)

Paper 17 (*Newton's Constant and Rest-Frame Schrödinger Evolution from
K_7 Graph-State Information Theory*) gives a quantum-information-theoretic
derivation of the bracket coefficients in the Paper 16 m_e/m_Pl
identity. The supporting numerics include K_N graph-state moment
identities, bracket-truncation probes, and IBM Heron R2 hardware
experiments (eight datasets across three backends).

```bash
# K_N graph-state moment identities (so(2n+1) family at N=7,9,11)
python3 analysis/nwt_qec_bracket_test.py
python3 analysis/nwt_qec_KN_generalization.py

# Bracket-truncation probes
python3 analysis/nwt_truncation_mechanism.py
python3 analysis/nwt_truncation_qdef.py
python3 analysis/nwt_truncation_channel_mixing.py

# IBM Heron R2 hardware submission and analysis (requires qiskit-ibm-runtime)
python3 analysis/nwt_qec_heron_experiment.py
python3 analysis/nwt_qec_heron_KN.py
python3 analysis/nwt_qec_heron_exp4.py
python3 analysis/nwt_qec_heron_exp5.py
python3 analysis/nwt_qec_heron_zne.py
python3 analysis/nwt_qec_heron_fetch.py

# PSL(2,7) edge-transitivity re-analysis (no QPU)
python3 analysis/nwt_qec_psl27_edge_transitivity.py

# Forward-prediction and zero-noise extrapolation
python3 analysis/nwt_qec_forward_prediction.py
python3 analysis/nwt_qec_zne_continuation.py
python3 analysis/nwt_qec_zne_ratio.py
python3 analysis/nwt_qec_zne_reanalysis.py

# Schrödinger derivation supports (Bremermann + b2.13 + PSL(2,7))
python3 analysis/nwt_qec_bit_quantum_from_bremermann.py
python3 analysis/nwt_qec_bps_compton_bridge.py
python3 analysis/nwt_qec_proportionality_constant.py
python3 analysis/nwt_qec_route_a_so7_lift.py
python3 analysis/nwt_qec_syndrome_attractor.py
python3 analysis/nwt_qec_time_evolution.py
python3 analysis/nwt_qec_entanglement_structure.py
python3 analysis/nwt_qec_interpretation_b_test.py

# Information-theoretic bookkeeping (β-decay Landauer floor + hyperon survey)
python3 analysis/nwt_beta_decay_landauer.py
python3 analysis/nwt_hyperon_landauer_survey.py

# Volovik direction (parked future-work, c emergence on the discrete medium)
python3 analysis/nwt_emergent_c.py
python3 analysis/nwt_volovik_c.py
python3 analysis/nwt_volovik_bogoliubov.py
python3 analysis/nwt_volovik_part_b.py
python3 analysis/nwt_volovik_two_mode.py
python3 analysis/nwt_volovik_closure.py
```

Raw IBM Heron R2 job outputs (2026-04-26, 8 datasets across
ibm_kingston / ibm_marrakesh / ibm_fez) are in
`analysis/heron_results/2026-04-26_*.txt`.

### Paper 18 reproducibility pipeline (analysis/)

Paper 18 (*Sakharov-Induced Einstein Gravity from the Null Worldtube
Condensate*) derives Einstein's equations as the long-wavelength
matter-loop dynamics of the NWT three-field condensate, in the manner
of Sakharov's 1968 induced-gravity mechanism.  The end-to-end
derivation is supplied as six step-by-step sympy reproduction
scripts:

```bash
# Paper 18 G1-G6 -- Sakharov-derivation step-by-step
python3 analysis/nwt_paper18_G1_setup.py             # matter action, T_munu, graviton coupling
python3 analysis/nwt_paper18_G2_matter_loop.py       # heat-kernel expansion on flat background, 1/G
python3 analysis/nwt_paper18_G3_K7_sakharov.py       # K_7 amplitude UV/IR bridge
python3 analysis/nwt_paper18_G4_linearized_einstein.py  # linearised Einstein, GW
python3 analysis/nwt_paper18_G5_curved_sakharov.py   # curved-background EH extraction
python3 analysis/nwt_paper18_G6_einstein_variation.py   # full nonlinear Einstein, Schwarzschild
```

The validated end-result is also exposed via the `nwt-substrate`
companion library (separate repo:
[github.com/JimGalasyn/nwt-substrate](https://github.com/JimGalasyn/nwt-substrate),
Zenodo [10.5281/zenodo.20012027](https://doi.org/10.5281/zenodo.20012027)
concept DOI):

```python
>>> from nwt_substrate.gravity import G_substrate_SI, verify_schwarzschild_vacuum_symbolic
>>> G_substrate_SI()                              # 6.674228e-11 m^3 kg^-1 s^-2  (−11 ppm CODATA)
>>> verify_schwarzschild_vacuum_symbolic()        # all 10 R_munu components vanish identically
```

The Paper 18 §4.4 figure (trefoil + K_7 on the same Heegaard torus,
the matter-gravity unification visualisation) is generated by
`papers/figures/paper18_torus_unification.py`.

## Repository Structure

```
null-worldtube/
├── papers/                             # All 18 papers (LaTeX + PDF)
├── simulations/
│   ├── paper14_integers/               # Crossing algebra, braid, Jones, end-to-end
│   ├── level2_abelian_higgs/           # JAX energy, holonomy, BPS, spectrum
│   ├── helmholtz_eigenvalue/           # BPS profile, eigenvalues, AB verification
│   ├── nwt/                            # Legacy analysis package (Papers 1-5)
│   └── output/                         # Generated data
├── analysis/                           # Analysis scripts (Papers 6-18)
│   └── heron_results/                  # IBM Heron R2 raw job outputs (Paper 17)
├── docs/                               # GitHub Pages website source
└── README.md
```

## Authors

James P. Galasyn (independent researcher) and Claude Théodore (Anthropic)

## License

MIT
