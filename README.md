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
| 15 | One Knot, All Forces (gravitational constant) | in preparation | 9 |
| 16 | Nuclear Binding from Topological Linking | in preparation | 31 |

**Note on framework evolution:** Papers 1–5 used an earlier formulation (photon on a (2,1) torus with κ=3, Skilton's α formula). Papers 6–15 use the refined framework (BPS vortex on trefoil T(2,3), crossing algebra, 1/α = 25π√3+1). The later papers supersede the earlier derivations where they conflict. See the [website](https://jimgalasyn.github.io/null-worldtube) for the current summary.

## Key Results (Papers 13–15)

- **80 SM observables** at 0.1% median residual (Paper 13)
- **1/α = 25π√3 + 1 = 137.035** at 7.6 ppm (Paper 11, verified Paper 14)
- **G = (8/7)²(1+α/7)² α²¹ ℏc/m_e²** at 0.03% (Paper 15)
- **All 15 topological integers** derived from trefoil crossing algebra (Paper 14)
- **Nuclear SEMF** at 1.66% RMS, zero free parameters (Paper 16)
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

## Repository Structure

```
null-worldtube/
├── papers/                             # All 16 papers (LaTeX + PDF)
├── simulations/
│   ├── paper14_integers/               # Crossing algebra, braid, Jones, end-to-end
│   ├── level2_abelian_higgs/           # JAX energy, holonomy, BPS, spectrum
│   ├── helmholtz_eigenvalue/           # BPS profile, eigenvalues, AB verification
│   ├── nwt/                            # Legacy analysis package (Papers 1-5)
│   └── output/                         # Generated data
├── analysis/                           # Analysis scripts (Papers 6-16)
├── docs/                               # GitHub Pages website source
└── README.md
```

## Authors

James P. Galasyn (independent researcher) and Claude Théodore (Anthropic)

## License

MIT
