# Null Worldtube Theory

An electron is a photon confined to a (2,1) torus knot. The worldtube of this circulation is a closed null surface in Minkowski spacetime. Using only standard Maxwell electrodynamics on toroidal topology, the framework derives 23 Standard Model parameters from one measured mass (m_e = 0.511 MeV) and three topological integers (p=2, q=1, k=3).

## Papers

| # | Title | File |
|---|-------|------|
| 1 | The Standard Model from a Torus Knot | `papers/paper1_lepton_spectrum` |
| 1S | Supplementary Material | `papers/paper1_supplementary` |
| 2 | The Torus Electron | `papers/paper2_torus_electron` |
| 2A | Annales de la Fondation Louis de Broglie submission | `papers/paper2_annales` |
| 2S | Supplementary Material | `papers/paper2_supplementary` |
| 3 | Three Integers and a Mass: Deriving the Standard Model Input Set | `papers/paper3_three_integers` |
| — | On Human–AI Collaboration in Theoretical Physics | `papers/paper_collaboration` |

Each paper has `.md`, `.tex`, and `.pdf` versions.

## Key Results

**Inputs**: m_e, p=2 (toroidal winding), q=1 (poloidal winding), k=R/r=3 (torus aspect ratio)

**Derived quantities** (23 predictions):
- Lepton masses via Koide angle θ_K = (6π+2)/9
- Quark masses via anchor formulas (m_t = (p/k)Λ, m_b = √5·α·Λ, m_τ = (20/21)α·Λ)
- Weinberg angle sin²θ_W = 3/13
- Strong coupling α_s = 16α
- Higgs mass m_H = (1/2 − α)Λ
- Full CKM and PMNS matrices
- Neutrino masses and mass ordering
- Fine-structure constant α⁻¹ = √(137² + π²) via Skilton's formula

Median error across all 23 predictions: ~1%.

## Simulation

```bash
cd simulations
python3 -m nwt --help              # all available analyses
python3 -m nwt --koide             # lepton Koide analysis
python3 -m nwt --quarks            # quark masses + EW/CKM/PMNS
python3 -m nwt --skilton           # α derivation from integers
python3 -m nwt --junction          # junction conditions
python3 -m nwt --pair-creation     # pair creation field analysis
python3 -m nwt --thevenin          # Thevenin impedance matching
python3 -m nwt --settling-spectrum # Kelvin wave soft photon spectrum
```

The backward-compat entry point `python3 null_worldtube.py --flag` also works.

Requires: `numpy`. Optional: `scipy` (decay dynamics, neutrino), `matplotlib` (figure generation).

## Repository Structure

```
null-worldtube/
├── papers/                            # LaTeX sources and PDFs
├── simulations/
│   ├── nwt/                           # Main analysis package
│   │   ├── __main__.py                # CLI parsing + lazy-import dispatch
│   │   ├── constants.py               # Physical constants, TorusParams
│   │   ├── core.py                    # Torus geometry, self-energy, resonance
│   │   ├── fields.py                  # Casimir, greybody, torus EM fields, LC circuit
│   │   ├── particles.py               # Basic analysis, find-radii, pair-production
│   │   ├── hydrogen.py                # H/He/Li orbital dynamics
│   │   ├── transitions.py             # Photon emission/absorption
│   │   ├── hadrons.py                 # Quarks, mesons, baryons, proton mass
│   │   ├── orbit.py                   # Kerr-Newman, Einstein, junction conditions
│   │   ├── pair_creation.py           # Euler-Heisenberg, Schwinger, gradient
│   │   ├── settling.py                # Transmission line, Kelvin waves, Thevenin
│   │   └── analyses.py                # Standalone analyses (Skilton, Koide, ...)
│   ├── null_worldtube.py              # Backward-compat shim → nwt.main()
│   ├── nonlinear_permittivity.py      # Born-Infeld / nonlinear permittivity
│   ├── plot_settling_spectrum.py      # Settling spectrum visualization
│   └── archive/                       # Earlier modular code
└── README.md
```

## Authors

James P. Galasyn and Claude Théodore
