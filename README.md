# Null Worldtube Theory

An electron is a photon confined to a (2,1) torus knot. The worldtube of this circulation is a closed null surface in Minkowski spacetime. Using only standard Maxwell electrodynamics on toroidal topology, the framework derives 23 Standard Model parameters from one measured mass (m_e = 0.511 MeV) and three topological integers (p=2, q=1, k=3).

## Papers

| # | Title | File |
|---|-------|------|
| 1 | The Standard Model from a Torus Knot | `papers/paper1_lepton_spectrum` |
| 1S | Supplementary Material | `papers/paper1_supplementary` |
| 2 | The Torus Electron | (published separately) |
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
python3 simulations/null_worldtube.py --help      # all available analyses
python3 simulations/null_worldtube.py --koide      # lepton Koide analysis
python3 simulations/null_worldtube.py --quarks     # quark masses + EW/CKM/PMNS
python3 simulations/null_worldtube.py --skilton    # α derivation from integers
python3 simulations/null_worldtube.py --junction   # junction conditions
```

Requires: `numpy`, `matplotlib`, `scipy`

## Repository Structure

```
null-worldtube/
├── papers/                        # LaTeX sources and PDFs
├── simulations/
│   ├── null_worldtube.py          # Main simulation (all analyses)
│   ├── nonlinear_permittivity.py  # Born-Infeld / nonlinear permittivity
│   ├── plot_settling_spectrum.py  # Settling spectrum visualization
│   └── archive/                   # Earlier modular code
└── README.md
```

## Authors

James P. Galasyn and Claude Théodore
