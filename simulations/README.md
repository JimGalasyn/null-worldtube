# Simulations

Null Worldtube Theory: simulation and analysis code.

## Main Code

```
simulations/
├── null_worldtube.py           # Main NWT simulation (~19,500 lines, all analyses)
├── nonlinear_permittivity.py   # Nonlinear permittivity / Born-Infeld analysis
├── plot_settling_spectrum.py   # Settling spectrum visualization
├── output/                     # Generated output files
└── archive/                    # Earlier modular code (nwt/, koide/, williamson/, FDTD)
```

## Usage

```bash
# Core analyses referenced in Paper 1
python3 simulations/null_worldtube.py --koide          # Lepton Koide analysis
python3 simulations/null_worldtube.py --quarks         # Quark masses + EW/CKM/PMNS
python3 simulations/null_worldtube.py --neutrino       # Neutrino masses and mixing
python3 simulations/null_worldtube.py --self-energy    # Alpha emergence
python3 simulations/null_worldtube.py --find-radii     # Self-consistent torus radii

# Analyses referenced in Paper 3
python3 simulations/null_worldtube.py --skilton         # Skilton's alpha formula
python3 simulations/null_worldtube.py --junction        # Junction conditions / geodesic curvature
python3 simulations/null_worldtube.py --gravity         # Self-consistent radii, Cornell potential
python3 simulations/null_worldtube.py --proton-mass     # Proton mass derivation

# Additional analyses
python3 simulations/null_worldtube.py --hydrogen        # Hydrogen atom model
python3 simulations/null_worldtube.py --weinberg        # Weinberg angle / electroweak
python3 simulations/null_worldtube.py --dark-matter     # Dark matter candidates
python3 simulations/null_worldtube.py --topology        # Topology survey
python3 simulations/null_worldtube.py --einstein        # Kerr-Newman geometry
python3 simulations/null_worldtube.py --pythagorean     # Pythagorean resonances
python3 simulations/null_worldtube.py --transmission-line  # Transmission line model
```

Run with `--help` for the full list of flags.

## Requirements

```
numpy matplotlib scipy
```

## Archive

The `archive/` directory contains earlier code:
- `nwt/` — Modular Python package (refactored from the monolith, Feb 2025)
- `koide/` — Koide formula explorations, Lorenz attractors, phase space
- `williamson/` — Williamson pivot field simulations
- `nwt_fdtd.py`, `nwt_fdtd_kn.py` — FDTD implementations
- Various `.png` output files from earlier runs
