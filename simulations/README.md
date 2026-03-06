# Simulations

Null Worldtube Theory: simulation and analysis code.

## Usage

```bash
cd simulations

# Core analyses (Paper 1)
python3 -m nwt --koide             # Lepton Koide analysis
python3 -m nwt --quarks            # Quark masses + EW/CKM/PMNS
python3 -m nwt --neutrino          # Neutrino masses and mixing
python3 -m nwt --self-energy       # Alpha emergence
python3 -m nwt --find-radii        # Self-consistent torus radii

# Derivation chain (Paper 2)
python3 -m nwt --skilton           # Skilton's alpha formula
python3 -m nwt --junction          # Junction conditions / geodesic curvature
python3 -m nwt --gravity           # Self-consistent radii, Cornell potential
python3 -m nwt --proton-mass       # Proton mass derivation

# Additional analyses
python3 -m nwt --hydrogen          # Hydrogen atom model
python3 -m nwt --weinberg          # Weinberg angle / electroweak
python3 -m nwt --dark-matter       # Dark matter candidates
python3 -m nwt --topology          # Topology survey
python3 -m nwt --einstein          # Kerr-Newman geometry
python3 -m nwt --pythagorean       # Pythagorean resonances
python3 -m nwt --transmission-line # Transmission line model
```

Run with `--help` for the full list of flags.

## Requirements

```
numpy matplotlib scipy
```

## Package Structure

```
simulations/
├── nwt/                           # Main analysis package
│   ├── __main__.py                # CLI parsing + lazy-import dispatch
│   ├── constants.py               # Physical constants, TorusParams
│   ├── core.py                    # Torus geometry, self-energy, resonance
│   ├── fields.py                  # Casimir, greybody, torus EM fields, LC circuit
│   ├── particles.py               # Basic analysis, find-radii, pair-production
│   ├── hydrogen.py                # H/He/Li orbital dynamics
│   ├── transitions.py             # Photon emission/absorption
│   ├── hadrons.py                 # Quarks, mesons, baryons, proton mass
│   ├── orbit.py                   # Kerr-Newman, Einstein, junction conditions
│   ├── pair_creation.py           # Euler-Heisenberg, Schwinger, gradient
│   ├── settling.py                # Transmission line, Kelvin waves, Thevenin
│   └── analyses.py                # Standalone analyses (Skilton, Koide, ...)
├── nonlinear_permittivity.py      # Born-Infeld / nonlinear permittivity
├── plot_settling_spectrum.py      # Settling spectrum visualization
└── output/                        # Generated output files
```
