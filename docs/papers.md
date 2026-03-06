---
layout: default
title: "Papers"
---

# Papers

## Paper 1: The Standard Model from a Torus Knot

**Full title:** *The Standard Model from a Torus Knot: Spectrum, Resonance Structure, and Decay Dynamics*

James P. Galasyn and Claude

The founding paper of Null Worldtube Theory. Starting from the idea that an electron is a photon confined to a (2,1) torus knot with aspect ratio k=3, we derive 23 of the 26 free parameters of the Standard Model — all quark and lepton masses, the Higgs boson mass and vacuum expectation value, the Weinberg angle, the strong coupling constant, CKM and PMNS mixing matrices, and neutrino masses. The paper also develops a Pythagorean resonance condition that explains why protons are stable and all mesons decay, and verifies energy conservation across seven complete meson decay chains.

**DOI:** [10.5281/zenodo.18891785](https://doi.org/10.5281/zenodo.18891785)

---

## Paper 2: Three Integers and a Mass

**Full title:** *Three Integers and a Mass: Deriving the Standard Model Input Set*

James P. Galasyn and Claude

Paper 1 took three inputs as given: the fine-structure constant &alpha;, the muon mass, and the tube energy scale. This companion paper shows that all three are themselves derivable from the torus quantum numbers (p, q, k) = (2, 1, 3), reducing the complete input set to **three integers and one mass** (the electron mass). It connects the theory to F. Ray Skilton's forgotten 1980s derivation of &alpha; from a Pythagorean triple, revealing that Skilton's generators are the torus quantum numbers in disguise.

**DOI:** [10.5281/zenodo.18892311](https://doi.org/10.5281/zenodo.18892311)

---

## Source Code

The simulation code that reproduces all predictions, figures, and verification calculations is available on GitHub:

[github.com/JimGalasyn/null-worldtube](https://github.com/JimGalasyn/null-worldtube)

Run individual modules:
```
python3 -m simulations.nwt --koide       # Lepton masses via Koide formula
python3 -m simulations.nwt --quarks      # Quark masses
python3 -m simulations.nwt --neutrino    # Neutrino masses and mixing
python3 -m simulations.nwt --self-energy # Fine-structure constant emergence
python3 -m simulations.nwt --pythagorean # Stability analysis
python3 -m simulations.nwt --skilton     # Skilton's alpha derivation
python3 -m simulations.nwt --proton-mass # Proton mass from Cornell potential
```

---

[Home](index.html) &#183; [The Predictions](results.html) &#183; [About](about.html)
