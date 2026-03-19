---
layout: default
title: "Papers"
---

# Papers

## Paper 1: The Standard Model from a Torus Knot

**Full title:** *The Standard Model from a Torus Knot: Spectrum, Resonance Structure, and Decay Dynamics*

James P. Galasyn and Claude Théodore

The founding paper of Null Worldtube Theory. Starting from the idea that an electron is a photon confined to a (2,1) torus knot with aspect ratio k=3, we derive 23 of the 26 free parameters of the Standard Model — all quark and lepton masses, the Higgs boson mass and vacuum expectation value, the Weinberg angle, the strong coupling constant, CKM and PMNS mixing matrices, and neutrino masses. The paper also develops a Pythagorean resonance condition that explains why protons are stable and all mesons decay, and verifies energy conservation across seven complete meson decay chains.

**DOI:** [10.5281/zenodo.18891785](https://doi.org/10.5281/zenodo.18891785)

---

## Paper 2: Three Integers and a Mass

**Full title:** *Three Integers and a Mass: Deriving the Standard Model Input Set*

James P. Galasyn and Claude Théodore

Paper 1 took three inputs as given: the fine-structure constant &alpha;, the muon mass, and the tube energy scale. This companion paper shows that all three are themselves derivable from the torus quantum numbers (p, q, k) = (2, 1, 3), reducing the complete input set to **three integers and one mass** (the electron mass). It connects the theory to F. Ray Skilton's forgotten 1980s derivation of &alpha; from a Pythagorean triple, revealing that Skilton's generators are the torus quantum numbers in disguise.

**DOI:** [10.5281/zenodo.18892311](https://doi.org/10.5281/zenodo.18892311)

---

## Paper 3: Nuclear Magic Numbers from Torus Topology

**Full title:** *Nuclear Magic Numbers from Torus Topology*

James P. Galasyn and Claude Théodore

The torus geometry that encoded the particle spectrum now reaches into the atomic nucleus. Starting from the pion mass relation m<sub>&pi;</sub> = 2m<sub>e</sub>/&alpha; and the k=3 aspect ratio, we derive the nuclear potential depth (50.2 MeV), the scalar-vector splitting of the relativistic mean field (V &minus; S = 853 MeV), and the resulting spin-orbit interaction — all with zero new free parameters. Solved in a Woods-Saxon shell model, these produce all seven nuclear magic numbers (2, 8, 20, 28, 50, 82, 126) and predict N=184 as the next closure. As a stress test, the NWT-derived nuclear masses correctly trace all four natural radioactive decay series to their Pb/Bi endpoints.

**DOI:** [10.5281/zenodo.19036783](https://doi.org/10.5281/zenodo.19036783)

---

## Paper 4: From Vortex Rings to the Top Quark

**Full title:** *From Vortex Rings to the Top Quark: Dynamical Derivations in the Null Worldtube Framework*

James P. Galasyn and Claude Th&eacute;odore

Papers 1&ndash;3 treated the torus geometry as given. This paper derives it dynamically. Modeling the electron as a quantized vortex ring in a superfluid vacuum described by the Gross-Pitaevskii equation, we derive the tube radius r/R = 1/&pi;&sup2; (matching the NWT value of 0.10392 to 2.5%), explain quark confinement through an incommensurability condition on torus-knot winding numbers, and show that the pion mass m<sub>&pi;</sub> = 2m<sub>e</sub>/&alpha; = 140.05 MeV (0.34%) and the top quark mass m<sub>t</sub> = 2k&sup2;m<sub>e</sub>/&alpha;&sup2; = 172,730 MeV (0.02%) emerge from the same framework. The Macken spacetime impedance Z<sub>s</sub> = c&sup3;/G connects gravitational and electromagnetic scales, while the Euler-Heisenberg QED vacuum polarization replaces Williamson's ad hoc "pivot field" as the physical mechanism for soliton confinement.

**DOI:** [10.5281/zenodo.19051589](https://doi.org/10.5281/zenodo.19051589)

---

## Paper 5: The Electron Mass from Phase Closure

**Full title:** *The Electron Mass from Phase Closure: Self-Consistent Torus Knots in the Superfluid Vacuum*

James P. Galasyn and Claude Th&eacute;odore

Papers 1&ndash;4 took the electron mass as a given input. This paper eliminates it as a free parameter. Three self-consistency conditions &mdash; phase closure on the (2,1) torus knot at first resonance m=3, dual resonance with the (1,4) proton mode, and vortex ring energy balance &mdash; together determine R/&xi; = &radic;5/2 from the Pythagorean identity 3&sup2; = (&radic;5)&sup2; + 2&sup2;, fix the aspect ratio &kappa; = 12/&radic;7, and yield m<sub>e</sub> = 9.19 &times; 10<sup>&minus;31</sup> kg (0.85% match). The electron mass is now a derived quantity: a consequence of the topology of a (2,1) torus knot in a superfluid vacuum, not an input.

**DOI:** [10.5281/zenodo.19072313](https://doi.org/10.5281/zenodo.19072313)

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
