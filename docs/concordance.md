---
layout: default
title: "Resolved-Mysteries Concordance"
---

# Resolved-Mysteries Concordance

A navigational companion to the NWT paper series (Papers 1&ndash;23): it maps
open problems and "mysteries" *posed* in earlier papers to where they were
later *resolved*, in both directions (forward: early &rarr; late; backward:
late &rarr; the problem it closed).

**Why this is a separate document.** The papers are **frozen published
records** &mdash; each carries a Zenodo DOI, so editing them would break
citation and re-versioning. This concordance provides the cross-referencing
value *without* touching the published papers. It is a living, additive
artifact; the papers stay as published.

**What it shows.** The point is not just navigation. A problem *posed* by an
early paper and *solved* later **from the same substrate algebra** is evidence
of the program's progressive coherence &mdash; closure, not ad-hoc patching.

**Provenance.** Compiled from a cross-paper self-consistency review of
Papers 1&ndash;23, anchored on the open-source [`nwt-substrate`](https://github.com/JimGalasyn/nwt-substrate)
library. Cited at paper granularity; a deeper per-paper "future-work /
conjecture" extraction would extend it (see *Extending this concordance*
below).

---

## 1. Resolved &mdash; a posed problem, later solved from the substrate

| # | Mystery / open problem | Posed in | Resolved in | Resolution |
|---|---|---|---|---|
| R1 | **The SO(7) substrate origin.** The &tau;-mass anchor m<sub>&tau;</sub> = (20/21)&alpha;&Lambda; has 21 = k(2k+1) = 3&times;7, and the paper flags *"2k+1 = 7 = dim of the fundamental rep of SO(7); its appearance warrants further investigation."* | **Paper 1 &sect;7.4** | **Papers 7&ndash;22** | The entire K<sub>7</sub> / so(7) / Spin(7) / Cl(0,7) substrate algebra. The single most consequential forward-reference in the corpus &mdash; a &tau;-formula footnote became the framework's foundation. |
| R2 | **Carrier-knot n<sub>q</sub> assignment (the "fitting critique").** n<sub>q</sub> &isin; {0, 2, 3, 5} entering the Paper-6 mass formula was assigned *empirically per sector*, not derived. | **Papers 6, 8** | **Paper 21a** | n<sub>q</sub> = F<sub>n</sub> via the (2, F<sub>n</sub>) Fibonacci torus-knot carrier family + Murasugi's determinant formula; per-walk sector from the cosmogenic Z<sub>3</sub> = AGL(1,7) rule (x &rarr; 2x mod 7). Closed-form, no fitting. |
| R3 | **sin<sup>2</sup>&theta;<sub>W</sub> from first principles.** Posed as 3/13 from a *13-mode count*, with "a rigorous derivation from the spectral geometry of the torus deferred to future work." | **Paper 1 &sect;8.1 / Paper 2** | **Papers 13, 14** | sin<sup>2</sup>&theta;<sub>W</sub> = (2 + &alpha;)/(dim 𝕆 + 1) = (2 + &alpha;)/9 = 0.22303. *(This is the **on-shell** angle &mdash; matches 1 &minus; M<sub>W</sub><sup>2</sup>/M<sub>Z</sub><sup>2</sup> to 0.009%; the effective-angle running is a separate open item.)* |
| R4 | **The reactor angle &theta;<sub>13</sub>.** Historically taken &approx; 0; its non-zero value was the open neutrino-mixing question. | (pre-NWT / Paper 1 PMNS) | **Paper 20** | &theta;<sub>13</sub> = arcsin&radic;(3&alpha;) = &radic;(3&alpha;) &approx; 8.5&deg;, from Spin(7) &sub; Spin(8) breaking &mdash; matches NuFIT (0.7%). A genuine prediction of the non-zero reactor angle. |
| R5 | **Why three fermion generations.** Paper 1 ties "3" to k = 3 Z<sub>3</sub> harmonic modes &mdash; conjectural. | **Paper 1 &sect;3.1** | **Papers 20, 21a** | The cosmogenic Z<sub>3</sub> = AGL(1,7) (&sigma;: x &rarr; 2x mod 7, QR cycle 1&rarr;2&rarr;4) / G<sub>2</sub> &rarr; SU(3) breaking via the S<sup>6</sup> stabilizer. The *same* Z<sub>3</sub> fixes &delta;<sub>CP</sub> and increments the Fibonacci carrier index. |
| R6 | **The CP phase &delta;<sub>CP</sub>.** Paper 1 gives &delta;<sub>CP</sub> = &pi; &minus; 2 as a knot-deficit angle &mdash; ad hoc. | **Paper 1 &sect;5.3** | **Paper 20 &sect;7.5** | &delta;<sub>CP</sub> = &minus;2&pi;/3 = &minus;120&deg; from the Z<sub>3</sub> = &pi;<sub>1</sub>(PSU(3)) winding number &mdash; the same Z<sub>3</sub> as the generations (R5). |
| R7 | **Proton mass from first principles.** Paper 1 &sect;13 "Known limitations": the proton mass used two *un-derived* variational coefficients (3/2 kinetic, 2 Coulomb). | **Paper 1 &sect;9.5, &sect;13** | **Papers 6, 13** | The (1, 3, 5, 5) nucleon-tuple correction + cinquefoil n<sub>q</sub> = 5 carrier; proton to ~0.1% with no variational fudge. |

---

## 2. Cross-paper unifications &mdash; one structure, several observables

| # | Structure | Appears in | As |
|---|---|---|---|
| U1 | **The trefoil Pythagorean 13 = (p<sup>2</sup> + q<sup>2</sup>) for T(2,3)** | Paper 13 **and** Paper 20 | the charged-lepton mass ratio m<sub>&tau;</sub>/m<sub>&mu;</sub> = 13 + 4 &minus; 25&alpha; (Paper 13) **and** the PMNS NLO rotation U<sub>&ell;</sub> = R(13&alpha;/2) (Paper 20 &sect;7.6). One trefoil, two independent observables &mdash; a substrate-monism signature. |
| U2 | **The denominator-7 / so(7) thread** | Paper 1 &rarr; 7&ndash;22 | the 7 first surfacing in Paper 1's &tau;-anchor (R1) recurs as k+2 = 7 (SU(2)<sub>5</sub> level), &#124;V(K<sub>7</sub>)&#124; = 7, &nu;<sub>RR</sub> = 5/7 filling, and &pi;/7 per-face flux &mdash; all forced by so(7) &rarr; k = 5. |

---

## 3. Refined / superseded &mdash; the story moved on (historical drift, record-only)

These early forms were *replaced* by cleaner later ones (not always flagged as
"open" &mdash; they simply evolved). Recorded so the lineage is legible; the
papers stand as published.

| Quantity | Early form | Current / late form |
|---|---|---|
| **&alpha; closed form** | Skilton &radic;(137<sup>2</sup> + &pi;<sup>2</sup>) (Paper 1) &rarr; 1/(&radic;2&middot;&pi;<sup>4</sup>) = 137.76 (P2, P4, P8a, P9, P10) | 1/(25&pi;&radic;3 + 1) = 137.0350 (P11+, library) |
| **sin<sup>2</sup>&theta;<sub>W</sub>** | 3/13 = 0.2308 (P2); 0.222 (P8/P9) | (2 + &alpha;)/9 = 0.22303 (P13, P14) |
| **&kappa; (torus aspect ratio)** | &pi;<sup>2</sup> = 9.870 (P4&ndash;P9); 12/&radic;7 (P5) | &kappa;<sub>Macken</sub> = 9.844 (P11+) |
| **Nucleon carrier knot** | trefoil n<sub>q</sub> = 3 (P15, P20) | cinquefoil n<sub>q</sub> = 5 (P13, library) |
| **n<sub>q</sub> meaning (&ldquo;3 lobes = 3 quarks&rdquo;)** | constituent count = # co-confined quarks; a baryon's 3 &ldquo;quarks&rdquo; = the trefoil's 3 crossings (P6 &sect;&ldquo;constituents&rdquo;; P8 hedges to scare-quoted &ldquo;quarks&rdquo;) | knot **determinant** = dimension of the carrier's SU(2) rep, n<sub>q</sub> = det T(2, F<sub>n</sub>) = F<sub>n</sub> (P21a: *&ldquo;n<sub>q</sub> denotes the carrier determinant F<sub>n</sub>, not Paper 6's constituent count&rdquo;*). The carrier is a substrate-algebra label, **not** a 3-D lobed object (its walk's spatial embedding is mostly the unknot), so &ldquo;lobes-as-quarks&rdquo; is retired. Integer preserved because crossing = det = n on T(2, n); they diverge off it (figure-eight 4<sub>1</sub>: 4 crossings, det 5). Decisive falsifier: the proton (3 valence quarks) now sits on the cinquefoil n<sub>q</sub> = 5. |
| **&tau;-lepton topology** | lepton / unknot T(2,1) (P2, P6, P8) | stealth-baryon T(3,4) (P7, P11, P12) |
| **&alpha;<sub>s</sub>** | 16&alpha; = p<sup>4</sup>&alpha; (Paper 1, *derived*, 0.97%) | PDG input 0.1179 (library) |
| **m<sub>p</sub>/m<sub>e</sub>** | 6&pi;<sup>5</sup> = 1836 (Paper 1, Lenz; correct) | &mdash; *(P14 &sect;5.3's 9/(2&alpha;)&middot;n<sub>q</sub><sup>3</sup> is off ~9&times;: a frozen-paper erratum)* |
| **BPS self-dual coupling** | Paper 16 quotes &lambda; = e<sup>2</sup>/2, &mu;<sub>BPS</sub> = &pi; (v=1) &mdash; but these are the **Ginzburg&ndash;Landau &frac12;-kinetic *code*** values (&frac12;&#124;D&psi;&#124;<sup>2</sup> + &frac12;&#124;B&#124;<sup>2</sup> + &frac18;(&hellip;)<sup>2</sup>; local gauged&lowbar;higgs, nwt-substrate), printed under Paper 16's *relativistic* Lagrangian &#124;D&psi;&#124;<sup>2</sup> &minus; &frac14;F<sup>2</sup> &minus; (&lambda;/4)(&hellip;)<sup>2</sup> &rArr; internally inconsistent (its own table shows &mu; = &pi; at v=1 vs stated 2&pi;v<sup>2</sup> = 2&pi;). | &lambda; = 2e<sup>2</sup>, &mu; = 2&pi;v<sup>2</sup> &mdash; correct for that relativistic Lagrangian (**Paper 11**; **jax-solitons** library PR #9); Paper 16's *own* BPS equation B = e(&#124;&psi;&#124;<sup>2</sup> &minus; v<sup>2</sup>) already implies 2e<sup>2</sup>. Same physics via &psi; = &radic;2 &phi;; the physical invariants (m<sub>H</sub> = m<sub>gauge</sub>, &kappa;<sub>GL</sub> = 1/&radic;2, flux 2&pi;n, helicity 8&pi;<sup>2</sup>&middot;lk) and the Paper-8a &alpha;-lock (the *aspect* &kappa; = &pi;<sup>2</sup>, a different &kappa;) are unchanged. **Action: correct Paper 16's quoted &lambda; &rarr; 2e<sup>2</sup>, &mu; &rarr; 2&pi;v<sup>2</sup>** (or reprint its Lagrangian in GL form). Published as **v2** &mdash; version DOI [10.5281/zenodo.20683110](https://doi.org/10.5281/zenodo.20683110); concept [10.5281/zenodo.19710845](https://doi.org/10.5281/zenodo.19710845) (always-latest). |

---

## 3b. By paper &mdash; stale-value index (check before citing a paper's value)

Per-paper view of &sect;3 (&rarr; = current value; see &sect;3 for detail).

- **Paper 1** &mdash; &alpha; &radic;(137<sup>2</sup>+&pi;<sup>2</sup>) &rarr; 1/(25&pi;&radic;3+1); sin<sup>2</sup>&theta;<sub>W</sub> 3/13 &rarr; (2+&alpha;)/9; &alpha;<sub>s</sub> 16&alpha; (derived) &rarr; PDG input.
- **Paper 2** &mdash; sin<sup>2</sup>&theta;<sub>W</sub> 3/13; &tau;-lepton topology unknot T(2,1) &rarr; stealth-baryon T(3,4); &alpha; 1/(&radic;2&middot;&pi;<sup>4</sup>).
- **Paper 4** &mdash; &alpha; 1/(&radic;2&middot;&pi;<sup>4</sup>); torus aspect &kappa;=&pi;<sup>2</sup> (9.870) &rarr; &kappa;<sub>Macken</sub>=9.844.
- **Paper 5** &mdash; torus aspect &kappa;=12/&radic;7 &rarr; &kappa;<sub>Macken</sub>=9.844.
- **Paper 6** &mdash; n<sub>q</sub> meaning (&ldquo;3 lobes = 3 quarks&rdquo;) &rarr; knot **determinant** (n<sub>q</sub>=det T(2,F<sub>n</sub>)=F<sub>n</sub>); &tau;-topology hedge.
- **Papers 8 / 8a** &mdash; n<sub>q</sub> meaning; &tau; topology; sin<sup>2</sup>&theta;<sub>W</sub> 0.222; &alpha; 1/(&radic;2&middot;&pi;<sup>4</sup>).
- **Papers 9 / 10** &mdash; &alpha; 1/(&radic;2&middot;&pi;<sup>4</sup>); (P9) sin<sup>2</sup>&theta;<sub>W</sub> 0.222, &kappa;=&pi;<sup>2</sup>.
- **Paper 14 &sect;5.3** &mdash; m<sub>p</sub>/m<sub>e</sub> = 9/(2&alpha;)&middot;n<sub>q</sub><sup>3</sup> off ~9&times; (frozen-paper erratum; correct 6&pi;<sup>5</sup>, Paper 1).
- **Papers 15 / 20** &mdash; nucleon carrier trefoil n<sub>q</sub>=3 &rarr; cinquefoil n<sub>q</sub>=5.
- **Paper 16** &mdash; BPS self-dual coupling &lambda;=e<sup>2</sup>/2, &mu;=&pi; &rarr; &lambda;=2e<sup>2</sup>, &mu;=2&pi;v<sup>2</sup>. **Corrected in v2 (2026-06-13)** &mdash; version DOI 10.5281/zenodo.20683110, concept 10.5281/zenodo.19710845 (always-latest).

---

## 4. Still open &mdash; flagged, not yet resolved

Honest record of mysteries the corpus poses that remain open (so the
concordance doesn't read as "all solved").

| # | Open problem | Status |
|---|---|---|
| O1 | **PMNS U<sub>&ell;</sub> first-principles derivation** (Paper 20 &sect;7.6). | Mechanism identified: the Z<sub>3</sub> = AGL(1,7) inter-generation structure, with U<sub>&ell;</sub> and &delta;<sub>CP</sub> sharing that Z<sub>3</sub>. But a uniform R(13&alpha;/2) rotation requires the charged-lepton off-diagonals to *track the mass-squared gaps* &mdash; not forced by the Z<sub>3</sub> alone. The amplitude 13&alpha;/2 must *emerge* from the K<sub>8</sub> edge structure, not be inserted. |
| O2 | **Mass-formula light-meson residual** (Paper 6). | &pi;/K/&rho; over-predicted ~+2&ndash;4%; the small-&beta; / Kelvin fat-ring correction hypothesis was *tested and refuted* (residual doesn't track &beta;; worst case K<sup>+</sup> at large &beta;). ~1% median is the formula's intrinsic zero-parameter accuracy. |
| O3 | **13-mode-counting derivation** (Paper 1 &sect;8.1). | The promised spectral-geometry derivation of "13 modes" was never done &mdash; it was *bypassed* by (2 + &alpha;)/9 (R3), not rigorously derived. |
| O4 | **Native-so(7) Hamiltonian / SU(2)<sub>5</sub> lattice realization.** | The chiral central charge c = 15/7 has been derived three independent ways and reproduced on IBM Heron hardware; but constructing the full modular-S / anyon-spin (&theta;<sub>w</sub>) structure of SU(2)<sub>5</sub> directly on the K<sub>7</sub> lattice (a Z<sub>7</sub>-vs-Z<sub>2</sub> mismatch in the Wilson-loop basis) remains an open construction problem. |

---

## Extending this concordance

This first pass matches problems *posed* in early papers to their later
resolutions. A *comprehensive* version would read each paper's "Future work /
Open problems / Conjecture" sections directly and match every flagged item to
its resolution &mdash; a per-paper pass. Add entries here as problems are posed
or closed; for *future* (not-yet-frozen) papers, include backward-references at
authoring time.

---

[Home](index.html) &#183; [The Predictions](results.html) &#183; [Papers](papers.html) &#183; [History](history.html) &#183; [About](about.html)
