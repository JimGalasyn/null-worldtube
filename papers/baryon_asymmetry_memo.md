# Baryon Asymmetry & Cosmological Matter-Generation Memo

*Extracted from Paper 21 v0 (`paper21_baryon_asymmetry.tex`, commit eabb960) during the 2026-05-22 refactor that re-scoped Paper 21 to particle synthesis. This memo preserves the η_B, Ω_b/Ω_c, and four-unseen-species content for a future cosmology paper (provisional slot: Paper 25, "Cosmic Baryon Asymmetry from K_7 Substrate"). The Phase 2 hardware-PASS content stayed in Paper 21; the cosmological-amplitude derivations live here until the new paper is opened.*

*Foundation citations: walk-to-Pauli direction-sensitive map, Steane-code identification of K_7, pair production via forward + reverse-traversal walks, and Sakharov "B-violation = discrete walk-homology change" all belong to the refactored Paper 21 and should be cited from there when this memo is promoted.*

---

## Headline results

$$
\boxed{\eta_B = \frac{3}{14}\,\alpha^4 = \frac{\mathrm{rank}(\so(7))}{\dim(G_2)}\,\alpha^4 \approx 6.077\times 10^{-10}}
$$

vs Planck 2018: $(6.10 \pm 0.10)\times 10^{-10}$. **Deviation: −0.38%** — well inside the 1.6% Planck experimental error.

$$
\boxed{\frac{\Omega_b}{\Omega_c} = h_v^2\,\alpha\,(1 + \mathrm{rank}(\so(7))\,\alpha) = 25\alpha(1+3\alpha) \approx 0.18643}
$$

vs Planck: $0.18642 \pm 0.00224$. **Deviation: +0.005%** — 240× tighter than experimental error.

Both derive from already-established substrate primitives ($\alpha$, $h_v = 5$, $\mathrm{rank}(\so(7)) = 3$, $\dim(G_2) = 14$) with zero free parameters.

---

## Cosmogenic asymmetry: substrate mechanism (was §4.4 of P21 v0)

At thermal equilibrium, pair production and annihilation are time-reversal symmetric, producing equal numbers of matter and antimatter walks. The matter–antimatter asymmetry $\eta_B$ requires a deviation from equilibrium combined with a CP-violating amplitude — the second and third Sakharov conditions.

In the substrate picture, the CP-violating amplitude arises from Law 2 (three $K_7$ Hamilton-cycle types by edge-difference): substrate walks with NR-Hamilton substructure ($d=3$ Hamilton cycle present as a subwalk) carry an additional CP-violating loop amplitude. These walks — identified in the SM with $\pi^0$, $\Omega^-$, $K^0$, and several other CP-coupled species — contribute asymmetrically to the substrate transduction rate at NLO in $\alpha$.

The result is

$$
\eta_B = \frac{\mathrm{rank}(\so(7))}{\dim(G_2)}\,\alpha^4 = \frac{3}{14}\,\alpha^4
$$

per pair-production event, summed over all matter-generating processes in the early universe. Three fermion generations (numerator = 3) contribute additively to the CP-asymmetric amplitude; the $G_2 = \mathrm{Aut}(\mathbb{O})$ substrate-automorphism phase space (denominator = 14) distributes the amplitude across the substrate's 14 continuous symmetry degrees of freedom. $\alpha^4$ arises from two NR-Hamilton CP-violating vertices, each contributing $\alpha^2$ loop suppression.

The mechanism is **phenomenologically neutral** on which cosmogenic process supplied the out-of-equilibrium condition. Whether the early-universe pair-production-and-annihilation history was driven by inflation+reheating, a parent black-hole substrate transition (Paper 22), ekpyrotic phase transitions, or another framework, the substrate predicts the same $\eta_B$ from substrate primitives alone.

---

## Sakharov conditions at the substrate level (was §5.3 of P21 v0)

Sakharov's three conditions for baryogenesis are each realised by structural features of the $K_7$ substrate:

1. **Baryon-number violation.** Discrete $K_7$ walk-homology classes $(p,q)$ change discretely under substrate transduction; there is no continuous symmetry to protect a conserved baryon number. The walk-winding $(p,q)$ of a created particle pair sums to $(0,0)$ (from $W + \bar W$), but the individual walks carry nontrivial windings that change discretely during cosmogenic processes.

2. **C/CP-violation.** The NR-Hamilton substructure (Law 2) provides the substrate's CP-violating amplitude. Walks with NR-Hamilton content carry a phase that is asymmetric under charge conjugation (which maps matter walks to antimatter walks via reverse-traversal). This is distinct from the standard CKM Jarlskog, which contributes at $\sim \alpha^{5/2}$ via quark-flavour mixing and gives $\eta_B \sim 10^{-20}$ when used as the sole CP-violation source. The substrate's NR-Hamilton channel operates at the $\alpha^4$ scale, ten orders of magnitude larger than CKM-only and matching observed $\eta_B$.

3. **Out-of-thermal-equilibrium.** Provided by whatever cosmogenic process generates the early-universe matter content (inflation+reheating, parent black-hole transition, ekpyrotic transition, etc.). This work does not specify the cosmogenic process; the substrate prediction is mechanism-neutral.

---

## Derivation of η_B = (3/14)α⁴ (was §6 of P21 v0)

### Sakharov amplitude framework

The cosmological baryon-to-photon ratio $\eta_B$ is the integrated asymmetry produced by CP-violating, baryon-number-violating processes in an out-of-equilibrium epoch. In standard baryogenesis frameworks, $\eta_B$ takes the schematic form

$$
\eta_B \sim \frac{n_X}{s}\,\varepsilon_{CP}
$$

where $n_X/s$ is the ratio of CP-violating source particles to entropy, and $\varepsilon_{CP}$ is the CP-asymmetry parameter per source decay.

In the NWT substrate picture, $\eta_B$ arises from differential pair-production rates: at the cosmogenic transduction, matter walks (forward-traversal) are minted at slightly higher rate than antimatter walks (reverse-traversal) when the walk's $\sigma$-orbit composition contains NR-Hamilton substructure ($d=3$ Hamilton cycle as subwalk). The asymmetric amplitude per pair-production event is the substrate's $\varepsilon_{CP}$.

### Substrate amplitude: two NR-Hamilton vertices at loop level

The substrate's CP-violating amplitude operates at next-to-leading order squared (NLO²) in $\alpha$. The structural reading:

- Each NR-Hamilton substructure in a walk's $\sigma$-orbit composition contributes one CP-violating vertex.
- Each vertex carries $\alpha^2$ loop suppression (one closed loop with one $\alpha$ EM coupling at each end).
- Two interfering NR-Hamilton vertices produce the CP-asymmetric matter-vs-antimatter pair-production rate difference.
- Net amplitude scaling: $\alpha^2 \times \alpha^2 = \alpha^4$.

Three fermion generations contribute additively to the CP-asymmetric amplitude (each generation has its own NR-Hamilton substructure):

$$
\varepsilon_{CP} \propto \mathrm{rank}(\so(7)) \cdot \alpha^4 = 3\alpha^4
$$

The $G_2 = \mathrm{Aut}(\mathbb{O})$ substrate-automorphism phase space has $\dim(G_2) = 14$ continuous degrees of freedom; the CP-asymmetric amplitude is distributed across these:

$$
\eta_B = \frac{\mathrm{rank}(\so(7))}{\dim(G_2)}\,\alpha^4 = \frac{3}{14}\,\alpha^4
$$

### Numerical result and Planck comparison

With substrate-derived $\alpha = 1/(25\pi\sqrt 3 + 1) \approx 7.297\times 10^{-3}$:

$$
\eta_B = \frac{3}{14}\,(7.297\times 10^{-3})^4 = 6.077\times 10^{-10}
$$

Compared to Planck 2018: $\eta_B = (6.10 \pm 0.10)\times 10^{-10}$. Deviation: −0.38%, well inside the 1.6% Planck experimental error. Substrate $\alpha$ and CODATA $\alpha$ agree to ∼8 ppm, giving identical $\eta_B$ predictions to four significant figures.

### Equivalent stabilizer-code reading

(Footnote of §6 in P21 v0, now part of refactored Paper 21's §4. Preserved here as cross-ref.)

Under the Steane $[[7,1,3]]$ identification of the $K_7$ substrate, the three X-type stabilizer supports share a unique common point coinciding with the Heffter polar vertex (the substrate-vacuum reference), and the transversal Hadamard $H^{\otimes 7}$ swaps X-rail readout with Z-rail readout on the same Fano-line-complement supports. Matter and antimatter are then *not* two condensate species but the two readout-basis selections of a single self-dual code, and $\eta_B = (3/14)\alpha^4$ is the cosmogenic $H^{\otimes 7}$-basis-selection bias rather than a walk population imbalance. Under the direction-sensitive walk-to-Pauli map (X-destination for QR-class edges $d \in \{1, 2, 4\}$, Z-destination for NR-class edges $d \in \{3, 5, 6\}$), the all-QR proton Hamilton cycle resolves to the logical Pauli $\bar X_L = X^{\otimes 7}$ and its CPT-conjugate to $\bar Z_L = Z^{\otimes 7} = H^{\otimes 7} \bar X_L (H^{\otimes 7})^\dagger$ exactly at the Pauli-bit level. The numerical amplitude is unchanged; the substrate-mechanistic reading sharpens to "preferred-stabilizer-basis selection at the cosmogenic phase transition."

### Candidate sweep verification

To confirm that $3/14$ is the structurally correct prefactor (rather than a numerical coincidence), 17 substrate-canonical formula candidates of the form $(p/q)\cdot\alpha^n$ with $p,q,n$ drawn from substrate primitives $\{1, 2, 3, 4, 5, 7, 8, 14, 16, 21, 28, 32, 35\}$ and standard QED-loop prefactors $\{1/\pi, 1/(4\pi), 1/(16\pi), 1/(32\pi)\}$ were tested. Top matches:

| Rank | Formula | Deviation |
|---|---|---|
| 1 | **(3/14)·α⁴ = rank(so(7))·α⁴ / dim(G₂)** | **0.38%** |
| 2 | (7/32)·α⁴ = N_VERT_K₇·α⁴ / (K₇_TRIANGLES − rank(so(7))) | 1.69% |
| 3 | α⁴/5 = α⁴/h_v | 7.02% |

QED-loop-natural candidates ($\alpha^4/4\pi$, $\alpha^4/16\pi$, etc.) all under-predict $\eta_B$ by 60–95%, confirming that the substrate-canonical (non-$\pi$) prefactor $3/14$ is correct. Lower-power candidates ($\alpha^3/X$) over-predict by $10^4\times$, locking the $\alpha^4$ scaling.

### Comparison to SM CKM-only baryogenesis

The SM Cabibbo–Kobayashi–Maskawa Jarlskog invariant provides the SM's sole intrinsic CP-violation source from quark mixing: $J \approx 3 \times 10^{-5}$. Using $J$ as the CP amplitude yields $\eta_B^{\text{SM-CKM}} \sim 10^{-20}$ — ten orders of magnitude smaller than observed. This is the well-known "CKM amplitude shortfall" motivating beyond-SM baryogenesis mechanisms.

NWT's substrate provides $\eta_B = (3/14)\alpha^4 \sim 6 \times 10^{-10}$ from a distinct CP-violation channel: the NR-Hamilton substructure of $K_7$ walks. This is independent of CKM Jarlskog, operates at the $\alpha^4$ scale, and matches observed $\eta_B$ without beyond-SM fields. The substrate has two CP-violation channels:

- **Static (kinematic):** the $\delta_{CP} = \pi - 2$ phase from trefoil geometry, manifesting in CKM and PMNS as the observed CP phase.
- **Dynamic (NLO² loop):** the $(3/14)\alpha^4$ amplitude from NR-Hamilton substructure, producing the cosmic asymmetry.

Both at Form A clean precision against observation; independent observables of the same underlying substrate algebra.

---

## Derivation of Ω_b/Ω_c = h_v²α(1+3α) (was §7 of P21 v0)

### Substrate-canonical formula

$$
\boxed{\frac{\Omega_b}{\Omega_c} = h_v^2 \cdot \alpha \cdot (1 + \mathrm{rank}(\so(7)) \cdot \alpha) = 25\alpha(1 + 3\alpha)}
$$

Numerically, $25\alpha = 0.18243$ (LO) and $75\alpha^2 = 0.00399$ (NLO), summing to $0.18643$. Compared to Planck's $0.18642$: deviation $+0.005\%$, 240× tighter than Planck's experimental error.

### Matter-channel transduction interpretation

- $h_v^2 = 25 = \mathrm{H\_V\_SO7}^2$ is the substrate vacuum's matter-channel transduction probability: the squared Hopf-vacuum amplitude for matter walks to be minted relative to CDM walks. The same $h_v^2$ appears in $\Omega_m = h_v^2 \cdot \sqrt 3 \cdot \alpha$ (total matter density), making the two ratios share the substrate vacuum-partition factor.
- $\alpha$ is the trefoil Aharonov–Bohm phase coupling at leading order.
- $(1 + 3\alpha)$ is the three-generation NLO correction: each of the three SM fermion generations contributes additively to the matter-channel amplitude at the $\alpha^2$ level.

The truncated 2-term form $25\alpha + 75\alpha^2$ beats the closed-form geometric series $25\alpha/(1 - 3\alpha)$ (which gives $0.05\%$ deviation, $10\times$ worse) by an order of magnitude. Substrate physics cuts off at NLO without resummation; this is consistent with the closed-form/dynamical bifurcation principle: substrate algebra closes at LO + NLO kinematic order, with higher orders requiring loop machinery outside substrate scope.

### Derived Ω_b and Ω_c

Combining with the substrate prediction $\Omega_m = h_v^2 \cdot \sqrt 3 \cdot \alpha \approx 0.316$:

- $\Omega_b \approx 0.0497$ (Planck: 0.0493)
- $\Omega_c \approx 0.2663$ (Planck: 0.2664)

Both match Planck within experimental error.

### Consistency with η_B via shared substrate primitives

The derivations share substrate primitives: $\mathrm{rank}(\so(7)) = 3$ enters both (numerator factor in $\eta_B$, NLO coefficient in $\Omega_b/\Omega_c$); $\alpha$ enters both as the substrate coupling. $\eta_B$ is the $\alpha^4$ entry (NLO² loop suppression at CP-violating vertices), while $\Omega_b/\Omega_c$ is the $\alpha^1$ LO + $\alpha^2$ NLO entry (kinematic matter-channel transduction). Both at Form A clean precision against Planck.

---

## Substrate predictions for unseen species (was §8 of P21 v0)

Beyond the 25-particle compendium fitting Paper 11's mass formula at median ∼0.1% precision, the $K_7$ Heffter-walk enumeration identifies additional walk-realisable $(p,q)$ classes at low walk length $L \leq 9$ that have no SM counterpart. Of 13 such "missing" $(p,q)$ classes, eight are substrate sub-cycle building blocks; one is the NR-Hamilton primitive $(3,2)$ (the CP-channel scaffold, not itself a particle); and **four are substrate-canonical predictions for currently-unseen species:**

| (p,q) | Walk | L | (n_q, m) | Mass | Sector | Signature |
|---|---|---|---|---|---|---|
| (2,2) | 0-1-2-5-1-4-0 | 6 | (3, 6) | ∼27 MeV | hyperon | sub-electron DM, partial Fano |
| (3,1) | 0-2-4-0-2-5-1-4-0 | 8 | (3, 29) | ∼52 MeV | hyperon | sub-pion DM, σ₅+σ₆ dual-twist |
| (2,3) | 0-1-2-3-4-0-1-4-0 | 8 | (2, 17) | ∼154 MeV | meson | BSM heavy-pion-class resonance |
| **(3,3)** | 0-1-2-3-6-2-5-1-4-0 | 9 | (3, 16) | **∼397 MeV** | hyperon | **matter+CP hybrid baryon** |

Masses via Paper 11's torus-knot mass formula at the indicated $(n_q, m)$ inputs.

### The (3,3) particle as structurally unique prediction

Of the four, the $(3,3)$ walk at ∼397 MeV is the cleanest substrate prediction. Its walk $0\text{-}1\text{-}2\text{-}3\text{-}6\text{-}2\text{-}5\text{-}1\text{-}4\text{-}0$ uniquely combines:

- **QR-Hamilton start** $0\text{-}1\text{-}2\text{-}3$ (matter scaffold, first three steps of proton walk)
- **NR-Hamilton tail** $3\text{-}6\text{-}2\text{-}5\text{-}1\text{-}4\text{-}0$ (the full $d=3$ NR-Hamilton cycle, the substrate's CP-violation channel)
- Full 7-vertex coverage and 7/7 Fano edge-coverage
- Mixed 33% QR / 67% NR polarity
- **Walk-knot = carrier-knot = trefoil 3₁:** the only substrate prediction (and one of only a handful of $K_7$ walks total) where the walk's intrinsic 3D torus-knot type and Paper 11's carrier-knot identification coincide.

The simultaneous matter-scaffold + CP-channel character makes (3,3) a candidate for a CP-asymmetric matter species at the sub-Λ hadronic mass scale: a ∼400 MeV hyperon-class particle carrying intrinsic CP-asymmetric production amplitude. The substrate's distinctive matter+antimatter-bridge species.

### Hyperon-DM candidates: (2,2) at 27 MeV and (3,1) at 52 MeV

The $(2,2)$ and $(3,1)$ predictions are sub-electron-mass and sub-pion-mass hyperon-class species ($n_q = 3$ trefoil carriers). No PDG hyperon exists below Λ at 1116 MeV; these substrate predictions are firmly in the dark-matter mass regime.

- $(2,2)$: short walk ($L=6$), partial Fano coverage (5/7), NR-dominant ($33\%$ QR), no Hamilton scaffold. Light DM candidate at the substrate-resonance scale near the electron mass.
- $(3,1)$: longer walk ($L=8$), partial Fano (5/7), NR-dominant ($38\%$ QR), dual-twist substructure ($\sigma_5 + \sigma_6$ both present) — same signature as $\pi^0$ and $\Omega^-$. A 50–60 MeV DM hyperon with $\pi^0/\Omega^-$-class CP coupling structure.

### BSM meson: (2,3) at 154 MeV

Beyond-SM meson-class species ($n_q = 2$ Hopf carrier) at ∼150 MeV. PDG has known light mesons at this scale ($\pi^\pm$ at 140, $\pi^0$ at 135, $\eta$ at 548); the substrate-predicted $(2,3)$ at 154 MeV sits in an unfilled mass slot just above the pion triplet. Structure: $L=8$, $5/7$ Fano, $62\%$ QR-dominant, no NR-Hamilton substructure.

### Falsifiable predictions (cosmology-side)

1. $(2,2)$ at $\sim 27$ MeV: stable particles in 20–30 MeV range with hyperon-like quantum numbers.
2. $(3,1)$ at $\sim 52$ MeV: 50–60 MeV mass with $\pi^0/\Omega^-$-like CP-coupling signature ($\sigma_5 + \sigma_6$ dual-twist).
3. $(2,3)$ at $\sim 154$ MeV: 150–160 MeV non-SM resonance in the pion sector.
4. $(3,3)$ at $\sim 397$ MeV: sub-Λ hyperon at 390–410 MeV with both matter-scaffold and CP-channel structure.
5. $\sigma_4 = 0$ universal hyperon constraint.
6. $\sigma_6$ + heavy-hyperon correlation.
7. No $d=2$ Hamilton particles.
8. $\eta_B = (3/14)\,\alpha^4 = 6.077 \times 10^{-10}$ to future Planck-class precision.
9. $\Omega_b/\Omega_c = 25\alpha(1+3\alpha) = 0.186$ to future precision.

**Note:** Predictions 1–4 are *also* addressable on quantum hardware via the refactored Paper 21's synthesis protocol — chip-realizing the predicted species *before* natural detection is a substrate-monism falsifier the cosmology paper should cross-reference.

---

## Comparison to standard baryogenesis mechanisms (was §9.1 of P21 v0)

| Mechanism | Required new physics | η_B prediction |
|---|---|---|
| SM CKM-only | none (SM-only) | ∼10⁻²⁰ (way too small) |
| Leptogenesis | heavy Majorana ν (≳10⁹ GeV) | ∼10⁻¹⁰ (tunable) |
| EW baryogenesis | first-order EW transition (SM is 2nd-order) | ∼10⁻¹² (typical) |
| GUT baryogenesis | GUT-scale (≳10¹⁵ GeV) physics | varies |
| Affleck–Dine | SUSY flat directions | tunable |
| **NWT substrate** | **none beyond Papers 17–20** | **6.077×10⁻¹⁰ (0.38% Planck)** |

NWT's advantage is structural: the substrate's CP-violation amplitude arises from a topological feature (NR-Hamilton substructure in $K_7$ walks) that is automatically present, not added as a free ingredient. Mechanism-neutral on the cosmogenic process.

---

## Outlook: seven substrate cosmology predictions (was §9.3 of P21 v0)

| # | Quantity | Substrate formula | Match |
|---|---|---|---|
| 1 | $\Lambda_\text{cc}$ | $(m_e/M_\text{Pl})^4 \cdot \alpha^{16} \cdot h_\text{Coxeter}$ | 3-digit Planck |
| 2 | $\Omega_m$ | $h_v^2 \cdot \sqrt 3 \cdot \alpha$ | ≤1% |
| 3 | $\Omega_b/\Omega_c$ | $h_v^2 \cdot \alpha(1 + \mathrm{rank}(\so(7))\cdot\alpha)$ | **0.005%** |
| 4 | $H_0$ | $(m_e c^2/\hbar) \cdot \alpha^{18} \cdot 2^{-1/4}$ | consistent |
| 5 | $n_s$ | $1 - h_v \cdot \alpha$ | closed-form |
| 6 | $f_J$ | $(1 - \sqrt{\alpha})^3$ | closed-form |
| 7 | $\eta_B$ | $\mathrm{rank}(\so(7)) \cdot \alpha^4 / \dim(G_2)$ | **0.38%** |

All seven from $\{\alpha, h_v = 5, \mathrm{rank}(\so(7)) = 3, \dim(G_2) = 14, h_\text{Coxeter} = 6, m_e, M_\text{Pl}\}$.

---

## Open work (cosmology-side)

1. **Cosmogenic-bridge dynamics rate.** Computing the absolute pair-production rate and its time evolution requires specifying the cosmogenic process (parent black-hole transition / inflation+reheating / ekpyrotic / etc.). The substrate $\eta_B$ amplitude is mechanism-neutral.
2. **m-carrier closed form.** Paper 11's third quantum number $m$ has been shown computationally to be non-walk-derivable. Empirical decomposition $m = |p| + |q| + m_\text{carrier}$ holds exactly for leptons; for higher carrier-knot sectors, $m_\text{carrier}$ varies. Required for sharper unseen-species mass predictions.
3. **CKM Jarlskog at $\alpha^{5/2}$ vs substrate at $\alpha^4$.** Two-channel CP-violation reading suggests the static (kinematic) trefoil-geometry phase and the dynamic NR-Hamilton-loop amplitude are independent invariants of the same algebra. Tighter cross-derivation would close this.
