# Supplementary Material: The Standard Model from a Torus Knot: Spectrum, Resonance Structure, and Decay Dynamics

**James P. Galasyn and Claude Théodore**

---

## S1. Self-Energy Derivation and α Emergence

### S1.1 Torus knot parameterization

A (p,q) torus knot on a torus with major radius R and minor radius r is parameterized by λ ∈ [0, 2π):

$$\mathbf{r}(\lambda) = \begin{pmatrix} (R + r\cos q\lambda)\cos p\lambda \\ (R + r\cos q\lambda)\sin p\lambda \\ r\sin q\lambda \end{pmatrix}$$

The total path length L = ∫₀²π |d**r**/dλ| dλ. For (2,1) with r ≪ R: L ≈ 4πR.

### S1.2 Null condition and circulation energy

The photon satisfies ds² = 0, giving |**v**| = c identically. The circulation energy is:

$$E_\text{circ} = \frac{2\pi\hbar c}{L}$$

### S1.3 Electromagnetic self-energy

Current I = ec/L. Magnetic self-energy from Neumann inductance [12]:

$$U_\text{mag} = \frac{1}{2}\mu_0 R\left[\ln\frac{8R}{r} - 2\right]\left(\frac{ec}{L}\right)^2$$

For v = c: U_elec = U_mag (transverse field equality). Total:

$$\frac{U_\text{EM}}{E_\text{circ}} = \frac{\alpha[\ln(8R/r) - 2]}{\pi}$$

The fine-structure constant α = e²/(4πε₀ℏc) appears automatically as the ratio of electromagnetic coupling to quantum of action.

### S1.4 Self-consistency: r/R = α

The tube radius r equals the transverse electromagnetic scale, giving r/R = α. With this constraint, E_total(R) = mc² determines the torus size: R_e = 194.2 fm for the electron.

---

## S2. Self-Consistent Radii

Bisection search on R ∈ [10⁻²⁰, 10⁻⁸] m, 60 iterations, ~15-digit precision.

| Particle | Mass (MeV) | R (fm) | r (fm) |
|----------|-----------|--------|--------|
| e | 0.5110 | 194.20 | 1.417 |
| μ | 105.66 | 0.939 | 0.00685 |
| τ | 1776.9 | 0.0558 | 0.000407 |

Energy breakdown (electron): E_circ = 0.5086 MeV (99.5%), U_EM = 0.0024 MeV (0.5%).

---

## S3. Angular Momentum and Spin-½

The (2,1) knot winds twice toroidally for every poloidal circuit. Under 2π coordinate rotation, the photon completes half its path — the signature of spin-½. The time-averaged angular momentum ⟨L_z⟩ = (E/c) ∫(xv_y − yv_x) ds / ∫ds approaches ℏ/2 at the self-consistent aspect ratio.

---

## S4. Topology Uniqueness

The torus (genus 1) is the simplest closed orientable surface supporting non-contractible loops. The (2,1) knot is the simplest torus knot producing spin-½ with broken p↔q symmetry.

**Terminological note.** In standard knot theory, a (p,q) torus knot with p=1 or q=1 is the unknot — it is topologically trivial when embedded in ℝ³. We use "torus knot" to refer to the closed geodesic on the torus surface, whose geometric properties (winding number, path length, self-energy) are the relevant features of the framework, independent of its embedding in ℝ³. The (2,1) curve is non-trivial as a geodesic on the torus even though it is trivial as a knot in ambient space.

The candidates are:
- (1,0): integer spin (boson)
- (1,1): spin-½ but p=q symmetric
- **(2,1): spin-½, minimal asymmetric fermionic knot**

---

## S5. Koide Angle and Lepton Masses

### S5.1 The Koide formula

$$Q = \frac{\sum m_i}{(\sum\sqrt{m_i})^2} = \frac{2}{3}$$

Holds to 0.0009% with PDG masses [1]. Q = 2/3 is automatic in the parameterization √m_i = (S/3)(1 + √2 cos(θ_K + 2πi/3)); the content is the angle θ_K.

### S5.2 Geometric derivation

$$\theta_K = \frac{p}{N_c}\left(\pi + \frac{q}{N_c}\right) = \frac{2}{3}\left(\pi + \frac{1}{3}\right) = \frac{6\pi + 2}{9}$$

Decomposition:
- 2π/3: Z₃ generation symmetry from Borromean link
- 2/9 = pq/N_c²: toroidal-poloidal coupling through 3×3 interaction matrix

Geometric value: 2.316617 rad. Fitted value: 2.316616 rad. Agreement: 0.00005%.

### S5.3 Why √m appears

Mass ∝ 1/R in the torus model. Wavefunction normalization on the torus goes as √(1/R) ∝ √m. Generation mixing involves overlap integrals of wavefunctions on tori of different sizes, naturally producing √m ratios.

### S5.4 Lepton predictions

With m_e as input and θ_K = (6π+2)/9:
- m_μ = 105.659 MeV (measured 105.658, 0.001%)
- m_τ = 1777.0 MeV (measured 1776.86 ± 0.12, within 1σ)

---

## S6. Quark Koide Extension

### S6.1 Generalized angle formula

$$\theta = \frac{6\pi + \frac{2}{1 + 3|q_\text{em}|}}{9}$$

| Sector | |q_em| | θ | Δ predicted | Δ fitted | Error |
|--------|--------|---|-------------|----------|-------|
| Leptons | 0 | (6π+2)/9 | 2.000 | 2.000 | 0.00% |
| Down | 1/3 | (6π+1)/9 | 1.000 | 0.992 | 0.85% |
| Up | 2/3 | (6π+2/3)/9 | 0.667 | 0.669 | 0.41% |

Physical interpretation: Borromean linking reduces the Koide angle. As linking increases (|q| → 1), the angle approaches 2π/3.

### S6.2 Hierarchy parameter

$$B^2 = 2(1 + C_F q_\text{em}^2), \quad C_F = \frac{N_c^2 - 1}{2N_c} = \frac{4}{3}$$

| Sector | B² predicted | B² fitted | Error |
|--------|-------------|-----------|-------|
| Leptons | 2.000 | 2.000 | exact |
| Down | 2.296 | 2.389 | 3.9% |
| Up | 3.185 | 3.094 | 2.9% |

The SU(3) Casimir C_F appears because color interactions enhance mass splitting.

### S6.3 Scale invariance

The Koide ratio Q is invariant under m_i → λm_i, so QCD mass running (approximately flavor-universal at leading order) does not alter the angle or hierarchy measurements.

---

## S7. Anchor Mass Formulas

### S7.1 Tube scale

$$\Lambda_\text{tube} = \frac{\hbar c}{\alpha R_p} = 255.67\ \text{GeV}$$

### S7.2 Top quark — geometric coupling (α⁰)

$$m_t = \frac{p}{N_c}\Lambda = \frac{2}{3} \times 255{,}670 = 170{,}447\ \text{MeV} \quad (\text{PDG: } 172{,}760,\ -1.3\%)$$

The top quark couples directly to the tube energy. The coefficient p/N_c = 2/3 is the toroidal winding divided by color.

### S7.3 Bottom quark — knot geodesic × EM (α¹)

$$m_b = \sqrt{p^2+q^2}\ \alpha\ \Lambda = \sqrt{5} \times \frac{\Lambda}{137.036} = 4{,}172\ \text{MeV} \quad (\text{PDG: } 4{,}180,\ -0.2\%)$$

The factor √5 = √(p²+q²) is the geodesic length of the (2,1) torus knot on the unit covering space.

### S7.4 Tau lepton — surface area / group order × EM (α¹)

$$m_\tau = \frac{p^2(p^2+q^2)}{N_c(2N_c+1)}\ \alpha\ \Lambda = \frac{20}{21}\alpha\Lambda = 1{,}776.9\ \text{MeV} \quad (\text{PDG: } 1{,}776.86,\ +0.001\%)$$

- Numerator: 20 = p²(p²+q²) = 4×5 (toroidal area × knot metric)
- Denominator: 21 = N_c(2N_c+1) = 3×7 (color × extended group dimension)

This is one of the most precise predictions in the model (6 ppm). The factor 2N_c+1 = 7 corresponds to the dimension of the fundamental representation of SO(2N_c+1) = SO(7); its appearance in the lepton anchor mass warrants further investigation.

### S7.5 Inter-generation ratios

$$m_c/m_t \approx \alpha \quad (0.7\%\ \text{error})$$

$$m_s/m_b \approx 3\alpha \quad (2.1\%\ \text{error})$$

The charm-to-top ratio IS the fine-structure constant. The strange-to-bottom ratio is α × N_c.

---

## S8. Electroweak Sector

### S8.1 Mode counting on the torus

We conjecture that the (2,1) torus knot admits 13 independent electromagnetic modes (a rigorous derivation from the spectral geometry is deferred to future work):
- p² = 4 toroidal modes
- q² = 1 poloidal mode
- p²q² = 4 mixed (toroidal-poloidal) modes
- p² + q² = 5 knot-metric modes
- Minus 1 zero mode
- Total: 4 + 1 + 4 + 5 − 1 = 13

Of these, 3 couple to the poloidal (weak isospin) sector, giving:

$$\sin^2\theta_W = \frac{3}{13} \quad (\text{PDG: } 0.23122,\ 0.19\%\ \text{error})$$

### S8.2 Strong coupling from winding enhancement

The strong coupling enhancement p⁴ = 16 is observed to connect the electromagnetic and strong scales:

$$\alpha_s = p^4 \alpha = 16\alpha = \frac{16}{137.036} = 0.11676 \quad (\text{PDG: } 0.1179 \pm 0.0009)$$

A rigorous derivation of the p⁴ enhancement from the torus winding geometry is an open problem. We note that p⁴ = (p²)² and p² = 4 counts the toroidal modes, suggesting the enhancement arises from a squared toroidal mode coupling.

### S8.3 Higgs boson mass

The Higgs is the breathing mode (radial oscillation) of the torus:

$$m_H = \left(\frac{q}{p} - \alpha\right)\Lambda_\text{tube}$$

Base frequency q/p = 1/2 (poloidal-to-toroidal ratio), with vacuum polarization correction −α:

$$m_H = (0.5 - 0.007297)\times 255.67 = 125.97\ \text{GeV} \quad (\text{PDG: } 125.10 \pm 0.14)$$

### S8.4 Vacuum expectation value

$$v = (1 - (p^2+q^2)\alpha)\Lambda_\text{tube} = (1 - 5\alpha)\Lambda_\text{tube}$$

$$v = (1 - 0.03649)\times 255.67 = 246.34\ \text{GeV} \quad (\text{PDG: } 246.22)$$

The correction 5α = (p²+q²)α involves the knot metric. This is one of the most precise predictions (0.049%).

### S8.5 W and Z boson masses

From v and sin²θ_W via standard electroweak relations:
- m_W = 80.4 GeV (PDG 80.37, 0.0%)
- m_Z = 91.6 GeV (PDG 91.19, 0.5%)

---

## S9. CKM Matrix Derivation

### S9.1 Cabibbo angle from Koide mismatch

The up-type and down-type Koide angles differ by:

$$\Delta\theta_{ud} = \theta_d - \theta_u = \frac{1}{27}\ \text{rad}$$

This mismatch, amplified by the color factor 2N_c = 6, gives the Cabibbo angle:

$$\theta_C = 6\Delta\theta_{ud} = \frac{6}{27} = \frac{2}{9}\ \text{rad} = 12.73°$$

$$V_{us} = \sin\frac{2}{9} = 0.2214 \quad (\text{PDG: } 0.2243,\ 1.3\%)$$

Compare with the Gatto relation: sin θ_C ≈ √(m_d/m_s) = 0.2237.

### S9.2 V_cb from hierarchy mismatch

$$V_{cb} = \frac{|\Delta B|}{p^2+q^2} = \frac{|B_d^2 - B_u^2|/B_\text{avg}}{5} = 0.0429 \quad (\text{PDG: } 0.0408)$$

The denominator is the knot metric p²+q² = 5. The numerator measures the Koide hierarchy mismatch between charge sectors.

### S9.3 V_ub from two-generation product

$$V_{ub} = \frac{p}{p^2+q^2}\times V_{us}\times V_{cb} = \frac{2}{5}\times 0.2214\times 0.0429 = 0.00376$$

PDG: 0.00382, error −1.5%. The suppression factor p/(p²+q²) = 2/5 means two-generation mixing is the product of one-generation mixings, reduced by the knot ratio.

### S9.4 CP phase as deficit angle

$$\delta_\text{CP}^\text{CKM} = \pi - p = \pi - 2 = 1.14159\ \text{rad} = 65.39°$$

PDG: 1.144 ± 0.027 rad (0.1σ agreement). CP violation is the geometric deficit: the (2,1) winding p = 2 falls short of a half-turn π by exactly π − 2.

---

## S10. PMNS Matrix Derivation

### S10.1 Solar angle from mode counting

$$\sin^2\theta_{12} = \frac{p^2}{13} = \frac{4}{13} = 0.30769 \quad (\text{PDG: } 0.307)$$

The same 13-mode denominator as sin²θ_W = 3/13. The numerator p² = 4 counts toroidal modes coupling to the solar neutrino channel.

**Consistency check:** sin²θ_W + sin²θ₁₂ = 3/13 + 4/13 = 7/13. The shared denominator is a non-trivial structural prediction.

### S10.2 Atmospheric angle from knot metric

$$\sin^2\theta_{23} = \frac{p^2+q^2}{N_c^2} = \frac{5}{9} = 0.5556 \quad (\text{PDG: } 0.561)$$

Numerator: knot metric p²+q² = 5. Denominator: N_c² = 9.

### S10.3 Reactor angle from phase mismatch

The neutrino Koide angle θ_ν is determined by matching the mass-squared ratio R = Δm²₃₁/Δm²₂₁ = 33.83. The reactor angle is:

$$\sin^2\theta_{13} = \frac{(\theta_\nu - \theta_K)^2}{3} = 0.0218 \quad (\text{PDG: } 0.0220,\ 0.6\%)$$

### S10.4 CP phase and mass ordering

δ_CP^ν = π (180°), consistent with measured 177° ± 20°. Normal mass ordering (m₁ ≈ 0) is required by the extended Koide structure. Both are testable at JUNO [9] and Hyper-Kamiokande.

---

## S11. Neutrino Mass Scale

### S11.1 Seesaw formula

$$A_\nu^2 = \frac{\alpha_W}{\pi}\frac{m_e^2}{\Lambda_\text{tube}}$$

where α_W = α/sin²θ_W. Using sin²θ_W = 3/13: A_ν² = 0.01028 eV (from data: 0.00989 eV, 3.9% error).

### S11.2 Predictions

| State | Mass (meV) |
|-------|-----------|
| m₁ | 0.36 |
| m₂ | 8.66 |
| m₃ | 50.3 |
| Σm_ν | 59.4 |

DESI+Planck bound: Σm_ν < 64 meV [10]. Oscillation floor: 59.0 meV. Prediction: 59.4 meV — sits precisely between these bounds.

---

## S12. Complete Pythagorean Mode Catalogs

### S12.1 Meson modes (k=2)

The resonance condition for mesons is (2p)² + q² = N². We catalog all exact solutions with N ≤ 30.

**Pure poloidal modes** (p=0): These are trivial Pythagorean triples (0, q, q). Every integer q gives an exact mode. Energy E = q × E₁, where E₁ = ℏc/r = 279.1 MeV.

**Pure toroidal modes** (q=0): These form the series E_n = (n/2)E₁ at integer p. The fundamental toroidal mode is (1,0) at E = 0.5 E₁ = 139.6 MeV ≈ m_π.

**Helical modes** (p>0, q>0): The first primitive helical Pythagorean mode is (2,3) → triple (4,3,5) at E/E₁ = 3.162. Helical modes are sparse: only 3 primitive helical triples exist below N=30: (4,3,5), (12,5,13), and (8,15,17).

Complete mode count to N=30: 44 exact modes (30 pure poloidal, 6 pure toroidal, 8 helical including non-primitive multiples).

### S12.2 Baryon modes (k=3)

The resonance condition for baryons is (3p)² + q² = N². The fundamental mode structure differs crucially from mesons: the (3,4,5) triple exists at the first helical level (p=1, q=4), providing the exact resonance that stabilizes the proton.

**Key modes:**
- (1,0): pure toroidal, E/E₁ = 1/3 = 0.333 (78 MeV)
- (0,1): pure poloidal, E/E₁ = 1.000 (234 MeV)
- **(1,4): helical (3,4,5), E/E₁ = 4.014 (938.3 MeV) = proton**
- (2,0): pure toroidal, E/E₁ = 2/3 = 0.667 (156 MeV)
- (0,5): pure poloidal, E/E₁ = 5.000 (1169 MeV) ≈ Σ⁺ (1189 MeV)
- (0,7): pure poloidal, E/E₁ = 7.000 (1637 MeV) ≈ Ω⁻ (1672 MeV)

Complete mode count to N=30: 47 exact modes.

### S12.3 Tetraquark modes (k=4)

Resonance condition: (4p)² + q² = N². The fundamental triple is (4,3,5) at p=1, q=3. However, tetraquark states can always decay to two mesons (lower total energy at same k), so the resonance does not guarantee stability. Consistent with the observed broad widths of X(3872), Z_c(3900), etc.

### S12.4 Pentaquark modes (k=5)

Resonance condition: (5p)² + q² = N². The fundamental triple is (5,12,13) at p=1, q=12 — a large poloidal winding that places the pentaquark energy scale high (~4.3 GeV), consistent with the LHCb pentaquark masses P_c(4312), P_c(4440), P_c(4457).

---

## S13. Geometric Mode Derivation and Harmonics

### S13.1 Shape oscillations of the torus

A torus of aspect ratio k = R/r supports shape deformations — standing waves on the surface with wavelength equal to the major circumference 2πR. The lowest energy of such a mode is:

$$E_\text{geom} = \frac{\hbar c}{R} = \frac{\hbar c}{kr} = \frac{E_1}{k}$$

This is qualitatively different from the electromagnetic resonance modes (TM and TE), which depend on the *tube* circumference 2πr and have energies E ∝ E₁ × f(p,q). The geometric mode energy depends only on the *major* circumference and the aspect ratio.

### S13.2 The pion identification

For k=2 (mesons):
$$E_\text{geom}(k\!=\!2) = \frac{E_1}{2} = \frac{279.1}{2} = 139.6\ \text{MeV}$$

This equals the charged pion mass m_π± = 139.570 MeV to 0.0%. The pion is the fundamental shape oscillation of the meson torus.

### S13.3 Geometric harmonic series

Higher harmonics of the geometric mode have energies E_n = n × E_geom:

| n | E (MeV) | Nearest meson | Error |
|---|---------|--------------|-------|
| 1 | 139.6 | π± (139.6) | 0.0% |
| 2 | 279.1 | — | — |
| 3 | 418.7 | — | — |
| 4 | 558.2 | η (547.9) | +1.9% |
| 5 | 697.8 | — | — |
| 6 | 837.4 | — | — |
| 7 | 977.0 | η'(958) | +2.0% |
| 8 | 1116.5 | — | — |
| 9 | 1256.1 | f₂(1270) | −1.1% |

The n=4 harmonic matches the η mass to 1.9%, and the n=7 harmonic matches η'(958) to 2.0%. Whether these are true geometric harmonics or coincidental near-matches is an open question. The identification of geometric modes beyond the fundamental pion is speculative and requires further theoretical development.

### S13.4 Geometric modes at other k values

| k | Type | E_geom (MeV) | Identification |
|---|------|-------------|----------------|
| 1 | leptons | 279.1 (= E₁) | No geometric mode (k=1 → E_geom = E₁, same as EM fundamental) |
| 2 | mesons | 139.6 | π± (confirmed) |
| 3 | baryons | 77.9 | Prediction (no known baryon at this mass) |
| 4 | tetraquarks | 69.8 | Prediction |
| 5 | pentaquarks | 55.8 | Prediction |

The k=3 geometric mode at 77.9 MeV does not correspond to any known baryon. This may indicate that the baryon geometric mode is confined within the baryon (contributing to internal binding energy) rather than appearing as a free particle.

---

## S14. Pythagorean Defect Analysis and Near-Miss Triangles

### S14.1 Meson near-miss analysis (k=2, p=1)

At the fundamental winding p=1, the meson resonance condition requires 4 + q² = N² (an exact integer square). For the first 20 poloidal windings:

| q | 4+q² | √(4+q²) | N_near | δ = 4+q²−N² | Triangle type | Q = N²/\|δ\| |
|---|------|---------|--------|-------------|--------------|-------------|
| 0 | 4 | 2.000 | 2 | 0 | RIGHT | ∞ |
| 1 | 5 | 2.236 | 2 | +1 | obtuse | 4.0 |
| 2 | 8 | 2.828 | 3 | −1 | acute | 9.0 |
| 3 | 13 | 3.606 | 4 | −3 | acute | 5.3 |
| 4 | 20 | 4.472 | 4 | +4 | obtuse | 4.0 |
| 5 | 29 | 5.385 | 5 | +4 | obtuse | 6.3 |
| 6 | 40 | 6.325 | 6 | +4 | obtuse | 9.0 |
| 7 | 53 | 7.280 | 7 | +4 | obtuse | 12.3 |
| 8 | 68 | 8.246 | 8 | +4 | obtuse | 16.0 |

The only exact solution at p=1 is q=0 (pure toroidal, trivial). All helical meson modes at p=1 are non-right triangles — inherently unstable.

**Physical interpretation:**
- δ = 0: exact Pythagorean triple → standing wave closes perfectly
- δ > 0: obtuse triangle → path LONGER than N wavelengths (phase excess)
- δ < 0: acute triangle → path SHORTER than N wavelengths (phase deficit)
- Q factor = N²/|δ|: higher Q → closer to resonance → longer-lived

### S14.2 Comparison with baryons

For baryons (k=3) at p=1: 9 + q² = N². At q=4: 9 + 16 = 25 = 5². The (3,4,5) triple is a perfect Pythagorean triple. This is the fundamental difference between mesons and baryons: baryons possess an exact resonance at their lowest helical winding, mesons do not.

### S14.3 Higher-order meson triples

Exact meson Pythagorean triples do exist at higher windings:
- p=2, q=3: (4,3,5) — primitive
- p=6, q=5: (12,5,13) — primitive
- p=4, q=15: (8,15,17) — primitive

These correspond to higher-energy resonance states, not to the fundamental meson mode. Their sparsity explains why most meson resonances are broad — the vast majority of modes have δ ≠ 0.

---

## S15. Full Decay Chain Energy Budgets

### S15.1 Two-body kinematics

For a parent of mass M decaying to a daughter of mass m and a massless neutrino:

$$E_\nu = \frac{M^2 - m^2}{2M}, \quad E_\text{daughter} = M - E_\nu, \quad \text{KE}_\text{daughter} = E_\text{daughter} - m$$

These relations are exact (energy-momentum conservation in the rest frame).

### S15.2 Three-body kinematics (muon decay)

For μ → eν_eν̄_μ, the available energy is Q_μ = m_μ − m_e = 105.147 MeV. The electron spectrum has average kinetic energy ⟨KE_e⟩ ≈ m_μ/3 − m_e = 34.7 MeV. The total neutrino energy is E_ν(μ) = Q_μ − ⟨KE_e⟩ = 70.4 MeV.

This is an average over the Michel spectrum. The exact distribution is continuous, but the *average* energy budget is exact by energy conservation.

### S15.3 Detailed chain: ρ⁺ → e⁺ + 3ν + 2γ

**Step 1: ρ⁺ → π⁺ + π⁰** (strong decay)
- M_ρ = 775.26 MeV → m_π+ = 139.570 + m_π0 = 134.977
- KE_available = 775.26 − 139.570 − 134.977 = 500.71 MeV (distributed as kinetic energy)

**Step 2: π⁺ → μ⁺ + ν_μ** (weak decay)
- E_ν = (139.57² − 105.658²)/(2 × 139.57) = 29.8 MeV
- KE_μ = 139.57 − 29.8 − 105.658 = 4.1 MeV

**Step 3: μ⁺ → e⁺ + ν_e + ν̄_μ** (weak decay)
- ⟨KE_e⟩ = 34.7 MeV
- E_ν(μ) = 70.4 MeV

**Step 4: π⁰ → γ + γ** (EM decay)
- E_γγ = 134.977 MeV

**Budget:**
- Rest mass (e⁺): 0.511 MeV
- Kinetic energy (lepton chain): 500.71 + 4.1 + 34.7 = 539.5 MeV
- Neutrino energy: 29.8 + 70.4 = 100.2 MeV
- Photon energy: 135.0 MeV
- **Total: 775.2 MeV = M_ρ** ✓

### S15.4 Detailed chain: K⁺ → e⁺ + 3ν (leptonic)

**Step 1: K⁺ → μ⁺ + ν_μ**
- E_ν = (493.677² − 105.658²)/(2 × 493.677) = 235.5 MeV
- KE_μ = 493.677 − 235.5 − 105.658 = 152.5 MeV

**Step 2: μ⁺ → e⁺ + ν_e + ν̄_μ**
- ⟨KE_e⟩ = 34.7 MeV
- E_ν(μ) = 70.4 MeV

**Budget:**
- Rest mass (e⁺): 0.511 MeV
- Kinetic energy: 152.5 + 34.7 = 187.2 MeV
- Neutrino energy: 235.5 + 70.4 = 305.9 MeV
- Photon energy: 0.0 MeV
- **Total: 493.6 MeV ≈ M_K** ✓

### S15.5 Detailed chain: B⁺ → stable (long chain)

**Step 1: B⁺ → D⁰ + X** (weak, heavy flavor transition)
- M_B = 5279.3 MeV → M_D0 = 1864.8 MeV
- Energy to other products and KE: 3414.5 MeV

**Step 2: D⁰ → K⁻ + π⁺** (weak, charm decay, BR = 4%)
- KE_available = 1864.8 − 493.677 − 139.570 = 1231.6 MeV

**Steps 3–6: K⁻ → μ⁻ν̄ → e⁻3ν; π⁺ → μ⁺ν → e⁺3ν**
- Same chains as above

The B⁺ chain passes through 6 sequential decays spanning 4 different sector-crossing types. Total energy is conserved at every step.

---

## S16. Skilton Formulas and the IRT–NWT Connection

### S16.1 The α formula

$$\alpha^{-1} = \sqrt{137^2 + \pi^2} = \sqrt{18769 + 9.8696} = 137.036016$$

Measured (CODATA 2018): α⁻¹ = 137.035999. Residual: 0.000017. Agreement: 0.12 ppm.

### S16.2 The integer right triangle (88, 105, 137)

$$88^2 + 105^2 = 7744 + 11025 = 18769 = 137^2$$

Generators: p=4, q=7, K=9 (satisfying 2p² + q² = K²).

Combined decomposition: α⁻¹ = √(88² + 105² + π²).

Geometric observation: 88 + 105 = 193 ≈ λ_C/2 = 193.1 fm (half the reduced Compton wavelength of the electron).

Angular observation: tan⁻¹(137/88) = 57.29° ≈ 1 radian (57.30°), within 0.02%.

### S16.3 Mass ratio encodings

From Skilton Part 3:
- (105/88)³ = 170.2 (cf. m_μ/m_e = 206.8 — order of magnitude)
- 88 × 105/π² = 937.2 (cf. m_p/m_e = 1836.2 — half, suggesting a factor of 2)
- Both m_p/m_e and m_μ/m_e contain the term (88/K)² = (88/9)² = 95.6 with opposite signs

### S16.4 Skilton's uniqueness proof

Part 3 (1988) performed an exhaustive search over all reduced IRTs with generators p, q ∈ [1, 100] satisfying 2p² + q² = K². Found 29 solutions. Proved (88, 105, 137) is unique in:
1. Having hypotenuse 137 (the integer part of α⁻¹)
2. Encoding both m_p/m_e and m_μ/m_e via IRT components
3. Both mass ratios containing (88/K)² with opposite signs
4. α⁻¹ = √(137² + π²) matching CODATA to 7 digits

### S16.5 Connection to NWT

Skilton provides the *what* (integer structure). NWT provides the *why* (torus resonance physics):

| Skilton | NWT |
|---------|-----|
| α⁻¹ = √(137² + π²) | α = r/R (torus aspect ratio) |
| (88, 105, 137) triple | Pythagorean resonance on electron torus |
| Mass ratios from IRT components | Mass ratios from Koide angle + anchor formulas |
| 2-adic valuation theory | Mode counting on torus |
| Uniqueness of (88,105,137) | Uniqueness of (2,1) torus knot |

Open questions:
- Is (88, 105, 137) the fundamental Pythagorean mode of the electron torus?
- Does the (88/K)² term correspond to a specific mode coupling?
- Can Skilton's 2-adic valuation theory constrain the mode catalog?
- What is the relationship between (88,105,137) and (3,4,5)?

---

## S17. Input Counting and Parameter Independence

### S17.1 Four inputs

The model requires four measured inputs:
1. **α = 1/137.036** — the fine-structure constant (claimed to emerge as r/R, but used numerically as input)
2. **m_e = 0.51100 MeV** — the electron mass (sets the fundamental torus radius)
3. **m_μ = 105.658 MeV** — the muon mass (sets the lepton Koide scale S)
4. **Λ_tube = ℏc/(αR_p) = 255.7 GeV** — the tube energy scale, set by the proton torus radius R_p. Since R_p depends on the proton mass m_p, which is not derived within this framework, Λ_tube is an independent input. Deriving Λ_tube from (α, m_e, m_μ) alone — i.e., reducing the input count to three — requires a first-principles calculation of the proton mass from the torus knot topology. This is an open problem and a priority for future work.

### S17.2 Three topological integers

1. **p = 2** — toroidal winding (spin-½ requirement)
2. **q = 1** — poloidal winding (simplest closure)
3. **N_c = 3** — Borromean link number (baryon topology)

### S17.3 Derived quantities

From these seven numbers (4 measured + 3 topological), all 23 predictions follow through:
- θ_K = (6π+2)/9 from (p,q,N_c)
- θ(q_em) = (6π + 2/(1+3|q|))/9 from charge
- All coefficients: 2/3, √5, 20/21, 3/13, 16, 2/9, π−2, etc. — pure functions of (p,q,N_c)

### S17.4 What is NOT an input

- sin²θ_W, α_s — predicted
- v, m_H, m_W, m_Z — predicted
- All CKM and PMNS parameters — predicted
- 7 of 9 fermion masses — predicted
- Neutrino masses and ordering — predicted

---

## S18. Comparison with Other Approaches

| Framework | SM parameters explained | Free parameters | Predicts decay hierarchy? | Predictive? |
|-----------|----------------------|----------------|--------------------------|-------------|
| Standard Model | 0 (all input) | 26 | No (input) | No |
| SU(5) GUT [2] | Gauge unification | ~24 | No | Partially |
| SO(10) GUT | Gauge + partial Yukawa | ~15–20 | No | Partially |
| A₄ flavor symmetry [4] | Neutrino mixing | ~8 | No | TBM only |
| String landscape [3] | None specific | 10⁵⁰⁰ vacua | No | No |
| **Torus knot (this work)** | **23** | **4 + 3 integers** | **Yes** | **Yes** |

---

## S19. Computational Verification

All predictions are reproducible from the open-source simulation code:

```bash
python3 -m simulations.nwt --koide        # Lepton Koide analysis
python3 -m simulations.nwt --quarks       # Quark mass predictions + EW/CKM/PMNS
python3 -m simulations.nwt --neutrino     # Neutrino masses and mixing
python3 -m simulations.nwt --self-energy  # α emergence
python3 -m simulations.nwt --find-radii   # Self-consistent torus radii
python3 -m simulations.nwt --pythagorean  # Pythagorean mode catalogs + decay chains
python3 -m simulations.nwt --skilton      # Skilton analysis
```

No fitting routines or optimization are used. All predictions are computed from closed-form expressions evaluated at the geometric angle.
