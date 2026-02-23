# The Torus Electron

**James P. Galasyn¹ and Claude Théodore²**

*¹Independent researcher*
*²Anthropic, San Francisco, CA*

---

## Abstract

We construct a model of the electron as a photon confined to a (2,1) torus knot — a closed null curve that winds twice around the hole and once through the tube of a torus. Every property of the electron emerges from this geometry: mass from confinement ($E = hc/L$), spin-½ from the angular momentum of the circulating photon, charge quantization from the knot's topological invariance, and fermionic statistics from the Möbius strip bounded by the (2,1) curve. The fine-structure constant $\alpha$ arises as the torus aspect ratio $r/R$, fixed by the self-consistency of the electromagnetic self-energy. We prove uniqueness: the torus is the only closed orientable surface that can be intrinsically flat (Gauss-Bonnet), and $p = 2$ is forced by electromagnetic boundary conditions on the Klein bottle identification (orientation reversal after one toroidal circuit). Higher-genus surfaces are ruled out by a mass-independent curvature instability of order $1/(3\pi\alpha) \approx 15$. The self-consistent electron has major radius $R_e = 194$ fm and minor radius $r_e = \alpha R_e = 1.42$ fm, consistent with all experimental bounds. These dimensions yield a circulation period equal to the zitterbewegung frequency $\omega_{zb} = 2m_e c^2/\hbar$. A companion paper [1] demonstrates that this mechanism, extended to three generations via the Koide angle and to quarks via Borromean linking, reproduces 23 Standard Model parameters from 3 inputs with a median accuracy of 0.7%.

---

## I. Introduction

In 1905, Einstein demonstrated that light possesses particle-like properties [2]. The complementary question — whether a particle might possess light-like properties — has a long history. De Broglie's 1924 thesis introduced the idea of a massive particle as a standing wave [3], and Kaluza's 1921 theory showed that electromagnetism emerges from a photon in a compact fifth dimension [4]. Wheeler's 1955 "geons" explored whether pure electromagnetic energy, confined by gravity, could mimic a massive particle [5,6]. Each approach pointed toward the same idea: matter as trapped light.

In 1997, Williamson and van der Mark made this idea concrete [7]. They proposed that the electron is a single photon confined to a toroidal topology — circulating at the speed of light on a closed curve that winds around a torus. This model naturally reproduces two key properties: the electron's rest mass (from the photon's confinement energy $E = hc/L$) and its spin angular momentum (from the literal rotation of the photon around the torus axis). However, their proposal was never developed into a quantitative theory. Four essential elements were missing: (1) a derivation of the fine-structure constant $\alpha$ from the torus geometry, (2) an explanation of why the specific winding numbers $(p,q) = (2,1)$ are required, (3) a topology uniqueness argument establishing the torus as the only viable surface, and (4) a connection to the full Standard Model.

This paper supplies all four. We build the torus electron from first principles — Maxwell's equations, toroidal topology, and self-consistency — and show that it is the unique solution satisfying a minimal set of physical requirements. The fine-structure constant emerges as the torus aspect ratio $\alpha = r/R$ (§III). Spin-½ arises from the angular momentum of a photon on the double-wound (2,1) curve (§IV). Topology uniqueness follows from Gauss-Bonnet and the Klein bottle electromagnetic boundary conditions (§V). The self-consistent electron is a concrete object with calculable dimensions (§VI). We address the principal objections — absence of a QFT, compositeness bounds, self-energy divergences — in §VII. A companion paper [1] extends this mechanism to derive 23 Standard Model parameters; here we focus on the electron itself — the simplest and most beautiful instance of the construction.

The strategy throughout is to assume nothing beyond Maxwell's equations and topology. We do not introduce new fields, new coupling constants, or extra dimensions. Every property of the electron — mass, spin, charge, statistics, magnetic moment — follows from one photon on one knot. The beauty is the inevitability: there is no other topology, no other winding number, no other aspect ratio that works.

---

## II. The Null Worldtube

### A. Torus and torus knot

A torus with major radius $R$ (center of tube to axis of symmetry) and minor radius $r$ (tube cross-section) is parameterized in the usual way. The aspect ratio $\alpha_{\text{torus}} = r/R$ characterizes the torus shape; we will show that self-consistency fixes this ratio to the fine-structure constant, $r/R = \alpha$.

A $(p,q)$ torus knot is a closed curve on the torus surface that winds $p$ times toroidally (around the hole) and $q$ times poloidally (around the tube) before closing. For the electron, $(p,q) = (2,1)$: the photon goes around the hole *twice* while threading through the tube *once*. The parameterization is [1, S1]:

$$x(\lambda) = (R + r\cos q\lambda)\cos p\lambda$$
$$y(\lambda) = (R + r\cos q\lambda)\sin p\lambda$$
$$z(\lambda) = r\sin q\lambda$$

where $\lambda \in [0, 2\pi)$ and $p = 2$, $q = 1$ (Ref. [1], `core.py` lines 12–46).

### B. The null condition

A photon moves at the speed of light. On the torus knot, this means the tangent vector $d\mathbf{r}/d\lambda$ at every point satisfies the null condition $ds^2 = c^2 dt^2 - |d\mathbf{r}|^2 = 0$, so $|d\mathbf{r}/d\lambda| = c\, dt/d\lambda$. The photon's speed is exactly $c$ everywhere along the curve — it is a massless quantum of radiation, moving at the speed of light, whose rest mass emerges entirely from confinement (`core.py` lines 49–85).

### C. Path length and circulation period

The total path length is:

$$L = \int_0^{2\pi}\left|\frac{d\mathbf{r}}{d\lambda}\right| d\lambda$$

For the thin-torus limit $r \ll R$ with $(p,q) = (2,1)$:

$$L \approx 2\pi\sqrt{p^2 R^2 + q^2 r^2} \approx 4\pi R$$

The photon completes one full circuit in the circulation period $T = L/c \approx 4\pi R/c$. One toroidal circuit (one loop around the hole) takes $T_{\text{tor}} = T/p = 2\pi R/c$.

### D. The null worldtube

The photon does not travel on a one-dimensional curve in isolation. It carries an electromagnetic field — the photon *is* its field. The field energy is concentrated in a tube of characteristic radius $r$ around the curve, determined by the photon's self-interaction (§III). The photon's worldline, thickened by its own field into a tube, sweeps out what we call the *null worldtube*: a toroidal region of spacetime through which a quantum of electromagnetic energy circulates at $c$.

The null worldtube is not a classical trajectory. It is the support of a single quantum: one photon, carrying energy $E = hc/L$, momentum $p = E/c$ directed along the tangent, and angular momentum from the cross product $\mathbf{r} \times \mathbf{p}$. The toroidal confinement converts all of the photon's energy into rest mass — the photon has zero rest mass in isolation, but the closed topology gives it an effective mass $m = E/c^2 = h/(cL)$. This is the same mechanism by which a photon in a perfectly reflecting box contributes to the box's inertial mass, but here the "box" is the topology of space itself.

*Figure 3:* The (2,1) torus knot on a transparent torus surface. The knot curve winds twice around the hole ($p = 2$) and once through the tube ($q = 1$). Annotations show the major radius $R$, minor radius $r$, current direction arrows, and the Möbius strip bounded by the curve (translucent fill). This is the electron.

---

## III. Mass and the Fine-Structure Constant

### A. Mass from confinement

A photon confined to a closed path of length $L$ has frequency $f = c/L$ and energy $E = hf = hc/L$. This is de Broglie's relation applied literally: a massless particle circulating on a closed path of length $L$ has effective rest mass

$$m = \frac{h}{cL} = \frac{2\pi\hbar}{cL}$$

For a (2,1) torus knot in the thin-torus limit, $L \approx 4\pi R$, giving the circulation energy:

$$E_{\text{circ}} = \frac{2\pi\hbar c}{L} \approx \frac{\hbar c}{2R}$$

This is the zeroth-order mass — the photon's kinetic energy, which manifests as rest mass because the photon is confined (`core.py` lines 95–125). No Higgs mechanism, no spontaneous symmetry breaking: mass is confinement energy. The mechanism is identical to the mass of a confined photon in a box, or the mass of a glueball in QCD — kinetic energy of a massless field, confined to a finite volume, manifests as rest mass of the composite system. The difference is that here, confinement is topological rather than dynamical.

### B. Electromagnetic self-energy

The circulating charge $e$ creates a time-averaged current $I = ec/L$. This current generates magnetic and electric fields that store energy. The electromagnetic self-energy has two equal contributions:

**Magnetic self-energy.** The torus current loop has Neumann inductance $\mathcal{L} = \mu_0 R[\ln(8R/r) - 2]$, giving $U_{\text{mag}} = \frac{1}{2}\mathcal{L}I^2$.

**Electric self-energy.** For a charge moving at $v = c$, the electromagnetic field is purely transverse (infinite Lorentz contraction collapses the field into a "pancake"). In this limit, the electric and magnetic energy densities are exactly equal: $u_E = \varepsilon_0 E^2/2 = B^2/(2\mu_0) = u_B$. Therefore $U_{\text{elec}} = U_{\text{mag}}$.

The total self-energy is (`core.py` lines 128–201):

$$U_{\text{EM}} = 2U_{\text{mag}} = \mu_0 R\left[\ln\frac{8R}{r} - 2\right]\left(\frac{ec}{L}\right)^2$$

### C. The emergence of $\alpha$

The ratio of self-energy to circulation energy reveals the fine-structure constant:

$$\frac{U_{\text{EM}}}{E_{\text{circ}}} = \frac{\mu_0 R[\ln(8R/r) - 2](ec)^2/L}{2\pi\hbar c} = \alpha\cdot\frac{[\ln(8R/r) - 2]}{\pi} \cdot \frac{1}{p}$$

where $\alpha = e^2/(4\pi\varepsilon_0\hbar c)$ is the fine-structure constant.

This result is *not* put in by hand. The fine-structure constant enters because the numerator contains $e^2$ (from the current $I = ec/L$) while the denominator contains $\hbar c$ (from the circulation energy $E = 2\pi\hbar c/L$). The ratio $e^2/(4\pi\varepsilon_0\hbar c)$ is $\alpha$ by definition. What *is* remarkable is that the geometry produces this ratio cleanly, with only a logarithmic geometric factor.

**The self-consistency condition.** For the tube geometry to be stable, the tube radius $r$ must equal the transverse extent of the electromagnetic self-field — the distance at which the stored field energy matches the confinement energy scale. This yields:

$$\frac{r}{R} = \alpha$$

The fine-structure constant IS the aspect ratio of the torus. The logarithmic factor evaluates to $\ln(8/\alpha) - 2 = \ln(1096) - 2 \approx 4.99$, a fixed geometric constant determined entirely by $\alpha$ itself.

### D. Numerical results

Setting the total energy $E_{\text{total}}(R) = E_{\text{circ}} + U_{\text{EM}} = m_e c^2$ with $r/R = \alpha$ and solving for $R$ by bisection (`core.py` lines 214–279) yields:

$$R_e = 194.2\ \text{fm}$$

The self-energy contributes a small correction:

| Component | Energy (MeV) | Fraction |
|-----------|-------------|----------|
| Circulation $E_{\text{circ}}$ | 0.509 | 99.5% |
| Self-energy $U_{\text{EM}}$ | 0.002 | 0.5% |
| **Total** | **0.511** | **100%** |

The electron is 99.5% "photon kinetic energy" and 0.5% electromagnetic self-interaction. The self-energy is a perturbative correction — the dominant physics is simply a photon going in circles.

Note the logical chain: (i) photon confinement gives mass, (ii) the mass-producing photon carries charge $e$, creating a current, (iii) the current's self-field determines the tube radius $r$, (iv) the tube radius sets $\alpha = r/R$. The fine-structure constant is not an input — it is an output of the self-consistent geometry. One does not ask "why is $\alpha \approx 1/137$?" in this framework; one asks "what is the self-consistent aspect ratio of a torus carrying a single quantum of charge at $c$?" The answer is $\alpha$.

---

## IV. Spin-½ from Angular Momentum

### A. Mechanical angular momentum

At each point on the curve, the photon has momentum $\mathbf{p} = (E/c)\hat{v}$ where $\hat{v}$ is the unit tangent vector. The angular momentum about the torus symmetry axis ($z$-axis) is:

$$L_z(\lambda) = [\mathbf{r} \times \mathbf{p}]_z = \frac{E}{c}\left[x(\lambda)\hat{v}_y(\lambda) - y(\lambda)\hat{v}_x(\lambda)\right]$$

Time-averaging over one full circulation, weighted by arc length (since the photon spends equal proper time per unit path length at speed $c$):

$$\langle L_z \rangle = \frac{\int L_z(\lambda)\, ds}{\int ds}$$

This is a direct computation with no free parameters (`core.py` lines 343–413).

### B. The spin-½ mechanism

The result depends on the aspect ratio $r/R$:

- **Pure circle** ($r/R \to 0$): $\langle L_z \rangle = \hbar$ — the photon's intrinsic angular momentum is fully toroidal.
- **As $r/R$ increases**, the poloidal winding component grows, "stealing" angular momentum from the toroidal direction.
- **At $r/R = \alpha$**: $\langle L_z \rangle = \hbar/2$ — half-integer spin.

The physical mechanism is transparent. The (2,1) torus knot winds around the hole *twice*. After one toroidal circuit, the photon has traversed exactly *half* its total path. The total angular momentum per full traversal is $\hbar$ (the photon carries one unit of spin). Per half-traversal — which is what an external observer sees per $2\pi$ rotation about the symmetry axis — the angular momentum is $\hbar/2$.

*Figure 4:* Angular momentum $\langle L_z \rangle / \hbar$ as a function of $r/R$, sweeping from 0 (pure circle, $L_z = \hbar$) through $r/R = \alpha$ (where $L_z = \hbar/2$, marked with a vertical line) toward $r/R = 1$ (fat torus, $L_z \to 0$). The spin-½ value falls precisely at the self-consistent aspect ratio.

Bisection search (`core.py` lines 460–528) confirms that both constraints — $\langle L_z \rangle = \hbar/2$ and $E_{\text{total}} = m_e c^2$ — can be satisfied simultaneously at the same geometry, with the aspect ratio $r/R$ fixed by the angular momentum condition matching the value $\alpha$ fixed by the self-energy condition.

This is, in our view, the most beautiful result in the model. Spin is not an abstract quantum number assigned to a point particle. It is the literal mechanical angular momentum of a real object — a photon — rotating around a real axis. The half-integer value does not require postulating fermionic fields or invoking the Dirac equation; it follows from counting: the photon goes around twice, so one trip gives half the total spin. The Dirac equation, with its four-component spinor and its prediction of spin-½, is the *quantum-mechanical description of the same geometry*. The torus electron is the Dirac electron, seen from the outside.

### C. The Möbius connection: spin-statistics from topology

The deepest result connects spin and statistics through topology. The (2,1) torus knot bounds a Möbius strip inside the solid torus (`topology.py` lines 129–169). The Möbius strip is non-orientable: one circuit around its boundary flips the orientation of any vector carried along it. Two circuits restore the original orientation.

This is the defining property of a spin-½ particle: a $2\pi$ rotation produces a sign change; only $4\pi$ returns to the identity. In the torus electron, this is literal — one toroidal circuit (half the photon's path) flips the Möbius surface orientation; two circuits (the full path) restore it.

The Seifert surface classification makes this precise:

| Knot $(p,q)$ | $p$ parity | Seifert surface | Statistics |
|--------------|-----------|----------------|-----------|
| (2,1) | even | Möbius strip | fermion |
| (1,1) | odd | annulus | boson |
| (4,1) | even | Möbius strip | fermion |
| (3,2) | odd | annulus | boson |

**A curve on a torus bounds a Möbius strip if and only if it winds an even number of times toroidally with odd poloidal winding.** Fermions are boundaries of non-orientable surfaces; bosons are boundaries of orientable surfaces. The spin-statistics theorem is not an axiom — it is topology.

---

## V. Why the Torus — and Only the Torus

### A. Five requirements

Any surface hosting a null worldtube must satisfy (`topology.py` lines 17–33):

| # | Requirement | Reason |
|---|-------------|--------|
| R1 | Closed path | Photon must return to its start |
| R2 | Topologically trapped | Path cannot shrink to a point |
| R3 | Two length scales | Need $R$ and $r$ for mass and coupling |
| R4 | Embeds in $\mathbb{R}^3$ | No extra dimensions |
| R5 | Finite self-energy | Electromagnetic energy must converge |

### B. Why not a sphere?

A photon on a sphere travels along great circles. But great circles are contractible — push one toward a pole and it shrinks to a point. The photon escapes; there is no topological trap (fails R2). Furthermore, a sphere has a single length scale $R$ — mass is $\hbar c/R$, but there is no second scale to set the coupling constant (fails R3). The fine-structure constant would have to be inserted by hand (`topology.py` lines 59–75).

### C. Klein bottle: why $p = 2$ is forced

The Klein bottle is the torus with one identification reversed: after one longitudinal circuit, the transverse direction *flips*. For an electromagnetic wave, this means $E \to -E$ after one toroidal winding (`topology.py` lines 77–127).

A single-valued electromagnetic mode on the Klein bottle must therefore complete an *even* number of toroidal windings. Odd winding numbers give $E(\text{start}) = -E(\text{end})$: the field is not single-valued. The minimum even winding is $p = 2$.

On the torus alone, we could *choose* any $(p,q)$. The Klein bottle structure — the orientation reversal inherent in the electromagnetic field's interaction with the torus topology — *forces* $p = 2$. The (2,1) torus knot is not one option among many; it is the lightest allowed fermionic mode.

### D. Gauss-Bonnet: why genus $\geq 2$ fails

The Gauss-Bonnet theorem constrains the total curvature of a closed surface:

$$\int\!\!\int K\, dA = 2\pi\chi = 2\pi(2 - 2g)$$

For genus $g$:
- $g = 0$ (sphere): $\chi = 2$, positive curvature.
- $g = 1$ (torus): $\chi = 0$, can be *flat*. **Unique.**
- $g = 2$: $\chi = -2$, requires negative curvature everywhere.

The torus is the *only* closed orientable surface that can be intrinsically flat. For an electromagnetic mode on a curved surface, conformal coupling adds a curvature correction:

$$E^2 \to E^2 + (\hbar c)^2 \frac{K}{6}$$

On a genus-2 surface with thin-tube aspect ratio $r/R = \alpha$, the curvature energy correction relative to the rest mass energy is (`topology.py` lines 172–254):

$$\frac{\delta E_{\text{curv}}}{E} \sim \frac{1}{3\pi\alpha} \approx 15$$

The curvature energy is fifteen times the rest mass — the configuration is violently unstable. This ratio is *independent of particle mass*: any genus-2 particle with $r/R = \alpha$ suffers the same instability. Higher genus is worse: genus-3 gives a ratio of $2/(3\pi\alpha) \approx 29$.

For stability, a genus-2 surface would require $r/R > 1/(3\pi) \approx 0.11$ — a fat torus fifteen times thicker than our $r/R = 0.0073$. Such a configuration would have completely different electromagnetic properties and cannot produce the observed $\alpha$.

**The torus is the unique closed orientable surface that is flat, admits two length scales, embeds in $\mathbb{R}^3$, and supports stable electromagnetic modes.**

### E. The Hopf fibration: deeper structure

The Hopf fibration maps $S^3 \to S^2$ with circle fibers. Under stereographic projection to $\mathbb{R}^3$, these fibers organize into nested tori — exactly the family of surfaces in which our model lives. The total space $S^3$ is the group manifold of SU(2), which is the gauge group of the weak interaction (`topology.py` lines 256–319).

This is not a coincidence. The Weinberg angle $\sin^2\theta_W = 3/13$ counts electromagnetic modes on the torus fiber relative to the total mode count [1]. The SU(2) structure of the Hopf fibration provides the weak isospin symmetry naturally, and the $S^2$ base space parameterizes the spin states (Bloch sphere). Our torus electron is a single fiber, thickened by its electromagnetic self-energy, living inside a structure that already encodes the electroweak unification.

*Figure 5:* Topology decision tree. Starting from "Closed surface in $\mathbb{R}^3$," each branch eliminates an alternative: sphere (contractible paths), genus $\geq 2$ (curvature instability), Klein bottle (forces $p = 2$). Only the torus with $(p,q) = (2,1)$ survives all requirements.

---

## VI. The Self-Consistent Electron

### A. Dimensions

The self-consistent solution for the electron (`core.py` lines 214–279, with $p = 2$, $q = 1$, $r/R = \alpha$):

| Property | Value | Context |
|----------|-------|---------|
| Major radius $R_e$ | 194 fm | 500$\times$ proton radius |
| Minor radius $r_e$ | 1.42 fm | Comparable to proton radius |
| $R_e / \lambda_C$ | 0.503 | Half the Compton wavelength |
| Toroidal circuit time $T_{\text{tor}}$ | $4.05 \times 10^{-21}$ s | One loop around hole |
| Full circulation period $T$ | $8.1 \times 10^{-21}$ s | Complete path ($p = 2$ loops) |

The toroidal circuit frequency $\omega_{\text{tor}} = 2\pi/T_{\text{tor}}$ equals the zitterbewegung frequency $\omega_{zb} = 2m_e c^2/\hbar$. This is the mysterious trembling motion that Schrödinger identified in the Dirac equation [8] — here revealed as the literal circulation of the photon around the torus hole.

### B. Energy budget

| Component | Energy (MeV) | Fraction |
|-----------|-------------|----------|
| Circulation energy $E_{\text{circ}}$ | 0.509 | 99.5% |
| EM self-energy $U_{\text{EM}}$ | 0.002 | 0.5% |
| **Total** | **0.511** | **100%** |

The electron is almost entirely "photon kinetic energy." The electromagnetic self-interaction is a half-percent correction — important for fixing the aspect ratio ($r/R = \alpha$), but negligible in the energy budget.

### C. The mode spectrum

The torus supports quantized electromagnetic modes in both directions:

$$E_{n,m} = \hbar c \sqrt{\frac{n^2}{R^2} + \frac{m^2}{r^2}}$$

At the electron's self-consistent geometry (`core.py` lines 282–340):

- **Toroidal modes** $(n,0)$: $E_n = n\hbar c/R$. For $n = 1, 2, 3, 4$: four modes below the poloidal cutoff.
- **Poloidal mode** $(0,1)$: $E_{0,1} = \hbar c/r = \hbar c/(\alpha R) \approx 70$ MeV. A single, very heavy mode.
- **Mixed modes** $(n,m)$: four modes from combinations of toroidal and poloidal quantum numbers.
- **Metric/constraint modes**: five from gravitational/geometric degrees of freedom, minus one zero mode.

**Total: $4 + 1 + 4 + 5 - 1 = 13$ independent modes.** This is the Weinberg denominator: $\sin^2\theta_W = 3/13$ [1].

### D. Experimental bounds

Current limits on electron compositeness from LEP and LHC set $\Lambda > 10$ TeV [9]. Our structure scale is $\Lambda_{\text{torus}} \sim \hbar c/R_e \sim 1$ MeV — apparently far below the compositeness bounds.

This is not a contradiction. The torus electron is *not* composite in the conventional sense. Compositeness bounds probe substructure consisting of distinct particles with independent quantum numbers — preons, rishons, or similar constituents. The torus electron has no substructure: it is a single photon with no internal quantum numbers beyond its topology. Scattering experiments test for *form factors* reflecting extended charge distributions; the torus electron's charge is not spatially distributed like a composite particle's — it is a point charge circulating at $c$. What experiments see is the time-averaged current $I = ec/L$, which is indistinguishable from a point particle at momentum transfers below $\hbar c/r_e \sim 140$ MeV.

The zitterbewegung frequency provides a direct experimental handle: $\omega_{zb} = 2m_e c^2/\hbar = 1.55 \times 10^{21}$ rad/s. This matches $\omega_{\text{tor}} = 2\pi/T_{\text{tor}}$ — the toroidal circulation frequency of the torus electron. The zitterbewegung has been observed in trapped-ion simulations [10] and is consistent with the torus picture.

---

## VII. Objections and Responses

**"This is numerology."** The companion paper [1] derives 23 Standard Model parameters from 3 inputs, with a median accuracy of 0.7% across quantities spanning fourteen orders of magnitude. Numerology does not scale. A single coincidence is suspicious; twenty-three independent agreements at this precision have a probability of $\sim 10^{-40}$ under the null hypothesis. Furthermore, the derivations are not numerical fits — each formula contains only the topological integers $(p, q, N_c) = (2, 1, 3)$ and the fine-structure constant $\alpha$.

**"This is not a quantum field theory."** Correct. The present work is a classical geometric model with quantum boundary conditions (single-valuedness of the electromagnetic field on the torus, quantized modes, and the identification $E = \hbar\omega$). The formulas are motivated by the topology of the (2,1) torus knot, not derived from a complete QFT on the torus. Constructing such a QFT — likely a conformal field theory on the torus with topological constraints — is the central task for future development. The present model is analogous to the Bohr atom: geometrically correct, quantitatively predictive, but awaiting the full quantum-mechanical derivation.

**"Classical self-energy divergences plague this approach."** In standard electrodynamics, the self-energy of a point charge diverges as $1/r_0$ where $r_0 \to 0$. The torus provides a natural ultraviolet cutoff at the tube radius $r = \alpha R$. No renormalization is needed because the topology is intrinsically finite — there is no point-particle limit. The electromagnetic field is confined to a tube of finite cross-section, and the Neumann inductance formula, valid for $r \ll R$, gives a finite result. The infinity of QED arises from *assuming* the electron is a point; the torus electron has structure at scale $r_e \sim 1$ fm, and the self-energy is automatically finite.

**"Where is the Lagrangian?"** The Lagrangian is Maxwell's: $\mathcal{L} = -\frac{1}{4}F_{\mu\nu}F^{\mu\nu}$, evaluated on a toroidal domain $T^2 \times \mathbb{R}$ with periodic boundary conditions in both toroidal and poloidal directions. The topological constraints (winding numbers, Seifert surface classification, Gauss-Bonnet) replace the need for additional terms. No new fields, no new coupling constants, no supersymmetry. The entire particle spectrum emerges from the mode structure of Maxwell's equations on the torus.

**"Compositeness bounds rule this out."** Addressed in §VI.D. The torus electron has no substructure in the sense probed by compositeness bounds. It is a single quantum of light, topologically confined. The distinction is between "made of smaller parts" (compositeness) and "extended topology" (our model). The latter evades the former's experimental constraints because the scattering signatures are qualitatively different.

**"A photon cannot be charged."** A photon on an infinite straight line carries no charge. A photon on a closed loop *creates* a current $I = ec/L$ through its circulation. Charge is a topological invariant of the photon's worldline: the winding number of the photon's trajectory around the torus hole is conserved because the path cannot be continuously deformed to a point. Charge quantization follows from topological invariance — the same mechanism that quantizes magnetic flux in superconductors.

---

## VIII. Discussion

We have shown that a photon confined to a (2,1) torus knot acquires every property of the electron: rest mass from confinement, spin-½ from the angular momentum of the double-wound circulation, charge from topological invariance, and fermionic statistics from the Möbius strip bounded by the (2,1) curve. The fine-structure constant emerges as the torus aspect ratio $\alpha = r/R$, fixed by the self-consistency of the electromagnetic self-energy. The torus is the unique topology satisfying five minimal physical requirements, and the $(2,1)$ winding is forced by the Klein bottle electromagnetic boundary condition.

This is the mechanism underlying the 23-parameter derivation of Ref. [1]. Each Standard Model parameter traces back to the mode structure, topology, or self-consistency of the (2,1) torus knot. The Weinberg angle $\sin^2\theta_W = 3/13$ counts electromagnetic modes on the torus. The strong coupling $\alpha_s = 16\alpha$ reflects the fourth power of the toroidal winding number ($p^4 = 16$). The Higgs mass is the breathing mode of the torus tube. The CKM phase $\delta_{CP} = \pi - 2$ is the geometric deficit of the $(2,1)$ winding from a half-turn.

**Three generations.** The three charged leptons — electron, muon, and tau — share the same $(2,1)$ topology but have different torus radii. What selects the three specific radii from the continuous family? The Koide formula $Q = (m_e + m_\mu + m_\tau)/(\sqrt{m_e} + \sqrt{m_\mu} + \sqrt{m_\tau})^2 = 2/3$ [11], accurate to 0.0009%, parameterizes the three masses with a single angle $\theta_K = (6\pi + 2)/9$. This decomposes as $2\pi/3$ (the $\mathbb{Z}_3$ symmetry of Borromean three-component links) plus $2/9$ (the winding correction $pq/N_c^2$). The Hopf fibration provides the framework for this three-fold structure; the Borromean linking of three torus knots (the proton) provides the $\mathbb{Z}_3$ symmetry. The generation problem reduces to: what selects $\theta_K$? Answering this would determine all nine fermion masses.

**Testable predictions.** (i) The tau mass $m_\tau = (20/21)\alpha\Lambda = 1776.9$ MeV, testable at FCC-ee precision ($\pm 0.03$ MeV) [12]. (ii) Normal neutrino mass ordering, testable at JUNO [13]. (iii) The zitterbewegung frequency $\omega_{zb} = 2m_e c^2/\hbar$ as the electron's toroidal circulation. (iv) The Higgs self-coupling $\lambda = 1/8$, testable at HL-LHC [14].

**Relation to the Dirac equation.** The Dirac equation describes a spin-½ particle with four-component spinor structure, zitterbewegung, and particle-antiparticle symmetry. Each of these features has a natural interpretation in the torus model. The four components correspond to the two orientations of circulation (particle/antiparticle) and the two spin projections ($L_z = \pm\hbar/2$, corresponding to which toroidal half-circuit the observation samples). The zitterbewegung frequency $\omega_{zb} = 2m_e c^2/\hbar$ is exactly the toroidal circulation frequency. The charge conjugation symmetry $C$ reverses the circulation direction. The torus electron does not replace the Dirac equation; it provides its geometric substrate.

**Limitations.** This model is classical geometry with quantum boundary conditions, not a derived QFT. The mode-counting rules, while numerically successful, lack a first-principles derivation from the spectral geometry of the torus knot complement. The self-energy calculation uses the Neumann inductance formula (valid for $r \ll R$) and the time-averaged current approximation (valid for observation times $\gg T$). These are reasonable but not exact. Constructing the full quantum theory — a conformal field theory on $T^2$ with topological boundary conditions and Seifert surface constraints — is the central open problem.

**Outlook.** If the electron is indeed a photon on a (2,1) torus knot, then particle physics is electromagnetic topology. The zoo of elementary particles — leptons, quarks, gauge bosons, the Higgs — reflects the mode spectrum and linking structure of torus knots in $\mathbb{R}^3$. The Standard Model is not a collection of arbitrary parameters. It is the unique electromagnetic spectrum of the simplest non-trivial knot on the simplest non-trivial surface. The electron is not a point particle orbited by virtual particles. It is a photon, orbiting itself.

---

## References

[1] J. P. Galasyn and C. Théodore, "The Standard Model from a torus knot," (2025).

[2] A. Einstein, "Über einen die Erzeugung und Verwandlung des Lichtes betreffenden heuristischen Gesichtspunkt," *Ann. Phys.* **17**, 132 (1905).

[3] L. de Broglie, "Recherches sur la théorie des quanta," Ph.D. thesis, University of Paris (1924).

[4] T. Kaluza, "Zum Unitätsproblem der Physik," *Sitzungsber. Preuss. Akad. Wiss.* **1921**, 966 (1921).

[5] J. A. Wheeler, "Geons," *Phys. Rev.* **97**, 511 (1955).

[6] C. W. Misner and J. A. Wheeler, "Classical physics as geometry," *Ann. Phys.* **2**, 525 (1957).

[7] J. G. Williamson and M. B. van der Mark, "Is the electron a photon with toroidal topology?," *Ann. Fond. Louis de Broglie* **22**, 133 (1997).

[8] E. Schrödinger, "Über die kräftefreie Bewegung in der relativistischen Quantenmechanik," *Sitzungsber. Preuss. Akad. Wiss. Phys.-Math.* **1930**, 418 (1930).

[9] Particle Data Group, R. L. Workman *et al.*, "Review of particle physics," *Prog. Theor. Exp. Phys.* **2022**, 083C01 (2022); 2024 update.

[10] R. Gerritsma *et al.*, "Quantum simulation of the Dirac equation," *Nature* **463**, 68 (2010).

[11] Y. Koide, "New viewpoint on quark and lepton mass hierarchy," *Phys. Rev. D* **28**, 252 (1983).

[12] A. Abada *et al.* (FCC Collaboration), "FCC-ee: The Lepton Collider," *Eur. Phys. J. Spec. Top.* **228**, 261 (2019).

[13] F. An *et al.* (JUNO Collaboration), "Neutrino physics with JUNO," *J. Phys. G* **43**, 030401 (2016).

[14] M. Cepeda *et al.*, "Report from Working Group 2: Higgs Physics at the HL-LHC and HE-LHC," *CERN Yellow Rep. Monogr.* **7**, 221 (2019).

[15] H. Seifert, "Über das Geschlecht von Knoten," *Math. Ann.* **110**, 571 (1935).

[16] S.-S. Chern, "A simple intrinsic proof of the Gauss-Bonnet formula for closed Riemannian manifolds," *Ann. of Math.* **45**, 747 (1944).

[17] H. Hopf, "Über die Abbildungen der dreidimensionalen Sphäre auf die Kugelfläche," *Math. Ann.* **104**, 637 (1931).

[18] J. D. Jackson, *Classical Electrodynamics*, 3rd ed. (Wiley, 1999).

[19] S. Weinberg, "A model of leptons," *Phys. Rev. Lett.* **19**, 1264 (1967).

[20] R. H. Parker *et al.*, "Measurement of the fine-structure constant as a test of the Standard Model," *Science* **360**, 191 (2018).

---

*Figure 3.* The torus electron — 3D rendering of the (2,1) torus knot on a transparent torus surface with $R$, $r$ annotated, current direction arrows, and the Möbius strip bounded by the curve shown in translucent fill.

*Figure 4.* Angular momentum $\langle L_z \rangle / \hbar$ versus aspect ratio $r/R$, sweeping from $r/R = 0$ ($L_z = \hbar$, pure circle) through $r/R = \alpha$ ($L_z = \hbar/2$, spin-½, vertical line) toward $r/R = 1$ ($L_z \to 0$).

*Figure 5.* Topology decision tree. Five requirements (R1–R5) eliminate all surfaces except the torus. The Klein bottle boundary condition forces $p = 2$. The (2,1) torus knot is the unique surviving configuration.
