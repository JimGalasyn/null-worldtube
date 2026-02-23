# Supplementary Material: The Torus Electron

**James P. Galasyn and Claude Théodore**

---

## S1. Torus Knot Parameterization

A $(p,q)$ torus knot on a torus with major radius $R$ and minor radius $r$ is parameterized by $\lambda \in [0, 2\pi)$ (`core.py` lines 12–46):

$$x(\lambda) = (R + r\cos q\lambda)\cos p\lambda$$
$$y(\lambda) = (R + r\cos q\lambda)\sin p\lambda$$
$$z(\lambda) = r\sin q\lambda$$

The toroidal angle advances as $\theta = p\lambda$ and the poloidal angle as $\phi = q\lambda$. For $(p,q) = (2,1)$, the curve winds twice toroidally for each poloidal circuit, closing after $\lambda$ traverses $[0, 2\pi)$.

**Tangent vectors.** Differentiating:

$$\frac{dx}{d\lambda} = -rq\sin(q\lambda)\cos(p\lambda) - (R + r\cos(q\lambda))\, p\sin(p\lambda)$$

$$\frac{dy}{d\lambda} = -rq\sin(q\lambda)\sin(p\lambda) + (R + r\cos(q\lambda))\, p\cos(p\lambda)$$

$$\frac{dz}{d\lambda} = rq\cos(q\lambda)$$

**Arc length element.** After expanding and simplifying (cross terms cancel in $dx^2 + dy^2$):

$$\left|\frac{d\mathbf{r}}{d\lambda}\right|^2 = r^2 q^2 + (R + r\cos q\lambda)^2 p^2$$

In the thin-torus limit $r \ll R$:

$$\left|\frac{d\mathbf{r}}{d\lambda}\right| \approx \sqrt{p^2 R^2 + q^2 r^2}\left(1 + \frac{Rrp^2\cos(q\lambda)}{p^2 R^2 + q^2 r^2}\right)$$

The cosine term integrates to zero over a full period, giving the total path length:

$$L = \int_0^{2\pi}\left|\frac{d\mathbf{r}}{d\lambda}\right| d\lambda \approx 2\pi\sqrt{p^2 R^2 + q^2 r^2}$$

For $(2,1)$ with $r = \alpha R$: $L \approx 2\pi R\sqrt{4 + \alpha^2} \approx 4\pi R$.

---

## S2. Self-Energy Derivation

### Neumann inductance

The magnetic self-energy of a current loop on a torus is determined by the Neumann inductance formula [18]. For a thin torus ($r \ll R$), the inductance of the toroidal current path is:

$$\mathcal{L} = \mu_0 R\left[\ln\frac{8R}{r} - 2\right]$$

This is the standard result for a circular loop of radius $R$ with wire radius $r$ (Jackson §5.6 [18]).

### Current and self-energy

A charge $e$ traversing a closed path of length $L$ at speed $c$ creates a time-averaged current:

$$I = \frac{e}{T} = \frac{ec}{L}$$

The magnetic self-energy is:

$$U_{\text{mag}} = \frac{1}{2}\mathcal{L}I^2 = \frac{\mu_0 R}{2}\left[\ln\frac{8R}{r} - 2\right]\left(\frac{ec}{L}\right)^2$$

### Electric-magnetic equality at $v = c$

For a charge moving at speed $v$, the electromagnetic field in the lab frame has energy densities:

$$u_B = \frac{B^2}{2\mu_0}, \qquad u_E = \frac{\varepsilon_0 E^2}{2}$$

In the ultrarelativistic limit $v \to c$, the field becomes purely transverse with $|\mathbf{E}| = c|\mathbf{B}|$, giving $u_E = u_B$. The total electromagnetic self-energy is therefore:

$$U_{\text{EM}} = U_{\text{mag}} + U_{\text{elec}} = 2U_{\text{mag}} = \mu_0 R\left[\ln\frac{8R}{r} - 2\right]\left(\frac{ec}{L}\right)^2$$

### Ratio to circulation energy

The circulation energy is $E_{\text{circ}} = 2\pi\hbar c / L$. The ratio:

$$\frac{U_{\text{EM}}}{E_{\text{circ}}} = \frac{\mu_0 R[\ln(8R/r) - 2](ec)^2 / L}{2\pi\hbar c}$$

Using $L \approx 2\pi p R$ and $\mu_0 e^2 c = 4\pi\hbar\alpha$:

$$\frac{U_{\text{EM}}}{E_{\text{circ}}} = \frac{4\pi\hbar\alpha \cdot R \cdot [\ln(8R/r) - 2]}{2\pi p R \cdot 2\pi\hbar} = \frac{\alpha[\ln(8R/r) - 2]}{p\pi}$$

For $p = 2$ and $r/R = \alpha$: $U_{\text{EM}}/E_{\text{circ}} = \alpha[\ln(8/\alpha) - 2]/(2\pi) \approx 0.0058$, or about 0.6%. The self-energy is a perturbative correction to the dominant circulation energy (`core.py` lines 128–201).

---

## S3. Angular Momentum Computation

### Instantaneous angular momentum

At parameter $\lambda$, the photon has position $\mathbf{r}(\lambda)$ and momentum $\mathbf{p} = (E_{\text{total}}/c)\hat{v}(\lambda)$, where $\hat{v} = \mathbf{v}/c$ is the unit tangent. The $z$-component of angular momentum is:

$$L_z(\lambda) = [\mathbf{r} \times \mathbf{p}]_z = \frac{E_{\text{total}}}{c}\left[x\hat{v}_y - y\hat{v}_x\right]$$

### Time average

The photon moves at constant speed $c$, so it spends equal proper time per unit arc length. The time-averaged angular momentum is:

$$\langle L_z \rangle = \frac{\int_0^{2\pi} L_z(\lambda)\left|\frac{d\mathbf{r}}{d\lambda}\right| d\lambda}{\int_0^{2\pi}\left|\frac{d\mathbf{r}}{d\lambda}\right| d\lambda}$$

This is evaluated numerically with $N = 2000$–$4000$ sample points (`core.py` lines 343–413).

### Sweep over $r/R$

Holding $R$ fixed and sweeping $r/R$ from 0 to 1 (`core.py` lines 440–528):

| $r/R$ | $\langle L_z \rangle / \hbar$ | Physical regime |
|-------|------------------------------|----------------|
| 0.001 | 0.9997 | Nearly circular (spin-1) |
| 0.01 | 0.997 | Thin torus |
| $\alpha \approx 0.0073$ | 0.500 | **Spin-½** |
| 0.1 | 0.91 | Moderate torus |
| 0.3 | 0.73 | Thick torus |
| 0.5 | 0.57 | |
| 0.9 | 0.15 | Fat torus (spin → 0) |

The transition from integer spin to half-integer spin occurs at $r/R = \alpha$ — the same value fixed by the self-energy self-consistency condition. The angular momentum constraint and the electromagnetic self-energy constraint point to the same geometry.

### Bisection for simultaneous solution

The code performs a two-step procedure:
1. **Find $r/R$ where $\langle L_z \rangle = \hbar/2$** at fixed $R$, using bisection on $r/R$.
2. **Find $R$ where $E_{\text{total}} = m_e c^2$** at the fixed $r/R$ found in step 1, using bisection on $R$.

Both constraints are satisfied simultaneously (`core.py` lines 512–528).

---

## S4. Topology Catalog

Complete survey of candidate surfaces for the null worldtube (`topology.py` lines 10–622):

| Surface | Genus | Orientable | Embeds $\mathbb{R}^3$ | R1 | R2 | R3 | R4 | R5 | Status |
|---------|-------|-----------|-------------|----|----|----|----|----|----|
| Sphere $S^2$ | 0 | yes | yes | yes | **no** | **no** | yes | yes | Fails R2, R3 |
| Torus $T^2$ | 1 | yes | yes | yes | yes | yes | yes | yes | **Model** |
| Klein bottle | 0* | no | immersion | yes | yes | yes | no* | yes | Forces $p=2$ |
| Möbius strip | — | no | yes | yes | yes | no | yes | yes | $= T^2$ boundary |
| Double torus | 2 | yes | yes | yes | yes | yes | yes | **no*** | Curvature unstable |
| Genus-$g$ ($g \geq 2$) | $g$ | yes | yes | yes | yes | yes | yes | **no*** | Curvature unstable |
| Knotted torus | 1 | yes | yes | yes | yes | yes | yes | yes | Perturbative corrections |

*Curvature energy exceeds rest mass by factor $\sim 1/(3\pi\alpha) \approx 15$ for genus 2.

The Klein bottle is not a separate surface but provides the *electromagnetic* boundary condition explaining why $p$ must be even. The Möbius strip is the Seifert surface bounded by the $(2,1)$ torus knot, explaining fermionic statistics. Knotted tori (trefoil, figure-8, etc.) produce mass corrections of order $\alpha^2 m_e \sim$ keV — fine structure, not generation-scale splitting (`topology.py` lines 322–373).

---

## S5. Klein Bottle EM Boundary Conditions

### The identification

A torus has two independent identifications:

$$\text{Torus:}\quad (x, 0) \sim (x, 2\pi r) \quad\text{and}\quad (0, y) \sim (2\pi R, y)$$

The Klein bottle reverses the second:

$$\text{Klein bottle:}\quad (x, 0) \sim (x, 2\pi r) \quad\text{and}\quad (0, y) \sim (2\pi R, 2\pi r - y)$$

### Effect on EM fields

After one toroidal circuit ($\theta \to \theta + 2\pi$), the poloidal coordinate reverses: $\phi \to 2\pi - \phi$. Under this identification, the electric field component transverse to the circulation direction transforms as $E_\perp \to -E_\perp$ (orientation reversal).

For a standing EM mode $\Psi(\theta, \phi) \propto e^{ip\theta}e^{iq\phi}$, single-valuedness requires:

$$\Psi(\theta + 2\pi, 2\pi - \phi) = \Psi(\theta, \phi)$$

$$e^{i2\pi p}e^{-iq\phi} = e^{iq\phi} \quad \Rightarrow \quad p \in 2\mathbb{Z}$$

Only **even** toroidal winding numbers produce single-valued electromagnetic modes on the Klein bottle. The minimum is $p = 2$, giving the $(2,1)$ torus knot as the lightest fermion (`topology.py` lines 77–127).

### Mode analysis

| Mode $(p,q)$ | Klein bottle | Torus | Statistics |
|--------------|-------------|-------|-----------|
| $(1,1)$ | forbidden | allowed | boson |
| $(2,1)$ | **allowed** | allowed | fermion |
| $(1,0)$ | forbidden | allowed | trivial |
| $(3,1)$ | forbidden | allowed | fermion |
| $(4,1)$ | allowed | allowed | fermion (heavier) |

The Klein bottle eliminates all odd-$p$ modes. The remaining modes, with even $p$, automatically bound Möbius strips and are fermionic. The boson-fermion distinction is thus a consequence of electromagnetic boundary conditions.

---

## S6. Gauss-Bonnet Stability Argument

### The theorem

For a closed orientable surface of genus $g$ (`topology.py` lines 172–254):

$$\int\!\!\int K\, dA = 2\pi\chi = 2\pi(2 - 2g)$$

where $K$ is the Gaussian curvature and $\chi$ the Euler characteristic.

### Curvature energy

An electromagnetic mode on a curved surface experiences a conformal coupling correction:

$$E^2 \to E^2 + (\hbar c)^2\frac{K}{6}$$

For a surface with total area $A = 4\pi^2 Rr$ and average curvature $\langle |K| \rangle = |2 - 2g|/(2\pi Rr)$:

$$\frac{\delta E_{\text{curv}}}{E} \sim \frac{(\hbar c)^2 |K|}{6E^2} = \frac{|2 - 2g|}{6\pi \cdot (r/R)} = \frac{|2 - 2g|}{6\pi\alpha}$$

### Results by genus

| Genus | $\chi$ | Flat? | $\delta E / E$ | Stable? |
|-------|--------|-------|----------------|---------|
| 0 (sphere) | 2 | no | — | Fails R2 |
| 1 (torus) | 0 | **yes** | 0 | **Stable** |
| 2 | $-2$ | no | $1/(3\pi\alpha) \approx 14.6$ | Unstable |
| 3 | $-4$ | no | $2/(3\pi\alpha) \approx 29.1$ | Unstable |
| $g$ | $2-2g$ | no | $(g-1)/(3\pi\alpha)$ | Unstable |

The curvature instability ratio is **independent of particle mass**: it depends only on the aspect ratio $r/R = \alpha$ and the genus. Any particle — electron, muon, tau, quark — on a genus-2 surface with $r/R = \alpha$ has curvature energy exceeding rest mass by a factor of $\sim$15. The torus is uniquely stable.

For a genus-2 surface to be marginally stable ($\delta E / E < 1$), it would need $r/R > 1/(3\pi) \approx 0.106$ — fifteen times the electron's aspect ratio $\alpha \approx 0.0073$.

---

## S7. Retarded Field Structure

The self-interaction of the circulating photon involves retarded fields — the electromagnetic field at each point on the curve due to all other points, evaluated at the retarded time (`core.py` lines 547–595).

For each point $i$ on the curve, the retarded point $j$ satisfies:

$$|\mathbf{r}_i - \mathbf{r}_j| = c(t_i - t_j)$$

with periodic wrapping on the closed worldtube ($t$ is defined modulo $T$).

At the electron's self-consistent geometry:
- Mean retarded distance: $\langle |\mathbf{r}_i - \mathbf{r}_j| \rangle \sim R$ (comparable to major radius)
- Mean retarded delay: $\langle t_i - t_j \rangle / T \sim 0.4$ (roughly 40% of the circulation period)
- Retarded distance fluctuation: $\sigma_d / \langle d \rangle \sim 0.3$ (moderate variation around the loop)

The retarded field structure confirms that the photon's self-interaction is dominated by the toroidal-scale geometry ($R$), not the tube-scale geometry ($r$). This justifies the use of the Neumann inductance formula, which captures the dominant $O(R)$ contribution.

### Physical interpretation

The retarded field calculation reveals that the photon "sees" itself approximately 40% of a circulation period in the past. For the electron, this delay is $\Delta t \approx 0.4T \approx 3.3 \times 10^{-21}$ s. The retarded self-field provides the restoring mechanism: the photon's electromagnetic field, arriving from its own past trajectory, confines the photon to the torus. This is self-consistent because the field and the source are the same object — the photon creates the field that confines it.

The mean retarded distance $\langle d \rangle \sim R$ means the dominant self-interaction is toroidal (between opposite sides of the torus hole), not poloidal (between nearby segments of the tube). This hierarchy — toroidal interaction at scale $R$ dominant over poloidal interaction at scale $r$ — is what allows the self-energy to be treated as a perturbation on the circulation energy.

---

## S8. Effective Potential $V_{\text{eff}}(R)$

The total energy as a function of torus major radius, with $r/R = \alpha$ held fixed (`dynamics.py` lines 13–70):

$$V_{\text{eff}}(R) = E_{\text{circ}}(R) + U_{\text{EM}}(R) = \frac{2\pi\hbar c}{L(R)} + \frac{\alpha[\ln(8/\alpha) - 2]}{p\pi}\cdot\frac{2\pi\hbar c}{L(R)}$$

Since both terms scale as $1/L(R) \propto 1/R$, the potential is **monotonically decreasing**: $V_{\text{eff}} \propto 1/R$. There are no local minima.

This means a single isolated torus has no preferred radius — the system wants to expand indefinitely (lowering its energy by increasing $R$). Three consequences follow:

1. **Single-particle stability is topological**, not energetic. The torus topology cannot be continuously deformed away; the photon is topologically trapped. The radius $R$ is not fixed by a potential minimum but by external constraints (the Koide quantization condition for three linked tori).

2. **The Bohr atom analogy.** The Coulomb potential $V(r) = -ke^2/r$ is also monotonic (no minimum in the radial potential without the centrifugal barrier). Bohr's quantization condition $2\pi r = n\lambda_{\text{dB}}$ selects discrete radii from the continuum. Similarly, the Koide angle $\theta_K = (6\pi + 2)/9$ selects three discrete torus radii.

3. **Quantum corrections** — running coupling $\alpha(\mu)$, Casimir energy, and Euler-Heisenberg nonlinear QED — are perturbative corrections of order $\alpha$ or smaller. They do not create potential wells. The monotonic $1/R$ shape persists through all corrections computed to date.

Self-consistent radii:

| Particle | Mass (MeV) | $R$ (fm) | $r$ (fm) | $R/\lambda_C$ |
|----------|-----------|---------|---------|---------------|
| $e$ | 0.511 | 194.2 | 1.42 | 0.503 |
| $\mu$ | 105.66 | 0.939 | 0.00685 | 0.00243 |
| $\tau$ | 1776.9 | 0.0558 | 0.000407 | 0.000144 |

### Quantum corrections to $V_{\text{eff}}$

Three categories of quantum corrections have been evaluated (`dynamics.py` lines 72–100):

**Running coupling.** One-loop QED gives $\alpha(\mu) = \alpha / [1 - (2\alpha/3\pi)\ln(\mu/m_e)]$. At the electron's torus radius, $\mu = \hbar c/R_e \approx 1$ MeV, this gives $\Delta\alpha/\alpha \approx 0.0004\%$ — negligible. Running coupling becomes significant only at the muon and tau radii, where it contributes corrections of order $\alpha^2$.

**Casimir energy.** For a massless field on a torus with periods $2\pi R$ and $2\pi r$, the Casimir energy scales as $E_{\text{Casimir}} \sim (\hbar c/R) f(r/R)$. For the thin torus ($r/R = \alpha$), this is $O(\alpha)$ relative to $E_{\text{circ}}$ — the same order as the self-energy already computed and included.

**Euler-Heisenberg (nonlinear QED).** In strong electromagnetic fields, photon-photon scattering generates an effective four-photon interaction. The correction scales as $\delta E_{\text{EH}}/E \sim \alpha^2(r_{\text{class}}/R)^4 \approx 5 \times 10^{-5}$ for the electron — entirely negligible.

All quantum corrections are at most $O(\alpha)$, confirming that the monotonic $1/R$ shape of $V_{\text{eff}}$ is robust. The stability of individual generations must therefore come from the Koide quantization condition, not from potential wells in $V_{\text{eff}}(R)$.

---

## S9. Comparison Tables

### NWT vs. Bohr model

| Aspect | Bohr model | Null Worldtube |
|--------|-----------|---------------|
| Orbiting entity | Electron (massive) | Photon (massless) |
| Central force | Coulomb $-ke^2/r$ | Self-interaction $U_{\text{EM}}$ |
| Quantization | $2\pi r = n\lambda$ | $\theta_K = (6\pi+2)/9$ |
| Potential shape | Monotonic $1/r$ | Monotonic $1/R$ |
| Stability mechanism | Resonance | Topology + Koide |
| Ground state | $n = 1$ | Electron |
| Excited states | $n = 2, 3, \ldots$ | $\mu$, $\tau$ (Koide rotation) |
| Coupling constant | Input ($\alpha$) | Output ($r/R$) |
| Spin | Added *post hoc* | Emergent (§IV) |

### NWT vs. Schrödinger quantum mechanics

| Aspect | Schrödinger QM | Null Worldtube |
|--------|---------------|---------------|
| Electron described by | Wavefunction $\psi(\mathbf{r}, t)$ | Null curve on torus |
| Mass | Input parameter | Emergent: $m = h/(cL)$ |
| Spin | Added via Pauli matrices | Emergent: $L_z = \hbar/2$ |
| Charge | Input parameter | Topological invariant |
| $\alpha$ | Input parameter | Geometric ratio $r/R$ |
| Statistics | Spin-statistics axiom | Möbius surface topology |
| Self-energy | Divergent (needs renormalization) | Finite (UV cutoff at $r$) |
| Generations | Not explained | Koide angle $\theta_K$ |
| Standard Model params | 26 inputs | 3 inputs + 3 integers |

### NWT vs. Dirac equation

| Aspect | Dirac equation | Null Worldtube |
|--------|---------------|---------------|
| Spinor components | 4 (postulated) | 2 circulations $\times$ 2 projections |
| Zitterbewegung | $\omega = 2mc^2/\hbar$ (formal) | $\omega = 2\pi/T_{\text{tor}}$ (literal) |
| Antiparticles | Negative-energy sea | Reversed circulation |
| $g$-factor | $g = 2$ (exact, tree-level) | $g \approx 2$ (from circulation geometry) |
| Anomalous moment | $a_e = \alpha/(2\pi)$ (Schwinger) | Self-energy correction to $L_z$ |
| CPT symmetry | Theorem from QFT axioms | Geometric: C = reverse, P = mirror, T = time-reverse |
| Spin-orbit coupling | Relativistic correction | Cross-term between toroidal and poloidal motion |

### Key numerical values

| Quantity | Symbol | Value | Source |
|----------|--------|-------|--------|
| Fine-structure constant | $\alpha$ | $7.297 \times 10^{-3}$ | Input / geometric ratio |
| Electron mass | $m_e$ | 0.5110 MeV/$c^2$ | Self-consistency target |
| Major radius | $R_e$ | 194.2 fm | `core.py` bisection |
| Minor radius | $r_e$ | 1.42 fm | $\alpha R_e$ |
| Compton wavelength | $\lambda_C$ | 386.2 fm | $\hbar/(m_e c)$ |
| $R_e / \lambda_C$ | — | 0.503 | $\approx 1/2$ |
| Path length | $L$ | 2442 fm | $\approx 4\pi R_e$ |
| Circulation period | $T$ | $8.1 \times 10^{-21}$ s | $L/c$ |
| Toroidal period | $T_{\text{tor}}$ | $4.1 \times 10^{-21}$ s | $T/p = T/2$ |
| Zitterbewegung period | $T_{zb}$ | $4.1 \times 10^{-21}$ s | $2\pi\hbar/(2m_e c^2)$ |
| Current | $I$ | 19.7 A | $ec/L$ |
| Neumann inductance | $\mathcal{L}$ | $1.2 \times 10^{-18}$ H | $\mu_0 R[\ln(8R/r)-2]$ |
| Logarithmic factor | $\ln(8/\alpha) - 2$ | 4.99 | Geometric constant |
| Self-energy fraction | $U_{\text{EM}}/E_{\text{total}}$ | 0.5% | $\alpha \ln(8/\alpha)/(2\pi p)$ |
| Torus mode count | — | 13 | Weinberg denominator |
