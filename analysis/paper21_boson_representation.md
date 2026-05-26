# Paper 21a — Boson Representation in NWT Substrate

*Working memo, 2026-05-23. Formalises Option (A)+(B): bosons as elements of the stabilizer-normalizer-quotient algebra of the multi-patch Steane code, realised on Heron-class hardware as inter-patch entangling operations.*

## 0. Problem statement

Paper 21 establishes that particles are closed walks on $K_7$ compiled to Pauli words on the Steane $[\![7,1,3]\!]$ code (§3–§4). Bosons are not particles in this sense — they have no persistent identity, no $(|p|, |q|)$ winding class, no carrier knot. They cannot be closed walks.

But under substrate-monism, every physical entity must be a $K_7$-substrate construction. So bosons must be **operations** on the substrate, not states. This memo formalises that claim.

The formalisation must:
1. Identify a specific operator class as the boson algebra.
2. Match the SM gauge structure (U(1) photon, SU(2) weak, SU(3) gluons).
3. Be hardware-realisable on Heron-class chips with the same 7-qubit-per-patch fidelity as Paper 21's Phase 2.
4. Make falsifiable bench-scale predictions distinct from the single-walk syndrome predictions already established.

Neutrinos are addressed separately (§6) because despite Jim's grouping with photons, they are fermions, not bosons.

## 1. Multi-patch substrate

**Definition 1.1 (Patch).** A *patch* $P_i$ is one copy of the Steane $[\![7,1,3]\!]$ code, with stabilizer group
$$
S_i = \langle g_{i,1}^X, g_{i,2}^X, g_{i,3}^X, g_{i,1}^Z, g_{i,2}^Z, g_{i,3}^Z \rangle \cong \mathbb{Z}_2^6
$$
acting on Hilbert space $\mathcal{H}_i = (\mathbb{C}^2)^{\otimes 7}$. The X-type generators have QR-translate supports $\{1,2,4\}+j$ ($j=0,1,2$) per Paper 21 §4.1; Z-type supports are the $H^{\otimes 7}$-image. Patch $i$ has its own polar vertex $P_i$ (substrate vacuum reference).

**Definition 1.2 (Multi-patch substrate).** The $N$-patch substrate is
$$
\mathcal{H}_N = \bigotimes_{i=1}^N \mathcal{H}_i, \qquad S_N = \bigotimes_{i=1}^N S_i.
$$
The codespace $\mathcal{C}_N = \bigotimes_i \mathcal{C}_i$ has dimension $2^N$ (one logical qubit per patch).

**Definition 1.3 (Fermion state).** A fermion in patch $i$ is an encoded Steane state $|\psi_w\rangle_i \in \mathcal{H}_i$ specified by walk class $w = (|p|, |q|)_i$ via the Paper 21 §6 syndrome map: $(s_X, s_Z)_i$ identifies the Fano-plane location of the walk's Pauli content modulo $S_i$.

The *vacuum* of the substrate is the all-patches-logical-zero state $|0\rangle_N = \bigotimes_i |\bar 0\rangle_i$.

## 2. Boson algebra: definition

The Pauli group on $\mathcal{H}_N$ is $\mathcal{P}_N$ (including phases). Its normalizer of $S_N$ is
$$
N(S_N) = \{ P \in \mathcal{P}_N : P S_N P^\dagger = S_N \},
$$
i.e., the elements of $\mathcal{P}_N$ that map codespace to codespace. The quotient
$$
\mathcal{B}_N := N(S_N) / S_N
$$
is the *logical Pauli algebra* of the multi-patch code. For one patch this has $4^1 = 4$ elements ($I, \bar X, \bar Y, \bar Z$); for $N$ patches it has $4^N$ elements.

**Definition 2.1 (Boson operator).** A *boson operator* on the $N$-patch substrate is a non-identity element $B \in \mathcal{B}_N$ supported on at least two patches (i.e., $B$ does not factor as $B_i \otimes I_{\bar i}$ for any single patch $i$).

By construction, $B$:
- Preserves the codespace (so it acts on fermion states without breaking the Steane encoding).
- Acts as a *logical* operation (not a stabilizer), so it changes the joint walk-state.
- Couples patches (so it mediates interaction between fermions in distinct regions).

Single-patch logical operators $\bar X_i, \bar Z_i$ are excluded because they correspond to logical Pauli operations *on a single walk* — these are absorbed into the walk-to-Pauli compilation of Paper 21 §4.2, not boson exchanges.

## 3. Photon operator

The photon is the U(1) gauge boson — neutral, massless, spin-1, self-conjugate under CPT (i.e., its own antiparticle).

### 3.1 Two-polarisation definition

**Definition 3.1 (Photon mode).** A *photon mode* on patches $(i, j)$ at vertex pair $(a, b)$ is the ordered pair of unitary operators
$$
V_\gamma^{(+)}(i,j; a,b) := X_a^{(i)} X_b^{(j)}, \qquad V_\gamma^{(-)}(i,j; a,b) := Z_a^{(i)} Z_b^{(j)}.
$$
Each component is unitary ($V_\gamma^{(\pm)2} = I$). The two components commute as operators on distinct qubits (since $X_a Z_a$ anti-commute on the same qubit, but $X_a X_b$ vs $Z_a Z_b$ each pick up two minus-signs that cancel). They are exchanged by the global CPT operator $H_N := \bigotimes_i H_i^{\otimes 7}$.

The two transverse polarisation eigenstates of the physical photon are the $\pm 1$ eigenstates of $V_\gamma^{(+)} + V_\gamma^{(-)}$ (the matter-symmetric combination) and $V_\gamma^{(+)} - V_\gamma^{(-)}$ (the matter-antimatter-antisymmetric combination). Spin-1 (two on-shell polarisations) emerges structurally from the CSS X-vs-Z duality of the Steane code.

### 3.2 Codespace preservation and the charge-vertex selection rule

**Theorem 3.2 (charge-vertex selection rule).** $V_\gamma^{(+)}(i,j; a,b) \in N(S_N)$ iff qubit $a$ is in the support of no Z-stabilizer on patch $i$ AND qubit $b$ is in the support of no Z-stabilizer on patch $j$. (Same rule for $V_\gamma^{(-)}$ with X/Z swapped.)

*Proof.* $X_a^{(i)}$ commutes with all X-stabilizers (Pauli X with Pauli X). It commutes with a Z-stabilizer $g_k^Z$ iff $a \notin \mathrm{supp}(g_k^Z)$ (otherwise $X_a$ and $Z_a$ in the support give an anticommutation). The inter-patch $X_b^{(j)}$ analogously must commute with all Z-stabilizers on patch $j$. □

**Corollary 3.3 (polar photon).** Under the Paper 21 §4.1 convention where the polar vertex $P$ is the unique qubit in no X- or Z-stabilizer support, the minimal-weight photon vertex is
$$
V_\gamma^{(\pm)}_{\mathrm{polar}}(i, j) := \bigl(X_P^{(i)} X_P^{(j)}, \; Z_P^{(i)} Z_P^{(j)}\bigr).
$$
This is the unique single-qubit-pair photon vertex compatible with both patches' Steane stabilizers.

**Substrate reading.** The polar vertex is the substrate vacuum reference (Paper 21 §4.1: "polar vertex = substrate vacuum reference"; pair production initiates at $P$ per §5). The result of Cor. 3.3 is therefore that *the photon vertex sits at the substrate vacuum reference of each patch* — photon exchange happens precisely where pair production happens. This is the substrate-monism statement that the photon mediates pair production (rather than is a separate carrier).

### 3.3 Photon-vertex graded algebra

Higher-weight photon vertices satisfy the selection rule by syndrome cancellation rather than individual-qubit avoidance. The general criterion:

**Theorem 3.4 (general photon vertex).** A multi-qubit operator $V = X_{A}^{(i)} X_{B}^{(j)}$ supported on qubit-subsets $A \subseteq P_i, B \subseteq P_j$ lies in $N(S_N)$ iff $A$ and $B$ have the same Z-syndrome (i.e., $\bigoplus_{a \in A} \mathrm{synd}_Z(a) = \bigoplus_{b \in B} \mathrm{synd}_Z(b)$), where $\mathrm{synd}_Z(a) \in \mathbb{F}_2^3$ is the vector of $a$'s membership in the three Z-stabilizers.

This gives a graded photon algebra indexed by support weight:

| Weight | Vertex | Substrate reading |
|--------|--------|-------------------|
| 1 + 1 | $X_P^{(i)} X_P^{(j)}$ | polar photon (vacuum-reference exchange) |
| 3 + 3 | $\bar X^{(i)}_{\rm min} \bar X^{(j)}_{\rm min}$ | logical photon (minimum-weight logical) |
| 4 + 4 | Fano-triangle photon | weight-4 stabilizer-coset photon |
| 7 + 7 | $X^{\otimes 7}_i X^{\otimes 7}_j$ | maximal logical photon (= Hamilton-cycle photon) |

The polar photon is the *softest* mode (lowest-weight support, coupling only to walks visiting the polar vertex). The Hamilton-cycle photon is the *hardest* (couples to all walks). Intermediate weights give a graded spectrum of photon channels that may correspond structurally to QED's continuous photon-energy spectrum, with substrate energy $\propto$ vertex weight.

*This is a substantive new claim deserving its own falsifier (§7).*

### 3.4 Properties

(P1) **Codespace preservation.** By Thm 3.2 and Cor. 3.3. ✓

(P2) **CPT self-conjugacy.** $H_N V_\gamma^{(+)} H_N^\dagger = V_\gamma^{(-)}$. The ordered-pair structure $(V_\gamma^{(+)}, V_\gamma^{(-)})$ is therefore exchanged by CPT — the two transverse polarisations are CPT-conjugates, while the photon mode as a whole (the pair) is self-CPT.

(P3) **Masslessness.** No closed walk → no $(|p|, |q|)$ → no carrier knot → no $n_q$. The Paper 11 mass formula does not apply. Structurally massless, not tuned-to-zero.

(P4) **Neutrality.** $V_\gamma^{(\pm)}$ commutes with the global $\bigotimes_k \bar Z_k$ operator (the matter/antimatter polarity label of the joint substrate). Photon exchange preserves matter/antimatter sector assignment of the connected walks. Photon carries no charge of its own.

(P5) **U(1) phase structure.** The one-parameter family $\exp(-i\theta V_\gamma^{(\pm)})$ for each polarisation generates a U(1) gauge action on the joint codespace. The angle $\theta$ is the photon's substrate analog of Wilson-line phase accumulation. The full U(1) × U(1) photon gauge action factorises across the two polarisations.

### 3.5 Coupling to walks

A walk $W$ in patch $i$ has Pauli content $P_W = \prod_v \mathrm{Op}(d_v)_v$. The photon $V_\gamma^{(\pm)}$ at vertex pair $(a, b)$ couples to $W$ iff $P_W$ has nontrivial Pauli at qubit $a$. Under the polar-photon Cor. 3.3, this means: the photon couples to a walk iff the walk has a Pauli deposit at the polar vertex.

Per Paper 21 §4.2's walk-to-Pauli map, a step $v_i \to v_{i+1}$ deposits Pauli at $v_{i+1}$. So a walk deposits Pauli at $P$ iff it *visits* $P$ during traversal. By Paper 21 §4.1 ("All matter-walk constructions start and end at $P$"), **every walk visits the polar vertex** — at minimum at start/end. Therefore every walk couples to the polar photon: this is the substrate statement of **photon-universality of electric coupling**.

QR-direction walks (matter) couple via $V_\gamma^{(+)} = X X$; NR-direction walks (antimatter) couple via $V_\gamma^{(-)} = Z Z$; mixed walks couple to both in proportion to their QR/NR fraction. This recovers the matter/antimatter symmetry of QED coupling.

## 4. Hardware realisation on Heron r2

A two-patch substrate uses 14 physical qubits, well within Heron r2's 156-qubit budget. Heavy-hex topology requires SWAP-bridges between the two 7-qubit Steane patches; estimated overhead 3–5 SWAPs per inter-patch CNOT, manageable.

**Pair production protocol** ($\gamma \to e^+ e^-$). Uses the polar photon vertex of Cor. 3.3:
1. Initialise patches 1, 2 in $|\bar 0\rangle$.
2. Apply $V_\gamma^{(+)}_{\mathrm{polar}}(1, 2) = X_{P_1} X_{P_2}$ via a single CNOT bridge between polar qubits of the two patches (followed by an X on either side, or equivalently a controlled-X with input prepared in $|+\rangle$).
3. Apply electron walk Pauli word on patch 1 (matter sector); apply positron walk Pauli word on patch 2 (= $H^{\otimes 7}$-conjugate of patch 1's walk, antimatter sector).
4. Measure all six stabilizers per patch + global polar-polar $X X$ and $Z Z$ correlators.
5. Expected syndrome: $(s_X, s_Z)_1 = (F_2, E_2)$ for $e^-$ per Paper 21a Table 1; $(s_X, s_Z)_2 = (E_2, F_2)$ for $e^+$ (CPT-conjugate). Global polar correlator $\langle X_{P_1} X_{P_2} \rangle = +1$ (matter polarisation channel).

This makes the photon vertex *explicit* (CNOT-bridge) where Paper 21a §5 has it *implicit* (joint vacuum preparation). The polar-polar inter-patch CNOT is the substrate-mechanical realisation of $\gamma \to e^+ e^-$.

**Compton scattering protocol** ($\gamma e^- \to \gamma e^-$):
1. Initialise patches 1, 2, 3 in $|\bar 0\rangle$.
2. Apply $V_\gamma^{(+)}_{\mathrm{polar}}(1, 2) = X_{P_1} X_{P_2}$ (incoming photon: entanglement between source patch 1 and electron patch 2).
3. Apply electron walk Pauli word on patch 2.
4. Photon "scatters" by transferring the polar entanglement from (1, 2) to (3, 2) via SWAP$_{P_1 P_3}$ followed by re-application of $V_\gamma^{(+)}_{\mathrm{polar}}(1, 2)$ (cancelling the original entanglement) and then $V_\gamma^{(+)}_{\mathrm{polar}}(3, 2)$.
5. Measure patch 2 syndromes (electron post-scatter) and patches 1, 3 polar correlators (incoming photon vacated, outgoing photon populated).
6. Expected: patch 2 syndrome unchanged (elastic scatter); polar correlation transferred from (1, 2) to (3, 2).

**Decay protocol** ($\mu^- \to e^- \bar\nu_e \nu_\mu$):
1. Three patches (21 qubits): muon on patch 1; electron + two neutrino targets on patches 2, 3, 4 (28 qubits — still well within budget).
2. Apply weak vertex (Y-Pauli inter-patch coupling, §5) to redistribute walk content.
3. Measure all four patches; check (a) syndrome conservation aggregated across all patches, (b) branching-ratio matching to PDG.

## 5. Other gauge bosons — sketch (not formalised yet)

**Gluons (SU(3)).** Eight generators expected; the X-stabilizer group has $2^3 = 8$ elements. Tentative identification: gluons live in the X-type sub-algebra of $N(S_N)/S_N$ supported on the QR-Heffter substrate. *Concrete formula deferred — need to check that the 8 stabilizer-coset X-operators close under commutation onto $\mathfrak{su}(3)$ structure constants. Plausible but open.*

**Weak bosons (W±, Z⁰, SU(2))**. Parity-violating, chiral. Natural candidate: Y-Pauli vertex operators $V_W^{ij}(a, b) = Y_a^{(i)} Y_b^{(j)}$. Under $H_N$, $Y \mapsto -Y$, so $Y_a Y_b \mapsto +Y_a Y_b$ (the double sign cancels) — Y-vertex is $H_N$-self-conjugate at the *operator* level. But Y has explicit imaginary structure ($Y = iXZ$), and the relative-phase $i$ is what gives chirality. The substrate signature of parity violation is then *complex coefficient in the Y-vertex*, not a real one as for the photon. *Concrete formula deferred — need to verify SU(2) structure constants close, and check that W± charge assignment matches the matter/antimatter polarity of inter-patch ZZ correlations.*

**Higgs.** No clean candidate yet. May correspond to the operation that pins a walk's carrier-knot via $n_q^q$ — i.e., the *mass-generation operation* not the *boson exchange*. This is consistent with Higgs being a *condensate*, not a clean propagating mode.

## 6. Neutrinos: K_8 Wilson amplitude, not Paper 6 torus-knot

Jim's question grouped neutrinos with photons. The grouping is structurally apt — but not for the reason originally posited in §7 of an earlier draft of this memo (the F_1-unknot reading). The correct picture is:

**Neutrinos are NOT in the $K_7$ closed-walk + Paper 6 torus-knot framework.** They require the $K_8$ extension developed in Papers 17–20, with the eighth vertex identified as the real octonion unit $1 \in \mathbb{R} \subset \mathbb{O}$ via the $\mathrm{Spin}(7) \subset \mathrm{Spin}(8)$ stabilization (Baez 2002 §3.2, Paper 20 §5.1). This bears out Jim's intuition that neutrinos sit *alongside* the boson question: both fall outside the $K_7$ walk-as-particle picture and require substrate extensions beyond it.

### 6.1 The $K_8$ Wilson amplitude recipe (Paper 20)

The unified Wilson recipe (Paper 20 eq. 5.1) for a phase soliton on a $K_n$ subgraph of $K_8$ is
$$
m \;=\; \frac{8}{N_v}\,\alpha^{N_e/2}\,\Bigl(1 + \frac{\alpha}{7}\Bigr)\bigl(1 + 3\alpha^2\bigr)\,m_{\rm Pl}
$$
where $N_v$ is the number of vertices and $N_e$ the number of edges of the accessible subgraph. The prefactor $8/N_v$ tracks the $\mathrm{Spin}(8)$-spinor-to-vector ratio; the $\alpha^{N_e/2}$ power is the Wilson loop over edges; the NLO/NNLO corrections $(1+\alpha/7)$ and $(1+3\alpha^2)$ are per-vertex and per-rank corrections.

Instantiating on $K_7$ ($N_v = 7, N_e = 21$) reproduces the electron mass to 8 significant figures:
$$
m_e / m_{\rm Pl} \;=\; (8/7)\,\alpha^{21/2}\,(1+\alpha/7)(1+3\alpha^2) \;=\; 4.185 \times 10^{-23} \;\;\;\text{(CODATA-matching)}
$$
Instantiating on $K_8$ ($N_v = 8, N_e = 28$) gives the lightest active neutrino:
$$
m_1 \;=\; (8/8)\,\alpha^{14}\,(1+\alpha/7)(1+3\alpha^2)\,m_{\rm Pl} \;=\; 14.84 \;\mathrm{meV}.
$$
The other generations follow from the observed mass-squared splittings: $m_2 = \sqrt{m_1^2 + \Delta m_{21}^2} \approx 17.16$ meV; $m_3 = \sqrt{m_1^2 + \Delta m_{31}^2} \approx 52.3$–$53.0$ meV depending on hierarchy convention.

**Reproducibility check** (independent numerical verification, this memo): with $\alpha = 1/137.035999084$, $m_{\rm Pl} = 1.221 \times 10^{19}$ GeV, $\Delta m^2_{21} = 7.42 \times 10^{-5}$ eV² and $\Delta m^2_{31} = 2.514 \times 10^{-3}$ eV² (NuFIT 6.0, normal ordering):
- $m_1 = 14.84$ meV ✓
- $m_2 = 17.16$ meV ✓
- $m_3 = 52.29$ meV (Paper 20 quotes 53.00; small difference traceable to $\Delta m^2_{31}$ choice)
- $\Sigma m_\nu = 84.3$ meV ✓ (Paper 20 quotes 85)

The Planck 2018 cosmological bound is $\Sigma m_\nu \lesssim 120$ meV. The substrate prediction sits at $\sim 70\%$ of the bound — falsifiable by the next iteration of CMB-S4 + DESI cosmological constraints, currently projected to reach $\sim 30$ meV sensitivity.

The KATRIN direct $m_{\nu_e}^{\rm eff}$ bound is $\lesssim 0.45$ eV (90% CL). The substrate prediction $m_{\nu_e}^{\rm eff} \lesssim 53$ meV sits four orders of magnitude below — well-satisfied, not falsifiable by KATRIN.

### 6.2 Why this bears on the $K_7$ boson representation

The boson algebra developed in §§1–4 lives on multi-patch $K_7$ substrate. The neutrino's $K_8$ extension means that any decay or scattering process involving neutrinos requires *mixed-graph* patches — some patches are $K_7$ (charged leptons, hadrons, mesons) and at least one is $K_8$ (the neutrino).

For the canonical weak-decay protocol $\mu^- \to e^- + \bar\nu_e + \nu_\mu$:
- $\mu^-$ (charged lepton): 7-qubit $K_7$ patch (Paper 21a §4).
- $e^-$ (charged lepton): 7-qubit $K_7$ patch.
- $\bar\nu_e, \nu_\mu$ (neutrinos): 8-qubit $K_8$ patches each.
- Total: $7 + 7 + 8 + 8 = 30$ qubits. Well within Heron r2's 156-qubit budget.

The weak vertex (open piece #2 in Paper 21a §10) is the operation that connects $K_7$ and $K_8$ patches. The Yukawa edges of $K_8$ (six edges from vertex 8 to the active/sterile orbits, per Paper 20 eq. 6.1) are the natural candidates for the weak-vertex inter-patch coupling.

### 6.3 Stabilizer code on $K_8$

Paper 21a's identification "$K_7$ = Steane $[\![7,1,3]\!]$" presumably extends to "$K_8$ = some $[\![8, k, d]\!]$ code." The natural candidate is the Reed–Muller $[\![8, 3, 2]\!]$ code (3 logical qubits on 8 physical), or a non-standard code adapted to the $\mathrm{Spin}(7) \subset \mathrm{Spin}(8)$ stabilization. *This is an open structural question — see §9.*

For hardware purposes, the 8-qubit neutrino patch is unambiguously bigger than a $K_7$ patch by one qubit. The eighth qubit hosts the real-octonion-unit / Higgs-vacuum direction and is the substrate analog of the "active-sterile distinguisher" axis.

### 6.4 Why neutrinos *feel* boson-like

The original grouping with photons is now justified differently than the F_1-unknot story:

1. Neutrinos sit *outside* the $K_7$ closed-walk catalog: they require a separate substrate framework ($K_8$ Wilson amplitudes), just as bosons require a separate framework (multi-patch $K_7$ stabilizer-normalizer-quotient).
2. Both neutrinos and bosons require special handling for chip realisation: neutrinos need a $K_8$ stabilizer code (8 qubits); bosons need inter-patch entangling vertices (none for single-walk circuits).
3. Both are essential for scattering and decay protocols (Phase 3 catalog classes 3–5 per Paper 21a §10's outlook).

Neutrinos *remain* fermions (Wilson-amplitude solitons with definite mass and definite particle/antiparticle distinction). They are not bosons. But the substrate machinery needed to host them is structurally adjacent to the boson-as-multi-patch-operation machinery of this memo.

## 7. Falsifiable predictions

1. **Polar-photon universality.** Two-patch pair-production protocols using the polar photon CNOT bridge must show $(0, 0)$ total winding and matter/antimatter syndrome anti-correlation between the two patches across all backends. Deviation from this anti-correlation pattern (matter-matter or antimatter-antimatter excess) falsifies the polar-photon identification.
2. **Polarisation duality.** The $V_\gamma^{(+)} = X_P X_P$ and $V_\gamma^{(-)} = Z_P Z_P$ modes must be exchanged by $H^{\otimes 7}$ on every backend. Asymmetry between X-coupled and Z-coupled polar correlators above shot noise falsifies the CSS-self-dual photon claim.
3. **Selection rule.** Photon vertices at non-polar single-qubit pairs (e.g., $X_a X_b$ for $a, b$ in stabilizer support) must produce *invalid* (codespace-leaving) syndromes when measured. A non-leaking syndrome at a non-polar single-qubit vertex falsifies Thm 3.2.
4. **Graded photon spectrum.** Photon vertices of different support weights should produce distinguishable signatures in pair-production efficiency / cross-section. If the polar (weight 1+1), Fano (weight 3+3 or 4+4), and Hamilton (weight 7+7) photons produce identical statistics, the graded algebra of §3.3 reduces to a single channel — falsifying the graded-spectrum claim.
5. **Neutrino mass-sum cosmological bound.** The Paper 20 $K_8$ Wilson amplitude predicts $\Sigma m_\nu = 85$ meV (this memo's independent verification: 84.3 meV). CMB-S4 + DESI projected sensitivity reaches $\sim 30$ meV by ~2030. A measured $\Sigma m_\nu < 50$ meV or $> 110$ meV would falsify the $K_8$ Wilson recipe with normal hierarchy.
6. **Neutrinoless double-beta decay $m_{\beta\beta}$.** The $K_8$ mass tower predicts a specific $m_{\beta\beta}$ as a function of PMNS angles + Majorana phases (Paper 20 §8); LEGEND-1000 sensitivity ($m_{\beta\beta} \sim 15$ meV) intersects the prediction band. Hard exclusion at this sensitivity falsifies the Majorana sterile assignment.
7. **$K_8$-vs-$K_7$ patch signature on Heron.** A bench-scale circuit running an 8-qubit $K_8$ neutrino patch alongside a 7-qubit $K_7$ charged-lepton patch should produce distinguishable Wilson-amplitude signatures (different $\alpha^{N_e/2}$ power scaling between the patch types). Failure to distinguish on the chip falsifies the $K_8$-extension hardware-realisability claim.

## 8. Where this slots into Paper 21a

Paper 21a (`papers/paper21a_theory.tex`) already adopts the conceptual framing this memo formalises. Specifically:

- **§3.6 "Scope: fermions and hadrons, not bosons"** (line 181) states "Bosons enter NWT not as walk-as-particle excitations but as *substrate-mediator operations* that act on the matter Hilbert space." This memo's Option (A) is the same claim.
- **§10.1 "Substrate-mediator algebra for the bosonic sector"** lists this as open piece #1 with three candidate readings: U(1)$_{em}$ = "sub-cycle propagator", SU(2)$_L$ = "NR-Hamilton substrate primitive", SU(3)$_c$ = "colour-labelled tri-cycle triplet". These are gestures; this memo replaces them with explicit operator-algebra content.

Proposed Paper 21a updates (modest, surgical):

1. **§10.1 closure for U(1).** Replace the "sub-cycle propagator" gesture with: "the photon is the symmetric CSS inter-patch vertex $V_\gamma^{ij}$ of [boson-representation memo], a $(X_a X_b + Z_a Z_b)$ element of $N(S_N)/S_N$ supported on at least two patches; the split-polarisation refinement gives the two transverse modes." Two to three sentences plus a forward-reference; does not require new §.
2. **§3.6 enrichment.** Add one paragraph after the existing bullet list pointing out that the substrate-mediator operations live in $N(S_N)/S_N$ — i.e., that "substrate-mediator" has a precise operator-algebra meaning, not just an informal one. Forward-reference to §10.1.
3. **Neutrino scope clarification.** Paper 21a §3 currently mentions neutrinos in the closed-walk-framework scope (§3.6: "matter content: charged leptons, neutrinos, mesons, hyperons, and nucleons"). This is misleading: neutrinos are *not* in the Paper 6 (p, q, m, n_q) framework, they are in the Paper 19/20 $K_8$ Wilson amplitude framework. Add a paragraph to §3 (or §10's open pieces) noting that neutrinos are handled by the $K_8$ extension (Paper 20), with active masses $\{14.84, 17.16, 53\}$ meV reproducing $\Sigma m_\nu = 85$ meV at the cosmological-bound level. Reference Paper 20 inline.

Gluon, $W^\pm$/$Z$, and Higgs identifications stay open in §10.1 (future work). The photon is the leading edge because:
- It's the simplest gauge structure (U(1), one generator).
- It's a prerequisite for §10.2 (interaction-vertex op): without the photon vertex made explicit, the QED scattering vertex has no substrate compilation.
- It's the gating piece for Paper 21b Phase 3 protocol class (2): pair-production circuits.

## 9. Open structural issues

**[CLOSED in §3.1]** Operator normalisation. The split-polarisation form $(V_\gamma^{(+)}, V_\gamma^{(-)}) = (X_a X_b, Z_a Z_b)$ resolves the $V_\gamma^2 \neq I$ issue by treating each polarisation as a distinct unitary, exchanged by CPT. Spin-1 emerges from the two-polarisation structure.

**[CLOSED in §3.2, §3.3]** Charge-vertex selection rule. Theorem 3.2 + Corollary 3.3 establish that the polar vertex is the unique single-qubit-pair photon vertex; Theorem 3.4 gives the graded algebra of higher-weight vertices via Z-syndrome cancellation.

**[CLOSED in §3.5]** Photon-walk coupling. Every walk visits the polar vertex (Paper 21a §4.1: walks start and end at $P$), so the polar photon couples universally to all walks. Recovers QED's photon-universality of electric coupling.

**[OPEN]** Stabilizer convention reconciliation. Paper 21a §4.1's description of X-stabilizer supports as "QR translates $\{1, 2, 4\} + i$" (weight 3) and "Fano-triangle complements through the polar vertex" (weight 4) are inconsistent. The polar-vertex-in-no-stabilizer reading required for Cor. 3.3 holds under either convention, but the higher-weight vertex algebra of Thm 3.4 depends on which is canonical. *Resolving the canonical convention is left to future work.*

**[OPEN]** Explicit Heron r2 circuit compilation. The polar-photon CNOT bridge is logically simple (1 CNOT between polar qubits of two patches), but heavy-hex topology requires SWAP-network compilation. Estimate: 3–5 SWAPs per inter-patch CNOT depending on patch placement. Pre-registration of qubit layout for Phase 3 protocol class (2) needed.

**[OPEN]** Wilson-loop formulation. The U(1) phase $\theta$ of $\exp(-i\theta V_\gamma^{(\pm)})$ should reduce to a Wilson loop integral; the substrate analog of $\alpha = e^2/4\pi$ at this level is unwritten. *Likely path: $\alpha$ from polar-vertex coupling frequency × Steane stabilizer measurement cadence on Heron.*

**[OPEN]** Graded photon spectrum. Section 3.3's substrate-energy-$\propto$-vertex-weight claim needs derivation. Plausibly: vertex weight ↔ number of substrate qubits perturbed ↔ substrate excitation count ↔ photon energy. Needs work.

**[OPEN]** Gluon SU(3) closure. Verify X-stabilizer coset operators close onto $\mathfrak{su}(3)$ structure constants under commutator. (Deferred to next iteration.)

**[OPEN]** Weak boson chirality. Derive parity violation from the imaginary structure of $Y$-vertex operator. (Deferred to next iteration.)

**[OPEN]** Higgs identification. No candidate yet. Plausibly the operation that pins a walk's carrier-knot via $n_q^q$ — mass-generation operation rather than boson exchange. Per Paper 20 §6, the $K_8$ edge from vertex 8 to vertex 0 (single fully-fixed scalar edge) is the "Higgs vacuum" direction. The Higgs boson may be a *substrate-mediator operation acting along this edge*, analogous to the photon vertex acting between two patches' polar qubits.

**[OPEN]** $K_8$ stabilizer code for neutrinos. The natural Heron-side realisation of neutrinos requires an 8-qubit stabilizer code analogous to Steane $[\![7,1,3]\!]$ for $K_7$. Candidates: Reed–Muller $[\![8, 3, 2]\!]$ (transversal $T$-gate code), or a custom $\mathrm{Spin}(7) \subset \mathrm{Spin}(8)$-adapted code that distinguishes the eighth vertex as the real octonion unit. *Action: design choice for Paper 20 follow-up; not blocking Paper 21a.*

**[OPEN]** Mixed-graph weak-vertex protocol. The $K_7$↔$K_8$ inter-patch coupling realising weak decay ($\mu \to e\bar\nu_e\nu_\mu$) requires a vertex operator that couples a 7-qubit patch to an 8-qubit patch. Substrate candidate: the six Yukawa edges of $K_8$ (Paper 20 eq. 6.1) restricted to the active orbit. *Concrete operator form open; would close decay-channel Phase 3 protocol class.*

**[OPEN]** Reconciliation of two mass frameworks. NWT has two distinct mass derivations: (a) Paper 6 torus-knot formula $m/m_e = (p^2+q^2)/(p_e^2+q_e^2) \cdot \beta/\beta_e \cdot \ln(8\beta)/\ln(8\beta_e) \cdot n_q^q$ for the 25-particle charged-lepton + hadron + meson compendium; (b) Paper 17–20 $K_n$ Wilson amplitude $m = (8/N_v)\alpha^{N_e/2}(1+\alpha/7)(1+3\alpha^2) m_{\rm Pl}$ for the electron at 8-sig-fig precision and the neutrinos at meV precision. Both reproduce the electron mass, but framework (a) gives the 25-particle structure while (b) gives the absolute scale + neutrino sector. The structural relationship between them is not in §9 of Paper 21a; *probably worth a short reconciliation paragraph there*.

## References

- Paper 21a (theory) and Paper 21b (experiment) of this bundle.
- nwt-substrate library, Zenodo concept DOI 10.5281/zenodo.20012027 (v0.2.0 = 10.5281/zenodo.20398451).
