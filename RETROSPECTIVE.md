# Null Worldtube Theory — retirement retrospective

**Status: PROGRAM RETIRED — July 2026.**
**DRAFT — the independent audit verdict
(`2026-07-12-constants-provenance-disputes`, verdict `4032d04`) is now
incorporated (§2.5); the companion case study is decoupled (in preparation);
the one remaining `[DOI]` slot (§6, this retrospective's Zenodo record) is
filled once the record is minted. Do not publish before.**

## 1. Summary

Null Worldtube Theory was a research program (2026) claiming that the
Standard Model's dimensionless constants, particle masses, and Newton's
constant follow from torus-knot topology and the K₇/Spin(7) structural
algebra with zero free dimensionless parameters. Twenty-two papers were
published on Zenodo; an open-source library (`nwt-substrate`) computed the
claims; reproduction code was published (`nwt-analysis`).

In July 2026 the program's own audit machinery — precision scoring at
experimental uncertainty, value-provenance linting, a quantified
look-elsewhere test, and an independent memory-blind audit protocol — was
turned on the full claim set. **The claims did not survive.** The program is
retired, the repositories are archived unmodified as the record, and this
document plus the case-study publication (in preparation) are the
authoritative statement of what was found.

The decision to retire was made by the author on the basis of the results
below, not external criticism. Nothing has been deleted: every paper,
commit, and dataset remains available, because the audit trail — a
year-long, AI-amplified theory-building effort that constructed the
instruments of its own falsification and then used them — is the program's
actual contribution.

## 2. What the audit found

All results below are reproducible from the pinned code
(`nwt-substrate`, branch `audit-dag-prereg-surface`, commit `a8f3335`;
modules `benchmarks/surface.py`, `benchmarks/neighbouring_value.py`,
`docs/BENCHMARK_TRIAGE_2026-07-12.md`).

**2.1 Precision accounting (S-NOW).** Scored against each measurement's own
1σ uncertainty rather than a percent-level tolerance, 31 of 36 registered
dimensionless claims are excluded as exact statements:

| claim | residual | experimental precision | exclusion |
|---|---|---|---|
| 1/α = 25π√3 + 1 | 7.6 ppm | 1.5 × 10⁻¹⁰ | ~50,000σ |
| m_p/m_e (walk formula) | 0.11% | 6 × 10⁻¹¹ | ~3.6 × 10⁶σ |
| v_EW/m_e | 27.7 ppm | 2.6 × 10⁻⁷ | ~107σ |
| m_e/M_Pl (substrate-pure) | ~75 ppm | 11 ppm | ~7σ |

The five claims that survive at 2σ do so either because their targets are
loosely measured (Cabibbo λ, sin²θ₁₃, a broad resonance) or because they are
the very claims whose provenance is under dispute as fitted or
post-selected (η_B, Ω_b/Ω_c — the latter compatible at 0.006σ, the
signature of a fit to the central value).

**2.2 Look-elsewhere volume, measured.** The constants were assembled from
a menu: a structural-integer prefactor × a half-integer power of α × up to
two rational-coefficient correction stages. Running the same procedure
against **random targets** (`neighbouring_value.py`) fits them to a median
~2 ppm, with **~83% landing inside CODATA G's ±22 ppm error bar** (full
menu; ~24% even on a minimal 6-integer menu). A ppm-scale match to any
constant is therefore the expected outcome of the procedure and carries no
information about nature.

**2.3 Input-hygiene failure in the flagship claim.** The published
statement that the derived G matches CODATA to **−11 ppm, "inside the ±22
ppm experimental error bar"** (Paper 18; repeated in this repository's
README and website) was produced with the **measured CODATA α as input**
(`gravity/coupling.py` defaults to `ALPHA_QED`). Computed with the theory's
own α = 1/(25π√3+1), the chain misses G by ~150 ppm — approximately 7σ
**outside** the bar. This correction is issued regardless of any other
finding.

**2.4 The benchmark suite, classified.** A failure-mode triage of all 38
public benchmarks (`docs/BENCHMARK_TRIAGE_2026-07-12.md`): 18 menu-fitted,
13 standard textbook results with the substrate structure inert (e.g. C₆₀'s
174 modes = 3N−6 + icosahedral group theory; magic numbers; Hückel
counting), 3 self-comparisons that cannot fail (including the electron
anomalous moment scored against its own Schwinger expression), 3 driven by
measured inputs (including a hardcoded PDG α_s graded against itself and
attributed to a "K₇ Wilson loop"), 1 untested forward claim. **The published
statement that the master table is computed "with no row-specific tuning"
(Paper 19) is contradicted by the library's own source** (per-particle walk
integers, corrected post-hoc; per-state rational coefficients) and is
likewise corrected.

**2.5 Independent audit verdict.** The memory-blind Auditor (audit
`2026-07-12-constants-provenance-disputes`, verdict `4032d04`; evidence: pinned
clones only, SHA-256 manifest 13/13, no program memory or chat read) adjudicated
the four disputed rows and **refuted the DERIVED self-tag on every one**:

- **η_B (3α⁴/14) → POST_SELECTED** — the kept-closest survivor of a 17-formula
  sweep against the observed value; the same menu pattern fits 54.9% of random
  targets inside this row's own 2σ bar.
- **Ω_b/Ω_c (25α + 75α²) → FITTED** — the leading term alone was already
  compatible at 1.77σ, and the added 75α² lands 0.006σ from the Planck central
  value (a ≈0.4%-by-luck over-compatibility, the fit-to-central signature).
- **m_e/M_Pl → MOTIVATED** — factor by factor: the exponent counted, the
  per-edge √α matched via the non-rigorous Chern–Simons step, 8/7 identified,
  and the (1+α/7) bracket retained as an admitted empirical factor.
- **ρ_Λ → MOTIVATED** (inherited through m_e/M_Pl) with its h_Cox and α¹⁶
  factors **UNDERSPECIFIED** — no pinned provenance of either; the unique menu
  integer that lands hits 100% of random targets at its stated bar.

The Auditor also fixed the claimed order at **NLO**: the NNLO α² bracket is
documented target-selection — its coefficient computed from CODATA, then rounded
to "the nearest structural integer" — and is retired from claim status (it may
remain as code, but may never be cited as the claim). Both dispute exhibits
(§2.2 look-elsewhere, §2.3 the measured-α G chain) were reproduced independently
and were seed- and menu-robust. The single vulnerability the Auditor disclosed
on the record: the four replacement tags lean on the program's own retrospective
admissions, so η_B's POST_SELECTED in particular is closed by pinning the primary
17-formula sweep artifact — supplied as a dated appendix (audit deposit
`CLAIM.md` Appendix A, 2026-07-12), which pins the memo and adds a reproducible
enumeration (`eta_B_menu_density.py`) confirming the same menu fits ~55–63% of
random targets at Planck precision.

**2.6 Related retirements.** The black-hole-cosmogenesis branch (Paper 22
and the AoE/Cold-Spot motivation) was retired separately on 2026-07-12
after a pre-registered, blind-reviewed joint null found the anomaly
constellation not established (the one provenance-clean external leg:
p = 0.16; the apparent disk alignment was a 2-of-12 selection artifact).
An earlier program (vortex-vision) was retired 2026-07-07. All three nulls
are documented in the case study (in preparation).

## 3. What is corrected, specifically

1. **G "inside the error bar"** — withdrawn (§2.3). The honest figure is
   ~150 ppm / ~7σ outside, substrate-pure.
2. **"No row-specific tuning"** — withdrawn (§2.4).
3. **"80 observables at 0.1% median residual"** and all similar aggregate
   accuracy claims — reframed: the residuals are as reported, but they are
   postdictions whose agreement is (a) excluded at experimental precision
   where targets are sharp, and (b) expected under the measured
   look-elsewhere volume where they are not. No claim in the series should
   be cited as a confirmed prediction.
4. **"Zero free parameters"** — withdrawn as an aggregate claim: the
   per-particle integer assignments, per-state rationals, and staged
   correction coefficients are free discrete parameters drawn from a menu
   whose fitting power is quantified in §2.2.

## 4. What survives — and, explicitly, what does not

A full salvage inventory was compiled at retirement under one rule of
inclusion: **a result survives if and only if it would be true had NWT never
been proposed** (`SALVAGE_INVENTORY.md`, to be published with the case
study, in preparation).
In summary:

**Salvaged:**

- **Soliton physics of the Eto–Hamada–Nitta model.** The reproduction of the
  published E≈6000 knot-binding benchmark (arXiv:2407.11731); the wrapped-∂a
  flux/linking lock; and stable torus-knot solitons *below the published
  N_link ≥ 4 floor* down to N_link = 1 (ring, trefoil, cinquefoil, septafoil,
  T(3,4), T(3,5); SHA-pinned field configurations). These are properties of
  the EHN model, true regardless of NWT.
- **The software.** `jax-solitons` is a general engine for classical
  field-theory solitons, independent of any NWT claim, and remains
  maintained (DOI 10.5281/zenodo.20774254), along with the model-agnostic
  numerical methods, certificate, and event-graph infrastructure.
- **The negative results.** "Simple α-polynomials over K₇/Spin(7) integers
  do not encode the fundamental constants at experimental precision" is
  itself an empirical finding — established here with instruments and
  receipts, across a claim family that has recurred in the literature for
  a century.
- **The method.** The audit architecture (σ-based scoring, provenance
  disputes with default-deny, the neighbouring-value instrument, the
  memory-blind auditor protocol, pre-registration with separate
  freeze/run commits) is domain-general; its validation is precisely that
  it falsified the program that built it.

**NOT salvaged (explicit, so nothing is resurrected by ambiguity —
mirroring `SALVAGE_INVENTORY.md` §F):**

- All dimensionless-constant derivations and the substrate→observables
  bridge (α, m_p/m_e, G ppm, n_s, r, N_e, …) — the 83%-of-random-targets
  result means every past "match" is attributable to apparatus freedom.
- The particle identifications (knot = hadron, det = n_q, framing = weak
  isospin, unknot = lepton, twist ladder = β-decay, composition-law mass
  additivity) — the *field configurations* survive; their particle names
  do not.
- The η_B / baryogenesis chain (μ₅→⟨Lk⟩ remains a true statement about
  forced injection in the simulation; its cosmological reading is dead),
  the K₇/CMB chirality claims, and all of cosmogenesis (retired separately
  2026-07-12 by the AoE joint null).
- The K₇/octonion/substrate algebra *as physics* — it survives only as the
  combinatorial bookkeeping it always was.

## 5. Reopen conditions (frozen)

The program retires with its remaining forward claims frozen, not erased.
To be precise about the relation to §4: these claims are **not salvaged** —
per the inventory's rule they are not established truths about anything —
they are the pre-registered conditions under which the retired claims would
earn re-examination. NWT would warrant reopening if and only if:

1. A **frozen K₈ dark-matter rung** is observed (the per-rung integers N_e
   are pinned at retirement; any post-hoc integer choice voids this
   clause) — e.g. a ~2 keV sterile-neutrino line (XRISM) or a ~98 GeV
   direct-detection state (LZ-G3);
2. A **forcing chain** is exhibited that derives a claimed constant
   *exclusively* — with no menu of alternative integers/coefficients —
   and demonstrably beats the measured noise-fit baseline of §2.2; or
3. The **S-FORWARD register** (post-2018 measurement updates, append-only)
   shows sustained convergence of new measurements toward a frozen
   prediction. Its first entry (CODATA-2022 α) moved away.

## 6. Record

- Papers 1–22: Zenodo DOIs, unchanged, each annotated with a related-
  identifier link to this retrospective's record `[DOI]`.
- Code: this repository, `nwt-substrate`, and `nwt-analysis` are archived
  read-only at their final states.
- Audit trail: pre-registrations, audit deposits, and verdicts will be
  published with the case study (in preparation).

## 7. Method note and acknowledgment

This program was built in intensive collaboration with AI systems (Claude),
which accelerated both the construction of the theory and — the case
study's central subject — the construction of the machinery that killed it.
The failure modes documented here (confirmation-amplification, fit-then-
interpret, yardstick equivocation, selection-as-signal) were catalogued
from the inside. Readers evaluating similar AI-amplified theory-building
programs may find the countermeasures more transferable than the physics.

The D12RG community's on-record cautions — in particular regarding the
rigor of the Chern–Simons functional-integral step — were correct, and are
acknowledged with thanks.

## 8. Lineage note

NWT descended from the toroidal-electron family of models (Williamson &
van der Mark's confined-photon electron and its successors). Two facts about
our own conduct toward that lineage belong in the record:

1. Our early survey treated the family's independent convergence on
   α-as-geometric-ratio as support. Under this retrospective's accounting,
   that convergence is what shared apparatus freedom predicts — several
   researchers drawing small-integer/π/√ combinations near 137 from the same
   menu is multiplicity, not independence (§2.2). Convergence-as-evidence
   was one of our own catalogued failure modes, and it operated on us
   through this lineage.
2. The one family mechanism we tested quantitatively — the Robinson
   charge-overlap model of nuclear binding — is falsified by that test: the
   mechanism yields net *repulsion* for the deuteron (+0.64 MeV predicted
   vs −2.22 MeV measured; no parameter in the scanned range binds it).
   The reproduction script is in the published analysis code. We report
   this as a result about a model we imported, not as commentary on any
   person or community.

We offer no verdicts on claims we have not tested. The instruments published
with this retrospective — precision scoring at experimental uncertainty,
input-hygiene checks, and the neighbouring-value look-elsewhere test — apply
to any closed-form-fits-constants program, ours first of all. Anyone who
wishes to audit a claims table, including their own, can.

— Jim Galasyn, July 2026
