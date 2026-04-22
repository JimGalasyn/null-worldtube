#!/usr/bin/env python3
"""
NWT Lagrangian -- L1: field-content specification.

With Paper 15 closed (b1 one-loop pipeline validated, b2 Spin(7)/Cl(0,7)/2T
chain complete), the next structural target is the NWT Lagrangian.

L1 pins down the MINIMAL FIELD CONTENT forced by the converging constraints
of Papers 6, 8, 10, 13, 14, and 15.  We do NOT yet write down all couplings:
L2 will add kinetic terms and the BPS potential; L3 will add the
Skyrme--Faddeev / Hopf topological stabilizer; L4/L5 verify that the
minimal theory reproduces the observed mass spectrum and gravitational
hierarchy.

==========================================================================
 CONSTRAINTS TO SATISFY
==========================================================================

  C1 (Paper 6)   Line tension at BPS point:
                    mu = 2 pi v^2      [v = condensate VEV]
                 Hierarchy:  m_{p,q,m} ~ mu * ln(8 beta) * (p^2 + q^2)
                 Implies:   ONE complex scalar with BPS U(1) gauge.

  C2 (Paper 8a) alpha = 1 / (25 pi sqrt(3) + 1)  emerges from AB phase on
                BPS vortex torus via Helmholtz eigensolver with
                kappa = R/xi, kappa^2 = (25 pi sqrt(3) + 1) / sqrt(2).
                Implies:   SAME U(1) as in C1 -- consistent.

  C3 (Paper 13) SM gauge group SU(3) x SU(2) x U(1) emerges from 2T
                crossing algebra; canonical fermions = (p,q,m) torus-knot
                solitons with gauge-consistent (p,q).
                Implies:   Gauge sector contains SM.  Either SM is direct
                           gauge data, OR it descends from a larger
                           symmetry broken along the Spin(7) chain.

  C4 (Paper 14) G = (8/7)^2 * alpha^21 * hbar c / m_e^2  with the
                trefoil/heptafoil/cinquefoil triangle.
                Implies:   Three distinguished torus knots T(2,3), T(2,5),
                           T(2,7) must be soliton solutions of the theory.

  C5 (Paper 15) m_e / m_Pl = (8/7)(1 + alpha/7) alpha^(21/2).
                Structural anatomy:  21 = dim so(7), 8/7 = dim(spinor)/
                dim(vector) for Spin(7) = B_3, (1+alpha/7) from seven
                K_7 vertices.  2T embeds in Spin(7) via Cl(0, 7) action
                on R^8 = O (octonions).
                Implies:   Octonion matter rep is natural; so(7) gauge
                           symmetry in the UV is structurally compatible.

  C6 (Paper 10) Dark sector = vortex defects at R/xi ~ 10^14 (string
                limit).  Same Lagrangian, same condensate, different
                topological sector.
                Implies:   No separate dark-sector Lagrangian; DM arises
                           as high-Q solitons of the same theory.

  C7 (b1 pipeline, 2026-04-22) BPS one-loop in 2D flat validated:
                Casimir Delta_mu = -0.279 reproduced with 2-DOF Grassmann
                ghost convention.
                Implies:   Faddeev--Popov ghost structure is fixed.

==========================================================================
 DERIVED MINIMAL FIELD CONTENT
==========================================================================

Claim: the minimal NWT Lagrangian has FOUR fields.

  F1.  psi : complex scalar (C-valued) --- the order parameter / condensate.
       This is NWT's fundamental field.  Supports U(1) gauge action.
       BPS vortex solitons (T(p,q) torus knots) are finite-energy
       configurations of psi.
       Physical role:  matter (via vortex defects) AND gauge condensate.

  F2.  A_mu : abelian gauge field.  Couples to psi via D_mu = d_mu - ie A_mu.
       At the BPS point lambda = e^2 / 2, the classical vortex mass is
       mu = 2 pi v^2 (Bogomolny 1976, Paper 6).

  F3.  n^a : S^2-valued Skyrme-Faddeev unit field, a = 1, 2, 3.
       Derived from psi via the Hopf map if psi is a CP^1 doublet
       (psi \in C^2), OR from a secondary auxiliary field.  Carries Hopf
       invariant Q_H = p * m for T(p, q) knot with phase winding m
       (Rybakov Path A).
       Physical role: stabilises finite-size knotted solitons; its Hopf
       number locks the topological sector.

  F4.  Omega : OPTIONAL ghost / Lagrange-multiplier field used to enforce
       the S^3/2I compactification boundary condition.  For the
       flat-space Lagrangian, Omega is absent; on S^3/2I it enforces
       anti-periodicity under the 2I binary-icosahedral identification.

The Spin(7) / Cl(0, 7) structure of Paper 15 is NOT an independent field
content: it emerges as the symmetry of the MODULI SPACE of torus-knot
solitons on the Heegaard torus of S^3/2I, not as a gauge symmetry of the
Lagrangian.  This is the key structural move: Spin(7) is an EMERGENT
symmetry of the topological sector, not a fundamental gauge symmetry.

Equivalently:  NWT is minimally an abelian Higgs + Skyrme-Faddeev model
(i.e. a relativistic condensate) on S^3/2I x R^{1,3}.  All of the rich
representation theory (Paper 15's 2T -> Spin(7), McKay, K_7, PSL(2,7))
acts on the torus-knot solution space, not on the Lagrangian fields
directly.

This is consistent with:
  (a) Rybakov's 2015 16-spinor framework -- psi in C^2 with Hopf-Whitehead
      reduction gives a 16-component Dirac-like real spinor on the
      S^3 fiber, the minimal Rybakov content.
  (b) Paper 6's ONE-complex-scalar BPS derivation.
  (c) Paper 13's emergence of SM group from crossings -- gauge dynamics
      lives in the knot intersection data, not in additional gauge
      fields beyond U(1).
"""

from __future__ import annotations

from dataclasses import dataclass


# ==========================================================================
# Field-content dataclass.
# ==========================================================================

@dataclass
class NWTField:
    name: str
    symbol: str
    target_space: str
    dof_real: int
    spin: str
    gauge_rep: str
    role: str
    papers: tuple


MINIMAL_CONTENT = [
    NWTField(
        name="F1 -- condensate",
        symbol="psi",
        target_space="C  (or  C^2 for CP^1 extension)",
        dof_real=2,
        spin="0 (scalar)",
        gauge_rep="U(1) charge -e",
        role="Order parameter; vortex cores are torus knots T(p,q)",
        papers=(6, 8, 10, 13, 14, 15),
    ),
    NWTField(
        name="F2 -- abelian gauge",
        symbol="A_mu",
        target_space="R (one-form)",
        dof_real=2,   # after gauge-fixing: 2 physical polarisations
        spin="1",
        gauge_rep="U(1) gauge connection",
        role="Carries magnetic flux of vortex; sets BPS scale via e",
        papers=(6, 8),
    ),
    NWTField(
        name="F3 -- Skyrme-Faddeev",
        symbol="n^a",
        target_space="S^2",
        dof_real=2,
        spin="0 (isotriplet unit)",
        gauge_rep="neutral",
        role="Stabilises finite-size knot; Hopf Q_H = p * m for T(p,q)",
        papers=(10, 14, 15),
    ),
]

OPTIONAL_CONTENT = [
    NWTField(
        name="F4 -- S^3/2I boundary ghost",
        symbol="Omega",
        target_space="R",
        dof_real=1,
        spin="0",
        gauge_rep="neutral",
        role="Lagrange multiplier for 2I anti-periodicity; only on S^3/2I",
        papers=(15,),
    ),
]


# ==========================================================================
# Constraint verification.
# ==========================================================================

@dataclass
class Constraint:
    label: str
    paper: int
    requirement: str
    field_satisfies: str


CONSTRAINT_CHECKLIST = [
    Constraint(
        "C1", 6,
        "mu = 2 pi v^2 BPS line tension",
        "F1 + F2 at lambda = e^2/2 (abelian Higgs BPS)",
    ),
    Constraint(
        "C2", 8,
        "alpha from AB phase on BPS vortex torus",
        "F1 + F2 provide the vortex; alpha emerges from "
        "Helmholtz eigenvalue of F2 on trefoil torus",
    ),
    Constraint(
        "C3", 13,
        "SU(3) x SU(2) x U(1) from 2T crossing algebra",
        "Crossings of F1 vortex cores generate the SM gauge data; "
        "no separate gauge fields needed beyond F2",
    ),
    Constraint(
        "C4", 14,
        "Trefoil/heptafoil/cinquefoil all exist as solitons",
        "F1 + F3 admits (2,q) torus knots for q = 3, 5, 7 "
        "provided Skyrme quartic c_4 > 0",
    ),
    Constraint(
        "C5", 15,
        "m_e/m_Pl = (8/7)(1+alpha/7) alpha^(21/2)",
        "Emerges from one-loop Casimir of F1 on S^3/2I with 2T/Spin(7) "
        "acting on the moduli space; no new Lagrangian field needed",
    ),
    Constraint(
        "C6", 10,
        "Dark sector as same Lagrangian, high-Q sector",
        "F1 + F2 + F3 with R/xi ~ 10^14 (string limit)",
    ),
    Constraint(
        "C7", 15,
        "Faddeev-Popov ghost structure (2 Grassmann DOF per complex ghost)",
        "Fixed at gauge-fixing stage of F2; validated in b1.5",
    ),
]


# ==========================================================================
# Output.
# ==========================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(" " + title)
    print("=" * 72)


def main() -> None:
    section("NWT Lagrangian -- L1: minimal field content")

    print()
    print("MINIMAL content (required):")
    print()
    for f in MINIMAL_CONTENT:
        print(f"  [{f.symbol:>5}]  {f.name}")
        print(f"       target  : {f.target_space}")
        print(f"       DOF     : {f.dof_real} real components")
        print(f"       spin    : {f.spin}")
        print(f"       gauge   : {f.gauge_rep}")
        print(f"       role    : {f.role}")
        print(f"       papers  : " + ", ".join(str(p) for p in f.papers))
        print()

    print("OPTIONAL content (background-dependent):")
    print()
    for f in OPTIONAL_CONTENT:
        print(f"  [{f.symbol:>5}]  {f.name}")
        print(f"       target  : {f.target_space}")
        print(f"       role    : {f.role}")
        print()

    section("Constraint audit")

    print()
    print(f"  {'cstr':>6}  {'paper':>6}  requirement")
    print("  " + "-" * 66)
    for c in CONSTRAINT_CHECKLIST:
        print(f"  {c.label:>6}  {c.paper:>6}  {c.requirement}")
        print(f"  {'':>6}  {'':>6}  -> {c.field_satisfies}")
        print()

    section("Structural summary")

    print("""
  THE MINIMAL NWT LAGRANGIAN HAS THREE FIELDS:  psi, A_mu, n^a.

  All of Papers 6-15's observational predictions follow from this
  three-field theory provided the couplings are tuned to the BPS point
  (lambda = e^2 / 2) and the Skyrme-Faddeev coefficient c_4 > 0 is chosen
  to set the knot scale xi via Derrick scaling.

  The Spin(7) / Cl(0, 7) / 2T structure of Paper 15 is EMERGENT: it acts
  on the moduli space of torus-knot solitons on the Heegaard torus of
  S^3/2I, not as a gauge symmetry of the Lagrangian.  This explains why
  the theory has SM (SU(3) x SU(2) x U(1)) visible gauge content -- the
  solitons CARRY knot-intersection data that organises into the SM group
  via 2T crossings -- while still exhibiting Spin(7) / McKay / K_7
  structure at the level of the amplitude (Paper 15).

  Next step (L2): kinetic + gauge-covariant + BPS potential terms,
  making contact with Paper 6 mu = 2 pi v^2.
""")


if __name__ == "__main__":
    main()
