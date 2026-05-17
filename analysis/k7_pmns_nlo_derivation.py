"""First-principles derivation of PMNS NLO corrections — exploratory.

The challenge: derive δθ_12 ≈ -1.8° and δθ_23 ≈ +2.7° from the K_7/K_8
substrate structure, not just match them numerically.

Honest finding (this script): the obvious candidate mechanism — per-vertex
(1+α/7) NLO corrections inherited from Paper 17 — is INSUFFICIENT in
magnitude (~0.1°) to account for observed deviations (~2-3°). The
dominant NLO source for θ_12 and θ_23 must come from a structural
effect we have NOT yet identified.

This is open work. Status documented honestly in Paper 20 §7.6.

What IS derivable:
  • TBM at LO from Spin(8) triality / Z_3 ⊂ S_3 outer automorphism
  • θ_13 = √(3α) from Spin(7) ⊂ Spin(8) breaking, 3 = dim(SU(3) fund)
  • δ_CP = -2π/3 from π_1(PSU(3)) winding number (today's result)

What is NOT yet derived (this script confirms):
  • δθ_12 ≈ -4α (numerical match, structural integer unexplained)
  • δθ_23 ≈ +6α to +7α (data can't distinguish, structure unknown)

This script:
  (1) Builds the TBM matrix (LO from Spin(8) triality).
  (2) Adds the θ_13 perturbation correctly (in PDG convention).
  (3) Tests the obvious NLO candidate: per-vertex (1+α/7) corrections.
  (4) Documents the magnitude shortfall — confirms this is open work.
"""
from __future__ import annotations

import math
import numpy as np


ALPHA = 7.2973525693e-3
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


# ---------------------------------------------------------------------------
# TBM matrix (Z_3 cyclic on (e,μ,τ), trimaximal ν_2)
# ---------------------------------------------------------------------------

s2 = 1.0 / math.sqrt(2)
s3 = 1.0 / math.sqrt(3)
s6 = 1.0 / math.sqrt(6)

U_TBM = np.array([
    [ 2*s6,  s3,    0   ],
    [-s6,    s3,    s2  ],
    [-s6,    s3,   -s2  ],
])

print("=" * 76)
print("STEP 1: TBM matrix from Z_3 ⊂ S_3 (Spin(8) triality) — derived")
print("=" * 76)
print()
print(f"  U_TBM = ")
for row in U_TBM:
    print(f"    [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
print()

# Verify
assert np.allclose(U_TBM @ U_TBM.T, np.eye(3))


# ---------------------------------------------------------------------------
# Add θ_13 via proper PDG rotation (not generic perturbation)
# ---------------------------------------------------------------------------

print("=" * 76)
print("STEP 2: Apply θ_13 = √(3α) rotation in PDG convention — derived")
print("=" * 76)

sin_13 = math.sqrt(3 * ALPHA)
cos_13 = math.sqrt(1 - sin_13**2)
theta_13 = math.degrees(math.asin(sin_13))

# PDG U = R_23(θ_23) R_13(θ_13, δ_CP) R_12(θ_12)
# Apply R_13 to U_TBM
R_13 = np.array([
    [cos_13, 0, sin_13],
    [0, 1, 0],
    [-sin_13, 0, cos_13],
])

# But TBM doesn't have R_13 ordering — we need to inject θ_13 carefully.
# Cleanest approach: extract angles from any unitary U.
def extract_angles(U):
    """Extract θ_12, θ_23, θ_13 from a unitary U following PDG."""
    s13 = abs(U[0, 2])
    c13 = math.sqrt(max(0, 1 - s13**2))
    s12 = abs(U[0, 1]) / c13 if c13 > 1e-12 else 0
    s23 = abs(U[1, 2]) / c13 if c13 > 1e-12 else 0
    th_12 = math.degrees(math.asin(min(1, s12)))
    th_23 = math.degrees(math.asin(min(1, s23)))
    th_13 = math.degrees(math.asin(min(1, s13)))
    return th_12, th_23, th_13


# Direct construction of PMNS = R_23(45°) × R_13(√(3α)) × R_12(arctan(1/√2))
# (i.e., apply θ_13 to the TBM angles)
def pmns(t12, t23, t13_deg, delta=0):
    """Build PMNS matrix from PDG angles."""
    s12, c12 = math.sin(t12), math.cos(t12)
    s23, c23 = math.sin(t23), math.cos(t23)
    s13, c13 = math.sin(t13_deg * DEG2RAD), math.cos(t13_deg * DEG2RAD)
    R23 = np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
    R13 = np.array([
        [c13, 0, s13 * complex(math.cos(delta), -math.sin(delta))],
        [0, 1, 0],
        [-s13 * complex(math.cos(delta), math.sin(delta)), 0, c13],
    ])
    R12 = np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])
    return R23 @ R13 @ R12


U_LO_plus_13 = pmns(
    math.atan(1 / math.sqrt(2)),       # θ_12 (TBM)
    math.pi / 4,                       # θ_23 (TBM)
    theta_13,                          # θ_13 = √(3α)
)

t12, t23, t13 = extract_angles(U_LO_plus_13)
print(f"  θ_12 = {t12:.3f}°  (TBM 35.26°, observed 33.45°, gap {t12 - 33.45:+.2f}°)")
print(f"  θ_23 = {t23:.3f}°  (TBM 45.00°, observed 47.70°, gap {t23 - 47.70:+.2f}°)")
print(f"  θ_13 = {t13:.3f}°  (= √(3α), observed 8.62°, gap {t13 - 8.62:+.2f}°)")
print()
print(f"  TBM + θ_13 alone leaves θ_12 and θ_23 at TBM values.")
print(f"  Remaining gaps: δθ_12 = {33.45 - t12:+.2f}°, δθ_23 = {47.70 - t23:+.2f}°")


# ---------------------------------------------------------------------------
# Test the per-vertex NLO ansatz
# ---------------------------------------------------------------------------

print()
print("=" * 76)
print("STEP 3: Test per-vertex (1+α/7) ansatz for δθ_12, δθ_23")
print("=" * 76)
print(r"""
  Paper 17's (1 + α/7) per-K_7-vertex correction applies to each
  Wilson edge. In the active flavor sector (orbit A = 3 vertices),
  the NLO factor is at most (1 + 3α/7) ≈ 1 + 0.003 per mass eigenstate.

  The asymmetry between ν_2 (3 vertices) and ν_3 (2 vertices, only μ, τ)
  is at most:
""")
delta_per_vertex = ALPHA / 7 * RAD2DEG
print(f"        max(δθ from per-vertex NLO) ~ α/7 in radians = {delta_per_vertex:.4f}°")
print()
print(f"  Observed deviations:")
print(f"        δθ_12 = -1.81°  =  {-1.81/delta_per_vertex:.0f} × (α/7 in deg)")
print(f"        δθ_23 = +2.70°  =  {+2.70/delta_per_vertex:.0f} × (α/7 in deg)")
print()
print(f"  → per-vertex (α/7) corrections are 30-50× too small.")
print(f"  → δθ_12 and δθ_23 require an O(α) effect, not O(α/7).")


# ---------------------------------------------------------------------------
# What WOULD give the right size?
# ---------------------------------------------------------------------------

print()
print("=" * 76)
print("STEP 4: What structural effect gives O(α) corrections to PMNS?")
print("=" * 76)
print(r"""
  Observed sizes:
        |δθ_12| / α (rad) = 1.81° / 0.4181° ≈ 4.33
        |δθ_23| / α (rad) = 2.70° / 0.4181° ≈ 6.46

  Both are O(α) (not O(α/7) and not O(α^(1/2))).

  CANDIDATE MECHANISMS:

  (A) Seesaw correction to active mass matrix.
      Integrating out the sterile generates δM_active ~ M_Dirac² /
      M_sterile ~ α^(9/2) × M_active ~ 10^(-10) × m_active. Too small.

  (B) Threshold corrections from the K_8 → K_7 reduction.
      The 8/7 spinor-vector ratio differs at NLO. Scale: α × (8/7
      something). Could be of right order O(α) ~ 0.42°.

  (C) Within-K_7 self-energy contributions from cross-A↔B edges.
      9 cross-orbit edges contribute to active flavor self-energy at
      O(α) via virtual sterile-loop processes. Scale: 9α/(K_7 edges)
      = 9α/21 ≈ 0.43α ≈ 0.18° — also too small.

  (D) NWT Lagrangian L_3 topological term (Hopf/Skyrme-Faddeev,
      Paper 16). At O(α), the Hopf charge mixes mass eigenstates
      with weights ~ generation × K_7 structure integer.

  Mechanism (B) and (D) are the leading candidates by scale. A first-
  principles derivation requires diagonalizing the L_3 effective mass
  matrix to O(α), which has not yet been done in the NWT program.

  ⟹ THIS IS A REAL OPEN PROBLEM, NOT LOW-HANGING FRUIT.

  Status documented in Paper 20 §7.6 as:
        "δθ_12 and δθ_23 are of order α (consistent with structural
         corrections), but the specific integer coefficients are not
         yet derived from first principles."
""")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 76)
print("SUMMARY: What we've derived vs. what is open")
print("=" * 76)
print(r"""
  DERIVED (first principles):
    θ_12, θ_23 at LO = TBM            ← Spin(8) triality (Paper 19 W3.3-K)
    θ_13 = √(3α)                      ← Spin(7) ⊂ Spin(8) breaking, 3 = dim SU(3)
    δ_CP = -2π/3                      ← π_1(PSU(3)) winding (today)
    sin δ_CP: J_CP = -0.0295          ← Jarlskog from above

  NUMERICALLY MATCHED (open):
    δθ_12 ≈ -4α (≈ -1.81°)            ← integer 4 unexplained
    δθ_23 ≈ +6α to +7α (≈ +2.5-2.9°)  ← can't distinguish with current data

  RULED OUT as dominant mechanism:
    Per-vertex (1+α/7) NLO            ← magnitude 30-50× too small
    Seesaw active-mass correction     ← α^(9/2), tiny

  REMAINING CANDIDATES (not derived):
    K_8 → K_7 threshold corrections   ← O(α), scale-plausible
    L_3 Hopf/Skyrme contributions     ← Paper 16, not yet expanded

  This is honest scientific progress: we've narrowed the search,
  ruled out the obvious mechanism, and identified the next steps.
  Paper 20 §7.6 reflects this honestly.
""")
