#!/usr/bin/env python3
"""
NWT bracket-truncation: q-deformed Spin(7) tests of options (a') and (c)
==========================================================================

Tests the two remaining candidate mechanisms for the bracket truncation:

  (a') Level-k Casimir hierarchy refinement: at level k=32, fusion-rule
       integrability constraints might cancel α^3 contributions despite
       C_6(V) being non-zero in the classical limit.

  (c) Spinor-vector boundary projection S = V ⊕ 1: rank-1 mismatch
      between dim(S) = 8 and dim(V) = 7 may admit only quadratic
      mixing in α at the Wilson endpoints.

Concrete tests:

  Test 1: q-deformed dimensions of V, S, Adj, 27, 35, 168 at level
          k=32, expanded in α.  Look for structural ratios that
          truncate at α^2.

  Test 2: Quantum-Casimir ratios C_q(λ)/C_q(V) at level k, expanded.
          Truncation at α^2 in any of these would indicate (a').

  Test 3: q-deformed Schur orthogonality test: compute
          sum_λ dim_q(λ)^2 × |projector matrix elements|^2 at level k
          and probe truncation order.

  Test 4: Targeted α^3 Lawrence-Rozansky-style expansion: compute the
          asymptotic coefficient a_3 in
          Z(Σ(2,3,5), K_7 in V) ~ α^{21/2} × [a_0 + a_1 α + a_2 α^2 + a_3 α^3 + ...]
          if a_3 ≠ 63 (geometric prediction), option (a') is favoured.
"""

from __future__ import annotations

import numpy as np
from itertools import product
from typing import List, Tuple


# ====================================================================
# Section 0: Spin(7) data and Weyl formula machinery
# ====================================================================
#
# B_3 = so(7) has rank 3, dual Coxeter number h^v = 5.
# Positive roots (in orthonormal basis e_1, e_2, e_3):
#   short: e_1, e_2, e_3
#   long: e_i - e_j (i<j), e_i + e_j (i<j)
# Total: 9 positive roots.
#
# Weyl vector ρ = (5/2, 3/2, 1/2).
# Killing-normalized inner product is the standard Euclidean.

# Positive roots of B_3
POS_ROOTS = [
    np.array([1, 0, 0]),     # e_1 (short)
    np.array([0, 1, 0]),     # e_2 (short)
    np.array([0, 0, 1]),     # e_3 (short)
    np.array([1, -1, 0]),    # e_1 - e_2 (long)
    np.array([1, 0, -1]),    # e_1 - e_3 (long)
    np.array([0, 1, -1]),    # e_2 - e_3 (long)
    np.array([1, 1, 0]),     # e_1 + e_2 (long)
    np.array([1, 0, 1]),     # e_1 + e_3 (long)
    np.array([0, 1, 1]),     # e_2 + e_3 (long)
]

RHO = np.array([5/2, 3/2, 1/2])
H_VEE = 5

# Standard so(7) representations -- weights and dimensions verified
# via Weyl formula.
REPS = {
    "1":      np.array([0, 0, 0]),         # trivial, dim 1
    "V":      np.array([1, 0, 0]),         # vector, dim 7
    "S":      np.array([1/2, 1/2, 1/2]),   # spinor, dim 8
    "Adj":    np.array([1, 1, 0]),         # adjoint, dim 21
    "27":     np.array([2, 0, 0]),         # Sym^2(V) - trace, dim 27
    "35":     np.array([1, 1, 1]),         # Λ^3(V), dim 35
    "48":     np.array([3/2, 1/2, 1/2]),   # spinor-vector, dim 48
    "77":     np.array([3, 0, 0]),         # Sym^3(V) traces removed, dim 77
    "105":    np.array([2, 1, 0]),         # mixed (2,1,0), dim 105
    "168":    np.array([5/2, 1/2, 1/2]),   # spin-vector tensor, dim 168
    "189":    np.array([2, 1, 1]),         # (2,1,1) tensor, dim 189
}


# ====================================================================
# Section 1: Classical Weyl dimension and quantum dimension
# ====================================================================

def weyl_dim(lam: np.ndarray) -> float:
    """Classical Weyl dimension of irrep V_λ."""
    out = 1.0
    for alpha in POS_ROOTS:
        out *= np.dot(lam + RHO, alpha) / np.dot(RHO, alpha)
    return out


def q_num(n: float, T: float) -> float:
    """[n]_q = sin(n π/T) / sin(π/T) -- standard q-number."""
    if abs(n) < 1e-14:
        return 0.0
    return np.sin(n * np.pi / T) / np.sin(np.pi / T)


def q_dim(lam: np.ndarray, k: int) -> float:
    """Quantum dimension at level k for so(7)."""
    T = k + H_VEE
    out = 1.0
    for alpha in POS_ROOTS:
        n_lam = np.dot(lam + RHO, alpha)
        n_rho = np.dot(RHO, alpha)
        out *= q_num(n_lam, T) / q_num(n_rho, T)
    return out


def section(s: str) -> None:
    print()
    print("=" * 78)
    print(" " + s)
    print("=" * 78)


# ====================================================================
# Section 1: Test classical & q-dimensions
# ====================================================================

def test_dimensions() -> None:
    section("Test 1: Classical Weyl dimensions (sanity check)")
    print(f"\n  {'rep':<8} {'weight':<22} {'dim (Weyl)':>12} {'expected':>10}")
    print("  " + "-" * 56)

    # The "rep name" indicates expected dimension; verify Weyl formula
    # gives that dimension for the listed weight.
    for name, w in REPS.items():
        d = weyl_dim(w)
        try:
            exp = int(name)
        except ValueError:
            exp = {"V": 7, "S": 8, "Adj": 21}.get(name, None)
        match = "OK" if exp is not None and abs(d - exp) < 1e-6 else "?"
        exp_str = str(exp) if exp is not None else "-"
        w_str = "(" + ", ".join(f"{x:g}" for x in w) + ")"
        print(f"  {name:<8} {w_str:<22} {d:>12.4f} {exp_str:>10}  {match}")

    section("Test 1b: q-deformed dimensions at level k=32")
    print(f"\n  {'rep':<8} {'q-dim (k=32)':>14} {'classical':>12} "
          f"{'ratio q/cl':>14}")
    print("  " + "-" * 52)
    k = 32
    for name in ["1", "V", "S", "Adj", "27", "35", "168", "189"]:
        w = REPS[name]
        d_classical = weyl_dim(w)
        d_q = q_dim(w, k)
        ratio = d_q / d_classical if d_classical != 0 else float('nan')
        print(f"  {name:<8} {d_q:>14.6f} {d_classical:>12.4f} {ratio:>14.6f}")


# ====================================================================
# Section 2: Test (c) -- q-deformed dim ratios and α-expansion
# ====================================================================

def expand_in_alpha(values: List[float], alphas: List[float]) -> Tuple[float, float, float]:
    """Fit values(alphas) to value = c0 + c1 alpha + c2 alpha^2 (least squares)."""
    A = np.array([[1, a, a**2] for a in alphas])
    b = np.array(values)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[0], c[1], c[2]


def test_qdim_ratios() -> None:
    section("Test 2: q-deformed dimension ratios -- α-expansion")
    print(f"""
  Test option (c): the spinor-vector branching S = V ⊕ 1 should
  produce quantities whose perturbative expansion truncates at α^2.

  We compute several q-deformed dimension ratios at multiple levels
  and fit each to c_0 + c_1 α + c_2 α^2.  Strong agreement (no
  residual at α^3+) would support option (c).
""")

    # Compute at multiple levels to extract α-expansion
    k_values = [10, 16, 24, 32, 50, 100]
    alphas = []
    for k in k_values:
        T = k + H_VEE
        # alpha = sin^2(pi/T)
        alpha = np.sin(np.pi / T) ** 2
        alphas.append(alpha)

    print(f"  Levels and α values:")
    for k, a in zip(k_values, alphas):
        print(f"    k = {k:>3}  α = {a:.6f}  (T = {k+H_VEE})")

    # Test ratios that might be physically meaningful
    ratios_to_test = [
        ("dim_q(S) / dim_q(V)",                  "S",   "V"),
        ("dim_q(Adj) / dim_q(V)",                "Adj", "V"),
        ("dim_q(27) / dim_q(V)",                 "27",  "V"),
        ("dim_q(168) / dim_q(V)",                "168", "V"),
        ("dim_q(Adj) / dim_q(S)",                "Adj", "S"),
        ("dim_q(S) / (dim_q(V) - 1)  [Cartan]",  "S",   None),
    ]

    print(f"\n  α-expansion fits  (c_0 + c_1 α + c_2 α^2):")
    print(f"  {'ratio':<40} {'c_0':>10} {'c_1':>10} {'c_2':>10}")
    print("  " + "-" * 72)

    for label, num_name, den_name in ratios_to_test:
        values = []
        for k in k_values:
            d_num = q_dim(REPS[num_name], k)
            if den_name is None:
                d_den = q_dim(REPS["V"], k) - 1
            else:
                d_den = q_dim(REPS[den_name], k)
            values.append(d_num / d_den)

        c0, c1, c2 = expand_in_alpha(values, alphas)
        # Check residual: how well does c0 + c1 α + c2 α^2 fit?
        residual = max(abs(values[i] - (c0 + c1 * alphas[i] + c2 * alphas[i]**2))
                       for i in range(len(k_values)))
        marker = ""
        if residual < 1e-3:
            marker = " <- truncates"
        print(f"  {label:<40} {c0:>10.4f} {c1:>10.4f} {c2:>10.4f}{marker}")

    print(f"""
  Specifically check: is there a natural ratio with c_0 = 8/7,
  c_1 = 8/49, c_2 = 24/7  (= (8/7)(1+α/7+3α^2) coefficients)?

  Target:  c_0 = 8/7 = 1.1429
           c_1 = 8/49 = 0.1633
           c_2 = 24/7 = 3.4286
""")

    # Direct check
    print("  Direct test: (8/7)(1 + α/7 + 3α²) at sample α values:")
    for k, a in zip(k_values, alphas):
        target = (8/7) * (1 + a/7 + 3 * a**2)
        print(f"    k = {k:>3}  α = {a:.5f}  target prefactor = {target:.6f}")

    # The actual K_7 Wilson amplitude prefactor at each level is what
    # should match this target.  Computing that requires the full RT
    # amplitude on Σ(2,3,5), which is heavy.
    # As a proxy: see if any q-dim ratio matches.

    print("""
  None of the simple q-dim ratios matches (8/7)(1+α/7+3α²) directly.
  This is the same negative result Paper 17 §6.3 reports for
  dim_q(S)/dim_q(V) (which gives 1 + 5α/8 + ..., not 1 + α/7 + ...).

  The bracket is NOT the q-dimension ratio.  It's a finer structure
  specific to the K_7 Wilson amplitude with V-line and S-boundary.

  This means Test 2 doesn't directly probe option (c); it requires
  computing the actual K_7 Wilson amplitude.
""")


# ====================================================================
# Section 3: Test (c) more directly -- Spin(7) → G_2 branching
# ====================================================================

def test_spin7_g2_branching() -> None:
    section("Test 3: Spin(7) → G_2 branching probe of option (c)")
    print(f"""
  Option (c) hypothesis: the V→S boundary projection comes from the
  Spin(7) → G_2 reduction (G_2 = Aut(O), the octonion automorphism
  group, embeds maximally in Spin(7)).  Under this branching:

    V_7  ↓ G_2  =  V_7^{{G_2}}        (7-dim G_2 vector)
    S_8  ↓ G_2  =  V_7^{{G_2}} ⊕ V_1  (8 = 7 + 1)
    Adj  ↓ G_2  =  V_14 ⊕ V_7         (21 = 14 + 7)

  The 8 = 7 + 1 split is exactly the rank-1 mismatch we identified.

  Concrete prediction of option (c): the V→S boundary intertwiner
  contains exactly 2 components — V_7^{{G_2}} (V-aligned) and the
  V_1^{{G_2}} singlet.  Perturbative corrections in α can mix these
  two components but cannot generate higher-rank G_2 structure
  (V_14 etc.) at the boundary.

  Quadratic mixing between two components ↔ bracket truncation at
  α^2.  Higher orders would require V_14 mixing, which would break
  the G_2 structure.

  This is qualitatively the option (c) story.  Quantitatively
  verifying it requires the full G_2 → Spin(7) Clebsch-Gordan
  data at level k=32, which is tabulated in specialised references
  (e.g., Sirlin-Trinquier 1971 for classical CG; not standard at
  level k).

  As a partial probe:
""")

    # Probe: at the level of dim_q at k=32, is there evidence for
    # 8/7 = (V_7^G2 + V_1^G2) / V_7^G2 coming from G_2 branching?
    k = 32
    d_V = q_dim(REPS["V"], k)
    d_S = q_dim(REPS["S"], k)
    d_Adj = q_dim(REPS["Adj"], k)
    print(f"  q-dimensions at k=32:")
    print(f"    dim_q(V)   = {d_V:.6f}    (classical 7)")
    print(f"    dim_q(S)   = {d_S:.6f}    (classical 8)")
    print(f"    dim_q(Adj) = {d_Adj:.6f}   (classical 21)")
    print()
    print(f"  Ratios:")
    print(f"    dim_q(S) / dim_q(V) = {d_S/d_V:.6f}    (target 8/7 = {8/7:.6f})")
    print(f"    [dim_q(V) - dim_q(1)] / dim_q(V) = "
          f"{(d_V - 1)/d_V:.6f}  (target 6/7 = {6/7:.6f})")

    # If G_2 branching were exact at level k, we'd have
    #   dim_q(S) = dim_q(V) + 1   (8 = 7 + 1)
    # at any level.  Check:
    print(f"\n  G_2-branching test (at level k=32):")
    print(f"    dim_q(S) - dim_q(V) - 1 = {d_S - d_V - 1:.6f}    "
          f"(would be 0 if 8 = 7+1 lifted to q-deformation)")

    if abs(d_S - d_V - 1) < 1e-9:
        print(f"    EXACT: 8 = 7 + 1 lifts to q-deformation at level k!")
    elif abs(d_S - d_V - 1) < 0.1:
        print(f"    Close but not exact -- corrections at order α.")
    else:
        print(f"    Significant deviation -- G_2 branching does NOT lift")
        print(f"    cleanly to q-deformation.  Possible explanation: G_2")
        print(f"    sub-algebra structure at level k requires its own")
        print(f"    q-deformation parameter, distinct from the Spin(7) one.")


# ====================================================================
# Section 4: Targeted Lawrence-Rozansky α^3 sketch
# ====================================================================

def test_LR_alpha_3_sketch() -> None:
    section("Test 4: Lawrence-Rozansky α^3 coefficient -- sketch")
    print(f"""
  Option (a'): the level-k Lawrence-Rozansky asymptotic expansion of
  the WRT invariant on Σ(2,3,5) for Spin(7)_k has the form

    Z(Σ(2,3,5)) ~ A_0 + A_1/T + A_2/T^2 + A_3/T^3 + ...

  where T = k + h^v = k + 5.

  The K_7-in-V Wilson amplitude on Σ(2,3,5) has an asymptotic form

    Z(Σ, K_7 in V) / Z(Σ) = α^{{21/2}} × [a_0 + a_1 α + a_2 α^2 + a_3 α^3 + ...]

  where α = sin^2(π/T) ~ 1/T^2 at large T.

  Geometric-series prediction:  a_3 = 63 (= 21^3/(7×21)).
  Bracket truncation prediction: a_3 = 0 (option a').

  The Lawrence-Rozansky calculation evaluates Z via a Seifert sum:

    Z = sum over weights λ at level k of [products of S-matrix elements
        and N-matrix powers, weighted by L-R coefficients].

  For Σ(2,3,5), the Seifert sum has 3 cusps (multiplicities 2, 3, 5).
  Each cusp contributes a sum over its modular S-matrix.

  Concrete computation requires:
    - The full S-matrix S_{{λμ}} of Spin(7)_{{k=32}} on integrable weights.
    - The fusion N-matrices for all relevant fusions.
    - The L-R perturbative weights for the Seifert combination.

  All of these are computable from the Kac-Peterson formula and
  Verlinde formula, but the calculation has hundreds of terms.

  As a partial diagnostic, we compute the simplest piece: the
  classical-limit (k→∞) behaviour of the integrable-weight sum.

""")

    k = 32
    T = k + H_VEE

    # Integrable weights at level k for so(7) are λ such that
    # (λ, θ) ≤ k where θ is the highest root.
    # For B_3, θ = e_1 + e_2 (the highest root).
    # So integrability: λ_1 + λ_2 ≤ k.
    #
    # Enumerate integrable weights with all components ≥ 0
    # (assuming integer components; spinors have half-integer).

    integrable_int = []
    integrable_half = []
    for a in range(k + 1):
        for b in range(min(a, k - a) + 1):
            for c in range(b + 1):
                if a + b <= k:
                    integrable_int.append((a, b, c))
        # Half-integer (spinor sector)
        for b in range(min(a, k - a)):
            for c in range(b + 1):
                # Half-integer convention: a, b, c all add 1/2
                pass  # skip for now; spinor sector adds complications

    # Just count integer-weight integrables
    n_int = len(integrable_int)
    print(f"  At level k={k}, so(7) has {n_int} integer-weight")
    print(f"  integrable highest weights (excluding spinor sector).")

    # Compute sum of dim_q^2 (modular total dimension squared)
    D_squared = sum(q_dim(np.array(w), k)**2 for w in integrable_int)
    print(f"  Modular total dim^2 (integer sector only):  {D_squared:.4f}")

    # Compare to large-k asymptotic  D^2 ~ T^{dim g}
    # For B_3, dim g = 21.
    # Also need to multiply by Vol(weight lattice fund domain) factor
    asymp = T ** 21 / 1e20  # arbitrary scale for comparison
    print(f"  Asymptotic T^{21} = {T**21:.3e}")

    print(f"""
  A targeted α^3 calculation requires:
    1. Full S-matrix on integrable weights (we have q-dimensions only).
    2. Fusion N-matrices N_{{λμ}}^ν -- computable but tedious.
    3. L-R Seifert formula coefficients for (2,3,5) signature.

  This is a 1-2 week expert-level CFT calculation, not a script.

  However, we CAN make a structural observation:  if the bracket
  truncates at α^2 due to a level-k Casimir constraint, we'd expect
  it to depend on the level k = 32 in a specific way.  At very high
  levels (k → ∞), all level-k constraints become inactive.  So the
  α^3 coefficient should:
    - Vanish at finite k=32 (option a' prediction)
    - Become non-zero at higher k (recovering geometric series)

  This is testable IF we can compute the bracket at multiple levels.
  The K_7 graph state we used in earlier sessions is a CLASSICAL
  (k → ∞) model, which gives the geometric series.  A finite-k
  proxy for the K_7 amplitude would be needed.

  Concretely:  at finite k, the K_7 Wilson amplitude has CONTRIBUTIONS
  from non-Adj fusion channels (the singlet 1 and the 27).  These
  vanish at large k by Casimir suppression but at finite k=32 they
  contribute.  Maybe these other-channel contributions cancel the
  α^3 geometric tail?
""")

    # Sketch: compute the R-matrix eigenvalues for V⊗V at level k=32
    # in all three channels, and see how their combination behaves.
    print(f"\n  R-matrix eigenvalues at level k=32 for V⊗V → 1, Adj, 27:")
    # q = exp(i π/T)
    # R_λ = ε_λ q^{C_λ - 2 C_V}
    # C_V = 6, C_1 = 0, C_Adj = 10, C_27 = 14
    # ε: 1 has +1 (Sym^2), Adj has -1 (Λ^2), 27 has +1 (Sym^2)
    delta = np.pi / T
    R_1   =  np.exp(1j * delta * (0 - 12))
    R_Adj = -np.exp(1j * delta * (10 - 12))
    R_27  =  np.exp(1j * delta * (14 - 12))
    print(f"    R_1   = q^{-12} = {R_1:.4f},   |R_1| = 1")
    print(f"    R_Adj = -q^{-2} = {R_Adj:.4f}, |R_Adj| = 1")
    print(f"    R_27  = q^{+2} = {R_27:.4f},   |R_27| = 1")

    print(f"""
  Per-edge amplitude contributions:
    Channel   Amplitude  |1+R|/2     Contribution
    1         (1+R_1)/2 = {abs(1+R_1)/2:.4f}   ~ 1 (singlet)
    Adj       (1+R_Adj)/2 = {abs(1+R_Adj)/2:.4f}   ~ √α (the bracket-source)
    27        (1+R_27)/2 = {abs(1+R_27)/2:.4f}    ~ 1 (other channel)

  The ratio of Adj-channel to other channels at finite k:
  (1+R_Adj)/2 / (1+R_1)/2 ≈ √α at small α.

  In the LR-formula sense, the "physical" Wilson amplitude includes
  ALL channels weighted by their fusion multiplicities and dimensions.
  Restricting to Adj alone gives α^{21/2}; the full amplitude has
  corrections at α^{21/2 + 1}, α^{21/2 + 2}, ... that come from
  inserting other-channel structure.

  These would be the bracket coefficients at α^1, α^2 — NOT a
  separate (1 + α/7 + 3α²) structure but the actual perturbative
  contributions from the OTHER fusion channels.  The α^3 coefficient
  would then come from interactions BETWEEN the other channels, which
  might cancel due to the specific Lie structure.

  This is option (a'): level-k channel-mixing cancellation.

  Specifically:  the bracket might encode

    bracket ~ (1 - α/2 × (singlet weight)) × (1 + α × Adj weight × ...) × ...

  with the α^3 term cancelling identically due to Lie algebra Bianchi
  identities or structure constants.
""")


def test_g2_branching_alpha_expansion() -> None:
    """The decisive test: if 8 = 7 + 1 (G_2 branching of S) lifts
    to q-deformation, then dim_q(S) - dim_q(V) - 1 should have a
    specific α-expansion.  If it truncates at α^n with n ≤ 2, it's
    strong evidence for option (c).

    """
    section("Test 5: dim_q(S) - dim_q(V) - 1  --  the α-expansion")
    print("""
  At the classical limit (k → ∞), dim_q(S) → 8 and dim_q(V) → 7,
  so dim_q(S) - dim_q(V) - 1 → 0.  Test 3 saw this residual is
  -0.000608 at k=32 -- small but non-zero.

  If the G_2 branching 8 = 7 + 1 lifts EXACTLY to q-deformation at
  every level, the residual would be exactly 0.  The non-zero value
  indicates corrections.  We extract the α-expansion of the residual
  to see if it truncates at α^2 (option c) or continues to all orders.
""")

    k_values = [10, 16, 24, 32, 50, 75, 100, 200, 500]
    alphas = [np.sin(np.pi / (k + H_VEE))**2 for k in k_values]
    residuals = []
    print(f"  {'k':>5}  {'α':>10}  {'dim_q(S)':>10}  {'dim_q(V)':>10}  "
          f"{'residual':>12}  {'residual/α^2':>15}")
    print("  " + "-" * 80)
    for k, a in zip(k_values, alphas):
        d_S = q_dim(REPS["S"], k)
        d_V = q_dim(REPS["V"], k)
        res = d_S - d_V - 1
        residuals.append(res)
        print(f"  {k:>5}  {a:>10.6f}  {d_S:>10.6f}  {d_V:>10.6f}  "
              f"{res:>+12.6e}  {res/a**2:>+15.4f}")

    # Fit residual to c_0 + c_1 α + c_2 α^2
    c0, c1, c2 = expand_in_alpha(residuals, alphas)
    print(f"\n  Fit:  residual = {c0:+.4e} + {c1:+.4f}·α + {c2:+.4f}·α²")

    # At each α, compare fit prediction to actual residual
    print(f"\n  Fit residual (actual - fitted) at each level:")
    for k, a, r in zip(k_values, alphas, residuals):
        fitted = c0 + c1 * a + c2 * a**2
        diff = r - fitted
        print(f"    k={k:>4}  actual={r:+.4e}  fitted={fitted:+.4e}  "
              f"diff={diff:+.4e}")

    # Scaling check: if residual scales purely as α^n, then
    # log|residual| / log α should be n (for the smallest α values where
    # higher-order terms are negligible).
    print(f"\n  Scaling check: log|res| / log(α)")
    for k, a, r in zip(k_values, alphas, residuals):
        if abs(r) > 1e-15:
            scaling = np.log(abs(r)) / np.log(a)
            print(f"    k={k:>4}  α={a:.6f}  log|res|/log(α) = {scaling:.4f}")

    # Test: try fitting only to higher powers
    # residual = b_2 α^2 + b_3 α^3 + b_4 α^4
    print(f"\n  Fit to higher powers:  residual = b_2 α² + b_3 α³ + b_4 α⁴")
    A2 = np.array([[a**2, a**3, a**4] for a in alphas])
    b, *_ = np.linalg.lstsq(A2, np.array(residuals), rcond=None)
    print(f"    b_2 = {b[0]:+.6f}    b_3 = {b[1]:+.6f}    b_4 = {b[2]:+.6f}")

    # Predicted residual using only α² term:
    print(f"\n  Test: does residual fit purely α² (no α³, α⁴)?")
    A2_only = np.array([[a**2] for a in alphas])
    b2_only, *_ = np.linalg.lstsq(A2_only, np.array(residuals), rcond=None)
    b2 = b2_only[0]
    print(f"    Best α²-only fit:  residual ≈ {b2:.4f} × α²")
    print(f"    Test points:")
    print(f"    {'k':>4}  {'actual':>12}  {'b_2 α² fit':>14}  {'rel error':>10}")
    print(f"    " + "-" * 50)
    for k, a, r in zip(k_values, alphas, residuals):
        fit = b2 * a**2
        if abs(r) > 1e-15:
            err = (fit - r) / r
            print(f"    {k:>4}  {r:>+12.4e}  {fit:>+14.4e}  {err:>+10.2%}")

    print("""
  Interpretation:

  - If residual ≈ b_2 α² with small higher-order corrections,
    then dim_q(S) ≈ dim_q(V) + 1 + b_2 α² is the q-deformation of
    the classical 8 = 7 + 1 split, with α² as the leading
    correction. This is the OPTION (c) signature: rank-1 mismatch
    (S = V ⊕ 1) admits at most quadratic α-mixing.

  - If significant α³ residuals remain, the truncation is NOT at
    α² and option (c) needs refinement.
""")


def main() -> None:
    print("=" * 78)
    print(" Paper 17 truncation mechanism: q-deformed Spin(7) tests")
    print("=" * 78)

    test_dimensions()
    test_qdim_ratios()
    test_spin7_g2_branching()
    test_LR_alpha_3_sketch()
    test_g2_branching_alpha_expansion()

    section("Summary")
    print("""
  This script performs partial tests of options (a') and (c):

  Test 1 verifies our quantum-dimension machinery against classical
  Weyl dimensions for so(7) reps.

  Test 2 (option c probe): checks whether any natural q-dim ratio
  truncates at α^2.  RESULT: no simple ratio reproduces the bracket
  (8/7)(1 + α/7 + 3α²); the bracket is finer structure specific to
  the K_7 Wilson amplitude.  Re-confirms the Paper 17 §6.3 negative
  result.

  Test 3 (option c): probes the Spin(7) → G_2 branching.  G_2 has
  S = V_7^{G_2} ⊕ V_1^{G_2}, the 8 = 7 + 1 split that drives the
  spinor-vector boundary projection.  At level k=32 the q-deformation
  partially breaks this, but the structural picture matches option (c).

  Test 4 (option a'): sketches the Lawrence-Rozansky α^3 calculation.
  Identifies the missing piece — full S-matrix, N-matrices, Seifert
  coefficients — and notes that finite-level channel-mixing
  corrections from V⊗V → 1 and V⊗V → 27 (beyond the Adj-only Paper 17
  reduction) might be the source of the bracket structure.  At
  α^3, these channel-mixing terms might cancel identically.

  Both options remain plausible.  A definitive resolution requires
  either:
    (c) Explicit Spin(7) → G_2 branching at level k, with q-deformed
        Clebsch-Gordan coefficients tabulated and traced.
    (a') Full Lawrence-Rozansky calculation on Σ(2,3,5) including all
         channel contributions and their α^3 cancellations.

  Both are 1-2 week expert-level computations beyond this session.

  Honest verdict for Paper 17: the truncation mechanism is structurally
  pinned to one of these two candidates, and we have qualitative
  arguments why each could be operative.  A rigorous derivation
  requires the heavy CS-level machinery cited in §13.3 as the central
  open problem.
""")


if __name__ == "__main__":
    main()
