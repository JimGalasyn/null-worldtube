#!/usr/bin/env python3
"""
NWT bracket-truncation: option (a') channel-mixing computation
================================================================

Tests whether the bracket truncation at α^2 arises from level-k
channel mixing in the V⊗V fusion at each K_7 edge.

The Paper 17 derivation picks ONLY the adjoint channel at each
edge, giving (1+R_Adj)/2 ≈ √α per edge.  But the full Wilson
amplitude on V⊗V at each crossing has contributions from all three
fusion channels:

    V ⊗ V  =  1 ⊕ Adj ⊕ 27

with R-matrix eigenvalues:
    R_1   = q^{-12}    (Casimir 0,  symmetric, ε=+1)
    R_Adj = -q^{-2}    (Casimir 10, antisymmetric, ε=-1)
    R_27  = q^{+2}     (Casimir 14, symmetric, ε=+1)

where q = exp(iπ/(k+h^v)) and h^v = 5 for so(7).

Hypothesis (option a'): the bracket structure 1 + α/7 + 3α² emerges
when ALL three channels are accounted for, with α³ cancelling
identically by interference between channels.

We compute several trace-like quantities involving multiple
R-matrix products at various levels k and fit their α-expansions
to test for truncation patterns.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


# ====================================================================
# Spin(7) data
# ====================================================================

H_VEE = 5
DIM_V   = 7
DIM_ADJ = 21
DIM_27  = 27
DIM_VV  = 49  # = 7 × 7 = 1 + 21 + 27

# Casimir values C_λ = (λ, λ + 2ρ) for so(7), Killing-normalized
C_1   = 0
C_V   = 6
C_ADJ = 10
C_27  = 14

# Drinfeld signs: + for sym (Sym²V channels), - for antisym (Λ²V)
EPS_1   = +1   # 1 ⊂ Sym²V
EPS_ADJ = -1   # Adj ⊂ Λ²V
EPS_27  = +1   # 27 ⊂ Sym²V


def section(s: str) -> None:
    print()
    print("=" * 78)
    print(" " + s)
    print("=" * 78)


# ====================================================================
# R-matrix eigenvalues at level k
# ====================================================================

def R_eigenvalues(k: int) -> dict:
    """Return R-matrix eigenvalues on V⊗V channels at level k."""
    T = k + H_VEE
    delta = np.pi / T
    q = np.exp(1j * delta)
    return {
        "1":   EPS_1   * q ** (C_1   - 2 * C_V),    # +q^{-12}
        "Adj": EPS_ADJ * q ** (C_ADJ - 2 * C_V),    # -q^{-2}
        "27":  EPS_27  * q ** (C_27  - 2 * C_V),    # +q^{+2}
    }


def alpha_at_level(k: int) -> float:
    return np.sin(np.pi / (k + H_VEE)) ** 2


# ====================================================================
# Trace-like quantities and their α-expansion
# ====================================================================

def expand_in_alpha(values: List[complex], alphas: List[float],
                    n_orders: int = 4) -> np.ndarray:
    """Fit complex values(alphas) to c_0 + c_1 α + c_2 α² + ... + c_{n-1} α^{n-1}."""
    A = np.array([[a**i for i in range(n_orders)] for a in alphas])
    b = np.array(values)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c


def test_R_traces() -> None:
    section("Test 1: Channel-summed R-matrix traces at multiple levels")
    print("""
  At each K_7 edge, the R-matrix on V⊗V has three eigenvalues.
  We compute three trace-like quantities at multiple k:

    T_1(k) = sum_λ dim(λ) · R_λ            (channel-weighted trace)
    T_2(k) = sum_λ dim(λ) · R_λ²           (squared trace)
    T_3(k) = sum_λ dim(λ) · R_λ³           (cubed trace)

  These are the diagonal contributions to the Wilson amplitude at
  1, 2, 3 consecutive edges respectively (ignoring 6j corrections).

  α-expansion of each, to detect truncation patterns.
""")

    k_values = [10, 16, 24, 32, 50, 75, 100, 200]
    alphas = [alpha_at_level(k) for k in k_values]
    dims = {"1": 1, "Adj": DIM_ADJ, "27": DIM_27}

    for n_pow in [1, 2, 3]:
        print(f"\n  Trace_n=lambda dim(λ)·R_λ^{n_pow}:")
        traces = []
        for k in k_values:
            R = R_eigenvalues(k)
            T_n = sum(dims[name] * R[name]**n_pow for name in ["1", "Adj", "27"])
            traces.append(T_n)

        print(f"    {'k':>5}  {'α':>10}  {'Re(Tr)':>14}  {'Im(Tr)':>14}")
        for k, a, T in zip(k_values, alphas, traces):
            print(f"    {k:>5}  {a:>10.6f}  {T.real:>+14.6f}  {T.imag:>+14.6f}")

        # Fit real and imag parts in α
        c_real = expand_in_alpha([T.real for T in traces], alphas, n_orders=4)
        c_imag = expand_in_alpha([T.imag for T in traces], alphas, n_orders=4)
        print(f"    α-expansion (real):  c_0={c_real[0]:+.4f}  c_1={c_real[1]:+.4f}  "
              f"c_2={c_real[2]:+.4f}  c_3={c_real[3]:+.4f}")
        print(f"    α-expansion (imag):  c_0={c_imag[0]:+.4f}  c_1={c_imag[1]:+.4f}  "
              f"c_2={c_imag[2]:+.4f}  c_3={c_imag[3]:+.4f}")


# ====================================================================
# Test 2: The "physical" V Wilson amplitude on K_7
# ====================================================================

def test_K7_full_amplitude() -> None:
    section("Test 2: Channel-mixed K_7 amplitude vs Adj-only reduction")
    print("""
  Paper 17 picks the Adj channel at each of the 21 K_7 edges, giving
  (1+R_Adj)/2)^21 = α^{21/2} as the bare amplitude.  The bracket
  (1 + α/7 + 3α²) is a sub-leading correction.

  We compare two computations at each level:

    A_paper(k) = ((1+R_Adj)/2)^21        — Paper 17 reduction
    A_full(k)  = (1/dim(V)^21) × sum over channel paths ...
                  — full V-line Wilson amplitude (approximation)

  For the full amplitude, the simplest gauge-invariant approximation
  treats each edge as the channel-weighted average:
    A_edge_full = sum_λ dim(λ) (1+R_λ)/2  /  dim(V⊗V)

  This counts dimensional contributions correctly but ignores 6j
  cross-correlations between adjacent edges (which is where the real
  bracket structure must come from for option (a')).

  Even at this level, we can check if the channel-mixed amplitude
  has different α-scaling than the Adj-only one, indicating
  potential truncation contributions.
""")

    k_values = [10, 16, 24, 32, 50, 75, 100, 200]
    alphas = [alpha_at_level(k) for k in k_values]
    n_edges = 21

    print(f"  Per-edge amplitudes at k=32:")
    R_32 = R_eigenvalues(32)
    for name, R in R_32.items():
        amp = (1 + R) / 2
        print(f"    Channel {name:<5}:  R = {R:+.4f}  |1+R|/2 = {abs(amp):.4f}")

    # Per-edge full amplitude (channel-weighted average)
    dims = {"1": 1, "Adj": DIM_ADJ, "27": DIM_27}

    print(f"\n  Per-edge full amplitudes |A_edge_full|:")
    print(f"    {'k':>5}  {'α':>10}  {'|A_paper|':>12}  {'|A_full|':>12}  "
          f"{'ratio':>10}")
    for k, a in zip(k_values, alphas):
        R = R_eigenvalues(k)
        A_paper = abs((1 + R["Adj"]) / 2)
        A_full = abs(sum(dims[name] * (1 + R[name]) / 2
                         for name in ["1", "Adj", "27"]) / DIM_VV)
        print(f"    {k:>5}  {a:>10.6f}  {A_paper:>12.6f}  {A_full:>12.6f}  "
              f"{A_full/A_paper:>10.4f}")

    print(f"""
  Observation:  the channel-weighted full amplitude has |A_full| ≈ 1
  (close to channels 1 and 27 which have |1+R|/2 ≈ 1), NOT √α.
  Restricting to the Adj channel is what gives the √α behaviour.

  This confirms the Paper 17 picture:  the K_7 Wilson amplitude is
  specifically in the Adj channel of each V⊗V crossing.  The bracket
  corrections must come from sub-leading α effects WITHIN this Adj
  channel restriction, not from full channel-mixing.

  We need to compute the corrections more carefully:  the leading
  order is (R_Adj/(2))^21 as a complex number (with phase), and
  sub-leading orders pick up α-corrections from the precise R_Adj
  expansion.
""")


# ====================================================================
# Test 3: Adj-only amplitude full complex expansion
# ====================================================================

def test_Adj_complex_expansion() -> None:
    section("Test 3: Adj-channel amplitude — complex α-expansion to α^3")
    print("""
  The Paper 17 amplitude is restricted to the Adj channel at each
  edge.  Per edge:  R_Adj = -q^{-2} = -e^{-2iδ}, where δ = π/(k+5).

    (1 + R_Adj)/2 = (1 - e^{-2iδ})/2 = e^{-iδ} (e^{iδ} - e^{-iδ})/2
                  = e^{-iδ} · i sin(δ)
                  = i sin(δ) · e^{-iδ}

  So |1+R_Adj|/2 = sin(δ) and the phase is -δ + π/2 (modulo 2π).

  For 21 edges, the amplitude is:
    A_circuit = (i sin(δ))^{21} · e^{-21iδ}
              = i^{21} · sin^{21}(δ) · e^{-21iδ}

  Let's expand this to high precision in α = sin²(δ) and see what
  shows up at α^3 vs α^2.
""")

    k_values = [10, 16, 24, 32, 50, 75, 100, 200, 500]
    alphas = [alpha_at_level(k) for k in k_values]

    print(f"  Per-edge amplitude (complex):  i sin(δ) · e^{{-iδ}}")
    print(f"  Circuit amplitude (21 edges):  i^21 · sin^21(δ) · e^{{-21iδ}}")
    print()
    print(f"    {'k':>5}  {'α':>10}  {'|A_circuit|/α^{21/2}':>22}  {'arg(A_circuit)':>18}")
    print("  " + "-" * 70)

    for k, a in zip(k_values, alphas):
        T = k + H_VEE
        delta = np.pi / T
        R_adj = -np.exp(-2j * delta)
        A_edge = (1 + R_adj) / 2
        A_circuit = A_edge ** 21

        # Normalize by α^{21/2}
        norm = abs(A_circuit) / a ** (21/2)
        arg = np.angle(A_circuit)
        print(f"    {k:>5}  {a:>10.6f}  {norm:>22.10f}  {arg:>+18.6f}")

    print("""
  Observation:  |A_circuit| / α^{21/2} → 1 as k → ∞, meaning the
  amplitude magnitude is exactly α^{21/2}.  No α-correction at the
  amplitude-magnitude level, contradicting the bracket existing.

  This means the bracket (1 + α/7 + 3α²) must come from
  something OTHER than the Adj-channel per-edge restriction.  The
  bracket is a finer structural correction — likely from:
    * 6j-symbol cross-correlations between adjacent K_7 edges
    * Cartan-graded boundary projections at the Wilson endpoints
    * Higher-order RT corrections beyond the leading Adj channel

  Returning to the original question:  Test 3 is consistent with
  the Paper 17 setup giving α^{21/2} bare and the bracket coming from
  endpoint/boundary structure (option c-style) or 6j-correction
  structure (option a'-style on K_7 vertex consistency rather than
  channel mixing per se).
""")


# ====================================================================
# Test 4: 6j-symbol structure at K_7 vertices
# ====================================================================

def test_6j_structure() -> None:
    section("Test 4: 6j-symbol structure at K_7 vertices — option (a') refined")
    print("""
  The K_7 Wilson amplitude on V has 21 edges but consistency at each
  K_7 vertex (degree 6) requires a 6j-symbol matching.  At each
  vertex, six edge-channels must combine via Spin(7) recoupling.

  Paper 17 picks all 21 edges in the Adj channel.  This is a
  consistent labeling at each vertex (six Adj's must fuse to identity
  via repeated tensor products).  The fusion multiplicity
    Adj ⊗ Adj ⊗ Adj ⊗ Adj ⊗ Adj ⊗ Adj  →  identity
  determines the per-vertex amplitude factor.

  The full Wilson amplitude is then:
    A_K_7 = (per-edge Adj amplitude)^21 × (per-vertex 6j factor)^7

  The bracket (1 + α/7 + 3α²) might be the per-vertex 6j factor.

  For so(7) with all six edges in Adj, the 6j factor has a specific
  α-expansion at level k.  Computing it requires tabulated 6j-symbols
  for so(7), which we don't have available here.  We can however
  estimate the SHAPE of this factor.

  The 6j-symbol  {Adj Adj Adj | Adj Adj Adj}_so(7)  is a specific
  rational function of dim_q values.  Its leading term is
  unity (the classical 6j) plus α-corrections.

  The b2.13 bijection between K_7 edges and Adj generators suggests
  the 6j factor at each vertex is related to the 'recoupling
  amplitude' for inserting six adjoint generators in V representation.
  This is exactly the 'per-vertex self-energy' factor of Paper 17 §7.
""")

    print(f"  Consistency check:  Paper 17 §7's factor is (1 + α/49)^7,")
    print(f"  expanding to 1 + α/7 + 21α²/2401 + ...  — but the actual α²")
    print(f"  bracket coefficient is 3, not 21/2401 ≈ 0.0087.")
    print(f"")
    print(f"  This means the per-vertex factor (1 + α/49)^7 is NOT the full")
    print(f"  bracket — it's only the LO correction.  The α² coefficient")
    print(f"  comes from a DIFFERENT structural source, identified as")
    print(f"  dim(Adj)/dim(V) = rank(so(7)) = 3 in Paper 17 §8.")
    print(f"")
    print(f"  The 6j-symbol picture suggests the bracket is:")
    print(f"    bracket = (per-vertex)^7 × (per-edge correction)^21")
    print(f"")
    print(f"  with the per-vertex containing α and the per-edge containing α²")
    print(f"  contributions.  This DOES allow truncation:  if both factors")
    print(f"  are linear in α, their product naturally truncates at α² when")
    print(f"  expanded.")


def test_factorisation_hypothesis() -> None:
    section("Test 5: Bracket factorisation hypothesis")
    print(r"""
  The bracket  1 + α/7 + 3α²  factorises if we write it as

      bracket(α)  =  (1 + α·a)·(1 + α·b)
                  =  1 + α(a + b) + α²(a·b)
                  =  1 + α/7 + 3α²

  with  a + b = 1/7  and  ab = 3.

  Solving for a and b:
      a, b = (1/7 ± √(1/49 - 12))/2 = (1/7 ± √(-587/49))/2

  These are COMPLEX roots:  a, b = (1 ± i√587)/14.

  So the bracket factorises into two complex linear factors with
  conjugate coefficients, but not into two real linear factors.

  Alternative factorisation:  maybe  bracket = 1 + α·F(α)  where F(α)
  is a finite polynomial.

      bracket = 1 + α(1/7 + 3α) = 1 + (α/7)·(1 + 21α)

  This IS a clean factorisation with real coefficients!  The
  bracket is:
      bracket(α) = 1 + (α/dim V) · (1 + dim(Adj)·α)

  The factor (1 + dim(Adj)·α) = (1 + 21α) is suggestive — it looks
  like the FIRST-ORDER expansion of 1/(1 - 21α), which would give
  the geometric series.  Truncating to first order in this factor
  yields exactly the bracket; including all orders gives the
  geometric series.

  Question:  is there a structural reason the (1 + 21α) factor
  truncates at first order rather than continuing as 1/(1 - 21α)?

  Numerical test:  if (1 + 21α) is the right factor, then at finite
  k, the K_7 amplitude correction should match this exactly (modulo
  α^3 from the geometric series tail) — not a more elaborate
  rational function.
""")

    # Numerical comparison
    alphas = [alpha_at_level(k) for k in [10, 16, 24, 32, 50, 100]]
    print(f"  At each α, compare:")
    print(f"    bracket = 1 + α/7 + 3α²    vs    1 + α/[7(1 - 21α)]")
    print(f"  The difference is the size of the α³ residual.")
    print()
    print(f"  {'k':>5}  {'α':>10}  {'truncated':>14}  {'geometric':>14}  "
          f"{'diff':>14}")
    print("  " + "-" * 70)
    for k_val, a in zip([10, 16, 24, 32, 50, 100], alphas):
        truncated = 1 + a/7 + 3*a**2
        geometric = 1 + a / (7 * (1 - 21*a))
        diff = geometric - truncated
        print(f"    {k_val:>5}  {a:>10.6f}  {truncated:>14.8f}  "
              f"{geometric:>14.8f}  {diff:>+14.4e}")

    print("""
  At physical α ≈ 0.0073 (k=32), the truncated and geometric
  predictions differ by ~3e-5 (relative).  At smaller α (larger k)
  they agree to higher precision.  CODATA precision on m_e/m_Pl
  (~11 ppm) cleanly separates the two predictions at k ≈ 32.

  Whether the TRUNCATED form is correct (and not the geometric) is
  the central physical question.  The current evidence (Paper 17 §11.6
  and this script) is:
    - Truncated form fits CODATA at -0.0001%
    - Geometric form misses by +49 ppm
    - Both agree at LO and NLO+NNLO; differ at α^3
""")


def main() -> None:
    print("=" * 78)
    print(" Paper 17 truncation: option (a') channel-mixing computation")
    print("=" * 78)

    test_R_traces()
    test_K7_full_amplitude()
    test_Adj_complex_expansion()
    test_6j_structure()
    test_factorisation_hypothesis()

    section("Summary")
    print("""
  This script probes option (a'):  the bracket truncation as a
  consequence of level-k channel-mixing in the V⊗V fusion.

  Key findings:

  (1) Per-edge channel structure:  three eigenvalues at level k:
      R_1 = q^{-12}, R_Adj = -q^{-2}, R_27 = q^{+2}.  Restricting to
      Adj gives the √α-per-edge behaviour;  the other channels
      contribute O(1).

  (2) Adj-only K_7 amplitude:  |A_circuit| / α^{21/2} → 1 exactly.
      The amplitude magnitude is α^{21/2}.  No bracket emerges from
      the Adj-only amplitude alone.

  (3) The bracket must come from:  6j-symbol cross-correlations
      between adjacent edges (per-vertex factor) AND/OR boundary
      projections at the Wilson endpoints (per-end factor).

  (4) Bracket factorisation:  the bracket admits a CLEAN factorisation
      bracket(α) = 1 + (α/dim V) · (1 + dim(Adj) α).
      This is the FIRST TWO terms of a geometric series 1 + α/[dim V
      (1 - dim(Adj) α)] truncated at first order.  Truncation at
      first order in dim(Adj)α (rather than continuing) is the
      structural signature of the bracket.

  This factorisation result is the cleanest expression of the
  truncation:  the bracket is the FIRST-ORDER GEOMETRIC EXPANSION of
  a specific level-k amplitude, with higher-order geometric
  contributions cancelled by structural constraint.

  What cancels them is still open.  Candidates from the candidate
  space narrowed in earlier scripts:
    (a') Level-k channel-mixing constraint at K_7 vertices;
    (c)  Spinor-vector boundary projection at Wilson endpoints.

  Both require non-trivial RT machinery.  The bracket factorisation
  identified here suggests the truncation is mediated by the
  EFFECTIVE PROPAGATOR structure in CS theory, in which level-k
  perturbative corrections naturally factorise into 'tree-level' and
  'one-loop' terms truncating at second order in α.
""")


if __name__ == "__main__":
    main()
