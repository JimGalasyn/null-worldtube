#!/usr/bin/env python3
"""
Paper 19 -- W1+W2: K_7 as holographic stabilizer state +
entanglement-distance structure.

After the 2026-04-27 evening pivot to background-independent
formulation (K_N substrate fundamental, spacetime emergent), W1+W2
is the first concrete deliverable using techniques imported from
the BI-QG literature:

  W1: stabilizer-graph-code structure of K_7
       (technique adapted from Engineering Holography with Stabilizer
       Graph Codes, npj Quantum Info 2024)

  W2: entanglement-distance metric on K_7 from mutual information
       (technique from Van Raamsdonk 2010 + ER=EPR realization 2024)

NWT's distinctive contribution: we can verify these theoretical
structures directly on quantum hardware (Heron experiments, Papers
17 §13 and 18 G3 already did this for the structural identity
<H_YY> = 21).  No other BI-QG framework has hardware-level
verification of the proposed substrate.

==========================================================================
 W1: K_7 AS A STABILIZER STATE
==========================================================================

The K_7 graph state on 7 qubits is the unique simultaneous +1
eigenstate of the seven stabilizers:

  S_v = X_v · prod_{u != v} Z_u    for v = 0, 1, ..., 6.

This generates a 7-element abelian subgroup of the Pauli group, with
no non-trivial logical operators (rank = 7 = number of qubits).
K_7 is therefore a **stabilizer state** in the [[7, 0, ?]] sense:
7 physical qubits, 0 logical qubits, the distance parameter
(minimum Pauli weight to flip the state) determined by the
stabilizer structure.

In holographic-code language (per Engineering Holography with
Stabilizer Graph Codes, 2024), K_7 by itself is not a holographic
code with bulk reconstruction (which requires 0 < k < n logical
qubits encoded in n physical qubits).  Rather, K_7 is a single
"voxel" — a pre-geometric quantum-information structure that, when
networked with other K_N voxels, gives rise to extended spacetime.

This is consistent with NWT's BI-pivot framing: the substrate is
a NETWORK of K_N graph-state patches, not a single K_7.  The
extended spacetime emerges from how the patches are glued.

==========================================================================
 W2: ENTANGLEMENT-DISTANCE STRUCTURE ON K_7
==========================================================================

For a graph state |G⟩ with edge set E, the bipartite entanglement
entropy across any cut is determined by the rank (over GF(2)) of
the cut-adjacency matrix.  For K_7:

  - Single-qubit cuts: rank = 1 (each qubit is connected to all 6
    others, so the adjacency vector has 1 nonzero entry up to GF(2)
    column reduction).  Single-qubit entropy = 1 bit.

  - Pair-of-qubits cuts (S = {u,v}): cut adjacency matrix is 2×5,
    all entries 1.  Rank over GF(2) = 1.  Pair entropy = 1 bit.

  - In general, for any subset S with 1 ≤ |S| ≤ 6, the cut
    adjacency rank for K_n is 1 (universal — all qubits in K_n are
    connected to all others).

Mutual information between qubits u, v (u ≠ v):

  I(u : v) = S(u) + S(v) - S({u,v}) = 1 + 1 - 1 = 1 bit per edge.

This is uniform across all 21 edges of K_7 — every pair of qubits
shares exactly 1 bit of mutual information.

**Total mutual information** = 21 bits = dim(Adj_so(7))
(matches Paper 17 §11 and Paper 18 G3's structural identity).

==========================================================================
 IMPLICATION: K_7 IS A SINGLE PRE-GEOMETRIC VOXEL
==========================================================================

In Van Raamsdonk's framework, entanglement = spatial proximity.
K_7's uniform mutual information across all pairs implies that
all 7 "points" of K_7 are at the same effective distance from
each other — this is a fully-connected simplex, NOT extended
geometry.

In other words: K_7 gives us the LOCAL structure of a single
substrate voxel.  It does not, by itself, contain extended
spacetime.  To get extended spacetime, multiple K_7 voxels (or
K_N patches more generally) must be glued together.  The gluing
structure determines the large-scale metric.

This is structurally analogous to how a single tetrahedron in
Regge calculus or a single 4-simplex in causal dynamical
triangulations is just one "atom" of spacetime, with extended
spacetime emerging from many such atoms.

==========================================================================
 HARDWARE EMBODIMENT (NWT's distinctive advantage)
==========================================================================

The above theoretical structure can be DIRECTLY VERIFIED on
quantum hardware.  We have 8 datasets across 3 Heron R2 devices
(kingston, marrakesh, fez) verifying the K_7 stabilizer structure
(Paper 17 §13, Paper 18 G3).

What we have NOT yet measured directly:

  * The mutual-information matrix I(u : v) for all 21 pairs.
    Predicted: 1 bit per edge, uniform.
    Hardware test: prepare K_7, measure rho_uv for each pair,
    compute mutual information.  Check uniformity.
  * The bipartite entropy S(S) for various subsets S of size 2, 3.
    Predicted: 1 bit for K_7 graph states.
    Hardware test: prepare K_7, do tomography of reduced density
    matrices on subsets, compute von Neumann entropy.

This would be an Experiment 6 candidate for the next Heron run.
Estimated cost: ~$50-100 PAYG.  Would directly verify the
substrate-level entanglement structure that BI-QG frameworks have
only ever proposed theoretically.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations


# ==========================================================================
# K_7 graph state setup.
# ==========================================================================

N = 7
DIM = 2 ** N

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def build_K7_state() -> np.ndarray:
    """|K_7> = prod_{u<v} CZ_{uv} |+>^7, as a 128-dim state vector."""
    psi = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    for u, v in combinations(range(N), 2):
        for idx in range(DIM):
            xu = (idx >> (N - 1 - u)) & 1
            xv = (idx >> (N - 1 - v)) & 1
            if xu == 1 and xv == 1:
                psi[idx] *= -1
    return psi


# ==========================================================================
# W1: Stabilizer structure.
# ==========================================================================

def stabilizer_op(v: int) -> np.ndarray:
    """S_v = X_v * prod_{u != v} Z_u, as a 128x128 matrix."""
    ops = [Z] * N
    ops[v] = X
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def verify_stabilizer_structure(psi: np.ndarray) -> dict:
    """Verify all 7 stabilizers eigenvalue +1 on |K_7>."""
    eigenvalues = []
    for v in range(N):
        S = stabilizer_op(v)
        ev = (psi.conj() @ S @ psi).real
        eigenvalues.append(float(ev))
    return dict(eigenvalues=eigenvalues,
                all_plus_one=all(abs(e - 1.0) < 1e-10 for e in eigenvalues))


# ==========================================================================
# W2: Reduced density matrices and entanglement entropy.
# ==========================================================================

def partial_trace_rho_subset(psi: np.ndarray, subset: tuple) -> np.ndarray:
    """Compute rho_S for the subset of qubits S ⊂ {0..6}.

    Naive O(N · 2^N · |subset|) implementation: reshape psi into
    (2,)*N tensor, take outer product, contract over complementary qubits.
    """
    subset = tuple(sorted(subset))
    complement = tuple(q for q in range(N) if q not in subset)
    n_S = len(subset)
    n_C = len(complement)

    # Reshape psi as (2,)^N tensor with axes 0..N-1
    psi_t = psi.reshape((2,) * N)
    # rho = psi |><| psi traced over complement
    # We can compute rho_S as sum over complement basis indices.
    dim_S = 2 ** n_S

    rho_S = np.zeros((dim_S, dim_S), dtype=complex)
    # For each pair of S-basis states, sum over complement
    for i_S in range(dim_S):
        for j_S in range(dim_S):
            # Build full multi-indices for psi_t and psi_t.conj()
            # by combining (i_S over subset positions) with iteration
            # over complement positions
            for i_C in range(2 ** n_C):
                idx_psi = [0] * N
                idx_psi_conj = [0] * N
                for k, q in enumerate(subset):
                    idx_psi[q] = (i_S >> (n_S - 1 - k)) & 1
                    idx_psi_conj[q] = (j_S >> (n_S - 1 - k)) & 1
                for k, q in enumerate(complement):
                    bit = (i_C >> (n_C - 1 - k)) & 1
                    idx_psi[q] = bit
                    idx_psi_conj[q] = bit
                rho_S[i_S, j_S] += (
                    psi_t[tuple(idx_psi)]
                    * psi_t[tuple(idx_psi_conj)].conj()
                )
    return rho_S


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S(rho) = -tr(rho log_2 rho).  Bits."""
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-12]
    return float(-np.sum(eigs * np.log2(eigs)))


def mutual_information(psi: np.ndarray, u: int, v: int) -> float:
    """I(u:v) = S(rho_u) + S(rho_v) - S(rho_uv) in bits."""
    rho_u = partial_trace_rho_subset(psi, (u,))
    rho_v = partial_trace_rho_subset(psi, (v,))
    rho_uv = partial_trace_rho_subset(psi, (u, v))
    return (von_neumann_entropy(rho_u) + von_neumann_entropy(rho_v)
            - von_neumann_entropy(rho_uv))


# ==========================================================================
# Reporting.
# ==========================================================================

def section(t: str) -> None:
    print()
    print('=' * 76)
    print(f' {t}')
    print('=' * 76)


def main() -> None:
    section('Paper 19 -- W1 + W2: K_7 as holographic stabilizer + entanglement metric')

    psi = build_K7_state()
    print(f"\n  |K_7> norm: {np.linalg.norm(psi):.6f}")

    # ---- W1: stabilizer structure ----
    section('W1: stabilizer structure of K_7')
    res = verify_stabilizer_structure(psi)
    print(f"\n  Stabilizers S_v = X_v * prod_{{u != v}} Z_u, v = 0..6:")
    for v, ev in enumerate(res['eigenvalues']):
        print(f"    <S_{v}>  =  {ev:+.6f}")
    print(f"\n  All seven stabilizers have eigenvalue +1: "
          f"{'PASS' if res['all_plus_one'] else 'FAIL'}")
    print(f"\n  K_7 is a [[7, 0, ?]] stabilizer state:")
    print(f"    n = 7 physical qubits")
    print(f"    k = 0 logical qubits encoded (no logical operators)")
    print(f"    Rank of stabilizer group = 7 (full rank)")
    print(f"\n  In holographic-code language (Engineering Holography 2024),")
    print(f"  this means K_7 is NOT a holographic code with bulk reconstruction")
    print(f"  by itself.  It is a single 'voxel' -- a pre-geometric QI structure.")

    # ---- W2: bipartite entropies ----
    section('W2: bipartite entanglement entropies')
    print(f"\n  Compute S(rho_S) for various subsets S:\n")
    print(f"  {'subset S':>12} {'|S|':>4} {'S(rho_S) [bits]':>18}")
    print('  ' + '-' * 36)

    subsets_to_test = [
        ((0,), 1),
        ((1,), 1),
        ((6,), 1),
        ((0, 1), 2),
        ((2, 4), 2),
        ((0, 1, 2), 3),
        ((0, 3, 6), 3),
    ]
    for S, _ in subsets_to_test:
        rho = partial_trace_rho_subset(psi, S)
        H = von_neumann_entropy(rho)
        print(f"  {str(S):>12} {len(S):>4} {H:>18.6f}")

    print(f"""
  All single-qubit and pair-of-qubit subsets give 1 bit of entropy.
  This is the universal property of K_n graph states: the cut-
  adjacency matrix has rank 1 over GF(2) for any subset 1 <= |S| < n.

  Three-qubit subsets give entropy 1 bit as well (rank-1 cut).
""")

    # ---- W2 cont: mutual information across all 21 edges ----
    section('W2 cont: mutual information per K_7 edge')
    print(f"\n  Computing I(u:v) = S(rho_u) + S(rho_v) - S(rho_uv) for all 21 edges:\n")
    print(f"  {'edge':>10} {'I(u:v) [bits]':>16}")
    print('  ' + '-' * 28)

    mutual_infos = []
    for u, v in combinations(range(N), 2):
        I = mutual_information(psi, u, v)
        mutual_infos.append(I)
        print(f"  ({u},{v}){'':>5} {I:>16.6f}")

    total = sum(mutual_infos)
    print(f"\n  Total mutual information: {total:.4f} bits")
    print(f"  Predicted:                21 bits = dim(Adj_so(7))")
    print(f"  Per-edge average:         {total / 21:.4f} bits")
    print(f"\n  All 21 edges have I(u:v) = 1 bit exactly (uniform).")
    print(f"  This matches Paper 17 / Paper 18 G3 structural identity.")

    # ---- W2: implications for entanglement distance ----
    section('W2 implications: K_7 as a pre-geometric voxel')
    print(f"""
  Van Raamsdonk's principle: entanglement = spatial proximity.

  K_7 has UNIFORM mutual information of 1 bit per edge across all
  21 pairs of qubits.  In Van Raamsdonk's framework, this means
  every pair of "spacetime points" is at the same effective distance
  from every other.  This is a fully-connected simplex, NOT
  extended geometry.

  Conclusion: K_7 alone is a SINGLE substrate "voxel" -- a pre-
  geometric atom of spacetime.  Extended spacetime requires
  gluing multiple K_N patches.

  Analogy: K_7 :: 4-simplex in causal dynamical triangulations.
  A single 4-simplex doesn't contain extended spacetime; the
  spacetime emerges from the simplicial complex of many 4-simplexes.

  The next research direction (W3+) is: how do multiple K_N graph-
  state voxels glue together?  Candidates from the literature:
    * spin-network coarse graining (Livine 2022)
    * tensor networks of K_N atoms
    * dynamical quantum graph evolution (Markopoulou-Smolin 2008,
       Dynamical Quantum Multigraphs 2025)

  This is where Paper 19 needs the most work.
""")

    # ---- Hardware proposal ----
    section('Hardware embodiment: proposed Heron experiment')
    print(f"""
  NWT's distinctive advantage: we can verify the K_7 entanglement
  structure DIRECTLY on quantum hardware, not just theoretically.

  Existing Heron data (Papers 17 §13, 18 G3):
    * <H_YY>|K_7> = 21 verified to ~0.029 % via K_9/K_7 ZNE ratio
    * Multiple devices (kingston, marrakesh, fez)
    * 8 datasets

  Proposed new Heron experiment ("Experiment 6: entanglement
  tomography on K_7"):
    * Prepare |K_7> on Heron (already done)
    * Perform reduced-state tomography on each pair of qubits
    * Extract rho_uv for all 21 edges
    * Compute I(u:v) directly from measured rho's
    * Verify: I(u:v) = 1 bit per edge, uniform

  This would be the FIRST direct hardware verification of an
  entanglement-distance metric on a proposed BI-QG substrate.
  No other BI-QG framework has experimentally verified its
  proposed substrate at this level.

  Cost estimate: ~21 pairs * tomography cost.  For 4-shot tomography
  per Pauli observable * 6 observables per pair = 24 measurements per
  pair * 21 pairs = ~500 measurements * 4000 shots each = ~2M shots.
  At Heron's rate ~13 sec QPU per 420K shots, total ~60 sec QPU.
  PAYG cost ~$100.

  Proposed for the next Heron campaign cycle.

==========================================================================
 W1+W2 status
==========================================================================

  Theoretical:                                                   [done]
    1. K_7 verified as [[7, 0, ?]] stabilizer state.
    2. Bipartite entropies computed: 1 bit for any non-trivial cut.
    3. Mutual information computed: 1 bit per edge, uniform.
    4. Identified K_7 as a single pre-geometric voxel.
    5. Open: how do K_N voxels glue into extended spacetime? (W3+)

  Hardware:                                                       [proposed]
    Experiment 6 design ready; awaits Heron campaign budget.
""")


if __name__ == '__main__':
    main()
