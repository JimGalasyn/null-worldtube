#!/usr/bin/env python3
"""
BPS μ=π → Compton period derivation for the QEC time-evolution framework.

Builds on `nwt_qec_time_evolution.py`, which postulated:
   natural time unit (for |K_7⟩-rotation under H_YY) = Compton period T_C
to make ⟨H_YY⟩ = 21 phase-ticks per cycle reduce to standard rest-mass
phase rotation.

This script tests whether the BPS condition μ_BPS = π (Paper 13's
Bogomolny saturation) independently fixes the natural unit, converting
the postulate into a derivation.

The bridge is action quantization:

   S_loop / ℏ  =  2π · (winding number)

For a closed worldline traversing one K_7 Eulerian circuit, the
classical action is S = m_e c² · T_circuit (rest frame).  Setting
S/ℏ = 2π · n (action quantization for n-fold winding):

   m_e c² · T_circuit / ℏ  =  2π · n
   ⇒  T_circuit  =  n · (2π ℏ / m_e c²)
                 =  n · T_C

So if the Heawood-embedded K_7 Eulerian circuit winds ONCE around the
Heegaard torus (n=1), the natural unit is exactly the Compton period
and the postulate becomes a derivation.

Pieces:
  (1) BPS μ=π: fixes m_e c² = π · L_trefoil.
  (2) Action quantization: S = 2πn ℏ for n-fold winding.
  (3) Heawood winding: the canonical K_7 Eulerian circuit on the
      Heegaard torus has winding n on π_1(T²) = Z².  Verify n.

If (3) gives n=1, the chain (BPS → m_e) + (action quantization)
+ (Heawood winding=1) jointly imply T_circuit = T_C without postulate.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import List, Tuple

import numpy as np


# =====================================================================
# 1. Physical constants
# =====================================================================

H_BAR = 1.054571817e-34   # J·s
M_E   = 9.1093837015e-31  # kg
C     = 299792458.0       # m/s
M_PL  = 2.176434e-8       # kg

omega_C = M_E * C**2 / H_BAR
T_C     = 2 * np.pi / omega_C
omega_Pl = M_PL * C**2 / H_BAR
T_Pl     = 2 * np.pi / omega_Pl


def section(title: str) -> None:
    print()
    print('=' * 78)
    print(' ' + title)
    print('=' * 78)


# =====================================================================
# 2. Action quantization argument
# =====================================================================

def action_quantization() -> None:
    section('Action quantization on the K_7 Eulerian circuit')
    print(f"""
  Standard QM: a closed worldline of period T accumulates phase
                 Δφ = ∫ (m c²) dt / ℏ = ω_C · T.
  Action quantization (Bohr-Sommerfeld for periodic orbits):
                 S_loop / ℏ  =  2π · n   for integer winding n.
  ⇒  m_e c² · T_loop / ℏ  =  2π · n
  ⇒  T_loop  =  n · (2π ℏ / m_e c²)  =  n · T_C.

  For n = 1:  T_loop = T_C exactly.  The natural unit IS the Compton
  period, *contingent on n=1*.

  Numerically:
     m_e c² / ℏ  =  ω_C  =  {omega_C:.4e} rad/s
     T_C         =  {T_C:.4e} s
     For n = 1:  T_loop = {T_C:.4e} s = 1 Compton period.
     For n = 21: T_loop = {21*T_C:.4e} s = 21 Compton periods.

  The H_YY phase rotation (⟨H_YY⟩ = 21 per natural unit) requires
  natural unit = T_C / (21/2π) = (2π/21) T_C ≈ {2*np.pi/21*T_C:.4e} s
  for one full cycle to elapse per |K_7⟩-rotation period.

  Equivalently: H_phys = (2π/21) · H_YY rotates |K_7⟩ at rate 2π per
  natural unit; setting natural unit = T_C gives standard rest-mass
  phase.  The factor (2π/21) is the per-edge phase 2π/21 rad, with
  21 edges traversed per Eulerian circuit summing to 2π.
""")


# =====================================================================
# 3. BPS μ=π fixes m_e
# =====================================================================

def bps_constraint() -> None:
    section('BPS condition μ=π and the rest mass')
    print("""
  Paper 13's BPS analysis (Bogomolny saturation on the trefoil):

     μ_BPS  =  π · ℏc / ƛ_C²      (line tension at BPS)

  where ƛ_C = ℏ/(m_e c) is the reduced Compton wavelength.  The
  trefoil's rest energy at BPS:

     m_e c²  =  μ_BPS · L_trefoil

  Solving for the trefoil tube length:

     L_trefoil  =  m_e c² / μ_BPS
                =  m_e c² / (π ℏc / ƛ_C²)
                =  m_e c² · ƛ_C² / (π ℏc)
                =  ƛ_C / π                       (after substitution)

  So at BPS μ=π, the trefoil's effective tube length is ƛ_C/π —
  one Compton wavelength divided by π.  This is the geometric
  scale that the trefoil supplies.

  CRITICAL CHECK: does BPS independently fix the K_7 Eulerian
  circuit's winding number on the Heegaard torus to 1?

  The Heegaard torus T² has fundamental group π_1(T²) = ℤ × ℤ.
  Any closed loop on T² has a winding (p, q) ∈ ℤ × ℤ.  For the
  Heawood-embedded K_7 graph, the Eulerian circuit's winding (p, q)
  is fixed by:
    (a) the embedding of K_7 on T² (Heawood map: unique up to
        symmetry),
    (b) the choice of Eulerian circuit (Hierholzer; one canonical
        choice modulo PSL(2,7) action on K_7).

  We compute (p, q) numerically below.
""")


# =====================================================================
# 4. Heawood winding number numerical check
# =====================================================================
#
# The Heawood map of K_7 on T² has 14 triangular faces.  We'll model
# T² as a parallelogram with identifications (x,y) ~ (x+1, y) ~ (x, y+1)
# and place the 7 K_7 vertices at the 7 points of a hexagonal lattice
# fundamental domain such that the Heawood adjacency is realized.
#
# For each pair (u, v), the edge {u,v} appears as a straight-line
# segment on T² (taking the shortest representative).
#
# An Eulerian circuit traverses each edge once.  The total winding
# of the circuit is the sum (Σ Δx, Σ Δy) of edge displacements,
# divided by the period.  Since each edge has fixed displacement
# (mod 1, 1), an Eulerian circuit's net winding is fixed by the
# embedding alone.
# =====================================================================

def heawood_vertex_positions() -> dict:
    """Place K_7 vertices on T² (fundamental domain [0,1)²) consistent
    with Heawood's triangulation.  Use the standard hexagonal lattice
    placement: 7 vertices arranged so that K_7 edges form an
    equilateral triangulation.

    Heawood's specific embedding has 7 vertices at positions
    ((i/7, (3i mod 7)/7)) — hexagonal lattice via shift map x→x+y,
    y→y on Z_7 × Z_7.

    More carefully, Heawood's triangulation has 14 triangles, each
    of equal area 1/14.  The vertex placement that achieves this is
    the orbit of (0,0) under the shift (x,y) → (x+1/7, y+a/7) for a
    suitable a, on the torus.  The standard a=3 gives the unique
    triangulating embedding up to symmetry.
    """
    a = 3
    return {v: (v / 7.0, (a * v % 7) / 7.0) for v in range(7)}


def torus_edge_displacement(u: int, v: int, positions: dict) -> Tuple[float, float]:
    """Shortest-representative displacement vector from u to v on T²."""
    px_u, py_u = positions[u]
    px_v, py_v = positions[v]
    dx = px_v - px_u
    dy = py_v - py_u
    # Wrap to [-0.5, 0.5)
    dx = (dx + 0.5) % 1.0 - 0.5
    dy = (dy + 0.5) % 1.0 - 0.5
    return dx, dy


def hierholzer_eulerian_circuit() -> List[int]:
    """Construct an Eulerian circuit on K_7 via Hierholzer's algorithm."""
    n = 7
    adj = {v: set(range(n)) - {v} for v in range(n)}
    stack = [0]
    circuit = []
    while stack:
        v = stack[-1]
        if adj[v]:
            u = min(adj[v])
            adj[v].remove(u)
            adj[u].remove(v)
            stack.append(u)
        else:
            circuit.append(stack.pop())
    return list(reversed(circuit))


def compute_circuit_winding() -> Tuple[float, float, int]:
    section('Heawood winding number on the canonical Eulerian circuit')

    positions = heawood_vertex_positions()
    circuit = hierholzer_eulerian_circuit()
    print(f"\n  Canonical Eulerian circuit: {circuit}")
    print(f"  Length: {len(circuit) - 1} edges (expected 21).")

    total_dx = 0.0
    total_dy = 0.0
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i + 1]
        dx, dy = torus_edge_displacement(u, v, positions)
        total_dx += dx
        total_dy += dy

    print(f"\n  Total displacement on T²: (Δx, Δy) = ({total_dx:.4f}, {total_dy:.4f})")
    print(f"  Winding numbers (p, q) = ({round(total_dx)}, {round(total_dy)})")

    p = round(total_dx)
    q = round(total_dy)
    n = abs(p) + abs(q)
    print(f"\n  Net |winding| = |p| + |q| = {n}")

    return total_dx, total_dy, n


# =====================================================================
# 5. Synthesis
# =====================================================================

def synthesis(p: float, q: float, n: int) -> None:
    section('Synthesis')

    print(f"""
  THE CHAIN (under BPS μ=π):

  (1) BPS condition:   μ · L_trefoil = m_e c²    (Paper 13)
                        with μ = π · ℏc/ƛ_C²
                        ⇒  L_trefoil = ƛ_C / π.

  (2) Action quantization:    S_loop / ℏ = 2π · n     (Bohr-Sommerfeld)
                              with S_loop = m_e c² · T_loop
                              ⇒ T_loop = n · T_C.

  (3) Heawood Eulerian circuit winding on T²:
                              (p, q) = ({p:.3f}, {q:.3f})
                              n = |p| + |q| = {n}.

  COMBINED:  T_circuit = n · T_C = {n} · T_C.

  For our QEC framework's H_YY rotation rate (⟨H_YY⟩ = 21 per natural
  unit), if natural unit = T_circuit = n · T_C = {n} · T_C, the phase
  rotation per "natural unit" is:

     |K_7(natural unit)⟩ = exp(-i · 21) |K_7⟩   from H_YY
     standard QM phase  = exp(-i · 2π · n) |K_7⟩  from rest-mass

  For these to match:  21  =  2π · n  ?  No — 2π · n = 2π · {n} = {2*np.pi*n:.4f},
  which is NOT 21.

  Discrepancy: ratio = 21 / (2π·n) = {21/(2*np.pi*n):.4f}.

  INTERPRETATION:

  The action-quantization derivation gives T_circuit = n · T_C, which
  is a clean consequence of BPS + Bohr-Sommerfeld.  But matching to
  H_YY's rotation rate 21 requires either:

    (a) n = 21 / (2π) ≈ 3.342  (not an integer; rules out simple
        action quantization with this normalization), OR

    (b) H_phys ≠ H_YY directly; rescale to H_phys = (2π/21) H_YY,
        which has spectrum 2π·j(j+1)·(2/21) − ... on the code subspace.
        Then ⟨H_phys⟩|K7 = 2π, matching one full Compton cycle per
        natural unit = T_C.  This is the original postulate, not a
        derivation.

  The cleanest reading: H_YY is the bracket-source operator (gives
  ⟨H_YY^n⟩ = 21^n moments), but is NOT the physical Hamiltonian
  whose evolution equals Schrödinger.  H_phys lives on the same
  Hilbert space but with different normalization, and BPS μ=π fixes
  m_e c² (= 2πℏ/T_C) but not the H_phys/H_YY normalization directly.

  RESULT:  BPS μ=π does NOT independently fix the natural unit = T_C
  for H_YY's rotation rate.  It fixes m_e (= ω_C ℏ) given the
  trefoil's L_trefoil, but the QEC normalization H_phys = (2π/21) H_YY
  requires a separate argument (e.g., direct identification of H_phys
  via the K_7 → so(7) → spinor-S Cartan structure).

  THE DERIVATION REMAINS INCOMPLETE.  Two routes forward:
    Route A: derive H_phys directly from the b2.13 K_7 → so(7)
             generator action on the 7-qubit space, project to the
             8-dim Cartan-graded subspace, and compute the natural
             time evolution operator's eigenstructure.  This may
             give H_phys = (2π/21) H_YY structurally.
    Route B: identify a different operator on |K_7⟩ that has natural
             eigenvalue 2π (not 21), and argue that THIS is the
             physical Hamiltonian.  Candidate: H_phys = (1/21)·H_YY
             with natural unit = (21/2π) T_C.

  Either way, the BPS μ=π hopefully-clean derivation didn't pan out
  with the obvious action-quantization argument.  More structural
  work needed.
""")


# =====================================================================
# 6. Main
# =====================================================================

def main() -> None:
    section('BPS μ=π → Compton period bridge for QEC time evolution')
    print(f"""
Probing whether the BPS condition μ_BPS = π (Paper 13's Bogomolny
saturation) independently fixes the QEC framework's natural time
unit = Compton period, converting the postulate of
`nwt_qec_time_evolution.py` into a derivation.

Compton period:   T_C  =  {T_C:.4e} s
Compton freq.:    ω_C  =  {omega_C:.4e} rad/s
Planck time:      T_Pl =  {T_Pl:.4e} s
m_e/m_Pl:                {M_E/M_PL:.5e}
T_C/T_Pl:                {T_C/T_Pl:.5e}  (= 2π · m_Pl/m_e)
""")

    action_quantization()
    bps_constraint()
    p, q, n = compute_circuit_winding()
    synthesis(p, q, n)


if __name__ == '__main__':
    main()
