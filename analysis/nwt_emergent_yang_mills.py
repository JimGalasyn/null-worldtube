#!/usr/bin/env python3
"""
Emergent Yang-Mills from the Crossing Lattice.

The central claim: the vortex crossing algebra IS a lattice gauge theory,
and the continuum limit gives Yang-Mills dynamics.

The analogy to Wilson's lattice QCD:

  Lattice QCD                 NWT Crossing Lattice
  ──────────────────────────  ───────────────────────────────
  Lattice sites               Vortex crossings
  Link variables U ∈ SU(N)    Strand-exchange operators e^{iθT}
  Plaquette                   Closed path around crossing region
  Wilson action Σ Re Tr(1-U)  Sum over crossing plaquettes
  Lattice spacing a           Distance between crossings
  Continuum limit a → 0       High-winding knots (many crossings)
  Coupling β = 2N/g²          From the AB crossing phase

The trefoil T(2,3) with 3 crossings is the COARSEST SU(3) lattice —
one plaquette.  The continuum physics emerges from the RG running
as you go to finer lattices (higher-crossing knots).
"""

import numpy as np

print("=" * 72)
print("EMERGENT YANG-MILLS FROM THE CROSSING LATTICE")
print("=" * 72)
print()

# ══════════════════════════════════════════════════════════════════════
# Step 1: The crossing lattice
# ══════════════════════════════════════════════════════════════════════

print("─── Step 1: The Crossing Lattice ───")
print()
print("The trefoil T(2,3) has 3 crossings, which define the COARSEST")
print("possible SU(3) lattice: a single plaquette with 3 vertices.")
print()
print("At each crossing, the strand-exchange operator is:")
print("  U_i = exp(i θ_AB × T_a)")
print("where θ_AB is the Aharonov-Bohm crossing phase and T_a = λ_a/2")
print("is the appropriate Gell-Mann generator.")
print()

# From our BPS computation:
theta_per_crossing = 2 * np.pi * 0.012714  # = 2π/(78.66)
theta = theta_per_crossing  # the AB phase in radians

print(f"From the BPS calculation (Paper 11):")
print(f"  θ_AB = 2π × (1/78.66) = {theta:.6f} rad")
print(f"  θ_AB/π = {theta/np.pi:.6f}")
print()

# ══════════════════════════════════════════════════════════════════════
# Step 2: The Wilson loop (one-plaquette action)
# ══════════════════════════════════════════════════════════════════════

print("─── Step 2: The Wilson Loop ───")
print()
print("For the trefoil's single plaquette with 3 crossings:")
print("  W = Tr(U₁ U₂ U₃)")
print()
print("Each U_i = exp(iθ T_i) where T_i are three of the eight")
print("Gell-Mann generators (the off-diagonal real ones from Paper 8):")
print("  T₁ = λ₁/2 (crossing 0↔1)")
print("  T₂ = λ₄/2 (crossing 0↔2)")
print("  T₃ = λ₆/2 (crossing 1↔2)")
print()

# Gell-Mann matrices (the three real off-diagonal ones from Paper 8)
lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)

T1 = lambda1 / 2
T4 = lambda4 / 2
T6 = lambda6 / 2

from scipy.linalg import expm

U1 = expm(1j * theta * T1)
U2 = expm(1j * theta * T4)
U3 = expm(1j * theta * T6)

# Wilson loop
W = U1 @ U2 @ U3
W_trace = np.trace(W)
W_real = W_trace.real

print(f"  U₁ = exp(iθ λ₁/2), diagonal: {np.diag(U1).real}")
print(f"  W = Tr(U₁ U₂ U₃) = {W_trace:.6f}")
print(f"  Re Tr(W) = {W_real:.6f}")
print(f"  Re Tr(W)/3 = {W_real/3:.6f}")
print()

# Wilson action for one plaquette
S_plaq = 1 - W_real / 3
print(f"  Wilson plaquette action: S = 1 - Re Tr(W)/3 = {S_plaq:.8f}")
print()

# ══════════════════════════════════════════════════════════════════════
# Step 3: Extract the lattice coupling
# ══════════════════════════════════════════════════════════════════════

print("─── Step 3: Lattice Coupling ───")
print()
print("In Wilson's lattice QCD, the action for one plaquette is:")
print("  S_W = β (1 - Re Tr(U_plaq)/N)")
print("where β = 2N/g² is the lattice coupling.")
print()
print("Comparing: our plaquette action S = 1 - Re Tr(W)/3 corresponds")
print("to β = 1 (the coefficient in front).")
print()
print("But this needs refinement.  For SMALL θ (weak coupling):")
print("  U_i ≈ 1 + iθ T_i - θ²T_i²/2 + ...")
print("  W ≈ Tr(1 + iθ(T₁+T₂+T₃) - θ²(...)/2 + ...)")
print("  Re Tr(W)/3 ≈ 1 - θ² C₂/3 + ...")
print("where C₂ is a group-theory factor.")
print()

# Compute the quadratic term explicitly
# W to second order: Tr(1 - θ²/2 × (T₁² + T₂² + T₃²) - θ²(T₁T₂ + T₂T₃ + T₁T₃))
T_sum_sq = T1@T1 + T4@T4 + T6@T6
T_cross = T1@T4 + T4@T6 + T1@T6
Tr_T2 = np.trace(T_sum_sq).real
Tr_Tcross = np.trace(T_cross).real

print(f"  Tr(T₁² + T₄² + T₆²) = {Tr_T2:.4f}")
print(f"  Tr(T₁T₄ + T₄T₆ + T₁T₆) = {Tr_Tcross:.4f}")
print()

# S ≈ θ²/2 × (Tr(T²) + Tr(T_cross))/3  for small θ
S_approx = theta**2 / 2 * (Tr_T2 + 2*Tr_Tcross) / 3
print(f"  S ≈ θ² × [{Tr_T2:.2f} + 2×{Tr_Tcross:.2f}]/(2×3)")
print(f"    = θ² × {(Tr_T2 + 2*Tr_Tcross)/6:.4f}")
print(f"    = {theta**2:.6f} × {(Tr_T2 + 2*Tr_Tcross)/6:.4f}")
print(f"    = {S_approx:.8f}")
print(f"  Exact: {S_plaq:.8f}")
print(f"  Agreement: {abs(S_approx - S_plaq)/S_plaq * 100:.2f}%")
print()

# The lattice coupling relation: S = β × g²_lat × C₂ / (2N)
# where g²_lat = θ² (the square of the crossing phase)
# So: β = S / (g²_lat × C₂/(2N)) = S × 2N / (θ² × C₂)
C2_eff = (Tr_T2 + 2*Tr_Tcross) / 3  # effective Casimir for our configuration
g2_lat = theta**2  # lattice coupling squared
beta_lat = 2 * 3 / g2_lat  # standard Wilson β = 2N/g²

print(f"  Lattice coupling:")
print(f"    g²_lat = θ² = {g2_lat:.6f}")
print(f"    β = 2N/g² = 6/{g2_lat:.6f} = {beta_lat:.2f}")
print(f"    α_lat = g²/(4π) = {g2_lat/(4*np.pi):.6f} = 1/{4*np.pi/g2_lat:.1f}")
print()

# ══════════════════════════════════════════════════════════════════════
# Step 4: Renormalization group running to physical scale
# ══════════════════════════════════════════════════════════════════════

print("─── Step 4: RG Running to Physical Scale ───")
print()
print("The lattice coupling β = 2N/g² is defined at the LATTICE SCALE,")
print("which for the trefoil is the crossing separation:")
print("  a ≈ R_electron ≈ βξ = (√5/2) × 386 fm ≈ 432 fm")
print("  → E_lattice = ℏc/a ≈ 197/432 ≈ 0.46 MeV")
print()
print("The PHYSICAL coupling α_s is measured at E = M_Z = 91.2 GeV.")
print("The RG running from E_lattice to M_Z uses the QCD β-function:")
print()
print("  1/α_s(μ) = 1/α_s(μ₀) + (b₀/2π) ln(μ/μ₀)")
print()
print("where b₀ = (11N_c - 2N_f)/(3) = (33 - 2N_f)/3 for SU(3).")
print()

# Lattice scale
E_lat = 0.46  # MeV
E_MZ = 91200  # MeV (Z boson mass)
alpha_lat = g2_lat / (4 * np.pi)

# QCD β-function coefficients (1-loop)
# For N_c = 3, N_f = 6 (all quark flavors active at high scale):
# b₀ = (11×3 - 2×6)/3 = (33-12)/3 = 7
# But at low scale (E ~ 1 MeV), only N_f = 3 (u, d, s):
# b₀ = (33-6)/3 = 9

# Use N_f = 3 for running from lattice to ~1 GeV, then N_f = 6 to M_Z
# Simplified: use effective b₀ ≈ 9 for the full range (rough)

b0 = 9.0  # effective for SU(3) with N_f ≈ 3-4

# 1-loop running: 1/α(μ) = 1/α(μ₀) + b₀/(2π) ln(μ/μ₀)
inv_alpha_MZ = 1/alpha_lat + b0/(2*np.pi) * np.log(E_MZ/E_lat)

alpha_s_pred = 1/inv_alpha_MZ

print(f"  At lattice scale ({E_lat:.2f} MeV):")
print(f"    α_lat = {alpha_lat:.6f} = 1/{1/alpha_lat:.1f}")
print(f"  After 1-loop running to M_Z ({E_MZ/1000:.1f} GeV):")
print(f"    Δ(1/α) = b₀/(2π) × ln(M_Z/E_lat)")
print(f"           = {b0:.0f}/(2π) × ln({E_MZ/E_lat:.0f})")
print(f"           = {b0/(2*np.pi):.3f} × {np.log(E_MZ/E_lat):.3f}")
print(f"           = {b0/(2*np.pi) * np.log(E_MZ/E_lat):.2f}")
print(f"    1/α_s(M_Z) = {1/alpha_lat:.2f} + {b0/(2*np.pi)*np.log(E_MZ/E_lat):.2f}")
print(f"               = {inv_alpha_MZ:.2f}")
print(f"    α_s(M_Z) = {alpha_s_pred:.4f}")
print()

alpha_s_exp = 0.1179  # PDG 2024
print(f"  Experimental: α_s(M_Z) = {alpha_s_exp:.4f} = 1/{1/alpha_s_exp:.1f}")
print(f"  Predicted:    α_s(M_Z) = {alpha_s_pred:.4f} = 1/{1/alpha_s_pred:.1f}")
print(f"  Ratio: {alpha_s_pred/alpha_s_exp:.3f}")
print()

# ── Try with different b₀ values ──
print("  Sensitivity to b₀ (number of active flavors):")
for nf in [2, 3, 4, 5, 6]:
    b0_nf = (33 - 2*nf) / 3
    inv_a = 1/alpha_lat + b0_nf/(2*np.pi) * np.log(E_MZ/E_lat)
    a_pred = 1/inv_a if inv_a > 0 else float('inf')
    print(f"    N_f={nf}: b₀={b0_nf:.1f}, α_s(M_Z) = {a_pred:.4f} = 1/{1/a_pred:.1f}")
print()

# ══════════════════════════════════════════════════════════════════════
# Step 5: The Yang-Mills action from the continuum limit
# ══════════════════════════════════════════════════════════════════════

print("─── Step 5: From Wilson Action to Yang-Mills ───")
print()
print("In the continuum limit (a → 0), Wilson's lattice action becomes:")
print()
print("  S_W = β Σ_plaq (1 - Re Tr(U_plaq)/N)")
print("      → (1/4g²) ∫ Tr(F_μν F^μν) d⁴x")
print()
print("where F_μν = ∂_μA_ν - ∂_νA_μ - ig[A_μ, A_ν] is the field strength.")
print()
print("In NWT:")
print("  • The 'lattice sites' are vortex crossings")
print("  • The 'link variables' are AB phase operators exp(iθ T_a)")
print("  • The 'plaquette' is the closed path around a crossing region")
print("  • The 'continuum limit' is the regime of many crossings")
print("    (high-winding knots with fine crossing structure)")
print()
print("The GAUGE FIELD A_μ^a emerges as the COARSE-GRAINED average of")
print("the discrete crossing phases over a region of spacetime.")
print("Specifically:")
print()
print("  A_μ^a(x) ≈ (1/V) Σ_{crossings in V} θ_i T_i^a / (Δx_μ)")
print()
print("where V is a coarse-graining volume and Δx_μ is the lattice spacing.")
print()
print("The LOCAL GAUGE INVARIANCE arises because the crossing operators")
print("transform as U → g U g⁻¹ under a change of basis at each crossing")
print("(i.e., which strand is 'over' vs 'under'). This is EXACTLY the")
print("lattice gauge transformation, which becomes local SU(3) in the")
print("continuum limit.")
print()

# ══════════════════════════════════════════════════════════════════════
# Step 6: The covariant derivative
# ══════════════════════════════════════════════════════════════════════

print("─── Step 6: The Covariant Derivative ───")
print()
print("On the lattice, parallel transport from site x to site x+μ̂ is:")
print("  ψ(x+μ̂) = U_μ(x) ψ(x)")
print()
print("In the continuum limit: U_μ(x) ≈ exp(igA_μa), so:")
print("  ψ(x+aμ̂) ≈ (1 + iga A_μ^a T^a) ψ(x)")
print("  → (ψ(x+aμ̂) - ψ(x))/a ≈ igA_μ^a T^a ψ(x)")
print("  → ∂_μψ → D_μψ = ∂_μψ - igA_μ^a T^a ψ")
print()
print("In NWT: when a quark (a crossing state) propagates from one")
print("crossing to the next, it picks up the AB phase of the vortex")
print("strand it follows. This IS parallel transport, and the")
print("accumulated phase IS the covariant derivative.")
print()

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("SUMMARY: YANG-MILLS FROM VORTEX CROSSINGS")
print("═" * 72)
print()
print("The promotion from crossing algebra to dynamical gauge theory")
print("is NOT a new construction — it IS lattice gauge theory, applied")
print("to the natural lattice of vortex crossings in the condensate.")
print()
print("  Crossing algebra (Paper 8/11)  =  one-plaquette lattice theory")
print("  Continuum limit (many crossings)  =  Yang-Mills dynamics")
print("  AB crossing phase θ  =  lattice gauge coupling")
print("  Strand-exchange operator  =  link variable U ∈ SU(N)")
print("  Basis change at crossing  =  local gauge transformation")
print("  Quark propagation between crossings  =  parallel transport")
print()
print("Quantitative results:")
print(f"  Lattice coupling: α_lat = θ²/(4π) = {alpha_lat:.6f} = 1/{1/alpha_lat:.1f}")
print(f"  After QCD running to M_Z: α_s = {alpha_s_pred:.4f} = 1/{1/alpha_s_pred:.1f}")
print(f"  Experimental: α_s(M_Z) = {alpha_s_exp:.4f} = 1/{1/alpha_s_exp:.1f}")
print()
print("The lattice-to-continuum bridge is well-understood physics")
print("(Wilson 1974, Creutz 1980). What is NEW here is the identification")
print("of the crossing lattice as the PHYSICAL origin of the gauge field,")
print("not an artificial discretization for numerical computation.")
