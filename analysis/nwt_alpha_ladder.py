#!/usr/bin/env python3
"""
The 1/α Energy Ladder: Particle Mass Hierarchy from EM Self-Interaction
=========================================================================

Each factor of 1/α represents one additional EM self-interaction.
The ladder m_e/α^n generates the major energy scales of physics:

  Level 0: m_e           = 0.511 MeV   (atomic)
  Level 1: m_e/α         = 70.0 MeV    (nuclear)
  Level 2: m_e/α²        = 9.60 GeV    (heavy quark)
  Level 3: m_e/α³        = 1.315 TeV   (electroweak)

Key results:
  m_π = 2 × Λ₁           = 140.1 MeV  (0.34% from measured)
  m_t = 2k² × Λ₂         = 172.7 GeV  (0.02% from measured!!)
  J/ψ = Λ₂/π             = 3.05 GeV   (1.4%)
  m_b = Λ₂/√5            = 4.29 GeV   (2.7%)
  Υ(1S) = Λ₂             = 9.60 GeV   (1.4%)
"""

import numpy as np

alpha = 7.2973525693e-3
m_e = 0.51099895  # MeV
k = 3  # torus aspect ratio
p = 2  # toroidal winding

print("=" * 72)
print("THE 1/α ENERGY LADDER")
print("=" * 72)

L = [m_e / alpha**n for n in range(4)]

print(f"\n  {'Level':<8} {'Scale':<12} {'Value':<15} {'Physics'}")
print(f"  {'─'*60}")
for n, (scale, phys) in enumerate([
    (L[0], "Electron (input)"),
    (L[1], "Nuclear / pion scale"),
    (L[2], "Heavy quark / bottomonium"),
    (L[3], "Electroweak"),
]):
    if scale < 1000:
        print(f"  {n:<8} m_e/α^{n:<5} {scale:<15.2f} MeV  {phys}")
    else:
        print(f"  {n:<8} m_e/α^{n:<5} {scale/1000:<15.3f} GeV  {phys}")

# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("LEVEL 1 PREDICTIONS (nuclear scale)")
print(f"{'─'*72}\n")

preds1 = [
    ("m_π±",   "2Λ₁",         2*L[1],       139.57,   "pair at charge boundary"),
    ("m_π⁰",   "2Λ₁",         2*L[1],       134.98,   "neutral (EM mass shift)"),
    ("f_π",    "(k+1)/k × Λ₁", (k+1)/k*L[1], 92.1,    "pion decay constant"),
    ("Λ_QCD",  "kΛ₁",          k*L[1],       220,      "QCD scale"),
]

print(f"  {'Quantity':<10} {'Formula':<18} {'Predicted':<12} {'Measured':<12} {'Δ%':>6}")
print(f"  {'─'*70}")
for name, formula, pred, meas, note in preds1:
    pct = (pred/meas - 1) * 100
    print(f"  {name:<10} {formula:<18} {pred:<12.1f} {meas:<12.1f} {pct:>+6.2f}%  {note}")

# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("LEVEL 2 PREDICTIONS (heavy quark scale)")
print(f"{'─'*72}\n")

preds2 = [
    ("J/ψ",    "Λ₂/π",         L[2]/np.pi,    3097,    "charmonium (1/π factor)"),
    ("m_b",    "Λ₂/√5",        L[2]/np.sqrt(5), 4180,  "b quark (knot invariant √5)"),
    ("m_b",    "(k+1)/k² × Λ₂", (k+1)/k**2*L[2], 4180, "b quark (torus formula)"),
    ("B_c",    "(2/3)Λ₂",      2/3*L[2],       6275,   "charmed B"),
    ("η_b",    "Λ₂",           L[2],            9399,   "bottomonium pseudoscalar"),
    ("Υ(1S)",  "Λ₂",           L[2],            9460,   "bottomonium ground state"),
    ("m_t",    "2k²Λ₂",        2*k**2*L[2],   172760,  "★ TOP QUARK"),
]

print(f"  {'Quantity':<10} {'Formula':<18} {'Predicted':<12} {'Measured':<12} {'Δ%':>6}")
print(f"  {'─'*70}")
for name, formula, pred, meas, note in preds2:
    pct = (pred/meas - 1) * 100
    u = "MeV" if meas < 10000 else "MeV"
    print(f"  {name:<10} {formula:<18} {pred:<12.0f} {meas:<12.0f} {pct:>+6.2f}%  {note}")

# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("★ THE TOP QUARK MASS")
print(f"{'─'*72}")

m_t_pred = 2 * k**2 * L[2]
m_t_meas = 172760
print(f"""
  m_t = 2k² × m_e/α² = 2 × {k}² × {L[2]/1000:.3f} GeV
      = {2*k**2} × {L[2]/1000:.3f} GeV
      = {m_t_pred/1000:.3f} GeV

  Measured: {m_t_meas/1000:.3f} GeV
  Error: {(m_t_pred/m_t_meas - 1)*100:+.4f}%

  Factors:  2  = pair creation factor (same as pion)
            k² = 9 = aspect ratio squared
            Λ₂ = m_e/α² = second rung of the 1/α ladder

  The top quark is the PAIR RESONANCE at Level 2,
  amplified by the full torus geometry (k²).
  Exactly as the pion is the pair resonance at Level 1.
""")

# ══════════════════════════════════════════════════════════════════════════
print(f"{'─'*72}")
print("THE PHYSICAL MECHANISM")
print(f"{'─'*72}")
print(f"""
  WHY 1/α per level:
    The electron's field at distance r from its charge center:
      E(r) = e/(4πε₀r²)
    At r = r_e = αR:  E/E_S = 1/α = 137  (super-Schwinger)

    Each self-interaction at this field strength amplifies the
    energy by 1/α. This is NOT perturbative — it's the full
    nonlinear QED response at the charge boundary.

  WHY 2 × Λ_n for the lightest pair:
    The pair (particle + antiparticle) has wavelength r_e/p.
    The factor p=2 from the (2,1) electron topology.
    At Level 1: m_π = 2Λ₁ (the pion)
    At Level 2: the 2 appears in m_t = 2k²Λ₂ (with k² amplification)

  WHY k² for the top quark:
    The top quark is the MOST MASSIVE fermion — it lives at the
    highest amplification of the pair resonance. The factor k²
    comes from the torus metric: the (k=3) aspect ratio enhances
    the interaction k² = 9 times through the scalar-vector splitting
    (the same k² that gives nuclear magic numbers in Paper 3).

  THE COMPLETE MASS HIERARCHY:
    m_e → m_π → m_Υ → m_t
    = m_e → 2m_e/α → m_e/α² → 2k²m_e/α²

  Input: m_e, α, p=2, k=3. Output: the particle mass spectrum.
""")
