#!/usr/bin/env python3
"""
NWT-Rybakov Path A, Step 1: Exact numerical integrals replace MVT estimates.

Strategy: keep Rybakov's trial functions (Eqs. 31, 36, 37, 45) unchanged,
but evaluate all action integrals EXACTLY using scipy.integrate.quad and
scipy.special.gamma, instead of the midpoint / mean-value-theorem estimates
Rybakov used in Eqs. 38-44 and 50-58.

This isolates the MVT error from the trial-function error.  If the exact
integrals give a structurally viable fit (all-positive coefficients for
the charged-lepton masses), Path A lives and the MVT was the bug.

Key derivation (traced from stationarity conditions):
  k^2  ∝  1 / [(n^2+1) I_1(n)]
  a^4  ∝  (n^2+1) I_1^4 / I_2
  f(n) ∝  I_2^{1/4} / [(n^2+1)^{1/4} I_1]

where
  I_1(n) = ∫_0^∞ t^{2ν-1} e^{-2t} dt = Γ(2ν) / 2^{2ν}    [sigma-model]
  I_2(n) = ∫_0^∞ t^{8ν-5} e^{-4t} dt = Γ(8ν-4) / 4^{8ν-4}  [Higgs potential]
"""

import numpy as np
from scipy.special import gamma as Gamma
from scipy.integrate import quad

# ── Rybakov's n-dependent parameters ──
def nu(n):  return 2.0 + np.sqrt(7*n*n + 11) / 7.0
def t0(n):  return (8.0*nu(n) - 5.0) / 3.0

# ── Exact action integrals ──
def I1_exact(n):
    """Sigma-model integral: ∫₀^∞ t^{2ν-1} e^{-2t} dt = Γ(2ν)/2^{2ν}"""
    nu_ = nu(n)
    return Gamma(2*nu_) / (2.0**(2*nu_))

def I2_exact(n):
    """Higgs potential integral: ∫₀^∞ t^{8ν-5} e^{-4t} dt = Γ(8ν-4)/4^{8ν-4}"""
    nu_ = nu(n)
    return Gamma(8*nu_ - 4) / (4.0**(8*nu_ - 4))

# ── Rybakov's MVT estimates (for comparison) ──
def I1_mvt(n):
    nu_ = nu(n)
    return (2*nu_ - 1)**(2*nu_ - 1) * np.exp(-2*(2*nu_ - 1))

def I2_mvt(n):
    t0_ = t0(n)
    return np.exp(-3*t0_) * t0_**(3*t0_)

# ── n-dependent mass functions from exact integrals ──
# From stationarity conditions:
#   k²  ∝ 1 / [(n²+1) I₁]
#   a⁴  ∝ (n²+1) I₁⁴ / I₂
#   f(n) = L₀/a ∝ I₂^{1/4} / [(n²+1)^{1/4} I₁]
# (All proportionality constants are n-independent.)

def f_exact(n):
    """Electrostatic mass function ∝ L₀/a(n) from EXACT integrals."""
    return I2_exact(n)**0.25 / ((n*n + 1)**0.25 * I1_exact(n))

def f_mvt(n):
    """Same function using Rybakov's MVT estimates."""
    return I2_mvt(n)**0.25 / ((n*n + 1)**0.25 * I1_mvt(n))

def M_exact(n):
    """Magnetic mass function ∝ f^3.  M(n) = p f(n)^3, p = (8/9^3)(3/π)^{1/2}."""
    return (8.0/9**3) * np.sqrt(3.0/np.pi) * f_exact(n)**3

def S_exact(n):
    """Higgs/kinetic/gravity mass function ∝ 1/f.  S(n) = (2/9)(π/3)^{1/4} / f(n)."""
    return (2.0/9.0) * (np.pi/3.0)**0.25 / f_exact(n)

# ── G(n): the Skyrme-Faddeev mass function ──
# G(n) involves the Skyrme-Faddeev integral which has the form:
#   I_SF = ∫₀^∞ [g(t)]^2 dt
# where g(t) = e^{-3t/2} t^{3(ν-1/2)} (2t - ν)
# Rybakov approximates this via midpoint of g^2.
# We compute it EXACTLY via numerical quadrature.
#
# Then G(n) also involves k_α^2, the Bohr magneton factors, etc.
# From the derivation, the Skyrme-Faddeev mass contribution has the form:
#   m_ε ∝ a(n)^{-1} × k_α^4 × κ₀² × I_SF(n)
# where k_α^2 = k^2 (5ν-3)/(4κ₀²(2ν-1))  [Eq. 35]
#
# But for the STRUCTURAL test we only need n-dependent ratios.
# The n-dependent part of m_ε involves:
#   k_α^4 ∝ k^4 × [(5ν-3)/(2ν-1)]^2
#   I_SF(n)
#   1/a(n) ∝ f(n)
# And k^4 ∝ 1/[(n²+1)^2 I₁^2]
# So the full n-dependent function is:
#   G_full(n) ∝ f(n) × [(5ν-3)/(2ν-1)]^2 / [(n²+1)^2 I₁^2] × I_SF(n)

def I_SF_exact(n):
    """Skyrme-Faddeev integral: ∫₀^∞ [g(t)]^2 dt
    where g(t) = e^{-3t/2} t^{3(ν-1/2)} (2t - ν)"""
    nu_ = nu(n)
    def g2(t):
        return (np.exp(-1.5*t) * t**(3*(nu_ - 0.5)) * (2*t - nu_))**2
    result, _ = quad(g2, 0, np.inf, limit=200)
    return result

def I_SF_mvt(n):
    """MVT estimate of I_SF from Rybakov's g^2 = [g(t1)^2 + g(t2)^2]/4"""
    nu_ = nu(n)
    a, b, c = 3.0, -(15*nu_/2 - 1), 3*nu_*(2*nu_ - 1)/2
    disc = b*b - 4*a*c
    sq = np.sqrt(disc)
    t1, t2 = (-b - sq)/(2*a), (-b + sq)/(2*a)
    def g(t): return np.exp(-1.5*t) * t**(3*(nu_ - 0.5)) * (2*t - nu_)
    return (g(t1)**2 + g(t2)**2) / 4.0


def G_exact(n):
    """Full n-dependent Skyrme-Faddeev mass function from EXACT integrals.
    G(n) ∝ f(n) × [(5ν-3)/(2ν-1)]^2 / [(n²+1)^2 I₁^2] × I_SF(n)
    """
    nu_ = nu(n)
    return (f_exact(n)
            * ((5*nu_ - 3) / (2*nu_ - 1))**2
            / ((n*n + 1)**2 * I1_exact(n)**2)
            * I_SF_exact(n))


# ── Also need the EM correction integral for R(n) ──
# For now, skip α²T R(n) since Rybakov says T << Z.  If the structural
# test passes with f, M, S, G we're good; if not, we add R later.


# ══════════════════════════════════════════════════════════════
#                        MAIN RESULTS
# ══════════════════════════════════════════════════════════════

M_E, M_MU, M_TAU = 0.510998928, 105.6583715, 1776.82

print("=" * 72)
print("PATH A — Exact integral n-dependent functions")
print("=" * 72)
print()

# ── Compare exact vs MVT ──
print("─── Exact vs MVT: individual integrals ───")
print(f"{'n':>3} {'I₁ exact':>12} {'I₁ MVT':>12} {'ratio':>8}   "
      f"{'I₂ exact':>12} {'I₂ MVT':>12} {'ratio':>8}   "
      f"{'I_SF exact':>12} {'I_SF MVT':>12} {'ratio':>8}")
for n in (1, 2, 3):
    i1e, i1m = I1_exact(n), I1_mvt(n)
    i2e, i2m = I2_exact(n), I2_mvt(n)
    ise, ism = I_SF_exact(n), I_SF_mvt(n)
    print(f"{n:>3} {i1e:>12.4e} {i1m:>12.4e} {i1e/i1m:>8.3f}   "
          f"{i2e:>12.4e} {i2m:>12.4e} {i2e/i2m:>8.3f}   "
          f"{ise:>12.4e} {ism:>12.4e} {ise/ism:>8.3f}")
print()

# ── n-dependent function ratios ──
print("─── n-dependent function ratios (normalized to n=1) ───")
fe = [f_exact(n) for n in (1,2,3)]
fm = [f_mvt(n) for n in (1,2,3)]
Se = [S_exact(n) for n in (1,2,3)]
Ge = [G_exact(n) for n in (1,2,3)]

print(f"  f_exact(n) ratios: 1, {fe[1]/fe[0]:.4f}, {fe[2]/fe[0]:.4f}")
print(f"  f_MVT(n)   ratios: 1, {fm[1]/fm[0]:.4f}, {fm[2]/fm[0]:.4f}")
print(f"  S_exact(n) ratios: 1, {Se[1]/Se[0]:.4f}, {Se[2]/Se[0]:.4f}")
print(f"  G_exact(n) ratios: 1, {Ge[1]/Ge[0]:.4f}, {Ge[2]/Ge[0]:.4f}")
print(f"  target m/m_e     : 1, {M_MU/M_E:.1f},  {M_TAU/M_E:.1f}")
print()

# ── Structural viability test ──
print("─── STRUCTURAL VIABILITY TEST (exact integrals) ───")
print("  m_n = C1 f(n) + C3 S(n) + C4 G(n)   [C2 = 0 for now]")
print()

f_vec = np.array(fe)
S_vec = np.array(Se)
G_vec = np.array(Ge)
m_exp = np.array([M_E, M_MU, M_TAU])

A = np.column_stack([f_vec, S_vec, G_vec])
coefs = np.linalg.solve(A, m_exp)
C1, C3, C4 = coefs

m_pred = C1*f_vec + C3*S_vec + C4*G_vec
all_pos = C1 > 0 and C3 > 0 and C4 > 0

print(f"  C1 = {C1:+.6e}  (coef of f,  should be positive)")
print(f"  C3 = {C3:+.6e}  (coef of S,  should be positive)")
print(f"  C4 = {C4:+.6e}  (coef of G,  should be positive)")
print(f"  ALL POSITIVE? {'YES ✓' if all_pos else 'NO ✗'}")
print()

for n, me, name in ((1, M_E, 'e'), (2, M_MU, 'μ'), (3, M_TAU, 'τ')):
    idx = n - 1
    print(f"  n={n} {name}:  m_pred = {m_pred[idx]:.6f}  m_exp = {me:.6f}  "
          f"err = {(m_pred[idx]-me)/me*100:+.2e}%")

# ── Wide C2 scan (include self-consistency term) ──
print()
print("─── C2 scan (m_n = C1 f + C2 M/m² + C3 S + C4 G) ───")
M_vec = np.array([M_exact(n) for n in (1,2,3)])
best_pos = (None, 1e99)
best_any = (None, 1e99)
for log10_C2 in np.linspace(-10, 50, 600):
    C2 = 10**log10_C2
    rhs = m_exp - C2 * M_vec / m_exp**2
    A_mat = np.column_stack([f_vec, S_vec, G_vec])
    try:
        cf = np.linalg.solve(A_mat, rhs)
    except np.linalg.LinAlgError:
        continue
    C1_, C3_, C4_ = cf
    mp = C1_*f_vec + C2*M_vec/m_exp**2 + C3_*S_vec + C4_*G_vec
    if not np.all(mp > 0):
        continue
    log_err = np.sum((np.log10(mp) - np.log10(m_exp))**2)
    if log_err < best_any[1]:
        best_any = ((C2, C1_, C3_, C4_), log_err)
    if C1_ > 0 and C3_ > 0 and C4_ > 0 and C2 > 0:
        if log_err < best_pos[1]:
            best_pos = ((C2, C1_, C3_, C4_), log_err)

if best_pos[0] is not None:
    C2, C1_, C3_, C4_ = best_pos[0]
    print(f"  BEST ALL-POSITIVE: C2={C2:.3e} C1={C1_:+.3e} "
          f"C3={C3_:+.3e} C4={C4_:+.3e}  log-err={best_pos[1]:.2e}")
else:
    print(f"  NO all-positive fit found in C2 scan")
    if best_any[0] is not None:
        C2, C1_, C3_, C4_ = best_any[0]
        print(f"  best any-sign: C2={C2:.3e} C1={C1_:+.3e} "
              f"C3={C3_:+.3e} C4={C4_:+.3e}  log-err={best_any[1]:.2e}")

print()
print("=" * 72)
print("PATH A STEP 1 COMPLETE")
print("=" * 72)
