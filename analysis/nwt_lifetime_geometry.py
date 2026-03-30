#!/usr/bin/env python3
"""
NWT Lifetime Matrix Elements and Geometric Carrier Sizes

Part 1: Lifetime from topological reconnection
  Γ = ν_circ × P_reconnect × N_crossings × (phase space factor)

  - ν_circ: mode circulation frequency around the carrier
  - P_reconnect: probability per pass that vortex cores overlap → topology change
  - N_crossings: number of crossing/reconnection sites
  - Phase space: available energy → daughter state density

Part 2: Geometric carrier sizes
  - R = β × ξ (major radius from phase closure)
  - r = R/κ (tube radius, κ = π²)
  - Physical path length on the carrier
  - Inter-core distances for Hopf links
"""

import numpy as np
from math import gcd

# ── Physical constants ────────────────────────────────────────────────
hbar = 1.054571817e-34   # J·s
c    = 2.99792458e8      # m/s
m_e  = 9.1093837015e-31  # kg
alpha = 7.2973525693e-3
e_charge = 1.602176634e-19  # C
MeV_to_J = 1.602176634e-13
MeV_to_kg = MeV_to_J / c**2

# Derived NWT constants
xi = hbar / (m_e * c)          # healing length = reduced Compton wavelength
m_star = m_e / np.sqrt(2)      # effective mass in condensate
kappa_circ = 2 * np.pi * hbar / m_star  # quantum of circulation
c_s = c / np.sqrt(2)           # speed of sound in condensate (= c/√2)

KAPPA = np.pi**2               # aspect ratio R/r
ME_MEV = 0.511
BETA_E = np.sqrt(5.0/4.0)     # electron β = √5/2

print("=" * 100)
print("NWT PHYSICAL CONSTANTS")
print("=" * 100)
print(f"  Healing length ξ = ƛ_C = {xi*1e15:.2f} fm = {xi:.4e} m")
print(f"  Effective mass m* = m_e/√2 = {m_star:.4e} kg")
print(f"  Circulation quantum κ = 2πℏ/m* = {kappa_circ:.4e} m²/s")
print(f"  Sound speed c_s = c/√2 = {c_s:.4e} m/s")
print(f"  Aspect ratio κ_torus = π² = {KAPPA:.4f}")
print(f"  Tube radius r = R/π² = R × {1/KAPPA:.6f}")


# ── Mass formula utilities ────────────────────────────────────────────
def beta_from_phase(p_eff, m_int):
    val = m_int**2 / p_eff**2 - 1
    return np.sqrt(val) if val > 0 else None

def mass_mev(p_eff, q_eff, m_int, n_q):
    beta = beta_from_phase(p_eff, m_int)
    if beta is None or beta <= 1e-6: return None
    pq = p_eff**2 + q_eff**2
    lr = np.log(8*beta) / np.log(8*BETA_E)
    if lr <= 0: return None
    geom = (pq/5.0) * (beta/BETA_E) * lr
    try: enh = float(n_q**q_eff) if n_q > 0 and q_eff > 0 else 1.0
    except: return None
    if not np.isfinite(enh) or enh > 1e30: return None
    mass = geom * enh * ME_MEV
    return mass if np.isfinite(mass) and mass > 0 else None


# ── Particle database ─────────────────────────────────────────────────
# (name, mode, carrier, m_int, n_q, mass_pdg_MeV, Γ_pdg_MeV, J, carrier_type)
PARTICLES = [
    # Leptons
    ('e⁻',     (2,1),  (1,1),   3, 0,    0.511,    0,         0.5, 'unknot'),
    ('μ⁻',     (3,2),  (1,4),  25, 0,  105.66,     3.01e-16,  0.5, 'unknot'),
    ('τ⁻',     (1,3),  (1,12),  8, 0, 1776.86,     2.27e-9,   0.5, 'unknot'),
    # Light mesons
    ('π⁰',     (3,1),  (6,3),  22, 2,  135.0,      7.73e-6,   0,   'Hopf(2)'),
    ('π±',     (3,1),  (4,2),  27, 2,  139.57,     2.53e-14,  0,   'Hopf(2)'),
    ('K±',     (2,1),  (6,3),  38, 2,  493.68,     5.32e-14,  0,   'Hopf(2)'),
    ('η',      (2,1),  (5,5),  17, 2,  547.86,     1.31e-3,   0,   'Hopf(2)'),
    ('ρ',      (3,1),  (5,5),  22, 2,  775.26,   149.1,       1,   'Hopf(2)'),
    ('ω',      (1,1),  (7,7),   9, 2,  782.66,     8.68,      1,   'Hopf(2)'),
    ('φ',      (1,1),  (5,5),  22, 2, 1019.46,     4.25,      1,   'Hopf(2)'),
    # Baryons
    ('p',      (2,1),  (2,3),  32, 3,  938.27,     0,         0.5, 'trefoil'),
    ('n',      (2,1),  (2,3),  32, 3,  939.57,     7.49e-28,  0.5, 'trefoil'),
    ('Λ',      (3,1),  (2,3),  35, 3, 1115.7,      2.50e-12,  0.5, 'trefoil'),
    ('Σ',      (2,1),  (2,3),  39, 3, 1189.4,      8.92e-12,  0.5, 'trefoil'),
    ('Δ',      (3,1),  (2,3),  38, 3, 1232.0,    117.0,       1.5, 'trefoil'),
    ('Ω⁻',    (1,2),  (3,2),  16, 3, 1672.5,      8.02e-12,  1.5, 'trefoil'),
    # Charm
    ('J/ψ',   (1,1),  (7,7),  16, 2, 3096.9,      0.0929,    1,   'Hopf(2)'),
    ('D⁰',    (1,1),  (5,5),  35, 2, 1864.8,      1.60e-9,   0,   'Hopf(2)'),
    # Bottom
    ('Υ',     (1,1),  (7,7),  35, 2, 9460.3,      0.054,     1,   'Hopf(2)'),
    ('B±',    (1,1),  (7,7),  23, 2, 5279.3,      4.02e-10,  0,   'Hopf(2)'),
    # EWK bosons (NEW — Hopf(2) with q_m=5)
    ('W±',    (1,5),  (2,2),  10, 2, 80379.0,  2085.0,        1,   'Hopf(2)'),
    ('Z⁰',   (1,5),  (2,2),  11, 2, 91188.0,  2495.0,        1,   'Hopf(2)'),
    ('H⁰',   (2,5),  (2,2),  26, 2,125100.0,     4.07e-3,    0,   'Hopf(2)'),
]


# ══════════════════════════════════════════════════════════════════════
# PART 1: GEOMETRIC CARRIER SIZES
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 1: GEOMETRIC CARRIER SIZES")
print("=" * 100)

print(f"""
  For each particle:
    R = β × ξ             (major radius of embedding torus)
    r = R / κ = R / π²    (tube radius)
    L = carrier path length (arc length of the carrier knot)
    d_crossing = minimum inter-core distance at crossings (for links/knots)

  For a torus knot T(p,q) with κ = R/r:
    L = 2π R √(p² + q²/κ²) ≈ 2πRp for p >> q/κ

  For a Hopf(n) link with n components:
    Each component is an unknot of radius R at angle 2πk/n
    Inter-ring separation at crossing ≈ 2r (tube diameter)
""")

print(f"  {'Name':>6s} {'mode':>7s} {'carrier':>9s} {'eff(p,q)':>10s} {'m':>3s} "
      f"{'β':>8s} {'R(fm)':>10s} {'r(fm)':>8s} {'L(fm)':>10s} {'R/ξ':>6s} "
      f"{'mass':>8s} {'Γ(MeV)':>10s}")
print(f"  {'─'*110}")

results = []
for name, (pm,qm), (pc,qc), m_int, nq, mass_pdg, gamma_pdg, J, ctype in PARTICLES:
    pe, qe = pm*pc, qm*qc
    beta = beta_from_phase(pe, m_int)
    if beta is None:
        continue

    R = beta * xi                    # major radius (m)
    r = R / KAPPA                    # tube radius (m)
    # Carrier path length: for torus knot, L = ∫₀²π √(p²(κ+cosθ)² + q²) dθ
    # Approximate: L ≈ 2πR × √(p_eff² + q_eff²/κ²)
    L = 2 * np.pi * R * np.sqrt(pe**2 + qe**2/KAPPA**2)

    # Inter-core distance at crossing
    if 'Hopf' in ctype:
        # Two rings on torus, crossing separation ~ 2r
        d_cross = 2 * r
    elif ctype == 'trefoil':
        # Trefoil self-crossing: strand separation ~ 2r at each crossing
        d_cross = 2 * r
    else:
        d_cross = None

    R_fm = R * 1e15
    r_fm = r * 1e15
    L_fm = L * 1e15
    R_over_xi = R / xi

    mass_pred = mass_mev(pe, qe, m_int, nq)

    d_str = f"{d_cross*1e15:.2f}" if d_cross else "n/a"
    gamma_str = f"{gamma_pdg:.2e}" if gamma_pdg > 0 else "stable"

    print(f"  {name:>6s} ({pm},{qm}) ({pc},{qc})  ({pe},{qe}){'':<3s} {m_int:3d} "
          f"{beta:8.4f} {R_fm:10.2f} {r_fm:8.2f} {L_fm:10.1f} {R_over_xi:6.3f} "
          f"{mass_pdg:8.1f} {gamma_str:>10s}")

    results.append({
        'name': name, 'mode': (pm,qm), 'carrier': (pc,qc),
        'eff': (pe,qe), 'm': m_int, 'nq': nq,
        'beta': beta, 'R': R, 'r': r, 'L': L,
        'd_cross': d_cross, 'mass_pdg': mass_pdg,
        'gamma_pdg': gamma_pdg, 'J': J, 'ctype': ctype,
    })


# ══════════════════════════════════════════════════════════════════════
# PART 2: LIFETIME / DECAY WIDTH MATRIX ELEMENTS
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 2: LIFETIME MATRIX ELEMENTS")
print("=" * 100)

print(f"""
  Decay rate from topological reconnection:

    Γ = ν_circ × P_reconnect × N_sites × Σ_channels(phase_space)

  Where:
    ν_circ = v_mode / L        circulation frequency (Hz)
    v_mode = c × p_eff/m_int   group velocity on carrier (rough estimate)
    P_reconnect = exp(-d_cross²/ξ²) × (ξ/d_cross)
                                reconnection probability per pass
    N_sites = 2 for Hopf(2), 3 for trefoil, etc.

  The reconnection probability is from superfluid vortex physics:
    When two vortex cores approach within ~ξ, reconnection is certain.
    At distance d > ξ, P ~ exp(-(d/ξ)²) (WKB tunneling through
    the energy barrier between linked and unlinked configurations).

  For a Hopf link with tube radius r = R/π², the crossing separation
  is d ~ 2r = 2R/π². The ratio d/ξ = 2β/π² determines stability:
    - Small β (small torus): d/ξ small → high P → fast decay
    - Large β (large torus): d/ξ large → low P → slow decay / stable
""")

print(f"\n  {'Name':>6s} {'ν_circ(Hz)':>12s} {'d/ξ':>8s} {'P_recon':>12s} "
      f"{'N_sites':>7s} {'Γ_pred(MeV)':>12s} {'Γ_pdg(MeV)':>12s} {'τ_pred(s)':>12s} {'τ_pdg(s)':>12s}")
print(f"  {'─'*110}")

for res in results:
    pe, qe = res['eff']
    beta = res['beta']
    R = res['R']
    r = res['r']
    L = res['L']
    d_cross = res['d_cross']
    mass_pdg = res['mass_pdg']
    gamma_pdg = res['gamma_pdg']
    m_int = res['m']
    nq = res['nq']
    name = res['name']
    ctype = res['ctype']

    if d_cross is None or d_cross <= 0:
        # Unknotted carrier (leptons) — decay is mode transition, not reconnection
        # Use m⁵ scaling from Fermi theory
        if name == 'e⁻':
            print(f"  {name:>6s} {'n/a':>12s} {'n/a':>8s} {'n/a':>12s} "
                  f"{'0':>7s} {'0 (stable)':>12s} {'stable':>12s} {'∞':>12s} {'∞':>12s}")
            continue
        elif name in ('μ⁻', 'τ⁻'):
            # Lepton decay: mode transition on unknot carrier
            # Γ ∝ G_F² m⁵ / (192π³) (standard Fermi formula)
            # Just use known values here
            if gamma_pdg > 0:
                tau_pdg = hbar / (gamma_pdg * MeV_to_J)
            else:
                tau_pdg = np.inf
            print(f"  {name:>6s} {'(mode trans)':>12s} {'n/a':>8s} {'n/a':>12s} "
                  f"{'n/a':>7s} {'(Fermi)':>12s} {gamma_pdg:.2e}{'':>3s} "
                  f"{'(Fermi)':>12s} {tau_pdg:.2e}{'':>3s}")
            continue

    # Circulation frequency
    # Mode group velocity: roughly v ~ c for relativistic modes
    # More precisely: v_mode = c × (p_eff / m_int) for the toroidal component
    v_mode = c * pe / m_int  # fraction of c
    if v_mode > c:
        v_mode = c  # cap at c

    nu_circ = v_mode / L  # Hz

    # Reconnection probability
    d_over_xi = d_cross / xi
    # WKB tunneling: P = (ξ/d) × exp(-(d/ξ - 1)²) for d > ξ
    # For d < ξ: P ≈ 1 (immediate reconnection)
    if d_over_xi < 1:
        P_recon = 1.0
    else:
        P_recon = (1.0 / d_over_xi) * np.exp(-(d_over_xi - 1)**2)

    # Number of reconnection sites
    if 'Hopf' in ctype:
        N_sites = 2  # two crossing points in Hopf link
    elif ctype == 'trefoil':
        N_sites = 3  # three crossings
    else:
        N_sites = 0

    # Phase space factor: number of open decay channels × kinematic factor
    # Rough: Σ channels ~ (mass_pdg / 140)² for hadronic (pion-scale)
    #        For leptonic: Σ ~ 3 (e, μ, τ channels)
    # Use dimensional: phase space ~ (mass_pdg)² / (8π) in natural units
    # This gives Γ in MeV when combined with ℏ
    ps_factor = (mass_pdg**2) / (8 * np.pi * 1000.0)  # rough, in MeV

    # Predicted width
    Gamma_Hz = nu_circ * P_recon * N_sites  # decay rate in Hz
    Gamma_MeV = Gamma_Hz * hbar / MeV_to_J  # convert to MeV

    # Include phase space correction
    # Γ_total = Γ_geometric × (phase space / reference)
    # Normalize to pion: π⁰ has Γ = 7.73 eV, which sets the scale
    # Actually, let's just report the geometric rate and compare

    # Predicted lifetime
    if Gamma_Hz > 0:
        tau_pred = 1.0 / Gamma_Hz
    else:
        tau_pred = np.inf

    # PDG lifetime
    if gamma_pdg > 0:
        tau_pdg = hbar / (gamma_pdg * MeV_to_J)
    else:
        tau_pdg = np.inf

    gamma_pdg_str = f"{gamma_pdg:.2e}" if gamma_pdg > 0 else "stable"
    tau_pdg_str = f"{tau_pdg:.2e}" if np.isfinite(tau_pdg) else "stable"
    tau_pred_str = f"{tau_pred:.2e}" if np.isfinite(tau_pred) else "stable"

    print(f"  {name:>6s} {nu_circ:12.4e} {d_over_xi:8.4f} {P_recon:12.4e} "
          f"{N_sites:7d} {Gamma_MeV:12.4e} {gamma_pdg_str:>12s} "
          f"{tau_pred_str:>12s} {tau_pdg_str:>12s}")


# ══════════════════════════════════════════════════════════════════════
# PART 3: THE d/ξ RATIO — KEY STABILITY PARAMETER
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 3: THE d/ξ RATIO — MASTER STABILITY PARAMETER")
print("=" * 100)

print(f"""
  The ratio d_crossing / ξ determines everything:
    d/ξ << 1: cores overlap, instant reconnection → resonance (Γ ~ mass)
    d/ξ ~ 1:  marginal, WKB tunneling → short-lived (Γ ~ MeV)
    d/ξ >> 1: exponential suppression → long-lived or stable

  For Hopf(2): d = 2r = 2R/π² = 2βξ/π²
    d/ξ = 2β/π²

  For trefoil: d ≈ 2r (same estimate)
    d/ξ = 2β/π²

  So the stability is controlled by β = R/ξ alone!
""")

print(f"  {'Name':>6s} {'β':>8s} {'d/ξ':>8s} {'P_recon':>12s} {'Γ_pdg':>12s} {'Category':>12s}")
print(f"  {'─'*70}")

for res in sorted(results, key=lambda r: r.get('d_cross', 0) or 0):
    beta = res['beta']
    d = res.get('d_cross')
    if d is None:
        category = 'lepton (mode)'
        d_xi = 0
        P = 0
    else:
        d_xi = d / xi
        P = (1.0/d_xi) * np.exp(-(d_xi-1)**2) if d_xi > 1 else 1.0
        if res['gamma_pdg'] > 100:
            category = 'resonance'
        elif res['gamma_pdg'] > 1e-6:
            category = 'short-lived'
        elif res['gamma_pdg'] > 0:
            category = 'long-lived'
        else:
            category = 'stable'

    gamma_str = f"{res['gamma_pdg']:.2e}" if res['gamma_pdg'] > 0 else "stable"
    d_str = f"{d_xi:.4f}" if d else "n/a"
    P_str = f"{P:.4e}" if d else "n/a"

    print(f"  {res['name']:>6s} {beta:8.4f} {d_str:>8s} {P_str:>12s} "
          f"{gamma_str:>12s} {category:>12s}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: CORRELATION — log(Γ) vs d/ξ
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 4: CORRELATION ANALYSIS — log(Γ) vs d/ξ and log(Γ) vs β")
print("=" * 100)

print(f"\n  If Γ ∝ exp(-(d/ξ)²), then log(Γ) should correlate with (d/ξ)²")
print(f"  Since d/ξ = 2β/π², this is equivalent to log(Γ) ∝ -β²\n")

# Collect data points for hadrons (not leptons)
x_vals = []  # d/ξ
y_vals = []  # log10(Γ in MeV)
names = []
for res in results:
    d = res.get('d_cross')
    g = res['gamma_pdg']
    if d is None or d <= 0: continue
    if g <= 0: continue  # skip stable
    x_vals.append(d/xi)
    y_vals.append(np.log10(g))
    names.append(res['name'])

x = np.array(x_vals)
y = np.array(y_vals)

if len(x) > 2:
    # Linear fit: log10(Γ) = a × (d/ξ)² + b
    x2 = x**2
    A = np.vstack([x2, np.ones(len(x2))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Also try log(Γ) vs d/ξ linearly
    A2 = np.vstack([x, np.ones(len(x))]).T
    slope2, intercept2 = np.linalg.lstsq(A2, y, rcond=None)[0]

    # And log(Γ) vs β
    betas = np.array([r['beta'] for r in results if r.get('d_cross') and r['gamma_pdg'] > 0])
    A3 = np.vstack([betas, np.ones(len(betas))]).T
    slope3, intercept3 = np.linalg.lstsq(A3, y, rcond=None)[0]

    # Correlation coefficients
    r_sq_1 = 1 - np.sum((y - (slope*x2 + intercept))**2) / np.sum((y - np.mean(y))**2)
    r_sq_2 = 1 - np.sum((y - (slope2*x + intercept2))**2) / np.sum((y - np.mean(y))**2)
    r_sq_3 = 1 - np.sum((y - (slope3*betas + intercept3))**2) / np.sum((y - np.mean(y))**2)

    print(f"  Model 1: log₁₀(Γ) = {slope:.4f} × (d/ξ)² + {intercept:.2f}  R² = {r_sq_1:.4f}")
    print(f"  Model 2: log₁₀(Γ) = {slope2:.4f} × (d/ξ) + {intercept2:.2f}  R² = {r_sq_2:.4f}")
    print(f"  Model 3: log₁₀(Γ) = {slope3:.4f} × β + {intercept3:.2f}       R² = {r_sq_3:.4f}")

    print(f"\n  {'Name':>6s} {'d/ξ':>8s} {'β':>8s} {'log₁₀Γ_obs':>12s} "
          f"{'log₁₀Γ_pred':>12s} {'Residual':>10s}")
    print(f"  {'─'*65}")
    for i, (xi_val, yi_val, nm) in enumerate(zip(x, y, names)):
        y_pred = slope * xi_val**2 + intercept
        resid = yi_val - y_pred
        print(f"  {nm:>6s} {xi_val:8.4f} {betas[i]:8.4f} {yi_val:12.4f} "
              f"{y_pred:12.4f} {resid:10.4f}")

    # What does the fit predict for the proton (stable)?
    for res in results:
        if res['name'] == 'p':
            d_p = res['d_cross'] / xi
            log_gamma_pred = slope * d_p**2 + intercept
            gamma_pred = 10**log_gamma_pred
            tau_pred = hbar / (gamma_pred * MeV_to_J)
            print(f"\n  PROTON STABILITY PREDICTION:")
            print(f"    d/ξ = {d_p:.4f}, β = {res['beta']:.4f}")
            print(f"    Extrapolated log₁₀(Γ) = {log_gamma_pred:.2f}")
            print(f"    Γ_pred = {gamma_pred:.2e} MeV")
            print(f"    τ_pred = {tau_pred:.2e} s")
            print(f"    τ_pdg  > 10³⁴ years = {1e34*3.15e7:.2e} s")
            break


# ══════════════════════════════════════════════════════════════════════
# PART 5: GEOMETRIC SIZE SUMMARY — PHYSICAL PICTURE
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("PART 5: PHYSICAL SIZE COMPARISON")
print("=" * 100)

print(f"\n  {'Name':>6s} {'R (fm)':>10s} {'r (fm)':>10s} {'d_cross(fm)':>12s} "
      f"{'L (fm)':>10s} {'R(pm)':>10s} {'Type':>12s}")
print(f"  {'─'*80}")

for res in sorted(results, key=lambda r: r['R']):
    R_fm = res['R'] * 1e15
    r_fm = res['r'] * 1e15
    d_fm = res['d_cross'] * 1e15 if res['d_cross'] else 0
    L_fm = res['L'] * 1e15
    R_pm = res['R'] * 1e12
    d_str = f"{d_fm:.2f}" if res['d_cross'] else "n/a"

    print(f"  {res['name']:>6s} {R_fm:10.2f} {r_fm:10.4f} {d_str:>12s} "
          f"{L_fm:10.1f} {R_pm:10.6f} {res['ctype']:>12s}")

print(f"""
  Reference scales:
    ξ = ƛ_C = {xi*1e15:.2f} fm (healing length / reduced Compton wavelength)
    Classical electron radius r_e = {alpha * xi * 1e15:.4f} fm = α × ξ
    Proton charge radius (exp) = 0.841 fm
    Nuclear scale ~ 1 fm
""")


if __name__ == '__main__':
    pass
