"""
Full BPS Nielsen-Olesen vortex: scalar f(rho) and gauge a(rho).

BPS first-order equations (at lambda = e^2, our units):
    f'(rho) = (n - a) f / rho
    a'(rho) = rho (1 - f^2) / 2

BCs: f(0)=0, f(inf)=1; a(0)=0, a(inf)=n.

Single shooting parameter c1 = f'(0+).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import k0


def bps_rhs(rho, y, n=1):
    f, a = y
    return [(n - a) * f / rho, rho * (1 - f**2) / 2.0]


def shoot_bps(c1, rho_start=1e-4, rho_end=15.0, n=1):
    y0 = [c1 * rho_start, rho_start**2 / 4.0]
    sol = solve_ivp(bps_rhs, [rho_start, rho_end], y0,
                    method='RK45', rtol=1e-11, atol=1e-13, max_step=0.02)
    if not sol.success:
        return 1e10
    return sol.y[0, -1] - 1.0


def find_c1_bps(rho_end=15.0):
    c1_lo, c1_hi = 0.5, 1.2
    r_lo = shoot_bps(c1_lo, rho_end=rho_end)
    r_hi = shoot_bps(c1_hi, rho_end=rho_end)
    print(f"  shoot(c1={c1_lo}) = {r_lo:+.4e}")
    print(f"  shoot(c1={c1_hi}) = {r_hi:+.4e}")
    if r_lo * r_hi > 0:
        print("Scanning...")
        for c1 in np.linspace(0.3, 2.5, 30):
            r = shoot_bps(c1, rho_end=rho_end)
            print(f"    c1={c1:.3f}  residual={r:+.4e}")
        raise RuntimeError("Bracket fails")
    return brentq(shoot_bps, c1_lo, c1_hi, args=(1e-4, rho_end, 1),
                  xtol=1e-10, rtol=1e-10)


def build_bps_profile(c1, rho_max=15.0, n_points=3000):
    rho_start = 1e-4
    y0 = [c1 * rho_start, rho_start**2 / 4.0]
    rho_grid = np.linspace(rho_start, rho_max, n_points)
    sol = solve_ivp(bps_rhs, [rho_start, rho_max], y0, t_eval=rho_grid,
                    method='RK45', rtol=1e-12, atol=1e-14, max_step=0.01)
    return sol.t, sol.y[0], sol.y[1]


def validate_bps(rho, f, a):
    print("=" * 60)
    print("BPS Profile Validation")
    print("=" * 60)

    mask = rho < 0.05
    c1_fit = np.polyfit(rho[mask], f[mask], 1)[0]
    print(f"Scalar small-rho slope c1 = {c1_fit:.8f}")
    print(f"  Literature (n=1 BPS): ~ 0.853 (de Vega-Schaposnik)")

    mask_a = (rho > 0.01) & (rho < 0.1)
    c_a_fit = np.polyfit(rho[mask_a]**2, a[mask_a], 1)[0]
    print(f"Gauge small-rho coeff c_a = {c_a_fit:.8f}  (expected 0.25)")

    mask = (rho > 4.0) & (rho < 10.0) & (f < 0.9999) & (f > 0)
    if mask.sum() > 10:
        y = np.log(1 - f[mask]) + 0.5 * np.log(rho[mask])
        slope = np.polyfit(rho[mask], y, 1)[0]
        print(f"Scalar large-rho decay rate = {-slope:.6f}")
        print(f"  Expected: sqrt(2) = {np.sqrt(2):.6f}")
        predictor = k0(np.sqrt(2) * rho[mask])
        A = np.sum((1 - f[mask]) * predictor) / np.sum(predictor**2)
        print(f"  Fit: 1-f ~ {A:.4f} * K0(sqrt(2) rho)")
        print(f"  Literature: A ~ 1.71")

    print(f"\nBoundary values:")
    print(f"  f(rho_max) = {f[-1]:.6f}  (expect 1)")
    print(f"  a(rho_max) = {a[-1]:.6f}  (expect 1)")
    print()


if __name__ == "__main__":
    print("Shooting for c1 in full BPS system...")
    c1 = find_c1_bps(rho_end=15.0)
    print(f"\nConverged c1 = {c1:.10f}\n")

    rho, f, a = build_bps_profile(c1, rho_max=15.0, n_points=3000)
    validate_bps(rho, f, a)

    print("Profile values:")
    print(f"{'rho':>6} | {'f':>10} | {'a':>10}")
    for r in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        print(f"{r:6.2f} | {np.interp(r, rho, f):10.6f} | {np.interp(r, rho, a):10.6f}")

    np.savez("bps_profile.npz", rho=rho, f=f, a=a, c1=c1)
    print("\nSaved to bps_profile.npz")
