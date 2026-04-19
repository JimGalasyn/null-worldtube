#!/usr/bin/env python3
"""
Quick scan: crossing angle of T(2,3) vs torus aspect ratio.

Does any physically-motivated aspect ratio give θ_cross = 60°
(i.e., sin(θ) = √3/2)?
"""
import numpy as np

def crossing_angle_T23(a_torus, R=np.pi**2):
    """Compute the crossing angle between strands at a T(2,3) crossing."""
    N = 2000
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = t[1] - t[0]
    a = a_torus

    x = (R + a*R*np.cos(3*t)) * np.cos(2*t)
    y = (R + a*R*np.cos(3*t)) * np.sin(2*t)
    z = a * R * np.sin(3*t)

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    sp = np.sqrt(dx**2 + dy**2 + dz**2)
    Tx = dx / sp; Ty = dy / sp; Tz = dz / sp

    # Find closest crossing pair
    best_dist = 1e10
    best_ij = None
    for i in range(N):
        for j in range(i + N//6, min(i + 5*N//6, N)):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
            if d < best_dist:
                best_dist = d
                best_ij = (i, j)

    if best_ij is None:
        return None, None

    i, j = best_ij
    T1 = np.array([Tx[i], Ty[i], Tz[i]])
    T2 = np.array([Tx[j], Ty[j], Tz[j]])
    dot = abs(np.dot(T1, T2))
    theta = np.degrees(np.arccos(np.clip(dot, 0, 1)))
    return theta, best_dist


print("Crossing angle of T(2,3) vs torus aspect ratio a = r/R")
print(f"{'a_torus':>10} {'θ_cross':>10} {'sin(θ)':>10} {'d_cross':>10} {'notes':>25}")
print(f"{'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*25}")

for a in [0.05, 0.08, 0.1, 1/np.pi**2, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    theta, dist = crossing_angle_T23(a)
    if theta is not None:
        sin_t = np.sin(np.radians(theta))
        notes = ""
        if abs(a - 1/np.pi**2) < 0.001:
            notes = "← κ = π² (NWT physical)"
        elif abs(a - 0.3) < 0.001:
            notes = "← holonomy_v3 default"
        elif abs(theta - 60) < 1:
            notes = f"← θ ≈ 60°!"
        print(f"  {a:8.4f} {theta:9.2f}° {sin_t:10.6f} {dist:10.3f} {notes:>25}")
