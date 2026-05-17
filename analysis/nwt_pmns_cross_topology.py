#!/usr/bin/env python3
"""
NWT PMNS Matrix: Cross-Topology Overlap

The neutrino is born on the parent carrier (trefoil) but propagates on
the vacuum condensate (unknot / electron carrier).  The PMNS matrix is
the overlap between eigenmodes of these two different topologies.

Physical picture:
  1. Baryon (trefoil T(2,3)) undergoes weak decay at a crossing
  2. R3 surgery creates: charged lepton + neutrino phase kink
  3. The neutrino's angular profile at the crossing is set by the
     TREFOIL eigenstate (flavor basis)
  4. The neutrino then propagates freely on the VACUUM condensate,
     decomposing into vacuum eigenmodes (mass basis)
  5. PMNS = overlap between trefoil eigenmodes and vacuum eigenmodes
     evaluated at the crossing point

The mixing arises because:
  - The trefoil has non-uniform curvature along its path
  - At each crossing, the local metric distorts the eigenmode profiles
  - Different n_s modes sample the crossing differently
  - The phase relationship between n_θ=+1 and n_θ=-1 components
    depends on the local curvature → differs between trefoil and vacuum
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time

KAPPA = np.pi**2
BETA_E = np.sqrt(5.0 / 4.0)


class TorusKnot:
    def __init__(self, pc, qc, kappa=KAPPA, beta=BETA_E):
        self.pc, self.qc = pc, qc
        self.kappa, self.beta = kappa, beta
        self.R, self.r = beta, beta / kappa
        self._build_arc_table()

    def _build_arc_table(self, N=2000):
        t = np.linspace(0, 2 * np.pi, N + 1)
        ds = np.array([self._ds_dt(ti) for ti in t])
        self._arc_cum = np.cumsum(0.5 * (ds[:-1] + ds[1:]) * np.diff(t))
        self._arc_cum = np.insert(self._arc_cum, 0, 0.0)
        self.L_total = self._arc_cum[-1]
        self._t_table = t

    def _ds_dt(self, t):
        theta = self.qc * t
        return np.sqrt(
            self.pc**2 * (self.kappa + np.cos(theta))**2 + self.qc**2
        ) * self.r

    def t_from_s(self, s_norm):
        return np.interp(s_norm * self.L_total, self._arc_cum, self._t_table)

    def curvature_normal(self, t):
        c, s = np.cos(self.qc * t), np.sin(self.qc * t)
        kc = self.kappa + c
        num2 = (self.pc**4 * kc**2 + self.qc**4
                + 2 * self.pc**2 * self.qc**2 * c * kc
                + self.pc**2 * self.qc**2 * s**2)
        denom = self.pc**2 * kc**2 + self.qc**2
        return np.sqrt(max(num2, 0)) / max(denom**1.5, 1e-30) / self.r

    def crossing_s_values(self):
        """Normalized arc-length positions of self-crossings."""
        pc, qc = self.pc, self.qc
        if pc == 1 or qc == 1:
            return []
        if pc == qc:
            t_vals = [2 * np.pi * k / pc for k in range(pc)]
        else:
            n_cross = min(pc * (qc - 1), qc * (pc - 1))
            t_vals = [2 * np.pi * k / n_cross for k in range(n_cross)]
        return [np.interp(t, self._t_table, self._arc_cum / self.L_total)
                for t in t_vals]


class KnotSolver:
    def __init__(self, knot, Ns=256, Nth=64):
        self.knot, self.Ns, self.Nth = knot, Ns, Nth
        self.ds = 1.0 / Ns
        self.dth = 2 * np.pi / Nth
        self.rL = knot.r / knot.L_total
        self._precompute()

    def _precompute(self):
        Ns, Nth, r = self.Ns, self.Nth, self.knot.r
        self.t_grid = np.array([
            self.knot.t_from_s(s)
            for s in np.linspace(0, 1, Ns, endpoint=False)
        ])
        self.th_grid = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
        self.kappa_n = np.array([self.knot.curvature_normal(t) for t in self.t_grid])
        self.h_s_norm = np.zeros((Ns, Nth))
        for i in range(Ns):
            for j in range(Nth):
                v = r * self.kappa_n[i] * np.cos(self.th_grid[j])
                self.h_s_norm[i, j] = max(1.0 - v, 0.05)

    def build_hamiltonian(self):
        Ns, Nth = self.Ns, self.Nth
        N = Ns * Nth
        ds, dth, rL2 = self.ds, self.dth, self.rL**2
        rows, cols, vals = [], [], []
        def idx(i, j): return (i % Ns) * Nth + (j % Nth)
        for i in range(Ns):
            for j in range(Nth):
                n = idx(i, j)
                hs2_p = 0.5 * (self.h_s_norm[i, j]**2 + self.h_s_norm[(i+1)%Ns, j]**2)
                hs2_m = 0.5 * (self.h_s_norm[i, j]**2 + self.h_s_norm[(i-1)%Ns, j]**2)
                coeff_s = rL2 / ds**2
                coeff_th = 1.0 / dth**2
                rk = self.knot.r * self.kappa_n[i]
                total = coeff_s * (1/hs2_p + 1/hs2_m) + 2*coeff_th - rk**2/4
                rows.append(n); cols.append(n); vals.append(total)
                rows.append(n); cols.append(idx(i+1, j)); vals.append(-coeff_s/hs2_p)
                rows.append(n); cols.append(idx(i-1, j)); vals.append(-coeff_s/hs2_m)
                rows.append(n); cols.append(idx(i, j+1)); vals.append(-coeff_th)
                rows.append(n); cols.append(idx(i, j-1)); vals.append(-coeff_th)
        return csr_matrix((vals, (rows, cols)), shape=(N, N))

    def solve(self, n_modes=80):
        print(f"    Solving {self.Ns}×{self.Nth} = {self.Ns*self.Nth} DOF...", end='')
        t0 = time.time()
        H = self.build_hamiltonian()
        ev, evec = eigsh(H, k=n_modes, sigma=-0.1, which='LM',
                         maxiter=10000, tol=1e-8)
        order = np.argsort(ev)
        print(f" {time.time()-t0:.1f}s")
        return ev[order], evec[:, order]

    def decompose(self, eigvec):
        psi = eigvec.reshape(self.Ns, self.Nth)
        psi_fft = np.fft.fft2(psi)
        power = np.abs(psi_fft)**2
        power[0, 0] = 0
        idx = np.argmax(power)
        i, j = np.unravel_index(idx, power.shape)
        ns = i if i <= self.Ns//2 else i - self.Ns
        nth = j if j <= self.Nth//2 else j - self.Nth
        purity = power[i, j] / np.sum(power)
        return ns, nth, purity

    def get_theta_profile_at_s(self, eigvec, s_idx):
        """Extract the θ-profile at a specific s-grid point."""
        psi = eigvec.reshape(self.Ns, self.Nth)
        return psi[s_idx, :].copy()

    def get_theta_profile_at_s_interp(self, eigvec, s_norm):
        """Extract θ-profile at arbitrary s (linear interpolation)."""
        psi = eigvec.reshape(self.Ns, self.Nth)
        s_idx_float = s_norm * self.Ns
        s_lo = int(s_idx_float) % self.Ns
        s_hi = (s_lo + 1) % self.Ns
        frac = s_idx_float - int(s_idx_float)
        return (1 - frac) * psi[s_lo, :] + frac * psi[s_hi, :]


def find_nth1_modes(solver, eigenvalues, eigenvectors, max_modes=6):
    """Find eigenmodes with n_θ = +1, one per distinct n_s."""
    modes = []
    seen_ns = set()
    for n in range(len(eigenvalues)):
        ns, nth, pur = solver.decompose(eigenvectors[:, n])
        if nth == 1 and ns not in seen_ns:
            modes.append({'idx': n, 'ns': ns, 'nth': nth,
                         'lam': eigenvalues[n], 'pur': pur})
            seen_ns.add(ns)
        if len(modes) >= max_modes:
            break
    return modes


def main():
    print("=" * 100)
    print("NWT PMNS MATRIX: CROSS-TOPOLOGY OVERLAP")
    print("=" * 100)

    # ── Solve on both topologies ────────────────────────────────────
    print("\n  TREFOIL T(2,3) — parent carrier (flavor basis):")
    knot_tref = TorusKnot(2, 3)
    solver_tref = KnotSolver(knot_tref, Ns=192, Nth=32)
    ev_tref, evec_tref = solver_tref.solve(n_modes=120)

    print("\n  ELECTRON (2,1) — vacuum carrier (mass basis):")
    knot_elec = TorusKnot(2, 1)
    solver_elec = KnotSolver(knot_elec, Ns=256, Nth=32)
    ev_elec, evec_elec = solver_elec.solve(n_modes=80)

    # Find n_θ=1 modes on each
    modes_tref = find_nth1_modes(solver_tref, ev_tref, evec_tref)
    modes_elec = find_nth1_modes(solver_elec, ev_elec, evec_elec)

    print(f"\n  Trefoil n_θ=1 modes (FLAVOR basis):")
    for rank, m in enumerate(modes_tref[:6]):
        label = " ← ν_" + ['e', 'μ', 'τ'][rank] if rank < 3 else ""
        print(f"    #{rank}: mode {m['idx']}, n_s={m['ns']:+d}, "
              f"λ={m['lam']:.4f}, pur={m['pur']:.3f}{label}")

    print(f"\n  Electron (2,1) n_θ=1 modes (MASS basis):")
    for rank, m in enumerate(modes_elec[:6]):
        label = " ← ν_" + str(rank+1) if rank < 3 else ""
        print(f"    #{rank}: mode {m['idx']}, n_s={m['ns']:+d}, "
              f"λ={m['lam']:.4f}, pur={m['pur']:.3f}{label}")

    # Mass eigenvalue splittings
    if len(modes_elec) >= 3:
        lam = [modes_elec[i]['lam'] for i in range(3)]
        print(f"\n  Mass eigenvalues: λ₁={lam[0]:.6f}, λ₂={lam[1]:.6f}, λ₃={lam[2]:.6f}")
        dl21 = lam[1] - lam[0]
        dl32 = lam[2] - lam[1]
        print(f"  Δλ₂₁ = {dl21:.6f}")
        print(f"  Δλ₃₂ = {dl32:.6f}")
        if abs(dl21) > 1e-12:
            print(f"  Δλ₃₂/Δλ₂₁ = {dl32/dl21:.4f}  (exp: ~33)")

    # ── Cross-topology overlap at crossing points ───────────────────
    print("\n\n" + "━" * 100)
    print("  CROSS-TOPOLOGY OVERLAP AT TREFOIL CROSSINGS")
    print("━" * 100)

    crossings = knot_tref.crossing_s_values()
    print(f"\n  Trefoil has {len(crossings)} crossings at s̃ = "
          f"{[f'{c:.4f}' for c in crossings]}")

    n_flavor = min(3, len(modes_tref))
    n_mass = min(3, len(modes_elec))

    # For each crossing, compute the overlap matrix between
    # trefoil n_θ=1 modes and vacuum n_θ=1 modes in θ-space.
    #
    # The θ-profile at the crossing encodes the local curvature
    # effects.  The FULL complex overlap (not just |c|²) captures
    # the phase mismatch.

    # Method 1: Profile overlap at each crossing
    print(f"\n  Method 1: θ-profile overlap at each crossing")
    for ci, s_cross in enumerate(crossings):
        print(f"\n    Crossing {ci} at s̃ = {s_cross:.4f}:")

        # Get θ-profiles of trefoil modes at this crossing
        flavor_profiles = []
        for rank in range(n_flavor):
            prof = solver_tref.get_theta_profile_at_s_interp(
                evec_tref[:, modes_tref[rank]['idx']], s_cross)
            norm = np.sqrt(np.sum(np.abs(prof)**2) * solver_tref.dth)
            if norm > 1e-30:
                prof /= norm
            flavor_profiles.append(prof)

        # Get θ-profiles of vacuum modes (averaged over s, since
        # the vacuum is approximately uniform)
        mass_profiles = []
        for rank in range(n_mass):
            # Average over several s-points on the vacuum
            avg_prof = np.zeros(solver_elec.Nth, dtype=complex)
            for s_idx in range(0, solver_elec.Ns, 8):
                prof = solver_elec.get_theta_profile_at_s(
                    evec_elec[:, modes_elec[rank]['idx']], s_idx)
                avg_prof += prof
            avg_prof /= (solver_elec.Ns // 8)
            norm = np.sqrt(np.sum(np.abs(avg_prof)**2) * solver_elec.dth)
            if norm > 1e-30:
                avg_prof /= norm
            mass_profiles.append(avg_prof)

        # Compute overlap matrix (both Nth=32, same θ-grid)
        U = np.zeros((n_flavor, n_mass), dtype=complex)
        for alpha in range(n_flavor):
            for i in range(n_mass):
                U[alpha, i] = np.sum(
                    mass_profiles[i].conj() * flavor_profiles[alpha]
                ) * solver_tref.dth

        U_sq = np.abs(U)**2
        # Normalize rows
        for alpha in range(n_flavor):
            rs = np.sum(U_sq[alpha])
            if rs > 1e-30:
                U_sq[alpha] /= rs

        labels_f = ['ν_e', 'ν_μ', 'ν_τ']
        labels_m = ['ν₁', 'ν₂', 'ν₃']
        print(f"    |U|² (crossing {ci}):")
        print(f"    {'':>6s}", end='')
        for lm in labels_m[:n_mass]:
            print(f" {lm:>9s}", end='')
        print()
        for alpha in range(n_flavor):
            print(f"    {labels_f[alpha]:>6s}", end='')
            for i in range(n_mass):
                print(f" {U_sq[alpha, i]:9.6f}", end='')
            print()

    # Method 2: Full 2D Fourier overlap
    # Decompose both topologies' eigenstates into (n_s, n_θ) Fourier
    # components and compute overlap in the common Fourier basis.
    print(f"\n\n  Method 2: Full 2D Fourier overlap")

    N_fourier_s = min(solver_tref.Ns, solver_elec.Ns) // 2
    N_fourier_th = min(solver_tref.Nth, solver_elec.Nth) // 2

    def fourier_decomp_2d(eigvec, Ns, Nth):
        psi = eigvec.reshape(Ns, Nth)
        fft = np.fft.fft2(psi)
        # Normalize
        norm = np.sqrt(np.sum(np.abs(fft)**2))
        if norm > 1e-30:
            fft /= norm
        return fft

    # Build common Fourier basis vectors
    # Truncate to common size and compare
    def truncate_fft(fft, Ns_orig, Nth_orig, Ns_trunc, Nth_trunc):
        """Extract low-frequency components into common-size array."""
        result = np.zeros((2*Ns_trunc+1, 2*Nth_trunc+1), dtype=complex)
        for i in range(-Ns_trunc, Ns_trunc+1):
            for j in range(-Nth_trunc, Nth_trunc+1):
                i_orig = i % Ns_orig
                j_orig = j % Nth_orig
                result[i + Ns_trunc, j + Nth_trunc] = fft[i_orig, j_orig]
        return result

    U_fourier = np.zeros((n_flavor, n_mass), dtype=complex)
    for alpha in range(n_flavor):
        fft_f = fourier_decomp_2d(evec_tref[:, modes_tref[alpha]['idx']],
                                   solver_tref.Ns, solver_tref.Nth)
        fft_f_trunc = truncate_fft(fft_f, solver_tref.Ns, solver_tref.Nth,
                                    N_fourier_s, N_fourier_th)
        for i in range(n_mass):
            fft_m = fourier_decomp_2d(evec_elec[:, modes_elec[i]['idx']],
                                       solver_elec.Ns, solver_elec.Nth)
            fft_m_trunc = truncate_fft(fft_m, solver_elec.Ns, solver_elec.Nth,
                                        N_fourier_s, N_fourier_th)
            U_fourier[alpha, i] = np.sum(fft_m_trunc.conj() * fft_f_trunc)

    U_sq_f = np.abs(U_fourier)**2
    for alpha in range(n_flavor):
        rs = np.sum(U_sq_f[alpha])
        if rs > 1e-30:
            U_sq_f[alpha] /= rs

    print(f"\n    |U|² (2D Fourier):")
    print(f"    {'':>6s}", end='')
    for lm in labels_m[:n_mass]:
        print(f" {lm:>9s}", end='')
    print()
    for alpha in range(n_flavor):
        print(f"    {labels_f[alpha]:>6s}", end='')
        for i in range(n_mass):
            print(f" {U_sq_f[alpha, i]:9.6f}", end='')
        print()

    # Method 3: Crossing-averaged with R3 weight
    # Average the crossing overlaps, weighted by the R3 operator
    # (cos θ × tunneling × 1/κ)
    print(f"\n\n  Method 3: R3-weighted crossing overlap")

    U_R3 = np.zeros((n_flavor, n_mass), dtype=complex)
    tunnel = np.exp(-2 * knot_tref.r)  # exp(-2r/ξ)

    for ci, s_cross in enumerate(crossings):
        # Trefoil flavor profiles at crossing
        for alpha in range(n_flavor):
            prof_f = solver_tref.get_theta_profile_at_s_interp(
                evec_tref[:, modes_tref[alpha]['idx']], s_cross)
            for i in range(n_mass):
                # Vacuum mass profile (s-averaged)
                avg_prof = np.zeros(solver_elec.Nth, dtype=complex)
                for s_idx in range(0, solver_elec.Ns, 8):
                    avg_prof += solver_elec.get_theta_profile_at_s(
                        evec_elec[:, modes_elec[i]['idx']], s_idx)
                avg_prof /= (solver_elec.Ns // 8)

                # R3-weighted overlap: include cos(θ) and tunneling
                cos_th = np.cos(solver_tref.th_grid)
                overlap = np.sum(
                    avg_prof.conj() * cos_th * prof_f
                ) * solver_tref.dth * tunnel / KAPPA

                U_R3[alpha, i] += overlap / len(crossings)

    U_sq_R3 = np.abs(U_R3)**2
    for alpha in range(n_flavor):
        rs = np.sum(U_sq_R3[alpha])
        if rs > 1e-30:
            U_sq_R3[alpha] /= rs

    print(f"\n    |U|² (R3-weighted):")
    print(f"    {'':>6s}", end='')
    for lm in labels_m[:n_mass]:
        print(f" {lm:>9s}", end='')
    print()
    for alpha in range(n_flavor):
        print(f"    {labels_f[alpha]:>6s}", end='')
        for i in range(n_mass):
            print(f" {U_sq_R3[alpha, i]:9.6f}", end='')
        print()

    # ── Extract mixing angles from best method ──────────────────────
    print("\n\n" + "━" * 100)
    print("  MIXING ANGLE EXTRACTION")
    print("━" * 100)

    for label, U_sq_mat in [("θ-profile at crossing 0", None),
                             ("2D Fourier", U_sq_f),
                             ("R3-weighted", U_sq_R3)]:
        if label == "θ-profile at crossing 0":
            # Recompute for crossing 0
            s_cross = crossings[0] if crossings else 0.0
            flavor_profiles = []
            for rank in range(n_flavor):
                prof = solver_tref.get_theta_profile_at_s_interp(
                    evec_tref[:, modes_tref[rank]['idx']], s_cross)
                norm = np.sqrt(np.sum(np.abs(prof)**2) * solver_tref.dth)
                if norm > 1e-30: prof /= norm
                flavor_profiles.append(prof)
            mass_profiles = []
            for rank in range(n_mass):
                avg_prof = np.zeros(solver_elec.Nth, dtype=complex)
                for s_idx in range(0, solver_elec.Ns, 8):
                    avg_prof += solver_elec.get_theta_profile_at_s(
                        evec_elec[:, modes_elec[rank]['idx']], s_idx)
                avg_prof /= (solver_elec.Ns // 8)
                norm = np.sqrt(np.sum(np.abs(avg_prof)**2) * solver_elec.dth)
                if norm > 1e-30: avg_prof /= norm
                mass_profiles.append(avg_prof)
            U_tmp = np.zeros((n_flavor, n_mass), dtype=complex)
            for alpha in range(n_flavor):
                for i in range(n_mass):
                    U_tmp[alpha, i] = np.sum(
                        mass_profiles[i].conj() * flavor_profiles[alpha]
                    ) * solver_tref.dth
            U_sq_mat = np.abs(U_tmp)**2
            for alpha in range(n_flavor):
                rs = np.sum(U_sq_mat[alpha])
                if rs > 1e-30: U_sq_mat[alpha] /= rs

        if n_flavor < 3 or n_mass < 3:
            continue

        sin2_13 = np.clip(U_sq_mat[0, 2], 0, 1)
        cos2_13 = 1.0 - sin2_13
        if cos2_13 > 1e-10:
            sin2_23 = np.clip(U_sq_mat[1, 2] / cos2_13, 0, 1)
            sin2_12 = np.clip(U_sq_mat[0, 1] / cos2_13, 0, 1)
        else:
            sin2_23 = sin2_12 = 0.5

        th12 = np.arcsin(np.sqrt(sin2_12)) * 180 / np.pi
        th23 = np.arcsin(np.sqrt(sin2_23)) * 180 / np.pi
        th13 = np.arcsin(np.sqrt(sin2_13)) * 180 / np.pi

        print(f"\n  {label}:")
        print(f"    θ₁₂ = {th12:7.2f}°  (exp: 33.41°)  err: {abs(th12-33.41)/33.41*100:5.1f}%")
        print(f"    θ₂₃ = {th23:7.2f}°  (exp: 49.0°)   err: {abs(th23-49.0)/49.0*100:5.1f}%")
        print(f"    θ₁₃ = {th13:7.2f}°  (exp:  8.54°)  err: {abs(th13-8.54)/8.54*100:5.1f}%")

    # ── Diagnostic: curvature at crossings ──────────────────────────
    print("\n\n" + "━" * 100)
    print("  DIAGNOSTIC: Curvature and metric at crossings")
    print("━" * 100)

    for ci, s_cross in enumerate(crossings):
        t_cross = knot_tref.t_from_s(s_cross)
        kn = knot_tref.curvature_normal(t_cross)
        s_idx = int(s_cross * solver_tref.Ns) % solver_tref.Ns
        h_vals = solver_tref.h_s_norm[s_idx, :]
        print(f"\n    Crossing {ci}: s̃={s_cross:.4f}, t={t_cross:.4f}")
        print(f"    κ_normal = {kn:.4f}")
        print(f"    h_s range: [{h_vals.min():.4f}, {h_vals.max():.4f}]")
        print(f"    h_s asymmetry: {(h_vals.max()-h_vals.min())/(h_vals.max()+h_vals.min()):.4f}")

        # Show how each mode's amplitude varies at this crossing
        print(f"    Mode amplitudes at crossing:")
        for rank in range(min(3, n_flavor)):
            psi = evec_tref[:, modes_tref[rank]['idx']].reshape(
                solver_tref.Ns, solver_tref.Nth)
            amp_at_cross = np.abs(psi[s_idx, :])
            phase_at_cross = np.angle(psi[s_idx, :])
            print(f"      Flavor {rank} (n_s={modes_tref[rank]['ns']}): "
                  f"|ψ|_max={amp_at_cross.max():.6f}, "
                  f"|ψ|_min={amp_at_cross.min():.6f}, "
                  f"phase range=[{phase_at_cross.min():.3f}, {phase_at_cross.max():.3f}]")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"""
  Cross-topology overlap between trefoil T(2,3) and electron (2,1) carriers.

  The PMNS mixing matrix arises from the geometric mismatch between:
    - Eigenmodes on the trefoil (3 crossings, non-uniform curvature)
    - Eigenmodes on the electron carrier (no crossings, different curvature)

  Three methods tested:
    1. θ-profile overlap at specific crossing points
    2. Full 2D Fourier overlap (topology-independent basis)
    3. R3-weighted overlap (includes cos(θ) angular coupling + tunneling)

  The R3-weighted method is physically the most correct: it weights the
  overlap by the actual weak interaction operator at the crossing.

  KEY: if all methods give near-identity, the mixing comes from somewhere
  else — possibly from the fact that the three lepton MASSES define different
  effective κ values (different aspect ratios for e, μ, τ), and the mode
  structure differs across these scales.
""")


if __name__ == '__main__':
    main()
