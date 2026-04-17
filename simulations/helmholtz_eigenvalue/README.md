# NWT Helmholtz Eigenvalue Calculation

Numerical framework for computing κ_SM from the BPS Helmholtz eigenvalue
problem, supporting the claims of NWT Paper 13.

## Status

**Step 1 complete (BPS profile solver):** `bps_profile.py`

- Shooting method for the first-order BPS equations (scalar f, gauge a)
- Converges to c₁ = f'(0) = 0.6032878538 for n=1
- BPS equations satisfied to RMS residual ~ 1e-7 in the interior
- Gauge small-ρ coefficient c_a = 0.2496 vs expected 0.25 (0.15% error)
- f → 0.9999... and a → 1.0000... at ρ = 15 (boundary conditions met)
- Saved to `bps_profile.npz` (fields: rho, f, a, c1)

## Convention notes (important for future work)

The BPS equations solved are:
    f'(ρ) = (1 - a(ρ)) f(ρ) / ρ
    a'(ρ) = ρ (1 - f²) / 2

with n = 1 (unit winding). This is the standard "Vachaspati textbook" form
at λ = e² = 1, v = 1.

**Unresolved:** the total BPS energy per unit length comes to 5.63 in these
units, vs naive expectations of π ≈ 3.14 or 2π ≈ 6.28. The four energy
components (scalar kinetic, scalar winding, gauge, potential) come out as
(1.84, 1.84, 0.65, 1.30) — with the expected BPS equipartitions
kinetic = winding and potential = 2 × gauge, but NOT all four equal.

This is almost certainly a Lagrangian normalization convention (factor of
½ in the gauge kinetic term) rather than a bug in the solver — the BPS
equations themselves are satisfied to 1e-7. Worth pinning down against
a specific reference (Taubes 1980 or de Vega–Schaposnik 1976) before
quoting c₁ = 0.6033 in any publication.

**For the eigenvalue problem this doesn't matter:** only the shape of f(ρ)
enters V″(f) = 3f² − 1 in the Helmholtz operator. The overall energy
normalization drops out of the eigenvalue equation.

## What's next (Step 2: straight-tube eigenvalue)

The radial Helmholtz operator for scalar perturbations δψ(ρ) around the
BPS background f₀(ρ), at fixed azimuthal mode m:

    [ -d²/dρ² - (1/ρ) d/dρ + m²/ρ² + 3f₀² - 1 ] ψ_m(ρ) = κ² ψ_m(ρ)

BCs: ψ_m(0) = 0 (regular for m ≥ 1; or ψ'_m(0) = 0 for m = 0),
     ψ_m(ρ → ∞) → 0 (bound mode) or plane-wave (scattering).

**Implementation plan:**

1. Chebyshev-Gauss-Lobatto grid on [0, ρ_max] with ρ_max = 10. Handle the
   ρ = 0 singularity by using the basis ψ_m(ρ) = ρ^|m| × g(ρ²) with g
   expanded in Chebyshev polynomials.

2. Build the discrete operator as a dense matrix. For N = 128 grid points,
   the matrix is 128 × 128 — fits trivially in memory, full diagonalization
   takes < 1 second.

3. Use `scipy.linalg.eigh` for the lowest few eigenvalues. No need for
   iterative solvers at this size.

4. **Literature benchmarks:** The Nielsen-Olesen stability spectrum is
   well-studied. The lowest m=0 eigenvalue should be a real number around
   ~1-2 in these units (the "breathing mode" of the vortex). The BPS point
   is marginal — perturbations are massless in some channels. Cross-check
   against Bogomolny-Vainshtein, Weinberg 1979, and Goodband-Hindmarsh.

5. If the m=0 mode gives κ ~ π (rather than 9.844 = π²), that's evidence
   the trefoil-geodesic calculation you want lives in a different mode
   (higher m, or a longitudinal mode) and the π² identification needs to
   emerge from the knot geometry (Step 3), not the straight tube.

## What's after that (Step 3: trefoil geometry)

Once Step 2 is validated, the trefoil calculation adds:

1. Build trefoil centerline r₀(s) with unit-speed parametrization
2. Compute curvature κ(s), torsion τ(s) around the knot (analytic for T(2,3))
3. Parallel-transport (Bishop) frame (N₁(s), N₂(s))
4. Tube coordinates (s, ρ, φ) with h_s = 1 - ρ(k₁ cos φ + k₂ sin φ)
5. 2D eigenvalue problem in (s, ρ) at fixed m=0, periodic in s
6. Grid: 256 (FFT) × 128 (Chebyshev) = 32K DOF — still fits on a 4090
7. LOBPCG or shift-invert Arnoldi for lowest eigenvalue

**The test:** does κ_SM(m=0) on the trefoil at a/R = 1/π² come out to 9.844?

If yes, Paper 13's identification of κ_SM as a Helmholtz eigenvalue is
numerically confirmed. If it comes out as something else (e.g., π instead
of π²), the definition needs refinement.

## Files

- `bps_profile.py` — BPS profile solver (DONE, validated)
- `bps_profile.npz` — saved profile on [1e-4, 15] with 3000 points
- `README.md` — this file

## Dependencies

numpy, scipy. Nothing GPU-specific yet; Step 3 is where JAX becomes useful.

## One thing to verify before Step 2

Pin down the BPS convention. Specifically: what does c₁ = 0.6033 correspond
to in the published literature? Checking two references:

- **Vachaspati, *Kinks and Domain Walls*** (CUP 2006), §4.2: gives the BPS
  vortex equations in our form; quotes c₁ numerically but I don't have
  the book handy.
- **Taubes, *Comm. Math. Phys.* 72, 277 (1980):** proves existence and
  uniqueness; establishes that at BPS, the profile is determined by n alone.

If the conventional value in a standard reference matches 0.6033, we're
good. If not, we may have a factor-of-√2 rescaling of ρ relative to that
reference — which would shift the eigenvalue by the same factor squared.
This matters for comparing a numerical κ_SM to the claimed 9.844.
