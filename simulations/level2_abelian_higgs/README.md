# Level 2: Full 3D Abelian Higgs Vortex Relaxation

## Goal

Initialize a knotted vortex in a 3D abelian Higgs condensate, relax
to the energy minimum, and measure the total energy.  The energy IS
the particle mass (E = mc²).  The integers from the Casimir framework
should emerge as OUTPUT of the relaxation, not INPUT.

## The Physics

### Abelian Higgs energy functional (static, 3D)

```
E[ψ, A] = ∫ d³x [ |Dψ|² + ½ B² + (λ/4)(|ψ|² - v²)² ]
```

where:
- ψ(x,y,z): complex scalar field (the condensate)
- A(x,y,z): U(1) gauge vector potential
- D = ∇ - ieA: covariant derivative
- B = ∇ × A: magnetic field
- λ = e²: BPS point (scalar mass = gauge mass)
- v = 1: vacuum expectation value

### What a vortex is

A vortex is a line defect where |ψ| → 0 and the phase of ψ winds by
2πn around the line.  For a knotted vortex, this line traces a torus
knot T(p,q) in 3D space.  The BPS profile f₀(ρ) from Step 1 gives
the radial shape of |ψ| near the vortex core.

### What relaxation produces

Starting from an initial T(p,q) vortex ansatz, gradient flow on E
finds the energy-minimizing field configuration in that topological
sector.  The topology is conserved (vortex lines can't disappear
without annihilating with anti-vortices), so the relaxed energy is
the mass of the lightest particle with that topology.

## Architecture

```
level2_abelian_higgs/
├── README.md              # This file
├── config.py              # Grid size, coupling constants, knot params
├── fields.py              # Field arrays (ψ, A), boundary conditions
├── energy.py              # E[ψ, A] via JAX (autodiff-ready)
├── initialize.py          # Vortex knot stamping onto grid
├── relax.py               # Gradient flow with gauge fixing
├── measure.py             # Energy, topological charge, winding number
├── run.py                 # Main: (p,q) → E_mass
└── results/               # Saved energies + field snapshots
```

## Fields and Grid

### Grid
- 3D uniform Cartesian: N × N × N
- N = 256 baseline (335 MB for 5 fields at fp32, fits on 4090)
- N = 512 for production (2.7 GB, still fits on 4090 16GB)
- Box size L_box = 4R where R = π²ξ ≈ 10ξ → L_box ≈ 40ξ
- Grid spacing dx = L_box/N ≈ 0.16ξ at N=256 (resolves the ξ-scale core)

### Fields (5 real DOF per grid point)
- ψ = ψ_re + i ψ_im (2 real fields)
- A = (A_x, A_y, A_z) (3 real fields)
- Total: N³ × 5 × 4 bytes

### Memory estimate
| N   | Points | Memory (fp32) | + gradients | Fits 4090? |
|-----|--------|---------------|-------------|------------|
| 128 | 2.1M   | 42 MB         | ~170 MB     | ✓ easily   |
| 256 | 16.8M  | 335 MB        | ~1.3 GB     | ✓          |
| 512 | 134M   | 2.7 GB        | ~11 GB      | ✓ tight    |
| 768 | 453M   | 9.1 GB        | ~36 GB      | ✗ need A100|

## Initialization: Stamping a Vortex Knot

### Step 1: Knot centerline
Parametrize T(p,q) on a torus (R, aR):
```python
r₀(t) = ((R + aR cos(qt)) cos(pt),
          (R + aR cos(qt)) sin(pt),
          aR sin(qt))       t ∈ [0, 2π)
```

### Step 2: Distance field
For each grid point x, compute the minimum distance ρ(x) to the
centerline and the corresponding arc-length parameter s(x):
```python
ρ(x) = min_s |x - r₀(s)|
```

This is the expensive step (O(N³ × N_centerline)), but only done once.
Can be accelerated with a KD-tree or by solving the eikonal equation.

### Step 3: Scalar field
```python
|ψ(x)| = f₀(ρ(x))        # BPS radial profile from Step 1
arg(ψ(x)) = n × θ(x)     # phase winds n times around vortex core
```

where θ(x) is the angle around the vortex core at the nearest
centerline point.  For n = 1 (unit winding), the phase increases
by 2π around each cross-section of the tube.

**Critical subtlety:** for a KNOTTED vortex, the phase must be
globally consistent.  Use the Seifert surface construction:
- Find a surface S bounded by the knot K
- Set arg(ψ) = ±π on the two sides of S (branch cut)
- Smooth the branch cut over the healing length ξ

Alternative: use the "writhe + twist" decomposition to compute the
phase directly from the knot's Frenet frame.

### Step 4: Gauge field
```python
A(x) = a(ρ(x)) × (ê_θ / ρ(x))   # BPS gauge profile, azimuthal
```

where ê_θ is the unit vector tangent to circles around the vortex
core, and a(ρ) is the gauge field from Step 1.

## Relaxation: Gradient Flow

### Equations of motion (imaginary-time Ginzburg-Landau)
```
∂ψ/∂τ = D²ψ - λ(|ψ|² - v²)ψ         # = -δE/δψ*
∂A/∂τ = ∇×B - Im(ψ* Dψ)              # = -δE/δA (supercurrent)
```

### Gauge fixing
Without gauge fixing, the relaxation wanders in gauge-orbit space.
Use Coulomb gauge: impose ∇·A = 0 at each step by projecting:
```python
A ← A - ∇(∇⁻²(∇·A))     # Helmholtz projection
```

In Fourier space this is trivial: subtract the longitudinal component.

### Step size and convergence
- Adaptive step size (L-BFGS or Adam optimizer via JAX)
- Monitor: E(τ), max|∂ψ/∂τ|, topological charge Q
- Converged when ΔE/E < 10⁻⁸ per step
- Typical: O(10³-10⁴) steps at N=256

### JAX implementation
```python
@jax.jit
def energy(psi_re, psi_im, Ax, Ay, Az, dx, e, lam, v):
    # Covariant derivatives via finite difference
    # Magnetic field via curl
    # Return scalar E = sum of all contributions × dx³
    ...

grad_energy = jax.grad(energy, argnums=(0,1,2,3,4))
```

JAX autodiff eliminates the need to hand-derive δE/δψ and δE/δA.
The `jit` decorator compiles to GPU-optimized XLA code.

## Measurement

### After relaxation, extract:

1. **Total energy** E = ∫ ε d³x → this IS the particle mass
2. **Topological charge** Q = (1/2π) ∫ B·dl around the vortex
3. **Winding number** = phase winding of ψ around a loop encircling
   the vortex core
4. **Effective coupling** α_eff = (ratio of kinetic to potential
   energy, or scattering cross-section in a perturbed calculation)
5. **Radial profile** |ψ| along a cut perpendicular to the vortex →
   compare to f₀(ρ) from Step 1; count radial NODES → this is m

### The decisive test

For the ELECTRON (2,1 unknot):
- Does E(2,1) at R = π²ξ give the correct mass-to-energy ratio?
- Does the radial profile have m = 3 nodes?

For the ELECTRON/TAU ratio:
- Initialize (2,1) at two different radii or radial excitations
- Does E(2,1,m=1900)/E(2,1,m=3) ≈ m_τ/m_e = 3477?

For the TREFOIL/ELECTRON ratio:
- Does E(2,3)/E(2,1) give a number that, combined with the AB phase
  and crossing energy, reproduces the proton-to-electron mass ratio?

## Roadmap

### Phase A: Circular vortex ring (unknot) — 1 week
- Implement fields, energy, relaxation for a CIRCULAR (not knotted)
  vortex ring at radius R
- Verify: relaxed energy = μ_BPS × 2πR (1 + Kelvin-Saffman correction)
- Verify: profile matches f₀(ρ) from Step 1
- This is the "hello world" of Level 2

### Phase B: Trefoil vortex — 1-2 weeks
- Add knot initialization (T(2,3) centerline + phase winding)
- Relax and measure energy
- Compare E(2,3)/E(2,1) to the arc-length ratio from Level 1
- Look for crossing-energy contributions beyond arc length

### Phase C: Mass spectrum — 2-4 weeks
- Scan over (p,q) topologies: (2,1), (2,3), (2,5), (1,4), (3,4), ...
- For each: initialize, relax, measure E
- Build the energy ratio table E(p,q)/E(2,1)
- Compare to PDG mass ratios
- Identify where the Casimir integers emerge (or don't)

### Phase D: Radial excitations — ongoing
- For (2,1) unknot: initialize with m radial nodes
- Find the excitation spectrum E(m) vs m
- Test whether E(m=1900)/E(m=3) ≈ m_τ/m_e
- This is where the mass HIERARCHY lives

## Dependencies

- **JAX** (with GPU support): `pip install jax[cuda12]`
- **numpy/scipy** for initialization
- **matplotlib** for visualization
- **NVIDIA 4090** (16 GB) for N ≤ 512

## What success looks like

The Level 2 program succeeds if:

1. The relaxed energy for a (2,1) unknot matches the BPS line-tension
   prediction to < 1%.

2. The energy RATIO E(2,3)/E(2,1) differs from the arc-length ratio
   by a measurable amount attributable to crossing energy — showing
   that the trefoil's topology contributes beyond mere length.

3. The radial excitation spectrum E(m) produces a mass hierarchy
   with the correct functional form (not just the correct numbers
   for specific m values).

4. At least one Casimir integer (e.g., the cinquefoil eigenvalue 25
   appearing in the m_e/v_EW relation) emerges from the energy
   functional without being put in by hand.

Any ONE of these would be a significant result.  All four together
would convert the NWT Casimir framework from "phenomenological
reconstruction" to "derived from the Lagrangian."
