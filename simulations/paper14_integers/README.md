# Paper 14: Integers as Output

Simulation infrastructure for deriving the Casimir integers from the
gauge-field holonomy on torus-knot vortices, WITHOUT importing them
from Paper 8's analytic crossing algebra.

## The Chain

```
U(1) abelian Higgs vortex on T(p,q)
  → project onto plane → identify crossings
    → compute AB holonomy at each crossing (from Gauss linking)
      → build crossing matrices (braid group generators)
        → compute Lie algebra from commutators
          → extract Casimir invariants (C_A, C_F, dim(adj), dim(reps))
            → verify these ARE the integers in Paper 13's mass formulas
```

## Modules

- `crossings.py`     — project knot, identify crossings, build braid word
- `holonomy.py`      — AB holonomy at each crossing from gauge field
- `braid_algebra.py` — braid generators → Lie algebra → Casimir invariants
- `jones.py`         — Jones polynomial from braid (links to IBM quantum impl)
- `integers_out.py`  — full chain: knot → integers → mass formulas → PDG check

## The Key Test

If the integers {2, 3, 4, 5, 7, 8, 9, 10, 13, 16, 25, 29} emerge
from the gauge-field holonomy computation without being put in by
hand, the "fitting" criticism is definitively answered.

## Existing Assets

- `simulations/helmholtz_eigenvalue/` — BPS profile, AB phase verification
- `simulations/level2_abelian_higgs/` — 3D JAX pipeline, Gauss linking, mass traceability
- IBM quantum Jones polynomial implementation (NWT repo)
