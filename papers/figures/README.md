# Figure Descriptions — Paper Series

## Figure 1: The (2,1) Torus Knot

**File:** `figure1_torus_knot.pdf` (to be generated)

3D rendering of the (2,1) torus knot on a transparent torus surface. The knot curve winds p=2 times toroidally and q=1 time poloidally. Annotations:
- R: major radius, r: minor radius, α = r/R
- Current I = ec/L along the knot (arrows)
- The 13 mode structure indicated schematically (4 toroidal + 1 poloidal + 4 mixed + 5 metric − 1 zero = 13)

Source: `null_worldtube_diagrams.py` (~line 175). Single-column width (86 mm).

---

## Figure 2: Complete Scorecard — Predicted vs Measured

**File:** `figure2_scorecard.pdf` (to be generated)

Log-log scatter plot of all 23 predicted Standard Model parameters versus their measured values. Diagonal line = perfect agreement. Points span 14 orders of magnitude from Σm_ν (~59 meV) to m_t (~173 GeV), with mixing angles and coupling constants on a separate linear-scale inset.

Color coding by sector:
- Red: fermion masses (9 points)
- Blue: gauge/Higgs parameters (6 points)
- Green: CKM elements (4 points)
- Orange: PMNS elements (4 points)

Error bars from PDG. Inset: histogram of |error %| with median line at 0.7%.

Double-column width (178 mm) — this is THE figure of the paper.

---

## Table I: Nine Charged Fermion Masses (in main text, Section III.D)

## Table II: Complete 23-Parameter Scorecard (in main text, Section VII)

---

# Paper 2: The Torus Electron

## Figure 3: The Torus Electron

**File:** `figure3_torus_electron.pdf` (to be generated)

3D rendering of the (2,1) torus knot on a transparent torus surface — THE image of Paper 2. The knot curve winds p=2 times toroidally and q=1 time poloidally. Elements:
- Transparent torus surface with wireframe
- Knot curve with gradient coloring showing circulation direction
- Current direction arrows along the knot path
- Annotations: R (major radius), r (minor radius), α = r/R
- Möbius strip bounded by the (2,1) curve shown in translucent fill (demonstrating the non-orientable surface → fermionic statistics)

Source: `null_worldtube_diagrams.py` (draw_torus_knot function). Single-column width (86 mm).

---

## Figure 4: Angular Momentum vs Aspect Ratio

**File:** `figure4_angular_momentum.pdf` (to be generated)

Sweep plot showing time-averaged angular momentum ⟨L_z⟩/ℏ as a function of the torus aspect ratio r/R. Elements:
- x-axis: r/R from 0 to 1 (log scale or linear)
- y-axis: ⟨L_z⟩/ℏ from 0 to 1
- Curve: monotonically decreasing from 1.0 (pure circle, r/R → 0) to ~0 (fat torus, r/R → 1)
- Vertical dashed line at r/R = α = 0.00730 (fine-structure constant)
- Horizontal dashed line at L_z/ℏ = 0.5 (spin-½)
- Intersection point highlighted: "Spin-½ at r/R = α"
- Labels: "Spin-1 (photon)" at upper left, "Spin-½ (electron)" at intersection

Source: `core.py` print_angular_momentum_analysis sweep (lines 440–528). Single-column width (86 mm).

---

## Figure 5: Topology Decision Tree

**File:** `figure5_topology_tree.pdf` (to be generated)

Flowchart showing how the five requirements (R1–R5) eliminate all alternative topologies, leaving only the torus with (2,1) winding. Structure:

```
[Closed surface in R³]
         |
    R1: Closed path?
    R2: Non-contractible?
         |
    ┌────┴────┐
    NO        YES
  [Sphere]    |
  (fails)   R3: Two scales?
              |
         ┌────┴────┐
         NO        YES
       [Sphere]    |
       (1 scale)  R5: Finite energy?
                    |
               ┌────┴────┐
               NO        YES
           [Genus ≥ 2]   |
           (curvature   [TORUS]
            unstable)     |
                    Klein bottle
                    boundary condition
                         |
                    p must be EVEN
                         |
                    (2,1) torus knot
                    ← UNIQUE SOLUTION
```

Color coding: red = eliminated, green = survives. Each eliminated branch annotated with the specific failure mode.

Source: `topology.py` full analysis (lines 10–622). Double-column width (178 mm).
