#!/usr/bin/env python3
"""
NWT Landauer Survey: Hyperons + Mesons + Leptons
=================================================

Tests whether the "near-Landauer" structural pattern of free-neutron
beta decay generalises to other free 1-body weak decays.

Hypothesis (structural):  in NWT's inertia-as-information framework,
free 1-body weak decays should satisfy

    Q  >=  Landauer_floor  =  n_new * log_2(24) * (m_e c^2 / 2 pi) * ln 2

where n_new is the number of newly created L_3 particles in the decay
(= n_final - 1 for a 1-body decay).

Beta decay sits at Q/Landauer = 1.51 (factor of 1.5 above floor).  The
question:  do other free weak decays show the same near-saturation, or
do hyperons / charm / bottom decays sit far above floor (like mu, tau)?

If only beta decay is near floor, that's a robust structural prediction
about the neutron's unique role.  If hyperons also cluster near floor,
the prediction generalises to a class of "barely-allowed" decays.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# -- constants -----------------------------------------------------------

M_E    = 0.51099895
LN2    = np.log(2.0)
PI     = np.pi
ALPHA  = 1.0 / 137.035999

T_EFF      = M_E / (2 * PI)
LAND_BIT   = T_EFF * LN2                  # 0.0564 MeV/bit
N_BITS_PER = np.log2(24.0)                # 4.585 bits per L_3 particle


# -- particle masses (MeV, PDG 2024) ------------------------------------

M = {
    # Leptons
    'e':       0.51099895,
    'mu':      105.6583755,
    'tau':     1776.86,
    'nu':      1e-9,                       # placeholder
    # Light mesons
    'pi+':     139.57039,
    'pi0':     134.9768,
    'eta':     547.862,
    # Strange mesons
    'K+':      493.677,
    'K0':      497.611,
    'KL':      497.611,
    # Charm
    'D+':      1869.66,
    'D0':      1864.84,
    'Ds':      1968.34,
    # Bottom
    'B+':      5279.34,
    'B0':      5279.65,
    # Baryons (octet)
    'p':       938.27208816,
    'n':       939.56542052,
    'Lambda':  1115.683,
    'Sigma+':  1189.37,
    'Sigma0':  1192.642,
    'Sigma-':  1197.449,
    'Xi0':     1314.86,
    'Xi-':     1321.71,
    'Omega-':  1672.45,
    # Charm baryons
    'Lc+':     2286.46,
    # Photon
    'gamma':   0.0,
}


@dataclass
class Decay:
    label: str
    initial: str
    final: list
    tau_s: float                       # mean lifetime
    BR: float = 1.0                    # branching ratio
    decay_class: str = 'weak'          # 'weak', 'EM', 'strong', 'lepton-weak', etc.
    note: str = ''


# Catalogue of free 1-body weak decays.
# n_new = len(final) - 1 (1 initial particle, n_final-1 newly created).

DECAYS = [
    # ----- LEPTONIC weak decays -----
    Decay('mu  -> e nu nu',         'mu',     ['e', 'nu', 'nu'],
          2.197e-6, 1.0, 'lepton-weak'),
    Decay('tau -> e nu nu',         'tau',    ['e', 'nu', 'nu'],
          2.903e-13, 0.178, 'lepton-weak'),
    Decay('tau -> mu nu nu',        'tau',    ['mu', 'nu', 'nu'],
          2.903e-13, 0.174, 'lepton-weak'),

    # ----- BARYON beta-like (semi-leptonic) -----
    Decay('n   -> p e nu',          'n',      ['p', 'e', 'nu'],
          880.2, 1.0, 'baryon-semilept', 'free neutron beta decay'),
    Decay('Lambda -> p e nu',       'Lambda', ['p', 'e', 'nu'],
          2.632e-10, 8.32e-4, 'baryon-semilept'),
    Decay('Lambda -> p mu nu',      'Lambda', ['p', 'mu', 'nu'],
          2.632e-10, 1.57e-4, 'baryon-semilept'),
    Decay('Sigma- -> n e nu',       'Sigma-', ['n', 'e', 'nu'],
          1.479e-10, 1.02e-3, 'baryon-semilept'),
    Decay('Sigma- -> Lambda e nu',  'Sigma-', ['Lambda', 'e', 'nu'],
          1.479e-10, 5.73e-5, 'baryon-semilept'),
    Decay('Xi-   -> Lambda e nu',   'Xi-',    ['Lambda', 'e', 'nu'],
          1.639e-10, 5.63e-4, 'baryon-semilept'),
    Decay('Xi-   -> Lambda mu nu',  'Xi-',    ['Lambda', 'mu', 'nu'],
          1.639e-10, 3.5e-4, 'baryon-semilept'),

    # ----- BARYON nonleptonic weak (1->2) -----
    Decay('Lambda -> p pi-',        'Lambda', ['p', 'pi+'],     # pi+ = pi- mass-wise
          2.632e-10, 0.639, 'baryon-nonlept', 'pi- final, mass = pi+'),
    Decay('Lambda -> n pi0',        'Lambda', ['n', 'pi0'],
          2.632e-10, 0.358, 'baryon-nonlept'),
    Decay('Sigma+ -> p pi0',        'Sigma+', ['p', 'pi0'],
          8.018e-11, 0.516, 'baryon-nonlept'),
    Decay('Sigma+ -> n pi+',        'Sigma+', ['n', 'pi+'],
          8.018e-11, 0.483, 'baryon-nonlept'),
    Decay('Sigma- -> n pi-',        'Sigma-', ['n', 'pi+'],
          1.479e-10, 0.999, 'baryon-nonlept'),
    Decay('Xi0   -> Lambda pi0',    'Xi0',    ['Lambda', 'pi0'],
          2.90e-10, 0.995, 'baryon-nonlept'),
    Decay('Xi-   -> Lambda pi-',    'Xi-',    ['Lambda', 'pi+'],
          1.639e-10, 0.999, 'baryon-nonlept'),
    Decay('Omega -> Lambda K-',     'Omega-', ['Lambda', 'K+'],
          8.21e-11, 0.678, 'baryon-nonlept'),
    Decay('Omega -> Xi0 pi-',       'Omega-', ['Xi0', 'pi+'],
          8.21e-11, 0.236, 'baryon-nonlept'),
    Decay('Omega -> Xi- pi0',       'Omega-', ['Xi-', 'pi0'],
          8.21e-11, 0.085, 'baryon-nonlept'),

    # ----- MESON 2-body weak -----
    Decay('pi+ -> mu nu',           'pi+',    ['mu', 'nu'],
          2.6033e-8, 0.9999, 'meson-2body'),
    Decay('pi+ -> e nu',            'pi+',    ['e', 'nu'],
          2.6033e-8, 1.23e-4, 'meson-2body', 'helicity-suppressed'),
    Decay('K+  -> mu nu',           'K+',     ['mu', 'nu'],
          1.238e-8, 0.6356, 'meson-2body'),
    Decay('K+  -> e nu',            'K+',     ['e', 'nu'],
          1.238e-8, 1.58e-5, 'meson-2body'),
    Decay('D+  -> mu nu',           'D+',     ['mu', 'nu'],
          1.04e-12, 3.74e-4, 'meson-2body'),
    Decay('D+  -> tau nu',          'D+',     ['tau', 'nu'],
          1.04e-12, 1.20e-3, 'meson-2body'),
    Decay('Ds  -> tau nu',          'Ds',     ['tau', 'nu'],
          5.04e-13, 5.32e-2, 'meson-2body'),
    Decay('B+  -> tau nu',          'B+',     ['tau', 'nu'],
          1.638e-12, 1.09e-4, 'meson-2body'),

    # ----- MESON beta-like (rare) -----
    Decay('pi+ -> pi0 e nu',        'pi+',    ['pi0', 'e', 'nu'],
          2.6033e-8, 1.036e-8, 'meson-beta',
          'pi-beta decay (degenerate isospin)'),
    Decay('K+  -> pi0 e nu',        'K+',     ['pi0', 'e', 'nu'],
          1.238e-8, 0.0507, 'meson-semilept', 'K_e3'),
    Decay('K+  -> pi0 mu nu',       'K+',     ['pi0', 'mu', 'nu'],
          1.238e-8, 0.0337, 'meson-semilept', 'K_mu3'),
    Decay('KL  -> pi+ e nu',        'KL',     ['pi+', 'e', 'nu'],
          5.116e-8, 0.4055, 'meson-semilept', 'K_L_e3'),
]


def landauer_floor(n_new: int) -> float:
    """Landauer floor energy for creating n_new new L_3 particles."""
    return n_new * N_BITS_PER * LAND_BIT


def Q_value(decay: Decay) -> float:
    """Q value = m_initial - sum(m_final)."""
    return M[decay.initial] - sum(M[p] for p in decay.final)


def section(s: str) -> None:
    print()
    print("=" * 86)
    print(" " + s)
    print("=" * 86)


def print_table(decays: list[Decay], title: str) -> None:
    section(title)
    print(f"  {'Process':<26} {'Q (MeV)':>10} {'n_new':>5} "
          f"{'Land (MeV)':>11} {'Q/Land':>9} {'tau (s)':>11} {'BR':>9}")
    print("  " + "-" * 84)

    for d in decays:
        Q = Q_value(d)
        n_new = len(d.final) - 1
        L = landauer_floor(n_new)
        ratio = Q / L if L > 0 else float('inf')

        if ratio < 0:
            ratio_str = f"{'<0!':>9}"
        elif ratio > 9999:
            ratio_str = f"{'>9999':>9}"
        else:
            ratio_str = f"{ratio:>9.3f}"

        print(f"  {d.label:<26} {Q:>10.4f} {n_new:>5d} "
              f"{L:>11.4f} {ratio_str} {d.tau_s:>11.2e} {d.BR:>9.2e}")


def main() -> None:
    section("NWT Landauer survey -- hyperons, mesons, leptons")

    print(f"""
Reference scales:
  T_eff    = m_e c^2 / (2 pi)  = {T_EFF:.4f} MeV
  Landauer/bit                  = {LAND_BIT:.4f} MeV/bit
  N_bits per L_3 particle       = {N_BITS_PER:.4f} bits
  Landauer floor for n_new=1    = {landauer_floor(1):.4f} MeV
  Landauer floor for n_new=2    = {landauer_floor(2):.4f} MeV
""")

    # Group by class
    classes = {
        'baryon-semilept':  'Baryon semileptonic decays (1 -> 3)',
        'baryon-nonlept':   'Baryon nonleptonic decays (1 -> 2)',
        'meson-2body':      'Meson 2-body weak decays (1 -> 2)',
        'meson-beta':       'Meson beta-like decays (1 -> 3, near-degenerate)',
        'meson-semilept':   'Meson semileptonic decays (1 -> 3)',
        'lepton-weak':      'Leptonic weak decays (1 -> 3)',
    }

    for cls, title in classes.items():
        decays = [d for d in DECAYS if d.decay_class == cls]
        if decays:
            print_table(decays, title)

    # Aggregate analysis
    section("Q/Landauer distribution by class")

    print(f"  {'Class':<32} {'count':>6} {'min Q/L':>10} "
          f"{'median Q/L':>11} {'max Q/L':>10}")
    print("  " + "-" * 72)

    for cls, title in classes.items():
        decays = [d for d in DECAYS if d.decay_class == cls]
        if not decays:
            continue
        ratios = []
        for d in decays:
            Q = Q_value(d)
            n_new = len(d.final) - 1
            L = landauer_floor(n_new)
            if L > 0 and Q > 0:
                ratios.append(Q / L)
        if not ratios:
            continue
        ratios = np.array(ratios)
        print(f"  {cls:<32} {len(ratios):>6d} "
              f"{ratios.min():>10.2f} {np.median(ratios):>11.2f} "
              f"{ratios.max():>10.2f}")

    # Find near-floor decays
    section("Near-Landauer-floor decays (ratio < 10)")

    near_floor = []
    for d in DECAYS:
        Q = Q_value(d)
        n_new = len(d.final) - 1
        L = landauer_floor(n_new)
        if L > 0 and 0 < Q / L < 10:
            near_floor.append((Q / L, d))

    near_floor.sort()
    print(f"  {'Process':<26} {'Q (MeV)':>10} {'Q/Land':>9} "
          f"{'class':<22} {'note':<30}")
    print("  " + "-" * 84)
    for ratio, d in near_floor:
        Q = Q_value(d)
        print(f"  {d.label:<26} {Q:>10.4f} {ratio:>9.3f} "
              f"{d.decay_class:<22} {d.note:<30}")

    section("Interpretation")

    if not near_floor:
        print("  No near-floor decays besides beta? (would be a surprise)")
    else:
        n = len(near_floor)
        print(f"""
  Found {n} weak decays with Q/Landauer < 10:
""")
        for ratio, d in near_floor:
            print(f"    {d.label:<26}  Q/L = {ratio:.3f}  ({d.decay_class})")

        print(f"""
  Pattern check:
    If only free neutron beta decay is unique near floor (ratio ~1.5)
    and all hyperon/meson decays are far above (ratio >>10), the
    structural prediction is "neutron is structurally near saturation"
    rather than "all weak decays cluster near floor".

    Watch for: are there OTHER decays at ratio O(1-10)?  If yes, do
    they correspond to near-degenerate mass splittings (like pi+/pi0
    or Sigma0/Lambda)?  This would generalise the pattern from
    "neutron is special" to "any near-degenerate isospin partner is
    near saturation".
""")


if __name__ == '__main__':
    main()
