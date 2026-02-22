"""
CLI entry point for the Null Worldtube package.

Usage:
    python -m nwt --hydrogen
    python -m simulations.nwt --hydrogen
"""

import argparse
from .constants import lambda_C, m_e_MeV, TorusParams


def main():
    parser = argparse.ArgumentParser(description='Closed null worldtube analysis')
    parser.add_argument('--scan', action='store_true', help='Scan parameter space')
    parser.add_argument('--energy', action='store_true', help='Detailed energy analysis')
    parser.add_argument('--self-energy', action='store_true', help='Self-energy analysis with α')
    parser.add_argument('--resonance', action='store_true', help='Resonance quantization analysis')
    parser.add_argument('--find-radii', action='store_true', help='Find self-consistent radii')
    parser.add_argument('--angular-momentum', action='store_true', help='Angular momentum analysis')
    parser.add_argument('--pair-production', action='store_true', help='Pair production analysis')
    parser.add_argument('--decay', action='store_true', help='Decay landscape analysis')
    parser.add_argument('--hydrogen', action='store_true', help='Hydrogen atom: orbits, shells, bonding')
    parser.add_argument('--transitions', action='store_true', help='Photon emission/absorption spectrum')
    parser.add_argument('--quarks', action='store_true', help='Quarks and hadrons: linked torus model')
    parser.add_argument('--skilton', action='store_true', help="Skilton's α formula and integer cosmology")
    parser.add_argument('--dark-matter', action='store_true', dest='dark_matter',
                        help='Dark matter candidates from TE torus modes')
    parser.add_argument('--weinberg', action='store_true',
                        help='Weinberg angle and electroweak masses from torus geometry')
    parser.add_argument('--gravity', action='store_true', help='Gravity from torus metric: GW modes, Planck mass')
    parser.add_argument('--topology', action='store_true',
                        help='Topology survey: alternative structures for worldtube model')
    parser.add_argument('--koide', action='store_true',
                        help='Koide angle from torus geometry: lepton mass predictions')
    parser.add_argument('--stability', action='store_true',
                        help='Stability analysis and phase space of torus configurations')
    parser.add_argument('--decay-dynamics', action='store_true', dest='decay_dynamics',
                        help='Dissipative decay dynamics: saddles, bifurcations, strange attractors')
    parser.add_argument('--neutrino', action='store_true',
                        help='Neutrino masses from twist-wave dispersion on the torus')
    parser.add_argument('--anim-ionization', action='store_true', dest='anim_ionization',
                        help='Animation: hydrogen photoionization (outputs GIF)')
    parser.add_argument('--R', type=float, default=1.0, help='Major radius in units of λ_C')
    parser.add_argument('--r', type=float, default=0.1, help='Minor radius in units of λ_C')
    parser.add_argument('--p', type=int, default=1, help='Toroidal winding number')
    parser.add_argument('--q', type=int, default=1, help='Poloidal winding number')
    args = parser.parse_args()

    if args.scan:
        from .particles import scan_torus_parameters
        scan_torus_parameters()
        return

    if args.find_radii:
        from .particles import print_find_radii
        print_find_radii()
        return

    if args.pair_production:
        from .particles import print_pair_production
        print_pair_production()
        return

    if args.decay:
        from .particles import print_decay_landscape
        print_decay_landscape()
        return

    if args.hydrogen:
        from .atoms import print_hydrogen_analysis
        print_hydrogen_analysis()
        return

    if args.transitions:
        from .transitions import print_transition_analysis
        print_transition_analysis()
        return

    if args.quarks:
        from .hadrons import print_quark_analysis
        print_quark_analysis()
        return

    if args.skilton:
        from .cosmology import print_skilton_analysis
        print_skilton_analysis()
        return

    if args.dark_matter:
        from .dark_matter import print_dark_matter_analysis
        print_dark_matter_analysis()
        return

    if args.weinberg:
        from .electroweak import print_weinberg_analysis
        print_weinberg_analysis()
        return

    if args.gravity:
        from .gravity import print_gravity_analysis
        print_gravity_analysis()
        return

    if args.topology:
        from .topology import print_topology_analysis
        print_topology_analysis()
        return

    if args.koide:
        from .leptons import print_koide_analysis
        print_koide_analysis()
        return

    if args.stability:
        from .dynamics import print_stability_analysis
        print_stability_analysis()
        return

    if args.decay_dynamics:
        from .dynamics import print_decay_dynamics
        print_decay_dynamics()
        return

    if args.neutrino:
        from .neutrinos import print_neutrino_analysis
        print_neutrino_analysis()
        return

    if args.anim_ionization:
        from .anim_ionization import run_animation
        run_animation()
        return

    # Default: basic analysis with given torus parameters
    from .particles import print_basic_analysis, print_self_energy_analysis, print_resonance_analysis
    from .core import print_angular_momentum_analysis, compute_total_energy

    params = TorusParams(
        R=args.R * lambda_C,
        r=args.r * lambda_C,
        p=args.p,
        q=args.q,
    )

    print_basic_analysis(params)

    if args.self_energy:
        print("\n")
        print_self_energy_analysis(params)

    if args.resonance:
        print("\n")
        print_resonance_analysis(params)

    if args.angular_momentum:
        print("\n")
        print_angular_momentum_analysis(params)

    if args.energy:
        print("\n\n")
        # Compare different winding numbers at the same torus size
        print("=" * 70)
        print("WINDING NUMBER COMPARISON (same torus geometry)")
        print("  Now includes self-energy corrections")
        print("=" * 70)
        for p in range(1, 5):
            for q in range(1, 4):
                tp = TorusParams(R=params.R, r=params.r, p=p, q=q)
                _, E_total_MeV, breakdown = compute_total_energy(tp)
                crossings = 2 * min(p, q) * (max(p, q) - 1) if p != q else 0
                print(f"  ({p},{q}): E_circ = {breakdown['E_circ_MeV']:8.4f}, "
                      f"E_self = {breakdown['U_total_MeV']:.6f}, "
                      f"E_total = {E_total_MeV:8.4f} MeV, "
                      f"crossings = {crossings}, "
                      f"E/m_e = {E_total_MeV/m_e_MeV:.4f}")


if __name__ == '__main__':
    main()
