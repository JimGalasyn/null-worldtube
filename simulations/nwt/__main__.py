"""CLI entry point for the null worldtube analysis package."""

import argparse
from .constants import TorusParams, lambda_C, m_e_MeV


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
    parser.add_argument('--einstein', action='store_true', help='Kerr-Newman geometry, frame dragging, mass equation')
    parser.add_argument('--junction', action='store_true',
                        help='Israel junction conditions, torus differential geometry, confinement')
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
    parser.add_argument('--quark-koide', action='store_true', dest='quark_koide',
                        help='Quark Koide extension: linking corrections to angle and hierarchy')
    parser.add_argument('--pythagorean', action='store_true',
                        help='Pythagorean mode catalog: resonant modes on torus by aspect ratio')
    parser.add_argument('--proton-mass', action='store_true', dest='proton_mass',
                        help='Proton mass from first principles: Cornell potential + NWT string tension')
    parser.add_argument('--orbit', action='store_true',
                        help='Orbit analysis: what fixes the electron size? Scale invariance survey')
    parser.add_argument('--casimir', action='store_true',
                        help='Casimir energy: can vacuum fluctuations set r/R = α?')
    parser.add_argument('--greybody', action='store_true',
                        help='Greybody factors: partial transparency of the NWT surface')
    parser.add_argument('--pair-creation', action='store_true', dest='pair_creation',
                        help='Pair creation: what determines the torus geometry?')
    parser.add_argument('--transient-impedance', action='store_true', dest='transient_impedance',
                        help='LC circuit transient impedance during pair creation')
    parser.add_argument('--transmission-line', action='store_true', dest='transmission_line',
                        help='Distributed transmission line / waveguide model')
    parser.add_argument('--euler-heisenberg', action='store_true', dest='euler_heisenberg',
                        help='EH Lagrangian, Schwinger rate, pair creation grid')
    parser.add_argument('--gradient-profile', action='store_true', dest='gradient_profile',
                        help='Field gradients, LCFA validity, violence parameter')
    parser.add_argument('--settling-spectrum', action='store_true', dest='settling_spectrum',
                        help='Settling radiation: Kelvin wave spectrum from vortex relaxation')
    parser.add_argument('--thevenin', action='store_true',
                        help='Thevenin impedance matching: constrain r/R from circuit + Kelvin wave consistency')
    parser.add_argument('--R', type=float, default=1.0, help='Major radius in units of λ_C')
    parser.add_argument('--r', type=float, default=0.1, help='Minor radius in units of λ_C')
    parser.add_argument('--p', type=int, default=1, help='Toroidal winding number')
    parser.add_argument('--q', type=int, default=1, help='Poloidal winding number')
    args = parser.parse_args()

    if args.scan:
        from .core import scan_torus_parameters
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
        from .hydrogen import print_hydrogen_analysis
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
        from .analyses import print_skilton_analysis
        print_skilton_analysis()
        return

    if args.dark_matter:
        from .analyses import print_dark_matter_analysis
        print_dark_matter_analysis()
        return

    if args.weinberg:
        from .analyses import print_weinberg_analysis
        print_weinberg_analysis()
        return

    if args.gravity:
        from .analyses import print_gravity_analysis
        print_gravity_analysis()
        return

    if args.einstein:
        from .orbit import print_einstein_analysis
        print_einstein_analysis()
        return

    if args.junction:
        from .orbit import print_junction_analysis
        print_junction_analysis()
        return

    if args.topology:
        from .analyses import print_topology_analysis
        print_topology_analysis()
        return

    if args.koide:
        from .analyses import print_koide_analysis
        print_koide_analysis()
        return

    if args.stability:
        from .analyses import print_stability_analysis
        print_stability_analysis()
        return

    if args.decay_dynamics:
        from .analyses import print_decay_dynamics
        print_decay_dynamics()
        return

    if args.neutrino:
        from .analyses import print_neutrino_analysis
        print_neutrino_analysis()
        return

    if args.quark_koide:
        from .analyses import print_quark_koide_analysis
        print_quark_koide_analysis()
        return

    if args.pythagorean:
        from .analyses import print_pythagorean_analysis
        print_pythagorean_analysis()
        return

    if args.proton_mass:
        from .hadrons import print_proton_mass_analysis
        print_proton_mass_analysis()
        return

    if args.orbit:
        from .orbit import print_orbit_analysis
        print_orbit_analysis()
        return

    if args.casimir:
        from .fields import print_casimir_analysis
        print_casimir_analysis()
        return

    if args.greybody:
        from .fields import print_greybody_analysis
        print_greybody_analysis()
        return

    if args.pair_creation:
        from .pair_creation import print_pair_creation_analysis
        print_pair_creation_analysis()
        return

    if args.transient_impedance:
        from .pair_creation import print_transient_impedance_analysis
        print_transient_impedance_analysis()
        return

    if args.transmission_line:
        from .settling import print_transmission_line_analysis
        print_transmission_line_analysis()
        return

    if args.euler_heisenberg:
        from .pair_creation import print_euler_heisenberg_analysis
        print_euler_heisenberg_analysis()
        return

    if args.gradient_profile:
        from .pair_creation import print_gradient_profile_analysis
        print_gradient_profile_analysis()
        return

    if args.settling_spectrum:
        from .settling import print_settling_spectrum_analysis
        print_settling_spectrum_analysis()
        return

    if args.thevenin:
        from .settling import print_thevenin_analysis
        print_thevenin_analysis()
        return

    params = TorusParams(
        R=args.R * lambda_C,
        r=args.r * lambda_C,
        p=args.p,
        q=args.q,
    )

    from .particles import print_basic_analysis
    print_basic_analysis(params)

    if args.self_energy:
        from .particles import print_self_energy_analysis
        print("\n")
        print_self_energy_analysis(params)

    if args.resonance:
        from .particles import print_resonance_analysis
        print("\n")
        print_resonance_analysis(params)

    if args.angular_momentum:
        from .core import print_angular_momentum_analysis
        print("\n")
        print_angular_momentum_analysis(params)

    if args.energy:
        from .core import compute_total_energy
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
