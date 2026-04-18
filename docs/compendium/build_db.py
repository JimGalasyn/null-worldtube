#!/usr/bin/env python3
"""
Build the NWT Particle Compendium SQLite database.

Populates from the mass-from-topology computation with full
traceability: PDG values, NWT formulas, integer origins,
family membership, and historical notes.
"""

import sqlite3
import numpy as np
import json
from pathlib import Path


DB_PATH = Path(__file__).parent / "nwt_particles.db"


def create_schema(conn):
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS particles")
    c.execute("DROP TABLE IF EXISTS integers")
    c.execute("DROP TABLE IF EXISTS families")

    c.execute("""
    CREATE TABLE particles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        symbol TEXT,
        latex TEXT,
        category TEXT,
        subcategory TEXT,

        -- Topology
        p INTEGER,
        q INTEGER,
        genus INTEGER,
        carrier TEXT,
        m_radial INTEGER,
        n_q INTEGER,

        -- Mass / value
        formula_text TEXT,
        formula_latex TEXT,
        computed REAL,
        unit TEXT DEFAULT 'MeV',
        pdg_value REAL,
        pdg_unit TEXT DEFAULT 'MeV',
        residual_pct REAL,

        -- Status
        status TEXT DEFAULT '✓',

        -- Integer traceability
        integers_used TEXT,
        integer_origins TEXT,

        -- Classification
        rational_tier TEXT,
        family TEXT,

        -- Papers
        first_paper INTEGER,
        notes TEXT
    )""")

    c.execute("""
    CREATE TABLE integers (
        value INTEGER PRIMARY KEY,
        identity TEXT,
        topological_origin TEXT,
        formula_appearances TEXT,
        derivation TEXT
    )""")

    c.execute("""
    CREATE TABLE families (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        formula_text TEXT,
        formula_latex TEXT,
        parameter TEXT,
        members TEXT,
        description TEXT
    )""")

    conn.commit()


def populate_integers(conn):
    """Fill the integer vocabulary table."""
    c = conn.cursor()

    integers = [
        (2, "C_A(SU(2))", "Hopf link crossing count",
         "sin²θ_W, hierarchy step, neutrino ladder",
         "Count crossings in minimal Hopf link diagram → 2"),
        (3, "C_A(SU(3))", "Trefoil crossing count",
         "sin²θ₁₃, α_s, D₃ phase, ψ(2S), Υ(4S)",
         "Count crossings in minimal trefoil diagram → 3"),
        (4, "C_A²(SU(2))", "Hopf crossings squared",
         "κ_SM structure, χ_cJ denominator step",
         "Square of Hopf crossing count: 2² = 4"),
        (5, "q_cinquefoil", "Cinquefoil poloidal winding",
         "m_π⁰ factor, K± ratio, Υ(1S)",
         "Poloidal winding number of T(2,5): q = 5"),
        (6, "C_A(SU(2))×C_A(SU(3))", "Hopf × trefoil composite",
         "χ_cJ numerator step, pentaquark k=6, Λ_b",
         "Product of Hopf and trefoil crossing counts: 2×3 = 6"),
        (7, "C_A²(SU(3)) − C_A(SU(2))", "Trefoil² minus Hopf",
         "cos²θ_W numerator, χ_cJ, ψ(3770), η_c, D*",
         "Difference of Casimir squares: 9 − 2 = 7"),
        (8, "dim(adj SU(3))", "Adjoint dimension of SU(3)",
         "η_c denom, h_c denom, P_c(4397) k-value",
         "Adjoint representation dimension: 3² − 1 = 8"),
        (9, "C_A²(SU(3))", "Trefoil crossings squared",
         "sin²θ_W denom, m_d, χ_c0 numerator, Υ(2S)",
         "Square of trefoil crossing count: 3² = 9"),
        (10, "dim(10) of SU(5)", "Antisymmetric 2-tensor of SU(5)",
         "y_s Yukawa, D-meson binding, pentaquark k=10",
         "Antisymmetric representation: 5×4/2 = 10"),
        (11, "C_A²(SU(3)) + C_A(SU(2))", "Trefoil² plus Hopf",
         "h_c, ψ(3770), D_s±, Υ(4S)",
         "Sum of Casimir squares: 9 + 2 = 11"),
        (13, "p²+q² (trefoil)", "Geodesic norm of T(2,3)",
         "κ_SM series, m_τ/m_μ, pentaquark ratio, B*",
         "Pythagorean sum: 2² + 3² = 13"),
        (16, "dim(16) of Spin(10)", "16-spinor representation",
         "α_s prefactor, baryon denominator, Δλ eigenvalue",
         "Eigenvalue difference: q²_cinq − q²_tref = 25 − 9 = 16"),
        (24, "dim(adj SU(5))", "Adjoint dimension of SU(5)",
         "B-meson binding (B⁰, B±), Υ(3S)",
         "Adjoint representation: 5² − 1 = 24"),
        (25, "q²_cinquefoil", "Cinquefoil toroidal eigenvalue",
         "1/α, m_τ/m_e, m_H NLO, v_EW, B_s⁰",
         "Toroidal eigenvalue: −∂²/∂φ² on e^{i5φ} → 5² = 25"),
        (29, "p²+q² (cinquefoil)", "Geodesic norm of T(2,5)",
         "Pentaquark mass ratio, κ_GUT series, χ_b0",
         "Pythagorean sum: 2² + 5² = 29"),
    ]

    c.executemany(
        "INSERT INTO integers VALUES (?, ?, ?, ?, ?)", integers)
    conn.commit()


def populate_families(conn):
    """Fill the family table."""
    c = conn.cursor()

    families = [
        ("χ_cJ(1P) triplet",
         "(9+6J)/(7+4J) × m_τ",
         r"\frac{9+6J}{7+4J} m_\tau",
         "J ∈ {0, 1, 2}",
         "χ_c0, χ_c1, χ_c2",
         "Spin-orbit coupling enters linearly in J via Hopf×trefoil "
         "(numerator) and Hopf² (denominator). Every integer is a "
         "Casimir invariant: 9=C²_A(SU3), 6=C_A(SU2)C_A(SU3), "
         "7=C²_A(SU3)−C_A(SU2), 4=C²_A(SU2)."),
        ("Pentaquark multiplet",
         "m_p × (29/13)² × (1 − kα)",
         r"m_p (29/13)^2 (1-k\alpha)",
         "k ∈ {6, 7, 9, 10}",
         "P_c(4312), P_c(4337), P_c(4440), P_c(4457)",
         "Cinquefoil-to-trefoil geodesic ratio squared times "
         "proton mass, with k-dependent binding correction. "
         "Each k is a Casimir invariant from the NWT vocabulary."),
        ("Charged lepton tower",
         "m_e, m_τ/(17−25α), 25m_e/(α(1−α)²)",
         r"m_e,\; m_\tau/(17-25\alpha),\; 25m_e/(\alpha(1-\alpha)^2)",
         "generation 1, 2, 3",
         "e, μ, τ",
         "All unknot (2,1) excitations with different radial indices."),
        ("Neutrino 2√α ladder",
         "m_ν,i+1 = 2√α(1+3α/4) × m_ν,i",
         r"m_{\nu,i+1} = 2\sqrt\alpha(1+\frac{3\alpha}{4}) m_{\nu,i}",
         "generation 1, 2, 3",
         "ν₁, ν₂, ν₃",
         "Geometric ladder with step 2√α from the transition rule."),
        ("Charm-top Yukawa complements",
         "y_c = α, y_t = 1−α",
         r"y_c = \alpha,\; y_t = 1-\alpha",
         "y_c + y_t = 1",
         "c, t",
         "The charm and top quarks are topological complements."),
    ]

    c.executemany(
        "INSERT INTO families (name, formula_text, formula_latex, "
        "parameter, members, description) VALUES (?, ?, ?, ?, ?, ?)",
        families)
    conn.commit()


def populate_particles(conn):
    """Fill the particle table from the mass-from-topology data."""
    c = conn.cursor()

    # Topological ingredients
    alpha = 1.0 / (25 * np.pi * np.sqrt(3) + 1)
    m_e = 0.51099895
    v_corr = 1 + 25 * alpha / (4 * np.sqrt(3))
    v_EW = m_e * 25 / alpha**2 * v_corr
    m_tau = 25 * m_e / (alpha * (1 - alpha)**2)
    m_mu = m_tau / (13 + 4 - 25 * alpha)
    m_c = alpha * v_EW / np.sqrt(2)
    m_b = 2 * np.pi * alpha * 91187.6
    m_pi0 = 2 * m_e / alpha * (1 - 5 * alpha)
    m_pi_ch = 2 * m_e / alpha * (1 - alpha / 2)
    m_K = m_pi_ch * 5 / np.sqrt(2)
    m_p = 938.272
    M_Z = 91187.6
    CF3 = 4.0 / 3.0
    m_nu3 = (CF3 + 2*alpha) * alpha**6 * v_EW * 1e6
    step = 2 * np.sqrt(alpha) * (1 + 3*alpha/4)

    def res(comp, pdg):
        return (comp - pdg) / pdg * 100 if pdg != 0 else 0

    # (name, symbol, latex, cat, subcat, p, q, genus, carrier, m_rad, n_q,
    #  formula_text, computed, unit, pdg, pdg_unit, status, integers, tier, family, paper, notes)
    particles = [
        # Leptons
        ("electron", "e", r"e", "lepton", "charged", 2, 1, 0, "unknot", 3, 0,
         "anchor", m_e, "MeV", 0.51100, "MeV", "A", "", "", "charged lepton tower", 1,
         "The scale anchor. Mass set by definition."),
        ("muon", "μ", r"\mu", "lepton", "charged", 2, 1, 0, "unknot", 157, 0,
         "m_τ/(17−25α)", m_mu, "MeV", 105.658, "MeV", "✓", "13,4,25", "A",
         "charged lepton tower", 1,
         "Second-generation lepton. 17 = p²+q²_tref + C²_A(SU2)."),
        ("tau", "τ", r"\tau", "lepton", "charged", 2, 1, 0, "unknot", 1900, 0,
         "25·m_e/(α(1−α)²)", m_tau, "MeV", 1776.86, "MeV", "✓", "25", "A",
         "charged lepton tower", 1,
         "Third-generation lepton. 25 = q²_cinquefoil."),
        ("nu_3", "ν₃", r"\nu_3", "lepton", "neutrino", None, None, None, "cinquefoil", None, 0,
         "(4/3+2α)·α⁶·v_EW", m_nu3, "meV", 50.15e-3, "meV", "✓", "4/3,2", "A",
         "neutrino ladder", 6,
         "Heaviest neutrino. C_F(SU3) = 4/3."),
        ("nu_2", "ν₂", r"\nu_2", "lepton", "neutrino", None, None, None, "cinquefoil", None, 0,
         "2√α(1+3α/4)·m_ν3", step*m_nu3, "meV", 8.61e-3, "meV", "✓", "2,3,4", "A",
         "neutrino ladder", 6,
         "Middle neutrino. Step = 2√α from transition rule."),
        ("nu_1", "ν₁", r"\nu_1", "lepton", "neutrino", None, None, None, "cinquefoil", None, 0,
         "4α(1+3α/2)·m_ν3", step**2*m_nu3, "meV", 1.48e-3, "meV", "✓", "2,3,4", "A",
         "neutrino ladder", 6,
         "Lightest neutrino. NWT PREDICTION (not yet measured)."),

        # Quarks
        ("up", "u", r"u", "quark", "light", 2, 3, 1, "trefoil", None, None,
         "225(1+2α)/(54+α)·m_e",
         9*25*(1+2*alpha)/(2*27+alpha)*m_e, "MeV", 2.16, "MeV", "✓",
         "225=9×25, 54=2×27", "A", None, 13,
         "Lightest quark. m_e scale (unknot sector)."),
        ("down", "d", r"d", "quark", "light", 2, 3, 1, "trefoil", None, None,
         "9(1+2α)·m_e",
         9*(1+2*alpha)*m_e, "MeV", 4.67, "MeV", "✓",
         "9", "A", None, 13,
         "9 = C²_A(SU3) trefoil Casimir squared."),
        ("strange", "s", r"s", "quark", "strange", 2, 3, 1, "trefoil", None, None,
         "10α²(1+α)·v_EW/√2",
         10*alpha**2*(1+alpha)*v_EW/np.sqrt(2), "MeV", 93.4, "MeV", "✓",
         "10", "A", None, 13,
         "10 = dim(10) of SU(5). v_EW scale."),
        ("charm", "c", r"c", "quark", "charm", 2, 3, 1, "trefoil", None, None,
         "α·v_EW/√2", m_c, "MeV", 1270.0, "MeV", "✓",
         "α", "A", "charm-top complements", 13,
         "y_c = α. Topological complement of top."),
        ("bottom", "b", r"b", "quark", "bottom", 2, 3, 1, "trefoil", None, None,
         "2π·α·M_Z", m_b, "MeV", 4180.0, "MeV", "✓",
         "2,π", "C", None, 13,
         "M_Z scale. Weakest derivation status among quarks."),
        ("top", "t", r"t", "quark", "top", 2, 3, 1, "trefoil", None, None,
         "(1−α)·v_EW/√2",
         (1-alpha)*v_EW/np.sqrt(2), "MeV", 172760.0, "MeV", "✓",
         "α", "A", "charm-top complements", 13,
         "y_t = 1−α. The heaviest SM fermion."),

        # Gauge + Higgs
        ("W boson", "W±", r"W^\pm", "gauge", "weak", None, None, None, "hopf", None, None,
         "M_Z·√((7−α)/9)",
         M_Z*np.sqrt((9-2-alpha)/9), "MeV", 80379.0, "MeV", "✓",
         "7,9", "A", None, 8,
         "From sin²θ_W = (2+α)/9 and M_Z."),
        ("Z boson", "Z⁰", r"Z^0", "gauge", "weak", None, None, None, "hopf", None, None,
         "anchor", M_Z, "MeV", 91187.6, "MeV", "A",
         "", "", None, 8,
         "Electroweak scale anchor."),
        ("Higgs", "H⁰", r"H^0", "gauge", "higgs", None, None, None, "hopf", None, None,
         "(½+α+25α²)·v_EW",
         (0.5+alpha+25*alpha**2)*v_EW, "MeV", 125250.0, "MeV", "✓",
         "25", "A", None, 13,
         "25 = q²_cinquefoil NLO correction."),

        # Light mesons
        ("pion neutral", "π⁰", r"\pi^0", "meson", "light", None, None, None, "S²×S²", None, 2,
         "2m_e/α·(1−5α)", m_pi0, "MeV", 134.977, "MeV", "✓",
         "2,5", "A", None, 1,
         "The lightest meson. 5 = q_cinquefoil."),
        ("pion charged", "π±", r"\pi^\pm", "meson", "light", None, None, None, "S²×S²", None, 2,
         "2m_e/α·(1−α/2)", m_pi_ch, "MeV", 139.570, "MeV", "✓",
         "2", "A", None, 1,
         "Charged pion. α/2 = α/C_A(SU2)."),
        ("kaon", "K±", r"K^\pm", "meson", "light", None, None, None, "S²×S²", None, 2,
         "m_π±·5/√2", m_K, "MeV", 493.677, "MeV", "✓",
         "5,2", "A", None, 13,
         "5 = q_cinquefoil, √2 = √C_A(SU2)."),
        ("eta", "η", r"\eta", "meson", "light", None, None, None, "S²×S²", None, 2,
         "10/9·m_K", 10.0/9*m_K, "MeV", 547.862, "MeV", "✓",
         "10,9", "A", None, 13,
         "10 = dim(10) of SU(5), 9 = C²_A(SU3)."),
        ("rho", "ρ", r"\rho", "meson", "light_vector", None, None, None, "S²×S²", None, 2,
         "11/7·m_K", 11.0/7*m_K, "MeV", 775.26, "MeV", "✓",
         "11,7", "C", None, 13,
         "11 = C²_A3+C_A2, 7 = C²_A3−C_A2. Ratio ≈ π/2."),

        # Charmonium (selected)
        ("J/psi", "J/ψ", r"J/\psi", "meson", "charmonium", None, None, None, "S²×S²", None, 2,
         "√(4m_c²+m_τ²)",
         np.sqrt(4*m_c**2+m_tau**2), "MeV", 3096.9, "MeV", "✓",
         "1", "D", None, 13,
         "Λ = m_τ exactly (BPS triplet unity)."),
        ("chi_c1", "χ_c1", r"\chi_{c1}", "meson", "charmonium", None, None, None, "S²×S²", None, 2,
         "√(4m_c²+(15m_τ/11)²)",
         np.sqrt(4*m_c**2+(15.0/11*m_tau)**2), "MeV", 3510.7, "MeV", "✓",
         "15=9+6, 11=7+4", "B", "χ_cJ(1P) triplet", 13,
         "J=1 of the parametric family (9+6J)/(7+4J)."),

        # Pentaquarks
        ("Pc_4312", "P_c(4312)", r"P_c(4312)", "baryon", "pentaquark", 2, 5, 2, "cinquefoil", None, None,
         "m_p·(29/13)²·(1−10α)",
         m_p*(29.0/13)**2*(1-10*alpha), "MeV", 4311.9, "MeV", "✓",
         "29,13,10", "C", "pentaquark multiplet", 13,
         "k=10 = dim(10) of SU(5). First cinquefoil hadron observed."),
        ("Pc_4397", "P_c(4397)", r"P_c(4397)", "baryon", "pentaquark", 2, 5, 2, "cinquefoil", None, None,
         "m_p·(29/13)²·(1−8α)",
         m_p*(29.0/13)**2*(1-8*alpha), "MeV", None, "MeV", "✓",
         "29,13,8", "C", "pentaquark multiplet", 13,
         "NWT PREDICTION. k=8 = dim(adj SU(3)). Principal falsification target for LHCb."),

        # Mixing parameters (selected)
        ("sin2_thW", "sin²θ_W", r"\sin^2\theta_W", "mixing", "electroweak", None, None, None, None, None, None,
         "(2+α)/9", (2+alpha)/9, "dimensionless", 0.22290, "dimensionless", "✓",
         "2,9", "A", None, 8,
         "The cleanest Casimir form: C_A(SU2)+α over C²_A(SU3)."),
        ("alpha_s", "α_s(M_Z)", r"\alpha_s(M_Z)", "mixing", "strong", None, None, None, None, None, None,
         "16α(3+α)/(3(1−α))",
         16*alpha*(3+alpha)/(3*(1-alpha)), "dimensionless", 0.1179, "dimensionless", "✓",
         "16,3", "A", None, 13,
         "16 = dim(16) of Spin(10). GUT structure in the SM coupling."),
        ("delta_CP", "δ_CP", r"\delta_\text{CP}", "mixing", "CP_phase", None, None, None, None, None, None,
         "π − 2", np.pi - 2, "rad", 1.142, "rad", "✓",
         "2", "A", None, 13,
         "π − C_A(SU2). Same for CKM and PMNS — topological unification."),
    ]

    for p in particles:
        (name, symbol, latex, cat, subcat, pp, qq, genus, carrier, m_rad, n_q,
         formula, computed, unit, pdg, pdg_unit, status, ints, tier, family,
         paper, notes) = p

        residual = res(computed, pdg) if pdg is not None else None

        c.execute("""
            INSERT INTO particles
            (name, symbol, latex, category, subcategory,
             p, q, genus, carrier, m_radial, n_q,
             formula_text, computed, unit, pdg_value, pdg_unit,
             residual_pct, status, integers_used, rational_tier,
             family, first_paper, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, symbol, latex, cat, subcat, pp, qq, genus, carrier,
              m_rad, n_q, formula, computed, unit, pdg, pdg_unit,
              residual, status, ints, tier, family, paper, notes))

    conn.commit()


def print_summary(conn):
    c = conn.cursor()

    print("=" * 60)
    print("NWT PARTICLE COMPENDIUM DATABASE")
    print("=" * 60)

    c.execute("SELECT COUNT(*) FROM particles")
    print(f"\n  Particles: {c.fetchone()[0]}")

    c.execute("SELECT COUNT(*) FROM integers")
    print(f"  Integers:  {c.fetchone()[0]}")

    c.execute("SELECT COUNT(*) FROM families")
    print(f"  Families:  {c.fetchone()[0]}")

    print(f"\n  By category:")
    c.execute("SELECT category, COUNT(*) FROM particles GROUP BY category ORDER BY COUNT(*) DESC")
    for cat, count in c.fetchall():
        print(f"    {cat:<12} {count}")

    print(f"\n  By status:")
    c.execute("SELECT status, COUNT(*) FROM particles GROUP BY status")
    for status, count in c.fetchall():
        print(f"    {status:<4} {count}")

    print(f"\n  Predictions (no PDG value):")
    c.execute("SELECT name, formula_text, computed, unit FROM particles WHERE pdg_value IS NULL")
    for row in c.fetchall():
        print(f"    {row[0]}: {row[1]} = {row[2]:.1f} {row[3]}")

    print(f"\n  Sample query — worst residuals:")
    c.execute("""SELECT symbol, residual_pct, formula_text
                 FROM particles WHERE residual_pct IS NOT NULL
                 ORDER BY ABS(residual_pct) DESC LIMIT 5""")
    for sym, res, formula in c.fetchall():
        print(f"    {sym:<8} {res:+.3f}%  {formula}")

    print(f"\n  Database: {DB_PATH}")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    create_schema(conn)
    populate_integers(conn)
    populate_families(conn)
    populate_particles(conn)
    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
