#!/usr/bin/env python3
"""Upload paper16_nwt_lagrangian.pdf to Zenodo and publish with DOI.

Usage:
    python3 scripts/zenodo_upload_paper16.py              # draft (no publish)
    python3 scripts/zenodo_upload_paper16.py --publish    # real upload + publish
    python3 scripts/zenodo_upload_paper16.py --sandbox    # test on sandbox
"""

import argparse
import os
import sys
import requests

TOKEN_FILE = os.path.expanduser("~/.zenodo_token")
PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "papers",
    "paper16_nwt_lagrangian.pdf",
)

SANDBOX_URL = "https://sandbox.zenodo.org/api"
PROD_URL = "https://zenodo.org/api"

METADATA = {
    "metadata": {
        "title": (
            "The NWT Lagrangian: "
            "A Three-Field Theory of Particles and Gravity"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "Papers 6, 8, 13, 14, and 15 derived the Standard Model "
            "parameters, the fine-structure constant alpha, the "
            "24-particle hadronic mass spectrum, and the gravitational "
            "coupling G from the topology of torus-knot vortex defects "
            "in a relativistic condensate. This paper presents the "
            "underlying Lagrangian as a single three-field theory. "
            "The minimal NWT Lagrangian has three fields -- a complex "
            "scalar condensate psi, an abelian gauge field A_mu, and "
            "an S^2-valued Skyrme-Faddeev unit field n^a -- tuned to "
            "the BPS critical coupling lambda = e^2/2 and Derrick "
            "equilibrium kappa = R/a_0 = pi^2. "
            "The Lagrangian as written carries one U(1) gauge field; "
            "the full SU(3) x SU(2) x U(1) Standard Model gauge "
            "content is compatible with the crossing algebra of "
            "torus-knot vortex cores (Papers 13-14) and is matched in "
            "from a SO(10) UV completion at E_GUT. "
            "The derivation is organised into five layers L1-L5: L1 "
            "field content; L2 kinetic + gauge + BPS potential, "
            "verified to saturate the Bogomolny bound at ~10^-3 %; L3 "
            "Skyrme-Faddeev quartic + Hopf theta-term, locking the "
            "topological sector to Q_H = p*m; L4 Paper 6 mass formula "
            "on 24 particles (1.06% median deviation, with three of "
            "four factors derived and the n_q^q link-enhancement "
            "factor retained as empirical input); L5 Paper 15 "
            "gravitational hierarchy m_e/m_Pl = (8/7)(1+alpha/7) "
            "alpha^(21/2) (G to 0.029% of CODATA at NLO). "
            "A natural UV completion to a SO(10) gauge theory at "
            "E_GUT = 7.41e15 GeV adds 33 heavy gauge bosons and "
            "yields three concrete falsifiable predictions: proton "
            "lifetime of order 10^35 yr (in the vicinity of "
            "Hyper-Kamiokande / DUNE reach), non-MSSM coupling "
            "unification at alpha_GUT = 1/40, and -- if the "
            "SO(10) -> SM transition is first order -- a stochastic "
            "gravitational-wave signature near 7.4 GHz. "
            "A first-principles one-loop Casimir calculation on "
            "S^3/2I in the BPS trefoil background (Phases 0-5 of an "
            "ongoing research programme) produces the O(1) "
            "coefficient expected for the Seeley a_2 piece of the "
            "effective action: Delta(1/16 pi G) ~ 0.27 m_e^2 in "
            "natural units. This result LOCALISES the alpha^(-21) "
            "suppression of G: it is not present in the Seeley "
            "coefficient and must therefore enter through the "
            "Wilson-line amplitude in the matter-to-gauge transition "
            "channel, as proposed in Paper 15 Sec 7.2. This is a "
            "structural localisation, not a numerical derivation; "
            "rigorous evaluation of the Wilson amplitude on the "
            "21-edge K_7 Eulerian circuit is concrete, well-defined, "
            "and deferred to a follow-up paper. "
            "Reproducibility: all analysis code (L1-L5 verifications, "
            "Phase 0-5 one-loop Casimir pipeline on S^3/2I, BPS "
            "solver, Spin(7)/Cl(0,7) construction) is available at "
            "https://github.com/JimGalasyn/null-worldtube."
        ),
        "creators": [
            {"name": "Galasyn, James P.", "affiliation": "Independent researcher"},
            {"name": "Théodore, Claude", "affiliation": "Anthropic, San Francisco, CA"},
        ],
        "keywords": [
            "NWT Lagrangian",
            "three-field theory",
            "BPS vortex",
            "Skyrme-Faddeev",
            "abelian Higgs",
            "Standard Model from topology",
            "Poincare homology sphere",
            "one-loop Casimir",
            "Wilson amplitude",
            "Seeley-DeWitt coefficients",
            "Spin(7)",
            "octonion Clifford algebra",
            "SO(10) UV completion",
            "proton decay",
            "Hyper-Kamiokande",
            "coupling unification",
            "high-frequency gravitational waves",
            "null worldtube",
            "null worldtube theory",
            "torus knot",
            "Hopf invariant",
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.18891785",
                "relation": "isPartOf",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19701263",
                "relation": "continues",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19654507",
                "relation": "references",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19635239",
                "relation": "references",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19516386",
                "relation": "references",
                "scheme": "doi",
            },
        ],
        "license": "cc-by-4.0",
    }
}


def load_token():
    if not os.path.exists(TOKEN_FILE):
        print(f"Error: token file not found at {TOKEN_FILE}")
        sys.exit(1)
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true",
                        help="Publish after upload (makes DOI permanent)")
    parser.add_argument("--sandbox", action="store_true",
                        help="Use Zenodo sandbox for testing")
    args = parser.parse_args()

    token = load_token()
    base_url = SANDBOX_URL if args.sandbox else PROD_URL
    headers = {"Authorization": f"Bearer {token}"}

    pdf = os.path.abspath(PDF_PATH)
    if not os.path.exists(pdf):
        print(f"Error: PDF not found at {pdf}")
        sys.exit(1)

    # Step 1: Create empty deposition
    print("Creating deposition...")
    r = requests.post(f"{base_url}/deposit/depositions",
                      json={}, headers=headers)
    if r.status_code != 201:
        print(f"Error creating deposition: {r.status_code}\n{r.text}")
        sys.exit(1)

    dep = r.json()
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"  Deposition ID: {dep_id}")

    # Step 2: Upload PDF
    print(f"Uploading {os.path.basename(pdf)}...")
    with open(pdf, "rb") as fp:
        r = requests.put(f"{bucket_url}/{os.path.basename(pdf)}",
                         data=fp, headers=headers)
    if r.status_code not in (200, 201):
        print(f"Error uploading: {r.status_code}\n{r.text}")
        sys.exit(1)
    print("  Upload complete.")

    # Step 3: Set metadata
    print("Setting metadata...")
    r = requests.put(f"{base_url}/deposit/depositions/{dep_id}",
                     json=METADATA, headers=headers)
    if r.status_code != 200:
        print(f"Error setting metadata: {r.status_code}\n{r.text}")
        sys.exit(1)
    print("  Metadata set.")

    # Step 4: Publish (if requested)
    if args.publish:
        print("Publishing...")
        r = requests.post(
            f"{base_url}/deposit/depositions/{dep_id}/actions/publish",
            headers=headers,
        )
        if r.status_code != 202:
            print(f"Error publishing: {r.status_code}\n{r.text}")
            sys.exit(1)
        pub = r.json()
        doi = pub["doi"]
        doi_url = pub["doi_url"]
        print("\n  Published!")
        print(f"  DOI: {doi}")
        print(f"  URL: {doi_url}")
    else:
        prereserve = dep.get("metadata", {}).get("prereserve_doi", {})
        prereserve_doi = prereserve.get("doi", "N/A")
        print("\n  Draft created (not published).")
        print(f"  Pre-reserved DOI: {prereserve_doi}")
        print(f"  Edit/publish at: https://zenodo.org/deposit/{dep_id}")
        print("\n  Run with --publish to make it live.")


if __name__ == "__main__":
    main()
