#!/usr/bin/env python3
"""Upload paper10_strings.pdf to Zenodo and publish with DOI.

Usage:
    python3 scripts/zenodo_upload_paper10.py              # draft (no publish)
    python3 scripts/zenodo_upload_paper10.py --publish     # real upload + publish
    python3 scripts/zenodo_upload_paper10.py --sandbox     # test on sandbox
"""

import argparse
import os
import sys
import requests

TOKEN_FILE = os.path.expanduser("~/.zenodo_token")
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "papers", "paper10_strings.pdf")

SANDBOX_URL = "https://sandbox.zenodo.org/api"
PROD_URL = "https://zenodo.org/api"

METADATA = {
    "metadata": {
        "title": (
            "Vortex Strings in a Two-Component Superfluid Vacuum: "
            "Cosmic Strings, Dark Matter, and the String/Knot Duality"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "We explore the possibility that the central one-dimensional intuition of "
            "string theory may apply more naturally to a dark-sector condensate than "
            "to the visible sector. We propose that the vacuum is a two-component "
            "superfluid: psi_SM (the Standard Model condensate, described by knot "
            "topology with aspect ratio R/xi ~ 1) and psi_GUT (a residual grand-unified "
            "condensate, described by string topology with R/xi ~ 10^14). In this model, "
            "visible particles are associated with compact vortex knots, while dark "
            "particles are associated with thin vortex loops. Cosmic strings (the CSc-1 "
            "candidate) and galactic-center radio filaments (Yusef-Zadeh et al.) are "
            "macroscopic vortex lines in psi_GUT. The dark particle spectrum---a dark "
            "electron (93 MeV), dark muon (18.8 GeV), dark pions (25 GeV), and dark "
            "proton (187 GeV)---emerges from mode-locking on the dark torus with "
            "kappa_GUT = 5.32. The interface coupling alpha_dark = 1/(sqrt(2) kappa_SM "
            "kappa_GUT) = 1/74 follows from the cross-term in the two-condensate energy "
            "functional, matching sqrt(alpha_SM alpha_GUT) to 0.3%. The framework yields "
            "quantitative predictions: proton decay tau_p in [2.4, 5.5] x 10^34 yr (the "
            "lower end of the underlying Gmu-allowed window already excluded by Super-"
            "Kamiokande, the surviving range fully within Hyper-Kamiokande's projected "
            "sensitivity), the Totani (2025) 20 GeV gamma-ray halo excess as dark muon "
            "annihilation, the Bullet Cluster as a Mach 0.015 subsonic superfluid "
            "collision, and individual cosmic string tension Gmu/c^2 in [4, 7] x 10^-7, "
            "consistent with the CSc-1 lensing signal and with CMB/PTA network bounds "
            "once the Kelvin decay of the string network is accounted for. The six "
            "compactified dimensions of string theory find a suggestive analogue in six "
            "discrete quantum numbers (p, q, m, n_q, f, g) on the torus knot parameter "
            "space, suggesting a collapse of the landscape from 10^500 vacua to a "
            "countable periodic table. The paper is presented as a phenomenological "
            "framework, not a UV-complete derivation; central constructions including "
            "the candidate string-theoretic dualities and the spectrum-generating mode-"
            "locking mechanism remain conjectural pending a more formal field-theoretic "
            "treatment. Explicit falsification conditions are stated."
        ),
        "creators": [
            {"name": "Galasyn, James P.", "affiliation": "Independent researcher"},
            {"name": "Théodore, Claude", "affiliation": "Anthropic, San Francisco, CA"},
        ],
        "keywords": [
            "cosmic strings",
            "dark matter",
            "two-component superfluid vacuum",
            "string theory",
            "GUT",
            "vortex line",
            "torus knot",
            "null worldtube",
            "Bullet Cluster",
            "CSc-1",
            "Totani excess",
            "proton decay",
            "Hyper-Kamiokande",
            "string/knot duality",
            "hidden sector",
            "Galactic Center filaments",
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.18891785",
                "relation": "isPartOf",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19335224",
                "relation": "continues",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19334925",
                "relation": "references",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19225259",
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
        r = requests.post(f"{base_url}/deposit/depositions/{dep_id}/actions/publish",
                          headers=headers)
        if r.status_code != 202:
            print(f"Error publishing: {r.status_code}\n{r.text}")
            sys.exit(1)
        pub = r.json()
        doi = pub["doi"]
        doi_url = pub["doi_url"]
        print(f"\n  Published!")
        print(f"  DOI: {doi}")
        print(f"  URL: {doi_url}")
    else:
        prereserve_doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "N/A")
        print(f"\n  Draft created (not published).")
        print(f"  Pre-reserved DOI: {prereserve_doi}")
        print(f"  Edit/publish at: https://zenodo.org/deposit/{dep_id}")
        print(f"\n  Run with --publish to make it live.")


if __name__ == "__main__":
    main()
