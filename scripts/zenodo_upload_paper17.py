#!/usr/bin/env python3
"""Upload paper17_alpha_closure.pdf to Zenodo and publish with DOI.

Usage:
    python3 scripts/zenodo_upload_paper17.py              # draft (no publish)
    python3 scripts/zenodo_upload_paper17.py --publish    # real upload + publish
    python3 scripts/zenodo_upload_paper17.py --sandbox    # test on sandbox
"""

import argparse
import os
import sys
import requests

TOKEN_FILE = os.path.expanduser("~/.zenodo_token")
PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "papers",
    "paper17_alpha_closure.pdf",
)

SANDBOX_URL = "https://sandbox.zenodo.org/api"
PROD_URL = "https://zenodo.org/api"

METADATA = {
    "metadata": {
        "title": (
            "Newton's Constant and Rest-Frame Schroedinger Evolution "
            "from K_7 Graph-State Information Theory"
        ),
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "Papers 13-16 derived the Standard Model parameters, the "
            "fine-structure constant alpha, the 24-particle hadron mass "
            "spectrum, and the gravitational coupling G from the topology "
            "of torus-knot vortex defects in a relativistic condensate. "
            "Paper 16 proposed the Planck-hierarchy expression "
            "m_e/m_Pl = (8/7)(1+alpha/7) alpha^(21/2) as a numerical "
            "identity to 0.014%. This paper gives a quantum-information-"
            "theoretic derivation of the factors in that expression. "
            "Within the NWT framework, m_e/m_Pl is interpreted as the "
            "encoded fidelity of an electron worldtube traversing the "
            "K_7 stabiliser state under alpha-strength interaction-event "
            "noise, and Newton's constant G is interpreted as the "
            "transduction ratio between momentum-changing interaction "
            "events and topological-information creation. "
            "The closed form, valid at NNLO under the framework's "
            "normalisation choices and with alpha as input from Paper 8a's "
            "Aharonov-Bohm calculation, agrees with CODATA m_e/m_Pl to "
            "0.0001% (a 140x improvement over Paper 14) and predicts G "
            "within 11 ppm of CODATA G, inside the 22 ppm experimental "
            "uncertainty on G itself. "
            "The information-theoretic spine is a structural identity: "
            "for the K_7 stabiliser graph state |K_7> on 7 qubits and the "
            "two-body Pauli-YY Hamiltonian H_YY, "
            "<H_YY^n>_{|K_N>} = dim(Adj_so(N))^n exactly for all n, "
            "verified across the so(2n+1) family at N in {7, 9, 11}. The "
            "bracket coefficients in m_e/m_Pl are the first two "
            "normalised moments of H_YY. A 3-body operator probe shows "
            "<Y_uY_vY_w>_{|K_7>} = 0 for all 35 basis triples, supporting "
            "truncation of the bracket at alpha^2 through a "
            "representation-theoretic mechanism rather than accidental "
            "cancellation. The classical prefactor 8/7 = dim(S)/dim(V) "
            "is the trace ratio of the Cartan-graded subspace of |K_7>. "
            "Beyond the gravitational-coupling derivation, the same K_7 "
            "graph-state framework yields a derived effective evolution "
            "equation i hbar d_t |K_7> = m_e c^2 |K_7> within the model, "
            "from Bremermann's bound, the Paper 16 b2.13 bijection, and "
            "the PSL(2,7) edge-transitive symmetry of K_7. "
            "The framework is supported by IBM Heron R2 hardware "
            "measurements in eight datasets across three backends "
            "(ibm_kingston, ibm_marrakesh, ibm_fez). Direct measurement "
            "of <H_YY> on hardware-prepared |K_7> gives values 16.00 +/- "
            "0.21, 16.4715 +/- 0.0448, and 15.2815 +/- 0.0493 across "
            "three runs, all consistent with the predicted noiseless "
            "value 21 within hardware-noise expectations, with Z-scores "
            "above the random-state null of +75, +368, and +310 sigma. "
            "Zero-noise extrapolation on the same backend sharpens the "
            "cross-group K_9/K_7 ratio to 1.767 +/- 0.024 vs the "
            "structural prediction 36/21 = 1.714, agreeing to 3.07% "
            "(+2.16 sigma deviation) and bracketing M(K_9) in [36, 39] "
            "at 3 sigma with the framework's M = 36 at the centre. "
            "The 3-body null test (12,000 shots/triple) confirms "
            "<Y_uY_vY_w> = 0 with chi^2_red = 2.26 and rules out a "
            "Fano-line anisotropy hypothesis at +0.7 sigma. The "
            "syndrome-attractor experiment finds P(0000000 | |K_7> "
            "input) = 0.482 vs the random-input expectation 1/128, with "
            "measurement entropy difference Delta H_2 = 2.66 bits, "
            "supporting |K_7> as the natural fixed point of the "
            "preparation-plus-uncompute dynamics within the protocol. "
            "Within NWT, the remaining input data reduce to a "
            "topological sector, a single absolute mass scale, and the "
            "post-2019 SI conventions. The model achieves a strong form "
            "of internal parsimony once the framework's normalisation "
            "choices are fixed. "
            "Reproducibility: all analysis code (K_N graph-state moment "
            "numerics, bracket-truncation probes, Heron submission and "
            "analysis scripts, PSL(2,7) edge-transitivity re-analysis, "
            "forward-prediction and zero-noise extrapolation analyses, "
            "and raw Heron job outputs) is available at "
            "https://github.com/JimGalasyn/null-worldtube."
        ),
        "creators": [
            {"name": "Galasyn, James P.", "affiliation": "Independent researcher"},
            {"name": "Théodore, Claude", "affiliation": "Anthropic, San Francisco, CA"},
        ],
        "keywords": [
            "null worldtube",
            "null worldtube theory",
            "NWT",
            "K_7 graph state",
            "stabiliser graph state",
            "Reshetikhin-Turaev Wilson amplitude",
            "Spin(7) Chern-Simons",
            "Newton's constant",
            "electron-Planck mass ratio",
            "rest-frame Schroedinger equation",
            "Bremermann's bound",
            "PSL(2,7)",
            "b2.13 bijection",
            "octonion composition algebra",
            "Fano plane",
            "Poincaré homology sphere",
            "Heawood map",
            "trefoil knot",
            "torus knot",
            "information-theoretic gravity",
            "Verlinde-Padmanabhan inertia-as-information",
            "quantum error correction",
            "syndrome-attractor",
            "IBM Heron R2",
            "zero-noise extrapolation",
            "graph-state moment identity",
            "cross-group falsification",
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.18891785",
                "relation": "isPartOf",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19710846",
                "relation": "continues",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.19701263",
                "relation": "references",
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
