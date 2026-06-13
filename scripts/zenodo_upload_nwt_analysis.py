#!/usr/bin/env python3
# ============================================================================
# FIRST-PUBLICATION ONLY. This mints a NEW concept DOI for the nwt-analysis
# reproduction library. Run it ONCE. For every later release, use
# scripts/zenodo_newversion.py (it keeps the same concept DOI so the link in
# docs/papers.md never changes). See ../PUBLISHING.md, section C.
# ============================================================================
"""Upload an nwt-analysis release artifact to Zenodo as a SOFTWARE record.

Build the artifact first (from the nwt-analysis repo):
    python -m build            # -> dist/nwt_analysis-<ver>.tar.gz
  or a source zip of a tagged commit.

Usage:
    python3 scripts/zenodo_upload_nwt_analysis.py --file ../nwt-analysis/dist/nwt_analysis-0.1.0.tar.gz
    python3 scripts/zenodo_upload_nwt_analysis.py --file <artifact> --publish
    python3 scripts/zenodo_upload_nwt_analysis.py --file <artifact> --sandbox
"""

import argparse
import os
import sys

import requests

TOKEN_FILE = os.path.expanduser("~/.zenodo_token")
SANDBOX_URL = "https://sandbox.zenodo.org/api"
PROD_URL = "https://zenodo.org/api"

METADATA = {
    "metadata": {
        "title": "nwt-analysis: reproduction code for the Null Worldtube Theory papers",
        "upload_type": "software",
        "description": (
            "Per-paper reproduction drivers for the Null Worldtube Theory (NWT) "
            "papers. Each module computes a published paper's figures and numbers "
            "on top of the NWT field-theory upstreams nwt-substrate (physics "
            "library) and jax-solitons (JAX/GPU soliton engine). This is the "
            "single, citable home for the papers' reproduction code: papers cite "
            "it by `pip install nwt-analysis` plus this concept DOI rather than "
            "linking loose scripts. The cross-repo reconciler asserts that every "
            "script a published paper cites is present here, so the "
            "data-availability links cannot silently rot. "
            "Source: https://github.com/JimGalasyn/nwt-analysis . "
            "See REPRODUCTIONS.md for the full script-to-paper map."
        ),
        "creators": [
            {"name": "Galasyn, James P.", "affiliation": "Independent researcher"},
            {"name": "Théodore, Claude", "affiliation": "Anthropic, San Francisco, CA"},
        ],
        "keywords": [
            "null worldtube theory", "reproduction", "research software",
            "topological physics", "neutrino sector", "quantum error correction",
            "knot invariants", "baryogenesis",
        ],
        "related_identifiers": [
            # the upstreams this library consumes
            {"identifier": "10.5281/zenodo.20012027",  # nwt-substrate concept DOI
             "relation": "requires", "scheme": "doi"},
        ],
        "license": "MIT",
    }
}


def load_token():
    if not os.path.exists(TOKEN_FILE):
        sys.exit(f"Error: token file not found at {TOKEN_FILE}")
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", dest="artifact", required=True,
                    help="release artifact (sdist .tar.gz / wheel / source zip)")
    ap.add_argument("--publish", action="store_true",
                    help="publish after upload (makes DOI permanent)")
    ap.add_argument("--sandbox", action="store_true", help="use Zenodo sandbox")
    a = ap.parse_args()

    token = load_token()
    base = SANDBOX_URL if a.sandbox else PROD_URL
    h = {"Authorization": f"Bearer {token}"}

    art = os.path.abspath(a.artifact)
    if not os.path.exists(art):
        sys.exit(f"Error: artifact not found at {art}")

    print("Creating deposition...")
    r = requests.post(f"{base}/deposit/depositions", json={}, headers=h)
    if r.status_code != 201:
        sys.exit(f"Error creating deposition: {r.status_code}\n{r.text}")
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"  Deposition ID: {dep_id}")

    print(f"Uploading {os.path.basename(art)}...")
    with open(art, "rb") as fp:
        r = requests.put(f"{bucket}/{os.path.basename(art)}", data=fp, headers=h)
    if r.status_code not in (200, 201):
        sys.exit(f"Error uploading: {r.status_code}\n{r.text}")
    print("  Upload complete.")

    print("Setting metadata...")
    r = requests.put(f"{base}/deposit/depositions/{dep_id}",
                     json=METADATA, headers=h)
    if r.status_code != 200:
        sys.exit(f"Error setting metadata: {r.status_code}\n{r.text}")
    print("  Metadata set.")

    if a.publish:
        print("Publishing...")
        r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish",
                          headers=h)
        if r.status_code != 202:
            sys.exit(f"Error publishing: {r.status_code}\n{r.text}")
        pub = r.json()
        print(f"\n  Published!\n  version DOI: {pub['doi']}\n  URL: {pub['doi_url']}")
        print("  Look up the CONCEPT DOI and add a 'Reproduction code' link to "
              "docs/papers.md (concept DOI is stable across versions):")
        print(f"    curl -s {base}/records/{pub['id']} "
              "| python3 -c \"import json,sys;print(json.load(sys.stdin)['conceptdoi'])\"")
    else:
        pre = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "N/A")
        host = "sandbox.zenodo.org" if a.sandbox else "zenodo.org"
        print(f"\n  Draft created (not published).\n  Pre-reserved DOI: {pre}")
        print(f"  Review/publish at: https://{host}/deposit/{dep_id}")
        print("  Re-run with --publish to make it live.")


if __name__ == "__main__":
    main()
