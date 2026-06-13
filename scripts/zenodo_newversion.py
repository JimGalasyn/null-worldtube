#!/usr/bin/env python3
"""Publish an UPDATED version of an existing Zenodo record (a new version under the
SAME concept DOI), so the concept-DOI link in docs/papers.md auto-resolves to it.

Works for any record type: a paper PDF (--pdf) OR a software artifact such as an
nwt-analysis release sdist/zip (--file). The flow is identical — only the uploaded
file differs.

For the FIRST publication of a NEW record, use a fresh-deposition script
(scripts/zenodo_upload_paperNN.py for papers, scripts/zenodo_upload_nwt_analysis.py
for the reproduction library) instead — those mint a new concept DOI.
See PUBLISHING.md for the full runbook and the new-vs-updated decision.

Recommended two-step (review what you publish):

  # 1. draft (REVERSIBLE — review on Zenodo, nothing minted):
  python3 scripts/zenodo_newversion.py --record-id 19710846 \
      --pdf ../null-worldtube-private/papers/paper16_nwt_lagrangian.pdf \
      --version 2 --note "Corrects the BPS self-dual coupling (erratum)."
  # 2. publish THAT reviewed draft (IRREVERSIBLE — a published record can't be deleted):
  python3 scripts/zenodo_newversion.py --publish-existing <draft_dep_id>

  # one-shot (draft + publish in one call; skips the review step):  ... --publish
  # sandbox dry-run (needs ~/.zenodo_token_sandbox):                ... --sandbox

--record-id is the LATEST published version's numeric record id (NOT the concept
record). Find it from the concept DOI zenodo.<conceptrecid>:
  curl -s https://zenodo.org/api/records/<conceptrecid> \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])"
"""

import argparse
import datetime as _dt
import os
import sys

import requests

PROD = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"


def load_token(sandbox):
    path = os.path.expanduser(
        "~/.zenodo_token_sandbox" if sandbox else "~/.zenodo_token")
    if not os.path.exists(path):
        sys.exit(f"Error: token file not found at {path}")
    with open(path) as f:
        return f.read().strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-id", type=int,
                    help="latest published version's record id (not the concept)")
    ap.add_argument("--file", dest="artifact",
                    help="path to the updated artifact (PDF for a paper, or an "
                         "sdist/wheel/zip for a software record e.g. nwt-analysis)")
    ap.add_argument("--pdf", dest="artifact",
                    help="alias for --file (kept for paper-publishing back-compat)")
    ap.add_argument("--version", help="new version label, e.g. 2")
    ap.add_argument("--publish-existing", type=int, metavar="DEP_ID",
                    help="publish an existing draft you already reviewed (IRREVERSIBLE) "
                         "instead of opening a new one; skips all other args")
    ap.add_argument("--note", default="",
                    help="version note appended to the description (e.g. the erratum)")
    ap.add_argument("--description-file", default=None,
                    help="replace the whole description with this file's contents "
                         "(use when the inherited description itself needs correcting)")
    ap.add_argument("--date", default=_dt.date.today().isoformat(),
                    help="publication_date (YYYY-MM-DD); defaults to today")
    ap.add_argument("--publish", action="store_true",
                    help="publish after upload — IRREVERSIBLE, mints the version DOI")
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()

    base = SANDBOX if a.sandbox else PROD
    h = {"Authorization": f"Bearer {load_token(a.sandbox)}"}

    # Publish a draft you already created + reviewed (the safe two-step flow).
    if a.publish_existing:
        r = requests.post(
            f"{base}/deposit/depositions/{a.publish_existing}/actions/publish",
            headers=h)
        if r.status_code != 202:
            sys.exit(f"publish failed: {r.status_code}\n{r.text}")
        o = r.json()
        print(f"PUBLISHED  version DOI: {o['doi']}\n  {o['doi_url']}")
        print("  (concept DOI unchanged; now resolves to this version.)")
        return

    if not (a.record_id and a.artifact and a.version):
        sys.exit("Error: --record-id, --file/--pdf, and --version are required "
                 "(unless using --publish-existing).")
    pdf = os.path.abspath(a.artifact)
    if not os.path.exists(pdf):
        sys.exit(f"Error: artifact not found at {pdf}")

    # 1. open a new-version draft from the existing record (inherits files + metadata)
    print(f"Opening new-version draft from record {a.record_id} ...")
    r = requests.post(
        f"{base}/deposit/depositions/{a.record_id}/actions/newversion", headers=h)
    if r.status_code != 201:
        sys.exit(f"newversion failed: {r.status_code}\n{r.text}")
    draft_url = r.json()["links"]["latest_draft"]
    d = requests.get(draft_url, headers=h).json()
    dep_id, bucket = d["id"], d["links"]["bucket"]
    print(f"  draft deposition id: {dep_id}")

    # 2. drop the inherited file(s) and upload the updated PDF
    for f in d.get("files", []):
        requests.delete(
            f"{base}/deposit/depositions/{dep_id}/files/{f['id']}", headers=h)
    with open(pdf, "rb") as fp:
        r = requests.put(f"{bucket}/{os.path.basename(pdf)}", data=fp, headers=h)
    if r.status_code not in (200, 201):
        sys.exit(f"upload failed: {r.status_code}\n{r.text}")
    print(f"  uploaded {os.path.basename(pdf)}")

    # 3. bump version + date; correct/append description; re-PUT metadata
    md = d["metadata"]
    md.pop("prereserve_doi", None)            # computed; not a writable field
    md["version"] = a.version
    md["publication_date"] = a.date
    if a.description_file:
        with open(a.description_file) as f:
            md["description"] = f.read().strip()
    if a.note:
        md["description"] = md.get("description", "").rstrip() + \
            f"\n\nVersion {a.version}: {a.note}"
    r = requests.put(f"{base}/deposit/depositions/{dep_id}",
                     json={"metadata": md}, headers=h)
    if r.status_code != 200:
        sys.exit(f"metadata failed: {r.status_code}\n{r.text}")
    print(f"  metadata set (version={a.version}, date={a.date})")

    # 4. publish (irreversible) or leave as a reviewable draft
    if a.publish:
        r = requests.post(
            f"{base}/deposit/depositions/{dep_id}/actions/publish", headers=h)
        if r.status_code != 202:
            sys.exit(f"publish failed: {r.status_code}\n{r.text}")
        out = r.json()
        print(f"\n  PUBLISHED  version DOI: {out['doi']}\n  {out['doi_url']}")
        print("  (concept DOI is unchanged and now resolves to this version.)")
    else:
        host = "sandbox.zenodo.org" if a.sandbox else "zenodo.org"
        print(f"\n  DRAFT ready (nothing minted). Review it, then publish THIS draft:")
        print(f"  Review : https://{host}/deposit/{dep_id}")
        print(f"  Publish: python3 scripts/zenodo_newversion.py --publish-existing {dep_id}"
              + (" --sandbox" if a.sandbox else ""))


if __name__ == "__main__":
    main()
