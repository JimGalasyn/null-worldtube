# Publishing to Zenodo — maintainer runbook

How to publish a **new** paper, an **updated** paper, or a release of the
**`nwt-analysis`** reproduction-code library. Papers and the reproduction library
are archived on Zenodo (one record each) and are **not** duplicated in this repo —
`docs/papers.md` links each artifact's **concept DOI**. (Field-theory library
releases are separate: `nwt-substrate/docs/RELEASING.md`,
`jax-solitons` release process.)

- New/updated **paper** → sections A / B.
- A **reproduction-code** release (`nwt-analysis`) → section C.
- Before publishing a paper, make sure its cited scripts actually exist in
  `nwt-analysis` (section C tie-in below).

## Key facts (read once)

- **Two DOIs per paper.** The **concept DOI** (`zenodo.<conceptrecid>`) is the
  "all versions" record — it never changes and always resolves to the **latest**
  version. Each published version also gets its own **version DOI**.
  `docs/papers.md` must link the **concept** DOI so corrections show automatically.
- **Publishing is IRREVERSIBLE.** A published Zenodo record cannot be deleted (only
  superseded by a new version). **Drafts are reversible** — always create the draft,
  review it on Zenodo, then publish.
- **Token:** `~/.zenodo_token` (production). Optional `~/.zenodo_token_sandbox` for
  `--sandbox` dry-runs against `sandbox.zenodo.org`.
- **New vs updated — do not confuse them.** The per-paper `zenodo_upload_paperNN.py`
  scripts create a **fresh deposition = a new concept DOI**. Running one to "update"
  a paper would orphan a *second* concept DOI that `docs/papers.md` doesn't point to.
  Updates MUST use the new-version flow (`zenodo_newversion.py`).
- **Errata tie-in:** a Tier-2 correction in `null-worldtube-private/papers/ERRATA_POLICY.md`
  (the printed value is wrong / internally inconsistent) is published as a **new
  version** here.

---

## A. New paper (first publication)

0. **Reproduction-code gate.** A paper's data-availability must point at code that
   exists. Confirm every script the paper cites is in `nwt-analysis`:
   ```bash
   cd null-worldtube-private && python scripts/reconcile.py --only cited
   ```
   If `cited:nwt-analysis` WARNs, promote the missing scripts into
   `nwt-analysis/src/nwt_analysis/paperNN_*/` (and release per section C) **before**
   publishing the paper. Cite the library in the paper's data-availability:
   `pip install nwt-analysis` + the nwt-analysis **concept** DOI.
1. Build the PDF from `null-worldtube-private/papers/paperNN_*.tex`.
2. Copy/adapt a `scripts/zenodo_upload_paperNN.py` (title, description, creators,
   keywords, `related_identifiers`, license `cc-by-4.0`). Point `PDF_PATH` at the
   built PDF.
3. **Draft first:** `python3 scripts/zenodo_upload_paperNN.py` → review the draft +
   pre-reserved DOI at the printed Zenodo URL.
4. **Publish:** `python3 scripts/zenodo_upload_paperNN.py --publish`.
5. Get the **concept** DOI of the new record (it's `version_doi - 1` only by luck —
   look it up):
   ```bash
   curl -s https://zenodo.org/api/records/<version_recid> \
     | python3 -c "import json,sys;print(json.load(sys.stdin)['conceptdoi'])"
   ```
6. Add a `## Paper NN` section to `docs/papers.md` linking that **concept** DOI;
   commit + push the public repo.

## B. Updated paper (new version of an existing record)

Use `scripts/zenodo_newversion.py` — it opens a new-version draft on the existing
record, replaces the file, bumps `version` + `publication_date`, and (optionally)
publishes. The concept DOI is unchanged, so **`docs/papers.md` needs no edit**.

1. Rebuild the corrected PDF in `null-worldtube-private/papers/`.
2. Find the **latest version's record id** from the concept DOI `zenodo.<conceptrecid>`:
   ```bash
   curl -s https://zenodo.org/api/records/<conceptrecid> \
     | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])"
   ```
3. **Draft first** (reversible — nothing minted):
   ```bash
   python3 scripts/zenodo_newversion.py --record-id <latest_recid> \
     --pdf ../null-worldtube-private/papers/paperNN_*.pdf \
     --version 2 --note "<one-line erratum/changelog>" \
     [--description-file corrected_description.txt]   # if the inherited blurb is wrong
   ```
   Review the draft at the printed Zenodo URL. **Correct any stale text** in the
   inherited description (e.g. a superseded constant) via `--description-file` — the
   new-version draft inherits the *old* description, which for an erratum is exactly
   what's wrong.
4. **Publish the reviewed draft (IRREVERSIBLE):**
   ```bash
   python3 scripts/zenodo_newversion.py --publish-existing <draft_dep_id>
   ```
   (`--publish-existing` publishes the *same* draft you inspected. The one-shot
   `--publish` flag also exists but skips the review step.)
5. Errata bookkeeping (Tier-2): the correction is already in the
   `resolved_mysteries_concordance.md` §3 + By-paper index and the paper's `.tex`
   carries a rendered erratum box + `% SUPERSEDED` header. No `docs/papers.md` change
   (concept DOI is stable). Optionally note the new version DOI in the concordance row.

---

## C. Reproduction-code release (`nwt-analysis`)

The per-paper reproduction drivers live in the `nwt-analysis` repo (a sibling of
this one), **not** as loose scripts here. It is published two ways that move
together: **PyPI** (so `pip install nwt-analysis` works) and a **Zenodo software
record** (so papers can cite an immutable concept DOI). The concept DOI is stable,
so `docs/papers.md` links it once and never edits it again.

### C.1 First release (mints the concept DOI — run ONCE)
1. From `nwt-analysis/`: bump `version` in `pyproject.toml`, ensure CI is green
   (`pytest` — every driver compiles + is discoverable), tag the commit.
2. Build the artifact: `python -m build` → `dist/nwt_analysis-<ver>.tar.gz`.
3. **Zenodo (draft first):**
   `python3 scripts/zenodo_upload_nwt_analysis.py --file ../nwt-analysis/dist/nwt_analysis-<ver>.tar.gz`
   — review the draft, then re-run with `--publish`.
4. Look up the **concept** DOI (the script prints the command) and add a
   `## Reproduction code` entry to `docs/papers.md` linking it + the PyPI page.
5. **PyPI:** `python -m twine upload dist/*`.

### C.2 Later releases (new version — keeps the concept DOI)
1. Bump `version`, CI green, tag, `python -m build`.
2. Find the latest version's record id from the concept DOI (same lookup as
   section B step 2), then **draft → publish** with the generalized uploader:
   ```bash
   python3 scripts/zenodo_newversion.py --record-id <latest_recid> \
     --file ../nwt-analysis/dist/nwt_analysis-<ver>.tar.gz \
     --version <ver> --note "<changelog: which papers' scripts were added/fixed>"
   python3 scripts/zenodo_newversion.py --publish-existing <draft_dep_id>
   ```
   (`--file` is the software-artifact form of `--pdf`; the flow is identical.)
3. `python -m twine upload dist/*`. No `docs/papers.md` edit (concept DOI stable).

### When to cut an `nwt-analysis` release
- A **new paper** is being published whose cited scripts were just promoted in
  (section A step 0) — release the code *before/with* the paper.
- The `reconcile.py --only cited` gate WARNs (a published paper cites a script not
  yet in the library).
- A driver was fixed for bit rot (an upstream API moved).

---

## Gotchas

- The fresh-upload scripts are **new-record only**; never use one to update.
- New-version drafts **inherit the prior files and metadata** — replace the file and
  fix any stale description text; don't assume the inherited blurb is current.
- **Concept ≠ version − 1** for re-released papers — always look up `conceptdoi`.
- Sanity-check all versions under a concept after publishing:
  ```bash
  curl -s "https://zenodo.org/api/records/?q=conceptrecid:<conceptrecid>&all_versions=true&sort=mostrecent&size=5" \
    | python3 -c "import json,sys;[print(r['metadata'].get('version'), r['doi']) for r in json.load(sys.stdin)['hits']['hits']]"
  ```

## First worked example

**Paper 16 v2 (2026-06-13)** — Tier-2 erratum (BPS self-dual coupling λ=e²/2 → 2e²).
Latest v1 record id `19710846`, concept `10.5281/zenodo.19710845`. Published via path B.
