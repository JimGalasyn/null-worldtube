# Publishing papers to Zenodo — maintainer runbook

How to publish a **new** paper or an **updated** version of an existing one. Papers
are archived on Zenodo (one record per paper) and are **not** duplicated in this
repo — `docs/papers.md` links each paper's **concept DOI**. (Library releases are a
separate process: `nwt-substrate/docs/RELEASING.md`.)

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
