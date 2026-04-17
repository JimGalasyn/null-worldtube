#!/usr/bin/env python3
"""
Residual Histogram for Paper 13 §10
=====================================

Generates a log-binned histogram of |% error| across the 75-particle
master spectrum.  The distribution demonstrates the central claim of
Paper 13: the NWT framework reaches PDG values at a median below 0.1%
with no fitted parameters.

Output: figures/paper13_residual_histogram.pdf  (and .png)
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_TEX = (Path(__file__).parent.parent / "papers" /
             "paper13_sm_capstone.tex")

# Table-row residuals can be wrapped ($-0.026\%$) or bare (+0.000\%).
# The residual is followed by a status column (& ✓ or & A) then \\.
TABLE_ROW_RE = re.compile(
    r"\$?([+-]?[0-9]+\.[0-9]+)\\%\$?\s*&")


def collect_residuals_from_paper():
    """Parse the appendix longtable for particle residuals."""
    text = PAPER_TEX.read_text()
    start = text.index(r"\begin{longtable}")
    end = text.index(r"\end{longtable}", start)
    appendix = text[start:end]
    return [abs(float(m)) for m in TABLE_ROW_RE.findall(appendix)]


def main():
    residuals = np.array(sorted(collect_residuals_from_paper()))
    residuals = residuals[residuals > 0]  # drop 0.000% exact-match rows

    n_total = len(residuals)
    median = float(np.median(residuals))
    mean = float(np.mean(residuals))
    under_01 = int(np.sum(residuals < 0.1))
    under_05 = int(np.sum(residuals < 0.5))
    under_10 = int(np.sum(residuals < 1.0))
    worst = residuals.max()

    print(f"Particles with residuals: {n_total}")
    print(f"  median |%|:     {median:.4f}")
    print(f"  mean   |%|:     {mean:.4f}")
    print(f"  count < 0.1%:   {under_01} ({100*under_01/n_total:.0f}%)")
    print(f"  count < 0.5%:   {under_05} ({100*under_05/n_total:.0f}%)")
    print(f"  count < 1.0%:   {under_10} ({100*under_10/n_total:.0f}%)")
    print(f"  worst:          {worst:.3f}%")

    # ── Figure ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    vmin = max(1e-4, residuals.min() * 0.8)
    vmax = max(worst * 1.4, 1.0)
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 28)

    ax.hist(residuals, bins=bins, color="#1f77b4",
            edgecolor="white", linewidth=0.6,
            label=f"{n_total} particles")

    ax.set_xscale("log")
    ax.set_xlabel(r"$|(m_\mathrm{NWT} - m_\mathrm{PDG}) / m_\mathrm{PDG}|$  (%)",
                  fontsize=11)
    ax.set_ylabel("Number of particles", fontsize=11)
    ax.set_title(f"NWT residual distribution — {n_total} particles, "
                 f"median {median:.3f}%", fontsize=12)

    # set ylim first so reference-line text positions are stable
    y_top = ax.get_ylim()[1]

    # Reference lines with adjusted label positions
    # median: move up to avoid bar collision
    ax.axvline(median, ls="-", color="#cc2222", lw=1.2, alpha=0.8)
    ax.text(median * 1.05, y_top * 0.98, f"median = {median:.3f}%",
            rotation=90, ha="left", va="top", color="#cc2222", fontsize=9)

    # 0.1%: move up to avoid bar collision
    ax.axvline(0.1, ls="--", color="#888888", lw=1.2, alpha=0.8)
    ax.text(0.1 * 1.05, y_top * 0.98, "0.1%",
            rotation=90, ha="left", va="top", color="#888888", fontsize=9)

    # 1%: move down to avoid legend collision
    ax.axvline(1.0, ls="--", color="#888888", lw=1.2, alpha=0.8)
    ax.text(1.0 * 1.05, y_top * 0.60, "1%",
            rotation=90, ha="left", va="top", color="#888888", fontsize=9)

    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()

    out_dir = Path(__file__).parent.parent / "papers" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "paper13_residual_histogram.pdf"
    png_path = out_dir / "paper13_residual_histogram.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"\nWrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
