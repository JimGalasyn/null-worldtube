"""
Render the Compton s-channel Feynman diagram for Paper 19's QED-lens
section, with each algebraic term in iM color-coded to match its
diagram element.

The diagram is rendered straight out of
`nwt_substrate.qed.compton.diagrams.s_channel` via the library's
`render_color_mapped()` interface (added in v0.1.3).  Each `Term`
in the diagram's `expression_terms` carries an `element_id` (e.g.
"in_e", "internal_e", "out_g"); the renderer paints each diagram
part in the color registered in `element_colors` for that id, and
the algebraic chunk in the expression appears in the same color.
The reader can immediately read off the algebra-to-picture
correspondence:  the teal `\bar u(p')` chunk IS the teal outgoing
electron line; the red `S_F(p+k)` chunk IS the red internal
propagator; etc.

Output:
  paper19_fig_qed_compton_s.pdf  (vector, paper-grade)
  paper19_fig_qed_compton_s.png  (raster, for git preview)

Run:  python3 diagrams/paper19_fig_qed_compton.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import nwt_substrate.qed as qed


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    diag = qed.compton.diagrams.s_channel

    diag.save_color_mapped(
        os.path.join(here, "paper19_fig_qed_compton_s.pdf"),
        figsize=(6.5, 5.6),
    )
    diag.save_color_mapped(
        os.path.join(here, "paper19_fig_qed_compton_s.png"),
        dpi=150,
        figsize=(6.5, 5.6),
    )

    print("Wrote paper19_fig_qed_compton_s.{pdf,png}")


if __name__ == "__main__":
    main()
