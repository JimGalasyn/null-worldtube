#!/usr/bin/env python3
"""
K_7 sector-rule look-elsewhere / degeneracy test (Paper 21a, rule eq:rule-I).

Round-2 referee concern (unanimous across Perplexity / Gemini / ChatGPT, 2026-05-25):
the per-walk sector assignment f(p mod 7, q mod 7) -> {lepton, meson, hyperon, nucleon}
was "validated on the same finite set it was fit to," with no demonstration that a
comparably simple rule would *not* fit equally well. (And mass-fit alone is known not to
select the assignment -- see analysis/nwt_assignment_degeneracy.py, which finds 11/13
canonical (p,q,m,n_q) dominated by a tighter cross-sector mass alternative; the selector
is sector + the physical quantum numbers, not mass.)

This script asks the look-elsewhere question for the SECTOR RULE specifically: given the
16 compendium (p,q) homology classes, is the fact that q mod 7 (plus one p mod 7 = 0
split at q=3) perfectly predicts the sector *surprising*, or expected by chance?

Method (permutation test). The substrate rule is realizable iff every group of classes
sharing a q-residue is monochromatic in sector -- except q=3, where a single (p mod 7 = 0)
conditional split is permitted (the rule's only p-dependence). We hold the sector MULTISET
fixed (1 nucleon, 2 leptons, 3 hyperons, 10 mesons -- the actual particle content) and ask:
over all ways to assign those labels to the 16 classes, what fraction are *also* realizable
by a rule of this form? That fraction is the look-elsewhere p-value for the q-mod-7 -> sector
regularity. Computed exactly (enumeration) and cross-checked by Monte Carlo.

SCOPE / honesty: this tests the sector rule GIVEN the (p,q) homology assignments (which come
from Paper 11's torus-knot mass formula). It does not address whether the (p,q) values
themselves are over-fit -- a separate, upstream question.

Run:  python3 analysis/k7_sector_rule_lookelsewhere.py
"""
from __future__ import annotations
import itertools, math, random
from collections import Counter

# 16 compendium classes: (p, q, sector)  -- Paper 21a Table 1
CLASSES = [
    (1, 3, "nucleon"), (2, 1, "lepton"), (3, 5, "meson"),  (1, 8, "lepton"),
    (2, 5, "meson"),   (2, 7, "meson"),  (3, 4, "hyperon"), (3, 7, "meson"),
    (4, 5, "meson"),   (4, 9, "meson"),  (5, 4, "hyperon"), (5, 7, "meson"),
    (6, 5, "meson"),   (7, 3, "meson"),  (7, 4, "hyperon"), (7, 5, "meson"),
]
LABELS = [c[2] for c in CLASSES]
MULTISET = Counter(LABELS)                       # {meson:10, hyperon:3, lepton:2, nucleon:1}

# group indices by q mod 7
groups: dict[int, list[int]] = {}
for i, (p, q, s) in enumerate(CLASSES):
    groups.setdefault(q % 7, []).append(i)


def realizable(labeling: list[str]) -> bool:
    """True iff labeling is producible by  sector = g(q mod 7)  with at most one
    (p mod 7 == 0) conditional split, and that split only at q == 3."""
    for r, idxs in groups.items():
        if r == 3:
            # split allowed by (p mod 7 == 0): partition by that bit, each part monochromatic
            for bit in (True, False):
                part = {labeling[i] for i in idxs if (CLASSES[i][0] % 7 == 0) == bit}
                if len(part) > 1:
                    return False
        else:
            if len({labeling[i] for i in idxs}) > 1:
                return False
    return True


def exact_pvalue() -> tuple[int, int, float]:
    """Exact count over all distinct label-assignments of the fixed multiset."""
    total = math.factorial(16)
    for n in MULTISET.values():
        total //= math.factorial(n)
    # enumerate distinct assignments via positions-per-sector; too many (16!/...) to brute
    # force directly, so enumerate over the constrained groups' sectors + multinomial rest.
    constrained = [r for r in groups if r != 3 and len(groups[r]) >= 1]
    # only multi-member non-q3 groups actually constrain; singletons are free
    multi = [r for r in groups if r != 3 and len(groups[r]) >= 2]
    free_positions = [i for r in groups if r not in multi for i in groups[r]]  # q3 + singletons
    sectors = list(MULTISET)
    favorable = 0
    for choice in itertools.product(sectors, repeat=len(multi)):
        used = Counter()
        ok = True
        for r, sec in zip(multi, choice):
            used[sec] += len(groups[r])
            if used[sec] > MULTISET[sec]:
                ok = False; break
        if not ok:
            continue
        # leftover labels fill the free positions (q3 + singletons) in any arrangement
        leftover = MULTISET - used
        nfree = sum(leftover.values())
        assert nfree == len(free_positions)
        ways = math.factorial(nfree)
        for n in leftover.values():
            ways //= math.factorial(n)
        favorable += ways
    return favorable, total, favorable / total


def mc_pvalue(trials: int = 2_000_000) -> float:
    base = list(LABELS)
    hits = 0
    for _ in range(trials):
        random.shuffle(base)
        if realizable(base):
            hits += 1
    return hits / trials


def main():
    print("=== K_7 sector-rule look-elsewhere test ===")
    print(f"16 compendium classes; sector multiset = {dict(MULTISET)}")
    print(f"observed labeling realizable by the rule? {realizable(LABELS)}")
    print("q-residue groups (size, sectors):")
    for r in sorted(groups):
        secs = Counter(LABELS[i] for i in groups[r])
        print(f"  q≡{r}: n={len(groups[r]):<2} {dict(secs)}")
    fav, tot, p = exact_pvalue()
    print(f"\nexact: {fav:,} favorable / {tot:,} total  ->  look-elsewhere p = {p:.3e}")
    print(f"   (i.e. 1 in {round(1/p):,} random sector-assignments is rule-realizable)")
    pmc = mc_pvalue()
    print(f"Monte Carlo cross-check (2e6 trials): p = {pmc:.3e}")
    # "free predictions": classes correctly classified beyond the per-group DOF
    free = sum(len(groups[r]) - 1 for r in groups if r != 3 and len(groups[r]) >= 2)
    print(f"\nrule DOF: one sector per populated q-residue (+1 p-split at q≡3)")
    print(f"multi-member residue groups predict {free} of 16 classes 'for free' "
          f"(beyond the first member of each group)")


if __name__ == "__main__":
    main()
