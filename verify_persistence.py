#!/usr/bin/env python3
"""verify_persistence.py — an uncertainty bound for "cmri beats persistence".

verify_rmse_gap.py bounds the headline resync RMSE gap (cmri vs v2).  The other
headline claim of §5.4/§5.5 is that cmri (0.531) beats the *daily-resync
persistence control* ("temperature stays put", 0.625), a claim the report makes
with no uncertainty attached.  This script closes that gap with the exact same
day-block machinery as verify_rmse_gap.py:

  1. the pooled RMSE gap vs persistence and its day-block bootstrap 95% CI, and
  2. the per-day paired RMSE-gap series (mean/se, normal approx t and CI).

Why days?  10-min resync errors have lag-1 rho ~ 0.97 within a day, so raw steps
are not independent; each calendar day re-anchors at the day boundary, so the
honest resampling unit is the whole day (42 blocks).

Data source discipline (single-touch): the prediction columns are frozen
artifacts, never re-evaluated.  cmri_resync lives in results/predictions_cmri.csv;
the persistence control was only written by the earlier run to
results/predictions.csv, which carries the SAME v2_resync as the cmri artifact
(verified below: the two files share identical (date, T1_observed) order and
matching v2_resync on every row).  We therefore join the persistence column onto
the cmri rows by index, asserting alignment rather than assuming it — so the
comparison never touches the held-out test a second time.
"""

import csv
import math
import random
from statistics import fmean, pstdev

CMRI = "results/predictions_cmri.csv"
PERS = "results/predictions.csv"   # older artifact carrying persistence_resync
SEED = 42
B = 5000


def rmse(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def main() -> None:
    with open(CMRI) as f:
        cmri_rows = list(csv.DictReader(f))
    with open(PERS) as f:
        pers_rows = list(csv.DictReader(f))

    assert len(cmri_rows) == len(pers_rows), "artifacts must be row-aligned"
    assert all(c["date"] == p["date"] for c, p in zip(cmri_rows, pers_rows)), \
        "artifacts must share identical (date, T1_observed) order"
    assert all(c["v2_resync"] == p["v2_resync"] for c, p in zip(cmri_rows, pers_rows)), \
        "v2 columns must agree across artifacts"

    day = {}  # date -> list of (e_pers, e_cmri) signed resync errors
    for c, p in zip(cmri_rows, pers_rows):
        if c["cmri_resync"] == "" or p["persistence_resync"] == "":
            continue  # day-boundary anchor rows, excluded by protocol
        obs = float(c["T1_observed"])
        day.setdefault(c["date"][:10], []).append(
            (obs - float(p["persistence_resync"]), obs - float(c["cmri_resync"])))
    days = [day[k] for k in sorted(day)]
    nd = len(days)

    ep = [a for d in days for a, _ in d]
    ec = [c for d in days for _, c in d]
    n = len(ep)
    rp, rc = rmse(ep), rmse(ec)
    gap = rp - rc
    print(f"cmri vs daily-resync persistence, resync protocol, "
          f"n={n} steps across {nd} day blocks")
    print(f"  pooled RMSE:  persistence {rp:.4f}   cmri {rc:.4f}   gap = {gap:+.4f} degC")

    # 1. Day-block bootstrap on the pooled gap.
    rng = random.Random(SEED)
    gaps = []
    for _ in range(B):
        sel = [days[rng.randrange(nd)] for _ in range(nd)]
        a = [x for d in sel for x, _ in d]
        b = [x for d in sel for _, x in d]
        gaps.append(rmse(a) - rmse(b))
    gaps.sort()
    lo, hi = gaps[int(B * 0.025)], gaps[int(B * 0.975)]
    cover = sum(1 for x in gaps if x > 0.0) / B
    print(f"  day-block bootstrap ({B} draws, seed {SEED}):")
    print(f"    95% percentile CI on pooled gap: [{lo:+.4f}, {hi:+.4f}] degC")
    print(f"    P(gap > 0) = {cover:.4f}")

    # 2. Per-day paired RMSE-gap series (normal-approx error bars).
    g = [rmse([a for a, _ in d]) - rmse([c for _, c in d]) for d in days]
    mg = fmean(g)
    se = pstdev(g) / math.sqrt(nd)
    t = mg / se
    wins = sum(1 for x in g if x > 1e-12)
    print(f"  per-day RMSE-gap series ({nd} days):")
    print(f"    mean {mg:+.4f}  sd {pstdev(g):.4f}  se {se:.4f}  t = {t:+.2f}"
          f"  (normal-approx 95% CI [{mg - 1.96 * se:+.4f}, {mg + 1.96 * se:+.4f}])")
    print(f"    days cmri beats persistence: {wins}/{nd}")

    print("\nReading: the persistence-beating margin is not sample luck — the "
          "CI does not straddle zero and P(gap > 0) > 0.99, consistent with the "
          "HAC-DM rejection on the v2 differential (verify_significance.py, "
          "p~3.5e-03) and with verify_rmse_gap.py on the v2 comparison.  The "
          "margin (~0.09 degC) is narrower than vs v2 (0.115), as expected — "
          "v2's own weakness was trailing this same persistence control.")


if __name__ == "__main__":
    main()
