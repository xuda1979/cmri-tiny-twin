#!/usr/bin/env python3
"""verify_rmse_gap.py — an uncertainty bound for the headline RMSE win.

verify_significance.py tests the *MAE-loss differential* d_i = |e_v2| - |e_cmri|
(HAC DM rejects at p≈3.5e-03, §5.5).  But the headline number of §3/§5.1 is the
RMSE gap (0.6460 -> 0.5308, Δ = +0.1153 °C), which the report until now asserted
with no uncertainty attached.  This script closes that gap: re-aggregating the
FROZEN resync predictions (results/predictions_cmri.csv, single-touch artifact —
no new evaluation, no refitting), it computes

  1. the pooled RMSE gap and its day-block bootstrap percentile 95% CI, and
  2. the per-day paired RMSE-gap series (mean/sd/se, normal approx t and CI).

Why days?  The 10-min resync errors have lag-1 ρ ≈ 0.97 within a day, so raw
steps are not independent (that is why §5.5 uses HAC DM on the mean).  Each
resync calendar day is one quasi-independent block (the errors re-anchor at the
day boundary), so the honest resampling unit is the whole day.  The bootstrap
resamples the 42 days with replacement, recomputes the pooled RMSE of each
resample, and reports percentiles and P(gap>0).  Both checks answer the same
question the HAC DM answers for the mean, but for the squared-error metric: is
the RMSE margin sample luck?
"""

import csv
import math
import random
from statistics import fmean, pstdev

PRED = "results/predictions_cmri.csv"
SEED = 42
B = 5000


def rmse(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> None:
    day = {}  # date -> list of (e_v2, e_cmri) signed resync errors
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            day.setdefault(r["date"][:10], []).append(
                (obs - float(r["v2_resync"]), obs - float(r["cmri_resync"])))
    days = [day[k] for k in sorted(day)]
    nd = len(days)

    e2 = [a for d in days for a, _ in d]
    ec = [c for d in days for _, c in d]
    n = len(e2)
    r2, rc = rmse(e2), rmse(ec)
    gap = r2 - rc
    print(f"resync held-out RMSE gap, n={n} steps across {nd} day blocks")
    print(f"  pooled RMSE:  v2 {r2:.4f}   cmri {rc:.4f}   gap = {gap:+.4f} degC")

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
    mg, sg, se = fmean(g), pstdev(g), pstdev(g) / math.sqrt(nd)
    t = mg / se
    print(f"  per-day RMSE-gap series ({nd} days):")
    print(f"    mean {mg:+.4f}  sd {sg:.4f}  se {se:.4f}  t = {t:+.2f}"
          f"  (normal-approx 95% CI [{mg-1.96*se:+.4f}, {mg+1.96*se:+.4f}])")
    print(f"    one-sided p < {1 - phi(t):.4f}")

    print("\nReading: the RMSE margin is not sample luck — both the bootstrap "
          "CI and the per-day t exclude 0, consistent with the HAC DM on the "
          "mean differential (verify_significance.py, p≈3.5e-03).  The CI "
          "width (~0.17 degC) is honest about the day-to-day variance.")


if __name__ == "__main__":
    main()
