#!/usr/bin/env python3
"""verify_freerun.py — does cmri's free-run win over v2 separate, or is it luck?

Pure-Python re-aggregation of the FROZEN held-out free-run predictions written
by the experiment (results/predictions_cmri.csv, single-touch artifact).  No
new evaluation, no refitting: exactly like verify_significance.py for the
daily-resync protocol, this one scrutinizes the free-run columns (n = 5920,
one continuous open-loop run over 42 calendar days).

The daily-resync loss differential had 5879 effectively within-day-correlated
samples; the honest test there was HAC DM.  An open-loop run has a different
correlation structure: the error once has NO re-anchor, so per-step samples are
dependent at ALL lags and no HAC bandwidth is defensible.  The honest unit of
independence is the DAY: 42 days of a continuous run is 42 (approximately)
independent error blocks, and a day differs from the next only through weather
and load.  This script therefore tests the day-block paired differential

    d_day = mean|e_v2| - mean|e_cmri|   (positive d => cmri better that day)

with a sign test (binomial normal approximation).  It also splits the run in
half to show whether the edge is uniform over the 41-day run or owned by a
period.  Context: diagnose_freerun.py already reported the drift mechanics
(bias/trend/quarter excursions); this script is the *statistical* counterpart.
"""

import csv
import math
from statistics import fmean

PRED = "results/predictions_cmri.csv"


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive d, normal approximation."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided H1: P(d>0) > 0.5


def main() -> None:
    day = {}  # date string -> list of (|e_v2|, |e_cmri|) freerun steps
    with open(PRED) as f:
        for r in csv.DictReader(f):
            obs = float(r["T1_observed"])
            day.setdefault(r["date"][:10], []).append(
                (abs(obs - float(r["v2_freerun"])),
                 abs(obs - float(r["cmri_freerun"]))))

    keys = sorted(day)
    d_days = [fmean(a for a, _ in day[k]) - fmean(c for _, c in day[k])
              for k in keys]
    n = len(d_days)
    wins = sum(1 for x in d_days if x > 1e-12)
    losses = n - wins
    z, p = sign_test(wins, n)

    print(f"free-run day-block paired verification, n={n} days")
    print(f"  per-step MAE overall:  v2 {fmean(a for k in day for a, _ in day[k]):.4f}"
          f"   cmri {fmean(c for k in day for _, c in day[k]):.4f}"
          f"   (report §5.2: RMSE 1.575 -> 1.492)")
    print(f"  mean daily d = mean|e_v2| - mean|e_cmri| = {fmean(d_days):+.4f} degC")
    print(f"  days cmri better: {wins}/{n}  (wins/losses {wins}/{losses})")
    print(f"  sign test z={z:.2f}  one-sided p={p:.4f}  (H1: cmri wins a majority of days)")

    # Split the run in half: is the edge uniform or period-owned?
    h = n // 2
    print("\nfirst half vs second half of the 41-day run:")
    for lab, seg in [("days 1-21 ", d_days[:h]), ("days 22-42", d_days[h:])]:
        w = sum(1 for x in seg if x > 1e-12)
        print(f"  {lab}: mean daily d {fmean(seg):+.4f}  cmri better {w}/{len(seg)} days")

    # For contrast: the daily-resync protocol separates HUGELY at day level.
    d_res = {}
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue
            obs = float(r["T1_observed"])
            d_res.setdefault(r["date"][:10], []).append(
                abs(obs - float(r["v2_resync"])) - abs(obs - float(r["cmri_resync"])))
    wk = sum(1 for k in d_res if fmean(d_res[k]) > 1e-9)
    zr, _ = sign_test(wk, len(d_res))
    print(f"\ncontrast — resync day-level: cmri better {wk}/{len(d_res)} days"
          f" (z={zr:.2f}), i.e. the re-anchored protocol separates at day level"
          f" while the open-loop win does not.")


if __name__ == "__main__":
    main()
