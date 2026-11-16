#!/usr/bin/env python3
"""verify_horizon.py — where does cmri's edge over v2 actually live?

Pure-Python, reads ONLY the already-frozen held-out predictions written to
results/predictions_cmri.csv (resync protocol).  No new test evaluation, no
model fitting: it re-aggregates the single frozen evaluation by steps-since-
day-anchor so each (day, step-index) is an unweighted paired comparison.

The 10-min daily-resync protocol re-anchors every calendar day, so the step
index within each day is the forecast horizon: step 0 is 1 model step past the
re-anchored observation, step 143 is the last step of a 144-step day.  At short
horizons both twins mostly track the anchor; at long horizons the coupled two-
zone structure (cmri) is expected to diverge less than the single-zone v2.

For each horizon h: mean |err| of cmri and v2 over all days, and the mean
paired differential d_h = |e_v2| - |e_cmri| (positive => cmri better).  Also
day-mean error (each calendar day a single block, averaged over the day) and
day win/loss counts, plus horizon-split sign tests (binomial, normal approx).
"""

import csv
import math
import statistics

PRED = "results/predictions_cmri.csv"
STEPS_PER_DAY = 144


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided


def main() -> None:
    # Gather (horizon, day, |e_v2|, |e_cmri|) from the frozen resync predictions.
    per_h = [[] for _ in range(STEPS_PER_DAY)]  # [h] -> (d, |e_v2|, |e_cmri|)
    days: dict[str, list[tuple[float, float]]] = {}
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            av = abs(obs - float(r["v2_resync"]))
            ac = abs(obs - float(r["cmri_resync"]))
            date = r["date"]
            key = date[:10]
            days.setdefault(key, []).append((av, ac))
            h = len(days[key]) - 1
            per_h[h].append((key, av, ac))

    # Horizon bands.  The first step is already 10 min past the anchor; the
    # within-day process is strongly autocorrelated, so split at 12 (2 h),
    # 36 (6 h), 72 (12 h) and 106 (17.7 h) steps.
    bands = [(0, 12, "0-12 steps (0-2 h)"),
             (12, 36, "12-36 steps (2-6 h)"),
             (36, 72, "36-72 steps (6-12 h)"),
             (72, 106, "72-106 steps (12-17.7 h)"),
             (106, STEPS_PER_DAY, "106+ steps (last 6 h)")]
    # Also a short/long split at 12 steps (2 h).
    short_long = [(0, 12, "steps 0-11 (first 2 h)"),
                  (12, STEPS_PER_DAY, "steps 12+ (after first 2 h)")]

    n_days = len(days)
    day_v2_mean = [statistics.fmean(av for av, _ in b) for b in days.values()]
    day_cm_mean = [statistics.fmean(ac for _, ac in b) for b in days.values()]
    cm_win_days = sum(1 for v2m, cmm in zip(day_v2_mean, day_cm_mean)
                      if cmm < v2m)

    print(f"resync held-out horizon analysis (frozen predictions), n_days={n_days}, "
          f"{STEPS_PER_DAY} steps/day = 24h")
    print(f"day-level: cmri wins {cm_win_days}/{n_days} days "
          f"(mean |err| per day)")

    print("\nper-horizon mean |err| (degC) and paired differential (degC):")
    print(f"  {'horizon':<26} {'v2':>8} {'cmri':>8} {'d=e_v2-e_cmri':>12}  days")
    for lo, hi, label in bands:
        v2, cm, dd = [], [], []
        for h in range(lo, hi):
            for key, av, ac in per_h[h]:
                v2.append(av); cm.append(ac); dd.append(av - ac)
        if not v2:
            continue
        days_in_band = {key for h in range(lo, hi) for key, _, _ in per_h[h]}
        win = sum(1 for x in dd if x > 1e-12)
        loss = sum(1 for x in dd if x < -1e-12)
        print(f"  {label:<26} {statistics.fmean(v2):>8.4f} {statistics.fmean(cm):>8.4f} "
              f"{statistics.fmean(dd):>+12.4f}  {len(days_in_band)} days "
              f"(w/l {win}/{loss})")

    print("\nshort vs long horizon (sign test on paired d within each band):")
    for lo, hi, label in short_long:
        v2, cm, dd = [], [], []
        for h in range(lo, hi):
            for key, av, ac in per_h[h]:
                v2.append(av); cm.append(ac); dd.append(av - ac)
        win = sum(1 for x in dd if x > 1e-12)
        loss = sum(1 for x in dd if x < -1e-12)
        z, p = sign_test(win, win + loss)
        print(f"  {label:<30} n={win+loss:<6} mean d={statistics.fmean(dd):+.4f} "
              f"win rate={win/(win+loss):.3f}  sign z={z:+.2f}  one-sided p={p:.2e}")

    # Honest day-level test per band: the step-level sign tests above overstate
    # significance because within-day errors have lag-1 rho ~ 0.97 (rcf. §7 item 1),
    # so the report's discipline is to resample whole calendar days.  For each
    # band, per-day mean paired d = mean|e_v2| - mean|e_cmri| over that band's
    # steps, then a binomial sign test on which days have positive d.
    print("\nday-block sign test per band (whole-day blocks = honest independence unit):")
    for lo, hi, label in bands:
        by_day: dict[str, list[float]] = {}
        for h in range(lo, hi):
            for key, av, ac in per_h[h]:
                by_day.setdefault(key, []).append(av - ac)
        if not by_day:
            continue
        m = [statistics.fmean(dd) for dd in by_day.values()]
        wins = sum(1 for x in m if x > 1e-12)
        z, p = sign_test(wins, len(m))
        print(f"  {label:<26} {len(m):>3} days  cmri better {wins:>2}  "
              f"mean day d {statistics.fmean(m):+.4f}  sign z {z:+.2f}  "
              f"one-sided p {p:.3f}")

    # Full-day block comparison: mean |err| over the whole day, one block per day.
    day_d = [statistics.fmean(av for av, _ in b) - statistics.fmean(ac for _, ac in b)
             for b in days.values()]
    day_v2 = [statistics.fmean(av for av, _ in b) for b in days.values()]
    day_cm = [statistics.fmean(ac for _, ac in b) for b in days.values()]
    print(f"\nday blocks (per-day mean |err|): cmri {statistics.fmean(day_cm):.4f} vs "
          f"v2 {statistics.fmean(day_v2):.4f} degC; mean diff {statistics.fmean(day_d):+.4f}, "
          f"win rate {sum(1 for x in day_d if x > 1e-12)}/{n_days}")


if __name__ == "__main__":
    main()
