#!/usr/bin/env python3
"""verify_quantiles.py — is cmri's resync edge body-owned or tail-owned?

Pure-Python re-aggregation of the FROZEN held-out resync predictions written by
the experiment (results/predictions_cmri.csv, single-touch artifact).  No new
evaluation, no refitting: exactly like verify_residuals.py, this one answers a
structural question about WHERE in the error distribution cmri's win lives.

Sections:
  1. Error quantiles of |e_v2| and |e_cmri| over all 5879 resync steps, with the
     gap at each level — does cmri help everywhere, or only on the big misses?
  2. Error-magnitude band attribution: steps grouped by |e_v2| band.  Per band,
     n, mean |err| of each model, mean paired d = |e_v2|-|e_cmri|, the band's
     share of total paired-d, and a whole-day-block sign test (the only honest
     independence unit, see §7).  A "variance-driven / tail-owned" reading
     predicts the large-error band separates at day level while the body bands
     are flat.
  3. Day-block bootstrap CI (B=5000, seed 42) on the mean of the daily 95th and
     99th percentile |err| gaps — the honest bound on "cmri avoids the big
     misses".

Cross-checks the committed §5.5/§7 headline numbers (overall paired d +0.0713,
win rate 0.579, day level 28/42).
"""

import csv
import math
import random
import statistics

PRED = "results/predictions_cmri.csv"
SEED = 42
B = 5000


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive paired d, normal approx."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided H1: P(d>0) > 0.5


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile over order statistics (numpy default)."""
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    x = (n - 1) * p
    i = int(x)
    f = x - i
    return s[i] * (1 - f) + (s[i + 1] if i + 1 < n else s[i]) * f


def main() -> None:
    # (|e_v2|, |e_cmri|, date) per resync step; plus per-day step lists for the
    # day-block tail tests.
    rows: list[tuple[float, float, str]] = []
    by_day: dict[str, list[tuple[float, float]]] = {}
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            av = abs(obs - float(r["v2_resync"]))
            ac = abs(obs - float(r["cmri_resync"]))
            key = r["date"][:10]
            rows.append((av, ac, key))
            by_day.setdefault(key, []).append((av, ac))
    n = len(rows)
    ev2 = [a for a, _, _ in rows]
    ecm = [c for _, c, _ in rows]
    dd = [a - c for a, c in zip(ev2, ecm)]

    # 1. Error-quantile profile.
    print(f"resync held-out error-distribution analysis (frozen predictions), "
          f"n={n} steps across {len(by_day)} days")
    print(f"reference: overall paired d = mean|e_v2|-mean|e_cmri| = "
          f"{statistics.fmean(dd):+.4f}, win rate "
          f"{sum(1 for x in dd if x > 1e-12)/n:.3f} (committed +0.0713 / 0.579)")
    print("\nerror quantiles (degC):")
    print(f"  {'q':>6} {'v2 |e|':>9} {'cmri |e|':>9} {'gap':>8}")
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        pv, pc = percentile(ev2, q), percentile(ecm, q)
        print(f"  {100*q:>5.0f}% {pv:>9.4f} {pc:>9.4f} {pv-pc:>+8.4f}")

    # 2. Error-magnitude band attribution.  Tail bands are wider to keep
    #    enough steps for a day-block sign test.
    bands = [(0.0, 0.2, "0-0.2"), (0.2, 0.4, "0.2-0.4"),
             (0.4, 0.6, "0.4-0.6"), (0.6, 0.9, "0.6-0.9"),
             (0.9, 1.3, "0.9-1.3"), (1.3, math.inf, "1.3+")]
    TOTAL_D = sum(dd)
    print("\nerror-magnitude bands (steps grouped by |e_v2|):")
    print(f"  {'band':>9} {'n':>6} {'share':>7} {'v2':>7} {'cmri':>7} "
          f"{'d':>8} {'days won':>9} {'z':>6} {'p':>7}")
    for lo, hi, label in bands:
        idx = [i for i, a in enumerate(ev2) if lo <= a < hi]
        if not idx:
            continue
        bdd = [ev2[i] - ecm[i] for i in idx]
        bsum = sum(bdd)
        # Day-block sign test within this band (whole day = honest unit).
        dday: dict[str, list[float]] = {}
        for i in idx:
            dday.setdefault(rows[i][2], []).append(ev2[i] - ecm[i])
        m = [statistics.fmean(v) for v in dday.values()]
        w = sum(1 for x in m if x > 1e-12)
        z, p = sign_test(w, len(m))
        print(f"  {label:>9} {len(idx):>6} {bsum/TOTAL_D:>+7.3f} "
              f"{statistics.fmean(ev2[i] for i in idx):>7.3f} "
              f"{statistics.fmean(ecm[i] for i in idx):>7.3f} "
              f"{statistics.fmean(bdd):>+8.4f} {w:>4}/{len(m)}   {z:+.2f}  {p:.3f}")
    body = sum(d for i, d in enumerate(dd) if ev2[i] < 0.6)
    tail = TOTAL_D - body
    print(f"  tail (|e_v2| >= 0.6, all bands) carries {tail/TOTAL_D:.1f}x the total "
          f"paired-d ({tail:+.1f} degC of {TOTAL_D:+.1f}; the <0.6 bands are net "
          f"negative, {body:+.1f} degC)")
    print("\nday-block sign tests per band are the honest test (within-day "
          "lag-1 rho ~ 0.99): a tail-owned edge means only the large-error "
          "bands separate at day level.")

    # 2b. body vs tail pooled paired mean, for the headline split.
    body = [dd[i] for i in range(n) if ev2[i] < 0.6]
    tail = [dd[i] for i in range(n) if ev2[i] >= 0.6]
    print(f"\nbody (|e_v2| < 0.6, n={len(body)}): mean paired d "
          f"{statistics.fmean(body):+.4f}  win {sum(1 for x in body if x > 1e-12)/len(body):.3f}")
    print(f"tail (|e_v2| >= 0.6, n={len(tail)}): mean paired d "
          f"{statistics.fmean(tail):+.4f}  win {sum(1 for x in tail if x > 1e-12)/len(tail):.3f}")

    # 3. Day-block bootstrap CI on the mean of the daily 95th/99th percentile
    #    |err| gaps.  The gap is v2 - cmri at the day's quantile (positive =>
    #    cmri's tail better).
    rng = random.Random(SEED)

    def day_q_gap(q: float) -> list[float]:
        return [percentile([a for a, _ in v], q) - percentile([c for _, c in v], q)
                for v in by_day.values()]

    for q in (0.95, 0.99):
        gaps = day_q_gap(q)
        m = statistics.fmean(gaps)
        wins = sum(1 for x in gaps if x > 1e-12)
        z, p = sign_test(wins, len(gaps))
        means = []
        for _ in range(B):
            s = [gaps[rng.randrange(len(gaps))] for _ in range(len(gaps))]
            means.append(statistics.fmean(s))
        means.sort()
        lo, hi = means[int(B * 0.025)], means[int(B * 0.975)]
        se = (statistics.pstdev(gaps) / math.sqrt(len(gaps) - 1)) if len(gaps) > 1 else math.nan
        print(f"\nmean of daily {int(q*100)}th-pct |err| gap (v2 - cmri): "
              f"{m:+.4f} degC, cmri better {wins}/{len(gaps)} days, sign z {z:+.2f} "
              f"p {p:.3f}; day-block bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}] "
              f"(B={B}, seed {SEED}); normal-approx se {se:.4f}, t = {m/se:+.2f}")

    # Cross-checks against committed numbers.
    overall_d = statistics.fmean(dd)
    overall_win = sum(1 for x in dd if x > 1e-12) / n
    day_means = [statistics.fmean(a - c for a, c in v) for v in by_day.values()]
    day_wins = sum(1 for x in day_means if x > 1e-12)
    ok = (abs(overall_d - 0.0713) < 0.0001 and abs(overall_win - 0.579) < 0.001
          and day_wins == 28)
    print(f"\ncross-check: overall d {overall_d:+.4f} win {overall_win:.3f} "
          f"day-level {day_wins}/{len(by_day)} — "
          f"{'PASS' if ok else 'FAIL'} against committed +0.0713/0.579/28-42")


if __name__ == "__main__":
    main()
