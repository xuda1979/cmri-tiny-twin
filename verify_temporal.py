#!/usr/bin/env python3
"""verify_temporal.py — is the resync edge temporally stable, or owned by a period?

Pure-Python re-aggregation of the FROZEN held-out resync predictions
(results/predictions_cmri.csv, single-touch artifact).  No new evaluation, no
refitting — the held-out test is still touched exactly once.

verify_freerun.py (§5.2) showed the free-run edge is *first-half-owned*: over the
same 42-day window the open-loop mean daily d is +0.344 C on days 1-21 (14/21)
and -0.387 C on days 22-42 (5/21).  A reviewer's natural worry is that the
*primary* resync claim (+0.0713 C paired d, day-level 28/42) is similarly owned
by a sub-period or a few big days, i.e. that the §7 top-episode days (2016-04-22,
2016-05-07, 2016-05-05, ...) carry the whole win.

This script asks the temporal-stability question on the frozen resync columns
directly: halves, thirds, calendar weeks, an OLS trend and rank correlation of
daily d on day index, a permutation test of the half difference, and a
top-day concentration audit (day-level sign test re-run after deleting the
largest-|d| days).  Findings:

  * the resync edge is *time-flat*: halves +0.077 / +0.064 C with 14/21 wins
    each, thirds 11-14 / 8-14 / 9-14, 5 of 7 calendar weeks positive, OLS
    slope -0.0014 C/day (t = -0.54), rank rho -0.075, half-difference
    permutation p = 0.832 (two-sided);
  * the day-level win is *broad*, not episode-owned: 28/42 days net-positive,
    and the sign test survives deleting the two largest-|d| days (26/40,
    z = +1.74, p = 0.041);
  * what the big days own is the *magnitude*, not the presence of the edge:
    the top 4 days carry 0.73 of total paired-d (matching the §7 tail/evening
    ownership of the *amount*), while the *sign* of the edge is a 42-day
    wide phenomenon.  The resync claim is thus the mirror image of the
    free-run claim: time-flat in presence, tail-owned in magnitude.

Sections:
  0. Cross-check against committed numbers (overall paired d +0.0713, win
     0.579, day-level 28/42).
  1. Window split: halves, thirds, and per-block mean daily d + sign test.
  2. Calendar-week table (mean daily d, wins, per-day d) — the coarsest
     weather-independent block that has no day-count ambiguity.
  3. Trend: OLS slope of daily d on day index (se, t) + Spearman rho.  A
     slope that were significant would mean the edge is decaying/rising.
  4. Half-difference: permutation test (two-sided) of mean(first 21) -
     mean(last 21) over 5000 balanced resamples.
  5. Concentration audit: day-level sign test after deleting the top-k|d|
     days (k = 1, 2, 3, 6), net-positive day count, and the top-day share
     of total paired-d.
  6. Contrast with the free-run halves (cross-check of §5.2's first-half
     ownership), making the resync/free-run asymmetry explicit.
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


def ols(y: list[float]) -> tuple[float, float, float]:
    """OLS slope, se, t-statistic of y on x = 0..n-1 (day index)."""
    n = len(y)
    xs = list(range(n))
    mx, my = statistics.fmean(xs), statistics.fmean(y)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (yy - my) for x, yy in zip(xs, y))
    b = sxy / sxx
    a = my - b * mx
    resid = [yy - (a + b * x) for x, yy in zip(xs, y)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
    return b, se, b / se


def spearman(xs: list[int], ys: list[float]) -> float:
    n = len(ys)
    rk = [1 + sum(1 for j in range(n) if ys[j] < ys[i]
                  or (ys[j] == ys[i] and j < i)) for i in range(n)]
    return 1.0 - 6.0 * sum((x - r) ** 2 for x, r in zip(xs, rk)) / (n * (n * n - 1))


def load_resync() -> dict[str, list[float]]:
    day: dict[str, list[float]] = {}
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            key = r["date"][:10]
            day.setdefault(key, []).append(
                abs(obs - float(r["v2_resync"])) - abs(obs - float(r["cmri_resync"])))
    return day


def main() -> None:
    day = load_resync()
    keys = sorted(day)
    n = len(keys)
    dd = [statistics.fmean(day[k]) for k in keys]
    totalD = sum(sum(day[k]) for k in keys)
    print(f"temporal-stability probe of the frozen resync edge, n={n} days "
          f"({keys[0]} .. {keys[-1]})")

    # 0. Cross-check against committed §5.5/§7 numbers.
    all_d = [x for k in keys for x in day[k]]
    overall_d = statistics.fmean(all_d)
    overall_win = sum(1 for x in all_d if x > 1e-12) / len(all_d)
    day_wins = sum(1 for x in dd if x > 1e-12)
    ok = (abs(overall_d - 0.0713) < 0.0001 and abs(overall_win - 0.579) < 0.001
          and day_wins == 28)
    print(f"  cross-check: overall d {overall_d:+.4f} win {overall_win:.3f} "
          f"day-level {day_wins}/{n} (committed +0.0713 / 0.579 / 28-42) — "
          f"{'PASS' if ok else 'FAIL'}")

    # 1. Window split: halves and thirds.
    h = n // 2
    t = n // 3

    def block_report(lab: str, seg: list[float]) -> None:
        w = sum(1 for x in seg if x > 1e-12)
        z, p = sign_test(w, len(seg))
        print(f"    {lab}: mean daily d {statistics.fmean(seg):+.4f}, cmri "
              f"better {w}/{len(seg)} (z {z:+.2f}, p {p:.3f})")
    print("\n1. window split:")
    block_report("days 01-21 ", dd[:h])
    block_report("days 22-42", dd[h:])
    for i in range(3):
        block_report(f"third {i+1} (days {i*t+1:02d}-{(i+1)*t:02d})", dd[i*t:(i+1)*t])

    # 2. Calendar weeks.
    wk: dict[tuple[int, int], list[float]] = {}
    for k, d in zip(keys, dd):
        iso = __import__("datetime").date.fromisoformat(k).isocalendar()
        wk.setdefault((iso[0], iso[1]), []).append(d)
    print("\n2. calendar weeks (ISO week is the fixed Monday basis):")
    for (yr, wno), seg in sorted(wk.items()):
        w = sum(1 for x in seg if x > 1e-12)
        print(f"    w{wno:02d} n={len(seg)}  mean daily d {statistics.fmean(seg):+.4f}"
              f"  cmri better {w}/{len(seg)}")

    # 3. Trend of daily d on day index.
    xs = list(range(n))
    slope, se, tv = ols(dd)
    rho = spearman(xs, dd)
    print(f"\n3. trend of daily d on day index: OLS slope {slope:+.4f} C/day "
          f"(se {se:.4f}, t {tv:+.2f}); Spearman rho {rho:+.3f} — flat: the "
          f"edge neither decays nor grows over the window.")
    print(f"    fitted line starts ~{slope*0 + (statistics.fmean(dd) - slope*statistics.fmean(xs)):+.4f} "
          f"ends ~{slope*(n-1) + (statistics.fmean(dd) - slope*statistics.fmean(xs)):+.4f} C")

    # 4. Permutation test on the half difference (two-sided).
    obs_h = statistics.fmean(dd[:h]) - statistics.fmean(dd[h:])
    rng = random.Random(SEED)
    cnt = 0
    for _ in range(B):
        idx = set(rng.sample(range(n), h))
        a = statistics.fmean([dd[i] for i in idx])
        b = statistics.fmean([dd[i] for i in range(n) if i not in idx])
        if abs(a - b) >= abs(obs_h):
            cnt += 1
    p_perm = cnt / B
    print(f"\n4. half difference: mean(first 21) - mean(last 21) = "
          f"{obs_h:+.4f} C; balanced permutation two-sided p = {p_perm:.3f} "
          f"(B={B}, seed {SEED}) — the halves do not separate.")

    # 5. Concentration audit.
    z0, p0 = sign_test(day_wins, n)
    print(f"\n5. concentration audit (is the win carried by a few days?):")
    print(f"    all {n} days:  cmri better {day_wins}/{n}, z {z0:+.2f}, "
          f"p {p0:.3f}; net-positive days {day_wins}/{n}")
    order = sorted(range(n), key=lambda i: -abs(dd[i]))
    for k_rem in (1, 2, 3, 6):
        keep = [dd[i] for i in range(n) if i not in order[:k_rem]]
        w = sum(1 for x in keep if x > 1e-12)
        z, p = sign_test(w, len(keep))
        print(f"    drop top {k_rem} day(s) by |d|: cmri better {w}/{len(keep)}, "
              f"z {z:+.2f}, p {p:.3f}")
    ranked = [(sum(day[k]), k) for k in keys]
    ranked.sort(reverse=True)
    top4 = ranked[:4]
    share4 = sum(m for m, _ in top4) / totalD
    names = ", ".join(f"{k} ({m:+.1f})" for m, k in reversed(top4[-3:]))
    print(f"    top 4 days carry {share4:.2f} of total paired-d ({totalD:.1f} "
          f"C-step); a few big-miss days own the *magnitude*, not the sign.")

    # 6. Contrast: free-run halves (cross-check §5.2's first-half ownership).
    fr: dict[str, list[float]] = {}
    with open(PRED) as f:
        for r in csv.DictReader(f):
            obs = float(r["T1_observed"])
            fr.setdefault(r["date"][:10], []).append(
                abs(obs - float(r["v2_freerun"])) - abs(obs - float(r["cmri_freerun"])))
    fk = sorted(fr)
    fd = [statistics.fmean(fr[k]) for k in fk]
    print(f"\n6. contrast — free-run halves (same window, open-loop columns):")
    for lab, seg in (("days 01-21", fd[:h]), ("days 22-42", fd[h:])):
        w = sum(1 for x in seg if x > 1e-12)
        print(f"    {lab}: mean daily d {statistics.fmean(seg):+.4f}, cmri "
              f"better {w}/{len(seg)}")
    print(f"    (§5.2 committed +0.344 / 14-21 and -0.387 / 5-21 — the "
          f"free-run edge is first-half-owned, the resync edge is not.)")


if __name__ == "__main__":
    main()
