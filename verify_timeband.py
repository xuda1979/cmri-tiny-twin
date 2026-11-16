#!/usr/bin/env python3
"""verify_timeband.py — joint time-of-day x error-magnitude location of the resync edge.

Pure-Python re-aggregation of the FROZEN held-out resync predictions
(results/predictions_cmri.csv, single-touch artifact).  No new evaluation, no
refitting — the held-out test is still touched exactly once.

The prior diagnostics pinned WHERE in time and WHERE in the error distribution
cmri's +0.0713 degC resync edge lives, but treated the two axes separately:
   - verify_residuals.py (§7 item 5): the edge is evening-owned — clock band
     18-23h is the only band that separates at day level (30/42 days, p = 0.004)
     and carries 87.6% of total paired-d;
   - verify_quantiles.py (§7 item 6): the edge is tail-owned — |e_v2| >= 0.6
     steps are the only ones where cmri separates at day level (mean d +0.34),
     and the 0-0.2 degC fast-tracking band is where it is day-level *worse*
     (13/42, z = -2.62).

This script asks the joint question those two leave open: does the edge live in
the evening *because* the evening error distribution is dominated by large
misses (composition), or does the evening *amplify* the same misses?  It builds
the 4 (|e_v2| band, same edges as verify_quantiles) x 2 (day 0-17h / evening
18-23h) table of whole-day-block sign tests, then compares the evening-tail
per-step edge against the same-day day-tail per-step edge on the days that have
both (25/42) — an honest within-day contrast of the two periods.

Finding: the tail edge is *global* — |e_v2| >= 0.6 separates at day level in
BOTH periods (day 21/28, p = 0.007; evening 26/34, p = 0.002) and the 0.4-0.6
mid band too (day p = 0.015, evening p = 0.044), while the body bands are flat
or negative in both (0-0.2 concession day p = 0.98 / evening p = 0.95; 0.2-0.4
flat both).  The evening's special role is (i) *composition* — 58% of evening
steps are |e_v2| >= 0.6 vs 18% daytime — and (ii) a modest genuine
*amplification* — on the 25 days with both, evening-tail per-step edge exceeds
day-tail by +0.19 degC (19/25 days, z = +2.40, p = 0.008; bootstrap CI
[+0.002, +0.383]).  The evening 87.6% share of total paired-d decomposes as:
evening-tail steps alone carry 92.0% of it; day-tail 43.5%; the mid bands a
net 27.6%; and the body bands a net -63.0% (the concession).  The edge is thus
evening-owned *by composition*, tail-owned *by band*, and the residual evening
effect is an amplification of an already-present tail edge.

Sections:
  0. Cross-check against committed numbers (overall d +0.0713, win 0.579,
     day-level 28/42; evening share 0.876).
  1. Joint (|e_v2| band) x (day/evening period) whole-day-block sign tests.
  2. Bulk decomposition: each cell's share of total paired-d, plus the tail
     composition of each period.
  3. Within-day evening-tail amplification on the 25 shared days (sign test +
     bootstrap CI on eve-tail minus day-tail per-step d).
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


def main() -> None:
    # per-day step rows: (|e_v2|, |e_cmri|, hour-of-day).
    by_day: dict[str, list[tuple[float, float, int]]] = {}
    all_d: list[float] = []
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            av = abs(obs - float(r["v2_resync"]))
            ac = abs(obs - float(r["cmri_resync"]))
            hour = int(r["date"][11:13])
            key = r["date"][:10]
            by_day.setdefault(key, []).append((av, ac, hour))
            all_d.append(av - ac)
    days = sorted(by_day)
    n = len(all_d)
    totalD = sum(all_d)
    overall_d = statistics.fmean(all_d)
    overall_win = sum(1 for x in all_d if x > 1e-12) / n
    print(f"time-of-day x error-magnitude joint probe on the frozen resync "
          f"predictions, n={n} steps across {len(days)} days")

    # 0. Cross-checks against committed §5.5/§7 numbers.
    day_means = [statistics.fmean(av - ac for av, ac, _ in v)
                 for v in by_day.values()]
    day_wins = sum(1 for x in day_means if x > 1e-12)
    eve_share = sum(a - c for v in by_day.values()
                    for a, c, h in v if h >= 18) / totalD
    ok = (abs(overall_d - 0.0713) < 0.0001 and abs(overall_win - 0.579) < 0.001
          and day_wins == 28 and abs(eve_share - 0.876) < 0.002)
    print(f"  cross-check: overall d {overall_d:+.4f} win {overall_win:.3f} "
          f"day-level {day_wins}/{len(days)} (committed +0.0713 / 0.579 / "
          f"28-42), evening share of paired-d {eve_share:.3f} (committed 0.876, "
          f"verify_residuals) — {'PASS' if ok else 'FAIL'}")

    # 1. Joint band x period whole-day-block sign test.  Band edges match
    #    verify_quantiles.py exactly so the marginal bands agree with §7 item 6.
    bands = [(0.0, 0.2, "0-0.2 fast"), (0.2, 0.4, "0.2-0.4 flat"),
             (0.4, 0.6, "0.4-0.6 mid"), (0.6, math.inf, "0.6+ tail")]
    print("\n1. whole-day-block sign test per (|e_v2| band) x (period):")
    print(f"   {'band':>11} {'period':>9} {'n':>6} {'d':>8} {'days won':>8} "
          f"{'z':>6} {'p':>6}")
    for lo, hi, bname in bands:
        for pname, pred in (("day 0-17", lambda h: h < 18),
                            ("eve 18-23", lambda h: h >= 18)):
            dday: dict[str, list[float]] = {}
            cell: list[float] = []
            for k in days:
                for av, ac, h in by_day[k]:
                    if lo <= av < hi and pred(h):
                        d = av - ac
                        cell.append(d)
                        dday.setdefault(k, []).append(d)
            if not cell:
                continue
            m = [statistics.fmean(v) for v in dday.values()]
            w = sum(1 for x in m if x > 1e-12)
            z, p = sign_test(w, len(m))
            print(f"   {bname:>11} {pname:>9} {len(cell):>6} "
                  f"{statistics.fmean(cell):>+8.4f} {w:>5}/{len(m):<3} "
                  f"{z:>+6.2f} {p:>6.3f}")

    # 2. Bulk decomposition of total paired-d by cell, and period tail share.
    print("\n2. share of total paired-d + period tail composition:")
    for lo, hi, bname in bands:
        for pname, pred in (("day 0-17", lambda h: h < 18),
                            ("eve 18-23", lambda h: h >= 18)):
            cell = [av - ac for v in by_day.values() for av, ac, h in v
                    if lo <= av < hi and pred(h)]
            if cell:
                print(f"   {bname:>11} {pname:>9}  {sum(cell)/totalD:+.3f}  "
                      f"({statistics.fmean(cell):+.4f} avg per step)")
    eve_steps = sum(1 for v in by_day.values() for _, _, h in v if h >= 18)
    eve_tail = sum(1 for v in by_day.values() for a, _, h in v
                   if h >= 18 and a >= 0.6)
    day_steps = n - eve_steps
    day_tail = sum(1 for v in by_day.values() for a, _, h in v
                   if h < 18 and a >= 0.6)
    print(f"   evening steps are {eve_tail}/{eve_steps} ({eve_tail/eve_steps:.2f}) "
          f"tail; day steps {day_tail}/{day_steps} ({day_tail/day_steps:.2f}) "
          f"tail — the evening day-level win rides on a period that is "
          f"{eve_steps}/{n:.2f} of steps yet delivers {eve_share:.3f} of the edge.")

    # 3. Within-day evening-tail amplification on days with both periods' tails.
    rows = []
    for k in days:
        dt = [av - ac for av, ac, h in by_day[k] if h < 18 and av >= 0.6]
        et = [av - ac for av, ac, h in by_day[k] if h >= 18 and av >= 0.6]
        if dt and et:
            rows.append((k, statistics.fmean(dt), statistics.fmean(et),
                         len(dt), len(et)))
    diff = [et - dt for _, dt, et, _, _ in rows]
    m3 = statistics.fmean(diff)
    w3 = sum(1 for x in diff if x > 1e-12)
    z3, p3 = sign_test(w3, len(diff))
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choice(diff) for _ in range(len(diff)))
                   for _ in range(B))
    lo, hi = means[int(B * 0.025)], means[int(B * 0.975)]
    se = (statistics.pstdev(diff) / math.sqrt(len(diff) - 1)) if len(diff) > 1 else math.nan
    eve_tail_share = sum(av - ac for v in by_day.values() for av, ac, h in v
                         if h >= 18 and av >= 0.6) / totalD
    print(f"\n3. evening-tail vs day-tail per-step d on {len(rows)} shared days "
          f"(delta = eve-cell mean minus day-cell mean):")
    print(f"   mean {m3:+.4f} degC, eve > day on {w3}/{len(rows)} days, sign "
          f"z {z3:+.2f} (one-sided p {p3:.3f}); day-block bootstrap 95% CI "
          f"[{lo:+.4f}, {hi:+.4f}] (B={B}, seed {SEED}); normal-approx se "
          f"{se:.4f}, t = {m3/se:+.2f}")
    print(f"   evening-tail steps alone carry {eve_tail_share:.3f} of total "
          f"paired-d, so the amplification compounds the single largest cell.\n"
          f"\nreading: the tail edge is period-global (0.6+ separates at day "
          f"level in both periods, p = 0.007 / 0.002), so §7 item 5's "
          f"'evening-owned' and item 6's 'tail-owned' readings are the same "
          f"edge: evening hours deliver mostly large misses (composite) and "
          f"amplify them (+0.19 degC, z = +2.40).  The 0-0.2 degC fast-tracking "
          f"concession is equally shared by the two periods — a broad body-level "
          f"tax, not an evening defect.")


if __name__ == "__main__":
    main()
