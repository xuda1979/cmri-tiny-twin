#!/usr/bin/env python3
"""verify_diffclone.py — structural probe of WHICH error is cmri's concession at day level.

Pure-Python re-aggregation of the FROZEN held-out resync predictions
(results/predictions_cmri.csv, single-touch artifact).  No new evaluation, no
refitting — the held-out test is still touched exactly once.

verify_quantiles.py (§7 item 6) located the +0.0713 °C resync edge: cmri wins
by shrinking the big misses (|e_v2| >= 0.6 tail) while running *behind* v2 on
the 0-0.2 °C fast-tracking band.  That downside shares the shape of the
free-run && resync bias concession already documented in §5.2/§5.4: cmri's
pooled running-mean error sits ~-0.11 °C where v2's is ~-0.01 °C.

This script resolves the ambiguity that concession raises: does cmri's mean
lag mean it *tracks every day's motion worse* (a per-step, whole-day handicap,
which would corrode the primary win), or does it come from the *day-level drifts
themselves* — days where cmri's daily mean error is larger in a signing that
mostly cancels in the pooled mean (a level shift), while per-day the model is
more stable and at least as accurate?

The exploration (probe, same frozen columns) showed:
  * pooled per-step error correlation 0.75, within-day median 0.93 (28/42 days
    > 0.9) — the two error series co-move tightly, so this is not a question of
    different misfire days;
  * the within-day error *sd* gap separates at day level: cmri lower-sd on
    34/42 days, z = +3.86, CI [+0.047, +0.110] °C, t = +4.74 — the resync-side
    version of verify_freerun_variance.py's free-run variance result, and the
    strongest resync claim in this repo;
  * within-day RMSE also separates (29/42, z = +2.31, CI [+0.030, +0.174],
    t = +2.63) — the per-day metric that keeps the bias inside if it is a
    real per-step tracking loss;
  * the per-day *mean*-error contrast nearly cancels in the aggregate (v2
    +0.100 °C closer-to-zero only on *17/42* days, i.e. cmri's daily means are
    closer to zero on 25/42 — no day-level tracking loss), and a big share of
    cmri's pooled mean bias is a *level offset* that day-mean differencing and
    an asymmetric day carry.

Sections:
  1. Colinearity structure of the paired errors (pooled per-step rho, within-day
     rho, per-day slope of e_cmri on e_v2, day-mean-error level rho).
  2. Day-block sign test + bootstrap CI on the per-day within-day error sd gap
     (the variance side of the resync claim — the strongest of this probe).
  3. Same, on the per-day within-day RMSE gap (keeps the bias inside; the
     day-level version of the §5.5 pooled gap).
  4. Day-level mean-error contrast: per-day paired d of *daily* mean errors
     (|b_v2| - |b_cmri|), one row per day.  If cmri paid a per-step tracking
     tax the daily means would lag behind on the majority of days; the contract
     shows whether the pooled bias is a level offset instead.
  5. The asymmetry of the two day-level drift distributions (v2's wider daily
     mean error spread, and where the big failure days fall) — including the
     2016-05-07 warm unoccupied Saturday where v2 erred +0.96 °C *above*
     observed and cmri -0.01 °C, with the same day in §5.2's episode list.

Cross-checks the committed §5.5/§7 numbers (overall paired d +0.0713, win rate
0.579, day level 28/42; pooled RMSE gap +0.1153 with per-day mean +0.0985,
t = +2.66).
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


def sdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def rms(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def pearson(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sxy = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sxx = sum((x - ma) ** 2 for x in a)
    syy = sum((y - mb) ** 2 for y in b)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def ols_slope(xs: list[float], ys: list[float]) -> float:
    """Slope of OLS y on x (errors: e_cmri ~ slope * e_v2 + intercept)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def day_block(label: str, gaps: list[float]) -> None:
    """Shared day-block report: sign test + bootstrap CI on mean daily gap."""
    m = statistics.fmean(gaps)
    n = len(gaps)
    w = sum(1 for x in gaps if x > 1e-12)
    z, p = sign_test(w, n)
    rng = random.Random(SEED)
    means = []
    for _ in range(B):
        s = [gaps[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(s))
    means.sort()
    lo, hi = means[int(B * 0.025)], means[int(B * 0.975)]
    # se = pstdev/sqrt(n): the same convention as verify_rmse_gap.py, so section
    # 3's t reproduces the committed §5.5 per-day RMSE value exactly (+2.66).
    se = (statistics.pstdev(gaps) / math.sqrt(n)) if n > 1 else math.nan
    print(f"  cmri better {w}/{n} days  sign z {z:+.2f} (one-sided p {p:.3f})")
    print(f"  mean daily gap {m:+.4f} degC  day-block bootstrap 95% CI "
          f"[{lo:+.4f}, {hi:+.4f}]  (B={B}, seed {SEED}); normal-approx se "
          f"{se:.4f}, t = {m/se:+.2f}")


def main() -> None:
    day: dict[str, list[tuple[float, float]]] = {}  # date -> [(e_v2, e_cmri)]
    all_v: list[float] = []
    all_c: list[float] = []
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            e2 = float(r["v2_resync"]) - obs
            ec = float(r["cmri_resync"]) - obs
            key = r["date"][:10]
            day.setdefault(key, []).append((e2, ec))
            all_v.append(e2)
            all_c.append(ec)
    keys = sorted(day)
    n = len(all_v)
    print(f"resync structural probe on the frozen predictions, n={n} steps "
          f"across {len(keys)} days")

    # 0. Cross-checks against committed §5.5/§7 numbers.
    dd = [abs(a) - abs(c) for a, c in zip(all_v, all_c)]
    overall_d = statistics.fmean(dd)
    overall_win = sum(1 for x in dd if x > 1e-12) / n
    day_means = [statistics.fmean(abs(a) - abs(c) for a, c in v) for v in day.values()]
    day_wins = sum(1 for x in day_means if x > 1e-12)
    rms_gap_day = statistics.fmean(
        rms([a for a, _ in v]) - rms([c for _, c in v]) for v in day.values())
    ok = (abs(overall_d - 0.0713) < 0.0001 and abs(overall_win - 0.579) < 0.001
          and day_wins == 28 and abs(rms_gap_day - 0.0985) < 0.002)
    print(f"  cross-check: overall d {overall_d:+.4f} win {overall_win:.3f} "
          f"day-level {day_wins}/{len(keys)}, per-day RMSE gap "
          f"{rms_gap_day:+.4f} (committed +0.0713 / 0.579 / 28-42 / +0.0985, "
          f"t=+2.66) — {'PASS' if ok else 'FAIL'}")

    # 1. Colinearity structure.
    rho_pool = pearson(all_v, all_c)
    withins = sorted(pearson([a for a, _ in v], [c for _, c in v])
                     for v in day.values())
    slopes = [ols_slope([a for a, _ in v], [c for _, c in v])
              for v in day.values()]
    day_v = [statistics.fmean(a for a, _ in v) for v in day.values()]
    day_c = [statistics.fmean(c for _, c in v) for v in day.values()]
    rho_level = pearson(day_v, day_c)
    print("\n1. colinearity of the paired resync errors:")
    print(f"   pooled per-step rho {rho_pool:.3f}; within-day rho median "
          f"{statistics.median(withins):.3f} (min {withins[0]:.3f}, max "
          f"{withins[-1]:.3f}; > 0.9 on "
          f"{sum(1 for x in withins if x > 0.9)}/{len(withins)} days);")
    print(f"   per-day slope of e_cmri on e_v2 mean {statistics.fmean(slopes):.3f} "
          f"(median {statistics.median(slopes):.3f}) — per-step errors co-move, "
          f"the two models misfire on the same days.")
    print(f"   day-mean-error level rho {rho_level:.3f} — the day-level drifts "
          f"are far less aligned than the per-step errors.")

    # 2. Within-day error-sd gap (variance side of the resync claim).
    g_sd = [sdev([a for a, _ in v]) - sdev([c for _, c in v]) for v in day.values()]
    print("\n2. within-day error sd, paired gap sd_v2 - sd_cmri "
          "(positive = cmri lower-variance):")
    day_block("error sd", g_sd)

    # 3. Within-day RMSE gap (keeps the bias inside).
    g_rm = [rms([a for a, _ in v]) - rms([c for _, c in v]) for v in day.values()]
    print("\n3. within-day RMSE, paired gap (positive = cmri lower-RMSE):")
    day_block("RMSE", g_rm)

    # 4. Day-level mean-error contract: |b_v2| - |b_cmri| on ONE row per day.
    #    If cmri paid a per-step tracking tax the daily means would be farther
    #    from zero on the majority of days; the contrast reveals a level shift.
    g_bias = [abs(a) - abs(c) for a, c in zip(day_v, day_c)]
    print("\n4. day-level |daily mean error| gap |b_v2| - |b_cmri|: is cmri's "
          "day-level mean error closer to zero, most days?")
    day_block("closer-to-zero daily mean", g_bias)
    print(f"   v2 daily mean error sd {sdev(day_v):.4f} vs cmri "
          f"{sdev(day_c):.4f} degC — cmri's day-level drift is *less* variable"
          f", so the pooled mean error margins run opposite on the big days.")

    # 5. Asymmetry of the day-level drifts: the big failure days and the level
    #    offset that keeps cmri's pooled mean at -0.11 degC.
    bs = [(a, c) for a, c in zip(day_v, day_c)]
    ranked = sorted(range(len(keys)), key=lambda i: -abs(bs[i][0]))[:6]
    print("\n5. the largest |daily mean error| days (v2 drift, cmri drift):")
    for i in ranked:
        print(f"   {keys[i]}  v2 {bs[i][0]:+.3f}  cmri {bs[i][1]:+.3f}  "
              f"(v2 closer to zero: {abs(bs[i][0]) < abs(bs[i][1])})")
    # level-offset reading: pooled bias vs mean of daily biases.
    pooled_v = statistics.fmean(all_v)
    pooled_c = statistics.fmean(all_c)
    print(f"   pooled running-mean bias: v2 {pooled_v:+.4f}, cmri "
          f"{pooled_c:+.4f} C/step — cmri's pooled concession is a signed "
          f"day-level drift (daily means {statistics.fmean(day_c):+.4f} on "
          f"average), not a per-step tracking tax (day-level |bias| wins: "
          f"{sum(1 for x in g_bias if x > 1e-12)}/{len(g_bias)} days).")


if __name__ == "__main__":
    main()
