#!/usr/bin/env python3
"""verify_significance.py — is cmri's win over v2 real or noise?

Pure-Python statistical tests on the paired held-out prediction errors already
stored in results/predictions_cmri.csv.  No new test evaluation happens here:
it only re-scores errors the frozen experiment wrote to disk, so it cannot be
used to hunt for a better config (the held-out test has been used once).

Daily-resync protocol, n = 5879.  The per-sample loss differential is

    d_i = |e_v2| - |e_cmri|       (positive d => cmri is the better of the two)

Reported statistics:
  - Win rate:   P(d > 0) = P(|e_cmri| < |e_v2|), with a sign test (binomial,
                normal approximation) under H0: P(d>0) = 0.5.
  - Paired t:   t on d.  With df = 5878 the t distribution is normal to
                <0.1%, so the p-value uses the normal approximation
                (exact enough, and immune to incomplete-beta underflow).
  - Diebold-Mariano: mean(d)/sd(d) * sqrt(n), N(0,1) two-sided.

H0: equal predictive accuracy.  A large positive statistic with small p means
cmri's margin is not sample luck.
"""

import csv
import math
from statistics import fmean, pstdev

PRED = "results/predictions_cmri.csv"


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_normal_p(z: float) -> float:
    return 2.0 * (1.0 - phi(abs(z)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive d, normal approximation."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided


def dm_hac(d: list[float], bw: int) -> tuple[float, float]:
    """Diebold-Mariano with Newey-West HAC long-run variance.

    The 10-min resync errors are strongly autocorrelated within a day, so the
    naive DM (which treats each step as independent) overstates significance.
    bw is the autocovariance truncation/bandwidth in steps — use one resync
    block (144 steps) plus, conservatively, two blocks.
    """
    n = len(d)
    m = fmean(d)
    g0 = sum((x - m) ** 2 for x in d) / n
    lrv = g0
    for l in range(1, bw + 1):
        gl = sum((d[i] - m) * (d[i + l] - m) for i in range(n - l)) / n
        w = 1.0 - l / (bw + 1.0)  # Bartlett kernel
        lrv += 2.0 * w * gl
    if lrv <= 0.0:
        return math.nan, math.nan
    stat = m / math.sqrt(lrv / n)
    return stat, two_sided_normal_p(stat)


def main() -> None:
    rows = []
    day_keys = []
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            rows.append((abs(obs - float(r["v2_resync"])),
                         abs(obs - float(r["cmri_resync"]))))
            day_keys.append(r["date"][:10])

    n = len(rows)

    def mae_loss(abserrs):
        return sum(abserrs) / n

    ab_v = [a for a, _ in rows]
    ab_c = [a for _, a in rows]
    win = sum(1 for av, ac in rows if ac < av)
    loss = sum(1 for av, ac in rows if ac > av)
    tie = n - win - loss
    d = [av - ac for av, ac in rows]  # absolute-loss differential

    rmse_c = math.sqrt(sum(a * a for a in ab_c) / n)
    rmse_v = math.sqrt(sum(a * a for a in ab_v) / n)

    md = fmean(d)
    sd = pstdev(d)
    t_stat = md / sd * math.sqrt(n) if sd > 0 else math.nan
    z_sign, p_sign = sign_test(win, n)
    p_t = two_sided_normal_p(t_stat) if t_stat == t_stat else math.nan

    # Resync blocks = calendar days (144 steps/day, 10-min data).  Day exists in
    # the resync protocol when day-boundary anchor rows were excluded correctly.
    d_by_day = {}
    for key, (av, ac) in zip(day_keys, rows):
        d_by_day.setdefault(key, []).append((av, ac))
    day_win = sum(1 for blk in d_by_day.values() if
                  fmean(b[1] for b in blk) < fmean(b[0] for b in blk))
    day_loss = len(d_by_day) - day_win
    d_blocks = [fmean(b[0] - b[1] for b in blk) for blk in d_by_day.values()]

    # DM with HAC long-run variance: naive DM assumes independent steps, which
    # the strong within-day autocorrelation violates.  Report the naive DM and
    # the HAC-corrected DM at bw = 576 (4 days) and bw = 1440 (10 days).
    dm_naive = t_stat if t_stat == t_stat else math.nan
    dm_naive_p = two_sided_normal_p(dm_naive) if dm_naive == dm_naive else math.nan
    dm_hac576, p_hac576 = dm_hac(d, 576)
    dm_hac1440, p_hac1440 = dm_hac(d, 1440)

    bw_auto = min(1440, n // 4)
    dm_hac_auto, p_hac_auto = dm_hac(d, bw_auto)

    print(f"resync held-out paired verification, n={n}")
    print(f"  cmri  RMSE={rmse_c:.4f}  MAE={mae_loss(ab_c):.4f}")
    print(f"  v2    RMSE={rmse_v:.4f}  MAE={mae_loss(ab_v):.4f}")
    print(f"  RMSE gap (v2 - cmri) = {rmse_v - rmse_c:+.4f} degC")
    print(f"  mean |err| differential d = {md:+.4f} degC")
    print(f"  win rate |e_cmri|<|e_v2|  = {win / n:.3f}  ({win} / {loss} / {tie} win/loss/tie)")
    print(f"  sign test                  z={z_sign:.2f}  one-sided p={p_sign:.2e}")
    print(f"  paired t (normal approx)   t={t_stat:.2f}  two-sided p={p_t:.2e}")
    print(f"  DM naive (steps iid)       DM={dm_naive:.2f}  two-sided p={dm_naive_p:.2e}")
    print(f"  DM HAC bw=576 (4 days)     DM={dm_hac576:.2f}  two-sided p={p_hac576:.2e}")
    print(f"  DM HAC bw=1440 (10 days)   DM={dm_hac1440:.2f}  two-sided p={p_hac1440:.2e}")
    print(f"  DM HAC bw=min(1440,n/4)    DM={dm_hac_auto:.2f}  two-sided p={p_hac_auto:.2e}")
    print(f"  day-level win rate         {day_win}/{len(d_by_day)} days (mean|err| per day), "
          f"block means {fmean(d_blocks):+.4f} degC")


if __name__ == "__main__":
    main()
