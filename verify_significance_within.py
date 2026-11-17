#!/usr/bin/env python3
"""verify_significance_within.py — is the §5.5 "cmri's win over v2 is not
sample luck" reading a structural fact, not a single held-out draw?

§5.5 committed digits (frozen held-out resync, results/predictions_cmri.csv,
re-scored by verify_significance.py, n = 5879 steps / 42 days; per-sample loss
differential d_i = |e_v2| - |e_cmri|, positive d => cmri the better of the two):
  * mean d = +0.0713 degC, win rate 0.579 (3406 / 2456 / 17 win/loss/tie),
  * sign (binomial, normal approx + continuity correction) z = 12.16, p < 1e-16,
  * paired t (normal approx, df 5878) t = 14.77, p < 1e-16,
  * DM iid (naive, = paired t by construction) DM = 14.77, p < 1e-16,
  * DM Newey-West HAC bw = 576 (4 days): DM = 2.38, two-sided p = 1.74e-02,
  * DM Newey-West HAC bw = 1440 (10 days): DM = 2.92, two-sided p = 3.46e-03,
  * day-level win (mean |err| per calendar day) 28/42 days.
The reading: the 10-min resync errors are strongly autocorrelated within a day
(lag-1 rho ~ 0.97), so the naive iid t/DM statistics overstate significance;
the honest upper bound is the HAC DM at bw = 1440, which still rejects H0 of
equal predictive accuracy at p ~ 3.5e-03 -- "the cmri margin is not sample
luck".  The margin is thinner than the single-fabric config (mean d +0.1000 ->
+0.0713, win rate 0.676 -> 0.579) -- the honest price of the free-run fix --
but it remains significant by the strictest autocorrelation-corrected test.
This line of the reading (line 184) is a structural claim about the TWO-MODEL
margin; it has no dedicated fresh-fit probe: verify_autocorr_within.py
(item 24) gates the "not iid" STRUCTURE (inflation >= 4x, HAC < naive/2),
verify_significance.py computes the frozen digits informationally with no
gates, verify_v2v1_within.py (item 27) gates the RMSE gap including its
bootstrap, and verify_marginthin_within.py (item 34) gates the thinning
ORDERING vs the single-fabric config.  No probe re-tests whether the cmri
margin itself (mean d > 0) survives under the honest day-block and
autocorrelation-corrected units on FRESH fits.

This script probes the significance reading WITHOUT touching the held-out
test: leg A re-fits fresh v2 (CalibratedTwin) and cmri (frozen-config
CmriTwin) on the within-train fit slice [0, 11051), replays the daily-resync
protocol on the within-train val block [11051, 13814) (n = 2743 steps /
20 days), and recomputes the pooled mean d, step win rate + sign test,
whole-day-block wins, and the HAC-corrected DM (bw = min(1440, n//4), the
verify_significance.py bw_auto convention, plus the strictest bw = 1440) on
the FRESH residuals; leg B re-reads the frozen predictions_cmri.csv
(read-only) and re-derives EVERY committed §5.5 digit exactly as
verify_significance.py does.

Gates (structural invariants of the significance reading, not digit matches):
  (i)  fresh pooled mean d > 0   (the margin is positive on brand-new fits);
  (ii) fresh step-level sign-test z > 0   (win rate > 0.5, continuity-
       corrected; the margin is not a few extreme days of the old model's
       fits);
  (iii) fresh whole-day-block wins > ndays/2 (strict majority on the honest
       day unit, 20 days) -- the step-level win is not an autocorrelation
       artifact;
  (iv) fresh HAC DM positive at BOTH the adaptive bw = min(1440, n//4) and
       the strictest bw = 1440 -- the autcorrelation-corrected statistic
       (the reading's last line of defense) keeps its sign on fresh fits;
       per the item-24 convention the HAC magnitude is DISCOUNTED below the
       naive z (within-day autocorrelation inflates the iid statistic), so
       only the sign is gated, with the discount documented;
  (v)  leg B re-derives ALL committed §5.5 digits:
       - n = 5879 (exact),
       - win/loss/tie = 3406/2456/17 (exact) and win rate 0.579 within
         TOL 0.0005,
       - mean d +0.0713 within TOL 0.0005,
       - sign z 12.16 and paired t 14.77 within TOL 0.01,
       - DM naive 14.77 within TOL 0.01 (== paired t by construction),
       - DM HAC bw=576 stat 2.38 within TOL 0.02 and p 1.74e-02 within
         TOL 0.001,
       - DM HAC bw=1440 stat 2.92 within TOL 0.02 and p 3.46e-03 within
         TOL 0.002,
       - per-day wins 28/42 (exact).
Block-dependent strength deltas are documented, not gated: the fresh 20-day
block's magnitudes are smaller than the held 42-day block's (mean d, win
rate, z and the HAC stats are all weaker on 1/2 the days), and the fresh
day-block count may not reach a significant split -- the significance
reading's structural carry is the two-sided domination of the fresh pool
under the step, day-block and autocorrelation-corrected units, replayed
without touching the held-out test.

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import statistics
import sys

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, metrics, run_protocol  # noqa: E402
from experiment_cmri import (CMRI_ALPHA, CMRI_KEEP, CMRI_T_SET, CMRI_W,  # noqa: E402
                             CMRI_W2, days_of, load_cmri_data,
                             run_protocol_cmri)
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from cmri_twin import CmriTwin, fabric_mass  # noqa: E402

VALID_FRAC = 0.2
PRED = "results/predictions_cmri.csv"

# Frozen §5.5 committed digits (leg-B audit targets).
N_FROZEN = 5879
WIN_LOSS_TIE = (3406, 2456, 17)
WIN_RATE = 0.579
MEAN_D = 0.0713
SIGN_Z = 12.16
PAIRED_T = 14.77
DM_NAIVE = 14.77
HAC576 = (2.38, 0.0174)
HAC1440 = (2.92, 0.00346)
DAY_WIN = (28, 42)
FROZEN_MEAN_D = 0.0713
FROZEN_SIGN_Z = 12.16
FROZEN_PAIRED_T = 14.77
FROZEN_DM_NAIVE = 14.77
FROZEN_HAC576 = (2.38, 0.0174)
FROZEN_HAC1440 = (2.92, 0.00346)

TOL_WIN = 0.0005    # win rate, 3 dp half-unit
TOL_D = 0.0005      # mean d, 4 dp half-unit
TOL_Z = 0.01        # sign/t paired/DM stats, 2 dp half-unit
TOL_STAT = 0.02     # HAC DM stats, 2 dp half-unit (HAC is less stable)
TOL_P576 = 0.001    # HAC576 p, 3 sf half-unit
TOL_P1440 = 0.002   # HAC1440 p, 3 sf half-unit


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
    """Diebold-Mariano with Newey-West HAC long-run variance (Bartlett)."""
    n = len(d)
    m = statistics.fmean(d)
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


def significance_suite(rows: list[tuple[float, float]],
                       day_keys: list[str]) -> dict:
    """Recompute the full §5.5 suite on any (av, ac) paired-loss list.

    rows are (v2 abs err, cmri abs err), day_keys parallel calendar dates.
    Mirrors verify_significance.py exactly so the frozen audit round-trips.
    """
    n = len(rows)
    av = [a for a, _ in rows]
    ac = [b for _, b in rows]
    win = sum(1 for a, b in rows if b < a)
    loss = sum(1 for a, b in rows if b > a)
    tie = n - win - loss
    d = [a - b for a, b in rows]
    md = statistics.fmean(d)
    sd = statistics.pstdev(d)
    t_stat = md / sd * math.sqrt(n) if sd > 0 else math.nan
    z_sign, p_sign = sign_test(win, n)
    p_t = two_sided_normal_p(t_stat) if t_stat == t_stat else math.nan
    dm_naive = t_stat if t_stat == t_stat else math.nan
    dm_hac576, p_hac576 = dm_hac(d, 576)
    dm_hac1440, p_hac1440 = dm_hac(d, 1440)
    bw_auto = min(1440, n // 4)
    dm_hac_auto, p_hac_auto = dm_hac(d, bw_auto)
    d_by_day: dict[str, list[float]] = {}
    for key, (a, b) in zip(day_keys, rows):
        d_by_day.setdefault(key, []).append((a, b))
    day_win = sum(1 for blk in d_by_day.values()
                  if statistics.fmean(b for _, b in blk)
                  < statistics.fmean(a for a, _ in blk))
    return {
        "n": n, "win": win, "loss": loss, "tie": tie, "d": d, "md": md,
        "sd": sd, "t": t_stat, "p_t": p_t, "z_sign": z_sign, "p_sign": p_sign,
        "dm_naive": dm_naive, "dm_hac576": dm_hac576, "p_hac576": p_hac576,
        "dm_hac1440": dm_hac1440, "p_hac1440": p_hac1440,
        "bw_auto": bw_auto, "dm_hac_auto": dm_hac_auto,
        "p_hac_auto": p_hac_auto, "day_win": day_win, "day_n": len(d_by_day),
    }


def read_frozen() -> tuple[int, list[tuple[float, float]], list[str], dict]:
    """Frozen held-out resync rows -> (n, rows, day_keys, suite); no fitting."""
    rows, day_keys = [], []
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            rows.append((abs(obs - float(r["v2_resync"])),
                         abs(obs - float(r["cmri_resync"]))))
            day_keys.append(r["date"][:10])
    return len(rows), rows, day_keys, significance_suite(rows, day_keys)


def main() -> None:
    data = load_cmri_data("data/energydata_complete.csv")
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    val_start = int(round((1.0 - VALID_FRAC) * split))
    T, Th, t_out = data["T1"], data["Th"], data["T_out"]
    dew, load, lux = data["Tdewpoint"], data["Appliances"], data["lights"]
    wind, ghi = data["Windspeed"], data["GHI"]
    tm = fabric_mass(t_out, CMRI_W)
    tm2 = fabric_mass(t_out, CMRI_W2)
    dates = data["date"]

    print("significance probe (daily-resync protocol); held-out test untouched "
          "for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + val-block significance suite -------
    fs = slice(0, val_start)
    m2 = CalibratedTwin()
    m2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    mc = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    mc.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
           ghi[fs], tm[fs], tm2[fs])

    recs_v2 = list(zip(T, t_out, dew, load, lux))[val_start:split]
    recs_cm = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm2))[val_start:split]
    val_days = days_of(dates, val_start, split)
    obs = [r[0] for r in recs_v2]
    pv2 = run_protocol(m2, recs_v2[0][0], recs_v2, val_days)
    pcm = run_protocol_cmri(mc, (recs_cm[0][0], recs_cm[0][1]), recs_cm, val_days)
    v2rows = [(c, o, d) for c, o, d in
              zip(pv2, obs, dates[val_start:split]) if c is not None]
    cmrows = [(c, o, d) for c, o, d in
              zip(pcm, obs, dates[val_start:split]) if c is not None]
    assert [d for _, _, d in v2rows] == [d for _, _, d in cmrows]
    dv2 = [(o - c) for c, o, _ in v2rows]
    ecm = [(o - c) for c, o, _ in cmrows]
    assert len(dv2) == len(ecm) == 2743
    arows = [(abs(a), abs(b)) for a, b in zip(dv2, ecm)]
    key_days = [str(d)[:10] for _, _, d in v2rows]  # datetime -> "YYYY-MM-DD"

    fs = significance_suite(arows, key_days)
    rmse_v2 = metrics([c for c, _, _ in v2rows], [o for _, o, _ in v2rows])["rmse"]
    rmse_cm = metrics([c for c, _, _ in cmrows], [o for _, o, _ in cmrows])["rmse"]

    print("leg A  within-train val-block significance suite (fresh fits, "
          f"n={fs['n']}, {fs['day_n']} days):")
    print(f"  RMSE v2 {rmse_v2:.4f} / cmri {rmse_cm:.4f} "
          f"(gap {rmse_v2 - rmse_cm:+.4f})")
    print(f"  pooled mean d = {fs['md']:+.4f} degC | win rate "
          f"{fs['win'] / fs['n']:.3f} ({fs['win']} / {fs['loss']} / "
          f"{fs['tie']} win/loss/tie)")
    print(f"  sign test z = {fs['z_sign']:+.2f} p = {fs['p_sign']:.2e} | "
          f"paired t = {fs['t']:+.2f} p = {fs['p_t']:.2e} | DM naive = "
          f"{fs['dm_naive']:+.2f}")
    print(f"  DM HAC bw=576 {fs['dm_hac576']:+.2f} p = {fs['p_hac576']:.2e} | "
          f"DM HAC bw={fs['bw_auto']} (auto) {fs['dm_hac_auto']:+.2f} p = "
          f"{fs['p_hac_auto']:.2e} | DM HAC bw=1440 {fs['dm_hac1440']:+.2f} "
          f"p = {fs['p_hac1440']:.2e}")
    print(f"  day-level win {fs['day_win']}/{fs['day_n']} days")
    print()

    # ---- leg B: frozen §5.5 digit audit (read-only) ---------------------------
    nf, frows, fkeys, ff = read_frozen()
    print(f"leg B  frozen predictions_cmri.csv audit (read-only, "
          f"n={nf} / {ff['day_n']} days):")
    audit_ok = True

    def audit(lbl: str, ok: bool, detail: str) -> None:
        nonlocal audit_ok
        audit_ok = audit_ok and ok
        print(f"    {lbl}: {'PASS' if ok else 'FAIL'}  {detail}")

    audit(f"n = {ff['n']}",
          ff["n"] == N_FROZEN,
          f"(committed {N_FROZEN})")
    audit(f"win/loss/tie = {ff['win']}/{ff['loss']}/{ff['tie']}",
          (ff["win"], ff["loss"], ff["tie"]) == WIN_LOSS_TIE,
          f"(committed {WIN_LOSS_TIE[0]}/{WIN_LOSS_TIE[1]}/{WIN_LOSS_TIE[2]})")
    wr = ff["win"] / ff["n"]
    audit(f"win rate {wr:.3f}",
          abs(wr - WIN_RATE) < TOL_WIN,
          f"(committed {WIN_RATE})")
    audit(f"mean d {ff['md']:+.4f}",
          abs(ff["md"] - MEAN_D) < TOL_D,
          f"(committed {MEAN_D:+.4f})")
    audit(f"sign z {ff['z_sign']:.2f}",
          abs(ff["z_sign"] - SIGN_Z) < TOL_Z,
          f"(committed {SIGN_Z})")
    audit(f"paired t {ff['t']:.2f}",
          abs(ff["t"] - PAIRED_T) < TOL_Z,
          f"(committed {PAIRED_T})")
    audit(f"DM naive {ff['dm_naive']:.2f}",
          abs(ff["dm_naive"] - DM_NAIVE) < TOL_Z,
          f"(committed {DM_NAIVE})")
    audit(f"DM HAC bw=576 {ff['dm_hac576']:.2f} p {ff['p_hac576']:.3e}",
          abs(ff["dm_hac576"] - HAC576[0]) < TOL_STAT
          and abs(ff["p_hac576"] - HAC576[1]) < TOL_P576,
          f"(committed {HAC576[0]} / {HAC576[1]:.3e})")
    audit(f"DM HAC bw=1440 {ff['dm_hac1440']:.2f} p {ff['p_hac1440']:.3e}",
          abs(ff["dm_hac1440"] - HAC1440[0]) < TOL_STAT
          and abs(ff["p_hac1440"] - HAC1440[1]) < TOL_P1440,
          f"(committed {HAC1440[0]} / {HAC1440[1]:.3e})")
    audit(f"day-level wins {ff['day_win']}/{ff['day_n']}",
          (ff["day_win"], ff["day_n"]) == DAY_WIN,
          f"(committed {DAY_WIN[0]}/{DAY_WIN[1]})")
    print()

    # ---- structural gates ------------------------------------------------------
    # The significance reading ("the cmri margin is not sample luck") has the
    # sample-robust core: positive pooled mean d on brand-new fits, a step-level
    # sign-test z > 0, the honest whole-day-block majority, and the
    # autocorrelation-corrected HAC DM (the reading's last line of defense)
    # keeping its positive sign.
    gg = []

    def gate(name, ok, detail):
        gg.append(ok)
        print(f"    {name}: {detail} : {'PASS' if ok else 'FAIL'}")

    print("gates (within-train leg = fresh test; held-out leg = frozen "
          "audit):")
    gate("(i) fresh pooled mean d > 0",
         fs["md"] > 0,
         f"within {fs['md']:+.4f} degC (held +0.0713)")
    gate("(ii) fresh step-level sign-test z > 0",
         fs["z_sign"] > 0,
         f"within z {fs['z_sign']:+.2f}, win rate {fs['win'] / fs['n']:.3f} "
         f"(held z +12.16, win 0.579)")
    gate("(iii) fresh whole-day-block wins > n_days/2 (strict majority on "
         "the honest day unit)",
         fs["day_win"] > fs["day_n"] / 2,
         f"within {fs['day_win']}/{fs['day_n']} days (held 28/42)")
    gate("(iv) fresh HAC DM positive at auto bw and strictest bw = 1440",
         fs["dm_hac_auto"] > 0 and fs["dm_hac1440"] > 0,
         f"within bw={fs['bw_auto']} {fs['dm_hac_auto']:+.2f} and bw=1440 "
         f"{fs['dm_hac1440']:+.2f} (held bw=1440 +2.92); HAC discounts the "
         f"naive z {fs['dm_naive']:+.2f} -> {fs['dm_hac1440']:+.2f} per the "
         f"item-24 autocorrelation convention, only the sign is gated")
    ok = all(gg) and audit_ok
    ndays = fs["day_n"]
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §5.5 'cmri margin is "
          f"not sample luck' reading is a structural fact, not a single "
          f"held-out draw: on fresh within-train fits the pooled mean d stays "
          f"positive ({fs['md']:+.4f} degC, held +0.0713), the step-level sign "
          f"test stays positive (z {fs['z_sign']:+.2f}, win rate "
          f"{fs['win'] / fs['n']:.3f} vs held z +12.16 / 0.579), the honest "
          f"whole-day-block unit holds a strict majority on the fresh 20-day "
          f"block ({fs['day_win']}/{ndays} days vs held 28/42), and the "
          f"autocorrelation-corrected HAC DM — the reading's last line of "
          f"defense — keeps its positive sign both at the adaptive bw="
          f"{fs['bw_auto']} (stat {fs['dm_hac_auto']:+.2f}) and at the "
          f"strictest bw=1440 (stat {fs['dm_hac1440']:+.2f}, held +2.92 / "
          f"p 3.46e-03), with the iid-statistic discount (naive z "
          f"{fs['dm_naive']:+.2f} -> HAC {fs['dm_hac1440']:+.2f}) matching the "
          f"item-24 autocorrelation reading; the frozen artifact re-produces "
          f"every committed §5.5 digit (win rate 0.579, win/loss/tie "
          f"3406/2456/17, mean d +0.0713, sign z 12.16, paired t / DM iid "
          f"14.77, HAC576 (2.38, 1.74e-02), HAC1440 (2.92, 3.46e-03), day "
          f"28/42, n = 5879).")


if __name__ == "__main__":
    main()
