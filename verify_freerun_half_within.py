#!/usr/bin/env python3
"""verify_freerun_half_within.py — is the §5.2 free-run EDGE-OWNERSHIP reading
("The edge is first-half-owned: days 1–21 (n=21) d = +0.344 °C, 14/21 days
cmri; days 22–42 (n=21) d = −0.387 °C, 5/21 days cmri — the last three weeks
of the open-loop run favor v2") a structural fact, not an artifact of the
single held-out draw?

§5.2 committed claims (frozen held-out free-run, verify_freerun.py on
results/predictions_cmri.csv, d_day = mean|e_v2| − mean|e_cmri|, positive d
=> cmri better that day, n = 42 days):
  * pooled mean daily d = −0.021 °C, cmri better 19/42 days, sign z = −0.77,
    one-sided p = 0.78 — the day-level win does NOT separate;
  * day-block bootstrap (B = 5000, seed 42) 95 % CI [−0.37, +0.31] °C,
    normal-approx se 0.18 °C (t = −0.12);
  * first half (days 1–21): d = +0.344 °C, cmri better 14/21 days;
    second half (days 22–42): d = −0.387 °C, cmri better 5/21 days — the
    edge is first-half-owned, "the same mid-run dip the quarter analysis
    shows (Q3 −2.10 °C/step, recovering in Q4)".

verify_freerun_variance_within (item 25) probes the variance mechanism and
verify_freerun_drift_within (item 28) the drift geometry with quarter shapes —
but neither gates the FIRST-HALF/SECOND-HALF ownership of the paired daily
differential itself. This script probes that reading WITHOUT touching the
held-out test: leg A re-fits fresh v2 (CalibratedTwin) and cmri
(CmriTwin keep=CMRI_KEEP) on the within-train fit slice [0, 11051) and replays
the open-loop free-run protocol on the within-train val block [11051, 13814)
(n = 2762 steps / 20 days), recomputing the fresh per-day paired d series and
splitting it first-10 vs last-10 days (the exact verify_freerun.py half-split
convention); leg B re-reads the frozen results/predictions_cmri.csv
(read-only) and re-aggregates every committed §5.2 honesty digit exactly as
verify_freerun.py does (seed 42, B = 5000, int(B*0.025)/int(B*0.975)
percentile CI, tie convention x > 1e-12).

Gates (power-aware structural invariants of the EDGE-OWNERSHIP reading, not
digit matches; the open-loop run on the within-train block carries the
documented drift-sign inversions of items 25/28 — positive slopes/bias and a
Q2 worst quarter — so ABSOLUTE signs of d cannot be gated; the invariant is
the RELATIVE first-vs-second-half direction):
  (i)  first-half mean daily d > second-half mean daily d on the fresh
       block — the edge (whichever direction the block's bias takes) is
       owned by the first half of the run (held: +0.344 vs −0.387);
  (ii) first-half cmri day-win count > second-half cmri day-win count on
       the fresh block (held: 14/21 vs 5/21);
  (iii) leg B re-audits the frozen artifact exactly as verify_freerun.py:
       n = 5920 / 42 days, pooled d −0.021, 19/42 wins, z −0.77 / p 0.78,
       bootstrap CI [−0.37, +0.31], se 0.18 / t −0.12, first-half
       +0.344 / 14/21, second-half −0.387 / 5/21.
Block-dependent strength deltas (the fresh pooled d and the fresh half-d
magnitudes and day counts are block-specific — 10 vs 10 days cannot pin
0.3-°C-level effects — and the within-train bias inversion shifts cmri's
absolute errors) are documented, not gated.

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from statistics import fmean

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, run_protocol  # noqa: E402
from experiment_cmri import (CMRI_ALPHA, CMRI_KEEP, CMRI_T_SET, CMRI_W,  # noqa: E402
                             CMRI_W2, days_of, load_cmri_data,
                             run_protocol_cmri)
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from cmri_twin import CmriTwin, fabric_mass  # noqa: E402

VALID_FRAC = 0.2
PRED = "results/predictions_cmri.csv"
SEED = 42
B = 5000
C_N = 5920                          # frozen free-run evaluable steps (42 days)
C_N_DAYS = 42

# committed §5.2 honesty digits (leg-B audit targets), exactly what
# verify_freerun.py prints on the frozen results/predictions_cmri.csv
C_POOL = -0.021                     # 3dp: mean daily d
C_WINS = 19                         # exact: days cmri better
C_Z = -0.77                         # 2dp: sign z
C_P = 0.78                          # 2dp: one-sided p (H1: cmri majority)
C_CI = (-0.37, 0.31)                # 2dp: bootstrap 95 % CI (°C)
C_SE = 0.18                         # 2dp: normal-approx se (°C)
C_T = -0.12                         # 2dp: t statistic
C_FIRST_D = 0.344                   # 3dp: first-half d (days 1–21)
C_FIRST_W = 14                      # exact: first-half days cmri better (of 21)
C_SECOND_D = -0.387                 # 3dp: second-half d (days 22–42)
C_SECOND_W = 5                      # exact: second-half days cmri better (of 21)

T_D = 0.0005                        # 3dp-digit tolerance
T_2 = 0.005                         # 2dp-digit tolerance


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive d, normal approximation —
    the exact verify_freerun.py convention (continuity correction)."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd
    return z, 1.0 - phi(z)


def day_series(pred_v2, pred_cm, obs, step_dates) -> tuple[dict, list, list]:
    """Group per-step |e| pairs by calendar day; return (day dict, sorted
    keys, per-day d = mean|e_v2| − mean|e_cmri| list aligned to sorted keys)."""
    day: dict[str, list[tuple[float, float]]] = {}
    for c2, cc, o, ds in zip(pred_v2, pred_cm, obs, step_dates):
        day.setdefault(ds.date().isoformat(), []).append(
            (abs(o - c2), abs(o - cc)))
    keys = sorted(day)
    d_days = [fmean(a for a, _ in day[k]) - fmean(c for _, c in day[k])
              for k in keys]
    return day, keys, d_days


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

    print("structural probe of the §5.2 free-run edge-OWNERSHIP reading "
          "(the paired daily differential d = mean|e_v2| − mean|e_cmri| is "
          "first-half-owned: days 1–21 +0.344 °C 14/21 vs days 22–42 −0.387 "
          "°C 5/21 held-out, while the pooled −0.021 °C / 19/42 stays "
          "unseparated); held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days); open-loop "
          f"free-run replay, single pass, no daily resync")
    print()

    # ---- leg A: fresh within-train fits + open-loop protocol replay ----------
    fs = slice(0, val_start)
    v2 = CalibratedTwin()
    v2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    cm = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    cm.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
           ghi[fs], tm[fs], tm2[fs])

    recs_v2 = list(zip(T, t_out, dew, load, lux))[val_start:split]
    recs_cm = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm2))[val_start:split]
    obs = T[val_start + 1:split]
    pv2 = run_protocol(v2, T[val_start], recs_v2, None)[1:]
    pcm = run_protocol_cmri(cm, (T[val_start], Th[val_start]), recs_cm, None)[1:]
    assert len(pv2) == len(pcm) == len(obs) == split - val_start - 1
    _, keys_a, d_a = day_series(pv2, pcm, obs, dates[val_start + 1:split])
    nda = len(d_a)
    assert nda == 20
    win_a = sum(1 for x in d_a if x > 1e-12)
    pooled_a = fmean(d_a)
    ha = nda // 2
    fda = fmean(d_a[:ha])
    sda = fmean(d_a[ha:])
    fwa = sum(1 for x in d_a[:ha] if x > 1e-12)
    swa = sum(1 for x in d_a[ha:] if x > 1e-12)

    print(f"leg A  within-train val block (fresh fits; open-loop replay, "
          f"n = {len(obs)} steps / {nda} days): d_day per calendar day, "
          f"split first-{ha} vs last-{ha} days (mirrors verify_freerun.py's "
          f"half-split)")
    print(f"  pooled mean daily d = {pooled_a:+.4f} °C, cmri better "
          f"{win_a}/{nda} days  (block delta, not gated: within-train the "
          f"open-loop run inverts to cmri-favorable bias, items 25/28)")
    print(f"  first {ha} days: d = {fda:+.4f} °C, cmri better {fwa}/{ha} days")
    print(f"  last {ha} days:  d = {sda:+.4f} °C, cmri better {swa}/{ha} days")

    # ---- leg B: frozen §5.2 honesty-digit audit (read-only) -------------------
    day: dict[str, list[tuple[float, float]]] = {}
    nstep = 0
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            obsb = float(r["T1_observed"])
            nstep += 1
            day.setdefault(r["date"][:10], []).append(
                (abs(obsb - float(r["v2_freerun"])),
                 abs(obsb - float(r["cmri_freerun"]))))
    keys = sorted(day)
    d_days = [fmean(a for a, _ in day[k]) - fmean(c for _, c in day[k])
              for k in keys]
    nb = len(d_days)
    pool = fmean(d_days)
    wins = sum(1 for x in d_days if x > 1e-12)
    z, p = sign_test(wins, nb)
    rng = random.Random(SEED)
    means = []
    for _ in range(B):
        s = [d_days[rng.randrange(nb)] for _ in range(nb)]
        means.append(fmean(s))
    means.sort()
    lo, hi = means[int(B * 0.025)], means[int(B * 0.975)]
    se = math.sqrt(sum((x - pool) ** 2 for x in d_days) / (nb - 1)) / math.sqrt(nb)
    hb = nb // 2
    fbd = fmean(d_days[:hb])
    fwb = sum(1 for x in d_days[:hb] if x > 1e-12)
    sbd = fmean(d_days[hb:])
    swb = sum(1 for x in d_days[hb:] if x > 1e-12)

    audit_ok = True
    print(f"\nleg B  frozen held-out (audit leg, {PRED}, read-only; re-"
          f"aggregated exactly as verify_freerun.py: seed {SEED}, B = {B})")
    print("  committed-digit audit (frozen artifact, read-only):")
    checks = [
        ("n steps", float(nstep), float(C_N), 0.5, None),
        ("n days", float(nb), float(C_N_DAYS), 0.5, None),
        ("pooled mean daily d (°C)", pool, C_POOL, T_D, None),
        ("days cmri better", float(wins), float(C_WINS), 0.5, None),
        ("sign z", z, C_Z, T_2, None),
        ("one-sided p", p, C_P, T_2, None),
        ("bootstrap CI lo (°C)", lo, C_CI[0], T_2, None),
        ("bootstrap CI hi (°C)", hi, C_CI[1], T_2, None),
        ("normal-approx se (°C)", se, C_SE, T_2, None),
        ("t statistic", pool / se, C_T, T_2, None),
        ("first-half d (°C)", fbd, C_FIRST_D, T_D, None),
        ("first-half days cmri better", float(fwb), float(C_FIRST_W), 0.5, None),
        ("second-half d (°C)", sbd, C_SECOND_D, T_D, None),
        ("second-half days cmri better", float(swb), float(C_SECOND_W), 0.5, None),
    ]
    for lbl, got, comm, tol, _ in checks:
        ok = abs(got - comm) <= tol
        audit_ok &= ok
        print(f"    {lbl:<28} {got:+.4f} (committed {comm:+.4f}): "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"      first half (days 1–21, n={hb}): d {fbd:+.4f}, cmri "
          f"{fwb}/{hb}; second half (days 22–42, n={hb}): d {sbd:+.4f}, "
          f"cmri {swb}/{hb}  (the committed +0.344 / 14/21 vs −0.387 / 5/21)")

    # ---- structural gates (the within-train leg is the real test) -------------
    g1 = fda > sda
    g2 = fwa > swa
    ok = g1 and g2 and audit_ok

    print()
    print("gates (within-train leg = fresh test; held-out leg = frozen audit; "
          "relative directions only — the within-train open-loop block carries "
          "the documented drift-sign/bias inversion of items 25/28, so "
          "absolute d signs are block deltas, not gates):")
    print(f"    (i)  first-half d > second-half d: {fda:+.4f} > {sda:+.4f} "
          f"within : {'PASS' if g1 else 'FAIL'}  (held +0.344 vs −0.387 — "
          f"the edge is first-half-owned)")
    print(f"    (ii) first-half cmri wins > second-half cmri wins: {fwa} > "
          f"{swa} within : {'PASS' if g2 else 'FAIL'}  (held 14/21 vs 5/21)")
    print()
    if ok:
        print(f"verdict: PASS — the §5.2 free-run edge-OWNERSHIP reading is a "
              f"structural fact, not a single held-out draw: on fresh "
              f"within-train fits the paired daily differential is again "
              f"first-half-owned (mean daily d {fda:+.4f} over the first "
              f"{ha} days vs {sda:+.4f} over the last {ha}, cmri better "
              f"{fwa}/{ha} vs {swa}/{ha} within; held +0.344/14/21 vs "
              f"−0.387/5/21) — the block's pooled level and half- magnitudes "
              f"are block-specific strength deltas (10 vs 10 days cannot pin "
              f"0.3-°C-level effects, and the within-train open-loop bias "
              f"inversion shifts cmri's absolute errors), not gates; and the "
              f"frozen artifact re-produces every committed §5.2 honesty "
              f"digit (pooled d {pool:+.3f}, cmri {wins}/{nb} days, z {z:+.2f} "
              f"p {p:.2f}, bootstrap 95 % CI [{lo:+.2f}, {hi:+.2f}], se "
              f"{se:.2f} t {pool / se:+.2f}, first-half {fbd:+.3f} {fwb}/{hb}, "
              f"second-half {sbd:+.3f} {swb}/{hb}, n = {nstep}).")
    else:
        print(f"verdict: FAIL — the §5.2 free-run edge-OWNERSHIP reading does "
              f"NOT replay on fresh within-train fits: the paired daily "
              f"differential {fda:+.4f} (first {ha} days) vs {sda:+.4f} (last "
              f"{ha} days) with cmri better {fwa}/{ha} vs {swa}/{ha} does not "
              f"reproduce the held-out first-half ownership (d_gates "
              f"{int(g1)}/{int(g2)} — see the printed gate lines), i.e. the "
              f"first-half/second-half split is a single-run feature of the "
              f"42-day held-out draw, not a structural property; the honest "
              f"bound stands as the day-block CI, and the quantile/drift "
              f"probes (items 25/28) carry the reproducible variance and "
              f"drift-geometry structure. The frozen audit leg: "
              f"{'PASS' if audit_ok else 'FAIL'}.")

    # block-delta documentation (always printed, PASS or FAIL)
    print()
    print("block-dependent strength deltas (documented, not gated): held-out "
          f"pooled d {pool:+.3f} / {wins}/{nb} days, first-half +0.344 "
          f"14/21, second-half −0.387 5/21, day-block CI [{lo:+.2f}, {hi:+.2f}] "
          f"(n = {nstep} / {nb} days); within-train pooled d {pooled_a:+.3f} / "
          f"{win_a}/{nda} days, first-half {fda:+.3f} {fwa}/{ha}, second-half "
          f"{sda:+.3f} {swa}/{ha} (n = {len(obs)} / {nda} days) — the 20-day "
          f"block splits 10 vs 10 days and carries the within-train open-loop "
          f"bias inversion of items 25/28, so magnitudes and the pooled level "
          f"are block-specific; the structural invariant gated here is the "
          f"relative first-vs-second-half direction only.")


if __name__ == "__main__":
    main()
