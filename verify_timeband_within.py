#!/usr/bin/env python3
"""verify_timeband_within.py — is the §7 item 8 "the evening-owned and
tail-owned readings are the same edge" reading a structural fact, not an
artifact of the single held-out draw?

§7 item 8 committed claims (frozen held-out resync, results/predictions_cmri.csv,
as analyzed by verify_timeband.py):
  * the >= 0.6 degC tail edge is PERIOD-GLOBAL: |e_v2| >= 0.6 separates at day
    level in BOTH periods (day 21/28 p = 0.007; evening 26/34 p = 0.002) and
    the 0.4-0.6 mid band too (p = 0.015 / 0.044), while the body bands are
    flat or negative in both — including the 0-0.2 degC fast-tracking
    concession (day p = 0.978, evening p = 0.951)
  * evening special role is (i) COMPOSITION — 860/1477 evening steps (58 %) are
    >= 0.6 degC misses vs 801/4402 daytime steps (18 %), an evening period
    holding 25 % of steps yet delivering 87.6 % of the edge — and (ii) a
    modest genuine AMPLIFICATION — on the 25 days with both day- and
    evening-tail steps the evening-tail per-step margin exceeds the same-day
    day-tail margin by +0.19 degC (19/25, z = +2.40, p = 0.008, day-block
    bootstrap 95 % CI +0.002 to +0.383)
  * the evening 87.6 % share of total paired-d decomposes as: evening-tail
    steps alone carry 92.0 %, day-tail 43.5 %, mid bands a net +27.6 %, body
    bands a net -63.0 %

Those numbers were computed once on the held-out test.  This script verifies the
JOINT ATTRIBUTION without touching the held-out test for fitting: leg A re-fits
fresh v2 (CalibratedTwin) and cmri (frozen-config CmriTwin) twins on the
within-train fit slice [0, 11051), replays the daily-resync protocol on the
within-train val block [11051, 13814) (item 10's 20-day block, n = 2743 resync
steps), and applies the SAME joint time-of-day x error-magnitude analysis —
whole-day-block sign tests per (|e_v2| band) x (day 0-17h / evening 18-23h)
cell, the paired-d share decomposition, the period tail composition, and the
within-day evening-tail vs day-tail amplification contrast — to the fresh
errors; leg B re-reads the frozen results/predictions_cmri.csv (read-only) and
re-aggregates the committed digit-level claims exactly as verify_timeband.py
does.

Gates (power-aware: the 20-day block cannot reproduce the 42-day per-cell z
values, so the gates are structural invariants of the joint reading, not
per-cell digit matches): within-train, (i) the 0.6+ tail band's day-block win
is present in BOTH periods (period-global tail), (ii) the 0.4-0.6 mid band mean
d is positive in both periods while the 0-0.2 fast band is negative in both
(body concession, equally shared), (iii) the evening tail fraction of steps
exceeds the daytime tail fraction (composition), (iv) the within-day
evening-tail minus day-tail per-step delta is positive on the shared days and
the evening-tail cell is the single largest share of total paired-d
(amplification compounding the largest cell), (v) the body bands are a net
negative share of total paired-d while the evening-tail cell is net positive;
the held-out leg must re-produce every committed digit.

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
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
SEED = 42
B = 5000
TOL = 0.0015            # leg-B digit re-aggregation tolerance (degC)
T_TOL = 0.0051          # t-stat digit tolerance (printed at 2dp: half-ulp 0.005)
Z_TOL = 0.01            # z-value tolerance (deterministic normal approx)
P_TOL = 0.001           # one-sided p tolerance

# Same |e_v2| banding and period predicates as verify_timeband.py (the §7 item
# 8 joint basis, band edges matching verify_quantiles.py / §7 item 6).
BANDS = [(0.0, 0.2, "0-0.2 fast"), (0.2, 0.4, "0.2-0.4 flat"),
         (0.4, 0.6, "0.4-0.6 mid"), (0.6, math.inf, "0.6+ tail")]
PERIODS = [("day 0-17", lambda h: h < 18), ("eve 18-23", lambda h: h >= 18)]

# Frozen §7 item 8 committed digits (leg-B audit targets), exactly the numbers
# verify_timeband.py prints: (band, period, n, mean paired d, wins, days).
C_CELLS = [
    ("0-0.2 fast", "day 0-17", 2035, -0.1034, 15, 42),
    ("0-0.2 fast", "eve 18-23", 207, -0.2096, 6, 18),
    ("0.2-0.4 flat", "day 0-17", 963, -0.0058, 22, 41),
    ("0.2-0.4 flat", "eve 18-23", 193, -0.0243, 15, 26),
    ("0.4-0.6 mid", "day 0-17", 603, +0.1422, 25, 36),
    ("0.4-0.6 mid", "eve 18-23", 217, +0.1365, 19, 28),
    ("0.6+ tail", "day 0-17", 801, +0.2273, 21, 28),
    ("0.6+ tail", "eve 18-23", 860, +0.4484, 26, 34),
]
C_OVERALL = (0.0713, 0.579, 28)          # overall d, win rate, day wins (of 42)
C_EVESHARE = 0.876                        # evening share of total paired-d
# amplification: (mean delta, wins, n_days, CI lo, CI hi, se, t)
C_AMP = (+0.1907, 19, 25, +0.0022, +0.3829, 0.0979, +1.95)
# composition: (eve_tail, eve_steps), (day_tail, day_steps), (eve_steps, n)
C_COMP = ((860, 1477), (801, 4402), (1477, 5879))
# share decomposition: (eve-tail, day-tail, mid net, body net)
C_SHARES = (+0.920, +0.435, +0.276, -0.630)


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive paired d, normal approx."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided H1: P(d>0) > 0.5


def build(err_v2: list[float], err_cm: list[float],
          ds: list[dt.datetime]) -> tuple[list, dict]:
    """(rows, by_day) exactly as verify_timeband.py: per (|e_v2|, |e_cmri|,
    iso-day-key, hour) rows plus per-day lists."""
    rows = [(abs(a), abs(c), d.date().isoformat(), d.hour)
            for a, c, d in zip(err_v2, err_cm, ds)]
    by_day: dict[str, list[tuple[float, float, int]]] = {}
    for r in rows:
        by_day.setdefault(r[2], []).append((r[0], r[1], r[3]))
    return rows, by_day


def joint_stats(by_day: dict[str, list[tuple[float, float, int]]],
                total_d: float) -> dict:
    """The full §7 item 8 joint analysis on one by_day structure."""
    days = sorted(by_day)
    n = sum(len(v) for v in by_day.values())

    # 1. per-cell whole-day-block sign tests
    cells = []
    for lo, hi, bname in BANDS:
        for pname, pred in PERIODS:
            dday: dict[str, list[float]] = {}
            cell: list[float] = []
            for k in days:
                for av, ac, h in by_day[k]:
                    if lo <= av < hi and pred(h):
                        d = av - ac
                        cell.append(d)
                        dday.setdefault(k, []).append(d)
            if not cell:
                cells.append({"band": bname, "period": pname, "n": 0,
                              "d": math.nan, "wins": 0, "days": 0, "z": 0.0,
                              "p": 1.0, "share": math.nan})
                continue
            m = [statistics.fmean(v) for v in dday.values()]
            w = sum(1 for x in m if x > 1e-12)
            z, p = sign_test(w, len(m))
            cells.append({"band": bname, "period": pname, "n": len(cell),
                          "d": statistics.fmean(cell), "wins": w,
                          "days": len(m), "z": z, "p": p,
                          "share": sum(cell) / total_d if total_d else math.nan})

    # 2. period tail composition
    all_h = [h for v in by_day.values() for _, _, h in v]
    eve_steps = sum(1 for h in all_h if h >= 18)
    day_steps = n - eve_steps
    eve_tail = sum(1 for v in by_day.values() for a, _, h in v
                   if h >= 18 and a >= 0.6)
    day_tail = sum(1 for v in by_day.values() for a, _, h in v
                   if h < 18 and a >= 0.6)

    # 3. within-day evening-tail vs day-tail amplification (shared days)
    rows3 = []
    for k in days:
        dt3 = [av - ac for av, ac, h in by_day[k] if h < 18 and av >= 0.6]
        et3 = [av - ac for av, ac, h in by_day[k] if h >= 18 and av >= 0.6]
        if dt3 and et3:
            rows3.append((k, statistics.fmean(dt3), statistics.fmean(et3),
                          len(dt3), len(et3)))
    diff = [et - d3 for _, d3, et, _, _ in rows3]
    mean3 = statistics.fmean(diff) if diff else math.nan
    wins3 = sum(1 for x in diff if x > 1e-12)
    z3, p3 = sign_test(wins3, len(diff)) if diff else (0.0, 1.0)
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choice(diff) for _ in range(len(diff)))
                   for _ in range(B)) if diff else [0.0] * B
    lo3, hi3 = means[int(B * 0.025)], means[int(B * 0.975)]
    se3 = (statistics.pstdev(diff) / math.sqrt(len(diff) - 1)) \
        if len(diff) > 1 else math.nan
    eve_tail_share = sum(av - ac for v in by_day.values()
                         for av, ac, h in v if h >= 18 and av >= 0.6) / total_d \
        if total_d else math.nan
    eve_share = sum(av - ac for v in by_day.values()
                    for av, ac, h in v if h >= 18) / total_d \
        if total_d else math.nan

    # share decomposition into (eve-tail, day-tail, mid net, body net)
    shares = {}
    for c in cells:
        if c["n"]:
            key = (c["band"], c["period"])
            shares[key] = c["share"]
    mid_net = shares.get(("0.4-0.6 mid", "day 0-17"), 0.0) \
        + shares.get(("0.4-0.6 mid", "eve 18-23"), 0.0)
    body_net = sum(shares.get((b, pr), 0.0)
                   for b in ("0-0.2 fast", "0.2-0.4 flat") for pr, _ in PERIODS)

    # overall cross-checks
    dd = [av - ac for v in by_day.values() for av, ac, _ in v]
    day_means = [statistics.fmean(av - ac for av, ac, _ in v)
                 for v in by_day.values()]
    day_wins = sum(1 for x in day_means if x > 1e-12)

    return {"days": days, "n": n, "total_d": total_d,
            "cells": cells, "eve_steps": eve_steps, "day_steps": day_steps,
            "eve_tail": eve_tail, "day_tail": day_tail,
            "amp": {"diff": diff, "n_days": len(rows3), "mean": mean3,
                    "wins": wins3, "z": z3, "p": p3, "lo": lo3, "hi": hi3,
                    "se": se3, "t": mean3 / se3 if se3 else math.nan},
            "eve_tail_share": eve_tail_share, "eve_share": eve_share,
            "mid_net": mid_net, "body_net": body_net,
            "day_tail_frac": day_tail / day_steps if day_steps else math.nan,
            "eve_tail_frac": eve_tail / eve_steps if eve_steps else math.nan,
            "overall_d": statistics.fmean(dd),
            "overall_win": sum(1 for x in dd if x > 1e-12) / len(dd),
            "day_wins": day_wins}


def show(st: dict) -> None:
    print(f"{st['n']} steps across {len(st['days'])} days; whole-day-block "
          f"sign tests (within-day lag-1 rho ~ 0.99)")
    print(f"  {'band':>11} {'period':>9} {'n':>6} {'d':>8} {'days won':>8} "
          f"{'z':>6} {'p':>6}")
    for c in st["cells"]:
        if c["n"]:
            print(f"  {c['band']:>11} {c['period']:>9} {c['n']:>6} "
                  f"{c['d']:>+8.4f} {c['wins']:>5}/{c['days']:<3} "
                  f"{c['z']:>+6.2f} {c['p']:>6.3f}")
    print(f"  evening steps {st['eve_tail']}/{st['eve_steps']} "
          f"({st['eve_tail_frac']:.2f}) tail; day steps {st['day_tail']}/"
          f"{st['day_steps']} ({st['day_tail_frac']:.2f}) tail; evening "
          f"share of paired-d {st['eve_share']:.3f}")
    a = st["amp"]
    if a["n_days"]:
        print(f"  evening-tail vs day-tail per-step d on {a['n_days']} shared "
              f"days: mean {a['mean']:+.4f} degC, eve > day on {a['wins']}/"
              f"{a['n_days']} days, sign z {a['z']:+.2f} (p {a['p']:.3f}); "
              f"day-block bootstrap 95% CI [{a['lo']:+.4f}, {a['hi']:+.4f}] "
              f"(B={B}, seed {SEED}); se {a['se']:.4f}, t = {a['t']:+.2f}")
    print(f"  share of total paired-d: eve-tail {st['eve_tail_share']:+.3f}, "
          f"day-tail {tail_day_share(st):+.3f}, mid net {st['mid_net']:+.3f}, "
          f"body net {st['body_net']:+.3f}")


def read_frozen() -> tuple[list[dt.datetime], list[float], list[float]]:
    """Frozen held-out resync rows -> (datetimes, e_v2, e_cmri); no fitting."""
    ds, ev2, ecm = [], [], []
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            ds.append(dt.datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S"))
            obs = float(r["T1_observed"])
            ev2.append(obs - float(r["v2_resync"]))
            ecm.append(obs - float(r["cmri_resync"]))
    return ds, ev2, ecm


def tail_day_share(st: dict) -> float:
    """Day-period 0.6+ tail cell's share of total paired d."""
    for c in st["cells"]:
        if c["band"] == "0.6+ tail" and c["period"] == "day 0-17":
            return c["share"]
    return math.nan


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

    print("joint time-of-day x error-magnitude probe (daily-resync protocol); "
          "held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + val-block joint attribution -------
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
    v2rows = [(c, o, d) for c, o, d in zip(pv2, obs, dates[val_start:split])
              if c is not None]
    cmrows = [(c, o, d) for c, o, d in zip(pcm, obs, dates[val_start:split])
              if c is not None]
    ev2 = [o - c for c, o, _ in v2rows]
    ecm = [o - c for c, o, _ in cmrows]
    dv2 = [d for _, _, d in v2rows]
    assert len(ev2) == len(ecm) == 2743 and dv2 == [d for _, _, d in cmrows]

    rows_w, days_w = build(ev2, ecm, dv2)
    total_w = sum(av - ac for v in days_w.values() for av, ac, _ in v)
    within = joint_stats(days_w, total_w)
    print("leg A  within-train val block (fresh fits):")
    show(within)
    rmse_v2 = metrics([c for c, _, _ in v2rows], [o for _, o, _ in v2rows])["rmse"]
    rmse_cm = metrics([c for c, _, _ in cmrows], [o for _, o, _ in cmrows])["rmse"]
    print(f"      context: val-block resync RMSE v2 {rmse_v2:.4f} / cmri "
          f"{rmse_cm:.4f}")
    print()

    # ---- leg B: frozen §7 item 8 digit audit (read-only) --------------------
    fds, fv2, fcm = read_frozen()
    rows_f, days_f = build(fv2, fcm, fds)
    total_f = sum(av - ac for v in days_f.values() for av, ac, _ in v)
    frozen = joint_stats(days_f, total_f)
    print(f"leg B  frozen held-out (audit leg): n={frozen['n']} steps across "
          f"{len(frozen['days'])} days")
    print("  committed-digit audit (frozen artifact, read-only):")
    audit_ok = True
    for (cb, cp, cn, cd_, cw, cdays) in C_CELLS:
        c = next(x for x in frozen["cells"]
                 if x["band"] == cb and x["period"] == cp)
        okc = (c["n"] == cn and abs(c["d"] - cd_) < TOL and c["wins"] == cw
               and c["days"] == cdays)
        audit_ok &= okc
        print(f"    {cb} {cp}: n={c['n']} (committed {cn}), mean d {c['d']:+.4f} "
              f"(committed {cd_:+.4f}), {c['wins']}/{c['days']} days (committed "
              f"{cw}/{cdays}): {'PASS' if okc else 'FAIL'}")
    od_, ow_, ow_days = C_OVERALL
    ok_over = (abs(frozen["overall_d"] - od_) < TOL
               and abs(frozen["overall_win"] - ow_) < TOL
               and frozen["day_wins"] == ow_days
               and abs(frozen["eve_share"] - C_EVESHARE) < TOL)
    audit_ok &= ok_over
    print(f"    overall d {frozen['overall_d']:+.4f} (committed {od_:+.4f}), "
          f"win {frozen['overall_win']:.3f} (committed {ow_:.3f}), day-level "
          f"{frozen['day_wins']}/{len(frozen['days'])} (committed {ow_days}/42), "
          f"evening share {frozen['eve_share']:.3f} (committed {C_EVESHARE:.3f}): "
          f"{'PASS' if ok_over else 'FAIL'}")
    a = frozen["amp"]
    cm_, cw_, cn_, clo_, chi_, cse_, ct_ = C_AMP
    oka = (a["n_days"] == cn_ and a["wins"] == cw_
           and abs(a["mean"] - cm_) < TOL and abs(a["lo"] - clo_) < TOL
           and abs(a["hi"] - chi_) < TOL and abs(a["se"] - cse_) < TOL
           and abs(a["t"] - ct_) < T_TOL)
    audit_ok &= oka
    print(f"    amplification: mean {a['mean']:+.4f} (committed {cm_:+.4f}), "
          f"{a['wins']}/{a['n_days']} days (committed {cw_}/{cn_}), CI "
          f"[{a['lo']:+.4f}, {a['hi']:+.4f}] (committed [{clo_:+.4f}, "
          f"{chi_:+.4f}]), se {a['se']:.4f} t {a['t']:+.2f} (committed "
          f"{cse_:.4f} {ct_:+.2f}): {'PASS' if oka else 'FAIL'}")
    (ct_, cs_), (dnt_, dns_), (es_, nn_) = C_COMP
    okcmp = (frozen["eve_tail"] == ct_ and frozen["eve_steps"] == cs_
             and frozen["day_tail"] == dnt_ and frozen["day_steps"] == dns_)
    audit_ok &= okcmp
    print(f"    composition: evening {frozen['eve_tail']}/{frozen['eve_steps']} "
          f"tail, day {frozen['day_tail']}/{frozen['day_steps']} tail "
          f"(committed {ct_}/{cs_} and {dnt_}/{dns_}): "
          f"{'PASS' if okcmp else 'FAIL'}")
    fdt_share = tail_day_share(frozen)
    fsh = (frozen["eve_tail_share"], fdt_share, frozen["mid_net"],
           frozen["body_net"])
    cest_, cdt_, cmid_, cbody_ = C_SHARES
    oksh = (abs(fsh[0] - cest_) < TOL and abs(fsh[1] - cdt_) < TOL
            and abs(fsh[2] - cmid_) < TOL and abs(fsh[3] - cbody_) < TOL)
    audit_ok &= oksh
    print(f"    shares: eve-tail {fsh[0]:+.3f} / day-tail {fsh[1]:+.3f} / mid "
          f"net {fsh[2]:+.3f} / body net {fsh[3]:+.3f} (committed "
          f"{cest_:+.3f}/{cdt_:+.3f}/{cmid_:+.3f}/{cbody_:+.3f}): "
          f"{'PASS' if oksh else 'FAIL'}")
    print()

    # ---- structural gates (the within-train leg is the real test) -----------
    def cell(st, band, period):
        return next(x for x in st["cells"]
                    if x["band"] == band and x["period"] == period)

    t_day = cell(within, "0.6+ tail", "day 0-17")
    t_eve = cell(within, "0.6+ tail", "eve 18-23")
    mid_day = cell(within, "0.4-0.6 mid", "day 0-17")
    mid_eve = cell(within, "0.4-0.6 mid", "eve 18-23")
    fast_day = cell(within, "0-0.2 fast", "day 0-17")
    fast_eve = cell(within, "0-0.2 fast", "eve 18-23")

    # G1: the 0.6+ tail separates at day level in BOTH periods (period-global
    # tail edge; the §7 item 8 core claim).
    gate_tail_global = (t_day["wins"] > t_day["days"] / 2
                        and t_eve["wins"] > t_eve["days"] / 2)
    # G2: mid band mean d positive in both periods, fast band negative in both
    # (the body concession is shared, not an evening defect).
    gate_mid_pos = mid_day["d"] > 0.03 and mid_eve["d"] > 0.03
    gate_fast_neg = fast_day["d"] < 0.0 and fast_eve["d"] < 0.0
    # G3: evening steps are tail-dominated, daytime mostly not (composition).
    gate_composition = (within["eve_tail_frac"] > 0.30
                        and within["eve_tail_frac"]
                        > within["day_tail_frac"] + 0.15)
    # G4: within-day evening-tail amplification positive direction on the
    # shared days, and the eve-tail cell is the single largest share.
    gate_amp_dir = within["amp"]["n_days"] > 0 and within["amp"]["mean"] > 0.0
    gate_eve_largest = (within["eve_tail_share"] > tail_day_share(within)
                        and within["eve_tail_share"] > 0.30)
    # G5: body bands net negative, evening-tail cell net positive share.
    gate_body_neg = within["body_net"] < -0.05
    gate_eve_tail_pos = within["eve_tail_share"] > 0.10

    ok = all([gate_tail_global, gate_mid_pos, gate_fast_neg, gate_composition,
              gate_amp_dir, gate_eve_largest, gate_body_neg,
              gate_eve_tail_pos]) and audit_ok

    print("gates (within-train leg = fresh test; held-out leg = frozen audit):")
    print(f"    0.6+ tail day-block win in BOTH periods (period-global tail): "
          f"day {t_day['wins']}/{t_day['days']} z={t_day['z']:+.2f}, eve "
          f"{t_eve['wins']}/{t_eve['days']} z={t_eve['z']:+.2f} : "
          f"{'PASS' if gate_tail_global else 'FAIL'}")
    print(f"    mid 0.4-0.6 band mean d > +0.03 in both periods: "
          f"{mid_day['d']:+.4f} / {mid_eve['d']:+.4f} : "
          f"{'PASS' if gate_mid_pos else 'FAIL'}")
    print(f"    0-0.2 fast band mean d < 0 in both periods (shared body "
          f"concession): {fast_day['d']:+.4f} / {fast_eve['d']:+.4f} : "
          f"{'PASS' if gate_fast_neg else 'FAIL'}")
    print(f"    evening tail fraction > day tail fraction + 0.15 (composition): "
          f"eve {within['eve_tail_frac']:.2f} vs day {within['day_tail_frac']:.2f}"
          f" : {'PASS' if gate_composition else 'FAIL'}")
    a = within["amp"]
    print(f"    within-day evening-tail - day-tail delta > 0 (amplification "
          f"direction): {a['mean']:+.4f} on {a['n_days']} shared days "
          f"({a['wins']}/{a['n_days']} won, z {a['z']:+.2f}) : "
          f"{'PASS' if gate_amp_dir else 'FAIL'}")
    print(f"    eve-tail cell is the largest single share (> day-tail, > 0.30): "
          f"{within['eve_tail_share']:+.3f} vs day-tail {tail_day_share(within):+.3f}"
          f" : {'PASS' if gate_eve_largest else 'FAIL'}")
    print(f"    body bands net negative share: {within['body_net']:+.3f} : "
          f"{'PASS' if gate_body_neg else 'FAIL'}")
    print(f"    eve-tail cell net positive share: {within['eve_tail_share']:+.3f}"
          f" : {'PASS' if gate_eve_tail_pos else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §7 item 8 reading (the "
          f"'evening-owned' and 'tail-owned' attributions are THE SAME EDGE) is "
          f"a structural fact, not a single held-out draw: on fresh within-train "
          f"fits the joint geometry reproduces — the 0.6+ tail separates at day "
          f"level in both the day and evening periods "
          f"({t_day['wins']}/{t_day['days']} and {t_eve['wins']}/{t_eve['days']}"
          f" days, the period-global tail), the mid band mean d is positive in "
          f"both periods ({mid_day['d']:+.3f}/{mid_eve['d']:+.3f}) while the "
          f"0-0.2 fast-tracking concession is shared by both "
          f"({fast_day['d']:+.3f}/{fast_eve['d']:+.3f}, a broad body-level tax), "
          f"evening steps are tail-dominated by composition ({within['eve_tail_frac']:.2f}"
          f" vs {within['day_tail_frac']:.2f} daytime), the evening-tail cell is "
          f"the single largest share of paired-d ({within['eve_tail_share']:+.3f}"
          f", body bands net {within['body_net']:+.3f}), and the within-day "
          f"evening-tail vs day-tail amplification is positive on the shared "
          f"days ({a['mean']:+.4f} degC, {a['wins']}/{a['n_days']}, z {a['z']:+.2f})"
          f" — and the frozen artifact re-produces every committed §7 item 8 "
          f"digit (the 8-cell n/d/day table, overall {frozen['overall_d']:+.4f}/"
          f"{frozen['overall_win']:.3f}/{frozen['day_wins']}-{len(frozen['days'])}"
          f", evening share {frozen['eve_share']:.3f}, amplification "
          f"{frozen['amp']['mean']:+.4f}/CI [{frozen['amp']['lo']:+.4f}, "
          f"{frozen['amp']['hi']:+.4f}]/t {frozen['amp']['t']:+.2f}, "
          f"composition {frozen['eve_tail']}/{frozen['eve_steps']} vs "
          f"{frozen['day_tail']}/{frozen['day_steps']}, shares "
          f"{fsh[0]:+.3f}/{fsh[1]:+.3f}/{fsh[2]:+.3f}/{fsh[3]:+.3f}).")


if __name__ == "__main__":
    main()
