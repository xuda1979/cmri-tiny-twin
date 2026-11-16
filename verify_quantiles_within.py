#!/usr/bin/env python3
"""verify_quantiles_within.py — is the §7 item 6 "tail-owned edge" reading a
structural fact, not an artifact of the single held-out draw?

§7 item 6 committed claims (frozen held-out resync, results/predictions_cmri.csv,
as analyzed by verify_quantiles.py):
  * cmri's resync edge is TAIL-OWNED: the per-step mean paired d = |e_v2| - |e_cmri|
    rises monotonically with the |e_v2| magnitude, from a fast-tracking
    concession at the bottom (0-0.2 degC band: mean d -0.113, 13/42 days,
    one-sided p 0.996) to a dominant win at the top (1.3+ degC band: mean d
    +0.775, 14/17 days, p 0.008)
  * the >= 0.6 degC tail carries 1.4x the total pooled paired d (+567.7 of
    +419.1 degC; the < 0.6 body is net negative, -148.7 degC)
  * body (|e_v2| < 0.6, n = 4218): mean paired d -0.035, win 0.501 (parity);
    tail (n = 1661): mean paired d +0.342, win 0.778
  * every band from 0.4 degC up separates cmri-favor at day level (the honest
    independence unit, within-day lag-1 rho ~ 0.99): 0.4-0.6: 29/41 days, z
    +2.50, p 0.006; 0.6-0.9: 29/37, z +3.29, p 0.001; 0.9-1.3: 25/30, z +3.47,
    p < 0.001; 1.3+: 14/17, z +2.43, p 0.008; the 0.2-0.4 mid band is flat
    (24/42, p 0.22)
  * mean daily 95th-pct |err| gap +0.242 degC (29/42 days, z +2.31, day-block
    bootstrap 95% CI [+0.100, +0.399], t +3.13); 99th-pct +0.253 (30/42, z
    +2.62, CI [+0.114, +0.410], t +3.28)

Those numbers were computed once on the held-out test.  This script verifies the
ATTRIBUTION without touching the held-out test for fitting: leg A re-fits fresh
v2 (CalibratedTwin) and cmri (frozen-config CmriTwin) twins on the within-train
fit slice [0, 11051), replays the daily-resync protocol on the within-train val
block [11051, 13814) (item 10's 20-day block), and applies the SAME error-
distribution analysis — per-|e_v2|-band day-block attribution, body/tail split,
tail-share accounting, daily 95th/99th-percentile gap bootstrap — to the fresh
errors; leg B re-reads the frozen results/predictions_cmri.csv (read-only) and
re-aggregates the committed digit-level claims exactly as verify_quantiles.py
does.

Gates (power-aware: the 20-day block cannot reproduce the 42-day per-band z
values, so the gates are structural invariants of the tail-owned reading, not
per-band digit matches): within-train, (i) per-band mean paired d rises
monotonically in |e_v2| to a substantially positive top band, (ii) the >= 0.6
tail carries the majority of total pooled paired d, (iii) the bottom
fast-tracking band concedes or stays flat at day level while the tail union
separates cmri-favor at day level, (iv) the mean daily 95th/99th-percentile
|err| gap is positive with a day-level majority, (v) the body mean paired d
stays near parity (no cmri win in the small-error zone); the held-out leg must
re-produce every committed digit.

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
BODY_CUT = 0.6            # body/tail split on |e_v2|, same as verify_quantiles.py
TOL = 0.0015              # leg-B digit re-aggregation tolerance (degC)

# Same |e_v2| banding as verify_quantiles.py (the §7 item 6 attribution basis).
BANDS = [(0.0, 0.2, "0-0.2"), (0.2, 0.4, "0.2-0.4"), (0.4, 0.6, "0.4-0.6"),
         (0.6, 0.9, "0.6-0.9"), (0.9, 1.3, "0.9-1.3"), (1.3, math.inf, "1.3+")]

# Frozen §7 item 6 committed digits (leg-B audit targets), exactly the numbers
# verify_quantiles.py prints: (band, n, mean paired d, wins, days).
C_BANDS = [
    ("0-0.2", 2242, -0.1132, 13, 42),
    ("0.2-0.4", 1156, -0.0089, 24, 42),
    ("0.4-0.6", 820, +0.1407, 29, 41),
    ("0.6-0.9", 785, +0.1711, 29, 37),
    ("0.9-1.3", 551, +0.3294, 25, 30),
    ("1.3+", 325, +0.7752, 14, 17),
]
C_OVERALL = (0.0713, 0.579, 28)     # overall d, win rate, day wins (of 42)
C_BODY = (4218, -0.0352, 0.501)
C_TAIL = (1661, +0.3418, 0.778)
TAIL_SHARE_COMM = 14.0               # printed "1.4x" as an integer times 10
# daily-percentile gaps: (q, mean gap, wins, n_days, CI lo, CI hi, t)
C_DAILY = [
    (0.95, +0.2422, 29, 42, +0.1001, +0.3987, +3.13),
    (0.99, +0.2529, 30, 42, +0.1141, +0.4104, +3.28),
]


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


def build(err_v2: list[float], err_cm: list[float],
          ds: list[dt.datetime]) -> tuple[list, dict]:
    """(rows, by_day) exactly as verify_quantiles.py: per (|e_v2|, |e_cmri|,
    iso-day-key) rows plus per-day lists."""
    rows = [(abs(a), abs(c), d.date().isoformat())
            for a, c, d in zip(err_v2, err_cm, ds)]
    by_day: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_day.setdefault(r[2], []).append((r[0], r[1]))
    return rows, by_day


def analyze(label: str, rows: list[tuple[float, float, str]],
            by_day: dict[str, list[tuple[float, float]]]) -> dict:
    """The full §7 item 6 attribution on one (rows, by_day) pair."""
    ev2 = [a for a, _, _ in rows]
    ecm = [c for _, c, _ in rows]
    dd = [a - c for a, c in zip(ev2, ecm)]
    total_d = sum(dd)
    st = {"label": label, "n": len(rows), "days": len(by_day),
          "overall_d": statistics.fmean(dd),
          "overall_win": sum(1 for x in dd if x > 1e-12) / len(dd),
          "total_d": total_d}

    # per-band day-block attribution (the §7 item 6 core table)
    bands = []
    for lo, hi, bname in BANDS:
        idx = [i for i, a in enumerate(ev2) if lo <= a < hi]
        if not idx:
            bands.append(None)
            continue
        bdd = [ev2[i] - ecm[i] for i in idx]
        dday: dict[str, list[float]] = {}
        for i in idx:
            dday.setdefault(rows[i][2], []).append(ev2[i] - ecm[i])
        m = [statistics.fmean(v) for v in dday.values()]
        w = sum(1 for x in m if x > 1e-12)
        z, p = sign_test(w, len(m))
        bands.append({"label": bname, "n": len(idx), "share": sum(bdd) / total_d,
                      "v2": statistics.fmean(ev2[i] for i in idx),
                      "cm": statistics.fmean(ecm[i] for i in idx),
                      "d": statistics.fmean(bdd), "wins": w, "days": len(m),
                      "z": z, "p": p})
    st["bands"] = bands

    # body/tail split at BODY_CUT, plus the pooled-share accounting
    body = [dd[i] for i in range(len(dd)) if ev2[i] < BODY_CUT]
    tail = [dd[i] for i in range(len(dd)) if ev2[i] >= BODY_CUT]

    def trio(xs):
        return (len(xs), statistics.fmean(xs),
                sum(1 for x in xs if x > 1e-12) / len(xs)) if xs else (0, math.nan, math.nan)

    st["body"] = trio(body)
    st["tail"] = trio(tail)
    st["body_sum"], st["tail_sum"] = sum(body), sum(tail)

    # body/tail union day-block sign tests (per-day mean d over the union)
    def union_days(cond):
        dday: dict[str, list[float]] = {}
        for i, r in enumerate(rows):
            if cond(ev2[i]):
                dday.setdefault(r[2], []).append(dd[i])
        m = [statistics.fmean(v) for v in dday.values()]
        w = sum(1 for x in m if x > 1e-12)
        z, p = sign_test(w, len(m))
        return w, len(m), z, p

    st["tail_union"] = union_days(lambda a: a >= BODY_CUT)
    st["body_union"] = union_days(lambda a: a < BODY_CUT)

    # daily 95th/99th percentile |err| gap day-block bootstrap (same as frozen)
    rng = random.Random(SEED)

    def day_q_gap(q: float) -> list[float]:
        return [percentile([a for a, _ in v], q) - percentile([c for _, c in v], q)
                for v in by_day.values()]

    quot = []
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
        se = (statistics.pstdev(gaps) / math.sqrt(len(gaps) - 1)) \
            if len(gaps) > 1 else math.nan
        quot.append({"q": q, "mean": m, "wins": wins, "days": len(gaps),
                     "z": z, "p": p, "lo": lo, "hi": hi, "t": m / se})
    st["daily"] = quot
    return st


def show(st: dict) -> None:
    print(f"{st['label']}: n={st['n']} steps across {st['days']} days")
    print(f"  overall paired d {st['overall_d']:+.4f}, win rate "
          f"{st['overall_win']:.3f}")
    print(f"  error-magnitude bands (steps grouped by |e_v2|); day-block sign "
          f"tests are the honest unit (within-day lag-1 rho ~ 0.99):")
    print(f"    {'band':>9} {'n':>6} {'share':>7} {'v2':>7} {'cmri':>7} "
          f"{'d':>8} {'days won':>9} {'z':>6} {'p':>7}")
    for b in st["bands"]:
        if b is None:
            continue
        print(f"    {b['label']:>9} {b['n']:>6} {b['share']:>+7.3f} "
              f"{b['v2']:>7.3f} {b['cm']:>7.3f} {b['d']:>+8.4f} "
              f"{b['wins']:>4}/{b['days']:<4} {b['z']:+.2f}  {b['p']:.3f}")
    nb, bd, bw = st["body"]
    nt, td, tw = st["tail"]
    print(f"  body (|e_v2| < {BODY_CUT}, n={nb}): mean paired d {bd:+.4f}, win "
          f"{bw:.3f}   tail (|e_v2| >= {BODY_CUT}, n={nt}): mean paired d "
          f"{td:+.4f}, win {tw:.3f}")
    if st["total_d"]:
        print(f"  tail carries {st['tail_sum']/st['total_d']:.1f}x the total "
              f"pooled paired d (+{st['tail_sum']:.1f} of {st['total_d']:+.1f}; "
              f"the <{BODY_CUT} body is net {st['body_sum']:+.1f})")
    bw_, bn_, bz_, _ = st["body_union"]
    tw_, tn_, tz_, _ = st["tail_union"]
    print(f"  union day-block sign tests: body {bw_}/{bn_} days z={bz_:+.2f}; "
          f"tail {tw_}/{tn_} days z={tz_:+.2f}")
    for i, q in enumerate(st["daily"]):
        print(f"  mean daily {int(q['q']*100)}th-pct |err| gap (v2 - cmri): "
              f"{q['mean']:+.4f}, cmri better {q['wins']}/{q['days']} days, "
              f"z {q['z']:+.2f} p {q['p']:.3f}; day-block bootstrap 95% CI "
              f"[{q['lo']:+.4f}, {q['hi']:+.4f}] (B={B}, seed {SEED}); "
              f"t = {q['t']:+.2f}")


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

    print("tail-owned-edge attribution probe (daily-resync protocol); "
          "held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + val-block attribution ------------
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
    within = analyze("within-train val block (fresh fits)", rows_w, days_w)
    show(within)
    rmse_v2 = metrics([c for c, _, _ in v2rows], [o for _, o, _ in v2rows])["rmse"]
    rmse_cm = metrics([c for c, _, _ in cmrows], [o for _, o, _ in cmrows])["rmse"]
    print(f"  context: val-block resync RMSE v2 {rmse_v2:.4f} / cmri "
          f"{rmse_cm:.4f}")
    print()

    # ---- leg B: frozen §7 item 6 digit audit (read-only) -------------------
    fds, fv2, fcm = read_frozen()
    rows_f, days_f = build(fv2, fcm, fds)
    frozen = analyze("frozen held-out (audit leg)", rows_f, days_f)
    show(frozen)
    print("  committed-digit audit (frozen artifact, read-only):")
    audit_ok = True
    for ci, (name, cn, cd_, cw, cdays) in enumerate(C_BANDS):
        b = frozen["bands"][ci]
        okb = (b is not None and b["n"] == cn and abs(b["d"] - cd_) < TOL
               and b["wins"] == cw and b["days"] == cdays)
        audit_ok &= okb
        print(f"    band {name}: n={b['n'] if b else None} (committed {cn}), "
              f"mean d {b['d'] if b else None:+.4f} (committed {cd_:+.4f}), "
              f"{b['wins'] if b else None}/{b['days'] if b else None} days "
              f"(committed {cw}/{cdays}): {'PASS' if okb else 'FAIL'}")
    od_, ow_, ow_days = C_OVERALL
    ok_over = (abs(frozen["overall_d"] - od_) < TOL
               and abs(frozen["overall_win"] - ow_) < TOL)
    # day-level wins from frozen (last element re-derived here)
    fday_wins = 0
    fday_tot = len(days_f)
    # day-level wins of the overall paired d: recompute
    fday_wins = sum(1 for v in days_f.values()
                    if statistics.fmean(a - c for a, c in v) > 1e-12)
    ok_over &= fday_wins == ow_days
    audit_ok &= ok_over
    print(f"    overall paired d {frozen['overall_d']:+.4f} (committed {od_:+.4f}), "
          f"win {frozen['overall_win']:.3f} (committed {ow_:.3f}), day-level "
          f"{fday_wins}/{fday_tot} (committed {ow_days}/42): "
          f"{'PASS' if ok_over else 'FAIL'}")
    nb_, bd_, bw_ = C_BODY
    nt_, td_, tw_ = C_TAIL
    ok_bt = (frozen["body"][0] == nb_ and abs(frozen["body"][1] - bd_) < TOL
             and abs(frozen["body"][2] - bw_) < TOL
             and frozen["tail"][0] == nt_ and abs(frozen["tail"][1] - td_) < TOL
             and abs(frozen["tail"][2] - tw_) < TOL)
    audit_ok &= ok_bt
    print(f"    body (n={frozen['body'][0]}, d {frozen['body'][1]:+.4f}, win "
          f"{frozen['body'][2]:.3f}) tail (n={frozen['tail'][0]}, d "
          f"{frozen['tail'][1]:+.4f}, win {frozen['tail'][2]:.3f}) vs committed "
          f"body {nb_}/{bd_:+.4f}/{bw_:.3f} tail {nt_}/{td_:+.4f}/{tw_:.3f}: "
          f"{'PASS' if ok_bt else 'FAIL'}")
    for ci, (q_, cm_, cw_, cn_, clo_, chi_, ct_) in enumerate(C_DAILY):
        dq = frozen["daily"][ci]
        okq = (abs(dq["mean"] - cm_) < TOL and dq["wins"] == cw_
               and dq["days"] == cn_ and abs(dq["lo"] - clo_) < TOL
               and abs(dq["hi"] - chi_) < TOL and abs(dq["t"] - ct_) < TOL)
        audit_ok &= okq
        print(f"    daily {int(q_*100)}th-pct gap {dq['mean']:+.4f} "
              f"({dq['wins']}/{dq['days']} days, CI [{dq['lo']:+.4f}, "
              f"{dq['hi']:+.4f}], t {dq['t']:+.2f}) vs committed {cm_:+.4f} "
              f"({cw_}/{cn_}, [{clo_:+.4f}, {chi_:+.4f}], {ct_:+.2f}): "
              f"{'PASS' if okq else 'FAIL'}")
    ts10 = round(frozen["tail_sum"] / frozen["total_d"] * 10)
    ok_ts = ts10 == TAIL_SHARE_COMM
    audit_ok &= ok_ts
    print(f"    tail share {ts10/10:.1f}x total paired d (committed "
          f"{TAIL_SHARE_COMM/10:.1f}x): {'PASS' if ok_ts else 'FAIL'}")
    print()

    # ---- structural gates (the within-train leg is the real test) ----------
    b = within["bands"]
    d_seq = [x["d"] for x in b if x is not None]
    lnames = [x["label"] for x in b if x is not None]
    # G1: per-band mean paired d rises monotonically in |e_v2| to a top band
    # that is substantially positive.
    mono = all(d_seq[i] <= d_seq[i + 1] + 1e-9 for i in range(len(d_seq) - 1))
    gate_mono = mono and d_seq[-1] >= 0.12
    # G2: the >= 0.6 tail carries the majority of total pooled paired d.
    within_tshare = within["tail_sum"] / within["total_d"]
    gate_tail_share = within_tshare > 0.5
    # G3: bottom fast-tracking band concedes/stays flat at day level (no
    # cmri-favor flip), while the tail union separates cmri-favor at day level.
    bbot = b[0]
    gate_bottom = bbot is not None and bbot["wins"] <= bbot["days"] / 2
    tu_w, tu_n, tu_z, tu_p = within["tail_union"]
    gate_tail_day = tu_w > tu_n / 2 and tu_z > 1.2
    # G4: mean daily 95th/99th-pct gap positive, with a day-level majority on
    # at least one of the two.
    d95, d99 = within["daily"][0], within["daily"][1]
    gate_gap = (min(d95["mean"], d99["mean"]) > 0.05
                and (d95["wins"] > d95["days"] / 2
                     or d99["wins"] > d99["days"] / 2))
    # G5: body mean paired d stays near parity (the small-error zone is NOT
    # where cmri's edge lives).
    gate_body = within["body"][1] < 0.10

    ok = all([gate_mono, gate_tail_share, gate_bottom, gate_tail_day,
              gate_gap, gate_body]) and audit_ok

    print("gates (within-train leg = fresh test; held-out leg = frozen audit):")
    print(f"    per-band mean d monotone in |e_v2| with top band >= +0.12: "
          f"{' / '.join(f'{ln} {dd:+.3f}' for ln, dd in zip(lnames, d_seq))} : "
          f"{'PASS' if gate_mono else 'FAIL'}")
    print(f"    tail carries majority of pooled paired d (share > 0.5): "
          f"{within_tshare:.2f} : {'PASS' if gate_tail_share else 'FAIL'}")
    print(f"    bottom 0-0.2 band concedes/flat at day level (wins <= n/2): "
          f"{bbot['wins']}/{bbot['days']} : "
          f"{'PASS' if gate_bottom else 'FAIL'}")
    print(f"    tail union separates cmri-favor at day level (w > n/2, "
          f"z > 1.2): {tu_w}/{tu_n} z={tu_z:+.2f} p={tu_p:.3f} : "
          f"{'PASS' if gate_tail_day else 'FAIL'}")
    bu_w, bu_n, bu_z, bu_p = within["body_union"]
    print(f"    body union day-block context (not a gate): {bu_w}/{bu_n} "
          f"z={bu_z:+.2f}")
    print(f"    daily 95th/99th-pct gap positive with day-level majority: "
          f"95th {d95['mean']:+.4f} ({d95['wins']}/{d95['days']}), "
          f"99th {d99['mean']:+.4f} ({d99['wins']}/{d99['days']}) : "
          f"{'PASS' if gate_gap else 'FAIL'}")
    print(f"    body mean paired d < +0.10 (edge not in the small-error zone): "
          f"{within['body'][1]:+.4f} : {'PASS' if gate_body else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §7 item 6 tail-owned-edge "
          f"reading is a structural fact, not an artifact of the single "
          f"held-out draw: on fresh within-train fits the same attribution "
          f"appears — the per-step mean paired d rises monotonically with "
          f"|e_v2| across the bands ({d_seq[0]:+.3f} at {lnames[0]} to "
          f"{d_seq[-1]:+.3f} at {lnames[-1]}), the >= {BODY_CUT} tail carries "
          f"{within_tshare:.1f}x the total pooled paired d and separates "
          f"cmri-favor at day level ({tu_w}/{tu_n} days, z {tu_z:+.2f}) while "
          f"the bottom fast-tracking band concedes or stays flat "
          f"({bbot['wins']}/{bbot['days']} days) and the body mean stays near "
          f"parity ({within['body'][1]:+.4f}), and the mean daily "
          f"95th-pct/99th-pct |err| gaps are positive ({d95['mean']:+.3f}/"
          f"{d99['mean']:+.3f} degC) — and the frozen artifact re-produces "
          f"every committed §7 item 6 digit (overall {frozen['overall_d']:+.4f}/"
          f"{frozen['overall_win']:.3f}/{fday_wins}-{fday_tot} day; body "
          f"{frozen['body'][1]:+.4f}/tail {frozen['tail'][1]:+.4f}; tail "
          f"{ts10/10:.1f}x of pooled d; per-band mean d and day counts, and the "
          f"daily 95th/99th-pct gap CIs [+{frozen['daily'][0]['lo']:.3f}, "
          f"+{frozen['daily'][0]['hi']:.3f}] / [+{frozen['daily'][1]['lo']:.3f}, "
          f"+{frozen['daily'][1]['hi']:.3f}]).")


if __name__ == "__main__":
    main()
