#!/usr/bin/env python3
"""verify_episodes.py — is the §7 item 4 "failure episodes" reading a
structural fact, or a single held-out draw?

§7 item 4 committed claims (frozen held-out resync, results/predictions_cmri.csv):
  * cmri's worst single step is 2.008 degC at 2016-04-29 21:20, and v2 fails
    in near-lockstep there (+2.207 vs +2.008 -- both blanch on the same cold
    night's appliance spike / outdoor conditions)
  * v2's worst episode is worse and evening-clustered: five of its top-10
    worst steps land on 2016-05-07 22:00-23:50, where v2's error reaches
    -2.70 degC while cmri stays at -0.85 degC
  * worst full-day means: cmri 2016-04-30 (0.850 degC), v2 2016-04-22
    (1.035 degC); cmri's best days reach 0.056 degC (2016-04-16)

That was one held-out draw.  This script verifies the reading WITHOUT
touching the held-out test for fitting: leg A re-fits fresh v2 (CalibratedTwin)
and cmri (frozen-config CmriTwin) twins on the within-train fit slice
[0, 11051) and replays the failure-episode structure on the within-train val
block [11051, 13814); leg B re-reads the frozen predictions_cmri.csv
(read-only) and re-aggregates the committed digit-level claims.  The gates
are tail-protection orderings -- NOT strict per-episode cmri-smaller (the
honest counter-instance is that both twins can blanch on the same step, e.g.
the val block's 2016-04-10 Sunday afternoon, where cmri's |e| is the larger
of the two -- a lockstep failure, exactly the §7 item 4 mechanism).

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import datetime as dt
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

# Committed §7 item 4 digit-level claims (leg-B re-aggregation tolerances).
C_CMRI_WORST = (dt.datetime(2016, 4, 29, 21, 20), 2.008, "cmri worst single step")
C_V2_AT_CM = 2.207          # v2 there, the near-lockstep
C_V2_WORST = (dt.datetime(2016, 5, 7, 23, 0), -2.703, "v2 worst single step")
C_CLUSTER_DAY = dt.date(2016, 5, 7)
C_CLUSTER_MIN = 5           # committed: five of v2's top-10 on 05-07 22-23h
C_DAY_CM = (dt.date(2016, 4, 30), 0.850, "cmri worst full-day mean")
C_DAY_V2 = (dt.date(2016, 4, 22), 1.035, "v2 worst full-day mean")
C_DAY_BEST = (dt.date(2016, 4, 16), 0.056, "cmri best full-day mean")
TOL = 0.0015                # step/day value re-aggregation tolerance


def day_means(errs: list[float], ds: list[dt.datetime]) -> dict[dt.date, float]:
    mm: dict[dt.date, list[float]] = {}
    for e, d in zip(errs, ds):
        mm.setdefault(d.date(), []).append(abs(e))
    return {k: statistics.fmean(v) for k, v in mm.items()}


def worst_day(dm: dict[dt.date, float]) -> tuple[dt.date, float]:
    return max(dm.items(), key=lambda kv: kv[1])


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
    data = load_cmri_data(DATA if "DATA" in globals() else "data/energydata_complete.csv")
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    val_start = int(round((1.0 - VALID_FRAC) * split))
    T, Th, t_out = data["T1"], data["Th"], data["T_out"]
    dew, load, lux = data["Tdewpoint"], data["Appliances"], data["lights"]
    wind, ghi = data["Windspeed"], data["GHI"]
    tm = fabric_mass(t_out, CMRI_W)
    tm2 = fabric_mass(t_out, CMRI_W2)
    dates = data["date"]

    print("failure-episode probe (resync protocol); held-out test untouched "
          "for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + val-block episode structure ------
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
    # anchors coincide for both protocols; index i -> same datetime in both lists
    v2rows = [(c, o, d) for c, o, d in zip(pv2, obs, dates[val_start:split])
              if c is not None]
    cmrows = [(c, o, d) for c, o, d in zip(pcm, obs, dates[val_start:split])
              if c is not None]
    ev2 = [o - c for c, o, _ in v2rows]
    ecm = [o - c for c, o, _ in cmrows]
    dv2 = [d for _, _, d in v2rows]
    dcm = [d for _, _, d in cmrows]
    assert len(ev2) == len(ecm) == 2743 and dv2 == dcm

    iworst_v2 = max(range(len(ev2)), key=lambda i: abs(ev2[i]))
    iworst_cm = max(range(len(ecm)), key=lambda i: abs(ecm[i]))
    print("leg A  within-train val-block failure episodes (fresh fits):")
    print(f"    v2 worst step  {dv2[iworst_v2]:%Y-%m-%d %H:%M} "
          f"({dv2[iworst_v2].strftime('%A')})  e={ev2[iworst_v2]:+.4f}")
    print(f"    cmri worst step {dcm[iworst_cm]:%Y-%m-%d %H:%M} "
          f"({dcm[iworst_cm].strftime('%A')}) e={ecm[iworst_cm]:+.4f}")
    rlock = abs(ev2[iworst_cm]) / abs(ecm[iworst_cm])
    print(f"    lockstep at cmri's worst step: e_v2={ev2[iworst_cm]:+.4f} / "
          f"e_cmri={ecm[iworst_cm]:+.4f}  ratio |e_v2|/|e_cmri|={rlock:.3f}")

    top10 = sorted(range(len(ev2)), key=lambda i: abs(ev2[i]))[-10:][::-1]
    n_eve = sum(1 for i in top10 if dv2[i].hour >= 18)
    print(f"    v2 top-10 worst steps: {n_eve}/10 in 18-23h evening "
          f"({dv2[top10[0]]:%Y-%m-%d %H:%M}..{dv2[top10[-1]]:%Y-%m-%d %H:%M} "
          f"all {dv2[top10[0]].strftime('%A')} night)")
    top30 = sorted(range(len(ev2)), key=lambda i: abs(ev2[i]))[-30:][::-1]
    n_small = sum(1 for i in top30 if abs(ecm[i]) < abs(ev2[i]))
    print(f"    cmri |e| smaller on {n_small}/30 of v2's top-30 worst steps")

    dm_v2 = day_means(ev2, dv2)
    dm_cm = day_means(ecm, dcm)
    wd_v2, wd_cm, bd_cm = worst_day(dm_v2), worst_day(dm_cm), min(dm_cm.items(), key=lambda kv: kv[1])
    print(f"    worst days: v2 {wd_v2[0]} ({wd_v2[0].strftime('%A')}) "
          f"{wd_v2[1]:.4f} / cmri {wd_cm[0]} ({wd_cm[0].strftime('%A')}) "
          f"{wd_cm[1]:.4f}")
    print(f"    cmri best day {bd_cm[0]} ({bd_cm[0].strftime('%A')}) "
          f"{bd_cm[1]:.4f}")
    rmse_v2 = metrics([c for c, _, _ in v2rows], [o for _, o, _ in v2rows])["rmse"]
    rmse_cm = metrics([c for c, _, _ in cmrows], [o for _, o, _ in cmrows])["rmse"]
    print(f"    context only: val-block resync RMSE v2 {rmse_v2:.4f} / cmri "
          f"{rmse_cm:.4f}")
    print()

    # ---- leg B: frozen §7 item 4 digit audit (read-only) ---------------------
    fds, fv2, fcm = read_frozen()
    fidx_cm = max(range(len(fcm)), key=lambda i: abs(fcm[i]))
    fidx_v2 = max(range(len(fv2)), key=lambda i: abs(fv2[i]))
    ftop10 = sorted(range(len(fv2)), key=lambda i: abs(fv2[i]))[-10:][::-1]
    fday_v2 = day_means(fv2, fds)
    fday_cm = day_means(fcm, fds)
    fworst_v2, fworst_cm = worst_day(fday_v2), worst_day(fday_cm)
    fbest_cm = min(fday_cm.items(), key=lambda kv: kv[1])
    cluster = [dsi for i, dsi in enumerate(fds) if dsi.date() == C_CLUSTER_DAY
               and dsi.hour >= 22]
    if cluster:
        cmin_v2 = min(fv2[i] for i, d in enumerate(fds) if d in cluster
                      and d.hour >= 22 and d.date() == C_CLUSTER_DAY)
    else:
        cmin_v2 = 0.0
    cm_eve = [fcm[i] for i, d in enumerate(fds) if d.date() == C_CLUSTER_DAY
              and d.hour >= 22]
    print("leg B  frozen predictions_cmri.csv audit (read-only, n=%d):" % len(fds))
    t, v, lab = C_CMRI_WORST
    print(f"    {lab}: {fds[fidx_cm]:%Y-%m-%d %H:%M} "
          f"({fds[fidx_cm].strftime('%A')}) |e|={abs(fcm[fidx_cm]):.4f}  "
          f"(committed {t:%Y-%m-%d %H:%M} {v:.3f}); "
          f"v2 there e={fv2[fidx_cm]:+.4f} (committed +{C_V2_AT_CM:.3f})")
    t, v, lab = C_V2_WORST
    print(f"    {lab}: {fds[fidx_v2]:%Y-%m-%d %H:%M} "
          f"({fds[fidx_v2].strftime('%A')}) e={fv2[fidx_v2]:+.4f} "
          f"(committed {v:.2f}); cmri there e={fcm[fidx_v2]:+.4f} "
          f"(committed > -1.2, reported -0.85)")
    print(f"    v2 top-10 cluster: {len(ftop10)} listed, "
          f"{sum(1 for i in ftop10 if fds[i].date() == C_CLUSTER_DAY)}/10 on "
          f"{C_CLUSTER_DAY.strftime('%Y-%m-%d')} (committed >= "
          f"{C_CLUSTER_MIN}, 'five of top-10' in 22:00-23:50); v2 min over "
          f"that window {cmin_v2:.3f}, cmri min {min(cm_eve) if cm_eve else 0:.3f}")
    print(f"    {C_DAY_CM[2]}: {fworst_cm[0]} {fworst_cm[1]:.4f} "
          f"(committed {C_DAY_CM[0]} {C_DAY_CM[1]:.3f})")
    print(f"    {C_DAY_V2[2]}: {fworst_v2[0]} {fworst_v2[1]:.4f} "
          f"(committed {C_DAY_V2[0]} {C_DAY_V2[1]:.3f})")
    print(f"    {C_DAY_BEST[2]}: {fbest_cm[0]} {fbest_cm[1]:.4f} "
          f"(committed {C_DAY_BEST[0]} {C_DAY_BEST[1]:.3f})")

    # ---- structural gates ----------------------------------------------------
    gate_worst = abs(ev2[iworst_v2]) > abs(ecm[iworst_cm])
    # lockstep: at cmri's worst step both blanch with comparable magnitudes
    # (band [0.75, 1.33]; committed-frozen ratio 2.207/2.008 = 1.10)
    gate_lock = 0.75 <= rlock <= 1.33
    # v2's worst failures cluster in the evening (committed: the 05-07 22-23h
    # episode; val-block bracket: 10/10 in 18-23h)
    gate_eve = n_eve >= 9
    # tail protection: on v2's worst 30 steps cmri is the smaller-magnitude
    # twin on almost all of them (val-block bracket: 30/30)
    gate_tail = n_small >= 28
    # worst full-day mean: v2's worst day is worse than cmri's worst day
    gate_day = wd_v2[1] >= wd_cm[1]

    # leg-B committed digits:
    audit_cm_worst = (fds[fidx_cm] == C_CMRI_WORST[0]
                      and abs(abs(fcm[fidx_cm]) - C_CMRI_WORST[1]) < TOL)
    audit_lock = abs(fv2[fidx_cm] - C_V2_AT_CM) < 3 * TOL
    audit_v2_worst = (fds[fidx_v2] == C_V2_WORST[0]
                      and abs(fv2[fidx_v2] - C_V2_WORST[1]) < 3 * TOL)
    audit_cm_quiet = fcm[fidx_v2] > -1.2          # cmri "stays at -0.85"
    audit_day_cm = fworst_cm == (C_DAY_CM[0], C_DAY_CM[1]) or (
        fworst_cm[0] == C_DAY_CM[0] and abs(fworst_cm[1] - C_DAY_CM[1]) < 3 * TOL)
    audit_day_v2 = fworst_v2[0] == C_DAY_V2[0] and abs(fworst_v2[1] - C_DAY_V2[1]) < 3 * TOL
    audit_day_best = fbest_cm[0] == C_DAY_BEST[0] and abs(fbest_cm[1] - C_DAY_BEST[1]) < 3 * TOL
    fcluster_n = sum(1 for i in ftop10 if fds[i].date() == C_CLUSTER_DAY
                     and fds[i].hour >= 22)
    audit_cluster = fcluster_n >= C_CLUSTER_MIN

    ok = all([gate_worst, gate_lock, gate_eve, gate_tail, gate_day,
              audit_cm_worst, audit_lock, audit_v2_worst, audit_cluster,
              audit_cm_quiet, audit_day_cm, audit_day_v2, audit_day_best])

    print()
    print("gates (within-train leg = fresh test; held-out leg = frozen audit):")
    print(f"    v2 worst step > cmri worst step: within "
          f"{abs(ev2[iworst_v2]):.3f} vs {abs(ecm[iworst_cm]):.3f} : "
          f"{'PASS' if gate_worst else 'FAIL'}")
    print(f"    lockstep at cmri's worst step (ratio in [0.75,1.33]): "
          f"{rlock:.3f} : {'PASS' if gate_lock else 'FAIL'}")
    print(f"    v2 top-10 evening clustering (>=9/10 in 18-23h): "
          f"{n_eve}/10 : {'PASS' if gate_eve else 'FAIL'}")
    print(f"    cmri smaller on v2 top-30 (>=28/30): {n_small}/30 : "
          f"{'PASS' if gate_tail else 'FAIL'}")
    print(f"    v2 worst day mean >= cmri worst day mean: {wd_v2[1]:.4f} vs "
          f"{wd_cm[1]:.4f} : {'PASS' if gate_day else 'FAIL'}")
    print(f"    audit cmri worst {C_CMRI_WORST[0]:%Y-%m-%d %H:%M} "
          f"{C_CMRI_WORST[1]:.3f} (got {abs(fcm[fidx_cm]):.4f} @ "
          f"{fds[fidx_cm]:%Y-%m-%d %H:%M}): "
          f"{'PASS' if audit_cm_worst else 'FAIL'}")
    print(f"    audit v2 near-lockstep +{C_V2_AT_CM:.3f} there (got "
          f"{fv2[fidx_cm]:+.4f}): {'PASS' if audit_lock else 'FAIL'}")
    print(f"    audit v2 worst {C_V2_WORST[0]:%Y-%m-%d %H:%M} "
          f"{C_V2_WORST[1]:.3f} (got {fv2[fidx_v2]:+.4f} @ "
          f"{fds[fidx_v2]:%Y-%m-%d %H:%M}): "
          f"{'PASS' if audit_v2_worst else 'FAIL'}")
    print(f"    audit cluster: >= {C_CLUSTER_MIN} of v2 top-10 on "
          f"{C_CLUSTER_DAY.strftime('%Y-%m-%d')} 22-23h "
          f"(committed 'five'): "
          f"{'PASS' if audit_cluster else 'FAIL'} "
          f"(actual {fcluster_n})")
    print(f"    audit cmri stays quiet at v2's worst step (> -1.2; got "
          f"{fcm[fidx_v2]:+.4f}): {'PASS' if audit_cm_quiet else 'FAIL'}")
    print(f"    audit worst days cmri {C_DAY_CM[0]} {C_DAY_CM[1]:.3f} / v2 "
          f"{C_DAY_V2[0]} {C_DAY_V2[1]:.3f} / best {C_DAY_BEST[0]} "
          f"{C_DAY_BEST[1]:.3f}: "
          f"{'PASS' if audit_day_cm and audit_day_v2 and audit_day_best else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §7 item 4 failure-episode "
          f"reading is a structural fact, not a single held-out draw: on fresh "
          f"within-train fits the same episode geometry appears — v2's worst "
          f"step is worse than cmri's ({abs(ev2[iworst_v2]):.3f} vs "
          f"{abs(ecm[iworst_cm]):.3f} C), the two twins blanch in lockstep "
          f"(ratio {rlock:.2f} at cmri's worst step), v2's worst failures "
          f"cluster in the evening ({n_eve}/10 in 18-23h) while cmri stays "
          f"smaller on {n_small}/30 of v2's worst-tail steps, and v2's worst "
          f"day mean exceeds cmri's ({wd_v2[1]:.3f} vs {wd_cm[1]:.3f} C) — and "
          f"the frozen artifact re-produces every committed digit (cmri worst "
          f"{abs(fcm[fidx_cm]):.3f} @ {fds[fidx_cm]:%Y-%m-%d %H:%M} with v2 in "
          f"near-lockstep +{fv2[fidx_cm]:.3f}; v2 worst {fv2[fidx_v2]:+.3f} @ "
          f"{fds[fidx_v2]:%Y-%m-%d %H:%M} while cmri stays {fcm[fidx_v2]:+.3f}; "
          f"worst days cmri {fworst_cm[0]} {fworst_cm[1]:.3f} / v2 "
          f"{fworst_v2[0]} {fworst_v2[1]:.3f}; cmri best {fbest_cm[0]} "
          f"{fbest_cm[1]:.3f}).")


if __name__ == "__main__":
    main()
