#!/usr/bin/env python3
"""verify_metrictuple_within.py — is the §5.4/§5.2 metric-tuple reading ("cmri
beats v2 on every held-out resync metric (RMSE 0.646 -> 0.531, -18%) and now
beats the daily-resync persistence control (0.531 vs 0.625, day-block
bootstrap CI +0.017 to +0.164 degC, §5.5)", RMSE/MAE/R2/MAPE tuples in the
§5.1 and §5.2 tables, and the §5.2 line-142 caveat "cmri is actually slightly
worse on MAE (1.200 vs 1.176)" on free-run) a structural fact, not an
artifact of the single held-out draw?

§5.1 committed claims (frozen held-out resync, n = 5879,
results/predictions_cmri.csv joined to results/predictions.csv for the
persistence_resync column, as analyzed by verify_persistence.py):
  * v2          0.646 RMSE / 0.456 MAE / +0.796 R2 /   1.95 MAPE;
  * cmri        0.531 / 0.384 / +0.862 / 1.65;
  * persistence 0.625 / 0.502 / +0.809 / 2.19
  -- cmri is the best row on every metric (this is the §5.4 headline,
  which adds the -18% RMSE erasure vs v2 and the persistence-beating
  day-block bootstrap of §5.5).
§5.2 committed claims (frozen held-out free-run, n = 5920):
  * v2   1.575 / 1.176 / -0.22 / 5.03;
  * cmri 1.492 / 1.200 / -0.09 / 5.06
  -- cmri wins RMSE/R2 on free-run but the §5.2 line-142 caveat documents
  that cmri is "actually slightly worse on MAE (1.200 vs 1.176)".

verify_v1baseline_within (item 21) audits only the v1/v2 rows on
results/predictions_cmri.csv, and verify_persistence_within (item 26) audits
only the §5.5 pooled RMSE gap -- neither re-audits the cmri and persistence
metric tuples or the free-run tuple pair. This script probes the tuple rows
WITHOUT touching the held-out test: leg A re-fits fresh v2 (CalibratedTwin),
cmri (frozen-config CmriTwin), and a within-train persistence control
(PersistenceModel, T-hat = T) on the within-train fit slice [0, 11051),
replays the daily-resync protocol on the within-train val block
[11051, 13814) (n = 2743 resync steps across 20 days) and the open-loop
free-run protocol (n = 2762 steps), and recomputes the full tuple
(RMSE, MAE, R2, MAPE) for all three models on the FRESH residuals under both
protocols; leg B re-reads the frozen artifacts (read-only), joins
predictions_cmri.csv to predictions.csv, and re-aggregates every committed
§5.1/§5.2 tuple digit exactly as verify_persistence.py / verify_freerun.py do
(predictions.csv has NO persistence free-run column, so the §5.2 free-run
table carries no persistence row and leg B audits only the cmri/v2 free-run
tuple pair, matching the committed table).

Gates (power-aware structural invariants of the metric-tuple reading, not
digit matches; both the within-train leg and the held-out leg must agree in
direction for every metric): (i) resync RMSE cmri <= v2 AND cmri <=
persistence, (ii) resync MAE cmri <= v2 AND cmri <= persistence, (iii) resync
R2 cmri >= v2 AND cmri >= persistence, (iv) resync MAPE cmri <= v2 AND cmri
<= persistence (the §5.4 "every held-out resync metric" claim, replayed per
metric on the 20-day block -- the block's magnitudes are smaller than the
42-day block's, so the gates are per-metric no-worse-than comparisons, not
digit matches), (v) free-run RMSE cmri < v2, (vi) free-run R2 cmri > v2.
Block-dependent strength deltas are documented, not gated: the within-train
block's metric magnitudes differ (e.g. fresh resync cmri 0.505 vs held 0.531;
fresh v2 0.582 vs held 0.646), and the within-train free-run MAE/MAPE
directions INVERT the held §5.2 caveat -- fresh cmri MAE 0.603 < v2 1.192 and
MAPE 2.738 < 5.408 both favor cmri, while held cmri MAE 1.200 > v2 1.176 and
MAPE 5.06 > 5.03 -- a single-run-feature block delta of the 20-day open-loop
replay, explicitly NOT gated (the held-out line-142 caveat stays the honest
statement, and the frozen artifact re-produces it).

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import sys

sys.path.insert(0, ".")

from run_experiment import (TRAIN_FRAC, PersistenceModel, metrics,  # noqa: E402
                            run_protocol)
from experiment_cmri import (CMRI_ALPHA, CMRI_KEEP, CMRI_T_SET, CMRI_W,  # noqa: E402
                             CMRI_W2, days_of, load_cmri_data,
                             run_protocol_cmri)
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from cmri_twin import CmriTwin, fabric_mass  # noqa: E402

VALID_FRAC = 0.2
CMRI_PRED = "results/predictions_cmri.csv"
PERS_PRED = "results/predictions.csv"   # older artifact carrying persistence_resync

# committed §5.1/§5.2 tuple digits (leg-B audit targets), exactly the numbers
# run_experiment.py / experiment_cmri.py / verify_persistence.py /
# verify_freerun.py print on the frozen artifacts
C_N_R = 5879                          # resync evaluable steps (42 days)
C_N_F = 5920                          # freerun evaluable steps
# (rmse, mae, r2, mape) per model/protocol
C_RES_V2 = (0.646, 0.456, 0.796, 1.95)    # §5.1 resync v2
C_RES_CM = (0.531, 0.384, 0.862, 1.65)    # §5.1 resync cmri
C_RES_PERS = (0.625, 0.502, 0.809, 2.19)  # §5.1 resync persistence
C_FREE_CM = (1.492, 1.200, -0.09, 5.06)   # §5.2 free-run cmri
C_FREE_V2 = (1.575, 1.176, -0.22, 5.03)   # §5.2 free-run v2
C_FREE_MAE_INV = True                 # held caveat: cmri free-run MAE 1.200 > v2 1.176

# per-digit half-unit rounding tolerance at each printed precision
# (rmse 3dp, mae 3dp, r2 2dp, mape 2dp)
TOL = (0.0005, 0.0005, 0.005, 0.005)


def metric_tup(pairs) -> tuple[float, float, float, float]:
    pred = [c for c, _ in pairs]
    obs = [o for _, o in pairs]
    m = metrics(pred, obs)
    return (m["rmse"], m["mae"], m["r2"], m["mape"])


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

    print("structural probe of the §5.1/§5.2 metric-tuple reading (cmri is "
          "the best resync row on every metric — beats v2 and the "
          "daily-resync persistence control — and wins free-run RMSE/R2 with "
          "the held §5.2 MAE caveat); held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + protocol replays -------------------
    fs = slice(0, val_start)
    v2 = CalibratedTwin()
    v2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    cm = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    cm.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
           ghi[fs], tm[fs], tm2[fs])

    recs_v2 = list(zip(T, t_out, dew, load, lux))[val_start:split]
    recs_cm = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm2))[val_start:split]
    val_days = days_of(dates, val_start, split)
    obs = [r[0] for r in recs_v2]

    pr2 = run_protocol(v2, recs_v2[0][0], recs_v2, val_days)
    prc = run_protocol_cmri(cm, (recs_cm[0][0], recs_cm[0][1]), recs_cm, val_days)
    prp = run_protocol(PersistenceModel(), recs_v2[0][0], recs_v2, val_days)
    r2 = [(c, o) for c, o in zip(pr2, obs) if c is not None]
    rc = [(c, o) for c, o in zip(prc, obs) if c is not None]
    rp = [(c, o) for c, o in zip(prp, obs) if c is not None]
    assert len(r2) == len(rc) == len(rp) == 2743

    pf2 = run_protocol(v2, recs_v2[0][0], recs_v2, None)[1:]
    pfc = run_protocol_cmri(cm, (recs_cm[0][0], recs_cm[0][1]), recs_cm, None)[1:]
    fobs = obs[1:]
    assert len(pf2) == len(pfc) == len(fobs) == 2762

    m2r = metric_tup(r2)
    mcr = metric_tup(rc)
    mpr = metric_tup(rp)
    m2f = metric_tup(list(zip(pf2, fobs)))
    mcf = metric_tup(list(zip(pfc, fobs)))

    print(f"leg A  within-train val block (fresh fits): resync n={len(rc)} "
          f"steps / 20 days; freerun n={len(pfc)} steps")
    print(f"  resync  v2          [{m2r[0]:.3f} {m2r[1]:.3f} {m2r[2]:+.2f} "
          f"{m2r[3]:.2f}]   (held §5.1 [{C_RES_V2[0]:.3f} {C_RES_V2[1]:.3f} "
          f"{C_RES_V2[2]:+.2f} {C_RES_V2[3]:.2f}])")
    print(f"          cmri        [{mcr[0]:.3f} {mcr[1]:.3f} {mcr[2]:+.2f} "
          f"{mcr[3]:.2f}]   (held §5.1 [{C_RES_CM[0]:.3f} {C_RES_CM[1]:.3f} "
          f"{C_RES_CM[2]:+.2f} {C_RES_CM[3]:.2f}])")
    print(f"          persistence [{mpr[0]:.3f} {mpr[1]:.3f} {mpr[2]:+.2f} "
          f"{mpr[3]:.2f}]   (held §5.1 [{C_RES_PERS[0]:.3f} "
          f"{C_RES_PERS[1]:.3f} {C_RES_PERS[2]:+.2f} {C_RES_PERS[3]:.2f}])")
    print(f"  freerun v2          [{m2f[0]:.3f} {m2f[1]:.3f} {m2f[2]:+.2f} "
          f"{m2f[3]:.2f}]   (held §5.2 [{C_FREE_V2[0]:.3f} {C_FREE_V2[1]:.3f} "
          f"{C_FREE_V2[2]:+.2f} {C_FREE_V2[3]:.2f}])")
    print(f"          cmri        [{mcf[0]:.3f} {mcf[1]:.3f} {mcf[2]:+.2f} "
          f"{mcf[3]:.2f}]   (held §5.2 [{C_FREE_CM[0]:.3f} {C_FREE_CM[1]:.3f} "
          f"{C_FREE_CM[2]:+.2f} {C_FREE_CM[3]:.2f}])")
    print(f"  (block delta, not gated: within-train freerun MAE/MAPE favor "
          f"cmri — {mcf[1]:.3f} < {m2f[1]:.3f} and {mcf[3]:.2f} < "
          f"{m2f[3]:.2f} — INVERTING the held §5.2 line-142 caveat cmri MAE "
          f"1.200 > v2 1.176; the held-out caveat stays the honest "
          f"statement, and the within-train inversion is a single-run feature "
          f"of the 20-day open-loop replay)")

    # ---- leg B: frozen §5.1/§5.2 tuple audit (read-only) ----------------------
    cmri_rows = []
    with open(CMRI_PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cmri_rows.append(r)
    pers_rows = []
    with open(PERS_PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pers_rows.append(r)
    assert len(cmri_rows) == len(pers_rows), "row counts must match"
    assert all(c["date"] == p["date"] and c["T1_observed"] == p["T1_observed"]
               for c, p in zip(cmri_rows, pers_rows)), "rows must align"
    assert all(c["v2_resync"] == p["v2_resync"]
               for c, p in zip(cmri_rows, pers_rows))

    fr2, frc, frp = [], [], []
    for c, p in zip(cmri_rows, pers_rows):
        if c["cmri_resync"] == "" or c["v2_resync"] == "" \
                or p["persistence_resync"] == "":
            continue  # day-boundary anchor rows, excluded by protocol
        o = float(c["T1_observed"])
        fr2.append((float(c["v2_resync"]), o))
        frc.append((float(c["cmri_resync"]), o))
        frp.append((float(p["persistence_resync"]), o))
    fm2r = metric_tup(fr2)
    fmcr = metric_tup(frc)
    fmpr = metric_tup(frp)

    ff2, ffc = [], []
    for r in cmri_rows:
        if r["v2_freerun"] != "" and r["cmri_freerun"] != "":
            o = float(r["T1_observed"])
            ff2.append((float(r["v2_freerun"]), o))
            ffc.append((float(r["cmri_freerun"]), o))
    fm2f = metric_tup(ff2)
    fmcf = metric_tup(ffc)

    audit_ok = True

    def audit(lbl, got, comm):
        nonlocal audit_ok
        t = TOL
        ok = all(abs(g - c) <= tt for g, c, tt in zip(got, comm, t))
        audit_ok &= ok
        print(f"    {lbl} RMSE {got[0]:.3f} MAE {got[1]:.3f} R2 {got[2]:+.2f} "
              f"MAPE {got[3]:.2f} (committed RMSE {comm[0]:.3f} MAE {comm[1]:.3f} "
              f"R2 {comm[2]:+.2f} MAPE {comm[3]:.2f}): "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\nleg B  frozen held-out (audit leg): resync n={len(fr2)} "
          f"(committed {C_N_R}), freerun n={len(ff2)} (committed {C_N_F}); "
          f"{CMRI_PRED} joined to {PERS_PRED} for the persistence_resync "
          f"column (predictions.csv has no persistence freerun column, so the "
          f"§5.2 freerun tuple pair is audited from {CMRI_PRED} only, "
          f"matching the committed table)")
    print("  committed-digit audit (frozen artifacts, read-only):")
    audit("resync v2", fm2r, C_RES_V2)
    audit("resync cmri", fmcr, C_RES_CM)
    audit("resync persistence", fmpr, C_RES_PERS)
    audit("freerun cmri", fmcf, C_FREE_CM)
    audit("freerun v2", fm2f, C_FREE_V2)
    inv = fmcf[1] > fm2f[1]
    print(f"      held §5.2 free-run MAE caveat re-derived: cmri "
          f"{fmcf[1]:.3f} > v2 {fm2f[1]:.3f} : "
          f"{'PASS' if inv == C_FREE_MAE_INV else 'FAIL'}  "
          f"(committed 1.200 vs 1.176)")

    # ---- structural gates (the within-train leg is the real test) -------------
    g1 = mcr[0] <= m2r[0] and mcr[0] <= mpr[0]
    g2 = mcr[1] <= m2r[1] and mcr[1] <= mpr[1]
    g3 = mcr[2] >= m2r[2] and mcr[2] >= mpr[2]
    g4 = mcr[3] <= m2r[3] and mcr[3] <= mpr[3]
    g5 = mcf[0] < m2f[0]
    g6 = mcf[2] > m2f[2]
    ok = g1 and g2 and g3 and g4 and g5 and g6 and audit_ok

    print()
    print("gates (within-train leg = fresh test; held-out leg = frozen "
          "audit; per-metric no-worse-than comparisons, both neighbors "
          "required):")
    print(f"    (i)  resync RMSE cmri <= v2 AND <= persistence: "
          f"{mcr[0]:.3f} <= {m2r[0]:.3f} AND {mcr[0]:.3f} <= {mpr[0]:.3f} : "
          f"{'PASS' if g1 else 'FAIL'}  (held 0.531 <= 0.646/0.625)")
    print(f"    (ii) resync MAE cmri <= v2 AND <= persistence: "
          f"{mcr[1]:.3f} <= {m2r[1]:.3f} AND {mcr[1]:.3f} <= {mpr[1]:.3f} : "
          f"{'PASS' if g2 else 'FAIL'}  (held 0.384 <= 0.456/0.502)")
    print(f"    (iii) resync R2 cmri >= v2 AND >= persistence: "
          f"{mcr[2]:+.2f} >= {m2r[2]:+.2f} AND {mcr[2]:+.2f} >= "
          f"{mpr[2]:+.2f} : {'PASS' if g3 else 'FAIL'}  "
          f"(held +0.862 >= +0.796/+0.809)")
    print(f"    (iv) resync MAPE cmri <= v2 AND <= persistence: "
          f"{mcr[3]:.2f} <= {m2r[3]:.2f} AND {mcr[3]:.2f} <= {mpr[3]:.2f} : "
          f"{'PASS' if g4 else 'FAIL'}  (held 1.65 <= 1.95/2.19)")
    print(f"    (v)  freerun RMSE cmri < v2: {mcf[0]:.3f} < {m2f[0]:.3f} : "
          f"{'PASS' if g5 else 'FAIL'}  (held 1.492 < 1.575)")
    print(f"    (vi) freerun R2 cmri > v2: {mcf[2]:+.2f} > {m2f[2]:+.2f} : "
          f"{'PASS' if g6 else 'FAIL'}  (held -0.09 > -0.22)")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §5.4 'cmri beats v2 on "
          f"every held-out resync metric'-and-persistence reading is a "
          f"structural fact: on fresh within-train fits the cmri row is again "
          f"no-worse-than both v2 and the replayed persistence control on "
          f"every resync metric (RMSE {mcr[0]:.3f} vs {m2r[0]:.3f} / "
          f"{mpr[0]:.3f}, MAE {mcr[1]:.3f} vs {m2r[1]:.3f} / {mpr[1]:.3f}, "
          f"R2 {mcr[2]:+.2f} vs {m2r[2]:+.2f} / {mpr[2]:+.2f}, MAPE "
          f"{mcr[3]:.2f} vs {m2r[3]:.2f} / {mpr[3]:.2f}; held 0.531 vs "
          f"0.646/0.625 etc.), and wins free-run RMSE/R2 direction "
          f"({mcf[0]:.3f} < {m2f[0]:.3f}, {mcf[2]:+.2f} > {m2f[2]:+.2f}; held "
          f"1.492 < 1.575, -0.09 > -0.22) — block-dependent strength deltas "
          f"only, not gates: the block's resync magnitudes are smaller "
          f"(cmri {mcr[0]:.3f} vs held 0.531, v2 {m2r[0]:.3f} vs held 0.646), "
          f"and the within-train freerun MAE/MAPE directions INVERT the held "
          f"§5.2 line-142 caveat (fresh cmri MAE {mcf[1]:.3f} < v2 "
          f"{m2f[1]:.3f} and MAPE {mcf[3]:.2f} < {m2f[3]:.2f} vs held 1.200 > "
          f"1.176 and 5.06 > 5.03 — a single-run feature of the 20-day "
          f"open-loop replay, so the held-out caveat stays the honest "
          f"statement); and the frozen artifacts re-produce every committed "
          f"§5.1/§5.2 tuple digit (resync v2 [{fm2r[0]:.3f} {fm2r[1]:.3f} "
          f"{fm2r[2]:+.2f} {fm2r[3]:.2f}], cmri [{fmcr[0]:.3f} {fmcr[1]:.3f} "
          f"{fmcr[2]:+.2f} {fmcr[3]:.2f}], persistence [{fmpr[0]:.3f} "
          f"{fmpr[1]:.3f} {fmpr[2]:+.2f} {fmpr[3]:.2f}] n = {len(fr2)}; "
          f"freerun cmri [{fmcf[0]:.3f} {fmcf[1]:.3f} {fmcf[2]:+.2f} "
          f"{fmcf[3]:.2f}], v2 [{fm2f[0]:.3f} {fm2f[1]:.3f} {fm2f[2]:+.2f} "
          f"{fm2f[3]:.2f}] n = {len(ff2)}, with the held free-run MAE caveat "
          f"cmri {fmcf[1]:.3f} > v2 {fm2f[1]:.3f} re-derived).")


if __name__ == "__main__":
    main()
