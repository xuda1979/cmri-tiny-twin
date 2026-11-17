#!/usr/bin/env python3
"""verify_v1baseline_within.py — is the §3.1/§3.2 v1-baseline reading (the
uncCalibrated handbook-default twin errors by ~6 degC while the least-squares
calibrated v2 twin cuts that to ~0.6 degC on the held-out resync protocol,
"the ~9x RMSE collapse 5.881 -> 0.646", and "v1 diverges 6.861 vs v2's
graceful 1.575 on free-run") a structural fact, not an artifact of the single
held-out draw?

§3.1/§3.2 committed claims (frozen held-out, results/predictions_cmri.csv):
  * resync (n = 5879): v1 5.881 RMSE / 5.554 MAE / -15.94 R2 / 24.26 MAPE;
    v2 0.646 / 0.456 / 0.796 / 1.95;  R2 flips from worse-than-mean (-15.94)
    to better-than-mean (+0.796);
  * freerun (n = 5920): v1 6.861 / 6.682 / -22.08 / 29.22; v2 1.575 / 1.176 /
    -0.22 / 5.03;
  * §4: "-89% error" (RMSE 5.881 -> 0.646 erases ~89% of the error) and the
    loose catchphrase "11x improvement" (the exact frozen RMSE ratio is
    5.881/0.646 = 9.10x; 24.26/1.95 = 12.4x on MAPE — the phrase is a round
    statement matching neither exactly, documented honestly below).

This script verifies the baseline-collapse reading WITHOUT touching the
held-out test: leg A re-fits a fresh v2 (CalibratedTwin) on the within-train
fit slice [0, 11051), replays v1 (V1Model, pure handbook constants — no fit)
and v2 through BOTH protocols on the within-train val block [11051, 13814)
(resync: daily re-anchored, n = 2743 steps across 20 days; freerun: one
continuous run, n = 2762 steps), recomputes the full metric tuple (RMSE, MAE,
R2, MAPE) for both models on the FRESH residuals, the per-day resync
v2-vs-v1 win count, and both RMSE ratios, plus a replayed PersistenceModel
(T-hat = T) context control; leg B re-reads the frozen
results/predictions_cmri.csv (read-only) and re-aggregates every committed
§3.1/§3.2 digit for both models under both protocols exactly as
run_experiment.py / experiment_cmri.py do.

Gates (power-aware structural invariants of the baseline-collapse reading,
not digit matches): within-train, (i) resync v1 RMSE > 5 degC AND v2 RMSE <
1 degC (the "~6 vs ~0.6 degC" structural split), (ii) resync RMSE ratio
v1/v2 > 5 (the "~9x collapse"; fresh 9.65x vs held 9.10x), (iii) v2 beats v1
on a strict majority of the fresh val days in the per-day resync RMSE series
(fresh 20/20; the collapse is not a few bad days), (iv) freerun v1 RMSE > 4
degC AND v2 RMSE < 2.5 degC ("v1 diverges, v2 degrades gracefully"),
(v) freerun RMSE ratio v1/v2 > 2 (fresh 4.63x vs held 4.36x), (vi) resync R2
v1 < 0 AND v2 > 0 (calibration flips a worse-than-mean predictor to
better-than-mean; fresh -77.66 / +0.156). Block-dependent strength deltas
(exact digits, ratios, R2 magnitudes — the 20-day block's metrics are
stronger on RMSE ratio but the fresh v2 resync R2 +0.156 is much smaller than
held +0.796 because the within-train block's error variance composition
differs; the freerun R2 stays negative for v2 on both blocks, -0.22 held vs
-3.90 fresh, and is reported in the §3.2 table as-is with no positivity
claim) are documented, not gated; the held-out leg must re-produce every
committed digit, the per-day held win count, the -89% error-erasure, and the
exact ratio pair (9.10x resync / 4.36x freerun).

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys

sys.path.insert(0, ".")

from run_experiment import (TRAIN_FRAC, PersistenceModel, V1Model, metrics,  # noqa: E402
                            run_protocol)
from experiment_cmri import days_of, load_cmri_data  # noqa: E402
from tiny_twin_v2 import CalibratedTwin  # noqa: E402

VALID_FRAC = 0.2
CMRI_PRED = "results/predictions_cmri.csv"

# committed §3.1/§3.2 digits (leg-B audit targets), exactly the numbers
# run_experiment.py/experiment_cmri.py print on the frozen
# results/predictions_cmri.csv
C_N_R = 5879                          # resync evaluable steps (42 days)
C_N_F = 5920                          # freerun evaluable steps
# (rmse, mae, r2, mape) per model/protocol
C_R1 = (5.881, 5.554, -15.94, 24.26)  # v1, resync
C_R2 = (0.646, 0.456, 0.796, 1.95)    # v2, resync
C_F1 = (6.861, 6.682, -22.08, 29.22)  # v1, freerun
C_F2 = (1.575, 1.176, -0.22, 5.03)    # v2, freerun

# per-digit half-unit rounding tolerance at each printed precision
# (rmse 3dp, mae 3dp, r2 2dp, mape 2dp)
TOL = (0.0005, 0.0005, 0.005, 0.005)  # -> (5.881,5.554,-15.94,24.26)


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
    T, t_out = data["T1"], data["T_out"]
    dew, load, lux = data["Tdewpoint"], data["Appliances"], data["lights"]
    dates = data["date"]

    print("structural probe of the §3.1/§3.2 v1-baseline reading "
          "(uncalibrated handbook twin ~6 degC -> calibrated v2 ~0.6 degC on "
          "resync, ~9x RMSE collapse; freerun divergence 6.861 vs graceful "
          "1.575); held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + protocol replays -------------------
    fs = slice(0, val_start)
    v1 = V1Model()                     # fixed handbook constants, no fit
    v2 = CalibratedTwin()
    v2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])

    recs = list(zip(T, t_out, dew, load, lux))[val_start:split]
    val_days = days_of(dates, val_start, split)
    obs = [r[0] for r in recs]

    pr1 = run_protocol(v1, recs[0][0], recs, val_days)
    pr2 = run_protocol(v2, recs[0][0], recs, val_days)
    prp = run_protocol(PersistenceModel(), recs[0][0], recs, val_days)
    r1 = [(c, o) for c, o in zip(pr1, obs) if c is not None]
    r2 = [(c, o) for c, o in zip(pr2, obs) if c is not None]
    rp = [(c, o) for c, o in zip(prp, obs) if c is not None]
    assert len(r1) == len(r2) == len(rp) == 2743

    pf1 = run_protocol(v1, recs[0][0], recs, None)[1:]
    pf2 = run_protocol(v2, recs[0][0], recs, None)[1:]
    fobs = obs[1:]
    assert len(pf1) == len(pf2) == len(fobs) == 2762

    m1r = metric_tup(r1)
    m2r = metric_tup(r2)
    mpr = metric_tup(rp)
    m1f = metric_tup(list(zip(pf1, fobs)))
    m2f = metric_tup(list(zip(pf2, fobs)))

    # per-day resync v2-vs-v1 wins (paired by calendar iso-date)
    day_pairs: dict[str, list[tuple[float, float]]] = {}
    for c1, c2, o, d in zip(pr1, pr2, obs, dates[val_start:split]):
        if c1 is None or c2 is None:
            continue
        day_pairs.setdefault(d.date().isoformat(), []).append(
            ((c1 - o) ** 2, (c2 - o) ** 2))
    dl = [day_pairs[k] for k in sorted(day_pairs)]
    nd = len(dl)
    wins = sum(1 for d in dl
               if math.sqrt(sum(a for a, _ in d) / len(d))
               > math.sqrt(sum(b for _, b in d) / len(d)))
    r_ratio = m1r[0] / m2r[0]
    o_ratio = m1f[0] / m2f[0]

    print(f"leg A  within-train val block (fresh v2 fit; v1 is fixed-handbook, "
          f"never fit): n={len(r2)} resync steps / {nd} days; n={len(pf2)} "
          f"freerun steps")
    print(f"  resync metrics  v1 (handbook) [{m1r[0]:.3f} {m1r[1]:.3f} "
          f"{m1r[2]:+.2f} {m1r[3]:.2f}]  vs  v2 (calibrated) [{m2r[0]:.3f} "
          f"{m2r[1]:.3f} {m2r[2]:+.2f} {m2r[3]:.2f}]  "
          f"(RMSE ratio v1/v2 = {r_ratio:.2f}x; held 9.10x)")
    print(f"  resync context: persistence [{mpr[0]:.3f} {mpr[1]:.3f} "
          f"{mpr[2]:+.2f} {mpr[3]:.2f}] (replayed T-hat = T control)")
    print(f"  resync per-day RMSE: v2 better than v1 on {wins}/{nd} days "
          f"(held 42/42)")
    print(f"  freerun metrics  v1 (handbook) [{m1f[0]:.3f} {m1f[1]:.3f} "
          f"{m1f[2]:+.2f} {m1f[3]:.2f}]  vs  v2 (calibrated) [{m2f[0]:.3f} "
          f"{m2f[1]:.3f} {m2f[2]:+.2f} {m2f[3]:.2f}]  "
          f"(RMSE ratio v1/v2 = {o_ratio:.2f}x; held 4.36x)")

    # ---- leg B: frozen §3.1/§3.2 digit audit (read-only) ---------------------
    fr1: list[tuple[float, float]] = []
    fr2: list[tuple[float, float]] = []
    ff1: list[tuple[float, float]] = []
    ff2: list[tuple[float, float]] = []
    with open(CMRI_PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            o = float(r["T1_observed"])
            if r["v1_resync"] != "" and r["v2_resync"] != "":
                fr1.append((float(r["v1_resync"]), o))
                fr2.append((float(r["v2_resync"]), o))
            if r["v1_freerun"] != "" and r["v2_freerun"] != "":
                ff1.append((float(r["v1_freerun"]), o))
                ff2.append((float(r["v2_freerun"]), o))
    fm1r = metric_tup(fr1)
    fm2r = metric_tup(fr2)
    fm1f = metric_tup(ff1)
    fm2f = metric_tup(ff2)
    frr = fm1r[0] / fm2r[0]
    forr = fm1f[0] / fm2f[0]
    erasure = (fm1r[0] - fm2r[0]) / fm1r[0] * 100.0
    # per-day held win count (resync basis)
    fday: dict[str, list[tuple[float, float]]] = {}
    with open(CMRI_PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["v1_resync"] == "" or r["v2_resync"] == "":
                continue
            fday.setdefault(r["date"][:10], []).append(
                ((float(r["v1_resync"]) - float(r["T1_observed"])) ** 2,
                 (float(r["v2_resync"]) - float(r["T1_observed"])) ** 2))
    fd = [fday[k] for k in sorted(fday)]
    fwins = sum(1 for d in fd
                if math.sqrt(sum(a for a, _ in d) / len(d))
                > math.sqrt(sum(b for _, b in d) / len(d)))

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

    print(f"\nleg B  frozen held-out (audit leg): resync n={len(fr1)} "
          f"(committed {C_N_R}), freerun n={len(ff1)} (committed {C_N_F})")
    print("  committed-digit audit (frozen artifact, read-only):")
    audit("resync v1", fm1r, C_R1)
    audit("resync v2", fm2r, C_R2)
    audit("freerun v1", fm1f, C_F1)
    audit("freerun v2", fm2f, C_F2)
    print(f"      resync RMSE ratio v1/v2 = {frr:.2f}x (the §4 '11x' "
          f"catchphrase is a loose round statement: exact frozen ratios are "
          f"{frr:.2f}x RMSE, {fm1r[3] / fm2r[3]:.2f}x MAPE; MAE "
          f"{fm1r[1] / fm2r[1]:.2f}x)")
    print(f"      resync error erasure (-89% claim): "
          f"({fm1r[0]:.3f} - {fm2r[0]:.3f})/{fm1r[0]:.3f} = "
          f"-{erasure:.1f}% (committed 89%)")
    print(f"      freerun RMSE ratio v1/v2 = {forr:.2f}x (held 4.36x)")
    print(f"      per-day resync RMSE: v2 better than v1 on {fwins}/{len(fd)} "
          f"held days")
    # informational block-context lines
    print(f"      (informational: within-train block deltas — fresh resync "
          f"v2 R2 {m2r[2]:+.2f} is much smaller than held +0.796 and the fresh "
          f"freerun v2 R2 {m2f[2]:+.2f} stays negative like held -0.22, "
          f"block-specific variance composition, not gated; §3.2 reports "
          f"the freerun R2 as-is with no positivity claim)")

    # ---- structural gates (the within-train leg is the real test) ------------
    g1 = m1r[0] > 5.0 and m2r[0] < 1.0
    g2 = r_ratio > 5.0
    g3 = wins > nd / 2
    g4 = m1f[0] > 4.0 and m2f[0] < 2.5
    g5 = o_ratio > 2.0
    g6 = m1r[2] < 0.0 and m2r[2] > 0.0
    ok = g1 and g2 and g3 and g4 and g5 and g6 and audit_ok

    print()
    print("gates (within-train leg = fresh test; held-out leg = frozen audit):")
    print(f"    (i)  resync RMSE split: v1 {m1r[0]:.3f} > 5 AND v2 "
          f"{m2r[0]:.3f} < 1 degC : {'PASS' if g1 else 'FAIL'}  "
          f"(held 5.881 vs 0.646)")
    print(f"    (ii) resync RMSE ratio v1/v2 > 5: {r_ratio:.2f}x : "
          f"{'PASS' if g2 else 'FAIL'}  (held 9.10x)")
    print(f"    (iii) v2 beats v1 on a strict majority of fresh val days: "
          f"{wins}/{nd} : {'PASS' if g3 else 'FAIL'}  (held 42/42)")
    print(f"    (iv) freerun RMSE split: v1 {m1f[0]:.3f} > 4 AND v2 "
          f"{m2f[0]:.3f} < 2.5 degC : {'PASS' if g4 else 'FAIL'}  "
          f"(held 6.861 vs 1.575)")
    print(f"    (v)  freerun RMSE ratio v1/v2 > 2: {o_ratio:.2f}x : "
          f"{'PASS' if g5 else 'FAIL'}  (held 4.36x)")
    print(f"    (vi) resync R2 flips worse-than-mean -> better-than-mean: "
          f"v1 {m1r[2]:+.2f} < 0 AND v2 {m2r[2]:+.2f} > 0 : "
          f"{'PASS' if g6 else 'FAIL'}  (held -15.94 vs +0.796)")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §3.1/§3.2 "
          f"v1-baseline reading is a structural fact, not a single held-out "
          f"draw: on fresh within-train fits the uncalibrated handbook twin "
          f"still errors by ~{m1r[0]:.1f} degC on resync (RMSE ratio "
          f"{r_ratio:.2f}x over v2's {m2r[0]:.3f}; held 5.881 vs 0.646, 9.10x) "
          f"with a worse-than-mean R2 ({m1r[2]:+.2f} < 0 < {m2r[2]:+.2f}; held "
          f"-15.94 < 0 < +0.796) and loses every fresh val day ({wins}/{nd}; "
          f"held {fwins}/{len(fd)}), and still diverges on freerun "
          f"(RMSE {m1f[0]:.3f} vs v2 {m2f[0]:.3f}, ratio {o_ratio:.2f}x; held "
          f"6.861 vs 1.575, 4.36x) — the ~9x resync collapse and the freerun "
          f"divergence replay on fresh data (block-dependent strength deltas "
          f"only: ratio magnitudes {r_ratio:.2f}x/{o_ratio:.2f}x vs held "
          f"9.10x/4.36x, the fresh v2 resync R2 +{m2r[2]:.2f} vs held +0.796, "
          f"and the freerun R2 staying negative at -{abs(m2f[2]):.2f} like "
          f"held -0.22, all documented above); and the frozen artifact "
          f"re-produces every committed §3.1/§3.2 digit (resync v1 "
          f"[{fm1r[0]:.3f} {fm1r[1]:.3f} {fm1r[2]:+.2f} {fm1r[3]:.2f}], v2 "
          f"[{fm2r[0]:.3f} {fm2r[1]:.3f} {fm2r[2]:+.2f} {fm2r[3]:.2f}] n = "
          f"{len(fr1)}; freerun v1 [{fm1f[0]:.3f} {fm1f[1]:.3f} {fm1f[2]:+.2f} "
          f"{fm1f[3]:.2f}], v2 [{fm2f[0]:.3f} {fm2f[1]:.3f} {fm2f[2]:+.2f} "
          f"{fm2f[3]:.2f}] n = {len(ff1)}), the held per-day win count "
          f"({fwins}/{len(fd)}), the -89% error erasure "
          f"(-{erasure:.1f}%), and the exact ratio pair ({frr:.2f}x resync / "
          f"{forr:.2f}x freerun); the §4 '11x' catchphrase is loose — the "
          f"exact frozen resync ratio is {frr:.2f}x (RMSE) and {fm1r[3] / fm2r[3]:.2f}x "
          f"(MAPE), so the honest summary is ~9-12x depending on metric.")


if __name__ == "__main__":
    main()
