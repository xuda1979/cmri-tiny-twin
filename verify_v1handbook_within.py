#!/usr/bin/env python3
"""verify_v1handbook_within.py — is the §2.1 v1-handbook reading a true
structural fact about the shipped code?  The reading says tiny_twin.py is "a
deliberately lazy first-order lumped RC thermal model" with FIXED
building-handbook defaults — `a_env = 0.02`, `g_set = 0.02`, `T_set = 21 °C`,
`a_nrg = 0.01` — a clamped thermostat `g_set·max(T_set − T, 0)`, and "v1 never
reads, fits, or otherwise touches the data: it is the uncalibrated baseline".

Prior probes gate the CONSEQUENCE of the reading (verify_v1baseline_within
item 35 gates the ~9x RMSE collapse; verify_metrictuple_within item 33 the
frozen tuples) but nothing audits that the shipped tiny_twin.py source itself
matches §2.1's description: no probe parsed the module for the committed
literals, checked that the step equation has the clamped-thermostat shape
described, verified the no-data-I/O claim structurally, or checked that an
independent-from-scratch build from ONLY the REPORT §2.1 sentence reproduces
the model's behavior and every committed digit.  This probe closes that gap.

Leg A (source-constant structural audit):
  (i)   tiny_twin.py's source literals (T_SET = 21.0, A_ENV = 0.02,
        G_SET = 0.02, A_NRG = 0.01, Q_SCALE = 100.0) equal the committed
        values, the module attributes export them, and REPORT.md §2.1's
        sentence names the exact same constants;
  (ii)  the step equation has the §2.1 clamped-thermostat shape: an envelope
        term a_env·(T_out − T), a clamped heating gain g_set·max(T_set − T, 0)
        (never negative heating), and an appliance term a_nrg·(load/100);
  (iii) the no-data-I/O claim holds structurally: the module performs no
        file/network/subprocess input anywhere (no open(, csv, pathlib,
        numpy/pandas, json, urllib, requests, subprocess, input(), .read,
        .write, os., import sys), the only import is the future-annotations
        boilerplate, and run_experiment.V1Model's adapter demonstrably drops
        dew/lux (varying them changes nothing) while load still matters;
  (iv)  an independent-from-scratch reimplementation built ONLY from the
        REPORT §2.1 equation + the four committed literals (V1Recon — with no
        import of tiny_twin) reproduces tiny_twin.step EXACTLY on a
        5x5x5 = 125-point sweep over temperature/outdoor/appliance load
        (max |delta| < 1e-12), i.e. the report's sentence is a faithful
        description of the code;
  (v)   fresh-consequence: the from-scratch §2.1 model (V1Recon) replayed
        against a fresh v2 (CalibratedTwin refit on [0,11051)) on the
        within-train val block [11051,13814) still reproduces the §3.1
        structural collapse without touching the held-out test — resync v1
        RMSE > 4 degC while fresh calibrated v2 RMSE < 1.5 degC (the
        "uncalibrated ~6 degC vs calibrated ~0.6 degC" split replays through
        the from-scratch construction).

Leg B (regeneration trace — the strongest "never touches data" evidence):
  (vi)  the frozen results/predictions_cmri.csv v1 columns are re-generated
        step-by-step from the raw data/energydata_complete.csv using ONLY
        tiny_twin.step (no fit of any kind): all 5879 resync + 5920 freerun
        frozen predictions match the regenerated series within 0.001 (they
        were written at 3dp), and the metric tuples re-derived from the
        regenerated series reproduce EVERY committed §3.1 digit (resync v1
        [5.881 5.554 -15.94 24.26], freerun v1 [6.861 6.682 -22.08 29.22])
        within the half-unit rounding tolerance.

Gates are the (i)-(vii) invariants above; exact block-magnitude digits on the
fresh val block stay informational (the from-scratch model is by construction
identical to the code model, so its fresh metrics equal the item-35 leg-A
numbers; only the resync split direction is gated here, to avoid duplicating
item 35's ratio gates).

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import math
import pathlib
import re
import statistics
import sys

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, V1Model, metrics, run_protocol   # noqa: E402
from experiment_cmri import days_of, load_cmri_data                     # noqa: E402
from tiny_twin_v2 import CalibratedTwin                                 # noqa: E402
import tiny_twin                                                        # noqa: E402

VALID_FRAC = 0.2
CMRI_PRED = "results/predictions_cmri.csv"
TINY = pathlib.Path("tiny_twin.py")
REPORT = pathlib.Path("REPORT.md")

# committed §2.1 handbook defaults (exactly as REPORT.md names them)
COMMITTED = {"T_SET": 21.0, "A_ENV": 0.02, "G_SET": 0.02, "A_NRG": 0.01,
             "Q_SCALE": 100.0}

# committed §3.1 v1 digits (leg-B audit targets)
C_N_R = 5879                          # frozen resync evaluable steps
C_N_F = 5920                          # frozen freerun evaluable steps
C_R1 = (5.881, 5.554, -15.94, 24.26)  # v1 resync  (rmse, mae, r2, mape)
C_F1 = (6.861, 6.682, -22.08, 29.22)  # v1 freerun
TOL = (0.0005, 0.0005, 0.005, 0.005)  # half-unit rounding at printed precision

# code-level I/O surfaces that would violate "never reads, fits, or otherwise
# touches the data" (deliberately NOT bare "read" etc: docstrings use the
# words "reads"/"fits" as prose, so only call/import-shaped tokens count)
FORBIDDEN = ["open(", "csv", "pathlib", "numpy", "pandas", "json", "urllib",
             "requests", "subprocess", "input(", "os.", ".read", ".write",
             "import sys", "imp.load", "__import__"]

# source literals as they must appear, allowing a trailing comment
# (name = <value>  [# comment]) — tiny_twin.py carries one per constant
LIT_RE = re.compile(
    r"^(T_SET|A_ENV|G_SET|A_NRG|Q_SCALE)\s*=\s*([\d.]+)\s*(#.*)?$", re.M)


class V1Recon:
    """Independent-from-scratch v1, built ONLY from the REPORT §2.1 sentence:

        T(t+1) = T(t) + a_env·(T_out − T(t)) + g_set·max(T_set − T(t), 0)
                 + a_nrg·Appliances(t)/100

    with a_env = 0.02, g_set = 0.02, T_set = 21 °C, a_nrg = 0.01.  No import
    of tiny_twin and no reading of its source: this class is the honest
    "what the report alone tells you to build" reproduction."""

    T_SET, A_ENV, G_SET, A_NRG = 21.0, 0.02, 0.02, 0.01

    def step(self, temp, t_out, dew, load, lux):
        heating = max(self.T_SET - temp, 0.0)          # clamped thermostat
        d_temp = (self.A_ENV * (t_out - temp)
                  + self.G_SET * heating
                  + self.A_NRG * (load / 100.0))
        return temp + d_temp

    def __call__(self, temp, t_out, dew, load, lux):
        return self.step(temp, t_out, dew, load, lux)


def metric_tup(pred, obs) -> tuple[float, float, float, float]:
    m = metrics(pred, obs)
    return (m["rmse"], m["mae"], m["r2"], m["mape"])


def main() -> None:
    # ---- leg A(i): source literals vs module attrs vs REPORT text -----------
    src = TINY.read_text(encoding="utf-8")
    lit = {k: float(v) for k, v, _ in LIT_RE.findall(src)}
    attr = {k: getattr(tiny_twin, k) for k in COMMITTED}
    report_text = REPORT.read_text(encoding="utf-8")
    rep_ok = bool(re.search(
        r"`a_env = 0\.02`, `g_set = 0\.02`, `T_set = 21 [^\`]*`, `a_nrg = 0\.01`",
        report_text))

    g_i = (set(lit) == set(COMMITTED)
           and all(abs(lit[k] - v) < 1e-12 for k, v in COMMITTED.items())
           and all(abs(attr[k] - v) < 1e-12 for k, v in COMMITTED.items())
           and rep_ok)

    # ---- leg A(ii): step equation shape (clamped thermostat, all 3 terms) ----
    # implement exactly the reported form; check the source spells each term
    g_ii = (re.search(r"max\([^)]*-\s*(temp|T)\s*,\s*0", src)      # clamp
            is not None
            and "a_env * (t_out - temp)" in src                    # envelope
            and "g_set * heating" in src                           # heating
            and "a_nrg * (load_wh / q_scale)" in src               # appliance
            and re.search(r"T[t_]*\s*\+=|d_temp\s*=", src)
            is not None)
    # behavioral clamp check: heating is never negative -> v1 never cools
    # toward the setpoint, only the envelope can cool
    body = tiny_twin.step(20.0, 30.0, 0.0)       # hot out, T below setpoint
    g_ii &= body > 20.0
    # above the setpoint the heating term is exactly zero: the prediction
    # equals the pure envelope term, and ANY t_set <= T gives the same answer
    # (the thermostat cannot add heat once T >= T_set)
    above_any = tiny_twin.step(24.0, 20.0, 0.0)  # T=24 > T_SET=21: heating off
    g_ii &= above_any == 24.0 + 0.02 * (20.0 - 24.0)      # pure envelope
    g_ii &= tiny_twin.step(24.0, 20.0, 0.0, t_set=10.0) == above_any
    # and heating only ever raises T: with T below the setpoint and a fixed
    # envelope, a bigger g_set means a warmer next T
    g_ii &= tiny_twin.step(18.0, 10.0, 0.0, t_set=21.0, g_set=1.0) > \
        tiny_twin.step(18.0, 10.0, 0.0, t_set=21.0, g_set=0.001)

    # ---- leg A(iii): no data I/O + adapter drops dew/lux ---------------------
    hits = [tok for tok in FORBIDDEN if tok in src]
    imports = [ln for ln in src.splitlines()
               if ln.startswith(("import ", "from "))]
    g_iii = (not hits
             and imports == ["from __future__ import annotations"]
             and "open(" not in src)
    vm = V1Model()
    dew_lux_free = (
        vm.step(20.0, 15.0, 8.0, 120.0, 50.0)
        == vm.step(20.0, 15.0, 40.0, 120.0, 900.0))   # dew/lux irrelevant
    load_matters = (
        vm.step(20.0, 15.0, 30.0, 0.0, 500.0)
        != vm.step(20.0, 15.0, 30.0, 1200.0, 500.0))  # appliance input kept
    g_iii &= dew_lux_free and load_matters

    # ---- leg A(iv): from-scratch reproduction == tiny_twin.step ---------------
    recon = V1Recon()
    worst = 0.0
    npt = 0
    for temp in (15.0, 18.0, 21.0, 24.0, 28.0):
        for t_out in (-5.0, 0.0, 10.0, 20.0, 30.0):
            for load in (0.0, 50.0, 200.0, 600.0, 1200.0):
                a = tiny_twin.step(temp, t_out, load)
                b = recon.step(temp, t_out, 0.0, load, 0.0)
                worst = max(worst, abs(a - b))
                npt += 1
    g_iv = worst < 1e-12
    # equivalence through the protocol-facing adapter too: recon answers
    # identically to V1Model when driven by run_protocol
    data = load_cmri_data("data/energydata_complete.csv")
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    val_start = int(round((1.0 - VALID_FRAC) * split))
    T, t_out, dew, load, lux = (data["T1"], data["T_out"], data["Tdewpoint"],
                                data["Appliances"], data["lights"])
    dates = data["date"]
    recs = list(zip(T, t_out, dew, load, lux))[val_start:split]
    val_days = days_of(dates, val_start, split)
    pvn = run_protocol(V1Model(), recs[0][0], recs, val_days)
    pvr = run_protocol(recon, recs[0][0], recs, val_days)
    g_iv &= all(abs(a - b) < 1e-12 for a, b in zip(pvn, pvr) if a is not None)

    # ---- leg A(v): fresh-consequence — §2.1 model vs fresh v2 on val block ----
    fs = slice(0, val_start)
    v2 = CalibratedTwin()
    v2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    obs = [r[0] for r in recs]
    pr1 = run_protocol(recon, recs[0][0], recs, val_days)
    pr2 = run_protocol(v2, recs[0][0], recs, val_days)
    r1 = [(c, o) for c, o in zip(pr1, obs) if c is not None]
    r2 = [(c, o) for c, o in zip(pr2, obs) if c is not None]
    m1r = metric_tup(*zip(*r1))
    m2r = metric_tup(*zip(*r2))
    g_v = m1r[0] > 4.0 and m2r[0] < 1.5

    # ---- leg B(vi): regenerate the frozen v1 columns from raw data ------------
    # resync: per-day re-anchor from observed T1, step within day (as
    # run_protocol/V1Model did); freerun: single continuous run from T[split].
    n_block = n - split                       # 5921 pred slots (0..5920)
    pred_r = [None] * n_block
    for day in days_of(dates, split):
        cur = T[split + day[0]]
        for a, b in zip(day[:-1], day[1:]):
            cur = tiny_twin.step(cur, t_out[split + a], load[split + a])
            pred_r[b] = cur
    pred_f = [None] * n_block
    cur = T[split]
    for b in range(1, n_block):
        cur = tiny_twin.step(cur, t_out[split + b - 1], load[split + b - 1])
        pred_f[b] = cur
    p_r = pred_r[1:]                          # CSV row k <-> pred slot k+1
    p_f = pred_f[1:]

    n_bad_r = n_bad_f = 0
    with open(CMRI_PRED, encoding="utf-8") as fh:
        for k, row in enumerate(csv.DictReader(fh)):
            if row["v1_resync"] != "" and abs(float(row["v1_resync"]) -
                                              p_r[k]) > 0.001:
                n_bad_r += 1
            if row["v1_freerun"] != "" and abs(float(row["v1_freerun"]) -
                                               p_f[k]) > 0.001:
                n_bad_f += 1
    gr = [p_r[k] for k in range(len(p_r)) if pred_r[k + 1] is not None]
    gor = [T[split + 1 + k] for k in range(len(p_r))
           if pred_r[k + 1] is not None]
    gf = [p_f[k] for k in range(len(p_f)) if pred_f[k + 1] is not None]
    gof = [T[split + 1 + k] for k in range(len(p_f))
           if pred_f[k + 1] is not None]
    assert len(gr) == C_N_R and len(gf) == C_N_F
    mr = metric_tup(gr, gor)
    mf = metric_tup(gf, gof)
    g_vi = n_bad_r == 0 and n_bad_f == 0 and (
        all(abs(g - c) <= t for g, c, t in zip(mr, C_R1, TOL))
        and all(abs(g - c) <= t for g, c, t in zip(mf, C_F1, TOL)))

    ok = g_i and g_ii and g_iii and g_iv and g_v and g_vi

    print("structural probe of the §2.1 v1-handbook reading (tiny_twin.py's "
          "fixed constants are the committed defaults, the step has the "
          "clamped-thermostat equation §2.1 describes, v1 never reads data, "
          "and the fresh val block replays the collapse through a from-scratch "
          "build); held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start}; val [{val_start},{split}) = "
          f"{split - val_start} samples ({len(days_of(dates, val_start, split))} days)")
    print()
    print("leg A  source-constant structural audit:")
    print(f"  (i)  literals: source {sorted(lit)} == committed {sorted(COMMITTED)} "
          f": module attrs == committed : REPORT §2.1 names same 4 : "
          f"{'PASS' if g_i else 'FAIL'}")
    print(f"  (ii) equation shape: clamp + envelope + heating + appliance "
          f"terms in source, heating never cools (T above T_set -> load "
          f"independent of T_set side): {'PASS' if g_ii else 'FAIL'}")
    print(f"  (iii) no data I/O: forbidden API tokens {hits} ; only import is "
          f"future-annotations {imports} ; V1Model drops dew/lux "
          f"({'PASS' if g_iii else 'FAIL'})")
    print(f"  (iv) from-scratch §2.1 recon == tiny_twin.step on {npt} sweep "
          f"points (max |d| {worst:.2e}) and through run_protocol on the val "
          f"block: {'PASS' if g_iv else 'FAIL'}")
    print(f"  (v)  fresh consequence on val block: recon-v1 resync "
          f"[{m1r[0]:.3f} {m1r[1]:.3f} {m1r[2]:+.2f} {m1r[3]:.2f}] vs "
          f"fresh-v2 [{m2r[0]:.3f} {m2r[1]:.3f} {m2r[2]:+.2f} {m2r[3]:.2f}] "
          f"(RMSE {m1r[0]:.3f} > 4 and {m2r[0]:.3f} < 1.5): "
          f"{'PASS' if g_v else 'FAIL'}")
    print(f"  (informational: the recon is identical to the code model, so "
          f"fresh magnitudes are the item-35 leg-A numbers; only the split "
          f"direction is gated here)")
    print()
    print("leg B  regeneration trace (frozen v1 columns rebuilt step-by-step "
          "from raw data, no fit of any kind):")
    print(f"  resync: {C_N_R} predictions regenerated, mismatches vs frozen "
          f"{n_bad_r}; freerun: {C_N_F} predictions, mismatches {n_bad_f}")
    print(f"  regenerated tuples: resync v1 [{mr[0]:.3f} {mr[1]:.3f} "
          f"{mr[2]:+.2f} {mr[3]:.2f}] (committed [{' '.join(f'{c:.3f}' if i < 2 else f'{c:+.2f}' for i, c in enumerate(C_R1))}]); "
          f"freerun v1 [{mf[0]:.3f} {mf[1]:.3f} {mf[2]:+.2f} {mf[3]:.2f}] "
          f"(committed [{' '.join(f'{c:.3f}' if i < 2 else f'{c:+.2f}' for i, c in enumerate(C_F1))}]): "
          f"{'PASS' if g_vi else 'FAIL'}")
    print()

    ok = all([g_i, g_ii, g_iii, g_iv, g_v, g_vi])
    print("gates (source structure + regeneration trace):")
    print(f"    (i)  committed literals in source = module attrs = REPORT "
          f"§2.1 text : {'PASS' if g_i else 'FAIL'}")
    print(f"    (ii) step equation has the §2.1 clamped-thermostat shape : "
          f"{'PASS' if g_ii else 'FAIL'}")
    print(f"    (iii) no data I/O anywhere in the module ; V1Model drops "
          f"dew/lux : {'PASS' if g_iii else 'FAIL'}")
    print(f"    (iv) from-scratch §2.1 recon reproduces tiny_twin.step "
          f"(<1e-12 on {npt} pts + val-block protocol replay) : "
          f"{'PASS' if g_iv else 'FAIL'}")
    print(f"    (v)  fresh val block: recon-v1 resync RMSE "
          f"{m1r[0]:.3f} > 4 AND fresh-v2 {m2r[0]:.3f} < 1.5 : "
          f"{'PASS' if g_v else 'FAIL'}")
    print(f"    (vi) all {C_N_R + C_N_F} frozen v1 predictions regenerated "
          f"(resync mismatch {n_bad_r}, freerun mismatch {n_bad_f}) and both "
          f"committed tuples re-derived exactly : "
          f"{'PASS' if g_vi else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §2.1 v1-handbook reading "
          f"is a true structural fact about the shipped code: tiny_twin.py "
          f"exports exactly the committed defaults (a_env = 0.02, g_set = "
          f"0.02, T_set = 21 degC, a_nrg = 0.01) in the clamped-thermostat "
          f"equation §2.1 prints, contains no data-I/O surface anywhere (only "
          f"the future-annotations import), and an independent build from ONLY "
          f"the §2.1 sentence reproduces tiny_twin.step to <1e-12 on every "
          f"sweep point — so the report is a faithful description of the code, "
          f"the model is genuinely data-free (regeneration of all "
          f"{C_N_R + C_N_F} frozen v1 predictions from raw data uses no fit, "
          f"and both committed v1 tuples [{'/'.join(f'{c:.3f}' for c in C_R1[:2])}"
          f" {'/'.join(f'{c:+.2f}' for c in C_R1[2:])}] resync, "
          f"[{'/'.join(f'{c:.3f}' for c in C_F1[:2])} "
          f"{'/'.join(f'{c:+.2f}' for c in C_F1[2:])}] freerun) re-derive), "
          f"and the §3.1 collapse replays on the fresh val block with the "
          f"from-scratch construction (resync {m1r[0]:.3f} degC uncalibrated "
          f"vs {m2r[0]:.3f} degC fresh-calibrated).")


if __name__ == "__main__":
    main()
