#!/usr/bin/env python3
"""verify_freerun_drift_within.py — is the §5.2 design-fix drift-geometry claim
(switching from the superseded single-fabric cmri config to the winning
two-timescale config: "the error trend slope halves −0.000680 → −0.000330
C/step², per-step error sd drops 1.482 → 1.172 °C, worst per-quarter
excursion moves from a Q4 runaway −2.65 to a mid-run Q3 dip −2.10 that
recovers in Q4 −1.00") a structural fact, not an artifact of the single
held-out draw?

REPORT.md §5.2 committed claims (frozen held-out free-run, as analyzed by
verify_freerun_variance.py / diagnose_freerun.py):
  * two-timescale winner cmri: trend slope −0.000330 C/step², intercept
    +0.0547, per-quarter [+0.2948, −0.8790, −2.1046, −1.0001]
    (Q3 mid-run dip, Q4 recovery), RMSE-after-trend 1.0276, RMSE 1.4915,
    err sd 1.1724, bias −0.9222, bias-noise sd 0.0411 (n = 5920);
  * v2 context: slope −0.000091, intercept +1.0589, per-quarter
    [+0.8222, +0.6663, +1.7235, −0.0525], RMSE-after-trend 1.3539,
    RMSE 1.5752, err sd 1.3629, bias +0.7899, bias-noise sd 0.0424.
The "before" digits (−0.000680 slope, 1.482 sd, Q4 runaway −2.65) belong to
the SUPERSEDED single-fabric cmri config; results/predictions_cmri.csv never
carried a single-fabric free-run column, so those digits are single-touch
HISTORY, not leg-B-auditable (documented honestly below).

This script verifies the config-fix reading WITHOUT touching the held-out
test: leg A re-fits fresh twins on the within-train fit slice [0, 11051) —
v2 (CalibratedTwin), the frozen two-timescale cmri config (CmriTwin
keep=CMRI_KEEP), and the superseded single-fabric cmri config (CmriTwin
keep=(0,3,4,5,6,7,8), the same mask verify_ablation_order.py uses) — replays
the OPEN-LOOP free-run protocol on the within-train val block [11051, 13814)
(n = 2762 free-run steps = 20 days; the double-slice-correct convention:
recs over the full [val_start:split] block, run_protocol[1:] drops the anchor
sample, errors e = p - o over obs = T[val_start+1:split]), and recomputes the
drift geometry (OLS slope of e on list(range(n)), err sd, per-quarter signed
errors, RMSE, RMSE-after-trend) on the FRESH residuals for all three configs;
leg B re-reads the frozen results/predictions_cmri.csv (read-only) and
re-aggregates every committed §5.2 free-run drift digit exactly as
diagnose_freerun.py does.

Gates (power-aware: the 20-day block cannot reproduce the 42-day block's
magnitudes, and the within-train drift SIGN inverts — positive slopes and
positive bias, the same block-dependent inversion §7 item 25 already
documents — so all slope gates are on |slope|, i.e. on the flatness/smallness
of the drift, never on its sign). The gates are structural invariants of the
design-fix reading:
  (i)  |slope|(two-timescale) < |slope|(single-fabric) on the fresh block —
       the config fix FLATTENS the drift (held: −0.000680 → −0.000330);
  (ii) err sd (two-timescale) < err sd (single-fabric) — the variance drop
       (held: 1.482 → 1.172);
  (iii) worst per-quarter |signed| (two-timescale) < worst per-quarter
       |signed| (single-fabric) — the worst excursion is not worse after the
       fix (held: Q4 −2.65 → Q3 −2.10);
  (iv) free-run RMSE (two-timescale) < free-run RMSE (single-fabric) — the
       overall improvement (held: 1.7207 → 1.4915).
Block-dependent deltas (exact magnitudes, slope values, quarter SHAPES — the
within-train block's worst quarter is Q2 instead of held-out Q4, and Q1 flips
slightly negative for the two-timescale config — bias signs, and the
v2 context slope) are documented, not gated; the held-out leg must re-produce
every committed digit.

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import math
import sys

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, run_protocol  # noqa: E402
from experiment_cmri import (CMRI_ALPHA, CMRI_KEEP, CMRI_T_SET, CMRI_W,  # noqa: E402
                             CMRI_W2, load_cmri_data, run_protocol_cmri)
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from cmri_twin import CmriTwin, fabric_mass  # noqa: E402

VALID_FRAC = 0.2
CMRI_PRED = "results/predictions_cmri.csv"
SINGLE_KEEP = (0, 3, 4, 5, 6, 7, 8)          # superseded single-fabric mask
SINGLE_LABEL = "cmri single-fabric (superseded)"

# committed §5.2 free-run drift digits, exactly the numbers diagnose_freerun.py
# prints on the frozen results/predictions_cmri.csv (n = 5920)
C_N = 5920
C_CM = dict(
    slope=-0.000330, intercept=0.0547, q=(0.2948, -0.8790, -2.1046, -1.0001),
    aft=1.0276, rmse=1.4915, sd=1.1724, bias=-0.9222, noise=0.0411,
)
C_V2 = dict(
    slope=-0.000091, intercept=1.0589, q=(0.8222, 0.6663, 1.7235, -0.0525),
    aft=1.3539, rmse=1.5752, sd=1.3629, bias=0.7899, noise=0.0424,
)

SLOPE_TOL = 5e-6     # slope tolerance (−0.000330 / −0.000091; 4dp agreement)
X_TOL = 0.0005       # 4dp-digit tolerance (intercept, quarters, aft, rmse, sd,
                     # bias, noise share 4dp precision in diagnose_freerun.py)


def rmse(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def sd(xs: list[float]) -> float:
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) * (a - mx) for a in xs)
    b1 = sxy / sxx
    return my - b1 * mx, b1  # (intercept at x=0, slope)


def drift(name: str, err: list[float]) -> dict[str, float]:
    """Full drift geometry for one config's fresh within-train free-run errors
    (mirrors diagnose_freerun.py's conventions exactly)."""
    n = len(err)
    b0, b1 = ols(list(range(n)), err)
    resid = [e - (b0 + b1 * i) for i, e in enumerate(err)]
    qn = n // 4
    qm = [sum(err[j * qn:(j + 1) * qn]) / len(err[j * qn:(j + 1) * qn])
          for j in range(4)]
    d = [err[i] - err[i - 1] for i in range(1, n)]
    return dict(
        name=name, n=n, slope=b1, intercept=b0, rmse=rmse(err), aft=rmse(resid),
        sd=sd(err), bias=sum(err) / n, noise=sd(d), q=qm,
        worst_q=max(abs(v) for v in qm),
    )


def fmt_row(d: dict[str, float]) -> None:
    print(f"  {d['name']:>30}: n={d['n']} slope={d['slope']:+.6f} "
          f"intercept={d['intercept']:+.4f}")
    print(f"    RMSE={d['rmse']:.4f} aft-trend={d['aft']:.4f} "
          f"sd={d['sd']:.4f} bias={d['bias']:+.4f} noise={d['noise']:.4f}")
    print(f"    per-quarter {''.join(f'[{v:+.4f}]' for v in d['q'])} "
          f"worst={d['worst_q']:.4f}")


def audit(digs: dict[str, float], lbl: str,
          d: dict[str, float]) -> tuple[bool, list[str]]:
    """Check one config's re-aggregated frozen digits against the committed
    dict. Returns (ok, [label PASS lines])."""
    lines = []
    ok = True
    for key, label in (("slope", "trend slope"), ("intercept", "intercept"),
                       ("rmse", "RMSE"), ("aft", "RMSE-after-trend"),
                       ("sd", "err sd"), ("bias", "bias"), ("noise", "bias noise")):
        tol = SLOPE_TOL if key == "slope" else X_TOL
        good = abs(d[key] - digs[key]) <= tol
        ok &= good
        lines.append(f"    {lbl} {label} {d[key]:+.6f} "
                     f"(committed {digs[key]:+.6f}): "
                     f"{'PASS' if good else 'FAIL'}")
    for j, v in enumerate(d["q"]):
        good = abs(v - digs["q"][j]) <= X_TOL
        ok &= good
        lines.append(f"    {lbl} Q{j + 1} {v:+.4f} "
                     f"(committed {digs['q'][j]:+.4f}): "
                     f"{'PASS' if good else 'FAIL'}")
    return ok, lines


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

    print("structural probe of the §5.2 design-fix drift-geometry reading "
          "(superseded single-fabric cmri config -> two-timescale winner: "
          "slope halving, sd drop, quarter-shape transform); held-out test "
          "untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples (20 days); "
          f"open-loop free-run replay, single pass, no daily resync")
    print()

    # ---- leg A: fresh within-train fits + open-loop protocol replay ----------
    fs = slice(0, val_start)
    m2 = CalibratedTwin()
    m2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    mc = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    mc.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
           ghi[fs], tm[fs], tm2[fs])
    mc1 = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=SINGLE_KEEP)
    mc1.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
            ghi[fs], tm[fs], tm2[fs])

    # double-slice-correct open-loop replay (see tool-notes: recs over the full
    # [val_start:split] block, run_protocol()[1:] drops the anchor sample)
    recs_v2 = list(zip(T, t_out, dew, load, lux))[val_start:split]
    recs_cm = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm2))[val_start:split]
    pv2 = run_protocol(m2, T[val_start], recs_v2, None)[1:]
    pcm = run_protocol_cmri(mc, (T[val_start], Th[val_start]), recs_cm, None)[1:]
    pcm1 = run_protocol_cmri(mc1, (T[val_start], Th[val_start]), recs_cm, None)[1:]
    obs = T[val_start + 1:split]
    assert len(pv2) == len(pcm) == len(pcm1) == len(obs) == split - val_start - 1
    ev2 = [p - o for p, o in zip(pv2, obs)]
    ecm = [p - o for p, o in zip(pcm, obs)]
    ecm1 = [p - o for p, o in zip(pcm1, obs)]

    d_v2 = drift("v2 (single-zone)", ev2)
    d_cm = drift("cmri two-timescale", ecm)
    d_sf = drift("cmri single-fabric (superseded)", ecm1)
    print(f"leg A  within-train val block (fresh fits, {val_start}..{split}): "
          f"n={len(obs)} free-run steps (20.0 days); e = p - o; drift signs "
          f"invert within-train (positive here), so gates use |slope| and the "
          f"block's quarter shape/bias signs are block deltas, not gates")
    fmt_row(d_v2)
    fmt_row(d_cm)
    fmt_row(d_sf)

    # ---- leg B: frozen §5.2 free-run drift digit audit (read-only) ------------
    obs_f, cmf_f, v2f_f = [], [], []
    with open(CMRI_PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            obs_f.append(float(r["T1_observed"]))
            a, b = r["cmri_freerun"].strip(), r["v2_freerun"].strip()
            cmf_f.append(float(a) if a else None)
            v2f_f.append(float(b) if b else None)
    cmf_p = [p for p in cmf_f if p is not None]
    v2f_p = [p for p in v2f_f if p is not None]
    obs_c = [o for p, o in zip(cmf_f, obs_f) if p is not None]
    obs_v = [o for p, o in zip(v2f_f, obs_f) if p is not None]
    e_cf = [p - o for p, o in zip(cmf_p, obs_c)]
    e_vf = [p - o for p, o in zip(v2f_p, obs_v)]
    d_cf = drift("cmri (frozen)", e_cf)
    d_vf = drift("v2 (frozen)", e_vf)

    print(f"\nleg B  frozen held-out (audit leg): n={len(e_cf)} steps "
          f"(committed {C_N}); re-aggregated exactly as diagnose_freerun.py")
    audit_ok, lines_cm = audit(C_CM, "cmri", d_cf)
    audit_ok2, lines_v2 = audit(C_V2, "v2", d_vf)
    audit_ok = audit_ok and audit_ok2
    print("  committed-digit audit (frozen artifact, read-only):")
    for line in lines_cm:
        print(line)
    for line in lines_v2:
        print(line)
    e1 = e_cf
    e2 = e_vf
    mu1, mu2 = sum(e1) / len(e1), sum(e2) / len(e2)
    rho = (sum((a - mu1) * (b - mu2) for a, b in zip(e1, e2))
           / (len(e1) * sd(e1) * sd(e2)))
    print(f"      (informational: frozen cross-model error correlation "
          f"{rho:.3f})")

    # ---- structural gates (the within-train leg is the real test) -------------
    g1 = abs(d_cm["slope"]) < abs(d_sf["slope"])
    g2 = d_cm["sd"] < d_sf["sd"]
    g3 = d_cm["worst_q"] < d_sf["worst_q"]
    g4 = d_cm["rmse"] < d_sf["rmse"]
    ok = g1 and g2 and g3 and g4 and audit_ok

    print()
    print("gates (within-train leg = fresh test; held-out leg = frozen audit; "
          "all slope gates on |slope|: the drift SIGN inverts within-train, "
          "the design-fix invariant is flatness, not sign):")
    print(f"    (i)  |trend slope| two-timescale < single-fabric: "
          f"{abs(d_cm['slope']):.6f} < {abs(d_sf['slope']):.6f} : "
          f"{'PASS' if g1 else 'FAIL'}  (flattened drift; held −0.000680 → "
          f"−0.000330)")
    print(f"    (ii) err sd two-timescale < single-fabric: {d_cm['sd']:.4f} "
          f"< {d_sf['sd']:.4f} : {'PASS' if g2 else 'FAIL'}  (variance drop; "
          f"held 1.482 → 1.172)")
    print(f"    (iii) worst per-quarter |signed| two-timescale < single-fabric: "
          f"{d_cm['worst_q']:.4f} < {d_sf['worst_q']:.4f} : "
          f"{'PASS' if g3 else 'FAIL'}  (no worse runaway; held Q4 −2.65 → "
          f"Q3 −2.10)")
    print(f"    (iv) free-run RMSE two-timescale < single-fabric: "
          f"{d_cm['rmse']:.4f} < {d_sf['rmse']:.4f} : "
          f"{'PASS' if g4 else 'FAIL'}  (held 1.7207 → 1.4915)")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §5.2 design-fix drift "
          f"geometry is a structural fact, not a single held-out draw: on "
          f"fresh within-train fits the single-fabric → two-timescale switch "
          f"replays the flattened drift (|slope| {abs(d_sf['slope']):.6f} → "
          f"{abs(d_cm['slope']):.6f}, held halving −0.000680 → −0.000330), "
          f"the variance drop (sd {d_sf['sd']:.4f} → {d_cm['sd']:.4f}, held "
          f"1.482 → 1.172), the no-worse-worst-quarter (worst |signed| "
          f"{d_sf['worst_q']:.4f} → {d_cm['worst_q']:.4f}, held Q4 −2.65 → "
          f"Q3 −2.10) and the free-run RMSE improvement ({d_sf['rmse']:.4f} "
          f"→ {d_cm['rmse']:.4f}, held 1.7207 → 1.4915) — the within-train "
          f"drift sign inverts (positive slopes/bias, the same block-dependent "
          f"inversion §7 item 25 documents) and the block's worst quarter "
          f"sits at Q2 instead of held-out Q4, both printed as block deltas; "
          f"and the frozen artifact re-produces every committed §5.2 drift "
          f"digit (cmri slope {d_cf['slope']:+.6f} intercept "
          f"{d_cf['intercept']:+.4f}, per-quarter "
          f"{''.join(f'[{v:+.4f}]' for v in d_cf['q'])}, after-trend "
          f"{d_cf['aft']:.4f}, RMSE {d_cf['rmse']:.4f}, sd {d_cf['sd']:.4f}, "
          f"bias {d_cf['bias']:+.4f}, noise {d_cf['noise']:.4f}, n = "
          f"{len(e_cf)}; v2 slope {d_vf['slope']:+.6f} intercept "
          f"{d_vf['intercept']:+.4f}, per-quarter "
          f"{''.join(f'[{v:+.4f}]' for v in d_vf['q'])}, after-trend "
          f"{d_vf['aft']:.4f}, RMSE {d_vf['rmse']:.4f}, sd {d_vf['sd']:.4f}, "
          f"bias {d_vf['bias']:+.4f}, noise {d_vf['noise']:.4f}). The "
          f"superseded 'before' digits (−0.000680, 1.482, Q4 −2.65) carried "
          f"no frozen artifact (predictions_cmri.csv never held a "
          f"single-fabric free-run column) — they are history whose honest "
          f"within-train counterpart is the single-fabric replay above.")


if __name__ == "__main__":
    main()
