#!/usr/bin/env python3
"""select_cv_slowfab.py — within-train prequential CV for the SLOW second
fabric capacitor (driver 9, T_mass2 - T) on the canonical cmri layout.

Recorded experiment: held-out free-run regression of the frozen config is a
test-block seasonal level shift, and the chosen structural fix is a second,
SLOW fabric capacitor (hours-to-days tau) as an exogenous slow reference for
the open-loop level.  This script selects w2 (and re-confirms mask/t_set/alpha)
STRICTLY within train: forward-chaining folds over [0, split), same growing-
origin machinery, fold-SD tie window and free-run tie-break as select_cv.py.

Feature layout (canonical cmri, CmriTwin):
  0 env (T_out-T)      3 coupling (Tother-T)  6 lights/100  8 fast fabric (T_mass-T)
  1 wind*(T_out-T)     4 Tdewpoint            7 thermostat   9 slow fabric (T_mass2-T)
  2 clear-sky GHI      5 appliances/100

Fast fabric w is FROZEN at CMRI_W=0.944 (tau=3h, the frozen config).  The new
knob is w2 = EMA weight for T_mass2 = fabric_mass(t_out, w2), grid spanning
tau in {6,12,24,48} h.  Test data (>= split) is NEVER touched here.

Selection: lowest mean validation-resync RMSE among stable candidates (mean
free-run RMSE < FREE_CUT).  Candidates within the best stable candidate's
FOLD-SD are tied on resync; tie-break by lowest mean free-run RMSE — the
physics-motivated selector for a term whose claim is anchoring the long-horizon
open-loop trend.  The frozen config (mask 0,3,4,5,6,7,8, no index 9) is
evaluated by the same machine as the baseline the slow capacitor must beat.
"""

from __future__ import annotations

import math
import statistics
import sys

sys.path.insert(0, ".")

from experiment_cmri import (CMRI_T_SET, CMRI_W, DATA, TRAIN_FRAC, days_of,
                             load_cmri_data)
from cmri_twin import CmriTwin, fabric_mass
from run_experiment import metrics

N_FOLDS = 4
FREE_CUT = 2.0

# Slow-fabric EMA weights w2 = exp(-600/tau_s) for tau = 6, 12, 24, 48 h.
W2_GRID = (0.972, 0.986, 0.993, 0.997)
TAU2_H = {0.972: 6, 0.986: 12, 0.993: 24, 0.997: 48}

FROZEN_KEEP = (0, 3, 4, 5, 6, 7, 8)
SLOW_KEEP = (0, 3, 4, 5, 6, 7, 8, 9)


def eval_block(m, recs, days):
    """Daily-resync kitchen RMSE, canonical two-zone state (t1, th)."""
    res = []
    for day in days:
        st = (recs[day[0]][0], recs[day[0]][1])
        for a, b in zip(day[:-1], day[1:]):
            r = recs[a]
            st = m.step(st[0], st[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
            res.append(st[0])
    obs = [recs[b][0] for day in days for b in day[1:]]
    return metrics(res, obs)["rmse"]


def eval_free(m, recs):
    st = (recs[0][0], recs[0][1])
    fr = []
    for j in range(1, len(recs)):
        r = recs[j - 1]
        st = m.step(st[0], st[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
        fr.append(st[0])
    return metrics(fr, [r[0] for r in recs[1:]])["rmse"]


def run() -> None:
    data = load_cmri_data(DATA)
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    dates = data["date"]
    T, Th = data["T1"], data["Th"]
    t_out, dew, load, lux, wind, ghi = (
        data["T_out"], data["Tdewpoint"], data["Appliances"],
        data["lights"], data["Windspeed"], data["GHI"],
    )
    assert split < n, "CV must be strictly within train; no test contact."

    tm = fabric_mass(t_out, CMRI_W)          # fast fabric, frozen w
    recs10 = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm))  # w2 placeholder

    def fit_arrays(hi, tm2):
        s = slice(0, hi)
        return (T[s], Th[s], t_out[s], dew[s], load[s], lux[s], wind[s],
                ghi[s], tm[s], tm2[s])

    bounds = [0] + [round(split * j / N_FOLDS) for j in range(1, N_FOLDS + 1)]
    NF = N_FOLDS - 1

    # candidates: (mask, w2); frozen mask ignores w2, slow mask sweeps W2_GRID.
    candidates = [(FROZEN_KEEP, None)] + [
        (SLOW_KEEP, w2) for w2 in W2_GRID]

    cands = []
    for keep, w2 in candidates:
        tm2 = tm if w2 is None else fabric_mass(t_out, w2)
        recs = [(*r[:-1], tm2[i]) for i, r in enumerate(recs10)]
        for tset in (20.0, 21.0):
            for alpha in (0.0, 100.0, 300.0, 1000.0, 5000.0):
                res_list, fr_list = [], []
                for f in range(NF):
                    fit_hi = bounds[f + 1]
                    lo, hi = bounds[f + 1], bounds[f + 2]
                    m = CmriTwin(alpha=alpha, t_set=tset, keep=keep)
                    m.fit(*fit_arrays(fit_hi, tm2))
                    res_list.append(eval_block(m, recs[lo:hi], days_of(dates, lo, hi)))
                    fr_list.append(eval_free(m, recs[lo:hi]))
                mr = sum(res_list) / NF
                mf = sum(fr_list) / NF
                sd = statistics.stdev(res_list) if NF > 1 else 0.0
                stable = mf < FREE_CUT
                cands.append((mr, mf, stable, sd, keep, tset, alpha, w2))
                wlab = "frozen" if w2 is None else f"tau2={TAU2_H[w2]}h"
                mlab = "slowfab" if keep == SLOW_KEEP else ("frozen " if keep == FROZEN_KEEP else "?")
                print(f"  mask={mlab} {wlab} tset={tset:4.1f} alpha={alpha:5.0f} "
                      f"mean_res={mr:.4f} mean_fr={mf:.4f} res_sd={sd:.4f} "
                      f"stable={'Y' if stable else 'N'}", flush=True)

    cands.sort(key=lambda c: (not c[2], c[0], c[1]))
    print("\n=== top 12 by mean validation resync RMSE (stable first) ===")
    for mr, mf, stable, sd, keep, tset, alpha, w2 in cands[:12]:
        wlab = "frozen" if w2 is None else f"tau2={TAU2_H[w2]}h"
        mlab = "slowfab" if keep == SLOW_KEEP else ("frozen " if keep == FROZEN_KEEP else "?")
        print(f"  mask={mlab} {wlab} tset={tset:4.1f} alpha={alpha:5.0f} "
              f"mean_res={mr:.4f} mean_fr={mf:.4f} res_sd={sd:.4f} "
              f"stable={'Y' if stable else 'N'}")

    stable_sorted = [c for c in cands if c[2]]
    best_res = stable_sorted[0][0] if stable_sorted else math.inf
    best_sd = stable_sorted[0][3] if stable_sorted else 0.0
    window = best_sd if stable_sorted else 0.0  # fold-SD, honest uncertainty scale
    tied = [c for c in stable_sorted if c[0] <= best_res + window]
    try:
        wr = min(tied, key=lambda c: c[1])
    except ValueError:
        wr = cands[0]
    print(f"\nbest stable mean_res={best_res:.4f} (res_sd={best_sd:.4f}, "
          f"fold-sd window={window:.4f}); {len(tied)} stable candidates within "
          f"fold-sd window (tie-broken by free-run)")

    # The frozen config evaluated by the same machine: what the slow capacitor
    # must beat.
    def find(keep, tset, alpha, w2):
        for c in cands:
            if c[4] == keep and c[5] == tset and c[6] == alpha and c[7] == w2:
                return c
        return None

    fc = find(FROZEN_KEEP, CMRI_T_SET, 300.0, None)
    print(f"\nfrozen config CV numbers: keep={FROZEN_KEEP} tset={CMRI_T_SET} "
          f"alpha=300 w2=None => mean_res={fc[0]:.4f} mean_fr={fc[1]:.4f} "
          f"stable={'Y' if fc[2] else 'N'} (slow capacitor must beat this)")

    try:
        strict = next(c for c in cands if c[2])
    except StopIteration:
        strict = cands[0]
    print("STRICT-SELECTED (resync-primary)",
          {"keep": strict[4], "t_set": strict[5], "alpha": strict[6],
           "w2": strict[7], "mean_res": round(strict[0], 4),
           "mean_fr": round(strict[1], 4)})
    print("SELECTED (fold-sd-tie, free-run tie-break)",
          {"keep": wr[4], "t_set": wr[5], "alpha": wr[6],
           "w2": wr[7], "mean_res": round(wr[0], 4), "mean_fr": round(wr[1], 4)})
    # Summary by w2 for the slow-fab mask: does longer tau2 monotonically
    # help free-run anchoring on train?
    print("\nslow-fab mask mean (over tset/alpha) per tau2:")
    for w2 in W2_GRID:
        rows = [c for c in cands if c[4] == SLOW_KEEP and c[7] == w2]
        mr = sum(c[0] for c in rows) / len(rows)
        mf = sum(c[1] for c in rows) / len(rows)
        print(f"  tau2={TAU2_H[w2]:>2}h  mean_res={mr:.4f}  mean_fr={mf:.4f}")


if __name__ == "__main__":
    run()
