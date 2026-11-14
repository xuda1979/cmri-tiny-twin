#!/usr/bin/env python3
"""freerun_diag_train.py — WITHIN-TRAIN free-run drift diagnostic.

Is the frozen cmri config's accelerating free-run drift (held-out test:
per-quarter mean signed error [+0.61, -0.39, -1.07, -2.65] C/step) a property
of the MODEL (reproducible on weather it was fit to → a config change can
fix it) or of the TEST-BLOCK WEATHER REALIZATION (a lucky/unlucky block → no
config change reliably helps)?

Design: fits the frozen config (keep nws+fab, t_set 21, alpha 300, w 0.944)
on early-training windows and free-runs over LATER training weather of the
same ~41-day length as the held-out free-run, then applies the identical
quarter-bias / accumulation statistics as diagnose_freerun.py. Also a shorter
horizon to test length-dependence, and v2 for comparison.

NO held-out test contact: everything stays in train[0, 13814).
"""

import math
import time

from experiment_cmri import (DATA, TRAIN_FRAC, run_protocol_cmri,
                             days_of, fabric_mass, load_cmri_data)
from cmri_twin import CmriTwin
from tiny_twin_v2 import CalibratedTwin
from run_experiment import metrics

SPLIT = 13814  # fixed train/test split; train is [0, SPLIT)


def _sd(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def drift_report(name, obs, pred):
    """Same statistics as diagnose_freerun.py::report on a prediction series."""
    pairs = [(c, o) for c, o in zip(pred, obs) if c is not None]
    pred, obs = [c for c, _ in pairs], [o for _, o in pairs]
    n = len(obs)
    err = [p - o for p, o in zip(pred, obs)]
    m = metrics(pred, obs)
    q = n // 4
    qb = [sum(err[j*q:(j+1)*q]) / len(err[j*q:(j+1)*q]) for j in range(4)]
    acc = []
    s = 0.0
    for e in err:
        s += e
        acc.append(s)
    d = [err[i] - err[i - 1] for i in range(1, n)]
    print(f"  [{name}] n={n}  RMSE={m['rmse']:.4f} MAE={m['mae']:.4f}")
    print(f"    mean per-step bias {sum(err)/n:+.4f} C/step (sd {_sd(err):.4f})")
    print(f"    per-quarter mean signed error: "
          + "  ".join(f"[{b:+.4f}]" for b in qb))
    print(f"    total accumulated error at end: {s:+.2f} C")
    print(f"    sd of per-step error increment (bias noise) = {_sd(d):.4f} C")
    return m["rmse"]


def main():
    data = load_cmri_data(DATA)
    T, t_out, dew, load, lux, wind, sol_ghi = (
        data["T1"], data["T_out"], data["Tdewpoint"],
        data["Appliances"], data["lights"], data["Windspeed"], data["GHI"],
    )
    T_h = data["Th"]
    n = len(T)
    tr = min(SPLIT, n)
    tm_all = fabric_mass(t_out, 0.944)

    def recs8(lo, hi):
        s = slice(lo, hi)
        return list(zip(T[s], T_h[s], t_out[s], dew[s], load[s], lux[s],
                        wind[s], sol_ghi[s], tm_all[s]))

    def fit_slice(lo, hi):
        s = slice(lo, hi)
        return (T[s], T_h[s], t_out[s], dew[s], load[s], lux[s], wind[s],
                sol_ghi[s], tm_all[s])

    # splits: (fit_lo, fit_hi, free_lo, free_hi), all within train.
    #  ~41-day free-run (~5920 steps) mirrors the held-out protocol; we also
    #  try shorter free-runs along the way to expose length dependence.
    splits = [
        ("train A: fit[0,7894) free[7894,13814) 41d", 0, 7894, 7894, tr),
        ("train B: fit[0,9874) free[9874,13814) 27d", 0, 9874, 9874, tr),
        ("train C: fit[0,11854) free[11854,13814) 14d", 0, 11854, 11854, tr),
        ("train D: fit[0,5920) free[5920,11840) 41d", 0, 5920, 5920, 11840),
    ]

    for label, flo, fhi, glo, ghi in splits:
        print(f"\n=== {label} ===")
        cmri = CmriTwin(alpha=300.0, t_set=21.0, keep=(0, 3, 4, 5, 6, 7, 8))
        cmri.fit(*fit_slice(flo, fhi))
        obs = T[glo + 1:ghi]
        p = run_protocol_cmri(cmri, (T[glo], T_h[glo]), recs8(glo, ghi), None)[1:]
        print(" cmri (frozen config):")
        drift_report("cmri", obs, p)
        v2 = CalibratedTwin()
        v2.fit(T[flo:fhi], t_out[flo:fhi], dew[flo:fhi],
               load[flo:fhi], lux[flo:fhi])
        p2 = [None] * (ghi - glo)
        t1 = T[glo]
        r5 = list(zip(T, t_out, dew, load, lux))
        for j in range(glo + 1, ghi):
            _, to, de, lo_, lu = r5[j - 1]
            t1 = v2.step(t1, to, de, lo_, lu)
            p2[j - glo] = t1
        print(" v2 (single-zone):")
        drift_report("v2", obs, p2)
        print()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s")
