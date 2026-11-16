#!/usr/bin/env python3
"""select_config.py — honest hyperparameter selection for the cmri twin.

Tune knobs (driver masks, ridge alpha, thermostat setpoint) on a validation
slice taken from the END of the 70% training block (train=first 56%,
val=last 14%), using the same daily-resync protocol as the final evaluation.
The winner is then frozen and only reported on the held-out 30% test in
experiment_cmri.py — no test peeking here.

Selection rule: minimise validation resync RMSE (kitchen), ties broken by
validation free-run RMSE.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tiny_twin_v2 import LinearModel  # noqa: E402
from run_experiment import TRAIN_FRAC, metrics  # noqa: E402
from experiment_cmri import DATA, days_of, load_cmri_data  # noqa: E402
from cmri_twin import CmriTwin  # noqa: E402

VALID_FRAC = 0.2  # fraction of the TRAIN block held out as validation


class KnobTwin:
    def __init__(self, mask_k, mask_h, alpha, t_set):
        self.mask_k, self.mask_h = mask_k, mask_h
        self.t_set = t_set
        self.k = LinearModel(alpha=alpha)
        self.h = LinearModel(alpha=alpha)

    def fit(self, T1, Th, t_out, dew, load, lux, wind, ghi):
        Xk, yk, Xh, yh = [], [], [], []
        for i in range(len(T1) - 1):
            fk = CmriTwin._k_feats(T1[i], Th[i], t_out[i], dew[i], load[i], lux[i], wind[i], ghi[i], t_out[i], self.t_set)
            fh = CmriTwin._h_feats(T1[i], Th[i], t_out[i], dew[i], load[i], lux[i], wind[i], ghi[i], t_out[i], self.t_set)
            Xk.append([fk[j] for j in self.mask_k])
            yk.append(T1[i + 1] - T1[i])
            Xh.append([fh[j] for j in self.mask_h])
            yh.append(Th[i + 1] - Th[i])
        self.k.fit(Xk, yk)
        self.h.fit(Xh, yh)

    def step(self, t1, th, t_out, dew, load, lux, wind, ghi):
        fk = CmriTwin._k_feats(t1, th, t_out, dew, load, lux, wind, ghi, t_out, self.t_set)
        fh = CmriTwin._h_feats(t1, th, t_out, dew, load, lux, wind, ghi, t_out, self.t_set)
        return (t1 + self.k.predict([fk[j] for j in self.mask_k]),
                th + self.h.predict([fh[j] for j in self.mask_h]))


def eval_on(m, recs, days):
    """Daily-resync kitchen RMSE (sim mode) + open-loop free-run RMSE."""
    res = []
    for day in days:
        t1, th = recs[day[0]][0], recs[day[0]][1]
        for a, b in zip(day[:-1], day[1:]):
            _, _, to, de, lo, lu, wi, gh = recs[a]
            t1, th = m.step(t1, th, to, de, lo, lu, wi, gh)
            res.append(t1)
    obs_res = [recs[b][0] for day in days for b in day[1:]]
    rmse_res = metrics(res, obs_res)["rmse"]

    t1, th = recs[0][0], recs[0][1]
    fr = []
    for j in range(1, len(recs)):
        _, _, to, de, lo, lu, wi, gh = recs[j - 1]
        t1, th = m.step(t1, th, to, de, lo, lu, wi, gh)
        fr.append(t1)
    rmse_fr = metrics(fr, [r[0] for r in recs[1:]])["rmse"]
    return rmse_res, rmse_fr


def run() -> None:
    data = load_cmri_data(DATA)
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    dates = data["date"]
    T = data["T1"]
    Th = data["Th"]
    t_out, dew, load, lux, wind, ghi = (
        data["T_out"], data["Tdewpoint"], data["Appliances"],
        data["lights"], data["Windspeed"], data["GHI"],
    )

    val_start = int(round((1.0 - VALID_FRAC) * split))
    print(f"train fit block [0,{val_start}), validation [{val_start},{split})")
    fs = slice(0, val_start)
    Tr = (T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs], ghi[fs])
    val_recs = list(zip(T, Th, t_out, dew, load, lux, wind, ghi))[val_start:split]
    val_days = days_of(dates, val_start, split)

    masks = {
        "full": (0, 1, 2, 3, 4, 5, 6, 7),
        "no_solar": (0, 1, 3, 4, 5, 6, 7),
        "no_wind_solar": (0, 3, 4, 5, 6, 7),
        "w_coupl_nosolar": (0, 1, 3, 4, 5, 6),
        "v2": (0, 4, 5, 6),
    }
    pairs = [
        ("full/full", "full", "full"),
        ("full/nosolar", "full", "no_solar"),
        ("full/nws", "full", "no_wind_solar"),
        ("nosolar/nosolar", "no_solar", "no_solar"),
        ("nws/nws", "no_wind_solar", "no_wind_solar"),
        ("coupl/coupl", "w_coupl_nosolar", "w_coupl_nosolar"),
    ]

    cands = []
    for (tag, mk, mh), tset, alpha in (
        ((t, mk, mh), tset, alpha)
        for (t, mk, mh) in pairs
        for tset in (20.0, 21.0)
        for alpha in (0.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
    ):
        m = KnobTwin(masks[mk], masks[mh], alpha, tset)
        m.fit(*Tr)
        r_res, r_fr = eval_on(m, val_recs, val_days)
        cands.append((r_res, r_fr, tag, tset, alpha))
        print(f"  mask={tag:>14} tset={tset:4.1f} alpha={alpha:6.0f} "
              f"val_res={r_res:.4f} val_fr={r_fr:.4f}", flush=True)

    cands.sort(key=lambda c: (c[0], c[1]))
    print("\n=== top 10 by validation resync RMSE ===")
    for r_res, r_fr, tag, tset, alpha in cands[:10]:
        print(f"  mask={tag:>14} tset={tset:4.1f} alpha={alpha:6.0f} "
              f"val_res={r_res:.4f} val_fr={r_fr:.4f}")
    print("RESULT", {k: v for k, v in zip(("mask", "t_set", "alpha", "val_res", "val_fr"),
                                          (cands[0][2], cands[0][3], cands[0][4], cands[0][0], cands[0][1]))})


if __name__ == "__main__":
    run()
