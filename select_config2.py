#!/usr/bin/env python3
"""select_config2.py — second round of honest config selection for the cmri twin.

Adds a physically-motivated SECOND-ORDER driver (thermal-inertia / momentum:
the recent rate of change T[t] - T[t-1]) to both zone equations, alongside the
existing 8 drivers.  Feature index layout:

  0 env (T_out-T)        5 appliances/100
  1 wind*(T_out-T)       6 lights/100
  2 clear-sky GHI        7 thermostat max(Tset-T,0)
  3 coupling (Tother-T)  8 momentum T[t]-T[t-1]
  4 Tdewpoint

Selection stays on the within-train validation slice (last 20% of the 70%
training block): minimise validation resync RMSE subject to free-run RMSE
below a stability cutoff (kitchen), tie-break on free-run.  Test is never
touched here.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tiny_twin_v2 import LinearModel  # noqa: E402
from run_experiment import TRAIN_FRAC, metrics  # noqa: E402
from experiment_cmri import DATA, days_of, load_cmri_data  # noqa: E402

VALID_FRAC = 0.2
FREE_CUT = 2.0  # stability: reject configs with val free-run RMSE >= this


class M2Twin:
    """Two-zone twin over extended 9-driver feature layout (index 8 momentum)."""

    def __init__(self, keep, alpha, t_set):
        self.keep = keep
        self.t_set = t_set
        self.k = LinearModel(alpha=alpha)
        self.h = LinearModel(alpha=alpha)

    @staticmethod
    def _feats(t1, th, t_out, dew, load, lux, wind, ghi, t_set, m1, mh):
        k = [t_out - t1, wind * (t_out - t1), ghi, th - t1, dew,
             load / 100.0, lux / 100.0, max(t_set - t1, 0.0), m1]
        h = [t_out - th, wind * (t_out - th), ghi, t1 - th, dew,
             load / 100.0, lux / 100.0, max(t_set - th, 0.0), mh]
        return k, h

    def fit(self, T1, Th, t_out, dew, load, lux, wind, ghi):
        Xk, yk, Xh, yh = [], [], [], []
        for i in range(1, len(T1) - 1):
            fk, fh = self._feats(T1[i], Th[i], t_out[i], dew[i], load[i], lux[i],
                                 wind[i], ghi[i], self.t_set,
                                 T1[i] - T1[i - 1], Th[i] - Th[i - 1])
            Xk.append([fk[j] for j in self.keep])
            yk.append(T1[i + 1] - T1[i])
            Xh.append([fh[j] for j in self.keep])
            yh.append(Th[i + 1] - Th[i])
        self.k.fit(Xk, yk)
        self.h.fit(Xh, yh)

    def step(self, state, t_out, dew, load, lux, wind, ghi):
        t1, th, m1, mh = state
        fk, fh = self._feats(t1, th, t_out, dew, load, lux, wind, ghi,
                             self.t_set, m1, mh)
        n1 = t1 + self.k.predict([fk[j] for j in self.keep])
        nh = th + self.h.predict([fh[j] for j in self.keep])
        return (n1, nh, n1 - t1, nh - th)


def eval_on(m, recs, days):
    """Daily-resync kitchen RMSE + open-loop free-run RMSE (extended state)."""
    res = []
    for day in days:
        # anchor: observed T, momentum = observed motion from previous sample
        i0 = day[0]
        a0, a1 = recs[i0][0], recs[i0 - 1][0]
        ah0, ah1 = recs[i0][1], recs[i0 - 1][1]
        st = (a0, ah0, a0 - a1, ah0 - ah1)
        for a, b in zip(day[:-1], day[1:]):
            _, _, to, de, lo, lu, wi, gh = recs[a]
            st = m.step(st, to, de, lo, lu, wi, gh)
            res.append(st[0])
    obs_res = [recs[b][0] for day in days for b in day[1:]]
    rmse_res = metrics(res, obs_res)["rmse"]

    st = (recs[0][0], recs[0][1], 0.0, 0.0)
    fr = []
    for j in range(1, len(recs)):
        _, _, to, de, lo, lu, wi, gh = recs[j - 1]
        st = m.step(st, to, de, lo, lu, wi, gh)
        fr.append(st[0])
    rmse_fr = metrics(fr, [r[0] for r in recs[1:]])["rmse"]
    return rmse_res, rmse_fr


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
    val_start = int(round((1.0 - VALID_FRAC) * split))

    fs = slice(0, val_start)
    Tr = (T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs], ghi[fs])
    val_recs = list(zip(T, Th, t_out, dew, load, lux, wind, ghi))[val_start:split]
    val_days = days_of(dates, val_start, split)

    masks = {
        "full": (0, 1, 2, 3, 4, 5, 6, 7),
        "nws": (0, 3, 4, 5, 6, 7),                 # no wind, no solar
        "nws+mom": (0, 3, 4, 5, 6, 7, 8),          # + momentum
        "coupl+mom": (0, 3, 4, 5, 6, 8),           # no thermostat, + momentum
        "env+mom": (0, 4, 5, 6, 8),                # v2 drivers + momentum
        "full+mom": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    }
    pairs = [("full",), ("nws",), ("nws+mom",), ("coupl+mom",), ("env+mom",), ("full+mom",)]

    cands = []
    for (name,) in pairs:
        keep = masks[name]
        for tset in (20.0, 21.0):
            for alpha in (0.0, 100.0, 300.0, 1000.0):
                m = M2Twin(keep, alpha, tset)
                m.fit(*Tr)
                r_res, r_fr = eval_on(m, val_recs, val_days)
                stable = r_fr < FREE_CUT
                cands.append((r_res, r_fr, stable, name, tset, alpha))
                print(f"  mask={name:>10} tset={tset:4.1f} alpha={alpha:5.0f} "
                      f"val_res={r_res:.4f} val_fr={r_fr:.4f} "
                      f"stable={'Y' if stable else 'N'}", flush=True)

    cands.sort(key=lambda c: (not c[2], c[0], c[1]))
    print("\n=== top 10 validation-resync candidates (stable first, v2 ref ~0.68 train-resync) ===")
    for r_res, r_fr, stable, name, tset, alpha in cands[:10]:
        print(f"  mask={name:>10} tset={tset:4.1f} alpha={alpha:5.0f} "
              f"val_res={r_res:.4f} val_fr={r_fr:.4f} stable={'Y' if stable else 'N'}")
    wr = next(c for c in cands if c[2])
    print("SELECTED",
          {"mask": wr[3], "t_set": wr[4], "alpha": wr[5],
           "val_res": round(wr[0], 4), "val_fr": round(wr[1], 4)})


if __name__ == "__main__":
    run()
