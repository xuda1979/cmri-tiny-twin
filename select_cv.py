#!/usr/bin/env python3
"""select_cv.py — honest within-train cross-validation for cmri twin knobs.

Forward-chaining (prequential) cross-validation over the 70% training block:
fit on [0, fold_end), evaluate daily-resync kitchen RMSE on
[fold_end, next_fold_end).  The held-out 30% test is NEVER touched here.

Feature layout for the candidate masks (index 8 = optional momentum):
  0 env (T_out-T)      3 coupling (Tother-T)   6 lights/100
  1 wind*(T_out-T)     4 Tdewpoint             7 thermostat max(Tset-T,0)
  2 clear-sky GHI      5 appliances/100        8 momentum T[t]-T[t-1]

Selection: lowest mean validation-resync RMSE among stable candidates
(mean free-run RMSE < 2.0), ties broken by free-run.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tiny_twin_v2 import LinearModel  # noqa: E402
from run_experiment import TRAIN_FRAC, metrics  # noqa: E402
from experiment_cmri import DATA, days_of, load_cmri_data  # noqa: E402

N_FOLDS = 4
FREE_CUT = 2.0


class M2Twin:
    """Two-zone twin over the 9-driver layout (index 8 = momentum driver)."""

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


def eval_block(m, recs, days):
    """Daily-resync kitchen RMSE (extended state anchored from observations)."""
    res = []
    for day in days:
        i0 = day[0]
        a0, a1 = recs[i0][0], recs[i0 - 1][0]
        ah0, ah1 = recs[i0][1], recs[i0 - 1][1]
        st = (a0, ah0, a0 - a1, ah0 - ah1)
        for a, b in zip(day[:-1], day[1:]):
            _, _, to, de, lo, lu, wi, gh = recs[a]
            st = m.step(st, to, de, lo, lu, wi, gh)
            res.append(st[0])
    obs = [recs[b][0] for day in days for b in day[1:]]
    return metrics(res, obs)["rmse"]


def eval_free(m, recs):
    st = (recs[0][0], recs[0][1], 0.0, 0.0)
    fr = []
    for j in range(1, len(recs)):
        _, _, to, de, lo, lu, wi, gh = recs[j - 1]
        st = m.step(st, to, de, lo, lu, wi, gh)
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

    recs = list(zip(T, Th, t_out, dew, load, lux, wind, ghi))
    # growing-origin prequential folds: fold f fits on [0, bounds[f+1])
    # and evaluates on [bounds[f+1], bounds[f+2])  (f in 0..NF-1, NF=N_FOLDS-1)
    bounds = [0] + [round(split * j / N_FOLDS) for j in range(1, N_FOLDS + 1)]
    NF = N_FOLDS - 1

    masks = {
        "full": (0, 1, 2, 3, 4, 5, 6, 7),
        "nws": (0, 3, 4, 5, 6, 7),
        "nws+mom": (0, 3, 4, 5, 6, 7, 8),
        "coupl+mom": (0, 3, 4, 5, 6, 8),
        "env+mom": (0, 4, 5, 6, 8),
        "full+mom": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    }

    def fit_arrays(hi):
        s = slice(0, hi)
        return (T[s], Th[s], t_out[s], dew[s], load[s], lux[s], wind[s], ghi[s])

    cands = []
    for name, keep in masks.items():
        for tset in (20.0, 21.0):
            for alpha in (0.0, 100.0, 300.0, 1000.0, 5000.0):
                res_list, fr_list = [], []
                for f in range(NF):
                    fit_hi = bounds[f + 1]
                    lo, hi = bounds[f + 1], bounds[f + 2]
                    m = M2Twin(keep, alpha, tset)
                    m.fit(*fit_arrays(fit_hi))
                    days = days_of(dates, lo, hi)
                    res_list.append(eval_block(m, recs, days))
                    fr_list.append(eval_free(m, recs[lo:hi]))
                mr = sum(res_list) / NF
                mf = sum(fr_list) / NF
                stable = mf < FREE_CUT
                cands.append((mr, mf, stable, name, tset, alpha))
                print(f"  mask={name:>10} tset={tset:4.1f} alpha={alpha:5.0f} "
                      f"mean_res={mr:.4f} mean_fr={mf:.4f} "
                      f"stable={'Y' if stable else 'N'}", flush=True)

    cands.sort(key=lambda c: (not c[2], c[0], c[1]))
    print("\n=== top 10 by mean validation resync RMSE (stable first) ===")
    for mr, mf, stable, name, tset, alpha in cands[:10]:
        print(f"  mask={name:>10} tset={tset:4.1f} alpha={alpha:5.0f} "
              f"mean_res={mr:.4f} mean_fr={mf:.4f} stable={'Y' if stable else 'N'}")
    try:
        wr = next(c for c in cands if c[2])
    except StopIteration:
        wr = cands[0]
    print("SELECTED",
          {"mask": wr[3], "t_set": wr[4], "alpha": wr[5],
           "mean_res": round(wr[0], 4), "mean_fr": round(wr[1], 4)})


if __name__ == "__main__":
    run()
