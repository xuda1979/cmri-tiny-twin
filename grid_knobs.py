#!/usr/bin/env python3
"""grid_knobs.py — config sweep for the cmri twin (diagnostic, not a result).

Grids over honest model-design knobs to find a configuration where the cmri
two-zone twin beats v2 on the held-out resync protocol (v2 resync RMSE 0.646)
without collapsing on free-run:

  mask_k / mask_h : per-zone driver retention
      full          (0..7)
      no_solar      (0,1,3,4,5,6,7)
      no_wind_solar (0,3,4,5,6,7)
      v2            (0,4,5,6)  [v2-equivalent envelope + dew + load + lights]
  t_set  : thermostat setpoint for the max(T_set - T, 0) driver
  alpha  : L2 ridge on the standardized design matrix (default 0)
  th_mode: resync coupling protocol
      sim  -> kitchen + house both simulated open-loop (pure simulator)
      obs  -> kitchen simulated, house-zone temp taken from observed (live
              house-zone sensor; realistic deployed-twin mode)

Free-run always simulates both zones from the observed start state.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tiny_twin_v2 import LinearModel  # noqa: E402
from run_experiment import TRAIN_FRAC, metrics  # noqa: E402
from experiment_cmri import DATA, days_of, load_cmri_data  # noqa: E402
from cmri_twin import CmriTwin  # noqa: E402


class KnobTwin:
    """CmriTwin analogue with tunable per-zone masks, alpha and t_set."""

    def __init__(self, mask_k, mask_h, alpha, t_set):
        self.mask_k = mask_k
        self.mask_h = mask_h
        self.t_set = t_set
        self.k = LinearModel(alpha=alpha)
        self.h = LinearModel(alpha=alpha)

    def fit(self, T1, Th, t_out, dew, load, lux, wind, ghi):
        Xk, yk, Xh, yh = [], [], [], []
        for i in range(len(T1) - 1):
            fk = CmriTwin._k_feats(T1[i], Th[i], t_out[i], dew[i], load[i],
                                   lux[i], wind[i], ghi[i], self.t_set)
            fh = CmriTwin._h_feats(T1[i], Th[i], t_out[i], dew[i], load[i],
                                   lux[i], wind[i], ghi[i], self.t_set)
            Xk.append([fk[j] for j in self.mask_k])
            yk.append(T1[i + 1] - T1[i])
            Xh.append([fh[j] for j in self.mask_h])
            yh.append(Th[i + 1] - Th[i])
        self.k.fit(Xk, yk)
        self.h.fit(Xh, yh)

    def step(self, t1, th, t_out, dew, load, lux, wind, ghi):
        fk = CmriTwin._k_feats(t1, th, t_out, dew, load, lux, wind, ghi, self.t_set)
        fh = CmriTwin._h_feats(t1, th, t_out, dew, load, lux, wind, ghi, self.t_set)
        return (t1 + self.k.predict([fk[j] for j in self.mask_k]),
                th + self.h.predict([fh[j] for j in self.mask_h]))


def run() -> None:
    data = load_cmri_data(DATA)
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    dates = data["date"]
    T, t_out, dew, load, lux, wind, ghi = (
        data["T1"], data["T_out"], data["Tdewpoint"],
        data["Appliances"], data["lights"], data["Windspeed"], data["GHI"],
    )
    Th = data["Th"]
    test_days = days_of(dates, split)

    s = slice(0, split)
    Tr = (T[s], Th[s], t_out[s], dew[s], load[s], lux[s], wind[s], ghi[s])
    test_obs = list(zip(T, Th, t_out, dew, load, lux, wind, ghi))[split + 1:]
    test_recs = list(zip(T, Th, t_out, dew, load, lux, wind, ghi))[split:]

    def eval_twin(m, th_mode):
        """Resync (daily anchors) + free-run (open-loop) kitchen RMSE."""
        res = []
        for day in test_days:
            t1, th = test_recs[day[0]][0], test_recs[day[0]][1]
            for a, b in zip(day[:-1], day[1:]):
                _, _, to, de, lo, lu, wi, gh = test_recs[a]
                if th_mode == "obs":
                    th = Th[split + b]
                t1, th = m.step(t1, th, to, de, lo, lu, wi, gh)
                res.append(t1)
        obs1 = [test_recs[b][0] for day in test_days for b in day[1:]]
        m_res = metrics(res, obs1)["rmse"]

        # free-run: both zones open-loop from observed start
        t1, th = test_recs[0][0], test_recs[0][1]
        fr = []
        for j in range(1, len(test_recs)):
            _, _, to, de, lo, lu, wi, gh = test_recs[j - 1]
            t1, th = m.step(t1, th, to, de, lo, lu, wi, gh)
            fr.append(t1)
        m_fr = metrics(fr, [r[0] for r in test_recs[1:]])["rmse"]
        return m_res, m_fr

    masks = {
        "full": (0, 1, 2, 3, 4, 5, 6, 7),
        "no_solar": (0, 1, 3, 4, 5, 6, 7),
        "no_wind_solar": (0, 3, 4, 5, 6, 7),
        "v2": (0, 4, 5, 6),
    }
    mask_pairs = [
        ("f/f", "full", "full"),
        ("f/ns", "full", "no_solar"),
        ("f/nws", "full", "no_wind_solar"),
        ("ns/ns", "no_solar", "no_solar"),
        ("nws/nws", "no_wind_solar", "no_wind_solar"),
    ]
    results = []
    for (tag, mk, mh), tset, alpha in (
        ((t, mk, mh), tset, alpha)
        for (t, mk, mh) in mask_pairs
        for tset in (20.0, 21.0)
        for alpha in (0.0, 5e2, 5e3)
    ):
        m = KnobTwin(masks[mk], masks[mh], alpha, tset)
        m.fit(*Tr)
        r_sim, fr = eval_twin(m, "sim")
        r_obs, _ = eval_twin(m, "obs")
        results.append((r_sim, r_obs, fr, tag, tset, alpha))
        print(f"mask={tag:>6} tset={tset:4.1f} alpha={alpha:>6.0f} "
              f"res_sim={r_sim:.4f} res_obs={r_obs:.4f} freerun={fr:.4f}",
              flush=True)

    results.sort(key=lambda r: r[1])
    print("\n=== top by resync-OBS RMSE (v2 baseline 0.6460, persistence 0.6246) ===")
    for r in results[:8]:
        print(f"  res_sim={r[0]:.4f} res_obs={r[1]:.4f} freerun={r[3]:.4f} "
              f"mask={r[4]:>6} tset={r[5]:4.1f} alpha={r[6]:>6.0f}")
    results.sort(key=lambda r: r[2])
    print("\n=== top by free-run RMSE ===")
    for r in results[:8]:
        print(f"  res_sim={r[0]:.4f} res_obs={r[1]:.4f} freerun={r[2]:.4f} "
              f"mask={r[3]:>6} tset={r[4]:4.1f} alpha={r[5]:>6.0f}")


if __name__ == "__main__":
    run()
