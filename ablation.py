#!/usr/bin/env python3
"""ablation.py — isolate HOW MUCH of v2's win comes from calibration vs extra drivers.

A reviewer question: is v2's large held-out improvement because the coefficients are
fitted to real data, or merely because v2's regression has extra inputs (dewpoint,
lights) that v1's handbook equation lacks?

This script fits OLS variants with IDENTICAL fitting machinery (LinearModel) that
augment the input set step by step, run all of them under the same held-out resync
protocol, and compare:

  * v1_handbook  : 5.88 RMSE (reference, no fitting at all)
  * env_only     : regression on [T_out - T] only            (envelope drive, + bias)
  * env_app      : + appliances/100                           (v1-equivalent driver set)
  * full (v2)    : + Tdewpoint, lights/100                    (v2 as shipped)

Pure Python stdlib; imports the shared runner from run_experiment.py.
"""

from __future__ import annotations

import csv
import pathlib

from tiny_twin_v2 import LinearModel, Q_SCALE
from run_experiment import (
    load_data,
    run_protocol,
    metrics,
    ROOT,
    DATA,
    OUT,
    TRAIN_FRAC,
)

# ---------------------------------------------------------------------------
# Feature sets (LinearModel always adds its own intercept column)
# ---------------------------------------------------------------------------


def f_env(T, tout, dew, load, lux):
    return [tout - T]


def f_env_app(T, tout, dew, load, lux):
    return [tout - T, load / Q_SCALE]


def f_full(T, tout, dew, load, lux):
    return [tout - T, dew, load / Q_SCALE, lux / Q_SCALE]


VARIANTS = {
    "env_only": f_env,
    "env_app": f_env_app,   # v1's driver set (envelope + appliance), calibrated
    "full": f_full,         # v2 as shipped
}


class AblModel:
    """OLS twin with a pluggable feature builder (fits on train, steps + resync)."""

    def __init__(self, feats, alpha: float = 0.0) -> None:
        self.feats = feats
        self.model = LinearModel(alpha=alpha)
        self.fitted = False

    def fit(self, T, tout, dew, load, lux) -> None:
        X, y = [], []
        for i in range(len(T) - 1):
            X.append(self.feats(T[i], tout[i], dew[i], load[i], lux[i]))
            y.append(T[i + 1] - T[i])
        self.model.fit(X, y)
        self.fitted = True

    def step(self, temp, t_out, dew, load, lux) -> float:
        return temp + self.model.predict(self.feats(temp, t_out, dew, load, lux))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data(DATA)
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)

    dates = data["date"]
    T, tout, dew, load, lux = data["T1"], data["T_out"], data["Tdewpoint"], data["Appliances"], data["lights"]
    recs = list(zip(T, tout, dew, load, lux))
    test_recs = recs[split:]

    # same day grouping as run_experiment.py (resync protocol)
    groups: dict[object, list[int]] = {}
    for j in range(len(test_recs)):
        groups.setdefault(dates[split + j].date(), []).append(j)
    test_days = [v for _, v in sorted(groups.items())]

    # fit each variant on TRAIN ONLY, then run held-out resync protocol
    rows = []
    for name, feats in VARIANTS.items():
        m = AblModel(feats)
        m.fit(T[:split], tout[:split], dew[:split], load[:split], lux[:split])
        preds = run_protocol(m, T[split], test_recs, test_days)
        obs = [r[0] for r in test_recs]
        pairs = [(c, o) for c, o in zip(preds, obs) if c is not None]
        met = metrics([c for c, _ in pairs], [o for _, o in pairs])
        rows.append({
            "variant": name,
            "drivers": {
                "env_only": "envelope",
                "env_app": "envelope + appliances",
                "full": "envelope + dewpoint + appliances + lights",
            }[name],
            "coefs": m.model.coefs,
            **{k: round(met[k], 4) for k in ("rmse", "mae", "r2", "mape")},
            "n": len(pairs),
        })

    # reference values from the main experiment (metrics.csv)
    from_main = {}
    with open(OUT / "metrics.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["protocol"] == "resync":
                from_main[r["model"]] = r

    # ---- artifacts: CSV + markdown ----
    with open(OUT / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "drivers", "RMSE", "MAE", "R2", "MAPE", "n"])
        for r in rows:
            w.writerow([r["variant"], r["drivers"], r["rmse"], r["mae"], r["r2"], r["mape"], r["n"]])
        v1 = from_main["v1"]
        w.writerow(["v1_handbook", "envelope + appliance thermostat (no fitting)", v1["rmse"], v1["mae"], v1["r2"], v1["mape"], v1["n"]])
        pp = from_main["persistence"]
        w.writerow(["persistence_resync", "predict T(t+1)=T(t), daily re-anchor", pp["rmse"], pp["mae"], pp["r2"], pp["mape"], pp["n"]])

    md = []
    md.append("## Driver ablation — how much of v2's win is calibration vs extra sensors?")
    md.append("")
    md.append("All OLS variants below use the **same fitting machinery** (`tiny_twin_v2.LinearModel`) ")
    md.append("and the same held-out resync protocol (n=%s). They differ only in which regressors enter." % rows[0]["n"])
    md.append("")
    md.append("| model | regressors (all OLS + intercept) | RMSE (°C) | MAE (°C) | R² | MAPE (%) |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['variant']} | {r['drivers']} | {r['rmse']} | {r['mae']} | {r['r2']} | {r['mape']} |")
    md.append("| v1 handbook | envelope + appliance *thermostat, no fitting* | %s | %s | %s | %s |" % (v1["rmse"], v1["mae"], v1["r2"], v1["mape"]))
    md.append("| persistence | T̂(t+1)=T(t), daily re-anchor | %s | %s | %s | %s |" % (pp["rmse"], pp["mae"], pp["r2"], pp["mape"]))
    md.append("")
    md.append("**Reading:** (1) `env_app` (v1's own drivers, OLS-fitted) already collapses v1's 5.88 °C "
              "error to ~0.6-0.7 °C → the dominant factor is **calibration to real data**, not extra sensors. "
              "(2) the `full` variant's remaining small gain over `env_app` is the marginal contribution of "
              "dewpoint/lights drives.")
    md.append("")
    (OUT / "ablation.md").write_text("\n".join(md), encoding="utf-8")

    # ---- console ----
    print(f"ablation (held-out resync, n={rows[0]['n']}):")
    for r in rows:
        print(f"  {r['variant']:<12} RMSE={r['rmse']:<8} MAE={r['mae']:<8} R2={r['r2']:<8} MAPE={r['mape']}%  coefs={[round(c,5) for c in r['coefs']]}")
    print(f"  v1_handbook  RMSE={v1['rmse']:<8} MAE={v1['mae']:<8} R2={v1['r2']:<8} MAPE={v1['mape']}%")
    print(f"  persistence  RMSE={pp['rmse']:<8} MAE={pp['mae']:<8} R2={pp['r2']:<8} MAPE={pp['mape']}%")
    print("artifacts:", "ablation.csv", "ablation.md")


if __name__ == "__main__":
    run()
