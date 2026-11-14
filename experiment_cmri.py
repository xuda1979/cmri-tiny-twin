#!/usr/bin/env python3
"""experiment_cmri.py — calibrate + compare the cmri coupled two-zone twin.

Same evaluation contract as run_experiment.py (70/30 time split, daily-resync
primary, free-run secondary, in-sample train diagnostics, one-step persistence
floor, daily-resync persistence control) extended to the coupled two-zone
cmri twin:

  kitchen zone  T1 : envelope + wind infiltration + clear-sky solar GHI
                     + kitchen<->house coupling + dewpoint + appliances
                     + lights + thermostat drive        (all OLS-calibrated)
  house zone    Th : same driver set on the mean of the non-outlier house
                     sensors T2,T3,T4,T5,T7,T8,T9 (T6 excluded: anomalous)

The cmri twin adds four physically-motivated drivers the single-zone twins
ignore (wind infiltration, solar irradiance, inter-zone coupling, building
fabric thermal mass T_mass = EMA of T_out with physical time constant tau;
see cmri_twin.fabric_mass). All coefficients are OLS-fitted on the training
segment only (same LinearModel normal-equations machinery as tiny_twin_v2).

Also reports a whole-twin ablation (resync protocol): stepping up from v2's
own driver set {env, dewpoint, appliances, lights} through +thermostat,
+coupling, +fabric-mass, +wind, +solar to the full cmri model, so we can see
*which* physics block buys the improvement.

Artifacts → results/metrics_cmri.csv, results/coefficients_cmri.csv,
results/predictions_cmri.csv, results/comparison_cmri.md,
results/ablation_cmri.csv, results/ablation_cmri.md, results/cmri_plot.svg.

Pure Python standard library, zero dependencies.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import pathlib
import sys

from run_experiment import (OUT, ROOT, TRAIN_FRAC, PersistenceModel, V1Model,
                            load_data, metrics, run_protocol)
from tiny_twin_v2 import CalibratedTwin, LinearModel, Q_SCALE

import cmri_twin
from cmri_twin import (CmriTwin, clear_sky_ghi, fabric_mass, house_zone)

DATA = ROOT / "data" / "energydata_complete.csv"

# ---------------------------------------------------------------------------
# Frozen winner of the within-train CV (select_cv.py): nws+fab mask, Tset 21,
# alpha 300, fabric-mass EMA w=0.944 (tau = 3 h).  The documented fold-SD tie
# rule broke the resync tie in favor of the lowest free-run RMSE, which is how
# the slow building-fabric driver (driver 8) got in over the strict resync
# winner (nws, tset 20, alpha 100): its whole claim is anchoring the long-
# horizon open-loop trend, so it is selected on free-run robustness, not
# resync noise.  wind and clear-sky solar are dropped from both zone equations
# (unstable on this dataset).  Test data is NEVER used for selection or tuning.
# ---------------------------------------------------------------------------
# Slow second fabric capacitor (driver 9): EMA weight w2 for T_mass2,
# tau2 = 12 h (w2 = exp(-600/43200) = e^(-1/72) ~ 0.9862).  Frozen winner of
# select_cv_slowfab.py, the within-train prequential CV over two-timescale
# fabric configs: 9-driver mask, t_set 21, alpha 300, w2 0.986 (12 h).  It
# beats the fabric-only config within train on BOTH resync (0.6057 vs 0.6254)
# and free-run (0.7499 vs 0.9012), so it earns the single fresh held-out
# evaluation.  CMRI_KEEP is this same CV winner: env, coupl, dew, appl, light,
# thermo, fast fabric, slow fabric.
CMRI_ALPHA = 300.0
CMRI_T_SET = 21.0
CMRI_W = 0.944  # fabric-mass EMA smoothing weight (tau = 3 h), see fabric_mass
CMRI_KEEP = (0, 3, 4, 5, 6, 7, 8, 9)  # env,cpl,dew,appl,light,therm,fab,slowfab
CMRI_W2 = 0.986  # slow fabric EMA weight (tau2 = 12 h); None = single-fabric

# ---------------------------------------------------------------------------
# Data loading (full sensor panel + weather + computed GHI)
# ---------------------------------------------------------------------------


def load_cmri_data(path: pathlib.Path) -> dict[str, list]:
    rows: dict[str, list] = {
        "date": [], "T1": [], "T_out": [], "Tdewpoint": [],
        "Appliances": [], "lights": [], "Windspeed": [],
    }
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for j in range(2, 10):
            rows[f"T{j}"] = []
        for line in reader:
            rows["date"].append(dt.datetime.strptime(line[idx["date"]], "%Y-%m-%d %H:%M:%S"))
            rows["T1"].append(float(line[idx["T1"]]))
            for j in range(2, 10):
                rows[f"T{j}"].append(float(line[idx[f"T{j}"]]))
            rows["T_out"].append(float(line[idx["T_out"]]))
            rows["Tdewpoint"].append(float(line[idx["Tdewpoint"]]))
            rows["Appliances"].append(float(line[idx["Appliances"]]))
            rows["lights"].append(float(line[idx["lights"]]))
            v = line[idx["Windspeed"]].strip()
            rows["Windspeed"].append(float(v) if v else 0.0)
    rows["GHI"] = [clear_sky_ghi(d) for d in rows["date"]]
    rows["Th"] = house_zone([rows[f"T{j}"] for j in range(2, 10)])
    return rows


# ---------------------------------------------------------------------------
# Dual-zone simulation protocols
# ---------------------------------------------------------------------------


def run_protocol_cmri(twin, state, recs, resync_days):
    """Step a two-zone twin over records; return kitchen predictions.

    recs: list of (T1_obs, Th_obs, t_out, Tdewpoint, Appliances, lights,
                   Windspeed, GHI, T_mass) -- optionally a 10th entry T_mass2
                   (slow fabric state) for two-timescale driver-set selection.
    resync_days: list of index-lists (one per day), or None for free-run.
    state: initial observed (T1, Th).
    Returns kitchen pred list of len(recs); None where no prediction exists
    (first sample of each resync day, or sample 0 in free-run).
    """
    n = len(recs)
    pred = [None] * n
    if resync_days is None:
        t1, th = state
        for j in range(1, n):
            r = recs[j - 1]
            _, _, t_out, dew, load, lux, wind, ghi, tm, *rest = r
            t1, th = twin.step(t1, th, t_out, dew, load, lux, wind, ghi, tm,
                               rest[0] if rest else None)
            pred[j] = t1
        return pred
    for day in resync_days:
        t1, th = recs[day[0]][0], recs[day[0]][1]
        for a, b in zip(day[:-1], day[1:]):
            r = recs[a]
            _, _, t_out, dew, load, lux, wind, ghi, tm, *rest = r
            t1, th = twin.step(t1, th, t_out, dew, load, lux, wind, ghi, tm,
                               rest[0] if rest else None)
            pred[b] = t1
    return pred


def days_of(dates, off, end=None):
    """Group indices [off, end) into day-index lists (0-based within block)."""
    end = len(dates) if end is None else end
    groups: dict[dt.date, list[int]] = {}
    for j in range(end - off):
        groups.setdefault(dates[off + j].date(), []).append(j)
    return [v for _, v in sorted(groups.items())]


# ---------------------------------------------------------------------------
# Ablation variant (restricted feature mask on both zone equations)
# ---------------------------------------------------------------------------


class CmriVariant:
    """Two-zone twin using only a subset of the 10 cmri drivers (both zones)."""

    def __init__(self, keep: tuple[int, ...]):
        self.keep = keep
        self.k = LinearModel()
        self.h = LinearModel()

    def fit(self, T1, Th, t_out, dew, load, lux, wind, ghi, t_mass,
            t_mass2=None, t_set=CMRI_T_SET):
        Xk, yk, Xh, yh = [], [], [], []
        for i in range(len(T1) - 1):
            fk = CmriTwin._k_feats(T1[i], Th[i], t_out[i], dew[i], load[i],
                                   lux[i], wind[i], ghi[i], t_mass[i], t_set,
                                   None if t_mass2 is None else t_mass2[i])
            fh = CmriTwin._h_feats(T1[i], Th[i], t_out[i], dew[i], load[i],
                                   lux[i], wind[i], ghi[i], t_mass[i], t_set,
                                   None if t_mass2 is None else t_mass2[i])
            Xk.append([fk[j] for j in self.keep])
            yk.append(T1[i + 1] - T1[i])
            Xh.append([fh[j] for j in self.keep])
            yh.append(Th[i + 1] - Th[i])
        self.k.fit(Xk, yk)
        self.h.fit(Xh, yh)

    def step(self, t1, th, t_out, dew, load, lux, wind, ghi, t_mass,
             t_mass2=None, t_set=CMRI_T_SET):
        fk = CmriTwin._k_feats(t1, th, t_out, dew, load, lux, wind, ghi,
                               t_mass, t_set, t_mass2)
        fh = CmriTwin._h_feats(t1, th, t_out, dew, load, lux, wind, ghi,
                               t_mass, t_set, t_mass2)
        return (t1 + self.k.predict([fk[j] for j in self.keep]),
                th + self.h.predict([fh[j] for j in self.keep]))


# Feature indices (both zone equations): 0 env, 1 wind, 2 solar, 3 coupling,
# 4 dewpoint, 5 appliances, 6 lights, 7 thermostat, 8 fabric mass (T_mass-T),
# 9 slow fabric mass (T_mass2-T).  Bias added by LinearModel.  Ordered additive
# chain for the ablation table: v2 drivers -> +thermostat -> +coupling
# -> +fabric mass -> +wind -> +solar -> full 9-driver cmri model; a slow-fabric
# row is appended only once CMRI_W2 is frozen (two-timescale fabric config).
ABLATIONS = [
    # (short name, keep-features, human driver description)
    ("v2_drivers", (0, 4, 5, 6), "envelope + Tdewpoint + appliances + lights"),
    ("plus_thermo", (0, 4, 5, 6, 7), " + thermostat drive (gain fitted)"),
    ("plus_coupl", (0, 3, 4, 5, 6, 7), " + inter-zone coupling (Th-T1)"),
    ("plus_fab", (0, 3, 4, 5, 6, 7, 8), " + building-fabric mass (T_mass-T)"),
    ("plus_wind", (0, 1, 3, 4, 5, 6, 7, 8), " + wind infiltration wind*(T_out-T)"),
    ("cmri_full", (0, 1, 2, 3, 4, 5, 6, 7, 8),
     " + clear-sky solar GHI (full 9-driver cmri model)"),
]
if CMRI_W2 is not None:
    ABLATIONS.append(("plus_slowfab", (0, 3, 4, 5, 6, 7, 8, 9),
                      " + slow fabric mass (T_mass2-T, second capacitor)"))


# ---------------------------------------------------------------------------
# SVG plot (observed + v2/Cmri resync + Cmri free-run)
# ---------------------------------------------------------------------------

from run_experiment import COL_FREE, COL_OBS, COL_V2, _polyline  # noqa: E402


def make_cmri_plot(test_dates, obs, panels, out_path: pathlib.Path) -> None:
    n = len(obs)
    W, ML, MR, MT, MB = 1300, 90, 40, 50, 36
    GAP, PH = 18, 250
    total_h = MT + len(panels) * PH + (len(panels) - 1) * GAP + MB
    plot_w = W - ML - MR

    def xof(i):
        return ML + i / (n - 1) * plot_w

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" viewBox="0 0 {W} {total_h}">',
        f'<rect width="{W}" height="{total_h}" fill="white"/>',
        f'<text x="{W/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">'
        "cmri tiny-twin comparison — UCI Appliances held-out test (10-min samples)</text>",
        f'<text x="{W/2}" y="42" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">'
        "observed vs predicted indoor temperature T1 (°C)</text>",
    ]

    first_date = test_dates[0]
    n_days = (test_dates[-1] - first_date).days + 1
    day_ticks = []
    i = 0
    for d in range(n_days):
        while i < n and (test_dates[i] - first_date).days < d:
            i += 1
        if i < n and (test_dates[i] - first_date).days == d:
            day_ticks.append((i, d))

    for p, (name, preds, col) in enumerate(panels):
        ytop = MT + p * (PH + GAP)

        def yof(v, lo, hi):
            return ytop + 6 + (hi - v) / (hi - lo) * (PH - 12)

        lo = min(min(obs), min(e for e in preds if e is not None))
        hi = max(max(obs), max(e for e in preds if e is not None))
        pad = max((hi - lo) * 0.06, 0.05)
        lo, hi = lo - pad, hi + pad

        out.append(f'<text x="{ML-8}" y="{ytop+6}" text-anchor="start" font-family="sans-serif" font-size="12" font-weight="bold">{name}</text>')
        out.append(f'<rect x="{ML}" y="{ytop}" width="{plot_w}" height="{PH}" fill="#fafafa" stroke="#cccccc"/>')
        for v in (lo, hi, (hi + lo) / 2):
            y = yof(v, lo, hi)
            out.append(f'<line x1="{ML-4}" y1="{y:.1f}" x2="{ML}" y2="{y:.1f}" stroke="#666"/>')
            out.append(f'<text x="{ML-7}" y="{y+3.5:.1f}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#333">{v:.1f}</text>')
        for x, d in day_ticks[:: max(1, n_days // 6)]:
            xv = xof(x)
            out.append(f'<line x1="{xv:.1f}" y1="{ytop}" x2="{xv:.1f}" y2="{ytop+PH}" stroke="#dddddd"/>')
            out.append(f'<text x="{xv:.1f}" y="{ytop+PH+14}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#333">{(first_date+dt.timedelta(days=d)).strftime("%m-%d")}</text>')
        out.append(_polyline([(xof(i), yof(obs[i], lo, hi)) for i in range(n)], COL_OBS))
        out.append(_polyline([(xof(i), yof(preds[i], lo, hi)) for i in range(n) if preds[i] is not None], col, 1.4))
        ly = ytop + 16
        out.append(f'<line x1="{ML+10}" y1="{ly}" x2="{ML+30}" y2="{ly}" stroke="{COL_OBS}" stroke-width="1.4"/><text x="{ML+34}" y="{ly+3.5}" font-family="sans-serif" font-size="10">observed T1</text>')
        out.append(f'<line x1="{ML+140}" y1="{ly}" x2="{ML+160}" y2="{ly}" stroke="{col}" stroke-width="1.4"/><text x="{ML+164}" y="{ly+3.5}" font-family="sans-serif" font-size="10">{name.split(" (")[0]}</text>')

    out.append("</svg>")
    out_path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_cmri_data(DATA)
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)

    dates = data["date"]
    T, t_out, dew, load, lux, wind, ghi = (
        data["T1"], data["T_out"], data["Tdewpoint"],
        data["Appliances"], data["lights"], data["Windspeed"], data["GHI"],
    )
    T_h = data["Th"]

    recs5 = list(zip(T, t_out, dew, load, lux))
    tm = fabric_mass(t_out, CMRI_W)
    tm2 = None if CMRI_W2 is None else fabric_mass(t_out, CMRI_W2)
    # recs carry t_mass2 as an optional 10th entry; run_protocol_cmri unpacks
    # with *rest so 9-tuples (t_mass2=None) stay the frozen-config path.
    recs8 = list(zip(T, T_h, t_out, dew, load, lux, wind, ghi, tm)) if tm2 is None \
        else list(zip(T, T_h, t_out, dew, load, lux, wind, ghi, tm, tm2))

    test_days = days_of(dates, split)
    train_days = days_of(dates, 0, split)

    def fit_arrays(lo, hi):
        s = slice(lo, hi)
        base = (T[s], T_h[s], t_out[s], dew[s], load[s], lux[s], wind[s], ghi[s], tm[s])
        return base if tm2 is None else base + (tm2[s],)

    # ---- calibrate on TRAINING ONLY (config = frozen CV winner) ----
    cmri = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    cmri.fit(*fit_arrays(0, split))

    v2 = CalibratedTwin()
    v2.fit(T[:split], t_out[:split], dew[:split], load[:split], lux[:split])

    fitted_variants = {}
    for name, keep, _ in ABLATIONS:
        m = CmriVariant(keep)
        m.fit(*fit_arrays(0, split))
        fitted_variants[name] = m

    v1 = V1Model()
    pers = PersistenceModel()

    def evaluate_cmri(model, base, end, days):
        p = run_protocol_cmri(model, (T[base], T_h[base]), recs8[base:end], days)[1:]
        obs = [r[0] for r in recs8[base + 1:end]]
        return [(c, o) for c, o in zip(p, obs) if c is not None]

    preds = {}
    obs_all = {}
    for label, model, days, base, end, kind in (
        ("v1_resync", v1, test_days, split, n, "s"),
        ("v2_resync", v2, test_days, split, n, "s"),
        ("persistence_resync", pers, test_days, split, n, "s"),
        ("v1_freerun", v1, None, split, n, "s"),
        ("v2_freerun", v2, None, split, n, "s"),
        ("v1_train", v1, train_days, 0, split, "s"),
        ("v2_train", v2, train_days, 0, split, "s"),
        ("cmri_resync", cmri, test_days, split, n, "c"),
        ("cmri_freerun", cmri, None, split, n, "c"),
        ("cmri_train", cmri, train_days, 0, split, "c"),
    ):
        if kind == "s":
            p = run_protocol(model, T[base], recs5[base:end], days)[1:]
        else:
            p = run_protocol_cmri(model, (T[base], T_h[base]), recs8[base:end], days)[1:]
        preds[label] = p
        obs_all[label] = [r[0] for r in recs5[base + 1:end]]

    evaluable = {k: [(c, o) for c, o in zip(preds[k], obs_all[k]) if c is not None] for k in preds}

    def row(label, model_name):
        # model_name in {"v1","v2","cmri","persistence"}
        pairs = evaluable[label]
        m = metrics([c for c, _ in pairs], [o for _, o in pairs])
        return {
            "model": model_name,
            "protocol": label.split("_", 1)[1],
            "rmse": round(m["rmse"], 4),
            "mae": round(m["mae"], 4),
            "r2": round(m["r2"], 4),
            "mape": round(m["mape"], 3),
            "n": len(pairs),
        }

    rows = [
        row("v1_resync", "v1"), row("v2_resync", "v2"),
        row("cmri_resync", "cmri"), row("persistence_resync", "persistence"),
        row("v1_freerun", "v1"), row("v2_freerun", "v2"), row("cmri_freerun", "cmri"),
        row("v1_train", "v1"), row("v2_train", "v2"), row("cmri_train", "cmri"),
    ]

    # ---- whole-twin ablation on the resync protocol ----
    ab_rows = []
    prev_pairs = None
    for name, _, desc in ABLATIONS:
        pairs = evaluate_cmri(fitted_variants[name], split, n, test_days)
        m = metrics([c for c, _ in pairs], [o for _, o in pairs])
        d = ""
        if prev_pairs is not None:
            dp = (m["rmse"] - metrics([c for c, _ in prev_pairs], [o for _, o in prev_pairs])["rmse"])
            d = f"{dp:+.4f}"
        ab_rows.append({
            "variant": name, "drivers": desc,
            "rmse": round(m["rmse"], 4), "mae": round(m["mae"], 4),
            "r2": round(m["r2"], 4), "mape": round(m["mape"], 3),
            "d_rmse": d, "n": len(pairs),
        })
        prev_pairs = pairs

    with open(OUT / "metrics_cmri.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "protocol", "rmse", "mae", "r2", "mape", "n"])
        w.writeheader()
        w.writerows(rows)

    with open(OUT / "ablation_cmri.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "drivers", "rmse", "mae", "r2", "mape", "d_rmse", "n"])
        w.writeheader()
        w.writerows(ab_rows)

    with open(OUT / "coefficients_cmri.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zone", "coefficient", "value", "role"])
        rolek = ["envelope gain (C per dT-step)", "wind-infiltration gain (C per m/s per dT-step)",
                 "solar gain (C per W/m^2 per step)", "coupling gain (C per dT-step)",
                 "dewpoint latent gain", "appliance heat gain (per 100 W)",
                 "lights heat gain (per 100 W)", "thermostat gain (C per max(Tsp-T,0)-step)",
                 "fabric-mass gain (C per dT-step)", "slow fabric-mass gain (C per dT-step)"]
        # coefs layout = [bias, kept-driver coefs...]; kept drivers follow
        # CMRI_KEEP order; names/roles come from the full 9-driver layout.
        def write_zone(zone, coefs, names):
            for idx, c in enumerate(coefs):
                if idx == 0:
                    w.writerow([zone, "bias", round(c, 6), "bias (C/step)"])
                else:
                    j = CMRI_KEEP[idx - 1]
                    w.writerow([zone, names[idx - 1], round(c, 6), rolek[j]])
        write_zone("kitchen", cmri.k.coefs, [cmri.K_NAMES[j] for j in CMRI_KEEP])
        write_zone("house", cmri.h.coefs, [cmri.H_NAMES[j] for j in CMRI_KEEP])

    with open(OUT / "predictions_cmri.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "T1_observed", "Th_observed",
                    "v1_resync", "v2_resync", "cmri_resync",
                    "v1_freerun", "v2_freerun", "cmri_freerun"])
        for i in range(split + 1, n):
            k = i - split - 1
            g = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
            w.writerow([
                dates[i].strftime("%Y-%m-%d %H:%M:%S"),
                f"{T[i]:.3f}", f"{T_h[i]:.3f}",
                g(preds["v1_resync"][k]), g(preds["v2_resync"][k]), g(preds["cmri_resync"][k]),
                g(preds["v1_freerun"][k]), g(preds["v2_freerun"][k]), g(preds["cmri_freerun"][k]),
            ])

    by = {(r["model"], r["protocol"]): r for r in rows}
    ab_by = {r["variant"]: r for r in ab_rows}

    def tb(key):
        return by[key]["rmse"]

    md = []
    md.append("# cmri tiny-twin vs single-zone twins — UCI Appliances (held-out 30%)")
    md.append("")
    md.append(f"- Dataset: `data/energydata_complete.csv`, {n} samples @10 min, {dates[0]} → {dates[-1]}.")
    md.append(f"- Split: first {split} samples (70%) calibrate, last {n - split} (30%) test — by time.")
    md.append("- Target: indoor kitchen temperature T1 (°C). All fitted models calibrated on train only; v1 never touches data.")
    md.append("- **cmri twin:** coupled two-zone RC. Kitchen T1 + house zone T_h = mean(T2,T3,T4,T5,T7,T8,T9); "
              "T6 excluded as an anomalous sensor. Driver set (all OLS-fitted): envelope, inter-zone coupling, "
              "Tdewpoint, appliances, lights, thermostat drive, and building-fabric thermal mass "
              "`(T_mass-T)` (EMA of T_out, tau = 3 h, exogenous). Wind infiltration `wind*(T_out-T)` and "
              "clear-sky solar GHI (Chievres AB, Belgium, DST-aware) are implemented and calibrated but were "
              "**selected out** by the within-train CV because they lose free-run stability on this dataset; "
              "the frozen winner is keep (0,3,4,5,6,7,8) `T_set=21.0 alpha=300 w=0.944`.")
    md.append("")
    md.append("## Held-out test metrics (resync protocol — primary)")
    md.append("")
    md.append("| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |")
    md.append("|---|---|---|---|---|---|")
    for mm in ("v1", "v2", "cmri", "persistence"):
        r = by[(mm, "resync")]
        md.append(f"| {mm} | {r['rmse']} | {r['mae']} | {r['r2']} | {r['mape']} | {r['n']} |")
    md.append("")
    md.append("## Free-run robustness (no resync — secondary)")
    md.append("")
    md.append("| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |")
    md.append("|---|---|---|---|---|---|")
    for mm in ("v1", "v2", "cmri"):
        r = by[(mm, "freerun")]
        md.append(f"| {mm} | {r['rmse']} | {r['mae']} | {r['r2']} | {r['mape']} | {r['n']} |")
    md.append("")
    md.append("## In-sample train diagnostics (resync — not the evaluation)")
    md.append("")
    md.append("| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |")
    md.append("|---|---|---|---|---|---|")
    for mm in ("v1", "v2", "cmri"):
        r = by[(mm, "train")]
        md.append(f"| {mm} | {r['rmse']} | {r['mae']} | {r['r2']} | {r['mape']} | {r['n']} |")
    md.append("")
    md.append("## Whole-twin ablation on the held-out resync protocol")
    md.append("")
    md.append("Features accumulate left-to-right (bold = \u0394 RMSE vs the previous row).")
    md.append("")
    md.append("| variant | drivers | RMSE (°C) | MAE (°C) | R² | \u0394 RMSE |")
    md.append("|---|---|---|---|---|---|")
    for a in ab_rows:
        md.append(f"| {a['variant']} | {a['drivers']} | {a['rmse']} | {a['mae']} | {a['r2']} | {a['d_rmse']} |")
    md.append("")
    md.append("## Calibrated cmri coefficients (train only)")
    md.append("")
    md.append("Config is the frozen within-train CV winner: drivers "
              f"{'+'.join(cmri.K_NAMES[j].split()[0] for j in CMRI_KEEP)} "
              f"(keep {list(CMRI_KEEP)}), T_set={CMRI_T_SET}, alpha={CMRI_ALPHA}.")
    md.append("")
    md.append("### Kitchen zone T1")
    md.append("")
    md.append("| coefficient | value | role |")
    md.append("|---|---|---|")
    for i, c in enumerate(cmri.k.coefs):
        nm = "bias" if i == 0 else cmri.K_NAMES[CMRI_KEEP[i - 1]]
        rl = "bias (C/step)" if i == 0 else rolek[CMRI_KEEP[i - 1]]
        md.append(f"| {nm} | {c:.6f} | {rl} |")
    md.append("")
    md.append("### House zone T_h")
    md.append("")
    md.append("| coefficient | value | role |")
    md.append("|---|---|---|")
    for i, c in enumerate(cmri.h.coefs):
        nm = "bias" if i == 0 else cmri.H_NAMES[CMRI_KEEP[i - 1]]
        rl = "bias (C/step)" if i == 0 else rolek[CMRI_KEEP[i - 1]]
        md.append(f"| {nm} | {c:.6f} | {rl} |")
    md.append("")
    md.append("## Noise floor (one-step persistence — irreducible reference)")
    md.append("")
    floor_pairs = [(T[j - 1], T[j]) for j in range(split + 1, n)]
    r_floor = metrics([c for c, _ in floor_pairs], [o for _, o in floor_pairs])
    md.append("| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| persistence (1-step) | {r_floor['rmse']:.4f} | {r_floor['mae']:.4f} "
              f"| {r_floor['r2']:.4f} | {r_floor['mape']:.3f} | {len(floor_pairs)} |")
    md.append("")
    md.append("## Verdict")
    md.append("")
    cmr_res, v2_res = by[("cmri", "resync")], by[("v2", "resync")]
    cmr_free, v2_free = by[("cmri", "freerun")], by[("v2", "freerun")]
    dv_res = v2_res["rmse"] - cmr_res["rmse"]
    dv_free = v2_free["rmse"] - cmr_free["rmse"]
    verdict = (
        f"- **Resync:** v2 RMSE = {v2_res['rmse']} °C → cmri RMSE = {cmr_res['rmse']} °C "
        f"(Δ **{dv_res:+.4f} °C**). cmri beats v2 on "
        f"{','.join(x for x in ('rmse', 'mae', 'r2', 'mape') if _beats(cmr_res, v2_res, x))}.\n"
        f"- **Free-run:** v2 RMSE = {v2_free['rmse']} °C (R² = {v2_free['r2']}) → "
        f"cmri RMSE = {cmr_free['rmse']} °C (R² = {cmr_free['r2']}) (Δ **{dv_free:+.4f} °C**).\n"
        "- Physics drivers: see the ablation table — the marginal RMSE gain per added "
        "driver is the evidence for which cmri physics term matters on this dataset."
    )
    md.append(verdict)
    md.append("")
    md.append(f"Plot: `results/cmri_plot.svg` · data: `results/predictions_cmri.csv` · "
              f"metrics: `results/metrics_cmri.csv` · ablation: `results/ablation_cmri.csv`")
    (OUT / "comparison_cmri.md").write_text("\n".join(md), encoding="utf-8")

    make_cmri_plot(dates[split:], obs_all["cmri_resync"],
                   [("v2 (resync)", preds["v2_resync"], COL_V2),
                    ("cmri (resync)", preds["cmri_resync"], "#2ca02c"),
                    ("cmri (free-run)", preds["cmri_freerun"], COL_FREE)],
                   OUT / "cmri_plot.svg")

    print(f"n={n}  split@{split}  (train {TRAIN_FRAC:.0%}, test {n - split})")
    for k in ("v1_resync", "v2_resync", "cmri_resync", "persistence_resync",
              "v1_freerun", "v2_freerun", "cmri_freerun",
              "v1_train", "v2_train", "cmri_train"):
        r = by[(k.split("_", 1)[0], k.split("_", 1)[1])]
        print(f"  {k:<12} RMSE={r['rmse']:<8} MAE={r['mae']:<8} R2={r['r2']:<7} MAPE={r['mape']}%  n={r['n']}")
    for a in ab_rows:
        print(f"  ablation {a['variant']:<12} RMSE={a['rmse']:<8} dRMSE={a['d_rmse']}")
    print("cmri kitchen coefs (kept-driver layout; bias first):")
    k_kept = ["bias"] + [cmri.K_NAMES[j] for j in CMRI_KEEP]
    print(f"  {dict(zip(k_kept, [round(c, 5) for c in cmri.k.coefs]))}")
    h_kept = ["bias"] + [cmri.H_NAMES[j] for j in CMRI_KEEP]
    print(f"  house: {dict(zip(h_kept, [round(c, 5) for c in cmri.h.coefs]))}")
    print("artifacts:", sorted(p.name for p in OUT.iterdir()))


def _beats(a, b, m):
    return a[m] < b[m] if m in ("rmse", "mae", "mape") else a[m] > b[m]


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        cmri_twin.selftest()
    else:
        run()
