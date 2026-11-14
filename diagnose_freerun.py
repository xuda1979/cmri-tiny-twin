#!/usr/bin/env python3
"""diagnose_freerun.py — quantify the free-run drift mechanism on the FROZEN
held-out predictions (results/predictions_cmri.csv, single-touch artifact).

No new evaluation, no refitting: this only re-reads the once-written free-run
predictions to answer *how* the error accumulates (constant per-step bias vs
time-varying drift), which dictates what a multi-horizon calibration should
target.  Question: cmri free-run RMSE 1.7207 > v2 1.5752 despite cmri winning
daily-resync by 20%.  Where does that edge get lost at 41 days?
"""

import csv
import math

from run_experiment import ROOT, metrics


def load():
    p = ROOT / "results" / "predictions_cmri.csv"
    obs, cmf, v2f = [], [], []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            obs.append(float(row["T1_observed"]))
            a, b = row["cmri_freerun"].strip(), row["v2_freerun"].strip()
            cmf.append(float(a) if a else None)
            v2f.append(float(b) if b else None)
    return obs, cmf, v2f


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) * (a - mx) for a in xs)
    b1 = sxy / sxx
    return my - b1 * mx, b1  # (intercept at x=0, slope)


def report(name, obs, pred):
    n = len(obs)
    err = [p - o for p, o in zip(pred, obs)]
    m = metrics(pred, obs)
    b0, b1 = ols(list(range(n)), err)
    resid = [e - (b0 + b1 * i) for i, e in enumerate(err)]
    rr = math.sqrt(sum(r * r for r in resid) / len(resid))
    # bias by quarter of the run
    q = n // 4
    def bias(lo, hi):
        return sum(err[lo:hi]) / len(err[lo:hi])
    print(f"\n== {name}: free-run over n={n} steps ({n/144:.1f} days)")
    print(f"  overall RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")
    print(f"  mean per-step bias  {sum(err)/n:+.4f} C/step  (sd {_sd(err):.4f})")
    print(f"  linear trend of error: intercept={b0:+.4f} slope={b1:+.6f} C/step^2")
    print(f"  RMSE after removing linear trend: {rr:.4f}  (vs raw {m['rmse']:.4f})")
    print(f"  per-quarter mean signed error: "
          + "  ".join(f"[{bias(j*q,(j+1)*q):+.4f} C/step]" for j in range(4)))
    # growth in cumulative error
    cum = []
    s = 0.0
    for e in err:
        s += e
        cum.append(s)
    print(f"  total accumulated error at end: {s:+.2f} C over {n} steps")
    print(f"  worst-case accumulated deviation: {min(cum):+.2f} .. {max(cum):+.2f} C")
    # 1-step lag structure of err growth: is err[t]-err[t-1] (bias series) white?
    d = [err[i] - err[i - 1] for i in range(1, n)]
    print(f"  sd of per-step error increment (bias noise) = {_sd(d):.4f} C")


def _sd(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def main():
    obs, cmf, v2f = load()
    cmfp = [p for p in cmf if p is not None]
    k = len(cmfp)  # free-run evaluable steps
    obs_c = [o for p, o in zip(cmf, obs) if p is not None]
    obs_v = [o for p, o in zip(v2f, obs) if p is not None]
    v2p = [p for p in v2f if p is not None]
    report("cmri (fabric, frozen)", obs_c, cmfp)
    report("v2 (single-zone)", obs_v, v2p)
    # cross-model per-step error correlation
    e1 = [p - o for p, o in zip(cmfp, obs_c)]
    e2 = [p - o for p, o in zip(v2p, obs_v)]
    mu1, mu2 = sum(e1) / len(e1), sum(e2) / len(e2)
    rho = (sum((a - mu1) * (b - mu2) for a, b in zip(e1, e2))
           / (len(e1) * e1_sd(e1) * e1_sd(e2)))
    print(f"\ncorrelation of cmri/v2 free-run errors: {rho:.3f}")


def e1_sd(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


if __name__ == "__main__":
    main()
