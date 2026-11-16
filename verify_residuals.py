#!/usr/bin/env python3
"""verify_residuals.py — residual diagnostics: WHERE does cmri go wrong (and right)?

Pure-Python, reads ONLY the already-frozen held-out predictions written to
results/predictions_cmri.csv (resync protocol).  No new test evaluation, no
model fitting: it re-aggregates the single frozen evaluation to characterise
the cmri-vs-v2 residuals instead of their means.

Sections:
  1. Residual autocorrelation (lag-1 rho per model) -- the 10-min resync errors
     are strongly correlated within a day; this is the lowest-order reason the
     naive iid tests in verify_significance.py overstate significance and why
     the HAC Diebold-Mariano is the honest test.
  2. Clock-hour error profile -- mean |err| of each model by hour of day, plus
     the paired differential d_h = |e_v2| - |e_cmri| (positive => cmri better).
     Shows whether the edge of section 5.6 is spread over all hours or owned by
     a few (expected: early morning heating ramp + evening).
  3. Weekday vs weekend -- does either twin degrade on unoccupied days (when
     the thermostat/setpoint driver is least informative)?
  4. Worst episodes -- the 10 individual steps with the largest cmri error and
     the 5 worst days by mean |err| for each model.
"""

import csv
import datetime as dt
import math
import random
import statistics

PRED = "results/predictions_cmri.csv"
SEED = 42
B = 5000


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive paired d, normal approx."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided H1: P(d>0) > 0.5


def main() -> None:
    rows = []
    with open(PRED) as f:
        for r in csv.DictReader(f):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            ev2 = obs - float(r["v2_resync"])
            ecm = obs - float(r["cmri_resync"])
            d = dt.datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S")
            rows.append((d, ev2, ecm))
    n = len(rows)

    # 1. Residual autocorrelation (lag-1).
    def lag1(xs: list[float]) -> float:
        mu = statistics.fmean(xs)
        xb = [x - mu for x in xs]
        den = sum(v * v for v in xb)
        num = sum(a * b for a, b in zip(xb[:-1], xb[1:]))
        return num / den if den > 0 else 0.0

    ev2 = [e for _, e, _ in rows]
    ecm = [e for _, _, e in rows]
    print(f"n={n} resync steps (frozen predictions)")
    print(f"lag-1 error autocorrelation:  v2  rho={lag1(ev2):.3f}"
          f"   cmri rho={lag1(ecm):.3f}")

    # 2. Clock-hour profile.
    by_hour = {h: [] for h in range(24)}  # h -> (av, ac)
    for d, ev2h, ecmh in rows:
        by_hour[d.hour].append((abs(ev2h), abs(ecmh)))
    print("\nper clock-hour mean |err| (degC), paired differential, win rate:")
    print(f"  {'hour':<6} {'v2':>7} {'cmri':>7} {'d=e_v2-e_cmri':>13} {'win':>5}")
    for h in range(24):
        av = [a for a, _ in by_hour[h]]
        ac = [c for _, c in by_hour[h]]
        dd = [a - c for a, c in by_hour[h]]
        win = sum(1 for x in dd if x > 1e-12)
        print(f"  {h:02d}h   {statistics.fmean(av):>7.4f} "
              f"{statistics.fmean(ac):>7.4f} {statistics.fmean(dd):>+13.4f} "
              f"{win/len(dd):>5.3f}")
    all_av = [a for h in range(24) for a, _ in by_hour[h]]
    all_ac = [c for h in range(24) for _, c in by_hour[h]]
    all_dd = [a - c for a, c in zip(all_av, all_ac)]
    all_win = sum(1 for x in all_dd if x > 1e-12)
    print(f"  all   {statistics.fmean(all_av):>7.4f} {statistics.fmean(all_ac):>7.4f} "
          f"{statistics.fmean(all_dd):>+13.4f} {all_win/len(all_dd):>5.3f}")

    # Day-phase bands (condensed view of the same data).
    phases = [(0, 6, "night 00-05h"), (6, 12, "morning 06-11h"),
              (12, 18, "afternoon 12-17h"), (18, 24, "evening 18-23h")]
    print("\nday-phase mean |err| (degC), paired differential:")
    for lo, hi, label in phases:
        av = [a for h in range(lo, hi) for a, _ in by_hour[h]]
        ac = [c for h in range(lo, hi) for _, c in by_hour[h]]
        dd = [a - c for a, c in zip(av, ac)]
        win = sum(1 for x in dd if x > 1e-12)
        print(f"  {label:<18} v2 {statistics.fmean(av):.4f}  cmri "
              f"{statistics.fmean(ac):.4f}  d {statistics.fmean(dd):+.4f}  "
              f"win {win/len(dd):.3f}")

    # 3. Weekday vs weekend (calendar day of the observation's local time).
    wd_v2, wd_cm, we_v2, we_cm = [], [], [], []
    for d, e2, ec in rows:
        (wd_v2 if d.weekday() < 5 else we_v2).append(abs(e2))
        (wd_cm if d.weekday() < 5 else we_cm).append(abs(ec))
    print("\nweekday vs weekend mean |err| (degC):")
    print(f"  weekday (n={len(wd_v2)}):  v2 {statistics.fmean(wd_v2):.4f}"
          f"   cmri {statistics.fmean(wd_cm):.4f}"
          f"   d {statistics.fmean(a-b for a,b in zip(wd_v2, wd_cm)):+.4f}")
    print(f"  weekend (n={len(we_v2)}):  v2 {statistics.fmean(we_v2):.4f}"
          f"   cmri {statistics.fmean(we_cm):.4f}"
          f"   d {statistics.fmean(a-b for a,b in zip(we_v2, we_cm)):+.4f}")

    # 4. Worst episodes.
    dd_all = [abs(a) - abs(c) for a, c in zip(ev2, ecm)]
    labs = ["cmri", "v2"]
    cols = {"cmri": ecm, "v2": ev2}
    for lab in labs:
        worst = sorted(range(n), key=lambda i: abs(cols[lab][i]))[-10:][::-1]
        print(f"\ntop-10 worst single steps by {lab} abs error:")
        for i in worst:
            d = rows[i][0]
            print(f"  {d:%Y-%m-%d %H:%M}  |err|={abs(cols[lab][i]):.3f} C "
                  f"v2={ev2[i]:+.3f} cmri={ecm[i]:+.3f}")

    # per-day mean |err| for worst-days listing.
    day_abs = {"cmri": {}, "v2": {}}  # date -> [abs err]
    for d, e2, ec in rows:
        key = d.date()
        day_abs["cmri"].setdefault(key, []).append(abs(ec))
        day_abs["v2"].setdefault(key, []).append(abs(e2))
    for lab in labs:
        means = [(k, statistics.fmean(v)) for k, v in day_abs[lab].items()]
        worst = sorted(means, key=lambda kv: -kv[1])[:5]
        print(f"\n5 worst days by {lab} daily mean |err|:")
        for k, m in worst:
            print(f"  {k} {m:.3f} C")
    # cmri best days, for contrast.
    means = [(k, statistics.fmean(v)) for k, v in day_abs["cmri"].items()]
    best = sorted(means, key=lambda kv: kv[1])[:3]
    print("\n3 best days by cmri daily mean |err|:")
    for k, m in best:
        print(f"  {k} {m:.3f} C")

    # Day-level paired differential: how many days cmri wins, and the margin by
    # which v2 wins on the days it does win.
    wins, losses = [], []
    for k in day_abs["cmri"]:
        dm = statistics.fmean(day_abs["v2"][k]) - statistics.fmean(day_abs["cmri"][k])
        (wins if dm > 0 else losses).append(dm)
    print(f"\nday-level: cmri wins {len(wins)} days (mean margin +{statistics.fmean(wins):.3f} C),"
          f" v2 wins {len(losses)} (mean margin {statistics.fmean(losses):+.3f} C)")

    # 5. Honest (day-block) significance of the phase and weekday claims.  The
    # per-step win rates and pooled d above overstate significance — residuals
    # have lag-1 rho ~ 0.99 within a day — so each claim gets the same whole-day
    # blocking this repo uses everywhere else (cf. verify_horizon.py / the RMSE
    # gap scripts): the calendar day is the only defensible independence unit.
    # For band B, per-day mean paired d = mean|e_v2| - mean|e_cmri| over B's
    # steps, then a sign test on which days have positive d.
    print("\nday-block sign test per day-phase band (whole-day blocks = honest unit):")
    for lo, hi, label in phases:
        by_day: dict[dt.date, list[float]] = {}
        for d, e2, ec in rows:
            if lo <= d.hour < hi:
                by_day.setdefault(d.date(), []).append(abs(e2) - abs(ec))
        m = [statistics.fmean(dd) for dd in by_day.values()]
        w = sum(1 for x in m if x > 1e-12)
        z, p = sign_test(w, len(m))
        print(f"  {label:<18} {len(m):>3} days  cmri better {w:>2}  "
              f"mean day d {statistics.fmean(m):+.4f}  sign z {z:+.2f}  "
              f"one-sided p {p:.3f}")

    # Weekday vs weekend day blocks: sign test per class, plus a day-block
    # bootstrap 95% CI on the pooled within-class mean paired d.  Both are
    # paired (|e_v2|-|e_cmri| per step), so a positive pooled d is a cmri win.
    print("\nweekday vs weekend — day-block sign tests + bootstrap CI "
          "on pooled mean paired d:")
    wd_days: dict[dt.date, list[float]] = {}
    we_days: dict[dt.date, list[float]] = {}
    for d, e2, ec in rows:
        dd = [abs(e2) - abs(ec)]
        (wd_days if d.weekday() < 5 else we_days).setdefault(d.date(), []).extend(dd)
    for lab, D in [("weekday", wd_days), ("weekend", we_days)]:
        keys = sorted(D)
        m = [statistics.fmean(D[k]) for k in keys]
        w = sum(1 for x in m if x > 1e-12)
        z, p = sign_test(w, len(m))
        rng = random.Random(SEED)
        means = []
        for _ in range(B):
            s = [m[rng.randrange(len(m))] for _ in range(len(m))]
            means.append(statistics.fmean(s))
        means.sort()
        lo, hi = means[int(B * 0.025)], means[int(B * 0.975)]
        mm = statistics.fmean(m)
        se = (statistics.pstdev(m) / math.sqrt(len(m) - 1)) if len(m) > 1 else math.nan
        print(f"  {lab} ({len(keys)} days): cmri better {w} days  "
              f"pooled mean d {mm:+.4f}  sign z {z:+.2f} (p {p:.3f})  "
              f"bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")

    # Cross-check the committed 87.6% evening share of total paired-d.
    band_d = {}
    for lo, hi, label in phases:
        band_d[label] = sum(abs(e2) - abs(ec) for d, e2, ec in rows
                            if lo <= d.hour < hi)
    share = band_d["evening 18-23h"] / sum(band_d.values())
    print(f"\nevening (18-23h) share of total paired-d across all phases: "
          f"{share:.3f} (committed value 0.876) — cross-check "
          f"{'PASS' if abs(share - 0.876) < 0.005 else 'FAIL'}")

    # Overall paired mean d across every step.
    print(f"\noverall paired d = |e_v2|-|e_cmri|: mean {statistics.fmean(dd_all):+.4f} C"
          f" (win rate {sum(1 for x in dd_all if x > 1e-12)/n:.3f})")


if __name__ == "__main__":
    main()
