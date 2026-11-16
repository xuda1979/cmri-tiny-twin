#!/usr/bin/env python3
"""verify_autocorr_within.py — is the §7 item 1 "residuals are not iid"
reading (lag-1 error autocorrelation rho = 0.987 v2 / 0.990 cmri; the
smallest-order reason the naive iid tests of §5.5 overstate significance,
and why the HAC Diebold-Mariano is the honest test) a structural fact, not
an artifact of the single held-out draw?

§7 item 1 committed claims (frozen held-out resync,
results/predictions_cmri.csv, as analyzed by verify_residuals.py /
verify_significance.py):
  * lag-1 error autocorrelation rho = 0.987 (v2) and 0.990 (cmri) over the
    n = 5879 resync steps — a 10-min residual at 20:00 explains ~98% of the
    variance of the one at 20:10;
  * the residual sequence is therefore NOT iid: the naive per-step tests of
    §5.5 (win rate, paired t, naive DM) overstate significance;
  * the HAC Diebold-Mariano (Newey-West long-run variance, Bartlett kernel,
    bw = 1440) is the honest test: DM = 2.92, two-sided p = 3.5e-03.

Those numbers were computed once on the held-out test. This script verifies
the autocorrelation reading WITHOUT touching the held-out test for fitting:
leg A re-fits fresh v2 (CalibratedTwin) and cmri (frozen-config CmriTwin)
twins on the within-train fit slice [0, 11051), replays the daily-resync
protocol on the within-train val block [11051, 13814) (the 20-day block,
n = 2743 resync steps), and applies the SAME autocorrelation machinery
(full-series lag-1 rho per model on the chronological residual sequence,
per-day within-day lag-1 rho median, rho over the consecutive cross-day
boundary pairs, and the naive-per-step z vs whole-day-block z inflation,
plus a fresh HAC DM at bw = 1440) to the fresh residual sequences; leg B
re-reads the frozen results/predictions_cmri.csv (read-only) and
re-aggregates every committed §7 item 1 digit exactly as
verify_residuals.py / verify_significance.py do.

Gates (power-aware: the 20-day block cannot reproduce the 42-day block's
strength, so the gates are structural invariants of the "not iid" reading,
not digit matches): within-train, (i) BOTH models' full-series lag-1 rho >
0.95 (the residual sequence carries strong day-scale serial dependence —
the exact claim of item 1), (ii) the per-day within-day median lag-1 rho >
0.9 for both models (the autocorrelation lives INSIDE days, at the 10-min
within-day scale), (iii) the consecutive cross-day boundary-pair rho is
near zero (< 0.3) while the within-day median minus the boundary rho > 0.5
(the day boundary resets the correlation — the calendar day is the honest
independence unit, the structural justification for every whole-day-block
test in this repo), (iv) the naive per-step z exceeds the whole-day-block z
by at least 4x within-train (the "iid tests overstate significance"
mechanism reproduces on fresh fits), (v) the fresh HAC DM (bw = 1440) has
|stat| < half the naive z (the honest test discounts the iid overstatement);
the held-out leg must re-produce every committed digit (rho = 0.987 / 0.990,
the naive z = 14.77 vs day-block z = 2.32 inflation ratio 6.4x, the HAC
bw = 576 p = 1.74e-02 and bw = 1440 p = 3.5e-03).

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, metrics, run_protocol  # noqa: E402
from experiment_cmri import (CMRI_ALPHA, CMRI_KEEP, CMRI_T_SET, CMRI_W,  # noqa: E402
                             CMRI_W2, days_of, load_cmri_data,
                             run_protocol_cmri)
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from cmri_twin import CmriTwin, fabric_mass  # noqa: E402

VALID_FRAC = 0.2
PRED = "results/predictions_cmri.csv"
TOL3 = 0.00055          # 3dp rho tolerance (0.987 / 0.990)
Z_TOL = 0.01            # z-value tolerance (deterministic normal approx)
P_TOL = 0.001           # two-sided p tolerance (HAC DM)
INFL_TOL = 0.06         # inflation-ratio 1dp tolerance (6.4x)
T_TOL = 0.02            # HAC DM stat 2dp tolerance

# Frozen §7 item 1 committed digits (leg-B audit targets), exactly the
# numbers verify_residuals.py / verify_significance.py / REPORT.md §7 item 1
# print.
C_FULL = (0.987, 0.990)                # lag-1 rho v2 / cmri (n = 5879)
C_NAIVE_Z = 14.77                      # naive per-step z (paired t on d)
C_BLOCK_Z = 2.32                       # whole-day-block z (per-day means)
C_INFL = 6.4                           # naive/block inflation factor (1dp)
C_HAC576 = (2.38, 0.0174)              # DM stat, two-sided p at bw = 576
C_HAC1440 = (2.92, 0.00346)            # DM stat, two-sided p at bw = 1440
# derived frozen re-aggregations (printed, audited to 3dp as cross-checks)
C_WD_MED = (0.982, 0.981)              # within-day median lag-1 rho v2 / cmri
C_BOUNDARY = (-0.058, -0.041)          # cross-day boundary-pair rho v2 / cmri
C_GAPS = 41
C_N = 5879


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_normal_p(z: float) -> float:
    return 2.0 * (1.0 - phi(abs(z)))


def lag1(xs: list[float]) -> float:
    """Lag-1 autocorrelation coefficient of a sequence (verify_residuals.py)."""
    mu = statistics.fmean(xs)
    xb = [x - mu for x in xs]
    den = sum(v * v for v in xb)
    num = sum(a * b for a, b in zip(xb[:-1], xb[1:]))
    return num / den if den > 0 else 0.0


def dm_hac(d: list[float], bw: int) -> tuple[float, float]:
    """Diebold-Mariano with Newey-West HAC long-run variance (identical to
    verify_significance.py's dm_hac: Bartlett kernel w = 1 - l/(bw+1))."""
    n = len(d)
    m = statistics.fmean(d)
    g0 = sum((x - m) ** 2 for x in d) / n
    lrv = g0
    for l in range(1, bw + 1):
        gl = sum((d[i] - m) * (d[i + l] - m) for i in range(n - l)) / n
        w = 1.0 - l / (bw + 1.0)  # Bartlett kernel
        lrv += 2.0 * w * gl
    if lrv <= 0.0:
        return math.nan, math.nan
    stat = m / math.sqrt(lrv / n)
    return stat, two_sided_normal_p(stat)


def pair_rho(pairs: list[tuple[float, float]]) -> float:
    """rho over a set of (a, b) pairs, mean-deviated against all members
    (matches /tmp/explore_ac.py's cross-day boundary rho construction)."""
    if not pairs:
        return 0.0
    vals = [v for p in pairs for v in p]
    mu = statistics.fmean(vals)
    num = sum((a - mu) * (b - mu) for a, b in pairs)
    den = sum((v - mu) ** 2 for v in vals)
    return num / den if den > 0 else 0.0


def autocorr_block(ev2: list[float], ecm: list[float],
                   ds: list) -> dict:
    """Full autocorrelation read on aligned (ev2, ecm, datetime) rows in
    chronological order (anchor rows already removed). Returns the full-series
    lag-1 rho per model, the per-day within-day lag-1 rho list, the cross-day
    boundary pairs per model and the per-day paired-d lists."""
    day_v2: dict = {}
    day_cm: dict = {}
    for a, c, d in zip(ev2, ecm, ds):
        k = d.date() if hasattr(d, "date") else d
        day_v2.setdefault(k, []).append(a)
        day_cm.setdefault(k, []).append(c)
    keys = sorted(day_v2)
    full = (lag1(ev2), lag1(ecm))
    wd_v2 = [lag1(day_v2[k]) for k in keys]
    wd_cm = [lag1(day_cm[k]) for k in keys]
    bp_v: list[tuple[float, float]] = []
    bp_c: list[tuple[float, float]] = []
    for i in range(len(ev2) - 1):
        k0 = ds[i].date() if hasattr(ds[i], "date") else ds[i]
        k1 = ds[i + 1].date() if hasattr(ds[i + 1], "date") else ds[i + 1]
        if k0 != k1:
            bp_v.append((ev2[i], ev2[i + 1]))
            bp_c.append((ecm[i], ecm[i + 1]))
    return {"keys": keys, "full": full,
            "wd": (statistics.median(wd_v2), statistics.median(wd_cm)),
            "wd_raw": (wd_v2, wd_cm),
            "boundary": (pair_rho(bp_v), pair_rho(bp_c)),
            "ngaps": len(bp_v), "ndays": len(keys)}


def z_inflation(d: list[float], ds: list) -> tuple[float, float, float]:
    """naive per-step z vs whole-day-block z and their inflation ratio.
    d = per-step paired d (|e_v2| - |e_cmri|); ds = aligned day keys."""
    n = len(d)
    md = statistics.fmean(d)
    sd = statistics.pstdev(d)
    z_naive = md / sd * math.sqrt(n) if sd > 0 else math.nan
    day: dict = {}
    for x, k in zip(d, ds):
        day.setdefault(k, []).append(x)
    blk = [statistics.fmean(v) for v in day.values()]
    z_block = (statistics.fmean(blk) / statistics.pstdev(blk)
               * math.sqrt(len(blk))) if len(blk) > 1 else math.nan
    return z_naive, z_block, (z_naive / z_block if z_block else math.nan)


def day_key_of(row_date) -> str:
    return row_date.date().isoformat() if hasattr(row_date, "date") else str(row_date)


def main() -> None:
    data = load_cmri_data("data/energydata_complete.csv")
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    val_start = int(round((1.0 - VALID_FRAC) * split))
    T, Th, t_out = data["T1"], data["Th"], data["T_out"]
    dew, load, lux = data["Tdewpoint"], data["Appliances"], data["lights"]
    wind, ghi = data["Windspeed"], data["GHI"]
    tm = fabric_mass(t_out, CMRI_W)
    tm2 = fabric_mass(t_out, CMRI_W2)
    dates = data["date"]

    print("autocorrelation-structure probe of the resync residuals "
          "(daily-resync protocol); held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + val-block autocorrelation ---------
    fs = slice(0, val_start)
    m2 = CalibratedTwin()
    m2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    mc = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    mc.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
           ghi[fs], tm[fs], tm2[fs])

    recs_v2 = list(zip(T, t_out, dew, load, lux))[val_start:split]
    recs_cm = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm2))[val_start:split]
    val_days = days_of(dates, val_start, split)
    obs = [r[0] for r in recs_v2]
    pv2 = run_protocol(m2, recs_v2[0][0], recs_v2, val_days)
    pcm = run_protocol_cmri(mc, (recs_cm[0][0], recs_cm[0][1]), recs_cm, val_days)
    v2rows = [(c, o, d) for c, o, d in zip(pv2, obs, dates[val_start:split])
              if c is not None]
    cmrows = [(c, o, d) for c, o, d in zip(pcm, obs, dates[val_start:split])
              if c is not None]
    ev2 = [o - c for c, o, _ in v2rows]
    ecm = [o - c for c, o, _ in cmrows]
    dv2 = [d for _, _, d in v2rows]
    assert len(ev2) == len(ecm) == 2743 and dv2 == [d for _, _, d in cmrows]

    w = autocorr_block(ev2, ecm, dv2)
    d_intra = [abs(a) - abs(c) for a, c in zip(ev2, ecm)]
    d_days = [day_key_of(d) for d in dv2]
    z_n, z_b, infl = z_inflation(d_intra, d_days)
    dm576, p576 = dm_hac(d_intra, 576)
    dm1440, p1440 = dm_hac(d_intra, 1440)
    frac_wd = (len(ev2) - 1 - w["ngaps"]) / (len(ev2) - 1) if len(ev2) > 1 else 0.0

    print(f"leg A  within-train val block (fresh fits): n={len(ev2)} steps "
          f"across {w['ndays']} days; consecutive-pair row count {len(ev2) - 1}, "
          f"of which within-day {100 * frac_wd:.1f}% and cross-day {w['ngaps']} "
          f"({100 * (1 - frac_wd):.1f}%)")
    print(f"  full-series lag-1 rho: v2 {w['full'][0]:.4f}  "
          f"cmri {w['full'][1]:.4f}")
    print(f"  within-day median lag-1 rho: v2 {w['wd'][0]:.4f}  "
          f"cmri {w['wd'][1]:.4f}  (per-day med {w['ndays']} days; the 10-min "
          f"scale correlation lives inside days)")
    print(f"  cross-day boundary-pair rho: v2 {w['boundary'][0]:.4f}  "
          f"cmri {w['boundary'][1]:.4f}  (n={w['ngaps']} day-gap pairs — the "
          f"day boundary resets the correlation)")
    print(f"  naive per-step z {z_n:+.2f} vs whole-day-block z {z_b:+.2f} — "
          f"inflation {infl:.1f}x (the iid tests overstate significance by the "
          f"within-day correlation)")
    print(f"  fresh HAC DM: bw=576 stat {dm576:.2f} p {p576:.2e}; "
          f"bw=1440 stat {dm1440:.2f} p {p1440:.2e} — the honest test "
          f"discounts the naive overstatement")
    rmse_v2 = metrics([c for c, _, _ in v2rows], [o for _, o, _ in v2rows])["rmse"]
    rmse_cm = metrics([c for c, _, _ in cmrows], [o for _, o, _ in cmrows])["rmse"]
    print(f"  context: val-block resync RMSE v2 {rmse_v2:.4f} / cmri "
          f"{rmse_cm:.4f}")

    # ---- leg B: frozen §7 item 1 digit audit (read-only) --------------------
    rows = []
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs_f = float(r["T1_observed"])
            rows.append((dt_parse(r["date"]),
                         obs_f - float(r["v2_resync"]),
                         obs_f - float(r["cmri_resync"])))
    fev2 = [e for _, e, _ in rows]
    fecm = [e for _, _, e in rows]
    fds = [d for d, _, _ in rows]
    fw = autocorr_block(fev2, fecm, fds)
    fd = [abs(a) - abs(c) for a, c in zip(fev2, fecm)]
    fd_days = [day_key_of(d) for d in fds]
    fz_n, fz_b, finfl = z_inflation(fd, fd_days)
    fdm576, fp576 = dm_hac(fd, 576)
    fdm1440, fp1440 = dm_hac(fd, 1440)

    audit_ok = True
    # A helper to round a frozen re-aggregation to the committed 3dp digit.
    def audit_r3(got, comm, label):
        nonlocal audit_ok
        ok = abs(got - comm) < TOL3
        audit_ok &= ok
        print(f"    {label} {got:+.4f} (committed {comm:+.3f}, 3dp): "
              f"{'PASS' if ok else 'FAIL'}")
        return ok

    print(f"\nleg B  frozen held-out (audit leg): n={len(rows)} steps "
          f"(committed {C_N})")
    print("  committed-digit audit (frozen artifact, read-only):")
    for (cgot, ccomm, clab, ctl) in ((fw["full"][0], C_FULL[0], "lag-1 rho v2", TOL3),
                                     (fw["full"][1], C_FULL[1], "lag-1 rho cmri", TOL3)):
        ok = abs(cgot - ccomm) < ctl
        audit_ok &= ok
        print(f"    {clab} {cgot:.4f} (committed {ccomm:.3f}): "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"      full-series n={len(rows)} rho {fw['full'][0]:.4f} / "
          f"{fw['full'][1]:.4f} — a 10-min residual at 20:00 explains "
          f"{100 * fw['full'][0] ** 2:.0f}% / {100 * fw['full'][1] ** 2:.0f}% "
          f"of the variance of the one at 20:10")

    def audit_z(got, comm, lab):
        nonlocal audit_ok
        ok = abs(got - comm) < Z_TOL
        audit_ok &= ok
        print(f"    {lab} {got:+.2f} (committed {comm:+.2f}): "
              f"{'PASS' if ok else 'FAIL'}")
        return ok

    audit_z(fz_n, C_NAIVE_Z, "naive per-step z")
    audit_z(fz_b, C_BLOCK_Z, "whole-day-block z")
    ok_i = abs(finfl - C_INFL) < INFL_TOL
    audit_ok &= ok_i
    print(f"    inflation {finfl:.1f}x (committed {C_INFL:.1f}x): "
          f"{'PASS' if ok_i else 'FAIL'}")

    def audit_hac(got, pgot, cstat, cp, lab):
        nonlocal audit_ok
        ok = abs(got - cstat) < T_TOL and abs(pgot - cp) < P_TOL
        audit_ok &= ok
        print(f"    {lab} stat {got:.2f} p {pgot:.2e} (committed "
              f"{cstat:.2f} / {cp:.2e}): {'PASS' if ok else 'FAIL'}")
        return ok

    audit_hac(fdm576, fp576, C_HAC576[0], C_HAC576[1], "HAC DM bw=576")
    audit_hac(fdm1440, fp1440, C_HAC1440[0], C_HAC1440[1], "HAC DM bw=1440")

    # derived frozen cross-checks (3dp, informational audits)
    for (cgot, ccomm, clab) in ((fw["wd"][0], C_WD_MED[0], "within-day median rho v2"),
                                (fw["wd"][1], C_WD_MED[1], "within-day median rho cmri"),
                                (fw["boundary"][0], C_BOUNDARY[0], "boundary rho v2"),
                                (fw["boundary"][1], C_BOUNDARY[1], "boundary rho cmri")):
        audit_r3(cgot, ccomm, clab)
    ok_g = fw["ngaps"] == C_GAPS
    audit_ok &= ok_g
    print(f"    cross-day gaps {fw['ngaps']} (committed {C_GAPS}): "
          f"{'PASS' if ok_g else 'FAIL'}")

    # ---- structural gates (the within-train leg is the real test) -----------
    g1 = w["full"][0] > 0.95 and w["full"][1] > 0.95
    g2 = w["wd"][0] > 0.9 and w["wd"][1] > 0.9
    g3 = (w["boundary"][0] < 0.3 and w["boundary"][1] < 0.3
          and (w["wd"][0] - w["boundary"][0]) > 0.5
          and (w["wd"][1] - w["boundary"][1]) > 0.5)
    g4 = infl >= 4.0
    g5 = abs(dm1440) < 0.5 * abs(z_n) if z_n == z_n else False

    ok = all([g1, g2, g3, g4, g5]) and audit_ok

    print()
    print("gates (within-train leg = fresh test; held-out leg = frozen audit):")
    print(f"    both models' full-series lag-1 rho > 0.95 (not iid): "
          f"{w['full'][0]:.4f} / {w['full'][1]:.4f} : "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"    within-day median lag-1 rho > 0.9 both (10-min scale "
          f"correlation lives inside days): {w['wd'][0]:.4f} / {w['wd'][1]:.4f} : "
          f"{'PASS' if g2 else 'FAIL'}")
    print(f"    boundary rho < 0.3 and within-day minus boundary > 0.5 (day "
          f"boundary resets the correlation; day = honest independent unit): "
          f"v2 {w['boundary'][0]:+.4f} (gap {w['wd'][0] - w['boundary'][0]:.4f}), "
          f"cmri {w['boundary'][1]:+.4f} (gap "
          f"{w['wd'][1] - w['boundary'][1]:.4f}) : "
          f"{'PASS' if g3 else 'FAIL'}")
    print(f"    naive z >= 4x day-block z (iid tests overstate significance): "
          f"{z_n:.2f} vs {z_b:.2f} = {infl:.1f}x : {'PASS' if g4 else 'FAIL'}")
    print(f"    HAC DM bw=1440 |stat| < naive z / 2 (the honest test "
          f"discounts the overstatement): |{dm1440:.2f}| < {abs(z_n) / 2:.2f} : "
          f"{'PASS' if g5 else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §7 item 1 reading "
          f"(the resync residuals are not iid — lag-1 rho 0.987/0.990, the "
          f"smallest-order reason the naive iid tests of §5.5 overstate "
          f"significance, and why the HAC Diebold-Mariano is the honest test) "
          f"is a structural fact, not a single held-out draw: on fresh "
          f"within-train fits the same autocorrelation geometry reproduces — "
          f"both models' full-series lag-1 rho stays above 0.95 "
          f"({w['full'][0]:.4f} / {w['full'][1]:.4f}), the correlation lives "
          f"INSIDE days at the 10-min scale (per-day median {w['wd'][0]:.4f} / "
          f"{w['wd'][1]:.4f} vs near-zero cross-day boundary rho "
          f"{w['boundary'][0]:+.4f} / {w['boundary'][1]:+.4f} between the "
          f"{w['ngaps']} day gaps — the calendar day is the honest independent "
          f"unit), the naive per-step z beats the whole-day-block z by "
          f"{infl:.1f}x ({z_n:.2f} vs {z_b:.2f}), and the fresh HAC DM "
          f"bw=1440 stat |{dm1440:.2f}| is a fraction of the naive z — and the "
          f"frozen artifact re-produces every committed §7 item 1 digit (lag-1 "
          f"rho {fw['full'][0]:.3f} / {fw['full'][1]:.3f} = {C_FULL[0]}/"
          f"{C_FULL[1]}, naive z {fz_n:.2f} vs day-block z {fz_b:.2f} "
          f"inflation {finfl:.1f}x, HAC bw=576 stat {fdm576:.2f} p {fp576:.2e}, "
          f"HAC bw=1440 stat {fdm1440:.2f} p {fp1440:.2e} = the committed "
          f"3.5e-03).")


def dt_parse(s: str):
    """Parse the frozen CSV date string into a datetime (verified format)."""
    import datetime as _dt
    return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
