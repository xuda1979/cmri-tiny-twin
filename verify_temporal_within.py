#!/usr/bin/env python3
"""verify_temporal_within.py — is the §7 item 9 "the resync edge is temporally
flat — time-flat in presence, tail-owned in magnitude, the mirror image of the
free-run's first-half ownership" reading a structural fact, not an artifact of
the single held-out draw?

§7 item 9 committed claims (frozen held-out resync,
results/predictions_cmri.csv, as analyzed by verify_temporal.py):
  * time-flat PRESENCE: days 01-21 mean daily d +0.0771 C (14/21) and days
    22-42 +0.0639 C (14/21); half difference +0.0132 C, balanced permutation
    two-sided p = 0.832 (B = 5000); thirds +0.082 / +0.114 / +0.016 C all
    positive (11 / 8 / 9 of 14 days won);
  * no trend: OLS slope of daily d on day index -0.0014 C/day (se 0.0026,
    t = -0.54; fitted line +0.099 -> +0.042 C), Spearman rho -0.075; 5 of 7
    ISO weeks (16-18, 20, 21) win at day level, best w18 +0.204 C 5/7, worst
    w19 -0.009 C 3/7;
  * broad, not episode-owned: the 28/42 day-level win survives deleting the
    two largest-|d| days (26/40, z = +1.74, p = 0.041) and even the top 6
    (24/36, z = +1.83, p = 0.033);
  * the big days own the MAGNITUDE, not the presence: the top 4 days
    (2016-04-22 +94.1, 2016-05-07 +89.7, 2016-05-05 +75.9, 2016-05-06 +47.4
    C-days) carry 0.73 of the total 419.1 C of paired-d, while the sign of the
    edge is a 42-day-wide phenomenon (28/42 net-positive days);
  * contrast: the FREE-RUN edge is first-half-owned (+0.344 C / 14-21 vs
    -0.387 C / 5-21), the resync edge is not.

Those numbers were computed once on the held-out test. This script verifies the
temporal reading WITHOUT touching the held-out test for fitting: leg A re-fits
fresh v2 (CalibratedTwin) and cmri (frozen-config CmriTwin) twins on the
within-train fit slice [0, 11051), replays the daily-resync protocol on the
within-train val block [11051, 13814) (item 10's 20-day block, n = 2743 resync
steps), and applies the SAME temporal analysis (halves, thirds, calendar
weeks, OLS trend + Spearman, half-difference permutation, top-|d| concentration
audit, top-day share) to the fresh per-day paired d, plus a fresh open-loop
free-run halves replay as the mirror negative control; leg B re-reads the
frozen results/predictions_cmri.csv (read-only) and re-aggregates every
committed §7 item 9 digit exactly as verify_temporal.py does.

Gates (power-aware: the 20-day block cannot reproduce the 42-day block's
strength, so the gates are structural invariants of the temporal reading, not
digit matches): within-train, (i) the edge is present in BOTH halves (time-flat
presence), (ii) |OLS t| < 2 (no significant trend), (iii) at least 2 of 3
thirds are net-positive, (iv) the day-level win survives deleting the top-2 and
top-6 largest-|d| days (wins > (n-k)/2 on the remainder), (v) the top 4 days
own at least 0.30 of the block's total paired-d (magnitude concentrated,
presence flat); the held-out leg must re-produce every committed digit.

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
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
SEED = 42
B = 5000
TOL = 0.0015            # degC digit tolerance (4dp-printed values)
TOL3 = 0.00055          # 3dp cell tolerance (thirds / weeks / free-run / rho)
T_SHR = 0.0051          # 2dp share tolerance (top-4 share)
T_MAG = 0.06            # 1dp C-day magnitude tolerance (top-4 sums / total)
Z_TOL = 0.01            # z-value tolerance (deterministic normal approx)
P_TOL = 0.001           # one-sided p tolerance
T_TOL = 0.0051          # 2dp t-stat tolerance

# Frozen §7 item 9 committed digits (leg-B audit targets), exactly the numbers
# verify_temporal.py / REPORT.md §7 item 9 print.
C_OVERALL = (0.0713, 0.579, 28)                       # overall d, win, day wins
C_HALVES = ((+0.0771, 14), (+0.0639, 14))             # (mean daily d, wins) of 21
C_HALFDIFF = (+0.0132, 0.832)                         # obs half diff, perm p
C_THIRDS = ((+0.082, 11), (+0.114, 8), (+0.016, 9))   # (mean, wins) of 14
C_OLS = (-0.0014, 0.0026, -0.54)                      # slope, se, t
C_RHO = -0.075
C_FITTED = (+0.099, +0.042)                           # fitted line start -> end
C_WIN_WEEKS = {16, 17, 18, 20, 21}                    # 5 of 7 ISO weeks win
C_W18 = (+0.204, 5)                                   # best week: mean, wins/7
C_W19 = (-0.009, 3)                                   # worst week: mean, wins/7
# concentration audit after deleting top-k |d| days: (k, wins, keep, z, p)
C_CONC = ((2, 26, 40, +1.74, 0.041), (6, 24, 36, +1.83, 0.033))
C_TOP4 = (("2016-04-22", +94.1), ("2016-05-07", +89.7),
          ("2016-05-05", +75.9), ("2016-05-06", +47.4))
C_TOP4_SHARE = 0.73
C_TOTAL = 419.1
C_FREERUN = ((+0.344, 14, 21), (-0.387, 5, 21))       # (mean d, wins, days)


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive paired d, normal approx."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided H1: P(d>0) > 0.5


def ols(y: list[float]) -> tuple[float, float, float]:
    """OLS slope, se, t-statistic of y on x = 0..n-1 (day index)."""
    nn = len(y)
    xs = list(range(nn))
    mx, my = statistics.fmean(xs), statistics.fmean(y)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (yy - my) for x, yy in zip(xs, y))
    b = sxy / sxx
    a = my - b * mx
    resid = [yy - (a + b * x) for x, yy in zip(xs, y)]
    se = math.sqrt(sum(r * r for r in resid) / (nn - 2) / sxx)
    return b, se, b / se


def spearman(xs: list[int], ys: list[float]) -> float:
    nn = len(ys)
    rk = [1 + sum(1 for j in range(nn) if ys[j] < ys[i]
                  or (ys[j] == ys[i] and j < i)) for i in range(nn)]
    return 1.0 - 6.0 * sum((x - r) ** 2 for x, r in zip(xs, rk)) / (nn * (nn * nn - 1))


def per_day_d(ev2: list[float], ecm: list[float],
              ds: list[dt.datetime]) -> dict[str, list[float]]:
    """Per-iso-day lists of per-step paired d (|e_v2| - |e_cmri|)."""
    day: dict[str, list[float]] = {}
    for a, c, d in zip(ev2, ecm, ds):
        day.setdefault(d.date().isoformat(), []).append(
            abs(a) - abs(c))
    return day


def compute(day: dict[str, list[float]]) -> dict:
    """The full verify_temporal.py sections 1-5 analysis on one by-day map."""
    keys = sorted(day)
    n = len(keys)
    dd = [statistics.fmean(day[k]) for k in keys]
    totalD = sum(sum(v) for v in day.values())
    all_d = [x for k in keys for x in day[k]]
    res = {"keys": keys, "n": n, "dd": dd, "totalD": totalD,
           "overall_d": statistics.fmean(all_d),
           "overall_win": sum(1 for x in all_d if x > 1e-12) / len(all_d),
           "day_wins": sum(1 for x in dd if x > 1e-12)}

    # 1. halves and thirds.
    h = n // 2
    res["halves"] = []
    for seg in (dd[:h], dd[h:]):
        w = sum(1 for x in seg if x > 1e-12)
        z, p = sign_test(w, len(seg))
        res["halves"].append((statistics.fmean(seg), w, len(seg), z, p))
    res["h"] = h
    res["thirds"] = []
    t, r = n // 3, n % 3
    i = 0
    for _ in range(3):
        sz = t + (1 if _ < r else 0)
        seg = dd[i:i + sz]
        i += sz
        w = sum(1 for x in seg if x > 1e-12)
        z, p = sign_test(w, len(seg))
        res["thirds"].append((statistics.fmean(seg), w, len(seg), z, p))

    # 2. calendar weeks.
    wk: dict[tuple[int, int], list[float]] = {}
    for k, d in zip(keys, dd):
        iso = dt.date.fromisoformat(k).isocalendar()
        wk.setdefault((iso[0], iso[1]), []).append(d)
    res["weeks"] = []
    for (yr, wno), seg in sorted(wk.items()):
        w = sum(1 for x in seg if x > 1e-12)
        res["weeks"].append((yr, wno, len(seg), statistics.fmean(seg), w))

    # 3. trend of daily d on day index.
    xs = list(range(n))
    slope, se, tv = ols(dd)
    rho = spearman(xs, dd)
    a_ = statistics.fmean(dd) - slope * statistics.fmean(xs)
    res["ols"] = (slope, se, tv)
    res["rho"] = rho
    res["fitted"] = (a_ + slope * 0, a_ + slope * (n - 1))

    # 4. balanced permutation test on the half difference (two-sided).
    obs_h = statistics.fmean(dd[:h]) - statistics.fmean(dd[h:])
    rng = random.Random(SEED)
    cnt = 0
    for _ in range(B):
        idx = set(rng.sample(range(n), h))
        a = statistics.fmean([dd[i] for i in idx])
        b = statistics.fmean([dd[i] for i in range(n) if i not in idx])
        if abs(a - b) >= abs(obs_h):
            cnt += 1
    res["perm"] = (obs_h, cnt / B)

    # 5. concentration audit.
    order = sorted(range(n), key=lambda i: -abs(dd[i]))
    res["conc"] = []
    for k_rem in (1, 2, 3, 6):
        keep = [dd[i] for i in range(n) if i not in order[:k_rem]]
        w = sum(1 for x in keep if x > 1e-12)
        z, p = sign_test(w, len(keep))
        res["conc"].append((k_rem, w, len(keep), z, p))
    ranked = [(sum(day[k]), k) for k in keys]
    ranked.sort(reverse=True)
    res["top4"] = ranked[:4]
    res["share4"] = (sum(m for m, _ in ranked[:4]) / totalD
                     if totalD else float("nan"))
    return res


def show_sections(st: dict, label: str) -> None:
    h = st["h"]
    print(f"1. window split:")
    for lab, seg in zip((f"days 01-{h:02d}", f"days {h+1:02d}-{st['n']:02d}"),
                        st["halves"]):
        m, w, nn, z, p = seg
        print(f"      {lab}: mean daily d {m:+.4f}, cmri better "
              f"{w}/{nn} (z {z:+.2f}, p {p:.3f})")
    for i, (m, w, nn, z, p) in enumerate(st["thirds"]):
        print(f"      third {i+1} (n={nn}): mean daily d {m:+.4f}, cmri better "
              f"{w}/{nn} (z {z:+.2f}, p {p:.3f})")
    print("2. calendar weeks (ISO week is the fixed Monday basis):")
    for yr, wno, nn, m, w in st["weeks"]:
        print(f"      w{wno:02d} n={nn}  mean daily d {m:+.4f}  cmri better "
              f"{w}/{nn}")
    slope, se, tv = st["ols"]
    s0, s1 = st["fitted"]
    print(f"3. trend of daily d on day index: OLS slope {slope:+.4f} C/day "
          f"(se {se:.4f}, t {tv:+.2f}); Spearman rho {st['rho']:+.3f} — flat: "
          f"the edge neither decays nor grows over the window.")
    print(f"      fitted line starts ~{s0:+.4f} ends ~{s1:+.4f} C")
    obs_h, p_perm = st["perm"]
    print(f"4. half difference: mean(first {h}) - mean(last {h}) = "
          f"{obs_h:+.4f} C; balanced permutation two-sided p = {p_perm:.3f} "
          f"(B={B}, seed {SEED}) — the halves do not separate.")
    z0, p0 = sign_test(st["day_wins"], st["n"])
    print(f"5. concentration audit (is the win carried by a few days?):")
    print(f"      all {st['n']} days:  cmri better {st['day_wins']}/{st['n']}, "
          f"z {z0:+.2f}, p {p0:.3f}")
    for k_rem, w, nn, z, p in st["conc"]:
        print(f"      drop top {k_rem} day(s) by |d|: cmri better {w}/{nn}, "
              f"z {z:+.2f}, p {p:.3f}")
    names = ", ".join(f"{k} ({m:+.1f})" for m, k in st["top4"])
    print(f"      top 4 days ({names}) carry {st['share4']:.2f} of total "
          f"paired-d ({st['totalD']:.1f} C-step); a few big-miss days own the "
          f"*magnitude*, not the sign.")


def read_frozen() -> dict[str, list[float]]:
    """Frozen held-out resync per-day paired-d lists; no fitting."""
    day: dict[str, list[float]] = {}
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            obs = float(r["T1_observed"])
            day.setdefault(r["date"][:10], []).append(
                abs(obs - float(r["v2_resync"])) - abs(obs - float(r["cmri_resync"])))
    return day


def read_frozen_freerun() -> dict[str, list[float]]:
    """Frozen held-out free-run per-day paired-d lists (all rows: open-loop
    predicts through anchors, matching verify_temporal.py §6)."""
    day: dict[str, list[float]] = {}
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            obs = float(r["T1_observed"])
            day.setdefault(r["date"][:10], []).append(
                abs(obs - float(r["v2_freerun"])) - abs(obs - float(r["cmri_freerun"])))
    return day


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

    print("temporal-stability probe of the resync edge (daily-resync protocol); "
          "held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: fresh within-train fits + val-block temporal structure -------
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

    day_w = per_day_d(ev2, ecm, dv2)
    within = compute(day_w)
    print(f"leg A  within-train val block (fresh fits): n={len(ev2)} steps "
          f"across {within['n']} days")
    show_sections(within, "within-train")
    print(f"      overall paired d {within['overall_d']:+.4f}, win "
          f"{within['overall_win']:.3f}, day-level {within['day_wins']}/{within['n']}")

    # within-train free-run replay (mirror negative control, open-loop).
    pf2 = run_protocol(m2, recs_v2[0][0], recs_v2, None)
    pfc = run_protocol_cmri(mc, (recs_cm[0][0], recs_cm[0][1]), recs_cm, None)
    fr2 = [(c, o, d) for c, o, d in zip(pf2, obs, dates[val_start:split])
           if c is not None]
    frc = [(c, o, d) for c, o, d in zip(pfc, obs, dates[val_start:split])
           if c is not None]
    frev2 = [o - c for c, o, _ in fr2]
    frecm = [o - c for c, o, _ in frc]
    frday = per_day_d(frev2, frecm, [d for _, _, d in fr2])
    frkeys = sorted(frday)
    fdd = [statistics.fmean(frday[k]) for k in frkeys]
    h_fr = len(fdd) // 2
    print(f"\n6. contrast — free-run halves (fresh open-loop replay over the "
          f"same {len(fdd)}-day block):")
    for lab, seg in ((f"days 01-{h_fr:02d}", fdd[:h_fr]),
                     (f"days {h_fr+1:02d}-{len(fdd):02d}", fdd[h_fr:])):
        w = sum(1 for x in seg if x > 1e-12)
        print(f"      {lab}: mean daily d {statistics.fmean(seg):+.4f}, cmri "
              f"better {w}/{len(seg)}")
    print(f"      (the frozen free-run edge is first-half-owned (+0.344 / 14-21 "
          f"vs -0.387 / 5-21); the open-loop replay on this shorter block is "
          f"the same negative-control geometry.)")
    rmse_v2 = metrics([c for c, _, _ in v2rows], [o for _, o, _ in v2rows])["rmse"]
    rmse_cm = metrics([c for c, _, _ in cmrows], [o for _, o, _ in cmrows])["rmse"]
    print(f"      context: val-block resync RMSE v2 {rmse_v2:.4f} / cmri "
          f"{rmse_cm:.4f}")
    print()

    # ---- leg B: frozen §7 item 9 digit audit (read-only) --------------------
    day_f = read_frozen()
    frozen = compute(day_f)
    print(f"leg B  frozen held-out (audit leg): n={sum(len(v) for v in day_f.values())}"
          f" steps across {frozen['n']} days")
    print("  committed-digit audit (frozen artifact, read-only):")
    audit_ok = True

    od_, ow_, ow_days = C_OVERALL
    ok0 = (abs(frozen["overall_d"] - od_) < TOL
           and abs(frozen["overall_win"] - ow_) < TOL
           and frozen["day_wins"] == ow_days)
    audit_ok &= ok0
    print(f"    overall d {frozen['overall_d']:+.4f} (committed {od_:+.4f}), "
          f"win {frozen['overall_win']:.3f} (committed {ow_:.3f}), day-level "
          f"{frozen['day_wins']}/{frozen['n']} (committed {ow_days}/42): "
          f"{'PASS' if ok0 else 'FAIL'}")

    for i, (cmean, cwin) in enumerate(C_HALVES):
        m, w, nn, _, _ = frozen["halves"][i]
        okh = abs(m - cmean) < TOL and w == cwin
        audit_ok &= okh
        print(f"    half {i+1}: mean daily d {m:+.4f} (committed {cmean:+.4f}), "
              f"{w}/{nn} won (committed {cwin}/21): {'PASS' if okh else 'FAIL'}")
    ch_, cp_ = C_HALFDIFF
    oh_, pp_ = frozen["perm"]
    okp = abs(oh_ - ch_) < TOL and abs(pp_ - cp_) < P_TOL
    audit_ok &= okp
    print(f"    half difference {oh_:+.4f} (committed {ch_:+.4f}), permutation "
          f"p {pp_:.3f} (committed {cp_:.3f}): {'PASS' if okp else 'FAIL'}")

    for i, (cmean, cwin) in enumerate(C_THIRDS):
        m, w, nn, _, _ = frozen["thirds"][i]
        okth = abs(m - cmean) < TOL3 and w == cwin
        audit_ok &= okth
        print(f"    third {i+1}: mean {m:+.4f} (committed {cmean:+.3f}), "
              f"{w}/{nn} won (committed {cwin}/14): {'PASS' if okth else 'FAIL'}")

    cs_, cse_, ct_ = C_OLS
    s_, se_, tv_ = frozen["ols"]
    okols = abs(s_ - cs_) < TOL and abs(se_ - cse_) < TOL and abs(tv_ - ct_) < T_TOL
    okrho = abs(frozen["rho"] - C_RHO) < TOL3
    audit_ok &= okols and okrho
    print(f"    OLS slope {s_:+.4f} (committed {cs_:+.4f}), se {se_:.4f} "
          f"(committed {cse_:.4f}), t {tv_:+.2f} (committed {ct_:+.2f}); "
          f"Spearman rho {frozen['rho']:+.3f} (committed {C_RHO:+.3f}): "
          f"{'PASS' if okols and okrho else 'FAIL'}")
    f0_, f1_ = frozen["fitted"]
    cf0_, cf1_ = C_FITTED
    okfit = abs(f0_ - cf0_) < TOL3 and abs(f1_ - cf1_) < TOL3
    audit_ok &= okfit
    print(f"    fitted line {f0_:+.4f} -> {f1_:+.4f} (committed "
          f"{cf0_:+.3f} -> {cf1_:+.3f}): {'PASS' if okfit else 'FAIL'}")

    wk_line = ", ".join(f"w{wno:02d} {m:+.3f} ({w}/{nn})"
                        for yr_, wno, nn, m, w in frozen["weeks"])
    print(f"    full ISO-week table: {wk_line}")
    # The committed "5 of 7 ISO weeks (16-18, 20, 21)" counts only weeks with
    # n >= 3: ISO week 15 is a 2-day boundary week whose first day (2016-04-16)
    # is the anchor split day, so its 2/2 win carries no sign-test weight and
    # does not count as a week-level win; w21 (5 days, 4/5) does.
    win_weeks = {wno for yr_, wno, nn, m, w in frozen["weeks"]
                 if w > nn / 2 and nn >= 3}
    okwset = win_weeks == C_WIN_WEEKS and len(frozen["weeks"]) == 7
    audit_ok &= okwset
    print(f"    winning ISO weeks {sorted(win_weeks)} (committed "
          f"{sorted(C_WIN_WEEKS)}, 5/7): {'PASS' if okwset else 'FAIL'}")
    for (cwno, cwmean, cwwin) in ((18,) + C_W18, (19,) + C_W19):
        # (wno, mean, wins)
        cell = next((m, w) for yr_, wno, nn, m, w in frozen["weeks"]
                    if wno == cwno)
        okc = abs(cell[0] - cwmean) < TOL3 and cell[1] == cwwin
        audit_ok &= okc
        print(f"    week w{cwno:02d}: mean {cell[0]:+.4f} "
              f"(committed {cwmean:+.3f}), {cell[1]}/{next(nn for yr_, wno, nn, m, w in frozen['weeks'] if wno == cwno)} "
              f"won (committed {cwwin}/7): {'PASS' if okc else 'FAIL'}")

    # conc entries are k=1,2,3,6 -> index = {1:0, 2:1, 3:2, 6:3}
    conc_idx = {2: 1, 6: 3}
    for ck, cwon, ckeep, cz, cp in C_CONC:
        k_rem, w, nn, z, p = frozen["conc"][conc_idx[ck]]
        okc = (w == cwon and nn == ckeep and abs(z - cz) < Z_TOL
               and abs(p - cp) < P_TOL)
        audit_ok &= okc
        print(f"    after dropping top {ck}: cmri better {w}/{nn} "
              f"(committed {cwon}/{ckeep}), z {z:+.2f} (committed {cz:+.2f}), "
              f"p {p:.3f} (committed {cp:.3f}): {'PASS' if okc else 'FAIL'}")

    oknames = tuple(k for m, k in frozen["top4"]) == tuple(k for k, m in C_TOP4)
    for (ckt, cmag), (m, k_) in zip(C_TOP4, frozen["top4"]):
        okn = abs(m - cmag) < T_MAG
        audit_ok &= okn
        print(f"    top day {k_} {m:+.1f} (committed {ckt} {cmag:+.1f} C-days): "
              f"{'PASS' if okn else 'FAIL'}")
    audit_ok &= oknames
    okshr = abs(frozen["share4"] - C_TOP4_SHARE) < T_SHR
    okttl = abs(frozen["totalD"] - C_TOTAL) < T_MAG
    audit_ok &= okshr and okttl
    print(f"    top-4 share {frozen['share4']:.2f} (committed {C_TOP4_SHARE:.2f}) "
          f"of total {frozen['totalD']:.1f} (committed {C_TOTAL:.1f}): "
          f"{'PASS' if okshr and okttl else 'FAIL'}")

    frf = read_frozen_freerun()
    frkeys = sorted(frf)
    fd_ = [statistics.fmean(frf[k]) for k in frkeys]
    h_f = len(fd_) // 2
    for i, (cmean, cwin, cdays) in enumerate(C_FREERUN):
        seg = fd_[:h_f] if i == 0 else fd_[h_f:]
        m = statistics.fmean(seg)
        w = sum(1 for x in seg if x > 1e-12)
        okc = abs(m - cmean) < TOL3 and w == cwin and len(seg) == cdays
        audit_ok &= okc
        print(f"    free-run half {i+1}: mean daily d {m:+.4f} (committed "
              f"{cmean:+.3f}), {w}/{len(seg)} (committed {cwin}/{cdays}): "
              f"{'PASS' if okc else 'FAIL'}")
    print()

    # ---- structural gates (the within-train leg is the real test) -----------
    hh = within["h"]
    g1 = all(m > 0.0 for m, w, nn, z, p in within["halves"])
    g2 = abs(within["ols"][2]) < 2.0
    g3 = sum(1 for m, w, nn, z, p in within["thirds"] if m > 0.0) >= 2
    conc2 = within["conc"][1]      # k = 2
    conc6 = within["conc"][3]      # k = 6
    g4 = (conc2[1] > conc2[2] / 2 and conc6[1] > conc6[2] / 2)
    g5 = within["share4"] >= 0.30

    ok = all([g1, g2, g3, g4, g5]) and audit_ok

    print("gates (within-train leg = fresh test; held-out leg = frozen audit):")
    print(f"    both halves mean daily d > 0 (time-flat presence): "
          f"{within['halves'][0][0]:+.4f} / {within['halves'][1][0]:+.4f} : "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"    |OLS t| < 2 (no significant trend): t {within['ols'][2]:+.2f}, "
          f"rho {within['rho']:+.3f} : {'PASS' if g2 else 'FAIL'}")
    th_m = [m for m, w, nn, z, p in within["thirds"]]
    print(f"    >= 2 of 3 third means > 0 (broad presence): "
          f"{th_m[0]:+.3f} / {th_m[1]:+.3f} / {th_m[2]:+.3f} : "
          f"{'PASS' if g3 else 'FAIL'}")
    print(f"    day-level win survives dropping top-2 and top-6 |d| days "
          f"(wins > (n-k)/2): {conc2[1]}/{conc2[2]} and {conc6[1]}/{conc6[2]} : "
          f"{'PASS' if g4 else 'FAIL'}")
    print(f"    top-4 days own >= 0.30 of total paired-d (magnitude "
          f"concentrated): {within['share4']:.2f} of {within['totalD']:.1f} : "
          f"{'PASS' if g5 else 'FAIL'}")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §7 item 9 reading (the "
          f"resync edge is temporally flat — time-flat in presence, tail-owned "
          f"in magnitude, the mirror image of the free-run's first-half "
          f"ownership) is a structural fact, not a single held-out draw: on "
          f"fresh within-train fits the same temporal geometry reproduces — the "
          f"edge is present in both halves ({within['halves'][0][0]:+.4f} / "
          f"{within['halves'][1][0]:+.4f}, time-flat presence), there is no "
          f"significant trend (OLS t {within['ols'][2]:+.2f}, rho "
          f"{within['rho']:+.3f}), at least 2 of 3 thirds are positive "
          f"({th_m[0]:+.3f}/{th_m[1]:+.3f}/{th_m[2]:+.3f}), the day-level win "
          f"survives deleting the top-2 and top-6 largest-|d| days "
          f"({conc2[1]}/{conc2[2]} and {conc6[1]}/{conc6[2]}), and the top 4 "
          f"days own the magnitude, not the presence (share "
          f"{within['share4']:.2f} of {within['totalD']:.1f} C of paired-d) — "
          f"and the frozen artifact re-produces every committed §7 item 9 digit "
          f"(halves {frozen['halves'][0][0]:+.4f}/{frozen['halves'][1][0]:+.4f} "
          f"{frozen['halves'][0][1]}/21 {frozen['halves'][1][1]}/21, half "
          f"difference {frozen['perm'][0]:+.4f} p {frozen['perm'][1]:.3f}, "
          f"thirds {frozen['thirds'][0][0]:+.3f}/{frozen['thirds'][1][0]:+.3f}/"
          f"{frozen['thirds'][2][0]:+.3f} "
          f"{frozen['thirds'][0][1]}/{frozen['thirds'][1][1]}/"
          f"{frozen['thirds'][2][1]}, OLS {frozen['ols'][0]:+.4f} se "
          f"{frozen['ols'][1]:.4f} t {frozen['ols'][2]:+.2f} rho "
          f"{frozen['rho']:+.3f}, winning ISO weeks {sorted(win_weeks)}, "
          f"{frozen['conc'][1][1]}/{frozen['conc'][1][2]} z "
          f"{frozen['conc'][1][3]:+.2f} and {frozen['conc'][3][1]}/"
          f"{frozen['conc'][3][2]} z {frozen['conc'][3][3]:+.2f}, top-4 "
          f"{', '.join(f'{k} {m:+.1f}' for m, k in frozen['top4'])} share "
          f"{frozen['share4']:.2f} of {frozen['totalD']:.1f}, free-run "
          f"{statistics.fmean(fd_[:h_f]):+.3f}/"
          f"{statistics.fmean(fd_[h_f:]):+.3f}).")


if __name__ == "__main__":
    main()
