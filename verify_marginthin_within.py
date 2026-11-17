#!/usr/bin/env python3
"""verify_marginthin_within.py — is the §5.5 MARGIN-THINNING reading ("the
margin is thinner than the single-fabric config: mean d +0.1000 → +0.0713 °C,
win rate 0.676 → 0.579 — the honest price of the free-run fix") a structural
fact about the two cmri driver selections, or an artifact of the single
held-out draw?

§5.5 committed claim (two-timescale frozen held-out resync,
results/predictions_cmri.csv, as re-scored by verify_significance.py):
  * mean d = +0.0713 °C over n = 5879 resync steps, win rate 0.579
    (3406/2456/17 win/loss/tie), sign z 12.16 p < 1e-16, paired t 14.77
    p < 1e-16, DM naive 14.77, DM Newey–West HAC bw=576 (4 days) 2.38
    p 1.74e-02, DM HAC bw=1440 (10 days) 2.92 p 3.46e-03, per-day win
    28/42 days — "the cmri margin is not sample luck";
  * the single-fabric history digits: mean d +0.1000 and win rate 0.676 —
    the pre-slow-fabric config's margin, the reference of the "thinner
    margin" sentence.  The two-timescale selection's margin is the honest
    price of the free-run fix of §5.2/§5.6.

Honesty anchor — which digits are frozen-auditable and which are not:
  * results/predictions_cmri.csv (and results/predictions.csv) are BOTH
    post-two-timescale era: NO single-fabric frozen artifact ever existed.
    The single-fabric margin digits (mean d +0.1000, win rate 0.676) are
    fresh-fit-only history numbers from the pre-slowfab §5.5, so their
    reproducible statement is the fresh leg-A replay below, exactly the
    honesty convention items 28/32/33 use for fresh-fit-only digits.
  * What CAN be frozen-audited is the two-timescale §5.5 margin block: mean
    d +0.0713, win rate 0.579, per-day 28/42, sign z 12.16, paired t 14.77,
    DM naive/√n 14.77, DM HAC bw=576 2.38 p 1.74e-02, DM HAC bw=1440 2.92
    p 3.46e-03, re-derived from the frozen CSV (read-only, same convention
    as verify_significance.py's leg B).

This script probes the POOLED MARGIN without touching the held-out test:
leg A re-fits THREE twins fresh on the within-train fit slice [0, 11051) —
v2 (CalibratedTwin, the mean-|err| reference), the superseded single-fabric
CmriTwin (alpha 300, t_set 21, keep=(0,3,4,5,6,7,8), fast fabric t_mass only,
a 9-array fit), and the frozen two-timescale CmriTwin (keep=(0,3,4,5,6,7,8,9),
fast + slow fabric, 10-array fit) — replays all three under the daily-resync
protocol on the within-train val block [11051, 13814) (n = 2743 steps / 20
days), and recomputes the same pooled margin statistics per cmri config
(pooled mean d = mean(|e_v2| − |e_cmri-config|), win rate = P(|e_cmri| <
|e_v2|), the exact verify_significance convention) so the two cmri selections
are compared margin-vs-margin on the SAME fresh block; leg B re-reads the
frozen results/predictions_cmri.csv (read-only) and re-aggregates every
committed two-timescale §5.5 digit (mean d, win rate + counts, sign z,
paired t, DM naive + HAC 576/1440, per-day win 28/42).

Gates (power-aware relative invariants of the MARGIN-THINNING reading; the
20-day block cannot pin the 42-day margin magnitudes, so only same-block
RELATIVE orderings are gated — every d and win-rate comparison is between
the two cmri configs measured on the SAME fresh replays):
  (i)  both cmri configs' pooled mean d > 0 within-train — the free-run fix
       did not erase the resync edge (held: +0.1000 and +0.0713 both > 0;
       "the margin is not sample luck" for both selections);
  (ii) two-timescale pooled mean d < single-fabric pooled mean d within-train
       — the margin is THINNER for the two-timescale selection (held: +0.0713
       < +0.1000, "the honest price of the free-run fix");
  (iii) two-timescale win rate < single-fabric win rate within-train — the
       step-level win share is also thinner (held: 0.579 < 0.676).
Block-dependent strength deltas (documented, not gated): the d and win-rate
LEVELS themselves (a 20-day block vs the 42-day held-out, and within-train
v2's absolute error level shifts both configs' d by the same v2 reference).
The frozen two-timescale §5.5 digits are audited regardless (leg B).

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import statistics
import sys

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, run_protocol  # noqa: E402
from experiment_cmri import (CMRI_ALPHA, CMRI_KEEP, CMRI_T_SET, CMRI_W,  # noqa: E402
                             CMRI_W2, days_of, load_cmri_data,
                             run_protocol_cmri)
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from cmri_twin import CmriTwin, fabric_mass  # noqa: E402

VALID_FRAC = 0.2
PRED = "results/predictions_cmri.csv"

# Superseded single-fabric cmri config (REPORT.md §5.2 line 140 / §6 item 30 /
# §7 item 11 provenance): pre-slow-capacitor driver mask, fast fabric only.
S_ALPHA = 300.0
S_T_SET = 21.0
S_KEEP = (0, 3, 4, 5, 6, 7, 8)

# Frozen two-timescale §5.5 margin digits (leg-B audit, same convention as
# verify_significance.py; re-derived from results/predictions_cmri.csv).
F_POOLED_D = 0.0713        # committed mean d = mean(|e_v2| − |e_cmri|)
F_WIN = 0.579              # committed win rate 3406/2456/17 win/loss/tie
F_WINC = 3406
F_N = 5879
F_DAYS = (28, 42)          # committed per-day win 28/42
F_SIGN_Z = 12.16           # sign test z, one-sided p < 1e-16
F_T_STAT = 14.77           # paired t (normal approx)
F_DM_NAIVE = 14.77         # DM iid approach = t
F_HAC576 = (2.38, 1.74e-02)
F_HAC1440 = (2.92, 3.46e-03)
TOL_D = 0.0001             # mean-d audit: 4 decimal places, exact
TOL_WIN = 0.0005           # win-rate audit: committed 3-decimal digit 0.579
                           # (printed {win/n:.3f}); 3406/5879 = 0.5794 at 4dp
TOL_STATS = 0.01           # z/t/DM statistics: 2 decimals

# History digits (fresh-fit-only, NO frozen artifact — both frozen CSVs are
# post-two-timescale era): the single-fabric margin reference.
B_SINGLE_D = 0.1000        # historical single-fabric mean d
B_SINGLE_WIN = 0.676       # historical single-fabric win rate


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_normal_p(z: float) -> float:
    return 2.0 * (1.0 - phi(abs(z)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive d, normal approximation."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided


def dm_hac(d: list[float], bw: int) -> tuple[float, float]:
    """Diebold-Mariano with Newey-West HAC long-run variance (Bartlett)."""
    n = len(d)
    m = statistics.fmean(d)
    g0 = sum((x - m) ** 2 for x in d) / n
    lrv = g0
    for l in range(1, bw + 1):
        gl = sum((d[i] - m) * (d[i + l] - m) for i in range(n - l)) / n
        w = 1.0 - l / (bw + 1.0)
        lrv += 2.0 * w * gl
    if lrv <= 0.0:
        return math.nan, math.nan
    stat = m / math.sqrt(lrv / n)
    return stat, two_sided_normal_p(stat)


def margin_stats(ev2: list[float], ecm: list[float]) -> dict[str, float]:
    """Pooled margin statistics on one error pair archive (verify_significance
    convention): d_i = |e_v2| − |e_cmri|, positive = cmri better."""
    d = [abs(a) - abs(c) for a, c in zip(ev2, ecm)]
    n = len(d)
    wins = sum(1 for a, c in zip(ev2, ecm) if abs(c) < abs(a))
    losses = sum(1 for a, c in zip(ev2, ecm) if abs(c) > abs(a))
    md = statistics.fmean(d)
    sd = statistics.pstdev(d)
    t_stat = md / sd * math.sqrt(n) if sd > 0 else math.nan
    z_sign, p_sign = sign_test(wins, n)
    return {
        "n": n, "md": md, "win": wins, "loss": losses,
        "win_rate": wins / n, "t": t_stat, "z_sign": z_sign,
        "p_sign": p_sign,
    }


def read_frozen() -> tuple[list[dt.datetime], list[float], list[float]]:
    """Frozen two-timescale held-out resync rows -> (datetimes, e_v2, e_cmri);
    no fitting."""
    ds, ev2, ecm = [], [], []
    with open(PRED, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cmri_resync"] == "" or r["v2_resync"] == "":
                continue  # day-boundary anchor rows, excluded by protocol
            ds.append(dt.datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S"))
            obs = float(r["T1_observed"])
            ev2.append(obs - float(r["v2_resync"]))
            ecm.append(obs - float(r["cmri_resync"]))
    return ds, ev2, ecm


def main() -> None:
    data = load_cmri_data("data/energydata_complete.csv")
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    val_start = int(round((1.0 - VALID_FRAC) * split))
    T, Th, t_out = data["T1"], data["Th"], data["T_out"]
    dew, load, lux = data["Tdewpoint"], data["Appliances"], data["lights"]
    wind, ghi = data["Windspeed"], data["GHI"]
    tm = fabric_mass(t_out, CMRI_W)      # fast fabric (tau = 3 h)
    tm2 = fabric_mass(t_out, CMRI_W2)    # slow fabric (tau2 = 12 h)
    dates = data["date"]

    print("margin-THINNING probe (daily-resync protocol; the §5.5 reading "
          "that the two-timescale cmri selection's resync margin is thinner "
          "than the single-fabric config's — mean d +0.1000 → +0.0713 °C, "
          "win rate 0.676 → 0.579, the honest price of the free-run fix); "
          "held-out test untouched for fitting")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- leg A: three fresh within-train fits + val-block replays ----------
    fs = slice(0, val_start)
    v2 = CalibratedTwin()
    v2.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    single = CmriTwin(alpha=S_ALPHA, t_set=S_T_SET, keep=S_KEEP)  # no tm2
    single.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs],
               wind[fs], ghi[fs], tm[fs])
    two = CmriTwin(alpha=CMRI_ALPHA, t_set=CMRI_T_SET, keep=CMRI_KEEP)
    two.fit(T[fs], Th[fs], t_out[fs], dew[fs], load[fs], lux[fs], wind[fs],
            ghi[fs], tm[fs], tm2[fs])

    recs_v2 = list(zip(T, t_out, dew, load, lux))[val_start:split]
    recs_s = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm))[val_start:split]
    recs_t = list(zip(T, Th, t_out, dew, load, lux, wind, ghi, tm, tm2))[val_start:split]
    val_days = days_of(dates, val_start, split)
    obs = [r[0] for r in recs_v2]
    pv2 = run_protocol(v2, recs_v2[0][0], recs_v2, val_days)
    ps = run_protocol_cmri(single, (recs_s[0][0], recs_s[0][1]), recs_s, val_days)
    pt = run_protocol_cmri(two, (recs_t[0][0], recs_t[0][1]), recs_t, val_days)

    def rows_of(pred):
        return [(c, o) for c, o, d in
                zip(pred, obs, dates[val_start:split]) if c is not None]

    r_v2 = rows_of(pv2)
    r_s = rows_of(ps)
    r_t = rows_of(pt)
    assert len(r_v2) == len(r_s) == len(r_t) == 2743

    def kept_date_keys(pred):
        """Date keys (day anchors) of the kept rows, same filter as rows_of."""
        return [d.date().isoformat()
                for c, d in zip(pred, dates[val_start:split]) if c is not None]

    key_v = kept_date_keys(pv2)
    key_s = kept_date_keys(ps)
    key_t = kept_date_keys(pt)
    assert key_v == key_s == key_t

    e_v2 = [o - c for c, o in r_v2]
    e_s = [o - c for c, o in r_s]
    e_t = [o - c for c, o in r_t]

    m_s = margin_stats(e_v2, e_s)
    m_t = margin_stats(e_v2, e_t)

    def day_win(ev2_, ecm_, keys):
        by_day: dict[str, list[tuple[float, float]]] = {}
        for k, a, c in zip(keys, ev2_, ecm_):
            by_day.setdefault(k, []).append((abs(a), abs(c)))
        return sum(1 for blk in by_day.values()
                   if statistics.fmean(b[1] for b in blk)
                   < statistics.fmean(b[0] for b in blk)), len(by_day)

    dw_s = day_win(e_v2, e_s, key_v)
    dw_t = day_win(e_v2, e_t, key_v)

    def fmt_m(label, m, dw):
        print(f"    {label:<26} mean d {m['md']:+.4f} °C  win rate "
              f"{m['win_rate']:.3f} ({m['win']}/{m['loss']} win/loss of "
              f"{m['n']})  day wins {dw[0]}/{dw[1]}")

    print("leg A  within-train val-block pooled margin (three fresh fits, "
          f"single-fabric keep={S_KEEP} vs frozen two-timescale keep="
          f"{CMRI_KEEP}; n = {len(e_v2)} steps / {len(val_days)} days; "
          "d = |e_v2| − |e_cmri|, positive = cmri better; win = |e_cmri| < "
          "|e_v2|):")
    print(f"    {'v2 reference (|e| itself)':<26} mean |e| "
          f"{statistics.fmean(abs(x) for x in e_v2):.4f} °C")
    print("  cmri configs (margin vs the same v2 reference):")
    fmt_m("single-fabric", m_s, dw_s)
    fmt_m("two-timescale", m_t, dw_t)
    print(f"      margin ordering: two-ts mean d {m_t['md']:+.4f} < single "
          f"{m_s['md']:+.4f}: {m_t['md'] < m_s['md']};  win rate "
          f"{m_t['win_rate']:.3f} < {m_s['win_rate']:.3f}: "
          f"{m_t['win_rate'] < m_s['win_rate']}")
    print()

    # ---- leg B: frozen two-timescale §5.5 digit audit (read-only) ----------
    fds, fv2, fcm = read_frozen()
    fkeys = [d.date().isoformat() for d in fds]
    fm = margin_stats(fv2, fcm)
    fdw = day_win(fv2, fcm, fkeys)
    fd = [abs(a) - abs(c) for a, c in zip(fv2, fcm)]
    fdm_hac576, fp_hac576 = dm_hac(fd, 576)
    fdm_hac1440, fp_hac1440 = dm_hac(fd, 1440)

    print(f"leg B  frozen predictions_cmri.csv audit (read-only, n={fm['n']}):")
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"    {name:<28} {got:+.4f} (committed {want:+.4f})  "
              f"{'PASS' if good else 'FAIL'}")
        return good

    check("mean d", fm["md"], F_POOLED_D, TOL_D)
    check("win rate", fm["win_rate"], F_WIN, TOL_WIN)
    good_n = fm["n"] == F_N
    ok &= good_n
    print(f"    n                              {fm['n']} (committed {F_N})  "
          f"{'PASS' if good_n else 'FAIL'}")
    good_dm = (fm["win"], fm["loss"]) == (F_WINC, F_N - F_WINC - 17)
    ok &= good_dm
    print(f"    win/loss counts (tie 17)       {fm['win']}/{fm['loss']} "
          f"(committed {F_WINC}/{F_N - F_WINC - 17})  "
          f"{'PASS' if good_dm else 'FAIL'}")
    check("sign z", fm["z_sign"], F_SIGN_Z, TOL_STATS)
    check("paired t (normal approx)", fm["t"], F_T_STAT, TOL_STATS)
    good_dw = fdw == F_DAYS
    ok &= good_dw
    print(f"    per-day win                    {fdw[0]}/{fdw[1]} days "
          f"(committed {F_DAYS[0]}/{F_DAYS[1]})  "
          f"{'PASS' if good_dw else 'FAIL'}")
    print(f"    DM HAC bw=576                  {fdm_hac576:.2f} p "
          f"{fp_hac576:.2e} (committed {F_HAC576[0]:.2f} p "
          f"{F_HAC576[1]:.2e})  "
          f"{'PASS' if (abs(fdm_hac576 - F_HAC576[0]) <= TOL_STATS and abs(fp_hac576 - F_HAC576[1]) <= 5e-4) else 'FAIL'}")
    ok &= abs(fdm_hac576 - F_HAC576[0]) <= TOL_STATS and abs(fp_hac576 - F_HAC576[1]) <= 5e-4
    print(f"    DM HAC bw=1440                 {fdm_hac1440:.2f} p "
          f"{fp_hac1440:.2e} (committed {F_HAC1440[0]:.2f} p "
          f"{F_HAC1440[1]:.2e})  "
          f"{'PASS' if (abs(fdm_hac1440 - F_HAC1440[0]) <= TOL_STATS and abs(fp_hac1440 - F_HAC1440[1]) <= 5e-4) else 'FAIL'}")
    ok &= abs(fdm_hac1440 - F_HAC1440[0]) <= TOL_STATS and abs(fp_hac1440 - F_HAC1440[1]) <= 5e-4
    print("  ℹ the single-fabric margin digits (mean d +"
          f"{B_SINGLE_D:.4f}, win rate {B_SINGLE_WIN:.3f}) have NO frozen "
          "artifact — results/predictions_cmri.csv and results/predictions.csv "
          "are BOTH post-two-timescale era — so those history digits are "
          "fresh-fit-only; their reproducible statement is the leg-A replay "
          "above (the same honesty convention as items 28/32/33).")
    print()

    # ---- structural gates (same-block relative invariants of the margin) ----
    g1 = m_s["md"] > 0.0 and m_t["md"] > 0.0     # neither fix erased the edge
    g2 = m_t["md"] < m_s["md"]                    # margin THINNER for two-ts
    g3 = m_t["win_rate"] < m_s["win_rate"]        # win share also thinner
    verdict = g1 and g2 and g3 and ok

    print("gates (within-train leg = fresh test; held-out leg = frozen audit; "
          "only same-block RELATIVE orderings between the two cmri configs "
          "are gated — the 20-day block cannot pin 42-day margin magnitudes, "
          "and within-train v2's absolute error level shifts both configs' d "
          "by the same v2 reference):")
    print(f"    (i)   both cmri pooled mean d > 0: single {m_s['md']:+.4f}, "
          f"two-ts {m_t['md']:+.4f}: {'PASS' if g1 else 'FAIL'}  (held "
          f"+0.1000 and +0.0713 both > 0)")
    print(f"    (ii)  two-ts pooled mean d < single: {m_t['md']:+.4f} < "
          f"{m_s['md']:+.4f}: {'PASS' if g2 else 'FAIL'}  (held +0.0713 < "
          f"+0.1000, the 'honest price of the free-run fix')")
    print(f"    (iii) two-ts win rate < single: {m_t['win_rate']:.3f} < "
          f"{m_s['win_rate']:.3f}: {'PASS' if g3 else 'FAIL'}  (held 0.579 < "
          f"0.676)")
    print()
    if verdict:
        print("verdict: PASS — the §5.5 margin-THINNING reading is a "
              "structural fact about the two cmri driver selections, not a "
              "single held-out draw: on fresh within-train fits both "
              f"selections' pooled margin is again positive (single "
              f"{m_s['md']:+.4f} / two-ts {m_t['md']:+.4f} °C, gate i; held "
              f"+0.1000/+0.0713) and the two-timescale margin is again "
              f"THINNER — mean d {m_t['md']:+.4f} < single {m_s['md']:+.4f} "
              f"(gate ii, the +0.1000 → +0.0713 sentence) and win rate "
              f"{m_t['win_rate']:.3f} < {m_s['win_rate']:.3f} (gate iii, the "
              f"0.676 → 0.579 sentence) — the honest price of the free-run "
              f"fix that §5.2/§5.6 quantify — and the frozen two-timescale "
              "§5.5 artifact re-produces every committed digit (mean d "
              f"{fm['md']:+.4f}, win rate {fm['win_rate']:.3f}, sign z "
              f"{fm['z_sign']:.2f}, paired t {fm['t']:.2f}, DM HAC 576 "
              f"{fdm_hac576:.2f} / 1440 {fdm_hac1440:.2f}, per-day "
              f"{fdw[0]}/{fdw[1]}).  The single-fabric side of the margin is "
              "documented honestly as fresh-fit-only (no single-fabric frozen "
              "artifact ever existed; both frozen CSVs are post-two-timescale "
              "era — the reproducible statement is this very leg-A replay).")
    else:
        print("verdict: FAIL — the §5.5 margin-THINNING reading does NOT "
              "replay on fresh within-train fits: pooled margins "
              f"{m_s['md']:+.4f} / {m_t['md']:+.4f} (gate i {int(g1)}), the "
              f"two-ts-vs-single ordering {m_t['md']:+.4f} vs {m_s['md']:+.4f} "
              f"(gate ii {int(g2)}) and win rates {m_t['win_rate']:.3f} vs "
              f"{m_s['win_rate']:.3f} (gate iii {int(g3)}) do not reproduce "
              "the held-out margin geometry — i.e. the thinning is a feature "
              "of the single held-out draw, not a structural property of the "
              "two driver selections.  The frozen two-timescale audit leg: "
              f"{'PASS' if ok else 'FAIL'}.")

    # block-delta documentation (always printed, PASS or FAIL)
    print()
    print("block-dependent strength deltas (documented, not gated): the d and "
          f"win-rate LEVELS (within-train single mean d {m_s['md']:+.4f} / "
          f"two-ts {m_t['md']:+.4f}, win rates {m_s['win_rate']:.3f} / "
          f"{m_t['win_rate']:.3f}; day wins {dw_s[0]}/{dw_s[1]} vs "
          f"{dw_t[0]}/{dw_t[1]}) are block-specific — a 20-day block vs the "
          "42-day held-out, and within-train v2's absolute error level shifts "
          "both configs' d by the same v2 reference — so only the RELATIVE "
          "orderings (gates ii and iii) and the both-positive shape (gate i) "
          "are structural; the frozen two-timescale §5.5 digits audit clean "
          "on the held-out artifact regardless.")


if __name__ == "__main__":
    main()
