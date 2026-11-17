#!/usr/bin/env python3
"""verify_horizontrade_within.py — is the §5.4/§5.6 MID-DAY-EDGE TRADE reading
("the two-timescale re-selection traded mid-day edge (which the single-fabric
config held, §5.6 previous) for a large free-run stability win" and "the
last-6 h edge ... grew from Δ0.19 to Δ0.25 °C per step") a structural fact
about the two cmri driver selections, or an artifact of the single held-out
draw?

§5.6 committed claims (two-timescale frozen held-out resync,
results/predictions_cmri.csv, as re-aggregated by verify_horizon.py /
verify_horizon_shape.py):
  * the five per-horizon band pairs (v2 / cmri mean |err|): 0.0736/0.0607 in
    the first 2 h, 0.1783/0.1417 at 2-6 h, 0.3179/0.3187 at 6-12 h (the
    mid-day null), 0.5389/0.5337 at 12-17.7 h, 0.8340/0.5847 in the last 6 h
    (d = +0.2493 ≈ Δ0.25, the dominant reset-horizon edge);
  * the single-fabric history digit: the same last-6 h band had d ≈ Δ0.19
    before the slow capacitor was added, i.e. the two-timescale selection
    "grew" the edge; and the single-fabric config "held" a mid-day edge the
    two-timescale re-selection gave up (traded for the open-loop gain).

Honesty anchor — which digits are frozen-auditable and which are not:
  * results/predictions_cmri.csv (and results/predictions.csv) are BOTH
    post-two-timescale era: NO single-fabric frozen artifact ever existed.
    The single-fabric band digits (Δ0.19 last-6 h, "mid-day edge held") are
    fresh-fit-only history numbers from the pre-slowfab §5.6, and the §5.4
    "traded mid-day edge" sentence is a cross-fit comparison — so their
    reproducible statement is the fresh leg-A replay below, exactly the
    honesty convention items 28/32 use for fresh-fit-only digits.
  * What CAN be frozen-audited is the two-timescale §5.6 band table: the
    five frozen band pairs re-derive from the frozen CSV (read-only, same
    convention as verify_horizon_shape.py's leg B) plus the day-block win
    counts 29/41 (last 6 h) and 27/42 (first 2 h).

This script probes the TRADE without touching the held-out test: leg A
re-fits THREE twins fresh on the within-train fit slice [0, 11051) — v2
(CalibratedTwin, the mean-|err| reference), the superseded single-fabric
CmriTwin (alpha 300, t_set 21, keep=(0,3,4,5,6,7,8), fast fabric t_mass only,
a 9-array fit), and the frozen two-timescale CmriTwin (keep=(0,3,4,5,6,7,8,9),
fast + slow fabric, 10-array fit) — replays all three under the daily-resync
protocol on the within-train val block [11051, 13814) (n = 2743 steps / 20
days), and recomputes the same five horizon bands per config (the same band
edges, the same step-since-day-anchor convention) so the two cmri selections
are compared band-by-band on the SAME fresh block; leg B re-reads the frozen
results/predictions_cmri.csv (read-only) and re-aggregates every committed
two-timescale §5.6 digit (five band pairs + day-block counts).

Gates (power-aware relative invariants of the TRADE reading; the 20-day block
cannot pin the 42-day magnitudes, so only same-block RELATIVE directions are
gated — d_c = band mean|e_v2| − band mean|e_cmri-config|, positive = cmri
better at that horizon, the exact verify_horizon convention):
  (i)  mid-day 6-12 h band: single-fabric d > two-timescale d within-train —
       the single fabric still holds MORE mid-day edge than the two-timescale
       selection (held: single fabric held the edge, the re-selection traded
       it away);
  (ii) last-6 h band: two-timescale d > single-fabric d within-train — the
       reset-horizon edge is LARGER for the two-timescale selection (held:
       Δ0.25 vs Δ0.19, the "grew from Δ0.19 to Δ0.25" sentence whose two
       sides are verified on one fresh block);
  (iii) leg B re-audits the frozen two-timescale §5.6 digits exactly as
       verify_horizon_shape.py: the five band pairs within TOL = 0.0015 and
       the day-block counts 29/41 (last 6 h) and 27/42 (first 2 h).
Block-dependent strength deltas (documented, not gated): the per-band mean
|err| levels and the d magnitudes (10-20-day block vs 42-day held-out), and
any within-train shift in v2's absolute error level, which shifts both
configs' d by the same v2 reference.

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
STEPS_PER_DAY = 144
TOL = 0.0015

# Superseded single-fabric cmri config (REPORT.md §5.2 line 140 / §6 item 30 /
# §7 item 11 provenance): pre-slow-capacitor driver mask, fast fabric only.
S_ALPHA = 300.0
S_T_SET = 21.0
S_KEEP = (0, 3, 4, 5, 6, 7, 8)

# Frozen two-timescale §5.6 band digits (leg-B audit), same convention as
# verify_horizon.py / verify_horizon_shape.py:
# (band, (lo, hi), v2 mean |err|, cmri mean |err|).
BANDS = [
    ((0, 12), "0-12 steps (0-2 h)", 0.0736, 0.0607),
    ((12, 36), "12-36 steps (2-6 h)", 0.1783, 0.1417),
    ((36, 72), "36-72 steps (6-12 h)", 0.3179, 0.3187),
    ((72, 106), "72-106 steps (12-17.7 h)", 0.5389, 0.5337),
    ((106, 144), "106+ steps (last 6 h)", 0.8340, 0.5847),
]
FROZEN_DAYBLOCK_LAST6 = (29, 41)   # committed: cmri better 29/41 days last 6 h
FROZEN_DAYBLOCK_FIRST2 = (27, 42)  # committed: 27/42 days first 2 h

# History digits (fresh-fit-only, NO frozen artifact — both frozen CSVs are
# post-two-timescale era): the single-fabric last-6 h edge and its mid-day
# edge delta are the §5.6 previous / §5.4 trade references.
B_SINGLE_LAST6_D = 0.19             # historical single-fabric last-6 h d
B_HIST_MID = "single-fabric held the mid-day edge the re-selection traded away"


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction, same as verify_horizon
    return z, 1.0 - phi(z)      # one-sided


def collect_one(err, dts):
    """Single config's per-horizon |err| archive: per_h[h] = [(day, |e|), ...]
    with h = steps since that day's resync anchor (verify_horizon convention)."""
    per_h = [[] for _ in range(STEPS_PER_DAY)]
    days: dict[str, list[float]] = {}
    for e, d in zip(err, dts):
        key = d.date().isoformat()
        av = abs(e)
        days.setdefault(key, []).append(av)
        h = len(days[key]) - 1
        per_h[h].append((key, av))
    return per_h, days


def band_mean(per_h, lo: int, hi: int) -> float:
    """Mean |err| over a horizon band of a single config."""
    return statistics.fmean(x for h in range(lo, hi) for _, x in per_h[h])


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

    print("horizon TRADE probe (daily-resync protocol; the §5.4/§5.6 reading "
          "that the two-timescale cmri selection traded mid-day edge for the "
          "open-loop gain, growing the last-6 h edge); held-out test untouched "
          "for fitting")
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
        return [(c, o, d) for c, o, d in zip(pred, obs, dates[val_start:split])
                if c is not None]

    r_v2 = rows_of(pv2)
    r_s = rows_of(ps)
    r_t = rows_of(pt)
    assert len(r_v2) == len(r_s) == len(r_t) == 2743
    e_v2 = [o - c for c, o, _ in r_v2]
    e_s = [o - c for c, o, _ in r_s]
    e_t = [o - c for c, o, _ in r_t]
    dv = [d for _, _, d in r_v2]
    assert dv == [d for _, _, d in r_s] == [d for _, _, d in r_t]

    p_v2, _ = collect_one(e_v2, dv)
    p_s, _ = collect_one(e_s, dv)
    p_t, _ = collect_one(e_t, dv)

    def band_row(lo, hi):
        m2 = band_mean(p_v2, lo, hi)
        ms = band_mean(p_s, lo, hi)
        mt = band_mean(p_t, lo, hi)
        return m2, ms, mt, m2 - ms, m2 - mt   # d_c = v2 − cmri config

    print("leg A  within-train val-block horizon bands (three fresh fits, "
          f"single-fabric keep={S_KEEP} vs frozen two-timescale keep="
          f"{CMRI_KEEP}; n = {len(e_v2)} steps / {len(val_days)} days; "
          "d_c = mean|e_v2| − mean|e_c|c]:")
    print(f"  {'horizon':<26} {'v2':>8} {'single':>8} {'two-ts':>8} "
          f"{'d_sgl':>9} {'d_two':>9}")
    ws = {}
    for (lo, hi), label, *_ in BANDS:
        m2, ms, mt, ds, dt_ = band_row(lo, hi)
        ws[label] = (m2, ms, mt, ds, dt_)
        print(f"  {label:<26} {m2:>8.4f} {ms:>8.4f} {mt:>8.4f} "
              f"{ds:>+9.4f} {dt_:>+9.4f}")

    mid2, mids, midt, dmid_s, dmid_t = ws["36-72 steps (6-12 h)"]
    last2, lasts, lastt, dlast_s, dlast_t = ws["106+ steps (last 6 h)"]
    print(f"      context: mid 6-12 h d_single {dmid_s:+.4f} vs d_two "
          f"{dmid_t:+.4f}; last-6 h d_two {dlast_t:+.4f} vs d_single "
          f"{dlast_s:+.4f}")
    print()

    # ---- leg B: frozen two-timescale §5.6 digit audit (read-only) ----------
    fds, fv2, fcm = read_frozen()
    fper_v, _ = collect_one(fv2, fds)
    fper_c, _ = collect_one(fcm, fds)
    print("leg B  frozen predictions_cmri.csv audit (read-only, n=%d):"
          % len(fds))
    audit_ok = True
    for (lo, hi), label, cv2, ccm in BANDS:
        av2 = band_mean(fper_v, lo, hi)
        acm = band_mean(fper_c, lo, hi)
        ok = abs(av2 - cv2) < TOL and abs(acm - ccm) < TOL
        audit_ok &= ok
        print(f"    {label:<26} v2 {av2:.4f} (committed {cv2:.4f})  cmri "
              f"{acm:.4f} (committed {ccm:.4f})  d {av2 - acm:+.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    fd = {label: (band_mean(fper_v, lo, hi), band_mean(fper_c, lo, hi))
          for (lo, hi), label, _, _ in BANDS}

    def frozen_dayblock(perh_v, perh_c, lo, hi):
        by_day: dict[str, list[tuple[float, float]]] = {}
        for h in range(lo, hi):
            rowc = {key: ac for key, ac in perh_c[h]}
            for key, av in perh_v[h]:
                by_day.setdefault(key, []).append((av, rowc[key]))
        m = [statistics.fmean(a for a, _ in v) - statistics.fmean(c for _, c in v)
             for v in by_day.values()]
        wins = sum(1 for x in m if x > 1e-12)
        return wins, len(m)

    fl6_w, fl6_n = frozen_dayblock(fper_v, fper_c, 106, 144)
    ff2_w, ff2_n = frozen_dayblock(fper_v, fper_c, 0, 12)
    ok_l6 = (fl6_w, fl6_n) == FROZEN_DAYBLOCK_LAST6
    ok_f2 = (ff2_w, ff2_n) == FROZEN_DAYBLOCK_FIRST2
    audit_ok &= ok_l6 and ok_f2
    print(f"    frozen day-block last-6 h: {fl6_w}/{fl6_n} days (committed "
          f"{FROZEN_DAYBLOCK_LAST6[0]}/{FROZEN_DAYBLOCK_LAST6[1]}): "
          f"{'PASS' if ok_l6 else 'FAIL'}")
    print(f"    frozen day-block first-2 h: {ff2_w}/{ff2_n} days (committed "
          f"{FROZEN_DAYBLOCK_FIRST2[0]}/{FROZEN_DAYBLOCK_FIRST2[1]}): "
          f"{'PASS' if ok_f2 else 'FAIL'}")
    print("  ℹ the single-fabric band digits (last-6 h d ≈ "
          f"{B_SINGLE_LAST6_D}, and the §5.4 'mid-day edge held') have NO "
          "frozen artifact — results/predictions_cmri.csv and "
          "results/predictions.csv are BOTH post-two-timescale era — so those "
          "history digits are fresh-fit-only; their reproducible statement is "
          "the leg-A replay above (the same honesty convention as items "
          "28/32).")
    print()

    # ---- structural gates (relative same-block invariants of the trade) ----
    g1 = dmid_s > dmid_t   # single fabric holds more mid-day edge than two-ts
    g2 = dlast_t > dlast_s  # two-ts last-6 h edge larger than single fabric
    ok = g1 and g2 and audit_ok

    print("gates (within-train leg = fresh test; held-out leg = frozen audit; "
          "only same-block RELATIVE directions are gated — the 20-day block "
          "cannot pin 42-day magnitudes, and within-train v2's absolute error "
          "level shifts both configs' d by the same reference):")
    print(f"    (i)  mid-day 6-12 h: single-fabric d > two-timescale d: "
          f"{dmid_s:+.4f} > {dmid_t:+.4f} within : "
          f"{'PASS' if g1 else 'FAIL'}  (held: the re-selection traded the "
          f"mid-day edge the single fabric held)")
    print(f"    (ii) last-6 h: two-timescale d > single-fabric d: "
          f"{dlast_t:+.4f} > {dlast_s:+.4f} within : "
          f"{'PASS' if g2 else 'FAIL'}  (held Δ0.25 vs Δ0.19, the 'grew from "
          f"Δ0.19 to Δ0.25' sentence)")
    print()
    if ok:
        print("verdict: PASS — the §5.4/§5.6 mid-day-edge TRADE reading is a "
              "structural fact about the two cmri driver selections, not a "
              "single held-out draw: on fresh within-train fits the "
              f"single-fabric config again holds MORE mid-day 6-12 h edge "
              f"(d {dmid_s:+.4f} vs the two-timescale {dmid_t:+.4f}, gate i; "
              "held: the re-selection traded it away) while the two-timescale "
              f"selection's reset-horizon last-6 h edge is again LARGER "
              f"(d {dlast_t:+.4f} vs single-fabric {dlast_s:+.4f}, gate ii; "
              "held Δ0.25 vs Δ0.19) — the same give-and-take the §5.2 "
              "open-loop gain quantifies — and the frozen two-timescale §5.6 "
              "artifact re-produces every committed digit (five band pairs "
              "within " + str(TOL) + f", day-blocks {fl6_w}/{fl6_n} and "
              f"{ff2_w}/{ff2_n}).  The single-fabric side of the trade is "
              "documented honestly as fresh-fit-only (no single-fabric frozen "
              "artifact ever existed; both frozen CSVs are post-two-timescale "
              "era — the reproducible statement is this very leg-A replay).")
    else:
        print("verdict: FAIL — the §5.4/§5.6 mid-day-edge TRADE reading does "
              "NOT replay on fresh within-train fits: the single-fabric "
              f"config's mid-day 6-12 h d {dmid_s:+.4f} vs two-timescale "
              f"{dmid_t:+.4f} (gate i {int(g1)}) and the last-6 h d "
              f"{dlast_t:+.4f} vs {dlast_s:+.4f} (gate ii {int(g2)}) do not "
              "reproduce the held-out trade directions — i.e. the edge "
              "give-and-take is a feature of the single held-out draw, not a "
              "structural property of the two driver selections.  The frozen "
              f"two-timescale audit leg: {'PASS' if audit_ok else 'FAIL'}.")

    # block-delta documentation (always printed, PASS or FAIL)
    print()
    print("block-dependent strength deltas (documented, not gated): the "
          f"per-band mean |err| levels (within-train mid 6-12 h v2 "
          f"{mid2:.4f} / two-ts {midt:.4f}; last-6 h v2 {last2:.4f} / two-ts "
          f"{lastt:.4f}) and the d magnitudes (mid {dmid_s:+.4f} vs "
          f"{dmid_t:+.4f}; last-6 h {dlast_t:+.4f} vs {dlast_s:+.4f}) are "
          "block-specific — a 20-day block vs the 42-day held-out, and "
          "within-train v2's absolute error level shifts both configs' d by "
          "the same v2 reference — so only the relative directions (gates i "
          "and ii) are structural; the frozen two-timescale §5.6 digits "
          "audit clean on the held-out artifact regardless.")


if __name__ == "__main__":
    main()
