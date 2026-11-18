#!/usr/bin/env python3
"""verify_solar_within.py — source-structural probe of the §5 clear-sky-GHI sentence.

REPORT.md §5 (line 122) describes the wind/solar preprocessing in the shipped
cmri twin as a *machinery fact* — a fixed, exogenous, computed signal that was
deliberately left OUT of the frozen winner:

    "Wind infiltration and clear-sky solar GHI (computed for the site,
     Chievres AB Belgium, DST-aware) are implemented but were **selected out**
     by within-train CV"

Every prior cmri probe gated the FROZEN coefficient tuples (items 29/30), the
metric tuples (items 35/37/39), the margin structure (items 33/34/38), or the
fabric-EMA machinery (item 42).  Not one audited the clear_sky_ghi machinery
itself: that the site constants are the committed (50.56 N, 3.83 E) of
Chievres AB, Belgium; that the offset is genuinely Europe/Brussels DST
(last-Sunday-of-month rules, UTC+1 winter / UTC+2 summer); that the solar
geometry is the committed declination / equation-of-time / hour-angle /
solar-noon = 12 − LON/15 construction with a Kasten-Young air mass and night
→ 0.0; that the loader computes GHI for every row of the real dataset (units
0–900 W/m²); and that the "selected out" reading is wired through the source:
CMRI_KEEP excludes driver index 2, the frozen ablation catalog carries the
solar-loss row (+0.0747), and the frozen coefficient artifact has no solar
columns at all.  This deterministic read-only probe closes that gap.

Leg A — source-structural gates (text scans of the shipped files):

  (i)  Site constants: cmri_twin.py ships LAT_DEG = 50.56, LON_DEG = 3.83
       with the "Chievres AB, Belgium" dataset citation.
  (ii) Europe/Brussels DST: the `_last_sunday` helper + `utc_offset_hours`
       encode last-Sun-Mar → last-Sun-Oct transitions (UTC+2 = start <= d <
       end, else UTC+1); behaviorally the 2016 transitions fall on
       2016-03-27 / 2016-10-30 and the +1/+2 bands hold.
  (iii) Solar geometry: the declination, equation-of-time, hour-angle,
       solar_noon_utc_h = 12.0 − LON_DEG/15.0 (~11.7447 h), Kasten-Young
       air-mass, solar-constant, and transmission factors appear verbatim.
  (iv) Night → 0.0: the `if sin_el <= 0.0: return 0.0` clamp is in source
       and behaviorally a winter 03:00 local datetime returns exactly 0.0
       while a summer-noon datetime returns large positive W/m².

Leg B — computational / frozen-artifact gates:

  (v)  Dataset units: load_cmri_data computes GHI for all 19,735 rows via the
       shipped clear_sky_ghi; every value lies in [0, 900] W/m², the dataset
       has both nights (zero rows) and clear-sky summer noon (> 400 W/m²),
       and a sampled direct replay through clear_sky_ghi matches the loader's
       column exactly.
  (vi) "Selected out" consequence: CMRI_KEEP = (0, 3, 4, 5, 6, 7, 8, 9)
       excludes index 2; K_NAMES/H_NAMES index 2 = "solar GHI"; the select_cv
       feature-layout docstrings label index 2 "clear-sky GHI"; the
       experiment_cmri.py prose states "selected out ... lose free-run
       stability"; and the ablation candidate "cmri_full" includes solar as
       the 9th driver.
  (vii) Frozen-artifact audit: results/ablation_cmri.csv re-derives the
       committed cmri_full solar row (d_rmse +0.0747, n = 5879, "clear-sky
       solar GHI" in the driver string) and results/coefficients_cmri.csv
       contains zero solar coefficient/role rows (18 rows: kitchen + house,
       bias + the 8 kept drivers).

DISCLOSURE (what is NOT gated, honest under the item-31 convention): §5's
"they lose free-run stability on this dataset" is the CAUSAL reading of a
single CV run — a selection-outcome claim, not a structural fact — so this
probe gates only the selected-out CONSEQUENCE (CMRI_KEEP excludes 2; the
frozen ablation loss +0.0747; no solar columns in the frozen coefficients),
never the free-run-stability reason itself.  Nor is there a fresh-fit solar
leg: solar is not a member of the frozen winner, so no frozen solar
coefficient artifact exists to re-derive — the reproducible statement is the
structural consequence + frozen-artifact audit, exactly as leg A/B(vii) above.

Pure Python standard library; deterministic; read-only.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

# noqa: E402 — project-relative imports after sys.path setup
import cmri_twin  # noqa: E402
from cmri_twin import LAT_DEG, LON_DEG, _last_sunday, clear_sky_ghi, utc_offset_hours  # noqa: E402
from experiment_cmri import CMRI_KEEP, load_cmri_data  # noqa: E402

SOLAR_NOON_UTC = 12.0 - LON_DEG / 15.0   # ~11.74467 h for Chievres 3.83 E
N_DATA = 19735                           # data/energydata_complete.csv rows

OK = "PASS"
FAIL = "FAIL"
_fails: list[str] = []


def check(sub: str, cond: bool, detail: str) -> bool:
    tag = OK if cond else FAIL
    if not cond:
        _fails.append(sub)
    print(f"  [{tag}] {sub:<56} {detail}")
    return cond


def gate_i(src: str) -> int:
    print("-- gate (i): site constants (Chievres AB, Belgium) --")
    n = 0
    n += check(
        "LAT_DEG = 50.56 literal", "LAT_DEG = 50.56" in src,
        "commitment latitude 50.56 N in cmri_twin.py source",
    )
    n += check(
        "LON_DEG = 3.83 literal", "LON_DEG = 3.83" in src,
        "commitment longitude 3.83 E in cmri_twin.py source",
    )
    n += check(
        "imported site constants match", LAT_DEG == 50.56 and LON_DEG == 3.83,
        f"module attributes LAT_DEG = {LAT_DEG}, LON_DEG = {LON_DEG}",
    )
    n += check(
        "Chievres dataset citation", "Chievres AB, Belgium" in src,
        "source comments name the dataset location",
    )
    return n


def gate_ii(src: str) -> int:
    print("-- gate (ii): Europe/Brussels DST rules --")
    n = 0
    n += check(
        "_last_sunday helper present", "def _last_sunday(year: int, month: int)" in src,
        "last-Sunday-of-month helper ships in cmri_twin.py",
    )
    n += check(
        "last-Sunday algorithm", "(nd.weekday() - 6) % 7" in src,
        "rolls back from the month's last day to its Sunday",
    )
    n += check(
        "UTC+1/UTC+2 band switch", "2.0 if start <= d < end else 1.0" in src,
        "summer band starts at last-Sun-Mar 01:00, ends last-Sun-Oct 01:00",
    )
    n += check(
        "2016 spring transition = 2016-03-27",
        _last_sunday(2016, 3) == __import__("datetime").date(2016, 3, 27),
        f"_last_sunday(2016, 3) = {_last_sunday(2016, 3)}",
    )
    n += check(
        "2016 fall transition = 2016-10-30",
        _last_sunday(2016, 10) == __import__("datetime").date(2016, 10, 30),
        f"_last_sunday(2016, 10) = {_last_sunday(2016, 10)}",
    )
    return n


def gate_binary_dst() -> int:
    n = 0
    n += check(
        "winter band UTC+1",
        utc_offset_hours(d(2016, 1, 15, 12, 0)) == 1.0
        and utc_offset_hours(d(2016, 11, 15, 12, 0)) == 1.0,
        "January and November local noon both outside the summer band",
    )
    n += check(
        "summer band UTC+2", utc_offset_hours(d(2016, 7, 15, 12, 0)) == 2.0,
        "July local noon inside the summer band",
    )
    n += check(
        "01:00 switch edges",
        utc_offset_hours(d(2016, 3, 27, 0, 59)) == 1.0
        and utc_offset_hours(d(2016, 3, 27, 1, 0)) == 2.0
        and utc_offset_hours(d(2016, 10, 30, 0, 59)) == 2.0
        and utc_offset_hours(d(2016, 10, 30, 1, 0)) == 1.0,
        "half-hour either side of the last-Sun 01:00 edge flips the band",
    )
    return n


def d(y: int, mo: int, da: int, h: int, mi: int):
    import datetime as _dt
    return _dt.datetime(y, mo, da, h, mi)


def gate_iii(src: str) -> int:
    print("-- gate (iii): solar-geometry formula strings --")
    n = 0
    n += check(
        "declination formula", "23.44 * math.sin(math.radians(360.0 * (n - 81) / 365))" in src,
        "n = day-of-year declination after the March-21st anomaly",
    )
    n += check(
        "equation-of-time formula", "9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)" in src,
        "EoT minutes from the standard elliptical-orbit series",
    )
    n += check(
        "solar noon = 12 - LON/15",
        "solar_noon_utc_h = 12.0 - LON_DEG / 15.0" in src,
        f"solar-noon construction as in the §5 sentence; = {SOLAR_NOON_UTC:.4f} h UTC",
    )
    n += check(
        "solar noon ~ 11.7447 h", abs(SOLAR_NOON_UTC - 11.7447) < 0.001,
        f"12.0 - 3.83/15.0 = {SOLAR_NOON_UTC:.4f} h for Chievres",
    )
    n += check(
        "hour-angle construction", "H = math.radians((solar_noon_utc_h - utc_h) * 15.0)" in src,
        "hour angle spread 15 deg/h about solar noon",
    )
    n += check(
        "elevation sine formula",
        "math.cos(math.radians(LAT_DEG)) * math.cos(decl) * math.cos(H)" in src,
        "sin(el) = sin(lat)sin(decl) + cos(lat)cos(decl)cos(H)",
    )
    n += check(
        "Kasten-Young air mass", "0.50572 * (el_deg + 6.07995) ** -1.6364" in src,
        "am = 1/(sin(el) + 0.50572*(el+6.07995)^-1.6364)",
    )
    n += check(
        "solar-constant TSI", "1361.0 * (1.0 + 0.033 * math.cos(math.radians(360.0 * (n - 2) / 365)))" in src,
        "1361 W/m^2 solar constant with annual eccentricity modulation",
    )
    n += check(
        "beam + diffuse return", "0.9 * i0 * sin_el * 0.7 ** (am ** 0.678)" in src,
        "direct-beam times diffuse transmission along the air mass",
    )
    return n


def gate_iv() -> int:
    print("-- gate (iv): night -> 0.0 clamp, behaviorally --")
    n = 0
    src = (ROOT / "cmri_twin.py").read_text(encoding="utf-8")
    n += check(
        "night clamp in source", "if sin_el <= 0.0:" in src and "return 0.0" in src,
        "sun below the horizon yields zero irradiance",
    )
    winter_night = clear_sky_ghi(d(2016, 1, 15, 3, 0))   # local 03:00 = UTC 02:00
    n += check(
        "winter 03:00 local -> 0.0", winter_night == 0.0,
        f"clear_sky_ghi(2016-01-15 03:00) = {winter_night:.3f} W/m^2",
    )
    summer_noon = clear_sky_ghi(d(2016, 7, 15, 13, 30))  # local 13:30 = UTC 11:30
    n += check(
        "summer noon -> > 400 W/m^2", summer_noon > 400.0,
        f"clear_sky_ghi(2016-07-15 13:30) = {summer_noon:.1f} W/m^2",
    )
    winter_noon = clear_sky_ghi(d(2016, 1, 15, 13, 0))   # local 13:00 = UTC 12:00
    n += check(
        "winter noon -> > 50 W/m^2", winter_noon > 50.0,
        f"clear_sky_ghi(2016-01-15 13:00) = {winter_noon:.1f} W/m^2",
    )
    return n


def gate_v(data: dict) -> int:
    print("-- gate (v): dataset units 0-900 W/m^2 over all rows --")
    n = 0
    ghi = data["GHI"]
    n += check("GHI computed for every row", len(ghi) == N_DATA,
               f"len(GHI) = {len(ghi)} == {N_DATA}")
    n += check("all 0 <= GHI <= 900", min(ghi) >= 0.0 and max(ghi) <= 900.0,
               f"[{min(ghi):.1f}, {max(ghi):.1f}] W/m^2 within the committed band")
    n += check("clear-sky summer noon observed", max(ghi) > 400.0,
               f"dataset max GHI = {max(ghi):.1f} W/m^2")
    zero_rows = sum(1.0 for g in ghi if g == 0.0)
    n += check("nights present (zero rows dominate)", zero_rows > 5000,
               f"{zero_rows:.0f} night rows at 0.0 W/m^2 over 137 days")
    replay_ok = all(ghi[i] == cmri_twin.clear_sky_ghi(data["date"][i]) for i in (0, 8000, 16000))
    n += check("loader column == direct replay", replay_ok,
               "sampled rows re-derive the shipped column exactly")
    return n


def gate_vi(twin_src: str, exp_src: str, cv_src: str, slow_src: str) -> int:
    print("-- gate (vi): 'selected out' consequence in the shipped source --")
    n = 0
    n += check(
        "CMRI_KEEP excludes driver 2", CMRI_KEEP == (0, 3, 4, 5, 6, 7, 8, 9) and 2 not in CMRI_KEEP,
        f"CMRI_KEEP = {CMRI_KEEP} — index 2 (solar GHI) absent",
    )
    n += check(
        "CMRI_KEEP literal in source",
        "CMRI_KEEP = (0, 3, 4, 5, 6, 7, 8, 9)" in exp_src.replace("  # ", " # ").replace("#", "#"),
        "the frozen keep mask as committed in experiment_cmri.py",
    )
    k2 = cmri_twin.CmriTwin.K_NAMES[2]
    h2 = cmri_twin.CmriTwin.H_NAMES[2]
    n += check(
        "K_NAMES/H_NAMES index 2 = solar GHI",
        k2.startswith("solar GHI") and h2.startswith("solar GHI"),
        f"kitchen '{k2}', house '{h2}'",
    )
    n += check(
        "select_cv.py index-2 label", "2 clear-sky GHI" in cv_src,
        "feature-layout docstring names driver 2 'clear-sky GHI'",
    )
    n += check(
        "select_cv_slowfab.py index-2 label", "2 clear-sky GHI" in slow_src,
        "slowfab selector likewise names driver 2 'clear-sky GHI'",
    )
    n += check(
        "selected-out prose in experiment_cmri.py",
        "selected out" in exp_src and "Chievres" in exp_src and "DST-aware" in exp_src,
        "the §5 sentence's selected-out reading is re-stated in source prose",
    )
    n += check(
        "ablation candidate includes solar",
        "clear-sky solar GHI (full 9-driver cmri model)" in exp_src,
        "cmri_full ablation entry adds solar as the 9th driver",
    )
    n += check(
        "GHI computed at load time",
        'rows["GHI"] = [clear_sky_ghi(d) for d in rows["date"]]' in exp_src,
        "solar is an exogenous preprocessing column, not a fit target",
    )
    return n


def gate_vii() -> int:
    print("-- gate (vii): frozen-artifact audit (ablation + coefficients) --")
    n = 0
    abp = ROOT / "results" / "ablation_cmri.csv"
    ab_rows = []
    with open(abp, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ab_rows.append(r)
    full = [r for r in ab_rows if r["variant"] == "cmri_full"]
    n += check(
        "ablation cmri_full row present", len(full) == 1,
        f"{len(full)} cmri_full row(s) in results/ablation_cmri.csv",
    )
    if full:
        r = full[0]
        n += check(
            "frozen solar loss d_rmse +0.0747", r["d_rmse"].strip() == "+0.0747",
            f"d_rmse = {r['d_rmse']} vs committed +0.0747",
        )
        n += check(
            "frozen ablation n = 5879", r["n"].strip() == "5879",
            f"n = {r['n']} vs committed 5879",
        )
        n += check(
            "driver string names solar",
            "clear-sky solar GHI" in r["drivers"] and "full 9-driver" in r["drivers"],
            f"drivers = '{r['drivers']}'",
        )
    else:
        for _ in range(3):
            n += 0
    cop = ROOT / "results" / "coefficients_cmri.csv"
    co_rows = []
    with open(cop, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            co_rows.append(r)
    solar_rows = [r for r in co_rows
                  if "solar" in r["coefficient"].lower() or "solar" in r["role"].lower()]
    n += check(
        "coefficients have exactly 18 rows", len(co_rows) == 18,
        f"{len(co_rows)} coefficient rows (kitchen 9 + house 9 = bias + the 8 kept drivers)",
    )
    n += check(
        "no solar coefficient rows", len(solar_rows) == 0,
        "frozen coefficients_cmri.csv has zero solar columns — the winner keeps 8 drivers",
    )
    return n


def main() -> None:
    twin_src = (ROOT / "cmri_twin.py").read_text(encoding="utf-8")
    exp_src = (ROOT / "experiment_cmri.py").read_text(encoding="utf-8")
    cv_src = (ROOT / "select_cv.py").read_text(encoding="utf-8")
    slow_src = (ROOT / "select_cv_slowfab.py").read_text(encoding="utf-8")
    data = load_cmri_data("data/energydata_complete.csv")

    total = 0
    total += gate_i(twin_src)
    total += gate_ii(twin_src)
    total += gate_binary_dst()
    total += gate_iii(twin_src)
    total += gate_iv()
    total += gate_v(data)
    total += gate_vi(twin_src, exp_src, cv_src, slow_src)
    total += gate_vii()

    print(f"\n=== SUMMARY: {total - len(_fails)}/{total} gates PASS ===")
    if _fails:
        print("failed gates:", ", ".join(_fails), file=sys.stderr)
        sys.exit(1)
    print("verify_solar_within.py: all gates PASS — the §5 clear-sky solar-GHI "
          "machinery reading is structurally verified (site, DST, geometry, "
          "night clamp, dataset units, selected-out consequence).")


if __name__ == "__main__":
    main()
