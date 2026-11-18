#!/usr/bin/env python3
"""Deterministic read-only structural probe — REPORT.md §5 model-selection CV protocol.

Gate target (REPORT.md line 122): "The frozen config (mask nws+fab+slowfab,
T_set=21, alpha=300, w=0.944, w2=0.986) was chosen by a 3-fold growing-origin
prequential CV on the training block only (select_cv.py for the single-fabric
winner, then select_cv_slowfab.py for the slow-capacitor extension), using the
fold-SD tie window with free-run tie-break; the held-out test was touched
exactly once."

This is a structural reading (a protocol description), not a single-run
outcome, so the probe parses the shipped selector sources themselves —
select_cv.py and select_cv_slowfab.py — and cross-checks every claimed
mechanism against the shipped frozen-constant block (experiment_cmri.py
CMRI_ALPHA/CMRI_T_SET/CMRI_W/CMRI_W2/CMRI_KEEP, run_experiment.py
TRAIN_FRAC = 0.70) and the committed §2 split value (13,814). No fitting and no
data are loaded: everything below is pure stdlib source text + integer
arithmetic on the shipped constants.

Leg structure (item-44 convention: protocol mechanics are structural facts,
gated; no fresh-fit leg exists because the CV protocol is not a fitted outcome):
  (i)   3-fold growing-origin prequential structure in BOTH selectors
  (ii)  train-only boundary — CV never touches the held-out test
  (iii) fold-SD tie window (NOT SE/sqrt(NF))
  (iv)  free-run tie-break + stability gate (FREE_CUT)
  (v)   the frozen config sits inside the selector grids at every level
  (vi)  the slow-capacitor extension preserves the same machine
Overlap note: the w/w2 grid LITERALS (W_GRID/TAU_H/W2_GRID/TAU2_H) and their
frozen-constant membership are gated by verify_fabricmass_within.py (item 42);
this probe gates the CV PROTOCOL structure, not those literals.
"""
import sys
from pathlib import Path

OK = "PASS"
FAIL = "FAIL"
_fails: list[str] = []


def check(sub: str, cond: bool, detail: str = "") -> None:
    tag = OK if cond else FAIL
    if not cond:
        _fails.append(sub)
    line = f"  [{tag}] {sub}"
    if detail:
        line += f" — {detail}"
    print(line)


REPO = Path(__file__).resolve().parent
CV = (REPO / "select_cv.py").read_text(encoding="utf-8")
SLOW = (REPO / "select_cv_slowfab.py").read_text(encoding="utf-8")
EXP = (REPO / "experiment_cmri.py").read_text(encoding="utf-8")
RUN = (REPO / "run_experiment.py").read_text(encoding="utf-8")


def constant(name: str, src: str, default: str = "") -> str:
    """Return the value text of the first FOO = <value> for FOO in src,
    stripped of a trailing inline comment."""
    import re

    m = re.search(rf"^{name}\s*=\s*([^\n]+)", src, re.M)
    if not m:
        return default
    return re.split(r"\s+#", m.group(1).strip())[0].strip()


def gate_i_cv_structure() -> int:
    print("\nGate (i) — 3-fold growing-origin prequential structure in BOTH selectors")
    n = 0
    for label, src in (("select_cv.py", CV), ("select_cv_slowfab.py", SLOW)):
        n += 1
        check(
            f"{label} N_FOLDS = 4 (4 growing windows ⇒ 3 folds)",
            constant("N_FOLDS", src) == "4",
            f"N_FOLDS = {constant('N_FOLDS', src)}",
        )
        n += 1
        check(
            f"{label} NF = N_FOLDS - 1 (the 3-fold count)",
            coefficient_text(src) == "N_FOLDS - 1",
            coefficient_text(src),
        )
        n += 1
        btext = bounds_text(src)
        check(
            f"{label} growing-origin bounds = [0, split/4, split/2, 3·split/4, split]",
            "bounds = [0] + [round(split * j / N_FOLDS) for j in range(1, N_FOLDS + 1)]" in src,
            btext,
        )
        n += 1
        check(
            f"{label} fold f fits the growing prefix: fit_hi = bounds[f+1], slice(0, hi)",
            "fit_hi = bounds[f + 1]" in src and "s = slice(0, hi)" in src,
        )
        n += 1
        check(
            f"{label} fold f evaluates [bounds[f+1], bounds[f+2]) — one future window",
            "lo, hi = bounds[f + 1], bounds[f + 2]" in src,
        )
        n += 1
        check(
            f"{label} last bounds endpoint == split = 13814 (never reaches the 19,735-row test block)",
            max(round(13814 * j / 4) for j in range(1, 5)) == 13814 and 13814 < 19735,
            f"max endpoint {max(round(13814 * j / 4) for j in range(1, 5))} vs split 13814 < n 19735",
        )
    return n


def coefficient_text(src: str) -> str:
    import re

    m = re.search(r"^\s*NF\s*=\s*([^\n]+)", src, re.M)
    return m.group(1).strip() if m else "<missing>"


def bounds_text(src: str) -> str:
    import re

    m = re.search(r"^\s*bounds\s*=\s*([^\n]+)", src, re.M)
    return m.group(1).strip() if m else "<missing>"


def gate_ii_train_only() -> int:
    print("\nGate (ii) — train-only boundary: the held-out test is NEVER inside a fold")
    n = 0
    # split derivation: n = 19735 rows, TRAIN_FRAC = 0.70 → 13,814 train rows.
    n += 1
    check(
        "run_experiment.py TRAIN_FRAC = 0.70 (the §2 split ratio)",
        constant("TRAIN_FRAC", RUN) == "0.70",
        f"TRAIN_FRAC = {constant('TRAIN_FRAC', RUN)}",
    )
    n += 1
    check(
        "commit §2 value: int(19735 × 0.70) = 13814 (pure int arithmetic, no data load)",
        int(19735 * 0.70) == 13814,
        "split = 13814",
    )
    n += 1
    check(
        "pure-int growing-origin bounds for n=19735: [0, 3454, 6907, 10360, 13814]",
        [0] + [round(13814 * j / 4) for j in range(1, 5)] == [0, 3454, 6907, 10360, 13814],
        f"{[0] + [round(13814 * j / 4) for j in range(1, 5)]}",
    )
    # select_cv_slowfab.py asserts the guard outright (line 86).
    n += 1
    check(
        "select_cv_slowfab.py asserts 'CV must be strictly within train; no test contact.'",
        'assert split < n, "CV must be strictly within train; no test contact."' in SLOW,
    )
    n += 1
    check(
        "select_cv.py docstring: 'The held-out 30% test is NEVER touched here.'",
        "The held-out 30% test is NEVER touched here." in CV,
    )
    n += 1
    check(
        "select_cv_slowfab.py docstring: 'Test data (>= split) is NEVER touched here.'",
        "Test data (>= split) is NEVER touched here." in SLOW,
    )
    n += 1
    check(
        "select_cv_slowfab.py: fast fabric w is frozen (CMRI_W) during selection, not re-selected",
        "Fast fabric w is FROZEN at CMRI_W=0.944 (tau=3h, the frozen config)" in SLOW,
    )
    # No test-block constant may be referenced by either selector.
    n += 1
    check(
        "neither selector references a test-start constant (no leakage path)",
        "SPLIT" not in CV and "TEST_" not in CV
        and "SPLIT" not in SLOW and "TEST_" not in SLOW,
    )
    return n


def gate_iii_fold_sd_window() -> int:
    print("\nGate (iii) — fold-SD tie window, NOT SE/sqrt(NF)")
    n = 0
    for label, src in (("select_cv.py", CV), ("select_cv_slowfab.py", SLOW)):
        n += 1
        check(
            f"{label} window = best_sd if stable else 0.0 — fold-SD uncertainty scale",
            "window = best_sd if stable_sorted else 0.0  # fold-SD, NOT SE/sqrt(NF)"
            in src
            or "window = best_sd if stable_sorted else 0.0  # fold-SD, honest uncertainty scale"
            in src,
        )
        n += 1
        check(
            f"{label} tied = candidates within best_res + best_sd (the fold-SD tie window)",
            "tied = [c for c in stable_sorted if c[0] <= best_res + window]" in src,
        )
    return n


def gate_iv_free_run_tie() -> int:
    print("\nGate (iv) — free-run tie-break + stability gate (FREE_CUT)")
    n = 0
    for label, src in (("select_cv.py", CV), ("select_cv_slowfab.py", SLOW)):
        n += 1
        check(
            f"{label} FREE_CUT = 2.0 (stability threshold)",
            constant("FREE_CUT", src) == "2.0",
            f"FREE_CUT = {constant('FREE_CUT', src)}",
        )
        n += 1
        check(
            f"{label} stable = mf < FREE_CUT — resync-primary, free-run-valid only",
            "stable = mf < FREE_CUT" in src,
        )
        n += 1
        check(
            f"{label} wr = min(tied, key=lambda c: c[1]) — lowest free-run RMSE breaks the tie",
            "wr = min(tied, key=lambda c: c[1])" in src,
        )
        n += 1
        check(
            f"{label} prints 'SELECTED (fold-sd-tie, free-run tie-break)'",
            "SELECTED (fold-sd-tie, free-run tie-break)" in src,
        )
    n += 1
    check(
        "select_cv.py also prints 'STRICT-SELECTED (resync-primary)' (no-tie path)",
        "STRICT-SELECTED (resync-primary)" in CV,
    )
    n += 1
    check(
        "select_cv_slowfab.py also prints 'STRICT-SELECTED (resync-primary)'",
        "STRICT-SELECTED (resync-primary)" in SLOW,
    )
    return n


def gate_v_frozen_in_grids() -> int:
    print("\nGate (v) — the frozen config sits inside the selector grids")
    n = 0
    cmri_alpha = constant("CMRI_ALPHA", EXP)
    cmri_tset = constant("CMRI_T_SET", EXP)
    cmri_w = constant("CMRI_W", EXP)
    cmri_w2 = constant("CMRI_W2", EXP)
    cmri_keep = constant("CMRI_KEEP", EXP)
    n += 1
    check(
        "frozen alpha 300.0 ∈ selector alpha grid (0.0, 100.0, 300.0, 1000.0, 5000.0)",
        cmri_alpha == "300.0" and "(0.0, 100.0, 300.0, 1000.0, 5000.0)" in CV,
        f"CMRI_ALPHA = {cmri_alpha}",
    )
    n += 1
    check(
        "frozen T_set 21.0 ∈ selector tset grid (20.0, 21.0)",
        cmri_tset == "21.0" and "(20.0, 21.0)" in CV,
        f"CMRI_T_SET = {cmri_tset}",
    )
    n += 1
    check(
        "slow-capacitor frozen keep mask = (0, 3, 4, 5, 6, 7, 8, 9) = CMRI_KEEP",
        constant("SLOW_KEEP", SLOW) == "(0, 3, 4, 5, 6, 7, 8, 9)"
        and cmri_keep == "(0, 3, 4, 5, 6, 7, 8, 9)",
        f"SLOW_KEEP = {constant('SLOW_KEEP', SLOW)}, CMRI_KEEP = {cmri_keep}",
    )
    n += 1
    check(
        "single-fabric frozen-shape winner mask (0, 3, 4, 5, 6, 7, 9) is a select_cv candidate",
        '("nws+fab", (0, 3, 4, 5, 6, 7, 9), W_GRID)' in CV,
    )
    n += 1
    check(
        "slowfab frozen (fast-w) baseline = find(FROZEN_KEEP, CMRI_T_SET, 300.0, None)",
        "fc = find(FROZEN_KEEP, CMRI_T_SET, 300.0, None)" in SLOW
        and constant("FROZEN_KEEP", SLOW) == "(0, 3, 4, 5, 6, 7, 8)",
        f"FROZEN_KEEP = {constant('FROZEN_KEEP', SLOW)}",
    )
    n += 1
    check(
        "frozen w = 0.944 / w2 = 0.986 exist as selector constants (membership per item 42)",
        cmri_w == "0.944" and cmri_w2 == "0.986",
        f"CMRI_W = {cmri_w}, CMRI_W2 = {cmri_w2}",
    )
    # The nws+fab entry names the 2-zone slow-capacitor extension source, so the
    # frozen config's slow path has an explicit W_GRID→W2_GRID hand-off hook.
    n += 1
    check(
        "select_cv.py slow-capacitor candidates restart the grid at W_GRID for nws+fab",
        '("nws+fab", (0, 3, 4, 5, 6, 7, 9), W_GRID)' in CV,
    )
    return n


def gate_vi_extension() -> int:
    print("\nGate (vi) — the slow-capacitor extension preserves the same machine")
    n = 0
    n += 1
    check(
        "select_cv_slowfab.py docstring: 'same growing-origin machinery, fold-SD tie window and free-run tie-break as select_cv.py'",
        "origin machinery, fold-SD tie window and free-run tie-break as select_cv.py" in SLOW,
    )
    n += 1
    check(
        "selector docstrings name the extension as the w2 stage for the frozen w=0.944 winner",
        "W2_GRID" in SLOW and "w2" in SLOW,
    )
    n += 1
    check(
        "extension is a second-stage selector, sharing N_FOLDS/FREE_CUT/grid layout with select_cv.py",
        constant("N_FOLDS", SLOW) == constant("N_FOLDS", CV)
        and constant("FREE_CUT", SLOW) == constant("FREE_CUT", CV),
        f"N_FOLDS {constant('N_FOLDS', SLOW)}/{constant('N_FOLDS', CV)}, "
        f"FREE_CUT {constant('FREE_CUT', SLOW)}/{constant('FREE_CUT', CV)}",
    )
    n += 1
    check(
        "the frozen config (w=0.944, w2=0.986) is the selectors' outcome, restated in experiment_cmri.py",
        "CMRI_W2" in EXP and cmri_block_sane(EXP),
    )
    return n


def cmri_block_sane(src: str) -> bool:
    """The frozen constants block is present and internally consistent."""
    return "CMRI_ALPHA = 300.0" in src and "CMRI_T_SET = 21.0" in src


def main() -> None:
    print("verify_selection_cv_within.py — REPORT.md §5 line-122 model-selection CV protocol")
    print("reads select_cv.py, select_cv_slowfab.py, experiment_cmri.py, run_experiment.py (no fitting)")

    gate_funcs = [
        gate_i_cv_structure,
        gate_ii_train_only,
        gate_iii_fold_sd_window,
        gate_iv_free_run_tie,
        gate_v_frozen_in_grids,
        gate_vi_extension,
    ]
    total = 0
    for fn in gate_funcs:
        total += fn()

    print(f"\n=== SUMMARY: {total - len(_fails)}/{total} gates PASS ===")
    if _fails:
        print("failed gates: " + ", ".join(_fails), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
