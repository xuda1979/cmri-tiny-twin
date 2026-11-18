#!/usr/bin/env python3
"""run_all_verifications.py — one command to run the full verification program.

Consolidates every check in REPORT.md §8 into a single deterministic suite:

  * the three machinery self-tests
      - python3 run_experiment.py --selftest   (OLS / Gaussian elimination)
      - python3 experiment_cmri.py --selftest  (couple coefficient recovery)
      - python3 cmri_twin.py                   (coupled 2-zone self-test)
  * every verify_*.py probe (46 scripts) in alphabetical order.

Each entry is run in a fresh subprocess, so scripts stay run-order-independent
(no cross-verify imports, no shared ephemeral state — they only read the
frozen results/*.csv artifacts and the data file).  The suite exits 0 only
when every entry exits 0; a failing entry prints a summary line to stderr and
the final summary reports it.

Pure Python standard library, zero dependencies.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).parent

# (command, human label) — machines first, then every probe alphabetically.
SELFTESTS = [
    ("run_experiment.py --selftest", "machinery self-test: OLS / Gaussian elimination"),
    ("experiment_cmri.py --selftest", "machinery self-test: cmri coupled-coefficient recovery"),
    ("cmri_twin.py", "machinery self-test: coupled 2-zone physics recovery"),
]


def discover_probes() -> list[str]:
    return sorted(p.name for p in ROOT.glob("verify_*.py"))


def run_one(cmd: str) -> tuple[bool, float]:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, *cmd.split()],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    dt = time.monotonic() - t0
    ok = proc.returncode == 0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {cmd:<45} {dt:6.1f}s")
    if not ok:
        print(proc.stdout, file=sys.stderr)
    return ok, dt


def main() -> int:
    print(f"run_all_verifications: {ROOT}")
    print("\n-- machinery self-tests --")
    results = [run_one(cmd) for cmd, _ in SELFTESTS]

    probes = discover_probes()
    print(f"\n-- probes ({len(probes)} verify_*.py, alphabetical) --")
    results += [run_one(cmd) for cmd in probes]

    n_pass = sum(1 for ok, _ in results if ok)
    n_fail = len(results) - n_pass
    total = sum(dt for _, dt in results)
    print(f"\n=== SUMMARY: {n_pass} passed, {n_fail} failed  ({len(results)} runs, {total:.1f}s) ===")
    if n_fail:
        print("failed:", file=sys.stderr)
        for cmd, _ in zip([c for c, _ in SELFTESTS] + probes, results):
            if not _[0]:
                print(f"  {cmd}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
