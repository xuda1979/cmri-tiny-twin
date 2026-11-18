#!/usr/bin/env python3
"""verify_report_inventory.py — is the report's reproducibility catalog consistent?

REPORT.md §8 ("Reproducibility") and §9 ("Files") are the hand-maintained
contract between the report and the on-disk inventory.  Three count literals
are bumped by hand on every new probe — the §8 runner line's "(N/N scripts",
the §9 runner row's "all N verify probes", and the run_all_verifications.py
docstring's "(N scripts)" — and §7's 44 residual-diagnostic items are each
supposed to have at least one probe citing them.  Count-literal drift is
exactly the failure class this probe exists to catch: a new probe added to
disk without the three coordinated report edits silently breaks the catalog
(and has drifted in past cycles before the verify_*.py convention matured).

This is a report-inventory consistency claim, not a model reading, so the
within-train / held-out leg A / leg B structure does not apply: there is no
fitting and nothing frozen.  The natural probe is a deterministic read-only
re-derivation of the catalog from REPORT.md + run_all_verifications.py +
the filesystem, gating on every §8/§9 invariant:

  (a) every Python program invoked in the §8 code block exists on disk
      (the machinery invocations and all verify probes);
  (b) the on-disk verify_*.py set and the §8 verify-invocation set are
      identical (no probe missing from §8, no §8 ref to a deleted probe);
  (c) the §9 Files-table verify rows are identical to the on-disk probe set;
  (d) every §8 verify invocation carries a "#" comment (the cited-reading
      trail each probe must state);
  (e) the §7 items cited across the §8 verify comments cover exactly 1..44 —
      every §7 item has a citing probe and no citation falls outside the
      range (the "every section reading has a probe" audit in machine form;
      legacy probes that predate per-item citation are permitted — the gate
      is on the union, not on per-line coverage);
  (f) every file row in the §9 table exists on disk (program rows, verify
      rows, and results/* artifact rows);
  (g) every results/* artifact on disk appears in the §9 table (on-disk
      results/ == §9 results/ rows, no unlisted artifact);
  (h) the three committed count literals (runner docstring "(N scripts)",
      §8 "(N/N scripts", §9 "all N verify probes") all equal the derived
      probe count — the drift-prone hand-bumped numbers cannot diverge from
      discovery.

Note what is NOT gated: §8's phrase "all 3 machinery self-tests" refers to
the three selftest TARGETS (run_experiment --selftest, experiment_cmri
--selftest, cmri_twin.py), not the 13 machinery invocation lines in the code
block — the block's machinery line count is informational, only the prog-existence
gate (a) applies to those lines.

Exit status: 0 iff every gate passes; any committed inventory digit that does
not re-verify is reported as an honest negative (the run_all_verifications
suite then fails loudly).

Pure Python standard library; deterministic; read-only on REPORT.md,
run_all_verifications.py, and the filesystem inventory.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPORT = pathlib.Path("REPORT.md")
RUNNER = pathlib.Path("run_all_verifications.py")
N_ITEMS = 44  # REPORT.md §7 residual-diagnostics item count (committed catalog)


def code_block(text: str, heading: str) -> str:
    """The first fenced code block directly under `heading`."""
    m = re.search(re.escape(heading) + r"\s*\n```\n(.*?)\n```", text, re.S)
    return m.group(1) if m else ""


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    runner_src = RUNNER.read_text(encoding="utf-8")
    block = code_block(text, "## 8. Reproducibility")
    sec9 = text.split("## 9. Files", 1)[1]

    on_disk = sorted(p.name for p in pathlib.Path(".").glob("verify_*.py"))
    n = len(on_disk)

    # (a) every `python3 <prog>` the §8 block invokes exists on disk
    programs = sorted({m.group(1) for m in re.finditer(r"^python3 (\S+\.py)", block, re.M)})
    missing_prog = sorted(p for p in programs if not pathlib.Path(p).is_file())

    # verify invocations: (name, trailing text) — comment presence checked in (d)
    verify_lines = [
        (m.group(1), m.group(2))
        for m in re.finditer(r"^python3 (verify_\w+\.py)(\s+#.*)?$", block, re.M)
    ]
    invoked = sorted({name for name, _ in verify_lines})

    # (e) §7 item citations across the §8 verify comments cover exactly 1..N_ITEMS
    cited: set[int] = set()
    for _, tail in verify_lines:
        for tok in re.findall(r"item[s]?[-/ ]+([\d/]+)", tail or ""):
            cited.update(int(d) for d in tok.split("/"))
    missing_items = sorted(set(range(1, N_ITEMS + 1)) - cited)
    out_of_range = sorted(d for d in cited if not 1 <= d <= N_ITEMS)

    # §9 Files table -> file rows (the FIRST backticked token of each row is
    # the path; role cells may carry backticked formulas, so only the path
    # column is extracted)
    rows = []
    for line in sec9.splitlines():
        m = re.match(r"^\|\s*`([^`]+)`", line)
        if m:
            rows.append(m.group(1))
    verify_rows = sorted(p for p in rows if p.startswith("verify_"))
    results_rows = sorted(p for p in rows if p.startswith("results/"))
    missing_file = sorted(p for p in rows if not pathlib.Path(p).is_file())

    results_on_disk = sorted("results/" + p.name for p in pathlib.Path("results").glob("*"))
    missing_results = sorted(set(results_on_disk) - set(results_rows))
    extra_results = sorted(set(results_rows) - set(results_on_disk))

    # (h) the three hand-bumped count literals all equal the derived count
    lit_runner = re.search(r"\((\d+) scripts\)", runner_src)
    lit_s8 = re.search(r"\((\d+)/(\d+) scripts", text)
    lit_s9 = re.search(r"all (\d+) verify probes", text)
    literals = [int(lit_runner.group(1)), int(lit_s8.group(1)),
                int(lit_s8.group(2)), int(lit_s9.group(1))]

    no_comment = [name for name, tail in verify_lines
                  if tail is None or "#" not in tail]

    gates = {
        "(a) every §8-invoked program exists on disk": (
            not missing_prog,
            f"{len(programs) - len(missing_prog)}/{len(programs)} programs exist"
            + (f"; missing: {', '.join(missing_prog)}" if missing_prog else ""),
        ),
        "(b) on-disk probe set == §8 invocation set": (
            set(on_disk) == set(invoked),
            f"disk {len(on_disk)} == §8 {len(invoked)} "
            f"(missing from §8: {', '.join(sorted(set(on_disk) - set(invoked))) or 'none'}; "
            f"§8 refs not on disk: {', '.join(sorted(set(invoked) - set(on_disk))) or 'none'})",
        ),
        "(c) §9 Files verify rows == on-disk probe set": (
            set(on_disk) == set(verify_rows),
            f"§9 {len(verify_rows)} == disk {len(on_disk)} "
            f"(missing from §9: {', '.join(sorted(set(on_disk) - set(verify_rows))) or 'none'}; "
            f"§9 rows not on disk: {', '.join(sorted(set(verify_rows) - set(on_disk))) or 'none'})",
        ),
        "(d) every §8 verify invocation carries a # comment": (
            not no_comment,
            f"{len(verify_lines) - len(no_comment)}/{len(verify_lines)} commented"
            + (f"; missing: {', '.join(no_comment)}" if no_comment else ""),
        ),
        "(e) §7 items cited in §8 comments == 1..44 exactly": (
            not missing_items and not out_of_range,
            (f"all {N_ITEMS} §7 items 1..{N_ITEMS} cited, none outside"
             if not missing_items and not out_of_range
             else f"missing: {missing_items or 'none'}; outside range: {out_of_range or 'none'}"),
        ),
        "(f) every §9 file row exists on disk": (
            not missing_file,
            f"{len(rows) - len(missing_file)}/{len(rows)} rows exist"
            + (f"; missing: {', '.join(missing_file)}" if missing_file else ""),
        ),
        "(g) every results/ artifact on disk appears in §9": (
            not missing_results and not extra_results,
            f"disk {len(results_on_disk)} == §9 {len(results_rows)} "
            f"(unlisted: {', '.join(missing_results) or 'none'}; "
            f"listed but absent: {', '.join(extra_results) or 'none'})",
        ),
        "(h) count literals == derived probe count": (
            all(v == n for v in literals),
            f"runner docstring {literals[0]}, §8 ({literals[1]}/{literals[2]} scripts), "
            f"§9 all {literals[3]} — all equal derived on-disk {n}",
        ),
    }

    print("§8/§9 report-inventory consistency probe (deterministic read-only "
          "re-derivation of the reproducibility catalog from REPORT.md + "
          "run_all_verifications.py + the filesystem; consistency claim, "
          "no fitting, nothing frozen)")
    print(f"files: {REPORT}, {RUNNER}")
    print()
    print("  catalog sizes derived independently:")
    print(f"    on-disk verify_*.py      : {n}")
    print(f"    §8 verify invocations    : {len(invoked)}")
    print(f"    §9 files-table verify rows: {len(verify_rows)}")
    print(f"    §8 programs invoked (existence gate (a)): {len(programs)}")
    print(f"    results/ on disk == §9 results rows: {len(results_on_disk)} == {len(results_rows)}")
    print()
    print("  committed-invariant gates:")
    all_ok = True
    for name, (ok, detail) in gates.items():
        all_ok &= ok
        print(f"    {name}: {detail} : {'PASS' if ok else 'FAIL'}")
    print()
    if all_ok:
        print(f"verdict: PASS — the §8 reproducibility list, the §9 Files table "
              f"and the on-disk inventory agree exactly: {n} verify probes are "
              f"one set across all three, every one of the {len(programs)} "
              f"programs the §8 block invokes exists, all {N_ITEMS} §7 "
              f"residual-diagnostic items have a citing probe (none outside "
              f"1..{N_ITEMS}), all {len(rows)} §9 file rows exist, all "
              f"{len(results_on_disk)} results/ artifacts are listed, and the "
              f"three hand-bumped count literals cannot drift from the derived "
              f"count.")
    else:
        print(f"verdict: FAIL — the following committed inventory invariant(s) "
              f"did NOT re-verify: "
              + ", ".join(name for name, (ok, _) in gates.items() if not ok)
              + " (honest negative, reported as-is per the §7 items 31/34 "
                "precedent)")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
