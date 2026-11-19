#!/usr/bin/env python3
"""verify_report_inventory.py — is the report's reproducibility catalog consistent?

REPORT.md §8 ("Reproducibility") and §9 ("Files") are the hand-maintained
contract between the report and the on-disk inventory.  Count literals are
bumped by hand on every new probe — the §8 runner line's "(N/N scripts",
the §9 runner row's "all N verify probes", the run_all_verifications.py
docstring's "(N scripts)", README.md's six verification-count sites,
REPORT.md's two "N machinery self-tests" prose literals, and REPORT.md's
results-artifact count literals ("all N committed results/ artifacts",
"N/N artifacts MATCH", "N == N") — and every §7 residual-diagnostic item
(1..45 at the time of writing) is supposed to have at least one probe
citing it.  Count-literal drift is exactly the failure class
this probe exists to catch: a new probe added to disk without the
coordinated report/README edits silently breaks the catalog (and has
drifted in past cycles before the verify_*.py convention matured).

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
  (e) the §7 items cited across the §8 verify comments cover exactly 1..M
      where M is the item count derived from REPORT.md's own bold item
      list (1..45 today) — every §7 item has a citing probe and no citation
      falls outside the range (the "every section reading has a probe"
      audit in machine form; legacy probes that predate per-item citation
      are permitted — the gate is on the union, not on per-line coverage);
  (f) every file row in the §9 table exists on disk (program rows, verify
      rows, and results/* artifact rows);
  (g) every results/* artifact on disk appears in the §9 table (on-disk
      results/ == §9 results/ rows, no unlisted artifact);
  (h) the three committed count literals (runner docstring "(N scripts)",
      §8 "(N/N scripts", §9 "all N verify probes") all equal the derived
      probe count — the drift-prone hand-bumped numbers cannot diverge from
      discovery;
  (i) README.md's six committed verification-count sites (the three "53
      checks" literals, the "50 targeted verification probes" layout line,
      the "runs every probe (53 checks: ...)" line, and the trailing
      "Last run: 53 passed, 0 failed") all equal the derived counts —
      the checks total is re-derived as the runner's machinery self-tests
      plus the on-disk probe count, so README cannot drift either;
  (j) REPORT.md's two "N machinery self-tests" prose literals (§8 runner
      line "all 3 machinery self-tests", §9 runner row "runs the 3
      machinery self-tests") equal the runner's derived self-test count —
      those hand-bumped prose sites are gated like every other digit;
  (k) REPORT.md's results-artifact count literals — the two "all N
      committed results/ artifacts" phrases (§8 closing-claim line, §9
      verify_reproducibility.py row), the §8 "N/N artifacts MATCH" verdict,
      and the §9 "(N == N," summary — all equal the derived on-disk
      results/ count, so adding an artifact to disk + §9 without bumping
      the prose cannot silently go stale (gate (g) covers rows, not prose);
  (l) the non-verify program catalog: the on-disk *.py programs outside
      verify_*.py and the §9 program rows (the first backticked token of
      each non-README, non-verify, non-results row) are the SAME set — no
      program file on disk without a §9 row, no §9 program row whose file
      is absent — and every program the §8 block invokes as machinery is
      one of those §9 program rows (the §8→§9 inclusion direction, so a
      machinery invocation can never point at a program undocumented in
      §9).  Gates (a) and (f) each check only the exists-on-disk direction
      of their catalogs; (l) closes the cross-catalog set-agreement window.

Note what is NOT gated: §8's machinery code-block lines (13 at the time of
writing) are covered by gates (a) and (l); their LINE COUNT is
informational, not a committed literal.  §9-only programs that §8 machinery
never invokes (today exactly tiny_twin.py — a model module, not a driver)
are reported, not gated: a module need not be re-run by the reproducibility
block.  Gate (j) gates only the two "N machinery self-tests" count digits,
which refer to the three selftest TARGETS (run_experiment --selftest,
experiment_cmri --selftest, cmri_twin.py), not the machinery invocation
lines in the block.  Gate (k) gates REPORT.md's results-count PROSE, not
verify_reproducibility.py's writer-declared WRITERS lists (those are that
probe's own internal catalog, checked by its own byte-level run).

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
README = pathlib.Path("README.md")


def code_block(text: str, heading: str) -> str:
    """The first fenced code block directly under `heading`."""
    m = re.search(re.escape(heading) + r"\s*\n```\n(.*?)\n```", text, re.S)
    return m.group(1) if m else ""


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    runner_src = RUNNER.read_text(encoding="utf-8")
    readme_src = README.read_text(encoding="utf-8")
    block = code_block(text, "## 8. Reproducibility")
    sec9 = text.split("## 9. Files", 1)[1]

    on_disk = sorted(p.name for p in pathlib.Path(".").glob("verify_*.py"))
    n = len(on_disk)

    # derived "checks" total as printed by the suite: machinery self-tests
    # (counted from the runner's SELFTESTS list) + every verify probe
    m_selftests = re.search(r"SELFTESTS = \[\n(.*?)\n\]", runner_src, re.S)
    n_selftests = len(re.findall(r'^\s*\("', m_selftests.group(1), re.M)) if m_selftests else 0
    n_checks = n_selftests + n

    # derived §7 item count from REPORT.md's own bold item list (replaces the
    # hand-bumped N_ITEMS literal — same failure class as the count literals:
    # if §7 grows or renumbers, the gate must track the report, not the probe)
    sec7 = text.split("## 7.", 1)[1].split("## 8.", 1)[0]
    item_nums = [int(m.group(1)) for m in re.finditer(r"^\*\*(\d+)\.\s", sec7, re.M)]
    items_ok = bool(item_nums) and item_nums == list(range(1, item_nums[-1] + 1))
    n_items = item_nums[-1] if items_ok else 0  # 0 -> gate (e) FAILs honestly

    # (a) every `python3 <prog>` the §8 block invokes exists on disk
    programs = sorted({m.group(1) for m in re.finditer(r"^python3 (\S+\.py)", block, re.M)})
    missing_prog = sorted(p for p in programs if not pathlib.Path(p).is_file())

    # verify invocations: (name, trailing text) — comment presence checked in (d)
    verify_lines = [
        (m.group(1), m.group(2))
        for m in re.finditer(r"^python3 (verify_\w+\.py)(\s+#.*)?$", block, re.M)
    ]
    invoked = sorted({name for name, _ in verify_lines})

    # (e) §7 item citations across the §8 verify comments cover exactly 1..n_items
    cited: set[int] = set()
    for _, tail in verify_lines:
        for tok in re.findall(r"item[s]?[-/ ]+([\d/]+)", tail or ""):
            cited.update(int(d) for d in tok.split("/"))
    missing_items = sorted(set(range(1, n_items + 1)) - cited)
    out_of_range = sorted(d for d in cited if not 1 <= d <= n_items)

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

    # (j) the two committed "N machinery self-tests" prose literals (§8 runner
    #     line, §9 runner row) equal the derived self-test count
    lit_s8_mach = re.findall(r"all (\d+) machinery self-tests", text)
    lit_s9_mach = re.findall(r"runs the (\d+) machinery self-tests", text)
    mach_literals = [int(v) for v in lit_s8_mach + lit_s9_mach]
    mach_ok = bool(mach_literals) and all(v == n_selftests for v in mach_literals)

    # (k) REPORT.md's results-artifact count literals vs the derived on-disk
    #     results/ count — same failure class as (h)/(i)/(j): if a new results/
    #     artifact were added to disk + §9 without bumping the prose digits,
    #     the "all 13 committed results/ artifacts" / "13/13 artifacts MATCH" /
    #     "(13 == 13," sites would silently go stale (gate (g) compares rows,
    #     not prose).  Non-empty guard: 0 matches -> FAIL, not vacuous-pass.
    sec8 = text.split("## 8. Reproducibility", 1)[1].split("## 9. Files", 1)[0]
    n_results = len(results_on_disk)
    lit_s8_res = [int(v) for v in re.findall(r"all (\d+) committed results/ artifacts", sec8)]
    lit_s8_match = [int(v) for t in re.findall(r"(\d+)/(\d+) artifacts MATCH", sec8) for v in t]
    lit_s9_res = [int(v) for v in re.findall(r"all (\d+) committed results/ artifacts", sec9)]
    lit_s9_sum = [int(v) for t in re.findall(r"\((\d+) == (\d+),", sec9) for v in t]
    res_literals = lit_s8_res + lit_s8_match + lit_s9_res + lit_s9_sum
    res_ok = bool(res_literals) and all(v == n_results for v in res_literals)

    # (l) non-verify program catalog: on-disk *.py programs == §9 program rows,
    #     and every §8 machinery invocation is one of those rows.  Gates (a)/(f)
    #     check only the exists-on-disk direction; (l) closes the cross-
    #     catalog set-agreement window (a program file added without a §9 row,
    #     a §9 program row deleted while the file remains, or a §8 machinery
    #     invocation pointing at a program undocumented in §9).  Non-empty
    #     guards as elsewhere: a zero-set FAILS, never vacuous-passes.
    prog_on_disk = sorted(p.name for p in pathlib.Path(".").glob("*.py")
                          if not p.name.startswith("verify_"))
    program_rows = sorted(p for p in rows
                          if p.endswith(".py") and not p.startswith("verify_"))
    machinery_progs = sorted(set(programs) - set(invoked))
    prog_missing_from_s9 = sorted(set(prog_on_disk) - set(program_rows))
    prog_not_on_disk = sorted(set(program_rows) - set(prog_on_disk))
    mach_not_in_s9 = sorted(set(machinery_progs) - set(program_rows))
    s9_only = sorted(set(program_rows) - set(machinery_progs))  # informational
    prog_ok = (bool(prog_on_disk) and bool(program_rows) and bool(machinery_progs)
               and not prog_missing_from_s9 and not prog_not_on_disk
               and not mach_not_in_s9)

    # (i) README.md's committed verification-count sites vs derived totals;
    #     non-empty guards: a regex that matches NOTHING must FAIL loudly, not
    #     vacuous-pass (the all([]) == True trap — the six sites are committed
    #     prose, their absence is a drift, not a pass)
    readme_checks = [int(v) for v in re.findall(r"(\d+) (?:verification checks|checks)", readme_src)]
    readme_probes = [int(v) for v in re.findall(r"(\d+) (?:targeted verification )?probes", readme_src)]
    readme_selftests = [int(v) for v in re.findall(r"(\d+) self-tests", readme_src)]
    readme_lastrun = [int(v) for v in re.findall(r"Last run: .*?(\d+) passed", readme_src)]
    readme_lists = {"checks": readme_checks, "probes": readme_probes,
                    "self-tests": readme_selftests, "last-run": readme_lastrun}
    readme_nonempty = all(readme_lists.values())

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
        "(e) §7 items cited in §8 comments == 1..N exactly": (
            not missing_items and not out_of_range,
            (f"all {n_items} §7 items 1..{n_items} cited, none outside"
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
        "(i) README count sites == derived checks/probes": (
            (readme_nonempty
             and all(v == n_checks for v in readme_checks)
             and all(v == n for v in readme_probes)
             and all(v == n_selftests for v in readme_selftests)
             and all(v == n_checks for v in readme_lastrun)),
            f"checks {readme_checks} == derived {n_checks}, "
            f"probes {readme_probes} == on-disk {n}, "
            f"self-tests {readme_selftests} == runner {n_selftests}, "
            f"last-run {readme_lastrun} == derived {n_checks}, "
            f"all six sites non-empty: {readme_nonempty}",
        ),
        "(j) §8/§9 machinery self-test literals == runner count": (
            mach_ok,
            f"§8 {' + '.join(str(v) for v in lit_s8_mach)} == §9 "
            f"{' + '.join(str(v) for v in lit_s9_mach)} == derived "
            f"self-tests {n_selftests}",
        ),
        "(k) §8/§9 results-artifact count literals == derived results/ count": (
            res_ok,
            f"all {n_results} committed results/ artifacts across "
            f"{len(res_literals)} prose literal(s) "
            f"({' + '.join(str(v) for v in res_literals) or 'none'}) "
            f"== derived on-disk {n_results}",
        ),
        "(l) non-verify program catalog == §9 program rows ⊇ §8 machinery": (
            prog_ok,
            f"disk {len(prog_on_disk)} == §9 {len(program_rows)} program rows, "
            f"§8 machinery {len(machinery_progs)} ⊆ §9 "
            f"(on-disk w/o §9 row: {', '.join(prog_missing_from_s9) or 'none'}; "
            f"§9 row w/o file: {', '.join(prog_not_on_disk) or 'none'}; "
            f"§8 machinery not in §9: {', '.join(mach_not_in_s9) or 'none'}; "
            f"§9-only, never invoked (informational): "
            f"{', '.join(s9_only) or 'none'})",
        ),
    }

    print("§8/§9 report-inventory consistency probe (deterministic read-only "
          "re-derivation of the reproducibility catalog from REPORT.md + "
          "run_all_verifications.py + README.md + the filesystem; "
          "consistency claim, no fitting, nothing frozen)")
    print(f"files: {REPORT}, {RUNNER}, {README}")
    print()
    print("  catalog sizes derived independently:")
    print(f"    on-disk verify_*.py      : {n}")
    print(f"    runner machinery self-tests (SELFTESTS tuples): {n_selftests}")
    print(f"    derived checks total    : {n_checks} = self-tests + probes")
    print(f"    derived §7 items        : {n_items} (bold item list, contiguous 1..{n_items})")
    print(f"    §8 verify invocations    : {len(invoked)}")
    print(f"    §9 files-table verify rows: {len(verify_rows)}")
    print(f"    §8 programs invoked (existence gate (a)): {len(programs)}")
    print(f"    non-verify *.py programs (disk == §9 program rows, gate (l)): {len(prog_on_disk)} "
          f"== {len(program_rows)}")
    print(f"    results/ on disk == §9 results rows: {len(results_on_disk)} == {len(results_rows)}")
    print(f"    results-artifact prose literals gated by (k): {len(res_literals)} "
          f"({', '.join(str(v) for v in res_literals) or 'none'})")
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
              f"programs the §8 block invokes exists, all {n_items} §7 "
              f"residual-diagnostic items have a citing probe (none outside "
              f"1..{n_items}), all {len(rows)} §9 file rows exist, all "
              f"{len(results_on_disk)} results/ artifacts are listed, the "
              f"{len(prog_on_disk)} non-verify programs form one catalog "
              f"across disk and §9 (with every §8 machinery invocation inside "
              f"it), and the "
              f"three hand-bumped count literals plus README.md's six "
              f"verification-count sites, REPORT.md's two machinery "
              f"self-test literals and its results-artifact count literals "
              f"cannot drift from the derived count.")
    else:
        print(f"verdict: FAIL — the following committed inventory invariant(s) "
              f"did NOT re-verify: "
              + ", ".join(name for name, (ok, _) in gates.items() if not ok)
              + " (honest negative, reported as-is per the §7 items 31/34 "
                "precedent)")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
