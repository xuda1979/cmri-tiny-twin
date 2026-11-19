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
      of their catalogs; (l) closes the cross-catalog set-agreement window;
  (m) this probe's OWN self-description: the "N/N gates PASS (§8, §9)"
      numerals on its §8 machinery line and its §9 Files row (REPORT.md's
      account of how many gates this probe runs) must equal this probe's
      derived gate total.  Those digits were hand-bumped in every gate
      commit - exactly the drift class this probe exists to catch, now
      applied to the report's description of the probe itself, which the
      probe can verify read-only against its own gate count.  A missing
      self-description site FAILs, never vacuous-passes;
  (n) this probe's self-description PROSE numerals, on the same two lines
      (m) gates: the probe count ("50 verify probes" on the §8 line, "(50
      probes across all three)" on the §9 row), the §7 item count ("all 45
      §7 items" / "all 45 §7 residual-diagnostic items", both equal to the
      derived bold-item count), the non-verify program count ("the 14
      non-verify programs"), and the §9 row's checks breakdown
      "(checks 53 == 3 self-tests + 50 probes, probes 50, self-tests 3,
      last-run 53)" (checks/self-tests/probes all equal to the runner-
      derived totals).  These digits are hand-bumped on the report's
      description of the probe itself and are all derivable read-only, so
      they cannot drift without this gate failing.  Each scoped regex must
      find its site — a numeral removed from the prose FAILs, never
      vacuous-passes;
  (o) the four neighbouring deterministic read-only probes' verdict
      numerals on the §8 machinery lines and the §9 dataset_provenance row
      (the "N/N gates PASS" verdict each of those probes ends its line
      with): the numeral must equal each probe's OWN live gate total, not
      this probe's.  Live totals are derived by RUNNING the four probes
      via subprocess (a HEAD-less, read-only, deterministic re-derivation:
      each is a self-contained reader with a fixed gate count) — the three
      SUMMARY-style probes (verify_solar_within.py -> 44, verify_selection_
      cv_within.py -> 45, verify_negativecontrols_within.py -> 34) print
      "=== SUMMARY: N/N gates PASS ===" whose denominator is the total; the
      dataset-provenance probe prints no SUMMARY line, so its total is the
      count of its plain gates["(…] dict keys (11) — a static count that
      IS exact for that probe because its gates dict is a literal (an
      in-probe self-description gate, out of scope here, would be the
      durable answer).  The §8 lines for
      solar/selection/negativecontrols and the dataset-provenance line on
      BOTH §8 and §9 each carry exactly one "N/N gates PASS" token that
      must equal the run-derived total; a missing numeral, a stray second
      token, or a numeral stale from a probe-gate change FAILs, never
      vacuous-passes;
  (p) the §9 Files-table GRAND-TOTAL prose numerals on the four SUMMARY-
      style probe rows that (o) does NOT cover — verify_fabricmass_within.py,
      verify_solar_within.py, verify_selection_cv_within.py, and verify_
      negativecontrols_within.py each end their row with a bold "N/N total"
      grand total (31/31, 44/44, 45/45 and 34/34 today) that sums their leg
      verdicts.  The discriminator is the literal "total" suffix: those rows
      also carry "X/Y gates PASS" leg sub-totals (4/4, 8/8, 12/12, 6/6,
      3/3, 5/5, 7/7) that a "N/N total" pattern does NOT match, so each row
      must carry EXACTLY ONE "N/N total" token (a stray sub-total digit is
      not a false pass).  Each must equal its probe's OWN live gate total,
      re-derived with the same read-only subprocess run as (o) — all four
      are SUMMARY-style probes whose printed "=== SUMMARY: N/N gates PASS
      ===" denominator is the exact total.  A hand-bump to any grand-total
      numeral, a missing token, or a second stray token FAILs, never
      vacuous-passes — this closes the last window of unguarded count
      literals in the verification catalog: (o) gated the §8 verdict lines
      and the §9 dataset_provenance row, (p) gates the §9 row grand totals,
      (q) gates the seven plain probes' §9 verdict-total numerals;
  (q) the §9 Files-table verdict-total "N/N gates PASS" numerals on the
      seven plain (non-SUMMARY-style) probe rows (o) and (p) do not touch:
      verify_freerun_episode_within.py (5/5), verify_v1handbook_within.py
      (6/6), verify_v2method_within.py (7/7), verify_housezone_audit.py
      (4/4), verify_horizontrade_within.py (2/2), verify_horizonsplit_
      within.py (4/4) and verify_metrictuple_within.py (6/6).  Each of
      these probes prints exactly ONE countable gates block in its live
      stdout — a "gates (...):" header line, one "(roman) ... : PASS" line
      per gate, then a "verdict:" line — so the numeral IS derivable
      read-only by RUNNING the probe and counting the gate lines between
      the header and the verdict.  The gate-line regex is deliberately
      tail-UNANCHORED (".*: PASS" with no end anchor) because housezone_
      audit / horizontrade / metrictuple gate lines carry trailing
      "  (held …)" annotations AFTER the ": PASS", and the block is
      anchored at the "gates (" header so a probe's pre-header leg-B
      "…: PASS" lines (housezone_audit prints 13) are never counted.  The
      v1handbook / v2method rows carry TWO "N/N gates PASS" tokens (an
      intermediate "N/N source-structural gates PASS" leg followed by the
      verdict total), so the comparison uses last-total-token semantics:
      each row needs >= 1 token and its LAST token must equal the live
      gate-line count — zero tokens, a stale last token, or a stray later
      token FAILs, never vacuous-passes;
  (r) the §1 / §2.3 dataset-provenance PROSE numerals in REPORT.md's body
      — the §1 Period/Structure bullets (lines 13-14 at the time of
      writing: "2016-01-11 17:00 → 2016-05-27 18:00 (137 days)" and
      "19,735 rows, all at a uniform 10-minute (600 s) resolution") and
      the §2.3 Split bullet (line 44: "first 70% (13,814 samples) train,
      last 30% (5,921 samples) test") — must re-derive read-only from
      `data/energydata_complete.csv` itself, the one source those facts
      describe.  Gates (a)-(q) cover only the §8/§9 reproducibility
      catalog and this probe's self-description; nothing before (r)
      ties the report BODY's provenance story to the actual data file
      (verify_dataset_provenance.py gates the CSV against its OWN
      literals, not against what REPORT.md's prose claims).  Derivation:
      count the CSV's data rows (19,735 today; the header is the first
      row), parse the first/last `date` cells (prose format "17:00"
      vs CSV "17:00:00" — the gate matches the PROSE forms), compute
      the calendar span as inclusive-days-minus-one, i.e. (last - first)
      in days (137, consistent with SPAN_DAYS in verify_dataset_
      provenance.py — the inclusive cell count is 138), and derive the
      split as int(rows x 0.70) = 13,814 train / 5,921 held-out.  The
      prose writes thousands separators ("19,735 rows", "13,814
      samples", "5,921 samples", "600 s"); every scoped site must match
      once (exactly one occurrence per bullet) and equal the derived
      value — a stale digit, a missing site, or a stray second match
      FAILs, never vacuous-passes;
  (s) the §7 residual-diagnostics body-prose VERDICT-TOTAL numerals — the
      six summary bullets that close with an overall "N/N gates PASS"
      verdict (REPORT.md lines 450/457/464/471/539/548 at the time of
      writing: housezone_audit "4/4", horizontrade "2/2", marginthin
      "1/3", metrictuple "6/6", selection_cv "(45/45 gates PASS)",
      negativecontrols "(34/34 gates PASS)") — must equal the citing
      probe's LIVE gate-line count: the "gates ("-to-"verdict:" block
      gate count, with the denominator matching "(: (PASS|FAIL)" — a
      ": PASS"-only matcher would wrongly derive marginthin as 1/1 when
      its gates ii/iii are FAIL lines — and the numerator matching only
      the ": PASS" lines (so marginthin honestly yields 1/3).  The four
      block-style probes (housezone_audit 4, horizontrade 2, marginthin 1,
      metrictuple 6) are subprocess-derived like (q); the two
      SUMMARY-style probes (selection_cv 45, negativecontrols 34) print
      no gate block, so their pass==total reuses the SAME live totals (o)
      derived by running them.  Gates (a)-(r) cover only the §8/§9
      catalog, the §1/§2.3 provenance prose, and this probe's own self-
      description; nothing before (s) ties the §7 residual-diagnostic
      verdicts' own totals to the gate lines the probes actually ran —
      a probe-gate change without the body-bullet bump FAILs, never
      vacuous-passes.
  (t) the §7/§9 "N/N fresh gates PASS" verdict-total prose numerals — the
      within-train FRESH-leg totals of the two block-style probes whose
      gates block also carries a frozen audit leg (item 36 rhocollapse §7
      body line 478 + §9 row line 668, item 37 episode §7 body line 485):
      the report discriminates the fresh leg ("N/N fresh gates PASS", gated
      here) from the frozen leg ("6/6 / 10/10 PASS" bare audit-total
      tokens — a separate literal class, gated by (u) below).  Fresh = the
      "gates ("-to-"verdict:" block lines marked PASS (denominator
      PASS|FAIL, as in (s)) whose text does NOT contain "frozen": for
      rhocollapse that excludes gate (iv) "frozen decomposition re-derives"
      (fresh 3 of 4 gate lines), for episode no gate text is frozen (fresh
      5).  rhocollapse is in no earlier gate's probe list and the
      word-"fresh" token form does not match any existing "N/N gates PASS"
      census regex, so before (t) a probe-gate change without the report
      bump could not FAIL — now it FAILs, never vacuous-passes.
  (u) the §7/§8/§9 FROZEN-leg bare "N/N PASS" audit-total prose numerals —
      the plain "N/N PASS"-tailed lines of the three block-style probes'
      frozen audit legs (freerun_half 14/14 at §7 body lines 446-447,
      rhocollapse 6/6 at §7 body line 480 + §9 row line 668, episode 10/10
      at §7 body line 487 + §8 machinery code-block line 600 + §9 row line
      672): the report discriminates the frozen leg ("6/6 / 10/10 PASS"
      bare audit-total tokens, gated here) from the fresh leg ("N/N fresh
      gates PASS", gated by (t)).  Derivation re-runs each probe and slices
      its stdout from the "leg B" header to the "gates (" header, counting
      the ": PASS" item lines by UNANCHORED ":\\s*PASS" search (denominator
      ":\\s*PASS" | ":\\s*FAIL") — unanchored rather than tail-anchored
      because episode's audit block carries a mid-line parenthetical after
      ": PASS" ("committed six episode days == ... : PASS  (committed
      [...])") that would drop its tail-anchored count to 9; no
      "frozen"-text exclusion is needed because the audit blocks carry no
      other gated line class.  The two remaining bare "N/N PASS" tokens are
      NOT gated: marginthin's "10/10" = 9 audit-block PASS lines plus a
      verdict-tail clause ("The frozen two-timescale audit leg: PASS.") so
      it is not a block-line-count literal, and horizonsplit's "17/17" =
      only 12 printed audit PASS lines (17 is a sub-check/comparison total)
      — both excluded, so no off-by-one can pass a derivable-count gate.

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

import csv
import pathlib
import re
import subprocess
import sys
from datetime import datetime

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

    # (m) this probe's OWN self-description: REPORT.md's account of this probe
    #     (its §8 machinery line and its §9 Files row) ends in a hand-bumped
    #     "N/N gates PASS (§8, §9)" numeral that reported the gate total at
    #     prose-write time — bumped in EVERY gate commit.  Derivable read-only
    #     from inside the probe: N must equal the derived gate total (len(gates)
    #     once (m) itself is included), so adding/removing a gate without the
    #     coordinated report bump fails here.  Non-empty guards: a missing or
    #     numeral-less self-description site FAILs, never vacuous-passes.
    #     re.findall over the two targeted scopes like gates (j)/(k): each
    #     self-description line must carry EXACTLY ONE "N/N gates PASS" token
    #     and every one must equal the derived total (a stray second token on
    #     either line is as much a drift as a wrong numeral).  Verified against
    #     the neighbouring probe lines' "N/N gates" numerals (lines 614/615/616
    #     etc.), which do not collide with these two lines' single tokens.
    self_s8 = re.findall(
        r"^python3 verify_report_inventory\.py[^\n]*?(\d+)/(\d+) gates PASS",
        block, re.M)
    self_s9 = re.findall(
        r"^\|\s*`verify_report_inventory\.py`[^\n]*?(\d+)/(\d+) gates PASS",
        sec9, re.M)

    # (n) this probe's self-description PROSE numerals, on the same two
    #     lines gate (m) scopes (captured in full so the scoped regexes
    #     below cannot collide with neighbouring probes' lines): the probe
    #     count ("50 verify probes" §8 / "(50 probes across all three)"
    #     §9), the §7 bold-item count ("all 45 §7 items" §8 / "all 45 §7
    #     residual-diagnostic items" §9) and its "1..45" range endpoint,
    #     the non-verify program count ("the 14 non-verify programs" on
    #     both), and the §9 row's checks breakdown "(checks 53 == 3
    #     self-tests + 50 probes, probes 50, self-tests 3, last-run 53)".
    #     Each is derivable read-only from the derived totals above and is
    #     hand-bumped in the report's account of this probe — same drift
    #     class as (m).  Every scoped site must carry exactly one numeral;
    #     a missing site or a stray second match FAILs, never
    #     vacuous-passes.
    s8_full = re.findall(r"^python3 verify_report_inventory\.py[^\n]*$",
                         block, re.M)
    s9_full = re.findall(r"^\|\s*`verify_report_inventory\.py`[^\n]*$",
                         sec9, re.M)
    line_s8 = s8_full[0] if s8_full else ""
    line_s9 = s9_full[0] if s9_full else ""
    n_progs = len(prog_on_disk)
    s8_probes = re.findall(r"(\d+) verify probes", line_s8)
    s9_probes = re.findall(r"\((\d+) probes across all three\)", line_s9)
    s8_items = re.findall(r"all (\d+) §7 items", line_s8)
    s9_items = re.findall(r"all (\d+) §7 residual-diagnostic items", line_s9)
    s8_range = re.findall(r"none outside (\d+)\.\.(\d+)", line_s8)
    s9_range = re.findall(r"none outside (\d+)\.\.(\d+)", line_s9)
    s8_progs = re.findall(r"the (\d+) non-verify programs", line_s8)
    s9_progs = re.findall(r"the (\d+) non-verify programs", line_s9)
    s9_break = re.findall(
        r"checks (\d+) == (\d+) self-tests \+ (\d+) probes, "
        r"probes (\d+), self-tests (\d+), last-run (\d+)", line_s9)
    prose_ok = (
        len(s8_probes) == 1 and int(s8_probes[0]) == n
        and len(s9_probes) == 1 and int(s9_probes[0]) == n
        and len(s8_items) == 1 and int(s8_items[0]) == n_items
        and len(s9_items) == 1 and int(s9_items[0]) == n_items
        and len(s8_range) == 1 and int(s8_range[0][0]) == 1
        and int(s8_range[0][1]) == n_items
        and len(s9_range) == 1 and int(s9_range[0][0]) == 1
        and int(s9_range[0][1]) == n_items
        and len(s8_progs) == 1 and int(s8_progs[0]) == n_progs
        and len(s9_progs) == 1 and int(s9_progs[0]) == n_progs
        and len(s9_break) == 1
        and [int(v) for v in s9_break[0]]
        == [n_checks, n_selftests, n, n, n_selftests, n_checks]
    )

    # (o) the four neighbouring deterministic read-only probes' verdict
    #     numerals ("N/N gates PASS") — the §8 machinery lines for
    #     verify_solar_within.py / verify_selection_cv_within.py /
    #     verify_negativecontrols_within.py, and the §8 line AND §9 Files
    #     row for verify_dataset_provenance.py — must equal each probe's OWN
    #     live gate total, not this probe's.  Live totals are derived
    #     read-only by RUNNING each probe (subprocess, same interpreter, the
    #     script's own directory as cwd): the three SUMMARY-style probes
    #     print "=== SUMMARY: N/N gates PASS ===" whose denominator is the
    #     live total; verify_dataset_provenance.py prints no SUMMARY line, so
    #     its total is the count of its gates["(…] dict-literal keys (that
    #     static count IS exact for THIS probe because its gates dict is a
    #     plain literal with no loop-expanded entries).  Each scoped line
    #     must carry EXACTLY ONE token equal to its probe's live total — the
    #     findall len==1 guards make a missing numeral, a stray second token,
    #     or a stale numeral (a probe-gate change without the coordinated
    #     report bump) FAIL, never vacuous-pass.
    _root = pathlib.Path(__file__).parent
    _sum_re = re.compile(r"=== SUMMARY: (\d+)/(\d+) gates PASS ===")
    _live = {}
    # fabricmass joins the live-total derivation for gate (p) — it is NOT in
    # _sh below because (o)'s §8 loop would then demand a verdict token on the
    # fabricmass §8 line, which has none; its §9 grand-total numeral is gated
    # only by (p).
    for _p in ("verify_dataset_provenance.py", "verify_solar_within.py",
               "verify_selection_cv_within.py",
               "verify_negativecontrols_within.py",
               "verify_fabricmass_within.py"):
        _r = subprocess.run([sys.executable, _p], cwd=_root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
        _m = _sum_re.search(_r.stdout)
        if _m:
            _live[_p] = int(_m.group(2))
        else:
            _live[_p] = len(re.findall(r'gates\["\(',
                                       (_root / _p).read_text()))
    _sh = {"verify_dataset_provenance.py": "dataset_provenance",
           "verify_solar_within.py": "solar",
           "verify_selection_cv_within.py": "selection_cv",
           "verify_negativecontrols_within.py": "negativecontrols"}
    _o_sites = []
    for _p in _sh:
        _prs = re.findall(
            rf"^python3 {_p.replace('.', r'\.')}[^\n]*?(\d+)/(\d+) gates PASS",
            block, re.M)
        _o_sites.append((f"§8 {_sh[_p]}", _prs, _p))
    _dp_s9 = re.findall(
        r"^\|\s*`verify_dataset_provenance\.py`[^\n]*?(\d+)/(\d+) gates PASS",
        sec9, re.M)
    _o_sites.append(("§9 dataset_provenance", _dp_s9,
                     "verify_dataset_provenance.py"))
    o_ok = all(len(prs) == 1
               and int(prs[0][0]) == _live[p] and int(prs[0][1]) == _live[p]
               for _, prs, p in _o_sites)
    o_detail = "; ".join(
        f"{lab} {prs[0][0] + '/' + prs[0][1] if len(prs) == 1 else ('missing' if not prs else 'multi')} "
        f"{'==' if len(prs) == 1 and int(prs[0][0]) == _live[p] and int(prs[0][1]) == _live[p] else '!='} "
        f"live {_live[p]}"
        for lab, prs, p in _o_sites)
    o_detail += f" ({len(_o_sites)} sites, single-match-each {'OK' if o_ok else 'FAIL'})"

    # (p) §9 Files-table GRAND-TOTAL "N/N total" numerals on the four SUMMARY-
    #     style probe rows (o) does not cover (o) matches only §8 lines + the
    #     §9 dataset_provenance row and its "gates PASS" token pattern): each
    #     of the four rows ends with a bold "N/N total" grand total that must
    #     equal that probe's live gate total from the SAME subprocess run as
    #     (o).  The four probes are exactly the SUMMARY-style ones in _live
    #     other than dataset_provenance (which prints no SUMMARY line and whose
    #     §9 row "11/11 gates PASS" token is already gated by (o)); fabricmass
    #     is NOT in _sh because (o)'s §8 line loop would then require a verdict
    #     token on §8 line 604, which has none.  The regex matches the literal
    #     "total" suffix so the many "X/Y gates PASS" / bare "X/Y" leg sub-total
    #     digits inside these same rows are NOT false positives; each row must
    #     carry EXACTLY ONE "N/N total" token (findall len==1 guard) so a
    #     missing grand total, a stale numeral, or a stray second token FAILs,
    #     never vacuous-passes.
    _p_probes = ("verify_fabricmass_within.py", "verify_solar_within.py",
                 "verify_selection_cv_within.py",
                 "verify_negativecontrols_within.py")
    _p_lab = {"verify_fabricmass_within.py": "fabricmass",
              "verify_solar_within.py": "solar",
              "verify_selection_cv_within.py": "selection_cv",
              "verify_negativecontrols_within.py": "negativecontrols"}
    _p_sites = []
    for _p in _p_probes:
        _prs = re.findall(
            rf"^\|\s*`{re.escape(_p)}`[^\n]*?(\d+)/(\d+) total",
            sec9, re.M)
        _p_sites.append((f"§9 {_p_lab[_p]}", _prs, _p))
    p_ok = all(len(prs) == 1
               and int(prs[0][0]) == _live[_p] and int(prs[0][1]) == _live[_p]
               for _, prs, _p in _p_sites)
    p_detail = "; ".join(
        f"{lab} {prs[0][0] + '/' + prs[0][1] if len(prs) == 1 else ('missing' if not prs else 'multi: ' + str(len(prs)))} "
        f"{'==' if len(prs) == 1 and int(prs[0][0]) == _live[_p] and int(prs[0][1]) == _live[_p] else '!='} "
        f"live {_live[_p]}"
        for lab, prs, _p in _p_sites)
    p_detail += f" ({len(_p_sites)} sites, single-match-each {'OK' if p_ok else 'FAIL'})"

    # (q) §9 verdict-total "N/N gates PASS" numerals on the seven plain
    #     (non-SUMMARY-style) probe rows (o)/(p) do not touch.  Each prints
    #     exactly ONE countable gates block in live stdout: a "gates ("
    #     header line, one "(roman) ... : PASS" line per gate, then a
    #     "verdict:" line — so the count IS derive-able read-only by RUNNING
    #     the probe.  The gate-line regex is tail-UNANCHORED (".*: PASS",
    #     no end anchor) because housezone_audit / horizontrade / metrictuple
    #     gate lines carry trailing "  (held …)" annotations AFTER the
    #     ": PASS"; the block is anchored at the "gates (" header so pre-
    #     header leg-B "…: PASS" lines (housezone_audit prints 13) are never
    #     counted.  v1handbook / v2method rows carry TWO "N/N gates PASS"
    #     tokens (an intermediate "N/N source-structural gates PASS" leg +
    #     the verdict total), so the comparison is last-total-token: >= 1
    #     token and the LAST one must equal the live count — zero tokens, a
    #     stale last token, or a stray later token FAILs, never vacuous-passes.
    _q_probes = ("verify_freerun_episode_within.py",
                 "verify_v1handbook_within.py", "verify_v2method_within.py",
                 "verify_housezone_audit.py", "verify_horizontrade_within.py",
                 "verify_horizonsplit_within.py", "verify_metrictuple_within.py")
    _q_lab = {"verify_freerun_episode_within.py": "freerun_episode",
              "verify_v1handbook_within.py": "v1handbook",
              "verify_v2method_within.py": "v2method",
              "verify_housezone_audit.py": "housezone_audit",
              "verify_horizontrade_within.py": "horizontrade",
              "verify_horizonsplit_within.py": "horizonsplit",
              "verify_metrictuple_within.py": "metrictuple"}
    _q_lines_re = re.compile(r"^\s*\(([ivxlcdm]+)\)\s+.*: PASS")
    _q_qs_re = re.compile(r"^\s*gates\s*\(")
    _q_verdict_re = re.compile(r"^\s*verdict\s*:")
    _q_counts = {}
    for _p in _q_probes:
        _r = subprocess.run([sys.executable, _p], cwd=_root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
        _lines = _r.stdout.splitlines()
        _start = next((i for i, ln in enumerate(_lines)
                       if _q_qs_re.match(ln)), None)
        if _start is None:
            _q_counts[_p] = -1  # missing gates header -> FAIL, not vacuous
            continue
        _end = next((i for i in range(_start + 1, len(_lines))
                     if _q_verdict_re.match(_lines[i])), None)
        if _end is None:
            _q_counts[_p] = -1  # missing verdict line -> FAIL
            continue
        _q_counts[_p] = sum(1 for ln in _lines[_start:_end]
                            if _q_lines_re.match(ln))
    _q_sites = []
    for _p in _q_probes:
        _toks = re.findall(
            rf"^\|\s*`{re.escape(_p)}`[^\n]*?(\d+)/(\d+) gates PASS",
            sec9, re.M)
        _q_sites.append((f"§9 {_q_lab[_p]}", _toks, _p))
    q_ok = all(len(toks) >= 1
               and int(toks[-1][0]) == _q_counts[p]
               and int(toks[-1][1]) == _q_counts[p]
               for _, toks, p in _q_sites)
    q_detail = "; ".join(
        f"{lab} {toks[-1][0] + '/' + toks[-1][1] if len(toks) >= 1 else 'missing'} "
        f"{'==' if len(toks) >= 1 and int(toks[-1][0]) == _q_counts[p] and int(toks[-1][1]) == _q_counts[p] else '!='} "
        f"live {_q_counts[p]} ({len(toks)} token(s))"
        for lab, toks, p in _q_sites)
    q_detail += f" ({len(_q_sites)} sites, last-token-each {'OK' if q_ok else 'FAIL'})"

    # (r) §1 / §2.3 dataset-provenance PROSE numerals in REPORT.md's body —
    #     the §1 Period/Structure bullets and the §2.3 Split bullet, NOT the
    #     §8/§9 catalog lines that gates (a)-(q) already cover — must re-derive
    #     read-only from data/energydata_complete.csv itself: the data-row
    #     count (header is the first row), the first/last `date` cells (the
    #     prose prints "17:00" WITHOUT seconds — the CSV cells and verify_
    #     dataset_provenance.py's literals carry "17:00:00", so the PROSE
    #     forms are matched verbatim), the calendar span as (last - first) in
    #     days (137; the inclusive cell count is 138), and the 70/30 split as
    #     int(rows x 0.70) / rows - that.  The prose writes thousands
    #     separators ("19,735 rows", "13,814 samples", "5,921 samples"); the
    #     bullet regexes are anchored at the "^**Period:** / **Structure:** /
    #     **Split:**" line forms so the §8 dataset_provenance machinery line's
    #     same tokens ("19,735 rows, 2016-01-11 17:00 → … (137 days)") cannot
    #     collide.  Every site must match EXACTLY once and equal the derived
    #     value — a stale digit, a missing bullet, or a stray second match
    #     FAILs, never vacuous-passes.
    _csv_path = _root / "data" / "energydata_complete.csv"
    _r_csv = []
    if _csv_path.is_file():
        with _csv_path.open(newline="") as _f:
            _rows_csv = list(csv.reader(_f))
        _r_csv = _rows_csv[1:]  # header is the first row
    _n_rows = len(_r_csv)
    _f_ymd = (datetime.strptime(_r_csv[0][0], "%Y-%m-%d %H:%M:%S")
              if _n_rows else None)
    _l_ymd = (datetime.strptime(_r_csv[-1][0], "%Y-%m-%d %H:%M:%S")
              if _n_rows else None)
    _span_days = (_l_ymd - _f_ymd).days if _f_ymd and _l_ymd else -1
    _step_s = ((datetime.strptime(_r_csv[1][0], "%Y-%m-%d %H:%M:%S")
                - _f_ymd).total_seconds() if _n_rows >= 2 else -1)
    _split70 = int(_n_rows * 0.70) if _n_rows else -1
    _held = _n_rows - _split70 if _n_rows else -1
    _f_prose = _f_ymd.strftime("%Y-%m-%d %H:%M") if _f_ymd else ""
    _l_prose = _l_ymd.strftime("%Y-%m-%d %H:%M") if _l_ymd else ""
    _r_prd = re.findall(
        r"^- \*\*Period:\*\* (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) → "
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \((\d+) days\)\.$", text, re.M)
    _r_str = re.findall(
        r"^- \*\*Structure:\*\* ([0-9,]+) rows, all at a uniform "
        r"10-minute \(([0-9,]+) s\) resolution\.$", text, re.M)
    _r_spl = re.findall(
        r"^- \*\*Split:\*\* first 70% \(([0-9,]+) samples\) train, "
        r"last 30% \(([0-9,]+) samples\) test.*$", text, re.M)
    _r_ok = (len(_r_prd) == 1 and _r_prd[0][0] == _f_prose
             and _r_prd[0][1] == _l_prose
             and int(_r_prd[0][2]) == _span_days
             and len(_r_str) == 1
             and int(_r_str[0][0].replace(",", "")) == _n_rows
             and int(_r_str[0][1].replace(",", "")) == int(_step_s)
             and len(_r_spl) == 1
             and int(_r_spl[0][0].replace(",", "")) == _split70
             and int(_r_spl[0][1].replace(",", "")) == _held)
    _r_detail = (f"CSV {_n_rows} rows, {_f_prose} → {_l_prose} ({_span_days} "
                 f"days), {int(_step_s)} s step, split {_split70}/{_held} vs "
                 f"prose Period "
                 f"{'/'.join(_r_prd[0]) if len(_r_prd) == 1 else 'missing'}, "
                 f"Structure "
                 f"{'/'.join(_r_str[0]) if len(_r_str) == 1 else 'missing'}, "
                 f"Split "
                 f"{'/'.join(_r_spl[0]) if len(_r_spl) == 1 else 'missing'}"
                 f" ({len(_r_prd)}+{len(_r_str)}+{len(_r_spl)} sites, "
                 f"single-match-each {'OK' if _r_ok else 'FAIL'})")

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

    # (m), (n), (o), (p), (q), (r), (s), (t) and (u) appended AFTER the dict:
    #     (m)'s
    #     "N/N
    #     gates
    #     PASS" numeral must equal the TOTAL gate count including all nine
    #     appends (len + 9 while the dict holds the a..l gates), and each of
    #     the two self-description lines must carry EXACTLY ONE "N/N gates
    #     PASS" token (findall — a missing site or a stray second token
    #     FAILs); (n) then gates the same two lines' PROSE numerals (50 probes / 45 items / 14
    #     programs / checks 53) against the derived counts; (o) gates the four
    #     neighbouring read-only probes' OWN verdict numerals against their
    #     subprocess-derived live gate totals (this probe's total is NOT used
    #     there — the §8 lines for solar/selection/negativecontrols and the
    #     §8 line + §9 row for dataset_provenance carry N/N tokens whose N is
    #     that probe's gate count, derived by running the probe); (p) gates
    #     the §9 Files-table GRAND-TOTAL "N/N total" numerals (with the literal
    #     "total" suffix, so the many "X/Y gates PASS"/bare "X/Y" leg sub-total
    #     digits inside those same rows are not matched) on the four SUMMARY-
    #     style probe rows (o) does not cover — fabricmass/solar/
    #     selection_cv/negativecontrols — against the same live totals;
    #     (r) gates the §1/§2.3 dataset-provenance prose numerals against a
    #     read-only derivation from data/energydata_complete.csv.
    n_gates = len(gates) + 9
    self_all = self_s8 + self_s9
    self_parity = (self_all
                   and all(num == den for num, den in self_all)
                   and all(int(num) == n_gates for num, _ in self_all))
    self_ok = self_parity
    gates["(m) self-description 'N/N gates PASS' literals == derived gate total"] = (
        self_ok,
        f"§8 line {(self_s8[0][0] + '/' + self_s8[0][1]) if self_s8 else 'missing'}, "
        f"§9 row {(self_s9[0][0] + '/' + self_s9[0][1]) if self_s9 else 'missing'} "
        f"== derived {n_gates} gates "
        f"({len(self_all)} token(s), N/N parity "
        f"{'OK' if self_parity else 'FAIL'})",
    )
    gates["(n) self-description prose numerals == derived totals"] = (
        prose_ok,
        f"§8 probes {s8_probes[0] if len(s8_probes) == 1 else '?'} == {n}, "
        f"items {s8_items[0] if len(s8_items) == 1 else '?'} == {n_items} "
        f"(range 1..{s8_range[0][1] if len(s8_range) == 1 else '?'}), "
        f"programs {s8_progs[0] if len(s8_progs) == 1 else '?'} == {n_progs}; "
        f"§9 probes {s9_probes[0] if len(s9_probes) == 1 else '?'} == {n}, "
        f"items {s9_items[0] if len(s9_items) == 1 else '?'} == {n_items} "
        f"(range 1..{s9_range[0][1] if len(s9_range) == 1 else '?'}), "
        f"programs {s9_progs[0] if len(s9_progs) == 1 else '?'} == {n_progs}, "
        f"breakdown {'/'.join(s9_break[0]) if s9_break else 'missing'} "
        f"== checks {n_checks} = {n_selftests} self-tests + {n} probes "
        f"(9 sites, single-match-each {'OK' if prose_ok else 'FAIL'})",
    )
    gates["(o) neighbouring probes' verdict numerals == their live totals"] = (
        o_ok,
        o_detail,
    )
    gates["(p) §9 grand-total 'N/N total' numerals == live totals"] = (
        p_ok,
        p_detail,
    )
    gates["(q) §9 plain-probe verdict-total numerals == gate-line counts"] = (
        q_ok,
        q_detail,
    )
    gates["(r) §1/§2.3 dataset-provenance prose numerals == derived from CSV"] = (
        _r_ok,
        _r_detail,
    )

    # (s) the §7 residual-diagnostics BODY-PROSE verdict-total numerals (the
    #     six summary bullets closing with an overall "N/N gates PASS": §7
    #     items 32/33/34/35/44/45 at REPORT.md lines 450/457/464/471/539/548)
    #     must equal the citing probe's LIVE gate-line count.  Four probes
    #     are block-style ((q) format: a "gates (" header, one "(roman)
    #     ... : MARK" line per gate, a "verdict:" line) and are subprocess-
    #     derived here like (q) — but the pass-count uses a ": PASS"-tail
    #     matcher while the TOTAL uses a ": (PASS|FAIL)" matcher, because
    #     marginthin's gates ii/iii are FAIL lines: a ": PASS"-only matcher
    #     would wrongly derive marginthin as 1/1 when its honest verdict is
    #     1/3.  housezone_audit / horizontrade / metrictuple gate lines
    #     carry trailing "  (held …)" annotations, so the block matcher is
    #     tail-UNANCHORED as in (q).  The two SUMMARY-style probes
    #     (selection_cv, negativecontrols) print no gate block, so their
    #     pass==total reuses the SAME live totals gate (o) derived by
    #     running them.  Each report site is a ^-anchored bullet-form regex
    #     with disambiguating prose context (so the identical §8/§9
    #     machine-line tokens cannot collide) that must match EXACTLY once
    #     and equal (pass_count, total_count) — a stale numeral, a missing
    #     bullet, or a stray second match FAILs, never vacuous-passes.
    _s_probes = ("verify_housezone_audit.py", "verify_horizontrade_within.py",
                 "verify_marginthin_within.py", "verify_metrictuple_within.py")
    _s_lab = {"verify_housezone_audit.py": "housezone_audit",
              "verify_horizontrade_within.py": "horizontrade",
              "verify_marginthin_within.py": "marginthin",
              "verify_metrictuple_within.py": "metrictuple",
              "verify_selection_cv_within.py": "selection_cv",
              "verify_negativecontrols_within.py": "negativecontrols"}
    _s_pass_re = re.compile(r"^\s*\(([ivxlcdm]+)\)\s+.*: PASS")
    _s_any_re = re.compile(r"^\s*\(([ivxlcdm]+)\)\s+.*: (PASS|FAIL)")
    _s_counts = {}
    for _p in _s_probes:
        _r = subprocess.run([sys.executable, _p], cwd=_root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
        _lines = _r.stdout.splitlines()
        _start = next((i for i, ln in enumerate(_lines)
                       if _q_qs_re.match(ln)), None)
        if _start is None:
            _s_counts[_p] = (-1, -1)  # missing gates header -> FAIL, not vacuous
            continue
        _end = next((i for i in range(_start + 1, len(_lines))
                     if _q_verdict_re.match(_lines[i])), None)
        if _end is None:
            _s_counts[_p] = (-1, -1)  # missing verdict line -> FAIL
            continue
        _blk = _lines[_start:_end]
        _s_counts[_p] = (sum(1 for ln in _blk if _s_pass_re.match(ln)),
                         sum(1 for ln in _blk if _s_any_re.match(ln)))
    for _p in ("verify_selection_cv_within.py",
               "verify_negativecontrols_within.py"):
        _s_counts[_p] = (_live[_p], _live[_p])
    _s_rx = {
        "verify_housezone_audit.py":
            r"^- \*\*Within-train val-block leg \(fresh fits, 20 days, "
            r"n = 2743 resync steps \+ 2762 free-run steps\):[^\n]*?"
            r"(\d+)/(\d+) gates PASS",
        "verify_horizontrade_within.py":
            r"^- \*\*Within-train val-block leg \(fresh fits, 20 days, "
            r"n = 2743\):\*\* the three-config band table[^\n]*?"
            r"(\d+)/(\d+) gates PASS",
        "verify_marginthin_within.py":
            r"^- \*\*Within-train val-block leg \(fresh fits, 20 days, "
            r"n = 2743\):\*\* v2 reference mean \|e\| 0\.4277[^\n]*?"
            r"(\d+)/(\d+) gates PASS",
        "verify_metrictuple_within.py":
            r"^- \*\*Within-train val-block leg \(fresh fits, 20 days, "
            r"n = 2743 resync / 2762 freerun\):[^\n]*?"
            r"(\d+)/(\d+) gates PASS",
        "verify_selection_cv_within.py":
            r"^\*\*Bottom line:\*\*[^\n]*?rather than a gateable process "
            r"claim \((\d+)/(\d+) gates PASS\)",
        "verify_negativecontrols_within.py":
            r'^\*\*Bottom line:\*\*[^\n]*?"irreducible reference" label '
            r"disclosed as an interpretation, not gated "
            r"\((\d+)/(\d+) gates PASS\)",
    }
    _s_sites = []
    for _p in _s_counts:
        _toks = re.findall(_s_rx[_p], text, re.M)
        _s_sites.append((f"§7 {_s_lab[_p]}", _toks, _p))
    s_ok = all(len(toks) == 1
               and (int(toks[0][0]), int(toks[0][1])) == _s_counts[p]
               for _, toks, p in _s_sites)
    s_detail = "; ".join(
        f"{lab} {toks[0][0] + '/' + toks[0][1] if len(toks) == 1 else ('missing' if not toks else 'multi: ' + str(len(toks)))} "
        f"{'==' if len(toks) == 1 and (int(toks[0][0]), int(toks[0][1])) == _s_counts[p] else '!='} "
        f"live {_s_counts[p][0]}/{_s_counts[p][1]}"
        for lab, toks, p in _s_sites)
    s_detail += f" ({len(_s_sites)} sites, single-match-each {'OK' if s_ok else 'FAIL'})"
    gates["(s) §7 residual-diagnostics verdict-total prose numerals == live gate-line counts"] = (
        s_ok,
        s_detail,
    )

    # (t) the §7/§9 "N/N fresh gates PASS" verdict-total prose numerals —
    #     the within-train FRESH-leg totals of the two block-style probes
    #     whose gates block also carries a frozen audit leg (item 36
    #     rhocollapse §7 body + §9 row, item 37 episode §7 body): the report
    #     discriminates the fresh leg ("N/N fresh gates PASS") from the
    #     frozen leg ("6/6 / 10/10 PASS" bare audit-total tokens — a separate
    #     literal class, gated by (u) below).  Fresh = the "gates ("-to-
    #     "verdict:" block lines marked PASS (denominator PASS|FAIL, as in
    #     (s)) whose text does NOT contain "frozen": for rhocollapse that
    #     excludes gate (iv) "frozen decomposition re-derives" (fresh 3 of 4
    #     gate lines), for episode no gate text is frozen (fresh 5).
    #     rhocollapse is in no earlier gate's probe list and the word-"fresh"
    #     token form does not match any existing "N/N gates PASS" census
    #     regex, so before (t) a probe-gate change without the report bump
    #     could not FAIL — now it FAILs, never vacuous-passes.
    _t_probes = ("verify_freerun_rhocollapse_within.py",
                 "verify_freerun_episode_within.py")
    _t_lab = {"verify_freerun_rhocollapse_within.py": "rhocollapse",
              "verify_freerun_episode_within.py": "freerun_episode"}
    _t_counts = {}
    for _p in _t_probes:
        _r = subprocess.run([sys.executable, _p], cwd=_root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
        _lines = _r.stdout.splitlines()
        _start = next((i for i, ln in enumerate(_lines)
                       if _q_qs_re.match(ln)), None)
        if _start is None:
            _t_counts[_p] = (-1, -1)  # missing gates header -> FAIL
            continue
        _end = next((i for i in range(_start + 1, len(_lines))
                     if _q_verdict_re.match(_lines[i])), None)
        if _end is None:
            _t_counts[_p] = (-1, -1)  # missing verdict line -> FAIL
            continue
        _blk = _lines[_start:_end]
        _t_counts[_p] = (sum(1 for ln in _blk
                             if _s_pass_re.match(ln) and "frozen" not in ln),
                         sum(1 for ln in _blk
                             if _s_any_re.match(ln) and "frozen" not in ln))
    _t_rx = {
        "verify_freerun_rhocollapse_within.py":
            r"^- \*\*Within-train val-block leg \(fresh fits, 20 days, "
            r"n = 2762 open-loop steps\):\*\* the collapse direction[^\n]*?"
            r"(\d+)/(\d+) fresh gates PASS",
        "verify_freerun_episode_within.py":
            r"^- \*\*Within-train val-block leg \(fresh fits, 20 days, "
            r"n = 2762 open-loop steps\):\*\* the episode geometry[^\n]*?"
            r"(\d+)/(\d+) fresh gates PASS",
    }
    _t_sites = []
    for _p in _t_probes:
        _toks = re.findall(_t_rx[_p], text, re.M)
        _t_sites.append((f"§7 body {_t_lab[_p]}", _toks, _p))
    _toks = re.findall(r"^\|\s*`verify_freerun_rhocollapse_within\.py`"
                       r"[^\n]*?(\d+)/(\d+) fresh gates PASS", sec9, re.M)
    _t_sites.append(("§9 row rhocollapse", _toks,
                     "verify_freerun_rhocollapse_within.py"))
    t_ok = all(len(toks) == 1
               and (int(toks[0][0]), int(toks[0][1])) == _t_counts[p]
               for _, toks, p in _t_sites)
    t_detail = "; ".join(
        f"{lab} {toks[0][0] + '/' + toks[0][1] if len(toks) == 1 else ('missing' if not toks else 'multi: ' + str(len(toks)))} "
        f"{'==' if len(toks) == 1 and (int(toks[0][0]), int(toks[0][1])) == _t_counts[p] else '!='} "
        f"live fresh {_t_counts[p][0]}/{_t_counts[p][1]}"
        for lab, toks, p in _t_sites)
    t_detail += f" ({len(_t_sites)} sites, single-match-each {'OK' if t_ok else 'FAIL'})"
    gates["(t) §7/§9 fresh-leg 'N/N fresh gates PASS' prose numerals == live fresh gate-line counts"] = (
        t_ok,
        t_detail,
    )

    # (u) the §7/§8/§9 FROZEN-leg bare "N/N PASS" audit-total prose
    #     numerals — the plain "N/N PASS"-tailed lines of the three
    #     block-style probes' frozen audit legs (freerun_half 14/14 at §7
    #     body lines 446-447, rhocollapse 6/6 at §7 body line 480 + §9 row
    #     line 668, episode 10/10 at §7 body line 487 + §8 machinery
    #     code-block line 600 + §9 row line 672): the report discriminates
    #     the frozen leg ("6/6 / 10/10 PASS" bare audit-total tokens, gated
    #     here) from the fresh leg ("N/N fresh gates PASS", gated by (t)).
    #     Fresh = the probe stdout slice from its "leg B" header to the
    #     "gates (" header, counting ": PASS" item lines by UNANCHORED
    #     ":\s*PASS" search (denominator ":\s*PASS" | ":\s*FAIL") —
    #     unanchored rather than tail-anchored because episode's audit
    #     block carries a mid-line parenthetical after ": PASS"
    #     ("committed six episode days == ... : PASS  (committed [...])")
    #     that would drop its tail-anchored count to 9; no "frozen"-text
    #     exclusion is needed because the audit blocks carry no other gated
    #     line class.
    _u_probes = ("verify_freerun_half_within.py",
                 "verify_freerun_rhocollapse_within.py",
                 "verify_freerun_episode_within.py")
    _u_lab = {"verify_freerun_half_within.py": "freerun_half",
              "verify_freerun_rhocollapse_within.py": "rhocollapse",
              "verify_freerun_episode_within.py": "freerun_episode"}
    _u_leg_re = re.compile(r"^\s*leg\s+B")
    _u_counts = {}
    for _p in _u_probes:
        _r = subprocess.run([sys.executable, _p], cwd=_root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800)
        _lines = _r.stdout.splitlines()
        _start = next((i for i, ln in enumerate(_lines)
                       if _u_leg_re.match(ln)), None)
        if _start is None:
            _u_counts[_p] = (-1, -1)  # missing leg-B header -> FAIL
            continue
        _end = next((i for i in range(_start + 1, len(_lines))
                     if _q_qs_re.match(_lines[i])), None)
        if _end is None:
            _u_counts[_p] = (-1, -1)  # missing gates header -> FAIL
            continue
        _blk = _lines[_start:_end]
        _u_counts[_p] = (sum(1 for ln in _blk
                             if re.search(r":\s*PASS", ln)),
                         sum(1 for ln in _blk
                             if re.search(r":\s*(PASS|FAIL)", ln)))
    _u_rx = {
        "verify_freerun_half_within.py": [
            r"^- \*\*Gate honesty:\*\*[^\n]*?leg B (\d+)/(\d+) PASS",
            r"^\*\*Bottom line:\*\*[^\n]*?(\d+)/(\d+) PASS: pooled",
        ],
        "verify_freerun_rhocollapse_within.py": [
            r"^- \*\*Held-out audit leg \(frozen artifact, n = 5920, "
            r"read-only\):\*\*[^\n]*?decomposition digit[^\n]*?"
            r"(\d+)/(\d+) PASS",
        ],
        "verify_freerun_episode_within.py": [
            r"^- \*\*Held-out audit leg \(frozen artifact, n = 5920, "
            r"read-only\):\*\*[^\n]*?episode digit[^\n]*?"
            r"(\d+)/(\d+) PASS",
        ],
    }
    _u_sites = []
    for _p in _u_probes:
        for _rx in _u_rx[_p]:
            _toks = re.findall(_rx, text, re.M)
            _u_sites.append((f"§7 body {_u_lab[_p]}", _toks, _p))
    _toks = re.findall(r"^python3 verify_freerun_episode_within\.py"
                       r"[^\n]*?(\d+)/(\d+) PASS", block, re.M)
    _u_sites.append(("§8 line freerun_episode", _toks,
                     "verify_freerun_episode_within.py"))
    _toks = re.findall(r"^\|\s*`verify_freerun_rhocollapse_within\.py`"
                       r"[^\n]*?(\d+)/(\d+) PASS", sec9, re.M)
    _u_sites.append(("§9 row rhocollapse", _toks,
                     "verify_freerun_rhocollapse_within.py"))
    _toks = re.findall(r"^\|\s*`verify_freerun_episode_within\.py`"
                       r"[^\n]*?(\d+)/(\d+) PASS", sec9, re.M)
    _u_sites.append(("§9 row freerun_episode", _toks,
                     "verify_freerun_episode_within.py"))
    u_ok = all(len(toks) == 1
               and (int(toks[0][0]), int(toks[0][1])) == _u_counts[p]
               for _, toks, p in _u_sites)
    u_detail = "; ".join(
        f"{lab} {toks[0][0] + '/' + toks[0][1] if len(toks) == 1 else ('missing' if not toks else 'multi: ' + str(len(toks)))} "
        f"{'==' if len(toks) == 1 and (int(toks[0][0]), int(toks[0][1])) == _u_counts[p] else '!='} "
        f"live {_u_counts[p][0]}/{_u_counts[p][1]}"
        for lab, toks, p in _u_sites)
    u_detail += f" ({len(_u_sites)} sites, single-match-each {'OK' if u_ok else 'FAIL'})"
    gates["(u) §7/§8/§9 frozen-leg bare 'N/N PASS' audit-total prose numerals == live frozen audit-block PASS-line counts"] = (
        u_ok,
        u_detail,
    )

    print("§8/§9 report-inventory consistency probe (deterministic read-only"
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
    print(f"    self-description 'N/N gates PASS' tokens gated by (m): {len(self_all)} "
          f"({' + '.join(num + '/' + den for num, den in self_all) or 'none'}) "
          f"== derived {n_gates} gates")
    n_sites = (len(s8_probes) + len(s9_probes) + len(s8_items) + len(s9_items)
               + len(s8_range) + len(s9_range) + len(s8_progs) + len(s9_progs)
               + len(s9_break))
    print(f"    self-description prose numerals gated by (n): {n_sites} sites "
          f"(probes {s8_probes[0] if len(s8_probes) == 1 else '?'} §8 / "
          f"{s9_probes[0] if len(s9_probes) == 1 else '?'} §9, items "
          f"{s8_items[0] if len(s8_items) == 1 else '?'}/"
          f"{s9_items[0] if len(s9_items) == 1 else '?'} == {n_items} "
          f"(ranges 1..{s8_range[0][1] if len(s8_range) == 1 else '?'} / "
          f"1..{s9_range[0][1] if len(s9_range) == 1 else '?'}), programs "
          f"{s8_progs[0] if len(s8_progs) == 1 else '?'}/"
          f"{s9_progs[0] if len(s9_progs) == 1 else '?'} == {n_progs}, §9 "
          f"breakdown {'/'.join(s9_break[0]) if s9_break else 'missing'} "
          f"== checks {n_checks} = {n_selftests} self-tests + {n} probes) "
          f": {'OK' if prose_ok else 'FAIL'}")
    print(f"    neighbouring-probe verdict numerals gated by (o): "
          f"{len(_o_sites)} sites (live totals by subprocess run: "
          + ", ".join(f"{_sh[_p]} {_live[_p]}" for _p in _sh)
          + f") : {'OK' if o_ok else 'FAIL'}")
    print(f"    §9 grand-total 'N/N total' numerals gated by (p): "
          f"{len(_p_sites)} sites "
          + ", ".join(f"{_p_lab[_p]} {_live[_p]}"
                      for _p in _p_probes)
          + f") : {'OK' if p_ok else 'FAIL'}")
    print(f"    §9 plain-probe verdict-total numerals gated by (q): "
          f"{len(_q_sites)} sites "
          + ", ".join(f"{_q_lab[_p]} {_q_counts[_p]}"
                      for _p in _q_probes)
          + f") : {'OK' if q_ok else 'FAIL'}")
    print(f"    §1/§2.3 dataset-provenance prose numerals gated by (r): "
          f"CSV {_n_rows} rows, {_f_prose} → {_l_prose} "
          f"({_span_days} days), {int(_step_s)} s step, split "
          f"{_split70}/{_held} : {'OK' if _r_ok else 'FAIL'}")
    print(f"    §7 residual-diagnostics verdict-total prose numerals gated "
          f"by (s): {len(_s_sites)} sites "
          + ", ".join(f"{_s_lab[_p]} {_s_counts[_p][0]}/{_s_counts[_p][1]}"
                      for _p in _s_counts)
          + f") : {'OK' if s_ok else 'FAIL'}")
    print(f"    §7/§9 fresh-leg 'N/N fresh gates PASS' verdict-total prose "
          f"numerals gated by (t): {len(_t_sites)} sites "
          + ", ".join(f"{_t_lab[_p]} {_t_counts[_p][0]}/{_t_counts[_p][1]}"
                      for _p in _t_probes)
          + f") : {'OK' if t_ok else 'FAIL'}")
    print(f"    §7/§8/§9 frozen-leg bare 'N/N PASS' audit-total prose "
          f"numerals gated by (u): {len(_u_sites)} sites "
          + ", ".join(f"{_u_lab[_p]} {_u_counts[_p][0]}/{_u_counts[_p][1]}"
                      for _p in _u_probes)
          + f") : {'OK' if u_ok else 'FAIL'}")
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
              f"cannot drift from the derived count, the four neighbouring "
              f"read-only probes' own verdict numerals re-verify against "
              f"their live gate totals (dataset_provenance "
              f"{_live['verify_dataset_provenance.py']}, solar "
              f"{_live['verify_solar_within.py']}, selection_cv "
              f"{_live['verify_selection_cv_within.py']}, negativecontrols "
              f"{_live['verify_negativecontrols_within.py']}), the §9 "
              f"Files-table grand-total 'N/N total' numerals on the four "
              f"SUMMARY-style probe rows re-verify against the same live "
              f"totals (fabricmass {_live['verify_fabricmass_within.py']}, "
              f"same solar/selection_cv/negativecontrols), the §9 "
              f"verdict-total 'N/N gates PASS' numerals on the seven plain "
              f"probe rows re-verify against their live gate-line counts "
              f"(freerun_episode {_q_counts['verify_freerun_episode_within.py']}, "
              f"v1handbook {_q_counts['verify_v1handbook_within.py']}, "
              f"v2method {_q_counts['verify_v2method_within.py']}, "
              f"housezone_audit {_q_counts['verify_housezone_audit.py']}, "
              f"horizontrade {_q_counts['verify_horizontrade_within.py']}, "
              f"horizonsplit {_q_counts['verify_horizonsplit_within.py']}, "
              f"metrictuple {_q_counts['verify_metrictuple_within.py']}), "
              f"and the §1/§2.3 dataset-provenance prose numerals re-verify "
              f"against a read-only derivation from "
              f"data/energydata_complete.csv ({_n_rows} rows, "
              f"{_f_prose} → {_l_prose} ({_span_days} days, "
              f"{int(_step_s)} s step, split {_split70}/{_held}), "
              f"the six §7 residual-diagnostics body-prose verdict-total "
              f"numerals re-verify against the citing probes' live gate-line "
              f"counts (housezone_audit "
              f"{_s_counts['verify_housezone_audit.py'][0]}/"
              f"{_s_counts['verify_housezone_audit.py'][1]}, horizontrade "
              f"{_s_counts['verify_horizontrade_within.py'][0]}/"
              f"{_s_counts['verify_horizontrade_within.py'][1]}, marginthin "
              f"{_s_counts['verify_marginthin_within.py'][0]}/"
              f"{_s_counts['verify_marginthin_within.py'][1]}, metrictuple "
              f"{_s_counts['verify_metrictuple_within.py'][0]}/"
              f"{_s_counts['verify_metrictuple_within.py'][1]}, selection_cv "
              f"{_s_counts['verify_selection_cv_within.py'][0]}/"
              f"{_s_counts['verify_selection_cv_within.py'][1]}, "
              f"negativecontrols "
              f"{_s_counts['verify_negativecontrols_within.py'][0]}/"
              f"{_s_counts['verify_negativecontrols_within.py'][1]}), "
              f"the §7/§9 fresh-leg 'N/N fresh gates PASS' numerals re-verify "
              f"against the same probes' live fresh gate-line counts"
              f" (rhocollapse "
              f"{_t_counts['verify_freerun_rhocollapse_within.py'][0]}/"
              f"{_t_counts['verify_freerun_rhocollapse_within.py'][1]}, "
              f"freerun_episode "
              f"{_t_counts['verify_freerun_episode_within.py'][0]}/"
              f"{_t_counts['verify_freerun_episode_within.py'][1]}), "
              f"and the §7/§8/§9 frozen-leg bare 'N/N PASS' audit-total "
              f"numerals re-verify against the same probes' live frozen "
              f"audit-block PASS-line counts"
              f" (freerun_half "
              f"{_u_counts['verify_freerun_half_within.py'][0]}/"
              f"{_u_counts['verify_freerun_half_within.py'][1]} in §7 body, "
              f"rhocollapse "
              f"{_u_counts['verify_freerun_rhocollapse_within.py'][0]}/"
              f"{_u_counts['verify_freerun_rhocollapse_within.py'][1]} in "
              f"§7 body and §9 row, freerun_episode "
              f"{_u_counts['verify_freerun_episode_within.py'][0]}/"
              f"{_u_counts['verify_freerun_episode_within.py'][1]} in §7 "
              f"body, §8 machinery line and §9 row), "
              f"and REPORT.md's own "
              f"description of this probe ({n_gates}/{n_gates} gates PASS "
              f"on both its §8 machinery line and its §9 row, whose prose "
              f"numerals — {n} probes, {n_items} §7 items, {n_progs} "
              f"non-verify programs, checks {n_checks} == {n_selftests} "
              f"self-tests + {n} probes — all match the derived totals) "
              f"re-verifies against the derived gate total.")
    else:
        print(f"verdict: FAIL — the following committed inventory invariant(s) "
              f"did NOT re-verify: "
              + ", ".join(name for name, (ok, _) in gates.items() if not ok)
              + " (honest negative, reported as-is per the §7 items 31/34 "
                "precedent)")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
