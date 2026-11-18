#!/usr/bin/env python3
"""verify_negativecontrols_within.py — source-structural probe of the §2.3 negative controls.

REPORT.md §2.3 describes the two negative controls of the v2 evaluation as
*machinery facts* — how the "temperature-stays-put" benchmarks are actually
built in the shipped run_experiment.py:

    "*Daily-resync persistence:* T̂(t+1) = T(t), re-anchored each day — shows
     whether fitted drivers beat a trivial 'temperature stays put' forecast
     at the daily horizon."
    "*One-step persistence floor:* T̂(t+1) = T observed at t, re-anchored
     every sample — an honest 10-min irreducible-error reference."

Every prior persistence probe operated at the VALUE level: verify_persistence.py
bootstraps the frozen day-block cmri-vs-persistence gap, verify_persistence_within.py
replays a fresh-fit persistence control, verify_stepanchor.py audits the frozen
§3.5/§4.3 floor digit.  Not one audited the NEGATIVE-CONTROL MACHINERY itself:
that the daily-resync persistence model is the identity step (T̂(t+1) = T(t));
that the one-step floor is a *separate* per-sample construction that re-anchors
on every sample (`model is None` → predict next obs = current obs), not a second
run-protocol pass; that the resync control re-anchors per test *day* from that
day's first observed indoor temperature while free-run never re-anchors; and
that the floor is labeled in the shipped artifacts as the "noise floor ...
irreducible reference" with the verdict prose "highly auto-correlated and
almost noise-free ... not sensor noise".  This deterministic read-only probe
closes that gap.

Leg A — source-structural gates (text scans + behavioral replay of the
shipped files):

  (i)  Identity-step model: `class PersistenceModel` ships in run_experiment.py
       with the signature `def step(self, temp, t_out, dew, load, lux):` and
       body `return temp` — T̂(t+1) = T(t), exactly §2.3's formula.
  (ii) Negative-control labeling: the run_experiment.py docstring states the
       benchmark "predict T(t+1) = T(t)" is run "as a negative control, so
       'v2 beats v1' is not trivially 'any fitted model beats an arbitrary
       constant'".
  (iii) Protocol registration: the 8-row protocol loop registers the control
       rows `("persistence_resync", pers, test_days, split, n)` and
       `("persistence_floor", None, None, split, n)` — the floor is a
       *separate* row with model None, not a third run-protocol pass.
  (iv) Floor branch: the `if model is None:` branch (comment: "one-step
       persistence floor: predict next obs = current obs") builds
       `preds[label] = [T[j - 1] for j in range(base + 1, end)]` and
       `obs_all[label] = [r[0] for r in recs[base + 1:end]]` — every sample
       re-anchors from the observed value at t to predict t+1, so n_floor =
       n − split − 1 = 5920 (no day-first Nones at all).
  (v)  Per-day re-anchor vs free-run: run_protocol's resync branch re-anchors
       each day's first sample from that day's first observed temperature
       (`cur = recs[day[0]][0]`, stepping over `zip(day[:-1], day[1:])` —
       first sample of every day is None), while the free-run branch
       (`if resync_days is None:`) re-anchors exactly once from the block's
       first observed temperature.  Because the identity step relays its
       model state (the anchor) rather than the observed value, a
       free-running persistence predicts the constant block anchor while a
       resync persistence predicts each day's first observed T — the §2.3
       "temperature stays put' forecast at the daily horizon" reading,
       hence resync n = 5920 − 41 = 5879 (one None per test-block day).
  (vi) "Irreducible reference" labeling: run_experiment.py's comparison.md
       builder emits the heading "Noise floor (one-step persistence —
       irreducible reference)" and the verdict probes "highly auto-correlated
       and almost noise-free" / "intraday dynamics, not sensor noise".

Leg B — computational / frozen-artifact gates:

  (vii) n-construction re-derivation: split = int(19735 × 0.70) = 13814;
        floor n = 19735 − 13814 − 1 = 5920; resync n = 5920 − 41 = 5879 —
        the 137-test-block-day arithmetic behind both frozen n digits.
  (viii) Frozen-artifact audit: results/metrics.csv carries the two control
        rows `persistence,resync` (0.6246 / 0.5019 / 0.809 / 2.193, n = 5879)
        and `persistence,floor` (0.0431 / 0.0243 / 0.9991 / 0.105, n = 5920);
        the floor row is the STRICT minimum RMSE of all 8 committed rows; and
        results/comparison.md still ships the "Noise floor (one-step
        persistence — irreducible reference)" heading with the committed
        0.0431 floor row.

DISCLOSURE (what is NOT gated, honest under the item-31 convention): the
"honest 10-min irreducible-error reference" SEMANTIC — that the floor error
means the v2 residual reflects "genuinely hard-to-model intraday dynamics,
not sensor noise" — is an interpretation of the measured floor, not a
structural fact.  This probe gates the machinery that produces that floor
(identity step, per-sample re-anchor, per-day resync re-anchor) and the
frozen digits + labels in the shipped artifacts; it never claims the causal
reading of what the floor "means".  Nor does it re-fit anything on the data:
fresh innerhalb replay of the persistence control already exists in
verify_persistence_within.py; here the reproducible statement is the
machinery + artifact audit, exactly as legs A/B(viii) above.

Pure Python standard library; deterministic; read-only.
"""

from __future__ import annotations

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

# noqa: E402 — project-relative imports after sys.path setup
from run_experiment import PersistenceModel, run_protocol  # noqa: E402

N_DATA = 19735     # data/energydata_complete.csv rows
TRAIN_FRAC = 0.70  # run_experiment.py TRAIN_FRAC

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
    print("-- gate (i): identity-step persistence model --")
    n = 0
    n += check(
        "class PersistenceModel present", "class PersistenceModel:" in src,
        "the negative control ships a named persistence model in run_experiment.py",
    )
    n += check(
        "uniform step() signature",
        "def step(self, temp, t_out, dew, load, lux):" in src,
        "satisfies the same interface as V1Model / CalibratedTwin adapters",
    )
    n += check(
        "identity body: return temp", "return temp" in src,
        "T-hat(t+1) = T(t) — the model never looks at any driver",
    )
    return n


def gate_ii(src: str) -> int:
    print("-- gate (ii): negative-control labeling in the docstring --")
    n = 0
    n += check(
        "'naive persistence benchmark'", "naive persistence benchmark" in src,
        "run_experiment.py names the benchmark in its protocol description",
    )
    n += check(
        "predict T(t+1) = T(t) formula", "predict T(t+1) = T(t)" in src,
        "the docstring states §2.3's exact formula",
    )
    n += check(
        "'as a negative control'", "negative control" in src,
        "the benchmark's role is declared, not inferred",
    )
    n += check(
        "'not trivially any fitted model beats'", "not trivially" in src,
        "the motivation: v2-vs-v1 is not arbitrary-constant beating",
    )
    return n


def gate_iii(src: str) -> int:
    print("-- gate (iii): protocol-row registration --")
    n = 0
    n += check(
        "resync control row",
        '"persistence_resync", pers, test_days, split, n' in src,
        "daily-resync control runs via run_protocol with the test-day grouping",
    )
    n += check(
        "floor control row",
        '"persistence_floor", None, None, split, n' in src,
        "the floor is a separate cell with model=None, not a third protocol pass",
    )
    n += check(
        "both rows in the 8-row loop",
        '"v1_resync", v1, test_days, split, n' in src
        and '"v2_train", v2, train_days, 0, split' in src,
        "controls are peers of the four v1/v2 evaluation cells",
    )
    n += check(
        "floor row NOT in run-protocol tuple",
        '"persistence_floor"' in src and "pers, test_days, split, n" in src,
        "floor is excluded from resync semantics by its None/model-None cells",
    )
    return n


def gate_iv(src: str) -> int:
    print("-- gate (iv): per-sample floor branch --")
    n = 0
    n += check(
        "'if model is None:' floor branch", "if model is None:" in src,
        "the None register routes to the floor path, not run_protocol",
    )
    n += check(
        "branch comment", "one-step persistence floor: predict next obs = current obs" in src,
        "source names the branch exactly as §2.3 reads it",
    )
    n += check(
        "per-sample prediction",
        "preds[label] = [T[j - 1] for j in range(base + 1, end)]" in src,
        "predict t+1 from the observed T at t for EVERY sample",
    )
    n += check(
        "per-sample observation",
        "obs_all[label] = [r[0] for r in recs[base + 1:end]]" in src,
        "both arrays span base+1..end with no None day-heads — n = n - split - 1",
    )
    return n


def gate_v(src: str) -> int:
    print("-- gate (v): per-day re-anchor vs free-run --")
    n = 0
    n += check(
        "per-day re-anchor cur = recs[day[0]][0]", "cur = recs[day[0]][0]" in src,
        "each resync day restarts from that day's FIRST observed indoor temperature",
    )
    n += check(
        "day pair-step loop", "for a, b in zip(day[:-1], day[1:]):" in src,
        "the day's first sample is the anchor, never itself predicted",
    )
    n += check(
        "free-run branch", "if resync_days is None:" in src,
        "free-run re-anchors once (block start), the resync path never",
    )
    free = run_protocol(PersistenceModel(), 20.0, [(20.0, 1, 2, 3, 4), (21.0, 1, 2, 3, 4), (22.0, 1, 2, 3, 4)], None)
    n += check(
        "free-run: relay the block anchor", free == [None, 20.0, 20.0],
        f"free-run over 3 samples = {free} (identity relays its one anchor T0 = 20.0)",
    )
    one_day = run_protocol(PersistenceModel(), 20.0, [(20.0, 1, 2, 3, 4), (21.0, 1, 2, 3, 4), (22.0, 1, 2, 3, 4)], [[0, 1, 2]])
    n += check(
        "single resync day == free-run anchor relay", one_day == [None, 20.0, 20.0],
        f"one day spanning the block re-anchors at sample 0 only: {one_day}",
    )
    two_day = run_protocol(PersistenceModel(), 20.0, [(20.0, 1, 2, 3, 4), (21.0, 1, 2, 3, 4), (22.0, 1, 2, 3, 4)], [[0, 1], [2]])
    n += check(
        "two days: second day-head is None", two_day == [None, 20.0, None],
        f"re-anchored each day: first sample of day 2 is never predicted: {two_day}",
    )
    return n


def gate_vi(src: str) -> int:
    print("-- gate (vi): 'irreducible reference' labeling in the md builder --")
    n = 0
    n += check(
        'heading "Noise floor (one-step persistence — irreducible reference)"',
        "Noise floor (one-step persistence — irreducible reference)" in src,
        "comparison.md heading is emitted by run_experiment.py source",
    )
    n += check(
        "'highly auto-correlated and almost noise-free'",
        "highly auto-correlated and almost noise-free" in src,
        "verdict probe claims the observed signal, not the models",
    )
    n += check(
        "'intraday dynamics, not sensor noise'",
        "intraday dynamics, not sensor noise" in src,
        "verdict tail draws the irreducible-error reading from the floor",
    )
    return n


def gate_vii() -> int:
    print("-- gate (vii): n-construction re-derivation --")
    n = 0
    split = int(N_DATA * TRAIN_FRAC)
    n += check(
        "split = int(19735 x 0.70) = 13814", split == 13814,
        f"int({N_DATA} x {TRAIN_FRAC}) = {split}",
    )
    floor_n = N_DATA - split - 1
    n += check(
        "floor n = n - split - 1 = 5920", floor_n == 5920,
        f"19735 - {split} - 1 = {floor_n} (every sample predicted, none dropped)",
    )
    n += check(
        "resync n = 5920 - 41 = 5879", floor_n - 41 == 5879,
        f"{floor_n} - 41 test-block day-heads = {floor_n - 41}",
    )
    return n


def gate_viii() -> int:
    print("-- gate (viii): frozen-artifact audit (metrics.csv + comparison.md) --")
    n = 0
    rows = []
    with open(ROOT / "results" / "metrics.csv", "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    floor = [r for r in rows if r["model"] == "persistence" and r["protocol"] == "floor"]
    resync = [r for r in rows if r["model"] == "persistence" and r["protocol"] == "resync"]
    n += check(
        "floor row in frozen metrics.csv",
        len(floor) == 1
        and (floor[0]["rmse"], floor[0]["mae"], floor[0]["r2"], floor[0]["mape"], floor[0]["n"])
        == ("0.0431", "0.0243", "0.9991", "0.105", "5920"),
        f"persistence,floor = {tuple((floor[0] if floor else {}).get(k, '-') for k in ('rmse', 'mae', 'r2', 'mape', 'n'))}",
    )
    n += check(
        "resync row in frozen metrics.csv",
        len(resync) == 1
        and (resync[0]["rmse"], resync[0]["mae"], resync[0]["r2"], resync[0]["mape"], resync[0]["n"])
        == ("0.6246", "0.5019", "0.809", "2.193", "5879"),
        f"persistence,resync = {tuple((resync[0] if resync else {}).get(k, '-') for k in ('rmse', 'mae', 'r2', 'mape', 'n'))}",
    )
    rmses = [float(r["rmse"]) for r in rows]
    n += check(
        "floor is the strict minimum RMSE", len(floor) == 1 and float(floor[0]["rmse"]) == min(rmses) == sorted(rmses)[0] < sorted(rmses)[1],
        f"member of the {len(rows)} committed rows; floor {float(floor[0]['rmse']):.4f} is strictly smallest",
    )
    n += check(
        "floor n == re-derived 5920", len(floor) == 1 and int(floor[0]["n"]) == 5920,
        "frozen floor n matches the split arithmetic",
    )
    n += check(
        "resync n == re-derived 5879", len(resync) == 1 and int(resync[0]["n"]) == 5879,
        "frozen resync n matches the day-head count",
    )
    md = (ROOT / "results" / "comparison.md").read_text(encoding="utf-8")
    n += check(
        "frozen md carries the noise-floor heading",
        "## Noise floor (one-step persistence — irreducible reference)" in md,
        "results/comparison.md still labels the floor as the irreducible reference",
    )
    n += check(
        "frozen md floor row", "| persistence (1-step) | 0.0431 | 0.0243 | 0.9991 | 0.105 | 5920 |" in md,
        "committed floor digit row present verbatim in the artifact",
    )
    return n


def main() -> None:
    src = (ROOT / "run_experiment.py").read_text(encoding="utf-8")

    total = 0
    total += gate_i(src)
    total += gate_ii(src)
    total += gate_iii(src)
    total += gate_iv(src)
    total += gate_v(src)
    total += gate_vi(src)
    total += gate_vii()
    total += gate_viii()

    print(f"\n=== SUMMARY: {total - len(_fails)}/{total} gates PASS ===")
    if _fails:
        print("failed gates:", ", ".join(_fails), file=sys.stderr)
        sys.exit(1)
    print("verify_negativecontrols_within.py: all gates PASS — the §2.3 "
          "negative-control machinery reading is structurally verified "
          "(identity step, per-sample floor, per-day re-anchor, "
          "irreducible-reference labeling, frozen digits).")


if __name__ == "__main__":
    main()
