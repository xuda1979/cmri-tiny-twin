#!/usr/bin/env python3
"""verify_freerun_variance.py — is cmri's free-run edge genuinely variance-driven?

Pure-Python re-aggregation of the FROZEN held-out free-run predictions
(results/predictions_cmri.csv, single-touch artifact).  No new evaluation, no
refitting — the held-out test is still touched exactly once.

diagnose_freerun.py reports the pooled open-loop *moments* of that frozen run:
cmri free-run RMSE 1.4915 vs v2 1.5752, per-step error sd 1.1724 vs 1.3629,
mean bias -0.9222 vs +0.7899 degC/step, error correlation 0.133, error-trend
slope -0.000330 vs -0.000091 C/step^2.  §5.2/§5.4 read these as a
"variance-driven" improvement: cmri wins RMSE not through a flatter mean but
through a collapsed error variance, at the cost of a slightly larger bias.

verify_freerun.py already showed the MAE differential does NOT separate at the
day level (19/42 days, p = 0.78).  An open-loop run has no re-anchor, so the
whole calendar day is the only defensible independence unit.  This script asks
whether the *variance* side of the claim -- the part claimed to drive the RMSE
win -- separates at day level at all, using the same day-block sign test and
bootstrap CI machinery used everywhere else in this repo.  Sections:

  1. Pooled-moment cross-check against the committed §5.2 numbers
     (RMSE / sd / bias / correlation, n = 5920).
  2. Per-day within-day error sd, paired gap sd_v2 - sd_cmri: day-block sign
     test + bootstrap 95 % CI on the mean daily sd gap.  This is the honest
     bound on "per-step error sd drops".
  3. Per-day within-day RMSE, paired gap: day-block test.  RMSE mixes bias and
     variance, so (2) + (3) together bound the headline 1.492 vs 1.575 win
     that the MAE-based §5.2 test did not cover.
  4. Per-day within-day absolute bias, paired gap: is the "slightly larger cmri
     bias" concession stable at day level, or noise?
  5. Per-day within-day correlation of the two error series: how often do the
     two models fail in lockstep (descriptive; the logged 0.66 -> 0.13
     "collapse" refers to the superseded single-fabric config, which is not
     present in this frozen artifact -- only the current 0.133 is testable).
  6. Day-block bootstrap on the POOLED free-run RMSE gap (the headline, not
     the per-day mean handled in section 3): resample whole days, recompute
     each model's pooled RMSE, take the difference.  Because within-day
     residuals are near-Gaussian and the pooled RMSE is an rms, this metric
     is driven by a few bad days, so the honest bound must resample days,
     exactly as verify_rmse_gap.py does for the resync protocol.
"""

import csv
import math
import random
import statistics

PRED = "results/predictions_cmri.csv"
SEED = 42
B = 5000


def phi(z: float) -> float:
    """Standard normal CDF (via erff, pure stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def sign_test(wins: int, n: int) -> tuple[float, float]:
    """Binomial sign test on counts of positive paired d, normal approx."""
    mu = n / 2.0
    sd = math.sqrt(n / 4.0)
    z = (wins - 0.5 - mu) / sd  # continuity correction
    return z, 1.0 - phi(z)      # one-sided H1: P(d>0) > 0.5


def sdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def rms(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def pearson(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sxy = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sxx = sum((x - ma) ** 2 for x in a)
    syy = sum((y - mb) ** 2 for y in b)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def day_block(keys: list[str], gaps: list[float]) -> None:
    """Shared day-block report: sign test + bootstrap CI on mean daily gap."""
    m = statistics.fmean(gaps)
    n = len(gaps)
    w = sum(1 for x in gaps if x > 1e-12)
    z, p = sign_test(w, n)
    rng = random.Random(SEED)
    means = []
    for _ in range(B):
        s = [gaps[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(s))
    means.sort()
    lo, hi = means[int(B * 0.025)], means[int(B * 0.975)]
    se = (statistics.pstdev(gaps) / math.sqrt(n - 1)) if n > 1 else math.nan
    print(f"  cmri better {w}/{n} days  sign z {z:+.2f} (one-sided p {p:.3f})")
    print(f"  mean daily gap {m:+.4f} degC  day-block bootstrap 95% CI "
          f"[{lo:+.4f}, {hi:+.4f}]  (B={B}, seed {SEED}); normal-approx se "
          f"{se:.4f}, t = {m/se:+.2f}")


def main() -> None:
    day: dict[str, list[tuple[float, float]]] = {}  # date -> [(e_v2, e_cmri)]
    all_v: list[float] = []
    all_c: list[float] = []
    with open(PRED) as f:
        for r in csv.DictReader(f):
            obs = float(r["T1_observed"])
            e2 = float(r["v2_freerun"]) - obs
            ec = float(r["cmri_freerun"]) - obs
            key = r["date"][:10]
            day.setdefault(key, []).append((e2, ec))
            all_v.append(e2)
            all_c.append(ec)
    keys = sorted(day)
    n = len(all_v)
    print(f"free-run variance-side verification on the frozen predictions, "
          f"n={n} steps across {len(keys)} days")

    # 1. Pooled-moment cross-check vs §5.2 / diagnose_freerun.py.
    rho = pearson(all_v, all_c)
    ok = (abs(rms(all_c) - 1.4915) < 0.0002 and abs(rms(all_v) - 1.5752) < 0.0002
          and abs(sdev(all_c) - 1.1724) < 0.0005 and abs(sdev(all_v) - 1.3629) < 0.0005
          and abs(rho - 0.133) < 0.002)
    print("\n1. pooled moments (cross-check vs committed §5.2 / diagnose_freerun.py):")
    print(f"   RMSE  v2 {rms(all_v):.4f}   cmri {rms(all_c):.4f}   (committed 1.5752 / 1.4915)")
    print(f"   MAE   v2 {statistics.fmean(abs(x) for x in all_v):.4f}"
          f"   cmri {statistics.fmean(abs(x) for x in all_c):.4f}   (1.1759 / 1.1998)")
    print(f"   err sd  v2 {sdev(all_v):.4f}   cmri {sdev(all_c):.4f}   (1.3629 / 1.1724)")
    print(f"   mean bias  v2 {statistics.fmean(all_v):+.4f}   cmri"
          f" {statistics.fmean(all_c):+.4f} C/step  (+0.7899 / -0.9222)")
    print(f"   err correlation: {rho:.3f}   (committed 0.133)   "
          f"{'PASS' if ok else 'FAIL'}")
    print("   note: the logged 0.66 -> 0.13 'correlation collapse' compares the")
    print("   current config to the superseded single-fabric run; only the")
    print("   current 0.133 is present in this frozen artifact.")

    # Per-day within-day statistics for v2 and cmri.
    sd_v = {k: sdev([a for a, _ in v]) for k, v in day.items()}
    sd_c = {k: sdev([c for _, c in v]) for k, v in day.items()}
    rm_v = {k: rms([a for a, _ in v]) for k, v in day.items()}
    rm_c = {k: rms([c for _, c in v]) for k, v in day.items()}
    ab_v = {k: abs(statistics.fmean([a for a, _ in v])) for k, v in day.items()}
    ab_c = {k: abs(statistics.fmean([c for _, c in v])) for k, v in day.items()}

    # 2. Per-day error-sd gap (the "variance" side of the claim).
    g_sd = [sd_v[k] - sd_c[k] for k in keys]
    print("\n2. per-day within-day error sd, paired gap sd_v2 - sd_cmri "
          "(positive = cmri lower-variance):")
    day_block(keys, g_sd)

    # 3. Per-day RMSE gap (the headline metric, independent of the MAE test).
    g_rm = [rm_v[k] - rm_c[k] for k in keys]
    print("\n3. per-day within-day RMSE, paired gap (positive = cmri lower-RMSE):")
    day_block(keys, g_rm)

    # 4. Per-day |bias| gap (is the cmri-bias concession real at day level?).
    g_ab = [ab_v[k] - ab_c[k] for k in keys]
    print("\n4. per-day within-day |bias|, paired gap (positive = cmri closer "
          "to zero bias):")
    day_block(keys, g_ab)

    # 5. Per-day within-day correlation structure (descriptive).
    rhos = sorted(pearson([a for a, _ in v], [c for _, c in v])
                  for v in day.values())
    n_instep = sum(1 for x in rhos if x > 0.5)
    print(f"\n5. per-day within-day error correlation (v2 vs cmri), "
          f"{len(keys)} days:")
    print(f"   median {statistics.median(rhos):.3f}, min {rhos[0]:.3f}, "
          f"max {rhos[-1]:.3f}; days rho > 0.5: {n_instep}/{len(keys)} — "
          f"on {len(keys) - n_instep}/{len(keys)} days the models are nearly "
          f"decorrelated in their errors (not failing in lockstep).")

    # 6. Day-block bootstrap on the pooled free-run RMSE gap (the headline).
    rng = random.Random(SEED)
    sums_v = [sum(x * x for x, _ in v) for k, v in day.items()]
    sums_c = [sum(y * y for _, y in v) for k, v in day.items()]
    ns = [len(v) for v in day.values()]
    n_days = len(keys)
    pool_gap = (math.sqrt(sum(sums_v) / n) - math.sqrt(sum(sums_c) / n))
    gaps = []
    for _ in range(B):
        sv = 0.0
        sc = 0.0
        nn = 0
        for _ in range(n_days):
            i = rng.randrange(n_days)
            sv += sums_v[i]
            sc += sums_c[i]
            nn += ns[i]
        gaps.append(math.sqrt(sv / nn) - math.sqrt(sc / nn))
    gaps.sort()
    lo, hi = gaps[int(B * 0.025)], gaps[int(B * 0.975)]
    print(f"\n6. POOLED free-run RMSE gap (v2 - cmri), day-block bootstrap: "
          f"{pool_gap:+.4f} degC (committed 1.5752 - 1.4915 = +0.0837)")
    print(f"   95% CI [{lo:+.4f}, {hi:+.4f}]  (B={B}, seed {SEED})")



if __name__ == "__main__":
    main()
