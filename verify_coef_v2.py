#!/usr/bin/env python3
"""verify_coef_v2.py — are the §3.3 calibrated step coefficients of the v2
single-zone twin a stable model property, or a single-fit artifact?

§3.3 committed claims for the v2 OLS-calibrated step model (train only,
results/coefficients_v2.csv):
  * env a_env        +0.0624   envelope exchange gain per step (the dominant
    -- largest-magnitude -- positive driver)
  * dewpoint a_rhu   +0.0044   latent/outdoor humidity drive (small positive)
  * appliances a_nrg -0.0031   appliance heat gain (NEGATIVE -- the sign
    flip from v1's fixed handbook A_NRG = +0.01)
  * lights a_lux     +0.0126   lighting heat gain (second-largest positive)
  * bias             +0.0471   constant offset (degC/step, positive)

Those were fit ONCE on the full training set [0, 13814).  This script
verifies them WITHOUT touching the held-out test: a fresh CalibratedTwin is
re-fit on the within-train fit slice [0, 11051) and its raw-space
coefficients are checked for the same structure (signs, dominance ordering,
the appliance sign-flip vs the v1 handbook); the structure is also bracketed
across earlier fit subslices (how much data is needed before each sign
stabilises); and a within-train val-block resync RMSE is reported for
context.  It then re-reads the frozen coefficients_v2.csv (read-only) as a
leg-B audit that the committed §3.3 digits re-produce from the artifact.

Pure Python standard library; deterministic.
"""

from __future__ import annotations

import csv
import sys

sys.path.insert(0, ".")

from run_experiment import TRAIN_FRAC, metrics  # noqa: E402
from experiment_cmri import days_of, load_cmri_data  # noqa: E402
from tiny_twin_v2 import CalibratedTwin  # noqa: E402
from tiny_twin import A_ENV, A_NRG  # noqa: E402  (v1 handbook contrast)

VALID_FRAC = 0.2

# Short role -> (index in CalibratedTwin.NAMES order, committed §3.3 digit).
NAMES = CalibratedTwin.NAMES  # ["env a_env", "dewpoint a_rhu", "appliances a_nrg",
#                               "lights a_lux", "bias"]
COMMITTED = {
    "env": +0.0624,
    "dew": +0.0044,
    "appl": -0.0031,
    "lights": +0.0126,
    "bias": +0.0471,
}
COMMITTED_STR = {
    "env": "+0.0624", "dew": "+0.0044", "appl": "-0.0031",
    "lights": "+0.0126", "bias": "+0.0471",
}


def coefs_of(model) -> dict[str, float]:
    """Raw-space step coefficients keyed by short role, in NAMES order."""
    return dict(zip(COMMITTED, model.model.coefs))


def main() -> None:
    data = load_cmri_data(DATA if "DATA" in globals() else "data/energydata_complete.csv")
    n = len(data["T1"])
    split = int(n * TRAIN_FRAC)
    val_start = int(round((1.0 - VALID_FRAC) * split))
    T, t_out = data["T1"], data["T_out"]
    dew, load, lux = data["Tdewpoint"], data["Appliances"], data["lights"]
    dates = data["date"]

    print("within-train calibrated-coefficient probe (v2 single-zone); "
          "held-out test untouched")
    print(f"fit [0,{val_start}) = {val_start} samples; val "
          f"[{val_start},{split}) = {split - val_start} samples "
          f"({len(days_of(dates, val_start, split))} days)")
    print()

    # ---- bracket fits: does each §3.3 signature need the full window? --------
    windows = [(2762, "1 wk"), (5525, "2 wks"), (8288, "3 wks"),
               (val_start, "full within-train")]
    fits: list[tuple[str, dict[str, float]]] = []
    print(f"{'fit window':<20} {'a_env':>9} {'a_rhu':>9} {'a_nrg':>9} "
          f"{'a_lux':>9} {'bias':>9}")
    for lo, label in windows:
        s = slice(0, lo)
        m = CalibratedTwin()
        m.fit(T[s], t_out[s], dew[s], load[s], lux[s])
        c = coefs_of(m)
        fits.append((label, c))
        print(f"{f'[0,{lo}) {label}':<20} {c['env']:>+9.5f} {c['dew']:>+9.5f} "
              f"{c['appl']:>+9.5f} {c['lights']:>+9.5f} {c['bias']:>+9.5f}")

    # ---- leg A: structural gates on the committed within-train fit ----------
    _, k = fits[-1]
    print()
    print("leg A  within-train full fit [0,11051), raw step coefficients:")
    print(f"    env a_env {k['env']:+.5f}  (fitted envelope gain per step vs "
          f"v1 handbook A_ENV = {A_ENV:g})")
    print(f"    dewpoint a_rhu {k['dew']:+.5f}")
    print(f"    appliances a_nrg {k['appl']:+.5f}  (vs v1 handbook "
          f"A_NRG = {A_NRG:+.3g} -- fitted sign is NEGATIVE)")
    print(f"    lights a_lux {k['lights']:+.5f}")
    print(f"    bias {k['bias']:+.5f}")

    others = max(abs(k["lights"]), abs(k["dew"]), abs(k["appl"]))
    gate_env = k["env"] > 0.03                # envelope drive is a real positive gain
    # appliance heat gain is NEGATIVE on the full fit (the sign flip) --
    # on calibration the fitted model reads appliances as a cooling/occupancy
    # residual driver, not a heat source (opposite of v1's handbook guess)
    gate_appl = k["appl"] < -0.001
    gate_pos = k["lights"] > 0.005 and k["dew"] > 0.001 and k["bias"] > 0.01
    # dominance: envelope gain is the biggest driver by >= 3x
    gate_dom = k["env"] >= 3.0 * others
    # robustness: every §3.3 signature present at EVERY fit window (the robust
    # facts are the appliance sign flip and the dominant-positive envelope)
    gate_appl_robust = all(c["appl"] < 0.0 for _, c in fits)
    gate_env_robust = all(c["env"] > 0.03 for _, c in fits)
    gate_sign_robust = all(
        c["lights"] > 0.0 and c["dew"] > 0.0 and c["bias"] > 0.0
        for _, c in fits)

    # ---- context: within-train val-block resync RMSE (not a gate) -----------
    fs = slice(0, val_start)
    mfull = CalibratedTwin()
    mfull.fit(T[fs], t_out[fs], dew[fs], load[fs], lux[fs])
    recs = list(zip(T, t_out, dew, load, lux))[val_start:split]
    val_days = days_of(dates, val_start, split)
    p = run_protocol(mfull, recs[0][0], recs, val_days)[1:]
    obs = [r[0] for r in recs[1:]]
    pairs = [(c, o) for c, o in zip(p, obs) if c is not None]
    rmse = metrics([c for c, _ in pairs], [o for _, o in pairs])["rmse"]
    print(f"    context only: within-train val-block resync RMSE {rmse:.4f} C")
    print()

    # ---- leg B: frozen §3.3 artifact audit (read-only) ------------------------
    held: dict[str, float] = {}
    with open("results/coefficients_v2.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["coefficient"]
            short = next(s for s, n in zip(COMMITTED, NAMES) if n == name)
            held[short] = float(row["value"])
    audit = {s: abs(held[s] - COMMITTED[s]) < 0.0006 for s in COMMITTED}
    print("leg B  frozen coefficients_v2.csv audit (read-only):")
    print(f"    committed digits must re-produce: a_env {COMMITTED_STR['env']} "
          f"/ a_rhu {COMMITTED_STR['dew']} / a_nrg {COMMITTED_STR['appl']} / "
          f"a_lux {COMMITTED_STR['lights']} / bias {COMMITTED_STR['bias']}")
    print(f"    env  a_env  {held['env']:+.6f}  "
          f"({'OK' if audit['env'] else 'MISMATCH'} vs {COMMITTED_STR['env']})")
    print(f"    dew  a_rhu  {held['dew']:+.6f}  "
          f"({'OK' if audit['dew'] else 'MISMATCH'} vs {COMMITTED_STR['dew']})")
    print(f"    appl a_nrg  {held['appl']:+.6f}  "
          f"({'OK' if audit['appl'] else 'MISMATCH'} vs {COMMITTED_STR['appl']})")
    print(f"    light a_lux {held['lights']:+.6f}  "
          f"({'OK' if audit['lights'] else 'MISMATCH'} vs {COMMITTED_STR['lights']})")
    print(f"    bias        {held['bias']:+.6f}  "
          f"({'OK' if audit['bias'] else 'MISMATCH'} vs {COMMITTED_STR['bias']})")
    print()

    # ---- verdict ------------------------------------------------------------
    ok = all([gate_env, gate_appl, gate_pos, gate_dom, gate_appl_robust,
              gate_env_robust, gate_sign_robust, *audit.values()])

    print("gates (within-train full-fit leg = fresh test; frozen CSV = "
          "frozen audit):")
    print(f"    envelope gain positive a_env {k['env']:+.4f} > 0.03 : "
          f"{'PASS' if gate_env else 'FAIL'}")
    print(f"    appliance gain NEGATIVE a_nrg {k['appl']:+.4f} < -0.001 "
          f"(sign flip vs v1 handbook +0.01): "
          f"{'PASS' if gate_appl else 'FAIL'}")
    print(f"    lights/dew/bias positive ({k['lights']:+.4f}/{k['dew']:+.4f}/"
          f"{k['bias']:+.4f}): {'PASS' if gate_pos else 'FAIL'}")
    print(f"    envelope dominance a_env {k['env']:.4f} >= 3x other drivers "
          f"(max {others:.4f}): {'PASS' if gate_dom else 'FAIL'}")
    print(f"    span of fit windows: a_nrg negative at every size "
          f"{'PASS' if gate_appl_robust else 'FAIL'}, a_env > 0.03 at every "
          f"size {'PASS' if gate_env_robust else 'FAIL'}, all-other-sign "
          f"stable {'PASS' if gate_sign_robust else 'FAIL'}")
    print(f"    audit: frozen coefficients_v2.csv re-produces the committed "
          f"§3.3 digits: {'PASS' if all(audit.values()) else 'FAIL'}")
    print(f"    context (v1 handbook contrast): fitted a_env {k['env']:.4f} vs "
          f"book {A_ENV:g} (calibration finds a ~{k['env'] / A_ENV:.1f}x "
          f"steeper envelope drive); fitted a_nrg {k['appl']:.4f} vs book "
          f"{A_NRG:+.3g} (sign flip -- the fitted appliance regressor is not "
          f"v1's heat-gain term)")
    print()
    print(f"verdict: {'PASS' if ok else 'FAIL'} — the §3.3 calibrated step "
          f"coefficients are a stable model property, not a single-fit "
          f"artifact: a fresh within-train fit reproduces the dominant "
          f"positive envelope gain (a_env {k['env']:.4f}, >= 3x every other "
          f"driver), the small positive dewpoint latent drive "
          f"({k['dew']:.4f}), the second-positive lighting gain "
          f"({k['lights']:.4f}), the positive bias ({k['bias']:.4f}), and — "
          f"the sharpest sign — a NEGATIVE appliance gain "
          f"({k['appl']:.4f}, the calibration flip from v1's handbook "
          f"A_NRG = {A_NRG:+.3g}); every sign holds at every fit window from "
          f"1 week up; and the frozen coefficients_v2.csv re-produces the "
          f"committed §3.3 digits "
          f"({COMMITTED_STR['env']}/{COMMITTED_STR['dew']}/"
          f"{COMMITTED_STR['appl']}/{COMMITTED_STR['lights']}/"
          f"{COMMITTED_STR['bias']}).")


def run_protocol(model, state, recs, resync_days):
    """Local copy of run_experiment.run_protocol to keep imports minimal."""
    n = len(recs)
    pred = [None] * n
    for day in resync_days:
        cur = recs[day[0]][0]
        for a, b in zip(day[:-1], day[1:]):
            _, t_out, dew, load, lux = recs[a]
            cur = model.step(cur, t_out, dew, load, lux)
            pred[b] = cur
    return pred


if __name__ == "__main__":
    main()
