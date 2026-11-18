# Tiny-Twin Improved — Towards a Calibrated Digital Twin

A short-code, dependencies-free exploration and validation of a *calibrated*
building-thermal digital twin. The full technical note — dataset provenance,
method, 52 verification checks, and conclusions — lives in
[`REPORT.md`](REPORT.md); this README is the entry point for the code.

## What this is

A physics-shaped RC step model of the indoor kitchen/utility-room temperature
(`T1`) of a real building, driven by the public *Appliances energy prediction*
dataset (UCI). Two twins are evaluated side by side:

- **v1** (`tiny_twin.py`) — the deliberately lazy baseline using fixed
  building-handbook coefficients (uncalibrated).
- **v2** (`tiny_twin_v2.py`) — calibrated by hand-rolled OLS against 137 days
  of real sensor data, plus a coupled two-zone variant `cmri_twin.py`.

## Layout

```
tiny_twin.py / tiny_twin_v2.py / cmri_twin.py   models
run_experiment.py / experiment_cmri.py          evaluation drivers
select_config*.py / ablation.py / grid_knobs.py hyperparameter / ablation studies
verify_*.py            49 targeted verification probes
run_all_verifications.py   runs every probe (52 checks: 3 self-tests + 49 probes)
```

## Run it

Zero external dependencies — only a Python 3.9+ interpreter.

```bash
python3 run_all_verifications.py     # full verification suite (52 checks)
python3 run_experiment.py          # full run -> results/
python3 run_experiment.py --selftest # OLS sanity check
```

## Results (2026-11-14)

Calibration with real data beats handbook defaults: v2 reduces the free-run
and one-step root-mean-square errors substantially, restores the correct
sign of the envelope coupling, and removes encode-rate bias in the residual
autocorrelation and quantiles. Full numbers in `REPORT.md` and
`results/`.

## Verification

`run_all_verifications.py` exits 0 only if all 52 checks pass:
OLS self-tests, coefficient honesty, persistence/variance baselines,
free-run trajectory fidelity, RMSE gap, residual whiteness, quantile
coverage, ablation ordering, dataset provenance, horizon roll-off, and the
v2-vs-v1 within-episode advantage. Last run: **52 passed, 0 failed**.

## License

Dual-licensed: the source code is released under the
**GNU Affero General Public License v3.0 (AGPL-3.0)** — see
[`LICENSE`](LICENSE). Any commercial or profit-making use, or use in a
product or service sold or licensed to others, requires a separate
written commercial authorization agreement with the copyright holder — see
[`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

Copyright (c) 2026 许达 (Xu Da) / 中国移动研究院 (China Mobile Research Institute).
