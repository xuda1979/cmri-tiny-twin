# Tiny-twin comparison — UCI Appliances (held-out test, last 30%)

- Dataset: `data/energydata_complete.csv`, 19735 samples @10 min, 2016-01-11 17:00:00 → 2016-05-27 18:00:00.
- Split: first 13814 samples (70%) train, last 5921 (30%) test — by time.
- Target: indoor kitchen temperature T1 (°C). v2 calibrated on train only; v1 never touches data.

## Held-out test metrics (resync protocol — primary)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 5.8813 | 5.5538 | -15.9385 | 24.255 | 5879 |
| v2 | 0.646 | 0.4555 | 0.7956 | 1.953 | 5879 |
| persistence | 0.6246 | 0.5019 | 0.809 | 2.193 | 5879 |

## Free-run robustness (no resync — secondary)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 6.8605 | 6.6819 | -22.0778 | 29.218 | 5920 |
| v2 | 1.5752 | 1.1759 | -0.2165 | 5.033 | 5920 |

## In-sample train diagnostics (resync — not the evaluation)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 6.5237 | 6.1425 | -26.57 | 29.184 | 13717 |
| v2 | 0.6592 | 0.5136 | 0.7185 | 2.456 | 13717 |

## Calibrated coefficients (v2, train only)

| coefficient | value | role |
|---|---|---|
| env a_env | 0.062420 | envelope gain a_env (per step) |
| dewpoint a_rhu | 0.004447 | dewpoint latent drive |
| appliances a_nrg | -0.003096 | appliance heat gain |
| lights a_lux | 0.012579 | lights heat gain |
| bias | 0.047120 | constant bias (degC/step) |

## Noise floor (one-step persistence — irreducible reference)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| persistence (1-step) | 0.0431 | 0.0243 | 0.9991 | 0.105 | 5920 |

## Verdict

- Held-out RMSE: v1 = 5.8813 °C, v2 = 0.646 °C (improvement **+5.235 °C**, 89.0% less error), resync persistence = 0.6246 °C.
- v2 beats v1 on **all four** held-out metrics (RMSE/MAE/R²/MAPE).
- One-step persistence floor: RMSE = 0.0431 °C at 10-min resolution — the observed signal itself is highly auto-correlated and almost noise-free, so the remaining v2 error (RMSE 0.646 °C when re-anchored daily) reflects genuinely hard-to-model intraday dynamics, not sensor noise.
- **Caveat:** v2 does not beat the daily-resync persistence benchmark on RMSE — noted for review; it still beats it on MAE/MAPE.

Plot: `results/plot.svg` · data: `results/predictions.csv` · metrics: `results/metrics.csv`