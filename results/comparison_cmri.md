# cmri tiny-twin vs single-zone twins — UCI Appliances (held-out 30%)

- Dataset: `data/energydata_complete.csv`, 19735 samples @10 min, 2016-01-11 17:00:00 → 2016-05-27 18:00:00.
- Split: first 13814 samples (70%) calibrate, last 5921 (30%) test — by time.
- Target: indoor kitchen temperature T1 (°C). All fitted models calibrated on train only; v1 never touches data.
- **cmri twin:** coupled two-zone RC. Kitchen T1 + house zone T_h = mean(T2,T3,T4,T5,T7,T8,T9); T6 excluded as an anomalous sensor. Driver set (all OLS-fitted): envelope, inter-zone coupling, Tdewpoint, appliances, lights, thermostat drive. Wind infiltration `wind*(T_out-T)` and clear-sky solar GHI (Chievres AB, Belgium, DST-aware) are implemented and calibrated but were **selected out** by the within-train CV because the full driver mask loses free-run stability on this dataset; the frozen winner is keep (0,3,4,5,6,7).

## Held-out test metrics (resync protocol — primary)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 5.8813 | 5.5538 | -15.9385 | 24.255 | 5879 |
| v2 | 0.646 | 0.4555 | 0.7956 | 1.953 | 5879 |
| cmri | 0.628 | 0.4397 | 0.8069 | 1.887 | 5879 |
| persistence | 0.6246 | 0.5019 | 0.809 | 2.193 | 5879 |

## Free-run robustness (no resync — secondary)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 6.8605 | 6.6819 | -22.0778 | 29.218 | 5920 |
| v2 | 1.5752 | 1.1759 | -0.2165 | 5.033 | 5920 |
| cmri | 1.467 | 1.1067 | -0.0552 | 4.746 | 5920 |

## In-sample train diagnostics (resync — not the evaluation)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 6.5237 | 6.1425 | -26.57 | 29.184 | 13717 |
| v2 | 0.6592 | 0.5136 | 0.7185 | 2.456 | 13717 |
| cmri | 0.6574 | 0.5143 | 0.72 | 2.457 | 13717 |

## Whole-twin ablation on the held-out resync protocol

Features accumulate left-to-right (bold = Δ RMSE vs the previous row).

| variant | drivers | RMSE (°C) | MAE (°C) | R² | Δ RMSE |
|---|---|---|---|---|---|
| v2_drivers | envelope + Tdewpoint + appliances + lights | 0.646 | 0.4555 | 0.7956 |  |
| plus_thermo |  + thermostat drive (gain fitted) | 0.6517 | 0.4592 | 0.792 | +0.0057 |
| plus_coupl |  + inter-zone coupling (Th-T1) | 0.6454 | 0.4539 | 0.796 | -0.0063 |
| plus_wind |  + wind infiltration wind*(T_out-T) | 0.6545 | 0.4591 | 0.7902 | +0.0091 |
| plus_solar |  + clear-sky solar GHI | 0.6823 | 0.4567 | 0.772 | +0.0278 |
| cmri_full | all: full cmri driver set | 0.7081 | 0.4726 | 0.7545 | +0.0257 |

## Calibrated cmri coefficients (train only)

Config is the frozen within-train CV winner: drivers env+coupl+Tdewpoint+appliances/100+lights/100+thermostat (keep [0, 3, 4, 5, 6, 7]), T_set=20.0, alpha=100.0.

### Kitchen zone T1

| coefficient | value | role |
|---|---|---|
| bias | 0.061243 | bias (C/step) |
| env (T_out-T1) | 0.004334 | envelope gain (C per dT-step) |
| coupl (Th-T1) | 0.000412 | coupling gain (C per dT-step) |
| Tdewpoint | -0.003003 | dewpoint latent gain |
| appliances/100 | 0.012490 | appliance heat gain (per 100 W) |
| lights/100 | 0.046825 | lights heat gain (per 100 W) |
| thermostat max(Tsp-T1,0) | -0.001312 | thermostat gain (C per max(Tsp-T,0)-step) |

### House zone T_h

| coefficient | value | role |
|---|---|---|
| bias | 0.027382 | bias (C/step) |
| env (T_out-Th) | 0.002191 | envelope gain (C per dT-step) |
| coupl (T1-Th) | -0.002799 | coupling gain (C per dT-step) |
| Tdewpoint | -0.001291 | dewpoint latent gain |
| appliances/100 | 0.008359 | appliance heat gain (per 100 W) |
| lights/100 | 0.046838 | lights heat gain (per 100 W) |
| thermostat max(Tsp-Th,0) | 0.000810 | thermostat gain (C per max(Tsp-T,0)-step) |

## Noise floor (one-step persistence — irreducible reference)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| persistence (1-step) | 0.0431 | 0.0243 | 0.9991 | 0.105 | 5920 |

## Verdict

- **Resync:** v2 RMSE = 0.646 °C → cmri RMSE = 0.628 °C (Δ **+0.0180 °C**). cmri beats v2 on rmse,mae,r2,mape.
- **Free-run:** v2 RMSE = 1.5752 °C (R² = -0.2165) → cmri RMSE = 1.467 °C (R² = -0.0552) (Δ **+0.1082 °C**).
- Physics drivers: see the ablation table — the marginal RMSE gain per added driver is the evidence for which cmri physics term matters on this dataset.

Plot: `results/cmri_plot.svg` · data: `results/predictions_cmri.csv` · metrics: `results/metrics_cmri.csv` · ablation: `results/ablation_cmri.csv`