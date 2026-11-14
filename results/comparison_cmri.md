# cmri tiny-twin vs single-zone twins — UCI Appliances (held-out 30%)

- Dataset: `data/energydata_complete.csv`, 19735 samples @10 min, 2016-01-11 17:00:00 → 2016-05-27 18:00:00.
- Split: first 13814 samples (70%) calibrate, last 5921 (30%) test — by time.
- Target: indoor kitchen temperature T1 (°C). All fitted models calibrated on train only; v1 never touches data.
- **cmri twin:** coupled two-zone RC. Kitchen T1 + house zone T_h = mean(T2,T3,T4,T5,T7,T8,T9); T6 excluded as an anomalous sensor. Driver set (all OLS-fitted): envelope, inter-zone coupling, Tdewpoint, appliances, lights, thermostat drive, and building-fabric thermal mass `(T_mass-T)` (EMA of T_out, tau = 3 h, exogenous). Wind infiltration `wind*(T_out-T)` and clear-sky solar GHI (Chievres AB, Belgium, DST-aware) are implemented and calibrated but were **selected out** by the within-train CV because they lose free-run stability on this dataset; the frozen winner is keep (0,3,4,5,6,7,8) `T_set=21.0 alpha=300 w=0.944`.

## Held-out test metrics (resync protocol — primary)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 5.8813 | 5.5538 | -15.9385 | 24.255 | 5879 |
| v2 | 0.646 | 0.4555 | 0.7956 | 1.953 | 5879 |
| cmri | 0.5308 | 0.3842 | 0.862 | 1.652 | 5879 |
| persistence | 0.6246 | 0.5019 | 0.809 | 2.193 | 5879 |

## Free-run robustness (no resync — secondary)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 6.8605 | 6.6819 | -22.0778 | 29.218 | 5920 |
| v2 | 1.5752 | 1.1759 | -0.2165 | 5.033 | 5920 |
| cmri | 1.4916 | 1.1998 | -0.0908 | 5.059 | 5920 |

## In-sample train diagnostics (resync — not the evaluation)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| v1 | 6.5237 | 6.1425 | -26.57 | 29.184 | 13717 |
| v2 | 0.6592 | 0.5136 | 0.7185 | 2.456 | 13717 |
| cmri | 0.6525 | 0.5149 | 0.7242 | 2.473 | 13717 |

## Whole-twin ablation on the held-out resync protocol

Features accumulate left-to-right (bold = Δ RMSE vs the previous row).

| variant | drivers | RMSE (°C) | MAE (°C) | R² | Δ RMSE |
|---|---|---|---|---|---|
| v2_drivers | envelope + Tdewpoint + appliances + lights | 0.646 | 0.4555 | 0.7956 |  |
| plus_thermo |  + thermostat drive (gain fitted) | 0.6487 | 0.4573 | 0.7939 | +0.0027 |
| plus_coupl |  + inter-zone coupling (Th-T1) | 0.6423 | 0.4521 | 0.798 | -0.0064 |
| plus_fab |  + building-fabric mass (T_mass-T) | 0.5486 | 0.3984 | 0.8526 | -0.0937 |
| plus_wind |  + wind infiltration wind*(T_out-T) | 0.5661 | 0.4066 | 0.843 | +0.0175 |
| cmri_full |  + clear-sky solar GHI (full 9-driver cmri model) | 0.6408 | 0.4345 | 0.7989 | +0.0747 |
| plus_slowfab |  + slow fabric mass (T_mass2-T, second capacitor) | 0.5526 | 0.4098 | 0.8505 | -0.0882 |

## Calibrated cmri coefficients (train only)

Config is the frozen within-train CV winner: drivers env+coupl+Tdewpoint+appliances/100+lights/100+thermostat+fabric+slow (keep [0, 3, 4, 5, 6, 7, 8, 9]), T_set=21.0, alpha=300.0.

### Kitchen zone T1

| coefficient | value | role |
|---|---|---|
| bias | 0.010031 | bias (C/step) |
| env (T_out-T1) | 0.005978 | envelope gain (C per dT-step) |
| coupl (Th-T1) | 0.003099 | coupling gain (C per dT-step) |
| Tdewpoint | -0.000826 | dewpoint latent gain |
| appliances/100 | 0.012010 | appliance heat gain (per 100 W) |
| lights/100 | 0.045959 | lights heat gain (per 100 W) |
| thermostat max(Tsp-T1,0) | 0.000186 | thermostat gain (C per max(Tsp-T,0)-step) |
| fabric mass (Tmass-T1) | -0.001692 | fabric-mass gain (C per dT-step) |
| slow fabric mass (Tmass2-T1) | -0.003116 | slow fabric-mass gain (C per dT-step) |

### House zone T_h

| coefficient | value | role |
|---|---|---|
| bias | -0.002036 | bias (C/step) |
| env (T_out-Th) | 0.005312 | envelope gain (C per dT-step) |
| coupl (T1-Th) | -0.001658 | coupling gain (C per dT-step) |
| Tdewpoint | 0.000020 | dewpoint latent gain |
| appliances/100 | 0.008149 | appliance heat gain (per 100 W) |
| lights/100 | 0.049723 | lights heat gain (per 100 W) |
| thermostat max(Tsp-Th,0) | 0.001142 | thermostat gain (C per max(Tsp-T,0)-step) |
| fabric mass (Tmass-Th) | -0.005086 | fabric-mass gain (C per dT-step) |
| slow fabric mass (Tmass2-Th) | 0.000303 | slow fabric-mass gain (C per dT-step) |

## Noise floor (one-step persistence — irreducible reference)

| model | RMSE (°C) | MAE (°C) | R² | MAPE (%) | n |
|---|---|---|---|---|---|
| persistence (1-step) | 0.0431 | 0.0243 | 0.9991 | 0.105 | 5920 |

## Verdict

- **Resync:** v2 RMSE = 0.646 °C → cmri RMSE = 0.5308 °C (Δ **+0.1152 °C**). cmri beats v2 on rmse,mae,r2,mape.
- **Free-run:** v2 RMSE = 1.5752 °C (R² = -0.2165) → cmri RMSE = 1.4916 °C (R² = -0.0908) (Δ **+0.0836 °C**).
- Physics drivers: see the ablation table — the marginal RMSE gain per added driver is the evidence for which cmri physics term matters on this dataset.

Plot: `results/cmri_plot.svg` · data: `results/predictions_cmri.csv` · metrics: `results/metrics_cmri.csv` · ablation: `results/ablation_cmri.csv`