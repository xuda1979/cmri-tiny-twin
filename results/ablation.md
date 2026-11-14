## Driver ablation — how much of v2's win is calibration vs extra sensors?

All OLS variants below use the **same fitting machinery** (`tiny_twin_v2.LinearModel`) 
and the same held-out resync protocol (n=5879). They differ only in which regressors enter.

| model | regressors (all OLS + intercept) | RMSE (°C) | MAE (°C) | R² | MAPE (%) |
|---|---|---|---|---|---|
| env_only | envelope | 0.7467 | 0.595 | 0.727 | 2.5401 |
| env_app | envelope + appliances | 0.7137 | 0.5246 | 0.7506 | 2.2427 |
| full | envelope + dewpoint + appliances + lights | 0.646 | 0.4555 | 0.7956 | 1.953 |
| v1 handbook | envelope + appliance *thermostat, no fitting* | 5.8813 | 5.5538 | -15.9385 | 24.255 |
| persistence | T̂(t+1)=T(t), daily re-anchor | 0.6246 | 0.5019 | 0.809 | 2.193 |

**Reading:** (1) `env_app` (v1's own drivers, OLS-fitted) already collapses v1's 5.88 °C error to ~0.6-0.7 °C → the dominant factor is **calibration to real data**, not extra sensors. (2) the `full` variant's remaining small gain over `env_app` is the marginal contribution of dewpoint/lights drives.
