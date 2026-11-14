"""tiny_twin.py — baseline "original tiny-twin" simulator (v1).

A deliberately simple, lazy lumped RC (resistor-capacitor) single-zone thermal
twin. It steps a first-order envelope ODE at the dataset resolution (10
minutes) with explicit thermostat-setpoint heating.

All coefficients are FIXED building-handbook defaults (module constants).
The model never reads, fits, or otherwise touches the real data in any form:
it is the uncalibrated baseline that the new calibrated twin
(tiny_twin_v2.py) is compared against.

Pure Python standard library, zero dependencies.
"""

from __future__ import annotations

# ---- Fixed handbook-default coefficients ----------------------------------
T_SET = 21.0      # thermostat heating setpoint (deg C)
A_ENV = 0.02      # envelope coupling: per-step fraction of (T_out - T)
G_SET = 0.02      # heating loop gain toward T_SET when T < T_SET
A_NRG = 0.01      # appliance heat gain (per 100-Wh load step)
Q_SCALE = 100.0   # appliance energy reported in Wh/10min; normalize to 100-Wh units


def step(
    temp: float,
    t_out: float,
    load_wh: float,
    t_set: float = T_SET,
    a_env: float = A_ENV,
    g_set: float = G_SET,
    a_nrg: float = A_NRG,
    q_scale: float = Q_SCALE,
) -> float:
    """One discrete 10-min step of the baseline twin.

    Returns the room temperature at the next sample:
        T += a_env*(T_out - T) + g_set*max(T_SET - T, 0) + a_nrg*(load/100)
    """
    heating = max(t_set - temp, 0.0)  # clamped thermostat: only heat when below setpoint
    d_temp = a_env * (t_out - temp) + g_set * heating + a_nrg * (load_wh / q_scale)
    return temp + d_temp
