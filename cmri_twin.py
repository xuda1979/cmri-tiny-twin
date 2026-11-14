"""cmri_twin.py — the cmri-tiny-twin: a more realistic, physically-motivated twin.

Beats the single-zone v1/v2 twins by adding the physics the first-order lumped
RC model ignores:

  1. TWO-ZONE COUPLING.  The kitchen (T1) exchanges heat with the rest of the
     house (T_h = mean of T2,T3,T4,T5,T7,T8,T9 — T6 excluded: an anomalous
     sensor, min -6.07 C, mean 7.9 C vs 19-22 C for every other zone sensor,
     consistent with an unheated/outlying space).  Both zones are simulated
     states coupled through (T_h - T1) / (T1 - T_h) conductance terms.

  2. WIND-DRIVEN INFILTRATION.  Air leakage scales with wind speed:
     wind * (T_out - T1) — mechanically distinct from the still-air envelope
     term (T_out - T1).

  3. CLEAR-SKY SOLAR GAIN.  A physical solar-irradiance proxy (clear-sky GHI,
     W/m^2) computed for the house location (Chievres, Belgium: 50.56 N,
     3.83 E), DST-aware (Belgium moved from UTC+1 to UTC+2 on 2016-03-27,
     inside this dataset), with solar-elevation geometry, equation-of-time and
     Kasten-Young air mass.

  4. FABRIC THERMAL MASS.  A building-fabric state T_mass (the internal mass
     of walls/floors, which equilibrates with the outside air over many
     hours, not minutes) as an exogenous EMA of T_out: with dt = 600 s and
     time constant tau, T_mass[t] = (1-w)*T_out[t] + w*T_mass[t-1],
     w = exp(-dt/tau).  Because first-order RC zones already track short
     time scales through the (T_out - T) term, the slow T_mass drives the
     long-horizon open-loop trend.  T_mass is a pure function of observed
     weather, so the step API stays single-step/exogenous (no model state
     to re-anchor).

     TWO-TIMESCALE FABRIC (driver 9).  Experiments show the open-loop drift
     on the held-out block is a slow seasonal level shift between the
     calibration and evaluation seasons.  A single fabric capacitor (driver
     8, tau = 3 h) tracks weather swings but not the seasonal level.  Driver
     9 adds a second, SLOWER capacitor T_mass2 (tau of hours-to-days) to the
     same driver set, giving the open-loop twin a slow exogenous reference
     for the seasonal level: dT now includes a9*(T_mass-T) + a10*(T_mass2-T).
     T_mass2 = fabric_mass(t_out, w2), w2 chosen strictly within train.

  5. YET the driver coefficients are all OLS-fitted on the real training
     segment (same closed-form normal-equations machinery as tiny_twin_v2),
     so every physics term is *calibrated to data*, never guessed:

        kitchen:  dT1 = a1*(T_out-T1) + a2*wind*(T_out-T1) + a3*GHI
                       + a4*(T_h-T1)  + a5*Tdewpoint + a6*Appl/100
                       + a7*lux/100   + a8*max(T_set-T1, 0)
                       + a9*(T_mass-T1) + a10*(T_mass2-T1) + b1
        house:    dTh = c1*(T_out-Th) + c2*wind*(T_out-Th) + c3*GHI
                       + c4*(T1-Th)   + c5*Tdewpoint + c6*Appl/100
                       + c7*lux/100   + c8*max(T_set-Th, 0)
                       + c9*(T_mass-Th) + c10*(T_mass2-Th) + b2

Pure Python standard library; reuses tiny_twin_v2.LinearModel.
"""

from __future__ import annotations

import datetime as dt
import math

from tiny_twin_v2 import LinearModel, Q_SCALE

# House location (dataset cite: Chievres AB, Belgium).
LAT_DEG = 50.56
LON_DEG = 3.83

# Sensor indices that make up the "rest of house" zone (T6 excluded).
TH_SENSORS = (2, 3, 4, 5, 7, 8, 9)

# Thermostat setpoint assumed for the heating control term (C). The *gain* of
# the term is still OLS-fitted; only the setpoint is fixed a priori.
T_SET = 20.0


def _last_sunday(year: int, month: int) -> dt.date:
    nd = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return nd - dt.timedelta(days=(nd.weekday() - 6) % 7)


def utc_offset_hours(d: dt.datetime) -> float:
    """Europe/Brussels: UTC+1 winter, UTC+2 summer (last Sun Mar -> last Sun Oct)."""
    start = dt.datetime.combine(_last_sunday(d.year, 3), dt.time(1, 0))
    end = dt.datetime.combine(_last_sunday(d.year, 10), dt.time(1, 0))
    return 2.0 if start <= d < end else 1.0


def clear_sky_ghi(d: dt.datetime) -> float:
    """Clear-sky horizontal irradiance (W/m^2) at the house location.

    Direct-beam + diffuse approximation using solar-elevation geometry and the
    Kasten-Young air mass.  Units are physical (0-900 W/m^2) so the fitted
    solar coefficient has a physical interpretation (C per W/m^2 per 10 min).
    """
    n = d.timetuple().tm_yday
    utc = d - dt.timedelta(hours=utc_offset_hours(d))
    decl = math.radians(23.44 * math.sin(math.radians(360.0 * (n - 81) / 365)))
    B = math.radians(360.0 * (n - 81) / 364)
    eot_min = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    solar_noon_utc_h = 12.0 - LON_DEG / 15.0
    utc_h = utc.hour + utc.minute / 60.0 + utc.second / 3600.0 + eot_min / 60.0
    H = math.radians((solar_noon_utc_h - utc_h) * 15.0)
    sin_el = (math.sin(math.radians(LAT_DEG)) * math.sin(decl)
              + math.cos(math.radians(LAT_DEG)) * math.cos(decl) * math.cos(H))
    if sin_el <= 0.0:
        return 0.0
    el_deg = math.degrees(math.asin(sin_el))
    am = 1.0 / (sin_el + 0.50572 * (el_deg + 6.07995) ** -1.6364)
    i0 = 1361.0 * (1.0 + 0.033 * math.cos(math.radians(360.0 * (n - 2) / 365)))
    return 0.9 * i0 * sin_el * 0.7 ** (am ** 0.678)


def fabric_mass(t_out: list[float], w: float) -> list[float]:
    """Building-fabric thermal state: external air temperature averaged into a
    slow thermal-mass capacitor.

    With dt = 600 s and physical time constant tau, w = exp(-dt/tau) (e.g.
    tau = 6 h -> w = exp(-1/36) ~ 0.973).  T_mass[t] = (1-w)*T_out[t] +
    w*T_mass[t-1], seeded at T_out[0].  This is a pure function of observed
    weather -- exogenous, fully re-anchored by re-computation -- so the twin's
    step API stays single-step with no hidden model state.
    """
    out = [t_out[0]]
    for i in range(1, len(t_out)):
        out.append((1.0 - w) * t_out[i] + w * out[-1])
    return out


class CmriTwin:
    """Coupled two-zone RC twin; both zone equations OLS-calibrated on train."""

    K_NAMES = ["env (T_out-T1)", "wind infil wind*(T_out-T1)", "solar GHI",
               "coupl (Th-T1)", "Tdewpoint", "appliances/100", "lights/100",
               "thermostat max(Tsp-T1,0)", "fabric mass (Tmass-T1)",
               "slow fabric mass (Tmass2-T1)"]
    H_NAMES = ["env (T_out-Th)", "wind infil wind*(T_out-Th)", "solar GHI",
               "coupl (T1-Th)", "Tdewpoint", "appliances/100", "lights/100",
               "thermostat max(Tsp-Th,0)", "fabric mass (Tmass-Th)",
               "slow fabric mass (Tmass2-Th)"]

    def __init__(self, alpha: float = 0.0, t_set: float = T_SET,
                 keep: tuple[int, ...] | None = None) -> None:
        self.alpha = alpha
        self.t_set = t_set
        # keep: subset of driver indices {0 env, 1 wind, 2 solar, 3 coupling,
        # 4 dewpoint, 5 appliances, 6 lights, 7 thermostat, 8 fabric mass
        # (T_mass-T), 9 slow fabric mass (T_mass2-T)}; None = all 10.
        self.keep = tuple(range(10)) if keep is None else tuple(keep)
        self.k = LinearModel(alpha=alpha)
        self.h = LinearModel(alpha=alpha)
        self.fitted = False

    @staticmethod
    def _k_feats(t1, th, t_out, dew, load, lux, wind, ghi, t_mass, t_set,
                 t_mass2=None):
        if t_mass2 is None:
            t_mass2 = t_mass
        return [t_out - t1, wind * (t_out - t1), ghi, th - t1, dew,
                load / Q_SCALE, lux / Q_SCALE, max(t_set - t1, 0.0),
                t_mass - t1, t_mass2 - t1]

    @staticmethod
    def _h_feats(t1, th, t_out, dew, load, lux, wind, ghi, t_mass, t_set,
                 t_mass2=None):
        if t_mass2 is None:
            t_mass2 = t_mass
        return [t_out - th, wind * (t_out - th), ghi, t1 - th, dew,
                load / Q_SCALE, lux / Q_SCALE, max(t_set - th, 0.0),
                t_mass - th, t_mass2 - th]

    def fit(self, T1, Th, t_out, dew, load, lux, wind, ghi,
            t_mass=None, t_mass2=None) -> None:
        # t_mass = building-fabric state (see fabric_mass).  None falls back to
        # T_out, which keeps the original 8-driver twin a valid special case.
        # t_mass2 = slow fabric state; None falls back to t_mass, then T_out.
        if t_mass is None:
            t_mass = t_out
        if t_mass2 is None:
            t_mass2 = t_out
        Xk, yk, Xh, yh = [], [], [], []
        for i in range(len(T1) - 1):
            fk = self._k_feats(T1[i], Th[i], t_out[i], dew[i], load[i], lux[i],
                               wind[i], ghi[i], t_mass[i], self.t_set,
                               t_mass2[i])
            fh = self._h_feats(T1[i], Th[i], t_out[i], dew[i], load[i], lux[i],
                               wind[i], ghi[i], t_mass[i], self.t_set,
                               t_mass2[i])
            Xk.append([fk[j] for j in self.keep])
            yk.append(T1[i + 1] - T1[i])
            Xh.append([fh[j] for j in self.keep])
            yh.append(Th[i + 1] - Th[i])
        self.k.fit(Xk, yk)
        self.h.fit(Xh, yh)
        self.fitted = True

    def step(self, t1, th, t_out, dew, load, lux, wind, ghi,
             t_mass=None, t_mass2=None) -> tuple[float, float]:
        if t_mass is None:
            t_mass = t_out
        if t_mass2 is None:
            t_mass2 = t_mass
        fk = self._k_feats(t1, th, t_out, dew, load, lux, wind, ghi, t_mass,
                           self.t_set, t_mass2)
        fh = self._h_feats(t1, th, t_out, dew, load, lux, wind, ghi, t_mass,
                           self.t_set, t_mass2)
        dk = self.k.predict([fk[j] for j in self.keep])
        dh = self.h.predict([fh[j] for j in self.keep])
        return t1 + dk, th + dh


def house_zone(T_sensor_sets: list[list[float]]) -> list[float]:
    """T_h = mean of the non-outlier house sensors, one per time step."""
    sel = [T_sensor_sets[j - 2] for j in TH_SENSORS]
    return [sum(v[i] for v in sel) / len(sel) for i in range(len(sel[0]))]


def selftest() -> None:
    """Recover known coupled-system coefficients from synthetic data."""
    import random

    rng = random.Random(7)
    n = 400
    t_out = [5.0 + 6.0 * math.sin(2 * math.pi * i / 100) for i in range(n)]
    load = [rng.uniform(0, 800) for _ in range(n)]
    lux = [rng.uniform(0, 300) for _ in range(n)]
    wind = [rng.uniform(0, 8) for _ in range(n)]
    ghi = [700.0 * max(0.0, math.sin(2 * math.pi * (i % 144) / 144)) for i in range(n)]
    dew = [3.0 + 2.0 * math.sin(2 * math.pi * i / 57) for i in range(n)]
    # Building-fabric states: fast EMA (tau = 6 h) + slow EMA (tau = 24 h),
    # the two-timescale fabric mass.  Both are exogenous functions of T_out.
    w_fab = math.exp(-1.0 / 36.0)
    w_fab2 = math.exp(-1.0 / 144.0)
    t_mass = fabric_mass(t_out, w_fab)
    t_mass2 = fabric_mass(t_out, w_fab2)

    # True coupled two-timescale system.  Coefficient arrays are in OLS output
    # layout [bias, env, wind, solar, coupling, dewpoint, appliances, lights,
    # thermostat, fabric, slow fabric]; features follow CmriTwin._k_feats/
    # _h_feats exactly: (t_out-t), wind*(t_out-t), ghi, (t_other-t), dew,
    # load/100, lux/100, max(T_SET-t,0), (t_mass-t), (t_mass2-t).
    K = [0.004, 0.05, 0.004, 0.00040, 0.02, 0.0015, 0.001, 0.0008, 0.01, 0.03, 0.005]
    H = [0.003, 0.04, 0.003, 0.00030, 0.015, 0.001, 0.0006, 0.0004, 0.007, 0.02, 0.002]

    def step_k(i, t1, th):
        return (K[1] * (t_out[i] - t1) + K[2] * wind[i] * (t_out[i] - t1) + K[3] * ghi[i]
                + K[4] * (th - t1) + K[5] * dew[i] + K[6] * load[i] / Q_SCALE
                + K[7] * lux[i] / Q_SCALE + K[8] * max(T_SET - t1, 0.0)
                + K[9] * (t_mass[i] - t1) + K[10] * (t_mass2[i] - t1) + K[0])

    def step_h(i, t1, th):
        return (H[1] * (t_out[i] - th) + H[2] * wind[i] * (t_out[i] - th) + H[3] * ghi[i]
                + H[4] * (t1 - th) + H[5] * dew[i] + H[6] * load[i] / Q_SCALE
                + H[7] * lux[i] / Q_SCALE + H[8] * max(T_SET - th, 0.0)
                + H[9] * (t_mass[i] - th) + H[10] * (t_mass2[i] - th) + H[0])

    t1, th = [20.0], [20.5]
    for i in range(n):
        t1.append(t1[i] + step_k(i, t1[i], th[i]))
        th.append(th[i] + step_h(i, t1[i], th[i]))
    twin = CmriTwin()
    twin.fit(t1[:-1], th[:-1], t_out[:-1], dew[:-1], load[:-1], lux[:-1], wind[:-1],
             ghi[:-1], t_mass=t_mass[:-1], t_mass2=t_mass2[:-1])
    for j in range(11):
        assert abs(twin.k.coefs[j] - K[j]) < 1e-5, ("k", j, twin.k.coefs[j], K[j])
        assert abs(twin.h.coefs[j] - H[j]) < 1e-5, ("h", j, twin.h.coefs[j], H[j])
    # Coupled forward-step consistency (both fabric states passed explicitly).
    a, b = 20.0, 20.4
    pa, pb = twin.step(a, b, 8.0, 2.0, 300.0, 0.0, 3.0, 400.0, 9.3, 11.7)
    ea = pa - (a + (K[1] * (8.0 - a) + K[2] * 3 * (8.0 - a) + K[3] * 400.0
                    + K[4] * (b - a) + K[5] * 2.0 + K[6] * 300.0 / Q_SCALE
                    + K[7] * 0.0 + K[8] * max(T_SET - a, 0.0)
                    + K[9] * (9.3 - a) + K[10] * (11.7 - a) + K[0]))
    eb = pb - (b + (H[1] * (8.0 - b) + H[2] * 3 * (8.0 - b) + H[3] * 400.0
                    + H[4] * (a - b) + H[5] * 2.0 + H[6] * 300.0 / Q_SCALE
                    + H[7] * 0.0 + H[8] * max(T_SET - b, 0.0)
                    + H[9] * (9.3 - b) + H[10] * (11.7 - b) + H[0]))
    assert abs(ea) < 1e-9 and abs(eb) < 1e-9, (ea, eb)
    # Backward compatibility: t_mass2=None makes the slow driver fall back to
    # the fast one (see _k_feats/_h_feats), and the frozen 9-keep mask
    # (0,3,4,5,6,7,8) excludes index 9 entirely -- so fit that mask on the
    # synthetic series with no t_mass2 and smoke-test the step API.
    z = CmriTwin(keep=(0, 3, 4, 5, 6, 7, 8))
    z.fit(t1[:-1], th[:-1], t_out[:-1], dew[:-1], load[:-1], lux[:-1], wind[:-1],
          ghi[:-1], t_mass=t_mass[:-1])
    za, zb = z.step(a, b, 8.0, 2.0, 300.0, 0.0, 3.0, 400.0, 9.3)
    assert math.isfinite(za) and math.isfinite(zb)
    # Feature fallback: omitting t_mass2 exactly duplicates the fast fabric
    # column (index 8 == index 9 == t_mass - t).
    fk = CmriTwin._k_feats(a, b, 8.0, 2.0, 300.0, 0.0, 3.0, 400.0, 9.3, T_SET)
    assert fk[8] == fk[9] == 9.3 - a
    print("cmri_twin self-test OK: coupled 2-zone calibration recovers known two-timescale physics coefficients")


if __name__ == "__main__":
    selftest()
