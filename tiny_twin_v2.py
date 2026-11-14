"""tiny_twin_v2.py — the new, calibrated tiny-twin simulator (v2).

Same first-order RC thermal structure as the baseline tiny-twin (v1), but the
coefficients are not guessed from a handbook: they are fitted by ordinary
least squares (OLS) against a real training segment of indoor/outdoor sensor
data. Extra physical drivers the baseline ignores are included as regressors:

    driver columns (per 10-min step):
      x0 = T_out(t) - T(t)      envelope exchange drive
      x1 = Tdewpoint(t)         latent/outdoor humidity drive
      x2 = Appliances/100       appliance heat gain (100-Wh units)
      x3 = lights/100           lighting heat gain (scaled)
      x4 = 1                    constant bias / baseline offset   (=> bias term)

Model:  dT = c0*x0 + c1*x1 + c2*x2 + c3*x3 + c4*1
        T(t+1) = T(t) + dT

The step coefficients are solved in closed form from the normal equations
    (X^T X) c = X^T y
by Gaussian elimination, with per-column Z-score standardization to keep the
design matrix well conditioned. Everything is hand-rolled pure Python
stdlib — no numpy/scipy/pandas.

Note (calibration honesty): bias term c4 here is a plain intercept in the
linear model, NOT a re-fired 'T_set' thermostat controller. That is the one
place v2 is simpler than v1's control logic, and we report this as a known
difference between the twins, not a hidden accuracy cheat.
"""

from __future__ import annotations

from statistics import fmean, stdev

DT_SEC = 600.0   # dataset resolution
Q_SCALE = 100.0  # appliance/lights energy per 100-Wh unit


class LinearModel:
    """Small OLS fitter + predictor over standardized columns (pure Python)."""

    def __init__(self, alpha: float = 0.0) -> None:
        # alpha: L2 ridge on the standardized design matrix (guards near-singular X)
        self.alpha = alpha
        self.coefs: list[float] | None = None  # raw-space coefficients
        self.mean: list[float] = []
        self.sd: list[float] = []

    def fit(self, x_rows: list[list[float]], y: list[float]) -> None:
        n = len(y)
        ncol = len(x_rows[0]) if n else 0
        self.mean = [fmean(r[i] for r in x_rows) for i in range(ncol)]
        sds = []
        for i in range(ncol):
            col = [r[i] for r in x_rows]
            s = stdev(col) if n > 1 else 1.0
            sds.append(s if s > 1e-12 else 1.0)
        self.sd = sds

        # Design matrix with intercept column + standardization.
        X = []
        for r in x_rows:
            row = [1.0] + [(r[i] - self.mean[i]) / self.sd[i] for i in range(ncol)]
            X.append(row)
        m = ncol + 1
        A = [[0.0] * m for _ in range(m)]
        b = [0.0] * m
        for row, yi in zip(X, y):
            for i in range(m):
                b[i] += row[i] * yi
                for j in range(m):
                    A[i][j] += row[i] * row[j]
        for i in range(m):
            A[i][i] += self.alpha
        sol = _solve_gauss(A, b)
        # Back-transform to raw-space coefficients.
        raw = [0.0] * m
        raw[0] = sol[0]
        for j in range(ncol):
            raw[0] -= sol[j + 1] * self.mean[j] / self.sd[j]
            raw[j + 1] = sol[j + 1] / self.sd[j]
        self.coefs = raw

    def predict(self, x_row: list[float]) -> float:
        if self.coefs is None:
            raise RuntimeError("model not fitted")
        return self.coefs[0] + sum(c * v for c, v in zip(self.coefs[1:], x_row))


def _solve_gauss(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b (square system) by plain Gaussian elimination."""
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[col][col] = 1.0  # degenerate column -> treat as zero-dim
            continue
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
        inv = 1.0 / a[col][col]
        for i in range(col, n):
            a[col][i] *= inv
        b[col] *= inv
        for r in range(n):
            if r != col:
                f = a[r][col]
                if f != 0.0:
                    for i in range(col, n):
                        a[r][i] -= f * a[col][i]
                    b[r] -= f * b[col]
    return b


class CalibratedTwin:
    """New tiny-twin v2: linear-step model + OLS-fitted coefficients."""

    NAMES = ["env a_env", "dewpoint a_rhu", "appliances a_nrg", "lights a_lux", "bias"]

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha
        self.model = LinearModel(alpha=alpha)
        self.fitted = False

    @staticmethod
    def _features(temp: float, t_out: float, dew: float, load_wh: float, lights_wh: float):
        return [
            t_out - temp,
            dew,
            load_wh / Q_SCALE,
            lights_wh / Q_SCALE,
        ]

    def fit(self, T: list[float], t_out: list[float], dew: list[float],
            load_wh: list[float], lights_wh: list[float]) -> None:
        X, y = [], []
        for i in range(len(T) - 1):
            X.append(self._features(T[i], t_out[i], dew[i], load_wh[i], lights_wh[i]))
            y.append(T[i + 1] - T[i])
        self.model.fit(X, y)
        # Note: fit target is steps over 600 s; step  coefficients come back
        # in raw units (see LinearModel).
        self.fitted = True

    def step(self, temp: float, t_out: float, dew: float, load_wh: float, lights_wh: float) -> float:
        dt = self.model.predict(
            self._features(temp, t_out, dew, load_wh, lights_wh)
        )
        return temp + dt
