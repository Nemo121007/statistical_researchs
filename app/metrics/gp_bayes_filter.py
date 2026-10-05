"""GP-EKF: байесовский фильтр с прогнозом на гауссовском процессе.

Ko и Fox (GP-BayesFilters) заменяют параметрическую модель перехода
распределением GP и линеаризуют его среднее для расширенного фильтра Калмана.
Здесь параметрическая часть — постоянная скорость, GP учит остаток перехода
по скорости и шагу времени. Наблюдение — известная линейная модель ГНСС:
измеряются восток и север. Входов управления у трека нет.

Score точки — нормированный квадрат инновации. Большее значение — сильнее
расхождение измерения с прогнозом фильтра.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve_triangular


class GpEkfModel:
    """Обучаемый GP-EKF для траектории в локальной проекции (восток, север), метры.

    ``xy`` — массив формы (N, 2). ``valid`` отмечает штатные точки, по которым
    собираются переходы. На шаге фильтра метки не используются. Опорные переходы
    берутся равномерно, параметр ``seed`` на этот выбор не влияет.
    """

    def __init__(
        self,
        support: int = 160,
        seed: int = 7,
        sigma_meas: float = 40.0,
        max_dt: float = 45.0,
        gap_dt: float = 180.0,
    ) -> None:
        self.support = int(support)
        self.seed = int(seed)
        self.sigma_meas = float(sigma_meas)
        self.max_dt = float(max_dt)
        self.gap_dt = float(gap_dt)
        self.lengthscale = 1.0
        self.noise = 0.3
        self.n_support = 0
        self._ready = False

    def fit(self, xy: np.ndarray, t_sec: np.ndarray, valid: np.ndarray) -> "GpEkfModel":
        """Учит остаток постоянной скорости на переходах между штатными точками."""
        xy = np.asarray(xy, dtype=np.float64)
        t_sec = np.asarray(t_sec, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy должен иметь форму (N, 2)")
        if len(t_sec) != len(xy) or len(valid) != len(xy):
            raise ValueError("xy, t_sec и valid должны быть одной длины")
        velocity_east, velocity_north, step_dt = _incoming_velocity(xy, t_sec)
        inputs, targets = _transition_sample(
            xy,
            velocity_east,
            velocity_north,
            step_dt,
            valid,
            self.max_dt,
            self.support,
        )
        self._fit_gps(inputs, targets)
        self._ready = True
        return self

    def scores(self, xy: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
        """NIS наблюдения на каждом шаге. У первой точки score равен 0."""
        if not self._ready:
            raise RuntimeError("Сначала вызовите fit")
        xy = np.asarray(xy, dtype=np.float64)
        t_sec = np.asarray(t_sec, dtype=np.float64)
        count = len(xy)
        if len(t_sec) != count:
            raise ValueError("xy и t_sec должны быть одной длины")
        scores = np.zeros(count, dtype=np.float64)
        if count < 2:
            return scores

        state = np.zeros(4, dtype=np.float64)
        state[:2] = xy[0]
        cov = np.diag([80.0**2, 80.0**2, 5.0**2, 5.0**2])
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        meas_cov = np.eye(2) * (self.sigma_meas**2)
        eye = np.eye(4)
        for index in range(1, count):
            dt = float(t_sec[index] - t_sec[index - 1])
            dt = 1.0 if dt <= 0.0 else dt
            gap = dt > self.gap_dt
            dt_model = min(dt, self.max_dt)
            pred, pred_cov = self._predict(state, cov, dt_model)
            innov = xy[index] - observation @ pred
            innov_cov = observation @ pred_cov @ observation.T + meas_cov
            solved = np.linalg.solve(innov_cov, innov)
            scores[index] = float(innov @ solved)
            gain = pred_cov @ observation.T @ np.linalg.inv(innov_cov)
            state = pred + gain @ innov
            updated = eye - gain @ observation
            cov = updated @ pred_cov @ updated.T + gain @ meas_cov @ gain.T
            cov = 0.5 * (cov + cov.T)
            if gap:
                state = np.array([xy[index, 0], xy[index, 1], 0.0, 0.0])
                cov = np.diag([80.0**2, 80.0**2, 5.0**2, 5.0**2])
        return scores

    def _fit_gps(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self._input_center = np.mean(inputs, axis=0)
        self._input_scale = np.std(inputs, axis=0)
        self._input_scale = np.where(self._input_scale < 1e-3, 1.0, self._input_scale)
        self._target_center = np.mean(targets, axis=0)
        self._target_scale = np.std(targets, axis=0)
        floors = np.array([4.0, 4.0, 0.08, 0.08])
        self._target_scale = np.maximum(self._target_scale, floors)
        standardized_in = (inputs - self._input_center) / self._input_scale
        standardized_out = (targets - self._target_center) / self._target_scale
        self.lengthscale, self.noise, chol, alpha = _select_hyperparameters(
            standardized_in, standardized_out
        )
        self._train_z = standardized_in
        self.n_support = int(len(standardized_in))
        self._chol = chol
        self._alpha = alpha
        self._inv_length_sq = np.full(3, 1.0 / (self.lengthscale**2))

    def _features(self, velocity_east: float, velocity_north: float, dt: float) -> np.ndarray:
        raw = np.array([velocity_east, velocity_north, dt], dtype=np.float64)
        return (raw - self._input_center) / self._input_scale

    def _gp_stats(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Среднее остатка, его дисперсия и якобиан среднего по (v_E, v_N)."""
        diff = features - self._train_z
        kernel = np.exp(-0.5 * np.sum(diff * diff * self._inv_length_sq, axis=1))
        mean_std = kernel @ self._alpha
        solved = solve_triangular(self._chol, kernel, lower=True, check_finite=False)
        variance_std = (1.0 + self.noise**2) - float(solved @ solved)
        variance_std = max(variance_std, 1e-8)
        weighted = kernel[:, None] * self._alpha
        mean_jac_z = -((diff * self._inv_length_sq).T @ weighted)
        mean = self._target_center + self._target_scale * mean_std
        variance = (self._target_scale**2) * variance_std
        velocity_scale = self._input_scale[:2]
        mean_jac_v = (self._target_scale[:, None] * mean_jac_z[:2].T) / velocity_scale
        return mean, variance, mean_jac_v

    def _predict(self, state: np.ndarray, cov: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        mean, variance, mean_jac_v = self._gp_stats(self._features(state[2], state[3], dt))
        transition = np.eye(4)
        transition[0, 2] = dt
        transition[1, 3] = dt
        linear = transition.copy()
        linear[:, 2] += mean_jac_v[:, 0]
        linear[:, 3] += mean_jac_v[:, 1]
        process = np.diag(np.maximum(variance, np.array([1.0, 1.0, 1e-4, 1e-4])))
        predicted = transition @ state + mean
        predicted_cov = linear @ cov @ linear.T + process
        return predicted, 0.5 * (predicted_cov + predicted_cov.T)


def _incoming_velocity(xy: np.ndarray, t_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Скорость точки по шагу, который в неё пришёл. У первой точки нули."""
    count = len(xy)
    velocity_east = np.zeros(count)
    velocity_north = np.zeros(count)
    step_dt = np.ones(count)
    if count < 2:
        return velocity_east, velocity_north, step_dt
    step_dt[1:] = np.diff(t_sec)
    safe_dt = np.clip(step_dt[1:], 1e-3, None)
    velocity_east[1:] = np.diff(xy[:, 0]) / safe_dt
    velocity_north[1:] = np.diff(xy[:, 1]) / safe_dt
    return velocity_east, velocity_north, step_dt


def _transition_sample(
    xy: np.ndarray,
    velocity_east: np.ndarray,
    velocity_north: np.ndarray,
    step_dt: np.ndarray,
    valid: np.ndarray,
    max_dt: float,
    support: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Пары штатных точек: вход (v_E, v_N, Δt), остаток после постоянной скорости."""
    count = len(xy)
    usable = []
    for index in range(1, count - 1):
        dt = float(step_dt[index + 1])
        if dt <= 0.0 or dt > max_dt:
            continue
        if not valid[index] or not valid[index + 1]:
            continue
        usable.append(index)
    if len(usable) < 8:
        raise RuntimeError("Для GP-EKF нужно хотя бы 8 штатных переходов с Δt ≤ max_dt")
    chosen = np.asarray(usable, dtype=int)
    if len(chosen) > support:
        grid = np.linspace(0, len(chosen) - 1, support).astype(int)
        chosen = chosen[grid]
    inputs = np.column_stack(
        [velocity_east[chosen], velocity_north[chosen], step_dt[chosen + 1]]
    )
    dt = step_dt[chosen + 1]
    residual_east = (xy[chosen + 1, 0] - xy[chosen, 0]) - velocity_east[chosen] * dt
    residual_north = (xy[chosen + 1, 1] - xy[chosen, 1]) - velocity_north[chosen] * dt
    residual_ve = velocity_east[chosen + 1] - velocity_east[chosen]
    residual_vn = velocity_north[chosen + 1] - velocity_north[chosen]
    targets = np.column_stack([residual_east, residual_north, residual_ve, residual_vn])
    return inputs, targets


def _select_hyperparameters(
    features: np.ndarray, targets: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Длина корреляции и шум — максимум лог-правдоподобия на короткой сетке."""
    best = None
    for lengthscale in (0.6, 1.0, 1.5, 2.2):
        for noise in (0.15, 0.3, 0.5):
            try:
                chol = _cholesky(features, lengthscale, noise)
            except LinAlgError:
                continue
            score = 0.0
            alphas = []
            for column in range(targets.shape[1]):
                alpha = _chol_solve(chol, targets[:, column])
                alphas.append(alpha)
                quad = float(targets[:, column] @ alpha)
                log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
                score += -0.5 * quad - 0.5 * log_det
            if best is None or score > best[0]:
                best = (score, lengthscale, noise, chol, np.column_stack(alphas))
    if best is None:
        raise LinAlgError("Ковариация GP-EKF не разложилась ни на одной паре гиперпараметров")
    _, lengthscale, noise, chol, alpha = best
    return lengthscale, noise, chol, alpha


def _unit_kernel(features: np.ndarray, lengthscale: float) -> np.ndarray:
    scale = features / lengthscale
    gram = scale @ scale.T
    sq = np.sum(scale * scale, axis=1)
    return np.exp(-0.5 * (sq[:, None] + sq[None, :] - 2.0 * gram))


def _cholesky(features: np.ndarray, lengthscale: float, noise: float) -> np.ndarray:
    gram = _unit_kernel(features, lengthscale)
    gram.flat[:: len(features) + 1] += noise**2
    jitter = 1e-8
    for _ in range(6):
        try:
            return np.linalg.cholesky(gram)
        except LinAlgError:
            gram.flat[:: len(features) + 1] += jitter
            jitter *= 10.0
    raise LinAlgError("Ковариация GP-EKF не разложилась")


def _chol_solve(chol: np.ndarray, vector: np.ndarray) -> np.ndarray:
    solved = solve_triangular(chol, vector, lower=True, check_finite=False)
    return solve_triangular(chol.T, solved, lower=False, check_finite=False)
