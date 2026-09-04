"""
Оценка параметров фильтров Калмана CV и RW.

Обучающие эксперименты:
    1.csv
    2.csv

Файл 3.csv не используется и оставляется для последующей валидации.

Для каждого файла выбираются непрерывные участки длиной не менее
MIN_SEGMENT_LENGTH точек:

    stand
    move
    stand_move

Оцениваются:

    sigma_meas_x
    sigma_meas_y
    sigma_meas

методами:

    1. std(x - mean(x)), std(y - mean(y)) на stand
    2. std(diff(x)) / sqrt(2), std(diff(y)) / sqrt(2) на stand
    3. MLE по инновациям CV
    4. MLE по инновациям RW

Для CV:
    sigma_acc

Для RW:
    sigma_rw

Результаты записываются в CSV.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import chi2

from app.working.data_processor import DataProcessor


# ============================================================
# Конфигурация
# ============================================================

MIN_SEGMENT_LENGTH = 100
WINDOW = 10

SIGMA_MEAS_MIN = 1e-3
SIGMA_MEAS_MAX = 1000.0

SIGMA_ACC_MIN = 1e-7
SIGMA_ACC_MAX = 10.0

SIGMA_RW_MIN = 1e-7
SIGMA_RW_MAX = 100.0

OUTPUT_FILENAME = "kalman_parameter_estimates.csv"


# ============================================================
# Участок трека
# ============================================================

@dataclass
class TrackSegment:
    """Непрерывный участок трека."""

    file_name: str
    segment_type: str
    segment_id: int
    df: pd.DataFrame

    @property
    def size(self) -> int:
        """Количество точек."""
        return len(self.df)


# ============================================================
# Основной класс
# ============================================================

class KalmanParameterEstimator:
    """Оценка параметров моделей Kalman CV и RW."""

    # ========================================================
    # Поиск непрерывных участков
    # ========================================================

    @staticmethod
    def extract_segments(
        df: pd.DataFrame,
        file_name: str,
        min_length: int = MIN_SEGMENT_LENGTH,
    ) -> list[TrackSegment]:
        """
        Выделяет непрерывные участки:

            stand
            move
            stand_move

        Для stand_move разрешены только статусы stand и move,
        поэтому переходы stand <-> move не разрывают участок.
        """
        if "status" not in df.columns:
            raise ValueError("В DataFrame отсутствует столбец 'status'")

        segments: list[TrackSegment] = []

        stand_mask = (df["status"] == "stand").to_numpy()

        segments.extend(
            KalmanParameterEstimator._extract_mask_segments(
                df=df,
                mask=stand_mask,
                file_name=file_name,
                segment_type="stand",
                min_length=min_length,
            )
        )

        move_mask = (df["status"] == "move").to_numpy()

        segments.extend(
            KalmanParameterEstimator._extract_mask_segments(
                df=df,
                mask=move_mask,
                file_name=file_name,
                segment_type="move",
                min_length=min_length,
            )
        )

        stand_move_mask = (
            df["status"].isin(["stand", "move"]).to_numpy()
        )

        segments.extend(
            KalmanParameterEstimator._extract_mask_segments(
                df=df,
                mask=stand_move_mask,
                file_name=file_name,
                segment_type="stand_move",
                min_length=min_length,
            )
        )

        return segments

    @staticmethod
    def _extract_mask_segments(
        df: pd.DataFrame,
        mask: NDArray[np.bool_],
        file_name: str,
        segment_type: str,
        min_length: int,
    ) -> list[TrackSegment]:
        """Выделяет непрерывные True-сегменты из boolean mask."""
        if len(df) != len(mask):
            raise ValueError("Длина mask не совпадает с DataFrame")

        if len(df) == 0:
            return []

        starts = mask & ~np.r_[False, mask[:-1]]
        ends = mask & ~np.r_[mask[1:], False]

        start_indices = np.flatnonzero(starts)
        end_indices = np.flatnonzero(ends)

        segments: list[TrackSegment] = []

        for segment_id, (start, end) in enumerate(
            zip(start_indices, end_indices)
        ):
            segment_df = df.iloc[start:end + 1].copy()

            if len(segment_df) < min_length:
                continue

            segments.append(
                TrackSegment(
                    file_name=file_name,
                    segment_type=segment_type,
                    segment_id=segment_id,
                    df=segment_df,
                )
            )

        return segments

    # ========================================================
    # Координаты
    # ========================================================

    @staticmethod
    def to_local_xy(
        df: pd.DataFrame,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Переводит один участок из lon/lat в локальные X/Y."""
        lon = df["lon"].to_numpy(dtype=np.float64)
        lat = df["lat"].to_numpy(dtype=np.float64)

        return DataProcessor.convert_to_local_cartesian(lon, lat)

    # ========================================================
    # sigma_meas: разброс положения
    # ========================================================

    @staticmethod
    def estimate_sigma_meas_from_position(
        segments: Iterable[TrackSegment],
    ) -> dict[str, float]:
        """
        Оценивает sigma_meas по разбросу положения на stand-участках.

        Для каждого участка:

            sigma_x = std(x - mean(x))
            sigma_y = std(y - mean(y))

        Итоговая оценка:

            sigma_meas = sqrt((sigma_x² + sigma_y²) / 2)
        """
        sum_sq_x = 0.0
        sum_sq_y = 0.0
        count = 0

        for segment in segments:
            if segment.segment_type != "stand":
                continue

            x, y = KalmanParameterEstimator.to_local_xy(segment.df)

            valid = np.isfinite(x) & np.isfinite(y)

            x = x[valid]
            y = y[valid]

            if len(x) == 0:
                continue

            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)

            sum_sq_x += np.sum(x_centered ** 2)
            sum_sq_y += np.sum(y_centered ** 2)

            count += len(x)

        if count == 0:
            return {
                "sigma_meas_x": np.nan,
                "sigma_meas_y": np.nan,
                "sigma_meas": np.nan,
                "count": 0,
            }

        sigma_x = np.sqrt(sum_sq_x / count)
        sigma_y = np.sqrt(sum_sq_y / count)

        sigma_meas = np.sqrt(
            (sigma_x ** 2 + sigma_y ** 2) / 2.0
        )

        return {
            "sigma_meas_x": float(sigma_x),
            "sigma_meas_y": float(sigma_y),
            "sigma_meas": float(sigma_meas),
            "count": count,
        }

    # ========================================================
    # sigma_meas: разности соседних stand-точек
    # ========================================================

    @staticmethod
    def estimate_sigma_meas_from_neighbor_difference(
        segments: Iterable[TrackSegment],
    ) -> dict[str, float]:
        """
        Оценивает sigma_meas через разности соседних stand-точек.

        При независимом шуме:

            Var(z_i - z_(i-1)) = 2 * sigma_meas²

        поэтому:

            sigma_meas = std(delta) / sqrt(2)
        """
        delta_x_list = []
        delta_y_list = []

        for segment in segments:
            if segment.segment_type != "stand":
                continue

            x, y = KalmanParameterEstimator.to_local_xy(segment.df)

            valid = np.isfinite(x) & np.isfinite(y)

            x = x[valid]
            y = y[valid]

            if len(x) < 2:
                continue

            delta_x_list.append(np.diff(x))
            delta_y_list.append(np.diff(y))

        if not delta_x_list:
            return {
                "sigma_meas_x": np.nan,
                "sigma_meas_y": np.nan,
                "sigma_meas": np.nan,
                "count": 0,
            }

        delta_x = np.concatenate(delta_x_list)
        delta_y = np.concatenate(delta_y_list)

        sigma_x = np.std(delta_x, ddof=0) / np.sqrt(2.0)
        sigma_y = np.std(delta_y, ddof=0) / np.sqrt(2.0)

        sigma_meas = np.sqrt(
            (sigma_x ** 2 + sigma_y ** 2) / 2.0
        )

        return {
            "sigma_meas_x": float(sigma_x),
            "sigma_meas_y": float(sigma_y),
            "sigma_meas": float(sigma_meas),
            "count": int(len(delta_x)),
        }

    # ========================================================
    # CV: likelihood одного локального окна
    # ========================================================

    @staticmethod
    def _cv_log_likelihood_window(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        time: NDArray[np.datetime64],
        sigma_acc: float,
        sigma_meas: float,
    ) -> tuple[float, float, float]:
        """
        Рассчитывает метрики последнего наблюдения локального окна CV.

        Returns:
            log_likelihood:
                Логарифм правдоподобия последнего наблюдения.

            mahalanobis_sq:
                Квадрат расстояния Махаланобиса.

            filter_distance:
                Евклидово расстояние между оценкой фильтра
                и фактическим измерением.
        """
        n = len(x)

        if n < 2:
            return np.nan, np.nan, np.nan

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        R = np.eye(
            2,
            dtype=np.float64,
        ) * sigma_meas ** 2

        I = np.eye(
            4,
            dtype=np.float64,
        )

        dt0 = float(
            (time[1] - time[0]) / np.timedelta64(1, "s")
        )

        if dt0 > 0.0:
            vx0 = (x[1] - x[0]) / dt0
            vy0 = (y[1] - y[0]) / dt0
        else:
            vx0 = 0.0
            vy0 = 0.0

        X_state = np.array(
            [x[0], y[0], vx0, vy0],
            dtype=np.float64,
        ).reshape(4, 1)

        P = np.eye(
            4,
            dtype=np.float64,
        ) * 500.0

        if dt0 > 0.0:
            vel_var = (
                2.0 * sigma_meas ** 2 / dt0 ** 2
            )
            P[2, 2] = vel_var
            P[3, 3] = vel_var
        else:
            P[2, 2] = 100.0
            P[3, 3] = 100.0

        current_log_likelihood = np.nan
        current_mahalanobis_sq = np.nan
        current_filter_distance = np.nan

        log_2pi = np.log(2.0 * np.pi)

        for k in range(1, n):
            dt = float(
                (time[k] - time[k - 1]) / np.timedelta64(1, "s")
            )

            dt = max(dt, 0.0)

            F = np.eye(
                4,
                dtype=np.float64,
            )

            F[0, 2] = dt
            F[1, 3] = dt

            dt2 = dt ** 2
            dt3 = dt ** 3
            dt4 = dt ** 4

            Q = np.zeros(
                (4, 4),
                dtype=np.float64,
            )

            Q[0, 0] = dt4 / 4.0
            Q[1, 1] = dt4 / 4.0
            Q[2, 2] = dt2
            Q[3, 3] = dt2
            Q[0, 2] = dt3 / 2.0
            Q[2, 0] = dt3 / 2.0
            Q[1, 3] = dt3 / 2.0
            Q[3, 1] = dt3 / 2.0

            Q *= sigma_acc ** 2

            X_pred = F @ X_state
            P_pred = F @ P @ F.T + Q

            z = np.array(
                [x[k], y[k]],
                dtype=np.float64,
            ).reshape(2, 1)

            innovation = z - H @ X_pred

            S = H @ P_pred @ H.T + R
            S = 0.5 * (S + S.T)

            sign, logdet = np.linalg.slogdet(S)

            if sign <= 0.0 or not np.isfinite(logdet):
                return -np.inf, np.nan, np.nan

            try:
                solved = np.linalg.solve(
                    S,
                    innovation,
                )

                mahalanobis_sq = float(
                    (innovation.T @ solved).item()
                )

                if (
                    not np.isfinite(mahalanobis_sq)
                    or mahalanobis_sq < 0.0
                ):
                    return -np.inf, np.nan, np.nan

                log_likelihood = -0.5 * (
                    2.0 * log_2pi
                    + logdet
                    + mahalanobis_sq
                )

                K = np.linalg.solve(
                    S,
                    H @ P_pred,
                ).T

                X_state = X_pred + K @ innovation

                I_KH = I - K @ H

                P = (
                    I_KH @ P_pred @ I_KH.T
                    + K @ R @ K.T
                )

                P = 0.5 * (P + P.T)

                current_log_likelihood = log_likelihood
                current_mahalanobis_sq = mahalanobis_sq

                current_filter_distance = float(
                    np.hypot(
                        X_state[0, 0] - x[k],
                        X_state[1, 0] - y[k],
                    )
                )

            except np.linalg.LinAlgError:
                return -np.inf, np.nan, np.nan

        return (
            current_log_likelihood,
            current_mahalanobis_sq,
            current_filter_distance,
        )

    # ========================================================
    # RW: likelihood одного локального окна
    # ========================================================

    @staticmethod
    def _rw_log_likelihood_window(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        time: NDArray[np.datetime64],
        sigma_rw: float,
        sigma_meas: float,
    ) -> tuple[float, float, float]:
        """
        Рассчитывает метрики последнего наблюдения локального окна RW.

        Текущая модель:

            Q = sigma_rw² * dt * I
        """
        n = len(x)

        if n < 2:
            return np.nan, np.nan, np.nan

        H = np.eye(
            2,
            dtype=np.float64,
        )

        R = np.eye(
            2,
            dtype=np.float64,
        ) * sigma_meas ** 2

        I = np.eye(
            2,
            dtype=np.float64,
        )

        X_state = np.array(
            [x[0], y[0]],
            dtype=np.float64,
        ).reshape(2, 1)

        P = np.eye(
            2,
            dtype=np.float64,
        ) * sigma_meas ** 2

        current_log_likelihood = np.nan
        current_mahalanobis_sq = np.nan
        current_filter_distance = np.nan

        log_2pi = np.log(2.0 * np.pi)

        for k in range(1, n):
            dt = float(
                (time[k] - time[k - 1]) / np.timedelta64(1, "s")
            )

            dt = max(dt, 0.0)

            Q = (
                np.eye(
                    2,
                    dtype=np.float64,
                )
                * sigma_rw ** 2
                * dt
            )

            X_pred = X_state
            P_pred = P + Q

            z = np.array(
                [x[k], y[k]],
                dtype=np.float64,
            ).reshape(2, 1)

            innovation = z - X_pred

            S = P_pred + R
            S = 0.5 * (S + S.T)

            sign, logdet = np.linalg.slogdet(S)

            if sign <= 0.0 or not np.isfinite(logdet):
                return -np.inf, np.nan, np.nan

            try:
                solved = np.linalg.solve(
                    S,
                    innovation,
                )

                mahalanobis_sq = float(
                    (innovation.T @ solved).item()
                )

                if (
                    not np.isfinite(mahalanobis_sq)
                    or mahalanobis_sq < 0.0
                ):
                    return -np.inf, np.nan, np.nan

                log_likelihood = -0.5 * (
                    2.0 * log_2pi
                    + logdet
                    + mahalanobis_sq
                )

                K = np.linalg.solve(
                    S,
                    P_pred,
                ).T

                X_state = X_pred + K @ innovation

                I_KH = I - K

                P = (
                    I_KH @ P_pred @ I_KH.T
                    + K @ R @ K.T
                )

                P = 0.5 * (P + P.T)

                current_log_likelihood = log_likelihood
                current_mahalanobis_sq = mahalanobis_sq

                current_filter_distance = float(
                    np.hypot(
                        X_state[0, 0] - x[k],
                        X_state[1, 0] - y[k],
                    )
                )

            except np.linalg.LinAlgError:
                return -np.inf, np.nan, np.nan

        return (
            current_log_likelihood,
            current_mahalanobis_sq,
            current_filter_distance,
        )

    # ========================================================
    # Сбор CV likelihood
    # ========================================================

    @staticmethod
    def collect_cv_likelihood(
        segments: Iterable[TrackSegment],
        sigma_acc: float,
        sigma_meas: float,
        window: int = WINDOW,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Рассчитывает innovation statistics CV для всех точек участков.

        Для каждой точки P[i]:

            P[i-window] ... P[i]

        После чего сохраняется оценка именно P[i].
        """
        log_likelihoods = []
        mahalanobis = []
        filter_distances = []

        for segment in segments:
            if segment.size <= window:
                continue

            x, y = KalmanParameterEstimator.to_local_xy(segment.df)
            time = segment.df["time"].to_numpy().astype("datetime64[ns]")

            for i in range(window, len(x)):
                x_window = x[i - window:i + 1]
                y_window = y[i - window:i + 1]
                time_window = time[i - window:i + 1]

                ll, m2, fd = (
                    KalmanParameterEstimator._cv_log_likelihood_window(
                        x=x_window,
                        y=y_window,
                        time=time_window,
                        sigma_acc=sigma_acc,
                        sigma_meas=sigma_meas,
                    )
                )

                if np.isfinite(ll):
                    log_likelihoods.append(ll)

                if np.isfinite(m2):
                    mahalanobis.append(m2)

                if np.isfinite(fd):
                    filter_distances.append(fd)

        return {
            "log_likelihood": np.asarray(
                log_likelihoods,
                dtype=np.float64,
            ),
            "mahalanobis": np.asarray(
                mahalanobis,
                dtype=np.float64,
            ),
            "filter_distance": np.asarray(
                filter_distances,
                dtype=np.float64,
            ),
        }

    # ========================================================
    # Сбор RW likelihood
    # ========================================================

    @staticmethod
    def collect_rw_likelihood(
        segments: Iterable[TrackSegment],
        sigma_rw: float,
        sigma_meas: float,
        window: int = WINDOW,
    ) -> dict[str, NDArray[np.float64]]:
        """Рассчитывает innovation statistics RW для всех точек участков."""
        log_likelihoods = []
        mahalanobis = []
        filter_distances = []

        for segment in segments:
            if segment.size <= window:
                continue

            x, y = KalmanParameterEstimator.to_local_xy(segment.df)
            time = segment.df["time"].to_numpy().astype("datetime64[ns]")

            for i in range(window, len(x)):
                x_window = x[i - window:i + 1]
                y_window = y[i - window:i + 1]
                time_window = time[i - window:i + 1]

                ll, m2, fd = (
                    KalmanParameterEstimator._rw_log_likelihood_window(
                        x=x_window,
                        y=y_window,
                        time=time_window,
                        sigma_rw=sigma_rw,
                        sigma_meas=sigma_meas,
                    )
                )

                if np.isfinite(ll):
                    log_likelihoods.append(ll)

                if np.isfinite(m2):
                    mahalanobis.append(m2)

                if np.isfinite(fd):
                    filter_distances.append(fd)

        return {
            "log_likelihood": np.asarray(
                log_likelihoods,
                dtype=np.float64,
            ),
            "mahalanobis": np.asarray(
                mahalanobis,
                dtype=np.float64,
            ),
            "filter_distance": np.asarray(
                filter_distances,
                dtype=np.float64,
            ),
        }

    # ========================================================
    # MLE CV
    # ========================================================

    @staticmethod
    def _negative_log_likelihood_cv(
        log_parameters: NDArray[np.float64],
        segments: list[TrackSegment],
        window: int,
    ) -> float:
        """Функция минимизации для MLE CV."""
        sigma_meas = float(np.exp(log_parameters[0]))
        sigma_acc = float(np.exp(log_parameters[1]))

        if (
            not np.isfinite(sigma_meas)
            or not np.isfinite(sigma_acc)
            or sigma_meas <= 0.0
            or sigma_acc <= 0.0
        ):
            return np.inf

        total_log_likelihood = 0.0
        observation_count = 0

        for segment in segments:
            if segment.size <= window:
                continue

            x, y = KalmanParameterEstimator.to_local_xy(segment.df)
            time = segment.df["time"].to_numpy().astype("datetime64[ns]")

            for i in range(window, len(x)):
                x_window = x[i - window:i + 1]
                y_window = y[i - window:i + 1]
                time_window = time[i - window:i + 1]

                ll, _, _ = (
                    KalmanParameterEstimator._cv_log_likelihood_window(
                        x=x_window,
                        y=y_window,
                        time=time_window,
                        sigma_acc=sigma_acc,
                        sigma_meas=sigma_meas,
                    )
                )

                if not np.isfinite(ll):
                    return np.inf

                total_log_likelihood += ll
                observation_count += 1

        if observation_count == 0:
            return np.inf

        return -total_log_likelihood

    @staticmethod
    def fit_cv_mle(
        segments: list[TrackSegment],
        sigma_meas_initial: float,
        sigma_acc_initial: float,
        window: int = WINDOW,
    ) -> dict[str, float | bool | str]:
        """Оценивает sigma_meas и sigma_acc методом MLE."""
        sigma_meas_initial = float(
            np.clip(
                sigma_meas_initial,
                SIGMA_MEAS_MIN,
                SIGMA_MEAS_MAX,
            )
        )

        sigma_acc_initial = float(
            np.clip(
                sigma_acc_initial,
                SIGMA_ACC_MIN,
                SIGMA_ACC_MAX,
            )
        )

        initial = np.log(
            [
                sigma_meas_initial,
                sigma_acc_initial,
            ]
        )

        bounds = [
            (
                np.log(SIGMA_MEAS_MIN),
                np.log(SIGMA_MEAS_MAX),
            ),
            (
                np.log(SIGMA_ACC_MIN),
                np.log(SIGMA_ACC_MAX),
            ),
        ]

        starts = [
            initial,
            initial + np.log([2.0, 2.0]),
            initial + np.log([0.5, 2.0]),
            initial + np.log([2.0, 0.5]),
            initial + np.log([0.5, 0.5]),
        ]

        best_result = None

        for start in starts:
            start = np.clip(
                start,
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            )

            result = minimize(
                KalmanParameterEstimator._negative_log_likelihood_cv,
                x0=start,
                args=(segments, window),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 300,
                    "ftol": 1e-9,
                },
            )

            if (
                best_result is None
                or result.fun < best_result.fun
            ):
                best_result = result

        if best_result is None:
            return {
                "success": False,
                "sigma_meas": np.nan,
                "sigma_acc": np.nan,
                "negative_log_likelihood": np.nan,
                "message": "optimizer returned no result",
            }

        sigma_meas = float(np.exp(best_result.x[0]))
        sigma_acc = float(np.exp(best_result.x[1]))

        return {
            "success": bool(best_result.success),
            "sigma_meas": sigma_meas,
            "sigma_acc": sigma_acc,
            "negative_log_likelihood": float(best_result.fun),
            "message": str(best_result.message),
        }

    # ========================================================
    # MLE RW
    # ========================================================

    @staticmethod
    def _negative_log_likelihood_rw(
        log_parameters: NDArray[np.float64],
        segments: list[TrackSegment],
        window: int,
    ) -> float:
        """Функция минимизации для MLE RW."""
        sigma_meas = float(np.exp(log_parameters[0]))
        sigma_rw = float(np.exp(log_parameters[1]))

        if (
            not np.isfinite(sigma_meas)
            or not np.isfinite(sigma_rw)
            or sigma_meas <= 0.0
            or sigma_rw <= 0.0
        ):
            return np.inf

        total_log_likelihood = 0.0
        observation_count = 0

        for segment in segments:
            if segment.size <= window:
                continue

            x, y = KalmanParameterEstimator.to_local_xy(segment.df)
            time = segment.df["time"].to_numpy().astype("datetime64[ns]")

            for i in range(window, len(x)):
                x_window = x[i - window:i + 1]
                y_window = y[i - window:i + 1]
                time_window = time[i - window:i + 1]

                ll, _, _ = (
                    KalmanParameterEstimator._rw_log_likelihood_window(
                        x=x_window,
                        y=y_window,
                        time=time_window,
                        sigma_rw=sigma_rw,
                        sigma_meas=sigma_meas,
                    )
                )

                if not np.isfinite(ll):
                    return np.inf

                total_log_likelihood += ll
                observation_count += 1

        if observation_count == 0:
            return np.inf

        return -total_log_likelihood

    @staticmethod
    def fit_rw_mle(
        segments: list[TrackSegment],
        sigma_meas_initial: float,
        sigma_rw_initial: float,
        window: int = WINDOW,
    ) -> dict[str, float | bool | str]:
        """Оценивает sigma_meas и sigma_rw методом MLE."""
        sigma_meas_initial = float(
            np.clip(
                sigma_meas_initial,
                SIGMA_MEAS_MIN,
                SIGMA_MEAS_MAX,
            )
        )

        sigma_rw_initial = float(
            np.clip(
                sigma_rw_initial,
                SIGMA_RW_MIN,
                SIGMA_RW_MAX,
            )
        )

        initial = np.log(
            [
                sigma_meas_initial,
                sigma_rw_initial,
            ]
        )

        bounds = [
            (
                np.log(SIGMA_MEAS_MIN),
                np.log(SIGMA_MEAS_MAX),
            ),
            (
                np.log(SIGMA_RW_MIN),
                np.log(SIGMA_RW_MAX),
            ),
        ]

        starts = [
            initial,
            initial + np.log([2.0, 2.0]),
            initial + np.log([0.5, 2.0]),
            initial + np.log([2.0, 0.5]),
            initial + np.log([0.5, 0.5]),
        ]

        best_result = None

        for start in starts:
            start = np.clip(
                start,
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            )

            result = minimize(
                KalmanParameterEstimator._negative_log_likelihood_rw,
                x0=start,
                args=(segments, window),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 300,
                    "ftol": 1e-9,
                },
            )

            if (
                best_result is None
                or result.fun < best_result.fun
            ):
                best_result = result

        if best_result is None:
            return {
                "success": False,
                "sigma_meas": np.nan,
                "sigma_rw": np.nan,
                "negative_log_likelihood": np.nan,
                "message": "optimizer returned no result",
            }

        sigma_meas = float(np.exp(best_result.x[0]))
        sigma_rw = float(np.exp(best_result.x[1]))

        return {
            "success": bool(best_result.success),
            "sigma_meas": sigma_meas,
            "sigma_rw": sigma_rw,
            "negative_log_likelihood": float(best_result.fun),
            "message": str(best_result.message),
        }

    # ========================================================
    # Проверка Mahalanobis
    # ========================================================

    @staticmethod
    def validate_model(
        mahalanobis: NDArray[np.float64],
        sigma_meas: float,
        process_sigma: float,
        optimizer_success: bool,
        process_name: str,
    ) -> dict[str, float | str]:
        """
        Проверяет калибровку модели по Mahalanobis².

        Для двумерной нормальной инновации при корректной модели:

            Mahalanobis² ~ chi2(df=2)

        Поэтому:

            E[M²] = 2
            P95[M²] = chi2.ppf(0.95, 2)
        """
        values = np.asarray(
            mahalanobis,
            dtype=np.float64,
        )

        values = values[np.isfinite(values)]

        expected_mean = 2.0
        expected_p95 = float(
            chi2.ppf(
                0.95,
                df=2,
            )
        )

        if values.size == 0:
            return {
                "validation_mean_mahalanobis": np.nan,
                "validation_p95_mahalanobis": np.nan,
                "validation_chi2_mean_expected": expected_mean,
                "validation_chi2_p95_expected": expected_p95,
                "validation_mean_ratio": np.nan,
                "validation_p95_ratio": np.nan,
                "validation_quality": "NO_DATA",
                "validation_message": f"{process_name}: нет данных",
            }

        mean_m2 = float(
            np.mean(values)
        )

        p95_m2 = float(
            np.percentile(
                values,
                95,
            )
        )

        mean_ratio = (
            mean_m2 / expected_mean
        )

        p95_ratio = (
            p95_m2 / expected_p95
        )

        if not optimizer_success:
            quality = "OPTIMIZER_FAILED"
        elif not (
            np.isfinite(sigma_meas)
            and sigma_meas > 0.0
            and np.isfinite(process_sigma)
            and process_sigma > 0.0
        ):
            quality = "INVALID_PARAMETERS"
        elif (
            0.7 <= mean_ratio <= 1.3
            and 0.7 <= p95_ratio <= 1.3
        ):
            quality = "GOOD"
        elif (
            0.5 <= mean_ratio <= 1.7
            and 0.5 <= p95_ratio <= 1.7
        ):
            quality = "CHECK"
        else:
            quality = "POOR"

        return {
            "validation_mean_mahalanobis": mean_m2,
            "validation_p95_mahalanobis": p95_m2,
            "validation_chi2_mean_expected": expected_mean,
            "validation_chi2_p95_expected": expected_p95,
            "validation_mean_ratio": mean_ratio,
            "validation_p95_ratio": p95_ratio,
            "validation_quality": quality,
            "validation_message": (
                f"{process_name}: "
                f"mean(M²)={mean_m2:.6g}, "
                f"P95(M²)={p95_m2:.6g}"
            ),
        }


# ============================================================
# Обработка одного файла
# ============================================================

def process_file(
    path: Path,
    output_rows: list[dict],
) -> None:
    """Обрабатывает один независимый обучающий эксперимент."""
    logging.info("Начало обработки файла: %s", path.name)

    df = DataProcessor.load_csv(path)

    logging.info(
        "%s: исходных точек: %d",
        path.name,
        len(df),
    )

    df = DataProcessor.pre_filter(df)

    logging.info(
        "%s: точек после pre_filter: %d",
        path.name,
        len(df),
    )

    segments = (
        KalmanParameterEstimator.extract_segments(
            df=df,
            file_name=path.name,
            min_length=MIN_SEGMENT_LENGTH,
        )
    )

    if not segments:
        logging.warning(
            "%s: подходящих участков не найдено",
            path.name,
        )
        return

    segment_sets = {
        "stand": [
            segment
            for segment in segments
            if segment.segment_type == "stand"
        ],
        "move": [
            segment
            for segment in segments
            if segment.segment_type == "move"
        ],
        "stand_move": [
            segment
            for segment in segments
            if segment.segment_type == "stand_move"
        ],
    }

    # --------------------------------------------------------
    # Информация об участках
    # --------------------------------------------------------

    for segment_type, current_segments in segment_sets.items():
        total_points = sum(
            segment.size
            for segment in current_segments
        )

        logging.info(
            "%s / %s: участков=%d, точек=%d",
            path.name,
            segment_type,
            len(current_segments),
            total_points,
        )

        for segment in current_segments:
            logging.debug(
                "%s / %s / segment=%d / N=%d",
                path.name,
                segment_type,
                segment.segment_id,
                segment.size,
            )

    # --------------------------------------------------------
    # sigma_meas из stand
    # --------------------------------------------------------

    stand_segments = segment_sets["stand"]

    if stand_segments:
        position_sigma = (
            KalmanParameterEstimator
            .estimate_sigma_meas_from_position(
                stand_segments
            )
        )

        neighbor_sigma = (
            KalmanParameterEstimator
            .estimate_sigma_meas_from_neighbor_difference(
                stand_segments
            )
        )

        n_stand_points = sum(
            segment.size
            for segment in stand_segments
        )

        logging.info(
            "%s / stand / position_std: "
            "sigma_x=%.8g, sigma_y=%.8g, sigma_meas=%.8g",
            path.name,
            position_sigma["sigma_meas_x"],
            position_sigma["sigma_meas_y"],
            position_sigma["sigma_meas"],
        )

        logging.info(
            "%s / stand / neighbor_diff: "
            "sigma_x=%.8g, sigma_y=%.8g, sigma_meas=%.8g",
            path.name,
            neighbor_sigma["sigma_meas_x"],
            neighbor_sigma["sigma_meas_y"],
            neighbor_sigma["sigma_meas"],
        )

        output_rows.append(
            {
                "file": path.name,
                "segment_type": "stand",
                "method": "stand_position_std",
                "n_segments": len(stand_segments),
                "n_points": n_stand_points,
                "sigma_meas_x": position_sigma["sigma_meas_x"],
                "sigma_meas_y": position_sigma["sigma_meas_y"],
                "sigma_meas": position_sigma["sigma_meas"],
                "sigma_acc": np.nan,
                "sigma_rw": np.nan,
                "negative_log_likelihood": np.nan,
                "optimizer_success": np.nan,
            }
        )

        output_rows.append(
            {
                "file": path.name,
                "segment_type": "stand",
                "method": "stand_neighbor_diff",
                "n_segments": len(stand_segments),
                "n_points": neighbor_sigma["count"],
                "sigma_meas_x": neighbor_sigma["sigma_meas_x"],
                "sigma_meas_y": neighbor_sigma["sigma_meas_y"],
                "sigma_meas": neighbor_sigma["sigma_meas"],
                "sigma_acc": np.nan,
                "sigma_rw": np.nan,
                "negative_log_likelihood": np.nan,
                "optimizer_success": np.nan,
            }
        )

        sigma_meas_initial = float(
            np.nanmean(
                [
                    position_sigma["sigma_meas"],
                    neighbor_sigma["sigma_meas"],
                ]
            )
        )

        if not np.isfinite(sigma_meas_initial):
            sigma_meas_initial = 2.4

    else:
        logging.warning(
            "%s: stand-участков длиной >= %d не найдено",
            path.name,
            MIN_SEGMENT_LENGTH,
        )

        sigma_meas_initial = 2.4

    # ========================================================
    # MLE для каждого типа участка
    # ========================================================

    for segment_type, current_segments in segment_sets.items():
        if not current_segments:
            logging.warning(
                "%s / %s: участки отсутствуют, MLE пропущен",
                path.name,
                segment_type,
            )
            continue

        total_points = sum(
            segment.size
            for segment in current_segments
        )

        logging.info(
            "%s / %s: запуск MLE, segments=%d, points=%d",
            path.name,
            segment_type,
            len(current_segments),
            total_points,
        )

        # ----------------------------------------------------
        # MLE CV
        # ----------------------------------------------------

        sigma_acc_initial = 0.04

        cv_result = (
            KalmanParameterEstimator.fit_cv_mle(
                segments=current_segments,
                sigma_meas_initial=sigma_meas_initial,
                sigma_acc_initial=sigma_acc_initial,
                window=WINDOW,
            )
        )

        sigma_meas_cv = float(
            cv_result["sigma_meas"]
        )

        sigma_acc_cv = float(
            cv_result["sigma_acc"]
        )

        logging.info(
            "%s / %s / CV MLE: "
            "success=%s, sigma_meas=%.8g, sigma_acc=%.8g, NLL=%.8g",
            path.name,
            segment_type,
            cv_result["success"],
            sigma_meas_cv,
            sigma_acc_cv,
            cv_result["negative_log_likelihood"],
        )

        cv_metrics = (
            KalmanParameterEstimator.collect_cv_likelihood(
                segments=current_segments,
                sigma_acc=sigma_acc_cv,
                sigma_meas=sigma_meas_cv,
                window=WINDOW,
            )
        )

        cv_validation = (
            KalmanParameterEstimator.validate_model(
                mahalanobis=cv_metrics["mahalanobis"],
                sigma_meas=sigma_meas_cv,
                process_sigma=sigma_acc_cv,
                optimizer_success=bool(
                    cv_result["success"]
                ),
                process_name="CV",
            )
        )

        logging.info(
            "%s / %s / CV validation: %s",
            path.name,
            segment_type,
            cv_validation["validation_message"],
        )

        cv_row = {
            "file": path.name,
            "segment_type": segment_type,
            "method": "MLE_CV",
            "n_segments": len(current_segments),
            "n_points": total_points,
            "sigma_meas_x": np.nan,
            "sigma_meas_y": np.nan,
            "sigma_meas": sigma_meas_cv,
            "sigma_acc": sigma_acc_cv,
            "sigma_rw": np.nan,
            "negative_log_likelihood": cv_result[
                "negative_log_likelihood"
            ],
            "optimizer_success": cv_result["success"],
        }

        cv_row.update(cv_validation)

        output_rows.append(cv_row)

        # ----------------------------------------------------
        # MLE RW
        # ----------------------------------------------------

        sigma_rw_initial = max(
            sigma_meas_initial / np.sqrt(10.0),
            SIGMA_RW_MIN,
        )

        rw_result = (
            KalmanParameterEstimator.fit_rw_mle(
                segments=current_segments,
                sigma_meas_initial=sigma_meas_initial,
                sigma_rw_initial=sigma_rw_initial,
                window=WINDOW,
            )
        )

        sigma_meas_rw = float(
            rw_result["sigma_meas"]
        )

        sigma_rw = float(
            rw_result["sigma_rw"]
        )

        logging.info(
            "%s / %s / RW MLE: "
            "success=%s, sigma_meas=%.8g, sigma_rw=%.8g, NLL=%.8g",
            path.name,
            segment_type,
            rw_result["success"],
            sigma_meas_rw,
            sigma_rw,
            rw_result["negative_log_likelihood"],
        )

        rw_metrics = (
            KalmanParameterEstimator.collect_rw_likelihood(
                segments=current_segments,
                sigma_rw=sigma_rw,
                sigma_meas=sigma_meas_rw,
                window=WINDOW,
            )
        )

        rw_validation = (
            KalmanParameterEstimator.validate_model(
                mahalanobis=rw_metrics["mahalanobis"],
                sigma_meas=sigma_meas_rw,
                process_sigma=sigma_rw,
                optimizer_success=bool(
                    rw_result["success"]
                ),
                process_name="RW",
            )
        )

        logging.info(
            "%s / %s / RW validation: %s",
            path.name,
            segment_type,
            rw_validation["validation_message"],
        )

        rw_row = {
            "file": path.name,
            "segment_type": segment_type,
            "method": "MLE_RW",
            "n_segments": len(current_segments),
            "n_points": total_points,
            "sigma_meas_x": np.nan,
            "sigma_meas_y": np.nan,
            "sigma_meas": sigma_meas_rw,
            "sigma_acc": np.nan,
            "sigma_rw": sigma_rw,
            "negative_log_likelihood": rw_result[
                "negative_log_likelihood"
            ],
            "optimizer_success": rw_result["success"],
        }

        rw_row.update(rw_validation)

        output_rows.append(rw_row)

    logging.info(
        "Обработка файла %s завершена",
        path.name,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    path_root = Path(__file__).parent

    train_paths = [
        path_root / "data" / "1.csv",
        path_root / "data" / "2.csv",
        path_root / "data" / "3.csv",
    ]

    output_path = (
        path_root
        / "statistics"
        / OUTPUT_FILENAME
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info(
        "Начало оценки параметров фильтров Калмана"
    )

    logging.info(
        "Обучающие файлы: %s",
        ", ".join(
            path.name
            for path in train_paths
        ),
    )

    logging.info(
        "Минимальная длина участка: %d",
        MIN_SEGMENT_LENGTH,
    )

    logging.info(
        "Окно Kalman: %d",
        WINDOW,
    )

    all_results: list[dict] = []

    for path in train_paths:
        process_file(
            path=path,
            output_rows=all_results,
        )

    result_df = pd.DataFrame(
        all_results
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    logging.info(
        "Результаты записаны в %s",
        output_path,
    )

    if result_df.empty:
        logging.warning(
            "Итоговый CSV пуст"
        )
    else:
        logging.info(
            "Получено строк результата: %d",
            len(result_df),
        )

        print(
            "\nИтоговая таблица:\n"
        )

        columns_to_print = [
            "file",
            "segment_type",
            "method",
            "n_segments",
            "n_points",
            "sigma_meas_x",
            "sigma_meas_y",
            "sigma_meas",
            "sigma_acc",
            "sigma_rw",
            "validation_mean_mahalanobis",
            "validation_p95_mahalanobis",
            "validation_mean_ratio",
            "validation_p95_ratio",
            "validation_quality",
        ]

        existing_columns = [
            column
            for column in columns_to_print
            if column in result_df.columns
        ]

        print(
            result_df[
                existing_columns
            ].to_string(
                index=False
            )
        )

    logging.info(
        "Оценка параметров завершена"
    )