
"""
Оценка параметров фильтров Калмана CV и RW.

Обучающие эксперименты:
    1.csv
    2.csv

Валидационный эксперимент:
    3.csv

1.csv и 2.csv обрабатываются как независимые эксперименты.
3.csv не участвует в подборе параметров.

Для каждой точки P[i] финальная innovation-статистика рассчитывается
по локальному окну:

    P[i-WINDOW] ... P[i]

Локальная система координат каждого окна начинается в первой точке окна.

MLE использует неперекрывающиеся окна:

    MLE_STRIDE = WINDOW + 1

Финальные статистики после калибровки рассчитываются для всех точек
с шагом 1.
"""

from __future__ import annotations

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
MLE_STRIDE = WINDOW + 1
BATCH_SIZE = 4096

OPTIMIZER_N_STARTS = 3
OPTIMIZER_MAXITER = 80
OPTIMIZER_FTOL = 1e-7
OPTIMIZER_LOG_EVERY = 10

SIGMA_MEAS_MIN = 1e-3
SIGMA_MEAS_MAX = 1000.0

SIGMA_ACC_MIN = 1e-7
SIGMA_ACC_MAX = 10.0

SIGMA_RW_MIN = 1e-7
SIGMA_RW_MAX = 100.0

DEFAULT_SIGMA_ACC_INITIAL = 0.04

OUTPUT_FILENAME = "file_2_kalman_parameter_estimates.csv"


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
        return len(self.df)


# ============================================================
# Подготовленные данные сегмента
# ============================================================

@dataclass
class PreparedSegment:
    """
    NumPy-представление TrackSegment для высокоскоростного MLE.

    Координаты остаются в lon/lat.
    Локальные X/Y строятся отдельно для каждого окна.
    """

    file_name: str
    segment_type: str
    segment_id: int
    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64]
    mark: NDArray[np.str_]
    dt: NDArray[np.float64]

    @property
    def size(self) -> int:
        return len(self.lon)

    @classmethod
    def from_segment(cls, segment: TrackSegment) -> PreparedSegment:
        lon = segment.df["lon"].to_numpy(dtype=np.float64)
        lat = segment.df["lat"].to_numpy(dtype=np.float64)
        time = pd.to_datetime(segment.df["time"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        mark = segment.df["status"].to_numpy(dtype=str)

        if len(lon) != len(lat) or len(lon) != len(time):
            raise ValueError(
                f"{segment.file_name} / {segment.segment_type} / "
                f"segment={segment.segment_id}: несовпадающая длина координат/time"
            )

        if len(lon) < 2:
            dt = np.empty(0, dtype=np.float64)
        else:
            dt = ((time[1:] - time[:-1]) / np.timedelta64(1, "s")).astype(np.float64)
            dt = np.maximum(dt, 0.0)

        if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
            raise ValueError(
                f"{segment.file_name} / {segment.segment_type} / "
                f"segment={segment.segment_id}: обнаружены NaN/Inf в координатах"
            )

        if not np.all(np.isfinite(dt)):
            raise ValueError(
                f"{segment.file_name} / {segment.segment_type} / "
                f"segment={segment.segment_id}: обнаружены некорректные значения time/dt"
            )

        return cls(
            file_name=segment.file_name,
            segment_type=segment.segment_type,
            segment_id=segment.segment_id,
            lon=lon,
            lat=lat,
            time=time,
            mark=mark,
            dt=dt,
        )


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

        Для stand_move разрешены только статусы stand и move.
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

        stand_move_mask = df["status"].isin(["stand", "move"]).to_numpy()
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
        """Выделяет непрерывные True-сегменты."""

        if len(df) != len(mask):
            raise ValueError("Длина mask не совпадает с DataFrame")

        if len(df) == 0:
            return []

        starts = mask & ~np.r_[False, mask[:-1]]
        ends = mask & ~np.r_[mask[1:], False]

        start_indices = np.flatnonzero(starts)
        end_indices = np.flatnonzero(ends)

        segments: list[TrackSegment] = []

        for segment_id, (start, end) in enumerate(zip(start_indices, end_indices)):
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
    # Подготовка данных
    # ========================================================

    @staticmethod
    def prepare_segments(segments: Iterable[TrackSegment]) -> list[PreparedSegment]:
        """
        Один раз преобразует DataFrame-сегменты в NumPy-представление.

        DataFrame и datetime не обрабатываются внутри objective.
        """

        prepared: list[PreparedSegment] = []

        for segment in segments:
            if segment.size <= WINDOW:
                continue

            prepared.append(PreparedSegment.from_segment(segment))

        return prepared

    # ========================================================
    # Координаты
    # ========================================================

    @staticmethod
    def to_local_xy(
        df: pd.DataFrame,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Переводит один участок из lon/lat в локальные X/Y."""

        lon = df["lon"].to_numpy(dtype=np.float64)
        lat = df["lat"].to_numpy(dtype=np.float64)

        return DataProcessor.convert_to_local_cartesian(lon, lat)

    # ========================================================
    # Формирование локальных окон
    # ========================================================

    @staticmethod
    def _build_local_windows(
        segment: PreparedSegment,
        start_indices: NDArray[np.int64],
        window: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Формирует локальные координаты для набора окон.

        Для каждого окна:

            P[start] ... P[start + window]

        начало локальной СК находится в P[start].
        """

        if len(start_indices) == 0:
            empty = np.empty((0, window + 1), dtype=np.float64)
            return empty, empty.copy(), np.empty((0, window), dtype=np.float64)

        starts = np.asarray(start_indices, dtype=np.int64)
        offsets = np.arange(window + 1, dtype=np.int64)
        indices = starts[:, None] + offsets[None, :]

        lon_windows = segment.lon[indices]
        lat_windows = segment.lat[indices]

        lon0 = lon_windows[:, 0]
        lat0 = lat_windows[:, 0]

        kx = DataProcessor.LEN_LAT * np.cos(np.radians(lat0))

        x_windows = (lon_windows - lon0[:, None]) * kx[:, None]
        y_windows = (lat_windows - lat0[:, None]) * DataProcessor.LEN_LAT

        dt_indices = starts[:, None] + np.arange(window, dtype=np.int64)[None, :] + 1
        dt_windows = segment.dt[dt_indices - 1]

        return x_windows, y_windows, dt_windows

    # ========================================================
    # sigma_meas: разброс положения
    # ========================================================

    @staticmethod
    def estimate_sigma_meas_from_position(
        segments: Iterable[TrackSegment],
    ) -> dict[str, float]:
        """
        Оценивает sigma_meas по разбросу положения на stand.

            sigma_x = std(x - mean(x))
            sigma_y = std(y - mean(y))

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

            sum_sq_x += float(np.sum(x_centered ** 2))
            sum_sq_y += float(np.sum(y_centered ** 2))
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
        sigma_meas = np.sqrt((sigma_x ** 2 + sigma_y ** 2) / 2.0)

        return {
            "sigma_meas_x": float(sigma_x),
            "sigma_meas_y": float(sigma_y),
            "sigma_meas": float(sigma_meas),
            "count": int(count),
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

        delta_x_list: list[NDArray[np.float64]] = []
        delta_y_list: list[NDArray[np.float64]] = []

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
        sigma_meas = np.sqrt((sigma_x ** 2 + sigma_y ** 2) / 2.0)

        return {
            "sigma_meas_x": float(sigma_x),
            "sigma_meas_y": float(sigma_y),
            "sigma_meas": float(sigma_meas),
            "count": int(len(delta_x)),
        }

    # ========================================================
    # CV: пакетное вычисление
    # ========================================================

    @staticmethod
    def _cv_batch_statistics(
        x_windows: NDArray[np.float64],
        y_windows: NDArray[np.float64],
        dt_windows: NDArray[np.float64],
        sigma_acc: float,
        sigma_meas: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Пакетная обработка CV-окон.

        Для каждого окна рассчитываются метрики последнего наблюдения.
        """

        batch_size = len(x_windows)

        if batch_size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy(), empty.copy()

        state_dim = 4

        R = np.eye(2, dtype=np.float64) * sigma_meas ** 2
        I = np.eye(state_dim, dtype=np.float64)
        H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        log_2pi = np.log(2.0 * np.pi)

        dt0 = dt_windows[:, 0]

        vx0 = np.divide(
            x_windows[:, 1] - x_windows[:, 0],
            dt0,
            out=np.zeros(batch_size),
            where=dt0 > 0.0,
        )

        vy0 = np.divide(
            y_windows[:, 1] - y_windows[:, 0],
            dt0,
            out=np.zeros(batch_size),
            where=dt0 > 0.0,
        )

        state = np.column_stack([
            x_windows[:, 0],
            y_windows[:, 0],
            vx0,
            vy0,
        ])

        P = np.broadcast_to(
            I * 500.0,
            (batch_size, state_dim, state_dim),
        ).copy()

        positive_dt0 = dt0 > 0.0
        vel_var = np.full(batch_size, 100.0, dtype=np.float64)

        vel_var[positive_dt0] = (
            2.0 * sigma_meas ** 2 / dt0[positive_dt0] ** 2
        )

        P[:, 2, 2] = vel_var
        P[:, 3, 3] = vel_var

        current_log_likelihood = np.full(batch_size, np.nan, dtype=np.float64)
        current_mahalanobis_sq = np.full(batch_size, np.nan, dtype=np.float64)
        current_filter_distance = np.full(batch_size, np.nan, dtype=np.float64)

        window = x_windows.shape[1] - 1

        for k in range(1, window + 1):
            dt = dt_windows[:, k - 1]

            F = np.zeros((batch_size, state_dim, state_dim), dtype=np.float64)
            F[:, 0, 0] = 1.0
            F[:, 1, 1] = 1.0
            F[:, 2, 2] = 1.0
            F[:, 3, 3] = 1.0
            F[:, 0, 2] = dt
            F[:, 1, 3] = dt

            dt2 = dt ** 2
            dt3 = dt ** 3
            dt4 = dt ** 4
            variance_acc = sigma_acc ** 2

            Q = np.zeros_like(P)
            Q[:, 0, 0] = dt4 / 4.0 * variance_acc
            Q[:, 1, 1] = dt4 / 4.0 * variance_acc
            Q[:, 2, 2] = dt2 * variance_acc
            Q[:, 3, 3] = dt2 * variance_acc
            Q[:, 0, 2] = dt3 / 2.0 * variance_acc
            Q[:, 2, 0] = dt3 / 2.0 * variance_acc
            Q[:, 1, 3] = dt3 / 2.0 * variance_acc
            Q[:, 3, 1] = dt3 / 2.0 * variance_acc

            state_pred = np.einsum("bij,bj->bi", F, state)
            FP = np.einsum("bij,bjk->bik", F, P)
            P_pred = np.einsum("bik,bjk->bij", FP, F) + Q

            z = np.column_stack([
                x_windows[:, k],
                y_windows[:, k],
            ])

            innovation = z - state_pred[:, :2]
            S = P_pred[:, :2, :2] + R
            S = 0.5 * (S + np.transpose(S, (0, 2, 1)))

            sign, logdet = np.linalg.slogdet(S)
            valid_S = (sign > 0.0) & np.isfinite(logdet)

            try:
                solved = np.linalg.solve(S, innovation[..., None])[..., 0]

                mahalanobis_sq = np.sum(innovation * solved, axis=1)

                valid = (
                    valid_S
                    & np.isfinite(mahalanobis_sq)
                    & (mahalanobis_sq >= 0.0)
                )

                log_likelihood = -0.5 * (
                    2.0 * log_2pi + logdet + mahalanobis_sq
                )

                solved_gain = np.linalg.solve(S, P_pred[:, :2, :])
                K = np.transpose(solved_gain, (0, 2, 1))

                state = state_pred + np.einsum(
                    "bij,bj->bi",
                    K,
                    innovation,
                )

                I_KH = I[None, :, :] - np.einsum(
                    "bij,jk->bik",
                    K,
                    H,
                )

                tmp = np.einsum(
                    "bij,bjk->bik",
                    I_KH,
                    P_pred,
                )

                P = np.einsum(
                    "bik,bjk->bij",
                    tmp,
                    I_KH,
                )

                P += np.einsum(
                    "bij,bjk->bik",
                    K @ R,
                    np.transpose(K, (0, 2, 1)),
                )

                P = 0.5 * (
                    P + np.transpose(P, (0, 2, 1))
                )

                current_log_likelihood[valid] = log_likelihood[valid]
                current_mahalanobis_sq[valid] = mahalanobis_sq[valid]
                current_filter_distance[valid] = np.hypot(
                    state[valid, 0] - x_windows[valid, k],
                    state[valid, 1] - y_windows[valid, k],
                )

            except np.linalg.LinAlgError:
                return (
                    np.full(batch_size, -np.inf),
                    np.full(batch_size, np.nan),
                    np.full(batch_size, np.nan),
                )

        return (
            current_log_likelihood,
            current_mahalanobis_sq,
            current_filter_distance,
        )

    # ========================================================
    # RW: пакетное вычисление
    # ========================================================

    @staticmethod
    def _rw_batch_statistics(
        x_windows: NDArray[np.float64],
        y_windows: NDArray[np.float64],
        dt_windows: NDArray[np.float64],
        sigma_rw: float,
        sigma_meas: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Пакетное вычисление локального RW-фильтра."""

        batch_size = len(x_windows)

        if batch_size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy(), empty.copy()

        R = np.eye(2, dtype=np.float64) * sigma_meas ** 2
        I = np.eye(2, dtype=np.float64)
        log_2pi = np.log(2.0 * np.pi)

        state = np.column_stack([
            x_windows[:, 0],
            y_windows[:, 0],
        ])

        P = np.broadcast_to(
            I * sigma_meas ** 2,
            (batch_size, 2, 2),
        ).copy()

        current_log_likelihood = np.full(batch_size, np.nan, dtype=np.float64)
        current_mahalanobis_sq = np.full(batch_size, np.nan, dtype=np.float64)
        current_filter_distance = np.full(batch_size, np.nan, dtype=np.float64)

        window = x_windows.shape[1] - 1

        for k in range(1, window + 1):
            dt = dt_windows[:, k - 1]

            Q = np.eye(2, dtype=np.float64)[None, :, :] * (
                sigma_rw ** 2 * dt
            )[:, None, None]

            state_pred = state
            P_pred = P + Q

            z = np.column_stack([
                x_windows[:, k],
                y_windows[:, k],
            ])

            innovation = z - state_pred
            S = P_pred + R
            S = 0.5 * (S + np.transpose(S, (0, 2, 1)))

            sign, logdet = np.linalg.slogdet(S)
            valid_S = (sign > 0.0) & np.isfinite(logdet)

            try:
                solved = np.linalg.solve(S, innovation[..., None])[..., 0]
                mahalanobis_sq = np.sum(innovation * solved, axis=1)

                valid = (
                    valid_S
                    & np.isfinite(mahalanobis_sq)
                    & (mahalanobis_sq >= 0.0)
                )

                log_likelihood = -0.5 * (
                    2.0 * log_2pi + logdet + mahalanobis_sq
                )

                K = np.transpose(
                    np.linalg.solve(S, P_pred),
                    (0, 2, 1),
                )

                state = state_pred + np.einsum(
                    "bij,bj->bi",
                    K,
                    innovation,
                )

                I_K = I[None, :, :] - K

                tmp = np.einsum(
                    "bij,bjk->bik",
                    I_K,
                    P_pred,
                )

                P = np.einsum(
                    "bik,bjk->bij",
                    tmp,
                    I_K,
                )

                P += np.einsum(
                    "bij,bjk->bik",
                    K @ R,
                    np.transpose(K, (0, 2, 1)),
                )

                P = 0.5 * (
                    P + np.transpose(P, (0, 2, 1))
                )

                current_log_likelihood[valid] = log_likelihood[valid]
                current_mahalanobis_sq[valid] = mahalanobis_sq[valid]
                current_filter_distance[valid] = np.hypot(
                    state[valid, 0] - x_windows[valid, k],
                    state[valid, 1] - y_windows[valid, k],
                )

            except np.linalg.LinAlgError:
                return (
                    np.full(batch_size, -np.inf),
                    np.full(batch_size, np.nan),
                    np.full(batch_size, np.nan),
                )

        return (
            current_log_likelihood,
            current_mahalanobis_sq,
            current_filter_distance,
        )

    # ========================================================
    # Индексы окон
    # ========================================================

    @staticmethod
    def _window_start_indices(
        segment: PreparedSegment,
        window: int,
        stride: int,
    ) -> NDArray[np.int64]:
        """Возвращает индексы начала локальных окон."""

        n_windows = segment.size - window

        if n_windows <= 0:
            return np.empty(0, dtype=np.int64)

        return np.arange(0, n_windows, stride, dtype=np.int64)

    # ========================================================
    # Оценка CV
    # ========================================================

    @staticmethod
    def _evaluate_cv_segments(
        segments: list[PreparedSegment],
        sigma_acc: float,
        sigma_meas: float,
        window: int,
        stride: int,
        collect_metrics: bool,
    ) -> dict[str, NDArray[np.float64] | float | int]:
        """Единый высокопроизводительный расчёт CV."""

        ll_list: list[NDArray[np.float64]] = []
        m2_list: list[NDArray[np.float64]] = []
        fd_list: list[NDArray[np.float64]] = []

        total_log_likelihood = 0.0
        observation_count = 0

        for segment in segments:
            starts = KalmanParameterEstimator._window_start_indices(
                segment=segment,
                window=window,
                stride=stride,
            )

            if len(starts) == 0:
                continue

            for begin in range(0, len(starts), BATCH_SIZE):
                end = min(begin + BATCH_SIZE, len(starts))
                current_starts = starts[begin:end]

                x_windows, y_windows, dt_windows = (
                    KalmanParameterEstimator._build_local_windows(
                        segment=segment,
                        start_indices=current_starts,
                        window=window,
                    )
                )

                log_likelihood, mahalanobis_sq, filter_distance = (
                    KalmanParameterEstimator._cv_batch_statistics(
                        x_windows=x_windows,
                        y_windows=y_windows,
                        dt_windows=dt_windows,
                        sigma_acc=sigma_acc,
                        sigma_meas=sigma_meas,
                    )
                )

                valid_ll = log_likelihood[np.isfinite(log_likelihood)]

                if len(valid_ll):
                    total_log_likelihood += float(np.sum(valid_ll))
                    observation_count += len(valid_ll)

                if collect_metrics:
                    valid = (
                        np.isfinite(log_likelihood)
                        & np.isfinite(mahalanobis_sq)
                        & np.isfinite(filter_distance)
                    )

                    if np.any(valid):
                        ll_list.append(log_likelihood[valid])
                        m2_list.append(mahalanobis_sq[valid])
                        fd_list.append(filter_distance[valid])

        if collect_metrics:
            ll = np.concatenate(ll_list) if ll_list else np.empty(0, dtype=np.float64)
            m2 = np.concatenate(m2_list) if m2_list else np.empty(0, dtype=np.float64)
            fd = np.concatenate(fd_list) if fd_list else np.empty(0, dtype=np.float64)

            return {
                "log_likelihood": ll,
                "mahalanobis": m2,
                "filter_distance": fd,
                "total_log_likelihood": float(np.sum(ll)),
                "observation_count": int(len(ll)),
            }

        return {
            "total_log_likelihood": total_log_likelihood,
            "observation_count": observation_count,
        }

    # ========================================================
    # Оценка RW
    # ========================================================

    @staticmethod
    def _evaluate_rw_segments(
        segments: list[PreparedSegment],
        sigma_rw: float,
        sigma_meas: float,
        window: int,
        stride: int,
        collect_metrics: bool,
    ) -> dict[str, NDArray[np.float64] | float | int]:
        """Единый высокопроизводительный расчёт RW."""

        ll_list: list[NDArray[np.float64]] = []
        m2_list: list[NDArray[np.float64]] = []
        fd_list: list[NDArray[np.float64]] = []

        total_log_likelihood = 0.0
        observation_count = 0

        for segment in segments:
            starts = KalmanParameterEstimator._window_start_indices(
                segment=segment,
                window=window,
                stride=stride,
            )

            if len(starts) == 0:
                continue

            for begin in range(0, len(starts), BATCH_SIZE):
                end = min(begin + BATCH_SIZE, len(starts))
                current_starts = starts[begin:end]

                x_windows, y_windows, dt_windows = (
                    KalmanParameterEstimator._build_local_windows(
                        segment=segment,
                        start_indices=current_starts,
                        window=window,
                    )
                )

                log_likelihood, mahalanobis_sq, filter_distance = (
                    KalmanParameterEstimator._rw_batch_statistics(
                        x_windows=x_windows,
                        y_windows=y_windows,
                        dt_windows=dt_windows,
                        sigma_rw=sigma_rw,
                        sigma_meas=sigma_meas,
                    )
                )

                valid_ll = log_likelihood[np.isfinite(log_likelihood)]

                if len(valid_ll):
                    total_log_likelihood += float(np.sum(valid_ll))
                    observation_count += len(valid_ll)

                if collect_metrics:
                    valid = (
                        np.isfinite(log_likelihood)
                        & np.isfinite(mahalanobis_sq)
                        & np.isfinite(filter_distance)
                    )

                    if np.any(valid):
                        ll_list.append(log_likelihood[valid])
                        m2_list.append(mahalanobis_sq[valid])
                        fd_list.append(filter_distance[valid])

        if collect_metrics:
            ll = np.concatenate(ll_list) if ll_list else np.empty(0, dtype=np.float64)
            m2 = np.concatenate(m2_list) if m2_list else np.empty(0, dtype=np.float64)
            fd = np.concatenate(fd_list) if fd_list else np.empty(0, dtype=np.float64)

            return {
                "log_likelihood": ll,
                "mahalanobis": m2,
                "filter_distance": fd,
                "total_log_likelihood": float(np.sum(ll)),
                "observation_count": int(len(ll)),
            }

        return {
            "total_log_likelihood": total_log_likelihood,
            "observation_count": observation_count,
        }

    # ========================================================
    # MLE CV
    # ========================================================

    @staticmethod
    def fit_cv_mle(
        segments: list[TrackSegment],
        sigma_meas_initial: float,
        sigma_acc_initial: float,
        window: int = WINDOW,
    ) -> dict[str, float | bool | str]:
        """Оценивает sigma_meas и sigma_acc методом MLE."""

        prepared = KalmanParameterEstimator.prepare_segments(segments)

        if not prepared:
            return {
                "success": False,
                "sigma_meas": np.nan,
                "sigma_acc": np.nan,
                "negative_log_likelihood": np.nan,
                "message": "нет подходящих сегментов",
            }

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

        initial = np.log([
            sigma_meas_initial,
            sigma_acc_initial,
        ])

        bounds = [
            (np.log(SIGMA_MEAS_MIN), np.log(SIGMA_MEAS_MAX)),
            (np.log(SIGMA_ACC_MIN), np.log(SIGMA_ACC_MAX)),
        ]

        starts = [
            initial,
            initial + np.log([2.0, 2.0]),
            initial + np.log([0.5, 0.5]),
        ]

        total_windows = sum(
            len(
                KalmanParameterEstimator._window_start_indices(
                    segment=segment,
                    window=window,
                    stride=MLE_STRIDE,
                )
            )
            for segment in prepared
        )

        logging.info(
            "CV MLE: подготовлено сегментов=%d, MLE windows=%d, stride=%d",
            len(prepared),
            total_windows,
            MLE_STRIDE,
        )

        best_result = None

        for start_id, start in enumerate(starts, start=1):
            start = np.clip(
                start,
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            )

            objective_calls = 0

            def objective(params: NDArray[np.float64]) -> float:
                nonlocal objective_calls

                objective_calls += 1

                sigma_meas = float(np.exp(params[0]))
                sigma_acc = float(np.exp(params[1]))

                if (
                    not np.isfinite(sigma_meas)
                    or not np.isfinite(sigma_acc)
                    or sigma_meas <= 0.0
                    or sigma_acc <= 0.0
                ):
                    return np.inf

                result = KalmanParameterEstimator._evaluate_cv_segments(
                    segments=prepared,
                    sigma_acc=sigma_acc,
                    sigma_meas=sigma_meas,
                    window=window,
                    stride=MLE_STRIDE,
                    collect_metrics=False,
                )

                total_ll = float(result["total_log_likelihood"])
                count = int(result["observation_count"])

                if count == 0:
                    return np.inf

                nll = -total_ll

                if (
                    objective_calls == 1
                    or objective_calls % OPTIMIZER_LOG_EVERY == 0
                ):
                    logging.info(
                        "CV MLE start=%d evaluation=%d: "
                        "sigma_meas=%.8g, sigma_acc=%.8g, NLL=%.8g",
                        start_id,
                        objective_calls,
                        sigma_meas,
                        sigma_acc,
                        nll,
                    )

                return nll

            logging.info(
                "CV MLE: старт оптимизации %d/%d",
                start_id,
                len(starts),
            )

            result = minimize(
                objective,
                x0=start,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": OPTIMIZER_MAXITER,
                    "ftol": OPTIMIZER_FTOL,
                },
            )

            logging.info(
                "CV MLE: старт %d завершён: success=%s, iterations=%s, evaluations=%s, message=%s",
                start_id,
                result.success,
                result.nit,
                result.nfev,
                result.message,
            )

            if best_result is None or result.fun < best_result.fun:
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
    def fit_rw_mle(
        segments: list[TrackSegment],
        sigma_meas_initial: float,
        sigma_rw_initial: float,
        window: int = WINDOW,
    ) -> dict[str, float | bool | str]:
        """Оценивает sigma_meas и sigma_rw методом MLE."""

        prepared = KalmanParameterEstimator.prepare_segments(segments)

        if not prepared:
            return {
                "success": False,
                "sigma_meas": np.nan,
                "sigma_rw": np.nan,
                "negative_log_likelihood": np.nan,
                "message": "нет подходящих сегментов",
            }

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

        initial = np.log([
            sigma_meas_initial,
            sigma_rw_initial,
        ])

        bounds = [
            (np.log(SIGMA_MEAS_MIN), np.log(SIGMA_MEAS_MAX)),
            (np.log(SIGMA_RW_MIN), np.log(SIGMA_RW_MAX)),
        ]

        starts = [
            initial,
            initial + np.log([2.0, 2.0]),
            initial + np.log([0.5, 0.5]),
        ]

        total_windows = sum(
            len(
                KalmanParameterEstimator._window_start_indices(
                    segment=segment,
                    window=window,
                    stride=MLE_STRIDE,
                )
            )
            for segment in prepared
        )

        logging.info(
            "RW MLE: подготовлено сегментов=%d, MLE windows=%d, stride=%d",
            len(prepared),
            total_windows,
            MLE_STRIDE,
        )

        best_result = None

        for start_id, start in enumerate(starts, start=1):
            start = np.clip(
                start,
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            )

            objective_calls = 0

            def objective(params: NDArray[np.float64]) -> float:
                nonlocal objective_calls

                objective_calls += 1

                sigma_meas = float(np.exp(params[0]))
                sigma_rw = float(np.exp(params[1]))

                if (
                    not np.isfinite(sigma_meas)
                    or not np.isfinite(sigma_rw)
                    or sigma_meas <= 0.0
                    or sigma_rw <= 0.0
                ):
                    return np.inf

                result = KalmanParameterEstimator._evaluate_rw_segments(
                    segments=prepared,
                    sigma_rw=sigma_rw,
                    sigma_meas=sigma_meas,
                    window=window,
                    stride=MLE_STRIDE,
                    collect_metrics=False,
                )

                total_ll = float(result["total_log_likelihood"])
                count = int(result["observation_count"])

                if count == 0:
                    return np.inf

                nll = -total_ll

                if (
                    objective_calls == 1
                    or objective_calls % OPTIMIZER_LOG_EVERY == 0
                ):
                    logging.info(
                        "RW MLE start=%d evaluation=%d: "
                        "sigma_meas=%.8g, sigma_rw=%.8g, NLL=%.8g",
                        start_id,
                        objective_calls,
                        sigma_meas,
                        sigma_rw,
                        nll,
                    )

                return nll

            logging.info(
                "RW MLE: старт оптимизации %d/%d",
                start_id,
                len(starts),
            )

            result = minimize(
                objective,
                x0=start,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": OPTIMIZER_MAXITER,
                    "ftol": OPTIMIZER_FTOL,
                },
            )

            logging.info(
                "RW MLE: старт %d завершён: success=%s, iterations=%s, evaluations=%s, message=%s",
                start_id,
                result.success,
                result.nit,
                result.nfev,
                result.message,
            )

            if best_result is None or result.fun < best_result.fun:
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
    # Финальный сбор CV
    # ========================================================

    @staticmethod
    def collect_cv_likelihood(
        segments: Iterable[TrackSegment],
        sigma_acc: float,
        sigma_meas: float,
        window: int = WINDOW,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Финальная CV-статистика.

        Здесь stride=1, то есть метрика считается для каждой точки.
        """

        prepared = KalmanParameterEstimator.prepare_segments(segments)

        result = KalmanParameterEstimator._evaluate_cv_segments(
            segments=prepared,
            sigma_acc=sigma_acc,
            sigma_meas=sigma_meas,
            window=window,
            stride=1,
            collect_metrics=True,
        )

        return {
            "log_likelihood": result["log_likelihood"],
            "mahalanobis": result["mahalanobis"],
            "filter_distance": result["filter_distance"],
        }

    # ========================================================
    # Финальный сбор RW
    # ========================================================

    @staticmethod
    def collect_rw_likelihood(
        segments: Iterable[TrackSegment],
        sigma_rw: float,
        sigma_meas: float,
        window: int = WINDOW,
    ) -> dict[str, NDArray[np.float64]]:
        """Финальная RW-статистика, stride=1."""

        prepared = KalmanParameterEstimator.prepare_segments(segments)

        result = KalmanParameterEstimator._evaluate_rw_segments(
            segments=prepared,
            sigma_rw=sigma_rw,
            sigma_meas=sigma_meas,
            window=window,
            stride=1,
            collect_metrics=True,
        )

        return {
            "log_likelihood": result["log_likelihood"],
            "mahalanobis": result["mahalanobis"],
            "filter_distance": result["filter_distance"],
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
        Проверяет калибровку по Mahalanobis².

        Для двумерной нормально распределённой инновации:

            M² ~ chi2(df=2)

        Ожидаемые значения:

            E[M²] = 2
            P95[M²] = chi2.ppf(0.95, 2)
        """

        values = np.asarray(mahalanobis, dtype=np.float64)
        values = values[np.isfinite(values)]

        expected_mean = 2.0
        expected_p95 = float(chi2.ppf(0.95, df=2))

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

        mean_m2 = float(np.mean(values))
        p95_m2 = float(np.percentile(values, 95))

        mean_ratio = mean_m2 / expected_mean
        p95_ratio = p95_m2 / expected_p95

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
# Загрузка файла
# ============================================================

def load_and_prepare_file(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, list[TrackSegment]]]:
    """Загружает, фильтрует и разбивает файл на сегменты."""

    logging.info("Загрузка файла: %s", path.name)

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

    segments = KalmanParameterEstimator.extract_segments(
        df=df,
        file_name=path.name,
        min_length=MIN_SEGMENT_LENGTH,
    )

    segment_sets = {
        "stand": [
            segment for segment in segments
            if segment.segment_type == "stand"
        ],
        "move": [
            segment for segment in segments
            if segment.segment_type == "move"
        ],
        "stand_move": [
            segment for segment in segments
            if segment.segment_type == "stand_move"
        ],
    }

    for segment_type, current_segments in segment_sets.items():
        total_points = sum(
            segment.size for segment in current_segments
        )

        logging.info(
            "%s / %s: segments=%d, points=%d",
            path.name,
            segment_type,
            len(current_segments),
            total_points,
        )

    return df, segment_sets


# ============================================================
# Строка результата
# ============================================================

def make_result_row(
    *,
    train_file: str,
    validation_file: str,
    segment_type: str,
    method: str,
    n_segments: int,
    n_points: int,
) -> dict:
    """Создаёт унифицированную строку CSV."""

    return {
        "train_file": train_file,
        "validation_file": validation_file,
        "segment_type": segment_type,
        "method": method,
        "n_segments": n_segments,
        "n_points": n_points,
        "sigma_meas_x": np.nan,
        "sigma_meas_y": np.nan,
        "sigma_meas": np.nan,
        "sigma_acc": np.nan,
        "sigma_rw": np.nan,
        "negative_log_likelihood": np.nan,
        "optimizer_success": np.nan,
        "optimizer_message": "",
        "validation_n_segments": np.nan,
        "validation_n_points": np.nan,
        "validation_mean_mahalanobis": np.nan,
        "validation_p95_mahalanobis": np.nan,
        "validation_chi2_mean_expected": np.nan,
        "validation_chi2_p95_expected": np.nan,
        "validation_mean_ratio": np.nan,
        "validation_p95_ratio": np.nan,
        "validation_quality": "",
        "validation_message": "",
    }


# ============================================================
# Обучение одного файла
# ============================================================

def train_one_file(
    train_path: Path,
    train_segment_sets: dict[str, list[TrackSegment]],
    validation_segment_sets: dict[str, list[TrackSegment]],
    validation_file_name: str,
    output_rows: list[dict],
) -> None:
    """
    Обучает CV и RW на одном train-файле.

    1.csv и 2.csv вызываются независимо.
    """

    logging.info("==================================================")
    logging.info("TRAIN experiment: %s", train_path.name)

    # --------------------------------------------------------
    # sigma_meas
    # --------------------------------------------------------

    stand_segments = train_segment_sets["stand"]

    if stand_segments:
        position_sigma = KalmanParameterEstimator.estimate_sigma_meas_from_position(
            stand_segments
        )

        neighbor_sigma = KalmanParameterEstimator.estimate_sigma_meas_from_neighbor_difference(
            stand_segments
        )

        logging.info(
            "%s / stand / position_std: sigma_x=%.8g, sigma_y=%.8g, sigma_meas=%.8g",
            train_path.name,
            position_sigma["sigma_meas_x"],
            position_sigma["sigma_meas_y"],
            position_sigma["sigma_meas"],
        )

        logging.info(
            "%s / stand / neighbor_diff: sigma_x=%.8g, sigma_y=%.8g, sigma_meas=%.8g",
            train_path.name,
            neighbor_sigma["sigma_meas_x"],
            neighbor_sigma["sigma_meas_y"],
            neighbor_sigma["sigma_meas"],
        )

        n_stand_points = sum(
            segment.size for segment in stand_segments
        )

        row = make_result_row(
            train_file=train_path.name,
            validation_file=validation_file_name,
            segment_type="stand",
            method="stand_position_std",
            n_segments=len(stand_segments),
            n_points=n_stand_points,
        )

        row.update({
            "sigma_meas_x": position_sigma["sigma_meas_x"],
            "sigma_meas_y": position_sigma["sigma_meas_y"],
            "sigma_meas": position_sigma["sigma_meas"],
        })

        output_rows.append(row)

        row = make_result_row(
            train_file=train_path.name,
            validation_file=validation_file_name,
            segment_type="stand",
            method="stand_neighbor_diff",
            n_segments=len(stand_segments),
            n_points=neighbor_sigma["count"],
        )

        row.update({
            "sigma_meas_x": neighbor_sigma["sigma_meas_x"],
            "sigma_meas_y": neighbor_sigma["sigma_meas_y"],
            "sigma_meas": neighbor_sigma["sigma_meas"],
        })

        output_rows.append(row)

        sigma_meas_initial = float(
            neighbor_sigma["sigma_meas"]
        )

        if not np.isfinite(sigma_meas_initial):
            sigma_meas_initial = float(
                position_sigma["sigma_meas"]
            )

        if not np.isfinite(sigma_meas_initial):
            sigma_meas_initial = 10.0

    else:
        logging.warning(
            "%s: stand-участков длиной >= %d не найдено",
            train_path.name,
            MIN_SEGMENT_LENGTH,
        )

        sigma_meas_initial = 10.0

    sigma_meas_initial = float(
        np.clip(
            sigma_meas_initial,
            SIGMA_MEAS_MIN,
            SIGMA_MEAS_MAX,
        )
    )

    logging.info(
        "%s: начальное sigma_meas для MLE = %.8g",
        train_path.name,
        sigma_meas_initial,
    )

    # --------------------------------------------------------
    # MLE для каждого типа участка
    # --------------------------------------------------------

    for segment_type, current_segments in train_segment_sets.items():
        if not current_segments:
            logging.warning(
                "%s / %s: сегменты отсутствуют, MLE пропущен",
                train_path.name,
                segment_type,
            )
            continue

        total_points = sum(
            segment.size for segment in current_segments
        )

        logging.info(
            "%s / %s: начало MLE, segments=%d, points=%d",
            train_path.name,
            segment_type,
            len(current_segments),
            total_points,
        )

        # ====================================================
        # CV
        # ====================================================

        cv_result = KalmanParameterEstimator.fit_cv_mle(
            segments=current_segments,
            sigma_meas_initial=sigma_meas_initial,
            sigma_acc_initial=DEFAULT_SIGMA_ACC_INITIAL,
            window=WINDOW,
        )

        sigma_meas_cv = float(cv_result["sigma_meas"])
        sigma_acc_cv = float(cv_result["sigma_acc"])

        logging.info(
            "%s / %s / CV MLE: success=%s, sigma_meas=%.8g, sigma_acc=%.8g, NLL=%.8g",
            train_path.name,
            segment_type,
            cv_result["success"],
            sigma_meas_cv,
            sigma_acc_cv,
            cv_result["negative_log_likelihood"],
        )

        row = make_result_row(
            train_file=train_path.name,
            validation_file=validation_file_name,
            segment_type=segment_type,
            method="MLE_CV",
            n_segments=len(current_segments),
            n_points=total_points,
        )

        row.update({
            "sigma_meas": sigma_meas_cv,
            "sigma_acc": sigma_acc_cv,
            "negative_log_likelihood": cv_result["negative_log_likelihood"],
            "optimizer_success": cv_result["success"],
            "optimizer_message": cv_result["message"],
        })

        # ----------------------------------------------------
        # Validation CV на 3.csv
        # ----------------------------------------------------

        validation_segments = validation_segment_sets[segment_type]

        if validation_segments:
            logging.info(
                "%s / %s / CV: расчёт validation на %s",
                train_path.name,
                segment_type,
                validation_file_name,
            )

            validation_metrics = KalmanParameterEstimator.collect_cv_likelihood(
                segments=validation_segments,
                sigma_acc=sigma_acc_cv,
                sigma_meas=sigma_meas_cv,
                window=WINDOW,
            )

            validation = KalmanParameterEstimator.validate_model(
                mahalanobis=validation_metrics["mahalanobis"],
                sigma_meas=sigma_meas_cv,
                process_sigma=sigma_acc_cv,
                optimizer_success=bool(cv_result["success"]),
                process_name="CV",
            )

            row.update({
                "validation_n_segments": len(validation_segments),
                "validation_n_points": sum(
                    segment.size for segment in validation_segments
                ),
            })

            row.update(validation)

            logging.info(
                "%s / %s / CV validation: %s",
                train_path.name,
                segment_type,
                validation["validation_message"],
            )

        else:
            logging.warning(
                "%s / %s: на %s нет подходящих validation-сегментов",
                train_path.name,
                segment_type,
                validation_file_name,
            )

        output_rows.append(row)

        # ====================================================
        # RW
        # ====================================================

        sigma_rw_initial = max(
            sigma_meas_initial / np.sqrt(10.0),
            SIGMA_RW_MIN,
        )

        rw_result = KalmanParameterEstimator.fit_rw_mle(
            segments=current_segments,
            sigma_meas_initial=sigma_meas_initial,
            sigma_rw_initial=sigma_rw_initial,
            window=WINDOW,
        )

        sigma_meas_rw = float(rw_result["sigma_meas"])
        sigma_rw = float(rw_result["sigma_rw"])

        logging.info(
            "%s / %s / RW MLE: success=%s, sigma_meas=%.8g, sigma_rw=%.8g, NLL=%.8g",
            train_path.name,
            segment_type,
            rw_result["success"],
            sigma_meas_rw,
            sigma_rw,
            rw_result["negative_log_likelihood"],
        )

        row = make_result_row(
            train_file=train_path.name,
            validation_file=validation_file_name,
            segment_type=segment_type,
            method="MLE_RW",
            n_segments=len(current_segments),
            n_points=total_points,
        )

        row.update({
            "sigma_meas": sigma_meas_rw,
            "sigma_rw": sigma_rw,
            "negative_log_likelihood": rw_result["negative_log_likelihood"],
            "optimizer_success": rw_result["success"],
            "optimizer_message": rw_result["message"],
        })

        # ----------------------------------------------------
        # Validation RW на 3.csv
        # ----------------------------------------------------

        validation_segments = validation_segment_sets[segment_type]

        if validation_segments:
            logging.info(
                "%s / %s / RW: расчёт validation на %s",
                train_path.name,
                segment_type,
                validation_file_name,
            )

            validation_metrics = KalmanParameterEstimator.collect_rw_likelihood(
                segments=validation_segments,
                sigma_rw=sigma_rw,
                sigma_meas=sigma_meas_rw,
                window=WINDOW,
            )

            validation = KalmanParameterEstimator.validate_model(
                mahalanobis=validation_metrics["mahalanobis"],
                sigma_meas=sigma_meas_rw,
                process_sigma=sigma_rw,
                optimizer_success=bool(rw_result["success"]),
                process_name="RW",
            )

            row.update({
                "validation_n_segments": len(validation_segments),
                "validation_n_points": sum(
                    segment.size for segment in validation_segments
                ),
            })

            row.update(validation)

            logging.info(
                "%s / %s / RW validation: %s",
                train_path.name,
                segment_type,
                validation["validation_message"],
            )

        else:
            logging.warning(
                "%s / %s: на %s нет подходящих validation-сегментов",
                train_path.name,
                segment_type,
                validation_file_name,
            )

        output_rows.append(row)

        logging.info(
            "%s / %s: MLE завершён",
            train_path.name,
            segment_type,
        )

    logging.info(
        "TRAIN experiment %s завершён",
        train_path.name,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    path_root = Path(__file__).parent

    train_paths = [
        path_root / "data" / "1.csv",
        path_root / "data" / "2.csv",
        path_root / "data" / "3.csv",
    ]

    validation_path = path_root / "data" / "3.csv"

    output_path = (
        path_root
        / "statistics"
        / OUTPUT_FILENAME
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.info("==================================================")
    logging.info("Начало калибровки фильтров Калмана")

    logging.info(
        "Training files: %s",
        ", ".join(path.name for path in train_paths),
    )

    logging.info(
        "Validation file: %s",
        validation_path.name,
    )

    logging.info(
        "MIN_SEGMENT_LENGTH=%d",
        MIN_SEGMENT_LENGTH,
    )

    logging.info(
        "WINDOW=%d",
        WINDOW,
    )

    logging.info(
        "MLE_STRIDE=%d",
        MLE_STRIDE,
    )

    logging.info(
        "BATCH_SIZE=%d",
        BATCH_SIZE,
    )

    all_results: list[dict] = []

    # ========================================================
    # Validation загружается один раз
    # ========================================================

    _, validation_segment_sets = load_and_prepare_file(
        validation_path
    )

    # ========================================================
    # Каждый train-файл обрабатывается независимо
    # ========================================================

    for train_path in train_paths:
        _, train_segment_sets = load_and_prepare_file(
            train_path
        )

        train_one_file(
            train_path=train_path,
            train_segment_sets=train_segment_sets,
            validation_segment_sets=validation_segment_sets,
            validation_file_name=validation_path.name,
            output_rows=all_results,
        )

    # ========================================================
    # Сохранение
    # ========================================================

    result_df = pd.DataFrame(all_results)

    result_df.to_csv(
        output_path,
        index=False,
    )

    logging.info(
        "Результаты записаны в: %s",
        output_path,
    )

    logging.info(
        "Количество строк результата: %d",
        len(result_df),
    )

    # ========================================================
    # Печать результата
    # ========================================================

    if result_df.empty:
        logging.warning("Итоговый CSV пуст")

    else:
        columns_to_print = [
            "train_file",
            "validation_file",
            "segment_type",
            "method",
            "n_segments",
            "n_points",
            "sigma_meas",
            "sigma_acc",
            "sigma_rw",
            "negative_log_likelihood",
            "optimizer_success",
            "validation_n_segments",
            "validation_n_points",
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

        print("\nИтоговая таблица:\n")
        print(
            result_df[existing_columns].to_string(index=False)
        )

    logging.info(
        "Оценка параметров завершена"
    )