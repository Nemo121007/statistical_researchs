"""
Калибровка параметров фильтров Калмана CV и RW.

Обучающие эксперименты:
    1.csv
    2.csv

Валидационный эксперимент:
    3.csv

ВАЖНО:

1. 1.csv и 2.csv являются независимыми обучающими экспериментами.

2. 3.csv НЕ используется при подборе параметров.

3. sigma_meas оценивается отдельным этапом:
       - только из stand;
       - stand разбивается на локальные неперекрывающиеся фрагменты;
       - отбираются локально-стационарные фрагменты;
       - sigma_meas оценивается по приращениям координат.

4. После оценки sigma_meas она ФИКСИРУЕТСЯ.

5. MLE CV оптимизирует только:
       sigma_acc

6. MLE RW оптимизирует только:
       sigma_rw

7. Для MLE используются неперекрывающиеся окна:

       P[0] ... P[WINDOW]
       P[WINDOW + 1] ... P[2*WINDOW + 1]
       ...

8. Для итоговой статистики используются все локальные окна
   с шагом 1:

       P[i-WINDOW] ... P[i]

9. Для validation рассчитываются:

       mean(M²)
       median(M²)
       P95(M²)
       P99(M²)
       доля M² > 5.991
       доля M² > 9.21
       доля M² > 13.82

10. Для двумерной Gaussian-инновации теоретически:

       M² ~ chi2(df=2)

   поэтому:

       E[M²]       = 2
       P95[M²]     = 5.991
       P99[M²]     = 9.210
       P99.9[M²]   = 13.816

11. DataProcessor не изменяется.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from app.working.data_processor import DataProcessor


# ============================================================
# Конфигурация
# ============================================================

MIN_SEGMENT_LENGTH = 100

# Локальное окно Kalman:
#
# P[i-WINDOW] ... P[i]
#
# При WINDOW=10 окно содержит 11 точек.
WINDOW = 10

# ------------------------------------------------------------
# Настройки локально-стационарных stand-фрагментов
# ------------------------------------------------------------

# При типичном шаге 10 секунд:
#
# 30 точек ~= 5 минут.
#
# Фрагменты НЕ перекрываются.
STATIONARY_FRAGMENT_LENGTH = 30

# Минимальный размер фрагмента.
STATIONARY_MIN_POINTS = 20

# Насколько сильно средний шаг может отличаться от нуля.
#
# Проверяется отдельно для X и Y:
#
#     |mean(diff)| <= Z * sigma_diff / sqrt(N)
#
# При Z=3 допускается 3-sigma отклонение.
STATIONARITY_Z = 3.0

# Локальная оценка sigma фрагмента должна находиться
# в разумном диапазоне относительно медианы по всем фрагментам.
#
# Это защищает от поврежденных участков GPS.
STATIONARY_SIGMA_MIN_FACTOR = 0.25
STATIONARY_SIGMA_MAX_FACTOR = 4.0

# Проверка временного шага внутри фрагмента.
#
# Например, при обычном dt=10 секунд значение 2.0
# допускает максимум около 20 секунд.
STATIONARY_MAX_DT_RATIO = 2.0

# Минимальное число отобранных stationarity-фрагментов.
STATIONARY_MIN_ACCEPTED_FRAGMENTS = 3

# ------------------------------------------------------------
# MLE
# ------------------------------------------------------------

MLE_STRIDE = WINDOW + 1

BATCH_SIZE = 4096

OPTIMIZER_MAXITER = 100
OPTIMIZER_XTOL = 1e-4
OPTIMIZER_LOG_EVERY = 10

SIGMA_MEAS_MIN = 1e-3
SIGMA_MEAS_MAX = 1000.0

SIGMA_ACC_MIN = 1e-7
SIGMA_ACC_MAX = 10.0

SIGMA_RW_MIN = 1e-7
SIGMA_RW_MAX = 100.0

DEFAULT_SIGMA_ACC_INITIAL = 0.04

DEFAULT_SIGMA_RW_INITIAL = 1.0

# ------------------------------------------------------------
# Mahalanobis
# ------------------------------------------------------------

MAHALANOBIS_P95_THEORETICAL = float(
    chi2.ppf(0.95, df=2)
)

MAHALANOBIS_P99_THEORETICAL = float(
    chi2.ppf(0.99, df=2)
)

MAHALANOBIS_P999_THEORETICAL = float(
    chi2.ppf(0.999, df=2)
)

MAHALANOBIS_THRESHOLD_P95 = MAHALANOBIS_P95_THEORETICAL
MAHALANOBIS_THRESHOLD_P99 = MAHALANOBIS_P99_THEORETICAL
MAHALANOBIS_THRESHOLD_P999 = MAHALANOBIS_P999_THEORETICAL

OUTPUT_FILENAME = "kalman_parameter_estimates.csv"


# ============================================================
# Segment
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
# Подготовленный segment
# ============================================================

@dataclass
class PreparedSegment:
    """NumPy-представление TrackSegment."""

    file_name: str
    segment_type: str
    segment_id: int

    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64]
    dt: NDArray[np.float64]

    @property
    def size(self) -> int:
        return len(self.lon)

    @classmethod
    def from_segment(
        cls,
        segment: TrackSegment,
    ) -> PreparedSegment:
        lon = segment.df["lon"].to_numpy(
            dtype=np.float64
        )

        lat = segment.df["lat"].to_numpy(
            dtype=np.float64
        )

        time = (
            pd.to_datetime(
                segment.df["time"],
                errors="coerce",
            )
            .to_numpy(dtype="datetime64[ns]")
        )

        if len(lon) != len(lat) or len(lon) != len(time):
            raise ValueError(
                f"{segment.file_name} / "
                f"{segment.segment_type} / "
                f"segment={segment.segment_id}: "
                "несовпадающая длина lon/lat/time"
            )

        if np.any(np.isnat(time)):
            raise ValueError(
                f"{segment.file_name} / "
                f"{segment.segment_type} / "
                f"segment={segment.segment_id}: "
                "обнаружены некорректные значения time"
            )

        if not np.all(np.isfinite(lon)):
            raise ValueError(
                f"{segment.file_name} / "
                f"{segment.segment_type} / "
                f"segment={segment.segment_id}: "
                "обнаружены NaN/Inf в lon"
            )

        if not np.all(np.isfinite(lat)):
            raise ValueError(
                f"{segment.file_name} / "
                f"{segment.segment_type} / "
                f"segment={segment.segment_id}: "
                "обнаружены NaN/Inf в lat"
            )

        if len(time) >= 2:
            dt = (
                (time[1:] - time[:-1])
                / np.timedelta64(1, "s")
            ).astype(np.float64)

            dt = np.maximum(
                dt,
                0.0,
            )
        else:
            dt = np.empty(
                0,
                dtype=np.float64,
            )

        if not np.all(np.isfinite(dt)):
            raise ValueError(
                f"{segment.file_name} / "
                f"{segment.segment_type} / "
                f"segment={segment.segment_id}: "
                "обнаружены некорректные dt"
            )

        return cls(
            file_name=segment.file_name,
            segment_type=segment.segment_type,
            segment_id=segment.segment_id,
            lon=lon,
            lat=lat,
            time=time,
            dt=dt,
        )


# ============================================================
# Локально-стационарный фрагмент
# ============================================================

@dataclass
class StationaryFragment:
    """Локальный фрагмент stand для оценки sigma_meas."""

    file_name: str
    segment_id: int
    fragment_id: int

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dt: NDArray[np.float64]

    sigma_x: float
    sigma_y: float

    mean_dx: float
    mean_dy: float

    var_dx: float
    var_dy: float

    accepted: bool = False
    reject_reason: str = ""


# ============================================================
# Основной класс
# ============================================================

class KalmanParameterEstimator:
    """Калибровка параметров Kalman CV и RW."""

    # ========================================================
    # Извлечение сегментов
    # ========================================================

    @staticmethod
    def extract_segments(
        df: pd.DataFrame,
        file_name: str,
        min_length: int = MIN_SEGMENT_LENGTH,
    ) -> list[TrackSegment]:
        """Выделяет stand, move и stand_move."""

        if "status" not in df.columns:
            raise ValueError(
                "В DataFrame отсутствует столбец 'status'"
            )

        segments: list[TrackSegment] = []

        stand_mask = (
            df["status"] == "stand"
        ).to_numpy()

        segments.extend(
            KalmanParameterEstimator._extract_mask_segments(
                df=df,
                mask=stand_mask,
                file_name=file_name,
                segment_type="stand",
                min_length=min_length,
            )
        )

        move_mask = (
            df["status"] == "move"
        ).to_numpy()

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
            df["status"]
            .isin(["stand", "move"])
            .to_numpy()
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
        """Выделяет непрерывные True-сегменты."""

        if len(df) != len(mask):
            raise ValueError(
                "Длина mask не совпадает с DataFrame"
            )

        if len(df) == 0:
            return []

        starts = (
            mask
            & ~np.r_[False, mask[:-1]]
        )

        ends = (
            mask
            & ~np.r_[mask[1:], False]
        )

        start_indices = np.flatnonzero(starts)
        end_indices = np.flatnonzero(ends)

        segments: list[TrackSegment] = []

        for segment_id, (start, end) in enumerate(
            zip(start_indices, end_indices)
        ):
            segment_df = (
                df.iloc[start:end + 1]
                .copy()
            )

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
    # Подготовка
    # ========================================================

    @staticmethod
    def prepare_segments(
        segments: Iterable[TrackSegment],
    ) -> list[PreparedSegment]:
        """Преобразует DataFrame в NumPy-представление."""

        result: list[PreparedSegment] = []

        for segment in segments:
            if segment.size <= WINDOW:
                continue

            result.append(
                PreparedSegment.from_segment(
                    segment
                )
            )

        return result

    # ========================================================
    # Локальная система координат
    # ========================================================

    @staticmethod
    def to_local_xy(
        df: pd.DataFrame,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Локальная система координат.

        Каждый переданный DataFrame получает собственное начало
        координат через DataProcessor.
        """

        lon = df["lon"].to_numpy(
            dtype=np.float64
        )

        lat = df["lat"].to_numpy(
            dtype=np.float64
        )

        return DataProcessor.convert_to_local_cartesian(
            lon,
            lat,
        )

    @staticmethod
    def _lon_lat_to_local_xy(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Локальные X/Y для NumPy-массива."""

        if len(lon) == 0:
            return (
                np.empty(
                    0,
                    dtype=np.float64,
                ),
                np.empty(
                    0,
                    dtype=np.float64,
                ),
            )

        lon0 = lon[0]
        lat0 = lat[0]

        kx = (
            DataProcessor.LEN_LAT
            * np.cos(np.radians(lat0))
        )

        x = (
            lon - lon0
        ) * kx

        y = (
            lat - lat0
        ) * DataProcessor.LEN_LAT

        return (
            x.astype(np.float64),
            y.astype(np.float64),
        )

    # ========================================================
    # Построение локальных stand-фрагментов
    # ========================================================

    @staticmethod
    def build_stationary_fragments(
        segments: Iterable[TrackSegment],
        fragment_length: int = STATIONARY_FRAGMENT_LENGTH,
    ) -> list[StationaryFragment]:
        """
        Разбивает stand-сегменты на неперекрывающиеся локальные фрагменты.

        Каждый фрагмент имеет собственную локальную систему координат.
        """

        fragments: list[StationaryFragment] = []

        for segment in segments:
            if segment.segment_type != "stand":
                continue

            lon = segment.df["lon"].to_numpy(
                dtype=np.float64
            )

            lat = segment.df["lat"].to_numpy(
                dtype=np.float64
            )

            time = (
                pd.to_datetime(
                    segment.df["time"],
                    errors="coerce",
                )
                .to_numpy(dtype="datetime64[ns]")
            )

            n = len(lon)

            fragment_id = 0

            for start in range(
                0,
                n,
                fragment_length,
            ):
                end = min(
                    start + fragment_length,
                    n,
                )

                if end - start < STATIONARY_MIN_POINTS:
                    continue

                fragment_lon = lon[start:end]
                fragment_lat = lat[start:end]
                fragment_time = time[start:end]

                if np.any(np.isnat(fragment_time)):
                    continue

                x, y = (
                    KalmanParameterEstimator
                    ._lon_lat_to_local_xy(
                        fragment_lon,
                        fragment_lat,
                    )
                )

                dt = (
                    (
                        fragment_time[1:]
                        - fragment_time[:-1]
                    )
                    / np.timedelta64(1, "s")
                ).astype(np.float64)

                valid_dt = (
                    np.isfinite(dt)
                    & (dt > 0.0)
                )

                if np.count_nonzero(valid_dt) < 3:
                    continue

                dx = np.diff(x)
                dy = np.diff(y)

                valid = (
                    valid_dt
                    & np.isfinite(dx)
                    & np.isfinite(dy)
                )

                dx = dx[valid]
                dy = dy[valid]
                dt_valid = dt[valid]

                if len(dx) < 3:
                    continue

                var_dx = float(
                    np.var(
                        dx,
                        ddof=1,
                    )
                )

                var_dy = float(
                    np.var(
                        dy,
                        ddof=1,
                    )
                )

                sigma_x = np.sqrt(
                    max(var_dx, 0.0) / 2.0
                )

                sigma_y = np.sqrt(
                    max(var_dy, 0.0) / 2.0
                )

                mean_dx = float(
                    np.mean(dx)
                )

                mean_dy = float(
                    np.mean(dy)
                )

                fragments.append(
                    StationaryFragment(
                        file_name=segment.file_name,
                        segment_id=segment.segment_id,
                        fragment_id=fragment_id,
                        x=x,
                        y=y,
                        dt=dt_valid,
                        sigma_x=float(sigma_x),
                        sigma_y=float(sigma_y),
                        mean_dx=mean_dx,
                        mean_dy=mean_dy,
                        var_dx=var_dx,
                        var_dy=var_dy,
                    )
                )

                fragment_id += 1

        return fragments

    # ========================================================
    # Отбор локально-стационарных фрагментов
    # ========================================================

    @staticmethod
    def select_stationary_fragments(
        fragments: list[StationaryFragment],
    ) -> tuple[
        list[StationaryFragment],
        dict[str, float],
    ]:
        """
        Отбирает локально-стационарные фрагменты.

        Критерии:

        1. Локальное sigma не должно быть аномально большим
           или малым относительно медианы.

        2. Среднее приращение X и Y не должно статистически
           отличаться от нуля.

        3. Временной шаг не должен содержать слишком большие
           разрывы.
        """

        valid_fragments = [
            fragment
            for fragment in fragments
            if np.isfinite(fragment.sigma_x)
            and np.isfinite(fragment.sigma_y)
            and fragment.sigma_x >= 0.0
            and fragment.sigma_y >= 0.0
        ]

        if not valid_fragments:
            return [], {
                "median_sigma_x": np.nan,
                "median_sigma_y": np.nan,
                "median_dt": np.nan,
            }

        sigma_x_values = np.array(
            [
                fragment.sigma_x
                for fragment in valid_fragments
            ],
            dtype=np.float64,
        )

        sigma_y_values = np.array(
            [
                fragment.sigma_y
                for fragment in valid_fragments
            ],
            dtype=np.float64,
        )

        median_sigma_x = float(
            np.median(
                sigma_x_values
            )
        )

        median_sigma_y = float(
            np.median(
                sigma_y_values
            )
        )

        dt_values = np.concatenate(
            [
                fragment.dt
                for fragment in valid_fragments
                if len(fragment.dt) > 0
            ]
        )

        median_dt = float(
            np.median(
                dt_values
            )
        ) if len(dt_values) else np.nan

        robust_diff_sigma_x = (
            median_sigma_x
            * np.sqrt(2.0)
        )

        robust_diff_sigma_y = (
            median_sigma_y
            * np.sqrt(2.0)
        )

        min_sigma_x = (
            median_sigma_x
            * STATIONARY_SIGMA_MIN_FACTOR
        )

        max_sigma_x = (
            median_sigma_x
            * STATIONARY_SIGMA_MAX_FACTOR
        )

        min_sigma_y = (
            median_sigma_y
            * STATIONARY_SIGMA_MIN_FACTOR
        )

        max_sigma_y = (
            median_sigma_y
            * STATIONARY_SIGMA_MAX_FACTOR
        )

        accepted: list[StationaryFragment] = []

        for fragment in valid_fragments:
            reasons: list[str] = []

            # ------------------------------------------------
            # Разброс X
            # ------------------------------------------------

            if (
                np.isfinite(median_sigma_x)
                and median_sigma_x > 0.0
            ):
                if not (
                    min_sigma_x
                    <= fragment.sigma_x
                    <= max_sigma_x
                ):
                    reasons.append(
                        "sigma_x_outlier"
                    )

            # ------------------------------------------------
            # Разброс Y
            # ------------------------------------------------

            if (
                np.isfinite(median_sigma_y)
                and median_sigma_y > 0.0
            ):
                if not (
                    min_sigma_y
                    <= fragment.sigma_y
                    <= max_sigma_y
                ):
                    reasons.append(
                        "sigma_y_outlier"
                    )

            # ------------------------------------------------
            # Локальный дрейф X
            # ------------------------------------------------

            n_diff = len(fragment.dt)

            if (
                robust_diff_sigma_x > 0.0
                and n_diff > 1
            ):
                se_mean_dx = (
                    robust_diff_sigma_x
                    / np.sqrt(n_diff)
                )

                if (
                    abs(fragment.mean_dx)
                    > STATIONARITY_Z
                    * se_mean_dx
                ):
                    reasons.append(
                        "drift_x"
                    )

            # ------------------------------------------------
            # Локальный дрейф Y
            # ------------------------------------------------

            if (
                robust_diff_sigma_y > 0.0
                and n_diff > 1
            ):
                se_mean_dy = (
                    robust_diff_sigma_y
                    / np.sqrt(n_diff)
                )

                if (
                    abs(fragment.mean_dy)
                    > STATIONARITY_Z
                    * se_mean_dy
                ):
                    reasons.append(
                        "drift_y"
                    )

            # ------------------------------------------------
            # Большой временной gap
            # ------------------------------------------------

            if (
                np.isfinite(median_dt)
                and median_dt > 0.0
            ):
                fragment_median_dt = float(
                    np.median(
                        fragment.dt
                    )
                )

                fragment_max_dt = float(
                    np.max(
                        fragment.dt
                    )
                )

                if (
                    fragment_median_dt
                    > median_dt
                    * STATIONARY_MAX_DT_RATIO
                ):
                    reasons.append(
                        "dt_median_outlier"
                    )

                if (
                    fragment_max_dt
                    > median_dt
                    * STATIONARY_MAX_DT_RATIO
                ):
                    reasons.append(
                        "dt_gap"
                    )

            if reasons:
                fragment.accepted = False
                fragment.reject_reason = ",".join(
                    reasons
                )
            else:
                fragment.accepted = True
                fragment.reject_reason = ""
                accepted.append(fragment)

        statistics = {
            "median_sigma_x": median_sigma_x,
            "median_sigma_y": median_sigma_y,
            "median_dt": median_dt,
            "total_fragments": len(fragments),
            "valid_fragments": len(valid_fragments),
            "accepted_fragments": len(accepted),
        }

        return accepted, statistics

    # ========================================================
    # Оценка sigma_meas
    # ========================================================

    @staticmethod
    def estimate_sigma_meas(
        segments: Iterable[TrackSegment],
    ) -> dict[str, float | int | str]:
        """
        Оценивает sigma_meas только по локально-стационарным
        stand-фрагментам.

        Для measurement noise:

            z_i = true_position + epsilon_i

        При независимом noise:

            diff(z_i) = epsilon_i - epsilon_(i-1)

        поэтому:

            Var(diff) = 2 * sigma_meas²

        и:

            sigma_meas = sqrt(Var(diff) / 2)

        Для итоговой оценки используется pooled within-fragment
        variance, то есть локальный средний drift не попадает
        напрямую в оценку дисперсии.
        """

        fragments = (
            KalmanParameterEstimator
            .build_stationary_fragments(
                segments=segments,
                fragment_length=STATIONARY_FRAGMENT_LENGTH,
            )
        )

        accepted, fragment_stats = (
            KalmanParameterEstimator
            .select_stationary_fragments(
                fragments
            )
        )

        if (
            len(accepted)
            < STATIONARY_MIN_ACCEPTED_FRAGMENTS
        ):
            logging.warning(
                "Недостаточно локально-стационарных "
                "stand-фрагментов: %d",
                len(accepted),
            )

            return {
                "sigma_meas_x": np.nan,
                "sigma_meas_y": np.nan,
                "sigma_meas": np.nan,
                "fragment_count_total": int(
                    len(fragments)
                ),
                "fragment_count_accepted": int(
                    len(accepted)
                ),
                "point_count": 0,
                "status": "INSUFFICIENT_FRAGMENTS",
            }

        weighted_var_x = 0.0
        weighted_var_y = 0.0

        weight_total_x = 0
        weight_total_y = 0

        for fragment in accepted:
            n = len(
                fragment.dt
            )

            if n < 2:
                continue

            weighted_var_x += (
                (n - 1)
                * fragment.var_dx
            )

            weighted_var_y += (
                (n - 1)
                * fragment.var_dy
            )

            weight_total_x += n - 1
            weight_total_y += n - 1

        if (
            weight_total_x == 0
            or weight_total_y == 0
        ):
            return {
                "sigma_meas_x": np.nan,
                "sigma_meas_y": np.nan,
                "sigma_meas": np.nan,
                "fragment_count_total": int(
                    len(fragments)
                ),
                "fragment_count_accepted": int(
                    len(accepted)
                ),
                "point_count": 0,
                "status": "NO_VARIANCE_DATA",
            }

        pooled_var_dx = (
            weighted_var_x
            / weight_total_x
        )

        pooled_var_dy = (
            weighted_var_y
            / weight_total_y
        )

        sigma_meas_x = np.sqrt(
            max(
                pooled_var_dx / 2.0,
                0.0,
            )
        )

        sigma_meas_y = np.sqrt(
            max(
                pooled_var_dy / 2.0,
                0.0,
            )
        )

        sigma_meas = np.sqrt(
            (
                sigma_meas_x ** 2
                + sigma_meas_y ** 2
            )
            / 2.0
        )

        point_count = sum(
            len(fragment.x)
            for fragment in accepted
        )

        logging.info(
            "sigma_meas: "
            "total_fragments=%d, "
            "accepted=%d, "
            "sigma_x=%.8g, "
            "sigma_y=%.8g, "
            "sigma=%.8g",
            len(fragments),
            len(accepted),
            sigma_meas_x,
            sigma_meas_y,
            sigma_meas,
        )

        return {
            "sigma_meas_x": float(
                sigma_meas_x
            ),
            "sigma_meas_y": float(
                sigma_meas_y
            ),
            "sigma_meas": float(
                sigma_meas
            ),
            "fragment_count_total": int(
                len(fragments)
            ),
            "fragment_count_accepted": int(
                len(accepted)
            ),
            "point_count": int(
                point_count
            ),
            "median_sigma_x": float(
                fragment_stats["median_sigma_x"]
            ),
            "median_sigma_y": float(
                fragment_stats["median_sigma_y"]
            ),
            "median_dt": float(
                fragment_stats["median_dt"]
            ),
            "status": "OK",
        }

    # ========================================================
    # Локальные окна
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
        """Строит локальные X/Y и dt для набора окон."""

        if len(start_indices) == 0:
            empty = np.empty(
                (0, window + 1),
                dtype=np.float64,
            )

            return (
                empty,
                empty.copy(),
                np.empty(
                    (0, window),
                    dtype=np.float64,
                ),
            )

        starts = np.asarray(
            start_indices,
            dtype=np.int64,
        )

        offsets = np.arange(
            window + 1,
            dtype=np.int64,
        )

        indices = (
            starts[:, None]
            + offsets[None, :]
        )

        lon_windows = (
            segment.lon[indices]
        )

        lat_windows = (
            segment.lat[indices]
        )

        lon0 = lon_windows[:, 0]
        lat0 = lat_windows[:, 0]

        kx = (
            DataProcessor.LEN_LAT
            * np.cos(
                np.radians(lat0)
            )
        )

        x_windows = (
            lon_windows
            - lon0[:, None]
        ) * kx[:, None]

        y_windows = (
            lat_windows
            - lat0[:, None]
        ) * DataProcessor.LEN_LAT

        dt_indices = (
            starts[:, None]
            + np.arange(
                window,
                dtype=np.int64,
            )[None, :]
            + 1
        )

        dt_windows = segment.dt[
            dt_indices - 1
        ]

        return (
            x_windows,
            y_windows,
            dt_windows,
        )

    # ========================================================
    # CV batch
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
        """Пакетный CV-фильтр для множества локальных окон."""

        batch_size = len(x_windows)

        if batch_size == 0:
            empty = np.empty(
                0,
                dtype=np.float64,
            )

            return (
                empty,
                empty.copy(),
                empty.copy(),
            )

        state_dim = 4

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        R = (
            np.eye(
                2,
                dtype=np.float64,
            )
            * sigma_meas ** 2
        )

        I = np.eye(
            state_dim,
            dtype=np.float64,
        )

        log_2pi = np.log(
            2.0 * np.pi
        )

        dt0 = dt_windows[:, 0]

        vx0 = np.divide(
            x_windows[:, 1]
            - x_windows[:, 0],
            dt0,
            out=np.zeros(
                batch_size,
                dtype=np.float64,
            ),
            where=dt0 > 0.0,
        )

        vy0 = np.divide(
            y_windows[:, 1]
            - y_windows[:, 0],
            dt0,
            out=np.zeros(
                batch_size,
                dtype=np.float64,
            ),
            where=dt0 > 0.0,
        )

        state = np.column_stack(
            [
                x_windows[:, 0],
                y_windows[:, 0],
                vx0,
                vy0,
            ]
        )

        P = np.broadcast_to(
            I * 500.0,
            (
                batch_size,
                state_dim,
                state_dim,
            ),
        ).copy()

        positive_dt0 = (
            dt0 > 0.0
        )

        vel_var = np.full(
            batch_size,
            100.0,
            dtype=np.float64,
        )

        vel_var[positive_dt0] = (
            2.0
            * sigma_meas ** 2
            / dt0[positive_dt0] ** 2
        )

        P[:, 2, 2] = vel_var
        P[:, 3, 3] = vel_var

        current_log_likelihood = np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

        current_mahalanobis_sq = np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

        current_filter_distance = np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

        window = (
            x_windows.shape[1] - 1
        )

        variance_acc = (
            sigma_acc ** 2
        )

        for k in range(
            1,
            window + 1,
        ):
            dt = dt_windows[
                :,
                k - 1,
            ]

            F = np.zeros(
                (
                    batch_size,
                    state_dim,
                    state_dim,
                ),
                dtype=np.float64,
            )

            F[:, 0, 0] = 1.0
            F[:, 1, 1] = 1.0
            F[:, 2, 2] = 1.0
            F[:, 3, 3] = 1.0

            F[:, 0, 2] = dt
            F[:, 1, 3] = dt

            dt2 = dt ** 2
            dt3 = dt ** 3
            dt4 = dt ** 4

            Q = np.zeros_like(P)

            Q[:, 0, 0] = (
                dt4 / 4.0
                * variance_acc
            )

            Q[:, 1, 1] = (
                dt4 / 4.0
                * variance_acc
            )

            Q[:, 2, 2] = (
                dt2
                * variance_acc
            )

            Q[:, 3, 3] = (
                dt2
                * variance_acc
            )

            Q[:, 0, 2] = (
                dt3 / 2.0
                * variance_acc
            )

            Q[:, 2, 0] = (
                dt3 / 2.0
                * variance_acc
            )

            Q[:, 1, 3] = (
                dt3 / 2.0
                * variance_acc
            )

            Q[:, 3, 1] = (
                dt3 / 2.0
                * variance_acc
            )

            state_pred = np.einsum(
                "bij,bj->bi",
                F,
                state,
            )

            FP = np.einsum(
                "bij,bjk->bik",
                F,
                P,
            )

            P_pred = (
                np.einsum(
                    "bik,bjk->bij",
                    FP,
                    F,
                )
                + Q
            )

            z = np.column_stack(
                [
                    x_windows[:, k],
                    y_windows[:, k],
                ]
            )

            innovation = (
                z
                - state_pred[:, :2]
            )

            S = (
                P_pred[:, :2, :2]
                + R
            )

            S = 0.5 * (
                S
                + np.transpose(
                    S,
                    (0, 2, 1),
                )
            )

            sign, logdet = (
                np.linalg.slogdet(S)
            )

            valid_s = (
                (sign > 0.0)
                & np.isfinite(logdet)
            )

            try:
                solved = np.linalg.solve(
                    S,
                    innovation[..., None],
                )[..., 0]

                mahalanobis_sq = np.sum(
                    innovation
                    * solved,
                    axis=1,
                )

                valid = (
                    valid_s
                    & np.isfinite(
                        mahalanobis_sq
                    )
                    & (
                        mahalanobis_sq
                        >= 0.0
                    )
                )

                log_likelihood = -0.5 * (
                    2.0 * log_2pi
                    + logdet
                    + mahalanobis_sq
                )

                solved_gain = (
                    np.linalg.solve(
                        S,
                        P_pred[:, :2, :],
                    )
                )

                K = np.transpose(
                    solved_gain,
                    (0, 2, 1),
                )

                state = (
                    state_pred
                    + np.einsum(
                        "bij,bj->bi",
                        K,
                        innovation,
                    )
                )

                I_KH = (
                    I[None, :, :]
                    - np.einsum(
                        "bij,jk->bik",
                        K,
                        H,
                    )
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

                KR = K @ R

                P += np.einsum(
                    "bij,bkj->bik",
                    KR,
                    K,
                )

                P = 0.5 * (
                    P
                    + np.transpose(
                        P,
                        (0, 2, 1),
                    )
                )

                current_log_likelihood[
                    valid
                ] = log_likelihood[
                    valid
                ]

                current_mahalanobis_sq[
                    valid
                ] = mahalanobis_sq[
                    valid
                ]

                current_filter_distance[
                    valid
                ] = np.hypot(
                    state[
                        valid,
                        0,
                    ]
                    - x_windows[
                        valid,
                        k,
                    ],
                    state[
                        valid,
                        1,
                    ]
                    - y_windows[
                        valid,
                        k,
                    ],
                )

            except np.linalg.LinAlgError:
                return (
                    np.full(
                        batch_size,
                        -np.inf,
                    ),
                    np.full(
                        batch_size,
                        np.nan,
                    ),
                    np.full(
                        batch_size,
                        np.nan,
                    ),
                )

        return (
            current_log_likelihood,
            current_mahalanobis_sq,
            current_filter_distance,
        )

    # ========================================================
    # RW batch
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
        """Пакетный RW-фильтр."""

        batch_size = len(x_windows)

        if batch_size == 0:
            empty = np.empty(
                0,
                dtype=np.float64,
            )

            return (
                empty,
                empty.copy(),
                empty.copy(),
            )

        R = (
            np.eye(
                2,
                dtype=np.float64,
            )
            * sigma_meas ** 2
        )

        I = np.eye(
            2,
            dtype=np.float64,
        )

        log_2pi = np.log(
            2.0 * np.pi
        )

        state = np.column_stack(
            [
                x_windows[:, 0],
                y_windows[:, 0],
            ]
        )

        P = np.broadcast_to(
            I * sigma_meas ** 2,
            (
                batch_size,
                2,
                2,
            ),
        ).copy()

        current_log_likelihood = np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

        current_mahalanobis_sq = np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

        current_filter_distance = np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

        window = (
            x_windows.shape[1] - 1
        )

        for k in range(
            1,
            window + 1,
        ):
            dt = dt_windows[
                :,
                k - 1,
            ]

            Q = (
                np.eye(
                    2,
                    dtype=np.float64,
                )[None, :, :]
                * (
                    sigma_rw ** 2
                    * dt
                )[:, None, None]
            )

            state_pred = state
            P_pred = P + Q

            z = np.column_stack(
                [
                    x_windows[:, k],
                    y_windows[:, k],
                ]
            )

            innovation = (
                z - state_pred
            )

            S = (
                P_pred + R
            )

            S = 0.5 * (
                S
                + np.transpose(
                    S,
                    (0, 2, 1),
                )
            )

            sign, logdet = (
                np.linalg.slogdet(S)
            )

            valid_s = (
                (sign > 0.0)
                & np.isfinite(logdet)
            )

            try:
                solved = np.linalg.solve(
                    S,
                    innovation[..., None],
                )[..., 0]

                mahalanobis_sq = np.sum(
                    innovation
                    * solved,
                    axis=1,
                )

                valid = (
                    valid_s
                    & np.isfinite(
                        mahalanobis_sq
                    )
                    & (
                        mahalanobis_sq
                        >= 0.0
                    )
                )

                log_likelihood = -0.5 * (
                    2.0 * log_2pi
                    + logdet
                    + mahalanobis_sq
                )

                K = np.transpose(
                    np.linalg.solve(
                        S,
                        P_pred,
                    ),
                    (0, 2, 1),
                )

                state = (
                    state_pred
                    + np.einsum(
                        "bij,bj->bi",
                        K,
                        innovation,
                    )
                )

                I_K = (
                    I[None, :, :]
                    - K
                )

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

                KR = K @ R

                P += np.einsum(
                    "bij,bkj->bik",
                    KR,
                    K,
                )

                P = 0.5 * (
                    P
                    + np.transpose(
                        P,
                        (0, 2, 1),
                    )
                )

                current_log_likelihood[
                    valid
                ] = log_likelihood[
                    valid
                ]

                current_mahalanobis_sq[
                    valid
                ] = mahalanobis_sq[
                    valid
                ]

                current_filter_distance[
                    valid
                ] = np.hypot(
                    state[
                        valid,
                        0,
                    ]
                    - x_windows[
                        valid,
                        k,
                    ],
                    state[
                        valid,
                        1,
                    ]
                    - y_windows[
                        valid,
                        k,
                    ],
                )

            except np.linalg.LinAlgError:
                return (
                    np.full(
                        batch_size,
                        -np.inf,
                    ),
                    np.full(
                        batch_size,
                        np.nan,
                    ),
                    np.full(
                        batch_size,
                        np.nan,
                    ),
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
        """Индексы начала локальных окон."""

        n_windows = (
            segment.size
            - window
        )

        if n_windows <= 0:
            return np.empty(
                0,
                dtype=np.int64,
            )

        return np.arange(
            0,
            n_windows,
            stride,
            dtype=np.int64,
        )

    # ========================================================
    # Evaluation CV
    # ========================================================

    @staticmethod
    def _evaluate_cv_segments(
        segments: list[PreparedSegment],
        sigma_acc: float,
        sigma_meas: float,
        window: int,
        stride: int,
        collect_metrics: bool,
    ) -> dict:
        """Общий пакетный расчёт CV."""

        ll_list: list[
            NDArray[np.float64]
        ] = []

        m2_list: list[
            NDArray[np.float64]
        ] = []

        fd_list: list[
            NDArray[np.float64]
        ] = []

        total_log_likelihood = 0.0
        observation_count = 0

        for segment in segments:
            starts = (
                KalmanParameterEstimator
                ._window_start_indices(
                    segment=segment,
                    window=window,
                    stride=stride,
                )
            )

            if len(starts) == 0:
                continue

            for begin in range(
                0,
                len(starts),
                BATCH_SIZE,
            ):
                end = min(
                    begin + BATCH_SIZE,
                    len(starts),
                )

                current_starts = (
                    starts[begin:end]
                )

                (
                    x_windows,
                    y_windows,
                    dt_windows,
                ) = (
                    KalmanParameterEstimator
                    ._build_local_windows(
                        segment=segment,
                        start_indices=current_starts,
                        window=window,
                    )
                )

                (
                    ll,
                    m2,
                    fd,
                ) = (
                    KalmanParameterEstimator
                    ._cv_batch_statistics(
                        x_windows=x_windows,
                        y_windows=y_windows,
                        dt_windows=dt_windows,
                        sigma_acc=sigma_acc,
                        sigma_meas=sigma_meas,
                    )
                )

                valid_ll = ll[
                    np.isfinite(ll)
                ]

                if len(valid_ll):
                    total_log_likelihood += float(
                        np.sum(valid_ll)
                    )

                    observation_count += (
                        len(valid_ll)
                    )

                if collect_metrics:
                    valid = (
                        np.isfinite(ll)
                        & np.isfinite(m2)
                        & np.isfinite(fd)
                    )

                    if np.any(valid):
                        ll_list.append(
                            ll[valid]
                        )

                        m2_list.append(
                            m2[valid]
                        )

                        fd_list.append(
                            fd[valid]
                        )

        if collect_metrics:
            ll = (
                np.concatenate(ll_list)
                if ll_list
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

            m2 = (
                np.concatenate(m2_list)
                if m2_list
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

            fd = (
                np.concatenate(fd_list)
                if fd_list
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

            return {
                "log_likelihood": ll,
                "mahalanobis": m2,
                "filter_distance": fd,
                "total_log_likelihood": float(
                    np.sum(ll)
                ),
                "observation_count": int(
                    len(ll)
                ),
            }

        return {
            "total_log_likelihood": (
                total_log_likelihood
            ),
            "observation_count": (
                observation_count
            ),
        }

    # ========================================================
    # Evaluation RW
    # ========================================================

    @staticmethod
    def _evaluate_rw_segments(
        segments: list[PreparedSegment],
        sigma_rw: float,
        sigma_meas: float,
        window: int,
        stride: int,
        collect_metrics: bool,
    ) -> dict:
        """Общий пакетный расчёт RW."""

        ll_list: list[
            NDArray[np.float64]
        ] = []

        m2_list: list[
            NDArray[np.float64]
        ] = []

        fd_list: list[
            NDArray[np.float64]
        ] = []

        total_log_likelihood = 0.0
        observation_count = 0

        for segment in segments:
            starts = (
                KalmanParameterEstimator
                ._window_start_indices(
                    segment=segment,
                    window=window,
                    stride=stride,
                )
            )

            if len(starts) == 0:
                continue

            for begin in range(
                0,
                len(starts),
                BATCH_SIZE,
            ):
                end = min(
                    begin + BATCH_SIZE,
                    len(starts),
                )

                current_starts = (
                    starts[begin:end]
                )

                (
                    x_windows,
                    y_windows,
                    dt_windows,
                ) = (
                    KalmanParameterEstimator
                    ._build_local_windows(
                        segment=segment,
                        start_indices=current_starts,
                        window=window,
                    )
                )

                (
                    ll,
                    m2,
                    fd,
                ) = (
                    KalmanParameterEstimator
                    ._rw_batch_statistics(
                        x_windows=x_windows,
                        y_windows=y_windows,
                        dt_windows=dt_windows,
                        sigma_rw=sigma_rw,
                        sigma_meas=sigma_meas,
                    )
                )

                valid_ll = ll[
                    np.isfinite(ll)
                ]

                if len(valid_ll):
                    total_log_likelihood += float(
                        np.sum(valid_ll)
                    )

                    observation_count += (
                        len(valid_ll)
                    )

                if collect_metrics:
                    valid = (
                        np.isfinite(ll)
                        & np.isfinite(m2)
                        & np.isfinite(fd)
                    )

                    if np.any(valid):
                        ll_list.append(
                            ll[valid]
                        )

                        m2_list.append(
                            m2[valid]
                        )

                        fd_list.append(
                            fd[valid]
                        )

        if collect_metrics:
            ll = (
                np.concatenate(ll_list)
                if ll_list
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

            m2 = (
                np.concatenate(m2_list)
                if m2_list
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

            fd = (
                np.concatenate(fd_list)
                if fd_list
                else np.empty(
                    0,
                    dtype=np.float64,
                )
            )

            return {
                "log_likelihood": ll,
                "mahalanobis": m2,
                "filter_distance": fd,
                "total_log_likelihood": float(
                    np.sum(ll)
                ),
                "observation_count": int(
                    len(ll)
                ),
            }

        return {
            "total_log_likelihood": (
                total_log_likelihood
            ),
            "observation_count": (
                observation_count
            ),
        }

    # ========================================================
    # MLE CV: оптимизируем только sigma_acc
    # ========================================================

    @staticmethod
    def fit_cv_mle(
        segments: list[TrackSegment],
        sigma_meas: float,
        sigma_acc_initial: float = DEFAULT_SIGMA_ACC_INITIAL,
        window: int = WINDOW,
    ) -> dict[str, float | bool | str | int]:
        """
        MLE CV.

        sigma_meas фиксирован.

        Оптимизируется только sigma_acc.
        """

        if not np.isfinite(sigma_meas) or sigma_meas <= 0.0:
            return {
                "success": False,
                "sigma_meas": np.nan,
                "sigma_acc": np.nan,
                "negative_log_likelihood": np.nan,
                "n_mle_windows": 0,
                "message": "invalid sigma_meas",
            }

        prepared = (
            KalmanParameterEstimator
            .prepare_segments(
                segments
            )
        )

        if not prepared:
            return {
                "success": False,
                "sigma_meas": sigma_meas,
                "sigma_acc": np.nan,
                "negative_log_likelihood": np.nan,
                "n_mle_windows": 0,
                "message": "нет подходящих сегментов",
            }

        sigma_acc_initial = float(
            np.clip(
                sigma_acc_initial,
                SIGMA_ACC_MIN,
                SIGMA_ACC_MAX,
            )
        )

        total_windows = sum(
            len(
                KalmanParameterEstimator
                ._window_start_indices(
                    segment=segment,
                    window=window,
                    stride=MLE_STRIDE,
                )
            )
            for segment in prepared
        )

        logging.info(
            "CV MLE: fixed sigma_meas=%.8g, "
            "segments=%d, MLE windows=%d, stride=%d",
            sigma_meas,
            len(prepared),
            total_windows,
            MLE_STRIDE,
        )

        objective_calls = 0

        log_min = np.log(
            SIGMA_ACC_MIN
        )

        log_max = np.log(
            SIGMA_ACC_MAX
        )

        def objective(
            log_sigma_acc: float,
        ) -> float:
            nonlocal objective_calls

            objective_calls += 1

            sigma_acc = float(
                np.exp(log_sigma_acc)
            )

            result = (
                KalmanParameterEstimator
                ._evaluate_cv_segments(
                    segments=prepared,
                    sigma_acc=sigma_acc,
                    sigma_meas=sigma_meas,
                    window=window,
                    stride=MLE_STRIDE,
                    collect_metrics=False,
                )
            )

            total_ll = float(
                result[
                    "total_log_likelihood"
                ]
            )

            count = int(
                result[
                    "observation_count"
                ]
            )

            if count == 0:
                return np.inf

            nll = -total_ll

            if (
                objective_calls == 1
                or objective_calls
                % OPTIMIZER_LOG_EVERY
                == 0
            ):
                logging.info(
                    "CV MLE evaluation=%d: "
                    "sigma_acc=%.8g, "
                    "NLL=%.8g",
                    objective_calls,
                    sigma_acc,
                    nll,
                )

            return nll

        initial_log = np.clip(
            np.log(
                sigma_acc_initial
            ),
            log_min,
            log_max,
        )

        # В одномерной задаче сначала делаем небольшой локальный
        # поиск вокруг первоначального значения, затем полный bounded
        # поиск в лог-пространстве.
        local_left = max(
            log_min,
            initial_log
            - np.log(10.0),
        )

        local_right = min(
            log_max,
            initial_log
            + np.log(10.0),
        )

        result = minimize_scalar(
            objective,
            bounds=(
                local_left,
                local_right,
            ),
            method="bounded",
            options={
                "maxiter": OPTIMIZER_MAXITER,
                "xatol": OPTIMIZER_XTOL,
            },
        )

        sigma_acc = float(
            np.exp(
                result.x
            )
        )

        return {
            "success": bool(
                result.success
            ),
            "sigma_meas": float(
                sigma_meas
            ),
            "sigma_acc": sigma_acc,
            "negative_log_likelihood": float(
                result.fun
            ),
            "n_mle_windows": int(
                total_windows
            ),
            "message": str(
                result.message
            ),
        }

    # ========================================================
    # MLE RW: оптимизируем только sigma_rw
    # ========================================================

    @staticmethod
    def fit_rw_mle(
        segments: list[TrackSegment],
        sigma_meas: float,
        sigma_rw_initial: float = DEFAULT_SIGMA_RW_INITIAL,
        window: int = WINDOW,
    ) -> dict[str, float | bool | str | int]:
        """
        MLE RW.

        sigma_meas фиксирован.

        Оптимизируется только sigma_rw.
        """

        if not np.isfinite(sigma_meas) or sigma_meas <= 0.0:
            return {
                "success": False,
                "sigma_meas": np.nan,
                "sigma_rw": np.nan,
                "negative_log_likelihood": np.nan,
                "n_mle_windows": 0,
                "message": "invalid sigma_meas",
            }

        prepared = (
            KalmanParameterEstimator
            .prepare_segments(
                segments
            )
        )

        if not prepared:
            return {
                "success": False,
                "sigma_meas": sigma_meas,
                "sigma_rw": np.nan,
                "negative_log_likelihood": np.nan,
                "n_mle_windows": 0,
                "message": "нет подходящих сегментов",
            }

        sigma_rw_initial = float(
            np.clip(
                sigma_rw_initial,
                SIGMA_RW_MIN,
                SIGMA_RW_MAX,
            )
        )

        total_windows = sum(
            len(
                KalmanParameterEstimator
                ._window_start_indices(
                    segment=segment,
                    window=window,
                    stride=MLE_STRIDE,
                )
            )
            for segment in prepared
        )

        logging.info(
            "RW MLE: fixed sigma_meas=%.8g, "
            "segments=%d, MLE windows=%d, stride=%d",
            sigma_meas,
            len(prepared),
            total_windows,
            MLE_STRIDE,
        )

        objective_calls = 0

        log_min = np.log(
            SIGMA_RW_MIN
        )

        log_max = np.log(
            SIGMA_RW_MAX
        )

        def objective(
            log_sigma_rw: float,
        ) -> float:
            nonlocal objective_calls

            objective_calls += 1

            sigma_rw = float(
                np.exp(log_sigma_rw)
            )

            result = (
                KalmanParameterEstimator
                ._evaluate_rw_segments(
                    segments=prepared,
                    sigma_rw=sigma_rw,
                    sigma_meas=sigma_meas,
                    window=window,
                    stride=MLE_STRIDE,
                    collect_metrics=False,
                )
            )

            total_ll = float(
                result[
                    "total_log_likelihood"
                ]
            )

            count = int(
                result[
                    "observation_count"
                ]
            )

            if count == 0:
                return np.inf

            nll = -total_ll

            if (
                objective_calls == 1
                or objective_calls
                % OPTIMIZER_LOG_EVERY
                == 0
            ):
                logging.info(
                    "RW MLE evaluation=%d: "
                    "sigma_rw=%.8g, "
                    "NLL=%.8g",
                    objective_calls,
                    sigma_rw,
                    nll,
                )

            return nll

        initial_log = np.clip(
            np.log(
                sigma_rw_initial
            ),
            log_min,
            log_max,
        )

        local_left = max(
            log_min,
            initial_log
            - np.log(10.0),
        )

        local_right = min(
            log_max,
            initial_log
            + np.log(10.0),
        )

        result = minimize_scalar(
            objective,
            bounds=(
                local_left,
                local_right,
            ),
            method="bounded",
            options={
                "maxiter": OPTIMIZER_MAXITER,
                "xatol": OPTIMIZER_XTOL,
            },
        )

        sigma_rw = float(
            np.exp(
                result.x
            )
        )

        return {
            "success": bool(
                result.success
            ),
            "sigma_meas": float(
                sigma_meas
            ),
            "sigma_rw": sigma_rw,
            "negative_log_likelihood": float(
                result.fun
            ),
            "n_mle_windows": int(
                total_windows
            ),
            "message": str(
                result.message
            ),
        }

    # ========================================================
    # Финальный CV
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

        Здесь stride=1.
        """

        prepared = (
            KalmanParameterEstimator
            .prepare_segments(
                segments
            )
        )

        result = (
            KalmanParameterEstimator
            ._evaluate_cv_segments(
                segments=prepared,
                sigma_acc=sigma_acc,
                sigma_meas=sigma_meas,
                window=window,
                stride=1,
                collect_metrics=True,
            )
        )

        return {
            "log_likelihood": result[
                "log_likelihood"
            ],
            "mahalanobis": result[
                "mahalanobis"
            ],
            "filter_distance": result[
                "filter_distance"
            ],
        }

    # ========================================================
    # Финальный RW
    # ========================================================

    @staticmethod
    def collect_rw_likelihood(
        segments: Iterable[TrackSegment],
        sigma_rw: float,
        sigma_meas: float,
        window: int = WINDOW,
    ) -> dict[str, NDArray[np.float64]]:
        """Финальная RW-статистика, stride=1."""

        prepared = (
            KalmanParameterEstimator
            .prepare_segments(
                segments
            )
        )

        result = (
            KalmanParameterEstimator
            ._evaluate_rw_segments(
                segments=prepared,
                sigma_rw=sigma_rw,
                sigma_meas=sigma_meas,
                window=window,
                stride=1,
                collect_metrics=True,
            )
        )

        return {
            "log_likelihood": result[
                "log_likelihood"
            ],
            "mahalanobis": result[
                "mahalanobis"
            ],
            "filter_distance": result[
                "filter_distance"
            ],
        }

    # ========================================================
    # Сводка Mahalanobis
    # ========================================================

    @staticmethod
    def summarize_mahalanobis(
        mahalanobis: NDArray[np.float64],
    ) -> dict[str, float | int]:
        """
        Рассчитывает основные характеристики распределения M².
        """

        values = np.asarray(
            mahalanobis,
            dtype=np.float64,
        )

        values = values[
            np.isfinite(values)
            & (values >= 0.0)
        ]

        if values.size == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "median": np.nan,
                "p95": np.nan,
                "p99": np.nan,
                "frac_gt_5_991": np.nan,
                "frac_gt_9_21": np.nan,
                "frac_gt_13_82": np.nan,
            }

        return {
            "n": int(values.size),
            "mean": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
            "p95": float(
                np.percentile(
                    values,
                    95,
                )
            ),
            "p99": float(
                np.percentile(
                    values,
                    99,
                )
            ),
            "frac_gt_5_991": float(
                np.mean(
                    values
                    > 5.991
                )
            ),
            "frac_gt_9_21": float(
                np.mean(
                    values
                    > 9.21
                )
            ),
            "frac_gt_13_82": float(
                np.mean(
                    values
                    > 13.82
                )
            ),
        }

    # ========================================================
    # Валидация
    # ========================================================

    @staticmethod
    def validate_model(
        mahalanobis: NDArray[np.float64],
        optimizer_success: bool,
        process_name: str,
    ) -> dict[str, float | int | str]:
        """
        Формирует validation-метрики.

        Теоретические вероятности для chi2(df=2):

            P(M² > 5.991) = 5%
            P(M² > 9.210) = 1%
            P(M² > 13.816) = 0.1%
        """

        summary = (
            KalmanParameterEstimator
            .summarize_mahalanobis(
                mahalanobis
            )
        )

        n = int(
            summary["n"]
        )

        if n == 0:
            return {
                "validation_n": 0,
                "validation_mean_mahalanobis": np.nan,
                "validation_median_mahalanobis": np.nan,
                "validation_p95_mahalanobis": np.nan,
                "validation_p99_mahalanobis": np.nan,
                "validation_frac_m2_gt_5_991": np.nan,
                "validation_frac_m2_gt_9_21": np.nan,
                "validation_frac_m2_gt_13_82": np.nan,
                "validation_expected_frac_gt_5_991": 0.05,
                "validation_expected_frac_gt_9_21": 0.01,
                "validation_expected_frac_gt_13_82": 0.001,
                "validation_mean_ratio": np.nan,
                "validation_median_ratio": np.nan,
                "validation_p95_ratio": np.nan,
                "validation_p99_ratio": np.nan,
                "validation_quality": "NO_DATA",
                "validation_message": (
                    f"{process_name}: нет validation-данных"
                ),
            }

        mean_m2 = float(
            summary["mean"]
        )

        median_m2 = float(
            summary["median"]
        )

        p95_m2 = float(
            summary["p95"]
        )

        p99_m2 = float(
            summary["p99"]
        )

        frac_5_991 = float(
            summary["frac_gt_5_991"]
        )

        frac_9_21 = float(
            summary["frac_gt_9_21"]
        )

        frac_13_82 = float(
            summary["frac_gt_13_82"]
        )

        mean_ratio = (
            mean_m2 / 2.0
        )

        median_ratio = (
            median_m2
            / float(
                chi2.ppf(
                    0.50,
                    df=2,
                )
            )
        )

        p95_ratio = (
            p95_m2
            / MAHALANOBIS_THRESHOLD_P95
        )

        p99_ratio = (
            p99_m2
            / MAHALANOBIS_THRESHOLD_P99
        )

        # ----------------------------------------------------
        # Качество модели.
        #
        # Основной акцент:
        #   P95, P99 и реальные хвостовые доли.
        #
        # Mean специально не используется как единственный
        # критерий, поскольку редкие gross errors могут
        # сделать его огромным.
        # ----------------------------------------------------

        close_p95 = (
            0.7
            <= p95_ratio
            <= 1.3
        )

        close_p99 = (
            0.7
            <= p99_ratio
            <= 1.3
        )

        frac_ok = (
            0.5 * 0.05
            <= frac_5_991
            <= 2.0 * 0.05
            and
            0.5 * 0.01
            <= frac_9_21
            <= 2.0 * 0.01
            and
            0.5 * 0.001
            <= frac_13_82
            <= 2.0 * 0.001
        )

        if not optimizer_success:
            quality = "OPTIMIZER_FAILED"
        elif close_p95 and close_p99 and frac_ok:
            quality = "GOOD"
        elif (
            p95_ratio < 0.5
            or p95_ratio > 2.0
            or p99_ratio < 0.5
            or p99_ratio > 2.0
        ):
            quality = "POOR"
        else:
            quality = "CHECK"

        return {
            "validation_n": n,
            "validation_mean_mahalanobis": mean_m2,
            "validation_median_mahalanobis": median_m2,
            "validation_p95_mahalanobis": p95_m2,
            "validation_p99_mahalanobis": p99_m2,
            "validation_frac_m2_gt_5_991": frac_5_991,
            "validation_frac_m2_gt_9_21": frac_9_21,
            "validation_frac_m2_gt_13_82": frac_13_82,
            "validation_expected_frac_gt_5_991": 0.05,
            "validation_expected_frac_gt_9_21": 0.01,
            "validation_expected_frac_gt_13_82": 0.001,
            "validation_mean_ratio": mean_ratio,
            "validation_median_ratio": median_ratio,
            "validation_p95_ratio": p95_ratio,
            "validation_p99_ratio": p99_ratio,
            "validation_quality": quality,
            "validation_message": (
                f"{process_name}: "
                f"N={n}, "
                f"mean={mean_m2:.6g}, "
                f"median={median_m2:.6g}, "
                f"P95={p95_m2:.6g}, "
                f"P99={p99_m2:.6g}, "
                f">5.991={frac_5_991:.6g}, "
                f">9.21={frac_9_21:.6g}, "
                f">13.82={frac_13_82:.6g}"
            ),
        }

    # ========================================================
    # Работа с одним train-файлом
    # ========================================================

    @staticmethod
    def train_one_file(
        train_path: Path,
        train_segment_sets: dict[str, list[TrackSegment]],
        validation_segment_sets: dict[str, list[TrackSegment]],
        validation_file_name: str,
        output_rows: list[dict],
    ) -> None:
        """
        Полностью калибрует один train-файл.

        sigma_meas оценивается один раз по stand,
        затем используется для CV и RW всех segment_type.
        """

        logging.info(
            "=================================================="
        )

        logging.info(
            "TRAIN experiment: %s",
            train_path.name,
        )

        # ====================================================
        # Этап 1. Оценка sigma_meas
        # ====================================================

        stand_segments = (
            train_segment_sets["stand"]
        )

        if not stand_segments:
            logging.error(
                "%s: отсутствуют подходящие stand-сегменты",
                train_path.name,
            )

            return

        sigma_result = (
            KalmanParameterEstimator
            .estimate_sigma_meas(
                stand_segments
            )
        )

        sigma_meas = float(
            sigma_result["sigma_meas"]
        )

        total_stand_points = sum(
            segment.size
            for segment in stand_segments
        )

        if not np.isfinite(sigma_meas):
            logging.error(
                "%s: sigma_meas не удалось оценить",
                train_path.name,
            )

            return

        logging.info(
            "%s: FIXED sigma_meas=%.8g",
            train_path.name,
            sigma_meas,
        )

        # ----------------------------------------------------
        # Строка результата sigma_meas
        # ----------------------------------------------------

        output_rows.append(
            {
                "train_file": train_path.name,
                "validation_file": validation_file_name,
                "segment_type": "stand",
                "method": "stationary_stand_diff",

                "n_segments": len(
                    stand_segments
                ),
                "n_points": total_stand_points,

                "sigma_meas_x": sigma_result[
                    "sigma_meas_x"
                ],
                "sigma_meas_y": sigma_result[
                    "sigma_meas_y"
                ],
                "sigma_meas": sigma_meas,

                "sigma_acc": np.nan,
                "sigma_rw": np.nan,

                "negative_log_likelihood": np.nan,
                "n_mle_windows": np.nan,
                "optimizer_success": True,
                "optimizer_message": (
                    "sigma_meas estimated "
                    "from stationary stand fragments"
                ),

                "stationary_fragments_total": (
                    sigma_result[
                        "fragment_count_total"
                    ]
                ),
                "stationary_fragments_accepted": (
                    sigma_result[
                        "fragment_count_accepted"
                    ]
                ),
            }
        )

        # ====================================================
        # Этап 2. MLE process noise
        # ====================================================

        for segment_type, current_segments in (
            train_segment_sets.items()
        ):
            if not current_segments:
                logging.warning(
                    "%s / %s: сегменты отсутствуют",
                    train_path.name,
                    segment_type,
                )

                continue

            total_points = sum(
                segment.size
                for segment in current_segments
            )

            logging.info(
                "%s / %s: MLE, "
                "segments=%d, "
                "points=%d, "
                "FIXED sigma_meas=%.8g",
                train_path.name,
                segment_type,
                len(current_segments),
                total_points,
                sigma_meas,
            )

            # =================================================
            # CV
            # =================================================

            cv_result = (
                KalmanParameterEstimator
                .fit_cv_mle(
                    segments=current_segments,
                    sigma_meas=sigma_meas,
                    sigma_acc_initial=(
                        DEFAULT_SIGMA_ACC_INITIAL
                    ),
                    window=WINDOW,
                )
            )

            sigma_acc = float(
                cv_result["sigma_acc"]
            )

            logging.info(
                "%s / %s / CV MLE: "
                "success=%s, "
                "sigma_meas=%.8g, "
                "sigma_acc=%.8g, "
                "NLL=%.8g",
                train_path.name,
                segment_type,
                cv_result["success"],
                sigma_meas,
                sigma_acc,
                cv_result[
                    "negative_log_likelihood"
                ],
            )

            cv_row = {
                "train_file": train_path.name,
                "validation_file": validation_file_name,
                "segment_type": segment_type,
                "method": "MLE_CV",

                "n_segments": len(
                    current_segments
                ),
                "n_points": total_points,

                "sigma_meas_x": sigma_result[
                    "sigma_meas_x"
                ],
                "sigma_meas_y": sigma_result[
                    "sigma_meas_y"
                ],
                "sigma_meas": sigma_meas,

                "sigma_acc": sigma_acc,
                "sigma_rw": np.nan,

                "negative_log_likelihood": (
                    cv_result[
                        "negative_log_likelihood"
                    ]
                ),
                "n_mle_windows": (
                    cv_result[
                        "n_mle_windows"
                    ]
                ),
                "optimizer_success": (
                    cv_result["success"]
                ),
                "optimizer_message": (
                    cv_result["message"]
                ),

                "stationary_fragments_total": (
                    sigma_result[
                        "fragment_count_total"
                    ]
                ),
                "stationary_fragments_accepted": (
                    sigma_result[
                        "fragment_count_accepted"
                    ]
                ),
            }

            # -------------------------------------------------
            # Validation CV
            # -------------------------------------------------

            validation_segments = (
                validation_segment_sets[
                    segment_type
                ]
            )

            if (
                validation_segments
                and cv_result["success"]
            ):
                logging.info(
                    "%s / %s / CV: "
                    "validation на %s",
                    train_path.name,
                    segment_type,
                    validation_file_name,
                )

                validation_metrics = (
                    KalmanParameterEstimator
                    .collect_cv_likelihood(
                        segments=validation_segments,
                        sigma_acc=sigma_acc,
                        sigma_meas=sigma_meas,
                        window=WINDOW,
                    )
                )

                validation = (
                    KalmanParameterEstimator
                    .validate_model(
                        mahalanobis=(
                            validation_metrics[
                                "mahalanobis"
                            ]
                        ),
                        optimizer_success=bool(
                            cv_result["success"]
                        ),
                        process_name="CV",
                    )
                )

                cv_row.update(
                    {
                        "validation_n_segments": len(
                            validation_segments
                        ),
                        "validation_n_points": sum(
                            segment.size
                            for segment in validation_segments
                        ),
                    }
                )

                cv_row.update(
                    validation
                )

                logging.info(
                    "%s / %s / CV validation: %s",
                    train_path.name,
                    segment_type,
                    validation[
                        "validation_message"
                    ],
                )

            output_rows.append(
                cv_row
            )

            # =================================================
            # RW
            # =================================================

            sigma_rw_initial = max(
                DEFAULT_SIGMA_RW_INITIAL,
                SIGMA_RW_MIN,
            )

            rw_result = (
                KalmanParameterEstimator
                .fit_rw_mle(
                    segments=current_segments,
                    sigma_meas=sigma_meas,
                    sigma_rw_initial=sigma_rw_initial,
                    window=WINDOW,
                )
            )

            sigma_rw = float(
                rw_result["sigma_rw"]
            )

            logging.info(
                "%s / %s / RW MLE: "
                "success=%s, "
                "sigma_meas=%.8g, "
                "sigma_rw=%.8g, "
                "NLL=%.8g",
                train_path.name,
                segment_type,
                rw_result["success"],
                sigma_meas,
                sigma_rw,
                rw_result[
                    "negative_log_likelihood"
                ],
            )

            rw_row = {
                "train_file": train_path.name,
                "validation_file": validation_file_name,
                "segment_type": segment_type,
                "method": "MLE_RW",

                "n_segments": len(
                    current_segments
                ),
                "n_points": total_points,

                "sigma_meas_x": sigma_result[
                    "sigma_meas_x"
                ],
                "sigma_meas_y": sigma_result[
                    "sigma_meas_y"
                ],
                "sigma_meas": sigma_meas,

                "sigma_acc": np.nan,
                "sigma_rw": sigma_rw,

                "negative_log_likelihood": (
                    rw_result[
                        "negative_log_likelihood"
                    ]
                ),
                "n_mle_windows": (
                    rw_result[
                        "n_mle_windows"
                    ]
                ),
                "optimizer_success": (
                    rw_result["success"]
                ),
                "optimizer_message": (
                    rw_result["message"]
                ),

                "stationary_fragments_total": (
                    sigma_result[
                        "fragment_count_total"
                    ]
                ),
                "stationary_fragments_accepted": (
                    sigma_result[
                        "fragment_count_accepted"
                    ]
                ),
            }

            # -------------------------------------------------
            # Validation RW
            # -------------------------------------------------

            if (
                validation_segments
                and rw_result["success"]
            ):
                logging.info(
                    "%s / %s / RW: "
                    "validation на %s",
                    train_path.name,
                    segment_type,
                    validation_file_name,
                )

                validation_metrics = (
                    KalmanParameterEstimator
                    .collect_rw_likelihood(
                        segments=validation_segments,
                        sigma_rw=sigma_rw,
                        sigma_meas=sigma_meas,
                        window=WINDOW,
                    )
                )

                validation = (
                    KalmanParameterEstimator
                    .validate_model(
                        mahalanobis=(
                            validation_metrics[
                                "mahalanobis"
                            ]
                        ),
                        optimizer_success=bool(
                            rw_result["success"]
                        ),
                        process_name="RW",
                    )
                )

                rw_row.update(
                    {
                        "validation_n_segments": len(
                            validation_segments
                        ),
                        "validation_n_points": sum(
                            segment.size
                            for segment in validation_segments
                        ),
                    }
                )

                rw_row.update(
                    validation
                )

                logging.info(
                    "%s / %s / RW validation: %s",
                    train_path.name,
                    segment_type,
                    validation[
                        "validation_message"
                    ],
                )

            output_rows.append(
                rw_row
            )

        logging.info(
            "TRAIN experiment %s завершён",
            train_path.name,
        )


# ============================================================
# Загрузка файла
# ============================================================

def load_file(
    path: Path,
) -> dict[str, list[TrackSegment]]:
    """Загружает и разбивает файл на segment sets."""

    logging.info(
        "Загрузка файла: %s",
        path.name,
    )

    df = DataProcessor.load_csv(
        path
    )

    logging.info(
        "%s: исходных точек=%d",
        path.name,
        len(df),
    )

    df = DataProcessor.pre_filter(
        df
    )

    logging.info(
        "%s: после pre_filter=%d",
        path.name,
        len(df),
    )

    segments = (
        KalmanParameterEstimator
        .extract_segments(
            df=df,
            file_name=path.name,
            min_length=MIN_SEGMENT_LENGTH,
        )
    )

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

    for segment_type, current_segments in (
        segment_sets.items()
    ):
        logging.info(
            "%s / %s: segments=%d, points=%d",
            path.name,
            segment_type,
            len(current_segments),
            sum(
                segment.size
                for segment in current_segments
            ),
        )

    return segment_sets


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    path_root = Path(__file__).parent

    train_paths = [
        path_root / "data" / "1.csv",
        path_root / "data" / "2.csv",
        path_root / "data" / "3.csv"
    ]

    validation_path = (
        path_root
        / "data"
        / "3.csv"
    )

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
        "=================================================="
    )

    logging.info(
        "Начало калибровки Kalman CV/RW"
    )

    logging.info(
        "Train files: %s",
        ", ".join(
            path.name
            for path in train_paths
        ),
    )

    logging.info(
        "Validation file: %s",
        validation_path.name,
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
        "STATIONARY_FRAGMENT_LENGTH=%d",
        STATIONARY_FRAGMENT_LENGTH,
    )

    logging.info(
        "STATIONARITY_Z=%.3f",
        STATIONARITY_Z,
    )

    logging.info(
        "BATCH_SIZE=%d",
        BATCH_SIZE,
    )

    # ========================================================
    # Загружаем validation один раз
    # ========================================================

    validation_segment_sets = load_file(
        validation_path
    )

    # ========================================================
    # Каждый train-файл независимо
    # ========================================================

    all_results: list[dict] = []

    for train_path in train_paths:
        train_segment_sets = load_file(
            train_path
        )

        KalmanParameterEstimator.train_one_file(
            train_path=train_path,
            train_segment_sets=train_segment_sets,
            validation_segment_sets=validation_segment_sets,
            validation_file_name=validation_path.name,
            output_rows=all_results,
        )

    # ========================================================
    # CSV
    # ========================================================

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

    # ========================================================
    # Печать
    # ========================================================

    if result_df.empty:
        logging.warning(
            "Итоговый CSV пуст"
        )

    else:
        columns_to_print = [
            "train_file",
            "validation_file",
            "segment_type",
            "method",
            "n_segments",
            "n_points",
            "sigma_meas_x",
            "sigma_meas_y",
            "sigma_meas",
            "sigma_acc",
            "sigma_rw",
            "negative_log_likelihood",
            "n_mle_windows",
            "optimizer_success",
            "stationary_fragments_total",
            "stationary_fragments_accepted",
            "validation_n",
            "validation_mean_mahalanobis",
            "validation_median_mahalanobis",
            "validation_p95_mahalanobis",
            "validation_p99_mahalanobis",
            "validation_frac_m2_gt_5_991",
            "validation_frac_m2_gt_9_21",
            "validation_frac_m2_gt_13_82",
            "validation_mean_ratio",
            "validation_median_ratio",
            "validation_p95_ratio",
            "validation_p99_ratio",
            "validation_quality",
        ]

        existing_columns = [
            column
            for column in columns_to_print
            if column in result_df.columns
        ]

        print(
            "\nИтоговая таблица:\n"
        )

        print(
            result_df[
                existing_columns
            ].to_string(
                index=False
            )
        )

    logging.info(
        "Калибровка завершена"
    )
