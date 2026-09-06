from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV
from app.working.kalman_filter_rw import KalmanFilterRW


LOGGER = logging.getLogger(__name__)


# ======================================================================
# КОНФИГУРАЦИЯ
# ======================================================================

WINDOW = 10

# Полный эксперимент.
POINT_SLICE = slice(None)

# Для быстрого тестирования можно временно использовать:
# POINT_SLICE = slice(120_000, 130_000)

BINS = 2000
SMOOTHING_SIGMA = 2.0
PLOT_DPI = 600

# Ограничивает только отображаемую область ECDF.
# Исходные значения при этом полностью участвуют
# в статистических расчётах.
DISPLAY_PERCENTILE = 95.0

# Сохранять ли point-level признаки.
# При полном объёме данных CSV может быть очень большим.
SAVE_POINT_FEATURES = False

# Количество проверяемых квантильных порогов
# при подборе threshold classifier.
THRESHOLD_GRID_SIZE = 1001

# Классы целевой классификации.
STATUS_VALUES = (
    "stand",
    "move",
    "anomaly",
)


# ======================================================================
# ПАРАМЕТРЫ МОДЕЛЕЙ
# ======================================================================


@dataclass(frozen=True)
class FilterParameters:
    """
    Параметры конкретной модели фильтра.

    sigma_meas:
        СКО шума измерений, м.

    process_noise:
        Параметр шума процесса.

        Для CV:
            process_noise = sigma_acc, м/с².

        Для RW:
            process_noise = sigma_rw, м/sqrt(с).
    """

    sigma_meas: float
    process_noise: float


@dataclass(frozen=True)
class ParameterProfile:
    """
    Набор параметров, используемый для одного эксперимента.

    Один профиль содержит отдельные параметры CV и RW.
    """

    name: str
    description: str
    cv: FilterParameters
    rw: FilterParameters


# ----------------------------------------------------------------------
# Параметры, полученные на предыдущем этапе калибровки.
#
# ВАЖНО:
#   sigma_acc используется только CV.
#   sigma_rw используется только RW.
#
# Здесь они хранятся отдельно, несмотря на общий термин
# process_noise внутри FilterParameters.
# ----------------------------------------------------------------------

KNOWN_PROFILE_FROM_1 = {
    "sigma_meas": 0.658406,
    "sigma_acc_cv": 0.070720,
    "sigma_rw": 6.211997,
}

KNOWN_PROFILE_FROM_2 = {
    "sigma_meas": 1.221545,
    "sigma_acc_cv": 0.047042,
    "sigma_rw": 9.313700,
}


# ======================================================================
# COLLECT STATISTICS
# ======================================================================


class CollectStatistics:
    """
    Сбор статистик распределений и статистик фильтров Калмана.

    Для CV используется модель постоянной скорости.

    Для RW используется модель случайного блуждания без скорости.

    Целевая метка status используется только после расчёта
    статистики фильтра и не передаётся самому фильтру.
    """

    SCHEME_LABELS = {
        "3_status": "3 класса: stand / move / anomaly",
        "2_status": "2 класса: stand+move / anomaly",
    }

    STATUS_LABELS = {
        "anomaly": "аномалия",
        "move": "движение",
        "stand": "стоянка",
        "stand_move": "движение и стоянка",
    }

    KIND_LABELS = {
        "distances": (
            "Расстояние между соседними точками, м",
            "Расстояние, м",
        ),
        "filter_distance": (
            "Расстояние между измерением и оценкой фильтра, м",
            "Расстояние, м",
        ),
        "log_likelihood": (
            "Логарифм правдоподобия",
            "Логарифм правдоподобия",
        ),
        "mahalanobis": (
            "Квадрат расстояния Махаланобиса",
            "Квадрат расстояния Махаланобиса",
        ),
        "sqrt_mahalanobis": (
            "Корень из квадрата расстояния Махаланобиса",
            "√M²",
        ),
        "log1p_filter_distance": (
            "log(1 + расстояние между измерением и фильтром)",
            "log(1 + расстояние), м",
        ),
    }

    MODEL_LABELS = {
        "CV": "модель постоянной скорости (CV)",
        "RW": "модель случайного блуждания (RW)",
    }

    # ------------------------------------------------------------------
    # ОБЩИЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------

    @staticmethod
    def combine_stand_and_move(
        *arrays: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Объединяет статистики классов stand и move.

        Используется при переходе от трёхклассовой задачи
        к бинарной:

            stand + move -> normal
            anomaly      -> anomaly
        """

        parts = [
            np.asarray(
                array,
                dtype=np.float64,
            ).reshape(-1)
            for array in arrays
        ]

        if not parts:
            return np.empty(
                0,
                dtype=np.float64,
            )

        return np.concatenate(parts)

    @staticmethod
    def _sample_std(
        values: NDArray[np.float64],
    ) -> float:
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size < 2:
            return float("nan")

        return float(
            np.std(
                values,
                ddof=1,
            )
        )

    @staticmethod
    def _linear_residuals(
        values: NDArray[np.float64],
        time_s: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Остатки линейной регрессии:

            value = a + b * t

        То есть остатки относительно модели
        постоянной скорости.
        """

        design = np.column_stack(
            (
                np.ones(
                    len(time_s),
                    dtype=np.float64,
                ),
                time_s,
            )
        )

        try:
            coefficients, _, _, _ = np.linalg.lstsq(
                design,
                values,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return (
                values
                - np.mean(values)
            )

        return (
            values
            - design @ coefficients
        )

    # ------------------------------------------------------------------
    # DISTANCE
    # ------------------------------------------------------------------

    @staticmethod
    def collect_distance_between_point(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        mark: NDArray[np.str_],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Собирает расстояния между соседними точками.

        Для P[i] -> P[i+1] локальная система координат
        определяется относительно P[i-10].

        Классификация интервала выполняется с приоритетом:

            anomaly > move > stand.
        """

        if not (
            len(lon)
            == len(lat)
            == len(mark)
        ):
            raise ValueError(
                "lon, lat и mark должны иметь одинаковую длину"
            )

        if len(lon) <= 11:
            empty = np.empty(
                0,
                dtype=np.float64,
            )

            return (
                empty,
                empty,
                empty,
            )

        origin_lat = lat[:-11]

        kx = (
            DataProcessor.LEN_LAT
            * np.cos(
                np.radians(origin_lat)
            )
        )

        delta_lon = (
            lon[11:]
            - lon[10:-1]
        )

        delta_lat = (
            lat[11:]
            - lat[10:-1]
        )

        dx = (
            delta_lon
            * kx
        )

        dy = (
            delta_lat
            * DataProcessor.LEN_LAT
        )

        distance = np.hypot(
            dx,
            dy,
        )

        mark_left = mark[10:-1]
        mark_right = mark[11:]

        anomaly_mask = (
            (mark_left == "anomaly")
            | (mark_right == "anomaly")
        )

        move_mask = (
            ~anomaly_mask
            & (
                (mark_left == "move")
                | (mark_right == "move")
            )
        )

        stand_mask = (
            ~anomaly_mask
            & ~move_mask
            & (mark_left == "stand")
            & (mark_right == "stand")
        )

        return (
            distance[anomaly_mask],
            distance[move_mask],
            distance[stand_mask],
        )

    # ------------------------------------------------------------------
    # КАЛМАН
    # ------------------------------------------------------------------

    @staticmethod
    def _create_filter(
        model: str,
        parameters: FilterParameters,
    ):
        """
        Создаёт экземпляр фильтра с параметрами соответствующей модели.

        CV:
            sigma_meas -> СКО измерения
            process_noise -> sigma_acc

        RW:
            sigma_meas -> СКО измерения
            process_noise -> sigma_rw
        """

        if model == "CV":
            return KalmanFilterCV(
                sigma_meas=parameters.sigma_meas,
                sigma_acc=parameters.process_noise,
            )

        if model == "RW":
            return KalmanFilterRW(
                sigma_meas=parameters.sigma_meas,
                sigma_rw=parameters.process_noise,
            )

        raise ValueError(
            f"Неизвестная модель: {model}"
        )

    @staticmethod
    def collect_kalman_metrics(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        time: NDArray[np.datetime64],
        mark: NDArray[np.str_],
        model: str,
        parameters: FilterParameters,
        window: int = 10,
    ) -> dict[
        str,
        NDArray[np.float64],
    ]:
        """
        Рассчитывает point-level статистики фильтра Калмана.

        Для точки P[i] используется окно:

            P[i-window] ... P[i]

        Последнее измерение окна является P[i].

        ВАЖНО:
            mark[i] не участвует в расчёте фильтра.
            Он используется исключительно как истинная
            целевая метка после получения статистики.

        Это предотвращает прямую утечку target -> feature.
        """

        n = len(lon)

        if not (
            len(lat)
            == n
            and len(time)
            == n
            and len(mark)
            == n
        ):
            raise ValueError(
                "lon, lat, time и mark должны иметь одинаковую длину"
            )

        result = {
            "log_likelihood": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
            "mahalanobis": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
            "filter_distance": np.full(
                n,
                np.nan,
                dtype=np.float64,
            ),
        }

        if n <= window:
            return result

        for i in range(
            window,
            n,
        ):
            lon_window = lon[
                i - window : i + 1
            ]

            lat_window = lat[
                i - window : i + 1
            ]

            time_window = time[
                i - window : i + 1
            ]

            # Все точки окна переводятся
            # в единую локальную систему координат.
            x_window, y_window = (
                DataProcessor
                .convert_to_local_cartesian(
                    lon_window,
                    lat_window,
                )
            )

            kalman_filter = (
                CollectStatistics
                ._create_filter(
                    model=model,
                    parameters=parameters,
                )
            )

            (
                filtered_x,
                filtered_y,
                log_likelihood,
                mahalanobis_sq,
            ) = kalman_filter.filter(
                x_window,
                y_window,
                time_window,
            )

            # Последнее значение относится
            # непосредственно к P[i].
            result[
                "log_likelihood"
            ][i] = float(
                log_likelihood[-1]
            )

            result[
                "mahalanobis"
            ][i] = float(
                mahalanobis_sq[-1]
            )

            result[
                "filter_distance"
            ][i] = float(
                np.hypot(
                    filtered_x[-1]
                    - x_window[-1],
                    filtered_y[-1]
                    - y_window[-1],
                )
            )

        return result

    # ------------------------------------------------------------------
    # ОЦЕНКА ПАРАМЕТРОВ ШУМА
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_kalman_noise_parameters(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        time: NDArray[np.datetime64],
    ) -> dict[str, float]:
        """
        Оценивает параметры шума по отдельному набору точек.

        sigma_meas:
            Оценка СКО измерительного шума.

        sigma_acc_cv:
            Оценка СКО ускорения для CV.
            Единицы: м/с².

        sigma_acc_rw:
            Интенсивность случайного блуждания для RW.
            Единицы: м/sqrt(с).

        Метод используется для диагностической
        self-calibration 3.csv.
        """

        empty = {
            "n_points": float(
                len(lon)
            ),
            "duration_s": float(
                "nan"
            ),
            "sigma_meas": float(
                "nan"
            ),
            "variance_meas": float(
                "nan"
            ),
            "sigma_acc_cv": float(
                "nan"
            ),
            "variance_acc_cv": float(
                "nan"
            ),
            "sigma_acc_rw": float(
                "nan"
            ),
            "variance_acc_rw": float(
                "nan"
            ),
        }

        if not (
            len(lon)
            == len(lat)
            == len(time)
        ) or len(lon) < 3:
            return empty

        x_local, y_local = (
            DataProcessor
            .convert_to_local_cartesian(
                lon,
                lat,
            )
        )

        time = np.asarray(
            time,
            dtype="datetime64[ns]",
        )

        time_s = (
            time - time[0]
        ) / np.timedelta64(
            1,
            "s",
        )

        time_s = np.asarray(
            time_s,
            dtype=np.float64,
        )

        duration_s = (
            float(
                time_s[-1]
                - time_s[0]
            )
            if time_s.size
            else float("nan")
        )

        empty[
            "duration_s"
        ] = duration_s

        residual_x = (
            CollectStatistics
            ._linear_residuals(
                x_local,
                time_s,
            )
        )

        residual_y = (
            CollectStatistics
            ._linear_residuals(
                y_local,
                time_s,
            )
        )

        sigma_meas_x = (
            CollectStatistics
            ._sample_std(
                residual_x
            )
        )

        sigma_meas_y = (
            CollectStatistics
            ._sample_std(
                residual_y
            )
        )

        if (
            np.isfinite(
                sigma_meas_x
            )
            and np.isfinite(
                sigma_meas_y
            )
        ):
            sigma_meas = float(
                np.sqrt(
                    0.5
                    * (
                        sigma_meas_x**2
                        + sigma_meas_y**2
                    )
                )
            )
        else:
            sigma_meas = float(
                "nan"
            )

        dt = np.diff(
            time_s
        )

        valid_dt = (
            dt > 1e-6
        )

        sigma_acc_cv = float(
            "nan"
        )

        sigma_acc_rw = float(
            "nan"
        )

        if np.count_nonzero(
            valid_dt
        ) >= 2:

            vx = (
                np.diff(
                    x_local
                )[valid_dt]
                / dt[valid_dt]
            )

            vy = (
                np.diff(
                    y_local
                )[valid_dt]
                / dt[valid_dt]
            )

            dt_valid = dt[
                valid_dt
            ]

            dt_acc = (
                0.5
                * (
                    dt_valid[:-1]
                    + dt_valid[1:]
                )
            )

            valid_acc = (
                dt_acc > 1e-6
            )

            if np.count_nonzero(
                valid_acc
            ) >= 2:

                ax = (
                    vx[1:]
                    - vx[:-1]
                ) / dt_acc

                ay = (
                    vy[1:]
                    - vy[:-1]
                ) / dt_acc

                ax = ax[
                    valid_acc
                ]

                ay = ay[
                    valid_acc
                ]

                sigma_ax = (
                    CollectStatistics
                    ._sample_std(
                        ax
                    )
                )

                sigma_ay = (
                    CollectStatistics
                    ._sample_std(
                        ay
                    )
                )

                if (
                    np.isfinite(
                        sigma_ax
                    )
                    and np.isfinite(
                        sigma_ay
                    )
                ):
                    sigma_acc_cv = float(
                        np.sqrt(
                            0.5
                            * (
                                sigma_ax**2
                                + sigma_ay**2
                            )
                        )
                    )

            residual_dx = (
                np.diff(
                    residual_x
                )[valid_dt]
            )

            residual_dy = (
                np.diff(
                    residual_y
                )[valid_dt]
            )

            intensity = (
                residual_dx**2
                + residual_dy**2
            ) / (
                2.0 * dt_valid
            )

            intensity = intensity[
                np.isfinite(
                    intensity
                )
                & (
                    intensity
                    >= 0.0
                )
            ]

            if intensity.size >= 2:
                sigma_acc_rw = float(
                    np.sqrt(
                        np.mean(
                            intensity
                        )
                    )
                )

        return {
            "n_points": float(
                len(lon)
            ),
            "duration_s": duration_s,
            "sigma_meas": sigma_meas,
            "variance_meas": (
                sigma_meas**2
                if np.isfinite(
                    sigma_meas
                )
                else float("nan")
            ),
            "sigma_acc_cv": sigma_acc_cv,
            "variance_acc_cv": (
                sigma_acc_cv**2
                if np.isfinite(
                    sigma_acc_cv
                )
                else float("nan")
            ),
            "sigma_acc_rw": sigma_acc_rw,
            "variance_acc_rw": (
                sigma_acc_rw**2
                if np.isfinite(
                    sigma_acc_rw
                )
                else float("nan")
            ),
        }

    # ------------------------------------------------------------------
    # SELF-CALIBRATED PROFILE 3.csv
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_profile_from_statuses(
        df: pd.DataFrame,
        sigma_meas: float,
    ) -> ParameterProfile:
        """
        Формирует диагностический профиль параметров для 3.csv.

        sigma_meas передаётся явно.

        sigma_acc для CV и sigma_rw для RW
        оцениваются отдельно по stand и move,
        после чего берётся медиана.

        Этот профиль НЕ является независимой validation-калибровкой,
        поскольку при его построении используются данные 3.csv.
        """

        status_parameters = {}

        for status in (
            "stand",
            "move",
        ):
            status_df = df[
                df["status"] == status
            ]

            if len(status_df) < 3:
                continue

            lon, lat, time = (
                DataProcessor
                .get_lon_lat(
                    status_df
                )
            )

            params = (
                CollectStatistics
                .estimate_kalman_noise_parameters(
                    lon,
                    lat,
                    time,
                )
            )

            status_parameters[
                status
            ] = params

        sigma_acc_values = []
        sigma_rw_values = []

        for params in (
            status_parameters.values()
        ):
            if np.isfinite(
                params["sigma_acc_cv"]
            ):
                sigma_acc_values.append(
                    params["sigma_acc_cv"]
                )

            if np.isfinite(
                params["sigma_acc_rw"]
            ):
                sigma_rw_values.append(
                    params["sigma_acc_rw"]
                )

        if not sigma_acc_values:
            raise RuntimeError(
                "Не удалось оценить "
                "sigma_acc для 3.csv"
            )

        if not sigma_rw_values:
            raise RuntimeError(
                "Не удалось оценить "
                "sigma_rw для 3.csv"
            )

        sigma_acc_cv = float(
            np.median(
                sigma_acc_values
            )
        )

        sigma_rw = float(
            np.median(
                sigma_rw_values
            )
        )

        LOGGER.info(
            "3.csv self-calibrated parameters: "
            "sigma_acc(CV)=%.8f, "
            "sigma_rw(RW)=%.8f, "
            "sigma_meas=%.8f",
            sigma_acc_cv,
            sigma_rw,
            sigma_meas,
        )

        return ParameterProfile(
            name="from_3",
            description=(
                "Диагностический self-calibrated "
                "профиль: sigma_meas и process-noise "
                "оценены непосредственно по 3.csv"
            ),
            cv=FilterParameters(
                sigma_meas=sigma_meas,
                process_noise=sigma_acc_cv,
            ),
            rw=FilterParameters(
                sigma_meas=sigma_meas,
                process_noise=sigma_rw,
            ),
        )

    # ------------------------------------------------------------------
    # DISTRIBUTIONS
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_distribution(
        values: NDArray[np.float64],
        bins: int = 2000,
        log_scale: bool = False,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:

        values = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            empty = np.empty(
                0,
                dtype=np.float64,
            )

            return (
                empty,
                empty,
                empty,
            )

        value_min = float(
            np.min(values)
        )

        value_max = float(
            np.max(values)
        )

        if value_min == value_max:
            width = max(
                abs(value_min) * 0.01,
                1.0,
            )

            edges = np.array(
                [
                    value_min - width,
                    value_max + width,
                ],
                dtype=np.float64,
            )

            counts = np.array(
                [values.size],
                dtype=np.float64,
            )

            density = (
                counts
                / (
                    values.size
                    * np.diff(edges)
                )
            )

            centers = (
                edges[:-1]
                + edges[1:]
            ) / 2.0

            return (
                centers,
                density,
                counts,
            )

        if not log_scale:
            edges = np.linspace(
                value_min,
                value_max,
                bins + 1,
                dtype=np.float64,
            )
        else:
            if value_min <= 0.0:
                raise ValueError(
                    "log_scale=True требует "
                    "строго положительных значений"
                )

            edges = np.geomspace(
                value_min,
                value_max,
                bins + 1,
                dtype=np.float64,
            )

        counts, edges = np.histogram(
            values,
            bins=edges,
        )

        counts = counts.astype(
            np.float64
        )

        widths = np.diff(
            edges
        )

        density = (
            counts
            / (
                values.size
                * widths
            )
        )

        centers = (
            edges[:-1]
            + edges[1:]
        ) / 2.0

        return (
            centers,
            density,
            counts,
        )

    @staticmethod
    def _build_cdf(
        values: NDArray[np.float64],
        bins: int = 2000,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:

        values = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            empty = np.empty(
                0,
                dtype=np.float64,
            )

            return (
                empty,
                empty,
            )

        sorted_values = np.sort(
            values
        )

        n = sorted_values.size

        sample_count = min(
            max(
                int(bins),
                10,
            ),
            n,
        )

        idx = np.unique(
            np.round(
                np.linspace(
                    0,
                    n - 1,
                    sample_count,
                )
            ).astype(
                np.int64
            )
        )

        x = sorted_values[
            idx
        ]

        cdf = (
            idx + 1
        ).astype(
            np.float64
        ) / n

        x = np.concatenate(
            [
                [sorted_values[0]],
                x,
                [sorted_values[-1]],
            ]
        )

        cdf = np.concatenate(
            [
                [0.0],
                cdf,
                [1.0],
            ]
        )

        return (
            x,
            cdf,
        )

    @staticmethod
    def _get_display_limits(
        values: NDArray[np.float64],
        display_percentile: float = 95.0,
    ) -> Tuple[
        float,
        float,
    ]:

        values = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            raise ValueError(
                "Пустой массив"
            )

        if not (
            0.0
            < display_percentile
            <= 100.0
        ):
            raise ValueError(
                "display_percentile должен "
                "находиться в диапазоне "
                "(0, 100]"
            )

        tail = (
            100.0
            - display_percentile
        ) / 2.0

        lower = float(
            np.percentile(
                values,
                tail,
            )
        )

        upper = float(
            np.percentile(
                values,
                100.0 - tail,
            )
        )

        if lower == upper:
            padding = max(
                abs(lower) * 0.01,
                1.0,
            )

            lower -= padding
            upper += padding

        return (
            lower,
            upper,
        )

    # ------------------------------------------------------------------
    # LABELS
    # ------------------------------------------------------------------

    @staticmethod
    def make_metric_key(
        scheme: str,
        status: str,
        kind: str,
        model: str | None = None,
    ) -> str:
        parts = [
            scheme,
            status,
            kind,
        ]

        if model is not None:
            parts.append(
                model
            )

        return "__".join(
            parts
        )

    @staticmethod
    def _split_metric_key(
        metric_name: str,
    ) -> Tuple[
        str,
        str,
        str,
        str | None,
    ]:

        parts = metric_name.split(
            "__"
        )

        if len(parts) == 3:
            scheme, status, kind = (
                parts
            )

            return (
                scheme,
                status,
                kind,
                None,
            )

        if len(parts) == 4:
            (
                scheme,
                status,
                kind,
                model,
            ) = parts

            return (
                scheme,
                status,
                kind,
                model,
            )

        raise ValueError(
            f"Некорректное имя метрики: "
            f"{metric_name}"
        )

    @staticmethod
    def _metric_labels(
        metric_name: str,
    ) -> Tuple[
        str,
        str,
        str,
        str | None,
        str,
        str,
        str,
    ]:

        (
            scheme,
            status,
            kind,
            model,
        ) = (
            CollectStatistics
            ._split_metric_key(
                metric_name
            )
        )

        title, xlabel = (
            CollectStatistics
            .KIND_LABELS[kind]
        )

        scheme_label = (
            CollectStatistics
            .SCHEME_LABELS
            .get(
                scheme,
                scheme,
            )
        )

        status_label = (
            CollectStatistics
            .STATUS_LABELS
            .get(
                status,
                status,
            )
        )

        context_parts = [
            scheme_label,
            status_label,
        ]

        if model is not None:
            context_parts.append(
                CollectStatistics
                .MODEL_LABELS
                .get(
                    model,
                    model,
                )
            )

        context = ", ".join(
            context_parts
        )

        return (
            scheme,
            status,
            kind,
            model,
            title,
            xlabel,
            context,
        )

    @staticmethod
    def _metric_output_path(
        output_dir: Path,
        metric_name: str,
    ) -> Path:

        (
            scheme,
            status,
            kind,
            model,
        ) = (
            CollectStatistics
            ._split_metric_key(
                metric_name
            )
        )

        folder = (
            output_dir
            / scheme
            / status
            / kind
        )

        filename = (
            f"{model}.png"
            if model is not None
            else f"{kind}.png"
        )

        return (
            folder
            / filename
        )

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def visualize_statistics(
        statistics: Mapping[
            str,
            NDArray[np.float64]
            | list
            | tuple,
        ],
        output_dir: Path,
        bins: int = 2000,
        smoothing_sigma: float = 2.0,
        dpi: int = 600,
        figsize: Tuple[
            float,
            float,
        ] = (
            18.0,
            7.5,
        ),
        display_percentile: float = 95.0,
    ) -> pd.DataFrame:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_rows = []

        for (
            metric_name,
            raw_values,
        ) in statistics.items():

            values = np.asarray(
                raw_values,
                dtype=np.float64,
            ).reshape(-1)

            values = values[
                np.isfinite(values)
            ]

            (
                scheme,
                status,
                kind,
                model,
                title,
                xlabel,
                context,
            ) = (
                CollectStatistics
                ._metric_labels(
                    metric_name
                )
            )

            if values.size == 0:

                summary_rows.append(
                    {
                        "metric": metric_name,
                        "title": title,
                        "scheme": scheme,
                        "status": status,
                        "kind": kind,
                        "model": (
                            model
                            if model is not None
                            else ""
                        ),
                        "count": 0,
                        "mean": np.nan,
                        "variance": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "p05": np.nan,
                        "p25": np.nan,
                        "p50": np.nan,
                        "p75": np.nan,
                        "p95": np.nan,
                        "max": np.nan,
                        "display_percentile": (
                            display_percentile
                        ),
                        "display_min": np.nan,
                        "display_max": np.nan,
                        "file": "",
                    }
                )

                continue

            mean = float(
                np.mean(values)
            )

            variance = float(
                np.var(values)
            )

            std = float(
                np.std(values)
            )

            minimum = float(
                np.min(values)
            )

            maximum = float(
                np.max(values)
            )

            (
                p05,
                p25,
                p50,
                p75,
                p95,
            ) = np.percentile(
                values,
                [
                    5,
                    25,
                    50,
                    75,
                    95,
                ],
            )

            (
                display_min,
                display_max,
            ) = (
                CollectStatistics
                ._get_display_limits(
                    values,
                    display_percentile,
                )
            )

            summary_rows.append(
                {
                    "metric": metric_name,
                    "title": title,
                    "scheme": scheme,
                    "status": status,
                    "kind": kind,
                    "model": (
                        model
                        if model is not None
                        else ""
                    ),
                    "count": int(
                        values.size
                    ),
                    "mean": mean,
                    "variance": variance,
                    "std": std,
                    "min": minimum,
                    "p05": float(p05),
                    "p25": float(p25),
                    "p50": float(p50),
                    "p75": float(p75),
                    "p95": float(p95),
                    "max": maximum,
                    "display_percentile": (
                        display_percentile
                    ),
                    "display_min": display_min,
                    "display_max": display_max,
                    "file": "",
                }
            )

            x_cdf, y_cdf = (
                CollectStatistics
                ._build_cdf(
                    values,
                    bins=bins,
                )
            )

            strictly_positive = bool(
                np.all(
                    values > 0.0
                )
            )

            fig, axes = plt.subplots(
                1,
                2,
                figsize=figsize,
            )

            fig.suptitle(
                f"{title}\n{context}",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            fig.text(
                0.5,
                0.90,
                (
                    f"{context} | "
                    f"Мат. ожидание = "
                    f"{mean:.8g} | "
                    f"Дисперсия = "
                    f"{variance:.8g} | "
                    f"N = {values.size:,} | "
                    f"Отображение = "
                    f"{display_percentile:g}%"
                ),
                ha="center",
                va="center",
                fontsize=11,
            )

            percentile_data = (
                (
                    float(p05),
                    "P5",
                    0.05,
                ),
                (
                    float(p25),
                    "P25",
                    0.25,
                ),
                (
                    float(p50),
                    "P50",
                    0.50,
                ),
                (
                    float(p75),
                    "P75",
                    0.75,
                ),
                (
                    float(p95),
                    "P95",
                    0.95,
                ),
            )

            def draw_cdf(
                ax,
                logarithmic_x: bool,
            ) -> None:

                ax.plot(
                    x_cdf,
                    y_cdf,
                    color="#222222",
                    linewidth=2.0,
                    label=(
                        "Функция "
                        "распределения F(x)"
                    ),
                )

                for (
                    percentile_value,
                    label,
                    probability,
                ) in percentile_data:

                    ax.axvline(
                        percentile_value,
                        linestyle="--",
                        linewidth=1.4,
                        label=(
                            f"{label} = "
                            f"{percentile_value:.6g}"
                        ),
                    )

                    ax.axhline(
                        probability,
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )

                ax.set_ylim(
                    -0.02,
                    1.05,
                )

                ax.set_ylabel(
                    "Функция распределения "
                    "F(x) = P(X ≤ x)",
                    fontsize=11,
                )

                ax.set_xlabel(
                    xlabel,
                    fontsize=11,
                )

                ax.set_xlim(
                    display_min,
                    display_max,
                )

                ax.grid(
                    True,
                    which="both",
                    alpha=0.25,
                )

                ax.legend(
                    fontsize=8,
                    loc="lower right",
                )

                if logarithmic_x:
                    ax.set_xscale(
                        "log"
                    )

                    ax.set_title(
                        "Функция распределения — "
                        "логарифмический масштаб X",
                        fontsize=13,
                    )

                else:
                    ax.set_title(
                        "Функция распределения — "
                        "линейный масштаб X",
                        fontsize=13,
                    )

            draw_cdf(
                axes[0],
                logarithmic_x=False,
            )

            if strictly_positive:

                draw_cdf(
                    axes[1],
                    logarithmic_x=True,
                )

            else:

                draw_cdf(
                    axes[1],
                    logarithmic_x=False,
                )

                scale = max(
                    abs(p50) * 1e-3,
                    1e-9,
                )

                axes[1].set_xscale(
                    "symlog",
                    linthresh=scale,
                )

                axes[1].set_title(
                    "Функция распределения — "
                    "симметричный "
                    "логарифмический масштаб X",
                    fontsize=13,
                )

            fig.subplots_adjust(
                left=0.06,
                right=0.98,
                bottom=0.12,
                top=0.82,
                wspace=0.17,
            )

            output_path = (
                CollectStatistics
                ._metric_output_path(
                    output_dir,
                    metric_name,
                )
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            summary_rows[-1][
                "file"
            ] = str(
                output_path
            )

            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.15,
            )

            plt.close(fig)

        summary_df = pd.DataFrame(
            summary_rows
        )

        summary_df.to_csv(
            output_dir / "summary.csv",
            index=False,
        )

        return summary_df


# ======================================================================
# CLASSIFICATION
# ======================================================================


class ClassificationEvaluator:
    """
    Классификационная часть эксперимента.

    Основная схема:

        TRAIN:
            1.csv + 2.csv

        VALIDATION:
            3.csv

    3.csv не участвует в выборе порогов.

    Исследуются две постановки:

        1. stand / move / anomaly

        2. stand+move / anomaly

    Для фильтра Калмана рассматриваются:

        CV
        RW
        CV + RW
    """

    FEATURE_COLUMNS = (
        "log_likelihood",
        "mahalanobis",
        "filter_distance",
        "sqrt_mahalanobis",
        "log1p_filter_distance",
    )

    BASE_FEATURE_COLUMNS = (
        "log_likelihood",
        "mahalanobis",
        "filter_distance",
    )

    @staticmethod
    def add_derived_features(
        df: pd.DataFrame,
        model: str,
    ) -> pd.DataFrame:
        """
        Добавляет производные признаки.

        sqrt_mahalanobis:
            sqrt(M²)

        log1p_filter_distance:
            log(1 + filter_distance)
        """

        df = df.copy()

        m2_col = (
            f"{model}__mahalanobis"
        )

        fd_col = (
            f"{model}__filter_distance"
        )

        df[
            f"{model}__sqrt_mahalanobis"
        ] = np.sqrt(
            np.maximum(
                df[m2_col].to_numpy(
                    dtype=np.float64
                ),
                0.0,
            )
        )

        df[
            f"{model}__log1p_filter_distance"
        ] = np.log1p(
            np.maximum(
                df[fd_col].to_numpy(
                    dtype=np.float64
                ),
                0.0,
            )
        )

        return df

    # ------------------------------------------------------------------
    # FEATURE FRAME
    # ------------------------------------------------------------------

    @staticmethod
    def build_feature_frame(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        time: NDArray[np.datetime64],
        mark: NDArray[np.str_],
        model: str,
        parameters: FilterParameters,
        window: int,
    ) -> pd.DataFrame:

        metrics = (
            CollectStatistics
            .collect_kalman_metrics(
                lon=lon,
                lat=lat,
                time=time,
                mark=mark,
                model=model,
                parameters=parameters,
                window=window,
            )
        )

        result = pd.DataFrame(
            {
                "status": mark,
                f"{model}__log_likelihood": (
                    metrics[
                        "log_likelihood"
                    ]
                ),
                f"{model}__mahalanobis": (
                    metrics[
                        "mahalanobis"
                    ]
                ),
                f"{model}__filter_distance": (
                    metrics[
                        "filter_distance"
                    ]
                ),
            }
        )

        result = (
            ClassificationEvaluator
            .add_derived_features(
                result,
                model,
            )
        )

        return result

    # ------------------------------------------------------------------
    # FEATURE LISTS
    # ------------------------------------------------------------------

    @staticmethod
    def get_model_columns(
        model: str,
        include_derived: bool = True,
    ) -> list[str]:

        columns = [
            f"{model}__log_likelihood",
            f"{model}__mahalanobis",
            f"{model}__filter_distance",
        ]

        if include_derived:
            columns.extend(
                [
                    f"{model}__sqrt_mahalanobis",
                    (
                        f"{model}__"
                        f"log1p_filter_distance"
                    ),
                ]
            )

        return columns

    @staticmethod
    def get_single_feature_columns(
        model: str,
    ) -> list[str]:

        return [
            f"{model}__log_likelihood",
            f"{model}__mahalanobis",
            f"{model}__filter_distance",
            f"{model}__sqrt_mahalanobis",
            f"{model}__log1p_filter_distance",
        ]

    # ------------------------------------------------------------------
    # BINARY TARGET
    # ------------------------------------------------------------------

    @staticmethod
    def _binary_target(
        statuses: pd.Series,
    ) -> NDArray[np.int64]:
        """
        Преобразует:

            stand -> 0
            move  -> 0
            anomaly -> 1
        """

        return (
            statuses.to_numpy()
            == "anomaly"
        ).astype(
            np.int64
        )

    @staticmethod
    def threshold_score(
        values: NDArray[np.float64],
        y_true: NDArray[np.int64],
        threshold: float,
        anomaly_if_high: bool,
    ) -> NDArray[np.int64]:

        del y_true

        if anomaly_if_high:
            return (
                values >= threshold
            ).astype(
                np.int64
            )

        return (
            values <= threshold
        ).astype(
            np.int64
        )

    @staticmethod
    def find_best_binary_threshold(
        values: NDArray[np.float64],
        y_true: NDArray[np.int64],
        anomaly_if_high: bool,
    ) -> dict[str, float]:

        mask = (
            np.isfinite(values)
            & np.isfinite(y_true)
        )

        values = values[
            mask
        ]

        y_true = y_true[
            mask
        ]

        if (
            values.size == 0
            or np.unique(
                y_true
            ).size < 2
        ):
            return {
                "threshold": np.nan,
                "balanced_accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            }

        quantiles = np.linspace(
            0.0,
            1.0,
            THRESHOLD_GRID_SIZE,
        )

        thresholds = np.unique(
            np.quantile(
                values,
                quantiles,
            )
        )

        best = None

        for threshold in thresholds:

            y_pred = (
                ClassificationEvaluator
                .threshold_score(
                    values,
                    y_true,
                    float(threshold),
                    anomaly_if_high,
                )
            )

            balanced_acc = (
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            precision = (
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )

            recall = (
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            )

            score = (
                0.5 * balanced_acc
                + 0.5 * f1
            )

            candidate = {
                "threshold": float(
                    threshold
                ),
                "balanced_accuracy": float(
                    balanced_acc
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(
                    f1
                ),
                "_score": float(
                    score
                ),
            }

            if (
                best is None
                or candidate["_score"]
                > best["_score"]
            ):
                best = candidate

        assert best is not None

        best.pop(
            "_score"
        )

        return best

    @staticmethod
    def evaluate_binary_threshold(
        values: NDArray[np.float64],
        y_true: NDArray[np.int64],
        threshold: float,
        anomaly_if_high: bool,
    ) -> dict[str, float]:

        mask = (
            np.isfinite(values)
            & np.isfinite(y_true)
        )

        values = values[
            mask
        ]

        y_true = y_true[
            mask
        ]

        if values.size == 0:
            return {
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "count": 0,
            }

        y_pred = (
            ClassificationEvaluator
            .threshold_score(
                values,
                y_true,
                threshold,
                anomaly_if_high,
            )
        )

        if anomaly_if_high:
            scores = values
        else:
            scores = -values

        try:
            roc_auc = (
                roc_auc_score(
                    y_true,
                    scores,
                )
            )
        except ValueError:
            roc_auc = np.nan

        try:
            pr_auc = (
                average_precision_score(
                    y_true,
                    scores,
                )
            )
        except ValueError:
            pr_auc = np.nan

        return {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc
            ),
            "pr_auc": float(
                pr_auc
            ),
            "count": int(
                values.size
            ),
        }

    # ------------------------------------------------------------------
    # BINARY LOGISTIC REGRESSION
    # ------------------------------------------------------------------

    @staticmethod
    def build_binary_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    @staticmethod
    def evaluate_binary_logistic(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        Pipeline,
    ]:

        train_mask = (
            train_df["status"].isin(
                STATUS_VALUES
            )
        )

        valid_mask = (
            valid_df["status"].isin(
                STATUS_VALUES
            )
        )

        train = train_df[
            train_mask
        ]

        valid = valid_df[
            valid_mask
        ]

        X_train = train[
            feature_columns
        ]

        X_valid = valid[
            feature_columns
        ]

        y_train = (
            ClassificationEvaluator
            ._binary_target(
                train["status"]
            )
        )

        y_valid = (
            ClassificationEvaluator
            ._binary_target(
                valid["status"]
            )
        )

        pipeline = (
            ClassificationEvaluator
            .build_binary_pipeline()
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        train_probability = (
            pipeline
            .predict_proba(
                X_train
            )[:, 1]
        )

        valid_probability = (
            pipeline
            .predict_proba(
                X_valid
            )[:, 1]
        )

        train_prediction = (
            train_probability >= 0.5
        ).astype(
            np.int64
        )

        valid_prediction = (
            valid_probability >= 0.5
        ).astype(
            np.int64
        )

        train_metrics = (
            ClassificationEvaluator
            .classification_binary_metrics(
                y_train,
                train_prediction,
                train_probability,
            )
        )

        valid_metrics = (
            ClassificationEvaluator
            .classification_binary_metrics(
                y_valid,
                valid_prediction,
                valid_probability,
            )
        )

        return (
            train_metrics,
            valid_metrics,
            pipeline,
        )

    @staticmethod
    def classification_binary_metrics(
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.int64],
        probability: NDArray[np.float64],
    ) -> dict[str, float]:

        try:
            roc_auc = (
                roc_auc_score(
                    y_true,
                    probability,
                )
            )
        except ValueError:
            roc_auc = np.nan

        try:
            pr_auc = (
                average_precision_score(
                    y_true,
                    probability,
                )
            )
        except ValueError:
            pr_auc = np.nan

        return {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc
            ),
            "pr_auc": float(
                pr_auc
            ),
            "count": int(
                y_true.size
            ),
        }

    # ------------------------------------------------------------------
    # MULTICLASS
    # ------------------------------------------------------------------

    @staticmethod
    def _three_class_target(
        statuses: pd.Series,
    ) -> NDArray[np.str_]:
        return statuses.to_numpy()

    @staticmethod
    def build_multiclass_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    @staticmethod
    def build_single_feature_tree() -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=3,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    @staticmethod
    def classification_multiclass_metrics(
        y_true: NDArray[np.str_],
        y_pred: NDArray[np.str_],
    ) -> dict[str, float]:

        return {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "stand_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=["stand"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "move_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=["move"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "anomaly_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=["anomaly"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "count": int(
                y_true.size
            ),
        }

    @staticmethod
    def evaluate_multiclass_logistic(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        Pipeline,
    ]:

        train = train_df[
            train_df["status"].isin(
                STATUS_VALUES
            )
        ]

        valid = valid_df[
            valid_df["status"].isin(
                STATUS_VALUES
            )
        ]

        X_train = train[
            feature_columns
        ]

        X_valid = valid[
            feature_columns
        ]

        y_train = (
            ClassificationEvaluator
            ._three_class_target(
                train["status"]
            )
        )

        y_valid = (
            ClassificationEvaluator
            ._three_class_target(
                valid["status"]
            )
        )

        pipeline = (
            ClassificationEvaluator
            .build_multiclass_pipeline()
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        train_pred = pipeline.predict(
            X_train
        )

        valid_pred = pipeline.predict(
            X_valid
        )

        train_metrics = (
            ClassificationEvaluator
            .classification_multiclass_metrics(
                y_train,
                train_pred,
            )
        )

        valid_metrics = (
            ClassificationEvaluator
            .classification_multiclass_metrics(
                y_valid,
                valid_pred,
            )
        )

        return (
            train_metrics,
            valid_metrics,
            pipeline,
        )

    @staticmethod
    def evaluate_multiclass_single_feature(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_column: str,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        Pipeline,
    ]:

        train = train_df[
            train_df["status"].isin(
                STATUS_VALUES
            )
        ]

        valid = valid_df[
            valid_df["status"].isin(
                STATUS_VALUES
            )
        ]

        pipeline = (
            ClassificationEvaluator
            .build_single_feature_tree()
        )

        pipeline.fit(
            train[
                [feature_column]
            ],
            train["status"],
        )

        train_pred = pipeline.predict(
            train[
                [feature_column]
            ]
        )

        valid_pred = pipeline.predict(
            valid[
                [feature_column]
            ]
        )

        train_metrics = (
            ClassificationEvaluator
            .classification_multiclass_metrics(
                train[
                    "status"
                ].to_numpy(),
                train_pred,
            )
        )

        valid_metrics = (
            ClassificationEvaluator
            .classification_multiclass_metrics(
                valid[
                    "status"
                ].to_numpy(),
                valid_pred,
            )
        )

        return (
            train_metrics,
            valid_metrics,
            pipeline,
        )

    # ------------------------------------------------------------------
    # ROC / PR
    # ------------------------------------------------------------------

    @staticmethod
    def metric_to_binary_score(
        values: NDArray[np.float64],
        feature_name: str,
    ) -> NDArray[np.float64]:

        # Чем меньше log-likelihood,
        # тем более аномальным считается наблюдение.
        if feature_name.endswith(
            "__log_likelihood"
        ):
            return -values

        # Для M² и filter_distance
        # большие значения являются более аномальными.
        return values

    @staticmethod
    def plot_binary_curves(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        output_dir: Path,
        model: str,
        profile_name: str,
    ) -> None:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_columns = (
            ClassificationEvaluator
            .get_single_feature_columns(
                model
            )
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18.0, 7.5),
        )

        for feature_column in (
            feature_columns
        ):

            train_values = (
                train_df[
                    feature_column
                ].to_numpy(
                    dtype=np.float64
                )
            )

            valid_values = (
                valid_df[
                    feature_column
                ].to_numpy(
                    dtype=np.float64
                )
            )

            y_train = (
                ClassificationEvaluator
                ._binary_target(
                    train_df[
                        "status"
                    ]
                )
            )

            y_valid = (
                ClassificationEvaluator
                ._binary_target(
                    valid_df[
                        "status"
                    ]
                )
            )

            train_mask = np.isfinite(
                train_values
            )

            valid_mask = np.isfinite(
                valid_values
            )

            train_scores = (
                ClassificationEvaluator
                .metric_to_binary_score(
                    train_values[
                        train_mask
                    ],
                    feature_column,
                )
            )

            valid_scores = (
                ClassificationEvaluator
                .metric_to_binary_score(
                    valid_values[
                        valid_mask
                    ],
                    feature_column,
                )
            )

            y_train_clean = (
                y_train[
                    train_mask
                ]
            )

            y_valid_clean = (
                y_valid[
                    valid_mask
                ]
            )

            if (
                np.unique(
                    y_train_clean
                ).size < 2
            ):
                continue

            if (
                np.unique(
                    y_valid_clean
                ).size < 2
            ):
                continue

            train_fpr, train_tpr, _ = (
                roc_curve(
                    y_train_clean,
                    train_scores,
                )
            )

            valid_fpr, valid_tpr, _ = (
                roc_curve(
                    y_valid_clean,
                    valid_scores,
                )
            )

            train_auc = (
                roc_auc_score(
                    y_train_clean,
                    train_scores,
                )
            )

            valid_auc = (
                roc_auc_score(
                    y_valid_clean,
                    valid_scores,
                )
            )

            axes[0].plot(
                train_fpr,
                train_tpr,
                linewidth=1.8,
                label=(
                    f"{feature_column} "
                    f"train AUC="
                    f"{train_auc:.4f}"
                ),
            )

            axes[0].plot(
                valid_fpr,
                valid_tpr,
                linestyle="--",
                linewidth=1.8,
                label=(
                    f"{feature_column} "
                    f"valid AUC="
                    f"{valid_auc:.4f}"
                ),
            )

            (
                train_precision,
                train_recall,
                _,
            ) = precision_recall_curve(
                y_train_clean,
                train_scores,
            )

            (
                valid_precision,
                valid_recall,
                _,
            ) = precision_recall_curve(
                y_valid_clean,
                valid_scores,
            )

            train_pr_auc = (
                average_precision_score(
                    y_train_clean,
                    train_scores,
                )
            )

            valid_pr_auc = (
                average_precision_score(
                    y_valid_clean,
                    valid_scores,
                )
            )

            axes[1].plot(
                train_recall,
                train_precision,
                linewidth=1.8,
                label=(
                    f"{feature_column} "
                    f"train AP="
                    f"{train_pr_auc:.4f}"
                ),
            )

            axes[1].plot(
                valid_recall,
                valid_precision,
                linestyle="--",
                linewidth=1.8,
                label=(
                    f"{feature_column} "
                    f"valid AP="
                    f"{valid_pr_auc:.4f}"
                ),
            )

        axes[0].plot(
            [0, 1],
            [0, 1],
            linestyle=":",
            linewidth=1.0,
        )

        axes[0].set_title(
            f"ROC — {model}, "
            f"{profile_name}"
        )

        axes[0].set_xlabel(
            "False Positive Rate"
        )

        axes[0].set_ylabel(
            "True Positive Rate"
        )

        axes[0].grid(
            True,
            alpha=0.25,
        )

        axes[0].legend(
            fontsize=7,
            loc="lower right",
        )

        axes[1].set_title(
            f"Precision-Recall — "
            f"{model}, "
            f"{profile_name}"
        )

        axes[1].set_xlabel(
            "Recall"
        )

        axes[1].set_ylabel(
            "Precision"
        )

        axes[1].grid(
            True,
            alpha=0.25,
        )

        axes[1].legend(
            fontsize=7,
            loc="lower left",
        )

        fig.tight_layout()

        path = (
            output_dir
            / (
                f"{profile_name}_"
                f"{model}_roc_pr.png"
            )
        )

        fig.savefig(
            path,
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ------------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------------

    @staticmethod
    def save_confusion_matrix(
        y_true: Sequence[str],
        y_pred: Sequence[str],
        output_path: Path,
        title: str,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        labels = [
            "stand",
            "move",
            "anomaly",
        ]

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        fig, ax = plt.subplots(
            figsize=(8.0, 7.0)
        )

        image = ax.imshow(
            matrix
        )

        fig.colorbar(
            image,
            ax=ax,
        )

        ax.set_xticks(
            range(
                len(labels)
            )
        )

        ax.set_yticks(
            range(
                len(labels)
            )
        )

        ax.set_xticklabels(
            labels
        )

        ax.set_yticklabels(
            labels
        )

        ax.set_xlabel(
            "Предсказанный класс"
        )

        ax.set_ylabel(
            "Истинный класс"
        )

        ax.set_title(
            title
        )

        for i in range(
            len(labels)
        ):
            for j in range(
                len(labels)
            ):
                ax.text(
                    j,
                    i,
                    str(
                        matrix[
                            i,
                            j,
                        ]
                    ),
                    ha="center",
                    va="center",
                )

        fig.tight_layout()

        fig.savefig(
            output_path,
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ------------------------------------------------------------------
    # SUMMARY GRAPHS
    # ------------------------------------------------------------------

    @staticmethod
    def plot_model_comparison(
        df: pd.DataFrame,
        output_dir: Path,
        metric: str,
        value_column: str,
        title: str,
    ) -> None:

        if df.empty:
            return

        fig, ax = plt.subplots(
            figsize=(15.0, 8.0)
        )

        combinations = (
            df[
                [
                    "profile",
                    "model",
                ]
            ]
            .drop_duplicates()
            .reset_index(
                drop=True
            )
        )

        x_positions = np.arange(
            len(combinations)
        )

        values = []
        labels = []

        for _, row in (
            combinations.iterrows()
        ):

            mask = (
                (df["profile"]
                 == row["profile"])
                & (
                    df["model"]
                    == row["model"]
                )
                & (
                    df["metric"]
                    == metric
                )
            )

            selected = df.loc[
                mask,
                value_column,
            ]

            values.append(
                float(
                    selected.iloc[0]
                )
                if not selected.empty
                else np.nan
            )

            labels.append(
                f"{row['profile']}\n"
                f"{row['model']}"
            )

        ax.bar(
            x_positions,
            values,
        )

        ax.set_xticks(
            x_positions
        )

        ax.set_xticklabels(
            labels,
            rotation=20,
            ha="right",
        )

        ax.set_ylabel(
            value_column
        )

        ax.set_title(
            title
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / (
                f"{metric}_"
                f"{value_column}.png"
            ),
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)


# ======================================================================
# DATA PREPARATION
# ======================================================================


def load_dataset(
    path: Path,
) -> pd.DataFrame:

    LOGGER.info(
        "Загрузка: %s",
        path,
    )

    df = DataProcessor.load_csv(
        path
    )

    df = DataProcessor.pre_filter(
        df
    )

    df = df.iloc[
        POINT_SLICE
    ].copy()

    LOGGER.info(
        "%s: %d точек после pre_filter",
        path.name,
        len(df),
    )

    return df


def build_statistics_for_model(
    df: pd.DataFrame,
    model: str,
    parameters: FilterParameters,
    scheme_output_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        NDArray[np.float64],
    ],
]:
    """
    Формирует:

        1. point-level DataFrame;
        2. набор распределений для визуализации.
    """

    lon, lat, time = (
        DataProcessor.get_lon_lat(
            df
        )
    )

    mark = df[
        "status"
    ].to_numpy()

    metrics = (
        CollectStatistics
        .collect_kalman_metrics(
            lon=lon,
            lat=lat,
            time=time,
            mark=mark,
            model=model,
            parameters=parameters,
            window=WINDOW,
        )
    )

    point_df = pd.DataFrame(
        {
            "status": mark,
            f"{model}__log_likelihood": (
                metrics[
                    "log_likelihood"
                ]
            ),
            f"{model}__mahalanobis": (
                metrics[
                    "mahalanobis"
                ]
            ),
            f"{model}__filter_distance": (
                metrics[
                    "filter_distance"
                ]
            ),
        }
    )

    point_df = (
        ClassificationEvaluator
        .add_derived_features(
            point_df,
            model,
        )
    )

    (
        distances_anomaly,
        distances_move,
        distances_stand,
    ) = (
        CollectStatistics
        .collect_distance_between_point(
            lon=lon,
            lat=lat,
            mark=mark,
        )
    )

    statistics = {}

    # ==============================================================
    # 3 КЛАССА
    # ==============================================================

    for status in (
        "anomaly",
        "move",
        "stand",
    ):

        mask = (
            point_df["status"]
            == status
        )

        for kind in (
            "log_likelihood",
            "mahalanobis",
            "filter_distance",
            "sqrt_mahalanobis",
            "log1p_filter_distance",
        ):

            key = (
                CollectStatistics
                .make_metric_key(
                    "3_status",
                    status,
                    kind,
                    model,
                )
            )

            statistics[key] = (
                point_df.loc[
                    mask,
                    f"{model}__{kind}",
                ]
                .to_numpy()
            )

    # Расстояния между точками.
    statistics[
        CollectStatistics
        .make_metric_key(
            "3_status",
            "anomaly",
            "distances",
            None,
        )
    ] = distances_anomaly

    statistics[
        CollectStatistics
        .make_metric_key(
            "3_status",
            "move",
            "distances",
            None,
        )
    ] = distances_move

    statistics[
        CollectStatistics
        .make_metric_key(
            "3_status",
            "stand",
            "distances",
            None,
        )
    ] = distances_stand

    # ==============================================================
    # 2 КЛАССА
    #
    # stand + move -> normal
    # anomaly      -> anomaly
    # ==============================================================

    for kind in (
        "log_likelihood",
        "mahalanobis",
        "filter_distance",
        "sqrt_mahalanobis",
        "log1p_filter_distance",
    ):

        move_values = point_df.loc[
            point_df["status"]
            == "move",
            f"{model}__{kind}",
        ].to_numpy()

        stand_values = point_df.loc[
            point_df["status"]
            == "stand",
            f"{model}__{kind}",
        ].to_numpy()

        anomaly_values = point_df.loc[
            point_df["status"]
            == "anomaly",
            f"{model}__{kind}",
        ].to_numpy()

        statistics[
            CollectStatistics
            .make_metric_key(
                "2_status",
                "stand_move",
                kind,
                model,
            )
        ] = (
            CollectStatistics
            .combine_stand_and_move(
                move_values,
                stand_values,
            )
        )

        statistics[
            CollectStatistics
            .make_metric_key(
                "2_status",
                "anomaly",
                kind,
                model,
            )
        ] = anomaly_values

    CollectStatistics.visualize_statistics(
        statistics=statistics,
        output_dir=scheme_output_dir,
        bins=BINS,
        smoothing_sigma=SMOOTHING_SIGMA,
        dpi=PLOT_DPI,
        display_percentile=DISPLAY_PERCENTILE,
    )

    return (
        point_df,
        statistics,
    )


# ======================================================================
# PROFILE CREATION
# ======================================================================


def build_parameter_profiles(
    df_3: pd.DataFrame,
) -> dict[
    str,
    ParameterProfile,
]:

    profiles = {
        "from_1": ParameterProfile(
            name="from_1",
            description=(
                "Параметры калибровки по 1.csv"
            ),
            cv=FilterParameters(
                sigma_meas=(
                    KNOWN_PROFILE_FROM_1[
                        "sigma_meas"
                    ]
                ),
                process_noise=(
                    KNOWN_PROFILE_FROM_1[
                        "sigma_acc_cv"
                    ]
                ),
            ),
            rw=FilterParameters(
                sigma_meas=(
                    KNOWN_PROFILE_FROM_1[
                        "sigma_meas"
                    ]
                ),
                process_noise=(
                    KNOWN_PROFILE_FROM_1[
                        "sigma_rw"
                    ]
                ),
            ),
        ),
        "from_2": ParameterProfile(
            name="from_2",
            description=(
                "Параметры калибровки по 2.csv"
            ),
            cv=FilterParameters(
                sigma_meas=(
                    KNOWN_PROFILE_FROM_2[
                        "sigma_meas"
                    ]
                ),
                process_noise=(
                    KNOWN_PROFILE_FROM_2[
                        "sigma_acc_cv"
                    ]
                ),
            ),
            rw=FilterParameters(
                sigma_meas=(
                    KNOWN_PROFILE_FROM_2[
                        "sigma_meas"
                    ]
                ),
                process_noise=(
                    KNOWN_PROFILE_FROM_2[
                        "sigma_rw"
                    ]
                ),
            ),
        ),
    }

    # --------------------------------------------------------------
    # Диагностический self-calibrated профиль 3.csv.
    #
    # sigma_meas = 0.656633 м
    #
    # Process-noise определяется отдельно:
    #   CV -> sigma_acc
    #   RW -> sigma_rw
    #
    # Этот профиль НЕ является независимой validation-калибровкой.
    # --------------------------------------------------------------

    profile_3 = (
        CollectStatistics
        .estimate_profile_from_statuses(
            df=df_3,
            sigma_meas=0.656633,
        )
    )

    profiles[
        "from_3"
    ] = profile_3

    return profiles


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    path_root = (
        Path(__file__)
        .parent
        .parent
        .parent
    )

    data_dir = (
        path_root
        / "data"
    )

    output_root = (
        path_root
        / "statistics"
        / "classification"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "1": data_dir / "1.csv",
        "2": data_dir / "2.csv",
        "3": data_dir / "3.csv",
    }

    # ==============================================================
    # Загрузка данных
    # ==============================================================

    datasets = {
        name: load_dataset(path)
        for name, path in paths.items()
    }

    df_train_1 = datasets["1"]
    df_train_2 = datasets["2"]
    df_valid = datasets["3"]

    # ==============================================================
    # Параметрические профили
    # ==============================================================

    profiles = build_parameter_profiles(
        df_valid
    )

    profile_rows = []

    for (
        profile_name,
        profile,
    ) in profiles.items():

        profile_rows.extend(
            [
                {
                    "profile": profile_name,
                    "model": "CV",
                    "sigma_meas_m": (
                        profile.cv.sigma_meas
                    ),
                    "sigma_acc_m_s2": (
                        profile.cv.process_noise
                    ),
                    "sigma_rw_m_sqrt_s": np.nan,
                    "description": (
                        profile.description
                    ),
                },
                {
                    "profile": profile_name,
                    "model": "RW",
                    "sigma_meas_m": (
                        profile.rw.sigma_meas
                    ),
                    "sigma_acc_m_s2": np.nan,
                    "sigma_rw_m_sqrt_s": (
                        profile.rw.process_noise
                    ),
                    "description": (
                        profile.description
                    ),
                },
            ]
        )

    profiles_df = pd.DataFrame(
        profile_rows
    )

    profiles_df.to_csv(
        output_root
        / "parameter_profiles.csv",
        index=False,
    )

    LOGGER.info(
        "Используемые профили:\n%s",
        profiles_df.to_string(
            index=False
        ),
    )

    # ==============================================================
    # Результаты классификации
    # ==============================================================

    binary_threshold_rows = []
    binary_ml_rows = []
    multiclass_single_rows = []
    multiclass_ml_rows = []

    # ==============================================================
    # Перебор параметрических профилей
    # ==============================================================

    for (
        profile_name,
        profile,
    ) in profiles.items():

        LOGGER.info(
            "=================================================="
        )

        LOGGER.info(
            "PROFILE: %s",
            profile_name,
        )

        profile_dir = (
            output_root
            / profile_name
        )

        profile_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==========================================================
        # CV / RW
        # ==========================================================

        model_feature_frames_train = {}
        model_feature_frames_valid = {}

        for (
            model,
            parameters,
        ) in (
            (
                "CV",
                profile.cv,
            ),
            (
                "RW",
                profile.rw,
            ),
        ):

            if model == "CV":
                LOGGER.info(
                    "Profile=%s, model=CV, "
                    "sigma_meas=%.8f m, "
                    "sigma_acc=%.8f m/s²",
                    profile_name,
                    parameters.sigma_meas,
                    parameters.process_noise,
                )
            else:
                LOGGER.info(
                    "Profile=%s, model=RW, "
                    "sigma_meas=%.8f m, "
                    "sigma_rw=%.8f m/sqrt(s)",
                    profile_name,
                    parameters.sigma_meas,
                    parameters.process_noise,
                )

            # ------------------------------------------------------
            # TRAIN 1
            # ------------------------------------------------------

            train1_dir = (
                profile_dir
                / "distributions"
                / "1"
                / model
            )

            train1_df, _ = (
                build_statistics_for_model(
                    df=df_train_1,
                    model=model,
                    parameters=parameters,
                    scheme_output_dir=train1_dir,
                )
            )

            # ------------------------------------------------------
            # TRAIN 2
            # ------------------------------------------------------

            train2_dir = (
                profile_dir
                / "distributions"
                / "2"
                / model
            )

            train2_df, _ = (
                build_statistics_for_model(
                    df=df_train_2,
                    model=model,
                    parameters=parameters,
                    scheme_output_dir=train2_dir,
                )
            )

            # ------------------------------------------------------
            # VALIDATION 3
            # ------------------------------------------------------

            valid_dir = (
                profile_dir
                / "distributions"
                / "3"
                / model
            )

            valid_df, _ = (
                build_statistics_for_model(
                    df=df_valid,
                    model=model,
                    parameters=parameters,
                    scheme_output_dir=valid_dir,
                )
            )

            # ------------------------------------------------------
            # Общий train
            #
            # TRAIN = 1.csv + 2.csv
            # ------------------------------------------------------

            train_df = pd.concat(
                [
                    train1_df,
                    train2_df,
                ],
                ignore_index=True,
            )

            model_feature_frames_train[
                model
            ] = train_df

            model_feature_frames_valid[
                model
            ] = valid_df

            if SAVE_POINT_FEATURES:

                train_df.to_csv(
                    profile_dir
                    / (
                        f"train_"
                        f"{model}_features.csv"
                    ),
                    index=False,
                )

                valid_df.to_csv(
                    profile_dir
                    / (
                        f"valid_"
                        f"{model}_features.csv"
                    ),
                    index=False,
                )

            # ======================================================
            # 2-КЛАССОВАЯ ЗАДАЧА
            #
            # stand + move -> normal
            # anomaly      -> anomaly
            # ======================================================

            single_feature_columns = (
                ClassificationEvaluator
                .get_single_feature_columns(
                    model
                )
            )

            for feature_column in (
                single_feature_columns
            ):

                values_train = (
                    train_df[
                        feature_column
                    ].to_numpy(
                        dtype=np.float64
                    )
                )

                y_train = (
                    ClassificationEvaluator
                    ._binary_target(
                        train_df[
                            "status"
                        ]
                    )
                )

                feature_name = (
                    feature_column
                    .split(
                        "__",
                        1,
                    )[1]
                )

                anomaly_if_high = (
                    feature_name
                    != "log_likelihood"
                )

                threshold_result = (
                    ClassificationEvaluator
                    .find_best_binary_threshold(
                        values=values_train,
                        y_true=y_train,
                        anomaly_if_high=(
                            anomaly_if_high
                        ),
                    )
                )

                values_valid = (
                    valid_df[
                        feature_column
                    ].to_numpy(
                        dtype=np.float64
                    )
                )

                y_valid = (
                    ClassificationEvaluator
                    ._binary_target(
                        valid_df[
                            "status"
                        ]
                    )
                )

                valid_result = (
                    ClassificationEvaluator
                    .evaluate_binary_threshold(
                        values=values_valid,
                        y_true=y_valid,
                        threshold=(
                            threshold_result[
                                "threshold"
                            ]
                        ),
                        anomaly_if_high=(
                            anomaly_if_high
                        ),
                    )
                )

                train_result = (
                    ClassificationEvaluator
                    .evaluate_binary_threshold(
                        values=values_train,
                        y_true=y_train,
                        threshold=(
                            threshold_result[
                                "threshold"
                            ]
                        ),
                        anomaly_if_high=(
                            anomaly_if_high
                        ),
                    )
                )

                binary_threshold_rows.append(
                    {
                        "profile": profile_name,
                        "model": model,
                        "feature": feature_name,
                        "threshold": (
                            threshold_result[
                                "threshold"
                            ]
                        ),
                        "threshold_direction": (
                            "high"
                            if anomaly_if_high
                            else "low"
                        ),
                        "train_accuracy": (
                            train_result[
                                "accuracy"
                            ]
                        ),
                        "train_balanced_accuracy": (
                            train_result[
                                "balanced_accuracy"
                            ]
                        ),
                        "train_precision": (
                            train_result[
                                "precision"
                            ]
                        ),
                        "train_recall": (
                            train_result[
                                "recall"
                            ]
                        ),
                        "train_f1": (
                            train_result[
                                "f1"
                            ]
                        ),
                        "train_roc_auc": (
                            train_result[
                                "roc_auc"
                            ]
                        ),
                        "train_pr_auc": (
                            train_result[
                                "pr_auc"
                            ]
                        ),
                        "valid_accuracy": (
                            valid_result[
                                "accuracy"
                            ]
                        ),
                        "valid_balanced_accuracy": (
                            valid_result[
                                "balanced_accuracy"
                            ]
                        ),
                        "valid_precision": (
                            valid_result[
                                "precision"
                            ]
                        ),
                        "valid_recall": (
                            valid_result[
                                "recall"
                            ]
                        ),
                        "valid_f1": (
                            valid_result[
                                "f1"
                            ]
                        ),
                        "valid_roc_auc": (
                            valid_result[
                                "roc_auc"
                            ]
                        ),
                        "valid_pr_auc": (
                            valid_result[
                                "pr_auc"
                            ]
                        ),
                    }
                )

            # ======================================================
            # BINARY LOGISTIC REGRESSION
            # ======================================================

            (
                train_metrics,
                valid_metrics,
                _,
            ) = (
                ClassificationEvaluator
                .evaluate_binary_logistic(
                    train_df=train_df,
                    valid_df=valid_df,
                    feature_columns=(
                        single_feature_columns
                    ),
                )
            )

            binary_ml_rows.append(
                {
                    "profile": profile_name,
                    "model": model,
                    "classifier": "logistic",
                    "feature_set": (
                        "all_statistics"
                    ),
                    "train_accuracy": (
                        train_metrics[
                            "accuracy"
                        ]
                    ),
                    "train_balanced_accuracy": (
                        train_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "train_precision": (
                        train_metrics[
                            "precision"
                        ]
                    ),
                    "train_recall": (
                        train_metrics[
                            "recall"
                        ]
                    ),
                    "train_f1": (
                        train_metrics[
                            "f1"
                        ]
                    ),
                    "train_roc_auc": (
                        train_metrics[
                            "roc_auc"
                        ]
                    ),
                    "train_pr_auc": (
                        train_metrics[
                            "pr_auc"
                        ]
                    ),
                    "valid_accuracy": (
                        valid_metrics[
                            "accuracy"
                        ]
                    ),
                    "valid_balanced_accuracy": (
                        valid_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "valid_precision": (
                        valid_metrics[
                            "precision"
                        ]
                    ),
                    "valid_recall": (
                        valid_metrics[
                            "recall"
                        ]
                    ),
                    "valid_f1": (
                        valid_metrics[
                            "f1"
                        ]
                    ),
                    "valid_roc_auc": (
                        valid_metrics[
                            "roc_auc"
                        ]
                    ),
                    "valid_pr_auc": (
                        valid_metrics[
                            "pr_auc"
                        ]
                    ),
                }
            )

            # ------------------------------------------------------
            # ROC / PR
            # ------------------------------------------------------

            ClassificationEvaluator.plot_binary_curves(
                train_df=train_df,
                valid_df=valid_df,
                output_dir=(
                    profile_dir
                    / "classification"
                ),
                model=model,
                profile_name=profile_name,
            )

            # ======================================================
            # 3-КЛАССОВАЯ ЗАДАЧА
            #
            # stand / move / anomaly
            # ======================================================

            for feature_column in (
                single_feature_columns
            ):

                (
                    train_metrics,
                    valid_metrics,
                    tree_model,
                ) = (
                    ClassificationEvaluator
                    .evaluate_multiclass_single_feature(
                        train_df=train_df,
                        valid_df=valid_df,
                        feature_column=feature_column,
                    )
                )

                feature_name = (
                    feature_column
                    .split(
                        "__",
                        1,
                    )[1]
                )

                multiclass_single_rows.append(
                    {
                        "profile": profile_name,
                        "model": model,
                        "feature": feature_name,
                        "classifier": (
                            "decision_tree"
                        ),
                        "train_accuracy": (
                            train_metrics[
                                "accuracy"
                            ]
                        ),
                        "train_balanced_accuracy": (
                            train_metrics[
                                "balanced_accuracy"
                            ]
                        ),
                        "train_macro_f1": (
                            train_metrics[
                                "macro_f1"
                            ]
                        ),
                        "train_weighted_f1": (
                            train_metrics[
                                "weighted_f1"
                            ]
                        ),
                        "train_stand_f1": (
                            train_metrics[
                                "stand_f1"
                            ]
                        ),
                        "train_move_f1": (
                            train_metrics[
                                "move_f1"
                            ]
                        ),
                        "train_anomaly_f1": (
                            train_metrics[
                                "anomaly_f1"
                            ]
                        ),
                        "valid_accuracy": (
                            valid_metrics[
                                "accuracy"
                            ]
                        ),
                        "valid_balanced_accuracy": (
                            valid_metrics[
                                "balanced_accuracy"
                            ]
                        ),
                        "valid_macro_f1": (
                            valid_metrics[
                                "macro_f1"
                            ]
                        ),
                        "valid_weighted_f1": (
                            valid_metrics[
                                "weighted_f1"
                            ]
                        ),
                        "valid_stand_f1": (
                            valid_metrics[
                                "stand_f1"
                            ]
                        ),
                        "valid_move_f1": (
                            valid_metrics[
                                "move_f1"
                            ]
                        ),
                        "valid_anomaly_f1": (
                            valid_metrics[
                                "anomaly_f1"
                            ]
                        ),
                    }
                )

                valid_pred = tree_model.predict(
                    valid_df[
                        [feature_column]
                    ]
                )

                ClassificationEvaluator.save_confusion_matrix(
                    y_true=valid_df[
                        "status"
                    ].to_numpy(),
                    y_pred=valid_pred,
                    output_path=(
                        profile_dir
                        / "classification"
                        / (
                            f"3class_"
                            f"{model}_"
                            f"{feature_name}_"
                            f"confusion.png"
                        )
                    ),
                    title=(
                        f"3 класса: "
                        f"{model}, "
                        f"{feature_name}, "
                        f"{profile_name}"
                    ),
                )

            # ======================================================
            # MULTINOMIAL LOGISTIC REGRESSION
            # ======================================================

            (
                train_metrics,
                valid_metrics,
                multiclass_model,
            ) = (
                ClassificationEvaluator
                .evaluate_multiclass_logistic(
                    train_df=train_df,
                    valid_df=valid_df,
                    feature_columns=(
                        single_feature_columns
                    ),
                )
            )

            multiclass_ml_rows.append(
                {
                    "profile": profile_name,
                    "model": model,
                    "classifier": "logistic",
                    "feature_set": (
                        "all_statistics"
                    ),
                    "train_accuracy": (
                        train_metrics[
                            "accuracy"
                        ]
                    ),
                    "train_balanced_accuracy": (
                        train_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "train_macro_f1": (
                        train_metrics[
                            "macro_f1"
                        ]
                    ),
                    "train_weighted_f1": (
                        train_metrics[
                            "weighted_f1"
                        ]
                    ),
                    "train_stand_f1": (
                        train_metrics[
                            "stand_f1"
                        ]
                    ),
                    "train_move_f1": (
                        train_metrics[
                            "move_f1"
                        ]
                    ),
                    "train_anomaly_f1": (
                        train_metrics[
                            "anomaly_f1"
                        ]
                    ),
                    "valid_accuracy": (
                        valid_metrics[
                            "accuracy"
                        ]
                    ),
                    "valid_balanced_accuracy": (
                        valid_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "valid_macro_f1": (
                        valid_metrics[
                            "macro_f1"
                        ]
                    ),
                    "valid_weighted_f1": (
                        valid_metrics[
                            "weighted_f1"
                        ]
                    ),
                    "valid_stand_f1": (
                        valid_metrics[
                            "stand_f1"
                        ]
                    ),
                    "valid_move_f1": (
                        valid_metrics[
                            "move_f1"
                        ]
                    ),
                    "valid_anomaly_f1": (
                        valid_metrics[
                            "anomaly_f1"
                        ]
                    ),
                }
            )

            valid_pred = (
                multiclass_model.predict(
                    valid_df[
                        single_feature_columns
                    ]
                )
            )

            ClassificationEvaluator.save_confusion_matrix(
                y_true=valid_df[
                    "status"
                ].to_numpy(),
                y_pred=valid_pred,
                output_path=(
                    profile_dir
                    / "classification"
                    / (
                        f"3class_"
                        f"{model}_"
                        f"logistic_confusion.png"
                    )
                ),
                title=(
                    f"3 класса: {model}, "
                    f"объединённые статистики, "
                    f"{profile_name}"
                ),
            )

            # ------------------------------------------------------
            # Classification report
            # ------------------------------------------------------

            report = classification_report(
                valid_df[
                    "status"
                ],
                valid_pred,
                labels=list(
                    STATUS_VALUES
                ),
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(
                report
            ).T

            report_df.to_csv(
                profile_dir
                / "classification"
                / (
                    f"3class_"
                    f"{model}_"
                    f"logistic_report.csv"
                )
            )

        # ==========================================================
        # CV + RW
        # ==========================================================

        train_cv = (
            model_feature_frames_train[
                "CV"
            ]
        )

        train_rw = (
            model_feature_frames_train[
                "RW"
            ]
        )

        valid_cv = (
            model_feature_frames_valid[
                "CV"
            ]
        )

        valid_rw = (
            model_feature_frames_valid[
                "RW"
            ]
        )

        # Точки полностью совпадают по порядку.
        combined_train = pd.DataFrame(
            {
                "status": (
                    train_cv[
                        "status"
                    ].to_numpy()
                ),
            }
        )

        combined_valid = pd.DataFrame(
            {
                "status": (
                    valid_cv[
                        "status"
                    ].to_numpy()
                ),
            }
        )

        for (
            model_df,
            model,
        ) in (
            (
                train_cv,
                "CV",
            ),
            (
                train_rw,
                "RW",
            ),
        ):

            for column in (
                ClassificationEvaluator
                .get_single_feature_columns(
                    model
                )
            ):
                combined_train[
                    column
                ] = model_df[
                    column
                ].to_numpy()

        for (
            model_df,
            model,
        ) in (
            (
                valid_cv,
                "CV",
            ),
            (
                valid_rw,
                "RW",
            ),
        ):

            for column in (
                ClassificationEvaluator
                .get_single_feature_columns(
                    model
                )
            ):
                combined_valid[
                    column
                ] = model_df[
                    column
                ].to_numpy()

        combined_feature_columns = [
            column
            for model in (
                "CV",
                "RW",
            )
            for column in (
                ClassificationEvaluator
                .get_single_feature_columns(
                    model
                )
            )
        ]

        # ==========================================================
        # BINARY CV + RW
        # ==========================================================

        (
            train_metrics,
            valid_metrics,
            _,
        ) = (
            ClassificationEvaluator
            .evaluate_binary_logistic(
                train_df=combined_train,
                valid_df=combined_valid,
                feature_columns=(
                    combined_feature_columns
                ),
            )
        )

        binary_ml_rows.append(
            {
                "profile": profile_name,
                "model": "CV+RW",
                "classifier": "logistic",
                "feature_set": (
                    "CV+RW_all_statistics"
                ),
                "train_accuracy": (
                    train_metrics[
                        "accuracy"
                    ]
                ),
                "train_balanced_accuracy": (
                    train_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "train_precision": (
                    train_metrics[
                        "precision"
                    ]
                ),
                "train_recall": (
                    train_metrics[
                        "recall"
                    ]
                ),
                "train_f1": (
                    train_metrics[
                        "f1"
                    ]
                ),
                "train_roc_auc": (
                    train_metrics[
                        "roc_auc"
                    ]
                ),
                "train_pr_auc": (
                    train_metrics[
                        "pr_auc"
                    ]
                ),
                "valid_accuracy": (
                    valid_metrics[
                        "accuracy"
                    ]
                ),
                "valid_balanced_accuracy": (
                    valid_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "valid_precision": (
                    valid_metrics[
                        "precision"
                    ]
                ),
                "valid_recall": (
                    valid_metrics[
                        "recall"
                    ]
                ),
                "valid_f1": (
                    valid_metrics[
                        "f1"
                    ]
                ),
                "valid_roc_auc": (
                    valid_metrics[
                        "roc_auc"
                    ]
                ),
                "valid_pr_auc": (
                    valid_metrics[
                        "pr_auc"
                    ]
                ),
            }
        )

        # ==========================================================
        # MULTICLASS CV + RW
        # ==========================================================

        (
            train_metrics,
            valid_metrics,
            combined_multiclass_model,
        ) = (
            ClassificationEvaluator
            .evaluate_multiclass_logistic(
                train_df=combined_train,
                valid_df=combined_valid,
                feature_columns=(
                    combined_feature_columns
                ),
            )
        )

        multiclass_ml_rows.append(
            {
                "profile": profile_name,
                "model": "CV+RW",
                "classifier": "logistic",
                "feature_set": (
                    "CV+RW_all_statistics"
                ),
                "train_accuracy": (
                    train_metrics[
                        "accuracy"
                    ]
                ),
                "train_balanced_accuracy": (
                    train_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "train_macro_f1": (
                    train_metrics[
                        "macro_f1"
                    ]
                ),
                "train_weighted_f1": (
                    train_metrics[
                        "weighted_f1"
                    ]
                ),
                "train_stand_f1": (
                    train_metrics[
                        "stand_f1"
                    ]
                ),
                "train_move_f1": (
                    train_metrics[
                        "move_f1"
                    ]
                ),
                "train_anomaly_f1": (
                    train_metrics[
                        "anomaly_f1"
                    ]
                ),
                "valid_accuracy": (
                    valid_metrics[
                        "accuracy"
                    ]
                ),
                "valid_balanced_accuracy": (
                    valid_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "valid_macro_f1": (
                    valid_metrics[
                        "macro_f1"
                    ]
                ),
                "valid_weighted_f1": (
                    valid_metrics[
                        "weighted_f1"
                    ]
                ),
                "valid_stand_f1": (
                    valid_metrics[
                        "stand_f1"
                    ]
                ),
                "valid_move_f1": (
                    valid_metrics[
                        "move_f1"
                    ]
                ),
                "valid_anomaly_f1": (
                    valid_metrics[
                        "anomaly_f1"
                    ]
                ),
            }
        )

        valid_pred = (
            combined_multiclass_model
            .predict(
                combined_valid[
                    combined_feature_columns
                ]
            )
        )

        ClassificationEvaluator.save_confusion_matrix(
            y_true=combined_valid[
                "status"
            ].to_numpy(),
            y_pred=valid_pred,
            output_path=(
                profile_dir
                / "classification"
                / (
                    "3class_CV_RW_"
                    "logistic_confusion.png"
                )
            ),
            title=(
                "3 класса: CV + RW, "
                "объединённые признаки, "
                f"{profile_name}"
            ),
        )

    # ==================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ==================================================================

    binary_threshold_df = pd.DataFrame(
        binary_threshold_rows
    )

    binary_ml_df = pd.DataFrame(
        binary_ml_rows
    )

    multiclass_single_df = pd.DataFrame(
        multiclass_single_rows
    )

    multiclass_ml_df = pd.DataFrame(
        multiclass_ml_rows
    )

    binary_threshold_df.to_csv(
        output_root
        / "binary_thresholds.csv",
        index=False,
    )

    binary_ml_df.to_csv(
        output_root
        / "binary_logistic.csv",
        index=False,
    )

    multiclass_single_df.to_csv(
        output_root
        / "multiclass_single_feature.csv",
        index=False,
    )

    multiclass_ml_df.to_csv(
        output_root
        / "multiclass_logistic.csv",
        index=False,
    )

    # ==================================================================
    # СРАВНИТЕЛЬНЫЕ ГРАФИКИ
    # ==================================================================

    comparison_dir = (
        output_root
        / "comparison"
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Binary threshold comparison
    # --------------------------------------------------------------

    if not binary_threshold_df.empty:

        summary = (
            binary_threshold_df
            .sort_values(
                "valid_f1",
                ascending=False,
            )
            .head(20)
        )

        summary.to_csv(
            comparison_dir
            / "top_binary_thresholds.csv",
            index=False,
        )

        fig, ax = plt.subplots(
            figsize=(18.0, 9.0)
        )

        labels = (
            summary["profile"]
            + " / "
            + summary["model"]
            + " / "
            + summary["feature"]
        )

        ax.bar(
            np.arange(
                len(summary)
            ),
            summary[
                "valid_f1"
            ],
        )

        ax.set_xticks(
            np.arange(
                len(summary)
            )
        )

        ax.set_xticklabels(
            labels,
            rotation=60,
            ha="right",
            fontsize=8,
        )

        ax.set_ylabel(
            "Validation F1"
        )

        ax.set_title(
            "Лучшие бинарные классификаторы "
            "по отдельным статистикам"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            comparison_dir
            / "binary_threshold_f1.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    # --------------------------------------------------------------
    # Binary logistic
    # --------------------------------------------------------------

    if not binary_ml_df.empty:

        fig, ax = plt.subplots(
            figsize=(15.0, 8.0)
        )

        labels = (
            binary_ml_df["profile"]
            + " / "
            + binary_ml_df["model"]
        )

        x = np.arange(
            len(binary_ml_df)
        )

        width = 0.22

        ax.bar(
            x - width,
            binary_ml_df[
                "valid_f1"
            ],
            width=width,
            label="F1",
        )

        ax.bar(
            x,
            binary_ml_df[
                "valid_roc_auc"
            ],
            width=width,
            label="ROC-AUC",
        )

        ax.bar(
            x + width,
            binary_ml_df[
                "valid_pr_auc"
            ],
            width=width,
            label="PR-AUC",
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            labels,
            rotation=30,
            ha="right",
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_ylabel(
            "Validation score"
        )

        ax.set_title(
            "Бинарная классификация: "
            "stand+move vs anomaly"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            comparison_dir
            / "binary_logistic_comparison.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    # --------------------------------------------------------------
    # Multiclass
    # --------------------------------------------------------------

    if not multiclass_ml_df.empty:

        fig, ax = plt.subplots(
            figsize=(15.0, 8.0)
        )

        labels = (
            multiclass_ml_df["profile"]
            + " / "
            + multiclass_ml_df["model"]
        )

        x = np.arange(
            len(
                multiclass_ml_df
            )
        )

        width = 0.24

        ax.bar(
            x - width,
            multiclass_ml_df[
                "valid_macro_f1"
            ],
            width=width,
            label="Macro F1",
        )

        ax.bar(
            x,
            multiclass_ml_df[
                "valid_balanced_accuracy"
            ],
            width=width,
            label="Balanced accuracy",
        )

        ax.bar(
            x + width,
            multiclass_ml_df[
                "valid_anomaly_f1"
            ],
            width=width,
            label="Anomaly F1",
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            labels,
            rotation=30,
            ha="right",
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_ylabel(
            "Validation score"
        )

        ax.set_title(
            "3-классовая классификация: "
            "stand / move / anomaly"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            comparison_dir
            / "multiclass_logistic_comparison.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ==================================================================
    # TEXT SUMMARY
    # ==================================================================

    LOGGER.info(
        "=================================================="
    )

    LOGGER.info(
        "Эксперимент завершён."
    )

    LOGGER.info(
        "Результаты: %s",
        output_root,
    )

    if not binary_threshold_df.empty:

        best_binary = (
            binary_threshold_df
            .sort_values(
                "valid_f1",
                ascending=False,
            )
            .iloc[0]
        )

        LOGGER.info(
            "BEST binary threshold: "
            "profile=%s model=%s feature=%s "
            "F1=%.6f ROC-AUC=%.6f PR-AUC=%.6f",
            best_binary[
                "profile"
            ],
            best_binary[
                "model"
            ],
            best_binary[
                "feature"
            ],
            best_binary[
                "valid_f1"
            ],
            best_binary[
                "valid_roc_auc"
            ],
            best_binary[
                "valid_pr_auc"
            ],
        )

    if not binary_ml_df.empty:

        best_binary_ml = (
            binary_ml_df
            .sort_values(
                "valid_f1",
                ascending=False,
            )
            .iloc[0]
        )

        LOGGER.info(
            "BEST binary logistic: "
            "profile=%s model=%s "
            "F1=%.6f ROC-AUC=%.6f PR-AUC=%.6f",
            best_binary_ml[
                "profile"
            ],
            best_binary_ml[
                "model"
            ],
            best_binary_ml[
                "valid_f1"
            ],
            best_binary_ml[
                "valid_roc_auc"
            ],
            best_binary_ml[
                "valid_pr_auc"
            ],
        )

    if not multiclass_ml_df.empty:

        best_multiclass = (
            multiclass_ml_df
            .sort_values(
                "valid_macro_f1",
                ascending=False,
            )
            .iloc[0]
        )

        LOGGER.info(
            "BEST multiclass: "
            "profile=%s model=%s "
            "Macro-F1=%.6f "
            "Balanced accuracy=%.6f "
            "Anomaly-F1=%.6f",
            best_multiclass[
                "profile"
            ],
            best_multiclass[
                "model"
            ],
            best_multiclass[
                "valid_macro_f1"
            ],
            best_multiclass[
                "valid_balanced_accuracy"
            ],
            best_multiclass[
                "valid_anomaly_f1"
            ],
        )


# ======================================================================
# ENTRY POINT
# ======================================================================


if __name__ == "__main__":
    main()