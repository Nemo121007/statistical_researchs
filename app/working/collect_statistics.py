import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV
from app.working.kalman_filter_rw import KalmanFilterRW

from typing import List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class CollectStatistics:
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
        Собирает статистику расстояний между последовательными точками.

        Для пары P[i] -> P[i + 1] локальная система координат
        определяется относительно точки P[i - 10].

        Расстояние вычисляется между соседними точками.

        Returns:
            Расстояния для категорий:
            - anomaly;
            - move;
            - stand.
        """
        if len(lon) != len(lat) or len(lon) != len(mark):
            raise ValueError("lon, lat и mark должны иметь одинаковую длину")

        if len(lon) <= 11:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty

        # Для:
        # i = 10 -> локальная система начинается в P[0]
        # i = 11 -> локальная система начинается в P[1]
        # i = 12 -> локальная система начинается в P[2]
        #
        # А сами расстояния:
        # P[10] -> P[11]
        # P[11] -> P[12]
        # P[12] -> P[13]
        origin_lat = lat[:-11]

        # Масштаб одного градуса долготы
        # в локальной системе координат.
        kx = DataProcessor.LEN_LAT * np.cos(np.radians(origin_lat))

        delta_lon = lon[11:] - lon[10:-1]
        delta_lat = lat[11:] - lat[10:-1]
        dx = delta_lon * kx
        dy = delta_lat * DataProcessor.LEN_LAT
        distance = np.hypot(dx, dy)

        # Статусы концов соответствующего интервала.
        mark_left = mark[10:-1]
        mark_right = mark[11:]

        # Приоритет:
        # anomaly -> move -> stand.
        anomaly_mask = (mark_left == "anomaly") | (mark_right == "anomaly")

        move_mask = ~anomaly_mask & ((mark_left == "move") | (mark_right == "move"))

        stand_mask = ~anomaly_mask & ~move_mask & (mark_left == "stand") & (mark_right == "stand")

        return distance[anomaly_mask], distance[move_mask], distance[stand_mask]

    @staticmethod
    def collect_estimation_kalman_filter_cv(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        time: NDArray[np.datetime64],
        mark: NDArray[np.str_],
        window: int = 10,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Собирает статистику оценок фильтра Калмана CV для каждой точки трека..
        Для точки P[i] используется локальное окно:
            P[i-window] ... P[i-1] P[i]
        Локальная система координат строится относительно P[i-window].
        Фильтр Калмана последовательно обрабатывает точки окна,
        после чего сохраняются метрики последнего наблюдения P[i]:
            log_likelihood[-1]
            mahalanobis_sq[-1]
            hypot(filtered_x[-1] - x[-1], filtered_y[-1] - y[-1])
        Метрика относится непосредственно к точке P[i] и классифицируется по mark[i].

        Returns:
            anomaly_log_likelihood: Логарифмы правдоподобия для anomaly.
            anomaly_mahalanobis_sq: Квадраты расстояния Махаланобиса для anomaly.
            anomaly_filter_distance: Линейные расстояния между оценкой фильтра и измерением для anomaly.
            move_log_likelihood: Логарифмы правдоподобия для move.
            move_mahalanobis_sq: Квадраты расстояния Махаланобиса для move.
            move_filter_distance: Линейные расстояния между оценкой фильтра и измерением для move.
            stand_log_likelihood: Логарифмы правдоподобия для stand.
            stand_mahalanobis_sq: Квадраты расстояния Махаланобиса для stand.
            stand_filter_distance: Линейные расстояния между оценкой фильтра и измерением для stand.
        """
        if not (len(lon) == len(lat) == len(time) == len(mark)):
            raise ValueError("lon, lat, time и mark должны иметь одинаковую длину")

        if window < 1:
            raise ValueError("window должен быть >= 1")

        if len(lon) <= window:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, empty, empty, empty, empty, empty

        anomaly_log_likelihood = []
        anomaly_mahalanobis_sq = []
        anomaly_filter_distance = []
        move_log_likelihood = []
        move_mahalanobis_sq = []
        move_filter_distance = []
        stand_log_likelihood = []
        stand_mahalanobis_sq = []
        stand_filter_distance = []

        for i in range(window, len(lon)):
            lon_window = lon[i - window : i + 1]
            lat_window = lat[i - window : i + 1]
            time_window = time[i - window : i + 1]

            # Все точки текущего окна находятся в одной
            # локальной системе координат с началом в P[i-window].
            x_window, y_window = DataProcessor.convert_to_local_cartesian(lon_window, lat_window)

            kf = KalmanFilterCV()
            filtered_x, filtered_y, log_likelihood, mahalanobis_sq = kf.filter(x_window, y_window, time_window)

            # Последнее значение соответствует именно P[i].
            current_log_likelihood = log_likelihood[-1]
            current_mahalanobis_sq = mahalanobis_sq[-1]
            current_filter_distance = float(
                np.hypot(
                    filtered_x[-1] - x_window[-1],
                    filtered_y[-1] - y_window[-1],
                )
            )
            current_mark = mark[i]

            if current_mark == "anomaly":
                anomaly_log_likelihood.append(current_log_likelihood)
                anomaly_mahalanobis_sq.append(current_mahalanobis_sq)
                anomaly_filter_distance.append(current_filter_distance)

            elif current_mark == "move":
                move_log_likelihood.append(current_log_likelihood)
                move_mahalanobis_sq.append(current_mahalanobis_sq)
                move_filter_distance.append(current_filter_distance)

            elif current_mark == "stand":
                stand_log_likelihood.append(current_log_likelihood)
                stand_mahalanobis_sq.append(current_mahalanobis_sq)
                stand_filter_distance.append(current_filter_distance)

        return (
            np.asarray(anomaly_log_likelihood, dtype=np.float64),
            np.asarray(anomaly_mahalanobis_sq, dtype=np.float64),
            np.asarray(anomaly_filter_distance, dtype=np.float64),
            np.asarray(move_log_likelihood, dtype=np.float64),
            np.asarray(move_mahalanobis_sq, dtype=np.float64),
            np.asarray(move_filter_distance, dtype=np.float64),
            np.asarray(stand_log_likelihood, dtype=np.float64),
            np.asarray(stand_mahalanobis_sq, dtype=np.float64),
            np.asarray(stand_filter_distance, dtype=np.float64),
        )

    @staticmethod
    def collect_estimation_kalman_filter_rw(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        time: NDArray[np.datetime64],
        mark: NDArray[np.str_],
        window: int = 10,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Собирает статистику оценок фильтра Калмана RW для каждой точки трека.
        Для точки P[i] используется локальное окно:
            P[i-window] ... P[i-1] P[i]
        Локальная система координат строится относительно P[i-window].
        RW-фильтр последовательно обрабатывает точки окна, после чего сохраняются метрики последнего наблюдения P[i]:
            log_likelihood[-1]
            mahalanobis_sq[-1]
            hypot(filtered_x[-1] - x[-1], filtered_y[-1] - y[-1])
        Метрика относится непосредственно к точке P[i] и
        классифицируется по mark[i].
        Returns:
            anomaly_log_likelihood: Логарифмы правдоподобия для anomaly.

            anomaly_mahalanobis_sq: Квадраты расстояния Махаланобиса для anomaly.

            anomaly_filter_distance: Линейные расстояния между оценкой фильтра и измерением для anomaly.

            move_log_likelihood: Логарифмы правдоподобия для move.

            move_mahalanobis_sq: Квадраты расстояния Махаланобиса для move.

            move_filter_distance: Линейные расстояния между оценкой фильтра и измерением для move.

            stand_log_likelihood: Логарифмы правдоподобия для stand.

            stand_mahalanobis_sq: Квадраты расстояния Махаланобиса для stand.

            stand_filter_distance: Линейные расстояния между оценкой фильтра и измерением для stand.
        """
        if not (len(lon) == len(lat) == len(time) == len(mark)):
            raise ValueError("lon, lat, time и mark должны иметь одинаковую длину")

        if window < 1:
            raise ValueError("window должен быть >= 1")

        if len(lon) <= window:
            empty = np.empty(0, dtype=np.float64)

            return empty, empty, empty, empty, empty, empty, empty, empty, empty

        anomaly_log_likelihood = []
        anomaly_mahalanobis_sq = []
        anomaly_filter_distance = []
        move_log_likelihood = []
        move_mahalanobis_sq = []
        move_filter_distance = []
        stand_log_likelihood = []
        stand_mahalanobis_sq = []
        stand_filter_distance = []

        for i in range(window, len(lon)):
            lon_window = lon[i - window : i + 1]
            lat_window = lat[i - window : i + 1]
            time_window = time[i - window : i + 1]
            # Все точки окна переводятся в одну локальную декартову систему координат.
            x_window, y_window = DataProcessor.convert_to_local_cartesian(lon_window, lat_window)
            kf = KalmanFilterRW()
            filtered_x, filtered_y, log_likelihood, mahalanobis_sq = kf.filter(x_window, y_window, time_window)

            # Последнее значение соответствует P[i].
            current_log_likelihood = log_likelihood[-1]
            current_mahalanobis_sq = mahalanobis_sq[-1]
            current_filter_distance = float(
                np.hypot(
                    filtered_x[-1] - x_window[-1],
                    filtered_y[-1] - y_window[-1],
                )
            )
            current_mark = mark[i]

            if current_mark == "anomaly":
                anomaly_log_likelihood.append(current_log_likelihood)
                anomaly_mahalanobis_sq.append(current_mahalanobis_sq)
                anomaly_filter_distance.append(current_filter_distance)

            elif current_mark == "move":
                move_log_likelihood.append(current_log_likelihood)
                move_mahalanobis_sq.append(current_mahalanobis_sq)
                move_filter_distance.append(current_filter_distance)

            elif current_mark == "stand":
                stand_log_likelihood.append(current_log_likelihood)
                stand_mahalanobis_sq.append(current_mahalanobis_sq)
                stand_filter_distance.append(current_filter_distance)

        return (
            np.asarray(anomaly_log_likelihood, dtype=np.float64),
            np.asarray(anomaly_mahalanobis_sq, dtype=np.float64),
            np.asarray(anomaly_filter_distance, dtype=np.float64),
            np.asarray(move_log_likelihood, dtype=np.float64),
            np.asarray(move_mahalanobis_sq, dtype=np.float64),
            np.asarray(move_filter_distance, dtype=np.float64),
            np.asarray(stand_log_likelihood, dtype=np.float64),
            np.asarray(stand_mahalanobis_sq, dtype=np.float64),
            np.asarray(stand_filter_distance, dtype=np.float64),
        )

    @staticmethod
    def combine_stand_and_move(
        *arrays: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Объединяет массивы категорий stand и move в один.

        Используется после раздельного сбора статистик: фильтр Калмана
        не зависит от status при вычислении метрик, поэтому объединение
        stand ∪ move эквивалентно повторному анализу с одним статусом.
        """
        parts = [np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays]
        if not parts:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(parts)

    @staticmethod
    def _linear_residuals(values: NDArray[np.float64], time_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Остатки линейной регрессии value = a + b * t (модель постоянной скорости)."""
        design = np.column_stack((np.ones(len(time_s), dtype=np.float64), time_s))
        try:
            coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        except np.linalg.LinAlgError:
            return values - np.mean(values)
        return values - design @ coefficients

    @staticmethod
    def _sample_std(values: NDArray[np.float64]) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            return float("nan")
        return float(np.std(values, ddof=1))

    @staticmethod
    def estimate_kalman_noise_parameters(
        lon: NDArray[np.float64],
        lat: NDArray[np.float64],
        time: NDArray[np.datetime64],
    ) -> dict[str, float]:
        """
        Оценивает СКО, соответствующие параметрам фильтров Калмана.

        sigma_meas:
            СКО шума измерений, м. Остатки координат после вычитания
            линейного тренда (постоянная скорость) по каждой оси,
            затем объединение осей.

        sigma_acc_cv:
            СКО ускорения, м/с². Конечные разности скорости.
            Это параметр процесса модели CV.

        sigma_acc_rw:
            Интенсивность случайного блуждания, м/с^{1/2}.
            Оценка из E[dx²] / dt после вычитания линейного тренда.
            Это параметр процесса модели RW: Q = sigma_acc² * dt.
        """
        empty = {
            "n_points": float(len(lon)),
            "duration_s": float("nan"),
            "sigma_meas": float("nan"),
            "variance_meas": float("nan"),
            "sigma_acc_cv": float("nan"),
            "variance_acc_cv": float("nan"),
            "sigma_acc_rw": float("nan"),
            "variance_acc_rw": float("nan"),
        }

        if not (len(lon) == len(lat) == len(time)) or len(lon) < 3:
            return empty

        x_local, y_local = DataProcessor.convert_to_local_cartesian(lon, lat)
        time = np.asarray(time, dtype="datetime64[ns]")
        time_s = (time - time[0]) / np.timedelta64(1, "s")
        time_s = np.asarray(time_s, dtype=np.float64)
        duration_s = float(time_s[-1] - time_s[0]) if time_s.size else float("nan")
        empty["duration_s"] = duration_s

        residual_x = CollectStatistics._linear_residuals(x_local, time_s)
        residual_y = CollectStatistics._linear_residuals(y_local, time_s)
        sigma_meas_x = CollectStatistics._sample_std(residual_x)
        sigma_meas_y = CollectStatistics._sample_std(residual_y)
        if np.isfinite(sigma_meas_x) and np.isfinite(sigma_meas_y):
            sigma_meas = float(np.sqrt(0.5 * (sigma_meas_x**2 + sigma_meas_y**2)))
        else:
            sigma_meas = float("nan")

        dt = np.diff(time_s)
        valid_dt = dt > 1e-6
        sigma_acc_cv = float("nan")
        sigma_acc_rw = float("nan")

        if np.count_nonzero(valid_dt) >= 2:
            vx = np.diff(x_local)[valid_dt] / dt[valid_dt]
            vy = np.diff(y_local)[valid_dt] / dt[valid_dt]
            dt_valid = dt[valid_dt]
            dt_acc = 0.5 * (dt_valid[:-1] + dt_valid[1:])
            valid_acc = dt_acc > 1e-6
            if np.count_nonzero(valid_acc) >= 2:
                ax = (vx[1:] - vx[:-1]) / dt_acc
                ay = (vy[1:] - vy[:-1]) / dt_acc
                ax = ax[valid_acc]
                ay = ay[valid_acc]
                sigma_ax = CollectStatistics._sample_std(ax)
                sigma_ay = CollectStatistics._sample_std(ay)
                if np.isfinite(sigma_ax) and np.isfinite(sigma_ay):
                    sigma_acc_cv = float(np.sqrt(0.5 * (sigma_ax**2 + sigma_ay**2)))

            residual_dx = np.diff(residual_x)[valid_dt]
            residual_dy = np.diff(residual_y)[valid_dt]
            intensity = (residual_dx**2 + residual_dy**2) / (2.0 * dt_valid)
            intensity = intensity[np.isfinite(intensity) & (intensity >= 0.0)]
            if intensity.size >= 2:
                sigma_acc_rw = float(np.sqrt(np.mean(intensity)))

        return {
            "n_points": float(len(lon)),
            "duration_s": duration_s,
            "sigma_meas": sigma_meas,
            "variance_meas": sigma_meas**2 if np.isfinite(sigma_meas) else float("nan"),
            "sigma_acc_cv": sigma_acc_cv,
            "variance_acc_cv": sigma_acc_cv**2 if np.isfinite(sigma_acc_cv) else float("nan"),
            "sigma_acc_rw": sigma_acc_rw,
            "variance_acc_rw": sigma_acc_rw**2 if np.isfinite(sigma_acc_rw) else float("nan"),
        }

    @staticmethod
    def collect_kalman_noise_for_intervals(
        intervals: Sequence[pd.DataFrame],
        example: str,
        scheme: str,
        status: str,
    ) -> List[dict]:
        """Оценивает параметры шума Калмана на каждом участке одного статуса."""
        rows: List[dict] = []
        for interval_index, interval_df in enumerate(intervals):
            lon, lat, time = DataProcessor.get_lon_lat(interval_df)
            parameters = CollectStatistics.estimate_kalman_noise_parameters(lon, lat, time)
            rows.append(
                {
                    "example": example,
                    "scheme": scheme,
                    "status": status,
                    "interval_index": interval_index,
                    **parameters,
                }
            )
        return rows

    @staticmethod
    def summarize_kalman_noise(interval_rows: Sequence[dict]) -> pd.DataFrame:
        """Агрегирует оценки шума по примеру, схеме статусов и статусу."""
        if not interval_rows:
            return pd.DataFrame()

        interval_df = pd.DataFrame(interval_rows)
        metric_names = (
            "sigma_meas",
            "variance_meas",
            "sigma_acc_cv",
            "variance_acc_cv",
            "sigma_acc_rw",
            "variance_acc_rw",
        )
        summary_rows = []
        grouped = interval_df.groupby(["example", "scheme", "status"], sort=False)
        for (example, scheme, status), group in grouped:
            row = {
                "example": example,
                "scheme": scheme,
                "status": status,
                "n_intervals": int(len(group)),
                "n_points_total": int(np.nansum(group["n_points"].to_numpy(dtype=np.float64))),
                "duration_s_total": float(np.nansum(group["duration_s"].to_numpy(dtype=np.float64))),
            }
            for metric_name in metric_names:
                values = group[metric_name].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    row[f"{metric_name}_mean"] = np.nan
                    row[f"{metric_name}_std"] = np.nan
                    row[f"{metric_name}_min"] = np.nan
                    row[f"{metric_name}_p05"] = np.nan
                    row[f"{metric_name}_p50"] = np.nan
                    row[f"{metric_name}_p95"] = np.nan
                    row[f"{metric_name}_max"] = np.nan
                    continue
                row[f"{metric_name}_mean"] = float(np.mean(values))
                row[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                row[f"{metric_name}_min"] = float(np.min(values))
                row[f"{metric_name}_p05"] = float(np.percentile(values, 5))
                row[f"{metric_name}_p50"] = float(np.percentile(values, 50))
                row[f"{metric_name}_p95"] = float(np.percentile(values, 95))
                row[f"{metric_name}_max"] = float(np.max(values))
            summary_rows.append(row)
        return pd.DataFrame(summary_rows)

    # ==========================================================
    # АГРЕГАЦИЯ
    # ==========================================================
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
        """
        Агрегирует большой массив для построения плотности.
        Все исходные значения участвуют в агрегации.
        Для линейного режима используются равномерные bins.
        Для логарифмического режима используются логарифмически
        распределённые bins. Логарифмический режим применим только
        к строго положительным значениям.
        Returns:
            centers: Центры bins.
            density: Плотность распределения в каждом bin.
            counts: Количество исходных наблюдений в каждом bin.
        """
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]

        if values.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty

        if bins < 10:
            raise ValueError("Количество bins должно быть >= 10")

        value_min = float(np.min(values))
        value_max = float(np.max(values))

        if value_min == value_max:
            width = max(
                abs(value_min) * 0.01,
                1.0,
            )

            edges = np.array([value_min - width, value_max + width], dtype=np.float64)

            counts = np.array([values.size], dtype=np.float64)

            density = counts / (values.size * np.diff(edges))

            centers = (edges[:-1] + edges[1:]) / 2.0

            return centers, density, counts

        # ------------------------------------------------------
        # Линейное распределение bins
        # ------------------------------------------------------
        if not log_scale:
            edges = np.linspace(value_min, value_max, bins + 1, dtype=np.float64)

        # ------------------------------------------------------
        # Логарифмическое распределение bins
        # ------------------------------------------------------
        else:
            if value_min <= 0.0:
                raise ValueError("log_scale=True требует строго положительных значений")

            edges = np.geomspace(value_min, value_max, bins + 1, dtype=np.float64)

        counts, edges = np.histogram(values, bins=edges)

        counts = counts.astype(np.float64)
        widths = np.diff(edges)
        density = counts / (values.size * widths)

        centers = (edges[:-1] + edges[1:]) / 2.0

        return centers, density, counts

    # ==========================================================
    # ПОСТРОЕНИЕ ПЛОТНОСТИ
    # ==========================================================
    @staticmethod
    def _build_density(
        values: NDArray[np.float64],
        bins: int = 2000,
        smoothing_sigma: float = 2.0,
        log_scale: bool = False,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Строит сглаженную плотность распределения.
        В отличие от gaussian_kde, здесь сначала выполняется
        агрегация всей выборки в bins, после чего сглаживается
        уже компактный массив плотности.
        Это позволяет эффективно работать с миллионами
        исходных значений.
        Args:
            values: Исходный массив.
            bins: Количество bins.
            smoothing_sigma: Степень сглаживания в единицах bins.
            log_scale: Использовать логарифмически расположенные bins.
        Returns:
            x: Координаты плотности.
            density: Сглаженная плотность.
        """
        x, density, _ = CollectStatistics.aggregate_distribution(values=values, bins=bins, log_scale=log_scale)

        if density.size == 0:
            return x, density

        if smoothing_sigma > 0.0 and density.size > 3:
            density = gaussian_filter1d(density, sigma=smoothing_sigma, mode="nearest")
            # После сглаживания устраняем небольшие
            # отрицательные значения, которые теоретически
            # могут появиться из-за численной погрешности.
            density = np.maximum(
                density,
                0.0,
            )
        return x, density

    # ==========================================================
    # ПОСТРОЕНИЕ ФУНКЦИИ РАСПРЕДЕЛЕНИЯ
    # ==========================================================
    @staticmethod
    def _build_cdf(
        values: NDArray[np.float64],
        bins: int = 2000,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Строит эмпирическую функцию распределения F(x) = P(X ≤ x).

        Все исходные значения участвуют в расчёте. Для отрисовки
        ECDF прореживается до bins точек по рангу, поэтому график
        остаётся монотонно неубывающим и заканчивается в 1.
        """
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]

        if values.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty

        sorted_values = np.sort(values)
        n = sorted_values.size

        sample_count = min(max(int(bins), 10), n)
        idx = np.unique(np.round(np.linspace(0, n - 1, sample_count)).astype(np.int64))

        x = sorted_values[idx]
        cdf = (idx + 1).astype(np.float64) / n

        x = np.concatenate([[sorted_values[0]], x, [sorted_values[-1]]])
        cdf = np.concatenate([[0.0], cdf, [1.0]])

        return x, cdf

    # ==========================================================
    # ВИЗУАЛИЗАЦИЯ
    # ==========================================================
    @staticmethod
    def _get_display_limits(
        values: NDArray[np.float64],
        display_percentile: float = 95.0,
    ) -> Tuple[float, float]:
        """
        Определяет границы видимой области графика.
        Метод НЕ удаляет значения из массива и НЕ влияет на расчёт
        статистических характеристик.
        Args:
            values: Полный массив значений. Предполагается, что NaN и inf уже удалены.
            display_percentile: Процент центральной части распределения, которую нужно показать на графике.
        Returns:
            Нижняя и верхняя границы отображаемой области X.
        """
        values = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if values.size == 0:
            raise ValueError("Невозможно определить границы отображения для пустого массива")

        if not 0.0 < display_percentile <= 100.0:
            raise ValueError("display_percentile должен находиться в диапазоне (0, 100]")

        tail_percentile = (100.0 - display_percentile) / 2.0

        lower = float(np.percentile(values, tail_percentile))

        upper = float(np.percentile(values, 100.0 - tail_percentile))

        # Для константного массива или ситуации,
        # когда оба перцентиля совпали.
        if lower == upper:
            value = lower
            padding = max(
                abs(value) * 0.01,
                1.0,
            )
            lower = value - padding
            upper = value + padding

        return lower, upper

    SCHEME_LABELS = {
        "3_status": "3 статуса",
        "2_status": "2 статуса",
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
    }

    MODEL_LABELS = {
        "CV": "модель постоянной скорости (CV)",
        "RW": "модель случайного блуждания (RW)",
    }

    @staticmethod
    def make_metric_key(
        scheme: str,
        status: str,
        kind: str,
        model: str | None = None,
    ) -> str:
        """Собирает уникальное имя метрики: scheme__status__kind[__model]."""
        parts = [scheme, status, kind]
        if model is not None:
            parts.append(model)
        return "__".join(parts)

    @staticmethod
    def _split_metric_key(metric_name: str) -> Tuple[str, str, str, str | None]:
        parts = metric_name.split("__")
        if len(parts) == 3:
            scheme, status, kind = parts
            return scheme, status, kind, None
        if len(parts) == 4:
            scheme, status, kind, model = parts
            return scheme, status, kind, model
        raise ValueError(
            "Имя метрики должно иметь вид "
            "scheme__status__kind или scheme__status__kind__model, "
            f"получено: {metric_name!r}"
        )

    @staticmethod
    def _metric_labels(
        metric_name: str,
    ) -> Tuple[str, str, str, str | None, str, str, str]:
        scheme, status, kind, model = CollectStatistics._split_metric_key(metric_name)
        if kind not in CollectStatistics.KIND_LABELS:
            raise ValueError(f"Неизвестный тип метрики: {kind}")

        title, xlabel = CollectStatistics.KIND_LABELS[kind]
        scheme_label = CollectStatistics.SCHEME_LABELS.get(scheme, scheme)
        status_label = CollectStatistics.STATUS_LABELS.get(status, status)
        context_parts = [scheme_label, status_label]
        if model is not None:
            context_parts.append(CollectStatistics.MODEL_LABELS.get(model, model))
        context = ", ".join(context_parts)
        return scheme, status, kind, model, title, xlabel, context

    @staticmethod
    def _metric_output_path(
        output_dir: Path,
        metric_name: str,
    ) -> Path:
        scheme, status, kind, model = CollectStatistics._split_metric_key(metric_name)
        folder = output_dir / scheme / status / kind
        filename = f"{model}.png" if model is not None else f"{kind}.png"
        return folder / filename

    @staticmethod
    def visualize_statistics(
        statistics: Mapping[
            str,
            NDArray[np.float64] | list | tuple,
        ],
        output_dir: Path,
        bins: int = 2000,
        smoothing_sigma: float = 2.0,
        dpi: int = 600,
        figsize: Tuple[float, float] = (18.0, 7.5),
        display_percentile: float = 95.0,
    ) -> pd.DataFrame:
        """
        Визуализирует переданные статистические массивы.
        Для каждой метрики:
            1. Из исходного массива удаляются только NaN и +/-inf.
            2. Все статистические характеристики рассчитываются
               по полной выборке.
            3. Строится ECDF по полной выборке.
            4. Для отрисовки ECDF используется прореживание,
               выполняемое внутри _build_cdf().
            5. Создаётся PNG высокого качества.
            6. График состоит из двух горизонтальных подграфиков:
                 - линейная шкала X;
                 - логарифмическая шкала X для положительных
                   распределений;
                 - symlog для распределений с отрицательными
                   значениями.
            7. На обоих графиках отмечаются P5, P25, P50, P75, P95.
            8. Область X принудительно ограничивается
               display_percentile центральной частью распределения.
            9. Хвостовые значения НЕ удаляются, а только скрываются
               за пределами отображаемой области.
           10. Сохраняется summary.csv.

        Args:
            statistics: Словарь: имя метрики -> полный массив значений.
            output_dir: Каталог для сохранения PNG.
            bins: Максимальное количество точек ECDF, используемых непосредственно для отрисовки.
            smoothing_sigma: Сохраняется в API для совместимости. Для ECDF не используется.
            dpi: DPI сохраняемого PNG.
            figsize: Размер фигуры в дюймах.
            display_percentile: Центральная часть распределения, отображаемая на графике.
        Returns:
            DataFrame со статистикой всех метрик.
        """
        if not 0.0 < display_percentile <= 100.0:
            raise ValueError("display_percentile должен находиться " "в диапазоне (0, 100]")

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for metric_name, raw_values in statistics.items():
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                scheme, status, kind, model, title, _, _ = CollectStatistics._metric_labels(metric_name)
                summary_rows.append(
                    {
                        "metric": metric_name,
                        "title": title,
                        "scheme": scheme,
                        "status": status,
                        "kind": kind,
                        "model": model if model is not None else "",
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
                        "display_percentile": display_percentile,
                        "display_min": np.nan,
                        "display_max": np.nan,
                        "file": "",
                    }
                )
                continue

            # ======================================================
            # Статистика по ВСЕМ значениям
            # ======================================================
            mean = float(np.mean(values))
            variance = float(np.var(values, ddof=0))
            std = float(np.std(values, ddof=0))
            minimum = float(np.min(values))
            maximum = float(np.max(values))

            p05, p25, p50, p75, p95 = np.percentile(values,[5, 25, 50, 75, 95],)
            display_min, display_max = CollectStatistics._get_display_limits(
                values=values,
                display_percentile=display_percentile,
            )

            # ======================================================
            # Сводная статистика
            # ======================================================
            scheme, status, kind, model, title, xlabel, context = CollectStatistics._metric_labels(metric_name)
            summary_rows.append(
                {
                    "metric": metric_name,
                    "title": title,
                    "scheme": scheme,
                    "status": status,
                    "kind": kind,
                    "model": model if model is not None else "",
                    "count": int(values.size),
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
                    "display_percentile": display_percentile,
                    "display_min": display_min,
                    "display_max": display_max,
                    "file": "",
                }
            )
            x_cdf, y_cdf = CollectStatistics._build_cdf(values=values,bins=bins)
            strictly_positive = np.all(values > 0.0)
            fig, axes = plt.subplots(1,2, figsize=figsize)
            fig.suptitle(f"{title}\n{context}", fontsize=16, fontweight="bold", y=0.98)

            fig.text(
                0.5,
                0.90,
                (
                    f"{context}    |    Мат. ожидание = {mean:.8g}    |    Дисперсия = {variance:.8g}    |    "
                    f"N = {values.size:,}    |    Отображение = {display_percentile:g}%"
                ),
                ha="center",
                va="center",
                fontsize=11,
            )
            percentile_data = (
                (float(p05), "P5", 0.05, "#1b9e77"),
                (float(p25), "P25", 0.25, "#d95f02"),
                (float(p50), "P50", 0.50, "#7570b3"),
                (float(p75), "P75", 0.75, "#e7298a"),
                (float(p95), "P95", 0.95, "#66a61e"),
            )

            # ======================================================
            # Функция отрисовки
            # ======================================================
            def draw_cdf(
                ax,
                logarithmic_x: bool,
            ) -> None:
                """
                Отрисовывает ECDF.
                xlim применяется только после построения полного
                распределения.
                """

                # --------------------------------------------------
                # ECDF
                # --------------------------------------------------

                ax.plot(x_cdf, y_cdf, color="#222222", linewidth=2.0, label="Функция распределения F(x)")

                for (percentile_value, label, probability, color) in percentile_data:
                    ax.axvline(
                        percentile_value,
                        color=color,
                        linestyle="--",
                        linewidth=1.4,
                        label=(f"{label} = " f"{percentile_value:.6g}"),
                    )

                    ax.axhline(
                        probability,
                        color=color,
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )

                ax.set_ylim(-0.02, 1.05)
                ax.set_ylabel("Функция распределения F(x) = P(X ≤ x)", fontsize=11,)
                ax.set_xlabel(xlabel, fontsize=11,)
                ax.set_xlim(display_min, display_max)
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8, loc="lower right")
                if logarithmic_x:
                    ax.set_xscale("log",)
                    ax.set_title("Функция распределения — " "логарифмический масштаб X", fontsize=13)
                else:

                    ax.set_title("Функция распределения — " "линейный масштаб X", fontsize=13)
            draw_cdf( axes[0], logarithmic_x=False)
            if strictly_positive:
                draw_cdf(axes[1], logarithmic_x=True)
            else:
                draw_cdf(axes[1], logarithmic_x=False)
                scale = max(abs(p50) * 1e-3, 1e-9,)
                axes[1].set_xscale("symlog", linthresh=scale)

                axes[1].set_title("Функция распределения — " "логарифмический " "масштаб X", fontsize=13,)

            fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.82, wspace=0.17,)

            output_path = CollectStatistics._metric_output_path(output_dir, metric_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_rows[-1]["file"] = str(output_path)

            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.15,
            )
            plt.close(fig)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    def add_metric(
            scheme: str,
            status: str,
            kind: str,
            values: NDArray[np.float64],
            model: str | None = None,
    ) -> None:
        key = CollectStatistics.make_metric_key(scheme, status, kind, model)
        statistics[key] = values

    path_root = Path(__file__).parent.parent.parent
    paths = [
        path_root / "data" / "1.csv",
        path_root / "data" / "2.csv",
        path_root / "data" / "3.csv",
    ]
    for path in paths:
        name_item = str(path.name).split(".")[0]
        logging.info(f"Обработка файла: {name_item}")
        df = DataProcessor.load_csv(path)
        df = DataProcessor.pre_filter(df)
        df = df[120_000:130_000]  # Ограничение для тестирования на небольшом объёме данных
        lon, lat, time = DataProcessor.get_lon_lat(df)
        mark = df["status"].to_numpy()

        distances_anomaly, distances_move, distances_stand = CollectStatistics.collect_distance_between_point(
            lon, lat, mark
        )

        (
            anomaly_log_likelihood_CV,
            anomaly_mahalanobis_sq_CV,
            anomaly_filter_distance_CV,
            move_log_likelihood_CV,
            move_mahalanobis_sq_CV,
            move_filter_distance_CV,
            stand_log_likelihood_CV,
            stand_mahalanobis_sq_CV,
            stand_filter_distance_CV,
        ) = CollectStatistics.collect_estimation_kalman_filter_cv(lon, lat, time, mark, window=10)

        (
            anomaly_log_likelihood_RW,
            anomaly_mahalanobis_sq_RW,
            anomaly_filter_distance_RW,
            move_log_likelihood_RW,
            move_mahalanobis_sq_RW,
            move_filter_distance_RW,
            stand_log_likelihood_RW,
            stand_mahalanobis_sq_RW,
            stand_filter_distance_RW,
        ) = CollectStatistics.collect_estimation_kalman_filter_rw(lon, lat, time, mark, window=10)

        statistics: dict[str, NDArray[np.float64]] = {}

        add_metric("3_status", "anomaly", "distances", distances_anomaly)
        add_metric("3_status", "move", "distances", distances_move)
        add_metric("3_status", "stand", "distances", distances_stand)

        add_metric("3_status", "anomaly", "log_likelihood", anomaly_log_likelihood_CV, "CV")
        add_metric("3_status", "anomaly", "mahalanobis", anomaly_mahalanobis_sq_CV, "CV")
        add_metric("3_status", "anomaly", "filter_distance", anomaly_filter_distance_CV, "CV")
        add_metric("3_status", "move", "log_likelihood", move_log_likelihood_CV, "CV")
        add_metric("3_status", "move", "mahalanobis", move_mahalanobis_sq_CV, "CV")
        add_metric("3_status", "move", "filter_distance", move_filter_distance_CV, "CV")
        add_metric("3_status", "stand", "log_likelihood", stand_log_likelihood_CV, "CV")
        add_metric("3_status", "stand", "mahalanobis", stand_mahalanobis_sq_CV, "CV")
        add_metric("3_status", "stand", "filter_distance", stand_filter_distance_CV, "CV")

        add_metric("3_status", "anomaly", "log_likelihood", anomaly_log_likelihood_RW, "RW")
        add_metric("3_status", "anomaly", "mahalanobis", anomaly_mahalanobis_sq_RW, "RW")
        add_metric("3_status", "anomaly", "filter_distance", anomaly_filter_distance_RW, "RW")
        add_metric("3_status", "move", "log_likelihood", move_log_likelihood_RW, "RW")
        add_metric("3_status", "move", "mahalanobis", move_mahalanobis_sq_RW, "RW")
        add_metric("3_status", "move", "filter_distance", move_filter_distance_RW, "RW")
        add_metric("3_status", "stand", "log_likelihood", stand_log_likelihood_RW, "RW")
        add_metric("3_status", "stand", "mahalanobis", stand_mahalanobis_sq_RW, "RW")
        add_metric("3_status", "stand", "filter_distance", stand_filter_distance_RW, "RW")

        # stand ∪ move — отдельный набор с уникальными именами (2_status).
        distances_stand_move = CollectStatistics.combine_stand_and_move(distances_move, distances_stand)
        stand_move_log_likelihood_CV = CollectStatistics.combine_stand_and_move(
            move_log_likelihood_CV, stand_log_likelihood_CV
        )
        stand_move_mahalanobis_sq_CV = CollectStatistics.combine_stand_and_move(
            move_mahalanobis_sq_CV, stand_mahalanobis_sq_CV
        )
        stand_move_filter_distance_CV = CollectStatistics.combine_stand_and_move(
            move_filter_distance_CV, stand_filter_distance_CV
        )
        stand_move_log_likelihood_RW = CollectStatistics.combine_stand_and_move(
            move_log_likelihood_RW, stand_log_likelihood_RW
        )
        stand_move_mahalanobis_sq_RW = CollectStatistics.combine_stand_and_move(
            move_mahalanobis_sq_RW, stand_mahalanobis_sq_RW
        )
        stand_move_filter_distance_RW = CollectStatistics.combine_stand_and_move(
            move_filter_distance_RW, stand_filter_distance_RW
        )

        add_metric("2_status", "anomaly", "distances", distances_anomaly)
        add_metric("2_status", "stand_move", "distances", distances_stand_move)

        add_metric("2_status", "anomaly", "log_likelihood", anomaly_log_likelihood_CV, "CV")
        add_metric("2_status", "anomaly", "mahalanobis", anomaly_mahalanobis_sq_CV, "CV")
        add_metric("2_status", "anomaly", "filter_distance", anomaly_filter_distance_CV, "CV")
        add_metric("2_status", "stand_move", "log_likelihood", stand_move_log_likelihood_CV, "CV")
        add_metric("2_status", "stand_move", "mahalanobis", stand_move_mahalanobis_sq_CV, "CV")
        add_metric("2_status", "stand_move", "filter_distance", stand_move_filter_distance_CV, "CV")

        add_metric("2_status", "anomaly", "log_likelihood", anomaly_log_likelihood_RW, "RW")
        add_metric("2_status", "anomaly", "mahalanobis", anomaly_mahalanobis_sq_RW, "RW")
        add_metric("2_status", "anomaly", "filter_distance", anomaly_filter_distance_RW, "RW")
        add_metric("2_status", "stand_move", "log_likelihood", stand_move_log_likelihood_RW, "RW")
        add_metric("2_status", "stand_move", "mahalanobis", stand_move_mahalanobis_sq_RW, "RW")
        add_metric("2_status", "stand_move", "filter_distance", stand_move_filter_distance_RW, "RW")

        statistics_dir = path_root / "statistics" / name_item

        CollectStatistics.visualize_statistics(
            statistics=statistics,
            output_dir=statistics_dir,
            bins=2000,
            smoothing_sigma=2.0,
            dpi=600,
        )

        exit(0)
