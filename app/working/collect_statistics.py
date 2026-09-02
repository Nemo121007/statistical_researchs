import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV
from app.working.kalman_filter_rw import KalmanFilterRW

from typing import Mapping, Tuple

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
    ) -> pd.DataFrame:
        """
        Визуализирует переданные статистические массивы.
        Для каждого массива:
        1. Все NaN и +/-inf удаляются.
        2. Среднее, дисперсия и перцентили рассчитываются
           по всей исходной выборке.
        3. Строится эмпирическая функция распределения F(x) = P(X ≤ x).
        4. Создаётся PNG высокого качества.
        5. График содержит два горизонтальных подграфика:
             - линейный масштаб;
             - логарифмический масштаб.
        6. На графиках отображаются P5, P25, P50, P75, P95
           разными цветами.
        7. В заголовке отображаются название, математическое
           ожидание и дисперсия.
        Для положительных метрик: второй график использует log X.
        Для метрик, которые могут иметь отрицательные значения
        (например log_likelihood): второй график использует symlog X.
        Статистика рассчитывается по всем исходным значениям.
        Дополнительно сохраняется summary.csv.
        Args:
            statistics: Словарь: имя метрики -> массив значений.
            output_dir: Каталог для сохранения PNG.
            bins: Количество точек ECDF для отрисовки.
            smoothing_sigma: Сохранён для совместимости API,
                на функцию распределения не влияет.

            dpi: DPI сохраняемых PNG.
            figsize: Размер каждого графика.

        Returns:
            DataFrame со статистикой всех метрик.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for metric_name, raw_values in statistics.items():
            # ==================================================
            # Подготовка исходных данных
            # ==================================================
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)

            values = values[np.isfinite(values)]

            if values.size == 0:
                print(f"[WARNING] {metric_name}: нет конечных значений")
                continue

            # ==================================================
            # Точная статистика по всей выборке
            # ==================================================

            mean = float(np.mean(values))

            variance = float(np.var(values, ddof=0))

            std = float(np.std(values, ddof=0))

            minimum = float(np.min(values))

            maximum = float(np.max(values))

            p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])

            # ==================================================
            # Сохраняем статистику
            # ==================================================

            summary_rows.append(
                {
                    "metric": metric_name,
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
                }
            )

            # ==================================================
            # Определяем возможность log X
            # ==================================================
            strictly_positive = np.all(values > 0.0)

            x_cdf, y_cdf = CollectStatistics._build_cdf(values=values, bins=bins)

            fig, axes = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(
                metric_name,
                fontsize=18,
                fontweight="bold",
                y=0.98,
            )

            fig.text(
                0.5,
                0.925,
                (f"E[X] = {mean:.8g}" f"    |    " f"Var[X] = {variance:.8g}" f"    |    " f"N = {values.size:,}"),
                ha="center",
                va="center",
                fontsize=12,
            )

            percentile_data = (
                (float(p05), "P5", 0.05, "#1b9e77"),
                (float(p25), "P25", 0.25, "#d95f02"),
                (float(p50), "P50", 0.50, "#7570b3"),
                (float(p75), "P75", 0.75, "#e7298a"),
                (float(p95), "P95", 0.95, "#66a61e"),
            )

            # ==================================================
            # Вложенная функция построения графика
            # ==================================================

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
            statistics:
                Словарь:
                    имя метрики -> полный массив значений.

            output_dir:
                Каталог для сохранения PNG.

            bins:
                Максимальное количество точек ECDF,
                используемых непосредственно для отрисовки.

            smoothing_sigma:
                Сохраняется в API для совместимости.
                Для ECDF не используется.

            dpi:
                DPI сохраняемого PNG.

            figsize:
                Размер фигуры в дюймах.

            display_percentile:
                Центральная часть распределения,
                отображаемая на графике.

                Например:

                    95 -> P2.5 ... P97.5
                    99 -> P0.5 ... P99.5
                    100 -> min ... max

                Значения за пределами диапазона не удаляются.

        Returns:
            DataFrame со статистикой всех метрик.
        """
        if not 0.0 < display_percentile <= 100.0:
            raise ValueError("display_percentile должен находиться " "в диапазоне (0, 100]")

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_rows = []

        for metric_name, raw_values in statistics.items():

            # ======================================================
            # Подготовка исходного массива
            # ======================================================

            values = np.asarray(
                raw_values,
                dtype=np.float64,
            ).reshape(-1)

            # Удаляем только NaN и +/-inf.
            # Реальные хвосты распределения НЕ удаляются.
            values = values[np.isfinite(values)]

            if values.size == 0:
                print(f"[WARNING] {metric_name}: " "нет конечных значений")
                continue

            # ======================================================
            # Статистика по ВСЕМ значениям
            # ======================================================

            mean = float(np.mean(values))

            variance = float(
                np.var(
                    values,
                    ddof=0,
                )
            )

            std = float(
                np.std(
                    values,
                    ddof=0,
                )
            )

            minimum = float(np.min(values))

            maximum = float(np.max(values))

            p05, p25, p50, p75, p95 = np.percentile(
                values,
                [5, 25, 50, 75, 95],
            )

            # ======================================================
            # Границы отображения
            #
            # ВАЖНО:
            # это НЕ фильтрация values.
            # Это только xlim графика.
            # ======================================================

            display_min, display_max = CollectStatistics._get_display_limits(
                values=values,
                display_percentile=display_percentile,
            )

            # ======================================================
            # Сводная статистика
            # ======================================================

            summary_rows.append(
                {
                    "metric": metric_name,
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
                }
            )

            # ======================================================
            # ECDF
            # ======================================================

            # Весь массив values передаётся в _build_cdf().
            #
            # _build_cdf() использует все значения для сортировки
            # и расчёта CDF, а bins влияет только на количество
            # точек для визуализации.
            x_cdf, y_cdf = CollectStatistics._build_cdf(
                values=values,
                bins=bins,
            )

            # ======================================================
            # Возможность log X
            # ======================================================

            strictly_positive = np.all(values > 0.0)

            # ======================================================
            # Создание figure
            # ======================================================

            fig, axes = plt.subplots(
                1,
                2,
                figsize=figsize,
            )

            # ======================================================
            # Заголовок
            # ======================================================

            fig.suptitle(
                metric_name,
                fontsize=18,
                fontweight="bold",
                y=0.98,
            )

            fig.text(
                0.5,
                0.925,
                (
                    f"E[X] = {mean:.8g}"
                    f"    |    "
                    f"Var[X] = {variance:.8g}"
                    f"    |    "
                    f"N = {values.size:,}"
                    f"    |    "
                    f"Display = {display_percentile:g}%"
                ),
                ha="center",
                va="center",
                fontsize=12,
            )

            # ======================================================
            # Перцентили
            #
            # ВАЖНО:
            # вычислены по всему исходному массиву.
            # ======================================================

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

                ax.plot(
                    x_cdf,
                    y_cdf,
                    color="#222222",
                    linewidth=2.0,
                    label="F(x) = P(X ≤ x)",
                )

                # --------------------------------------------------
                # Перцентили
                # --------------------------------------------------

                for (
                    percentile_value,
                    label,
                    probability,
                    color,
                ) in percentile_data:

                    ax.axvline(
                        percentile_value,
                        color=color,
                        linestyle="--",
                        linewidth=1.4,
                        label=(f"{label} = " f"{percentile_value:.6g}"),
                    )

                    # Горизонтальная линия соответствующего
                    # значения CDF.
                    ax.axhline(
                        probability,
                        color=color,
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )

                # --------------------------------------------------
                # Y
                # --------------------------------------------------

                ax.set_ylim(
                    -0.02,
                    1.05,
                )

                ax.set_ylabel(
                    "F(x) = P(X ≤ x)",
                    fontsize=11,
                )

                ax.set_xlabel(
                    metric_name,
                    fontsize=11,
                )

                # --------------------------------------------------
                # Главная часть изменения:
                #
                # принудительное окно отображения.
                #
                # values и x_cdf остаются полными.
                # --------------------------------------------------

                ax.set_xlim(
                    display_min,
                    display_max,
                )

                # --------------------------------------------------
                # Сетка
                # --------------------------------------------------

                ax.grid(
                    True,
                    which="both",
                    alpha=0.25,
                )

                ax.legend(
                    fontsize=8,
                    loc="lower right",
                )

                # --------------------------------------------------
                # Масштаб X
                # --------------------------------------------------

                if logarithmic_x:

                    ax.set_xscale(
                        "log",
                    )

                    ax.set_title(
                        "Функция распределения — " "логарифмический масштаб X",
                        fontsize=13,
                    )

                else:

                    ax.set_title(
                        "Функция распределения — " "линейный масштаб X",
                        fontsize=13,
                    )

            # ======================================================
            # Левый график — линейный
            # ======================================================

            draw_cdf(
                axes[0],
                logarithmic_x=False,
            )

            # ======================================================
            # Правый график
            # ======================================================

            if strictly_positive:

                # Для положительных величин используем
                # настоящий логарифмический X.

                draw_cdf(
                    axes[1],
                    logarithmic_x=True,
                )

            else:

                # Для величин, которые могут быть отрицательными,
                # обычный log X невозможен.
                #
                # Поэтому сначала рисуем обычный ECDF,
                # устанавливаем xlim, а затем переключаем X
                # на symmetric log scale.

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
                    "Функция распределения — " "симметричный логарифмический " "масштаб X",
                    fontsize=13,
                )

            # ======================================================
            # Отступы
            # ======================================================

            fig.subplots_adjust(
                left=0.06,
                right=0.98,
                bottom=0.12,
                top=0.84,
                wspace=0.17,
            )

            # ======================================================
            # Имя файла
            # ======================================================

            safe_name = (
                metric_name.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace(":", "_")
                .replace("*", "_")
                .replace("?", "_")
                .replace("<", "_")
                .replace(">", "_")
                .replace("|", "_")
            )

            output_path = output_dir / f"{safe_name}.png"

            # ======================================================
            # Сохранение PNG
            # ======================================================

            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.15,
            )

            plt.close(fig)

            print(
                f"[OK] {metric_name}: "
                f"N={values.size:,}; "
                f"E[X]={mean:.8g}; "
                f"Var[X]={variance:.8g}; "
                f"P5={p05:.8g}; "
                f"P50={p50:.8g}; "
                f"P95={p95:.8g}; "
                f"display={display_min:.8g}..."
                f"{display_max:.8g}; "
                f"file={output_path}"
            )

        # ==========================================================
        # Summary
        # ==========================================================

        summary_df = pd.DataFrame(summary_rows)

        summary_path = output_dir / "summary.csv"

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    path_root = Path(__file__).parent.parent.parent
    path = path_root / "data" / "1.csv"

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

    statistics = {
        "distances_anomaly": distances_anomaly,
        "distances_move": distances_move,
        "distances_stand": distances_stand,
        "anomaly_log_likelihood_CV": anomaly_log_likelihood_CV,
        "anomaly_mahalanobis_sq_CV": anomaly_mahalanobis_sq_CV,
        "anomaly_filter_distance_CV": anomaly_filter_distance_CV,
        "move_log_likelihood_CV": move_log_likelihood_CV,
        "move_mahalanobis_sq_CV": move_mahalanobis_sq_CV,
        "move_filter_distance_CV": move_filter_distance_CV,
        "stand_log_likelihood_CV": stand_log_likelihood_CV,
        "stand_mahalanobis_sq_CV": stand_mahalanobis_sq_CV,
        "stand_filter_distance_CV": stand_filter_distance_CV,
        "anomaly_log_likelihood_RW": anomaly_log_likelihood_RW,
        "anomaly_mahalanobis_sq_RW": anomaly_mahalanobis_sq_RW,
        "anomaly_filter_distance_RW": anomaly_filter_distance_RW,
        "move_log_likelihood_RW": move_log_likelihood_RW,
        "move_mahalanobis_sq_RW": move_mahalanobis_sq_RW,
        "move_filter_distance_RW": move_filter_distance_RW,
        "stand_log_likelihood_RW": stand_log_likelihood_RW,
        "stand_mahalanobis_sq_RW": stand_mahalanobis_sq_RW,
        "stand_filter_distance_RW": stand_filter_distance_RW,
    }

    statistics_dir = path_root / "statistics"

    summary = CollectStatistics.visualize_statistics(
        statistics=statistics,
        output_dir=statistics_dir,
        bins=2000,
        smoothing_sigma=2.0,
        dpi=600,
    )

    print("\nИтоговая статистика:")
    print(summary.to_string(index=False))
