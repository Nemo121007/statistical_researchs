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
        kx = (DataProcessor.LEN_LAT * np.cos(np.radians(origin_lat)))

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
        anomaly_mask = ((mark_left == "anomaly") | (mark_right == "anomaly"))

        move_mask = (~anomaly_mask & ((mark_left == "move") | (mark_right == "move")))

        stand_mask = (~anomaly_mask & ~move_mask & (mark_left == "stand") & (mark_right == "stand"))

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
        Метрика относится непосредственно к точке P[i] и классифицируется по mark[i].

        Returns:
            anomaly_log_likelihood: Логарифмы правдоподобия для anomaly.
            anomaly_mahalanobis_sq: Квадраты расстояния Махаланобиса для anomaly.
            move_log_likelihood: Логарифмы правдоподобия для move.
            move_mahalanobis_sq: Квадраты расстояния Махаланобиса для move.
            stand_log_likelihood: Логарифмы правдоподобия для stand.
            stand_mahalanobis_sq: Квадраты расстояния Махаланобиса для stand.
        """
        if not (len(lon) == len(lat) == len(time) == len(mark)):
            raise ValueError("lon, lat, time и mark должны иметь одинаковую длину")

        if window < 1:
            raise ValueError("window должен быть >= 1")

        if len(lon) <= window:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, empty, empty

        anomaly_log_likelihood = []
        anomaly_mahalanobis_sq = []
        move_log_likelihood = []
        move_mahalanobis_sq = []
        stand_log_likelihood = []
        stand_mahalanobis_sq = []

        for i in range(window, len(lon)):
            lon_window = lon[i - window:i + 1]
            lat_window = lat[i - window:i + 1]
            time_window = time[i - window:i + 1]

            # Все точки текущего окна находятся в одной
            # локальной системе координат с началом в P[i-window].
            x_window, y_window = (DataProcessor.convert_to_local_cartesian(lon_window, lat_window))

            kf = KalmanFilterCV()
            (_, _, log_likelihood, mahalanobis_sq) = kf.filter(x_window, y_window, time_window)

            # Последнее значение соответствует именно P[i].
            current_log_likelihood = log_likelihood[-1]
            current_mahalanobis_sq = mahalanobis_sq[-1]
            current_mark = mark[i]

            if current_mark == "anomaly":
                anomaly_log_likelihood.append(current_log_likelihood)
                anomaly_mahalanobis_sq.append(current_mahalanobis_sq)

            elif current_mark == "move":
                move_log_likelihood.append(current_log_likelihood)
                move_mahalanobis_sq.append(current_mahalanobis_sq)

            elif current_mark == "stand":
                stand_log_likelihood.append(current_log_likelihood)
                stand_mahalanobis_sq.append(current_mahalanobis_sq)

        return (
            np.asarray(anomaly_log_likelihood, dtype=np.float64),
            np.asarray(anomaly_mahalanobis_sq, dtype=np.float64),
            np.asarray(move_log_likelihood, dtype=np.float64),
            np.asarray(move_mahalanobis_sq, dtype=np.float64),
            np.asarray(stand_log_likelihood, dtype=np.float64),
            np.asarray(stand_mahalanobis_sq, dtype=np.float64),
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
    ]:
        """
        Собирает статистику оценок фильтра Калмана RW для каждой точки трека.
        Для точки P[i] используется локальное окно:
            P[i-window] ... P[i-1] P[i]
        Локальная система координат строится относительно P[i-window].
        RW-фильтр последовательно обрабатывает точки окна, после чего сохраняются метрики последнего наблюдения P[i]:
            log_likelihood[-1]
            mahalanobis_sq[-1]
        Метрика относится непосредственно к точке P[i] и
        классифицируется по mark[i].
        Returns:
            anomaly_log_likelihood: Логарифмы правдоподобия для anomaly.

            anomaly_mahalanobis_sq: Квадраты расстояния Махаланобиса для anomaly.

            move_log_likelihood: Логарифмы правдоподобия для move.

            move_mahalanobis_sq: Квадраты расстояния Махаланобиса для move.

            stand_log_likelihood: Логарифмы правдоподобия для stand.

            stand_mahalanobis_sq: Квадраты расстояния Махаланобиса для stand.
        """
        if not (len(lon) == len(lat) == len(time) == len(mark)):
            raise ValueError("lon, lat, time и mark должны иметь одинаковую длину")

        if window < 1:
            raise ValueError("window должен быть >= 1")

        if len(lon) <= window:
            empty = np.empty(0, dtype=np.float64)

            return empty, empty, empty, empty, empty, empty

        anomaly_log_likelihood = []
        anomaly_mahalanobis_sq = []
        move_log_likelihood = []
        move_mahalanobis_sq = []
        stand_log_likelihood = []
        stand_mahalanobis_sq = []

        for i in range(window, len(lon)):
            lon_window = lon[i - window:i + 1]
            lat_window = lat[i - window:i + 1]
            time_window = time[i - window:i + 1]
            # Все точки окна переводятся в одну локальную декартову систему координат.
            x_window, y_window = DataProcessor.convert_to_local_cartesian(lon_window, lat_window)
            kf = KalmanFilterRW()
            _, _, log_likelihood, mahalanobis_sq = kf.filter(x_window, y_window, time_window)

            # Последнее значение соответствует P[i].
            current_log_likelihood = log_likelihood[-1]
            current_mahalanobis_sq = mahalanobis_sq[-1]
            current_mark = mark[i]

            if current_mark == "anomaly":
                anomaly_log_likelihood.append(current_log_likelihood)
                anomaly_mahalanobis_sq.append(current_mahalanobis_sq)

            elif current_mark == "move":
                move_log_likelihood.append(current_log_likelihood)
                move_mahalanobis_sq.append(current_mahalanobis_sq)

            elif current_mark == "stand":
                stand_log_likelihood.append(current_log_likelihood)
                stand_mahalanobis_sq.append(current_mahalanobis_sq)

        return (
            np.asarray(anomaly_log_likelihood, dtype=np.float64),
            np.asarray(anomaly_mahalanobis_sq, dtype=np.float64),
            np.asarray(move_log_likelihood, dtype=np.float64),
            np.asarray(move_mahalanobis_sq, dtype=np.float64),
            np.asarray(stand_log_likelihood, dtype=np.float64),
            np.asarray(stand_mahalanobis_sq, dtype=np.float64),
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
            width = max(abs(value_min) * 0.01, 1.0,)

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
        x, density, _ = (
            CollectStatistics.aggregate_distribution(values=values, bins=bins, log_scale=log_scale)
        )

        if density.size == 0:
            return x, density

        if smoothing_sigma > 0.0 and density.size > 3:
            density = gaussian_filter1d(density, sigma=smoothing_sigma, mode="nearest")
            # После сглаживания устраняем небольшие
            # отрицательные значения, которые теоретически
            # могут появиться из-за численной погрешности.
            density = np.maximum(density,0.0,)
        return x, density

    # ==========================================================
    # ВИЗУАЛИЗАЦИЯ
    # ==========================================================
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
        3. Вся выборка агрегируется в bins.
        4. По bins строится сглаженная плотность.
        5. Создаётся PNG высокого качества.
        6. График содержит два горизонтальных подграфика:
             - линейный масштаб;
             - логарифмический масштаб.
        7. На графиках отображаются P5, P25, P50, P75, P95.
        8. В заголовке отображаются название, математическое
           ожидание и дисперсия.
        Для положительных метрик: второй график использует log X.
        Для метрик, которые могут иметь отрицательные значения
        (например log_likelihood): второй график использует symlog X.
        Статистика рассчитывается по всем исходным значениям, агрегация применяется только к построению плотности.
        Дополнительно сохраняется summary.csv.
        Args:
            statistics: Словарь: имя метрики -> массив значений.
            output_dir: Каталог для сохранения PNG.
            bins: Количество bins для агрегации.
            smoothing_sigma: Параметр сглаживания плотности.

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

            p05, p25, p50, p75, p95 = np.percentile(values,[5, 25, 50, 75, 95])

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
            # ==================================================
            # Линейная плотность
            # ==================================================
            x_linear, density_linear = (
                CollectStatistics._build_density(
                    values=values,
                    bins=bins,
                    smoothing_sigma=smoothing_sigma,
                    log_scale=False,
                )
            )

            # ==================================================
            # Плотность для второго графика
            # ==================================================

            if strictly_positive:
                x_log, density_log = (
                    CollectStatistics._build_density(
                        values=values,
                        bins=bins,
                        smoothing_sigma=smoothing_sigma,
                        log_scale=True,
                    )
                )

                log_mode = "log"

            else:
                # Для log_likelihood обычная логарифмическая
                # шкала X невозможна.
                #
                # Строим плотность в линейных координатах,
                # а отображаем X через symlog.
                x_log = x_linear
                density_log = density_linear
                log_mode = "symlog"

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
                (
                    f"E[X] = {mean:.8g}"
                    f"    |    "
                    f"Var[X] = {variance:.8g}"
                    f"    |    "
                    f"N = {values.size:,}"
                ),
                ha="center",
                va="center",
                fontsize=12,
            )

            percentile_data = (
                (float(p05), "P5"),
                (float(p25), "P25"),
                (float(p50), "P50"),
                (float(p75), "P75"),
                (float(p95), "P95"),
            )

            # ==================================================
            # Вложенная функция построения графика
            # ==================================================

            def draw_density(
                ax,
                x_data,
                density_data,
                logarithmic_x: bool,
            ) -> None:

                positive_density = density_data > 0.0

                ax.plot(
                    x_data[positive_density],
                    density_data[positive_density],
                    linewidth=2.0,
                )

                for percentile_value, label in (
                    percentile_data
                ):
                    ax.axvline(percentile_value,
                        linestyle="--",
                        linewidth=1.1,
                        label=(f"{label} = {percentile_value:.6g}"),
                    )

                if logarithmic_x:
                    ax.set_xscale("log")
                    ax.set_title(
                        "Плотность распределения — логарифмический масштаб X",
                        fontsize=13,
                    )
                else:
                    ax.set_title(
                        "Плотность распределения — линейный масштаб X",
                        fontsize=13,
                    )
                ax.set_xlabel(metric_name, fontsize=11)
                ax.set_ylabel("Плотность", fontsize=11)
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8, loc="best")

            # ==================================================
            # Левый график
            # ==================================================
            draw_density(
                axes[0],
                x_linear,
                density_linear,
                logarithmic_x=False,
            )

            # ==================================================
            # Правый график
            # ==================================================

            if log_mode == "log":
                draw_density(axes[1], x_log, density_log, logarithmic_x=True)
            else:
                draw_density(axes[1], x_log, density_log, logarithmic_x=False)

                # Для отрицательных и положительных значений
                # используется symmetric logarithmic scale.
                scale = max(abs(p50) * 1e-3, 1e-9)

                axes[1].set_xscale("symlog", linthresh=scale)

                axes[1].set_title(
                    "Плотность распределения — симметричный логарифмический масштаб X",
                    fontsize=13,
                )

            # ==================================================
            # Отступы
            # ==================================================
            fig.subplots_adjust(
                left=0.06,
                right=0.98,
                bottom=0.12,
                top=0.84,
                wspace=0.17,
            )

            # ==================================================
            # Имя файла
            # ==================================================

            safe_name = (
                metric_name
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace(":", "_")
                .replace("*", "_")
                .replace("?", "_")
                .replace("<", "_")
                .replace(">", "_")
                .replace("|", "_")
            )

            output_path = (output_dir / f"{safe_name}.png")

            # ==================================================
            # Сохранение PNG
            # ==================================================
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
                f"file={output_path}"
            )

        # ======================================================
        # Сводная таблица
        # ======================================================

        summary_df = pd.DataFrame(summary_rows)

        summary_path = (output_dir / "summary.csv")

        summary_df.to_csv(summary_path, index=False)

        return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    path = Path("/home/ubuntu/PycharmProjects/statistical_researchs/data/1.csv")

    df = DataProcessor.load_csv(path)
    df = DataProcessor.pre_filter(df)
    lon, lat, time = DataProcessor.get_lon_lat(df)
    mark = df["status"].to_numpy()

    distances_anomaly, distances_move, distances_stand = (
        CollectStatistics.collect_distance_between_point(lon, lat, mark)
    )

    (
        anomaly_log_likelihood_CV,
        anomaly_mahalanobis_sq_CV,
        move_log_likelihood_CV,
        move_mahalanobis_sq_CV,
        stand_log_likelihood_CV,
        stand_mahalanobis_sq_CV,
    ) = CollectStatistics.collect_estimation_kalman_filter_cv(lon, lat, time, mark, window=10)

    (
        anomaly_log_likelihood_RW,
        anomaly_mahalanobis_sq_RW,
        move_log_likelihood_RW,
        move_mahalanobis_sq_RW,
        stand_log_likelihood_RW,
        stand_mahalanobis_sq_RW,
    ) = CollectStatistics.collect_estimation_kalman_filter_rw(lon, lat, time, mark, window=10)
