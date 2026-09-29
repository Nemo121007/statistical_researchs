import datetime
import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from app.help_scripts.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle


class DataProcessor:
    """Класс для подготовки и обработки исходных данных."""

    LEN_LAT = 111132.0  # Длина одного градуса широты в метрах

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        """Загружает данные из CSV файла."""
        return pd.read_csv(path, parse_dates=["time"])

    @staticmethod
    def pre_filter(df: DataFrame) -> DataFrame:
        """
        Предварительная фильтрация данных:
        - Удаляет точки с sat < 3.
        - Удаляет точки с is_water == False.
        - Удаляет точки с NULL/NaN в lon или lat.
        """
        mask = df[["lon", "lat"]].isna().any(axis=1).sum()
        logging.debug("Удалено %d точек с NULL/NaN в lon/lat", mask)
        df = df.dropna(subset=["lon", "lat"])

        mask = df["sat"] >= 3
        logging.debug("Удалено %d точек с sat < 3", (~mask).sum())
        df = df[mask]

        mask = df["is_water"]
        logging.debug(
            "Удалено %d точек с is_water == False",
            (~mask).sum(),
        )
        df = df[mask]
        return df

    @staticmethod
    def parse_intervals(
        df: pd.DataFrame,
        max_point_in_interval: int = 300,
        min_point_in_interval: int = 10,
        distance_threshold: float = 500,
    ) -> Tuple[
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
    ]:
        """
        Разбивает трек на интервалы типов anomaly, stand и move.

        Алгоритм:

        1. По изменению status формируются непрерывные участки.
        2. Участки распределяются по типам:
           anomaly / stand / move.
        3. Каждый участок дополнительно разделяется:
           - каждые max_point_in_interval точек;
           - при расстоянии между соседними точками > distance_threshold.
        4. Интервалы размером менее min_point_in_interval удаляются.

        Все операции поиска границ выполняются векторно.
        """

        if df.empty:
            return [], [], []

        if max_point_in_interval < 1:
            raise ValueError("max_point_in_interval должен быть >= 1.")

        if min_point_in_interval < 1:
            raise ValueError("min_point_in_interval должен быть >= 1.")

        if max_point_in_interval < min_point_in_interval:
            raise ValueError("max_point_in_interval должен быть >= min_point_in_interval.")

        if distance_threshold <= 0:
            raise ValueError("distance_threshold должен быть > 0.")

        status_changed = df["status"].ne(df["status"].shift())
        interval_ids = status_changed.cumsum()

        list_anomaly_df: List[pd.DataFrame] = []
        list_stand_df: List[pd.DataFrame] = []
        list_move_df: List[pd.DataFrame] = []

        for _, group in df.groupby(interval_ids, sort=False):
            if group.empty:
                continue

            status = group["status"].iloc[0]

            if status not in {"anomaly", "stand", "move"}:
                continue

            if len(group) < min_point_in_interval:
                continue

            group = group.copy()

            lons = group["lon"].to_numpy(dtype=float)
            lats = group["lat"].to_numpy(dtype=float)

            group_size = len(group)
            if group_size > 1:
                lat_array = lats
                lon_array = lons
                distances = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_array, lon_array)
                distances = np.asarray(distances, dtype=float)
                distance_split = distances > distance_threshold
            else:
                distance_split = np.empty(0, dtype=bool)

            max_points_split = np.arange(group_size) % max_point_in_interval == 0
            max_points_split[0] = False
            split_mask = max_points_split.copy()

            if group_size > 1:
                split_mask[1:] |= distance_split
            subinterval_ids = split_mask.cumsum()

            for _, subinterval in group.groupby(subinterval_ids, sort=False):
                if len(subinterval) < min_point_in_interval:
                    continue

                if status == "anomaly":
                    list_anomaly_df.append(subinterval.copy())
                elif status == "stand":
                    list_stand_df.append(subinterval.copy())
                else:
                    list_move_df.append(subinterval.copy())

        return list_anomaly_df, list_stand_df, list_move_df

    @staticmethod
    def get_lon_lat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Извлекает массивы долгот, широт и времени (time как np.datetime64)."""
        time = pd.to_datetime(df["time"]).to_numpy()
        return df["lon"].to_numpy(), df["lat"].to_numpy(), time

    @staticmethod
    def convert_to_local_cartesian(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Переводит сферические координаты (lon, lat) в локальные прямоугольные (x, y) в метрах.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if len(lat) == 0:
            return np.array([]), np.array([])

        valid_mask = ~np.isnan(lat) & ~np.isnan(lon)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.full_like(lon, np.nan, dtype=float), np.full_like(lat, np.nan, dtype=float)

        first_valid_idx = valid_indices[0]
        lat0 = lat[first_valid_idx]
        lon0 = lon[first_valid_idx]

        ky = DataProcessor.LEN_LAT
        lat0_rad = np.radians(lat0)
        kx = DataProcessor.LEN_LAT * np.cos(lat0_rad)

        x_local = (lon - lon0) * kx
        y_local = (lat - lat0) * ky

        return x_local, y_local

    @staticmethod
    def visualize_and_save(
        x_true: np.ndarray,
        y_true: np.ndarray,
        x_filt: np.ndarray,
        y_filt: np.ndarray,
        save_path: Path,
    ):
        """
        Строит и сохраняет график сравнения исходной и сглаженной траекторий.

        На графике отображаются:
        - Исходная траектория (синяя сплошная линия).
        - Сглаженная траектория (красная пунктирная линия).
        - Количество точек данных в заголовке.

        Args:
            x_true: Массив координат X исходной траектории (в метрах).
            y_true: Массив координат Y исходной траектории (в метрах).
            x_filt: Массив координат X сглаженной траектории (в метрах).
            y_filt: Массив координат Y сглаженной траектории (в метрах).
            save_path: Путь (объект pathlib.Path) для сохранения файла изображения.

        Returns:
            None. Функция сохраняет изображение на диск и закрывает фигуру.
        """
        num_points = len(x_true)

        plt.figure(figsize=(10, 8))
        plt.plot(
            x_true,
            y_true,
            c="blue",
            label=f"Исходные данные ({num_points} точек)",
            alpha=0.6,
        )
        plt.plot(x_filt, y_filt, "r--", label="Фильтр Калмана", linewidth=2)

        plt.xlabel("X (метры)")
        plt.ylabel("Y (метры)")
        plt.title(f"Траектория: {save_path.stem} | Точек: {num_points}")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def process_track_list(
        df_list: List[pd.DataFrame],
        save_dir: Path,
        processor,  # DataProcessor
        kalman_filter,  # Union[KalmanFilterCV, KalmanFilterRW]
        label: str,
    ):
        """
        Обрабатывает список треков: фильтрует и сохраняет визуализацию.

        Args:
            df_list: Список DataFrame для обработки.
            save_dir: Папка для сохранения картинок.
            processor: Экземпляр класса DataProcessor.
            kalman_filter: Экземпляр класса фильтра (должен иметь метод filter).
            label: Название типа данных (для логирования).
        """
        print(f"Обработка {label}: {len(df_list)} шт.")

        for i, df in enumerate(df_list):
            lon, lat = processor.get_lon_lat(df)
            x, y = processor.convert_to_local_cartesian(lon, lat)
            time = df["time"].to_numpy()

            # Вызываем метод filter, который есть и у KalmanFilterCV, и у KalmanFilterRW
            x_filt, y_filt = kalman_filter.filter(x, y, time)

            save_path = save_dir / f"track_{i}.png"
            DataProcessor.visualize_and_save(x, y, x_filt, y_filt, save_path)

    @staticmethod
    def plot_array_and_hist(arr1, arr2=None, name="", bins=100, save_path: Path = None):
        """Метод, визуализирующий массивы в виде графиков и гистограмм (2x2).

        Args:
            arr1: первый массив значений
            arr2: второй массив значений (опционально)
            name: название графика
            bins: количество интервалов для гистограммы
            save_path: путь для сохранения визуализации
        """
        arr1_np = np.asarray(arr1).flatten()
        clean_arr1 = arr1_np[~np.isnan(arr1_np)]

        clean_arr2 = np.array([])
        if arr2 is not None:
            arr2_np = np.asarray(arr2).flatten()
            clean_arr2 = arr2_np[~np.isnan(arr2_np)]

        if len(clean_arr1) == 0 and len(clean_arr2) == 0:
            print("Предупреждение: оба массива пусты или содержат только NaN. Графики не строятся.")
            return

        # --- СОЗДАНИЕ ГРАФИКОВ ---
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Графики и гистограммы \n {name}\n"
            f"Численных измерений: log_likehood ({len(clean_arr1)}), mahalanobis_sq ({len(clean_arr2)})",
            fontsize=14,
        )

        # --- Верхний левый: Линейный график первого массива ---
        if len(clean_arr1) > 0:
            axs[0, 0].plot(clean_arr1, linewidth=1.5, color="blue")
            axs[0, 0].set_title("Линейный график (log_likehood)")
            axs[0, 0].set_xlabel("Индекс")
            axs[0, 0].set_ylabel("Значение")
            axs[0, 0].grid(True)

            # --- Верхний правый: Гистограмма первого массива ---
            axs[0, 1].hist(clean_arr1, bins=bins, edgecolor="black", alpha=0.7, color="blue")
            axs[0, 1].set_title(f"Гистограмма частот (log_likehood, интервалы: {bins})", fontsize=10)
            axs[0, 1].set_xlabel("Значение")
            axs[0, 1].set_ylabel("Частота")
            axs[0, 1].grid(True, linestyle="--", alpha=0.6, axis="y")

        # --- Нижний левый: Линейный график второго массива ---
        if len(clean_arr2) > 0:
            axs[1, 0].plot(clean_arr2, linewidth=1.5, color="green")
            axs[1, 0].set_title("Линейный график (mahalanobis_sq)")
            axs[1, 0].set_xlabel("Индекс")
            axs[1, 0].set_ylabel("Значение")
            axs[1, 0].grid(True)

            # --- Нижний правый: Гистограмма второго массива ---
            axs[1, 1].hist(clean_arr2, bins=bins, edgecolor="black", alpha=0.7, color="green")
            axs[1, 1].set_title(f"Гистограмма частот (mahalanobis_sq, интервалы: {bins})", fontsize=10)
            axs[1, 1].set_xlabel("Значение")
            axs[1, 1].set_ylabel("Частота")
            axs[1, 1].grid(True, linestyle="--", alpha=0.6, axis="y")

        plt.tight_layout()

        # --- Логика сохранения или отображения ---
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    # pylint: disable=cyclic-import
    from app.working.kalman_filter_cv import KalmanFilterCV
    from app.working.kalman_filter_rw import KalmanFilterRW

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    path = Path(__file__).parent.parent.parent / "data" / "1.csv"
    df = DataProcessor.load_csv(path)
    df = DataProcessor.pre_filter(df)
    df = df[:100000]

    list_anomaly_df, list_stand_df, list_move_df = DataProcessor.parse_intervals(df)
    logging.debug(list_anomaly_df[-1])
