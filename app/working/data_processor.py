from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Предполагаем, что этот импорт корректен в структуре вашего проекта
from app.help_scripts.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle


class DataProcessor:
    """Класс для подготовки и обработки исходных данных."""

    LEN_LAT = 111132.0  # Длина одного градуса широты в метрах

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        """Загружает данные из CSV файла."""
        return pd.read_csv(path)

    def parse_intervals(self, df: pd.DataFrame, n: int = 300, distance_threshold: float = 500)\
            -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Разбивает DataFrame на списки валидных и невалидных интервалов.
        """
        list_valid_df = self._extend_intervals(df, target_point=1, n=n, distance_threshold=distance_threshold)
        list_invalid_df = self._extend_intervals(df, target_point=-1, n=n, distance_threshold=distance_threshold)
        return list_valid_df, list_invalid_df

    def _extend_intervals(self, df: pd.DataFrame, target_point: int = 1, n: int = 300,
                          distance_threshold: float = 500) -> List[pd.DataFrame]:
        """
        Внутренняя логика разбиения на интервалы с проверкой дистанции.
        """
        if df.empty:
            return []

        clean_df = df[df["validate_point"] == target_point].copy()
        if clean_df.empty:
            return []

        time_diff = clean_df["time"].diff()
        split_mask = time_diff > 10
        group_ids = split_mask.cumsum()

        result_list = []

        for _, group in clean_df.groupby(group_ids):
            if len(group) <= 1:
                continue

            # Логика обработки группы (разбиение на чанки и проверка дистанции)
            groups_to_process = []
            if len(group) > n:
                for i in range(0, len(group), n):
                    groups_to_process.append(group.iloc[i:i + n])
            else:
                groups_to_process.append(group)

            for chunk in groups_to_process:
                lon, lat = self.get_lon_lat(chunk)
                if len(lon) < 2:
                    continue

                # Проверка дистанции
                lon_ends = np.array([lon[0], lon[-1]])
                lat_ends = np.array([lat[0], lat[-1]])
                distance = float(
                    np.nansum(CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_ends, lon_ends)))

                if distance < distance_threshold:
                    continue

                result_list.append(chunk)

        return result_list

    @staticmethod
    def get_lon_lat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Извлекает массивы долгот и широт."""
        return df['lon'].to_numpy(), df['lat'].to_numpy()

    def convert_to_local_cartesian(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Переводит сферические координаты (lon, lat) в локальные прямоугольные (x, y) в метрах.
        """
        if len(lat) == 0:
            return np.array([]), np.array([])

        lat0 = lat[0]
        lon0 = lon[0]

        ky = self.LEN_LAT
        lat0_rad = np.radians(lat0)
        kx = self.LEN_LAT * np.cos(lat0_rad)

        x_local = (lon - lon0) * kx
        y_local = (lat - lat0) * ky

        return x_local, y_local

    @staticmethod
    def visualize_and_save(x_true: np.ndarray, y_true: np.ndarray,
                           x_filt: np.ndarray, y_filt: np.ndarray,
                           save_path: Path):
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
        plt.plot(x_true, y_true, c='blue', label=f'Исходные данные ({num_points} точек)', alpha=0.6)
        plt.plot(x_filt, y_filt, 'r--', label='Фильтр Калмана', linewidth=2)

        plt.xlabel('X (метры)')
        plt.ylabel('Y (метры)')
        plt.title(f'Траектория: {save_path.stem} | Точек: {num_points}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def process_track_list(df_list: List[pd.DataFrame],
                           save_dir: Path,
                           processor,  # DataProcessor
                           kalman_filter,  # Union[KalmanFilterCV, KalmanFilterRW]
                           label: str):
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
            time = df['time'].to_numpy()

            # Вызываем метод filter, который есть и у KalmanFilterCV, и у KalmanFilterRW
            x_filt, y_filt = kalman_filter.filter(x, y, time)

            save_path = save_dir / f'track_{i}.png'
            DataProcessor.visualize_and_save(x, y, x_filt, y_filt, save_path)
