"""Модуль для вычисления расстояний по большому кругу на сфере Земли.
Содержит класс CalculatorDistancesLengthLargeCircle для точного вычисления
географических расстояний между точками с использованием формул гаверсинуса
"""

from typing import Union

import numpy as np
from numpy._typing import NDArray

from ....core.config import CorrectorSettings


class CalculatorDistancesLengthLargeCircle:
    """Калькулятор расстояний по большому кругу на сфере Земли.
    Предоставляет методы для вычисления расстояний между точками
    с использованием длины большого круга (гаверсинус).
    """

    @staticmethod
    def vectorized_min_distance_to_points(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Вычисляет расстояния от точки до набора точек.
        Args:
            lat1: Широта точки в градусах [-90, 90].
            lon1: Долгота точки в градусах [-180, 180].
            lat2: Массив широт точек в градусах [-90, 90].
            lon2: Массив долгот точек в градусах [-180, 180].
        Returns:
            Массив расстояний в метрах от точки до каждой точки из набора.
        Raises:
            ValueError: Если координаты выходят за допустимые пределы.
        """
        # Валидация входных данных
        CalculatorDistancesLengthLargeCircle._validate_coordinates(lat1, lon1)
        CalculatorDistancesLengthLargeCircle._validate_coordinates(lat2, lon2)

        if lat2.shape != lon2.shape:
            raise ValueError("Массивы широт и долгот должны иметь одинаковую форму")

        # Переводим все координаты в радианы
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        return CalculatorDistancesLengthLargeCircle.haversine_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad)

    @staticmethod
    def vectorized_segment_distances(lat_array: np.ndarray, lon_array: np.ndarray) -> NDArray[np.float64]:
        """Векторизованное вычисление расстояний между последовательными точками сегмента.
        Args:
            lat_array: Массив широт сегмента формы [N].
            lon_array: Массив долгот сегмента формы [N].
        Returns:
            Массив расстояний между последовательными точками формы [N-1].
        Raises:
            ValueError: Если координаты некорректны, сегмент слишком короткий
                        или массивы широт и долгот имеют разную длину.
        """
        if lat_array.shape != lon_array.shape:
            raise ValueError("Массивы широт и долгот должны иметь одинаковую форму")

        if lat_array.shape[0] < 2:
            raise ValueError("Сегмент должен содержать хотя бы 2 точки")

        CalculatorDistancesLengthLargeCircle._validate_coordinates(lat_array, lon_array)

        # Все точки, кроме последней
        lat1 = lat_array[:-1]
        lon1 = lon_array[:-1]

        # Все точки, кроме первой
        lat2 = lat_array[1:]
        lon2 = lon_array[1:]

        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        return CalculatorDistancesLengthLargeCircle.haversine_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad)

    @staticmethod
    def vectorized_great_circle_distance(
        lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
    ) -> np.ndarray[np.float64]:
        """Вычисляет расстояния по большому кругу между массивами точек.
        Args:
            lat1: Массив широт первых точек в градусах [-90, 90].
            lon1: Массив долгот первых точек в градусах [-180, 180].
            lat2: Массив широт вторых точек в градусах [-90, 90].
            lon2: Массив долгот вторых точек в градусах [-180, 180].
        Returns:
            Массив расстояний в метрах между соответствующими парами точек.
        Raises:
            ValueError: Если координаты некорректны или массивы имеют разную форму.
        """
        # Валидация входных данных
        CalculatorDistancesLengthLargeCircle._validate_coordinates(lat1, lon1)
        CalculatorDistancesLengthLargeCircle._validate_coordinates(lat2, lon2)

        if lat1.shape != lat2.shape or lon1.shape != lon2.shape:
            raise ValueError("Все массивы должны иметь одинаковую форму")

        # Переводим координаты в радианы
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        return CalculatorDistancesLengthLargeCircle.haversine_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad)

    @staticmethod
    def haversine_distance(
        lat1_rad: np.ndarray,
        lon1_rad: np.ndarray,
        lat2_rad: np.ndarray,
        lon2_rad: np.ndarray,
    ) -> NDArray[np.float64]:
        """Вычисляет расстояние по формуле гаверсинуса.
        Args:
            lat1_rad: Широты первой точки в радианах
            lon1_rad: Долготы первой точки в радианах
            lat2_rad: Широты второй точки в радианах
            lon2_rad: Долготы второй точки в радианах
        Returns:
            Расстояния в метрах
        """
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return CorrectorSettings.Earth_radius_meters * c

    @staticmethod
    def _validate_coordinates(
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> None:
        """Проверяет корректность географических координат.
        Args:
            lat: Широта(ы)
            lon: Долгота(ы)
            name: Имя переменной для сообщений об ошибках
        Raises:
            ValueError: Если координаты некорректны
        """
        # Приводим к массивам для единообразной обработки
        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = np.asarray(lon, dtype=float)

        # Создаем маски для валидных (не NaN) значений
        lat_valid = ~np.isnan(lat_arr)
        lon_valid = ~np.isnan(lon_arr)

        # Проверяем диапазоны только для валидных значений
        if np.any(lat_valid) and (np.any(lat_arr[lat_valid] < -90) or np.any(lat_arr[lat_valid] > 90)):
            raise ValueError(f"{(lon_arr, lat_arr)}: Широта должна быть в диапазоне [-90, 90]")

        if np.any(lon_valid) and (np.any(lon_arr[lon_valid] < -180) or np.any(lon_arr[lon_valid] > 180)):
            raise ValueError(f"{(lon_arr, lat_arr)}: Долгота должна быть в диапазоне [-180, 180]")
