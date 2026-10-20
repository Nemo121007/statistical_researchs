"""Ряд траектории для расчёта метрик."""

from __future__ import annotations

import numpy as np


class TrajectorySeries:
    """Один ряд траектории: время, координаты и метка класса.

    Метка 0 — аномалия, значение больше нуля — штатное состояние.

    Attributes:
        time: Одномерный массив времени.
        lon: Долгота в градусах.
        lat: Широта в градусах.
        label: Метка класса в каждой точке.
    """

    def __init__(
        self,
        time: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        label: np.ndarray,
    ) -> None:
        """
        Args:
            time: Метки времени.
            lon: Долготы.
            lat: Широты.
            label: Метки; 0 — аномалия.

        Raises:
            ValueError: Некорректная форма, длина, координаты или метки.
        """
        self.time = np.asarray(time)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.lat = np.asarray(lat, dtype=np.float64)
        self.label = np.asarray(label, dtype=np.float64)
        self._validate()

    @property
    def anomaly_mask(self) -> np.ndarray:
        """True в точках, где label == 0."""
        return np.equal(self.label, 0.0)

    @property
    def time_seconds(self) -> np.ndarray:
        """Время ряда в секундах (float64)."""
        return self.time_to_seconds(self.time)

    def assert_same_index(self, other: TrajectorySeries) -> None:
        """Проверяет, что с другим рядом совпадает число точек.

        Args:
            other: Второй ряд (prediction или ground truth).

        Raises:
            ValueError: Разная длина рядов.
        """
        if self.time.shape != other.time.shape:
            raise ValueError("prediction и ground truth должны содержать одинаковое число точек.")

    def validate_coordinates(self) -> None:
        """Проверяет lat и lon на допустимый диапазон.

        Raises:
            ValueError: Неконечные значения или выход за пределы.
        """
        if not np.all(np.isfinite(self.lat)) or not np.all(np.isfinite(self.lon)):
            raise ValueError("Координаты должны быть конечными числами.")
        if np.any((self.lat < -90.0) | (self.lat > 90.0)):
            raise ValueError("Широта должна лежать в диапазоне [-90, 90].")
        if np.any((self.lon < -180.0) | (self.lon > 180.0)):
            raise ValueError("Долгота должна лежать в диапазоне [-180, 180].")

    @staticmethod
    def time_to_seconds(time: np.ndarray) -> np.ndarray:
        """Приводит массив времени к float-секундам.

        Args:
            time: Одномерный массив (числа, datetime64 или строки дат).

        Returns:
            Массив float64.

        Raises:
            ValueError: Пустой массив или неконечные значения.
        """
        values = np.asarray(time)
        if values.size == 0:
            raise ValueError("Ряд не должен быть пустым.")
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]").astype(np.float64) / 1e9
        if values.dtype.kind in {"U", "S", "O"}:
            return values.astype("datetime64[ns]").astype(np.float64) / 1e9
        seconds = values.astype(np.float64)
        if not np.all(np.isfinite(seconds)):
            raise ValueError("Время должно состоять из конечных чисел или дат.")
        return seconds

    def _validate(self) -> None:
        if self.time.ndim != 1 or self.lon.ndim != 1 or self.lat.ndim != 1 or self.label.ndim != 1:
            raise ValueError("time, lon, lat и label должны быть одномерными массивами.")
        if not self.time.shape == self.lon.shape == self.lat.shape == self.label.shape:
            raise ValueError("time, lon, lat и label должны иметь одинаковую длину.")
        if self.time.size == 0:
            raise ValueError("Ряд не должен быть пустым.")
        if not np.all(np.isfinite(self.label)):
            raise ValueError("Метки должны быть конечными числами.")
        self.validate_coordinates()
