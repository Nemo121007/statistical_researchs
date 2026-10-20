"""Расстояния по большому кругу (формула гаверсинуса) на сфере Земли."""

from typing import Union

import numpy as np
from numpy._typing import NDArray

EARTH_RADIUS_M: float = 6371000.0


def segment_distances_m(lat: np.ndarray, lon: np.ndarray) -> NDArray[np.float64]:
    """Расстояния между последовательными точками линии в метрах.

    Args:
        lat: Широты точек в градусах [-90, 90].
        lon: Долготы точек в градусах [-180, 180].

    Returns:
        Расстояния между соседними парами точек.

    Raises:
        ValueError: Некорректные координаты, N < 2 или разная длина lat и lon.
    """
    if lat.shape != lon.shape:
        raise ValueError("Массивы широт и долгот должны иметь одинаковую форму")

    if lat.shape[0] < 2:
        raise ValueError("Сегмент должен содержать хотя бы 2 точки")

    _validate_coordinates(lat=lat, lon=lon)

    return great_circle_distances_m(lat[:-1], lon[:-1], lat[1:], lon[1:])


def great_circle_distances_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> NDArray[np.float64]:
    """Расстояния по большому кругу между соответствующими парами точек, в метрах.

    Args:
        lat1: Широты первых точек в градусах [-90, 90].
        lon1: Долготы первых точек в градусах [-180, 180].
        lat2: Широты вторых точек в градусах [-90, 90].
        lon2: Долготы вторых точек в градусах [-180, 180].

    Returns:
        Расстояния в метрах; форма совпадает с формой входных массивов.

    Raises:
        ValueError: Некорректные координаты или несовпадающие формы массивов.
    """
    _validate_coordinates(lat1, lon1)
    _validate_coordinates(lat2, lon2)

    if lat1.shape != lat2.shape or lon1.shape != lon2.shape:
        raise ValueError("Все массивы должны иметь одинаковую форму")

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_M * c


def _validate_coordinates(
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
) -> None:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)

    lat_valid = ~np.isnan(lat_arr)
    lon_valid = ~np.isnan(lon_arr)

    if np.any(lat_valid) and (np.any(lat_arr[lat_valid] < -90) or np.any(lat_arr[lat_valid] > 90)):
        raise ValueError(f"{(lon_arr, lat_arr)}: Широта должна быть в диапазоне [-90, 90]")

    if np.any(lon_valid) and (np.any(lon_arr[lon_valid] < -180) or np.any(lon_arr[lon_valid] > 180)):
        raise ValueError(f"{(lon_arr, lat_arr)}: Долгота должна быть в диапазоне [-180, 180]")
