"""Перевод WGS-84 (широта/долгота) в локальную метрическую плоскость ENU (Отчёт.tex)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy._typing import NDArray

# WGS-84
_WGS84_SEMI_MAJOR_AXIS_M: float = 6378137.0
_WGS84_FLATTENING: float = 1.0 / 298.257223563
_WGS84_ECCENTRICITY_SQ: float = 2.0 * _WGS84_FLATTENING - _WGS84_FLATTENING**2


def _geodetic_to_ecef(
    lat_rad: NDArray[np.float64],
    lon_rad: NDArray[np.float64],
    height_m: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Преобразует геодезические координаты (широта, долгота, высота) в ECEF (Earth-Centered, Earth-Fixed).
    Args:
        lat_rad: Широта в радианах.
        lon_rad: Долгота в радианах.
        height_m: Высота над эллипсоидом, метры.

    Returns:
        Координаты в ECEF, метры.

    """
    sin_phi = np.sin(lat_rad)
    cos_phi = np.cos(lat_rad)
    sin_lam = np.sin(lon_rad)
    cos_lam = np.cos(lon_rad)
    prime_vertical = _WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQ * sin_phi**2)
    radius = prime_vertical + height_m
    x = radius * cos_phi * cos_lam
    y = radius * cos_phi * sin_lam
    z = (prime_vertical * (1.0 - _WGS84_ECCENTRICITY_SQ) + height_m) * sin_phi
    return x, y, z


def _validate_coordinates(lat: np.ndarray, lon: np.ndarray) -> None:
    """
    Проверяет корректность координат.
    Args:
        lat: Массив широт.
        lon: Массив долгот.
    """
    if lat.shape != lon.shape:
        raise ValueError("Массивы широт и долгот должны иметь одинаковую форму")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Координаты должны быть конечными числами")
    if np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("Широта должна лежать в диапазоне [-90, 90]")
    if np.any((lon < -180.0) | (lon > 180.0)):
        raise ValueError("Долгота должна лежать в диапазоне [-180, 180]")


class LocalEnuConverter:
    """Локальная система ENU: $x$ — восток ($E$), $y$ — север ($N$), метры.

    Опорная точка $(lat_0, lon_0, h_0)$ задаёт начало координат. Преобразование:
    геодезические координаты WGS-84 → ECEF → ENU относительно опоры (см. ``report/Отчёт.tex``).
    """

    def __init__(
        self,
        origin_lat: float,
        origin_lon: float,
        origin_height_m: float = 0.0,
    ) -> None:
        """
        Args:
            origin_lat: Широта опорной точки, градусы.
            origin_lon: Долгота опорной точки, градусы.
            origin_height_m: Высота над эллипсоидом в опоре, метры.
        """
        _validate_coordinates(np.asarray([origin_lat]), np.asarray([origin_lon]))
        phi0 = np.deg2rad(origin_lat)
        lam0 = np.deg2rad(origin_lon)
        x0, y0, z0 = _geodetic_to_ecef(
            np.asarray(phi0, dtype=np.float64),
            np.asarray(lam0, dtype=np.float64),
            np.asarray(origin_height_m, dtype=np.float64),
        )
        self._origin_ecef = (float(x0), float(y0), float(z0))

        sin_phi0 = float(np.sin(phi0))
        cos_phi0 = float(np.cos(phi0))
        sin_lam0 = float(np.sin(lam0))
        cos_lam0 = float(np.cos(lam0))
        self._east_row = np.array([-sin_lam0, cos_lam0, 0.0], dtype=np.float64)
        self._north_row = np.array(
            [-sin_phi0 * cos_lam0, -sin_phi0 * sin_lam0, cos_phi0],
            dtype=np.float64,
        )

    @classmethod
    def from_first_point(
        cls,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        height_m: Optional[Union[float, np.ndarray]] = None,
    ) -> LocalEnuConverter:
        """Создаёт экземпляр класса из координат первой точки.

        Args:
            lat: Широта первой точки, градусы.
            lon: Долгота первой точки, градусы.
            height_m: Высота над эллипсоидом в первой точке, метры.

        Returns:
            Экземпляр класса LocalEnuConverter.
        """
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)
        if lat_arr.size == 0:
            raise ValueError("Ряд координат не должен быть пустым")
        h0 = 0.0
        if height_m is not None:
            h_arr = np.asarray(height_m, dtype=np.float64)
            h0 = float(h_arr.ravel()[0])
        return cls(float(lat_arr.ravel()[0]), float(lon_arr.ravel()[0]), h0)

    @property
    def origin_ecef(self) -> Tuple[float, float, float]:
        """
        Координаты опоры в ECEF, метры.
        Returns:
            Кортеж из трех координат (x, y, z) в ECEF.
        """
        return self._origin_ecef

    def to_local_xy(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        height_m: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Переводит широту и долготу в локальные $x$ (восток) и $y$ (север), метры.

        Args:
            lat: Широта в градусах.
            lon: Долгота в градусах.
            height_m: Высота над эллипсоидом; по умолчанию 0.

        Returns:
            Пара массивов ``x``, ``y`` той же формы, что ``lat`` и ``lon``.
        """
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)
        _validate_coordinates(lat_arr, lon_arr)
        if height_m is None:
            h_arr = np.zeros_like(lat_arr, dtype=np.float64)
        else:
            h_arr = np.broadcast_to(np.asarray(height_m, dtype=np.float64), lat_arr.shape).copy()

        phi = np.deg2rad(lat_arr)
        lam = np.deg2rad(lon_arr)
        x_ecef, y_ecef, z_ecef = _geodetic_to_ecef(phi, lam, h_arr)
        dx = x_ecef - self._origin_ecef[0]
        dy = y_ecef - self._origin_ecef[1]
        dz = z_ecef - self._origin_ecef[2]
        stack = np.stack((dx, dy, dz), axis=0)
        east = self._east_row @ stack
        north = self._north_row @ stack
        return east, north


if __name__ == "__main__":
    # Демо: опора в первой точке, смещение ~1 км на северо-восток
    lat0, lon0 = 43.04, 131.87
    lat = np.array([lat0, lat0 + 0.009, lat0 + 0.009])
    lon = np.array([lon0, lon0, lon0 + 0.012])
    converter = LocalEnuConverter.from_first_point(lat, lon)
    x, y = converter.to_local_xy(lat, lon)
    print("Локальные координаты (м), опора — первая точка:")
    for index in range(lat.size):
        print(f"  i={index}: x={x[index]:.2f}, y={y[index]:.2f}")
    print(f"  опора ECEF: {converter.origin_ecef}")
