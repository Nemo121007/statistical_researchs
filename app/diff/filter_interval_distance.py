import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6_371_000.0  # метры
DISTANCE_SPLIT_THRESHOLD = 500.0  # метры
SPEED_THRESHOLD = 20.0  # м/с


def _haversine_distance_m(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float,
) -> float:
    """
    Метод, рассчитывающий расстояние между двумя точками
    Args:
        lat1_rad: широта первой точки (в радианах)
        lon1_rad: долгота первой точки (в радианах)
        lat2_rad: широта второй точки (в радианах)
        lon2_rad: долгота второй точки (в радианах)
    Returns:
        Расстояние между двумя точками (в метрах)
    """
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS * c)


def _segment_distances_m(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """

    Args:
        lat_rad:
        lon_rad:

    Returns:

    """
    if len(lat_rad) < 2:
        return np.array([], dtype=float)

    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS * c


def _get_distance_intervals(
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    distance_threshold: float,
) -> list:
    """Вспомогательный метод для получения интервалов по расстоянию."""
    lat_finite_rad = np.radians(lat_finite)
    lon_finite_rad = np.radians(lon_finite)

    segment_distances = _segment_distances_m(lat_finite_rad, lon_finite_rad)
    split_index = np.flatnonzero(segment_distances > distance_threshold) + 1

    chunks = np.split(np.arange(len(lat_finite)), split_index)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size > 0]


def _apply_speed_reachability_mask(
    intervals: list,
    validate_mask: np.ndarray,
    finite_idx: np.ndarray,
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
) -> None:
    """Вспомогательный метод для применения маски на основе достижимости."""
    if not intervals:
        return

    first_start, first_end = intervals[0]
    validate_mask[finite_idx[first_start : first_end + 1]] = 1
    last_valid_end = first_end

    for start, end in intervals[1:]:
        dt = time_finite[start] - time_finite[last_valid_end]

        if not np.isfinite(dt) or dt <= 0:
            continue

        bridge_dist = _haversine_distance_m(
            np.radians(lat_finite[last_valid_end]),
            np.radians(lon_finite[last_valid_end]),
            np.radians(lat_finite[start]),
            np.radians(lon_finite[start]),
        )

        if bridge_dist / dt < speed_threshold:
            validate_mask[finite_idx[start : end + 1]] = 1
            last_valid_end = end


def filter_distance_intervals_and_speed(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    distance_threshold: float = DISTANCE_SPLIT_THRESHOLD,
    speed_threshold: float = SPEED_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, производящий разделение интервалов по расстоянию между точками интервала
    Args:
        lon: массив долгот
        lat: массив широт
        time: массив временных меток
        distance_threshold: предельное удаление точек одного интервала (в метрах)
        speed_threshold: предельная скорость объекта
    Returns:
        Массив долгот, Массив широт, Массив временных меток, Массив флагов валидности точек
    """
    if lon.shape != lat.shape or lon.shape != time.shape:
        raise ValueError("lon, lat и time должны иметь одинаковую длину")

    n = len(lon)
    if n == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=np.int8),
        )

    lon_copy = np.asarray(lon, dtype=float).copy()
    lat_copy = np.asarray(lat, dtype=float).copy()
    time_copy = np.asarray(time, dtype=float).copy()

    finite_mask = np.isfinite(lon_copy) & np.isfinite(lat_copy) & np.isfinite(time_copy)
    finite_idx = np.flatnonzero(finite_mask)

    validate_mask = np.zeros(n, dtype=np.int8)

    if finite_idx.size == 0:
        return lon_copy, lat_copy, time_copy, validate_mask

    if finite_idx.size == 1:
        validate_mask[finite_idx[0]] = 1
        return lon_copy, lat_copy, time_copy, validate_mask

    lon_finite = lon_copy[finite_mask]
    lat_finite = lat_copy[finite_mask]
    time_finite = time_copy[finite_mask]

    intervals = _get_distance_intervals(lat_finite, lon_finite, distance_threshold)

    _apply_speed_reachability_mask(
        intervals,
        validate_mask,
        finite_idx,
        lat_finite,
        lon_finite,
        time_finite,
        speed_threshold,
    )

    return lon_copy, lat_copy, time_copy, validate_mask


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"
    list_time = []

    processor = DataProcessor()
    df = processor.load_csv(data_path)

    for i in range(15):
        lon_df, lat_df, time_df = processor.get_lon_lat(df)

        start_time = time.time()
        check_lon, check_lat, check_time, check_validate_point = filter_distance_intervals_and_speed(
            lon_df,
            lat_df,
            time_df,
            distance_threshold=500,
            speed_threshold=20.0,
        )
        end_time = time.time()
        execution_time = end_time - start_time
        list_time.append(execution_time)

        control_df = pd.DataFrame(
            {
                "lon": lon_df,
                "lat": lat_df,
                "time": time_df,
                "validate_point": df["validate_point"].to_numpy(),
                "in_water": df["in_water"].to_numpy(),
            }
        )

        experimental_df = pd.DataFrame(
            {
                "lon": check_lon,
                "lat": check_lat,
                "time": check_time,
                "validate_point": check_validate_point,
            }
        )
        print(i)

        if i == 14:
            CalculatingStatistics.calculate_statistics(experimental_df, control_df, np.array(list_time))

            mask = df["validate_point"].to_numpy() == 1
            lon_df = np.where(mask, lon_df, np.nan)
            lat_df = np.where(mask, lat_df, np.nan)

            mask = check_validate_point == 1
            check_lon = np.where(mask, check_lon, np.nan)
            check_lat = np.where(mask, check_lat, np.nan)

            step = 100000
            path_dir = Path(__file__).parent.parent.parent
            for i in range(0, len(lon_df), step):
                end_i = i + step
                lon = lon_df[i:end_i]
                lat = lat_df[i:end_i]
                time = time_df[i:end_i]
                number = i // step
                path = path_dir / f"control_{number}.geojson"
                IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])

            for i in range(0, len(lon_df), step):
                end_i = i + step
                lon = check_lon[i:end_i]
                lat = check_lat[i:end_i]
                time = check_time[i:end_i]
                number = i // step
                path = path_dir / f"experiment_{number}.geojson"
                IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])
