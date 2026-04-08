import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6371000.0  # meters
SPEED_THRESHOLD = 20.0  # m/s
MAX_POINTS_IN_INTERVAL = 180


def haversine_single(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Метод, вычисляющий расстояние между двумя точками
    Args:
        lat1: широта первой точки
        lon1: долгота первой точки
        lat2: широта второй точки
        lon2: долгота второй точки
    Returns:
        Расстояние между двумя точками
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS * c)


def segment_distances(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """
    Метод, вычисляющий расстояния между последовательными точками
    Args:
        lat_rad: широты точек (в рад.)
        lon_rad: долготы точек (в рад.)
    Returns:
        Массив расстояний между последовательными точками (в метрах)
    """
    if len(lat_rad) < 2:
        return np.array([], dtype=float)

    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS * c


def _split_interval_by_max_points(
    start: int, end: int, max_points: int
) -> List[Tuple[int, int]]:
    """
    Метод, разделяющий интервалы по максимальному количеству точек
    Args:
        start: Стартовый индекс
        end: Конечный индекс
        max_points: Максимальное количество точек в интервале
    Returns:
        Список кортежей (start, end) для каждого подинтервала
    """
    if end < start:
        return []

    result: List[Tuple[int, int]] = []
    cur = start

    while cur <= end:
        cur_end = min(cur + max_points - 1, end)
        result.append((cur, cur_end))
        cur = cur_end + 1

    return result


def _get_sub_intervals(
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
    max_points_in_interval: int,
) -> List[Tuple[int, int]]:
    """Вспомогательный метод для разбиения трека на подинтервалы."""
    lat_rad = np.radians(lat_finite)
    lon_rad = np.radians(lon_finite)

    dist = segment_distances(lat_rad, lon_rad)
    dt = np.diff(time_finite)

    speed = np.full_like(dist, np.inf, dtype=float)
    valid_dt_mask = np.isfinite(dt) & (dt > 0)
    speed[valid_dt_mask] = dist[valid_dt_mask] / dt[valid_dt_mask]

    split_index = np.flatnonzero(speed > speed_threshold) + 1
    chunks = np.split(np.arange(len(time_finite)), split_index)

    speed_intervals = [
        (int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size > 0
    ]

    sub_intervals: List[Tuple[int, int]] = []
    for start, end in speed_intervals:
        sub_intervals.extend(
            _split_interval_by_max_points(start, end, max_points_in_interval)
        )

    return sub_intervals


def _apply_reachability_mask(
    sub_intervals: List[Tuple[int, int]],
    validate_mask: np.ndarray,
    finite_idx: np.ndarray,
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
) -> None:
    """Применяет маску валидности на основе достижимости интервалов."""
    if not sub_intervals:
        return

    first_start, first_end = sub_intervals[0]
    validate_mask[finite_idx[first_start : first_end + 1]] = 1
    last_valid_end = first_end

    for start, end in sub_intervals[1:]:
        dt_bridge = time_finite[start] - time_finite[last_valid_end]
        if not np.isfinite(dt_bridge) or dt_bridge <= 0:
            continue

        bridge_dist = haversine_single(
            np.radians(lat_finite[last_valid_end]),
            np.radians(lon_finite[last_valid_end]),
            np.radians(lat_finite[start]),
            np.radians(lon_finite[start]),
        )

        if bridge_dist / dt_bridge < speed_threshold:
            validate_mask[finite_idx[start : end + 1]] = 1
            last_valid_end = end


def filter_intervals_by_speed_reachability_and_max_points(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    speed_threshold: float = SPEED_THRESHOLD,
    max_points_in_interval: int = MAX_POINTS_IN_INTERVAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, разбивающий интервал на основе максимального количества точек
    Args:
        lon: Массив долгот
        lat: Массив широт
        time: Массив временных меток
        speed_threshold: Предельное значение скорости
        max_points_in_interval: Максимальное количество точек в интервале
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

    sub_intervals = _get_sub_intervals(
        lat_finite, lon_finite, time_finite, speed_threshold, max_points_in_interval
    )

    _apply_reachability_mask(
        sub_intervals,
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
        check_lon, check_lat, check_time, check_validate_point = (
            filter_intervals_by_speed_reachability_and_max_points(
                lon_df,
                lat_df,
                time_df,
                speed_threshold=20.0,
                max_points_in_interval=MAX_POINTS_IN_INTERVAL,
            )
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
            CalculatingStatistics.calculate_statistics(
                experimental_df, control_df, np.array(list_time)
            )

            mask = df["validate_point"].to_numpy() == 1
            lon_df = np.where(mask, lon_df, np.nan)
            lat_df = np.where(mask, lat_df, np.nan)

            mask = check_validate_point == 1
            check_lon = np.where(mask, check_lon, np.nan)
            check_lat = np.where(mask, check_lat, np.nan)

            step = 100000
            path_dir = Path(__file__).parent.parent.parent
            for i in range(0, len(lon_df), step):
                lon = lon_df[i : i + step]
                lat = lat_df[i : i + step]
                time = time_df[i : i + step]
                number = i // step
                path = path_dir / f"control_{number}.geojson"
                IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])

            for i in range(0, len(lon_df), step):
                lon = check_lon[i : i + step]
                lat = check_lat[i : i + step]
                time = check_time[i : i + step]
                number = i // step
                path = path_dir / f"experiment_{number}.geojson"
                IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])
