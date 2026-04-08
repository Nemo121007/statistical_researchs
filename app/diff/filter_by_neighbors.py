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
NEIGHBORS_EACH_SIDE = 1
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
    Метод, вычисляющий расстояния между последовательными точками.
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
    start: int,
    end: int,
    max_points: int,
) -> List[Tuple[int, int]]:
    """
    Метод, разделяющий интервалы по максимальному количеству точек.
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


def _split_by_speed_and_max_points(
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
    max_points_in_interval: int,
) -> List[Tuple[int, int]]:
    """
    Метод, разделяющий трек на интервалы по скорости и количеству точек в интервале
    Args:
        lat_finite: Массив широт для finite-точек
        lon_finite: Массив долгот для finite-точек
        time_finite: Массив временных меток для finite-точек
        speed_threshold: Предельное значение скорости
        max_points_in_interval: Максимальное количество точек в интервале
    Returns:
        Список кортежей (start, end) для каждого интервала
    """
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


def _filter_intervals_by_reachability(
    intervals: List[Tuple[int, int]],
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
    neighbors_each_side: int,
) -> List[Tuple[int, int]]:
    """Выполняет фильтрацию интервалов по достижимости соседей."""

    def _get_edge(
        interval: Tuple[int, int],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        start, end = interval
        return (
            (lat_finite[start], lon_finite[start], time_finite[start]),
            (lat_finite[end], lon_finite[end], time_finite[end]),
        )

    def _is_jump_exceeded(
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> bool:
        p1_lat, p1_lon, p1_ts = p1
        p2_lat, p2_lon, p2_ts = p2
        dt = p2_ts - p1_ts
        if not np.isfinite(dt) or dt <= 0:
            return False

        dist = haversine_single(
            np.radians(p1_lat),
            np.radians(p1_lon),
            np.radians(p2_lat),
            np.radians(p2_lon),
        )
        return (dist / dt) > speed_threshold

    def _check_exceeded(pos: int, idx: int, alive_indices: List[int]) -> bool:
        c_s, c_e = _get_edge(intervals[idx])
        for offset in range(1, neighbors_each_side + 1):
            if pos - offset >= 0:
                _, l_e = _get_edge(intervals[alive_indices[pos - offset]])
                if _is_jump_exceeded(l_e, c_s):
                    return True

            if pos + offset < len(alive_indices):
                r_s, _ = _get_edge(intervals[alive_indices[pos + offset]])
                if _is_jump_exceeded(c_e, r_s):
                    return True
        return False

    alive = [True] * len(intervals)

    while True:
        alive_indices = [i for i, a in enumerate(alive) if a]
        to_remove = set()

        for pos, idx in enumerate(alive_indices):
            if _check_exceeded(pos, idx, alive_indices):
                to_remove.add(idx)

        if not to_remove:
            break

        for idx in to_remove:
            alive[idx] = False

    return [intervals[i] for i, a in enumerate(alive) if a]


def filter_intervals_neighbor_reachability(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    speed_threshold: float = SPEED_THRESHOLD,
    neighbors_each_side: int = NEIGHBORS_EACH_SIDE,
    max_points_in_interval: int = MAX_POINTS_IN_INTERVAL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, фильтрующий интервалы на основе количества последовательно-достижимых соседей
    Args:
        lon: Массив долгот
        lat: Массив широт
        time: Массив временных меток
        speed_threshold: Предельное значение скорости
        neighbors_each_side: Количество последовательных соседей с каждой стороны
        max_points_in_interval: Максимальное количество точек в интервале
    Returns:
        Массив долгот, Массив широт, Массив временных меток, Массив флагов валидности точек
    """
    if lon.shape != lat.shape or lon.shape != time.shape:
        raise ValueError("lon, lat и time должны иметь одинаковую длину")

    n = len(lon)
    lon_copy = np.asarray(lon, dtype=float).copy()
    lat_copy = np.asarray(lat, dtype=float).copy()
    time_copy = np.asarray(time, dtype=float).copy()

    validate_mask = np.zeros(n, dtype=np.int8)

    finite_mask = np.isfinite(lon_copy) & np.isfinite(lat_copy) & np.isfinite(time_copy)
    finite_idx = np.flatnonzero(finite_mask)

    if finite_idx.size == 0:
        return lon_copy, lat_copy, time_copy, validate_mask

    lon_finite = lon_copy[finite_mask]
    lat_finite = lat_copy[finite_mask]
    time_finite = time_copy[finite_mask]

    if finite_idx.size == 1:
        validate_mask[finite_idx[0]] = 1
        return lon_copy, lat_copy, time_copy, validate_mask

    intervals = _split_by_speed_and_max_points(
        lat_finite,
        lon_finite,
        time_finite,
        speed_threshold=speed_threshold,
        max_points_in_interval=max_points_in_interval,
    )

    if not intervals:
        return lon_copy, lat_copy, time_copy, validate_mask

    valid_intervals = _filter_intervals_by_reachability(
        intervals,
        lat_finite,
        lon_finite,
        time_finite,
        speed_threshold,
        neighbors_each_side,
    )

    for start, end in valid_intervals:
        if end - start < 1:
            continue
        validate_mask[finite_idx[start : end + 1]] = 1

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
            filter_intervals_neighbor_reachability(
                lon_df,
                lat_df,
                time_df,
                speed_threshold=20.0,
                neighbors_each_side=2,
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
