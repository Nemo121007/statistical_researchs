import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6371000.0  # meters
SPEED_THRESHOLD = 20.0  # m/s
NEIGHBOR_SPEED_THRESHOLD = 20.0
DISTANCE_SPLIT_THRESHOLD = 500.0  # meters
MIN_CUMULATIVE_DISTANCE = 500.0
MIN_ENDPOINT_DISTANCE = 500.0
MIN_POINTS_IN_INTERVAL = 30
MAX_POINTS_IN_INTERVAL = 180
NEIGHBORS_EACH_SIDE = 1


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

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
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


def _get_speed_distance_intervals(
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
    distance_threshold: float,
    max_points_in_interval: int,
) -> List[Tuple[int, int]]:
    """
    Метод, разделяющий трек на интервалы по превышению скорости и расстояния.
    Args:
        lat_finite: Массив широт (конечные значения)
        lon_finite: Массив долгот (конечные значения)
        time_finite: Массив временных меток (конечные значения)
        speed_threshold: Порог скорости (м/с) для разрыва интервала
        distance_threshold: Порог расстояния (м) для разрыва интервала
        max_points_in_interval: Максимальное количество точек в одном интервале
    Returns:
        Список кортежей (start, end) с индексами начала и конца подинтервалов
    """
    lat_rad = np.radians(lat_finite)
    lon_rad = np.radians(lon_finite)

    dist = segment_distances(lat_rad, lon_rad)
    dt = np.diff(time_finite)

    speed = np.full_like(dist, np.inf, dtype=float)
    valid_dt_mask = np.isfinite(dt) & (dt > 0)
    speed[valid_dt_mask] = dist[valid_dt_mask] / dt[valid_dt_mask]

    break_mask = (~valid_dt_mask) | (speed > speed_threshold) | (dist > distance_threshold)
    split_index = np.flatnonzero(break_mask) + 1

    chunks = np.split(np.arange(len(time_finite)), split_index)
    speed_intervals = [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size > 0]

    sub_intervals: List[Tuple[int, int]] = []
    for start, end in speed_intervals:
        sub_intervals.extend(_split_interval_by_max_points(start, end, max_points_in_interval))

    return sub_intervals


def _filter_intervals_by_metrics(
    intervals: List[Tuple[int, int]],
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    finite_idx: np.ndarray,
    validate_mask: np.ndarray,
    min_points_in_interval: Optional[int],
    min_cumulative_distance: Optional[float],
    min_endpoint_distance: Optional[float],
) -> List[Tuple[int, int]]:
    """
    Метод, фильтрующий интервалы по минимальным метрикам (количество точек, расстояние).
    Args:
        intervals: Список интервалов (начало, конец)
        lat_finite: Массив широт
        lon_finite: Массив долгот
        finite_idx: Индексы конечных значений в исходном массиве
        validate_mask: Массив флагов валидности точек
        min_points_in_interval: Минимальное количество точек в интервале
        min_cumulative_distance: Минимальное накопленное расстояние (м)
        min_endpoint_distance: Минимальное расстояние между начальной и конечной точками (м)
    Returns:
        Список валидных интервалов
    """
    if not intervals:
        return []

    lat_rad = np.radians(lat_finite)
    lon_rad = np.radians(lon_finite)

    segment_dist = segment_distances(lat_rad, lon_rad)
    distance_prefix = np.concatenate(([0.0], np.cumsum(segment_dist)))

    valid_intervals: List[Tuple[int, int]] = []

    for start, end in intervals:
        interval_size = end - start + 1

        if interval_size < 2:
            continue

        if min_points_in_interval is not None and interval_size < min_points_in_interval:
            continue

        cumulative_distance = float(distance_prefix[end] - distance_prefix[start])

        if min_cumulative_distance is not None and cumulative_distance < min_cumulative_distance:
            continue

        endpoint_distance = haversine_single(
            float(lat_rad[start]),
            float(lon_rad[start]),
            float(lat_rad[end]),
            float(lon_rad[end]),
        )

        if min_endpoint_distance is not None and endpoint_distance < min_endpoint_distance:
            continue

        validate_mask[finite_idx[start : end + 1]] = 1
        valid_intervals.append((start, end))

    return valid_intervals


def _filter_intervals_by_neighbor_reachability(
    intervals: List[Tuple[int, int]],
    lat_finite: np.ndarray,
    lon_finite: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
    neighbors_each_side: int,
) -> List[Tuple[int, int]]:
    """
    Метод, фильтрующий интервалы по достижимости соседей (не превышая порог скорости).
    Args:
        intervals: Возможные валидные интервалы
        lat_finite: Массив широт
        lon_finite: Массив долгот
        time_finite: Массив временных меток
        speed_threshold: Порог скорости для прыжка между интервалами
        neighbors_each_side: Количество соседей с каждой стороны для проверки
    Returns:
        Список интервалов, прошедших проверку соседями
    """
    if not intervals:
        return []

    def _get_edges(interval: Tuple[int, int]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Получает координаты и время для краев интервала.
        Args:
            interval: Кортеж с индексами (начало, конец) интервала
        Returns:
            Два кортежа (lat, lon, time) для начала и конца интервала
        """
        start, end = interval
        return (
            (float(lat_finite[start]), float(lon_finite[start]), float(time_finite[start])),
            (float(lat_finite[end]), float(lon_finite[end]), float(time_finite[end])),
        )

    def _is_jump_exceeded(
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> bool:
        """
        Проверяет, превышена ли допустимая скорость при переходе между точками.
        Args:
            p1: Кортеж (lat, lon, time) первой точки
            p2: Кортеж (lat, lon, time) второй точки
        Returns:
            True, если скорость превышена, иначе False
        """
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

    alive = [True] * len(intervals)

    while True:
        alive_indices = [i for i, flag in enumerate(alive) if flag]
        to_remove = set()

        for pos, idx in enumerate(alive_indices):
            c_start, c_end = _get_edges(intervals[idx])

            exceeded = False

            for offset in range(1, neighbors_each_side + 1):
                if pos - offset >= 0:
                    _, l_end = _get_edges(intervals[alive_indices[pos - offset]])

                    if _is_jump_exceeded(l_end, c_start):
                        exceeded = True
                        break

                if not exceeded and pos + offset < len(alive_indices):
                    r_start, _ = _get_edges(intervals[alive_indices[pos + offset]])

                    if _is_jump_exceeded(c_end, r_start):
                        exceeded = True
                        break

            if exceeded:
                to_remove.add(idx)

        if not to_remove:
            break

        for idx in to_remove:
            alive[idx] = False

    return [intervals[i] for i, flag in enumerate(alive) if flag]


def filter_combined_intervals(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    speed_threshold: float = SPEED_THRESHOLD,
    distance_threshold: float = DISTANCE_SPLIT_THRESHOLD,
    max_points_in_interval: int = MAX_POINTS_IN_INTERVAL,
    min_points_in_interval: Optional[int] = None,
    min_cumulative_distance: Optional[float] = None,
    min_endpoint_distance: Optional[float] = None,
    neighbors_each_side: int = NEIGHBORS_EACH_SIDE,
    neighbor_speed_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, применяющий комбинированную фильтрацию для определения валидных интервалов.
    Args:
        lon: Исходный массив долгот
        lat: Исходный массив широт
        time: Исходный массив временных меток
        speed_threshold: Порог скорости для разделения
        distance_threshold: Порог расстояния для разделения
        max_points_in_interval: Максимальное количество точек в интервале
        min_points_in_interval: Минимальное количество точек в интервале
        min_cumulative_distance: Минимальное накопленное расстояние (м)
        min_endpoint_distance: Минимальное расстояние от начала до конца интервала (м)
        neighbors_each_side: Радиус проверки соседних интервалов
        neighbor_speed_threshold: Порог скорости до соседей
    Returns:
        Кортежи (lon, lat, time, validate_mask), где validate_mask - флаги валидности исходных точек
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

    # Используем исходные массивы напрямую, если нет необходимости в копировании
    # И создаем один маскирующий массив
    finite_mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(time)
    finite_idx = np.flatnonzero(finite_mask)

    validate_mask = np.zeros(n, dtype=np.int8)

    if finite_idx.size == 0:
        return (
            np.asarray(lon, dtype=float).copy(),
            np.asarray(lat, dtype=float).copy(),
            np.asarray(time, dtype=float).copy(),
            validate_mask,
        )

    if finite_idx.size == 1:
        validate_mask[finite_idx[0]] = 1
        return (
            np.asarray(lon, dtype=float).copy(),
            np.asarray(lat, dtype=float).copy(),
            np.asarray(time, dtype=float).copy(),
            validate_mask,
        )

    if neighbor_speed_threshold is None:
        neighbor_speed_threshold = speed_threshold

    intervals = _get_speed_distance_intervals(
        lat_finite=lat[finite_mask],
        lon_finite=lon[finite_mask],
        time_finite=time[finite_mask],
        speed_threshold=speed_threshold,
        distance_threshold=distance_threshold,
        max_points_in_interval=max_points_in_interval,
    )

    if not intervals:
        return (
            np.asarray(lon, dtype=float).copy(),
            np.asarray(lat, dtype=float).copy(),
            np.asarray(time, dtype=float).copy(),
            validate_mask,
        )

    intervals = _filter_intervals_by_metrics(
        intervals=intervals,
        lat_finite=lat[finite_mask],
        lon_finite=lon[finite_mask],
        finite_idx=finite_idx,
        validate_mask=validate_mask,
        min_points_in_interval=min_points_in_interval,
        min_cumulative_distance=min_cumulative_distance,
        min_endpoint_distance=min_endpoint_distance,
    )

    if not intervals:
        return (
            np.asarray(lon, dtype=float).copy(),
            np.asarray(lat, dtype=float).copy(),
            np.asarray(time, dtype=float).copy(),
            validate_mask,
        )

    valid_intervals = _filter_intervals_by_neighbor_reachability(
        intervals=intervals,
        lat_finite=lat[finite_mask],
        lon_finite=lon[finite_mask],
        time_finite=time[finite_mask],
        speed_threshold=neighbor_speed_threshold,
        neighbors_each_side=neighbors_each_side,
    )

    validate_mask[:] = 0
    for start, end in valid_intervals:
        validate_mask[finite_idx[start : end + 1]] = 1

    return (
        np.asarray(lon, dtype=float).copy(),
        np.asarray(lat, dtype=float).copy(),
        np.asarray(time, dtype=float).copy(),
        validate_mask,
    )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"
    list_time = []

    processor = DataProcessor()
    df = processor.load_csv(data_path)

    for i in range(15):
        lon_df, lat_df, time_df = processor.get_lon_lat(df)

        start_time = time.time()
        check_lon, check_lat, check_time, check_validate_point = filter_combined_intervals(
            lon_df,
            lat_df,
            time_df,
            speed_threshold=SPEED_THRESHOLD,
            distance_threshold=DISTANCE_SPLIT_THRESHOLD,
            min_points_in_interval=MIN_POINTS_IN_INTERVAL,
            max_points_in_interval=MAX_POINTS_IN_INTERVAL,
            min_cumulative_distance=MIN_CUMULATIVE_DISTANCE,
            min_endpoint_distance=MIN_ENDPOINT_DISTANCE,
            neighbors_each_side=NEIGHBORS_EACH_SIDE,
            neighbor_speed_threshold=NEIGHBOR_SPEED_THRESHOLD,
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
