import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6371000.0  # meters
SPEED_THRESHOLD = 20.0  # m/s


def haversine_single(lat1, lon1, lat2, lon2) -> float:
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


def filter_intervals(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    speed_threshold: float = SPEED_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, фильтрующий временной ряд по интервалам и скорости
    Args:
        lon: Массив долгот
        lat: Массив широт
        time: Массив временных меток
        speed_threshold: Предельное значение скорости
    Returns:
        Массив долгот, Массив широт, Массив временных меток, Массив флагов валидности точек
    """
    # pylint: disable=too-many-locals
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

    lon_finite = lon_copy[finite_mask]
    lat_finite = lat_copy[finite_mask]
    time_finite = time_copy[finite_mask]

    validate_mask = np.zeros(n, dtype=np.int8)

    if len(time_finite) == 0:
        lon_copy[:] = np.nan
        lat_copy[:] = np.nan
        return lon_copy, lat_copy, time_copy, validate_mask

    if len(time_finite) == 1:
        validate_mask[finite_idx[0]] = 1
        return lon_copy, lat_copy, time_copy, validate_mask

    distance = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_finite, lon_finite)
    diff_time = np.diff(time_finite)

    speed = distance / diff_time
    split_index = np.where(speed > speed_threshold)[0] + 1

    chunks = np.split(np.arange(len(time_finite)), split_index)
    intervals: List[Tuple[int, int]] = [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size > 0]

    if not intervals:
        return lon_copy, lat_copy, time_copy, validate_mask

    # Первый интервал всегда валидный
    first_start, first_end = intervals[0]
    validate_mask[finite_idx[first_start : first_end + 1]] = 1
    last_valid_end = first_end

    for start, end in intervals[1:]:
        dt = time_finite[start] - time_finite[last_valid_end]

        if not np.isfinite(dt) or dt <= 0:
            continue

        dist = haversine_single(
            np.radians(lat_finite[last_valid_end]),
            np.radians(lon_finite[last_valid_end]),
            np.radians(lat_finite[start]),
            np.radians(lon_finite[start]),
        )
        bridge_speed = dist / dt

        if bridge_speed < speed_threshold:
            validate_mask[finite_idx[start : end + 1]] = 1
            last_valid_end = end

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
        check_lon, check_lat, check_time, check_validate_point = filter_intervals(
            lon_df,
            lat_df,
            time_df,
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
