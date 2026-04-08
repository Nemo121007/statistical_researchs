import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6371000.0  # meters
SPEED_THRESHOLD = 20.0  # m/s
MIN_ENDPOINT_DISTANCE = 500.0  # meters


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

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS * c


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


def _find_chunks_by_speed(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    time_finite: np.ndarray,
    speed_threshold: float,
) -> list:
    """
    Вспомогательный метод, разделяющий массивы на чанки по скорости
    Args:
        lat_rad: Широта точек (в рад.)
        lon_rad: Долгота точек (в рад.)
        time_finite: Временные метки точек
        speed_threshold: Предельное значение скорости
    Returns:
        Список кортежей (start_index, end_index) для каждого чанка
    """
    dist = segment_distances(lat_rad, lon_rad)
    dt = np.diff(time_finite)

    speed = np.full_like(dist, np.inf, dtype=float)
    valid_dt_mask = np.isfinite(dt) & (dt > 0)
    speed[valid_dt_mask] = dist[valid_dt_mask] / dt[valid_dt_mask]

    split_index = np.flatnonzero(speed > speed_threshold) + 1
    chunks = np.split(np.arange(len(time_finite)), split_index)

    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size > 0]


def filter_intervals_by_speed_and_endpoint_distance(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    speed_threshold: float = SPEED_THRESHOLD,
    min_endpoint_distance: float = MIN_ENDPOINT_DISTANCE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, фильтрующий интервалы по минимальному расстоянию между началом и концом интервала
    Args:
        lon: Массив долгот
        lat: Массив широт
        time: Массив временных меток
        speed_threshold: Предельное значение скорости
        min_endpoint_distance: Минимальное расстояние между первой и последней точкой интервала
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
        return lon_copy, lat_copy, time_copy, validate_mask

    lon_finite = lon_copy[finite_mask]
    lat_finite = lat_copy[finite_mask]
    time_finite = time_copy[finite_mask]

    lon_rad = np.radians(lon_finite)
    lat_rad = np.radians(lat_finite)

    # Разбиение на интервалы по скорости между соседними finite-точками
    intervals = _find_chunks_by_speed(lat_rad, lon_rad, time_finite, speed_threshold)

    for start, end in intervals:
        if end > start:
            if (
                haversine_single(
                    lat_rad[start],
                    lon_rad[start],
                    lat_rad[end],
                    lon_rad[end],
                )
                >= min_endpoint_distance
            ):
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
        check_lon, check_lat, check_time, check_validate_point = filter_intervals_by_speed_and_endpoint_distance(
            lon_df,
            lat_df,
            time_df,
            speed_threshold=20.0,
            min_endpoint_distance=MIN_ENDPOINT_DISTANCE,
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
