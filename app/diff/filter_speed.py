import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6371000  # метры


def haversine_single(lat1, lon1, lat2, lon2) -> float:
    """
    Метод, считающий расстояние между двумя точками
    Args:
        lat1: широта первой точки
        lon1: долгота первой точки
        lat2: широта первой точки
        lon2: долгота первой точки
    Returns:
        Расстояние между двумя точками
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return float(EARTH_RADIUS * c)


def filter_speed(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод, фильтрующий временной ряд по скорости
    Args:
         lon: Массив долгот
         lat: Массив широт
         time: Массив временных меток
    Returns:
         Массив долгот, Массив широт, Массив временных меток, Массив флагов валидности точек
    """
    lon = np.asarray(lon, dtype=float).copy()
    lat = np.asarray(lat, dtype=float).copy()
    time = np.asarray(time).copy()

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    if lon.shape != lat.shape or lon.shape != time.shape:
        raise ValueError("lon, lat и time должны иметь одинаковую длину")

    n = lon.size
    valid_indices = np.where(~np.isnan(lon) & ~np.isnan(lat))[0]
    if n == 0 or len(valid_indices) == 0:
        return np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([])

    mask = np.zeros(n, dtype=np.int8)
    mask[valid_indices[0]] = 1

    last_valid_idx = valid_indices[0]

    for i in range(valid_indices[0] + 1, n):
        # Если текущая точка уже NaN — ведём себя как при превышении скорости
        if np.isnan(lon[i]) or np.isnan(lat[i]):
            lon[i] = np.nan
            lat[i] = np.nan
            mask[i] = 0
            continue

        dt = time[i] - time[last_valid_idx]

        distance = haversine_single(
            lat_rad[last_valid_idx],
            lon_rad[last_valid_idx],
            lat_rad[i],
            lon_rad[i],
        )

        speed = distance / dt

        if speed < 20:
            mask[i] = 1
            last_valid_idx = i
        else:
            # lon[i] = np.nan
            # lat[i] = np.nan
            mask[i] = 0

    return lon, lat, time, mask


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"
    list_time = []

    processor = DataProcessor()
    df = processor.load_csv(data_path)

    for i in range(15):
        lon_df, lat_df, time_df = processor.get_lon_lat(df)

        start_time = time.time()
        check_lon, check_lat, check_time, check_validate_point = filter_speed(
            lon_df, lat_df, time_df
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
