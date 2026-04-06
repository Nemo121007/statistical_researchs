from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from app.working.data_processor import DataProcessor
from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.help_scripts.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle


EARTH_RADIUS = 6371000  # метры


def haversine_single(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS * c


def filter_speed(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'post_processing' / 'example.csv'

    processor = DataProcessor()
    df = processor.load_csv(data_path)

    lon, lat, time = processor.get_lon_lat(df)

    check_lon, check_lat, check_time, check_validate_point = filter_speed(lon, lat, time)

    control_df = pd.DataFrame({
        'lon': check_lon,
        'lat': check_lat,
        'time': check_time,
        'validate_point': df['validate_point'].to_numpy(),
        'in_water': df['in_water'].to_numpy(),
    })

    experimental_df = pd.DataFrame({
        'lon': check_lon,
        'lat': check_lat,
        'time': check_time,
        'validate_point': check_validate_point,
    })

    CalculatingStatistics.calculate_statistics(experimental_df, control_df)
