import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.help_scripts.calculating_statistics import CalculatingStatistics
from app.working.data_processor import DataProcessor

EARTH_RADIUS = 6371000.0  # meters
SPEED_THRESHOLD = 20.0  # m/s
NEIGHBORS_EACH_SIDE = 1


def haversine_single(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS * c)


def filter_intervals_by_speed_neighbor_reachability(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    speed_threshold: float = SPEED_THRESHOLD,
    neighbors_each_side: int = NEIGHBORS_EACH_SIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Фильтр:
    1) берёт только finite-точки;
    2) делит трек на интервалы по скорости между соседними finite-точками;
    3) проверяет интервалы на достижимость не последовательно, а относительно соседей;
    4) удаляет интервалы, для которых прыжок до хотя бы одного соседа превышает порог скорости.

    Логика удаления итеративная: после удаления части интервалов список живых
    интервалов пересчитывается заново.
    """
    if lon.shape != lat.shape or lon.shape != time.shape:
        raise ValueError("lon, lat и time должны иметь одинаковую длину")

    n = len(lon)
    lon_copy = np.asarray(lon, dtype=float).copy()
    lat_copy = np.asarray(lat, dtype=float).copy()
    time_copy = np.asarray(time, dtype=float).copy()

    validate_mask = np.zeros(n, dtype=np.int8)

    finite_mask = (
        np.isfinite(lon_copy) & np.isfinite(lat_copy) & np.isfinite(time_copy)
    )
    finite_idx = np.flatnonzero(finite_mask)

    if finite_idx.size == 0:
        return lon_copy, lat_copy, time_copy, validate_mask

    lon_finite = lon_copy[finite_mask]
    lat_finite = lat_copy[finite_mask]
    time_finite = time_copy[finite_mask]

    if finite_idx.size == 1:
        validate_mask[finite_idx[0]] = 1
        return lon_copy, lat_copy, time_copy, validate_mask

    lat_rad = np.radians(lat_finite)
    lon_rad = np.radians(lon_finite)

    def _segment_distances() -> np.ndarray:
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

    def _split_by_speed() -> List[Tuple[int, int]]:
        dist = _segment_distances()
        dt = np.diff(time_finite)

        speed = np.full_like(dist, np.inf, dtype=float)
        valid_dt_mask = np.isfinite(dt) & (dt > 0)
        speed[valid_dt_mask] = dist[valid_dt_mask] / dt[valid_dt_mask]

        split_index = np.flatnonzero(speed > speed_threshold) + 1
        chunks = np.split(np.arange(finite_idx.size), split_index)

        return [
            (int(chunk[0]), int(chunk[-1]))
            for chunk in chunks
            if chunk.size > 0
        ]

    def _get_edge(interval: Tuple[int, int]) -> Tuple[float, float, float, float, float, float]:
        start, end = interval
        return (
            lat_finite[start],
            lon_finite[start],
            time_finite[start],
            lat_finite[end],
            lon_finite[end],
            time_finite[end],
        )

    def _is_jump_exceeded(
        p1_lat: float,
        p1_lon: float,
        p1_ts: float,
        p2_lat: float,
        p2_lon: float,
        p2_ts: float,
    ) -> bool:
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

    intervals = _split_by_speed()
    if not intervals:
        return lon_copy, lat_copy, time_copy, validate_mask

    alive = [True] * len(intervals)

    while True:
        alive_indices = [i for i, a in enumerate(alive) if a]
        to_remove = set()

        for pos, idx in enumerate(alive_indices):
            c_s_lat, c_s_lon, c_s_ts, c_e_lat, c_e_lon, c_e_ts = _get_edge(
                intervals[idx]
            )

            exceeded = False

            for offset in range(1, neighbors_each_side + 1):
                # Левый сосед
                if pos - offset >= 0:
                    left_idx = alive_indices[pos - offset]
                    l_s_lat, l_s_lon, l_s_ts, l_e_lat, l_e_lon, l_e_ts = _get_edge(
                        intervals[left_idx]
                    )

                    if _is_jump_exceeded(
                        l_e_lat, l_e_lon, l_e_ts,
                        c_s_lat, c_s_lon, c_s_ts,
                    ):
                        exceeded = True
                        break

                # Правый сосед
                if not exceeded and pos + offset < len(alive_indices):
                    right_idx = alive_indices[pos + offset]
                    r_s_lat, r_s_lon, r_s_ts, r_e_lat, r_e_lon, r_e_ts = _get_edge(
                        intervals[right_idx]
                    )

                    if _is_jump_exceeded(
                        c_e_lat, c_e_lon, c_e_ts,
                        r_s_lat, r_s_lon, r_s_ts,
                    ):
                        exceeded = True
                        break

            if exceeded:
                to_remove.add(idx)

        if not to_remove:
            break

        for idx in to_remove:
            alive[idx] = False

    for idx, is_alive in enumerate(alive):
        if not is_alive:
            continue
        start, end = intervals[idx]
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
        check_lon, check_lat, check_time, check_validate_point = filter_intervals_by_speed_neighbor_reachability(
            lon_df,
            lat_df,
            time_df,
            speed_threshold=20.0,
            neighbors_each_side=3,
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
                lon = lon_df[i: i + step]
                lat = lat_df[i: i + step]
                time = time_df[i: i + step]
                number = i // step
                path = path_dir / f"control_{number}.geojson"
                IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])

            for i in range(0, len(lon_df), step):
                lon = check_lon[i: i + step]
                lat = check_lat[i: i + step]
                time = check_time[i: i + step]
                number = i // step
                path = path_dir / f"experiment_{number}.geojson"
                IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])
