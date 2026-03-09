from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geopy.distance import distance
from pandas.core.interchange.dataframe_protocol import DataFrame
from pykalman import KalmanFilter
from tqdm import tqdm  # Импорт для прогресс-бара

from help_scripts.IOPs_geojson import IOPs_geojson
from help_scripts.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle
from settings.settings import DefaultLocate


class Filtration:
    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        data = pd.read_csv(path)
        # Заменить data["lon", "lat"] на NaN, если data["in_water"] == False, validate_point == 0
        mask = (data["in_water"] == False)
        data.loc[mask, ["lon", "lat"]] = np.nan
        # Взять первые 10000 строк
        # data = data.head(500000)
        return data

    @staticmethod
    def discretize_df(df: pd.DataFrame) -> pd.DataFrame:
        # Рассчитать расстояние между точками
        lat_array = df["lat"].values.astype(float)
        lon_array = df["lon"].values.astype(float)

        # если в  lat_array или lon_array попадает Nan, то брать для рассчёта расстояний последнее не Nan Значение
        lat_filled = pd.Series(lat_array).ffill().bfill().values
        lon_filled = pd.Series(lon_array).ffill().bfill().values
        # Расстояния между последовательными точками (длина N-1)
        distances = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_filled, lon_filled)

        # Объединить точки в группы, если между точками не более 100 м.
        # Задать каждой группе уникальный id.
        group_ids = np.zeros(len(df), dtype=int)
        current_group = 0
        for i, dist in enumerate(distances):
            group_ids[i] = current_group
            if dist > 100:
                current_group += 1
        group_ids[-1] = current_group  # последняя точка

        df = df.copy()
        df["group_id"] = group_ids
        return df

    @staticmethod
    def filter_size_groups(df: pd.DataFrame, min_size: int = 30) -> pd.DataFrame:
        # Оставить только группы, в которых не менее min_size точек
        # Для остальных групп lon, lat = Nan, validate_point = 0
        valid_mask = df["lon"].notna() & df["lat"].notna()
        valid_counts = df[valid_mask].groupby("group_id").size()
        group_sizes = df["group_id"].map(valid_counts).fillna(0)
        small_group_mask = group_sizes < min_size
        df = df.copy()
        df.loc[small_group_mask, ["lon", "lat"]] = np.nan
        df.loc[small_group_mask, "validate_point"] = 0
        return df

    @staticmethod
    def filter_groups_by_length(df: pd.DataFrame, min_length_meters: float = 100.0) -> pd.DataFrame:
        """
        Заменяет координаты группы на NaN, если суммарная длина маршрута группы
        меньше min_length_meters.

        :param df: DataFrame с колонками lat, lon, group_id (NaN допустимы).
        :param min_length_meters: минимальная кумулятивная длина группы в метрах.
        :return: DataFrame с отфильтрованными группами.
        """
        df = df.copy()
        removed_groups = 0

        for group_id, group in df.groupby("group_id"):
            valid = group[group["lon"].notna() & group["lat"].notna()]
            if valid.empty:
                continue

            lat_array = valid["lat"].values.astype(float)
            lon_array = valid["lon"].values.astype(float)

            if len(lat_array) < 2:
                cumulative_length = 0.0
            else:
                segments = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_array, lon_array)
                cumulative_length = segments.sum()

            if cumulative_length < min_length_meters:
                df.loc[group.index, ["lon", "lat"]] = np.nan
                df.loc[group.index, "validate_point"] = 0
                removed_groups += 1

        print(f"filter_groups_by_length: удалено {removed_groups} групп с длиной < {min_length_meters} м")
        return df

    @staticmethod
    def filter_groups_by_speed(df: pd.DataFrame, max_avg_speed_ms: float = 15.0) -> pd.DataFrame:
        """
        Заменяет координаты группы на NaN, если средняя скорость группы
        превышает max_avg_speed_ms.

        Средняя скорость = суммарная длина маршрута / суммарное время группы.

        :param df: DataFrame с колонками lat, lon, time, group_id (NaN допустимы).
        :param max_avg_speed_ms: максимально допустимая средняя скорость в м/с.
        :return: DataFrame с отфильтрованными группами.
        """
        df = df.copy()
        removed_groups = 0

        for group_id, group in df.groupby("group_id"):
            valid = group[group["lon"].notna() & group["lat"].notna()].copy()
            if len(valid) < 2:
                continue

            lat_array = valid["lat"].values.astype(float)
            lon_array = valid["lon"].values.astype(float)
            time_array = valid["time"].values.astype(int)

            total_length_meters = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                lat_array, lon_array
            ).sum()

            total_time_seconds = int(time_array[-1]) - int(time_array[0])

            if total_time_seconds <= 0:
                continue

            avg_speed_ms = total_length_meters / total_time_seconds

            if avg_speed_ms > max_avg_speed_ms:
                df.loc[group.index, ["lon", "lat"]] = np.nan
                df.loc[group.index, "validate_point"] = 0
                removed_groups += 1

        print(f"filter_groups_by_speed: удалено {removed_groups} групп со средней скоростью > {max_avg_speed_ms} м/с")
        return df

    @staticmethod
    def filter_groups_by_sampled_length(
        df: pd.DataFrame,
        sample_step: int = 100,
        min_sampled_length_meters: float = 500.0,
    ) -> pd.DataFrame:
        """
        Удаляет группы, у которых кумулятивное расстояние между точками,
        взятыми через каждые sample_step точек, меньше min_sampled_length_meters.

        Если очередная точка выборки — NaN, берётся последняя не-NaN точка.

        :param df: DataFrame с колонками lat, lon, group_id (NaN допустимы).
        :param sample_step: шаг выборки точек внутри группы.
        :param min_sampled_length_meters: минимальная кумулятивная длина выборки в метрах.
        :return: DataFrame с отфильтрованными группами.
        """
        df = df.copy()
        removed_groups = 0

        for group_id, group in df.groupby("group_id"):
            lat_array = group["lat"].values.astype(float)
            lon_array = group["lon"].values.astype(float)

            # Берём точки через каждые sample_step
            sampled_indices = range(0, len(lat_array), sample_step)

            sampled_lat = []
            sampled_lon = []
            last_valid_lat = None
            last_valid_lon = None

            # Если точек меньше шага — берём первую и последнюю не-NaN точки
            if len(lat_array) < sample_step:
                valid_mask = ~np.isnan(lat_array) & ~np.isnan(lon_array)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) >= 2:
                    first_idx, last_idx = valid_indices[0], valid_indices[-1]
                    sampled_lat = [lat_array[first_idx], lat_array[last_idx]]
                    sampled_lon = [lon_array[first_idx], lon_array[last_idx]]
            else:
                for idx in sampled_indices:
                    if not np.isnan(lat_array[idx]) and not np.isnan(lon_array[idx]):
                        last_valid_lat = lat_array[idx]
                        last_valid_lon = lon_array[idx]

                    if last_valid_lat is not None:
                        sampled_lat.append(last_valid_lat)
                        sampled_lon.append(last_valid_lon)

            if len(sampled_lat) < 2:
                df.loc[group.index, ["lon", "lat"]] = np.nan
                df.loc[group.index, "validate_point"] = 0
                removed_groups += 1
                continue

            sampled_lat = np.array(sampled_lat)
            sampled_lon = np.array(sampled_lon)

            cumulative_length = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                sampled_lat, sampled_lon
            ).sum()

            if cumulative_length < min_sampled_length_meters:
                df.loc[group.index, ["lon", "lat"]] = np.nan
                df.loc[group.index, "validate_point"] = 0
                removed_groups += 1

        print(
            f"filter_groups_by_sampled_length: удалено {removed_groups} групп "
            f"с выборочной длиной (шаг={sample_step}) < {min_sampled_length_meters} м"
        )
        return df

    @staticmethod
    def filter_groups_by_edge_speed(
        df: pd.DataFrame,
        max_edge_speed_ms: float = 15.0,
        neighbors_each_side: int = 1,
    ) -> pd.DataFrame:
        """
        Удаляет группы, у которых скорость между крайними точками соседних групп
        превышает max_edge_speed_ms хотя бы с одной стороны.

        Алгоритм итеративный: повторяется до тех пор, пока на очередном проходе
        не перестанут удаляться новые группы. Это необходимо, потому что после
        удаления группы её соседи могут образовать новые аномальные пары.
        Уже удалённые группы (края == None) пропускаются при проверке соседей.

        :param df: DataFrame с колонками lat, lon, time, group_id (NaN допустимы).
        :param max_edge_speed_ms: максимально допустимая скорость между краями групп в м/с.
        :param neighbors_each_side: количество соседних групп с каждой стороны для проверки
                                    (1 → ±1 группа, 2 → ±2 группы и т.д.).
        :return: DataFrame с отфильтрованными группами.
        """
        df = df.copy()
        sorted_group_ids = sorted(df["group_id"].unique())
        total_removed = 0

        def build_edges(current_df, group_ids):
            """Строит словарь краёв групп по текущему состоянию df."""
            edges = {}
            for gid in group_ids:
                group = current_df[current_df["group_id"] == gid]
                valid = group[group["lon"].notna() & group["lat"].notna()]
                if valid.empty:
                    edges[gid] = None
                    continue
                first_row = valid.iloc[0]
                last_row = valid.iloc[-1]
                edges[gid] = {
                    "first": (first_row["lat"], first_row["lon"], int(first_row["time"])),
                    "last":  (last_row["lat"],  last_row["lon"],  int(last_row["time"])),
                }
            return edges

        while True:
            group_edges = build_edges(df, sorted_group_ids)

            # Список только живых групп (не NaN) — по нему ищем соседей
            alive_group_ids = [gid for gid in sorted_group_ids if group_edges[gid] is not None]

            groups_to_remove = set()

            for alive_idx, group_id in enumerate(alive_group_ids):
                current_edges = group_edges[group_id]
                exceeded = False

                # Проверяем neighbors_each_side живых соседей слева
                for offset in range(1, neighbors_each_side + 1):
                    left_alive_idx = alive_idx - offset
                    if left_alive_idx >= 0:
                        left_id = alive_group_ids[left_alive_idx]
                        left_edges = group_edges[left_id]
                        dist_m = distance(
                            (left_edges["last"][0], left_edges["last"][1]),
                            (current_edges["first"][0], current_edges["first"][1]),
                        ).meters
                        dt = current_edges["first"][2] - left_edges["last"][2]
                        if dt > 0 and dist_m / dt > max_edge_speed_ms:
                            exceeded = True
                            break

                # Проверяем neighbors_each_side живых соседей справа
                if not exceeded:
                    for offset in range(1, neighbors_each_side + 1):
                        right_alive_idx = alive_idx + offset
                        if right_alive_idx < len(alive_group_ids):
                            right_id = alive_group_ids[right_alive_idx]
                            right_edges = group_edges[right_id]
                            dist_m = distance(
                                (current_edges["last"][0], current_edges["last"][1]),
                                (right_edges["first"][0], right_edges["first"][1]),
                            ).meters
                            dt = right_edges["first"][2] - current_edges["last"][2]
                            if dt > 0 and dist_m / dt > max_edge_speed_ms:
                                exceeded = True
                                break

                if exceeded:
                    groups_to_remove.add(group_id)

            if not groups_to_remove:
                break  # Стабилизировались — выходим

            for group_id in groups_to_remove:
                mask = df["group_id"] == group_id
                df.loc[mask, ["lon", "lat"]] = np.nan
                df.loc[mask, "validate_point"] = 0

            total_removed += len(groups_to_remove)

        print(
            f"filter_groups_by_edge_speed: удалено {total_removed} групп "
            f"со скоростью между краями > {max_edge_speed_ms} м/с "
            f"(проверка ±{neighbors_each_side} соседей)"
        )
        return df

    @staticmethod
    def debug_distances(df: pd.DataFrame) -> None:
        """Выводит расстояния между каждым десятым не-NaN значением df."""
        valid = df[df["lon"].notna() & df["lat"].notna()].reset_index(drop=True)
        sampled = valid.iloc[::6]
        lat_array = sampled["lat"].values.astype(float)
        lon_array = sampled["lon"].values.astype(float)
        distances = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_array, lon_array)
        for i, dist in enumerate(distances):
            idx_from = sampled.index[i]
            idx_to = sampled.index[i + 1]
            print(f"[{idx_from}] -> [{idx_to}]: {dist:.2f} м")

    @staticmethod
    def suppress_stationary_oscillations(
        df: pd.DataFrame,
        lookahead_steps: int = 6,
        max_displacement_meters: float = 60.0,
    ) -> pd.DataFrame:
        """
        Заменяет колебания координат стоящего объекта на повтор последней «стабильной» точки.

        Алгоритм: для каждой точки i вычисляется расстояние до точки i + lookahead_steps.
        Если расстояние меньше max_displacement_meters, считается, что объект фактически
        не сдвинулся, и координаты точки i + 1 заменяются координатами точки i.

        :param df: DataFrame с колонками lat, lon (NaN допустимы).
        :param lookahead_steps: количество шагов вперёд для оценки смещения (n).
        :param max_displacement_meters: порог расстояния в метрах (m).
        :return: DataFrame с подавленными колебаниями.
        """
        df = df.copy()
        lat = df["lat"].values.astype(float)
        lon = df["lon"].values.astype(float)

        replaced_count = 0

        for i in range(len(lat) - lookahead_steps):
            future_idx = i + lookahead_steps

            # Пропускаем, если любая из двух точек — NaN
            if np.isnan(lat[i]) or np.isnan(lon[i]):
                continue
            if np.isnan(lat[future_idx]) or np.isnan(lon[future_idx]):
                continue

            displacement = distance((lat[i], lon[i]), (lat[future_idx], lon[future_idx])).meters

            if displacement < max_displacement_meters:
                # Объект стоит на месте — заменяем следующую точку координатами текущей
                next_idx = i + 1
                lat[next_idx] = lat[i]
                lon[next_idx] = lon[i]
                replaced_count += 1

        df["lat"] = lat
        df["lon"] = lon
        print(f"suppress_stationary_oscillations: заменено {replaced_count} точек из {len(df)}")
        return df

    @staticmethod
    def print_results(output_path: Path, df: pd.DataFrame):
        # Составлять list_arrays по идентификаторам групп
        list_arrays = []
        for _, group in df.groupby("group_id"):
            time_array = group["time"].values
            lat_array = group["lat"].values
            lon_array = group["lon"].values
            list_arrays.append([time_array, lat_array, lon_array])
        IOPs_geojson.write_geojson_from_arrays(output_path, list_arrays)


if __name__ == '__main__':
    path = DefaultLocate.DATA_POSTPROCESSED_DIR / "example_located.csv"

    data = Filtration.load_csv(path)
    data = Filtration.suppress_stationary_oscillations(data)

    # Filtration.debug_distances(data)
    data = Filtration.discretize_df(data)
    data = Filtration.filter_size_groups(data)

    # data = Filtration.filter_groups_by_length(data, min_length_meters=500.0)
    data = Filtration.filter_groups_by_speed(data)
    data = Filtration.filter_groups_by_sampled_length(data, min_sampled_length_meters=1000)
    # data = Filtration.filter_groups_by_edge_speed(data, neighbors_each_side=3)

    output_path = DefaultLocate.OUTPUT_DIR / "example_located_filtered.geojson"
    Filtration.print_results(output_path, data)
