import pandas as pd
import numpy as np
import datetime
from pandas.core.interchange.dataframe_protocol import DataFrame

from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate
from geopy.distance import geodesic
from geomag.geomag import GeoMag
import math

# Инициализируем модели один раз для эффективности
gm = GeoMag()
g = geodesic()
g.measure((0, 0), (1, 1))

Earth_radius_meters: float = 6371000


def haversine_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние по формуле гаверсинуса."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return Earth_radius_meters * c


def spherical_law_of_cosines_distance(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние по сферической теореме косинусов."""
    dlon = lon2 - lon1
    cos_angle = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    c = np.arccos(cos_angle)
    return Earth_radius_meters * c


def speed_comparison(df: DataFrame, name: str):
    """
    Рассчитывает и сохраняет статистику по скорости.
    """
    # --- РАСЧЕТ СКОРОСТИ ---
    df["diff_time"] = df["time"].diff()
    df["distance_haversine"] = haversine_distance(
        np.radians(df["lat"].shift(1)),
        np.radians(df["lon"].shift(1)),
        np.radians(df["lat"]),
        np.radians(df["lon"])
    )
    df["speed_haversine"] = df["distance_haversine"] / df["diff_time"]

    # --- ПОДГОТОВКА ДАННЫХ ДЛЯ СТАТИСТИКИ ---
    df["speed_haversine"] = df["speed_haversine"].replace([np.inf, -np.inf], np.nan)
    df.rename(columns={'speed': 'speed_true'}, inplace=True)
    df['speed_diff'] = df['speed_true'] - df['speed_haversine']

    stats_df = df[['speed_haversine', 'speed_diff']].dropna()
    valid_points_count = len(stats_df)

    if valid_points_count == 0:
        print(f"Пропускаем файл {name}, так как нет валидных точек для анализа скорости.")
        return

    # --- ВЫЧИСЛЕНИЕ СТАТИСТИК (ИСПРАВЛЕНО) ---
    result = {}
    result['valid_points_count'] = valid_points_count

    for col_name in ['speed_haversine', 'speed_diff']:
        prefix = 'speed_derived' if col_name == 'speed_haversine' else 'speed_diff'

        # Эта часть теперь выполняется для обеих колонок
        result[f'{prefix}_mean_mps'] = stats_df[col_name].mean()
        result[f'{prefix}_min_mps'] = stats_df[col_name].min()
        result[f'{prefix}_max_mps'] = stats_df[col_name].max()

        for p in [25, 50, 75, 90, 95]:
            result[f'{prefix}_{p}p_mps'] = stats_df[col_name].quantile(p / 100.0)

    result_df = pd.DataFrame([result])
    print(f"\n--- Итоговые результаты по скорости для {name} ---")
    print(result_df)

    path = DefaultLocate.DATA_DIR / f"{name}_results_speed_comparison.csv"
    result_df.to_csv(path)
    print(f"\nРезультаты по скорости для {name} сохранены в файл: {path}")


def source_comparison(df: DataFrame, name: str):
    """
    Рассчитывает и сохраняет статистику по курсу.
    """
    # Указываем дату для расчета магнитного склонения
    data = datetime.datetime(year=2025, month=6, day=1)

    # --- РАСЧЕТ АЗИМУТОВ ---
    lat1 = df['lat'].shift(1)
    lon1 = df['lon'].shift(1)
    lat2 = df['lat']
    lon2 = df['lon']

    pairs_df = pd.DataFrame({
        'lat1': lat1[1:], 'lon1': lon1[1:],
        'lat2': lat2[1:], 'lon2': lon2[1:]
    }).dropna()

    geographic_azimuths = []
    magnetic_azimuths = []

    for index, row in pairs_df.iterrows():
        if pd.isna(row[['lat1', 'lon1', 'lat2', 'lon2']]).any():
            geographic_azimuths.append(np.nan)
            magnetic_azimuths.append(np.nan)
            continue

        inverse_result = g.geod.Inverse(row['lat1'], row['lon1'], row['lat2'], row['lon2'])
        geo_azimuth = (inverse_result['azi1'] + 360) % 360
        geographic_azimuths.append(geo_azimuth)

        try:
            mag_result = gm.GeoMag(dlat=row['lat1'], dlon=row['lon1'], h=0, time=data.date())
            declination = mag_result.dec
            magnetic_azimuth = (geo_azimuth - declination + 360) % 360
            magnetic_azimuths.append(magnetic_azimuth)
        except Exception as e:
            print(f"Не удалось рассчитать магнитное склонение для точки ({row['lat1']}, {row['lon1']}): {e}")
            magnetic_azimuths.append(np.nan)

    # --- ПРИСВАИВАНИЕ И РАСЧЕТ СТАТИСТИК ---
    temp_azimuth_df = pd.DataFrame({
        'geographic_azimuth': geographic_azimuths,
        'magnetic_azimuth': magnetic_azimuths
    })
    temp_azimuth_df.index = pairs_df.index
    df = df.join(temp_azimuth_df, how='left')

    df['geo_diff'] = (df['heading'] - df['geographic_azimuth'] + 180) % 360 - 180
    df['mag_diff'] = (df['heading'] - df['magnetic_azimuth'] + 180) % 360 - 180

    stats_df = df[['geo_diff', 'mag_diff']].dropna()
    valid_points_count = len(stats_df)

    if valid_points_count == 0:
        print(f"Пропускаем файл {name}, так как нет валидных точек для анализа курса.")
        return

    # --- ВЫЧИСЛЕНИЕ СТАТИСТИК (ИСПРАВЛЕНО) ---
    temp_result = {}
    # Добавляем имя файла в результат для корректного сохранения
    temp_result['filename'] = name
    temp_result['valid_points_count'] = valid_points_count

    for col_name in ['geo_diff', 'mag_diff']:
        prefix = 'geographic' if 'geo' in col_name else 'magnetic'
        temp_result[f'{prefix}_diff_mean'] = stats_df[col_name].mean()
        temp_result[f'{prefix}_diff_min'] = stats_df[col_name].min()
        temp_result[f'{prefix}_diff_max'] = stats_df[col_name].max()
        for p in [25, 50, 75, 90, 95]:
            temp_result[f'{prefix}_{p}p'] = stats_df[col_name].quantile(p / 100.0)

    result_df = pd.DataFrame([temp_result])
    print(f"\n--- Итоговые результаты по разнице курсов для {name} ---")
    print(result_df)

    path = DefaultLocate.DATA_DIR / f"{name}_course_comparison.csv"
    result_df.to_csv(path)
    print(f"\nРезультаты по курсу для {name} сохранены в файл: {path}")


def main():
    path_file = DefaultLocate.DATA_PREPROCESSED_DIR / "5_part_2.csv"
    df = pd.read_csv(path_file)

    mark_time_true = (27789404, 27848806)
    mark_time_anomaly = (27778143, 27789404)

    df_true = df[(df["time"] >= mark_time_true[0]) & (df["time"] <= mark_time_true[1])].copy()
    df_anomaly = df[(df["time"] >= mark_time_anomaly[0]) & (df["time"] <= mark_time_anomaly[1])].copy()

    time_true, lon_true, lat_true = df_true["time"], df_true["lat"], df_true["lon"]
    time_anomaly, lat_anomaly, lon_anomaly = df_anomaly["time"], df_anomaly["lon"], df_anomaly["lat"]

    path = DefaultLocate.OUTPUT_DIR
    IOPs_geojson.write_geojson_from_arrays(
        output_path=path / "true_positions.geojson",
        list_arrays=[[time_true, lon_true, lat_true]],
    )
    IOPs_geojson.write_geojson_from_arrays(
        output_path=path / "anomaly_positions.geojson",
        list_arrays=[[time_anomaly, lon_anomaly, lat_anomaly]],
    )

    # Вызываем функции для обоих датафреймов
    speed_comparison(df_true, "true_positions")
    speed_comparison(df_anomaly, "anomaly_positions")
    source_comparison(df_true, "true_positions")
    source_comparison(df_anomaly, "anomaly_positions")


if __name__ == '__main__':
    main()