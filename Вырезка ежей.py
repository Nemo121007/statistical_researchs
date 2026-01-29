from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString
from scipy.signal import savgol_filter

from shared_files.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle
from utils.IOPs.IOPs_geojson import IOPs_geojson

def filter_hedgehog(list_lon: List[float], list_lat: List[float], step: int = 10):

    lon_past = list_lon[:-step]
    lon_next = list_lon[step:]
    lat_past = list_lat[:-step]
    lat_next = list_lat[step:]
    distance = CalculatorDistancesLengthLargeCircle.vectorized_great_circle_distance(
        lat1=np.array(lat_past),
        lon1=np.array(lon_past),
        lat2=np.array(lat_next),
        lon2=np.array(lon_next)
    )
    # Строим интервальный вариационный ряд
    n_intervals = 500
    hist, bin_edges = np.histogram(distance, bins=n_intervals)

    # Создаём DataFrame для наглядности
    variation_series = pd.DataFrame({
        'Интервал_от': bin_edges[:-1],
        'Интервал_до': bin_edges[1:],
        'Частота': hist,
        'Относительная_частота': hist / len(distance),
        'Накопленная_частота': np.cumsum(hist)
    })

    print("\n--- Интервальный вариационный ряд (расстояния между точками) ---")
    print(variation_series.to_string(index=False))

    print(f"\nВсего наблюдений: {len(distance)}")
    print(f"Среднее расстояние: {np.mean(distance):.2f} м")
    print(f"Медиана: {np.median(distance):.2f} м")
    print(f"Стандартное отклонение: {np.std(distance):.2f} м")

    result_list_lat, result_list_lon = [], []
    for i in range(10, len(list_lon) - 10):
        if distance[i - 10] > 300:
            result_list_lat.append(list_lat[i])
            result_list_lon.append(list_lon[i])
    print(f"\nПосле фильтрации ёжика осталось точек: {len(result_list_lat)}")

    return result_list_lon, result_list_lat

def print_count_NaN(df):
    # Подсчет пропущенных значений в каждом столбце
    null_lat = df['lat'].isna().sum()
    null_lon = df['lon'].isna().sum()

    # Подсчет строк, где хотя бы одно из полей пропущено
    null_any = df[['lat', 'lon']].isna().any(axis=1).sum()

    # Подсчет строк, где оба поля пропущены
    null_both = df[['lat', 'lon']].isna().all(axis=1).sum()

    print(f"\nПропущенные значения:")
    print(f"  - В столбце 'lat': {null_lat}")
    print(f"  - В столбце 'lon': {null_lon}")
    print(f"  - Строк с пропущенными lat ИЛИ lon: {null_any}")
    print(f"  - Строк с пропущенными lat И lon (оба): {null_both}")


def interpolate_missing_coords(df, lat_col='lat', lon_col='lon'):
    """
    Заменяет NaN/null значения координат линейной интерполяцией между известными точками.

    Args:
        df: DataFrame с координатами
        lat_col: название колонки с широтой
        lon_col: название колонки с долготой

    Returns:
        DataFrame с интерполированными координатами
    """
    df = df.copy()

    # Заменяем null на NaN для единообразной обработки
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

    # Линейная интерполяция для каждой колонки
    df[lat_col] = df[lat_col].interpolate(method='linear', limit_direction='both')
    df[lon_col] = df[lon_col].interpolate(method='linear', limit_direction='both')

    return df

# 1. Читаем данные
df_read = pd.read_csv('данные1.csv')

# 2. СОРТИРУЕМ ДАННЫЕ ПО СТОЛБЦУ "v0" и сохраняем результат
#    Указываем ignore_index=True, чтобы сбросить старые индексы и создать новые (0, 1, 2...)
df_read = df_read.sort_values("v0", ignore_index=True)
df_read = interpolate_missing_coords(df_read, lat_col='v1', lon_col='v2')
# 3. Теперь извлекаем отсортированные данные
lons = df_read["v2"]
lats = df_read["v1"]

# 4. Создаем DataFrame для анализа
df = pd.DataFrame()
df["lon"] = lons
df["lat"] = lats
df["time"] = np.arange(len(df))  # Временные метки как последовательные числа


print("--- Пример данных (первые 5 точек ПОСЛЕ сортировки) ---")
print(df.head())


# --- 2. Функция для расчета расстояния ---
def calculate_total_distance(points_df, name):
    """
    Рассчитывает суммарное расстояние между последовательными точками.
    """
    total_distance = 0
    for i in range(len(points_df) - 1):
        point1 = (points_df.iloc[i]['lat'], points_df.iloc[i]['lon'])
        point2 = (points_df.iloc[i + 1]['lat'], points_df.iloc[i + 1]['lon'])
        total_distance += geodesic(point1, point2).meters

    # Преобразуем DataFrame в массивы для write_geojson_from_arrays
    time_array = points_df['time'].values if 'time' in points_df.columns else np.arange(len(points_df))
    lat_array = points_df['lat'].values
    lon_array = points_df['lon'].values

    IOPs_geojson.write_geojson_from_arrays(
        output_path=Path(__file__).parent / f"{name}_output.geojson",
        list_arrays=[[time_array, lat_array, lon_array]],  # Список с одним элементом [time, lat, lon]
    )
    return total_distance


dist_haversin = np.nansum(CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
    lat_array=df['lat'].to_numpy(),
    lon_array=df['lon'].to_numpy()
))
print_count_NaN(df)

list_lon, list_lat = filter_hedgehog(
    list_lon=df['lon'].tolist(),
    list_lat=df['lat'].tolist(),
)
result_df = pd.DataFrame({
    'lon': list_lon,
    'lat': list_lat
})
dist_hedgehog = calculate_total_distance(result_df, "hedgehog")


# --- 3. "Наивный" расчет на (теперь отсортированных) шумных данных ---
# noisy_distance = calculate_total_distance(df, "start")
print(f"\n--- Расчет расстояния ---")
print(f"Расстояние по формуле гаверсин: {dist_haversin:.2f} метров = {dist_haversin/1000:.2f} км = {dist_haversin / 1000 /1.85:.2f} м. миль")
# print(f"Расстояние на исходных данных: {noisy_distance:.2f} метров = {noisy_distance/1000:.2f} км = {noisy_distance / 1000 /1.85:.2f} м. миль")
print(f"Расстояние после фильтрации ёжика: {dist_hedgehog:.2f} метров = {dist_hedgehog/1000:.2f} км = {dist_hedgehog / 1000 /1.85:.2f} м. миль")


# --- 4. Очистка и сглаживание данных ---

# Метод 1: Упрощение (Алгоритм Дугласа-Пекера)
line = LineString(zip(df['lon'], df['lat']))
# tolerance можно подбирать. 0.00005 ~ 5.5 метров. Начните с этого значения.
simplified_line = line.simplify(tolerance=0.00005, preserve_topology=True)
simplified_coords = list(simplified_line.coords)
df_simplified = pd.DataFrame(simplified_coords, columns=['lon', 'lat'])

# Метод 2: Сглаживание (Фильтр Савицки-Голея)
# window_length должен быть нечетным и меньше длины данных
# polyorder - порядок полинома (обычно 2 или 3)
window_length = 21 # должно быть нечетным
polyorder = 3
df_smoothed = df.copy()
df_smoothed['lat'] = savgol_filter(df['lat'], window_length, polyorder)
df_smoothed['lon'] = savgol_filter(df['lon'], window_length, polyorder)


# --- 5. Расчет расстояния на очищенных данных ---
simplified_distance = calculate_total_distance(df_simplified, "simplified")
smoothed_distance = calculate_total_distance(df_smoothed, "smoothed")

print(f"Расстояние после упрощения (Дуглас-Пекер): {simplified_distance:.2f} метров = {simplified_distance/1000:.2f} км = {simplified_distance / 1.85 / 1000:.2f} м. миль")
print(f"Расстояние после сглаживания (Савицки-Голей): {smoothed_distance:.2f} метров = {smoothed_distance/1000:.2f} км = {smoothed_distance / 1.85 / 1000:.2f} м. миль")
