from pathlib import Path

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString
from scipy.signal import savgol_filter

from help_scripts.IOPs_geojson import IOPs_geojson


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

# --- 3. "Наивный" расчет на (теперь отсортированных) шумных данных ---
noisy_distance = calculate_total_distance(df, "start")
print(f"\n--- Расчет расстояния ---")
print(f"Расстояние на исходных данных: {noisy_distance:.2f} метров = {noisy_distance/1000:.2f} км = {noisy_distance / 1000 /1.85:.2f} м. миль")


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
