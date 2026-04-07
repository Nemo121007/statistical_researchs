"""Скрипт для измерения и сравнения производительности и точности"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic

Earth_radius_meters: float = 6371000


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние по формуле гаверсинуса.
    Координаты должны быть в радианах. Возвращает расстояние в метрах.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return Earth_radius_meters * c


def spherical_law_of_cosines_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние по сферической теореме косинусов.
    Координаты должны быть в радианах. Возвращает расстояние в метрах.
    """
    dlon = lon2 - lon1
    cos_angle = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    # Обрезаем значения для численной стабильности
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    c = np.arccos(cos_angle)
    return Earth_radius_meters * c


def main():
    # pylint: disable=missing-function-docstring
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    path_file = Path(__file__).parent
    files = [file.name for file in path_file.glob("*.csv")]

    results_list = []

    for file in files:
        print(f"Processing file: {path_file / file}")
        df = pd.read_csv(path_file / file)
        # Удаляем строки, где есть NaN в 'lat' или 'lon'
        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

        if len(df) < 2:
            print(f"Пропускаем файл {file}, так как в нем менее 2 точек.")
            continue

        temp_result = {"filename": file}

        lat_rad = np.radians(df["lat"])
        lon_rad = np.radians(df["lon"])

        lat1 = lat_rad.shift(1)
        lon1 = lon_rad.shift(1)
        lat2 = lat_rad
        lon2 = lon_rad

        # Удаляем первую строку, так как для нее нет предыдущей точки
        calc_data = pd.DataFrame(
            {"lat1": lat1[1:], "lon1": lon1[1:], "lat2": lat2[1:], "lon2": lon2[1:]}
        ).dropna()

        # Метод сферической теоремы косинусов
        start_time = time.perf_counter()
        dist_cosine = spherical_law_of_cosines_distance(
            calc_data["lat1"].values,
            calc_data["lon1"].values,
            calc_data["lat2"].values,
            calc_data["lon2"].values,
        )
        temp_result["cosine_time_sec"] = time.perf_counter() - start_time

        # Метод гаверсинуса
        start_time = time.perf_counter()
        dist_haversine = haversine_distance(
            calc_data["lat1"].values,
            calc_data["lon1"].values,
            calc_data["lat2"].values,
            calc_data["lon2"].values,
        )
        temp_result["haversine_time_sec"] = time.perf_counter() - start_time

        # geopy
        start_time = time.perf_counter()
        # Конвертируем радианы обратно в градусы для geopy
        lat1_deg = np.degrees(calc_data["lat1"])
        lon1_deg = np.degrees(calc_data["lon1"])
        lat2_deg = np.degrees(calc_data["lat2"])
        lon2_deg = np.degrees(calc_data["lon2"])

        dist_geopy = [
            geodesic((lat1, lon1), (lat2, lon2)).meters
            for lat1, lon1, lat2, lon2 in zip(lat1_deg, lon1_deg, lat2_deg, lon2_deg)
        ]
        temp_result["geopy_time_sec"] = time.perf_counter() - start_time

        # Geopy как эталон
        comparison_df = pd.DataFrame(
            {"haversine": dist_haversine, "cosine": dist_cosine, "geopy": dist_geopy}
        )

        # Вычисляем и сохраняем суммарное расстояние для каждого метода
        total_haversine = np.sum(dist_haversine)
        total_cosine = np.sum(dist_cosine)
        total_geopy = sum(dist_geopy)

        temp_result["haversine_total_m"] = total_haversine
        temp_result["cosine_total_m"] = total_cosine
        temp_result["geopy_total_m"] = total_geopy

        # Вычисляем и сохраняем разницу суммарных расстояний с эталоном (geopy)
        temp_result["haversine_total_diff_m"] = total_haversine - total_geopy
        temp_result["cosine_total_diff_m"] = total_cosine - total_geopy

        for method in ["haversine", "cosine"]:
            error = comparison_df[method] - comparison_df["geopy"]
            abs_error = error.abs()

            # Минимальная ошибка (минимальный модуль ошибки)
            temp_result[f"{method}_min_abs_error_m"] = abs_error.min()

            # Максимальная ошибка (максимальный модуль ошибки)
            temp_result[f"{method}_max_abs_error_m"] = abs_error.max()

            # Максимальная переоценка (максимальная положительная ошибка)
            positive_errors = error[error > 0]
            temp_result[f"{method}_max_overestimation_m"] = (
                positive_errors.max() if not positive_errors.empty else 0.0
            )

            # Максимальная недооценка (максимальная отрицательная ошибка по модулю)
            negative_errors = error[error < 0]
            temp_result[f"{method}_max_underestimation_m"] = (
                abs(negative_errors.min()) if not negative_errors.empty else 0.0
            )

            # Средняя ошибка (сохраняем для понимания смещения)
            temp_result[f"{method}_mean_error_m"] = error.mean()

            # Перцентили ошибок
            for p in [25, 50, 75, 90, 95]:
                temp_result[f"{method}_{p}p_error_m"] = error.quantile(p / 100.0)

        print(f"  - Результаты для {file}: {temp_result}")

        results_list.append(temp_result)

    if results_list:
        result_df = pd.DataFrame(results_list)
        print("\n--- Итоговые результаты ---")
        print(result_df)

        path = Path(__file__).parent / "results_distance_comparison.csv"
        # Сохранить result в CSV файл
        result_df.to_csv(path, index=False)
        print(f"\nРезультаты сохранены в файл: {path}")
    else:
        print("Нет файлов для обработки.")


if __name__ == "__main__":
    main()
