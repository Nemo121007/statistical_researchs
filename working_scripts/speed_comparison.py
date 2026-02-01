import datetime
import pandas as pd
import numpy as np

from settings.settings import DefaultLocate

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


def main():
    path_file = DefaultLocate.DATA_PREPROCESSED_DIR
    files = [file.name for file in path_file.glob('*.csv')]

    results_list = []

    for file in files:
        print(f"Processing file: {path_file / file}")
        df = pd.read_csv(path_file / file)

        # Предполагаем, что в df есть колонка 'speed' с истинной скоростью
        if 'speed' not in df.columns:
            print(f"Пропускаем файл {file}, так как отсутствует колонка 'speed'.")
            continue

        # --- РАСЧЕТ СКОРОСТИ ---

        # Рассчитываем разницу во времени между последовательными точками
        df["diff_time"] = df["time"].diff()

        # Рассчитываем расстояние по гаверсинусу между последовательными точками
        df["distance_haversine"] = haversine_distance(
            np.radians(df["lat"].shift(1)),
            np.radians(df["lon"].shift(1)),
            np.radians(df["lat"]),
            np.radians(df["lon"])
        )

        # Рассчитываем производную скорость (м/с)
        df["speed_haversine"] = df["distance_haversine"] / df["diff_time"]

        # --- ПОДГОТОВКА ДАННЫХ ДЛЯ СТАТИСТИКИ ---

        # Заменяем бесконечность (inf) на NaN
        df["speed_haversine"].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Переименуем исходную колонку для ясности
        df.rename(columns={'speed': 'speed_true'}, inplace=True)

        # Вычисляем разницу с истинной скоростью
        df['speed_diff'] = df['speed_true'] - df['speed_haversine']

        # Создаем DataFrame только с нужными колонками и удаляем все строки с NaN
        # Это гарантирует, что мы считаем статистику только по точкам,
        # где есть и истинная скорость, и рассчитанная.
        stats_df = df[['speed_haversine', 'speed_diff']].dropna()

        # Сохраняем количество обработанных (валидных) точек
        valid_points_count = len(stats_df)

        # Проверяем, есть ли валидные данные для расчета статистик
        if valid_points_count == 0:
            print(f"Пропускаем файл {file}, так как нет валидных точек для анализа скорости.")
            continue

        # --- ВЫЧИСЛЕНИЕ СТАТИСТИК ---
        temp_result = {'filename': file}
        temp_result['valid_points_count'] = valid_points_count

        # Рассчитываем статистику для производной скорости и для разницы
        for col_name in ['speed_haversine', 'speed_diff']:
            # Определяем префикс для названия колонок в результате
            if col_name == 'speed_haversine':
                prefix = 'speed_derived'
            else:
                prefix = 'speed_diff'

            temp_result[f'{prefix}_mean_mps'] = stats_df[col_name].mean()
            temp_result[f'{prefix}_min_mps'] = stats_df[col_name].min()
            temp_result[f'{prefix}_max_mps'] = stats_df[col_name].max()

            for p in [25, 50, 75, 90, 95]:
                temp_result[f'{prefix}_{p}p_mps'] = stats_df[col_name].quantile(p / 100.0)

        print(f"  - Результаты для {file}: {temp_result}")
        results_list.append(temp_result)

    # --- ФИНАЛЬНОЕ СОХРАНЕНИЕ ---
    if results_list:
        result_df = pd.DataFrame(results_list)
        print("\n--- Итоговые результаты по скорости ---")
        # Устанавливаем 'filename' как индекс для лучшей читаемости
        result_df.set_index('filename', inplace=True)
        print(result_df)

        # Сохраняем в отдельный файл
        path = DefaultLocate.DATA_DIR / "results_speed_comparison.csv"
        result_df.to_csv(path)
        print(f"\nРезультаты по скорости сохранены в файл: {path}")
    else:
        print("Нет файлов для обработки или не удалось получить результаты.")


if __name__ == '__main__':
    main()