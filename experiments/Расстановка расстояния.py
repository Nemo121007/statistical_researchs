import pandas as pd
import numpy as np
import datetime
from pathlib import Path

from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate

data_distance = [
    (datetime.datetime(year=2025, month=7, day=1, hour=0), 0),
    (datetime.datetime(year=2025, month=7, day=2, hour=0), 108),
    (datetime.datetime(year=2025, month=7, day=3, hour=0), 127.4),
    (datetime.datetime(year=2025, month=7, day=4, hour=0), 63.7),
    (datetime.datetime(year=2025, month=7, day=5, hour=0), 170.6),
    (datetime.datetime(year=2025, month=7, day=6, hour=0), 29.7),
    (datetime.datetime(year=2025, month=7, day=7, hour=0), 177.6),
    (datetime.datetime(year=2025, month=7, day=8, hour=0), 69.1),
    (datetime.datetime(year=2025, month=7, day=9, hour=0), 142.5),
    (datetime.datetime(year=2025, month=7, day=10, hour=0), 98.3),
    (datetime.datetime(year=2025, month=7, day=11, hour=0), 91),
    (datetime.datetime(year=2025, month=7, day=12, hour=0), 0),
    (datetime.datetime(year=2025, month=7, day=13, hour=0), 0),
    (datetime.datetime(year=2025, month=7, day=14, hour=0), 36.5),
    (datetime.datetime(year=2025, month=7, day=15, hour=0), 72),
    (datetime.datetime(year=2025, month=7, day=16, hour=0), 141.2),
    (datetime.datetime(year=2025, month=7, day=17, hour=0), 89.6),
    (datetime.datetime(year=2025, month=7, day=18, hour=0), 121.5),
    (datetime.datetime(year=2025, month=7, day=19, hour=0), 91.3),
    (datetime.datetime(year=2025, month=7, day=20, hour=0), 101.5),
    (datetime.datetime(year=2025, month=7, day=21, hour=0), 136),
    (datetime.datetime(year=2025, month=7, day=22, hour=0), 57.8),
    (datetime.datetime(year=2025, month=7, day=23, hour=0), 164.1),
    (datetime.datetime(year=2025, month=7, day=24, hour=0), 70),
    (datetime.datetime(year=2025, month=7, day=25, hour=0), 84.5),
    (datetime.datetime(year=2025, month=7, day=26, hour=0), 108),
    (datetime.datetime(year=2025, month=7, day=27, hour=0), 101.5),
    (datetime.datetime(year=2025, month=7, day=28, hour=0), 131.2),
    (datetime.datetime(year=2025, month=7, day=29, hour=0), 0),
    (datetime.datetime(year=2025, month=7, day=30, hour=0), 0),
    (datetime.datetime(year=2025, month=7, day=31, hour=0), 0),
]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_distance(raw_df: pd.DataFrame, anonimize_df: pd.DataFrame, data_distance: list[tuple[datetime.datetime, float]]):
    """
    Проставляет контрольные расстояния, пропуская NaN значения в raw_df.
    """
    # 1. Приводим время к datetime
    raw_df['time'] = pd.to_datetime(raw_df['time'])

    # Создаем столбец control_distance
    anonimize_df['control_distance'] = np.nan

    # Определяем колонки для сопоставления (те, что есть в обоих df, исключая time и control_distance)
    common_cols = list(set(raw_df.columns) & set(anonimize_df.columns))
    cols_to_match = [col for col in common_cols if col not in ['time', 'control_distance']]

    print(f"Колонки для поиска совпадений: {cols_to_match}")

    # Предварительно считаем маску валидности данных для raw_df (нет NaN в ключевых полях)
    # Это ускорит работу внутри цикла
    raw_valid_mask = raw_df[cols_to_match].notna().all(axis=1)

    for item in data_distance:
        # Проверка корректности входных данных
        if len(item) < 2:
            print(f"Предупреждение: Некорректный формат данных {item}. Пропуск.")
            continue

        target_time, distance_val = item

        # 1. Ищем кандидатов по времени
        time_mask = raw_df['time'] >= target_time

        # 2. Комбинируем с маской валидности (время >= искомого И данные не NaN)
        final_search_mask = time_mask & raw_valid_mask

        if not final_search_mask.any():
            print(f"Предупреждение: Не найдено валидных строк (без NaN) после времени {target_time} в raw_df.")
            continue

        # Берем индекс первой подходящей строки
        found_index = final_search_mask.idxmax()
        found_row_raw = raw_df.loc[found_index]

        # 3. Ищем совпадение в anonimize_df по значениям
        match_mask = pd.Series(True, index=anonimize_df.index)

        for col in cols_to_match:
            # Сравниваем значения
            match_mask &= (anonimize_df[col] == found_row_raw[col])

        matched_rows = anonimize_df[match_mask]

        if matched_rows.empty:
            # Если совпадения нет, выводим отладочную информацию
            print(f"Ошибка: Не найдено совпадение в anonimize_df.")
            print(f"Время поиска: {target_time}")
            print(f"Найденная строка raw_df (index {found_index}): {found_row_raw[cols_to_match].to_dict()}")
            # Можно раскомментировать raise, если нужно остановить выполнение
            # raise ValueError("Данные не совпадают")
            continue

        found_index_anon = matched_rows.index[0]

        if len(matched_rows) > 1:
            print(f"Предупреждение: Найдено несколько совпадений для времени {target_time}. Используется первое (индекс {found_index_anon}).")

        # 4. Записываем расстояние
        anonimize_df.loc[found_index_anon, 'control_distance'] = distance_val

    # 5. Сохранение
    output_path = DefaultLocate.DATA_POSTPROCESSED_DIR / "anonimize_result.csv"
    anonimize_df.to_csv(output_path, index=True)
    print(f"Файл успешно сохранен: {output_path}")


if __name__ == "__main__":
    raw_df = load_csv(path=(DefaultLocate.DATA_DIR.parent / "raw" / "2.csv"))
    anonimize_df = load_csv(path=(DefaultLocate.DATA_POSTPROCESSED_DIR / "example.csv"))

    write_distance(raw_df, anonimize_df, data_distance)