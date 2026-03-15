import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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


def write_distance(
    raw_df: pd.DataFrame,
    anonimize_df: pd.DataFrame,
    data_distance: list[tuple[datetime.datetime, float]],
):
    """
    1. Добавляет validity_point из anonimize_df в raw_df.
    2. Проставляет control_distance в raw_df.
    3. Сохраняет копию raw_df под названием 2_full.
    4. Анонимизирует время и сохраняет под названием example.
    """

    # --- Шаг 1: Подготовка данных и добавление validity_point ---

    # Приводим время к datetime и сортируем raw_df
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    raw_df.sort_values(by="time", inplace=True, ignore_index=True)

    raw_df["validate_point"] = anonimize_df["validity_point"]

    # Инициализируем столбец расстояний
    raw_df["control_distance"] = np.nan

    # --- Шаг 2: Проставление расстояний ---
    cum_distance_val = 0
    for item in data_distance:
        target_time, distance_val = item
        cum_distance_val += distance_val

        # Ищем первую строку в raw_df, где время >= target_time И данные валидны
        time_mask = raw_df["time"] >= target_time

        if not time_mask.any():
            print(
                f"Предупреждение: Не найдено валидных строк после времени {target_time}."
            )
            continue

        found_index = time_mask.idxmax()

        # Записываем расстояние прямо в raw_df
        raw_df.loc[found_index, "control_distance"] = cum_distance_val

    # --- Шаг 3: Сохранение полной версии (2_full) ---
    path_full = DefaultLocate.DATA_POSTPROCESSED_DIR / "2_full.csv"
    # Восстанавливаем исходный формат времени (строка) для сохранения, если нужно, или оставляем ISO
    # Обычно при сохранении в CSV pandas сам приведет дату к строке.
    raw_df.to_csv(path_full, index=True)
    print(f"Файл 2_full успешно сохранен: {path_full}")

    # --- Шаг 4: Анонимизация и сохранение (example) ---
    # Анонимизируем время: считаем секунды от первой записи
    start_time = raw_df["time"].iloc[0]
    raw_df["time"] = (raw_df["time"] - start_time).dt.total_seconds()

    path_example = DefaultLocate.DATA_POSTPROCESSED_DIR / "example.csv"
    raw_df.to_csv(path_example, index=True)
    print(f"Файл example успешно сохранен: {path_example}")

    # --- Шаг 5: Вывод статистики ---
    not_nan_count = raw_df["control_distance"].notna().sum()
    print(
        f"\nКоличество строк в df example, у которых control_distance != NaN: {not_nan_count}"
    )


if __name__ == "__main__":
    raw_df = load_csv(path=(DefaultLocate.DATA_DIR.parent / "raw" / "2.csv"))
    anonimize_df = load_csv(path=(DefaultLocate.DATA_POSTPROCESSED_DIR / "example.csv"))

    write_distance(raw_df, anonimize_df, data_distance)
