import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from help_scripts.calculator_distances_length_large_circle import (
    CalculatorDistancesLengthLargeCircle,
)
from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate


def load_csv(name: str) -> pd.DataFrame:
    """
    Загружает CSV файл в DataFrame.
    """
    path = DefaultLocate.DATA_POSTPROCESSED_DIR / name
    return pd.read_csv(path)
    # # Для примера создадим тестовый датафрейм, если файл не грузится:
    # data = {
    #     'time': pd.date_range(start='2023-01-01', periods=10, freq='h'),
    #     'lat': np.random.rand(10) * 60,
    #     'lon': np.random.rand(10) * 30,
    #     'satellites': np.random.randint(5, 15, 10),
    #     'speed': np.random.rand(10) * 10,
    #     'heading': np.random.rand(10) * 360,
    #     'validity_point': [1, 1, 0, 2, 2, 0, 0, 1, 0, 3]  # Пример данных
    # }
    # return pd.DataFrame(data)


def discretize_path(df: pd.DataFrame, count_max_len: int):
    """
    Разделяет DataFrame на список DataFrame на основе столбца validity_point.

    Логика:
    1. Группируем последовательные строки с одинаковым validity_point.
    2. Если validity_point == 0, строка присоединяется к группе предыдущей строки.

    Args:
        df: Исходный DataFrame с данными судна.
        count_max_len: (Не используется в текущей логике, но оставлен для совместимости с сигнатурой)

    Returns:
        list[pd.DataFrame]: Список разделенных DataFrame.
    """
    if df.empty:
        return []

    # Создаем вспомогательную серию для определения групп.
    # Заменяем 0 на NaN, чтобы потом заполнить их предыдущим значением.
    # Это позволяет "0" принадлежать той же группе, что и предыдущая строка.
    valid_groups = df["validity_point"].replace(0, np.nan)

    # Заполняем NaN значениями сверху (forward fill).
    # Теперь, например, последовательность [1, 1, 0, 2] станет [1, 1, 1, 2]
    valid_groups = valid_groups.ffill()

    # Если первые строки были 0, ffill не сработает (нечем заполнять).
    # Заполняем оставшиеся NaN нулями (или можно оставить как есть,
    # но для стабильности группировки лучше заполнить).
    valid_groups = valid_groups.fillna(0)

    # Определяем моменты смены группы.
    # Сравниваем текущее значение со сдвинутым (предыдущим).
    # shift(1) сдвигает на одну строку вниз.
    mask = valid_groups != valid_groups.shift(1)

    # cumsum создает уникальный идентификатор для каждой непрерывной группы
    group_ids = mask.cumsum()

    # Разбиваем исходный DataFrame по group_ids
    # groupby вернет кортежи (id, DataFrame)
    df_list = [group_df for _, group_df in df.groupby(group_ids)]

    # Опционально: сбрасываем индексы в полученных кусках
    df_list = [part.reset_index(drop=True) for part in df_list]

    return df_list


if __name__ == "__main__":
    # name = "2_part_1.csv"
    # df = load_csv(name)
    #
    # # Вызываем функцию разделения
    # parts = discretize_path(df, count_max_len=0)
    #
    # print(f"\nПолучено частей: {len(parts)}")
    # for i, part in enumerate(parts):
    #     print(f"\nЧасть {i + 1}:")
    #     print(part[['time', 'validity_point']])

    # Загрузка данных
    path = DefaultLocate.DATA_POSTPROCESSED_DIR / "example_located.csv"
    df = pd.read_csv(path)
    print(f"Загруженные данные из {path}: {len(df)} строк")

    # --- Условие 1: satellites < 2 ---
    # Заменяем lat, lon на NaN, validate_point на -1
    mask_satellites = df["satellites"] < 2

    df.loc[mask_satellites, "lat"] = np.nan
    df.loc[mask_satellites, "lon"] = np.nan
    df.loc[mask_satellites, "validate_point"] = -1

    # --- Условие 2: in_water = False и validate_point = 1 ---
    # Заменяем lat, lon на NaN, validate_point на -1
    mask_land = df["in_water"] == False

    df.loc[mask_land, "lat"] = np.nan
    df.loc[mask_land, "lon"] = np.nan
    df.loc[mask_land, "validate_point"] = -1

    distance = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
        lat_array=df["lat"].to_numpy(),
        lon_array=df["lon"].to_numpy(),
    )
    distance = np.concatenate(([0], distance))
    df["distance"] = distance

    # Сохранение результатов в новом файле
    output_path = DefaultLocate.DATA_POSTPROCESSED_DIR / "example_cleaned.csv"
    df.to_csv(output_path, index=True)

    print(f"Обработка завершена. Файл сохранен: {output_path}")
    print(f"Статистика изменений:")
    print(f"  Точек с satellites < 2 (статус -1): {mask_satellites.sum()}")
    print(f"  Точек на суше (статус -1): {mask_land.sum()}")
    print(f"  Всего точек с статусом -1: {(mask_satellites | mask_land).sum()}")
    print(
        f"  Всего валидных точек (статус != -1): {len(df) - (mask_satellites | mask_land).sum()}"
    )
