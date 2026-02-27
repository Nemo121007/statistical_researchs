import pandas as pd
import numpy as np
import datetime
from pathlib import Path

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
    valid_groups = df['validity_point'].replace(0, np.nan)

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
    mask = (valid_groups != valid_groups.shift(1))

    # cumsum создает уникальный идентификатор для каждой непрерывной группы
    group_ids = mask.cumsum()

    # Разбиваем исходный DataFrame по group_ids
    # groupby вернет кортежи (id, DataFrame)
    df_list = [group_df for _, group_df in df.groupby(group_ids)]

    # Опционально: сбрасываем индексы в полученных кусках
    df_list = [part.reset_index(drop=True) for part in df_list]

    return df_list


if __name__ == "__main__":
    name = "2_part_1.csv"
    df = load_csv(name)

    # Вызываем функцию разделения
    parts = discretize_path(df, count_max_len=0)

    print(f"\nПолучено частей: {len(parts)}")
    for i, part in enumerate(parts):
        print(f"\nЧасть {i + 1}:")
        print(part[['time', 'validity_point']])