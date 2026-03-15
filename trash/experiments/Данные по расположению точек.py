from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from help_scripts.area_collector import AreaCollector
from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate
from shapely.geometry import Point
from tqdm import tqdm


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def check_locate(data: pd.DataFrame, area_collector: AreaCollector) -> List[bool]:
    """
    Проверяет, находится ли каждая точка из data внутри какой-либо области из area_collector.
    Возвращает список булевых значений.
    """
    results = []

    # Итерируемся по строкам датафрейма
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Проверка точек"):
        # Создаем объект Point (x=lon, y=lat)
        point = Point(row["lon"], row["lat"])

        # Получаем кандидатов по bounding box (предварительная фильтрация)
        list_area = area_collector.get_areas_by_bounding_box(
            min_lon=row["lon"],
            max_lon=row["lon"],
            min_lat=row["lat"],
            max_lat=row["lat"],
        )

        is_inside = False
        # Проверяем каждого кандидата точной геометрией
        for area in list_area.values():
            if area.shapely_polygon.contains(point):
                is_inside = True
                break  # Если нашли хотя бы одну область, можно прекращать проверку

        results.append(is_inside)

    return results


if __name__ == "__main__":
    path = DefaultLocate.DATA_POSTPROCESSED_DIR / "example.csv"
    df = load_csv(path)
    print(f"Загруженные данные из {path}:")

    path = DefaultLocate.DATA_DIR / "europe_light.geojson"
    _, _, area_collector = IOPs_geojson.fast_read_json(path)
    print(f"Загруженные данные из {path}:")

    # 3. Проверка вхождения
    print("Начинаю проверку вхождения точек...")
    mask_inside_polygon = check_locate(df, area_collector)

    df["in_water"] = mask_inside_polygon

    # 4. Статистика
    count_inside = df["in_water"].sum()
    print(f"Количество точек, расположенных внутри area: {count_inside} из {len(df)}")

    # 5. Сохранение результата
    output_path = DefaultLocate.DATA_POSTPROCESSED_DIR / "example_located.csv"
    df.to_csv(output_path, index=True)
    print(f"Результат успешно сохранен в файл: {output_path}")
