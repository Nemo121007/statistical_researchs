import pandas as pd
from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate

name = "2_part_1.csv"

def get_mask_from_segments(df, segments, column_name="time"):
    """
    Вспомогательная функция для создания маски по интервалам.
    """
    if not segments:
        return pd.Series(False, index=df.index)
    
    mask = pd.Series(False, index=df.index)
    for start, end in segments:
        mask |= (df[column_name] >= start) & (df[column_name] <= end)
    return mask


def main():
    # Чтение исходного файла
    df = pd.read_csv(DefaultLocate.DATA_PREPROCESSED_DIR / name)

    # --- 1. Определение списков интервалов ---

    # 1 список: интервалы идентификаторов, которые необходимо вывести в json (область интереса)
    list_output_segments = [
        (55 * 100 * 1000, 60 * 100 * 1000),     # Пример: весь диапазон или конкретные участки
    ]

    # 2 список: интервалы идентификаторов валидных точек
    list_valid_segments = [
        (2659075, 2659475),
        (3972308, 3991859),
        (3992579, 3993009),
        (3994319, 4069774),
        (4071654, 4074234),
        (4075824, 4077654),
        (4087805, 4155650),
        (4157809, 4168240),
        (4173621, 4237865),
        (4242895, 4246655),
        (4248445, 4249575),
        (4271417, 4273277),
        (4274517, 4281857),
        (4282107, 4341251),
        (4415436, 4497241),
        (4881513, 4886694),
        (4801808, 4832860),
        (4763136, 4764506),
        (4762016, 4762946),
        (4659466, 4761926),
        (4636965, 4657336),
        (4630034, 4636605),
        (4622347, 4624917),
        (4605707, 4615028),
        (4579476, 4591207),
        (4560145, 4577306),
        (4557335, 4558805),
        (4553574, 4555875),
        (4550354, 4552374),
        (4543794, 4548954),
        (4527443, 4528383),
        (4518112, 4521242),
        (4507641, 4512932),
        (4505871, 4506311),
        ####################################
        (5499312, 5499992),
        (5491572, 5498022),
        (5489742, 5490182),
        (5479731, 5487942),
        (5438759, 5471261),
        (5435998, 5437628),
        (5365194, 5434208),
        (5328212, 5363034),
        (5319671, 5326632),
        (5317401, 5319631),
        (5283469, 5285319),
        (5250957, 5258577),
        (5058105, 5062895),
        (5071625, 5072695),
        ##################################
        (5991074, 5999995),
        (5977104, 5983244),
        (5500002, 5502493),
        (5960574, 5962044),
        (5504253, 5512614),
        (5532035, 5537066),
        (5945583, 5958544),
        (5932942, 5943733),
        (5538676, 5545576),
        (5545616, 5549916),
        (5929662, 5930842),
        (5552276, 5552886),
        (5926752, 5928102),
        (5555357, 5556927),
        (5923572, 5924802),
        (5559127, 5560417),
        (5563667, 5565727),
        (5920161, 5922202),
        (5566947, 5567867),
        (5568317, 5569127),
        (5906301, 5906831),
        (5580678, 5581858),
        (5903740, 5904690),
        (5903030, 5903470),
        (5893380, 5897900),
        (5583368, 5605710),
        (5876036, 5878195),
        (5883853, 5886871),
    ]

    # --- 2. Формирование масок ---

    # Маска для областей, которые нужно вывести (Список 1)
    mask_output = get_mask_from_segments(df, list_output_segments, "time")

    # Маска для валидных сегментов (Список 2)
    mask_valid_segments = get_mask_from_segments(df, list_valid_segments, "time")

    # Маска наличия координат (не NaN)
    mask_has_coords = ~(df["lat"].isna() | df["lon"].isna())

    # --- 3. Логика заполнения столбца validity_point ---
    
    # Изначально заполняем -1 (невалидные)
    df["validity_point"] = -1

    # 0 - для null в полях lat и lon (приоритет для отсутствующих координат)
    df.loc[~mask_has_coords, "validity_point"] = 0

    # 1 - для валидных точек (должны быть в Списке 2 И иметь координаты)
    df.loc[mask_valid_segments & mask_has_coords, "validity_point"] = 1

    # --- 4. Запись модифицированного CSV ---
    # Сохраняем ПОЛНЫЙ датафрейм (копию исходного) с добавленным столбцом validity_point
    
    path_csv = DefaultLocate.DATA_POSTPROCESSED_DIR / name
    df.to_csv(path_csv, index=False)
    print(f"CSV сохранен: {path_csv}")

    # --- 5. Выгрузка GeoJSON файлов ---

    # GeoJSON 1: Точки из списка 2, которые попадают в рамки списка 1
    # Условие: (Попадает в Список 1) И (Попадает в Список 2) И (Есть координаты)
    mask_gj1 = mask_output & mask_valid_segments & mask_has_coords
    df_gj1 = df[mask_gj1]

    if not df_gj1.empty:
        path_gj1 = DefaultLocate.OUTPUT_DIR / "valid_points_intersection.geojson"
        IOPs_geojson.write_geojson_from_arrays(
            output_path=path_gj1,
            list_arrays=[[df_gj1["time"].values.tolist(), 
                          df_gj1["lat"].values.tolist(), 
                          df_gj1["lon"].values.tolist()]],
        )
        print(f"GeoJSON 1 (Валидные в рамках) сохранен: {path_gj1}")
    else:
        print("Нет данных для GeoJSON 1.")

    # GeoJSON 2: Все точки списка 1, за исключением тех, которые обозначены списком 2
    # Условие: (Попадает в Список 1) И (НЕ попадает в Список 2)
    # Примечание: Точки с NaN координатами в GeoJSON не запишутся (отфильтруются dropna или проверкой mask_has_coords при необходимости)
    
    mask_gj2 = mask_output & ~mask_valid_segments
    df_gj2 = df[mask_gj2].dropna(subset=['lat', 'lon']) # Убираем NaN, так как геометрия невозможна

    if not df_gj2.empty:
        path_gj2 = DefaultLocate.OUTPUT_DIR / "remainder_points.geojson"
        IOPs_geojson.write_geojson_from_arrays(
            output_path=path_gj2,
            list_arrays=[[df_gj2["time"].values.tolist(), 
                          df_gj2["lat"].values.tolist(), 
                          df_gj2["lon"].values.tolist()]],
        )
        print(f"GeoJSON 2 (Остальные точки) сохранен: {path_gj2}")
    else:
        print("Нет данных для GeoJSON 2.")


if __name__ == "__main__":
    main()