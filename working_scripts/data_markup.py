
import pandas as pd

from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate

# # 1_part_1.csv
#
# name = "1_part_1.csv"
# list_invalid_segments = [
#     (0,  8983500),
# ]
#
# list_valid_segments = [
#     # (10006040, 10300000),
#     (1408540, 1491780),
#     (1502060, 1508080),
#     (2449270, 2485750),
#     (3356930, 3397230),
#     (4520880, 4552920),
#     (4626500, 4650260),
#     (4871460, 4919830),
#     (5866300, 5878120),
#     (5883420, 5933640),
#     (6395060, 6404740),
#     (6487430, 6494450),
#     (6499900, 6502780),
#     (6763130, 6798730),
#     (6910080, 6999730),
#     (6999730, 7875850),
#     (7882290, 7941080),
#     (7946310, 7999940),
#     (7999950, 8213290),
#     (8543190, 8546800),
#     (8721830, 8725990),
#     (8726730, 8983500),
#     (int(10 * 1000000), 1000000000000)
# ]
# 2_part_1.csv

name = "2_part_1.csv"


def main():
    df = pd.read_csv(DefaultLocate.DATA_PREPROCESSED_DIR / name)

    list_invalid_segments = [
        (0,  0),
    ]

    list_valid_segments = [
        # (10006040, 10300000),
        (int(100 * 1000000), 1000000000000)
    ]

    # Инициализация столбца единицами
    df["validity_point"] = 0

    # Присвоение -1, если lat или lon содержит NaN/null
    df.loc[df["lat"].isna() | df["lon"].isna(), "validity_point"] = -1

    if list_invalid_segments:
        # Создаём маску для всех сегментов сразу
        mask = pd.Series(False, index=df.index)
        for start, end in list_invalid_segments:
            mask |= (df["time"] >= start) & (df["time"] <= end)
        df.loc[mask, "validity_point"] = -1

    if list_valid_segments:
        # Создаём маску для всех сегментов сразу
        mask = pd.Series(False, index=df.index)
        for start, end in list_valid_segments:
            mask |= (df["time"] >= start) & (df["time"] <= end)
        df.loc[mask, "validity_point"] = 1

    df.loc[df["lat"].isna() | df["lon"].isna(), "validity_point"] = -1

    path = DefaultLocate.DATA_POSTPROCESSED_DIR / name
    df.to_csv(path, index=False)

    valid_df = df[df["validity_point"] == 0]
    list_time = valid_df["time"].values.tolist()
    list_lon = valid_df["lon"].values.tolist()
    list_lat = valid_df["lat"].values.tolist()

    path = DefaultLocate.OUTPUT_DIR / "2_1validity.geojson"
    IOPs_geojson.write_geojson_from_arrays(
        output_path=path,
        list_arrays=[[list_time, list_lat, list_lon]],
    )


if __name__ == "__main__":
    main()
