from pathlib import Path
from typing import List, Tuple

import geojson
import pandas as pd
from geojson import FeatureCollection

def parce_file(path: Path, output_path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    df = df.dropna()
    df = df.reset_index(drop=True)
    # Удаляем строки, где lon и lat совпадают с предыдущей строкой
    mask = (df['lon'] == df['lon'].shift(1)) & (df['lat'] == df['lat'].shift(1))
    df = df[~mask]
    df = df.reset_index(drop=True)

    features_collection = FeatureCollection([])
    list_nodes = []
    for index, row in df.iterrows():
        node = geojson.Feature(
            id=index,
            geometry=geojson.Point((row['lon'], row['lat'])),
        )
        features_collection.features.append(node)
        list_nodes.append((row['lon'], row['lat']))
    features_collection.features.append(
        geojson.Feature(
            id=-1,
            geometry=geojson.LineString(coordinates=list_nodes),
        )
    )

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    with open(output_path, "w") as file:
        geojson.dump(features_collection, file)

def extract_parce_data(path: Path, output_path: Path, indexes: List[Tuple[float, float]]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    df = df.dropna()
    df = df.reset_index(drop=True)

    past_lot, last_lat = -1.0, -1.0
    for ind, (start, end) in enumerate(indexes):
        df_subset = df.iloc[start:end + 1]
        features_collection = FeatureCollection([])
        list_nodes = []
        for index, row in df_subset.iterrows():
            if past_lot == row['lon'] and last_lat == row['lat']:
                continue
            node = geojson.Feature(
                id=index,
                geometry=geojson.Point((row['lon'], row['lat'])),
            )
            features_collection.features.append(node)
            list_nodes.append([row['lon'], row['lat']])
            past_lot = row['lon']
            last_lat = row['lat']
        features_collection.features.append(
            geojson.Feature(
                id=-1,
                geometry=geojson.LineString(coordinates=list_nodes),
            )
        )

        subset_output_path = output_path.parent / f"{output_path.stem}_{ind}.geojson"
        with open(subset_output_path, "w") as file:
            geojson.dump(features_collection, file)
        subset_output_path = output_path.parent / f"{output_path.stem}_{ind}.csv"
        df_subset.to_csv(subset_output_path, index=False)





if __name__ == "__main__":
    # path = Path(__file__).parent / "data"
    # list_paths = list(path.glob("*.csv"))
    # for file_path in list_paths:
    #     output_path = Path(__file__).parent / "output" / f"{file_path.stem}.geojson"
    #     parce_file(file_path, output_path)
    path = Path(__file__).parent / "data" / "example_8.csv"
    output_path = Path(__file__).parent / "output" / "example_8.geojson"
    # indexes_0 = [(1, 1282), (1680, 7248), (8074, 29333), (29341, 30505), (32852, 35419)]
    # indexes_2 = [(2420, 5122), (5544, 20000), (20000, 27502), (29443, 33427),
    #             (33811, 39660), (39872, 50000), (50000, 57518)]
    # indexes_4 = [(33200, 33407), (33503, 34438), (11568, 27069), (946, 8777)]
    # indexes_6 = [(18134, 18401), (18531, 18685), (33703, 33790 ), (33519, 33790), (33790, 33943),
    #            (52113, 60000), (60000, 65548), (69414, 82486), (82487, 100000)]
    indexes = [(95385, 120000), (73471, 89227), (60000,  69287), (50000, 60000),
               (40000, 50000), (30000, 40000), (20929, 30000)]
    extract_parce_data(path, output_path, indexes)