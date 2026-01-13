"""Модуль для извлечения и обработки путей из CSV файлов."""

from pathlib import Path

import pandas as pd

from application.modules.bg_services.gps.utils.IOPs.IOPs_geojson import IOPs_geojson
from application.modules.bg_services.gps.corrector.tracker.models.way_model import Way
from application.modules.bg_services.gps.corrector.tracker.models.node_model import Node
from application.modules.bg_services.gps.corrector.tracker.collectors.way_collector import (
    WayCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.node_collector import (
    NodeCollector,
)

# pylint: disable=line-too-long
from application.modules.bg_services.gps.corrector.tracker.shared_files.calculator_distances_length_large_circle import (
    CalculatorDistancesLengthLargeCircle,
)


class ExtractingPaths:
    """Класс для извлечения и обработки путей из CSV файлов."""

    @staticmethod
    def extract_data():
        """Извлекает и обрабатывает данные из CSV файлов, создавая GeoJSON и CSV файлы."""
        path = Path(__file__).parent / "data"
        files = [file.name for file in path.iterdir() if file.is_file()]
        for file in files:
            name = str(file).split(".")[0]
            df = pd.read_csv(path / file)
            print("Processing file:", file)
            df["time"] = pd.to_datetime(df["time"])
            min_value = df["time"].min()
            df["time"] = (df["time"] - min_value).dt.total_seconds()
            df = df.sort_values(by="time", ascending=True)

            segment = df[["lat", "lon"]].to_numpy()
            distance = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(segment=segment)
            list_node = []
            temp_rows = []
            node_collector = NodeCollector()
            c = 0
            for i, dist in enumerate(distance):
                if dist < 50:
                    node = Node(
                        node_id=len(list_node),
                        lat=df.iloc[i]["lat"],
                        lon=df.iloc[i]["lon"],
                    )
                    list_node.append(node)
                    node_collector.add_node(node)
                    temp_rows.append(df.iloc[i])
                elif len(list_node) > 100:
                    c = c + 1
                    way = Way(way_id=len(list_node), nodes=list_node)
                    way_collector = WayCollector()
                    way_collector.add_way(way)
                    writer = IOPs_geojson()
                    writer.write_geojson(
                        file_output_path=path / "geojson" / f"{name}_{c}.geojson",
                        ways_collector=way_collector,
                        nodes_collector=node_collector,
                        list_print_points=list_node,
                    )

                    # Создаём DataFrame из списка строк
                    temp_df = pd.DataFrame(temp_rows)
                    path_csv = path / "csv" / f"{name}_{c}.csv"
                    temp_df.to_csv(path_csv, index=False)
                    print(f"GeoJSON file created: {path / 'geojson' / f'{name}_{c}.geojson'}")

                    list_node = []
                    temp_rows = []  # Сбрасываем список строк
                    node_collector = NodeCollector()
                else:
                    list_node = []
                    temp_rows = []
                    node_collector = NodeCollector()


if __name__ == "__main__":
    ExtractingPaths.extract_data()
