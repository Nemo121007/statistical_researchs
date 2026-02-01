"""Модуль для извлечения и обработки путей из CSV файлов."""
import logging
import pandas as pd

from IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate


class ExtractingPaths:
    """Класс для извлечения и обработки путей из CSV файлов."""

    @staticmethod
    def extract_data():
        """Извлекает и обрабатывает данные из CSV файлов, создавая GeoJSON и CSV файлы."""
        path_file = DefaultLocate.DATA_RAW_DIR
        files = [file.name for file in path_file.glob('*.csv')]
        for file in files:
            name = str(file).split(".")[0]
            df = pd.read_csv(path_file / file)
            logging.info(str(path_file/ file))
            logging.info("Processing file: %s", file)
            df["time"] = pd.to_datetime(df["time"])
            min_value = df["time"].min()
            df["time"] = (df["time"] - min_value).dt.total_seconds()
            df = df.sort_values(by="time", ascending=True)

            chunk_size = 1000000  # Размер части

            def split_dataframe(df, chunk_size):
                """Генератор для разделения DataFrame на части."""
                for start in range(0, len(df), chunk_size):
                    yield df.iloc[start:start + chunk_size]

            for i, chunk in enumerate(split_dataframe(df, chunk_size)):
                logging.info(f"Обработка части {i + 1}, размер: {len(chunk)}")
                path = DefaultLocate.DATA_PREPROCESSED_DIR / f"{name}_part_{i + 1}"

                min_time = chunk["time"].min()
                chunk = chunk.copy()
                chunk["time"] = chunk["time"] - min_time
                chunk["time"] = chunk["time"].astype("Int64")
                chunk["satellites"] = chunk["satellites"].astype("Int64")

                chunk.to_csv(path.with_suffix(".csv"), index=False)
                logging.info(f"Сохранен CSV файл: {path.with_suffix('.csv')}")

                time_array = chunk["time"]
                lat_array = chunk["lat"]
                lon_array = chunk["lon"]

                path = DefaultLocate.DATA_DIR / "output" / f"{name}_part_{i + 1}.geojson"
                IOPs_geojson.write_geojson_from_arrays(
                    output_path=path,
                    list_arrays=[[time_array, lat_array, lon_array]],
                )

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ExtractingPaths.extract_data()
