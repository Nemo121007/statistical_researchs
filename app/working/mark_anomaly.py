from pathlib import Path

import numpy as np

from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

invalid_interval = [
    []
]

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"

    output_path_geojson = project_root / 'test.geojson'
    output_path_pict = project_root

    processor = DataProcessor()
    df = processor.load_csv(data_path)
    df = df.dropna(subset=['lat', 'lon'])

    start_i = 141100
    stop_i = 151893

    test_df = df[start_i:stop_i]
    lon, lat, dt = processor.get_lon_lat(test_df)
    time_moment = np.arange(start_i, stop_i, dtype=np.int64)

    IOPs_geojson.write_geojson_from_arrays(
        output_path_geojson,
        [
            (time_moment, lat, lon),
        ]
    )

    df['validate_point'] = -1
    for start_local, stop_local in invalid_interval:
        df['validate_point'][start_local:stop_local] = 1
    output_path_pict = output_path_geojson.parent / 'correct.csv'
    df.to_csv(output_path_geojson, index=False)
