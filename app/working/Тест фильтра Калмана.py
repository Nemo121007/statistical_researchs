from pathlib import Path

import numpy as np

from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV


class ExperimentsFilter:
    pass


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"

    output_path_geojson = project_root / 'test.geojson'
    output_path_pict = project_root

    processor = DataProcessor()
    df = processor.load_csv(data_path)
    df = df.dropna(subset=['lat', 'lon'])

    # start_i = 132860
    # stop_i = 134000
    start_i = 141607
    stop_i = 151893
    test_df = df[start_i:stop_i]
    lon, lat, dt = processor.get_lon_lat(test_df)
    time_moment = np.arange(start_i, stop_i, dtype=np.int64)

    filter_lon = np.full(len(time_moment), np.nan)
    filter_lat = np.full(len(time_moment), np.nan)
    filter_time_moment = np.arange(start_i, stop_i, dtype=np.int64)
    step = 30
    kf = KalmanFilterCV(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)
    for i in range(0, len(time_moment), step):
        local_lon = lon[i:i+step]
        local_lat = lat[i:i+step]
        local_dt = dt[i:i+step]
        x, y = processor.convert_to_local_cartesian(local_lon, local_lat)

        x_filt, y_filt, likelihood, mahalanobis_sq = kf.filter(x, y, local_dt)
        # likelihood.sort()
        # print(likelihood[:20])
        # path_true = project_root / f"{start_i}-{stop_i}_KF_CV.png"
        # DataProcessor.plot_array_and_hist(
        #     likelihood, mahalanobis_sq, name=f"{start_i}-{stop_i}_CV", bins=100, save_path=path_true
        # )
        # path_true = project_root / f"{start_i}-{stop_i}_KF_CV_graph.png"
        # DataProcessor.visualize_and_save(x, y, x_filt, y_filt, path_true)

        if np.nanmin(likelihood) < -100:
            continue
        else:
            filter_lon[i:i+step] = lon[i:i+step]
            filter_lat[i:i+step] = lat[i:i+step]

    IOPs_geojson.write_geojson_from_arrays(
        output_path_geojson,
        [
            (time_moment, lat, lon),
            (filter_time_moment, filter_lat, filter_lon),
        ]
    )
