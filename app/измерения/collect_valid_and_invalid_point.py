from pathlib import Path

from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV
from app.working.kalman_filter_rw import KalmanFilterRW

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"
    processor = DataProcessor()

    df = processor.load_csv(data_path)
    list_valid_df, list_invalid_df = processor.parse_intervals(df, 1000000, 1000000)

    # valid_counts = []
    # for i, v_df in enumerate(list_invalid_df):
    #     count = v_df[['lon', 'lat']].notna().all(axis=1).sum()
    #     start_lon = v_df['lon'].iloc[0]
    #     end_lon = v_df['lon'].iloc[-1]
    #     start_lat = v_df['lat'].iloc[0]
    #     end_lat = v_df['lat'].iloc[-1]
    #     dist = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
    #         lat_array=np.array([start_lat, end_lat]),
    #         lon_array=np.array([start_lon, end_lon]),
    #     )
    #     if dist < 5000:
    #         continue
    #     valid_counts.append((i, count))
    #
    # top_10 = sorted(valid_counts, key=lambda x: x[1], reverse=True)[10:20]
    # print("Топ 10 индексов DataFrame с наибольшим количеством непустых lon и lat:")
    # for idx, count in top_10:
    #     print(f"Индекс {idx}: {count} записей")

    valid_index = 379, 174, 166, 411, 167, 371, 348, 421
    invalid_index = (
        85,
        2844,
        1474,
        2521,
        3260,
        3231,
        1051,   # 7
        2083,   # 8
        1798,   # 9
        1289,   # 10
        1720,   # 11
        1115,   # 12
        3279,   # 13
    )
    index = 1474
    name = "false 3"

    df = list_invalid_df[index]

    lon, lat, time = processor.get_lon_lat(df)

    path = project_root / "valid_interval.geojson"
    IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])

    x, y = processor.convert_to_local_cartesian(lon, lat)

    kf = KalmanFilterRW(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    path_true = project_root / f"{name}_RW.png"
    _, _, likelihood, mahalanobis_sq = kf.filter(x, y, time)
    likelihood = likelihood.tolist()
    DataProcessor.plot_array_and_hist(likelihood, mahalanobis_sq, name=f"{name}_RW", bins=100, save_path=path_true)
    print("false_RW.png")

    kf = KalmanFilterCV(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    path_true = project_root / f"{name}_CV.png"
    _, _, likelihood, mahalanobis_sq = kf.filter(x, y, time)
    likelihood = likelihood.tolist()
    DataProcessor.plot_array_and_hist(likelihood, mahalanobis_sq, name=f"{name}_CV", bins=100, save_path=path_true)
    print("false_CV.png")
