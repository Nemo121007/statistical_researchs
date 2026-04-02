from pathlib import Path

from app.working.kalman_filter_rw import KalmanFilterRW
from app.working.kalman_filter_cv import KalmanFilterCV
from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'post_processing' / 'example.csv'
    processor = DataProcessor()

    df = processor.load_csv(data_path)
    list_valid_df, list_invalid_df = processor.parse_intervals(df, 1000000, 1000000)

    index = 2, 5, 13, 16
    index = 16

    df = list_valid_df[index]

    lon, lat = processor.get_lon_lat(df)
    time = df['time']

    path = project_root / 'valid_interval.geojson'
    IOPs_geojson.write_geojson_from_arrays(path, [[time, lat, lon]])

    x, y = processor.convert_to_local_cartesian(lon, lat)
    time = time.to_numpy()

    kf = KalmanFilterRW(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    path_true = project_root / 'true_RW.png'
    _, _, likelihood = kf.filter(x, y, time)
    likelihood = likelihood.tolist()
    DataProcessor.plot_array_and_hist(likelihood, bins=100, save_path=path_true)
    print('true_RW.png')

    # path_false = project_root / 'false_RW.png'
    # _, _, likelihood = kf.filter(x, y, time)
    # likelihood = likelihood.tolist()
    # DataProcessor.plot_array_and_hist(likelihood, bins=100, save_path=path_false)

    kf = KalmanFilterCV(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    path_true = project_root / 'true_CV.png'
    _, _, likelihood = kf.filter(x, y, time)
    likelihood = likelihood.tolist()
    DataProcessor.plot_array_and_hist(likelihood, bins=100, save_path=path_true)
    print('true_CV.png')

    # path_false = project_root / 'false_CV.png'
    # _, _, likelihood = kf.filter(x, y, time)
    # likelihood = likelihood.tolist()
    # DataProcessor.plot_array_and_hist(likelihood, bins=100, save_path=path_true)
    # print('false_CV.png')
