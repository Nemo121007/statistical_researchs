from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV
from app.working.kalman_filter_rw import KalmanFilterRW


def analyze_filter_consistency(
        mahalanobis_sq: np.ndarray,
        list_likelihood: np.ndarray,
        mark: np.ndarray,
        save_path: Path = None
):
    """
    Визуализирует

    Args:
        mahalanobis_sq: Массив квадратов расстояний Махаланобиса.
        save_path: Путь pathlib.Path.
                   Если указан, сохраняет .png и .txt по этому пути.
                   Если None, выводит график на экран и текст в консоль.
    """
    all_count = len(mahalanobis_sq)
    valid_count_mahalanobis = np.nansum(np.where(np.isnan(mahalanobis_sq), 0, 1))
    valid_count_likelihood = np.nansum(np.where(np.isnan(list_likelihood), 0, 1))


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Модель Filter Kalman CV\nИсходное количество объектов: {all_count}\n"
                 f"Не nan объектов mahalanobis_sq: {valid_count_mahalanobis}\n"
                 f"Не nan объектов likelihood: {valid_count_likelihood}")

    x = np.arange(all_count)
    
    # Маски для True и False в mark
    mark_true = mark == True
    mark_false = mark == False

    # Маски для валидных данных (без nan и inf)
    valid_msq = ~np.isnan(mahalanobis_sq) & ~np.isinf(mahalanobis_sq)
    valid_lh = ~np.isnan(list_likelihood) & ~np.isinf(list_likelihood)

    # Маски для каждого графика
    idx_false_msq = mark_false & valid_msq
    idx_true_msq = mark_true & valid_msq
    
    idx_false_lh = mark_false & valid_lh
    idx_true_lh = mark_true & valid_lh

    # График 1: mahalanobis_sq
    ax1.plot(x[idx_false_msq], mahalanobis_sq[idx_false_msq], 'bs', label='False')
    ax1.plot(x[idx_true_msq], mahalanobis_sq[idx_true_msq], 'g^', label='True')
    ax1.set_ylabel('Mahalanobis Sq')
    ax1.grid(True)
    ax1.legend()

    # График 2: likelihood
    ax2.plot(x[idx_false_lh], list_likelihood[idx_false_lh], 'bs', label='False')
    ax2.plot(x[idx_true_lh], list_likelihood[idx_true_lh], 'g^', label='True')
    ax2.set_xlabel('Номер измерения')
    ax2.set_ylabel('Likelihood')
    ax2.grid(True)
    ax2.legend()
    
    # Второй график(Отдельный график): плотность распределения False
    fig_hist_false, ax_hist_false = plt.subplots(figsize=(8, 6))
    data_hist_false = mahalanobis_sq[idx_false_msq]
    if len(data_hist_false) > 0:
        # Отфильтруем жесткие выбросы (оставим 95-й процентиль) чтобы график не сжимался
        p95_false = np.percentile(data_hist_false, 95)
        data_hist_false = data_hist_false[data_hist_false <= p95_false]

        mean_false = np.mean(data_hist_false)
        var_false = np.var(data_hist_false)
        std_false = np.std(data_hist_false)
        if std_false > 0:
            data_hist_false = (data_hist_false - mean_false) / std_false
        ax_hist_false.hist(data_hist_false, bins=50, density=True, alpha=0.6, color='b')
        ax_hist_false.set_title(f"Плотность распределения нормализованного Mahalanobis (False, без выбросов)\nИсходное Мат.ожидание: {mean_false:.4f}, Дисперсия: {var_false:.4f}")
    else:
        ax_hist_false.set_title("Плотность распределения Mahalanobis (False) - Нет данных")
    ax_hist_false.grid(True)
    
    # Третий график(Отдельный график): плотность распределения True
    fig_hist_true, ax_hist_true = plt.subplots(figsize=(8, 6))
    data_hist_true = mahalanobis_sq[idx_true_msq]
    if len(data_hist_true) > 0:
        # Отфильтруем жесткие выбросы (оставим 95-й процентиль) чтобы график не сжимался
        p95_true = np.percentile(data_hist_true, 95)
        data_hist_true = data_hist_true[data_hist_true <= p95_true]

        mean_true = np.mean(data_hist_true)
        var_true = np.var(data_hist_true)
        std_true = np.std(data_hist_true)
        if std_true > 0:
            data_hist_true = (data_hist_true - mean_true) / std_true
        ax_hist_true.hist(data_hist_true, bins=50, density=True, alpha=0.6, color='g')
        ax_hist_true.set_title(f"Плотность распределения нормализованного Mahalanobis (True, без выбросов)\nИсходное Мат.ожидание: {mean_true:.4f}, Дисперсия: {var_true:.4f}")
    else:
        ax_hist_true.set_title("Плотность распределения Mahalanobis (True) - Нет данных")
    ax_hist_true.grid(True)

    if save_path:
        # Приводим путь к объекту Path
        if not isinstance(save_path, Path):
            save_path = Path(save_path)

        png_path = save_path.with_suffix('.png')

        # Создаем директории, если их нет
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем главный график
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        
        # Сохраняем дополнительные графики с суффиксами
        fig_hist_false.savefig(save_path.parent / f"{save_path.stem}_hist_false.png", dpi=150)
        plt.close(fig_hist_false)
        fig_hist_true.savefig(save_path.parent / f"{save_path.stem}_hist_true.png", dpi=150)
        plt.close(fig_hist_true)

        print(f"Результаты сохранены:\n  График: {png_path}")
    else:
        plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "correct.csv"

    output_path_pict = project_root

    processor = DataProcessor()
    df = processor.load_csv(data_path)
    list_valid_intervals, list_invalid_intervals = processor.parse_intervals(df, 700_000, 100_000)
    all_mahalanobis_sq = []
    all_likelihood = []
    all_mark = []

    kf = KalmanFilterCV(sigma_acc=0.04, sigma_meas=1 * 2.4)
    for interval in list_valid_intervals:
        step = 5
        if len(interval) < step:
            continue
        interval = interval.dropna(subset=['lat', 'lon'])

        lon, lat, dt = processor.get_lon_lat(interval)
        mark = interval['validate_point'].values
        mark = np.where(mark == 1, True, False)

        for i in range(step, len(dt) + 1):
            local_lon = lon[i - step:i]
            local_lat = lat[i - step:i]
            local_dt = dt[i - step:i]
            x, y = processor.convert_to_local_cartesian(local_lon, local_lat)

            x_filt, y_filt, likelihood, mahalanobis_sq = kf.filter(x, y, local_dt)
            all_mahalanobis_sq.append(mahalanobis_sq[-1])
            all_likelihood.append(likelihood[-1])
            if len(all_mahalanobis_sq) % 1000 == 0:
                print(len(all_mahalanobis_sq))
        
        # Сохраняем метки для данного интервала со сдвигом на размер окна
        all_mark.extend(mark[step - 1:len(dt)])
                
    for interval in list_invalid_intervals:
        step = 5
        if len(interval) < step:
            continue
        interval = interval.dropna(subset=['lat', 'lon'])

        lon, lat, dt = processor.get_lon_lat(interval)
        mark = interval['validate_point'].values
        mark = np.where(mark == 1, True, False)

        for i in range(step, len(dt) + 1):
            local_lon = lon[i - step:i]
            local_lat = lat[i - step:i]
            local_dt = dt[i - step:i]
            x, y = processor.convert_to_local_cartesian(local_lon, local_lat)

            x_filt, y_filt, likelihood, mahalanobis_sq = kf.filter(x, y, local_dt)
            all_mahalanobis_sq.append(mahalanobis_sq[-1])
            all_likelihood.append(likelihood[-1])
            if len(all_mahalanobis_sq) % 1000 == 0:
                print(len(all_mahalanobis_sq))

        # Сохраняем метки для данного интервала со сдвигом на размер окна
        all_mark.extend(mark[step - 1:len(dt)])

    pict_path = project_root / 'Kalman_CV_non_accumulate.png'
    if len(all_mahalanobis_sq) > 0:
        analyze_filter_consistency(
            mahalanobis_sq=np.array(all_mahalanobis_sq), 
            list_likelihood=np.array(all_likelihood),
            mark=np.array(all_mark), 
            save_path=pict_path
        )
    else:
        print("Не удалось собрать данные для анализа.")
