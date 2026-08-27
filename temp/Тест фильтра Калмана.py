from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from app.help_scripts.IOPs_geojson import IOPs_geojson
from app.working.data_processor import DataProcessor
from app.working.kalman_filter_cv import KalmanFilterCV


def analyze_filter_consistency(
        mahalanobis_sq: np.ndarray,
        alpha: float = 0.05,
        save_path: Path = None
):
    """
    Анализирует и визуализирует согласованность фильтра.
    Сохраняет графики в .png и текстовый отчет в .txt.

    Args:
        mahalanobis_sq: Массив квадратов расстояний Махаланобиса.
        alpha: Уровень значимости.
        save_path: Путь pathlib.Path.
                   Если указан, сохраняет .png и .txt по этому пути.
                   Если None, выводит график на экран и текст в консоль.
    """
    # Убираем NaN значения
    valid_data = mahalanobis_sq[~np.isnan(mahalanobis_sq)]
    n = len(valid_data)

    if n == 0:
        print("Нет данных для анализа.")
        return

    # Вычисление P-value
    p_values = 1 - chi2.cdf(valid_data, df=2)

    # Теоретический порог
    threshold_chi2 = chi2.ppf(1 - alpha, df=2)

    # Статистика
    practical_mean = np.mean(valid_data)
    theoretical_mean = 2.0

    outliers_count = np.sum(valid_data > threshold_chi2)
    practical_outlier_rate = outliers_count / n

    # --- Формирование текстового отчета ---
    report_lines = []
    report_lines.append(f"--- Анализ согласованности (Alpha = {alpha}) ---")
    report_lines.append(f"Всего обработано точек: {n}")
    report_lines.append(f"Теоретическое среднее d^2: {theoretical_mean:.2f}")
    report_lines.append(f"Практическое среднее d^2:  {practical_mean:.2f}")

    if practical_mean > theoretical_mean * 1.5:
        report_lines.append(">>> ВНИМАНИЕ: Практическое среднее значительно выше теоретического.")
        report_lines.append("    Возможные причины: занижены шумы (sigma_acc/sigma_meas) или аномальное измерение.")
    elif practical_mean < theoretical_mean * 0.5:
        report_lines.append(">>> ВНИМАНИЕ: Практическое среднее значительно ниже теоретического.")
        report_lines.append("    Возможные причины: завышены шумы (фильтр не реагирует на данные).")
    else:
        report_lines.append(">>> Статистика соответствует теории.")

    report_lines.append(f"\nТеоретический порог (Chi2): {threshold_chi2:.2f}")
    report_lines.append(f"Обнаружено выбросов: {outliers_count} ({practical_outlier_rate*100:.2f}%)")
    report_lines.append(f"Ожидаемая доля выбросов (Alpha): {alpha:.2f}%")

    report_text = "\n".join(report_lines)

    # Визуализация
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Гистограмма
    viz_limit = np.percentile(valid_data, 99.5)
    if viz_limit < 10: viz_limit = 10

    axs[0].hist(
        valid_data, bins=50, range=(0, viz_limit), density=True, alpha=0.6, color='g', label='Практические измерения (Гистограмма)'
    )

    x_range = np.linspace(0, viz_limit, 100)
    theoretical_pdf = chi2.pdf(x_range, df=2)
    axs[0].plot(x_range, theoretical_pdf, 'r-', lw=2, label=r'Теоретическая функция $\chi^2(2)$')

    axs[0].axvline(
        threshold_chi2, color='b', linestyle='--', label=r'Порог $\alpha={}$ ({:.2f})'.format(alpha, threshold_chi2)
    )

    axs[0].set_title(f'Распределение расстояния Махаланобиса (Среднее: {practical_mean:.2f})')
    axs[0].set_xlabel('$d_k^2$')
    axs[0].set_ylabel('Плотность вероятности')
    axs[0].legend()
    axs[0].grid(True)

    # P-value во времени
    axs[1].semilogy(p_values, 'g.', markersize=2, label='P-value измерений')
    axs[1].axhline(alpha, color='r', linestyle='--', label=r'Уровень значимости $\alpha={}$'.format(alpha))

    axs[1].fill_between(range(len(p_values)), 0, alpha, color='red', alpha=0.1, label='Аномальная зона')

    axs[1].set_title('P-value измерений во времени')
    axs[1].set_xlabel('Индекс измерения')
    axs[1].set_ylabel('P-value (log scale)')
    axs[1].set_ylim(1e-4, 1.1)
    axs[1].legend()
    axs[1].grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()

    if save_path:
        # Приводим путь к объекту Path
        if not isinstance(save_path, Path):
            save_path = Path(save_path)

        # Формируем пути для файлов
        png_path = save_path.with_suffix('.png')
        txt_path = save_path.with_suffix('.txt')

        # Создаем директории, если их нет
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем текст
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # Сохраняем график
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

        print(f"Результаты сохранены:\n  График: {png_path}\n  Отчет:  {txt_path}")
    else:
        # Вывод в консоль и на экран, если путь не задан
        print(report_text)
        plt.show()


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

    # Смешанный участок 1
    # start_i = 141100
    # stop_i = 151893

    # Смешанный участок 2
    start_i = 132860
    stop_i = 134000

    # Эталонный участок 1
    # start_i = 142507
    # stop_i = 143023

    # Эталонный участок 2
    # start_i = 134000
    # stop_i = 140659

    # Аномальный участок 1
    # start_i = 132910
    # stop_i = 132981

    # Аномальный участок 2
    # start_i = 141609
    # stop_i = 142498

    test_df = df[start_i:stop_i]
    lon, lat, dt = processor.get_lon_lat(test_df)
    time_moment = np.arange(start_i, stop_i, dtype=np.int64)

    filter_lon = np.full(len(time_moment), np.nan)
    filter_lat = np.full(len(time_moment), np.nan)
    filter_time_moment = np.arange(start_i, stop_i, dtype=np.int64)

    all_mahalanobis_sq = []

    step = 30
    kf = KalmanFilterCV(sigma_acc=0.04, sigma_meas=1 * 2.4)
    for i in range(0, len(time_moment), step):
        local_lon = lon[i:i+step]
        local_lat = lat[i:i+step]
        local_dt = dt[i:i+step]
        x, y = processor.convert_to_local_cartesian(local_lon, local_lat)

        x_filt, y_filt, likelihood, mahalanobis_sq = kf.filter(x, y, local_dt)

        valid_vals = mahalanobis_sq[~np.isnan(mahalanobis_sq)]
        if len(valid_vals) > 0:
            all_mahalanobis_sq.extend(valid_vals)

        # likelihood.sort()
        # print(likelihood[:20])
        # path_true = project_root / f"{start_i}-{stop_i}_KF_CV.png"
        # DataProcessor.plot_array_and_hist(
        #     likelihood, mahalanobis_sq, name=f"{start_i}-{stop_i}_CV", bins=100, save_path=path_true
        # )
        # path_true = project_root / f"{start_i}-{stop_i}_KF_CV_graph.png"
        # DataProcessor.visualize_and_save(x, y, x_filt, y_filt, path_true)

        if len(likelihood) and np.nanmin(likelihood) < -100:
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

    pict_path = project_root / 'mixed_2.png'
    if len(all_mahalanobis_sq) > 0:
        analyze_filter_consistency(np.array(all_mahalanobis_sq), alpha=0.05, save_path=pict_path)
    else:
        print("Не удалось собрать данные для анализа.")
