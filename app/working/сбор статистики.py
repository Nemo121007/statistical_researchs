from math import sqrt
from pathlib import Path
import random  # Добавлен импорт

import numpy as np

from movement_model import load_csv, parce_df, get_lon_lat, convert_to_local_cartesian


random.seed(42)

if __name__ == '__main__':
    path = Path(__file__).parent.parent.parent / 'data' / 'post_processing' / 'example.csv'
    df = load_csv(path)
    df = df[df['satellites'] >= 2]
    list_valid_df, list_invalid_df = parce_df(df)
    list_target_df = list_invalid_df

    # Вычисляем количество интервалов для выборки (1/3 с округлением вниз)
    n_select = len(list_target_df) // 3

    # Выбираем случайные интервалы
    if n_select > 0:
        selected_dfs = random.sample(list_target_df, n_select)
    else:
        selected_dfs = list_target_df

    all_meas_errors = []
    all_accelerations = []

    # Итерируемся по выбранным интервалам
    for valid_df in selected_dfs:
        lon, lat = get_lon_lat(valid_df)
        x, y = convert_to_local_cartesian(lon, lat)
        time = valid_df['time'].to_numpy()

        # Сбор ошибок измерений (для R)
        for i in range(1, len(x) - 1):
            dt = time[i+1] - time[i-1]
            if dt == 0:
                continue

            x_mid = (x[i-1] + x[i+1]) / 2
            y_mid = (y[i-1] + y[i+1]) / 2

            err = sqrt((x[i] - x_mid)**2 + (y[i] - y_mid)**2)
            all_meas_errors.append(err)

        # Сбор ускорений (для Q)
        for i in range(1, len(x) - 1):
            dt1 = time[i] - time[i-1]
            dt2 = time[i+1] - time[i]
            if dt1 == 0 or dt2 == 0:
                continue

            # Скорости
            vx1 = (x[i] - x[i-1]) / dt1
            vy1 = (y[i] - y[i-1]) / dt1
            vx2 = (x[i+1] - x[i]) / dt2
            vy2 = (y[i+1] - y[i]) / dt2

            # Ускорения
            dt_mid = (dt1 + dt2) / 2
            ax = (vx2 - vx1) / dt_mid
            ay = (vy2 - vy1) / dt_mid

            a_total = sqrt(ax**2 + ay**2)
            all_accelerations.append(a_total)

    # Итоговые значения
    if all_meas_errors:
        sigma_meas = np.nanstd(all_meas_errors)
    else:
        sigma_meas = 0.0 # Значение по умолчанию, если данных нет

    if all_accelerations:
        # TODO: Возможно, стоит взять перцентиль?
        sigma_acc_raw = np.nanstd(all_accelerations)
    else:
        sigma_acc_raw = 0.0

    print(f"Sigma Meas (R): {sigma_meas:.4f}")
    print(f"Sigma Acc (Q): {sigma_acc_raw:.4f}")
