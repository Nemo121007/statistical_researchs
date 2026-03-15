from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

LEN_LAT = 111132.0  # Длина одного градуса широты в метрах


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def parce_df(df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    list_valid_df = _extend_intervals(df, target_point=1)
    list_invalid_df = _extend_intervals(df, target_point=-1)
    return list_valid_df, list_invalid_df


def _extend_intervals(df: pd.DataFrame, target_point: int = 1, n: int = 300) -> List[pd.DataFrame]:
    """
    Разбивает DataFrame на список непрерывных временных интервалов.
    Если длина интервала превышает n, он разбивается на части длиной не более n.

    Args:
        df: DataFrame с колонкой 'time' и 'validate_point'.
        target_point: Значение для фильтрации колонки 'validate_point'.
        n: Максимальная длина датафрейма в выходном списке.
    Returns:
        Список DataFrame, каждый из которых представляет непрерывный временной интервал.
    """
    if df.empty:
        return []

    clean_df = df[df["validate_point"] == target_point].copy()
    if clean_df.empty:
        return []

    time_diff = clean_df["time"].diff()
    split_mask = time_diff > 10
    group_ids = split_mask.cumsum()

    result_list = []

    # Группируем и обрабатываем каждую группу
    for _, group in clean_df.groupby(group_ids):
        if len(group) <= 1:
            continue

        # Если группа длиннее n, разбиваем её на части
        if len(group) > n:
            for i in range(0, len(group), n):
                chunk = group.iloc[i:i + n]
                result_list.append(chunk)
        else:
            result_list.append(group)

    return result_list


def get_lon_lat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлекает массивы долгот и широт из DataFrame.

    Args:
        df: DataFrame с колонками 'lon' и 'lat'.
    Returns:
        Кортеж (lon_array, lat_array) — массивы долгот и широт
    """
    return df['lon'].to_numpy(), df['lat'].to_numpy()


def convert_to_local_cartesian(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Переводит сферические координаты (lon, lat) в локальные прямоугольные (x, y) в метрах.

    Начало координат (0, 0) соответствует первой точке трека.

    Args:
        lon: Массив долгот (в градусах).
        lat: Массив широт (в градусах).

    Returns:
        Кортеж (x_local, y_local) — координаты в метрах.
    """
    if len(lat) == 0:
        return np.array([]), np.array([])

    # Запоминаем начальную точку трека
    lat0 = lat[0]
    lon0 = lon[0]

    # Задаем коэффициенты масштабирования
    # Длина градуса широты (примерно постоянна)
    ky = LEN_LAT

    # Длина градуса долготы (зависит от широты)
    # Переводим широту в радианы для корректного вычисления косинуса
    lat0_rad = np.radians(lat0)
    kx = LEN_LAT * np.cos(lat0_rad)

    # Перевод входных данных в метры
    x_local = (lon - lon0) * kx
    y_local = (lat - lat0) * ky

    return x_local, y_local


def _get_transition_matrix(dt: float) -> np.ndarray:
    """
    Формирует матрицу перехода F для шага dt.
    F = [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    """
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _get_process_noise_matrix(dt: float, sigma_acc: float) -> np.ndarray:
    """
    Формирует ковариационную матрицу шума процесса Q.
    Модель дискретного белого шума ускорения.
    """
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4

    # Блок для координаты x (и аналогично для y)
    # [[dt^4/4, dt^3/2],
    #  [dt^3/2, dt^2]]

    Q = np.zeros((4, 4))

    # Индексы: 0 - x, 1 - y, 2 - vx, 3 - vy
    Q[0, 0] = dt4 / 4
    Q[1, 1] = dt4 / 4
    Q[2, 2] = dt2
    Q[3, 3] = dt2

    Q[0, 2] = dt3 / 2
    Q[2, 0] = dt3 / 2

    Q[1, 3] = dt3 / 2
    Q[3, 1] = dt3 / 2

    return Q * (sigma_acc ** 2)


def kalman_filter_cv(x: np.ndarray, y: np.ndarray, time: np.ndarray,
                     sigma_acc: float = 0.04, sigma_meas: float = 2.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применяет фильтр Калмана к траектории судна в прямоугольных координатах.

    Args:
        x: Массив координат X (в метрах).
        y: Массив координат Y (в метрах).
        time: Массив времени (в секундах).
        sigma_acc: СКО шума ускорения (параметр процесса Q). Определяет "маневренность".
        sigma_meas: СКО шума измерений (параметр R). Определяет доверие к измерениям.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Сглаженные координаты X и Y.
    """
    if len(x) < 2:
        return x, y

    # Матрица измерений H (измеряем только позицию)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    # Ковариация шума измерений R
    R = np.eye(2) * (sigma_meas ** 2)

    # Инициализация состояния X_state = [x, y, vx, vy]
    # Используем X_state, чтобы не путать с входным массивом x
    X_state = np.array([x[0], y[0], 0.0, 0.0]).reshape(4, 1)

    # Инициализация ковариации ошибки P
    P = np.eye(4) * 500.0  # Начальная неопределенность
    P[2, 2] = 100.0        # Неопределенность скорости
    P[3, 3] = 100.0

    # Массивы для результатов
    filtered_x = np.zeros(len(x))
    filtered_y = np.zeros(len(y))
    filtered_x[0] = x[0]
    filtered_y[0] = y[0]

    I = np.eye(4)

    # Основной цикл
    for k in range(1, len(time)):
        dt = time[k] - time[k-1]
        if dt <= 0:
            dt = 1e-5 # Защита от нулевого или отрицательного времени

        # --- Prediction (Этап предсказания) ---
        F = _get_transition_matrix(dt)
        Q = _get_process_noise_matrix(dt, sigma_acc)

        X_pred = F @ X_state
        P_pred = F @ P @ F.T + Q

        # --- Update (Этап коррекции) ---
        # Вектор измерений (берем исходные x и y)
        z = np.array([x[k], y[k]]).reshape(2, 1)

        # Невязка (Innovation)
        y_err = z - (H @ X_pred)

        # Ковариация невязки
        S = H @ P_pred @ H.T + R

        # Коэффициент усиления Калмана
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Обновление состояния
        X_state = X_pred + K @ y_err

        # Обновление ковариации
        P = (I - K @ H) @ P_pred

        filtered_x[k] = X_state[0, 0]
        filtered_y[k] = X_state[1, 0]

    return filtered_x, filtered_y


if __name__ == '__main__':
    path = Path(__file__).parent.parent.parent / 'data' / 'post_processing' / 'example.csv'
    df = load_csv(path)
    list_valid_df, list_invalid_df = parce_df(df)


