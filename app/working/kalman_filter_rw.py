from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from app.working.data_processor import DataProcessor


class KalmanFilterRW:
    """Класс реализации фильтра Калмана для модели случайного блуждания (без скорости)."""

    def __init__(self, sigma_proc: float = 0.04, sigma_meas: float = 2.4):
        """
        Инициализация параметров фильтра.

        Args:
            sigma_proc: Интенсивность шума процесса (q).
                        Определяет, насколько сильно точка может сместиться за 1 секунду (в метрах).
                        (Т.е. sqrt(Q) ~ sigma_proc * sqrt(dt)).
            sigma_meas: СКО шума измерений (в метрах).
        """
        self.sigma_proc = sigma_proc
        self.sigma_meas = sigma_meas

    def filter(self, x: np.ndarray, y: np.ndarray, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет фильтр Калмана к траектории.

        Args:
            x: Массив координат X (в метрах).
            y: Массив координат Y (в метрах).
            time: Массив времени (в секундах).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Сглаженные координаты X и Y.
        """
        if len(x) < 2:
            return x, y

        # Размерность вектора состояния (x, y)
        n_dim = 2

        # Матрица перехода F = I (для Random Walk следующее положение равно текущему + шум)
        F = np.eye(n_dim)

        # Матрица измерений H = I (измеряем положение напрямую)
        H = np.eye(n_dim)

        # Ковариация шума измерений R
        R = np.eye(n_dim) * (self.sigma_meas ** 2)

        # Инициализация состояния X_state = [x, y]
        X_state = np.array([x[0], y[0]]).reshape(n_dim, 1)

        # Инициализация ковариации ошибки P
        P = np.eye(n_dim) * (self.sigma_meas ** 2)

        # Массивы для результатов
        filtered_x = np.zeros(len(x))
        filtered_y = np.zeros(len(y))
        filtered_x[0] = x[0]
        filtered_y[0] = y[0]

        I = np.eye(n_dim)

        for k in range(1, len(time)):
            dt = time[k] - time[k - 1]
            if dt <= 0:
                dt = 1e-5

            # --- Формирование матрицы шума процесса Q ---
            # Используем зависимость от времени: Q = q * dt
            # Это означает, что неопределенность положения растет со временем
            Q = np.eye(n_dim) * (self.sigma_proc ** 2 * dt)

            # --- Prediction (Этап предсказания) ---
            # X_pred = F @ X_state (так как F=I, просто X_state)
            X_pred = F @ X_state

            # P_pred = P + Q
            P_pred = P + Q

            # --- Update (Этап коррекции) ---
            # Вектор измерений z = [x, y]
            z = np.array([x[k], y[k]]).reshape(n_dim, 1)

            # Невязка (Innovation): y_err = z - X_pred
            y_err = z - (H @ X_pred)

            # Ковариация невязки: S = P_pred + R
            S = P_pred + R

            # Коэффициент усиления Калмана: K = P_pred * S^-1
            K = P_pred @ np.linalg.inv(S)

            # Обновление состояния
            X_state = X_pred + K @ y_err

            # Обновление ковариации
            P = (I - K) @ P_pred

            filtered_x[k] = X_state[0, 0]
            filtered_y[k] = X_state[1, 0]

        return filtered_x, filtered_y


def visualize_and_save(x_true: np.ndarray, y_true: np.ndarray,
                       x_filt: np.ndarray, y_filt: np.ndarray,
                       save_path: Path):
    """Строит и сохраняет график сравнения траекторий."""
    num_points = len(x_true)

    plt.figure(figsize=(10, 8))
    plt.plot(x_true, y_true, c='blue', label=f'Исходные данные ({num_points} точек)', alpha=0.6)
    plt.plot(x_filt, y_filt, 'r--', label='Фильтр Калмана', linewidth=2)

    plt.xlabel('X (метры)')
    plt.ylabel('Y (метры)')
    plt.title(f'Траектория: {save_path.stem} | Точек: {num_points}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.savefig(save_path)
    plt.close()


def process_track_list(df_list: List[pd.DataFrame],
                       save_dir: Path,
                       processor: DataProcessor,
                       kalman_filter: KalmanFilterRW,
                       label: str):
    """
    Обрабатывает список треков: фильтрует и сохраняет визуализацию.

    Args:
        df_list: Список DataFrame для обработки.
        save_dir: Папка для сохранения картинок.
        processor: Экземпляр класса DataProcessor.
        kalman_filter: Экземпляр класса KalmanFilterCV.
        label: Название типа данных (для логирования).
    """
    print(f"Обработка {label}: {len(df_list)} шт.")

    for i, df in enumerate(df_list):
        lon, lat = processor.get_lon_lat(df)
        x, y = processor.convert_to_local_cartesian(lon, lat)
        time = df['time'].to_numpy()

        x_filt, y_filt = kalman_filter.filter(x, y, time)

        save_path = save_dir / f'track_{i}.png'
        visualize_and_save(x, y, x_filt, y_filt, save_path)


if __name__ == '__main__':
    # Определение путей
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'post_processing' / 'example.csv'

    # Директории для сохранения картинок
    pict_dir = project_root / 'data' / 'pict'
    true_dir = pict_dir / 'true'
    false_dir = pict_dir / 'false'

    # Инициализация директорий
    true_dir.mkdir(parents=True, exist_ok=True)
    false_dir.mkdir(parents=True, exist_ok=True)

    # Инициализация классов
    processor = DataProcessor()

    # Параметры фильтра идентичны для обоих списков, поэтому создаем один экземпляр
    # (метод filter сбрасывает состояние внутри себя)
    kf = KalmanFilterRW(sigma_proc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    # Загрузка и парсинг
    df = processor.load_csv(data_path)
    list_valid_df, list_invalid_df = processor.parse_intervals(df)

    # Обработка валидных интервалов
    process_track_list(list_valid_df, true_dir, processor, kf, "валидных интервалов")

    # Обработка невалидных интервалов
    process_track_list(list_invalid_df, false_dir, processor, kf, "невалидных интервалов")

    print("Готово.")
