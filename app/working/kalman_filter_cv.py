from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from app.working.data_processor import DataProcessor


class KalmanFilterCV:
    """Класс реализации фильтра Калмана для модели постоянной скорости."""

    def __init__(self, sigma_acc: float = 0.04, sigma_meas: float = 2.4):
        """
        Инициализация параметров фильтра.

        Args:
            sigma_acc: СКО шума ускорения.
            sigma_meas: СКО шума измерений.
        """
        self.sigma_acc = sigma_acc
        self.sigma_meas = sigma_meas

    @staticmethod
    def _get_transition_matrix(dt: float) -> np.ndarray:
        """Формирует матрицу перехода F."""
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _get_process_noise_matrix(self, dt: float) -> np.ndarray:
        """Формирует ковариационную матрицу шума процесса Q."""
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        Q = np.zeros((4, 4))
        Q[0, 0] = dt4 / 4
        Q[1, 1] = dt4 / 4
        Q[2, 2] = dt2
        Q[3, 3] = dt2

        Q[0, 2] = dt3 / 2
        Q[2, 0] = dt3 / 2
        Q[1, 3] = dt3 / 2
        Q[3, 1] = dt3 / 2

        return Q * (self.sigma_acc ** 2)

    def filter(self, x: np.ndarray, y: np.ndarray, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Применяет фильтр Калмана и вычисляет логарифм правдоподобия.
        """
        if len(x) < 2:
            return x, y, 0.0

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        R = np.eye(2) * (self.sigma_meas ** 2)

        X_state = np.array([x[0], y[0], 0.0, 0.0]).reshape(4, 1)

        P = np.eye(4) * 500.0
        P[2, 2] = 100.0
        P[3, 3] = 100.0

        filtered_x = np.zeros(len(x))
        filtered_y = np.zeros(len(y))
        filtered_x[0] = x[0]
        filtered_y[0] = y[0]

        I = np.eye(4)

        # Переменная для накопления правдоподобия
        total_log_likelihood = 0.0

        # Константа для формулы (2 * pi)
        log_2pi = np.log(2 * np.pi)
        dim = 2  # Размерность измерения (x, y)

        for k in range(1, len(time)):
            dt = time[k] - time[k - 1]
            if dt <= 0:
                dt = 1e-5

            # --- Prediction ---
            F = self._get_transition_matrix(dt)
            Q = self._get_process_noise_matrix(dt)

            X_pred = F @ X_state
            P_pred = F @ P @ F.T + Q

            # --- Update ---
            z = np.array([x[k], y[k]]).reshape(2, 1)

            # Невязка (Innovation)
            y_err = z - (H @ X_pred)

            # Ковариация невязки (Innovation Covariance)
            S = H @ P_pred @ H.T + R

            # --- Вычисление P(y_k | y_(1:k-1)) ---
            det_S = np.linalg.det(S)
            # Используем логарифм для численной устойчивости
            if det_S > 0:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = float(y_err.T @ S_inv @ y_err)
                step_log_likelihood = -0.5 * (dim * log_2pi + np.log(det_S) + mahalanobis_dist)
                total_log_likelihood += step_log_likelihood
            else:
                # Защита: если матрица вырождена, используем псевдообратную матрицу
                # для продолжения фильтрации, но правдоподобие не считаем
                S_inv = np.linalg.pinv(S)

            # Остальной код обновления
            K = P_pred @ H.T @ S_inv  # S_inv уже вычислен
            X_state = X_pred + K @ y_err
            P = (I - K @ H) @ P_pred

            filtered_x[k] = X_state[0, 0]
            filtered_y[k] = X_state[1, 0]

        return filtered_x, filtered_y, total_log_likelihood


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
                       kalman_filter: KalmanFilterCV,
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
    kf = KalmanFilterCV(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    # Загрузка и парсинг
    df = processor.load_csv(data_path)
    list_valid_df, list_invalid_df = processor.parse_intervals(df)

    # Обработка валидных интервалов
    process_track_list(list_valid_df, true_dir, processor, kf, "валидных интервалов")

    # Обработка невалидных интервалов
    process_track_list(list_invalid_df, false_dir, processor, kf, "невалидных интервалов")

    print("Готово.")
