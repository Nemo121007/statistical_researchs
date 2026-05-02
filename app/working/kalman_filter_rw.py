from pathlib import Path
from typing import Tuple

import numpy as np

from app.working.data_processor import DataProcessor


class KalmanFilterRW:
    """Класс реализации фильтра Калмана для модели случайного блуждания (без скорости)."""

    def __init__(self, sigma_acc: float = 0.04, sigma_meas: float = 2.4):
        """
        Инициализация параметров фильтра.

        Args:
            sigma_acc: СКО шума ускорения.
            sigma_meas: СКО шума измерений.
        """
        self.sigma_acc = sigma_acc
        self.sigma_meas = sigma_meas

    def filter(
            self, x: np.ndarray, y: np.ndarray, time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # pylint: disable=too-many-locals
        """
        Применяет фильтр Калмана к траектории и вычисляет логарифм правдоподобия.

        Args:
            x: Массив координат X (в метрах).
            y: Массив координат Y (в метрах).
            time: Массив времени (в секундах).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Сглаженные координаты X.
            - Сглаженные координаты Y.
            - Массив логарифмов правдоподобия (log-likelihood).
        """
        if len(x) < 2:
            return x, y, np.array([]), np.array([])

        n_dim = 2

        # Матрица перехода F = I
        F = np.eye(n_dim)

        # Матрица измерений H = I
        H = np.eye(n_dim)

        # Ковариация шума измерений R
        R = np.eye(n_dim) * (self.sigma_meas**2)

        # Инициализация состояния X_state = [x, y]
        X_state = np.array([x[0], y[0]]).reshape(n_dim, 1)

        # Инициализация ковариации ошибки P
        P = np.eye(n_dim) * (self.sigma_meas**2)

        # Массивы для результатов
        filtered_x = np.zeros(len(x))
        filtered_y = np.zeros(len(y))
        filtered_x[0] = x[0]
        filtered_y[0] = y[0]

        I = np.eye(n_dim)

        # Переменная для накопления правдоподобия
        log_likelihood = np.full(len(x), np.nan)
        mahalanobis_sq = np.full(len(x), np.nan)

        log_2pi = np.log(2 * np.pi)
        dim = 2  # Размерность измерения

        for k in range(1, len(time)):
            dt = time[k] - time[k - 1]
            if dt <= 0:
                dt = 0.0

            # --- Формирование матрицы шума процесса Q ---
            Q = np.eye(n_dim) * (self.sigma_acc**2 * dt)

            # --- Prediction (Этап предсказания) ---
            X_pred = F @ X_state
            P_pred = P + Q

            # --- Update (Этап коррекции) ---
            z = np.array([x[k], y[k]]).reshape(n_dim, 1)

            # Невязка (Innovation)
            y_err = z - (H @ X_pred)

            # Ковариация невязки
            S = H @ P_pred @ H.T + R

            # --- Вычисление P(y_k | y_(1:k-1)) ---
            det_S = np.linalg.det(S)
            # Используем логарифм для численной устойчивости
            if det_S > 0:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = (y_err.T @ S_inv @ y_err).item()
                log_likelihood[k] = -0.5 * (dim * log_2pi + np.log(det_S) + mahalanobis_dist)
                mahalanobis_sq[k] = mahalanobis_dist
            else:
                # Защита: если матрица вырождена, используем псевдообратную матрицу
                # для продолжения фильтрации, но правдоподобие не считаем
                S_inv = np.linalg.pinv(S)

            # Коэффициент усиления Калмана
            K = P_pred @ S_inv

            # Обновление состояния
            X_state = X_pred + K @ y_err

            # Обновление ковариации
            P = (I - K) @ P_pred

            filtered_x[k] = X_state[0, 0]
            filtered_y[k] = X_state[1, 0]

        return filtered_x, filtered_y, log_likelihood, mahalanobis_sq


if __name__ == "__main__":
    # Определение путей
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "post_processing" / "example.csv"

    # Директории для сохранения картинок
    pict_dir = project_root / "data" / "pict"
    true_dir = pict_dir / "true"
    false_dir = pict_dir / "false"

    # Инициализация директорий
    true_dir.mkdir(parents=True, exist_ok=True)
    false_dir.mkdir(parents=True, exist_ok=True)

    # Инициализация классов
    processor = DataProcessor()

    # Параметры фильтра идентичны для обоих списков, поэтому создаем один экземпляр
    # (метод filter сбрасывает состояние внутри себя)
    kf = KalmanFilterRW(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    # Загрузка и парсинг
    df = processor.load_csv(data_path)
    list_valid_df, list_invalid_df = processor.parse_intervals(df)

    # Обработка валидных интервалов
    processor.process_track_list(list_valid_df, true_dir, processor, kf, "валидных интервалов")

    # Обработка невалидных интервалов
    processor.process_track_list(list_invalid_df, false_dir, processor, kf, "невалидных интервалов")

    print("Готово.")
