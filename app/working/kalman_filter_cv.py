from pathlib import Path
from typing import Tuple

import numpy as np

from app.working.data_processor import DataProcessor


class KalmanFilterCV:
    """Класс реализации фильтра Калмана для модели постоянной скорости."""

    def __init__(self, sigma_acc: float = 0.04, sigma_meas: float = 2.4):
        """
        Инициализация фильтра Калмана (модель постоянной скорости).
        
        Args:
            sigma_acc: Стандартное отклонение шума ускорения (процесса).
            sigma_meas: Стандартное отклонение шума измерений.
        """
        self.sigma_acc = sigma_acc
        self.sigma_meas = sigma_meas

    @staticmethod
    def _get_transition_matrix(dt: float) -> np.ndarray:
        """
        Формирует матрицу перехода состояний (F) для модели с постоянной скоростью.
        
        Args:
            dt: Интервал времени между измерениями (дельта времени).

        Returns:
            np.ndarray: Матрица перехода состояний размером 4x4.
        """
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _get_process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Формирует ковариационную матрицу шума процесса (Q).
        
        Args:
            dt: Интервал времени между измерениями (дельта времени).

        Returns:
            np.ndarray: Ковариационная матрица шума процесса размером 4x4.
        """
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        Q = np.zeros((4, 4))
        Q[0, 0] = dt4 / 4
        Q[1, 1] = dt4 / 4
        Q[2, 2] = dt2
        Q[3, 3] = dt2

        Q[0, 2] = dt3 / 2
        Q[2, 0] = dt3 / 2
        Q[1, 3] = dt3 / 2
        Q[3, 1] = dt3 / 2

        return Q * (self.sigma_acc**2)

    def filter(
        # pylint: disable=too-many-locals
        self,
        x: np.ndarray,
        y: np.ndarray,
        time: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Применяет фильтр Калмана к серии измерений координат.
        
        Args:
            x: Массив измеренных координат X.
            y: Массив измеренных координат Y.
            time: Массив временных меток измерений.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Кортеж из четырех массивов:
                - Отфильтрованные координаты X.
                - Отфильтрованные координаты Y.
                - Логарифмы функции правдоподобия для каждого шага.
                - Квадраты расстояния Махаланобиса для каждого шага.
        """
        if len(x) < 2:
            return x, y, np.array([]), np.array([])

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * (self.sigma_meas**2)
        I = np.eye(4)

        # Инициализация скорости
        dt0 = time[1] - time[0]
        vx0 = 0.0
        vy0 = 0.0
        # Защита от деления на ноль или отрицательного времени
        if dt0 > 0:
            vx0 = (x[1] - x[0]) / dt0
            vy0 = (y[1] - y[0]) / dt0
        X_state = np.array([x[0], y[0], vx0, vy0]).reshape(4, 1)

        P = np.eye(4) * 500.0
        # Если время dt0 валидно, можно уменьшить неопределенность скорости
        if dt0 > 0:
            # Дисперсия скорости = 2 * дисперсия измерения / dt^2
            # (по формуле распространения ошибки для разности двух точек)
            vel_var = 2 * (self.sigma_meas**2) / (dt0**2)
            P[2, 2] = vel_var
            P[3, 3] = vel_var
        else:
            P[2, 2] = 100.0
            P[3, 3] = 100.0

        filtered_x = np.zeros(len(x))
        filtered_y = np.zeros(len(y))
        filtered_x[0] = x[0]
        filtered_y[0] = y[0]

        # Переменная для накопления правдоподобия
        log_likelihood = np.full(len(x), np.nan)
        mahalanobis_sq = np.full(len(x), np.nan)

        # Константа для формулы (2 * pi)
        log_2pi = np.log(2 * np.pi)
        dim = 2  # Размерность измерения (x, y)

        for k in range(1, len(time)):
            dt = time[k] - time[k - 1]
            if dt <= 0:
                dt = 0.0

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
                # Вычисление расстояния Махаланобиса
                mahalanobis_dist = (y_err.T @ S_inv @ y_err).item()
                log_likelihood[k] = -0.5 * (dim * log_2pi + np.log(det_S) + mahalanobis_dist)
                mahalanobis_sq[k] = mahalanobis_dist
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
    kf = KalmanFilterCV(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    # Загрузка и парсинг
    df = processor.load_csv(data_path)
    list_valid_df, list_invalid_df = processor.parse_intervals(df)

    # Обработка валидных интервалов
    processor.process_track_list(list_valid_df, true_dir, processor, kf, "валидных интервалов")

    # Обработка невалидных интервалов
    processor.process_track_list(list_invalid_df, false_dir, processor, kf, "невалидных интервалов")

    print("Готово.")
