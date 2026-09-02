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
        time содержит np.datetime64. Все временные интервалы
        внутри фильтра преобразуются в секунды.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - filtered_x: отфильтрованные координаты X;
            - filtered_y: отфильтрованные координаты Y;
            - log_likelihood: логарифм правдоподобия каждого наблюдения;
            - mahalanobis_sq: квадрат расстояния Махаланобиса для каждого наблюдения.
        """
        if not (len(x) == len(y) == len(time)):
            raise ValueError("x, y и time должны иметь одинаковую длину")

        if len(x) < 2:
            return (
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.full(len(x), np.nan, dtype=np.float64),
                np.full(len(x), np.nan, dtype=np.float64),
            )

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        time = np.asarray(time, dtype="datetime64[ns]")

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        R = np.eye(2, dtype=np.float64) * (self.sigma_meas ** 2)
        I = np.eye(4, dtype=np.float64,)
        # ---------------------------------------------------------
        # Инициализация
        # ---------------------------------------------------------

        dt0 = (time[1] - time[0]) / np.timedelta64(1, "s")
        dt0 = float(dt0)
        vx0 = 0.0
        vy0 = 0.0

        if dt0 > 0.0:
            vx0 = (x[1] - x[0]) / dt0
            vy0 = (y[1] - y[0]) / dt0

        X_state = np.array(
            [
                x[0],
                y[0],
                vx0,
                vy0,
            ],
            dtype=np.float64,
        ).reshape(4, 1)

        # Начальная ковариация.
        P = np.eye(4, dtype=np.float64) * 500.0

        if dt0 > 0.0:
            # Если две координаты имеют независимую
            # погрешность sigma_meas, то:
            #
            # Var(x2 - x1) = 2 * sigma_meas²
            #
            # Var(v) = 2 * sigma_meas² / dt²
            vel_var = 2.0 * self.sigma_meas ** 2 / dt0 ** 2
            P[2, 2] = vel_var
            P[3, 3] = vel_var
        else:
            P[2, 2] = 100.0
            P[3, 3] = 100.0

        # ---------------------------------------------------------
        # Результаты
        # ---------------------------------------------------------
        filtered_x = np.zeros(len(x), dtype=np.float64)
        filtered_y = np.zeros(len(y), dtype=np.float64)

        filtered_x[0] = x[0]
        filtered_y[0] = y[0]
        log_likelihood = np.full(len(x), np.nan, dtype=np.float64)
        mahalanobis_sq = np.full(len(x), np.nan, dtype=np.float64)

        log_2pi = np.log(2.0 * np.pi)
        dim = 2
        # ---------------------------------------------------------
        # Основной цикл
        # ---------------------------------------------------------
        for k in range(1, len(time)):
            # np.datetime64 -> секунды
            dt = (time[k] - time[k - 1]) / np.timedelta64(1, "s")
            dt = float(dt)
            # Некорректный dt не должен ломать фильтр.
            if dt <= 0.0:
                dt = 0.0
            # -----------------------------------------------------
            # Prediction
            # -----------------------------------------------------
            F = self._get_transition_matrix(dt)
            Q = self._get_process_noise_matrix(dt)

            X_pred = F @ X_state
            P_pred = F @ P @ F.T + Q

            # -----------------------------------------------------
            # Update
            # -----------------------------------------------------
            z = np.array(
                [
                    x[k],
                    y[k],
                ],
                dtype=np.float64,
            ).reshape(2, 1)

            # Innovation
            innovation = z - H @ X_pred

            # Innovation covariance
            S = H @ P_pred @ H.T + R

            # Накапливаем симметричность из-за
            # численных погрешностей.
            S = 0.5 * (S + S.T)

            # -----------------------------------------------------
            # log likelihood и Mahalanobis
            # -----------------------------------------------------
            sign, logdet = np.linalg.slogdet(S)
            if sign > 0.0 and np.isfinite(logdet):
                try:
                    # S @ solution = innovation
                    solved_innovation = np.linalg.solve(S, innovation)
                    mahalanobis_dist = float((innovation.T @ solved_innovation).item())

                    if mahalanobis_dist >= 0.0:
                        mahalanobis_sq[k] = mahalanobis_dist
                        log_likelihood[k] = -0.5 * (dim * log_2pi + logdet + mahalanobis_dist)
                    else:
                        # Теоретически для корректной
                        # положительно определённой S
                        # этого быть не должно.
                        solved_innovation = np.linalg.pinv(S) @ innovation

                except np.linalg.LinAlgError:
                    solved_innovation = np.linalg.pinv(S) @ innovation
            else:
                solved_innovation = np.linalg.pinv(S) @ innovation

            # -----------------------------------------------------
            # Kalman Gain
            # -----------------------------------------------------
            # K = P_pred H^T S^-1
            # Вместо вычисления S^-1:
            kalman_gain = np.linalg.solve(S, H @ P_pred).T
            # -----------------------------------------------------
            # State update
            # -----------------------------------------------------
            X_state = X_pred + kalman_gain @ innovation
            # -----------------------------------------------------
            # Joseph form covariance update
            # -----------------------------------------------------
            I_KH = I - kalman_gain @ H
            P = I_KH @ P_pred @ I_KH.T + kalman_gain @ R @ kalman_gain.T
            # Возвращаем симметричность.
            P = 0.5 * (P + P.T)

            filtered_x[k] = X_state[0, 0]
            filtered_y[k] = X_state[1, 0]

        return filtered_x, filtered_y, log_likelihood, mahalanobis_sq,


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
