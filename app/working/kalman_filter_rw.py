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
            self,
            x: np.ndarray,
            y: np.ndarray,
            time: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Применяет фильтр Калмана RW к серии измерений.
        time содержит np.datetime64.
        Внутри фильтра dt переводится в секунды.
        """
        if not len(x) == len(y) == len(time):
            raise ValueError("x, y и time должны иметь одинаковую длину")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        time = np.asarray(time, dtype="datetime64[ns]")

        if len(x) < 2:
            return (x, y, np.full(len(x), np.nan, dtype=np.float64,), np.full(len(x), np.nan, dtype=np.float64))

        n_dim = 2

        F = np.eye(n_dim, dtype=np.float64)

        H = np.eye(n_dim, dtype=np.float64)

        R = (np.eye(n_dim, dtype=np.float64) * self.sigma_meas ** 2)

        I = np.eye(n_dim, dtype=np.float64)

        X_state = np.array(
            [
                x[0],
                y[0],
            ],
            dtype=np.float64,
        ).reshape(n_dim, 1)

        P = np.eye(n_dim, dtype=np.float64) * self.sigma_meas ** 2

        filtered_x = np.zeros(len(x), dtype=np.float64,)

        filtered_y = np.zeros(len(y), dtype=np.float64)

        filtered_x[0] = x[0]
        filtered_y[0] = y[0]
        log_likelihood = np.full(len(x), np.nan, dtype=np.float64)
        mahalanobis_sq = np.full(len(x), np.nan, dtype=np.float64)

        log_2pi = np.log(2.0 * np.pi)
        dim = 2

        for k in range(1, len(time)):
            # np.datetime64 -> секунды
            dt = (time[k] - time[k - 1]) / np.timedelta64(1, "s")
            dt = float(dt)
            if dt <= 0.0:
                dt = 0.0

            # Шум процесса RW.
            Q = np.eye(n_dim, dtype=np.float64) * (self.sigma_acc ** 2 * dt)

            # Prediction.
            X_pred = F @ X_state
            P_pred = P + Q

            # Measurement.
            z = np.array(
                [
                    x[k],
                    y[k],
                ],
                dtype=np.float64,
            ).reshape(n_dim, 1)

            # Innovation.
            innovation = z - H @ X_pred

            # Innovation covariance.
            S = H @ P_pred @ H.T + R

            S = 0.5 * (S + S.T)

            # Log-likelihood и Mahalanobis².
            sign, logdet = np.linalg.slogdet(S)

            if sign > 0.0 and np.isfinite(logdet):
                try:
                    solved_innovation = np.linalg.solve(S, innovation)
                    mahalanobis_dist = float((innovation.T @ solved_innovation).item())

                    if mahalanobis_dist >= 0.0:
                        mahalanobis_sq[k] = mahalanobis_dist
                        log_likelihood[k] = -0.5 * (dim * log_2pi + logdet + mahalanobis_dist)
                    else:
                        # Теоретически такого быть не должно
                        # для корректной S.
                        solved_innovation = np.linalg.pinv(S) @ innovation
                except np.linalg.LinAlgError:
                    solved_innovation = (np.linalg.pinv(S) @ innovation)
            else:
                solved_innovation = (np.linalg.pinv(S)@ innovation)

            # Kalman Gain.
            K = np.linalg.solve(S, H @ P_pred,).T

            # State update.
            X_state = (X_pred + K @ innovation)

            # Joseph form.
            I_KH = I - K @ H
            P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
            P = 0.5 * P + P.T

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
