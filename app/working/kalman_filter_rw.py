from pathlib import Path
from typing import Tuple

import numpy as np


class KalmanFilterRW:
    """
    Фильтр Калмана для модели случайного блуждания.

    Состояние:
        X = [x, y]^T

    Модель перехода:
        X_k = X_{k-1} + w_k

    Матрица перехода:
        F = I

    Process noise:
        Q = sigma_rw^2 * dt * I

    где:
        sigma_rw — интенсивность случайного блуждания,
        единицы измерения: м / sqrt(с).

    Measurement noise:
        R = sigma_meas^2 * I

    где:
        sigma_meas — СКО шума измерений, м.
    """

    def __init__(
        self,
        sigma_rw: float = 1.0,
        sigma_meas: float = 2.4,
    ):
        """
        Инициализация параметров фильтра.

        Args:
            sigma_rw:
                Интенсивность случайного блуждания,
                м / sqrt(с).

            sigma_meas:
                СКО шума измерений, м.
        """
        if sigma_rw < 0.0:
            raise ValueError(
                "sigma_rw должен быть >= 0"
            )

        if sigma_meas <= 0.0:
            raise ValueError(
                "sigma_meas должен быть > 0"
            )

        self.sigma_rw = float(sigma_rw)
        self.sigma_meas = float(sigma_meas)

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

        Args:
            x:
                Координаты X в локальной декартовой системе, м.

            y:
                Координаты Y в локальной декартовой системе, м.

            time:
                Временные метки np.datetime64.

        Returns:
            filtered_x:
                Отфильтрованные координаты X.

            filtered_y:
                Отфильтрованные координаты Y.

            log_likelihood:
                Логарифм правдоподобия каждого наблюдения.

            mahalanobis_sq:
                Квадрат расстояния Махаланобиса
                инновации каждого наблюдения.
        """

        # ==============================================================
        # Проверка входных данных
        # ==============================================================

        if not (
            len(x)
            == len(y)
            == len(time)
        ):
            raise ValueError(
                "x, y и time должны иметь одинаковую длину"
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        y = np.asarray(
            y,
            dtype=np.float64,
        )

        time = np.asarray(
            time,
            dtype="datetime64[ns]",
        )

        # ==============================================================
        # Недостаточно точек для одного шага фильтра
        # ==============================================================

        if len(x) < 2:
            return (
                x.copy(),
                y.copy(),
                np.full(
                    len(x),
                    np.nan,
                    dtype=np.float64,
                ),
                np.full(
                    len(x),
                    np.nan,
                    dtype=np.float64,
                ),
            )

        # ==============================================================
        # Размерность состояния
        # ==============================================================

        n_dim = 2

        # ==============================================================
        # Матрицы модели
        # ==============================================================

        # RW:
        #
        # X_k = X_{k-1} + w_k
        #
        # Поэтому F = I.
        F = np.eye(
            n_dim,
            dtype=np.float64,
        )

        # Измерение непосредственно наблюдает
        # состояние [x, y].
        H = np.eye(
            n_dim,
            dtype=np.float64,
        )

        # Единичная матрица.
        I = np.eye(
            n_dim,
            dtype=np.float64,
        )

        # ==============================================================
        # Measurement noise
        # ==============================================================

        R = (
            np.eye(
                n_dim,
                dtype=np.float64,
            )
            * self.sigma_meas**2
        )

        # ==============================================================
        # Начальное состояние
        # ==============================================================

        X_state = np.array(
            [
                x[0],
                y[0],
            ],
            dtype=np.float64,
        ).reshape(
            n_dim,
            1,
        )

        # Начальная ковариация.
        P = (
            np.eye(
                n_dim,
                dtype=np.float64,
            )
            * self.sigma_meas**2
        )

        # ==============================================================
        # Результаты
        # ==============================================================

        filtered_x = np.empty(
            len(x),
            dtype=np.float64,
        )

        filtered_y = np.empty(
            len(y),
            dtype=np.float64,
        )

        filtered_x[0] = x[0]
        filtered_y[0] = y[0]

        log_likelihood = np.full(
            len(x),
            np.nan,
            dtype=np.float64,
        )

        mahalanobis_sq = np.full(
            len(x),
            np.nan,
            dtype=np.float64,
        )

        log_2pi = np.log(
            2.0 * np.pi
        )

        dim = n_dim

        # ==============================================================
        # Основной цикл фильтра
        # ==============================================================

        for k in range(
            1,
            len(time),
        ):
            # ----------------------------------------------------------
            # Интервал времени в секундах
            # ----------------------------------------------------------

            dt = (
                time[k]
                - time[k - 1]
            ) / np.timedelta64(
                1,
                "s",
            )

            dt = float(dt)

            if dt < 0.0:
                raise ValueError(
                    "time должен быть "
                    "монотонно неубывающим"
                )

            # ----------------------------------------------------------
            # Process noise
            #
            # Q = sigma_rw^2 * dt * I
            # ----------------------------------------------------------

            Q = (
                np.eye(
                    n_dim,
                    dtype=np.float64,
                )
                * (
                    self.sigma_rw**2
                    * dt
                )
            )

            # ----------------------------------------------------------
            # Prediction
            #
            # X_pred = F X
            #
            # P_pred = F P F^T + Q
            # ----------------------------------------------------------

            X_pred = (
                F @ X_state
            )

            P_pred = (
                F
                @ P
                @ F.T
                + Q
            )

            # ----------------------------------------------------------
            # Текущее измерение
            # ----------------------------------------------------------

            z = np.array(
                [
                    x[k],
                    y[k],
                ],
                dtype=np.float64,
            ).reshape(
                n_dim,
                1,
            )

            # ----------------------------------------------------------
            # Innovation
            #
            # ν = z - H X_pred
            # ----------------------------------------------------------

            innovation = (
                z
                - H @ X_pred
            )

            # ----------------------------------------------------------
            # Innovation covariance
            #
            # S = H P_pred H^T + R
            # ----------------------------------------------------------

            S = (
                H
                @ P_pred
                @ H.T
                + R
            )

            # Защита от небольшой численной
            # асимметрии.
            S = 0.5 * (
                S + S.T
            )

            # ----------------------------------------------------------
            # Проверка S
            # ----------------------------------------------------------

            sign, logdet = (
                np.linalg.slogdet(S)
            )

            if (
                sign <= 0.0
                or not np.isfinite(logdet)
            ):
                raise np.linalg.LinAlgError(
                    "Матрица S не является "
                    "положительно определённой"
                )

            # ----------------------------------------------------------
            # Решение:
            #
            # S * S^-1 innovation = innovation
            #
            # Без явного вычисления inv(S).
            # ----------------------------------------------------------

            try:
                solved_innovation = (
                    np.linalg.solve(
                        S,
                        innovation,
                    )
                )
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "Не удалось решить систему "
                    "S * x = innovation"
                ) from exc

            # ----------------------------------------------------------
            # Mahalanobis^2
            #
            # M² = ν^T S^-1 ν
            # ----------------------------------------------------------

            mahalanobis_dist = float(
                (
                    innovation.T
                    @ solved_innovation
                ).item()
            )

            # Защита от отрицательной
            # численной погрешности.
            mahalanobis_dist = max(
                mahalanobis_dist,
                0.0,
            )

            mahalanobis_sq[k] = (
                mahalanobis_dist
            )

            # ----------------------------------------------------------
            # Log-likelihood
            #
            # log p(z_k | z_1...z_{k-1})
            #
            # = -1/2 * (
            #       d log(2π)
            #       + log det(S)
            #       + M²
            #   )
            # ----------------------------------------------------------

            log_likelihood[k] = (
                -0.5
                * (
                    dim * log_2pi
                    + logdet
                    + mahalanobis_dist
                )
            )

            # ----------------------------------------------------------
            # Kalman Gain
            #
            # K = P_pred H^T S^-1
            #
            # Снова не вычисляем inv(S).
            # ----------------------------------------------------------

            try:
                K = np.linalg.solve(
                    S,
                    H @ P_pred,
                ).T
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "Не удалось вычислить "
                    "Kalman Gain"
                ) from exc

            # ----------------------------------------------------------
            # State update
            #
            # X = X_pred + K ν
            # ----------------------------------------------------------

            X_state = (
                X_pred
                + K @ innovation
            )

            # ----------------------------------------------------------
            # Joseph form
            #
            # P =
            #   (I-KH) P_pred (I-KH)^T
            #   + K R K^T
            #
            # Такая форма лучше сохраняет
            # положительную полуопределённость
            # ковариационной матрицы.
            # ----------------------------------------------------------

            I_KH = (
                I
                - K @ H
            )

            P = (
                I_KH
                @ P_pred
                @ I_KH.T
                + K
                @ R
                @ K.T
            )

            # Симметризация.
            P = 0.5 * (
                P + P.T
            )

            # ----------------------------------------------------------
            # Сохранение результата
            # ----------------------------------------------------------

            filtered_x[k] = (
                X_state[0, 0]
            )

            filtered_y[k] = (
                X_state[1, 0]
            )

        return (
            filtered_x,
            filtered_y,
            log_likelihood,
            mahalanobis_sq,
        )


if __name__ == "__main__":
    # ==============================================================
    # Пример создания фильтра
    # ==============================================================

    project_root = (
        Path(__file__)
        .parent
        .parent
        .parent
    )

    data_path = (
        project_root
        / "data"
        / "post_processing"
        / "example.csv"
    )

    pict_dir = (
        project_root
        / "data"
        / "pict"
    )

    true_dir = (
        pict_dir
        / "true"
    )

    false_dir = (
        pict_dir
        / "false"
    )

    true_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # Инициализация RW-фильтра
    #
    # sigma_rw:
    #     м / sqrt(с)
    #
    # sigma_meas:
    #     м
    # ==============================================================

    kf = KalmanFilterRW(
        sigma_rw=0.0001 * 0.04,
        sigma_meas=1 * 2.4,
    )

    print(
        "KalmanFilterRW initialized:"
    )

    print(
        f"  sigma_rw   = {kf.sigma_rw}"
    )

    print(
        f"  sigma_meas = {kf.sigma_meas}"
    )