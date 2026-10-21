"""Геопространственные метрики на заполненных траекториях."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.evaluation.series import TrajectorySeries
from app.geo.local_enu import LocalEnuConverter
from app.viz import plot_track_maps


def _path_length_xy(x_m: np.ndarray, y_m: np.ndarray) -> float:
    """Длина ломаной в локальной ENU (метры)."""
    if x_m.size < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x_m), np.diff(y_m))))


def _filled_to_local_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    converter = LocalEnuConverter(origin_lat, origin_lon)
    return converter.to_local_xy(lat, lon)


class GeospatialMetrics:
    """RMSE, максимальное отклонение и distance loss между траекториями."""

    @staticmethod
    def compute(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> Dict[str, float]:
        """
        Считает геопространственные метрики.
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Dict[str, float]: словарь с метриками.
        """
        deviations = GeospatialMetrics.pointwise_deviations_m(prediction, ground_truth)
        rmse = GeospatialMetrics.rmse_m(deviations)
        max_dev = float(np.max(deviations)) if deviations.size else 0.0
        distance_loss = GeospatialMetrics.distance_loss(prediction, ground_truth)
        return {
            "rmse_m": rmse,
            "max_deviation_m": max_dev,
            "distance_loss_m": distance_loss,
        }

    @staticmethod
    def filled_trajectories(
        prediction: TrajectorySeries, ground_truth: TrajectorySeries
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Координаты ground truth и prediction после заполнения аномалий.
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: координаты (lat, lon) для ground truth и prediction.
        """
        ground_truth.assert_same_index(prediction)
        time_s = ground_truth.time_seconds
        gt_lat_f, gt_lon_f = GeospatialMetrics.__fill_anomaly_coordinates(
            time_s, ground_truth.lat, ground_truth.lon, ground_truth.anomaly_mask
        )
        pred_lat_f, pred_lon_f = GeospatialMetrics.__fill_anomaly_coordinates(
            time_s, prediction.lat, prediction.lon, prediction.anomaly_mask
        )
        return gt_lat_f, gt_lon_f, pred_lat_f, pred_lon_f

    @staticmethod
    def pointwise_deviations_m(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> np.ndarray:
        """
        Расстояние $d_i$ в локальной ENU (м) между заполненными траекториями в каждой точке.
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            np.ndarray: массив геодезических расстояний между точками.
        """
        gt_lat, gt_lon, pred_lat, pred_lon = GeospatialMetrics.filled_trajectories(prediction, ground_truth)
        origin_lat = float(gt_lat[0])
        origin_lon = float(gt_lon[0])
        gt_x, gt_y = _filled_to_local_xy(gt_lat, gt_lon, origin_lat, origin_lon)
        pred_x, pred_y = _filled_to_local_xy(pred_lat, pred_lon, origin_lat, origin_lon)
        return np.hypot(pred_x - gt_x, pred_y - gt_y)

    @staticmethod
    def rmse_m(deviations: np.ndarray) -> float:
        """RMSE по точкам с ненулевым отклонением $d_i > 0$ (см. Отчёт.tex)."""
        positive = deviations[deviations > 0.0]
        if positive.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(positive))))

    @staticmethod
    def distance_loss(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> float:
        """
        Абсолютный distance loss длин заполненных траекторий.
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            float: значение distance loss.
        """
        gt_lat, gt_lon, pred_lat, pred_lon = GeospatialMetrics.filled_trajectories(prediction, ground_truth)
        origin_lat = float(gt_lat[0])
        origin_lon = float(gt_lon[0])
        gt_x, gt_y = _filled_to_local_xy(gt_lat, gt_lon, origin_lat, origin_lon)
        pred_x, pred_y = _filled_to_local_xy(pred_lat, pred_lon, origin_lat, origin_lon)
        length_truth = _path_length_xy(gt_x, gt_y)
        length_prediction = _path_length_xy(pred_x, pred_y)
        loss = abs(length_prediction - length_truth)
        return loss

    @staticmethod
    def reference_path_length_m(ground_truth: TrajectorySeries) -> float:
        """
        Длина заполненной эталонной траектории (для сравнения в экспериментах).
        Args:
            ground_truth: эталонная траектория.

        Returns:
            float: длина заполненной эталонной траектории.
        """
        time_s = ground_truth.time_seconds
        lat_f, lon_f = GeospatialMetrics.__fill_anomaly_coordinates(
            time_s, ground_truth.lat, ground_truth.lon, ground_truth.anomaly_mask
        )
        origin_lat = float(lat_f[0])
        origin_lon = float(lon_f[0])
        x_m, y_m = _filled_to_local_xy(lat_f, lon_f, origin_lat, origin_lon)
        return _path_length_xy(x_m, y_m)

    @staticmethod
    def prediction_path_length_m(prediction: TrajectorySeries) -> float:
        """
        Длина заполненной траектории prediction.
        Args:
            prediction: траектория с предсказаниями модели.

        Returns:
            float: длина заполненной траектории prediction.
        """
        time_s = prediction.time_seconds
        lat_f, lon_f = GeospatialMetrics.__fill_anomaly_coordinates(
            time_s, prediction.lat, prediction.lon, prediction.anomaly_mask
        )
        origin_lat = float(lat_f[0])
        origin_lon = float(lon_f[0])
        x_m, y_m = _filled_to_local_xy(lat_f, lon_f, origin_lat, origin_lon)
        return _path_length_xy(x_m, y_m)

    @staticmethod
    def __fill_anomaly_coordinates(
            time_s: np.ndarray,
            lat: np.ndarray,
            lon: np.ndarray,
            is_anomaly: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Строит траекторию с интерполяцией в точках аномалии между штатными соседями.

        В штатных точках координаты сохраняются. Для аномалии без пары соседей остаются исходные координаты.
        Args:
            time_s: массив времени
            lat: массив широт
            lon: массив долгот
            is_anomaly: массив булевых значений, указывающих на аномалии

        Returns:
            tuple[np.ndarray, np.ndarray]: заполненные массивы широт и долгот
        """
        lat_out = np.asarray(lat, dtype=np.float64).copy()
        lon_out = np.asarray(lon, dtype=np.float64).copy()
        anomaly = np.asarray(is_anomaly, dtype=bool)
        n = lat_out.size
        normal = ~anomaly

        for index in range(n):
            if not anomaly[index]:
                continue
            left = np.flatnonzero(normal[:index])
            right = np.flatnonzero(normal[index + 1:])
            if left.size == 0 or right.size == 0:
                continue
            j = int(left[-1])
            k = int(right[0]) + index + 1
            dt = time_s[k] - time_s[j]
            if dt <= 0.0:
                continue
            alpha = (time_s[index] - time_s[j]) / dt
            lat_out[index] = lat_out[j] + alpha * (lat_out[k] - lat_out[j])
            lon_left = lon_out[j]
            delta_lon = (lon_out[k] - lon_left + 180.0) % 360.0 - 180.0
            lon_out[index] = (lon_left + alpha * delta_lon + 180.0) % 360.0 - 180.0

        return lat_out, lon_out


if __name__ == "__main__":
    """21 точка: петля на индексах 8–12, задержка детекции и одно ложное срабатывание."""
    count = 21
    time = np.arange(count, dtype=np.float64)
    latitude = 43.04 + 0.001 * time
    longitude = 131.87 + 0.001 * time
    labels = np.ones(count, dtype=np.float64)
    labels[8:13] = 0.0
    latitude = latitude.copy()
    longitude = longitude.copy()
    latitude[8:13] += np.array([0.02, 0.05, 0.08, 0.05, 0.02])
    longitude[8:13] += np.array([0.01, 0.04, 0.01, -0.03, -0.01])
    predicted = labels.copy()
    predicted[8:10] = 1.0
    predicted[15] = 0.0
    ground_truth = TrajectorySeries(time, longitude, latitude, labels)
    prediction = TrajectorySeries(time, longitude.copy(), latitude.copy(), predicted)
    metrics = GeospatialMetrics.compute(prediction, ground_truth)
    print("Геопространственные метрики (демо-трек):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2f}" if math.isfinite(value) else f"  {name}: —")
    plot_track_maps(prediction, ground_truth)
    plt.show()
