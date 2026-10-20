"""Временные метрики детекции аномалий."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.evaluation.series import TrajectorySeries
from app.viz import plot_track_maps


def _validate_time_axis(ground_truth: TrajectorySeries, prediction: TrajectorySeries) -> np.ndarray:
    """
    Возвращает время эталонной траектории в секундах после проверок для delay-метрик.
    Args:
        ground_truth: эталонная траектория.
        prediction: траектория с предсказаниями модели.

    Returns:
        np.ndarray: массив времени в секундах для эталонной траектории.
    Raises:
        ValueError: если траектории имеют несовпадающие индексы или некорректные значения времени.
    """
    ground_truth.assert_same_index(prediction)
    time_s = ground_truth.time_seconds
    if np.any(~np.isfinite(prediction.time_seconds)):
        raise ValueError("Время prediction должно быть конечным.")
    if time_s.size >= 2 and np.any(np.diff(time_s) <= 0.0):
        raise ValueError("Время ground truth должно строго возрастать.")
    return time_s


def _true_runs(mask: np.ndarray) -> list[Tuple[int, int]]:
    """
    Находит непрерывные участки, где mask истинен.

    Args:
        mask: массив булевых значений, где True указывает на наличие аномалии.

    Raises:
        ValueError: если mask не является массивом numpy.

    Returns:
        list[Tuple[int, int]]: список кортежей (start, end) для каждого непрерывного участка True.
    """
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


class TemporalMetrics:
    """Задержка обнаружения аномальных эпизодов (Detection Delay)."""

    @staticmethod
    def compute(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> Dict[str, float]:
        """Считает задержки по эпизодам и агрегаты.

        Необнаруженный эпизод получает задержку, равную его длительности (time_end - time_start).

        Returns:
            Ключи delay_mean, delay_median, delay_p95, delay_max.
        """
        values = np.asarray(
            [delay for delay, _, _, _ in TemporalMetrics.per_event_delays(prediction, ground_truth)],
            dtype=np.float64,
        )
        if values.size == 0:
            return {
                "delay_mean": float("nan"),
                "delay_median": float("nan"),
                "delay_p95": float("nan"),
                "delay_max": float("nan"),
            }
        return {
            "delay_mean": float(np.mean(values)),
            "delay_median": float(np.median(values)),
            "delay_p95": float(np.percentile(values, 95)),
            "delay_max": float(np.max(values)),
        }

    @staticmethod
    def per_event_delays(
        prediction: TrajectorySeries, ground_truth: TrajectorySeries
    ) -> List[Tuple[float, int, int, bool]]:
        """
        Задержка по каждому истинному эпизоду: (delay, start, end, detected).
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Список кортежей (delay, start, end, detected) для каждого истинного эпизода.
        """
        time_s = _validate_time_axis(ground_truth, prediction)
        pred_anomaly = prediction.anomaly_mask
        gt_anomaly = ground_truth.anomaly_mask
        result: List[Tuple[float, int, int, bool]] = []
        for start, end in _true_runs(gt_anomaly):
            detected_slice = pred_anomaly[start : end + 1]
            if np.any(detected_slice):
                detect_at = start + int(np.argmax(detected_slice))
                delay = float(time_s[detect_at] - time_s[start])
                result.append((delay, start, end, True))
            else:
                delay = float(time_s[end] - time_s[start])
                result.append((delay, start, end, False))
        return result


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
    metrics = TemporalMetrics.compute(prediction, ground_truth)
    print("Временные метрики (демо-трек):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if math.isfinite(value) else f"  {name}: —")
    plot_track_maps(prediction, ground_truth)
    plt.show()
