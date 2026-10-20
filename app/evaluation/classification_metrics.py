"""Точечные метрики классификации (Precision, Recall, F1)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.evaluation.series import TrajectorySeries


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


class ClassificationMetrics:
    """Метрики бинарной классификации: аномалия — положительный класс (метка 0)."""

    @staticmethod
    def compute(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> Dict[str, float]:
        """
        Считает precision, recall и F1 по точкам ряда.
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Dict[str, float]: словарь с метриками.
        """
        precision, recall, f1 = ClassificationMetrics.point_scores(prediction, ground_truth)
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def point_scores(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> Tuple[float, float, float]:
        """
        Возвращает (precision, recall, F1).
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Tuple[float, float, float]: (precision, recall, F1).
        """
        ground_truth.assert_same_index(prediction)
        pred_anomaly = prediction.anomaly_mask
        gt_anomaly = ground_truth.anomaly_mask
        true_positive = float(np.sum(gt_anomaly & pred_anomaly))
        false_positive = float(np.sum(~gt_anomaly & pred_anomaly))
        false_negative = float(np.sum(gt_anomaly & ~pred_anomaly))
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative)
        return precision, recall, f1


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
    metrics = ClassificationMetrics.compute(prediction, ground_truth)
    print("Классификационные метрики (демо-трек):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if math.isfinite(value) else f"  {name}: —")
    ClassificationMetrics.plot_track(prediction, ground_truth)
    plt.show()
