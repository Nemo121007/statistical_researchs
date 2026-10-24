"""Оркестратор: все группы метрик для пары prediction / ground truth."""

from __future__ import annotations

from typing import Dict

import numpy as np

from app.evaluation.classification_metrics import ClassificationMetrics
from app.evaluation.geospatial_metrics import GeospatialMetrics
from app.evaluation.series import TrajectorySeries
from app.evaluation.temporal_metrics import TemporalMetrics


class TrajectoryMetrics:
    """Собирает классификационные, временные и геопространственные метрики."""

    @staticmethod
    def compute_all(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> Dict[str, float]:
        """
        Вычисляет все метрики для заданных предсказаний и эталонных данных.

        Args:
            prediction: Ряд prediction.
            ground_truth: Эталонный ряд.

        Returns:
            Словарь с метриками.
        """
        metrics: Dict[str, float] = {}
        metrics.update(ClassificationMetrics.compute(prediction, ground_truth))
        metrics.update(TemporalMetrics.compute(prediction, ground_truth))
        metrics.update(GeospatialMetrics.compute(prediction, ground_truth))
        return metrics


if __name__ == "__main__":
    """21 точка: петля на индексах 8–12, задержка детекции и одно ложное срабатывание."""
    import math

    import matplotlib.pyplot as plt

    from app.viz import plot_track_maps

    _REPORT_ROWS = (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("mcc", "MCC"),
        ("delay_mean", "Delay mean, с"),
        ("delay_median", "Delay median, с"),
        ("delay_p95", "Delay p95, с"),
        ("delay_max", "Delay max, с"),
        ("rmse_m", "RMSE, м"),
        ("max_deviation_m", "Max deviation, м"),
        ("distance_loss_m", "Distance loss, м"),
    )

    def _format_metric_value(key: str, value: float) -> str:
        if not math.isfinite(value):
            return "—"
        if key in {"precision", "recall", "f1", "mcc"} or key.startswith("delay_"):
            return f"{value:.4f}"
        return f"{value:.2f}"

    def _print_metrics_table(metrics: Dict[str, float], title: str = "", prefix: str = "  ") -> None:
        if title:
            print(title)
        for key, label in _REPORT_ROWS:
            if key not in metrics:
                continue
            print(f"{prefix}{label}: {_format_metric_value(key, metrics[key])}")

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
    print(
        "Тестовый трек: 21 точка. Аномалия — петля на индексах 8–12. "
        "Модель обнаруживает её с индекса 10 и даёт ложное срабатывание на индексе 15."
    )
    delayed_metrics = TrajectoryMetrics.compute_all(prediction, ground_truth)
    _print_metrics_table(delayed_metrics, title="Метрики с задержкой детекции:")

    perfect_prediction = TrajectorySeries(
        ground_truth.time,
        ground_truth.lon,
        ground_truth.lat,
        ground_truth.label.copy(),
    )
    perfect_metrics = TrajectoryMetrics.compute_all(perfect_prediction, ground_truth)
    _print_metrics_table(
        perfect_metrics,
        title="Предсказанные метки совпадают с ground truth:",
    )

    plot_track_maps(
        prediction,
        ground_truth,
        title="Тестовый трек (модель с задержкой детекции)",
    )
    plt.show()
