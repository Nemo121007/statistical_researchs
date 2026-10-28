"""Точечные метрики классификации (Precision, Recall, F1, MCC)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.evaluation.series import TrajectorySeries


class ClassificationMetrics:
    """Метрики бинарной классификации: аномалия — положительный класс (метка 0)."""

    @staticmethod
    def compute(prediction: TrajectorySeries, ground_truth: TrajectorySeries) -> Dict[str, float]:
        """
        Считает precision, recall, F1 и MCC по точкам ряда.
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Dict[str, float]: словарь с метриками.
        """
        precision, recall, f1, mcc = ClassificationMetrics.point_scores(prediction, ground_truth)
        return {"precision": precision, "recall": recall, "f1": f1, "mcc": mcc}

    @staticmethod
    def confusion_counts(
        prediction: TrajectorySeries,
        ground_truth: TrajectorySeries,
    ) -> Tuple[float, float, float, float]:
        """
        Возвращает (TP, FP, FN, TN); положительный класс — аномалия (label == 0).
        Args:
            prediction: Траектория, предсказанная моделью
            ground_truth: Истинная траектория

        Returns:
            Tuple[float, float, float, float]: (TP, FP, FN, TN).
        """
        ground_truth.assert_same_index(prediction)
        pred_anomaly = prediction.anomaly_mask
        gt_anomaly = ground_truth.anomaly_mask
        true_positive = float(np.sum(gt_anomaly & pred_anomaly))
        false_positive = float(np.sum(~gt_anomaly & pred_anomaly))
        false_negative = float(np.sum(gt_anomaly & ~pred_anomaly))
        true_negative = float(np.sum(~gt_anomaly & ~pred_anomaly))
        return true_positive, false_positive, false_negative, true_negative

    @staticmethod
    def mcc(
        prediction: TrajectorySeries,
        ground_truth: TrajectorySeries,
    ) -> float:
        """
        Коэффициент корреляции Мэттьюса по точкам ряда.

        Args: 
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            float: коэффициент корреляции Мэттьюса.
        """
        counts = ClassificationMetrics.confusion_counts(prediction, ground_truth)
        return ClassificationMetrics.__mcc_from_counts(*counts)

    @staticmethod
    def point_scores(
        prediction: TrajectorySeries,
        ground_truth: TrajectorySeries,
    ) -> Tuple[float, float, float, float]:
        """
        Возвращает (precision, recall, F1, MCC).
        Args:
            prediction: траектория с предсказаниями модели.
            ground_truth: эталонная траектория.

        Returns:
            Tuple[float, float, float, float]: (precision, recall, F1, MCC).
        """
        true_positive, false_positive, false_negative, true_negative = (
            ClassificationMetrics.confusion_counts(prediction, ground_truth)
        )
        precision = ClassificationMetrics.__ratio(true_positive, true_positive + false_positive)
        recall = ClassificationMetrics.__ratio(true_positive, true_positive + false_negative)
        f1 = ClassificationMetrics.__ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative)
        mcc = ClassificationMetrics.__mcc_from_counts(
            true_positive, false_positive, false_negative, true_negative
        )
        return precision, recall, f1, mcc

    @staticmethod
    def __ratio(numerator: float, denominator: float) -> float:
        """
        Вычисляет отношение числителя к знаменателю.

        Args:
            numerator: числитель.
            denominator: знаменатель.

        Returns:
            float: отношение числителя к знаменателю.
        """
        if denominator <= 0.0:
            return float("nan")
        return numerator / denominator

    @staticmethod
    def __mcc_from_counts(
            true_positive: float,
            false_positive: float,
            false_negative: float,
            true_negative: float,
    ) -> float:
        """
        Вычисляет коэффициент корреляции Мэттьюса по точкам ряда.

        Args:
            true_positive: true positive.
            false_positive: false positive.
            false_negative: false negative.
            true_negative: true negative.

        Returns:
            float: коэффициент корреляции Мэттьюса.
        """
        denominator = math.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        if denominator <= 0.0:
            return 0.0
        return (true_positive * true_negative - false_positive * false_negative) / denominator


if __name__ == "__main__":
    # 21 точка: петля на индексах 8–12, задержка детекции и одно ложное срабатывание.
    # app.viz импортируется здесь: на уровне модуля он замыкает цикл app.viz → app.evaluation → app.viz.
    from app.viz import plot_track_maps

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
    plot_track_maps(prediction, ground_truth)
    plt.show()
