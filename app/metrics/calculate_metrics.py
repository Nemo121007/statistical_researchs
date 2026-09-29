from typing import Tuple

import numpy as np


class CalculateMetrics:
    @staticmethod
    def calculate_point_precision(
            prediction: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            ground_truth: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[float, float, float]:
        """
        Вычисляет precision, recall и f1-score
        Args:
            prediction: кортеж предсказанных значений: время, долгота, широта, метка типа точки
            ground_truth: кортеж истинных значений: время, долгота, широта, метка типа точки

        Returns:
            precision: точность предсказаний
            recall: полнота предсказаний
            f1_score: F1-мера предсказаний
        """
        return 0, 0, 0
