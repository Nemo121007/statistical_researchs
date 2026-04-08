import time
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from app.help_scripts.calculator_distances_length_large_circle import \
    CalculatorDistancesLengthLargeCircle


class CalculatingStatistics:
    """
    Класс для расчета статистических метрик качества (Recall)
    сравнения экспериментальных данных с контрольными.
    """

    @staticmethod
    def calculate_statistics(
        experimental_df: pd.DataFrame,
        control_df: pd.DataFrame,
        time_working: np.ndarray,
        output_path_log: Path = None,
    ) -> str:
        """
        Основной метод для запуска расчета статистики.
        Выполняет слияние данных один раз и передает их в методы расчета.
        """
        if "time" not in experimental_df.columns or "time" not in control_df.columns:
            raise ValueError("DataFrames must contain 'time' column.")

        df = CalculatingStatistics._merge_dataframes(experimental_df, control_df)

        # Расчеты
        stat_point = CalculatingStatistics._get_point_statistic(df)
        stat_edge = CalculatingStatistics._get_edge_statistic(df)
        lengths = CalculatingStatistics._get_statistic_abs_length(df)

        report_lines = [
            "=" * 55,
            "REPORT: Route Validation Statistics".center(55),
            "=" * 55,
            "\n[1] Point Metrics (Counts)",
            f"{'  Count valid point in control_df:':<40} {stat_point['count_valid_point_control_df']:>14}",
            f"{'  Count valid point in experimental_df:':<40} {stat_point['count_valid_point_experimental_df']:>14}",
            f"{'  Count valid point in merge_df:':<40} {stat_point['count_valid_point_merge_df']:>14}",
            "-" * 55,
            f"{'  True Positives (TP):':<40} {stat_point['TP']:>14}",
            f"{'  False Positives (FP):':<40} {stat_point['FP']:>14}",
            f"{'  False Negatives (FN):':<40} {stat_point['FN']:>14}",
            f"{'  True Negatives (TN):':<40} {stat_point['TN']:>14}",
            f"{'  In_water:':<40} {stat_point['in_water']:>14}",
            "-" * 55,
            f"{'  Accuracy:':<40} {stat_point['accuracy']:>14.4f}",
            f"{'  Precision:':<40} {stat_point['precision']:>14.4f}",
            f"{'  Recall:':<40} {stat_point['recall']:>14.4f}",
            f"{'  F1 Score:':<40} {stat_point['f_score']:>14.4f}",
            "\n[2] Length Metrics (Meters)",
            f"{'  Count valid edges in control_df:':<40} {stat_edge['count_valid_edge_control_df']:>14}",
            f"{'  Count valid edges in experimental_df:':<40} {stat_edge['count_valid_edge_experimental_df']:>14}",
            f"{'  Count valid edges in merge_df:':<40} {stat_edge['count_valid_edge_merge_df']:>14}",
            f"{'  Length valid edges in control_df:':<40} {lengths['ctrl_distance']:>14.4f}",
            f"{'  Length valid edges in experimental_df:':<40} {lengths['exp_distance']:>14.4f}",
            "-" * 55,
            f"{'  True Positives (TP):':<40} {stat_edge['TP']:>14}",
            f"{'  False Positives (FP):':<40} {stat_edge['FP']:>14}",
            f"{'  False Negatives (FN):':<40} {stat_edge['FN']:>14}",
            f"{'  True Negatives (TN):':<40} {stat_edge['TN']:>14}",
            f"{'  In_water:':<40} {stat_edge['in_water']:>14}",
            "-" * 55,
            f"{'  Accuracy:':<40} {stat_edge['accuracy']:>14.4f}",
            f"{'  Precision:':<40} {stat_edge['precision']:>14.4f}",
            f"{'  Recall:':<40} {stat_edge['recall']:>14.4f}",
            f"{'  F1 Score:':<40} {stat_edge['f_score']:>14.4f}",
            "-" * 55,
            f"{'  Length True Positives (TP):':<40} {stat_edge['length_TP']:>14.4f}",
            f"{'  Length False Positives (FP):':<40} {stat_edge['length_FP']:>14.4f}",
            f"{'  Length False Negatives (FN):':<40} {stat_edge['length_FN']:>14.4f}",
            f"{'  Length True Negatives (TN):':<40} {stat_edge['length_TN']:>14.4f}",
            "-" * 55,
            f"{'  Avg time working:':<40} {np.average(time_working[5:]):>14.4f}",
            "=" * 55,
        ]

        full_report = "\n".join(report_lines)
        print(full_report)

        if output_path_log:
            output_path_log.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_log, "w", encoding="utf-8") as f:
                f.write(full_report)

        return full_report

    @staticmethod
    def _get_point_statistic(merge_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        if "ctrl_validate_point" not in merge_df.columns:
            raise ValueError("merge_df must contain 'ctrl_validate_point' column.")
        if "exp_validate_point" not in merge_df.columns:
            raise ValueError("merge_df must contain 'exp_validate_point' column.")
        # Приводим к булевому виду, NaN считаем как False
        ctrl = merge_df["ctrl_validate_point"] == 1
        exp = merge_df["exp_validate_point"] == 1

        # Количества валидных точек
        count_valid_point_control_df = int(np.nansum(ctrl))
        count_valid_point_experimental_df = int(np.nansum(exp))
        count_valid_point_merge_df = int(np.nansum(ctrl & exp))
        in_water = merge_df["in_water"].fillna(0).astype(bool)
        count_in_water = int((exp & in_water).sum())

        # Матрица ошибок
        TP = int(np.nansum(ctrl & exp))
        FP = int(np.nansum(~ctrl & exp))
        FN = int(np.nansum(ctrl & ~exp))
        TN = int(np.nansum(~ctrl & ~exp))

        total = TP + FP + FN + TN

        # Метрики с защитой от деления на ноль
        accuracy = (TP + TN) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "count_valid_point_control_df": count_valid_point_control_df,
            "count_valid_point_experimental_df": count_valid_point_experimental_df,
            "count_valid_point_merge_df": count_valid_point_merge_df,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "in_water": count_in_water,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
        }

    @staticmethod
    def _get_edge_statistic(merge_df: pd.DataFrame) -> Dict[str, float]:
        """
        Статистика по рёбрам.

        Логика merge-ребра берётся из вашего правила:
            ctrl_validate_point[i - 1] == 1 and exp_validate_point[i] == 1

        При этом:
        - control edges считаются как соседние точки внутри control-трека
        - experimental edges считаются как соседние точки внутри experimental-трека
        - попадание ребра в воду проверяется по двум точкам ребра
        """
        # pylint: disable=too-many-locals

        required_columns = {
            "ctrl_validate_point",
            "exp_validate_point",
            "ctrl_lat",
            "ctrl_lon",
            "exp_lat",
            "exp_lon",
            "in_water",
        }
        missing = required_columns - set(merge_df.columns)
        if missing:
            raise ValueError(f"merge_df не содержит атрибутов: {sorted(missing)}")

        n = len(merge_df)
        if n < 2:
            return {
                "count_valid_edge_control_df": 0,
                "count_valid_edge_experimental_df": 0,
                "count_valid_edge_merge_df": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f_score": 0.0,
                "in_water": 0,
                "length_TP": 0.0,
                "length_FP": 0.0,
                "length_FN": 0.0,
                "length_TN": 0.0,
            }

        # Булевы маски точек
        ctrl_valid = merge_df["ctrl_validate_point"].eq(1).to_numpy()
        exp_valid = merge_df["exp_validate_point"].eq(1).to_numpy()

        in_water = merge_df["in_water"].fillna(0).astype(bool).to_numpy()

        # Рёбра внутри каждого трека
        ctrl_edge_valid = ctrl_valid[:-1] & ctrl_valid[1:]
        exp_edge_valid = exp_valid[:-1] & exp_valid[1:]

        # merge рёбер:
        merge_edge_valid = ctrl_edge_valid & exp_edge_valid

        # Ребро в воде, если обе его точки в воде
        edge_in_water = in_water[:-1] & in_water[1:]

        # Counts
        count_valid_edge_control_df = int(ctrl_edge_valid.sum())
        count_valid_edge_experimental_df = int(exp_edge_valid.sum())
        count_valid_edge_merge_df = int(merge_edge_valid.sum())
        count_in_water = int((merge_edge_valid & edge_in_water).sum())

        # Матрица ошибок по рёбрам:
        # сравниваем последовательные рёбра внутри control и experimental треков
        TP_mask = merge_edge_valid
        FP_mask = (~ctrl_edge_valid) & exp_edge_valid
        FN_mask = ctrl_edge_valid & (~exp_edge_valid)
        TN_mask = (~ctrl_edge_valid) & (~exp_edge_valid)

        TP = int(TP_mask.sum())
        FP = int(FP_mask.sum())
        FN = int(FN_mask.sum())
        TN = int(TN_mask.sum())

        total = TP + FP + FN + TN

        accuracy = (TP + TN) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        edge_lengths = CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
            merge_df["ctrl_lat"].to_numpy(dtype=float),
            merge_df["ctrl_lon"].to_numpy(dtype=float),
        )

        # Длинна считается если оба конца ребра принадлежат к одному ребру
        length_TP = float(np.nansum(edge_lengths[TP_mask]))
        length_FP = float(np.nansum(edge_lengths[FP_mask]))
        length_FN = float(np.nansum(edge_lengths[FN_mask]))
        length_TN = float(np.nansum(edge_lengths[TN_mask]))

        return {
            "count_valid_edge_control_df": count_valid_edge_control_df,
            "count_valid_edge_experimental_df": count_valid_edge_experimental_df,
            "count_valid_edge_merge_df": count_valid_edge_merge_df,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "in_water": count_in_water,
            "length_TP": length_TP,
            "length_FP": length_FP,
            "length_FN": length_FN,
            "length_TN": length_TN,
        }

    @staticmethod
    def _get_statistic_abs_length(merge_df: pd.DataFrame) -> Dict[str, float]:
        mask = merge_df["ctrl_validate_point"] == 1
        ctrl_lat = (merge_df.loc[mask, "ctrl_lat"]).to_numpy(dtype=float)
        ctrl_lon = (merge_df.loc[mask, "ctrl_lon"]).to_numpy(dtype=float)
        if np.nansum(mask) >= 2:
            ctrl_distance = np.nansum(
                CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                    ctrl_lat,
                    ctrl_lon,
                )
            )
        else:
            ctrl_distance = np.nan

        mask = merge_df["exp_validate_point"] == 1
        exp_lat = (merge_df.loc[mask, "exp_lat"]).to_numpy(dtype=float)
        exp_lon = (merge_df.loc[mask, "exp_lon"]).to_numpy(dtype=float)
        if np.nansum(mask) >= 2:
            exp_distance = np.nansum(
                CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                    exp_lat,
                    exp_lon,
                )
            )
        else:
            exp_distance = np.nan
        return {
            "ctrl_distance": ctrl_distance,
            "exp_distance": exp_distance,
        }

    @staticmethod
    def _merge_dataframes(experimental_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
        """
        Объединяет экспериментальный и контрольный DataFrame в один.
        Добавляет префиксы exp_ и ctrl_ ко всем колонкам.
        Всегда включает 'time', так как она нужна для расчета интервалов.

        Args:
            experimental_df: DataFrame с экспериментальными данными.
            control_df: DataFrame с контрольными данными.
        Returns:
            Объединенный DataFrame с префиксами для каждой колонки.
        """
        if len(experimental_df) != len(control_df):
            raise ValueError(f"Длинны DF не совпадают: {len(experimental_df)} != {len(control_df)}")
        # Список колонок для слияния (time нужна для _extend_intervals)
        cols_to_merge = ["lat", "lon", "validate_point", "time"]

        exp_part = experimental_df[cols_to_merge].rename(
            columns={
                "lat": "exp_lat",
                "lon": "exp_lon",
                "validate_point": "exp_validate_point",
                "time": "exp_time",
            }
        )

        ctrl_part = control_df[cols_to_merge].rename(
            columns={
                "lat": "ctrl_lat",
                "lon": "ctrl_lon",
                "validate_point": "ctrl_validate_point",
                "time": "ctrl_time",
            }
        )
        ctrl_part["in_water"] = control_df["in_water"]

        return pd.concat([exp_part.reset_index(drop=True), ctrl_part.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    list_time = []
    path = Path(__file__).parent.parent.parent / "data" / "post_processing" / "example.csv"
    df_exp = pd.read_csv(path)
    df_control = pd.read_csv(path)

    for i in range(15):
        start_time = time.time()
        end_time = time.time()
        execution_time = end_time - start_time
        list_time.append(execution_time)

    CalculatingStatistics.calculate_statistics(
        experimental_df=df_exp,
        control_df=df_control,
        time_working=np.array(list_time),
    )
