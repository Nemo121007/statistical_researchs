from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.help_scripts.calculator_distances_length_large_circle import (
    CalculatorDistancesLengthLargeCircle,
)


class CalculatingStatistics:
    """
    Класс для расчета статистических метрик качества (Recall)
    сравнения экспериментальных данных с контрольными.
    """

    @staticmethod
    def calculate_statistics(
        experimental_df: pd.DataFrame,
        control_df: pd.DataFrame,
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
        exp_pts, ctrl_pts, point_recall = CalculatingStatistics._get_recall(df)
        class_metrics = CalculatingStatistics.get_validation_metrics(df)
        ctrl_len, exp_len, matched_len = CalculatingStatistics._get_recall_length(df)
        fp_len = CalculatingStatistics._get_false_positive_length(df)
        invalid_fp_len = CalculatingStatistics._get_invalid_false_positive_length(df)

        report_lines = [
            "=" * 50,
            "REPORT: Route Validation Statistics",
            "=" * 50,
            "\n[1] Point Metrics (Counts)",
            f"  Control Valid Points: {ctrl_pts}",
            f"  Matched Experimental Points (TP):  {exp_pts}",
            f"  Point Recall:         {point_recall:.4f}",
            "\n[2] Classification Metrics (Confusion Matrix)",
            f"  True Positives (TP):  {class_metrics['TP']}",
            f"  False Positives (FP): {class_metrics['FP']}",
            f"  False Negatives (FN): {class_metrics['FN']}",
            f"  True Negatives (TN):  {class_metrics['TN']}",
            "-" * 50,
            f"  Accuracy:             {class_metrics['Accuracy']:.4f}",
            f"  Precision:            {class_metrics['Precision']:.4f}",
            f"  Recall:               {class_metrics['Recall']:.4f}",
            f"  F1 Score:             {class_metrics['F1']:.4f}",
            "\n[3] Length Metrics (Meters)",
            f"  Control Path Length:    {ctrl_len:.2f} m",
            f"  Experimental Path Len:  {exp_len:.2f} m",
            f"  Matched Path Length:    {matched_len:.2f} m",
            f"  False Positive Length:  {fp_len:.2f} m",
            "-" * 50,
            f"  Length Recall:          {matched_len / ctrl_len if ctrl_len > 0 else 0.0:.4f}",
            f"  Invalid Control Length (FP): {invalid_fp_len:.2f} m",
            "=" * 50,
        ]

        full_report = "\n".join(report_lines)
        print(full_report)

        if output_path_log:
            output_path_log.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_log, "w", encoding="utf-8") as f:
                f.write(full_report)

        return full_report

    @staticmethod
    def get_correct_classify_points(
        experimental_df: pd.DataFrame, control_df: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """
        Возвращает список DataFrame, соответствующих верно классифицированным интервалам (True Positives).

        Интервал считается верно классифицированным, если точка в эксперименте помечена как валидная (1),
        в контроле она тоже валидна (1), и их координаты полностью совпадают.

        Args:
            experimental_df: DataFrame с экспериментальными данными.
            control_df: DataFrame с контрольными данными.

        Returns:
            List[pd.DataFrame]: Список интервалов, которые являются True Positives.
        """
        # Слияние данных
        df = CalculatingStatistics._merge_dataframes(experimental_df, control_df)

        # Формирование условия True Positive
        # Эксперимент = 1, Контроль = 1, Координаты совпадают
        tp_mask = (
            (df["exp_validate_point"] == 1)
            & (df["ctrl_validate_point"] == 1)
            & (df["exp_lat"] == df["ctrl_lat"])
            & (df["exp_lon"] == df["ctrl_lon"])
        )

        # Выделение точек
        tp_points = df.loc[
            tp_mask, ["exp_lat", "exp_lon", "exp_validate_point", "exp_time"]
        ].copy()

        # Приведение к формату, ожидаемому _extend_intervals (lat, lon, time, validate_point)
        tp_points.rename(
            columns={
                "exp_lat": "lat",
                "exp_lon": "lon",
                "exp_time": "time",
                "exp_validate_point": "validate_point",
            },
            inplace=True,
        )

        return CalculatingStatistics._extend_intervals(tp_points)

    @staticmethod
    def get_incorrect_classify_points(
        experimental_df: pd.DataFrame, control_df: pd.DataFrame
    ) -> List[pd.DataFrame]:
        """
        Возвращает список DataFrame, соответствующих неверно классифицированным интервалам (False Positives).

        Интервал считается неверно классифицированным, если точка в эксперименте помечена как валидная (1),
        но при этом:
        - Либо в контроле она невалидна (0).
        - Либо координаты не совпадают с контрольными.

        Примечание: Функция возвращает именно "лишние" или "ошибочные" интервалы эксперимента (False Positives).
        Пропущенные точки (False Negatives) не возвращаются, так как они отсутствуют в валидной части эксперимента.

        Args:
            experimental_df: DataFrame с экспериментальными данными.
            control_df: DataFrame с контрольными данными.

        Returns:
            List[pd.DataFrame]: Список интервалов, которые являются False Positives.
        """
        # Слияние данных
        df = CalculatingStatistics._merge_dataframes(experimental_df, control_df)

        # Формирование условия True Positive (для последующего отрицания)
        is_tp = (
            (df["exp_validate_point"] == 1)
            & (df["ctrl_validate_point"] == 1)
            & (df["exp_lat"] == df["ctrl_lat"])
            & (df["exp_lon"] == df["ctrl_lon"])
        )

        # Условие False Positive
        # Эксперимент считает точку валидной, но это НЕ True Positive
        fp_mask = (df["exp_validate_point"] == 1) & (~is_tp)

        # Выделение точек
        fp_points = df.loc[
            fp_mask, ["exp_lat", "exp_lon", "exp_validate_point", "exp_time"]
        ].copy()

        # Приведение к формату, ожидаемому _extend_intervals
        fp_points.rename(
            columns={
                "exp_lat": "lat",
                "exp_lon": "lon",
                "exp_time": "time",
                "exp_validate_point": "validate_point",
            },
            inplace=True,
        )

        return CalculatingStatistics._extend_intervals(fp_points)

    @staticmethod
    def _merge_dataframes(
        experimental_df: pd.DataFrame, control_df: pd.DataFrame
    ) -> pd.DataFrame:
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

        return pd.concat(
            [exp_part.reset_index(drop=True), ctrl_part.reset_index(drop=True)], axis=1
        )

    @staticmethod
    def _get_recall(df: pd.DataFrame) -> Tuple[int, int, float]:
        """
        Рассчитывает точечную полноту (Point Recall).
        Принимает уже смерженный DataFrame.

        Args:
            df: DataFrame, содержащий колонки с префиксами exp_ и ctrl_.
        Returns:
            experimental_count: Количество совпавших валидных точек в эксперименте.
            control_count: Количество валидных точек в контроле.
            recall_count: Доля совпавших точек от общего количества валидных точек в контроле (Recall).
        """
        control_count = df[df["ctrl_validate_point"] == 1].shape[0]

        match_condition = (
            (df["ctrl_validate_point"] == 1)
            & (df["exp_validate_point"] == 1)
            & (df["exp_lat"] == df["ctrl_lat"])
            & (df["exp_lon"] == df["ctrl_lon"])
        )
        experimental_count = df[match_condition].shape[0]

        recall_count = experimental_count / control_count if control_count > 0 else 0.0
        return experimental_count, control_count, recall_count

    @staticmethod
    def _get_recall_length(df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Рассчитывает метрики протяженности маршрутов.
        Принимает уже смерженный DataFrame.
        Извлекает необходимые колонки для расчета длин.

        Args:
            df: DataFrame, содержащий колонки с префиксами exp_ и ctrl_.
        Returns:
            control_length: Общая длина контрольного маршрута (метры).
            experimental_length: Общая длина экспериментального маршрута (метры).
            matched_length: Длина совпавших сегментов между экспериментом и контролем (метры).
        """
        # Подготовка данных контрольного маршрута (извлекаем из общего df)
        ctrl_path = df[
            ["ctrl_lat", "ctrl_lon", "ctrl_validate_point", "ctrl_time"]
        ].rename(
            columns={
                "ctrl_lat": "lat",
                "ctrl_lon": "lon",
                "ctrl_validate_point": "validate_point",
                "ctrl_time": "time",
            }
        )
        control_list_df = CalculatingStatistics._extend_intervals(ctrl_path)
        control_length = np.nansum(
            [
                np.nansum(
                    CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                        interval["lat"].values, interval["lon"].values
                    )
                )
                for interval in control_list_df
            ]
        )

        # Подготовка данных экспериментального маршрута
        exp_path = df[["exp_lat", "exp_lon", "exp_validate_point", "exp_time"]].rename(
            columns={
                "exp_lat": "lat",
                "exp_lon": "lon",
                "exp_validate_point": "validate_point",
                "exp_time": "time",
            }
        )
        exp_list_df = CalculatingStatistics._extend_intervals(exp_path)
        exp_length = np.nansum(
            [
                np.nansum(
                    CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                        interval["lat"].values, interval["lon"].values
                    )
                )
                for interval in exp_list_df
            ]
        )

        # Расчет совпавших сегментов
        match_condition = (
            (df["ctrl_validate_point"] == 1)
            & (df["exp_validate_point"] == 1)
            & (df["exp_lat"] == df["ctrl_lat"])
            & (df["exp_lon"] == df["ctrl_lon"])
        )

        # Извлекаем совпавшие точки, используя экспериментальные координаты и время
        matched_path = df[match_condition][
            ["exp_lat", "exp_lon", "exp_validate_point", "exp_time"]
        ].rename(
            columns={
                "exp_lat": "lat",
                "exp_lon": "lon",
                "exp_validate_point": "validate_point",
                "exp_time": "time",
            }
        )

        concat_exp_list_df = CalculatingStatistics._extend_intervals(matched_path)
        concat_exp_length = np.nansum(
            [
                np.nansum(
                    CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                        interval["lat"].values, interval["lon"].values
                    )
                )
                for interval in concat_exp_list_df
            ]
        )

        return float(control_length), float(exp_length), float(concat_exp_length)

    @staticmethod
    def _extend_intervals(df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Разбивает DataFrame на список непрерывных временных интервалов.

        Args:
            df: DataFrame с колонкой 'time' и 'validate_point'.
        Returns:
            Список DataFrame, каждый из которых представляет непрерывный временной интервал с validate_point == 1.
        """
        if df.empty:
            return []

        clean_df = df[df["validate_point"] == 1].copy()
        if clean_df.empty:
            return []

        time_diff = clean_df["time"].diff()
        split_mask = time_diff > 10
        group_ids = split_mask.cumsum()
        return [group for _, group in clean_df.groupby(group_ids) if len(group) > 1]

    @staticmethod
    def _get_false_positive_length(df: pd.DataFrame) -> float:
        """
        Вычисляет длину ошибочно классифицированных интервалов.
        Принимает уже смерженный DataFrame.

        Args:
            df: DataFrame, содержащий колонки с префиксами exp_ и ctrl_.
        Returns:
            false_positive_length: Общая длина сегментов, которые были ошибочно классифицированы
            как валидные в эксперименте, но не совпали с контролем (метры).
        """
        is_exp_valid = df["exp_validate_point"] == 1
        is_true_positive = (
            (df["ctrl_validate_point"] == 1)
            & (df["exp_lat"] == df["ctrl_lat"])
            & (df["exp_lon"] == df["ctrl_lon"])
        )

        false_positive_condition = is_exp_valid & (~is_true_positive)
        fp_df = df[false_positive_condition].copy()

        if fp_df.empty:
            return 0.0

        # Извлекаем нужные колонки для расчета длины
        fp_path = fp_df[
            ["exp_lat", "exp_lon", "exp_validate_point", "exp_time"]
        ].rename(
            columns={
                "exp_lat": "lat",
                "exp_lon": "lon",
                "exp_validate_point": "validate_point",
                "exp_time": "time",
            }
        )

        fp_intervals = CalculatingStatistics._extend_intervals(fp_path)
        fp_length = np.nansum(
            [
                np.nansum(
                    CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                        interval["lat"].values, interval["lon"].values
                    )
                )
                for interval in fp_intervals
            ]
        )

        return float(fp_length)

    @staticmethod
    def get_validation_metrics(df: pd.DataFrame) -> dict:
        """
        Строит Confusion Matrix.
        Принимает уже смерженный DataFrame.

        Args:
            df: DataFrame, содержащий колонки с префиксами exp_ и ctrl_.
        Returns:
            Словарь с метриками: TP, FP, FN, TN, Accuracy, Precision, Recall, F1.
        """
        # Базовые условия валидности
        is_exp_pos = df["exp_validate_point"] == 1
        is_ctrl_pos = df["ctrl_validate_point"] == 1

        # Считаем TP напрямую, инлайним проверку координат
        tp = np.nansum(
            is_exp_pos
            & is_ctrl_pos
            & (df["exp_lat"] == df["ctrl_lat"])
            & (df["exp_lon"] == df["ctrl_lon"])
        )

        # FP и FN выводим через разницу множеств (без лишних масок)
        # FP = Все предсказанные положительные (exp_pos) минус истинные (tp)
        fp = np.nansum(is_exp_pos) - tp
        # FN = Все реальные положительные (ctrl_pos) минус истинные (tp)
        fn = np.nansum(is_ctrl_pos) - tp

        # TN вычисляем как остаток
        tn = len(df) - tp - fp - fn

        # Промежуточные метрики для F1 (используются дважды)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Accuracy": (tp + tn) / len(df) if len(df) > 0 else 0.0,
            "Precision": precision,
            "Recall": recall,
            "F1": (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            ),
        }

    @staticmethod
    def _get_invalid_false_positive_length(df: pd.DataFrame) -> float:
        condition = (
            (df["ctrl_validate_point"] == 0) &
            (df["exp_validate_point"] == 1)
        )

        invalid_fp_df = df[condition]

        if invalid_fp_df.empty:
            return 0.0

        path = invalid_fp_df[
            ["exp_lat", "exp_lon", "exp_validate_point", "exp_time"]
        ].rename(
            columns={
                "exp_lat": "lat",
                "exp_lon": "lon",
                "exp_validate_point": "validate_point",
                "exp_time": "time",
            }
        )

        intervals = CalculatingStatistics._extend_intervals(path)

        length = np.nansum([
            np.nansum(
                CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(
                    interval["lat"].values,
                    interval["lon"].values
                )
            )
            for interval in intervals
        ])

        return float(length)


if __name__ == "__main__":
    # # Пример использования
    # data = {
    #     'lat': [55.75, 55.76, 55.77, 55.78, 55.79],
    #     'lon': [37.60, 37.61, 37.62, 37.63, 37.64],
    #     'time': [1, 2, 3, 20, 21],
    #     'validate_point': [1, 1, 1, 1, 1]
    # }
    # df_control = pd.DataFrame(data)
    #
    # data_exp = {
    #     'lat': [55.75, 55.76, 55.77, 55.78, 55.79],
    #     'lon': [37.60, 37.61, 37.62, 37.63, 37.64],
    #     'time': [1, 2, 3, 20, 21],
    #     'validate_point': [1, 1, 0, 1, 0]
    # }
    # df_exp = pd.DataFrame(data_exp)

    path = (
        Path(__file__).parent.parent.parent / "data" / "post_processing" / "example.csv"
    )
    df_exp = pd.read_csv(path)
    df_control = pd.read_csv(path)

    CalculatingStatistics.calculate_statistics(
        experimental_df=df_exp,
        control_df=df_control,
    )
