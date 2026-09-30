"""Метрики качества детекции аномалий на траектории.

Каждый ряд передаётся кортежем из четырёх массивов одинаковой длины:
время, долгота, широта, метка. Метка 0 — аномалия, метка больше 0 — штатное
состояние. Для PR-AUC бинарной маски недостаточно: нужен отдельный anomaly score,
в котором большее значение означает более сильную аномальность.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

Series = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

_EARTH_RADIUS_M = 6_371_000.0
_RECALL_GRID_SIZE = 8192


def _to_seconds(time: np.ndarray) -> np.ndarray:
    """Приводит время к float. Для datetime64 единица измерения — секунды."""
    values = np.asarray(time)
    if values.size == 0:
        raise ValueError("Ряд не должен быть пустым.")
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.float64) / 1e9
    if values.dtype.kind in {"U", "S", "O"}:
        parsed = values.astype("datetime64[ns]")
        return parsed.astype(np.float64) / 1e9
    seconds = values.astype(np.float64)
    if not np.all(np.isfinite(seconds)):
        raise ValueError("Время должно состоять из конечных чисел или дат.")
    return seconds


def _unpack(series: Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Проверяет кортеж (время, долгота, широта, метка) и возвращает массивы."""
    if len(series) != 4:
        raise ValueError("Ожидается кортеж (время, долгота, широта, метка).")
    time, lon, lat, label = (np.asarray(part) for part in series)
    if time.ndim != 1 or lon.ndim != 1 or lat.ndim != 1 or label.ndim != 1:
        raise ValueError("Каждый элемент кортежа должен быть одномерным массивом.")
    if not time.shape == lon.shape == lat.shape == label.shape:
        raise ValueError("Время, долгота, широта и метка должны иметь одинаковую длину.")
    if time.size == 0:
        raise ValueError("Ряд не должен быть пустым.")
    lon = lon.astype(np.float64)
    lat = lat.astype(np.float64)
    label = label.astype(np.float64)
    if not np.all(np.isfinite(label)):
        raise ValueError("Метки должны быть конечными числами.")
    return time, lon, lat, label


def _validate_coordinates(lat: np.ndarray, lon: np.ndarray) -> None:
    """Проверяет географические координаты в градусах."""
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Координаты должны быть конечными числами.")
    if np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("Широта должна лежать в диапазоне [-90, 90].")
    if np.any((lon < -180.0) | (lon > 180.0)):
        raise ValueError("Долгота должна лежать в диапазоне [-180, 180].")


def _prepare(
    prediction: Series, ground_truth: Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Выравнивает prediction и ground truth по общему индексу точек.

    Возвращает время ground truth в секундах или исходных числовых единицах,
    координаты prediction, маску предсказанных аномалий, координаты ground truth
    и маску истинных аномалий.
    """
    pred_time, pred_lon, pred_lat, pred_label = _unpack(prediction)
    gt_time, gt_lon, gt_lat, gt_label = _unpack(ground_truth)
    if pred_time.shape != gt_time.shape:
        raise ValueError("prediction и ground truth должны содержать одинаковое число точек.")
    _validate_coordinates(pred_lat, pred_lon)
    _validate_coordinates(gt_lat, gt_lon)
    time = _to_seconds(gt_time)
    if np.any(~np.isfinite(_to_seconds(pred_time))):
        raise ValueError("Время prediction должно быть конечным.")
    if time.size >= 2 and np.any(np.diff(time) <= 0.0):
        raise ValueError("Время ground truth должно строго возрастать.")
    return (
        time,
        pred_lon,
        pred_lat,
        np.equal(pred_label, 0.0),
        gt_lon,
        gt_lat,
        np.equal(gt_label, 0.0),
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Возвращает частное. При нулевом знаменателе метрика не определена."""
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _harmonic_mean(precision: float, recall: float) -> float:
    """Считает F1. Если precision или recall не определены, результат не определён."""
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    total = precision + recall
    if total <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / total)


def _true_runs(mask: np.ndarray) -> list[Tuple[int, int]]:
    """Возвращает включительные границы непрерывных участков, где mask истинна."""
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _time_range(time_s: np.ndarray) -> Tuple[float, float]:
    """Диапазон ряда [t_0, t_N), где t_N продолжает последний шаг дискретизации."""
    if time_s.size == 1:
        return float(time_s[0]), float(time_s[0] + 1.0)
    step = float(time_s[-1] - time_s[-2])
    return float(time_s[0]), float(time_s[-1] + step)


def _events_from_mask(time_s: np.ndarray, mask: np.ndarray) -> Tuple[list[Tuple[float, float]], Tuple[float, float]]:
    """Переводит аномальные точки в полуинтервалы [t_i, t_{i+1})."""
    t_range = _time_range(time_s)
    events: list[Tuple[float, float]] = []
    for start, end in _true_runs(mask):
        t_start = float(time_s[start])
        t_stop = float(time_s[end + 1]) if end + 1 < time_s.size else t_range[1]
        if t_stop > t_start:
            events.append((t_start, t_stop))
    return events, t_range


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Геодезическое расстояние на сфере радиуса 6371 км, в метрах."""
    lat1_rad = np.radians(np.asarray(lat1, dtype=np.float64))
    lat2_rad = np.radians(np.asarray(lat2, dtype=np.float64))
    lon1_deg = np.asarray(lon1, dtype=np.float64)
    lon2_deg = np.asarray(lon2, dtype=np.float64)
    dlon = np.radians((lon2_deg - lon1_deg + 180.0) % 360.0 - 180.0)
    dlat = lat2_rad - lat1_rad
    haversine = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return _EARTH_RADIUS_M * 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    """Численный интеграл. np.trapezoid есть в NumPy 2, np.trapz — в NumPy 1.26."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(values, grid))


def _path_length_m(lat: np.ndarray, lon: np.ndarray) -> float:
    """Суммарная длина ломаной по последовательным точкам, в метрах."""
    if lat.size < 2:
        return 0.0
    segments = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.sum(segments))


def _fill_reference_gap(
    time_s: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_ref: np.ndarray,
    lon_ref: np.ndarray,
    start: int,
    end: int,
) -> None:
    """Линейно интерполирует эталон между штатными соседями аномального участка."""
    alpha = (time_s[start : end + 1] - time_s[start - 1]) / (time_s[end + 1] - time_s[start - 1])
    lon_left = lon[start - 1]
    delta_lon = (lon[end + 1] - lon_left + 180.0) % 360.0 - 180.0
    lat_ref[start : end + 1] = lat[start - 1] + alpha * (lat[end + 1] - lat[start - 1])
    lon_ref[start : end + 1] = (lon_left + alpha * delta_lon + 180.0) % 360.0 - 180.0


def _reference_track(
    time_s: np.ndarray, lat: np.ndarray, lon: np.ndarray, gt_anomaly: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит эталонные координаты линейной интерполяцией через аномальные участки.

    Штатные точки сохраняют свои координаты. Для аномального участка берутся
    ближайшие штатные точки слева и справа. Если одной из них нет, эталон для
    RMSE и Хаусдорфа на этом участке не определён: в маске valid стоит False,
    а в сам трек эталона подставляется исходная координата.
    Долгота интерполируется по кратчайшей дуге.
    """
    lat_ref = lat.copy()
    lon_ref = lon.copy()
    valid = np.ones(lat.shape, dtype=bool)
    last = lat.size - 1
    for start, end in _true_runs(gt_anomaly):
        if start == 0 or end == last:
            valid[start : end + 1] = False
            continue
        _fill_reference_gap(time_s, lat, lon, lat_ref, lon_ref, start, end)
    return lat_ref, lon_ref, valid


def _false_negative_distances(
    time_s: np.ndarray,
    gt_lon: np.ndarray,
    gt_lat: np.ndarray,
    gt_anomaly: np.ndarray,
    pred_anomaly: np.ndarray,
) -> np.ndarray:
    """Геодезические отклонения false negative точек от интерполированного эталона."""
    lat_ref, lon_ref, valid = _reference_track(time_s, gt_lat, gt_lon, gt_anomaly)
    usable = gt_anomaly & ~pred_anomaly & valid
    if not np.any(usable):
        return np.empty(0, dtype=np.float64)
    distances = _haversine_m(gt_lat[usable], gt_lon[usable], lat_ref[usable], lon_ref[usable])
    return distances[np.isfinite(distances)]


def _binary_counts(gt_anomaly: np.ndarray, pred_anomaly: np.ndarray) -> Tuple[float, float, float]:
    """Считает TP, FP и FN. Положительный класс — аномалия."""
    true_positive = float(np.sum(gt_anomaly & pred_anomaly))
    false_positive = float(np.sum(~gt_anomaly & pred_anomaly))
    false_negative = float(np.sum(gt_anomaly & ~pred_anomaly))
    return true_positive, false_positive, false_negative


def _pr_auc(gt_anomaly: np.ndarray, scores: np.ndarray) -> float:
    """Площадь под precision-recall кривой методом трапеций."""
    anomaly_scores = np.asarray(scores, dtype=np.float64)
    if anomaly_scores.ndim != 1 or anomaly_scores.shape != gt_anomaly.shape:
        raise ValueError("anomaly_scores должен быть одномерным массивом той же длины, что и ряд.")
    if not np.all(np.isfinite(anomaly_scores)):
        raise ValueError("anomaly_scores должен состоять из конечных чисел.")
    positives = float(np.sum(gt_anomaly))
    if positives <= 0.0:
        return float("nan")

    order = np.argsort(-anomaly_scores, kind="mergesort")
    labels = gt_anomaly.astype(np.float64)[order]
    sorted_scores = anomaly_scores[order]
    score_changes = np.flatnonzero(np.diff(sorted_scores))
    group_ends = np.concatenate((score_changes, [labels.size - 1]))
    true_positives = np.cumsum(labels)[group_ends]
    false_positives = (group_ends + 1.0) - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return _trapezoid(precision, recall)


def _affiliation_zones(events: list[Tuple[float, float]], t_range: Tuple[float, float]) -> list[Tuple[float, float]]:
    """Делит ось времени на зоны affiliation по серединам промежутков между событиями."""
    zones: list[Tuple[float, float]] = []
    last = len(events) - 1
    for index, (start, stop) in enumerate(events):
        left = t_range[0] if index == 0 else 0.5 * (events[index - 1][1] + start)
        right = t_range[1] if index == last else 0.5 * (stop + events[index + 1][0])
        zones.append((left, right))
    return zones


def _events_inside(events: list[Tuple[float, float]], zone: Tuple[float, float]) -> list[Tuple[float, float]]:
    """Оставляет части предсказанных событий, попавшие в зону affiliation."""
    zone_start, zone_stop = zone
    clipped: list[Tuple[float, float]] = []
    for start, stop in events:
        left = max(start, zone_start)
        right = min(stop, zone_stop)
        if left < right:
            clipped.append((left, right))
    return clipped


def _integrate_min(d_start: float, d_stop: float, margin: float) -> float:
    """Интеграл min(z, margin) по z от d_start до d_stop."""
    if d_stop <= margin:
        return 0.5 * (d_stop * d_stop - d_start * d_start)
    if d_start >= margin:
        return margin * (d_stop - d_start)
    capped_head = 0.5 * (margin * margin - d_start * d_start)
    return capped_head + margin * (d_stop - margin)


def _integrate_precision_survival(
    d_start: float, d_stop: float, margin: float, event_length: float, zone_length: float
) -> float:
    """Интеграл survival function precision по расстоянию до истинного события."""
    width = d_stop - d_start
    integral_distance = 0.5 * (d_stop * d_stop - d_start * d_start)
    integral_capped = _integrate_min(d_start, d_stop, margin)
    numerator = event_length * width + integral_capped + integral_distance
    return width - numerator / zone_length


def _precision_side_integral(
    start: float,
    stop: float,
    pivot: float,
    event: Tuple[float, float],
    zone: Tuple[float, float],
    before: bool,
) -> float:
    """Интеграл precision по части предсказания, лежащей строго вне истинного события."""
    if stop <= start:
        return 0.0
    if before:
        d_start, d_stop = pivot - stop, pivot - start
    else:
        d_start, d_stop = start - pivot, stop - pivot
    zone_length = zone[1] - zone[0]
    if zone_length <= 0.0 or d_stop <= d_start:
        return 0.0
    margin = max(0.0, min(event[0] - zone[0], zone[1] - event[1]))
    return _integrate_precision_survival(d_start, d_stop, margin, event[1] - event[0], zone_length)


def _precision_integral(start: float, stop: float, event: Tuple[float, float], zone: Tuple[float, float]) -> float:
    """Интеграл локальной precision probability по одному предсказанному интервалу."""
    event_start, event_stop = event
    total = 0.0
    overlap_start = max(start, event_start)
    overlap_stop = min(stop, event_stop)
    if overlap_start < overlap_stop:
        total += overlap_stop - overlap_start
    if start < event_start:
        total += _precision_side_integral(start, min(stop, event_start), event_start, event, zone, True)
    if stop > event_stop:
        total += _precision_side_integral(max(start, event_stop), stop, event_stop, event, zone, False)
    return total


def _precision_probability(
    preds: list[Tuple[float, float]], event: Tuple[float, float], zone: Tuple[float, float]
) -> float:
    """Локальная affiliation precision одного истинного события. Без предсказаний не определена."""
    if not preds:
        return float("nan")
    total_length = sum(stop - start for start, stop in preds)
    if total_length <= 0.0:
        return float("nan")
    integral = sum(_precision_integral(start, stop, event, zone) for start, stop in preds)
    return float(min(1.0, max(0.0, integral / total_length)))


def _recall_grid(start: float, stop: float, zone: Tuple[float, float], preds: list[Tuple[float, float]]) -> np.ndarray:
    """Сетка интегрирования recall: равномерные узлы и изломы расстояния до предсказаний."""
    knots = [start, stop, 0.5 * (zone[0] + zone[1])]
    for left, right in preds:
        knots.extend((left, right))
    for (_, right), (left, _) in zip(preds, preds[1:]):
        knots.append(0.5 * (right + left))
    inside = [knot for knot in knots if start <= knot <= stop]
    uniform = np.linspace(start, stop, _RECALL_GRID_SIZE)
    return np.unique(np.concatenate((uniform, np.asarray(inside, dtype=np.float64))))


def _distance_to_events(times: np.ndarray, events: list[Tuple[float, float]]) -> np.ndarray:
    """Расстояние от каждой точки до ближайшего предсказанного интервала."""
    distances = np.full(times.shape, np.inf, dtype=np.float64)
    for start, stop in events:
        delta = np.where(times < start, start - times, np.where(times > stop, times - stop, 0.0))
        distances = np.minimum(distances, delta)
    return distances


def _recall_probability(
    preds: list[Tuple[float, float]], event: Tuple[float, float], zone: Tuple[float, float]
) -> float:
    """Локальная affiliation recall одного истинного события. Без предсказаний равна 0."""
    if not preds:
        return 0.0
    event_start, event_stop = event
    length = event_stop - event_start
    if length <= 0.0:
        return 0.0
    zone_length = zone[1] - zone[0]
    if zone_length <= 0.0:
        return 0.0
    grid = _recall_grid(event_start, event_stop, zone, preds)
    distances = _distance_to_events(grid, preds)
    margin = np.minimum(grid - zone[0], zone[1] - grid)
    penalty = np.minimum(distances, np.maximum(margin, 0.0)) + distances
    survival = 1.0 - penalty / zone_length
    np.clip(survival, 0.0, 1.0, out=survival)
    return float(min(1.0, max(0.0, _trapezoid(survival, grid) / length)))


def _affiliation_from_events(
    pred_events: list[Tuple[float, float]],
    gt_events: list[Tuple[float, float]],
    t_range: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Affiliation precision, recall и F1 по уже выделенным временным интервалам.

    Precision усредняется по событиям, в зону которых попало хотя бы одно предсказание.
    Recall усредняется по всем истинным событиям: пропуск даёт локальный recall 0.
    """
    if not gt_events:
        return float("nan"), float("nan"), float("nan")
    precisions: list[float] = []
    recalls: list[float] = []
    for event, zone in zip(gt_events, _affiliation_zones(gt_events, t_range)):
        preds = _events_inside(pred_events, zone)
        precisions.append(_precision_probability(preds, event, zone))
        recalls.append(_recall_probability(preds, event, zone))
    defined = [value for value in precisions if not math.isnan(value)]
    precision = float(sum(defined) / len(defined)) if defined else float("nan")
    recall = float(sum(recalls) / len(recalls))
    return precision, recall, _harmonic_mean(precision, recall)


def _point_scores(gt_anomaly: np.ndarray, pred_anomaly: np.ndarray) -> Tuple[float, float, float]:
    """Point-wise precision, recall и F1 по бинарным маскам аномалий."""
    true_positive, false_positive, false_negative = _binary_counts(gt_anomaly, pred_anomaly)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1_score = _ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative)
    return precision, recall, f1_score


def _detection_delay_summary(time_s: np.ndarray, gt_anomaly: np.ndarray, pred_anomaly: np.ndarray) -> Dict[str, float]:
    """Mean, median и P95 задержки только по обнаруженным событиям."""
    delays: list[float] = []
    undetected = 0
    runs = _true_runs(gt_anomaly)
    for start, end in runs:
        detected = pred_anomaly[start : end + 1]
        if not np.any(detected):
            undetected += 1
            continue
        detect_at = start + int(np.argmax(detected))
        delays.append(float(time_s[detect_at] - time_s[start]))
    delay_values = np.asarray(delays, dtype=np.float64)
    if delay_values.size == 0:
        mean = median = p95 = float("nan")
    else:
        mean = float(np.mean(delay_values))
        median = float(np.median(delay_values))
        p95 = float(np.percentile(delay_values, 95))
    event_count = len(runs)
    return {
        "mean": mean,
        "median": median,
        "p95": p95,
        "n_events": float(event_count),
        "n_detected": float(event_count - undetected),
        "n_undetected": float(undetected),
    }


class CalculateMetrics:
    """Расчёт метрик из metrics.md для пары prediction / ground truth."""

    @staticmethod
    def calculate_all_metrics(
        prediction: Series,
        ground_truth: Series,
        anomaly_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Считает все метрики документа и возвращает их одним словарём.

        anomaly_scores нужен только для PR-AUC. Если его нет, pr_auc равен nan:
        по одной бинарной маске площадь под PR-кривой не определена.
        Задержка считается в секундах для datetime и в единицах времени ряда иначе.
        Пространственные метрики возвращаются в метрах, DLR — безразмерным.
        """
        precision, recall, f1_score = CalculateMetrics.calculate_point_precision(prediction, ground_truth)
        affiliation_precision, affiliation_recall, affiliation_f1 = CalculateMetrics.calculate_affiliation_f1(
            prediction, ground_truth
        )
        distance_loss, distance_loss_ratio = CalculateMetrics.calculate_distance_loss(prediction, ground_truth)
        delays = CalculateMetrics.calculate_detection_delay(prediction, ground_truth)
        if anomaly_scores is None:
            pr_auc = float("nan")
        else:
            pr_auc = CalculateMetrics.calculate_pr_auc(anomaly_scores, ground_truth)
        metrics = {
            "point_precision": precision,
            "point_recall": recall,
            "point_f1": f1_score,
            "pr_auc": pr_auc,
            "affiliation_precision": affiliation_precision,
            "affiliation_recall": affiliation_recall,
            "affiliation_f1": affiliation_f1,
            "geodesic_rmse": CalculateMetrics.calculate_geodesic_rmse(prediction, ground_truth),
            "hausdorff_distance": CalculateMetrics.calculate_hausdorff_distance(prediction, ground_truth),
            "distance_loss": distance_loss,
            "distance_loss_ratio": distance_loss_ratio,
        }
        for name, value in delays.items():
            metrics[f"detection_delay_{name}"] = value
        return metrics

    @staticmethod
    def calculate_point_precision(prediction: Series, ground_truth: Series) -> Tuple[float, float, float]:
        """Считает point-wise precision, recall и F1.

        Аномалия — точка с меткой 0:
        TP = sum 1[y=0 и ŷ=0], FP = sum 1[y>0 и ŷ=0], FN = sum 1[y=0 и ŷ>0].
        F1 = 2TP / (2TP + FP + FN).
        Время и координаты в расчёте не участвуют. При нулевом знаменателе
        соответствующая величина равна nan.

        Args:
            prediction: время, долгота, широта, предсказанная метка.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            precision, recall и F1 точечной классификации аномалий.
        """
        _, _, _, pred_anomaly, _, _, gt_anomaly = _prepare(prediction, ground_truth)
        return _point_scores(gt_anomaly, pred_anomaly)

    @staticmethod
    def calculate_pr_auc(anomaly_scores: np.ndarray, ground_truth: Series) -> float:
        """Считает площадь под precision-recall кривой anomaly score.

        Для каждого порога tau аномалия объявляется при s_i >= tau. Интеграл
        Precision(recall) считается методом трапеций. Истинная аномалия — y_i = 0.
        Если в ground truth нет аномалий, PR-AUC не определён.

        Args:
            anomaly_scores: непрерывная оценка аномальности, большее значение — хуже.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            PR-AUC в диапазоне от 0 до 1 либо nan.
        """
        _, _, _, gt_label = _unpack(ground_truth)
        return _pr_auc(np.equal(gt_label, 0.0), anomaly_scores)

    @staticmethod
    def calculate_affiliation_f1(prediction: Series, ground_truth: Series) -> Tuple[float, float, float]:
        """Считает affiliation precision, recall и F1 по временной близости событий.

        Непрерывные аномальные участки переводятся в интервалы [t_i, t_{i+1}).
        Ось времени режется на зоны ближайшего истинного события. Локальные
        precision и recall — средние survival function расстояния до события
        внутри своей зоны (Huet, Navarro, Rossi, KDD 2022). Координаты не используются.
        Precision усредняется только по зонам, где есть предсказание; recall —
        по всем истинным событиям.

        Args:
            prediction: время, долгота, широта, предсказанная метка.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            affiliation precision, recall и F1. Без истинных событий все три равны nan.
        """
        time, _, _, pred_anomaly, _, _, gt_anomaly = _prepare(prediction, ground_truth)
        gt_events, t_range = _events_from_mask(time, gt_anomaly)
        pred_events, _ = _events_from_mask(time, pred_anomaly)
        return _affiliation_from_events(pred_events, gt_events, t_range)

    @staticmethod
    def calculate_detection_delay(prediction: Series, ground_truth: Series) -> Dict[str, float]:
        """Считает задержку первой детекции каждого истинного аномального участка.

        Delay_k = t_detect - t_start, где t_detect — первая точка участка с ŷ=0.
        Если до конца участка аномалия не предсказана, событие не входит в среднее,
        медиану и P95 и учитывается в n_undetected. Нулевая задержка означает
        детекцию на первой точке события, а не пропуск.

        Args:
            prediction: время, долгота, широта, предсказанная метка.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            mean, median, p95, n_events, n_detected и n_undetected.
            При отсутствии обнаруженных событий агрегаты равны nan.
        """
        time, _, _, pred_anomaly, _, _, gt_anomaly = _prepare(prediction, ground_truth)
        return _detection_delay_summary(time, gt_anomaly, pred_anomaly)

    @staticmethod
    def calculate_geodesic_rmse(prediction: Series, ground_truth: Series) -> float:
        """Считает RMSE геодезического отклонения пропущенных аномальных точек.

        Эталон внутри аномального участка — линейная интерполяция между ближайшими
        штатными точками ground truth слева и справа. В сумму входят только
        false negative: y=0 и ŷ!=0. Если таких точек нет, штраф равен 0.
        Если для них нельзя построить эталон, результат равен nan.

        Args:
            prediction: время, долгота, широта, предсказанная метка.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            RMSE в метрах.
        """
        time, _, _, pred_anomaly, gt_lon, gt_lat, gt_anomaly = _prepare(prediction, ground_truth)
        false_negative = gt_anomaly & ~pred_anomaly
        if not np.any(false_negative):
            return 0.0
        distances = _false_negative_distances(time, gt_lon, gt_lat, gt_anomaly, pred_anomaly)
        if distances.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(distances))))

    @staticmethod
    def calculate_hausdorff_distance(prediction: Series, ground_truth: Series) -> float:
        """Считает направленное временно-сопоставленное расстояние Хаусдорфа.

        H = max d_geo(P_i, P_i*) по false negative точкам. P_i* — та же
        интерполированная эталонная позиция, что и в geodesic RMSE.
        Без пропусков расстояние равно 0. Без эталона для пропусков — nan.

        Args:
            prediction: время, долгота, широта, предсказанная метка.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            Максимальное геодезическое отклонение пропущенной точки, в метрах.
        """
        time, _, _, pred_anomaly, gt_lon, gt_lat, gt_anomaly = _prepare(prediction, ground_truth)
        if not np.any(gt_anomaly & ~pred_anomaly):
            return 0.0
        distances = _false_negative_distances(time, gt_lon, gt_lat, gt_anomaly, pred_anomaly)
        if distances.size == 0:
            return float("nan")
        return float(np.max(distances))

    @staticmethod
    def calculate_distance_loss(prediction: Series, ground_truth: Series) -> Tuple[float, float]:
        """Считает абсолютный и относительный distance loss длины траектории.

        L_ref — длина эталона: штатные точки ground truth и линейная интерполяция
        аномальных промежутков. L_pred — длина ломаной по точкам, которые
        классификатор оставил штатными (ŷ!=0), в координатах prediction.
        DL = |L_pred - L_ref|, DLR = DL / L_ref.

        Args:
            prediction: время, долгота, широта, предсказанная метка.
            ground_truth: время, долгота, широта, истинная метка.

        Returns:
            DL в метрах и безразмерный DLR. При нулевой эталонной длине DLR равен nan.
        """
        time, pred_lon, pred_lat, pred_anomaly, gt_lon, gt_lat, gt_anomaly = _prepare(prediction, ground_truth)
        lat_ref, lon_ref, _ = _reference_track(time, gt_lat, gt_lon, gt_anomaly)
        reference_length = _path_length_m(lat_ref, lon_ref)
        accepted = ~pred_anomaly
        predicted_length = _path_length_m(pred_lat[accepted], pred_lon[accepted])
        loss = abs(predicted_length - reference_length)
        return loss, _ratio(loss, reference_length)


def _sample_track() -> Tuple[Series, Series, np.ndarray]:
    """Короткий трек: штатный ход на северо-восток и петля на точках 8–12."""
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
    scores = np.full(count, 0.15)
    scores[8:13] = np.array([0.55, 0.62, 0.90, 0.95, 0.88])
    scores[15] = 0.70
    ground_truth = (time, longitude, latitude, labels)
    prediction = (time, longitude.copy(), latitude.copy(), predicted)
    return prediction, ground_truth, scores


def _print_metrics(title: str, metrics: Dict[str, float]) -> None:
    """Печатает словарь метрик выровненными строками."""
    print(title)
    for name, value in metrics.items():
        print(f"  {name:<32} {value:12.4f}")


if __name__ == "__main__":
    try:
        from plot_sample_track import plot_sample_track  # pylint: disable=import-outside-toplevel
    except ImportError:
        from app.metrics.plot_sample_track import plot_sample_track  # pylint: disable=import-outside-toplevel

    delayed_prediction, truth, anomaly_score = _sample_track()
    print(
        "Тестовый трек: 21 точка, шаг 1. Аномалия — петля на индексах 8–12. "
        "Модель обнаруживает её с индекса 10 и даёт ложное срабатывание на индексе 15."
    )
    delayed_metrics = CalculateMetrics.calculate_all_metrics(delayed_prediction, truth, anomaly_score)
    _print_metrics("Метрики с задержкой детекции:", delayed_metrics)

    truth_time, truth_lon, truth_lat, truth_label = truth
    perfect_prediction = (truth_time, truth_lon, truth_lat, truth_label.copy())
    perfect_metrics = CalculateMetrics.calculate_all_metrics(perfect_prediction, truth, anomaly_score)
    _print_metrics("Те же точки, предсказанные метки совпадают с ground truth:", perfect_metrics)
    plot_sample_track(delayed_prediction, truth)
