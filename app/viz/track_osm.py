"""Визуализация GPS-треков на подложке OpenStreetMap"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from app.evaluation.series import TrajectorySeries

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from PIL import Image

MapLimits = Tuple[float, float, float, float]
BasemapExtent = Tuple[float, float, float, float]
BasemapBundle = Tuple[np.ndarray, BasemapExtent]

_OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_MAP_USER_AGENT = "statistical-researchs-viz/0.1"
_MAP_CAPTION = "Подложка: OpenStreetMap (нужен доступ в интернет)."
_COLOR_NORMAL = "#24527a"
_COLOR_ANOMALY = "#b23a48"
_COLOR_REFERENCE = "#e8590c"
_COLOR_TRACK = "#071a33"
_COLOR_HIT = "#1f7a4d"
_COLOR_FALSE = "#d97706"
_COLOR_ACCEPTED = "#1f2933"
_TILE_MEMORY: dict[Tuple[int, int, int], np.ndarray] = {}


def plot_track_maps(
    prediction: TrajectorySeries,
    ground_truth: TrajectorySeries,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 8.5),
    annotate: Optional[bool] = None,
    bbox: Optional[Sequence[float]] = None,
    use_basemap: bool = True,
) -> Optional[Tuple[Path, Path]]:
    """Рисует трек на карте OSM: ground truth и предсказание модели.

    Формируются две отдельные фигуры (не в одном ряду). На каждой — растровая подложка
    из тайлов OpenStreetMap (если use_basemap и есть сеть), линия маршрута, точки
    с разными маркерами по типу (штатная / аномалия / ошибки классификации), легенда и
    при необходимости подписи с индексами точек.

    Args:
        prediction: Ряд prediction (поля time, lon, lat, label).
        ground_truth: Эталонный ряд; метка 0 — аномалия.
        output_path: Базовый путь для PNG. Сохраняются *_ground_truth* и *_model*.
            Если None, фигуры показываются интерактивно (в ноутбуке — через IPython).
        title: Общий заголовок; к нему добавляются суффиксы панелей.
        figsize: Размер фигуры в дюймах.
        annotate: Подписывать индексы точек. По умолчанию True, если точек не больше 80.
        bbox: Окно карты [юг, запад, север, восток] в градусах. Иначе — авто-поля вокруг трека.
        use_basemap: Загружать тайлы OSM; при ошибке сети рисуется только трек на сером фоне.

    Returns:
        Пути к сохранённым файлам (ground_truth, model) или None, если не сохраняли.
    """
    time = TrajectorySeries.time_to_seconds(ground_truth.time)
    lon = ground_truth.lon
    lat = ground_truth.lat
    labels = ground_truth.label
    predicted = prediction.label
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    limits = _limits_from_bbox(bbox) if bbox is not None else _padded_limits(lon, lat)
    basemap = _load_basemap(limits[0], limits[2], limits[1], limits[3]) if use_basemap else None
    reference_lat, reference_lon, reference_valid = _reference_coordinates(time, lat, lon, labels)
    if annotate is None:
        annotate = labels.size <= 80
    indices = _story_indices(labels, predicted) if annotate else []
    base_title = title or ""
    caption = _MAP_CAPTION if use_basemap else "Без подложки карты."
    if annotate:
        caption += " Числа — индексы ряда."

    base_output = Path(output_path) if output_path is not None else None
    truth_path, model_path = _map_output_paths(base_output) if base_output is not None else (None, None)
    interactive = base_output is None

    figure_truth, axis_truth = plt.subplots(1, 1, figsize=figsize)
    _draw_truth_map(
        axis_truth,
        lon,
        lat,
        labels,
        reference_lon,
        reference_lat,
        reference_valid,
        basemap,
        limits,
    )
    if annotate:
        _annotate_indices(axis_truth, lon, lat, indices)
    _finish_map_figure(
        figure_truth,
        f"{base_title} — изначальный трек",
        caption,
        truth_path,
        interactive,
    )

    figure_model, axis_model = plt.subplots(1, 1, figsize=figsize)
    _draw_prediction_map(axis_model, lon, lat, labels, predicted, basemap, limits)
    if annotate:
        _annotate_indices(axis_model, lon, lat, indices)
    _finish_map_figure(
        figure_model,
        f"{base_title} — разметка модели",
        caption,
        model_path,
        interactive,
    )

    if base_output is None:
        return None
    return truth_path, model_path


def _tile_index(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """
    Номер тайла Web Mercator, в котором лежит точка
    Args:
        lon: Долгота точки
        lat: Широта точки
        zoom: Масштаб карты

    Returns:
        x_tile: Индекс тайла по долготе
        y_tile: Индекс тайла по широте
    """
    scale = 2.0**zoom
    x_tile = int((lon + 180.0) / 360.0 * scale)
    lat_rad = math.radians(lat)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale)
    last = int(scale) - 1
    return min(max(x_tile, 0), last), min(max(y_tile, 0), last)


def _tile_bounds(x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    Границы тайла: запад, юг, восток, север в градусах.
    Args:
        x_tile: Индекс тайла по долготе
        y_tile: Индекс тайла по широте
        zoom: Масштаб карты

    Returns:
        west: Западная граница тайла
        south: Южная граница тайла
        east: Восточная граница тайла
        north: Северная граница тайла
    """
    scale = 2.0**zoom
    west = x_tile / scale * 360.0 - 180.0
    east = (x_tile + 1) / scale * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / scale))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y_tile + 1) / scale))))
    return west, south, east, north


def _tile_span(west: float, south: float, east: float, north: float, zoom: int) -> Tuple[int, int, int, int]:
    """
    Индексы тайлов, покрывающих прямоугольник: x0, y_north, x1, y_south.
    Args:
        west: Западная граница прямоугольника
        south: Южная граница прямоугольника
        east: Восточная граница прямоугольника
        north: Северная граница прямоугольника
        zoom: Масштаб карты

    Returns:
        x_west: Индекс тайла по долготе для западной границы
        y_north: Индекс тайла по широте для северной границы
        x_east: Индекс тайла по долготе для восточной границы
        y_south: Индекс тайла по широте для южной границы
    """
    x_west, y_north = _tile_index(west, north, zoom)
    x_east, y_south = _tile_index(east, south, zoom)
    return x_west, y_north, x_east, y_south


def _choose_zoom(west: float, south: float, east: float, north: float, max_tiles: int = 20) -> int:
    """
    Самый подробный масштаб, который укладывается в заданное число тайлов.
    Args:
        west: Западная граница прямоугольника
        south: Южная граница прямоугольника
        east: Восточная граница прямоугольника
        north: Северная граница прямоугольника
        max_tiles: Максимальное число тайлов

    Returns:
        zoom: Выбранный масштаб карты
    """
    for zoom in range(15, -1, -1):
        x_west, y_north, x_east, y_south = _tile_span(west, south, east, north, zoom)
        if (x_east - x_west + 1) * (y_south - y_north + 1) <= max_tiles:
            return zoom
    return 0


def _download_tile(zoom: int, x_tile: int, y_tile: int) -> np.ndarray:
    """
    Один тайл OSM: скачивание и раскладка в RGB-массив.
    Args:
        zoom: Масштаб карты
        x_tile: Индекс тайла по долготе
        y_tile: Индекс тайла по широте

    Returns:
        image: RGB-массив тайла
    """
    key = (zoom, x_tile, y_tile)
    cached = _TILE_MEMORY.get(key)
    if cached is not None:
        return cached
    url = _OSM_TILE_URL.format(z=zoom, x=x_tile, y=y_tile)
    request = urllib.request.Request(url, headers={"User-Agent": _MAP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read()
    image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    _TILE_MEMORY[key] = image
    return image


def _load_basemap(
    west: float,
    south: float,
    east: float,
    north: float,
) -> Optional[BasemapBundle]:
    """
    Склеивает тайлы OSM для прямоугольника (запад, юг, восток, север).
    Args:
        west: Западная граница прямоугольника
        south: Южная граница прямоугольника
        east: Восточная граница прямоугольника
        north: Северная граница прямоугольника

    Returns:
        image: RGB-массив подложки
        extent: (запад, восток, юг, север) в градусах
    """
    try:
        zoom = _choose_zoom(west, south, east, north)
        x_west, y_north, x_east, y_south = _tile_span(west, south, east, north, zoom)
        rows = []
        for y_tile in range(y_north, y_south + 1):
            row = [_download_tile(zoom, x_tile, y_tile) for x_tile in range(x_west, x_east + 1)]
            rows.append(np.concatenate(row, axis=1))
        image = np.concatenate(rows, axis=0)
        bound_west, _, _, bound_north = _tile_bounds(x_west, y_north, zoom)
        _, bound_south, bound_east, _ = _tile_bounds(x_east, y_south, zoom)
        return image, (bound_west, bound_east, bound_south, bound_north)
    except OSError as error:
        print(f"Подложку OSM загрузить не удалось ({error}). Рисуется только трек.")
        return None


def _limits_from_bbox(bbox: Sequence[float]) -> MapLimits:
    """
    Преобразует bbox [юг, запад, север, восток] в (запад, восток, юг, север).
    Args:
        bbox: Bounding box в формате [юг, запад, север, восток]

    Returns:
        limits: Границы прямоугольника в формате (запад, восток, юг, север)
    """
    if len(bbox) != 4:
        raise ValueError("bbox должен содержать четыре числа: [юг, запад, север, восток]")
    south, west, north, east = (float(value) for value in bbox)
    if south >= north:
        raise ValueError("bbox: южная граница должна быть меньше северной")
    if west >= east:
        raise ValueError("bbox: западная граница должна быть меньше восточной")
    return west, east, south, north


def _padded_limits(lon: np.ndarray, lat: np.ndarray) -> MapLimits:
    """
    Прямоугольник вокруг точек с полями: запад, восток, юг, север.
    Args:
        lon: Массив долгот
        lat: Массив широт

    Returns:
        limits: Границы прямоугольника в формате (запад, восток, юг, север)
    """
    lon_span = max(float(np.max(lon) - np.min(lon)), 0.01)
    lat_span = max(float(np.max(lat) - np.min(lat)), 0.01)
    return (
        float(np.min(lon) - 0.45 * lon_span),
        float(np.max(lon) + 0.18 * lon_span),
        float(np.min(lat) - 0.22 * lat_span),
        float(np.max(lat) + 0.18 * lat_span),
    )


def _draw_basemap(axis: Axes, basemap: Optional[BasemapBundle], limits: MapLimits) -> None:
    """
    Кладёт карту на оси и выставляет географический масштаб.
    Args:
        axis: Оси matplotlib
        basemap: Подложка OSM
        limits: Границы прямоугольника в формате (запад, восток, юг, север)
    """
    west, east, south, north = limits
    if basemap is not None:
        image, extent = basemap
        map_west, map_east, map_south, map_north = extent
        axis.imshow(
            image,
            extent=(map_west, map_east, map_south, map_north),
            origin="upper",
            interpolation="bilinear",
            zorder=0,
            clip_on=True,
        )
    else:
        axis.set_facecolor("#f4f4f4")
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.margins(0)
    axis.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)), adjustable="box")
    axis.set_xlabel("Долгота, °")
    axis.set_ylabel("Широта, °")
    axis.tick_params(labelsize=8)


def _track_line(axis: Axes, lon: np.ndarray, lat: np.ndarray, **style: Any) -> None:
    """
    Линия трека с обрезкой по границам окна карты.
    Args:
        axis: Оси matplotlib
        lon: Массив долгот
        lat: Массив широт
        **style: Дополнительные параметры стиля
    """
    style.setdefault("clip_on", True)
    style.setdefault("alpha", 1.0)
    axis.plot(lon, lat, **style)


def _broken_polyline(lon: np.ndarray, lat: np.ndarray, omit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Координаты с NaN в omit: matplotlib не рисует отрезки через разрыв.
    Args:
        lon: Массив долгот
        lat: Массив широт
        omit: Массив булевых значений, указывающих на точки для замены на NaN

    Returns:
        lon_plot: Массив долгот с NaN в omit
        lat_plot: Массив широт с NaN в omit
    """
    lon_plot = np.asarray(lon, dtype=np.float64).copy()
    lat_plot = np.asarray(lat, dtype=np.float64).copy()
    omit = np.asarray(omit, dtype=bool)
    if omit.shape != lon_plot.shape:
        raise ValueError("omit должен совпадать по длине с lon/lat")
    lon_plot[omit] = np.nan
    lat_plot[omit] = np.nan
    return lon_plot, lat_plot


def _to_seconds(time: np.ndarray) -> np.ndarray:
    """
    Приводит время к float-секундам. Для datetime64 единица — секунды.
    Args:
        time: Массив времени

    Returns:
        Массив времени в секундах
    """
    values = np.asarray(time)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.float64) / 1e9
    if values.dtype.kind in {"U", "S", "O"}:
        return values.astype("datetime64[ns]").astype(np.float64) / 1e9
    return values.astype(np.float64)


def _marker_sizes(count: int) -> Tuple[int, int, int, int]:
    """
    Размеры маркеров: TP (штатная), TN (аномалия), FN (пропуск), FP (ложная тревога).
    Args:
        count: Количество точек

    Returns:
        Tuple[int, int, int, int]: Размеры маркеров (TP, TN, FN, FP)
    """
    if count > 8_000:
        return 8, 14, 18, 16
    if count > 400:
        return 18, 36, 48, 42
    return 32, 70, 90, 80


def _story_indices(labels: np.ndarray, predicted: np.ndarray, max_labels: int = 80) -> list[int]:
    """
    Индексы для подписей: края ряда, аномалии и характерные ошибки модели.
    Args:
        labels: Массив меток
        predicted: Массив предсказаний
        max_labels: Максимальное количество подписей

    Returns:
        indices: Список индексов для подписей
    """
    count = int(labels.size)
    chosen = {0, count - 1}
    anomaly = np.flatnonzero(np.equal(labels, 0.0))
    if anomaly.size:
        chosen.add(int(anomaly[0]) - 1)
        chosen.add(int(anomaly[-1]) + 1)
        step = max(1, anomaly.size // max(1, max_labels // 4))
        chosen.update(int(index) for index in anomaly[::step][: max_labels // 2])
    false_positive = np.flatnonzero(~np.equal(labels, 0.0) & np.equal(predicted, 0.0))
    if false_positive.size:
        step = max(1, false_positive.size // max(1, max_labels // 4))
        chosen.update(int(index) for index in false_positive[::step][: max_labels // 4])
    indices = sorted(index for index in chosen if 0 <= index < count)
    if len(indices) > max_labels:
        stride = max(1, len(indices) // max_labels)
        indices = indices[::stride][:max_labels]
    return indices


def _annotate_indices(axis: Axes, lon: np.ndarray, lat: np.ndarray, indices: list[int]) -> None:
    """
    Подписывает номера точек поверх карты.
    Args:
        axis: Оси matplotlib
        lon: Массив долгот
        lat: Массив широт
        indices: Список индексов для подписей
    """
    offsets = {0: (-14, -11), 7: (6, -12), 13: (8, 8), 15: (-16, 7), 20: (7, -9)}
    for index in indices:
        text = axis.annotate(
            str(index),
            (lon[index], lat[index]),
            textcoords="offset points",
            xytext=offsets.get(index, (5, 4)),
            fontsize=8,
            color=_COLOR_ACCEPTED,
            zorder=5,
            clip_on=True,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground="white")])


def _scatter_mask(axis: Axes, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, **style: Any) -> None:
    """
    Рисует подмножество точек, если оно не пустое.
    Args:
        axis: Оси matplotlib
        lon: Массив долгот
        lat: Массив широт
        mask: Маска для выбора точек
        **style: Дополнительные параметры стиля
    """
    if np.any(mask):
        style.setdefault("zorder", 4)
        style.setdefault("edgecolors", "white")
        style.setdefault("linewidths", 0.6)
        style.setdefault("clip_on", True)
        style.setdefault("alpha", 1.0)
        axis.scatter(lon[mask], lat[mask], **style)


def _draw_truth_map(
    axis: Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    reference_lon: np.ndarray,
    reference_lat: np.ndarray,
    reference_valid: np.ndarray,
    basemap: Optional[BasemapBundle],
    limits: MapLimits,
) -> None:
    """
    Левая панель: записанный трек, аномальная петля и эталонная интерполяция.
    Args:
        axis: Оси matplotlib
        lon: Массив долгот
        lat: Массив широт
        labels: Массив меток
        reference_lon: Массив долгот эталона
        reference_lat: Массив широт эталона
        reference_valid: Массив допустимых значений эталона
        basemap: Базовая карта
        limits: Предельные значения
    """
    _draw_basemap(axis, basemap, limits)
    normal_size, hit_size, _, _ = _marker_sizes(lon.size)
    line_width = 0.7 if lon.size > 8_000 else 2.9
    halo_width = 2.0 if lon.size > 8_000 else 5.6
    anomaly = np.equal(labels, 0.0)
    track_lon, track_lat = _broken_polyline(lon, lat, anomaly)
    ref_lon, ref_lat = _broken_polyline(reference_lon, reference_lat, ~reference_valid)
    _track_line(axis, track_lon, track_lat, color="white", linewidth=halo_width, zorder=2, solid_capstyle="round")
    _track_line(axis, track_lon, track_lat, color=_COLOR_TRACK, linewidth=line_width, zorder=3, solid_capstyle="round")
    _track_line(
        axis,
        ref_lon,
        ref_lat,
        color=_COLOR_REFERENCE,
        linestyle=(0, (7, 4)),
        linewidth=max(1.2, line_width),
        zorder=4,
        solid_capstyle="round",
    )
    normal = ~np.equal(labels, 0.0)
    _scatter_mask(axis, lon, lat, normal, s=normal_size, c=_COLOR_NORMAL, zorder=5, linewidths=0.15)
    _scatter_mask(axis, lon, lat, ~normal, s=hit_size, c=_COLOR_ANOMALY, zorder=6, linewidths=0.15)
    axis.set_title("Изначальный трек: штатный ход и аномалия")
    axis.legend(
        handles=[
            Line2D([0], [0], color=_COLOR_TRACK, linewidth=2.6, marker="o", label="Штатные точки"),
            Line2D([0], [0], color=_COLOR_ANOMALY, marker="o", linestyle="None", label="Аномалия, y = 0"),
            Line2D([0], [0], color=_COLOR_REFERENCE, linestyle=(0, (7, 4)), linewidth=2.6, label="Эталон через разрыв"),
        ],
        loc="lower right",
        framealpha=0.94,
        fontsize=8,
    )


def _draw_prediction_map(
    axis: Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    basemap: Optional[BasemapBundle],
    limits: MapLimits,
) -> None:
    """
    Правая панель: разметка модели и типы ошибок (TP, TN, FN, FP).
    Args:
        axis: Оси matplotlib
        lon: Массив долгот
        lat: Массив широт
        labels: Массив меток
        predicted: Массив предсказаний
        basemap: Базовая карта
        limits: Предельные значения
    """
    _draw_basemap(axis, basemap, limits)
    gt_anomaly = np.equal(labels, 0.0)
    pred_anomaly = np.equal(predicted, 0.0)
    accepted = ~pred_anomaly
    normal_size, hit_size, miss_size, false_size = _marker_sizes(lon.size)
    line_width = 0.5 if lon.size > 8_000 else 1.4
    _track_line(axis, lon[accepted], lat[accepted], color=_COLOR_ACCEPTED, linewidth=line_width, zorder=2)
    tp_mask = ~gt_anomaly & accepted
    tn_mask = gt_anomaly & pred_anomaly
    fn_mask = gt_anomaly & accepted
    fp_mask = ~gt_anomaly & pred_anomaly
    _scatter_mask(axis, lon, lat, tp_mask, s=normal_size, c=_COLOR_NORMAL, zorder=3, linewidths=0.15)
    _scatter_mask(axis, lon, lat, tn_mask, s=hit_size, c=_COLOR_HIT, zorder=4, linewidths=0.15)
    _scatter_mask(
        axis, lon, lat, fn_mask, s=miss_size, c=_COLOR_ANOMALY, marker="X", zorder=5, linewidths=0.6
    )
    _scatter_mask(
        axis, lon, lat, fp_mask, s=false_size, c=_COLOR_FALSE, marker="^", zorder=5, linewidths=0.15
    )
    axis.set_title("Разметка модели")
    axis.legend(
        handles=[
            Line2D([0], [0], color=_COLOR_NORMAL, marker="o", linestyle="None", label="TP — штатная верно"),
            Line2D([0], [0], color=_COLOR_HIT, marker="o", linestyle="None", label="TN — аномалия верно"),
            Line2D([0], [0], color=_COLOR_ANOMALY, marker="X", linestyle="None", label="FN — пропуск аномалии"),
            Line2D([0], [0], color=_COLOR_FALSE, marker="^", linestyle="None", label="FP — ложная тревога"),
            Line2D([0], [0], color=_COLOR_ACCEPTED, label="Маршрут, принятый моделью"),
        ],
        loc="lower right",
        framealpha=0.94,
        fontsize=8,
    )


def _reference_coordinates(
    time: np.ndarray, lat: np.ndarray, lon: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Эталонная линия: интерполяция через внутренние аномальные разрывы (как в метриках).
    Args:
        time: Массив времени
        lat: Массив широт
        lon: Массив долгот
        labels: Массив меток

    Returns:
        reference_lat: Массив широт эталона
        reference_lon: Массив долгот эталона
        reference_valid: Булев массив допустимых значений эталона
    """
    lat_ref = lat.copy()
    lon_ref = lon.copy()
    valid = np.ones(lat.shape, dtype=bool)
    anomaly = np.equal(labels, 0.0)
    if not np.any(anomaly):
        return lat_ref, lon_ref, valid
    padded = np.concatenate(([False], anomaly, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    last = lat.size - 1
    for start, end in zip(starts.tolist(), ends.tolist()):
        if start == 0 or end == last:
            valid[start : end + 1] = False
            continue
        alpha = (time[start : end + 1] - time[start - 1]) / (time[end + 1] - time[start - 1])
        lon_left = lon[start - 1]
        delta_lon = (lon[end + 1] - lon_left + 180.0) % 360.0 - 180.0
        lat_ref[start : end + 1] = lat[start - 1] + alpha * (lat[end + 1] - lat[start - 1])
        lon_ref[start : end + 1] = (lon_left + alpha * delta_lon + 180.0) % 360.0 - 180.0
    return lat_ref, lon_ref, valid


def _map_output_paths(base: Path) -> Tuple[Path, Path]:
    """
    Пути для двух карт: изначальный трек и разметка модели.
    Args:
        base: Базовый путь

    Returns:
        Tuple[Path, Path]: Пути к файлам *_ground_truth* и *_model*
    """
    suffix = base.suffix or ".png"
    stem = base.stem if base.suffix else base.name
    return (
        base.parent / f"{stem}_ground_truth{suffix}",
        base.parent / f"{stem}_model{suffix}",
    )


def _present_figure(figure: plt.Figure) -> None:
    """
    Показывает фигуру в ноутбуке без преждевременного закрытия (inline backend).
    Args:
        figure: Фигура matplotlib
    """
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(figure)
            plt.close(figure)
            return
    except Exception:
        pass
    plt.show()


def _finish_map_figure(
    figure: plt.Figure,
    suptitle: str,
    caption: str,
    output_path: Optional[Path],
    interactive: bool,
) -> None:
    """
    Общее оформление фигуры: заголовок, подпись, сохранение или показ.
    Args:
        figure: Фигура matplotlib
        suptitle: Заголовок фигуры
        caption: Подпись фигуры
        output_path: Путь для сохранения фигуры
        interactive: Флаг интерактивности
    """
    figure.suptitle(suptitle, fontsize=14)
    figure.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#4b5563")
    figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        print(f"Карта тестового трека: {output_path.resolve()}")
    elif interactive:
        _present_figure(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Пример отрисовки трека на фоне карты OSM.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Путь к PNG. Если не задан, картинка показывается интерактивно.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        default=None,
        help="Окно карты: юг запад север восток (градусы).",
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Не загружать тайлы OSM, только трек на осях.",
    )
    cli_args = parser.parse_args()

    count = 21
    demo_time = np.arange(count, dtype=np.float64)
    demo_latitude = 43.04 + 0.001 * demo_time
    demo_longitude = 131.87 + 0.001 * demo_time
    demo_labels = np.ones(count, dtype=np.float64)
    demo_labels[8:13] = 0.0
    demo_latitude = demo_latitude.copy()
    demo_longitude = demo_longitude.copy()
    demo_latitude[8:13] += np.array([0.02, 0.05, 0.08, 0.05, 0.02])
    demo_longitude[8:13] += np.array([0.01, 0.04, 0.01, -0.03, -0.01])
    demo_predicted = demo_labels.copy()
    demo_predicted[8:10] = 1.0
    demo_predicted[15] = 0.0
    demo_ground_truth = TrajectorySeries(demo_time, demo_longitude, demo_latitude, demo_labels)
    demo_prediction = TrajectorySeries(
        demo_time, demo_longitude.copy(), demo_latitude.copy(), demo_predicted
    )

    plot_track_maps(
        demo_prediction,
        demo_ground_truth,
        output_path=cli_args.output,
        bbox=cli_args.bbox,
        use_basemap=not cli_args.no_basemap,
    )
