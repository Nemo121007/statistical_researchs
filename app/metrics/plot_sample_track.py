"""Отрисовка трека на карте OpenStreetMap.

Каждый ряд — кортеж (время, долгота, широта, метка). Метка 0 — аномалия.
"""

import argparse
import io
import math
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

TrackSeries = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

_MAP_TILE_URL = "https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
_MAP_USER_AGENT = "statistical-researchs-metrics-demo/0.1"
_COLOR_NORMAL = "#24527a"
_COLOR_ANOMALY = "#b23a48"
_COLOR_REFERENCE = "#e8590c"
_COLOR_TRACK = "#071a33"
_COLOR_HIT = "#1f7a4d"
_COLOR_FALSE = "#d97706"
_COLOR_ACCEPTED = "#1f2933"
_TILE_CACHE: dict[Tuple[int, int, int], np.ndarray] = {}


def _tile_index(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Номер тайла Web Mercator, в котором лежит точка."""
    scale = 2.0**zoom
    x_tile = int((lon + 180.0) / 360.0 * scale)
    lat_rad = math.radians(lat)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale)
    last = int(scale) - 1
    return min(max(x_tile, 0), last), min(max(y_tile, 0), last)


def _tile_bounds(x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float, float, float]:
    """Границы тайла: запад, юг, восток, север в градусах."""
    scale = 2.0**zoom
    west = x_tile / scale * 360.0 - 180.0
    east = (x_tile + 1) / scale * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / scale))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y_tile + 1) / scale))))
    return west, south, east, north


def _tile_span(west: float, south: float, east: float, north: float, zoom: int) -> Tuple[int, int, int, int]:
    """Индексы тайлов, покрывающих прямоугольник: x0, y_north, x1, y_south."""
    x_west, y_north = _tile_index(west, north, zoom)
    x_east, y_south = _tile_index(east, south, zoom)
    return x_west, y_north, x_east, y_south


def _choose_zoom(west: float, south: float, east: float, north: float, max_tiles: int = 20) -> int:
    """Самый подробный масштаб, который укладывается в заданное число тайлов."""
    for zoom in range(15, -1, -1):
        x_west, y_north, x_east, y_south = _tile_span(west, south, east, north, zoom)
        if (x_east - x_west + 1) * (y_south - y_north + 1) <= max_tiles:
            return zoom
    return 0


def _download_map_tile(x_tile: int, y_tile: int, zoom: int) -> np.ndarray:
    """Скачивает один растровый тайл подложки. Повторный запрос берётся из памяти."""
    key = (zoom, x_tile, y_tile)
    cached = _TILE_CACHE.get(key)
    if cached is not None:
        return cached
    url = _MAP_TILE_URL.format(zoom=zoom, x=x_tile, y=y_tile)
    request = urllib.request.Request(url, headers={"User-Agent": _MAP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read()
    image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    _TILE_CACHE[key] = image
    return image


def _load_basemap(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    """Собирает подложку. Возвращает изображение и границы (запад, восток, юг, север)."""
    try:
        chosen = _choose_zoom(west, south, east, north) if zoom is None else zoom
        x_west, y_north, x_east, y_south = _tile_span(west, south, east, north, chosen)
        rows = []
        for y_tile in range(y_north, y_south + 1):
            row = [_download_map_tile(x_tile, y_tile, chosen) for x_tile in range(x_west, x_east + 1)]
            rows.append(np.concatenate(row, axis=1))
        image = np.concatenate(rows, axis=0)
        bound_west, _, _, bound_north = _tile_bounds(x_west, y_north, chosen)
        _, bound_south, bound_east, _ = _tile_bounds(x_east, y_south, chosen)
        return image, (bound_west, bound_east, bound_south, bound_north)
    except (OSError, ValueError) as error:
        print(f"Подложку карты загрузить не удалось ({error}). Точки нарисованы без карты.")
        return None


def _padded_limits(lon: np.ndarray, lat: np.ndarray) -> Tuple[float, float, float, float]:
    """Прямоугольник вокруг точек с полями: запад, восток, юг, север."""
    lon_span = max(float(np.max(lon) - np.min(lon)), 0.01)
    lat_span = max(float(np.max(lat) - np.min(lat)), 0.01)
    return (
        float(np.min(lon) - 0.45 * lon_span),
        float(np.max(lon) + 0.18 * lon_span),
        float(np.min(lat) - 0.22 * lat_span),
        float(np.max(lat) + 0.18 * lat_span),
    )


def _draw_basemap(axis, basemap, limits: Tuple[float, float, float, float]) -> None:
    """Кладёт карту на оси и выставляет географический масштаб."""
    if basemap is not None:
        image, extent = basemap
        west, east, south, north = extent
        axis.imshow(image, extent=(west, east, south, north), origin="upper", interpolation="bilinear", zorder=0)
    west, east, south, north = limits
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)), adjustable="box")
    axis.set_xlabel("Долгота, °")
    axis.set_ylabel("Широта, °")
    axis.tick_params(labelsize=8)


def _story_indices(labels: np.ndarray, predicted: np.ndarray) -> list[int]:
    """Индексы, которые нужны, чтобы глазами сверить пример: края, аномалия, ложные тревоги."""
    count = int(labels.size)
    chosen = {0, count - 1}
    anomaly = np.flatnonzero(np.equal(labels, 0.0))
    if anomaly.size:
        chosen.add(int(anomaly[0]) - 1)
        chosen.update(int(index) for index in anomaly)
        chosen.add(int(anomaly[-1]) + 1)
    false_positive = np.flatnonzero(~np.equal(labels, 0.0) & np.equal(predicted, 0.0))
    chosen.update(int(index) for index in false_positive)
    return sorted(index for index in chosen if 0 <= index < count)


def _annotate_indices(axis, lon: np.ndarray, lat: np.ndarray, indices: list[int]) -> None:
    """Подписывает номера точек поверх карты."""
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
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.6, foreground="white")])


def _scatter_mask(axis, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, **style) -> None:
    """Рисует подмножество точек, если оно не пустое."""
    if np.any(mask):
        style.setdefault("zorder", 4)
        style.setdefault("edgecolors", "white")
        style.setdefault("linewidths", 0.6)
        axis.scatter(lon[mask], lat[mask], **style)


def _draw_truth_map(axis, lon, lat, labels, reference_lon, reference_lat, basemap, limits) -> None:
    """Левая панель: записанный трек, аномальная петля и эталонная интерполяция."""
    _draw_basemap(axis, basemap, limits)
    axis.plot(lon, lat, color="white", linewidth=5.6, zorder=2, solid_capstyle="round")
    axis.plot(lon, lat, color=_COLOR_TRACK, linewidth=2.9, zorder=3, solid_capstyle="round")
    axis.plot(
        reference_lon,
        reference_lat,
        color=_COLOR_REFERENCE,
        linestyle=(0, (7, 4)),
        linewidth=2.8,
        zorder=4,
        solid_capstyle="round",
    )
    normal = ~np.equal(labels, 0.0)
    _scatter_mask(axis, lon, lat, normal, s=32, c=_COLOR_NORMAL, zorder=5)
    _scatter_mask(axis, lon, lat, ~normal, s=70, c=_COLOR_ANOMALY, zorder=6)
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


def _draw_prediction_map(axis, lon, lat, labels, predicted, basemap, limits) -> None:
    """Правая панель: какие точки модель оставила штатными и где ошиблась."""
    _draw_basemap(axis, basemap, limits)
    gt_anomaly = np.equal(labels, 0.0)
    pred_anomaly = np.equal(predicted, 0.0)
    accepted = ~pred_anomaly
    axis.plot(lon[accepted], lat[accepted], color=_COLOR_ACCEPTED, linewidth=1.4, zorder=2)
    _scatter_mask(axis, lon, lat, ~gt_anomaly & accepted, s=32, c=_COLOR_NORMAL, zorder=3)
    _scatter_mask(axis, lon, lat, gt_anomaly & pred_anomaly, s=70, c=_COLOR_HIT)
    _scatter_mask(axis, lon, lat, gt_anomaly & accepted, s=90, c=_COLOR_ANOMALY, marker="X", zorder=5, linewidths=1.2)
    _scatter_mask(axis, lon, lat, ~gt_anomaly & pred_anomaly, s=80, c=_COLOR_FALSE, marker="^", zorder=5)
    axis.set_title("Разметка модели")
    axis.legend(
        handles=[
            Line2D([0], [0], color=_COLOR_NORMAL, marker="o", linestyle="None", label="Верно, штатная, TP"),
            Line2D([0], [0], color=_COLOR_HIT, marker="o", linestyle="None", label="Обнаружена аномалия, TN"),
            Line2D([0], [0], color=_COLOR_ANOMALY, marker="X", linestyle="None", label="Пропуск, FN"),
            Line2D([0], [0], color=_COLOR_FALSE, marker="^", linestyle="None", label="Ложная тревога, FP"),
            Line2D([0], [0], color=_COLOR_ACCEPTED, label="Маршрут, принятый моделью"),
        ],
        loc="lower right",
        framealpha=0.94,
        fontsize=8,
    )


def _reference_coordinates(
    time: np.ndarray, lat: np.ndarray, lon: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Линейный эталон через аномальный разрыв между соседними штатными точками."""
    lat_ref = lat.copy()
    lon_ref = lon.copy()
    anomaly = np.equal(labels, 0.0)
    if not np.any(anomaly):
        return lat_ref, lon_ref
    padded = np.concatenate(([False], anomaly, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    last = lat.size - 1
    for start, end in zip(starts.tolist(), ends.tolist()):
        if start == 0 or end == last:
            continue
        alpha = (time[start : end + 1] - time[start - 1]) / (time[end + 1] - time[start - 1])
        lon_left = lon[start - 1]
        delta_lon = (lon[end + 1] - lon_left + 180.0) % 360.0 - 180.0
        lat_ref[start : end + 1] = lat[start - 1] + alpha * (lat[end + 1] - lat[start - 1])
        lon_ref[start : end + 1] = (lon_left + alpha * delta_lon + 180.0) % 360.0 - 180.0
    return lat_ref, lon_ref


def plot_sample_track(
    prediction: TrackSeries,
    ground_truth: TrackSeries,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Рисует prediction и ground truth.

    Без ``output_path`` открывает интерактивное окно. С путём сохраняет PNG по указанному адресу.
    ``title`` заменяет общий заголовок рисунка.
    """
    time, lon, lat, labels = ground_truth
    _, _, _, predicted = prediction
    time = np.asarray(time, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    limits = _padded_limits(lon, lat)
    basemap = _load_basemap(limits[0], limits[2], limits[1], limits[3])
    reference_lat, reference_lon = _reference_coordinates(time, lat, lon, labels)
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 7.2))
    _draw_truth_map(axes[0], lon, lat, labels, reference_lon, reference_lat, basemap, limits)
    _draw_prediction_map(axes[1], lon, lat, labels, predicted, basemap, limits)
    indices = _story_indices(labels, predicted)
    _annotate_indices(axes[0], lon, lat, indices)
    _annotate_indices(axes[1], lon, lat, indices)
    figure.suptitle(title or "Тестовый трек у Владивостока: точки 8–12 уходят в петлю", fontsize=13)
    figure.text(
        0.5,
        0.01,
        "Подложка: OpenStreetMap. Синтетические точки. Числа - индексы ряда.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10, wspace=0.18)
    if output_path is None:
        plt.show()
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Карта тестового трека: {output.resolve()}")
    return output


def _standalone_demo_track() -> Tuple[TrackSeries, TrackSeries]:
    """Локальный пример для запуска модуля без calculate_metrics."""
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
    ground_truth = (time, longitude, latitude, labels)
    prediction = (time, longitude.copy(), latitude.copy(), predicted)
    return prediction, ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Отрисовка тестового трека на карте OpenStreetMap.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Путь к PNG. Если не задан, картинка показывается интерактивно.",
    )
    args = parser.parse_args()
    demo_prediction, demo_ground_truth = _standalone_demo_track()
    plot_sample_track(demo_prediction, demo_ground_truth, output_path=args.output)
