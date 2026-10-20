"""Чтение и запись GeoJSON для треков и OSM-подобных коллекторов."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import geojson
import numpy as np
from geojson import Feature
from shapely.geometry import LineString, Polygon

from app.geo.area_collector import Area, AreaCollector
from app.geo.way_collector import Way, WayCollector

TrackArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


def write_tracks_geojson(output_path: Path, tracks: Sequence[TrackArrays]) -> None:
    """
    Записывает треки в GeoJSON: точки с временем и линии по каждому треку.
    Args:
        output_path: Путь к выходному файлу.
        tracks: Последовательность троек (time, lat, lon) — согласованные 1D-массивы.
    """
    features_collection = geojson.FeatureCollection([])
    point_id = 0

    for way_id, (time_array, lat_array, lon_array) in enumerate(tracks):
        line_coords: List[List[float]] = []
        for time, lat, lon in zip(time_array, lat_array, lon_array, strict=False):
            if lon is None or lat is None or np.isnan(lon) or np.isnan(lat):
                continue
            point_id += 1
            line_coords.append([float(lon), float(lat)])
            features_collection.features.append(
                Feature(
                    id=point_id,
                    geometry=geojson.Point((float(lon), float(lat))),
                    properties={"time": str(time)},
                )
            )
        if line_coords:
            features_collection.features.append(
                Feature(
                    id=way_id,
                    geometry=geojson.LineString(line_coords),
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        geojson.dump(features_collection, f, ensure_ascii=False, indent=2)
    logging.info(
        "GeoJSON файл записан в %s с %s объектами",
        str(output_path),
        len(features_collection.features),
    )


def write_collectors_geojson(
    output_path: Path,
    ways_collector: WayCollector | None = None,
    areas_collector: AreaCollector | None = None,
) -> None:
    """
    Записывает пути и полигоны из коллекторов в GeoJSON.
    Args:
        output_path: Путь к выходному файлу.
        ways_collector: Коллекция линейных объектов.
        areas_collector: Коллекция полигонов.
    """
    has_ways = ways_collector and len(ways_collector.ways) > 0
    has_areas = areas_collector and len(areas_collector.areas) > 0
    if not has_ways and not has_areas:
        raise ValueError("Нет данных для записи в GeoJSON файл")

    feature_collection = geojson.FeatureCollection([])
    if ways_collector:
        feature_collection.features.extend(_ways_to_features(ways_collector))
        logging.debug("Записано путей: %s", len(ways_collector.ways))
    if areas_collector:
        feature_collection.features.extend(_areas_to_features(areas_collector))
        logging.debug("Записано полигонов: %s", len(areas_collector.areas))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)
    logging.info(
        "GeoJSON файл записан в %s с %s объектами",
        str(output_path),
        len(feature_collection.features),
    )


def read_collectors_geojson(
    input_path: Path,
) -> Tuple[WayCollector, AreaCollector]:
    """
    Читает пути и полигоны из GeoJSON в коллекторы (геометрия Shapely).
    Args:
        input_path: Путь к входному файлу FeatureCollection.

    Returns:
        Пара (WayCollector, AreaCollector).
    """
    if not input_path or not isinstance(input_path, Path):
        raise ValueError("Путь к файлу для чтения не задан или неверного типа")
    if not input_path.exists():
        raise ValueError(f"Файл не найден: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = geojson.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error("Ошибка при чтении GeoJSON файла: %s", str(e))
        raise

    if not isinstance(data, geojson.FeatureCollection):
        raise ValueError("GeoJSON файл не содержит FeatureCollection")

    way_collector = WayCollector()
    area_collector = AreaCollector()
    for feature in data.features:
        geometry = feature.geometry
        if geometry["type"] == "LineString":
            way_id = way_collector.resolve_id(feature.id)
            way_collector.add_way(Way(way_id, LineString(geometry["coordinates"])))
        elif geometry["type"] == "Polygon":
            rings = geometry["coordinates"]
            area_id = area_collector.resolve_id(feature.id)
            area_collector.add_area(Area(area_id, Polygon(rings[0], rings[1:])))

    logging.info(
        "GeoJSON файл %s прочитан. Путей: %s, площадей: %s",
        input_path,
        len(way_collector.ways),
        len(area_collector.areas),
    )
    return way_collector, area_collector


def _ways_to_features(ways_collector: WayCollector) -> List[Feature]:
    """
    Преобразует коллекцию путей в список GeoJSON-фич.
    Args:
        ways_collector: Коллекция путей для преобразования.

    Returns:
        Список GeoJSON-фич.
    """
    features: List[Feature] = []
    for way in ways_collector.ways.values():
        coordinates = [list(coord) for coord in way.shapely_line.coords]
        features.append(
            Feature(
                id=way.id,
                boundingbox=way.bbox,
                geometry=geojson.LineString(coordinates),
                properties={"tags": way.tags},
            )
        )
    return features


def _areas_to_features(areas_collector: AreaCollector) -> List[Feature]:
    """
    Преобразует коллекцию полигонов в список GeoJSON-фич.
    Args:
        areas_collector: Коллекция полигонов для преобразования.

    Returns:
        Список GeoJSON-фич.
    """
    features: List[Feature] = []
    for area in areas_collector.areas.values():
        polygon = area.shapely_polygon
        all_rings = [list(polygon.exterior.coords)]
        all_rings.extend([list(ring.coords) for ring in polygon.interiors])
        features.append(
            Feature(
                id=area.id,
                boundingbox=area.bbox,
                geometry=geojson.Polygon(all_rings),
                properties={"tags": area.tags},
            )
        )
    return features
