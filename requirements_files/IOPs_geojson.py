# pylint: disable=invalid-name
"""Класс для операций ввода-вывода GeoJSON."""

import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import List, Tuple

import geojson
from shapely.geometry import Polygon, LineString

from . import (
    AreaCollector,
)
from .node_collector import (
    NodeCollector,
)
from .way_collector import (
    WayCollector,
)
from .area_model import Area
from .node_model import Node
from .way_model import Way


class IOPs_geojson:
    """Класс для операций ввода-вывода GeoJSON.
    write_geojson - запись данных в GeoJSON файл. Предназначен для парсинга.
    read_geojson - чтение данных из GeoJSON файла. Предназначен для парсинга.
    fast_read_json - быстрое чтение данных из GeoJSON файла без создания всех узлов и путей.
        Предназначен для реального пользования
    """

    def __init__(self):
        pass

    @staticmethod
    def write_geojson(
        file_output_path: Path,
        nodes_collector: NodeCollector = None,
        areas_collector: AreaCollector = None,
        ways_collector: WayCollector = None,
        list_print_points: List[Node] = None,
    ) -> None:
        """Записывает данные в GeoJSON файл.
        В первую очередь предназначен для парсинга
        Args:
            file_output_path: Путь к выходному файлу.
            nodes_collector: Коллекция узлов для записи.
            areas_collector: Коллекция площадей для записи.
            ways_collector: Коллекция путей для записи.
            list_print_points: Список точек для записи.
        Raises:
            ValueError: Если путь к файлу не задан или нет данных для записи.
            OSError: Если произошла ошибка при записи файла.
        """
        if (  # pylint: disable=too-many-boolean-expressions
            (not nodes_collector or len(nodes_collector.nodes) == 0)
            and (not ways_collector or len(ways_collector.ways) == 0)
            and (not areas_collector or len(areas_collector.areas) == 0)
            and (not list_print_points or len(list_print_points) == 0)
        ):
            raise ValueError("Нет данных для записи в GeoJSON файл")

        feature_collection = geojson.FeatureCollection([])

        if ways_collector:
            list_features = IOPs_geojson._convert_way_collector_to_list_features(ways_collector)
            feature_collection.features.extend(list_features)
            logging.debug(f"Записано путей: {len(ways_collector.ways)}")

        if areas_collector:
            list_features = IOPs_geojson._convert_area_collector_to_list_features(areas_collector)
            feature_collection.features.extend(list_features)
            logging.debug(f"Записано полигонов: {len(areas_collector.areas)}")

        if list_print_points:
            list_features = IOPs_geojson._convert_list_point_to_list_features(list_print_points)
            feature_collection.features.extend(list_features)
            logging.debug(f"Записано точек: {len(list_print_points)}")

        try:
            file_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_output_path, "w", encoding="utf-8") as f:
                geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)

            logging.info(
                f"GeoJSON файл записан в {str(file_output_path)} "
                f"с {len(feature_collection.features)} объектами"
            )
        except OSError as e:
            logging.error(f"Ошибка при записи GeoJSON файла: {str(e)}")
            raise

    @staticmethod
    def _convert_way_collector_to_list_features(
        ways_collector: WayCollector,
    ) -> List[geojson.Feature]:
        """Преобразует коллекцию путей в список GeoJSON фич.
        Args:
            way_collector: Коллекция путей для преобразования.
        Returns:
            Список GeoJSON фич.
        """
        features: List[geojson.Feature] = []
        for way in ways_collector.ways.values():
            coordinates = [(node.lon, node.lat) for node in way.nodes]
            list_node_id_from_way = [node.id for node in way.nodes]
            properties = {"OSM_id_nodes": list_node_id_from_way, "tags": way.tags}
            feature = geojson.Feature(
                id=way.id,
                boundingbox=(way.min_lon, way.min_lat, way.max_lon, way.max_lat),
                geometry=geojson.LineString(coordinates),
                properties=properties,
            )
            features.append(feature)
        return features

    @staticmethod
    def _convert_area_collector_to_list_features(
        areas_collector: AreaCollector,
    ) -> List[geojson.Feature]:
        features: List[geojson.Feature] = []
        for area in areas_collector.areas.values():
            if len(area.outer_border) < 3:
                logging.warning(
                    f"Пропущена площадь {area.id} - внешняя граница содержит менее 3 узлов"
                )
                continue
            if area.outer_border[0] != area.outer_border[-1]:
                logging.warning(
                    f"Пропущена площадь {area.id} - внешняя граница не замкнута. Исправление..."
                )
                area.outer_border.append(area.outer_border[0])
            all_rings = [tuple((node.lon, node.lat) for node in area.outer_border)]
            list_node_id_from_area = [[node.id for node in area.outer_border]]
            for inner_border in area.inner_borders:
                if len(inner_border) < 3:
                    logging.warning(
                        f"Пропущена внутренняя граница в площади {area.id} "
                        f"- содержит менее 3 узлов"
                    )
                    continue
                if inner_border[0] != inner_border[-1]:
                    logging.warning(
                        f"Пропущена внутренняя граница в площади {area.id} "
                        f"- не замкнута. Исправление..."
                    )
                    inner_border.append(inner_border[0])
                inner_coordinates = tuple((node.lon, node.lat) for node in inner_border)
                list_node_id_from_area.append([node.id for node in inner_border])
                all_rings.append(inner_coordinates)
            properties = {"OSM_id_nodes": list_node_id_from_area, "tags": area.tags}
            feature = geojson.Feature(
                id=area.id,
                boundingbox=(
                    area.min_lon,
                    area.min_lat,
                    area.max_lon,
                    area.max_lat,
                ),
                geometry=geojson.Polygon(all_rings),
                properties=properties,
            )
            features.append(feature)
        return features

    @staticmethod
    def _convert_list_point_to_list_features(
        list_points: List[Node],
    ) -> List[geojson.Feature]:
        features = []
        for node in list_points:
            feature = geojson.Feature(
                id=node.id,
                geometry=geojson.Point((node.lon, node.lat)),
            )
            features.append(feature)
        return features

    @staticmethod
    def read_geojson(
        file_read_path: Path,
    ) -> Tuple[NodeCollector, WayCollector, AreaCollector]:
        """Читает данные из GeoJSON файла.
        Args:
            file_read_path: Путь к входному файлу.
        Returns:
            Кортеж из трех элементов: (NodeCollector, WayCollector, AreaCollector).
        Raises:
            ValueError: Если путь к файлу не задан, файл не найден или формат неверен.
            OSError: Если произошла ошибка при чтении файла.
        """
        if not file_read_path or not isinstance(file_read_path, Path):
            raise ValueError("Путь к файлу для чтения не задан или неверного типа")
        if not file_read_path.exists():
            raise ValueError(f"Файл не найден: {file_read_path}")

        try:
            with open(file_read_path, "r", encoding="utf-8") as f:
                data = geojson.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise JSONDecodeError("Ошибка при чтении GeoJSON файла", doc=str(e), pos=0) from e

        if not isinstance(data, geojson.FeatureCollection):
            raise ValueError("GeoJSON файл не содержит FeatureCollection")

        node_collector = NodeCollector()
        way_collector = WayCollector()
        area_collector = AreaCollector()
        for feature in data.features:
            feature: geojson.Feature = feature
            if feature.geometry["type"] == "LineString":
                way_id = feature.id
                coordinates = feature.geometry["coordinates"]
                id_nodes = feature.properties.get("OSM_id_nodes", [])
                if len(coordinates) != len(id_nodes):
                    t = feature.properties.get("OSM_id", -1)
                    logging.warning(
                        f"Количество координат и ID узлов не совпадает " f"для пути OSM_id={t}"
                    )
                    logging.error(f"Ошибка обработки точек пути: {id}\nПропуск пути")
                nodes: List[Node] = []
                for i, coordinate in enumerate(coordinates):
                    if node_collector.nodes.get(id_nodes[i]):
                        node = node_collector.nodes[id_nodes[i]]
                    else:
                        node = Node(node_id=id_nodes[i], lon=coordinate[0], lat=coordinate[1])
                        node_collector.add_node(node)
                    nodes.append(node)
                tags = feature.properties.get("tags", {})
                way = Way(way_id=way_id, tags=tags, nodes=nodes)

                for node in way.nodes:
                    node.add_way(way)

                way_collector.add_way(way)

            elif feature.geometry["type"] == "Polygon":
                area_id = feature.id

                all_rings = feature.geometry["coordinates"]
                id_nodes_rings = feature.properties.get("OSM_id_nodes", [])

                outer_ring_coords = all_rings[0]
                outer_ring_ids = id_nodes_rings[0]

                if len(outer_ring_coords) != len(outer_ring_ids):
                    logging.warning(
                        f"Количество координат и ID узлов не совпадает для внешней границы "
                        f"площади OSM_id={area_id}"
                    )
                    logging.error(
                        f"Ошибка обработки точек внешней границы площади: "
                        f"{area_id}\n Пропуск площади"
                    )
                    continue

                outer_border: List[Node] = []
                for i, coordinate in enumerate(outer_ring_coords):
                    if node_collector.nodes.get(outer_ring_ids[i], None):
                        node = node_collector.nodes[outer_ring_ids[i]]
                    else:
                        node = Node(
                            node_id=outer_ring_ids[i],
                            lon=coordinate[0],
                            lat=coordinate[1],
                        )
                        node_collector.add_node(node)
                    outer_border.append(node)

                inner_borders: List[List[Node]] = []
                for j in range(1, len(all_rings)):
                    inner_ring_coords = all_rings[j]
                    inner_ring_ids = id_nodes_rings[j]

                    if len(inner_ring_coords) != len(inner_ring_ids):
                        logging.warning(
                            f"Количество координат и ID узлов не совпадает "
                            f"для внутренней границы {j} площади OSM_id={area_id}",
                        )
                        logging.error(
                            f"Ошибка обработки точек внутренней границы площади: {area_id}\n"
                            f"Пропуск внутренней границы",
                        )
                        continue

                    inner_border: List[Node] = []
                    for i, coordinate in enumerate(inner_ring_coords):
                        node = Node(
                            node_id=inner_ring_ids[i],
                            lon=coordinate[0],
                            lat=coordinate[1],
                        )
                        inner_border.append(node)
                    inner_borders.append(inner_border)

                tags = feature.properties.get("tags", {})
                area = Area(
                    area_id=area_id,
                    tags=tags,
                    outer_border=outer_border,
                    inner_borders=inner_borders,
                )

                for node in area.outer_border:
                    node.add_area(area)
                for inner_border in area.inner_borders:
                    for node in inner_border:
                        node.add_area(area)

                area_collector.add_area(area)

        logging.info(
            f"GeoJSON файл {file_read_path} прочитан. "
            f"Узлов: {len(node_collector.nodes)} "
            f"Путей: {len(way_collector.ways)} "
            f"Площадей: {len(area_collector.areas)}"
        )
        return node_collector, way_collector, area_collector

    @staticmethod
    def fast_read_json(
        file_read_path: Path,
    ) -> Tuple[NodeCollector, WayCollector, AreaCollector]:
        """Быстрое чтение данных из GeoJSON файла без создания всех узлов и путей.
        Предназначен для реального пользования.
        Args:
            file_read_path: Путь к входному файлу.
        Returns:
            Кортеж из трех элементов: (NodeCollector, WayCollector, AreaCollector).
        Raises:
            ValueError: Если путь к файлу не задан, файл не найден или формат неверен.
            OSError: Если произошла ошибка при чтении файла.
        """
        if not file_read_path or not isinstance(file_read_path, Path):
            raise ValueError("Путь к файлу для чтения не задан или неверного типа")
        if not file_read_path.exists():
            raise ValueError(f"Файл не найден: {file_read_path}")

        try:
            with open(file_read_path, "r", encoding="utf-8") as f:
                data = geojson.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logging.error(f"Ошибка при чтении GeoJSON файла: {str(e)}")
            raise

        if not isinstance(data, geojson.FeatureCollection):
            raise ValueError("GeoJSON файл не содержит FeatureCollection")

        node_collector = NodeCollector()
        way_collector = WayCollector()
        area_collector = AreaCollector()
        for feature in data.features:
            feature: geojson.Feature = feature
            if feature.geometry["type"] == "LineString":
                way_id = feature.id
                coordinates = feature.geometry["coordinates"]

                way = Way(way_id=way_id)
                way.shapely_line_string = LineString(coordinates)
                way_collector.add_way(way)

            elif feature.geometry["type"] == "Polygon":
                area_id = feature.id
                all_rings = feature.geometry["coordinates"]

                polygon = Polygon(all_rings[0], all_rings[1:])

                area = Area(area_id=area_id)
                area.shapely_polygon = polygon
                area_collector.add_area(area)

        logging.info(
            f"GeoJSON файл {file_read_path} прочитан. "
            f"Узлов: {len(node_collector.nodes)}, "
            f"Путей: {len(way_collector.ways)}, "
            f"Площадей: {len(area_collector.areas)}",
        )
        return node_collector, way_collector, area_collector
