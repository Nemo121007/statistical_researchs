"""Модуль для обработки и объединения GeoJSON файлов,
включая выделение и объединение береговых линий,
а также удаление дублирующихся объектов на основе тегов OSM."""

import logging
import random
from typing import List

import geojson
import shapely
import numpy as np
from shapely.prepared import prep
from shapely.geometry import Polygon, shape
from geojson import Feature, FeatureCollection, coords

from application.modules.bg_services.gps.utils.IOPs.IOPs_geojson import IOPs_geojson
from application.modules.bg_services.gps.corrector.tracker.models.area_model import Area
from application.modules.bg_services.gps.corrector.tracker.models.node_model import Node
from application.modules.bg_services.gps.corrector.tracker.collectors.area_collector import (
    AreaCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.node_collector import (
    NodeCollector,
)
from application.modules.bg_services.gps.utils.settings import (
    TagsOSM,
    DefaultLocate,
    DefaultLocateRegionGeoJson,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Принудительная переконфигурация
)


class ParceGeojson:
    """Класс для обработки и объединения GeoJSON файлов,
    включая выделение и объединение береговых линий,
    а также удаление дублирующихся объектов на основе тегов OSM."""

    @staticmethod
    def concat_geojson_features(
        list_feature_collections: List[FeatureCollection],
        rewrite_duplicates: bool = True,
        adding_duplicates: bool = False,
    ) -> FeatureCollection:
        """Объединяет несколько коллекций GeoJSON в одну,
        удаляя дублирующиеся объекты на основе тегов OSM.
        Каждый последующий объект при дублировании перезаписывает прошлый объект.
        Args:
            list_feature_collections: список коллекций GeoJSON для объединения.
        """
        if not isinstance(list_feature_collections, list):
            raise TypeError("Ожидается список FeatureCollection")
        if not all(isinstance(fc, FeatureCollection) for fc in list_feature_collections):
            raise TypeError("Все элементы должны быть типа FeatureCollection")

        dict_areas_collectors = {}
        dict_ways_collectors = {}
        for feature_collection in list_feature_collections:
            for feature in feature_collection.features:
                if feature.geometry.type == "LineString":
                    key = feature.properties.get("tags").get("OSM_way_id")
                    if key is None:
                        raise ValueError(f"Отсутствует OSM_way_id в свойствах: {feature.id}")
                    if key not in dict_ways_collectors:
                        dict_ways_collectors[key] = feature
                    elif rewrite_duplicates:
                        # dict_ways_collectors[key] = feature
                        logging.debug(f"Дублирование OSM_way_id найдено и перезаписано: {key}")
                    elif adding_duplicates:
                        new_key = key
                        while new_key in dict_ways_collectors:
                            int(random.randint(1, 1000000000000000))
                        dict_ways_collectors[new_key] = feature
                        logging.debug(f"Дублирование OSM_way_id найдено и добавлено с новым ключом: {new_key}")
                    else:
                        logging.debug(f"Дублирование OSM_way_id найдено и пропущено: {key}")
                elif feature.geometry.type == "Polygon":
                    key = feature.properties.get("tags").get("OSM_area_id")
                    if key is None:
                        raise ValueError(f"Отсутствует OSM_area_id в свойствах: {feature.id}")
                    if key not in dict_areas_collectors:
                        dict_areas_collectors[key] = feature
                    elif rewrite_duplicates:
                        # dict_areas_collectors[key] = feature
                        logging.debug(f"Дублирование OSM_area_id найдено и перезаписано: {key}")
                    elif adding_duplicates:
                        new_key = key
                        while new_key in dict_areas_collectors:
                            new_key = int(random.randint(1, 1000000000000))
                        dict_areas_collectors[new_key] = feature
                        logging.debug(f"Дублирование OSM_area_id найдено и добавлено с новым ключом: {new_key}")
                    else:
                        logging.debug(f"Дублирование OSM_area_id найдено и пропущено: {key}")

        logging.info("Объединение объектов...")
        feature_collection = FeatureCollection([])
        for feature in dict_ways_collectors.values():
            feature.id = len(feature_collection.features)
            feature_collection.features.append(feature)
        for feature in dict_areas_collectors.values():
            feature.id = len(feature_collection.features)
            feature_collection.features.append(feature)
        logging.debug(f"Итоговое количество объектов: {len(feature_collection.features)}")
        return feature_collection

    @staticmethod
    def concat_coastline(list_feature_collections: List[FeatureCollection], output_path=None) -> FeatureCollection:
        """Объединяет береговые линии из нескольких коллекций GeoJSON в одну.
        Args:
            list_feature_collections: список коллекций GeoJSON дляобъединения.
            output_path: путь для сохранения объединенного GeoJSON файла.
        Returns:
            Объединенная коллекция береговых линий.
        """
        if output_path is None:
            output_path = DefaultLocate.OUTPUT_DIR / "temp_map.geojson"
        unity_feature_collection = ParceGeojson.concat_geojson_features(list_feature_collections)
        unity_coastlines = ParceGeojson.extracting_coastline(unity_feature_collection)
        with open(output_path, "w", encoding="utf-8") as file:
            geojson.dump(unity_coastlines, file, ensure_ascii=False, indent=2)
            logging.info(f"Сохранен объединенный файл GeoJSON: {str(output_path)}")
            logging.debug(f"Итоговое количество объектов: {len(unity_coastlines.features)}")

        coastline = ParceGeojson.concat_coastline_sea(feature_collection=unity_coastlines, ind_start_line=711)
        logging.info(f"Количество объектов 'costline' после объединения: {len(coastline.features)}")
        output_path = DefaultLocate.OUTPUT_DIR / "coastline_merged.geojson"
        with open(output_path, "w", encoding="utf-8") as file:
            geojson.dump(coastline, file, ensure_ascii=False, indent=2)
            logging.info(f"Сохранен объединенный файл GeoJSON: {str(output_path)}")
            logging.debug(f"Итоговое количество объектов: {len(coastline.features)}")
        return coastline

    @staticmethod
    def extracting_coastline(
        feature_collection: FeatureCollection,
    ) -> FeatureCollection:
        """Выделяет береговые линии из коллекции GeoJSON.
        Args:
            feature_collection: коллекция GeoJSON для обработки.
        Returns:
            Коллекция береговых линий.
        """
        coastline_features = FeatureCollection([])
        for feature in feature_collection.features:
            if feature.geometry.type != "LineString":
                continue
            tags = feature.properties.get("tags", {})
            if tags.get("natural") == "coastline":
                coastline_features.features.append(feature)
        logging.info(f"Количество выделенных объектов 'costline': {len(coastline_features.features)}")
        return coastline_features

    @staticmethod
    def concat_coastline_sea(feature_collection: FeatureCollection, ind_start_line: int = 711) -> FeatureCollection:
        """Объединяет береговые линии в одну непрерывную линию.
        Args:
            feature_collection: коллекция береговых линий для объединения.
            ind_start_line: индекс начальной линии для объединения.
        Returns:
            Коллекция с объединенной береговой линией.
        """
        dict_line = {}
        for feature in feature_collection.features:
            dict_line[feature.id] = feature

        result_line = []
        result_line_ids = []

        line = dict_line[ind_start_line].geometry.coordinates
        line.reverse()
        result_line.extend(line)
        line = dict_line[ind_start_line].properties.get("OSM_id_nodes")
        line.reverse()
        result_line_ids.extend(line)
        del dict_line[ind_start_line]

        flag_chance = True
        while flag_chance:
            flag_chance = False
            for feature in dict_line.values():
                if result_line_ids[-1] == feature.properties.get("OSM_id_nodes")[0]:
                    list_point: List = feature.geometry.coordinates
                    list_point_ids: list = feature.properties.get("OSM_id_nodes")

                    if feature.id == 144:
                        list_point = list_point[::-1]
                        list_point_ids = list_point_ids[::-1]

                    result_line.extend(list_point[1:])
                    result_line_ids.extend(list_point_ids[1:])
                    flag_chance = True
                    logging.debug(
                        f"Объединение объектов: {result_line}"
                        f"(len: {len(result_line)}) и "
                        f"{feature.id}(len: {len(list_point)})"
                    )
                    del dict_line[feature.id]
                    break
                if result_line_ids[-1] == feature.properties.get("OSM_id_nodes")[-1]:
                    list_point: List = feature.geometry.coordinates
                    list_point_ids = feature.properties.get("OSM_id_nodes")

                    list_point.reverse()
                    list_point_ids.reverse()

                    result_line.extend(list_point[1:])
                    result_line_ids.extend(list_point_ids[1:])
                    flag_chance = True
                    logging.debug(
                        f"Объединение объектов: len: {len(result_line)} и " f"{feature.id}(len: {len(list_point)})"
                    )
                    del dict_line[feature.id]
                    break

        new_feature = Feature(
            id=ind_start_line,
            geometry=geojson.LineString(result_line),
            properties={
                "OSM_id_nodes": result_line_ids,
                "tags": {
                    "natural": "coastline",
                    "origin": "manual cast",
                },
            },
        )
        dict_line[ind_start_line] = new_feature

        feature_collection = FeatureCollection([])
        for feature in dict_line.values():
            feature_collection.features.append(feature)

        return feature_collection

    @staticmethod
    def remove_feature_from_tag(
        features_collection: FeatureCollection, key_tag: str, value_tag: str
    ) -> (FeatureCollection, FeatureCollection):
        """Удаляет объекты из коллекции GeoJSON на основе заданного тега и его значения."""
        feature_collections = FeatureCollection([])
        removed_features = FeatureCollection([])
        for feature in features_collection.features:
            tags = feature.properties.get("tags", {})
            if tags.get(key_tag) == value_tag:
                logging.debug(f"Удаление объекта по тегу {key_tag}:{value_tag} - {feature.id}")
                removed_features.features.append(feature)
            else:
                feature_collections.features.append(feature)
        return feature_collections, removed_features

    @staticmethod
    def remove_duplicate_line(
        features_collection: FeatureCollection,
    ) -> FeatureCollection:
        """Удаляет дублирующиеся линии, покрытые полигонами из коллекции GeoJSON.
        Args:
            list_features: список объектов GeoJSON дляобработки.
        Returns:
            Коллекция GeoJSON без дублирующихся линий.
        """
        logging.debug("Начало удаления дублирующихся линий...")
        result_dict_ways = {}
        dict_ways_collectors = {}
        dict_areas_collectors = {}
        for feature in features_collection.features:
            if feature.geometry.type == "LineString":
                dict_ways_collectors[feature.id] = feature
            elif feature.geometry.type == "Polygon":
                dict_areas_collectors[feature.id] = feature

        prep_shapely_ways = {}
        for way in dict_ways_collectors.values():
            shapely_geom = shape(way.geometry)
            prep_shapely_ways[way.id] = prep(shapely_geom)
        prep_shapely_areas = {}
        for area in dict_areas_collectors.values():
            shapely_geom = shape(area.geometry)
            prep_shapely_areas[area.id] = prep(shapely_geom)

        for way in dict_ways_collectors.values():
            for area in dict_areas_collectors.values():
                shapely_area = prep_shapely_areas[area.id]
                polygon_way = shapely.LineString(way.geometry.coordinates)
                if shapely_area.covers(polygon_way) and not prep_shapely_areas[area.id].contains(polygon_way):
                    logging.debug(f"Удаление дублирующей линии, покрытой полигоном: {way.id}")
                    break
            else:
                result_dict_ways[way.id] = way

        result_feature_collections = FeatureCollection([])
        for way in result_dict_ways.values():
            result_feature_collections.features.append(way)
        for area in dict_areas_collectors.values():
            result_feature_collections.features.append(area)
        return result_feature_collections

    @staticmethod
    def remove_duplicate_features(feature_collection: FeatureCollection) -> FeatureCollection:
        """Удаляет дублирующиеся объекты из коллекции GeoJSON на основе их геометрии."""
        unique_features = {}
        for feature in feature_collection.features:
            geom_wkt = shape(feature.geometry).wkt
            if geom_wkt not in unique_features:
                unique_features[geom_wkt] = feature
            else:
                logging.debug(f"Удаление дублирующего объекта: {feature.id}")

        result_feature_collections = FeatureCollection([])
        false_features = FeatureCollection([])
        for feature in unique_features.values():
            try:
                result_feature_collections.features.append(
                    Feature(
                        type=feature.type,
                        boundingbox=feature.boundingbox,
                        id=len(result_feature_collections.features),
                        geometry=feature.geometry,
                        properties=feature.properties,
                    )
                )
            except:
                false_features.features.append(feature)

        with open(
            DefaultLocate.OUTPUT_DIR / "false_features.geojson",
            "w",
            encoding="utf-8",
        ) as f:
            geojson.dump(false_features, f, ensure_ascii=False, indent=2)
            logging.info(
                f"Сохранен файл GeoJSON с ложными объектами: {str(DefaultLocate.OUTPUT_DIR / 'false_features.geojson')}"
            )
            logging.debug(f"Итоговое количество ложных объектов: {len(false_features.features)}")

        return result_feature_collections

    @staticmethod
    def remove_feature_in_inner_boards(
        feature_collection: FeatureCollection,
    ) -> FeatureCollection:
        """Удаляет объекты, находящиеся внутри внутренних границ полигонов"""
        dict_features = {}
        for feature in feature_collection.features:
            if feature.geometry.type == "Polygon":
                dict_features[feature.id] = feature

        result_dict_search_features = dict_features.copy()
        for feature in dict_features.values():
            if len(feature.geometry.coordinates) > 1:
                for inner_board in feature.geometry.coordinates[1:]:
                    inner_polygon = Polygon(inner_board)
                    for search_feature in dict_features.values():
                        if len(search_feature.geometry.coordinates) != 1:
                            continue
                        search_polygon = Polygon(search_feature.geometry.coordinates[0])
                        if (
                            search_feature.id in result_dict_search_features
                            and inner_polygon.covers(search_polygon)
                            and not inner_polygon.contains(search_polygon)
                        ):
                            del result_dict_search_features[search_feature.id]
                            logging.debug(f"Удаление дублирования внутренней границы: {search_feature.id}")
        result_feature_collection = FeatureCollection([])
        for feature in feature_collection.features:
            if feature.geometry.type != "Polygon":
                result_feature_collection.features.append(feature)
            elif feature.id in result_dict_search_features:
                result_feature_collection.features.append(feature)
        return result_feature_collection

    @staticmethod
    def maker_easier(feature_collection: FeatureCollection) -> FeatureCollection:
        result_feature_collections = FeatureCollection([])
        for feature in feature_collection.features:
            new_feature = Feature(
                type=feature.type,
                boundingbox=feature.boundingbox,
                id=feature.id,
                geometry=feature.geometry,
            )
            result_feature_collections.features.append(new_feature)
        return result_feature_collections

    @staticmethod
    def split_to_grid(feature: geojson.Feature, cell_size: float) -> FeatureCollection:
        """Разбивает полигональный объект GeoJSON на сетку ячеек заданного размера."""
        feature_collection = FeatureCollection([])

        # Получаем геометрию из Feature
        geom = feature.geometry
        if geom.type != "Polygon":
            raise ValueError("Geometry type must be Polygon")

        # Преобразуем GeoJSON-координаты в Shapely-полигон
        exterior = [tuple(coord) for coord in geom.coordinates[0]]
        interiors = [[tuple(coord) for coord in ring] for ring in geom.coordinates[1:]]
        polygon = shapely.geometry.Polygon(exterior, interiors)

        tags = feature.properties if feature.properties else {}
        if tags.get('OSM_id_nodes'):
            del tags['OSM_id_nodes']

        min_x, min_y, max_x, max_y = polygon.bounds
        feature_id = 1  # Счетчик для ID

        x_coords = np.arange(min_x, max_x + cell_size, cell_size)
        y_coords = np.arange(min_y, max_y + cell_size, cell_size)

        for x in x_coords[:-1]:
            for y in y_coords[:-1]:
                cell = shapely.geometry.Polygon(
                    [
                        (x, y),
                        (x + cell_size, y),
                        (x + cell_size, y + cell_size),
                        (x, y + cell_size),
                    ]
                )

                # Обрезаем ячейку по границам полигона
                intersection = cell.intersection(polygon)
                if intersection.is_empty:
                    continue

                if intersection.geom_type == "Polygon":
                    geometries = [intersection]
                elif intersection.geom_type == "MultiPolygon":
                    geometries = list(intersection.geoms)
                else:
                    continue  # Пропускаем не-полигональные результаты

                for geom in geometries:
                    # Преобразуем Shapely-геометрию в GeoJSON
                    exterior_coords = list(geom.exterior.coords)
                    interior_coords = [list(interior.coords) for interior in geom.interiors]

                    # Формируем GeoJSON-координаты
                    if interior_coords:
                        geojson_coords = [exterior_coords] + interior_coords
                    else:
                        geojson_coords = [exterior_coords]

                    # Создаем GeoJSON-полигон
                    geojson_poly = geojson.Polygon(geojson_coords)

                    min_lat, max_lat, min_lon, max_lon = None, None, None, None

                    for line in geojson_poly.coordinates:
                        for lon, lat in line:
                            if min_lat is None or lat < min_lat:
                                min_lat = lat
                            if max_lat is None or lat > max_lat:
                                max_lat = lat
                            if min_lon is None or lon < min_lon:
                                min_lon = lon
                            if max_lon is None or lon > max_lon:
                                max_lon = lon

                    # Создаем Feature
                    new_feature = geojson.Feature(
                        id=feature_id,
                        geometry=geojson_poly,
                        properties=tags.copy(),
                        boundingbox=(min_lon, min_lat, max_lon, max_lat),
                    )
                    feature_collection.features.append(new_feature)
                    feature_id += 1

        return feature_collection

    @staticmethod
    def print_tags(feature_collection: FeatureCollection) -> None:
        """Выводит уникальные теги и их значения из коллекции GeoJSON.
        Args:
            feature_collection: коллекция GeoJSON для обработки.
        """
        tags_collector = {}
        for feature in feature_collection.features:
            if feature.geometry.type == "Polygon":
                continue
            tags = feature.properties.get("tags", {})

            if tags.get("material") == "concrete":
                pass

            for key, value in feature.properties.get("tags").items():
                # if value in TagsOSM.WHITE_LIST.get(key, []):
                #     continue
                # if any(keyword in str(key) for keyword in TagsOSM.SKIP_KEYWORDS):
                #     continue
                if key not in tags_collector:
                    tags_collector[key] = [value]
                elif value not in tags_collector[key]:
                    tags_collector[key].append(value)

        for key, value in tags_collector.items():
            print(f"{key} : {value}\n")

    def construct_sea(self):
        """Конструирует полигон моря на основе береговой линии."""
        # pylint: disable=redefined-outer-name
        path = DefaultLocate.OUTPUT_DIR / "Балтика_линия.geojson"
        with open(path, "r", encoding="utf-8") as f:
            data: FeatureCollection = geojson.load(f)
            logging.info(f"Загружен файл GeoJSON: {path}")

        # list_line = [
        #     geojson.LineString(coordinates=[(12.570572, 56.062064), (12.635993, 56.096688)]),
        #     geojson.LineString(coordinates=[(10.803766, 55.341511), (11.190707, 55.462733)]),
        #     geojson.LineString(coordinates=[(9.741694, 55.520021), (9.757104, 55.516913)])
        # ]
        # listr_features = [
        #     Feature(
        #         id=-1,
        #         geometry=list_line[0],
        #         properties={
        #             "tags": {
        #                 "origin": "manual cast",
        #                 "natural": "coastline",
        #             },
        #             "OSM_id_nodes": [25094861, 6721887050],
        #         },
        #     ),
        #     Feature(
        #         id=-2,
        #         geometry=list_line[1],
        #         properties={
        #             "tags": {
        #                 "origin": "manual cast",
        #                 "natural": "coastline",
        #             },
        #             "OSM_id_nodes": [22631324, 1751367225],
        #         },
        #     ),
        #     Feature(
        #         id=-3,
        #         geometry=list_line[2],
        #         properties={
        #             "tags": {
        #                 "origin": "manual cast",
        #                 "natural": "coastline",
        #             },
        #             "OSM_id_nodes": [22558322, 9575022836],
        #         },
        #     ),
        # ]
        #
        # if list_line is not None:
        #     feature_collection = FeatureCollection(listr_features)
        #     with open (
        #         DefaultLocate.OUTPUT_DIR / "temp_line.geojson",
        #         "w",
        #         encoding="utf-8",
        #     ) as f:
        #         geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)
        #         logging.info(
        #             f"Сохранен временный файл GeoJSON: {str(DefaultLocate.OUTPUT_DIR / 'temp_line.geojson')}"
        #         )

        # data.features.extend(listr_features)

        # data = ParceGeojson.extracting_coastline(data)

        feature_collection = FeatureCollection([])
        for feature in data.features:
            if (
                feature.id
                == 16261
                # or feature.id == 595 or
                # feature.id == 337 or feature.id == 260 or
                # feature.id == 2059 or feature.id == 952
            ) and feature.geometry.type == "LineString":
                continue
            feature_collection.features.append(feature)
        data = feature_collection

        data = ParceGeojson.concat_coastline_sea(feature_collection=data, ind_start_line=1)

        feature_collection = FeatureCollection([])
        for feature in data.features:
            if not (feature.properties.get("tags").get("origin") == "manual cast"):
                continue
            else:
                feature_collection.features.append(feature)
        data = feature_collection

        with open(
            DefaultLocate.OUTPUT_DIR / " <Балтика_merged.geojson",
            "w",
            encoding="utf-8",
        ) as f:
            geojson.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(
                f"Сохранен объединенный файл GeoJSON: {str(DefaultLocate.OUTPUT_DIR / 'coastline_merged.geojson')}"
            )
            logging.debug(f"Итоговое количество объектов: {len(data.features)}")

        list_coord = []
        for coord in data.features[0].geometry.coordinates:
            list_coord.append(coord)
        data = FeatureCollection(
            [
                Feature(
                    id=-1,
                    geometry=geojson.Polygon(coordinates=[list_coord]),
                    properties={
                        "tags": {
                            "origin": "manual cast",
                            "name": "Балтийское море",
                            "name:en": "Baltic sea",
                            "natural": "water",
                            "water": "sea",
                            "OSM_area_id": -1,
                        },
                    },
                )
            ]
        )
        with open(
            DefaultLocate.OUTPUT_DIR / "Балтика_merged.geojson",
            "w",
            encoding="utf-8",
        ) as f:
            geojson.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(
                f"Сохранен объединенный файл GeoJSON: {str(DefaultLocate.OUTPUT_DIR / 'coastline_merged.geojson')}"
            )
            logging.debug(f"Итоговое количество объектов: {len(data.features)}")

        # temp_feature = Feature()
        # for feature in data.features:
        #     if feature.geometry.type == "LineString" and feature.id == 140:
        #         temp_feature = feature
        #         break
        #
        # outer_node = []
        # node_collector = NodeCollector()
        # for i, coord in enumerate(temp_feature.geometry.coordinates):
        #     id_node = temp_feature.properties["OSM_id_nodes"][i]
        #     node = Node(
        #         node_id=id_node,
        #         lat=coord[1],
        #         lon=coord[0],
        #     )
        #     outer_node.append(node)
        #     node_collector.add_node(node)
        # area = Area(
        #     area_id=-1,
        #     tags={
        #         "origin": "manual cast",
        #         "name": "Азовское море",
        #         "name:en": "Azov sea",
        #         "natural": "water",
        #         "water": "sea",
        #         "OSM_area_id": -1,
        #     },
        #     outer_border=outer_node,
        # )
        # area_collector = AreaCollector()
        # area_collector.add_area(area)
        # path = DefaultLocate.OUTPUT_DIR / "merge_1.geojson"
        # IOPs_geojson.write_geojson(
        #     file_output_path=path, areas_collector=area_collector
        # )


if __name__ == "__main__":
    # a = ParceGeojson()
    # a.construct_sea()
    # exit(0)

    # data = FeatureCollection([
    #     Feature(
    #         id=-1,
    #         boundingbox=(36.653310, 45.368423, 36.653897, 45.42181),
    #         geometry=geojson.LineString([(36.653897, 45.368423), (36.653310, 45.421813)]),
    #         properties=
    #         {
    #             "origin": "manual cast",
    #             "name": "затычка",
    #         },
    #     ),
    #     Feature(
    #         id=-2,
    #         boundingbox=(36.678276, 45.368423, 36.678863, 45.412918),
    #         geometry=geojson.LineString([(36.678276, 45.368423), (36.678863, 45.412918)]),
    #         properties=
    #         {
    #             "origin": "manual cast",
    #             "name": "затычка",
    #         },
    #     ),
    #     Feature(
    #         id=-3,
    #         boundingbox=(36.713815, 45.422227, 36.715284, 45.373392),
    #         geometry=geojson.LineString([(36.715284, 45.373392), (36.713815, 45.422227)]),
    #         properties=
    #         {
    #             "origin": "manual cast",
    #             "name": "затычка",
    #         },
    #     ),
    #     Feature(
    #         id=-4,
    #         boundingbox=(39.227646, 47.085887, 39.234667, 47.086217),
    #         geometry=geojson.LineString([(39.234667, 47.085887), (39.227646, 47.086217)]),
    #         properties=
    #         {
    #             "origin": "manual cast",
    #             "name": "затычка",
    #         },
    #     ),
    # ])
    # with open(
    #         DefaultLocate.OUTPUT_DIR / "temp.geojson",
    #         "w",
    #         encoding="utf-8",
    # ) as f:
    #     geojson.dump(data, f, ensure_ascii=False, indent=2)
    #     logging.info(f"Сохранен объединенный файл GeoJSON: {str(DefaultLocate.OUTPUT_DIR / "temp.geojson")}")
    # exit(0)


    flag_start = None
    list_data = []
    data_result = FeatureCollection([])
    for path in [
        # DefaultLocateRegionGeoJson.central_fed_district,
        # DefaultLocateRegionGeoJson.crimean_fed_district,
        # # DefaultLocateRegionGeoJson.far_eastern_fed_district,
        # DefaultLocateRegionGeoJson.north_caucasus_fed_district,
        # DefaultLocateRegionGeoJson.northwestern_fed_district,
        # # DefaultLocateRegionGeoJson.siberian_fed_district,
        # DefaultLocateRegionGeoJson.south_fed_district,
        # # DefaultLocateRegionGeoJson.ural_fed_district,
        # # DefaultLocateRegionGeoJson.ukraine,
        # DefaultLocateRegionGeoJson.volga_fed_district,
        # DefaultLocate.GEOJSON_DIR / "bulgaria.geojson",
        # DefaultLocate.GEOJSON_DIR / "georgia.geojson",
        # DefaultLocate.GEOJSON_DIR / "romania.geojson",
        # DefaultLocate.GEOJSON_DIR / "turkey.geojson",
        # DefaultLocate.GEOJSON_DIR / "Azov_sea.geojson",
        # DefaultLocate.GEOJSON_DIR / "Black_sea.geojson",
        # DefaultLocate.GEOJSON_DIR / "latvia.geojson",
        # DefaultLocate.GEOJSON_DIR / "lithuania.geojson",
        # DefaultLocate.GEOJSON_DIR / "denmark.geojson",
        # DefaultLocate.GEOJSON_DIR / "estonia.geojson",
        # DefaultLocate.GEOJSON_DIR / "germany.geojson",
        # DefaultLocate.GEOJSON_DIR / "northwestern-fed-district.geojson",
        # DefaultLocate.GEOJSON_DIR / "sweden.geojson",
        # DefaultLocate.GEOJSON_DIR / "finland.geojson",
        # # DefaultLocate.GEOJSON_DIR / "poland.geojson",
        # DefaultLocate.GEOJSON_DIR / "Azov_sea.geojson",
        # DefaultLocate.GEOJSON_DIR / "Black_sea.geojson",
        # DefaultLocate.GEOJSON_DIR / "Baltic_sea.geojson",
        # DefaultLocate.OUTPUT_DIR / "temp.geojson",
        # DefaultLocate.OUTPUT_DIR / "europe.geojson",
        # DefaultLocate.OUTPUT_DIR / "grid_Azov_sea.geojson",
        # DefaultLocate.OUTPUT_DIR / "grid_Baltic_sea.geojson",
        # DefaultLocate.OUTPUT_DIR / "grid_Black_sea.geojson",
        # DefaultLocate.OUTPUT_DIR / "grid_caspian_sea.geojson",
        # DefaultLocate.OUTPUT_DIR / "grid_white_sea.geojson",
        # DefaultLocate.OUTPUT_DIR / "construct_europe.geojson"
        # DefaultLocate.OUTPUT_DIR / "test.geojson"
        DefaultLocate.GEOJSON_DIR / "europe.geojson",
        DefaultLocate.OUTPUT_DIR / "make_hand_line.geojson",
    ]:
        with open(path, "r", encoding="utf-8") as f:
            data: FeatureCollection = geojson.load(f)
            logging.info(f"Загружен файл GeoJSON: {path}")


        # data = ParceGeojson.maker_easier(data)
        # data = ParceGeojson.split_to_grid(
        #     feature=f,
        #     cell_size=1
        # )
        # list_node = []
        # for feature in data.features:
        #     for list_coord in feature.geometry.coordinates:
        #         for coord in list_coord:
        #             node = geojson.Feature(
        #                 id = len(list_data) + 1,
        #                 geometry=geojson.Point(coord),
        #             )
        #             list_node.append(node)
        # data.features.extend(list_node)

        # data = ParceGeojson.extracting_coastline(data)
        # logging.debug(f"Количество объектов до очистки: {len(data.features)}")
        # for key, values in TagsOSM.BLACKLIST.items():
        #     for value in values:
        #         data, removed_features = ParceGeojson.remove_feature_from_tag(
        #             features_collection=data, key_tag=key, value_tag=value
        #         )
        data_result.features.extend(data.features)

        # # for key, values in TagsOSM.TEST_BLACKLIST.items():
        # #     for value in values:
        # #         data, removed_features = ParceGeojson.remove_feature_from_tag(
        # #             features_collection=data, key_tag=key, value_tag=value
        # #         )
        # logging.debug(f"Количество объектов после очистки: {len(data.features)}")
        # list_data.append(data)
        # ParceGeojson.print_tags(data)
        #
        # with open(
        #         DefaultLocate.OUTPUT_DIR / f"grid_{path.stem}.geojson",
        #         "w",
        #         encoding="utf-8",
        #     ) as f:
        #         geojson.dump(data, f, ensure_ascii=False, indent=2)

        # a = data.features[0].get("properties").get("tags")
        # a["OSM_way_id"] = (-1 * len(list_data)) - 1
        # data = FeatureCollection(
        #     [
        #         Feature(
        #             id=-1 * (len(list_data) + 1),
        #             geometry=data.features[0].geometry,
        #             properties={
        #                 "tags": a,
        #                 "OSM_id_nodes": data.features[0].properties.get("OSM_id_nodes"),
        #             },
        #         )
        #     ]
        # )
        # list_data.append(data)

    # data_result = ParceGeojson.concat_geojson_features(
    #     list_feature_collections=list_data, rewrite_duplicates=False, adding_duplicates=True
    # )
    data = ParceGeojson.remove_duplicate_features(data_result)
    logging.debug(f"Количество объектов после объединения: {len(data_result.features)}")
    with open(
        DefaultLocate.OUTPUT_DIR / "test_unity.geojson",
        "w",
        encoding="utf-8",
    ) as f:
        geojson.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Сохранен объединенный файл GeoJSON: {str(DefaultLocate.OUTPUT_DIR / ".geojson")}")
    #
    # data = ParceGeojson.maker_easier(data)
    # with open(
    #     DefaultLocate.OUTPUT_DIR / "test_easter.geojson",
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     geojson.dump(data, f, ensure_ascii=False, indent=2)
    #     logging.info(f"Сохранен объединенный файл GeoJSON: {str(DefaultLocate.OUTPUT_DIR / "easter.geojson")}")
