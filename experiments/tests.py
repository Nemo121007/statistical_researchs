"""Запуски для GPS утилит."""

import logging
from datetime import datetime
from pyexpat import features
from typing import List
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
from geojson import FeatureCollection, Feature, LineString, Point

from application.business_layer.db_access.sql_reader_postgres import SqlReaderPostgresResource
from application.modules.bg_services.gps.core import Logger
from application.modules.bg_services.gps.core.config import CorrectorSettings, Config
from application.modules.bg_services.gps.corrector.tracker.collectors.area_collector import AreaCollector
from application.modules.bg_services.gps.corrector.tracker.shared_files.calculator_distances_length_large_circle import \
    CalculatorDistancesLengthLargeCircle
from application.modules.bg_services.gps.utils.IOPs.IOPs_geojson import IOPs_geojson
from application.modules.bg_services.gps.corrector.tracker.models.way_model import Way
from application.modules.bg_services.gps.corrector.tracker.models.node_model import Node
from application.modules.bg_services.gps.corrector.tracker.trackers.GPS_tracker import (
    GPS_tracker,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.way_collector import (
    WayCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.node_collector import (
    NodeCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.geo_object_storage import (
    GeoObjectStorage,
)
from application.modules.bg_services.gps.utils.settings import DefaultLocate


class T:
    """Запуски для GPS утилит."""

    @staticmethod
    def gps():
        """Тестовый запуск GPS утилит."""
        path = Path(__file__).parent / "test_data" / "Дафна.csv"
        df = pd.read_csv(path, usecols=["time", "lat", "lon"], na_values=["", " ", "NULL", "null"])
        df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values(by="time").reset_index(drop=True)
        df = df.iloc[40:].reset_index(drop=True)
        # # Преобразуем timedelta в datetime, добавляя к базовой дате
        # base_date = datetime(2024, 1, 1, 0, 0, 0)  # Базовая дата
        # time_deltas = pd.to_timedelta(df["time"])
        # time_array = (base_date + time_deltas).tolist()

        time_array = df["time_dt"].tolist()[:]
        lat_array = df["lat"].astype(float).tolist()[:]
        lon_array = df["lon"].astype(float).tolist()[:]

        IOPs_geojson.write_geojson_from_arrays(
            output_path=DefaultLocate.OUTPUT_DIR / "output_1.geojson",
            list_arrays=[[time_array, lat_array, lon_array]],
        )
        logging.info(f"Исходных точек: {len(time_array)}. Файл  {DefaultLocate.OUTPUT_DIR / 'output_1.geojson'} создан.")

        # # Читаем в storage полигоны из geojson
        storage = GeoObjectStorage()
        path = DefaultLocate.GEOJSON_DIR / "europe.geojson"
        nodes, ways, areas = IOPs_geojson().fast_read_json(path)
        storage.nodes_collector = nodes
        storage.ways_collector = ways
        storage.area_collector = areas

        config = Config.load()
        sync_db_dsn = config.sync_db_settings.dsn.model_dump()
        with SqlReaderPostgresResource(sync_db_dsn) as sql_reader:

            # Загружаем хранилище в трекер
            gps_tracker = GPS_tracker(
                    times=time_array,
                    lats=lat_array,
                    lons=lon_array,
                    geo_object_storage=storage,
                    # sql_reader=sql_reader
                )

        # area = storage.area_collector.get_area(4230)
        # a = area.shapely_polygon.area
        # print(area.shapely_polygon.area)

        # gps_tracker.filter_w()
        # gps_tracker.filter_w()
        # exit(0)
        gps_tracker.raw_filter_speed()
        time_raw = gps_tracker.time_array
        lat_raw = gps_tracker.lat_array
        lon_raw = gps_tracker.lon_array

        IOPs_geojson.write_geojson_from_arrays(
            output_path=DefaultLocate.OUTPUT_DIR / "output_2.geojson",
            list_arrays=[[time_raw, lat_raw, lon_raw]],
        )
        logging.info(f"После фильтрации по скорости точек: {len(time_raw)}. Файл  {DefaultLocate.OUTPUT_DIR / 'output_2.geojson'} создан.")

        time_array, lat_array, lon_array, _ = gps_tracker.filter_track(
            distance_threshold=500.0,
            pre_initialization=False,
        )

        IOPs_geojson.write_geojson_from_arrays(
            output_path=DefaultLocate.OUTPUT_DIR / "output_3.geojson",
            list_arrays=[[time_array, lat_array, lon_array]],
        )
        logging.info(f"После фильтрации по расстоянию точек: {len(time_array)}. Файл  {DefaultLocate.OUTPUT_DIR / 'output_3.geojson'} создан.")

    @staticmethod
    def check_navigate():
        # # Читаем в storage полигоны из geojson
        storage = GeoObjectStorage()
        path = Path(__file__).parent.parent / "data" / "output" / "test_unity.geojson"
        nodes, ways, areas = IOPs_geojson().fast_read_json(path)
        storage.nodes_collector = nodes
        storage.ways_collector = ways
        storage.area_collector = areas

        area = storage.area_collector.get_area(20)
        print(area.shapely_polygon.area)

        # Загружаем хранилище в трекер
        gps_tracker = GPS_tracker(
            geo_object_storage=storage,
            times=[],
            lats=[],
            lons=[],
        )

        time_now = datetime.now()
        time_yesterday = time_now.replace(day=time_now.day - 1)
        # point_start = (time_yesterday, 39.317259, 47.115293)
        # point_end = (time_now, 38.922397, 47.067882)
        point_start = (time_yesterday, 36.677444, 45.636959)
        point_end = (time_now, 36.522054, 44.787366)
        track = gps_tracker._repair_track(past_node=point_start, current_node=point_end)
        list_nodes = []
        for _, lon, lat in track:
            node = Node(node_id=len(list_nodes), lat=lat, lon=lon)
            list_nodes.append(node)
        way = Way(way_id=3, nodes=list_nodes)
        way_collector = WayCollector()
        way_collector.add_way(way)
        path = DefaultLocate.OUTPUT_DIR / "navigate_output.geojson"
        IOPs_geojson.write_geojson(file_output_path=path, ways_collector=way_collector, list_print_points=list_nodes)
        pass

    @staticmethod
    def check():
        storage = GeoObjectStorage()
        path = Path(__file__).parent.parent / "data" / "output" / "central-fed-district_merged.geojson"
        with open(path, "r", encoding="utf-8") as f:
            data: FeatureCollection = geojson.load(f)

        feature_collection = FeatureCollection([])
        for geom in data.geometries:
            if geom.type == "Polygon":
                feature = geojson.Feature(
                    id=len(feature_collection.features),
                    geometry=geojson.Polygon(geom.coordinates),
                )
                feature_collection.features.append(feature)
            elif geom.type == "LineString":
                feature = geojson.Feature(
                    id=len(feature_collection.features),
                    geometry=geojson.LineString(geom.coordinates),
                )
                feature_collection.features.append(feature)

        with open(
            DefaultLocate.OUTPUT_DIR / f"reader_{path.stem}.geojson",
            "w",
            encoding="utf-8",
        ) as f:
            geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)


    @staticmethod
    def check_polygon():
        config = Config.load()
        sync_db_dsn = config.sync_db_settings.dsn.model_dump()
        with SqlReaderPostgresResource(sync_db_dsn) as sql_reader:
            a = GPS_tracker(
                times=[],
                lats=[],
                lons=[],
                sql_reader=sql_reader
            )
            # 47.126205, 39.330891
            features = a._get_all_object_from_bounding_box(
                max_lat=90.0,
                max_lon=90.0,
                min_lat=0.0,
                min_lon=0.0,
            )

        feature_collection = FeatureCollection([])
        for item in features:
            feature_collection.features.append(item)
        with open(
            DefaultLocate.OUTPUT_DIR / f"reader.geojson",
            "w",
            encoding="utf-8",
        ) as f:
            geojson.dump(feature_collection, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with Logger():
        T.gps()
        # T.check_navigate()
        # # # T.check()
        # T.check_polygon()
        # config = Config.load()
        # sync_db_dsn = config.sync_db_settings.dsn.model_dump()
        # with SqlReaderPostgresResource(sync_db_dsn) as sql_reader:
        #     a = GPS_tracker(
        #         times=[],
        #         lats=[],
        #         lons=[],
        #         sql_reader=sql_reader
        #     )
        #     a._get_all_object_from_bounding_box(
        #         max_lat=45.293706,
        #         max_lon=32.827239,
        #         min_lat=45.489973,
        #         min_lon=33.263603,
        #     )
