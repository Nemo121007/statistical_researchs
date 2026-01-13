"""Пример чтения OSM и записи в GeoJSON."""

import logging

from application.modules.bg_services.gps.utils.IOPs.reader_osm import ReaderOSM
from application.modules.bg_services.gps.utils.IOPs.IOPs_geojson import IOPs_geojson
from application.modules.bg_services.gps.utils.settings import (
    DefaultLocate,
    DefaultLocateRegion,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.geo_object_storage import (
    GeoObjectStorage,
)

if __name__ == "__main__":
    for name in [
        # "central-fed-district",
        # "crimean-fed-district",
        # "far-eastern-fed-district",
        # "north-caucasus-fed-district",
        # "northwestern-fed-district",
        # "siberian-fed-district",
        # "south-fed-district",
        # "ukraine",
        # "ural-fed-district",
        # "volga-fed-district",
        "bulgaria",
        "georgia",
        "romania",
        "turkey",
    ]:
        # Пример пайплайна парсинга
        # 1. Чтение way из osm по тегам, ассоциированным с судоходством
        path = DefaultLocateRegion.south_fed_district.parent
        path = path / f"{name}.osm.pbf"

        storage = GeoObjectStorage()
        reader = ReaderOSM()
        logging.debug(f"Файл для чтения: {str(path)}")
        nodes, ways, areas = reader.read_osm(path_osm=path, read_ways=True)

        storage.node_collector = nodes
        storage.ways_collector = ways
        storage.area_collector = areas

        # 2. Чтение Area, которые имеют непосредственные общие точки с загруженными путями
        new_nodes, new_ways, new_areas = reader.read_osm(path_osm=path, read_areas=True)
        storage.node_collector = new_nodes
        storage.area_collector = new_areas

        # 3. Запись в geojson всех собранных объектов
        IOPs_geojson().write_geojson(
            file_output_path=DefaultLocate.OUTPUT_DIR / f"{name}.geojson",
            nodes_collector=storage.node_collector,
            ways_collector=storage.ways_collector,
            areas_collector=storage.area_collector,
        )

        del nodes, ways, areas, reader, storage
