"""Чтение и фильтрация данных из OSM файлов.
Внимание: в данным классе переопределены методы area и way из osmium.SimpleHandler."""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import geojson
import osmium

from application.modules.bg_services.gps.utils.IOPs.IOPs_geojson import IOPs_geojson
from application.modules.bg_services.gps.utils.settings import TagsOSM, DefaultLocate
from application.modules.bg_services.gps.corrector.tracker.models.way_model import Way
from application.modules.bg_services.gps.corrector.tracker.models.area_model import Area
from application.modules.bg_services.gps.corrector.tracker.models.node_model import Node
from application.modules.bg_services.gps.corrector.tracker.collectors.way_collector import (
    WayCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.area_collector import (
    AreaCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.node_collector import (
    NodeCollector,
)

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("osm_reader.log", encoding="utf-8"),  # Запись в файл
    ],
)

logger = logging.getLogger(__name__)


def update_line(message: str) -> None:
    """Обновление строки в консоли.
    Позволяет динамически обновлять строку в консоли без создания новой строки.
    Args:
        message: Сообщение для отображения в консоли
    """
    # '\r' — вернуть каретку в начало строки
    # '\x1b[K' — ANSI последовательность: очистить от курсора до конца строки
    sys.stdout.write("\r" + message + "\x1b[K")
    sys.stdout.flush()


class ReaderOSM(osmium.SimpleHandler):
    """Чтение OSM файлов и фильтрация данных.
    Собирает узлы, пути и области в соответствующие коллекции.
    Фильтрация основана на тегах, чтобы оставить только релевантные данные.
    Attributes:
        _node_collector (NodeCollector): Коллекция узлов.
        _way_collector (WayCollector): Коллекция путей.
        _area_collector (AreaCollector): Коллекция областей.
    """

    def __init__(self):
        super().__init__()
        self._node_collector = NodeCollector()
        self._way_collector = WayCollector()
        self._area_collector = AreaCollector()
        self._read_ways = False
        self._read_areas = False
        self._path_read_file = None

        self.counter_ways = 0
        self.counter_areas = 0

    @property
    def node_collector(self) -> NodeCollector:
        """Возвращает коллекцию узлов."""
        return self._node_collector

    @node_collector.setter
    def node_collector(self, collector: NodeCollector) -> None:
        """Устанавливает коллекцию узлов."""
        if not isinstance(collector, NodeCollector):
            raise TypeError("collector должен быть типа NodeCollector")
        self._node_collector = collector

    @property
    def way_collector(self) -> WayCollector:
        """Возвращает коллекцию путей."""
        return self._way_collector

    @way_collector.setter
    def way_collector(self, collector: WayCollector) -> None:
        """Устанавливает коллекцию путей."""
        if not isinstance(collector, WayCollector):
            raise TypeError("collector должен быть типа WayCollector")
        self._way_collector = collector

    @property
    def area_collector(self) -> AreaCollector:
        """Возвращает коллекцию областей."""
        return self._area_collector

    @area_collector.setter
    def area_collector(self, collector: AreaCollector) -> None:
        if not isinstance(collector, AreaCollector):
            raise TypeError("collector должен быть типа AreaCollector")
        self._area_collector = collector

    def read_osm(
        self, path_osm: Path, read_ways: bool = False, read_areas: bool = False
    ) -> Tuple[NodeCollector, WayCollector, AreaCollector]:
        """Чтение OSM файла и заполнение коллекций узлов, путей и областей.
        Args:
            path_osm: Путь к OSM файлу
            read_ways: Флаг для чтения путей
            read_areas: Флаг для чтения областей
        Returns:
            Кортеж из трех коллекций: (NodeCollector, WayCollector, AreaCollector)
        Raises:
            ValueError: Если не указан ни один из объектов для чтения
            TypeError: Если path_osm не является экземпляром Path
            ValueError: Если файл по path_osm не существует
        """
        if not read_ways and not read_areas:
            raise ValueError("Укажите объект для чтения: read_ways или read_areas")
        if read_ways and read_areas:
            logging.warning(
                "\nЧитаются одновременно пути и области. " "Это может привести к некорректному распознаванию объектов."
            )
        self._read_ways = read_ways
        self._read_areas = read_areas
        self._path_read_file = path_osm
        if not isinstance(path_osm, Path):
            raise TypeError("path_osm must be an instance of Path")
        if not path_osm.exists():
            raise ValueError(f"OSM файл не найден: {path_osm}")

        self.apply_file(path_osm, locations=True, idx="flex_mem")

        return self._node_collector, self._way_collector, self._area_collector

    def area(self, a: osmium.osm.Area) -> None:
        """Обработка области OSM.
        Фильтрация областей по тегам и добавление релевантных областей в коллекцию.
        Данный метод является переопределением метода area итератора osmium.SimpleHandler,
        вызываемого при обработке каждой области в OSM файле.
        Не вызывать напрямую!
        Args:
            a: Область OSM для обработки
        """
        if not self._read_areas:
            return

        # Попытка динамического обновления прогресса
        self.counter_areas = self.counter_areas + 1
        update_line(f"Обработано областей: {self.counter_areas}. " f"Последняя обработанная область ID: {a.id}")

        if not self._filtered_areas(a):
            return

        tags = {tag.k: tag.v for tag in a.tags}
        tags["OSM_area_id"] = str(a.id)

        # Перебираем все кольца (внешние и внутренние)
        for oring in a.outer_rings():
            outer_border: List[Node] = []
            for node in oring:
                if self._node_collector.nodes.get(node.ref):
                    node = self._node_collector.nodes.get(node.ref)
                else:
                    node = Node(node_id=node.ref, lon=node.lon, lat=node.lat)
                outer_border.append(node)

            inner_borders: List[List[Node]] = []
            for bord in a.inner_rings(oring):
                inner_border: List[Node] = []
                for inner_node in bord:
                    if self._node_collector.nodes.get(inner_node.ref):
                        node = self._node_collector.nodes.get(inner_node.ref)
                    else:
                        node = Node(
                            node_id=inner_node.ref,
                            lon=inner_node.lon,
                            lat=inner_node.lat,
                        )
                    inner_border.append(node)
                inner_borders.append(inner_border)

            area = Area(
                area_id=len(self._area_collector.areas) + 1,
                tags=tags,
                outer_border=outer_border,
                inner_borders=inner_borders,
            )

            for border_node in area.outer_border:
                border_node.add_area(area)
                if not self._node_collector.nodes.get(border_node.id):
                    self._node_collector.add_node(border_node)
                border_node.add_area(area)
            for item in area.inner_borders:
                for border_node in item:
                    border_node.add_area(area)
                    if not self._node_collector.nodes.get(border_node.id):
                        self._node_collector.add_node(border_node)
                    border_node.add_area(area)

            self._area_collector.add_area(area)

            if len(self._way_collector.ways) % 100 == 0:
                logging.debug(
                    "\n%s) Добавлена область ID: %s, "
                    + "узлы внешней границы: %s, "
                    + "внутренние границы: %s,"
                    + " время: %s,"
                    + " %s",
                    str(len(self._area_collector.areas)),
                    str(a.id),
                    str(len(outer_border)),
                    str(len(inner_borders)),
                    str(datetime.now()),
                    str(self._path_read_file),
                )

    def _filtered_areas(self, area: osmium.osm.Area) -> bool:
        """Фильтрация областей по тегам.
        Args:
            area: Область OSM для фильтрации
        Returns:
            True если область релевантна, иначе False
        """
        # if area.tags.get("natural") in [
        #     "bay",
        #     "beach",
        #     "cape",
        #     "coastline",
        #     "peninsula",
        #     "peninsula",
        #     "strait",
        #     "water",
        # ]:
        #     return True
        # if area.tags.get("water") in [
        #     "river",
        #     "oxbow",
        #     "canal",
        #     "lock",
        #     "lake",
        #     "reservoir",
        #     "pond",
        #     "lagoon",
        # ]:
        #     return True
        # if area.tags.get("waterway") in ["river", "riverbank", "fairway"]:
        #     return True
        for tag, keys in TagsOSM.AREA_TAGS_INCLUDE.items():
            if area.tags.get(tag) in keys:
                break
        else:
            return False

        flag_terminate = True
        for item in area.outer_rings():
            for node in item:
                if node.ref in self._node_collector.nodes.keys() and self.node_collector.nodes[node.ref].way_count > 0:
                    flag_terminate = False
                    break
            if not flag_terminate:
                break
        if flag_terminate:
            return False

        return True

    def way(self, w: osmium.osm.Way) -> None:
        """Обработка пути OSM.
        Фильтрация путей по тегам и добавление релевантных путей.
        Данный метод является переопределением метода way итератора osmium.SimpleHandler,
        вызываемого при обработке каждого пути в OSM файле.
        Не вызывать напрямую!
        Args:
            w: Путь OSM для обработки
        """
        if not self._read_ways:
            return

        # Попытка динамического обновления прогресса
        self.counter_ways = self.counter_ways + 1
        update_line(f"Обработано путей: {self.counter_ways}. " f"Последний обработанный путь ID: {w.id}")

        if self._filtered_ways(w):
            tags = {tag.k: tag.v for tag in w.tags}
            tags["OSM_way_id"] = str(w.id)
            way = Way(
                way_id=len(self._way_collector.ways) + 1,
                tags=tags,
                nodes=[Node(node_id=node.ref, lat=node.lat, lon=node.lon) for node in w.nodes],
            )
            for border_node in way.nodes:
                node = self._node_collector.nodes.get(border_node.id)
                if node:
                    node.add_way(way=way)
                else:
                    border_node.add_way(way)
                    self._node_collector.add_node(border_node)

            self._way_collector.add_way(way)

            if len(self._way_collector.ways) % 100 == 0:
                logging.debug(
                    "\n%s) Добавлен путь ID: %s, узлы: %s, время: %s, %s",
                    str(len(self._way_collector.ways)),
                    str(w.id),
                    str(len(w.nodes)),
                    str(datetime.now()),
                    str(self._path_read_file),
                )

    @staticmethod
    def _filtered_ways(way: osmium.osm.Way) -> bool:
        """Фильтрация путей по тегам.
        Args:
            way: Путь OSM для фильтрации
        Returns:
            True если путь релевантен, иначе False
        """
        # Хорошая фильтрация для водных путей
        # if way.tags.get("boat") == "yes":
        #     return True
        # if way.tags.get("ship") == "yes":
        #     return True
        # if way.tags.get("CEMT"):
        #     return True
        if way.tags.get("natural") == "coastline":
            return True

        # for tag, keys in TagsOSM.AREA_TAGS_INCLUDE.items():
        #     if way.tags.get(tag) in keys:
        #         break
        # else:
        #     return False
        # return True

        #################################################
        # if way.tags.get('natural') in ['bay', 'beach', 'cape', 'coastline',
        # 'peninsula', 'peninsula', 'strait', 'water']:
        #     return True
        # if way.tags.get('water') in ['river', 'oxbow', 'canal', 'lock',
        # 'lake', 'reservoir', 'pond', 'lagoon']:
        #     return True
        # if way.tags.get('waterway') in ['river', 'riverbank', 'fairway']:
        #     return True
        ##################################################
        return False


if __name__ == "__main__":
    list_files = [
        # DefaultLocate.OSM_DIR / "denmark.osm.pbf",
        # DefaultLocate.OSM_DIR / "estonia.osm.pbf",
        # DefaultLocate.OSM_DIR / "finland.osm.pbf",
        # DefaultLocate.OSM_DIR / "germany-latest.osm.pbf",
        # DefaultLocate.OSM_DIR / "latvia.osm.pbf",
        # DefaultLocate.OSM_DIR / "lithuania.osm.pbf",
        # DefaultLocate.OSM_DIR / "sweden.osm.pbf",
        DefaultLocate.OSM_DIR
        / "poland.osm.pbf",
    ]

    for path in list_files:
        reader = ReaderOSM()
        nodes, ways, areas = reader.read_osm(
            path_osm=path,
            read_ways=True,
        )
        nodes, ways, areas = reader.read_osm(
            path_osm=path,
            read_areas=True,
        )

        logger.info(
            f"\nФайл: {path}\n"
            f"Узлов: {len(nodes.nodes)}\n"
            f"Путей: {len(ways.ways)}\n"
            f"Областей: {len(areas.areas)}\n"
        )

        output = DefaultLocate.OUTPUT_DIR / f"{path.stem}_full.geojson"
        IOPs_geojson.write_geojson(
            file_output_path=output,
            ways_collector=ways,
            areas_collector=areas,
            nodes_collector=nodes,
        )
    # path = DefaultLocate.OUTPUT_DIR / "sweden_full.geojson"
    # with open(path, 'r', encoding='utf-8') as f:
    #     data = geojson.load(f)
    # logger.info(f"Файл: {path}\nГеометрий: {len(data['features'])}\n")
    # with open("sweden_1_full.geojson", 'w', encoding='utf-8') as f:
    #     geojson.dump(data, f, ensure_ascii=False)
