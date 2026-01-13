#!/usr/bin/env python3
"""Хранилище геообъектов: узлы, линии, полигоны."""

import logging

from application.modules.bg_services.gps.corrector.tracker.collectors.way_collector import (
    WayCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.area_collector import (
    AreaCollector,
)
from application.modules.bg_services.gps.corrector.tracker.collectors.node_collector import (
    NodeCollector,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="\n%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Принудительная переконфигурация
)


class GeoObjectStorage:
    """Хранилище геообъектов: узлы, линии, полигоны.
    Позволяет хранить и управлять коллекциями геообъектов.
    Attributes:
        _node_collector (NodeCollector): Коллекция узлов.
        _ways_collector (WayCollector): Коллекция линий.
        _area_collector (AreaCollector): Коллекция полигонов.
    """

    def __init__(
        self,
        nodes_collector: NodeCollector = None,
        ways_collector: WayCollector = None,
        areas_collector: AreaCollector = None,
    ):
        self._node_collector = nodes_collector or NodeCollector()
        self._ways_collector = ways_collector or WayCollector()
        self._area_collector = areas_collector or AreaCollector()

    def __repr__(self):
        return (
            f"GeoObjectStorage(areas={len(self.area_collector.areas)}, "
            f"ways={len(self.ways_collector.ways)}, "
            f"nodes={len(self.node_collector.nodes)})"
        )

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
    def ways_collector(self) -> WayCollector:
        """Возвращает коллекцию линий."""
        return self._ways_collector

    @ways_collector.setter
    def ways_collector(self, collector: WayCollector) -> None:
        """Устанавливает коллекцию линий."""
        if not isinstance(collector, WayCollector):
            raise TypeError("collector должен быть типа WayCollector")
        self._ways_collector = collector

    @property
    def area_collector(self) -> AreaCollector:
        """Возвращает коллекцию полигонов."""
        return self._area_collector

    @area_collector.setter
    def area_collector(self, collector: AreaCollector) -> None:
        if not isinstance(collector, AreaCollector):
            raise TypeError("collector должен быть типа AreaCollector")
        self._area_collector = collector

    @property
    def global_bounding_box(self):
        """Вычисляет глобальный bounding box для всех геообъектов."""
        # min_lat, min_lon, max_lat, max_lon = None, None, None, None
        return 0, 90, 0, 90  # TODO:  реализую расчет глобального бокса. Пока затычка
