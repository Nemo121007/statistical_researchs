"""Модуль для сбора и управления путями OSM."""

import logging
from typing import Dict, Optional

from .way_model import Way


class WayCollector():
    """Класс для сбора и управления путями OSM.
    Обеспечивает уникальность хранимых путей и их многократное использование.
    (Используется при парсинге данных OSM)
    Attributes:
        _ways (Dict[str, Way]): Словарь путей, ключ - id пути, значение - объект Way.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ways: Dict[int, Way] = {}

    def __repr__(self):
        """Возвращает строковое представление объекта WayCollector.
        Returns:
            Строка с информацией о количестве путей.
        """
        return f"WayCollector(ways={len(self._ways)}"

    @property
    def ways(self) -> Dict[int, Way]:
        """Словарь всех собранных путей, ключ - id пути, значение - объект Way."""
        return self._ways

    def add_way(self, way: Way) -> None:
        """Добавляет путь в коллекцию.
        При добавлении пути с уже существующим ID, он будет перезаписан.
        Args:
            way: Объект Way для добавления
        Raises:
            TypeError: Если way не является экземпляром Way
        """
        if not isinstance(way, Way):
            raise TypeError("way должен быть типа Way")
        if way.id in self._ways:
            logging.debug(f"Way с id {way.id}уже существует в коллекции. Он будет перезаписан")
        self._ways[way.id] = way

    def get_way(self, way_id: int) -> Optional[Way]:
        """Возвращает путь по его идентификатору.
        Args:
            way_id: Идентификатор пути для поиска
        Returns:
            Объект Way если найден, иначе None
        """
        return self._ways.get(way_id)

    # region ways
    def remove_way(self, way_id: int) -> bool:
        """Удаляет путь по его идентификатору.
        Args:
            way_id: Идентификатор пути для удаления
        Returns:
            True если путь был удален, иначе False
        """
        if way_id in self._ways:
            way = self._ways[way_id]
            # Удаляем путь из всех связанных узлов
            for node in way.nodes:
                node.remove_way(way)
            # Удаляем путь из коллекции
            del self._ways[way_id]
            return True
        return False

    def get_ways_by_bounding_box(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> Dict[str, Way]:
        """Возвращает пути, пересекающиеся с заданной областью.
        Args:
            min_lat: Минимальная широта области
            min_lon: Минимальная долгота области
            max_lat: Максимальная широта области
            max_lon: Максимальная долгота области
        Returns:
            Словарь путей, пересекающихся с областью
        """
        result = {}
        for way in self._ways.values():
            if (
                way.min_lat <= max_lat
                and way.max_lat >= min_lat
                and way.min_lon <= max_lon
                and way.max_lon >= min_lon
            ):
                result[way.id] = way
        return result

    # endregion ways
