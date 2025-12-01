"""Модуль для сбора и управления областями (Area) OSM"""

import logging
from typing import Dict, Optional

from .area_model import Area


class AreaCollector:
    """Класс для сбора и управления областями (Area).
    Обеспечивает уникальность хранимых областей и их многократное использование.
    (Используется при парсинге данных OSM)
    Attributes:
        _areas (Dict[int, Area]): Словарь областей, ключ - id области
    """

    def __init__(self):
        self._areas: Dict[int, Area] = {}

    def __repr__(self):
        return f"AreaCollector(areas={len(self.areas)})"

    @property
    def areas(self) -> Dict[int, Area]:
        """Словарь всех собранных областей, ключ - id области, значение - объект Area."""
        return self._areas

    def add_area(self, area: Area) -> None:
        """Добавляет область в коллекцию.
        При добавлении области с уже существующим ID, она будет перезаписана.
        Args:
            area: Объект Area для добавления
        Raises:
            TypeError: Если area не является экземпляром Area
        """
        if not isinstance(area, Area):
            raise TypeError("area должен быть типа Area")
        if area.id in self._areas:
            logging.warning(f"Area с id {area.id} уже существует в коллекции. Он будет перезаписан")
        self._areas[area.id] = area

    def replace_area(self, old_area: Area, new_area: Area) -> None:
        """Заменяет существующую область на новую.
        Args:
            old_area: Существующая область для замены
            new_area: Новая область для добавления
        Raises:
            TypeError: Если new_area не является экземпляром Area
        """
        if not isinstance(new_area, Area):
            raise TypeError("new_area должен быть типа Area")
        if old_area.id in self._areas:
            del self._areas[old_area.id]
        self._areas[new_area.id] = new_area

    def remove_area(self, area_id: int) -> bool:
        """Удаляет область из коллекции по идентификатору.
        Args:
            area_id: Идентификатор области для удаления
        Returns:
            True если область была удалена, иначе False
        """
        if area_id in self._areas:
            del self._areas[area_id]
            return True
        return False

    def clean_areas(self) -> None:
        """Очищает коллекцию областей."""
        self._areas.clear()

    def get_area(self, area_id: int) -> Optional[Area]:
        """Возвращает область по её идентификатору.
        Args:
            area_id: Идентификатор области для поиска
        Returns:
            Объект Area если найден, иначе None
        """
        return self.areas.get(area_id, None)

    def get_areas_by_bounding_box(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> Dict[str, Area]:
        """Возвращает области, пересекающиеся с заданной ограничивающей рамкой.
        Args:
            min_lat: Минимальная широта
            min_lon: Минимальная долгота
            max_lat: Максимальная широта
            max_lon: Максимальная долгота
        Returns:
            Словарь областей, пересекающихся с заданной рамкой, ключ - id
        """
        result = {}
        for area_id, area in self._areas.items():
            if (
                area.min_lat <= max_lat
                and area.max_lat >= min_lat
                and area.min_lon <= max_lon
                and area.max_lon >= min_lon
            ):
                result[area_id] = area
        return result
