"""Полигоны и коллектор площадей."""

from __future__ import annotations

from typing import Any, Dict, Optional

from shapely.geometry import Polygon


class Area:
    """
    Класс, представляющий полигональную область (Area) с уникальным идентификатором,
    геометрией в виде полигона и дополнительными тегами.
    Attributes:
        area_id (int): Уникальный идентификатор полигона.
        shapely_polygon (Polygon): Геометрия полигона в виде объекта Shapely
        tags (Dict[str, Any]): Дополнительные теги или метаданные для полигона
        bbox (tuple[float, float, float, float]): Ограничивающий прямоугольник полигона в виде
            кортежа (minx, miny, maxx, maxy).
    """
    area_id: int
    shapely_polygon: Polygon
    tags: Dict[str, Any]
    bbox: tuple[float, float, float, float]

    def __init__(self, area_id: int, shapely_polygon: Polygon, tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            area_id: Идентификатор полигона.
            shapely_polygon: Полигон, представляющий область.
            tags: Дополнительные теги или метаданные для полигона.
        """
        self.area_id: int = area_id
        self.tags: Dict[str, Any] = tags if tags is not None else {}
        self.shapely_polygon = shapely_polygon

    @property
    def id(self) -> int:
        """
        Returns: идентификатор полигона.
        """
        return self.area_id

    @property
    def shapely_polygon(self) -> Polygon:
        """
        Returns: полигон, представляющий область.
        """
        return self._shapely_polygon

    @shapely_polygon.setter
    def shapely_polygon(self, value: Polygon) -> None:
        """
        Установить новый полигон для области и обновить ограничивающую рамку (bbox).
        Args:
            value: Новый полигон, который будет установлен для области.
        """
        self._shapely_polygon = value
        self._bbox = value.bounds

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Получить ограничивающую рамку (bounding box) полигона."""
        return self._bbox


class AreaCollector:
    def __init__(self) -> None:
        self.areas: Dict[int, Area] = {}
        self._next_id: int = 1

    def resolve_id(self, area_id: int | None = None) -> int:
        """
        Разрешает уникальный идентификатор для нового полигона. Если area_id передан, он будет использоваться,
        Args:
            area_id: Идентификатор полигона, который нужно использовать. Если None, будет сгенерирован новый уникальный.
        """
        if area_id is not None:
            area_id = int(area_id)
            self._next_id = max(self._next_id, area_id + 1)
            return area_id
        while self._next_id in self.areas:
            self._next_id += 1
        resolved = self._next_id
        self._next_id += 1
        return resolved

    def add_area(self, area: Area) -> None:
        """
        Добавляет полигон в коллекцию. Если полигон с таким же идентификатором уже существует, будет вызвано исключение.
        Args:
            area: Объект Area, который нужно добавить в коллекцию.
        """
        if area.area_id in self.areas:
            raise ValueError(f"Площадь с id={area.area_id} уже есть в коллекторе")
        self.areas[area.area_id] = area
        self._next_id = max(self._next_id, area.area_id + 1)

    def get_area(self, area_id: int) -> Optional[Area]:
        """
        Получает полигон по его идентификатору.
        Args:
            area_id: Идентификатор полигона, который нужно получить.

        Returns:
            Объект Area, если полигон найден, иначе None.
        """
        return self.areas[area_id] if area_id in self.areas else None
