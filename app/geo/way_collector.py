"""Линейные объекты и коллектор путей."""

from __future__ import annotations

from typing import Any, Dict, Optional

from shapely.geometry import LineString


class Way:
    """
    Класс, представляющий линейный путь (Way) с уникальным идентификатором,
    геометрией в виде линии и дополнительными тегами.
    Attributes:
        way_id (int): Уникальный идентификатор пути.
        shapely_line (LineString): Геометрия пути в виде объекта Shapely.
        tags (Dict[str, Any]): Дополнительные теги или метаданные для пути.
        bbox (tuple[float, float, float, float]): Ограничивающий прямоугольник пути в виде
            кортежа (minx, miny, maxx, maxy).
    """

    def __init__(self, way_id: int, shapely_line: LineString, tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            way_id: Идентификатор пути.
            shapely_line: Линия, представляющая путь.
            tags: Дополнительные теги или метаданные для пути.
        """
        self.way_id: int = way_id
        self.tags: Dict[str, Any] = tags if tags is not None else {}
        self.shapely_line = shapely_line

    @property
    def id(self) -> int:
        """
        Returns: идентификатор пути.
        """
        return self.way_id

    @property
    def shapely_line(self) -> LineString:
        """
        Returns: линия, представляющая путь.
        """
        return self._shapely_line

    @shapely_line.setter
    def shapely_line(self, value: LineString) -> None:
        """
        Установить новую линию для пути и обновить ограничивающую рамку (bbox).
        Args:
            value: Новая линия, которая будет установлена для пути.
        """
        self._shapely_line = value
        self._bbox = value.bounds

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Получить ограничивающую рамку (bounding box) пути."""
        return self._bbox


class WayCollector:
    def __init__(self) -> None:
        self.ways: Dict[int, Way] = {}
        self._next_id: int = 1

    def resolve_id(self, way_id: int | None = None) -> int:
        """
        Разрешает уникальный идентификатор для нового пути. Если way_id передан, он будет использоваться.
        Args:
            way_id: Идентификатор пути, который нужно использовать. Если None, будет сгенерирован новый уникальный.
        """
        if way_id is not None:
            way_id = int(way_id)
            self._next_id = max(self._next_id, way_id + 1)
            return way_id
        while self._next_id in self.ways:
            self._next_id += 1
        resolved = self._next_id
        self._next_id += 1
        return resolved

    def add_way(self, way: Way) -> None:
        """
        Добавляет путь в коллекцию. Если путь с таким же идентификатором уже существует, будет вызвано исключение.
        Args:
            way: Объект Way, который нужно добавить в коллекцию.
        """
        if way.way_id in self.ways:
            raise ValueError(f"Путь с id={way.way_id} уже есть в коллекторе")
        self.ways[way.way_id] = way
        self._next_id = max(self._next_id, way.way_id + 1)

    def get_way(self, way_id: int) -> Optional[Way]:
        """
        Получает путь по его идентификатору.
        Args:
            way_id: Идентификатор пути, который нужно получить.

        Returns:
            Объект Way, если путь найден, иначе None.
        """
        return self.ways[way_id] if way_id in self.ways else None
