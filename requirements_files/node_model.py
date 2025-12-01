"""Модуль, содержащий класс Node для представления узла OpenStreetMap."""

from __future__ import annotations
from typing import List, TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from way_model import Way
    from area_model import Area


class Node:
    """Класс для представления узла OpenStreetMap.

    Узел представляет точку на карте с географическими координатами и связанными
    метаданными. Может быть частью путей (Way),
    а также иметь связи с соседними узлами для построения графа.

    Attributes:
        _id: Уникальный идентификатор узла.
        _lat: Широта узла в градусах.
        _lon: Долгота узла в градусах.
        _ways: Список путей, в которых участвует данный узел.
        _neighbors: Список соседних узлов для построения графа связей.
    """

    def __init__(
        self,
        node_id: int,
        lat: float = None,
        lon: float = None,
        ways: Optional[List["Way"]] = None,
        areas: Optional[List["Area"]] = None,
    ) -> None:
        """Инициализирует объект Node.
        Args:
            node_id: Уникальный идентификатор узла
            lat: Широта узла в градусах
            lon: Долгота узла в градусах
            ways: Список путей, содержащих данный узел
        Raises:
            ValueError: Если координаты некорректны
        """
        self._id = node_id
        if lat is not None and lat is not None:
            self._lat, self._lon = self._validate_coordinates(lat, lon)
        self._ways = list(ways) if ways else []
        self._areas = list(areas) if areas else []
        self._neighbors: List["Node"] = []

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: Node) -> bool:
        if not isinstance(other, Node):
            return False
        return self._id == other.id

    @staticmethod
    def _validate_coordinates(lat: float, lon: float) -> Tuple[float, float]:
        """Валидирует координаты.
        Args:S
            lat: Широта
            lon: Долгота
        Returns:
            Кортеж (lat, lon) если координаты валидны
        Raises:
            ValueError: Если координаты некорректны
        """
        if not -90 <= lat <= 90:
            raise ValueError(f"Широта должна быть в диапазоне [-90, 90], получено: {lat}")
        if not -180 <= lon <= 180:
            raise ValueError(f"Долгота должна быть в диапазоне [-180, 180], получено: {lon}")
        return lat, lon

    def __repr__(self) -> str:
        return (
            f"Node(id={self._id}, coords=({self._lat:.4f}, {self._lon:.4f}), "
            f"ways={self.way_count}, neighbors={self.neighbor_count})"
        )

    @property
    def id(self) -> int:
        """Уникальный идентификатор узла."""
        return self._id

    @property
    def coordinates(self) -> Tuple[float, float]:
        """Координаты узла (широта, долгота)."""
        return self._lat, self._lon

    @coordinates.setter
    def coordinates(self, coords: Tuple[float, float]) -> None:
        """Устанавливает координаты узла.
        Args:
            coords: Кортеж (широта, долгота)
        Raises:
            ValueError: Если координаты некорректны
        """
        lat, lon = coords
        self._lat, self._lon = self._validate_coordinates(lat, lon)

    @property
    def lat(self):
        """Широта узла."""
        return self._lat

    @property
    def lon(self):
        """Долгота узла."""
        return self._lon

    @property
    def ways(self) -> List["Way"]:
        """Список путей узла (копия для предотвращения изменений)."""
        return self._ways

    @property
    def areas(self) -> List["Area"]:
        """Список областей, к которым принадлежит узел (копия для предотвращения изменений)."""
        return self._areas

    @property
    def neighbors(self) -> List["Node"]:
        """Список соседних узлов (копия для предотвращения изменений)."""
        return self._neighbors

    # region Методы работы с путями
    def add_way(self, way: "Way") -> None:
        """Добавляет путь к узлу.
        Args:
            way: Объект Way для добавления
        Raises:
            ValueError: Если путь некорректен
        """
        if way is None:
            raise ValueError("Путь не может быть None")
        if way not in self._ways:
            self._ways.append(way)

    def remove_way(self, way: "Way") -> bool:
        """Удаляет путь из списка путей узла.
        Args:
            way: Объект Way для удаления
        Returns:
            True если путь был удален, иначе False
        """
        try:
            self._ways.remove(way)
            return True
        except ValueError:
            return False

    def clear_ways(self) -> None:
        """Удаляет все пути узла."""
        self._ways.clear()

    @property
    def way_count(self) -> int:
        """Количество путей, связанных с узлом."""
        return len(self._ways)

    # endregion

    def add_area(self, area: "Area") -> None:
        """Добавляет область к узлу.
        Args:
            area: Объект Area для добавления
        Raises:
            ValueError: Если область некорректна
        """
        if area is None:
            raise ValueError("Область не может быть None")
        if area not in self._areas:
            self._areas.append(area)

    # region Методы работы с соседями
    def add_neighbor(self, neighbor: "Node") -> None:
        """Добавляет соседний узел.
        Args:
            neighbor: Узел для добавления в качестве соседа
        Raises:
            ValueError: Если сосед некорректен
        """
        if neighbor is None:
            raise ValueError("Сосед не может быть None")
        if neighbor is self:
            raise ValueError("Узел не может быть соседом самому себе")
        if neighbor not in self._neighbors:
            self._neighbors.append(neighbor)
            # Добавляем обратную связь
            neighbor.add_neighbor(self)

    def remove_neighbor(self, neighbor: "Node") -> bool:
        """Удаляет соседний узел.
        Args:
            neighbor: Узел для удаления из списка соседей
        Returns:
            True если сосед был удален, иначе False
        """
        try:
            self._neighbors.remove(neighbor)
            # Удаляем обратную связь
            neighbor.remove_neighbor(self)
            return True
        except ValueError:
            return False

    def clear_neighbors(self) -> None:
        """Удаляет всех соседей узла."""
        for neighbor in self._neighbors:
            self.remove_neighbor(neighbor)

    @property
    def neighbor_count(self) -> int:
        """Количество соседних узлов."""
        return len(self._neighbors)

    # endregion

    # region Свойства состояния узла
    @property
    def is_isolated_neighbors(self) -> bool:
        """Проверяет, является ли узел изолированным по соседям.
        Узел считается изолированным по соседям, если у него нет соседей.
        Returns:
            True если у узла нет соседей, иначе False
        """
        return not self._neighbors

    @property
    def is_isolated_ways(self) -> bool:
        """Проверяет, является ли узел изолированным по путям.
        Узел считается изолированным по путям, если у него нет путей.
        Returns:
            True если у узла нет путей, иначе False
        """
        return not self._ways

    @property
    def degree(self) -> int:
        """Степень узла (количество связей с другими узлами)."""
        return len(self._neighbors)

    @property
    def is_connected(self) -> bool:
        """Проверяет, подключен ли узел к какой-либо инфраструктуре.
        Returns:
            True если у узла есть соседи, пути
        """
        return not (self.is_isolated_neighbors and self.is_isolated_ways)

    # endregion
