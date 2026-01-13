"""Модуль для представления пути (Way) из OpenStreetMap."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
)

import numpy as np
from shapely.geometry import LineString

if TYPE_CHECKING:
    from node_model import Node


class Way:
    """Класс для представления пути (Way) из OpenStreetMap.

    Путь представляет упорядоченную последовательность узлов, которая может
    образовывать линию (река, граница) или замкнутую область. Содержит
    метаданные в виде тегов.

    Attributes:
        _id: Уникальный идентификатор пути.
        _tags: Словарь тегов с метаданными пути.
        _nodes: Упорядоченный список узлов, составляющих путь.
        _is_polygon: Флаг, указывающий, является ли путь полигоном (замкнутым путём).
        _min_lat: Минимальная широта узлов пути.
        _max_lat: Максимальная широта узлов пути.
        _min_lon: Минимальная долгота узлов пути.
        _max_lon: Максимальная долгота узлов пути.
    """

    def __init__(
        self,
        way_id: int,
        tags: Optional[Dict[str, str]] = None,
        nodes: Optional[List["Node"]] = None,
        shapely_line: Optional[LineString] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
    ) -> None:
        """Инициализирует объект Way.

        Args:
            way_id: Уникальный идентификатор пути
            tags: Словарь тегов с метаданными пути
            nodes: Список узлов пути

        Raises:
            ValueError: Если идентификатор некорректен
        """
        self._id: int = way_id
        self._tags: Dict[str, str] = tags if tags else {}
        self._nodes: List["Node"] = list(nodes) if nodes else []
        self._is_polygon: bool = False
        self._min_lat: Optional[float] = min_lat
        self._max_lat: Optional[float] = max_lat
        self._min_lon: Optional[float] = min_lon
        self._max_lon: Optional[float] = max_lon
        self._neighbor_ways: Dict[Way, List[Node]] = {}
        self._shapely_line: Optional[LineString] = shapely_line

        if self._nodes:
            # Обновляем связи с узлами
            for node in self._nodes:
                node.add_way(self)

            lats = [node.coordinates[0] for node in self._nodes]
            lons = [node.coordinates[1] for node in self._nodes]

            self._min_lat = min(lats)
            self._min_lon = min(lons)
            self._max_lat = max(lats)
            self._max_lon = max(lons)

    def __repr__(self) -> str:
        """Возвращает строковое представление объекта Way."""
        return (
            f"Way(id={self._id}, name={self._tags.get('name')}, " f"tags={len(self._tags)}, nodes={len(self._nodes)})"
        )

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: Way) -> bool:
        if not isinstance(other, Way):
            return False
        return self._id == other.id

    @property
    def id(self) -> int:
        """Уникальный идентификатор пути."""
        return self._id

    @property
    def tags(self) -> Dict[str, str]:
        """Словарь тегов пути."""
        return self._tags

    @property
    def nodes(self) -> List["Node"]:
        """Список узлов пути."""
        return self._nodes

    @property
    def is_polygon(self) -> bool:
        """Проверяет, является ли путь полигоном (замкнутым)."""
        return len(self._nodes) >= 2 and self._nodes[0] == self._nodes[-1]

    @property
    def min_lat(self) -> Union[float, None]:
        """Минимальная широта узлов пути."""
        return self._min_lat

    @property
    def max_lat(self) -> Union[float, None]:
        """Максимальная широта узлов пути."""
        return self._max_lat

    @property
    def min_lon(self) -> Union[float, None]:
        """Минимальная долгота узлов пути."""
        return self._min_lon

    @property
    def max_lon(self) -> Union[float, None]:
        """Максимальная долгота узлов пути."""
        return self._max_lon

    @property
    def neighbor_ways(self) -> Dict[Way, List["Node"]]:
        """Словарь соседних путей с общими узлами."""
        return self._neighbor_ways

    @property
    def shapely_line(self) -> Optional[LineString]:
        """Возвращает объект LineString из библиотеки Shapely, представляющий путь.

        Returns:
            LineString или None, если путь пустой
        """
        if self._shapely_line is None:
            raise ValueError("Путь не содержит объекта shapely.LineString")
        return self._shapely_line

    @shapely_line.setter
    def shapely_line(self, line: LineString) -> None:
        """Устанавливает объект LineString из библиотеки Shapely, представляющий путь.
        Обновляет границы пути.
        Args:
            line: Объект LineString для установки
        Raises:
            ValueError: Если линия некорректна
        """
        if not isinstance(line, LineString):
            raise ValueError("Линия должна быть объектом shapely.LineString")
        self._shapely_line = line
        self._min_lon, self._min_lat, self._max_lon, self._max_lat = line.bounds

    # region Методы работы с тегами
    def add_tag(self, key: str, value: str) -> None:
        """Добавляет тег к пути.

        Args:
            key: Ключ тега
            value: Значение тега

        Raises:
            ValueError: Если ключ или значение некорректны
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Ключ тега должен быть непустой строкой")
        if not isinstance(value, str):
            raise ValueError("Значение тега должно быть строкой")
        self._tags[key] = value

    def get_tag(self, key: str, default: str = None) -> Optional[str]:
        """Возвращает значение тега по ключу.

        Args:
            key: Ключ тега
            default: Значение по умолчанию, если тег не найден

        Returns:
            Значение тега или default
        """
        return self._tags.get(key, default)

    def has_tag(self, key: str) -> bool:
        """Проверяет наличие тега.

        Args:
            key: Ключ тега

        Returns:
            True если тег существует, иначе False
        """
        return key in self._tags

    def remove_tag(self, key: str) -> bool:
        """Удаляет тег по ключу.

        Args:
            key: Ключ тега для удаления

        Returns:
            True если тег был удален, иначе False
        """
        return self._tags.pop(key, None) is not None

    def clear_tags(self) -> None:
        """Удаляет все теги пути."""
        self._tags.clear()

    # endregion

    # region Методы работы с узлами
    def add_node(self, node: "Node") -> None:
        """Добавляет узел в конец пути.

        Args:
            node: Объект Node для добавления

        Raises:
            ValueError: Если узел некорректен
        """
        if node is None:
            raise ValueError("Узел не может быть None")

        self._nodes.append(node)
        node.add_way(self)

        if len(self._nodes) >= 2 and self._nodes[0] == self._nodes[-1]:
            self._is_polygon = True

        if self._min_lat is None or node.coordinates[0] < self._min_lat:
            self._min_lat = node.coordinates[0]
        if self._max_lat is None or node.coordinates[0] > self._max_lat:
            self._max_lat = node.coordinates[0]
        if self._min_lon is None or node.coordinates[1] < self._min_lon:
            self._min_lon = node.coordinates[1]
        if self._max_lon is None or node.coordinates[1] > self._max_lon:
            self._max_lon = node.coordinates[1]

    def remove_node(self, node: "Node") -> bool:
        """Удаляет узел из пути.

        Args:
            node: Объект Node для удаления

        Returns:
            True если узел был удален, иначе False
        """
        try:
            node.remove_way(self)
            self._nodes.remove(node)

            lats = [node.coordinates[0] for node in self._nodes]
            lons = [node.coordinates[1] for node in self._nodes]

            self._min_lat = min(lats)
            self._min_lon = min(lons)
            self._max_lat = max(lats)
            self._max_lon = max(lons)

            return True
        except ValueError:
            return False

    def clear_nodes(self) -> None:
        """Удаляет все узлы из пути."""
        for node in self._nodes:
            self.remove_node(node)

    @property
    def node_count(self) -> int:
        """Количество узлов в пути."""
        return len(self._nodes)

    def get_node_ids(self) -> List[int]:
        """Возвращает список идентификаторов узлов пути.

        Returns:
            Список идентификаторов узлов в порядке следования
        """
        return [node.id for node in self._nodes]

    def has_node(self, node: "Node") -> bool:
        """Проверяет, содержится ли узел в пути.

        Args:
            node: Узел для проверки

        Returns:
            True если узел содержится в пути, иначе False
        """
        return node in self._nodes

    def add_neighbor_way(self, way: "Way", shared_node: "Node") -> None:
        """Добавляет соседний путь с общим узлом.

        Args:
            way: Соседний путь для добавления
            shared_node: Общий узел между путями

        Raises:
            ValueError: Если путь или узел некорректны
        """
        if way is None or shared_node is None:
            raise ValueError("Путь и узел не могут быть None")
        if way.id == self.id:
            raise ValueError("Путь не может быть соседом самому себе")
        if shared_node not in self._nodes or shared_node not in way.nodes:
            raise ValueError("Общий узел должен принадлежать обоим путям")

        if way.id in self._neighbor_ways:
            self._neighbor_ways[way].append(shared_node)
        else:
            self._neighbor_ways[way] = [shared_node]

    def set_neighbor_way(self) -> None:
        """Очищает и пересчитывает соседние пути на основе общих узлов."""
        self._neighbor_ways.clear()
        for node in self._nodes:
            for way in node.ways:
                if way.id != self.id:
                    self.add_neighbor_way(way, node)

    def get_coordinates_nodes(
        self, get_format: str = "tuples", include_ids: bool = True
    ) -> Union[List[Tuple[int, float, float]], List[Tuple[float, float]], np.ndarray]:
        """Возвращает координаты всех узлов пути в различных форматах.

        Args:
            get_format: Формат вывода:
                'tuples' - список кортежей (id, lat, lon)
                'coords' - список кортежей (lat, lon)
                'arrays' - три отдельных массива (ids, lats, lons)
                'numpy' - numpy массив shape (n, 2) или (n, 3)
            include_ids: Включать ли идентификаторы в результат (только для 'numpy')

        Returns:
            Координаты узлов в указанном формате

        Raises:
            ValueError: Если формат некорректен
        """
        if not self._nodes:
            return [] if get_format != "numpy" else np.array([])

        if get_format == "tuples":
            return [(node.id, node.coordinates[0], node.coordinates[1]) for node in self._nodes]
        if get_format == "coords":
            return [(node.coordinates[0], node.coordinates[1]) for node in self._nodes]
        if get_format == "arrays":
            return [(node.id, node.coordinates[0], node.coordinates[1]) for node in self._nodes]
        if get_format == "numpy":
            data = [(node.coordinates[0], node.coordinates[1]) for node in self._nodes]
            if include_ids:
                data = [(node.id, node.coordinates[0], node.coordinates[1]) for node in self._nodes]
            return np.array(data)
        raise ValueError(f"Неизвестный формат: {format}")

    @property
    def center(self) -> Tuple[float, float]:
        """Возвращает центр пути (средние координаты).

        Returns:
            Кортеж (center_lat, center_lon) или (0,0) если путь пустой
        """
        if not self._nodes:
            return 0.0, 0.0

        lats = [node.coordinates[0] for node in self._nodes]
        lons = [node.coordinates[1] for node in self._nodes]

        return sum(lats) / len(lats), sum(lons) / len(lons)
