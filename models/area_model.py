"""Модуль, содержащий класс Area для представления области (Area) OpenStreetMap."""

from typing import (
    Dict,
    List,
    Optional,
)

from shapely.geometry import Point, Polygon

from ..models.node_model import Node


class Area:
    """Класс для представления области (Area) OpenStreetMap.

    Область представляет замкнутую географическую область, определяемую внешней
    границей и, возможно, внутренними границами (например, озерами или островами).
    Область может иметь связанные метаданные в виде тегов.

    Attributes:
        _id: Уникальный идентификатор области.
        _tags: Словарь тегов, связанных с областью.
        _outer_border: Список узлов, определяющих внешнюю границу области.
        _inner_borders: Список списков узлов, определяющих внутренние границы области.
        _min_lon: Минимальная долгота области (для ограничивающего прямоугольника).
        _max_lon: Максимальная долгота области (для ограничивающего прямоугольника).
        _min_lat: Минимальная широта области (для ограничивающего прямоугольника).
        _max_lat: Максимальная широта области (для ограничивающего прямоугольника).
        _shapely_polygon: Объект Polygon из библиотеки Shapely, представляющий область.
    """

    def __init__(
        self,
        area_id: int,
        tags: Optional[Dict[str, str]] = None,
        outer_border: List[Node] = None,
        inner_borders: List[List[Node]] = None,
        shapely_polygon: Polygon = None,
        min_lon: float = None,
        max_lon: float = None,
        min_lat: float = None,
        max_lat: float = None,
    ) -> None:
        if inner_borders and not outer_border:
            raise ValueError("Если заданы внутренние границы, должна быть задана и внешняя граница")
        self._id: int = area_id
        self._tags: Optional[Dict[str, str]] = tags or {}
        self._outer_border: Optional[List[Node]] = outer_border or []
        self._inner_borders: List[List[Node]] = inner_borders or []
        self._min_lon: Optional[float] = min_lon
        self._max_lon: Optional[float] = max_lon
        self._min_lat: Optional[float] = min_lat
        self._max_lat: Optional[float] = max_lat

        if outer_border or inner_borders:
            self._calculate_bounding_box()

        self._shapely_polygon = shapely_polygon

    def __repr__(self):
        return (
            f"Area(id={self._id}, name={self._tags.get('name')}, "
            f"outer_border_nodes={len(self._outer_border)}, "
            f"inner_borders_count={len(self._inner_borders)})"
        )

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Area):
            return False
        return self._id == other.id

    @property
    def id(self) -> int:
        """Уникальный идентификатор области."""
        return self._id

    @property
    def tags(self) -> Dict[str, str]:
        """Словарь тегов, связанных с областью."""
        return self._tags

    @tags.setter
    def tags(self, new_tags: Dict[str, str]) -> None:
        if not isinstance(new_tags, dict):
            raise ValueError("Tags должен быть словарем")
        self._tags = new_tags

    @property
    def outer_border(self) -> List[Node]:
        """Список узлов, определяющих внешнюю границу области."""
        return self._outer_border

    @outer_border.setter
    def outer_border(self, new_outer_border: List[Node]) -> None:
        if not isinstance(new_outer_border, list) or not all(isinstance(node, Node) for node in new_outer_border):
            raise ValueError("Outer border должен быть списком объектов Node")
        self._outer_border = new_outer_border

    @property
    def inner_borders(self) -> List[List[Node]]:
        """ "Список списков узлов, определяющих внутренние границы области."""
        return self._inner_borders

    @inner_borders.setter
    def inner_borders(self, new_inner_borders: List[List[Node]]) -> None:
        if not isinstance(new_inner_borders, list) or not all(
            isinstance(border, list) and all(isinstance(node, Node) for node in border) for border in new_inner_borders
        ):
            raise ValueError("Inner borders должен быть списком списков объектов Node")
        self._inner_borders = new_inner_borders

    @property
    def bounding_box(self) -> Optional[List[float]]:
        """Возвращает ограничивающий прямоугольник области в формате
        [min_lon, min_lat, max_lon, max_lat] или None, если границы не заданы.
        """
        if self._min_lon is None or self._max_lon is None or self._min_lat is None or self._max_lat is None:
            return None
        return [self._min_lon, self._min_lat, self._max_lon, self._max_lat]

    @property
    def min_lon(self) -> Optional[float]:
        """Минимальная долгота области."""
        return self._min_lon

    @property
    def max_lon(self) -> Optional[float]:
        """Максимальная долгота области."""
        return self._max_lon

    @property
    def min_lat(self) -> Optional[float]:
        """Минимальная широта области."""
        return self._min_lat

    @property
    def max_lat(self) -> Optional[float]:
        """Максимальная широта области."""
        return self._max_lat

    def _calculate_bounding_box(self) -> None:
        """Вычисляет ограничивающий прямоугольник области на основе границ."""
        for border in [self._outer_border] + self._inner_borders:
            for node in border:
                lat, lon = node.coordinates
                if self._min_lat is None or lat < self._min_lat:
                    self._min_lat = lat
                if self._max_lat is None or lat > self._max_lat:
                    self._max_lat = lat
                if self._min_lon is None or lon < self._min_lon:
                    self._min_lon = lon
                if self._max_lon is None or lon > self._max_lon:
                    self._max_lon = lon

    @property
    def shapely_polygon(self) -> Optional[Polygon]:
        """Возвращает объект Polygon из библиотеки Shapely, представляющий область.
        Данный объект задаётся вручную. Используется для быстрой проверки области точек.
        При установке пересчитывается ограничивающий прямоугольник.
        """
        return self._shapely_polygon

    @shapely_polygon.setter
    def shapely_polygon(self, polygon: Polygon) -> None:
        self._shapely_polygon = polygon
        self._min_lon, self._min_lat, self._max_lon, self._max_lat = polygon.bounds

    # region tags

    def add_tag(self, key: str, value: str) -> None:
        """Добавляет или обновляет тег области.
        Args:
            key: Ключ тега
            value: Значение тега
        Raises:
            ValueError: Если ключ не является непустой строкой или значение не строка
        """
        if not key or not isinstance(key, str):
            raise ValueError("Ключ тега должен быть непустой строкой")
        if not isinstance(value, str):
            raise ValueError("Значение тега должно быть строкой")
        self._tags[key] = value

    def get_tag(self, key: str) -> Optional[str]:
        """Возвращает значение тега по ключу или None, если тег не найден.
        Args:
            key: Ключ тега
        Returns:
            Значение тега или None, если тег не найден
        """
        return self._tags.get(key, None)

    def has_tag(self, key: str) -> bool:
        """Проверяет наличие тега по ключу.
        Args:
            key: Ключ тега
        Returns:
            True, если тег с таким ключом существует, иначе False
        """
        return key in self._tags

    def remove_tag(self, key: str) -> bool:
        """Удаляет тег по ключу.
        Args:
            key: Ключ  тега
        Returns:
            True, если тег был удален, иначе False
        Raises:
            ValueError: Если ключ является пустой строкой
        """
        if key in self._tags:
            del self._tags[key]
            return True
        return False

    def clear_tags(self) -> None:
        """Удаляет все теги области."""
        self._tags.clear()

    #  endregion

    # region borders
    def set_outer_border(self, border: List[Node]) -> None:
        """Устанавливает внешнюю границу области.
        Args:
            border: Список узлов, определяющих внешнюю границу
        Raises:
            ValueError: Если border не является списком объектов Node
        """
        if not isinstance(border, list) or not all(isinstance(node, Node) for node in border):
            raise ValueError("Outer border должен быть списком объектов Node")
        self._outer_border = border
        self._calculate_bounding_box()

    def remove_outer_border(self) -> None:
        """Удаляет внешнюю границу области.
        Raises:
            ValueError: Если есть внутренние границы
        """
        if self._inner_borders:
            raise ValueError("Нельзя удалить внешнюю границу, если есть внутренние границы")
        self._outer_border = []
        self._min_lon = self._max_lon = self._min_lat = self._max_lat = None

    def add_inner_border(self, border: List[Node]) -> None:
        """Добавляет внутреннюю границу области.
        Args:
            border: Список узлов, определяющих внутреннюю границу
        Raises:
            ValueError: Если border не является списком объектов Node
                        или если внешняя граница не установлена
        """
        if not isinstance(border, list) or not all(isinstance(node, Node) for node in border):
            raise ValueError("Inner border должен быть списком объектов Node")
        if not self._outer_border:
            raise ValueError("Нельзя добавить внутреннюю границу без внешней границы")
        self._inner_borders.append(border)

    def remove_inner_border(self, border: List[Node]) -> bool:
        """Удаляет внутреннюю границу области.
        Args:
            border: Список узлов, определяющих внутреннюю границу для удаления
        Returns:
            True если граница была удалена, иначе False
        """
        try:
            self._inner_borders.remove(border)
            self._min_lon = self._max_lon = self._min_lat = self._max_lat = None
            return True
        except ValueError:
            return False

    def clear_inner_borders(self) -> None:
        """Удаляет все внутренние границы области."""
        self._inner_borders.clear()
        self._min_lon = self._max_lon = self._min_lat = self._max_lat = None

    # endregion

    def contains_point(self, lat: float, lon: float) -> bool:
        """Проверяет, находится ли заданная точка внутри области.
        Точка считается внутри области, если она находится внутри внешней границы
        и вне всех внутренних границ.
        Для определения принадлежности точки используется библиотека Shapely.
        Args:
            lat: Широта точки
            lon: Долгота точки
        Returns:
            True если точка внутри области, иначе False
        Raises:
            ValueError: Если внешняя граница не установлена
        """
        polygon = Polygon([(node.lon, node.lat) for node in self._outer_border])
        point = Point(lon, lat)
        if not polygon.contains(point):
            return False
        for inner_border in self._inner_borders:
            inner_polygon = Polygon([(node.lon, node.lat) for node in inner_border])
            if inner_polygon.contains(point):
                return False
        return True
