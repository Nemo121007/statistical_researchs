"""Модуль для сбора и управления узлами OSM"""

import logging
from typing import Dict

from .node_model import Node


class NodeCollector:
    """Класс для сбора и управления узлами.
    Обеспечивает уникальность хранимых узлов и их многократное использование.
    (Используется при парсинге данных OSM)
    Attributes:
        _nodes (Dict[int, Node]): Словарь узлов, ключ - id узла, значение - объект Node.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes: Dict[int, Node] = {}

    def __repr__(self):
        return f"NodeCollector(nodes={len(self._nodes)}"

    @property
    def nodes(self) -> Dict[int, Node]:
        """Словарь всех собранных узлов, ключ - id узла, значение - объект Node."""
        return self._nodes

    def add_node(self, node: Node) -> None:
        """Добавляет узел в коллекцию.
        При добавлении узла с уже существующим ID, он будет перезаписан.
        Args:
            node: Объект Node для добавления
        Raises:
            ValueError: Если узел некорректен
        """
        if node is None:
            raise ValueError("Узел не может быть None")
        if node.id in self._nodes:
            logging.warning(f"Узел с ID {node.id} уже существует и будет перезаписан")
        self._nodes[node.id] = node

    def remove_node(self, node_id: int) -> bool:
        """Удаляет узел из коллекции по идентификатору.

        Args:
            node_id: Идентификатор узла для удаления

        Returns:
            True если узел был удален, иначе False
        """
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    def clear_nodes(self) -> None:
        """Очищает коллекцию узлов."""
        self._nodes.clear()

    def clear_isolated_nodes(
        self, from_isolated_neighbors: bool = False, from_isolated_ways: bool = False
    ) -> None:
        """Удаляет изолированные узлы из коллекции.

        Args:
            from_isolated_neighbors: Если True, удаляет узлы без соседей.
            from_isolated_ways: Если True, удаляет узлы без путей.
            Если оба параметра False, удаляет узлы без соседей и путей.
        """
        initial_count = len(self._nodes)
        if from_isolated_neighbors:
            self._nodes = {
                node_id: node for node_id, node in self._nodes.items() if node.is_isolated_neighbors
            }
        if from_isolated_ways:
            self._nodes = {
                node_id: node for node_id, node in self._nodes.items() if node.is_isolated_ways
            }
        if not from_isolated_neighbors and not from_isolated_ways:
            self._nodes = {
                node_id: node for node_id, node in self._nodes.items() if not node.is_connected
            }
        removed_count = initial_count - len(self._nodes)
        logging.info(f"Удалено изолированных узлов: {removed_count}")
