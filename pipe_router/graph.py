"""
Undirected graph class definition.
"""
from typing import Optional, List, Callable


class UndirectedGraph:
    """
    Represents an undirected graph. Each node and each edge can store a value.
    """

    def __init__(self):
        """
        Builds an undirected graph object.
        """
        self.edges = {}  # adjacency format: dict of dicts; e.g. {node1: {node2: val2, node3: val3}, ...}
        self.node_values = {}  # format: dict; e.g. {node1: val1, node2: val2, ...}

    def get_node_val(self, key: any) -> any:
        """
        Gets the value stored in the given node.

        :param key: node key
        :return: stored value
        """
        return self.node_values[key]

    def get_edge(self, key1: any, key2: any) -> any:
        """
        Gets the value stored in the edge connecting two nodes

        :param key1: key for first node
        :param key2: key for second node
        :return: stored value
        """
        if key1 not in self.graph or key2 not in self.graph:
            missing_keys = [key for key in [key1, key2] if key not in self.graph]
            raise KeyError(f'The following keys are missing: {missing_keys}')
        return self.graph[key1][key2]

    def set_node_val(self, key: any, value: any) -> None:
        """
        Sets the value of the given node.

        :param key: node key
        :param value: value to store
        :return: None
        """
        self.node_values[key] = value

    def set_edge(self, key1: any, key2: any, value: Optional[any] = None) -> None:
        """
        Sets the value stored in the edge between two nodes.

        :param key1: key for first node
        :param key2: key for second node
        :param value: value to store
        :return: None
        """
        if key1 not in self.graph and key2 not in self.graph:
            self.graph[key1] = {key2: value}
            self.graph[key2] = self.graph[key1]
        else:
            self.graph[key1][key2] = value

    def set_all_edges(self, val: any) -> None:
        """
        Sets the value of all edges to the given value.

        :param val: value to store in every edge
        :return: None
        """
        for key1 in self.graph:
            for key2 in self.graph[key1]:
                self.graph[key1][key2] = val

    def __str__(self):
        return self.__class__.__name__ + '\n' + str(self.edges)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.edges)
