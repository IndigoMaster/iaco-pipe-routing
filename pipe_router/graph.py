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
        try:
            return self.edges[key1][key2]
        except KeyError:
            pass

        if key1 not in self.edges or key2 not in self.edges:
            raise KeyError(f'The following adjacency keys are missing: {[key for key in [key1, key2] if key not in self.edges]}')
        if key2 not in self.edges[key1]:
            raise KeyError(f'Node key "{key2}" is not connected to node key "{key1}"')

    def get_connections_to(self, key: any) -> List[any]:
        """
        Gets all nodes (keys) connected to this node (key).

        :param key: key to inspect
        :return: list of keys, possibly empty
        """
        if key not in self.edges:
            return []
        else:
            return list(self.edges[key].keys())

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
        if key1 not in self.edges.keys():
            self.edges[key1] = {key2: value}
        else:
            # self.edges[key1] |= {key2: value}
            self.edges[key1][key2] = value

        if key2 not in self.edges.keys():
            self.edges[key2] = {key1: value}
        else:
            # self.edges[key2] |= {key1: value}
            self.edges[key2][key1] = value

    def set_all_edges(self, val: any) -> None:
        """
        Sets the value of all edges to the given value.

        :param val: value to store in every edge
        :return: None
        """
        for key1 in self.edges.keys():
            for key2 in self.edges[key1].keys():
                self.edges[key1][key2] = val

    def update_edges(self, func: Callable[[any], any]) -> None:
        """
        Updates every graph edge using the provided function object.

        :param func: must accept graph edge value input and return an updated value
        """
        for key1 in self.edges:
            for key2 in self.edges[key1]:
                val = func(self.get_edge(key1, key2))
                self.set_edge(key1, key2, val)

    def max_value(self) -> any:
        """
        Gets the maximum value or object stored on a graph edge.

        :return: max value or object
        """
        _key1 = next(iter(self.edges.keys()))
        _key2 = next(iter(self.edges[_key1].keys()))
        max_val = self.edges[_key1][_key2]
        for key1 in self.edges:
            for key2 in self.edges[key1]:
                max_val = max(max_val, self.edges[key1][key2])
        return max_val

    def normalize_edges(self) -> None:
        """
        Normalizes all values stored on graph edges to [0,1] range.
        Assumes all objects/values support this operation.

        :return: None
        """
        max_val = self.max_value()
        self.update_edges(lambda val: val / max_val)

    def __str__(self):
        return self.__class__.__name__ + '\n' + str(self.edges)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.edges)
