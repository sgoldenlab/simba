import itertools
import os
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numba import jit
from pyvis.network import Network

from simba.utils.checks import (check_float, check_instance, check_int,
                                check_iterable_length, check_str,
                                check_valid_array, check_valid_hex_color)
from simba.utils.data import create_color_palette, find_ranked_colors, get_mode
from simba.utils.errors import CountError, InvalidInputError


class NetworkMixin(object):
    """
    Methods to create and analyze time-dependent graphs from pose-estimation data.

    When working with pose-estimation data for more than two animals - over extended periods - it can be beneficial to
    represent the data as a graph where the animals feature as nodes are their relationship strengths are represented as edges.

    When formatted as a graph, we can compute (i) how the relationships between animal pairs change across time and recordings,
    (ii) the relative importance's and hierarchies of individual animals within the group, or (iii) identify sub-groups with the network.

    The critical component determining the results is how edge weights are represented. These edge weight values could be the amount of time animal bounding
    boxes overlap each other, aggregate distances between the animals, or how much time animals engange in coordinated behaviors. These values can be computed
    through other methods within SimBA mixin methods.

    Very much wip and so far primarily depend on `networkx <https://github.com/networkx/networkx>`_.

    References
    ----------
    See below references for mature and reliable packages (12/2023):

    .. [1] `networkx <https://github.com/networkx/networkx>`_
    .. [2] `igraph <https://github.com/networkx/networkx>`_
    """

    def __init__(self):
        pass

    @staticmethod
    def create_graph(data: Dict[Tuple[str, str], float]) -> nx.Graph():
        """
        Create a single undirected graph with single edges from on dictionary.

        :param Dict[Tuple[str, str], float] data: A dictionary where keys are tuples representing node pairs and values are the corresponding edge weights.
        :returns nx.Graph: A networkx graph with nodes and edges defined by the input data.

        :example:
        >>> data = {('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5}
        >>> graph = NetworkMixin.create_graph(data=data)
        """

        check_instance(
            source=NetworkMixin.create_graph.__name__,
            instance=data,
            accepted_types=dict,
        )
        for k, v in data.items():
            check_instance(
                source=NetworkMixin.create_graph.__name__,
                instance=k,
                accepted_types=tuple,
            )
            check_iterable_length(
                source=f"{NetworkMixin.create_graph.__name__} {k}",
                val=len(k),
                exact_accepted_length=2,
            )
            check_instance(
                source=f"{NetworkMixin.create_graph.__name__} {v}",
                instance=v,
                accepted_types=(float, int),
            )

        G = nx.Graph()
        for node_names in data.keys():
            G.add_node(node_names[0])
            G.add_node(node_names[1])
        for node_names, edge_weight in data.items():
            G.add_edge(node_names[0], node_names[1], weight=edge_weight)
        return G

    @staticmethod
    def create_multigraph(data: Dict[Tuple[str, str], List[float]]) -> nx.MultiGraph:
        """
        Create a multi-graph from a dictionary of node pairs and associated edge weights.

        For example, creates a multi-graph where node edges represent animal relationship weights at different
        timepoints.

        :param Dict[Tuple[str, str], List[float]] data: A dictionary where keys are tuples representing node pairs, and values are lists of edge weights associated with each pair.
        :returns nx.MultiGraph: A NetworkX multigraph with nodes and edges specified by the input data. Each edge is labeled and weighted based on the provided information.

        :example:
        >>> data = {('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]}
        >>> NetworkMixin().create_multigraph(data=data)
        """

        check_instance(
            source=NetworkMixin.create_multigraph.__name__,
            instance=data,
            accepted_types=dict,
        )
        results, dict_data, G = {}, [], nx.MultiGraph()
        for k, v in data.items():
            check_instance(
                source=NetworkMixin.create_multigraph.__name__,
                instance=k,
                accepted_types=tuple,
            )
            check_instance(
                source=f"{NetworkMixin.create_multigraph.__name__} {v}",
                instance=v,
                accepted_types=list,
            )
            dict_data.append(len(v))
        check_iterable_length(
            source=f"{NetworkMixin.create_multigraph.__name__} data",
            val=len(list(set([x for x in dict_data]))),
            exact_accepted_length=1,
        )
        for node_names in data.keys():
            G.add_node(node_names[0])
            G.add_node(node_names[1])
        for node_names, edge_weights in data.items():
            for edge_cnt, edge_weight in enumerate(edge_weights):
                G.add_edge(
                    node_names[0],
                    node_names[1],
                    weight=edge_weight,
                    label=f"edge_{edge_cnt}",
                )
        return G

    @staticmethod
    def graph_page_rank(graph: nx.Graph,
                            weights: Optional[str] = "weight",
                            alpha: Optional[float] = 0.85,
                            max_iter: Optional[int] = 100) -> Dict[str, float]:

        """
        Calculate the PageRank of nodes in a graph.

        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> NetworkMixin().graph_page_rank(graph=graph)
        """

        check_instance(
            source=NetworkMixin.graph_page_rank.__name__,
            instance=graph,
            accepted_types=nx.Graph,
        )
        check_str(
            name=f"{NetworkMixin.graph_page_rank.__name__} weights", value=weights
        )
        check_float(name=f"{NetworkMixin.graph_page_rank.__name__} alpha", value=alpha)
        check_int(
            name=f"{NetworkMixin.graph_page_rank.__name__} max_iter",
            value=max_iter,
            min_value=1,
        )
        edge_weights = tuple(
            set(
                itertools.chain.from_iterable(
                    d.keys() for *_, d in graph.edges(data=True)
                )
            )
        )
        check_str(
            name=f"{NetworkMixin.graph_page_rank.__name__} weights",
            value=weights,
            options=edge_weights,
        )

        return nx.pagerank(graph, alpha=alpha, max_iter=max_iter, weight=weights)

    @staticmethod
    def graph_katz_centrality(graph: nx.Graph, weights: Optional[str] = "weight", alpha: Optional[float] = 0.85,):
        """
        Katz centrality is an algorithm in NetworkX that measures the relative influence of a node in a network.

        See networkx documentation

        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> NetworkMixin().graph_katz_centrality(graph=graph)
        """
        check_instance(source=NetworkMixin.graph_katz_centrality.__name__, instance=graph, accepted_types=nx.Graph,)
        check_str(name=f"{NetworkMixin.graph_katz_centrality.__name__} weights", value=weights)
        check_float(name=f"{NetworkMixin.graph_katz_centrality.__name__} alpha", value=alpha)
        edge_weights = tuple(
            set(
                itertools.chain.from_iterable(
                    d.keys() for *_, d in graph.edges(data=True)
                )
            )
        )
        check_str(
            name=f"{NetworkMixin.graph_katz_centrality.__name__} weights",
            value=weights,
            options=edge_weights,
        )

        return nx.katz_centrality_numpy(graph, alpha=alpha, weight=weights)

    @staticmethod
    def graph_current_flow_closeness_centrality(graph: nx.Graph, weights: Optional[str] = "weight"):

        """
        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> NetworkMixin().graph_current_flow_closeness_centrality(graph=graph)
        """

        check_instance(
            source=NetworkMixin.graph_current_flow_closeness_centrality.__name__,
            instance=graph,
            accepted_types=nx.Graph,
        )
        check_str(
            name=f"{NetworkMixin.graph_current_flow_closeness_centrality.__name__} weights",
            value=weights,
        )
        edge_weights = tuple(
            set(
                itertools.chain.from_iterable(
                    d.keys() for *_, d in graph.edges(data=True)
                )
            )
        )
        check_str(
            name=f"{NetworkMixin.graph_current_flow_closeness_centrality.__name__} weights",
            value=weights,
            options=edge_weights,
        )
        return nx.current_flow_closeness_centrality(
            graph, weight=weights, solver="full"
        )

    @staticmethod
    def girvan_newman(
        graph: nx.Graph,
        levels: Optional[int] = 1,
        most_valuable_edge: Optional[object] = None,
    ):
        """
        :example:
        >>> graph = NetworkMixin.create_graph({ ('Animal_1', 'Animal_2'): 0.0, ('Animal_1', 'Animal_3'): 0.0, ('Animal_1', 'Animal_4'): 0.0, ('Animal_1', 'Animal_5'): 0.0, ('Animal_2', 'Animal_3'): 1.0, ('Animal_2', 'Animal_4'): 1.0, ('Animal_2', 'Animal_5'): 1.0, ('Animal_3', 'Animal_4'): 1.0, ('Animal_3', 'Animal_5'): 1.0, ('Animal_4', 'Animal_5'): 1.0})
        >>> NetworkMixin().girvan_newman(graph=graph, levels = 1)
        >>> [({'Animal_1'}, {'Animal_2', 'Animal_3', 'Animal_4', 'Animal_5'})]
        """

        check_instance(
            source=NetworkMixin.multigraph_page_rank.__name__,
            instance=graph,
            accepted_types=nx.Graph,
        )
        if levels > G.number_of_nodes():
            raise CountError(
                msg=f"Number of nodes ({G.number_of_nodes()}) is less than numer of girvan newman levels ({levels}).",
                source=NetworkMixin.__class__.__name__,
            )

        communities = list(
            nx.algorithms.community.girvan_newman(
                graph, most_valuable_edge=most_valuable_edge
            )
        )
        return communities[:levels]

    @staticmethod
    def multigraph_page_rank(
        graph: nx.MultiGraph,
        weights: Optional[str] = "weight",
        alpha: Optional[float] = 0.85,
        max_iter: Optional[int] = 100,
    ) -> Dict[str, List[float]]:
        """
        Calculate multi-graph PageRank scores for each node in a MultiGraph.

        For example, each node-pair in a graph has N undirected edges representing the weighted relationship between the two nodes atobserved point in time.
        Calculates the page rank of each node at each observed time point.

        :param nx.MultiGraph graph: The input MultiGraph, created by ``NetworkMixin.create_multigraph()``.

        :example:
        >>> multigraph = NetworkMixin().create_multigraph(data={('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]})
        >>> NetworkMixin().multigraph_page_rank(graph=multigraph)
        >>> {'Animal_1': [0.06122524589028524, 0.06122524589028524, 0.06122524589028524, 0.32739635847890775], 'Animal_2': [0.06122524589028524, 0.40816213116457223, 0.06122524589028524, 0.442259400816002], 'Animal_3': [0.40816213116457223, 0.06122524589028524, 0.40816213116457223, 0.04545454545454547], 'Animal_4': [0.06122524589028524, 0.40816213116457223, 0.06122524589028524, 0.13943514979599955], 'Animal_5': [0.40816213116457223, 0.06122524589028524, 0.40816213116457223, 0.04545454545454547]}
        """

        check_instance(
            source=NetworkMixin.multigraph_page_rank.__name__,
            instance=graph,
            accepted_types=nx.MultiGraph,
        )
        edge_labels = list(set(data["label"] for _, _, data in graph.edges(data=True)))
        check_iterable_length(
            source=f"{NetworkMixin.multigraph_page_rank.__name__} edge_labels",
            val=len(edge_labels),
            min=1,
        )
        check_str(
            name=f"{NetworkMixin.graph_page_rank.__name__} weights", value=weights
        )
        check_float(name=f"{NetworkMixin.graph_page_rank.__name__} alpha", value=alpha)
        check_int(
            name=f"{NetworkMixin.graph_page_rank.__name__} max_iter",
            value=max_iter,
            min_value=1,
        )
        results = {x: [] for x in list(graph.nodes())}
        for edge_label in edge_labels:
            filtered_graph = nx.Graph(
                graph.edge_subgraph(
                    [
                        (u, v, k)
                        for u, v, k, data in graph.edges(keys=True, data=True)
                        if data.get("label") == edge_label
                    ]
                )
            )
            page_ranks = NetworkMixin().graph_page_rank(
                graph=filtered_graph, weights=weights, alpha=alpha, max_iter=max_iter
            )
            for k, v in page_ranks.items():
                results[k].append(v)

    @staticmethod
    def visualize(
        graph: Union[nx.Graph, nx.MultiGraph],
        save_path: Optional[Union[str, os.PathLike]] = None,
        node_size: Optional[Union[float, Dict[str, float]]] = 25.0,
        palette: Optional[Union[str, Dict[str, str]]] = "magma",
        img_size: Optional[Tuple[int, int]] = (500, 500),
    ) -> Union[None, Network]:
        """
        Visualizes a network graph using the vis.js library and saves the result as an HTML file.

        .. raw:: html
           :file: ../docs/_static/img/network_ex.html

        .. note::
           Multi-networks created by ``simba.mixins.network_mixin.create_multigraph`` can be a little messy to look at. Instead,
           creates seperate objects and files with single edges from each time-point.

        :param Union[nx.Graph, nx.MultiGraph] graph: The input graph to be visualized.
        :param Optional[Union[str, os.PathLike]] save_path: The path to save the HTML file. If None, the graph is not saved but returned. Default: None.
        :param Optional[Union[float, Dict[str, float]]] node_size: The size of nodes. Can be a single float or a dictionary mapping node names to their respective sizes. Default: 25.0.
        :param Optional[Union[str, Dict[str, str]]] palette: The color palette for nodes. Can be a single string representing a palette name or a dictionary mapping node names to their respective colors. Default; magma.
        :param Optional[Tuple[int, int]] img_size: The size of the resulting image in pixels, represented as (width, height). Default: 500x500.

        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> graph_pg = NetworkMixin().graph_page_rank(graph=graph)
        """

        check_instance(
            source=NetworkMixin.visualize.__name__,
            instance=graph,
            accepted_types=(nx.MultiGraph, nx.Graph),
        )
        check_instance(
            source=NetworkMixin.visualize.__name__,
            instance=node_size,
            accepted_types=(int, float, dict),
        )
        multi_graph = False
        if graph.is_multigraph() and save_path is not None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            multi_graph = True
        check_instance(
            source=NetworkMixin.visualize.__name__,
            instance=img_size,
            accepted_types=tuple,
        )
        for i in img_size:
            check_int(
                name=f"{NetworkMixin.visualize.__name__} img_size",
                value=i,
                min_value=100,
            )
        if isinstance(node_size, dict):
            if sorted(graph.nodes) != sorted(list(node_size.keys())):
                raise InvalidInputError(
                    msg=f"node_size keys do not match graph node names: {sorted(graph.nodes)} != {sorted(list(node_size.keys()))}"
                )
            for v in node_size.values:
                check_float(
                    name=f"{NetworkMixin.visualize.__name__} node_size",
                    value=v,
                    min_value=0,
                )
        else:
            check_int(
                name=f"{NetworkMixin.visualize.__name__} node size",
                value=node_size,
                min_value=1,
            )
        if isinstance(palette, dict):
            if sorted(graph.nodes) != sorted(list(palette.keys())):
                raise InvalidInputError(
                    msg=f"palette keys do not match graph node names: {sorted(graph.nodes)} != {sorted(list(node_size.keys()))}"
                )
            for v in palette.values():
                check_valid_hex_color(color_hex=str(v))
        else:
            clrs = create_color_palette(
                pallete_name=palette, as_hex=True, increments=len(list(graph.nodes()))
            )

        if not multi_graph:
            network_graph = Network(f"{img_size[0]}px", f"{img_size[1]}px")
            network_graph.set_edge_smooth("dynamic")
            for node_cnt, node_name in enumerate(graph):
                if isinstance(node_size, dict):
                    node_node_size = node_size[node_name]
                else:
                    node_node_size = node_size
                if isinstance(palette, dict):
                    node_clr = palette[node_name]
                else:
                    node_clr = clrs[node_cnt]
                network_graph.add_node(
                    n_id=node_name, shape="dot", color=node_clr, size=node_node_size
                )
            for source, target, edge_attrs in graph.edges(data=True):
                network_graph.add_edge(source, target, value=edge_attrs["weight"])
            if save_path is not None:
                network_graph.save_graph(save_path)

            return network_graph

        else:
            edge_labels = list(
                set(data["label"] for _, _, data in graph.edges(data=True))
            )
            check_iterable_length(
                source=f"{NetworkMixin.multigraph_page_rank.__name__} edge_labels",
                val=len(edge_labels),
                min=1,
            )
            for edge_label in edge_labels:
                network_graph = Network(f"{img_size[0]}px", f"{img_size[1]}px")
                network_graph.set_edge_smooth("dynamic")
                network_graph.force_atlas_2based()
                graph_save_path = os.path.join(save_path, f"{edge_label}.html")
                filtered_graph = nx.Graph(
                    graph.edge_subgraph(
                        [
                            (u, v, k)
                            for u, v, k, data in graph.edges(keys=True, data=True)
                            if data.get("label") == edge_label
                        ]
                    )
                )
                for node_cnt, node_name in enumerate(filtered_graph):
                    if isinstance(node_size, dict):
                        node_node_size = node_size[node_name]
                    else:
                        node_node_size = node_size
                    if isinstance(palette, dict):
                        node_clr = palette[node_name]
                    else:
                        node_clr = palette[node_cnt]
                    network_graph.add_node(
                        n_id=node_name, shape="dot", color=node_clr, size=node_node_size
                    )
                for source, target, edge_attrs in filtered_graph.edges(data=True):
                    network_graph.add_edge(source, target, value=edge_attrs["weight"])
                if save_path is not None:
                    network_graph.save_graph(graph_save_path)

    @staticmethod
    @jit(nopython=True)
    def simpson_index(x: np.ndarray) -> float:
        """
        Calculate Simpson's diversity index for a given array of values.

        Simpson's diversity index is a measure of diversity that takes into account the number of different categories
        present in the input data as well as the relative abundance of each category. Answer how homogenous a cluster or community is
        for categorical input variable.

        .. math::

           D = \frac{\sum(n(n-1))}{N(N-1)}

        where:
        - \( n \) is the number of individuals of a particular category,
        - \( N \) is the total number of individuals,
        - \( \sum \) represents the sum over all categories.

        :param np.ndarray x: 1-dimensional numpy array containing the values representing categories for which Simpson's index is calculated.
        :return float: Simpson's diversity index value for the input array `x`

        """

        unique_v = np.unique(x)
        n_unique = np.unique(x).shape[0]
        results = np.full((n_unique, 3), np.nan)
        for i in range(unique_v.shape[0]):
            v = unique_v[i]
            cnt = np.argwhere(x == v).flatten().shape[0]
            squared = cnt * (cnt - 1)
            results[i, :] = np.array([v, cnt, squared])
        return (np.sum(results[:, 2])) / (x.shape[0] * (x.shape[0] - 1))

    @staticmethod
    def berger_parker(x: np.ndarray) -> float:
        """
        Berger-Parker index for the given one-dimensional array.
        The Berger-Parker index is a measure of category dominance, calculated as the ratio of
        the frequency of the most abundant category to the total number of observations. Answer how dominated a cluster or community is
        by categorical variable.

        The Berger-Parker index (BP) is calculated using the formula:

        .. math::

           BP = \\frac{f_{\max}}{N}

        where:
        - \( f_{\max} \) is the frequency of the most abundant category,
        - \( N \) is the total number of observations.

        :param np.ndarray x: One-dimensional numpy array containing the values for which the Berger-Parker index is calculated.
        :return float: Berger-Parker index value for the input array `x`

        :example:
        >>> x = np.random.randint(0, 25, (100,)).astype(np.float32)
        >>> z = NetworkMixin.berger_parker(x=x)
        """
        check_valid_array(
            source=f"{NetworkMixin.berger_parker.__name__} x",
            accepted_ndims=(1,),
            data=x,
            accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8),
        )
        return get_mode(x=x) / x.shape[0]

    @staticmethod
    @jit(nopython=True)
    def shannon_diversity_index(x: np.ndarray) -> float:
        """
        Calculate the Shannon Diversity Index for a given array of categories. The Shannon Diversity Index is a measure of diversity in a
        categorical feature, taking into account both the number of different categories (richness)
        and their relative abundances (evenness). Answer how homogenous a cluster or community is
        for categorical variable. A low value indicates that one or a few categories dominate.

        .. math::

           H = -\sum_{i=1}^{n} (p_i \cdot \log(p_i))

        where:
        - \( p_i \) is the proportion of individuals belonging to the i-th category,
        - \( n \) is the total number of categories.

        :param np.ndarray x: One-dimensional numpy array containing the categories for which the Shannon Diversity Index is calculated.
        :return float: Shannon Diversity Index value for the input array `x`

        :example:
        >>> x = np.random.randint(0, 100, (100, ))
        >>> NetworkMixin.shannon_diversity_index(x=x)
        """

        unique_v = np.unique(x)
        n_unique = np.unique(x).shape[0]
        results = np.full((n_unique,), np.nan)
        for i in range(unique_v.shape[0]):
            v = unique_v[i]
            cnt = np.argwhere(x == v).flatten().shape[0]
            pi = cnt / x.shape[0]
            results[i] = pi * np.log(pi)
        return np.sum(np.abs(results))

    @staticmethod
    def margalef_diversification_index(x: np.array) -> float:
        """
        Calculate the Margalef Diversification Index for a given array of values.

        The Margalef Diversification Index is a measure of category diversity. It quantifies the richness of a community/cluster
        relative to the number of individuals. A high Margalef Diversification Index indicates a high diversity of categories relative to the number of observations.
        A low Margalef Diversification Index suggests a lower diversity of categories relative to the number of observations.

        The Margalef Diversification Index (D) is calculated using the formula:

        .. math::

           D = \\frac{(S - 1)}{\\log(N)}

        where:
        - \( S \) is the number of unique categories,
        - \( N \) is the total number of individuals.

        :param np.array x: One-dimensional numpy array containing nominal values for which the Margalef Diversification Index is calculated.
        :return float: Margalef Diversification Index value for the input array `x`


        :example:
        >>> x = np.random.randint(0, 100, (100,))
        >>> NetworkMixin.margalef_diversification_index(x=x)
        """
        check_valid_array(
            source=f"{NetworkMixin.margalef_diversification_index.__name__} x",
            accepted_ndims=(1,),
            data=x,
            accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8),
            min_axis_0=2,
        )
        n_unique = np.unique(x).shape[0]
        return (n_unique - 1) / np.log(x.shape[0])

    @staticmethod
    def menhinicks_index(x: np.array) -> float:
        """
        Calculate the Menhinick's Index for a given array of values.

        Menhinick's Index is a measure of category richness. It quantifies the number of categories relative to the square root of the total number of observations.
        A high Menhinick's Index suggests a high diversity of categories relative to the number of observations.
        A low Menhinick's Index indicates a lower diversity of categories relative to the number of observations.

        Menhinick's Index (D) is calculated using the formula:

        .. math::

           D = \\frac{S}{\\sqrt{N}}

        where:
        - \( S \) is the number of unique categories,
        - \( N \) is the total number of observations.

         :param np.array x: One-dimensional numpy array containing the integer values representing nominal values for which Menhinick's Index is calculated.
         :return float: Menhinick's Index value for the input array `x`

        :example:
        >>> x = np.random.randint(0, 5, (1000,))
        >>> NetworkMixin.menhinicks_index(x=x)
        """
        check_valid_array(
            source=f"{NetworkMixin.menhinicks_index.__name__} x",
            accepted_ndims=(1,),
            data=x,
            accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8),
            min_axis_0=2,
        )
        return np.unique(x).shape[0] / np.sqrt(x.shape[0])

    @staticmethod
    def brillouins_index(x: np.array) -> float:
        """
        Calculate Brillouin's Diversity Index for a given array of values.

        Brillouin's Diversity Index is a measure of cluster/community diversity that accounts for both richness
        and evenness of distribution.

        Brillouin's Diversity Index (H) is calculated using the formula:

        .. math::

           H = \\frac{1}{\\log(S)} \\sum_{i=1}^{S} \\frac{N_i(N_i - 1)}{n(n-1)}

        where:
        - \( H \) is Brillouin's Diversity Index,
        - \( S \) is the total number of unique species,
        - \( N_i \) is the count of individuals in the i-th species,
        - \( n \) is the total number of individuals.

        :param np.array x: One-dimensional numpy array containing the values for which Brillouin's Index is calculated.
        :return float: Brillouin's Diversity Index value for the input array `x`

        :example:
        >>> x = np.random.randint(0, 10, (100,))
        >>> NetworkMixin.brillouins_index(x)
        """

        check_valid_array(
            source=f"{NetworkMixin.brillouins_index.__name__} x",
            accepted_ndims=(1,),
            data=x,
            accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8),
            min_axis_0=2,
        )
        n_total = x.shape[0]
        n_unique = np.unique(x, return_counts=True)[1]
        if n_unique.shape[0] == 1:
            return 1.0
        else:
            S = len(n_unique)
            h = 0
            for count in n_unique:
                h += count * (count - 1)
            h /= n_total * (n_total - 1)
            h /= np.log(S)
            return h

    @staticmethod
    def sorensen_dice_coefficient(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Sørensen's Similarity Index between two communities/clusters.

        The Sørensen similarity index, also known as the overlap index, quantifies the overlap between two populations by comparing the number of shared categories to the total number of categories in both populations. It ranges from zero, indicating no overlap, to one, representing perfect overlap

        Sørensen's Similarity Index (S) is calculated using the formula:

        .. math::

            S = \\frac{2 \times |X \cap Y|}{|X| + |Y|}

        where:
        - \( S \) is Sørensen's Similarity Index,
        - \( X \) and \( Y \) are the sets representing the categories in the first and second communities, respectively,
        - \( |X \cap Y| \) is the number of shared categories between the two communities,
        - \( |X| \) and \( |Y| \) are the total number of categories in the first and second communities, respectively.


        :param x: 1D numpy array with nominal values for the first cluster/community.
        :param y: 1D numpy array with nominal values for the second cluster/community.
        :return: Sørensen's Similarity Index between x and y.

        :example:
        >>> x = np.random.randint(0, 10, (100,))
        >>> y = np.random.randint(0, 10, (100,))
        >>> NetworkMixin.sorensen_dice_coefficient(x=x, y=y)
        """

        check_valid_array(
            source=f"{NetworkMixin.sorensen_dice_coefficient.__name__} x",
            accepted_ndims=(1,),
            data=x,
            accepted_dtypes=(np.int32, np.int64, np.int8, int),
            min_axis_0=2,
        )
        check_valid_array(
            source=f"{NetworkMixin.sorensen_dice_coefficient.__name__} y",
            accepted_ndims=(1,),
            data=y,
            accepted_dtypes=(np.int32, np.int64, np.int8, int),
            min_axis_0=2,
        )
        x, y = set(x), set(y)
        return 2 * len(x.intersection(y)) / (len(x) + len(y))


# graph = NetworkMixin.create_graph(
#     {
#         ("Animal_1", "Animal_2"): 0.0,
#         ("Animal_1", "Animal_3"): 0.0,
#         ("Animal_1", "Animal_4"): 0.0,
#         ("Animal_1", "Animal_5"): 0.1,
#         ("Animal_2", "Animal_3"): 1.0,
#         ("Animal_2", "Animal_4"): 1.0,
#         ("Animal_2", "Animal_5"): 1.5,
#         ("Animal_3", "Animal_4"): 1.0,
#         ("Animal_3", "Animal_5"): 1.0,
#         ("Animal_4", "Animal_5"): 1.0,
#     }
# )
#
# G = nx.Graph(graph)
#
# NetworkMixin().graph_current_flow_closeness_centrality(graph=graph)
# #
#
# communities = list(nx.algorithms.community.(G, weight='weight'))
#
# partition = nx.algorithms.community.kernighan_lin_bisection(G, weight='weight')
# partition = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
# partition = list(nx.algorithms.community.girvan_newman(G))
#
# tuple(sorted(c) for c in next(partition))

# graph_data = {
#     ('Animal_1', 'Animal_2'): 0.0,
#     ('Animal_1', 'Animal_3'): 0.0,
#     ('Animal_1', 'Animal_4'): 0.0,
#     ('Animal_1', 'Animal_5'): 0.0,
#     ('Animal_2', 'Animal_3'): 1.5,  # Increase weight
#     ('Animal_2', 'Animal_4'): 1.0,  # Increase weight
#     ('Animal_2', 'Animal_5'): 1.5,  # Increase weight
#     ('Animal_3', 'Animal_4'): 2.9,  # Increase weight
#     ('Animal_3', 'Animal_5'): 2.0,  # Increase weight
#     ('Animal_4', 'Animal_5'): 1.0   # Decrease weight
# }

# result = list(NetworkMixin().asyn_lpa_communities(graph=graph, weight='weight'))
#
#
# for i in list(result):
#     print(i)
#


# graph_pg = NetworkMixin().graph_page_rank(graph=graph)
# graph_clrs = find_ranked_colors(data=graph_pg, palette='jet', as_hex=True)
# NetworkMixin().visualize(graph=graph,
#                          save_path='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test/project/project_data/network_html/graph_101.html',
#                          palette=graph_clrs)


# multigraph = NetworkMixin().create_multigraph(data={('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]})
#
#
# #graph_pg = NetworkMixin().graph_page_rank(graph=graph)
# graph_clrs = find_ranked_colors(data=graph_pg, palette='jet', as_hex=True)
#
# NetworkMixin().visualize(graph=graph,
#                          save_path='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test/project/project_data/network_html.html',
#                          palette=graph_clrs)


# multigraph = NetworkMixin().create_multigraph(data={('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]})
# NetworkMixin().visualize(graph=multigraph, save_path='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test/project/project_data/network_html')


# graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
# NetworkMixin().graph_page_rank(graph=graph)
# multigraph = NetworkMixin().create_multigraph(data=data)


# multigraph = NetworkMixin().create_multigraph(data=data)
#
# #multigraph.edges.data()
#
# NetworkMixin().visualize(graph=multigraph, save_dir='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test/project/project_data/network_html')


#
# for video_name, G in self.graphs.items():
#     if node_colors is not None:
#         clrs = create_single_color_lst(pallete_name='magma', increments=len(self.animal_names), as_hex=True)
#         node_colors = {k: v for k, v in sorted(node_colors.items(), key=lambda item: item[1], reverse=True)}
#         for node_cnt, node_name in enumerate(node_colors.keys()):
#             node_colors[node_name] = clrs[node_cnt]
#
#     network_graph = Network(style_attr['size'][0], style_attr['size'][1])
#     for node_name, node_attrs in G.nodes(data=True):
#         network_graph.add_node(node_name, color=node_colors[node_name])
#
#     for source, target, edge_attrs in G.edges(data=True):
#         edge_attrs['value'] = edge_attrs['weight']
#         network_graph.add_edge(str(source), str(target), **edge_attrs)
#
#     network_graph.save_graph('nx.html')
#


#
#
# data = {('Animal_1', 'Animal_2'): 0.0,
#         ('Animal_1', 'Animal_3'): 0.2,
#         ('Animal_1', 'Animal_4'): 0.3,
#         ('Animal_2', 'Animal_3'): 0.9,
#         ('Animal_2', 'Animal_4'): 1.0,
#         ('Animal_3', 'Animal_4'): 0.4}
# graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
# NetworkMixin().graph_current_flow_closeness_centrality(graph=graph)
#


#
#
# data = {('Animal_1', 'Animal_2'): [0, 0, 0, 6],
#         ('Animal_1', 'Animal_3'): [0, 0, 0, 0],
#         ('Animal_1', 'Animal_4'): [0, 0, 0, 0],
#         ('Animal_1', 'Animal_5'): [0, 0, 0, 0],
#         ('Animal_2', 'Animal_3'): [0, 0, 0, 0],
#         ('Animal_2', 'Animal_4'): [5, 0, 0, 2],
#         ('Animal_2', 'Animal_5'): [0, 0, 0, 0],
#         ('Animal_3', 'Animal_4'): [0, 0, 0, 0],
#         ('Animal_3', 'Animal_5'): [0, 2, 22, 0],
#         ('Animal_4', 'Animal_5'): [0, 0, 0, 0]}


# graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
# #NetworkMixin().graph_page_rank(graph=graph)
# multigraph = NetworkMixin().create_multigraph(data=data)
#
# #multigraph.edges.data()
#
# NetworkMixin().visualize(graph=multigraph, save_dir='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test/project/project_data/network_html')


# NetworkMixin().multigraph_page_rank(graph=multigraph)


# NetworkMixin.page_rank(graph=graph)
# NetworkMixin.katz_centrality(graph=graph)
# NetworkMixin.current_flow_closeness_centrality(graph=graph)
