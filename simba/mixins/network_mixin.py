import os

import networkx as nx
from pyvis.network import Network
from typing import Dict, Tuple, Optional, List, Union
import itertools

from simba.utils.checks import (check_instance,
                                check_iterable_length,
                                check_float,
                                check_int,
                                check_str,
                                check_if_dir_exists)
from simba.utils.data import create_color_palette

class NetworkMixin(object):

    """
    Methods to create and analyze time-dependent graphs from pose-estimation data.

    Very much wip and so far primarily depend on `networkx <https://github.com/networkx/networkx>`_.
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

        check_instance(source=NetworkMixin.create_graph.__name__, instance=data, accepted_types=dict)
        for k, v in data.items():
            check_instance(source=NetworkMixin.create_graph.__name__, instance=k, accepted_types=tuple)
            check_iterable_length(source=f'{NetworkMixin.create_graph.__name__} {k}', val=len(k), exact_accepted_length=2)
            check_instance(source=f'{NetworkMixin.create_graph.__name__} {v}', instance=v, accepted_types=float)

        G = nx.Graph()
        for node_names in data.keys():
            G.add_node(node_names[0]); G.add_node(node_names[1])
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

        check_instance(source=NetworkMixin.create_multigraph.__name__, instance=data, accepted_types=dict)
        results, dict_data, G = {}, [], nx.MultiGraph()
        for k, v in data.items():
            check_instance(source=NetworkMixin.create_multigraph.__name__, instance=k, accepted_types=tuple)
            check_instance(source=f'{NetworkMixin.create_multigraph.__name__} {v}', instance=v, accepted_types=list)
            dict_data.append(len(v))
        check_iterable_length(source=f'{NetworkMixin.create_multigraph.__name__} data', val=len(list(set([x for x in dict_data]))), exact_accepted_length=1)
        for node_names in data.keys():
            G.add_node(node_names[0]); G.add_node(node_names[1])
        for node_names, edge_weights in data.items():
            for edge_cnt, edge_weight in enumerate(edge_weights):
                G.add_edge(node_names[0], node_names[1], weight=edge_weight, label=f'edge_{edge_cnt}')
        return G

    @staticmethod
    def graph_page_rank(graph: nx.Graph,
                        weights: Optional[str] = 'weight',
                        alpha: Optional[float] = 0.85,
                        max_iter: Optional[int] = 100) -> Dict[str, float]:

        """
        Calculate the PageRank of nodes in a graph.

        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> NetworkMixin().graph_page_rank(graph=graph)
        """

        check_instance(source=NetworkMixin.graph_page_rank.__name__, instance=graph, accepted_types=nx.Graph)
        check_str(name=f'{NetworkMixin.graph_page_rank.__name__} weights', value=weights)
        check_float(name=f'{NetworkMixin.graph_page_rank.__name__} alpha', value=alpha)
        check_int(name=f'{NetworkMixin.graph_page_rank.__name__} max_iter', value=max_iter, min_value=1)
        edge_weights = tuple(set(itertools.chain.from_iterable(d.keys() for *_, d in graph.edges(data=True))))
        check_str(name=f'{NetworkMixin.graph_page_rank.__name__} weights', value=weights, options=edge_weights)

        return nx.pagerank(graph, alpha=alpha, max_iter=max_iter, weight=weights)

    @staticmethod
    def graph_katz_centrality(graph: nx.Graph,
                              weights: Optional[str] = 'weight',
                              alpha: Optional[float] = 0.85):
        """
        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> NetworkMixin().graph_katz_centrality(graph=graph)
        """
        check_instance(source=NetworkMixin.graph_katz_centrality.__name__, instance=graph, accepted_types=nx.Graph)
        check_str(name=f'{NetworkMixin.graph_katz_centrality.__name__} weights', value=weights)
        check_float(name=f'{NetworkMixin.graph_katz_centrality.__name__} alpha', value=alpha)
        edge_weights = tuple(set(itertools.chain.from_iterable(d.keys() for *_, d in graph.edges(data=True))))
        check_str(name=f'{NetworkMixin.graph_katz_centrality.__name__} weights', value=weights, options=edge_weights)

        return nx.katz_centrality_numpy(graph, alpha=alpha, weight=weights)

    @staticmethod
    def graph_current_flow_closeness_centrality(graph: nx.Graph,
                                                weights: Optional[str] = 'weight'):

        """

        :example:
        >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
        >>> NetworkMixin().graph_current_flow_closeness_centrality(graph=graph)
        """

        check_instance(source=NetworkMixin.graph_current_flow_closeness_centrality.__name__, instance=graph, accepted_types=nx.Graph)
        check_str(name=f'{NetworkMixin.graph_current_flow_closeness_centrality.__name__} weights', value=weights)
        edge_weights = tuple(set(itertools.chain.from_iterable(d.keys() for *_, d in graph.edges(data=True))))
        check_str(name=f'{NetworkMixin.graph_current_flow_closeness_centrality.__name__} weights', value=weights, options=edge_weights)
        return nx.current_flow_closeness_centrality(graph, weight=weights, solver='full')

    @staticmethod
    def multigraph_page_rank(graph: nx.MultiGraph,
                             weights: Optional[str] = 'weight',
                             alpha: Optional[float] = 0.85,
                             max_iter: Optional[int] = 100) -> Dict[str, List[float]]:

        """
        Calculate multi-graph PageRank scores for each node in a MultiGraph.

        For example, each node-pair in a graph has N undirected edges representing the weighted relationship between the two nodes atobserved point in time. Calculates the page rank of each node at each observed time point.

        :param nx.MultiGraph graph: The input MultiGraph, created by ``NetworkMixin.create_multigraph()``.

        :example:
        >>> multigraph = NetworkMixin().create_multigraph(data={('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]})
        >>> NetworkMixin().multigraph_page_rank(graph=multigraph)
        >>> {'Animal_1': [0.06122524589028524, 0.06122524589028524, 0.06122524589028524, 0.32739635847890775], 'Animal_2': [0.06122524589028524, 0.40816213116457223, 0.06122524589028524, 0.442259400816002], 'Animal_3': [0.40816213116457223, 0.06122524589028524, 0.40816213116457223, 0.04545454545454547], 'Animal_4': [0.06122524589028524, 0.40816213116457223, 0.06122524589028524, 0.13943514979599955], 'Animal_5': [0.40816213116457223, 0.06122524589028524, 0.40816213116457223, 0.04545454545454547]}
        """

        check_instance(source=NetworkMixin.multigraph_page_rank.__name__, instance=graph, accepted_types=nx.MultiGraph)
        edge_labels = list(set(data['label'] for _, _, data in graph.edges(data=True)))
        check_iterable_length(source=f'{NetworkMixin.multigraph_page_rank.__name__} edge_labels', val=len(edge_labels), min=1)
        check_str(name=f'{NetworkMixin.graph_page_rank.__name__} weights', value=weights)
        check_float(name=f'{NetworkMixin.graph_page_rank.__name__} alpha', value=alpha)
        check_int(name=f'{NetworkMixin.graph_page_rank.__name__} max_iter', value=max_iter, min_value=1)
        results = {x: [] for x in list(graph.nodes())}
        for edge_label in edge_labels:
            filtered_graph = nx.Graph(graph.edge_subgraph([(u, v, k) for u, v, k, data in graph.edges(keys=True, data=True) if data.get('label') == edge_label]))
            page_ranks = NetworkMixin().graph_page_rank(graph=filtered_graph, weights=weights, alpha=alpha, max_iter=max_iter)
            for k, v in page_ranks.items():
                results[k].append(v)

    @staticmethod
    def visualize(graph: Union[nx.Graph, nx.MultiGraph],
                  save_dir: Union[str, os.PathLike],
                  palette: Optional[str] = 'magma',
                  node_size: Optional[Union[int, Dict[str, float]]] = 5,
                  img_size: Optional[Tuple[int, int]] = (500, 500)) -> None:

        check_instance(source=NetworkMixin.visualize.__name__, instance=graph, accepted_types=(nx.MultiGraph, nx.Graph))
        check_instance(source=NetworkMixin.visualize.__name__, instance=img_size, accepted_types=tuple)
        check_if_dir_exists(in_dir=save_dir)
        save_path = os.path.join(save_dir, 'network.html')
        check_int(name=f'{NetworkMixin.visualize.__name__} node size', value=node_size, min_value=1)
        for i in img_size: check_int(name=f'{NetworkMixin.visualize.__name__} img_size', value=i, min_value=100)
        clrs = create_color_palette(pallete_name=palette, as_hex=True, increments=len(list(graph.nodes())))

        network_graph = Network(f'{img_size[0]}px', f'{img_size[1]}px')
        network_graph.set_edge_smooth('dynamic')
        for node_cnt, node_name in enumerate(graph):
            network_graph.add_node(n_id=node_name, shape='dot', color=clrs[node_cnt])

        network_graph.save_graph(save_path)


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
#

#NetworkMixin().multigraph_page_rank(graph=multigraph)



#NetworkMixin.page_rank(graph=graph)
#NetworkMixin.katz_centrality(graph=graph)
#NetworkMixin.current_flow_closeness_centrality(graph=graph)