from simba.mixins.network_mixin import NetworkMixin
import networkx as nx
import os
from typing import Union, Optional, Tuple, Dict
from pyvis.network import Network
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import check_instance, check_valid_tuple, check_if_dir_exists, check_float, check_int, check_valid_hex_color, check_valid_lst
from simba.utils.errors import InvalidInputError
from simba.utils.data import create_color_palette




def visualize(graph: Union[nx.Graph, nx.MultiGraph],
              save_path: Optional[Union[str, os.PathLike]] = None,
              node_size: Optional[Union[float, Dict[str, float]]] = 25.0,
              palette: Optional[Union[str, Dict[str, str]]] = "magma",
              node_shape: Optional[Literal['dot', 'ellipse', 'circle']] = 'dot',
              smooth_type: Optional[Literal['dynamic', 'continuous', 'discrete', 'diagonalCross', 'straightCross', 'horizontal', 'vertical', 'curvedCW', 'curvedCCW', 'cubicBezier']] = 'dynamic',
              img_size: Optional[Tuple[int, int]] = (500, 500)) -> Union[None, Network]:

    """
    Visualizes a network graph using the vis.js library and saves the result as an HTML file.

    .. raw:: html
       :file: ../docs/_static/img/network_ex.html

    .. note::
       Multi-networks created by ``simba.mixins.network_mixin.create_multigraph`` can be a little messy to look at. Instead,
       creates seperate objects and files with single edges from each time-point.

    :param Union[nx.Graph, nx.MultiGraph] graph: The input graph to be visualized.
    :param Optional[Union[str, os.PathLike]] save_path: The path to save the HTML file. If multi-graph,  pass a directory path. If None, the graph(s) are returned but not saved.
    :param Optional[Union[float, Dict[str, float]]] node_size: The size of nodes. Can be a single float or a dictionary mapping node names to their respective sizes. Default: 25.0.
    :param Optional[Union[str, Dict[str, str]]] palette: The color palette for nodes. Can be a single string representing a palette name or a dictionary mapping node names to their respective colors. Default; magma.
    :param Optional[Tuple[int, int]] img_size: The size of the resulting image in pixels, represented as (width, height). Default: 500x500.
    :param Optional[Literal['dot', 'ellipse', 'circle']] node_shape: The shape of the nodes. Default: `dot`.
    :param Optional[Literal] smooth_type: The dynamics of the interactive graph.

    :example:
    >>> graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
    >>> graph_pg = visualize(graph=graph, node_size={'Animal_1': 10, 'Animal_2': 26, 'Animal_3': 50}, save_path='/Users/simon/Downloads/graph.html', node_shape='box', palette='spring')
    >>> multigraph = NetworkMixin().create_multigraph(data={('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]})
    >>> graph_pg = visualize(graph=multigraph, node_size={'Animal_1': 10, 'Animal_2': 26, 'Animal_3': 50, 'Animal_4': 50, 'Animal_5': 50}, save_path='/Users/simon/Downloads/graphs', node_shape='box', palette='spring', smooth_type='diagonalCross')

    """

    check_instance(source=NetworkMixin.visualize.__name__, instance=graph, accepted_types=(nx.MultiGraph, nx.Graph))
    check_instance(source=NetworkMixin.visualize.__name__, instance=node_size, accepted_types=(int, float, dict))
    multi_graph = False
    if graph.is_multigraph(): multi_graph = True
    if multi_graph:
        check_if_dir_exists(in_dir=save_path, source=NetworkMixin.visualize.__name__)
    check_valid_tuple(x=img_size, accepted_lengths=(2,), valid_dtypes=(int,))
    if isinstance(node_size, dict):
        if sorted(graph.nodes) != sorted(list(node_size.keys())):
            raise InvalidInputError(msg=f"node_size keys do not match graph node names: {sorted(graph.nodes)} != {sorted(list(node_size.keys()))}")
        for v in node_size.values():
            check_float(name=f"{NetworkMixin.visualize.__name__} node_size", value=v, min_value=0)
    else:
        check_int(name=f"{NetworkMixin.visualize.__name__} node size", value=node_size, min_value=1)
    if isinstance(palette, dict):
        if sorted(graph.nodes) != sorted(list(palette.keys())):
            raise InvalidInputError(
                msg=f"palette keys do not match graph node names: {sorted(graph.nodes)} != {sorted(list(node_size.keys()))}")
        for v in palette.values():
            check_valid_hex_color(color_hex=str(v))
        clrs = palette
    else:
        clrs = create_color_palette(pallete_name=palette, as_hex=True, increments=len(list(graph.nodes())))

    if not multi_graph:
        network_graph = Network(f"{img_size[0]}px", f"{img_size[1]}px")
        network_graph.set_edge_smooth(smooth_type)
        for node_cnt, node_name in enumerate(graph):
            if isinstance(node_size, dict): node_node_size = node_size[node_name]
            else: node_node_size = node_size
            if isinstance(palette, dict):
                node_clr = palette[node_name]
            else:
                node_clr = clrs[node_cnt]
            network_graph.add_node(n_id=node_name, shape=node_shape, color=node_clr, size=node_node_size)
        for source, target, edge_attrs in graph.edges(data=True):
            network_graph.add_edge(source, target, value=edge_attrs["weight"])
        if save_path is not None:
            network_graph.save_graph(save_path)
        return network_graph

    else:
        results = {}
        edge_labels = list(set(data["label"] for _, _, data in graph.edges(data=True)))
        check_valid_lst(source=f"{NetworkMixin.multigraph_page_rank.__name__} edge_labels", data=edge_labels, min_len=1)
        for edge_label in edge_labels:
            network_graph = Network(f"{img_size[0]}px", f"{img_size[1]}px")
            network_graph.set_edge_smooth(smooth_type)
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
                    node_clr = clrs[node_cnt]
                network_graph.add_node(n_id=node_name, shape="dot", color=node_clr, size=node_node_size)
            for source, target, edge_attrs in filtered_graph.edges(data=True):
                network_graph.add_edge(source, target, value=edge_attrs["weight"])
            if save_path is not None:
                network_graph.save_graph(graph_save_path)
            results[edge_label] = network_graph
        return results
    #

# graph = NetworkMixin.create_graph(data={('Animal_1', 'Animal_2'): 1.0, ('Animal_1', 'Animal_3'): 0.2, ('Animal_2', 'Animal_3'): 0.5})
# graph_pg = visualize(graph=graph, node_size={'Animal_1': 10, 'Animal_2': 26, 'Animal_3': 50}, save_path='/Users/simon/Downloads/graph.html', node_shape='box', palette='spring')
# multigraph = NetworkMixin().create_multigraph(data={('Animal_1', 'Animal_2'): [0, 0, 0, 6], ('Animal_1', 'Animal_3'): [0, 0, 0, 0], ('Animal_1', 'Animal_4'): [0, 0, 0, 0], ('Animal_1', 'Animal_5'): [0, 0, 0, 0], ('Animal_2', 'Animal_3'): [0, 0, 0, 0], ('Animal_2', 'Animal_4'): [5, 0, 0, 2], ('Animal_2', 'Animal_5'): [0, 0, 0, 0], ('Animal_3', 'Animal_4'): [0, 0, 0, 0], ('Animal_3', 'Animal_5'): [0, 2, 22, 0], ('Animal_4', 'Animal_5'): [0, 0, 0, 0]})
# graph_pg = visualize(graph=multigraph, node_size={'Animal_1': 10, 'Animal_2': 26, 'Animal_3': 50, 'Animal_4': 50, 'Animal_5': 50}, save_path='/Users/simon/Downloads/graphs', node_shape='box', palette='spring', smooth_type='diagonalCross')
