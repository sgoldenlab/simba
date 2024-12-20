import itertools
import os
from typing import Dict, List, Optional, Tuple, Union
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import networkx as nx
import numpy as np
from numba import jit
from pyvis.network import Network
from simba.mixins.network_mixin import NetworkMixin

from simba.utils.checks import (check_float, check_instance, check_int,
                                check_iterable_length, check_str,
                                check_valid_array, check_valid_hex_color,
                                check_valid_tuple, check_if_dir_exists, check_valid_lst)
from simba.utils.data import create_color_palette, find_ranked_colors, get_mode
from simba.utils.errors import CountError, InvalidInputError
from itertools import combinations


def graph_load_centrality(g: nx.Graph):
    check_instance(source=f'{graph_load_centrality.__name__} g', instance=g, accepted_types=(nx.Graph, ))
    load_centrality = nx.load_centrality(g, weight='weight')
    load_centrality = {key: value * 100 for key, value in load_centrality.items()}

    weighted_betweenness_centrality = nx.betweenness_centrality(g, weight='weight')
    harmonic_centrality = nx.harmonic_centrality(g, distance='weight')
    weighted_closeness_centrality = nx.closeness_centrality(g, distance='weight')
    weighted_degree_centrality = {n: sum(data['weight'] for _, _, data in g.edges(n, data=True)) for n in g.nodes()}
    eigenvector_centrality = nx.eigenvector_centrality(g, weight='weight')

    return eigenvector_centrality






animals = ['Simon', 'JJ', 'Nastacia', 'Liana', 'Roël']
animals = list(combinations(animals, 2))

weights = list(range(len(animals)*10, 0, -5))


#
#weights = np.random.randint(0, 10, size=(len(animals)))
graph_input = {}
for i in range(len(animals)): graph_input[animals[i]] = int(weights[i])
g = NetworkMixin.create_graph(data=graph_input)
load = graph_load_centrality(g=g)


page_rank = NetworkMixin.graph_page_rank(graph=g)
page_rank = {key: value * 100 if value > 0 else 0 for key, value in page_rank.items()}
katz = NetworkMixin.graph_katz_centrality(graph=g)
katz = {key: value * 100 if value > 0 else 0 for key, value in katz.items()}

graph_current_flow_closeness_centrality = NetworkMixin.graph_current_flow_closeness_centrality(graph=g)
graph_current_flow_closeness_centrality = {key: value * 1 if value > 0 else 0 for key, value in graph_current_flow_closeness_centrality.items()}


graph_clrs = find_ranked_colors(data=katz, palette='magma', as_hex=True)




NetworkMixin.visualize(graph=g,
                       node_size=load,
                       save_path='/Users/simon/Desktop/envs/simba/simba/simba/sandbox/graph.html',
                       node_shape='dot',
                       smooth_type='dynamic', palette=graph_clrs)









