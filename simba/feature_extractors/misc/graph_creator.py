import itertools
import pickle

import networkx as nx
import pandas as pd
from pyvis.network import Network

from simba.misc_tools import create_single_color_lst


class GraphCreator(object):
    def __init__(
        self,
        animals: list,
        data_path: str,
        edge_weight_settings: dict,
        filter_settings: dict,
    ):
        self.data = pd.read_csv(data_path)
        self.edge_weight_settings, self.animal_names = edge_weight_settings, animals
        self.filter_settings = filter_settings

    def __weigh_time(self, df: pd.DataFrame):
        results = pd.DataFrame(columns=["source", "target", "value"])
        if self.edge_weight_settings["Aggregation_method"] is "Sum":
            results.loc[len(results)] = [
                df["ROI 1"].values[0],
                df["ROI 2"].values[0],
                df["BOUT TIME (s)"].sum(),
            ]
        elif self.edge_weight_settings["Aggregation_method"] is "Mean":
            results.loc[len(results)] = [
                df["ROI 1"].values[0],
                df["ROI 2"].values[0],
                df["BOUT TIME (s)"].mean(),
            ]
        return results

    def __weigh_events(self, df: pd.DataFrame):
        results = pd.DataFrame(columns=["source", "target", "value"])
        results.loc[len(results)] = [
            df["ROI 1"].values[0],
            df["ROI 2"].values[0],
            len(df),
        ]

        return results

    def save(self, save_path: str):
        if not hasattr(self, "graphs"):
            raise ValueError()
        with open(save_path, "wb") as f:
            pickle.dump(self.graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        self.graphs = {}
        for video_name in self.data["VIDEO"].unique():
            G = nx.Graph()
            for animal in self.animal_names:
                G.add_node(animal)
            weight_df_lst = []
            node_pairs = list(itertools.combinations(self.animal_names, 2))
            for node_pair in node_pairs:
                node_pair_df = self.data.loc[
                    (self.data["ROI 1"].isin(node_pair))
                    & (self.data["ROI 2"].isin(node_pair))
                ]

                if self.edge_weight_settings["Variable"] == "Time":
                    if self.filter_settings["Time_threshold"] is not None:
                        node_pair_df = node_pair_df[
                            node_pair_df["BOUT TIME (s)"]
                            > self.filter_settings["Time_threshold"]
                        ]
                    if len(node_pair_df) > 0:
                        weight_df_lst.append(self.__weigh_time(df=node_pair_df))

                if self.edge_weight_settings["Variable"] == "Counts":
                    if len(node_pair_df) > 0:
                        weight_df_lst.append(self.__weigh_events(df=node_pair_df))

            weight_arr = pd.concat(weight_df_lst, axis=0).reset_index(drop=True).values
            for i in range(weight_arr.shape[0]):
                G.add_edge(weight_arr[i][0], weight_arr[i][1], weight=weight_arr[i][2])
            self.graphs[video_name] = G

    def page_rank(self):
        for video_name, G in self.graphs.items():
            return nx.pagerank(G, alpha=0.9, max_iter=600, weight="weight")

    def visualize(
        self,
        node_colors: dict or None,
        node_size: dict or str or None,
        style_attr: dict = {"size": ["500px", "500px"]},
    ):
        for video_name, G in self.graphs.items():
            if node_colors is not None:
                clrs = create_single_color_lst(
                    pallete_name="magma", increments=len(self.animal_names), as_hex=True
                )
                node_colors = {
                    k: v
                    for k, v in sorted(
                        node_colors.items(), key=lambda item: item[1], reverse=True
                    )
                }
                for node_cnt, node_name in enumerate(node_colors.keys()):
                    node_colors[node_name] = clrs[node_cnt]

            network_graph = Network(style_attr["size"][0], style_attr["size"][1])
            for node_name, node_attrs in G.nodes(data=True):
                network_graph.add_node(node_name, color=node_colors[node_name])

            for source, target, edge_attrs in G.edges(data=True):
                edge_attrs["value"] = edge_attrs["weight"]
                network_graph.add_edge(str(source), str(target), **edge_attrs)

            network_graph.save_graph("nx.html")


test = GraphCreator(
    data_path="/Users/simon/Desktop/envs/simba_dev/tests/test_data/misc_test_files/termite_rois.csv",
    animals=["Animal_1", "Animal_2", "Animal_3", "Animal_4", "Animal_5"],
    edge_weight_settings={"Variable": "Time", "Aggregation_method": "Sum"},
    filter_settings={"Time_threshold": 1.0, "Count_threshold": None},
)

test.run()
test.save(save_path="Test.pickle")


#
# pr = test.page_rank()
# test.visualize(node_colors=pr, node_size=pr)
