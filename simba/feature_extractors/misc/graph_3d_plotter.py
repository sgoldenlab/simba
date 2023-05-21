import pickle

import networkx as nx
import numpy as np
import pyvista as pv
from fa2 import ForceAtlas2


class Graph3DPlotter(object):
    def __init__(self, networks_path: dict):
        with open(networks_path, "rb") as handle:
            self.networks = pickle.load(handle)

        force_atlas = ForceAtlas2(verbose=False)
        self.node_locations = {}
        for i in self.networks.keys():
            node_locations = force_atlas.forceatlas2_networkx_layout(
                self.networks[i], pos=None, iterations=100
            )
            for node in node_locations.keys():
                node_location = list(node_locations[node])
                node_location.append(0.0)
                node_locations[node] = node_location
            self.node_locations[i] = node_locations

    def plot(self, video_name: str):
        self.plotter = pv.Plotter()
        self.plotter.set_background("darkgrey", top="dodgerblue")
        node_locations = self.node_locations[video_name]
        node_array = np.full((len(node_locations.keys()), 3), 0.0)
        print(node_array)

        for cnt, animal_name in enumerate(node_locations):
            node_array[cnt] = np.array(node_locations[animal_name])
        node_cloud = pv.wrap(node_array)
        self.plotter.add_mesh(
            node_cloud,
            scalars=None,
            render_points_as_spheres=True,
            point_size=45,
            cmap="Accent",
            ambient=0.5,
            categories=True,
            name="non_target_baseline",
        )

        for edge in self.networks[video_name].edges.data():
            line_points = np.array(
                [
                    self.node_locations[video_name][edge[0]],
                    self.node_locations[video_name][edge[1]],
                ]
            )
            self.plotter.add_lines(line_points, color="yellow", width=edge[2]["weight"])

        self.plotter.show()


test = Graph3DPlotter(networks_path="/Users/simon/Desktop/envs/simba_dev/Test.pickle")
test.plot(video_name="termites_test")
