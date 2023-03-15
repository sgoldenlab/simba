import pandas as pd
import pyvista as pv
import numpy as np
from simba.unsupervised.misc import read_pickle


class EmbeddingPlotter(object):
    def __init__(self,
                 data_path: str,
                 settings: dict):

        self.data, self.settings = read_pickle(data_path=data_path)['DATA'], settings
        self.embedding_data = self.data[['X', 'Y']].values
        if settings['CLUSTERS']:
            self.embedding_data = np.hstack((self.embedding_data, self.data['CLUSTER'].values.reshape(-1, 1)))

    def plot(self):
        pv.close_all()
        env = pv.Plotter(lighting='light_kit', polygon_smoothing=True)
        env.set_background(color=self.settings['COLORS']['color'], top=self.settings['COLORS']['top'])
        for cluster in np.unique(self.embedding_data[:, 2]):
            target_cluster_idx = np.argwhere(self.embedding_data[:, 2] == cluster).flatten()
            non_target_cluster_idx = [x for x in list(range(0, self.embedding_data.shape[0])) if x not in target_cluster_idx]
            target_cluster_arr = self.embedding_data[target_cluster_idx]
            non_target_cluster_arr = self.embedding_data[non_target_cluster_idx]
            target_mesh = env.add_mesh(target_cluster_arr, scalars=None, render_points_as_spheres=True, point_size=settings['POINT_SIZE'], cmap=self.settings['COLORS']['cmap'], ambient=0.5, categories=True, name='target_mesh')

        #
        #
        #
        #
        # self.plotter.legend_visible = False
        #
        # node_cloud = pv.wrap(self.embedding_data[:, 0:3])
        #
        # self.plotter.enable_eye_dome_lighting()
        #
        env.show()

    #
settings = {'video_speed': 0.4,
            'CLUSTERS': True,
            'POINT_SIZE': 15,
            'COLORS': {'color': 'gold', 'top': 'royalblue', 'cmap': 'Accent'}}


test = EmbeddingPlotter(data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/dreamy_spence_awesome_elion.pickle',
                        settings=settings)
test.plot()
