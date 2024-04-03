import os
from typing import Optional, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.statistics_mixin import Statistics
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.errors import CountError
from simba.utils.read_write import read_pickle
from simba.utils.warnings import CountWarning

LOF = "local outlier factor"
EE = "elliptic envelope"


class OutlierDetector(ConfigReader):

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_path: Union[str, os.PathLike],
        algorithm: Literal[LOF, EE],
        cluster_sliced: Optional[bool] = True,
    ):

        ConfigReader.__init__(
            self, config_path=config_path, read_video_info=False, create_logger=False
        )
        data = read_pickle(data_path=data_path, verbose=True)
        self.x = data[Unsupervised.DR_MODEL.value]["MODEL"].embedding_
        idx = np.array(
            list(map(np.array, np.array(data["DATA"]["BOUTS_FEATURES"].index.values)))
        )
        self.x = np.hstack((idx, self.x))
        if cluster_sliced:
            self.y = data[Clustering.CLUSTER_MODEL.value][
                Unsupervised.MODEL.value
            ].labels_
            self.unique_labels = [x for x in list(np.unique(self.y)) if x != -1]
            if len(self.unique_labels) <= 1:
                CountWarning(
                    msg="Too few clusters for performing cluster_sliced, reverting to False",
                    source=self.__class__.__name__,
                )
                cluster_sliced = False
        else:
            self.y = None
        self.cluster_sliced, self.algorithm = cluster_sliced, algorithm

    def run(self):
        if self.cluster_sliced:
            self.results = []
            for i in self.unique_labels:
                c = self.x[np.argwhere(self.y == i)].reshape(-1, 5)
                c_id, c = c[:, 0:3], c[:, 3:5].astype(np.float32)
        #         if self.algorithm is LOF:
        #             s = Statistics.local_outlier_factor(data=c, k=0.2, normalize=True)
        #         elif self.algorithm is EE:
        #             s = Statistics.elliptic_envelope(data=c, normalize=True)
        #         self.results.append(np.hstack([c, np.full((c.shape[0], 1), i).reshape(-1, 1), s.reshape(-1, 1)]))
        #     self.results = np.concatenate(self.results, axis=0)
        #     unclustered = self.x[np.argwhere(self.y == -1)].reshape(-1, 2)
        #     if unclustered.shape[0] > 0:
        #         unclustered = np.hstack((unclustered, np.full((unclustered.shape[0], 1), -1), np.full((unclustered.shape[0], 1), np.max(self.results[:, 3]))))
        #         self.results = np.vstack((self.results, unclustered))
        #
        # else:
        #     if self.algorithm is LOF:
        #         s = Statistics.local_outlier_factor(data=self.x, k=0.2, normalize=True)
        #     elif self.algorithm is EE:
        #         s = Statistics.elliptic_envelope(data=self.x, normalize=True)
        #     self.results = np.vstack([self.x, s.reshape(-1, 1)])
        #
        #
        # print(self.results)

        # img = PlottingMixin.continuous_scatter(data=self.results, columns=('X', 'Y', 'LOF'), size=20, palette='seismic')

        # results = pd.DataFrame(np.vstack((results, unclustered)), columns=('X', 'Y', 'CLUSTER', 'LOF'))

        # img = PlottingMixin.continuous_scatter(data=self.results, columns=('X', 'Y', 'LOF'), size=20, palette='seismic')

        #
        #
        #
        #
        #
        # results = []
        # for i in [x for x in unique_clusters if x != -1]:
        #     c = x[np.argwhere(y == i)].reshape(-1, 2)
        #     lof = Statistics.local_outlier_factor(data=c, k=100)
        #     results.append(np.hstack([c, np.full((c.shape[0], 1), i).reshape(-1, 1), lof.reshape(-1, 1)]))
        #
        # results = np.concatenate(results, axis=0)
        # unclustered = x[np.argwhere(y == -1)].reshape(-1, 2)
        # unclustered = np.hstack((unclustered, np.full((unclustered.shape[0], 1), -1), np.full((unclustered.shape[0], 1), np.max(results[:, 3]))))
        # results = pd.DataFrame(np.vstack((results, unclustered)), columns=('X', 'Y', 'CLUSTER', 'LOF'))
        #
        # #results['LOF'] = (results['LOF'] - results['LOF'].min()) / (y - x) * (b - a) + a
        #
        # results = (results - results.mean()) / results.std()
        # results=(results-results.min())/(results.max()-results.min())
        #
        #
        # img = PlottingMixin.continuous_scatter(data=results, columns=('X', 'Y', 'LOF'), size=20, palette='seismic')
        #
        #


# data_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters/beautiful_beaver.pickle'
data_path = "/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/small_clusters/adoring_hoover.pickle"
x = OutlierDetector(
    data_path=data_path,
    config_path="/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini",
    algorithm=LOF,
    cluster_sliced=True,
)
x.run()
