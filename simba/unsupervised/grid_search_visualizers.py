__author__ = "Simon Nilsson"

import numpy as np
import glob, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.utils.printing import stdout_success
from simba.utils.checks import check_if_dir_exists, check_if_filepath_list_is_empty


class GridSearchVisualizer(UnsupervisedMixin):
    """
    Visualize grid-searched hyper-parameters in .png format.

    :param model_dir: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param save_dir: directory holding one or more unsupervised results in pickle ``data_map.yaml`` format.
    :param settings: User-defined image attributes (e.g., continous and catehorical palettes)

    :example:
    >>> settings = {'CATEGORICAL_PALETTE': 'Pastel1', 'SCATTER_SIZE': 10}
    >>> visualizer = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models_042023', save_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/images', settings=settings)
    >>> visualizer.continuous_visualizer(continuous_vars=['START_FRAME'])
    >>> visualizer.categorical_visualizer(categoricals=['CLUSTER'])
    """
    def __init__(self,
                 model_dir: str,
                 save_dir: str,
                 settings: dict):

        super().__init__()
        check_if_dir_exists(in_dir=save_dir)
        check_if_dir_exists(in_dir=model_dir)
        self.save_dir, self.settings, self.model_dir = save_dir, settings, model_dir
        self.data_path = glob.glob(model_dir + '/*.pickle')
        check_if_filepath_list_is_empty(filepaths=self.data_path, error_msg=f'SIMBA ERROR: No pickle files in {model_dir}')

    def __join_data(self, data: object):
        embedding_data = pd.DataFrame(data[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].embedding_, columns=['X', 'Y'])
        bouts_data = data[Unsupervised.DATA.value][Unsupervised.BOUTS_FEATURES.value]
        target_data = data[Unsupervised.DATA.value][Unsupervised.BOUTS_TARGETS.value]
        cluster_data = pd.DataFrame(data[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value].labels_.reshape(-1, 1).astype(np.int8), columns=['CLUSTER'])
        data = pd.concat([embedding_data, bouts_data, target_data, cluster_data], axis=1)
        return data.loc[:, ~data.columns.duplicated()].copy()

    def categorical_visualizer(self,
                               categoricals: list):
        data = self.read_pickle(self.model_dir)
        for k, v in data.items():
            self.check_key_exist_in_object(object=v, key=Unsupervised.DR_MODEL.value)
        for k, v in data.items():
            data = self.__join_data(data=v)
            for variable in categoricals:
                save_path = os.path.join(self.save_dir, f'{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{variable}.png')
                if os.path.isfile(save_path):
                    continue
                else:
                    sns.scatterplot(data=data, x='X', y='Y', hue=variable, palette=sns.color_palette(self.settings['CATEGORICAL_PALETTE'], len(data[variable].unique())))
                    plt.savefig(save_path)
                    stdout_success(msg=f'Saved {save_path}...')
                    plt.close('all')
        self.timer.stop_timer()
        stdout_success(msg='All cluster images created.', elapsed_time=self.timer.elapsed_time_str)

    def continuous_visualizer(self,
                             continuous_vars: list):
        data = self.read_pickle(self.model_dir)
        for k, v in data.items():
            self.check_key_exist_in_object(object=v, key=Unsupervised.DR_MODEL.value)
        for k, v in data.items():
            data = self.__join_data(data=v)
            for variable in continuous_vars:
                save_path = os.path.join(self.save_dir, f'{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{variable}.png')
                fig, ax = plt.subplots()
                plt.xlabel('X')
                plt.ylabel('Y')
                points = ax.scatter(data['X'], data['Y'], c=data[variable], s=self.settings['SCATTER_SIZE'], cmap=self.settings['CONTINUOUS_PALETTE'])
                cbar = fig.colorbar(points)
                cbar.set_label(variable, loc='center')
                title = v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]
                plt.title(title, ha="center", fontsize=15, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 0})
                fig.savefig(save_path)
                plt.close('all')
                stdout_success(msg=f'Saved {save_path}...')
        self.timer.stop_timer()
        stdout_success(msg='All cluster images created.', elapsed_time=self.timer.elapsed_time_str)



# settings = {'PALETTE': 'Pastel1'}
# test = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models',
#                             save_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/images',
#                             settings=settings)
# test.cluster_visualizer()


# settings = {'CATEGORICAL_PALETTE': 'Pastel1', 'SCATTER_SIZE': 10}
# test = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models_042023',
#                             save_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/images',
#                             settings=settings)
# #test.continuous_visualizer(continuous_vars=['START_FRAME'])
# test.categorical_visualizer(categoricals=['CLUSTER'])