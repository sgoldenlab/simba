__author__ = "Simon Nilsson"

import glob
import os
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, UMLOptions, Unsupervised
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_keys_exist_in_dict, check_instance,
                                check_int, check_str, check_that_column_exist,
                                check_valid_lst)
from simba.utils.enums import Formats, Options
from simba.utils.printing import stdout_success
from simba.utils.read_write import read_pickle


class GridSearchVisualizer(UnsupervisedMixin):
    """
    Visualize grid-searched latent spaces in .png format.

    .. image:: _static/img/GridSearchVisualizer.png
       :width: 800
       :align: center

    :param model_dir: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param save_dir: directory holding one or more unsupervised results in pickle ``data_map.yaml`` format.
    :param settings: User-defined image attributes (e.g., continous and catehorical palettes)

    :example:
    >>> settings = {'CATEGORICAL_PALETTE': 'Pastel1', 'CONTINUOUS_PALETTE': 'magma', 'SCATTER_SIZE': 10}
    >>> visualizer = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models_042023', save_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/images', settings=settings)
    >>> visualizer.continuous_visualizer(continuous_vars=['START_FRAME'])
    >>> visualizer.categorical_visualizer(categoricals=['CLUSTER'])
    """

    def __init__(
        self,
        model_dir: Union[str, os.PathLike],
        save_dir: Union[str, os.PathLike],
        settings: Dict[str, Any],
    ):
        super().__init__()
        check_if_dir_exists(in_dir=save_dir)
        check_if_dir_exists(in_dir=model_dir)
        self.save_dir, self.settings, self.model_dir = save_dir, settings, model_dir
        self.data_paths = glob.glob(model_dir + f"/*.{Formats.PICKLE.value}")
        check_if_keys_exist_in_dict(
            data=self.settings,
            key=["CATEGORICAL_PALETTE", "CONTINUOUS_PALETTE", "SCATTER_SIZE"],
            name=f"{self.__class__.__name__} settings",
        )
        check_if_filepath_list_is_empty(
            filepaths=self.data_paths,
            error_msg=f"SIMBA ERROR: No pickle files found in {model_dir}",
        )
        check_int(
            name=f"{self.__class__.__name__} scatter size",
            value=self.settings["SCATTER_SIZE"],
            min_value=1,
        )
        check_str(
            name=f"{self.__class__.__name__} categorical palette",
            value=self.settings["CATEGORICAL_PALETTE"],
            options=Options.PALETTE_OPTIONS_CATEGORICAL.value,
        )
        check_str(
            name=f"{self.__class__.__name__} continuous palette",
            value=self.settings["CONTINUOUS_PALETTE"],
            options=Options.PALETTE_OPTIONS.value,
        )

    def __extract_plot_data(self, data: dict):
        embedding_data = pd.DataFrame(
            data[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].embedding_,
            columns=["X", "Y"],
        )
        bouts_data = data[Unsupervised.DATA.value][
            Unsupervised.BOUTS_FEATURES.value
        ].reset_index()
        target_data = data[Unsupervised.DATA.value][Unsupervised.BOUTS_TARGETS.value]
        cluster_data = pd.DataFrame(
            data[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value]
            .labels_.reshape(-1, 1)
            .astype(np.int8),
            columns=["CLUSTER"],
        )
        data = pd.concat(
            [embedding_data, bouts_data, target_data, cluster_data], axis=1
        )
        return data.loc[:, ~data.columns.duplicated()].copy()

    def categorical_visualizer(self, categorical_vars: List[str]):
        check_valid_lst(
            data=categorical_vars,
            source=self.__class__.__name__,
            valid_dtypes=(str,),
            min_len=1,
        )
        print(
            f"Creating {len(categorical_vars)} categorical plot(s) from {len(self.data_paths)} data files..."
        )
        for file_cnt, file_path in enumerate(self.data_paths):
            v = read_pickle(data_path=file_path)
            check_if_keys_exist_in_dict(
                data=v,
                key=[Unsupervised.DR_MODEL.value, Unsupervised.DATA.value],
                name=file_path,
            )
            data = self.__extract_plot_data(data=v)
            for variable in categorical_vars:
                check_that_column_exist(
                    df=data,
                    column_name=variable,
                    file_name=v[Unsupervised.DR_MODEL.value][
                        Unsupervised.HASHED_NAME.value
                    ],
                )
                save_path = os.path.join(
                    self.save_dir,
                    f"{v[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value]}_{variable}.png",
                )
                _ = PlottingMixin.categorical_scatter(
                    data=data,
                    columns=("X", "Y", variable),
                    palette=self.settings["CATEGORICAL_PALETTE"],
                    size=self.settings["SCATTER_SIZE"],
                    save_path=save_path,
                    show_box=False,
                )
                stdout_success(msg=f"Saved {save_path}...")
                plt.close("all")
        self.timer.stop_timer()
        stdout_success(
            msg=f"{int(len(categorical_vars) * len(self.data_paths))} categorical plot(s) created.",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def continuous_visualizer(self, continuous_vars: List[str]):
        check_valid_lst(
            data=continuous_vars,
            source=self.__class__.__name__,
            valid_dtypes=(str,),
            min_len=1,
        )
        print(
            f"Creating {len(continuous_vars)} categorical plot(s) from {len(self.data_paths)} data files..."
        )
        for file_cnt, file_path in enumerate(self.data_paths):
            v = read_pickle(data_path=file_path)
            check_if_keys_exist_in_dict(
                data=v,
                key=[Unsupervised.DR_MODEL.value, Unsupervised.DATA.value],
                name=file_path,
            )
            data = self.__extract_plot_data(data=v)
            for variable in continuous_vars:
                check_that_column_exist(
                    df=data,
                    column_name=variable,
                    file_name=v[Unsupervised.DR_MODEL.value][
                        Unsupervised.HASHED_NAME.value
                    ],
                )
                save_path = os.path.join(
                    self.save_dir,
                    f"{v[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value]}_{variable}.png",
                )
                _ = PlottingMixin.continuous_scatter(
                    data=data,
                    columns=("X", "Y", variable),
                    palette=self.settings["CONTINUOUS_PALETTE"],
                    size=self.settings["SCATTER_SIZE"],
                    save_path=save_path,
                    show_box=False,
                )
                plt.close("all")
                stdout_success(msg=f"Saved {save_path}...")
        self.timer.stop_timer()
        stdout_success(
            msg=f"{int(len(continuous_vars) * len(self.data_paths))} continuous plot(s) created.",
            elapsed_time=self.timer.elapsed_time_str,
        )


# #
# settings = {'CATEGORICAL_PALETTE': 'tab20', 'CONTINUOUS_PALETTE': 'magma', 'SCATTER_SIZE': 10}
# test = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters',
#                             save_dir='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/cluster_vis',
#                             settings=settings)
# test.categorical_visualizer(categorical_vars=['CLUSTER'])
# test.continuous_visualizer(continuous_vars=['PROBABILITY'])


# settings = {'CATEGORICAL_PALETTE': 'Pastel1', 'CONTINUOUS_PALETTE': 'magma', 'SCATTER_SIZE': 10}
# test = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/clustering_test_2',
#                             save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_viz_2',
#                             settings=settings)
# test.categorical_visualizer(categorical_vars=['CLASSIFIER'])


# settings = {'CATEGORICAL_PALETTE': 'Pastel1', 'CONTINUOUS_PALETTE': 'magma', 'SCATTER_SIZE': 10}
# test = GridSearchVisualizer(model_dir='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_mdls',
#                             save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_vis',
#                             settings=settings)
# #test.categorical_visualizer(categorical_vars=['VIDEO', 'CLUSTER'])
# test.continuous_visualizer(continuous_vars=['START_FRAME'])

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
