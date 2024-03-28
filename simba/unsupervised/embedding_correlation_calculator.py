__author__ = "Simon Nilsson"

import os
from typing import Any, Dict, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict, check_instance)
from simba.utils.printing import stdout_success
from simba.utils.read_write import read_pickle

METHOD = "method"
PEARSON = "pearson"
KENDALL = "kendall"
PALETTE = "palette"
SHAP = "shap"
PLOTS = "plots"
CREATE = "create"
SPEARMAN = "spearman"
CORRELATIONS = "correlations"


class EmbeddingCorrelationCalculator(UnsupervisedMixin, ConfigReader):
    """
    Class for correlating dimensionality reduction features with original features for explainability purposes.

    .. image:: _static/img/EmbeddingCorrelationCalculator.png
       :width: 800
       :align: center

    :param str config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param str data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param dict settings: dict holding which statistical tests to use and how to create plots.

    :Example:
    >>> settings = {'correlation_methods': ['pearson', 'kendall', 'spearman'], 'plots': {'create': True, 'correlations': 'pearson', 'palette': 'jet'}}
    >>> calculator = EmbeddingCorrelationCalculator(config_path='unsupervised/project_folder/project_config.ini', data_path='unsupervised/cluster_models/quizzical_rhodes.pickle', settings=settings)
    >>> calculator.run()
    """

    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        config_path: Union[str, os.PathLike],
        settings: Dict[str, Any],
    ):

        check_file_exist_and_readable(file_path=config_path)
        check_instance(
            source=f"{self.__class__.__name__} settings",
            instance=settings,
            accepted_types=(dict,),
        )
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        check_file_exist_and_readable(file_path=data_path)
        self.settings, self.data_path = settings, data_path
        check_if_keys_exist_in_dict(
            data=settings,
            key=[CORRELATIONS, PLOTS],
            name=f"{self.__class__.__name__} settings",
        )
        self.data = read_pickle(data_path=self.data_path)
        check_if_keys_exist_in_dict(
            data=self.data,
            key=[Unsupervised.METHODS.value, Unsupervised.DR_MODEL.value],
            name=self.data_path,
        )
        self.save_path = os.path.join(
            self.logs_path,
            f"embedding_correlations_{self.data[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}.csv",
        )

    def run(self):
        print("Calculating embedding correlations...")
        self.x_df = self.data[Unsupervised.METHODS.value][
            Unsupervised.SCALED_DATA.value
        ]
        self.y_df = pd.DataFrame(
            self.data[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].embedding_,
            columns=["X", "Y"],
            index=self.x_df.index,
        )
        results = pd.DataFrame()
        for correlation_method in self.settings[CORRELATIONS]:
            results[f"{correlation_method}_Y"] = self.x_df.corrwith(
                self.y_df["Y"], method=correlation_method
            )
            results[f"{correlation_method}_X"] = self.x_df.corrwith(
                self.y_df["X"], method=correlation_method
            )
        results.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Embedding correlations saved in {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )

        if self.settings[PLOTS][CREATE]:
            print("Creating embedding correlation plots...")
            df = pd.concat([self.x_df, self.y_df], axis=1)
            save_dir = os.path.join(
                self.logs_path,
                f"embedding_correlation_plots_{self.data[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}",
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for feature_cnt, feature_name in enumerate(
                self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value]
            ):
                save_path = os.path.join(save_dir, f"{feature_name}.png")
                _ = PlottingMixin.continuous_scatter(
                    data=df,
                    columns=["X", "Y", feature_name],
                    palette=self.settings[PLOTS][PALETTE],
                    title=feature_name,
                    save_path=save_path,
                    show_box=False,
                )
                print(
                    f"Saving image {str(feature_cnt+1)}/{str(len(df.columns))} ({feature_name})"
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Embedding correlation calculations complete",
            elapsed_time=self.timer.elapsed_time_str,
        )


# settings = {'correlations': ['pearson', 'kendall', 'spearman'], 'plots': {'create': True, 'correlations': 'pearson', 'palette': 'jet'}}
# calculator = EmbeddingCorrelationCalculator(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini',
#                                             data_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters/beautiful_beaver.pickle',
#                                             settings=settings)
# calculator.run()
