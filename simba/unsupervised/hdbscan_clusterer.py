__author__ = "Simon Nilsson"

try:
    from cuml.cluster import hdbscan
    from cuml.cluster.hdbscan import HDBSCAN

    gpu_flag = True
except ModuleNotFoundError:
    from hdbscan import HDBSCAN
    import hdbscan

import glob
import itertools
import os
import random
from copy import deepcopy
from typing import Any, Dict, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.unsupervised.umap_embedder import UmapEmbedder
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_keys_exist_in_dict,
                                check_if_list_contains_values, check_instance,
                                check_str, check_that_directory_is_empty)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import drop_df_fields, read_pickle, write_pickle


class HDBSCANClusterer(UnsupervisedMixin):
    """
    Methods for grid-search HDBSCAN model fit and transform.
    Defaults to GPU and cuml.cluster.HDBSCAN. If GPU unavailable, then hdbscan.HDBSCAN.
    """

    def __init__(self):
        super().__init__()

    def fit(self, data_path: str, save_dir: str, hyper_parameters: dict):
        """
        :param data_path: Path holding pickled unsupervised dimensionality reduction results in ``data_map.yaml`` format
        :param save_dir: Empty directory where to save the HDBSCAN results.
        :param hyper_parameters: dict holding hyperparameters in list format
        :return:

        :Example I: Grid-search fit:
        >>> hyper_parameters = {'alpha': [1.0], 'min_cluster_size': [10], 'min_samples': [1], 'cluster_selection_epsilon': [20]}
        >>> embedding_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
        >>> save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models'
        >>> config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
        >>> clusterer = HDBSCANClusterer(data_path=embedding_dir, save_dir=save_dir)
        >>> clusterer.fit(hyper_parameters=hyper_parameters)
        """

        self.save_dir, self.data_path = save_dir, data_path
        check_that_directory_is_empty(directory=self.save_dir)
        if os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path)
            check_if_filepath_list_is_empty(
                filepaths=glob.glob(data_path + "/*.pickle"),
                error_msg=f"SIMBA ERROR: No pickle files in {data_path}",
            )
        else:
            check_file_exist_and_readable(file_path=data_path)
            self.data_path = data_path

        self.search_space = list(
            itertools.product(
                *[
                    hyper_parameters[Clustering.ALPHA.value],
                    hyper_parameters[Clustering.MIN_CLUSTER_SIZE.value],
                    hyper_parameters[Clustering.MIN_SAMPLES.value],
                    hyper_parameters[Clustering.EPSILON.value],
                ]
            )
        )
        self.embeddings = read_pickle(data_path=self.data_path)
        print(
            f"Fitting {str(len(self.search_space) * len(self.embeddings.keys()))} HDBSCAN model(s)..."
        )
        self.__fit_hdbscan()
        self.timer.stop_timer()
        stdout_success(
            msg=f"{str(len(self.search_space) * len(self.embeddings.keys()))} saved in {self.save_dir}",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def __fit_hdbscan(self):
        self.model_counter = 0
        for k, v in self.embeddings.items():
            fit_timer = SimbaTimer()
            fit_timer.start_timer()
            embedder = v[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value]
            for cnt, h in enumerate(self.search_space):
                results, self.model = {}, {}
                self.model_counter += 1
                self.model_timer = SimbaTimer()
                self.model_timer.start_timer()
                self.model[Unsupervised.HASHED_NAME.value] = random.sample(
                    self.model_names, 1
                )[0]
                self.model[Unsupervised.PARAMETERS.value] = {
                    Clustering.ALPHA.value: h[0],
                    Clustering.MIN_CLUSTER_SIZE.value: h[1],
                    Clustering.MIN_SAMPLES.value: h[2],
                    Clustering.EPSILON.value: h[3],
                }
                self.model[Unsupervised.MODEL.value] = HDBSCAN(
                    algorithm="best",
                    alpha=self.model[Unsupervised.PARAMETERS.value][
                        Clustering.ALPHA.value
                    ],
                    approx_min_span_tree=True,
                    gen_min_span_tree=True,
                    min_cluster_size=self.model[Unsupervised.PARAMETERS.value][
                        Clustering.MIN_CLUSTER_SIZE.value
                    ],
                    min_samples=self.model[Unsupervised.PARAMETERS.value][
                        Clustering.MIN_SAMPLES.value
                    ],
                    cluster_selection_epsilon=self.model[Unsupervised.PARAMETERS.value][
                        Clustering.EPSILON.value
                    ],
                    p=None,
                    prediction_data=True,
                )

                self.model[Unsupervised.MODEL.value].fit(embedder.embedding_)
                results[Unsupervised.DATA.value] = v[Unsupervised.DATA.value]
                results[Unsupervised.METHODS.value] = v[Unsupervised.METHODS.value]
                results[Unsupervised.DR_MODEL.value] = v[Unsupervised.DR_MODEL.value]
                results[Clustering.CLUSTER_MODEL.value] = self.model
                self.__save(data=results)

    def __save(self, data: dict) -> None:
        write_pickle(
            data=data,
            save_path=os.path.join(
                self.save_dir, f"{self.model[Unsupervised.HASHED_NAME.value]}.pickle"
            ),
        )
        self.model_timer.stop_timer()
        stdout_success(
            msg=f"Model {self.model_counter}/{len(self.search_space) * len(list(self.embeddings.keys()))} ({self.model[Unsupervised.HASHED_NAME.value]}) saved...",
            elapsed_time=self.model_timer.elapsed_time,
        )

    def transform(
        self,
        data_path: Union[str, os.PathLike],
        model: Union[Union[str, os.PathLike, Dict[str, Any]]],
        save_dir: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[Dict[str, Any]] = None,
        save_format: Optional[Literal["csv", "pickle"]] = "csv",
    ) -> Union[None, pd.DataFrame]:
        """
        :param data_path: Path to directory holding pickled unsupervised dimensionality reduction results in ``data_map.yaml`` format
        :param model: Path to pickle holding hdbscan model in ``data_map.yaml`` format.
        :param save_dir: Empty directory where to save the HDBSCAN results. If none, then keep results in memory under self.results.
        :param settings: User-defined params.

        :Example I: Transform:
        >>> data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/logs/unsupervised_data_20240218134920.pickle'
        >>> mdl_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_mdls/hopeful_khorana.pickle'
        >>> clusterer = HDBSCANClusterer()
        >>> settings = {'DATA_FORMAT': 'scaled', 'CLASSIFICATIONS': True}
        >>> results = clusterer.transform(data_path=data_path, model=mdl_path, settings=settings)
        """

        print("Transforming data using HDBSCAN model...")
        timer = SimbaTimer(start=True)
        if isinstance(model, str):
            check_file_exist_and_readable(file_path=model)
            model = read_pickle(data_path=model)
        check_file_exist_and_readable(file_path=data_path)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
            check_str(
                name=f"{HDBSCANClusterer.transform.__name__} save_format",
                value=save_format,
                options=(
                    "csv",
                    "pickle",
                ),
            )
        data = read_pickle(data_path=data_path)
        check_if_keys_exist_in_dict(
            data=data,
            key=[Unsupervised.BOUTS_FEATURES.value, Unsupervised.BOUTS_TARGETS.value],
            name=data_path,
        )
        check_if_keys_exist_in_dict(
            data=model,
            key=[
                Unsupervised.METHODS.value,
                Unsupervised.DR_MODEL.value,
                Clustering.CLUSTER_MODEL.value,
            ],
            name=data_path,
        )
        dr_mdl_name = model[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]
        cl_mdl_name = model[Clustering.CLUSTER_MODEL.value][
            Unsupervised.HASHED_NAME.value
        ]
        data_df = deepcopy(data[Unsupervised.BOUTS_FEATURES.value])
        data_df = drop_df_fields(
            data=data_df,
            fields=model[Unsupervised.METHODS.value][
                Unsupervised.LOW_VARIANCE_FIELDS.value
            ],
        )
        scaled_data = TrainModelMixin.scaler_transform(
            data=data_df,
            scaler=model[Unsupervised.METHODS.value][Unsupervised.SCALER.value],
            name=dr_mdl_name,
        )
        check_if_list_contains_values(
            data=list(scaled_data.columns),
            values=model[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value],
            name=self.__class__.__name__,
        )
        results = pd.DataFrame(
            model[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].transform(
                scaled_data
            ),
            columns=["X", "Y"],
            index=scaled_data.index,
        )
        results["CLUSTER_LABELS"] = hdbscan.approximate_predict(
            model[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value],
            results[["X", "Y"]].values,
        )[0]
        if settings is not None:
            check_instance(
                source=f"{UmapEmbedder.transform.__name__} settings",
                instance=settings,
                accepted_types=(dict,),
            )
            if "DATA_FORMAT" in settings.keys():
                if settings["DATA_FORMAT"] == Unsupervised.SCALED.value:
                    results = pd.concat([scaled_data, results], axis=1)
                elif settings["DATA_FORMAT"] == Unsupervised.RAW.value:
                    results = pd.concat([data_df, results], axis=1)
            if "CLASSIFICATIONS" in settings.keys():
                target_data = data[Unsupervised.BOUTS_TARGETS.value].set_index(
                    ["VIDEO", "START_FRAME", "END_FRAME"]
                )
                results = results.join(target_data, how="inner")
        if save_dir:
            if save_format == Formats.CSV.value:
                save_path = os.path.join(
                    save_dir,
                    f"Transformed_{dr_mdl_name}_{cl_mdl_name}_{self.datetime}.{Formats.CSV.value}",
                )
                results.to_csv(path_or_buf=save_path)
            else:
                save_path = os.path.join(
                    save_dir,
                    f"Transformed_{dr_mdl_name}_{cl_mdl_name}_{self.datetime}.{Formats.PICKLE.value}",
                )
                write_pickle(data=results, save_path=save_path)
            timer.stop_timer()
            stdout_success(
                msg=f"Transformed data saved at {save_path} (elapsed time: {timer.elapsed_time_str}s)"
            )
        else:
            timer.stop_timer()
            stdout_success(
                msg=f"Data transformed using UMAP model {dr_mdl_name} and HDBSCAN model {cl_mdl_name} (elapsed time: {timer.elapsed_time_str}s)"
            )
            return results


# hyper_parameters = {'alpha': [1.0], 'min_cluster_size': [15], 'min_samples': [1], 'cluster_selection_epsilon': [1, 0.5]}
# embedding_dir = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/small_embeddings'
# save_dir = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/small_clusters'
# config_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini'
# clusterer = HDBSCANClusterer()
# clusterer.fit(hyper_parameters=hyper_parameters, data_path=embedding_dir, save_dir=save_dir)
#
# #
#


# hyper_parameters = {'alpha': [1.0], 'min_cluster_size': [10], 'min_samples': [1], 'cluster_selection_epsilon': [20]}
# embedding_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# clusterer = HDBSCANClusterer(data_path=embedding_dir, save_dir=save_dir)
# clusterer.fit(hyper_parameters=hyper_parameters)


# data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/logs/unsupervised_data_20240218134920.pickle'
# mdl_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_mdls/hopeful_khorana.pickle'
# clusterer = HDBSCANClusterer()
# settings = {'DATA_FORMAT': 'scaled', 'CLASSIFICATIONS': True}
# results = clusterer.transform(data_path=data_path, model=mdl_path, settings=settings)


# save_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
# clusterer = HDBSCANClusterer(data_path=data_path, save_dir=save_path)
# clusterer.transform(model='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/awesome_curran.pickle', settings={'DATA': None}, data_path=data_path)

# settings = {'feature_values': True, 'scaled_features': True, 'save_format': 'csv'}
# clusterer_model_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/amazing_burnell.pickle'
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230215093552.pickle'
# save_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'

# _ = HDBSCANTransform(clusterer_model_path=clusterer_model_path,
#                      data_path=data_path,
#                      save_dir=save_path,
#                      settings=settings)
