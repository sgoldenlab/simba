__author__ = "Simon Nilsson"

import datetime
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
from simba.unsupervised.enums import Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_if_list_contains_values, check_instance,
                                check_str, check_that_directory_is_empty,
                                check_umap_hyperparameters)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (drop_df_fields, read_pickle, write_df,
                                    write_pickle)

try:
    from cuml import UMAP

    gpu_flag = True
except ModuleNotFoundError:
    from umap import UMAP


class UmapEmbedder(UnsupervisedMixin):
    """
    Methods for grid-search UMAP model fit and transform.
    Defaults to GPU and cuml.UMAP if GPU available. If GPU unavailable, then umap.UMAP.

    :param data_path: Path holding pickled data-set created by `simba.unsupervised.dataset_creator.DatasetCreator.
    :param save_dir: Empty directory where to save the UMAP results.
    :param hyper_parameters: dict holding UMAP hyperparameters in list format.

    :Example I: Fit.
    >>> hyper_parameters = {'n_neighbors': [10, 2], 'min_distance': [1.0], 'spread': [1.0], 'scaler': 'MIN-MAX', 'variance': 0.25, "multicolinearity": 0.5}
    >>> data_path = 'unsupervised/project_folder/logs/unsupervised_data_20230416145821.pickle'
    >>> save_dir = 'unsupervised/dr_models'
    >>> config_path = 'unsupervised/project_folder/project_config.ini'
    >>> embedder = UmapEmbedder(data_path=data_path, save_dir=save_dir)
    >>> embedder.fit(hyper_parameters=hyper_parameters)
    """

    def __init__(self):
        self.datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        super().__init__()

    def fit(
        self,
        data_path: Union[str, os.PathLike],
        save_dir: [str, os.PathLike],
        hyper_parameters: dict,
    ):
        check_file_exist_and_readable(file_path=data_path)
        check_that_directory_is_empty(directory=save_dir)
        self.data_path = data_path
        self.data = read_pickle(data_path=data_path)
        self.umap_df = (
            deepcopy(self.data[Unsupervised.BOUTS_FEATURES.value])
            .reset_index()
            .set_index(
                [
                    Unsupervised.VIDEO.value,
                    Unsupervised.START_FRAME.value,
                    Unsupervised.END_FRAME.value,
                ]
            )
        )
        self.save_dir = save_dir

        self.low_var_cols, self.hyper_parameters = None, hyper_parameters
        check_umap_hyperparameters(hyper_parameters=hyper_parameters)
        self.search_space = list(
            itertools.product(
                *[
                    hyper_parameters[Unsupervised.N_NEIGHBORS.value],
                    hyper_parameters[Unsupervised.MIN_DISTANCE.value],
                    hyper_parameters[Unsupervised.SPREAD.value],
                ]
            )
        )
        print(f"Building {len(self.search_space)} UMAP model(s)...")
        if hyper_parameters[Unsupervised.VARIANCE.value] > 0:
            self.low_var_cols = TrainModelMixin.find_low_variance_fields(
                data=self.umap_df,
                variance_threshold=hyper_parameters[Unsupervised.VARIANCE.value],
            )
            self.umap_df = drop_df_fields(data=self.umap_df, fields=self.low_var_cols)
        # if hyper_parameters[Unsupervised.MULTICOLLINEARITY.value] > 0:
        #     print(hyper_parameters[Unsupervised.MULTICOLLINEARITY.value])
        self.scaler = TrainModelMixin.define_scaler(
            scaler_name=hyper_parameters[Unsupervised.SCALER.value]
        )
        self.scaler.fit(self.umap_df)
        self.scaled_umap_data = TrainModelMixin.scaler_transform(
            data=self.umap_df, scaler=self.scaler
        )
        self.__create_methods_log()
        self.__fit_umaps()
        self.timer.stop_timer()
        stdout_success(
            msg=f"{len(self.search_space)} models saved in {self.save_dir} directory",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def __create_methods_log(self):
        self.methods = {}
        self.methods[Unsupervised.SCALER.value] = self.scaler
        self.methods[Unsupervised.SCALER_TYPE.value] = self.hyper_parameters[
            Unsupervised.SCALER.value
        ]
        self.methods[Unsupervised.SCALED_DATA.value] = self.scaled_umap_data
        self.methods[Unsupervised.VARIANCE.value] = self.hyper_parameters[
            Unsupervised.VARIANCE.value
        ]
        self.methods[Unsupervised.LOW_VARIANCE_FIELDS.value] = self.low_var_cols
        self.methods[Unsupervised.FEATURE_NAMES.value] = self.scaled_umap_data.columns

    def __fit_umaps(self):
        for cnt, h in enumerate(self.search_space):
            self.model_count = cnt
            self.model = {}
            self.model_timer = SimbaTimer()
            self.model_timer.start_timer()
            self.model[Unsupervised.HASHED_NAME.value] = random.sample(
                self.model_names, 1
            )[0]
            self.model[Unsupervised.PARAMETERS.value] = {
                Unsupervised.N_NEIGHBORS.value: h[0],
                Unsupervised.MIN_DISTANCE.value: h[1],
                Unsupervised.SPREAD.value: h[2],
            }
            self.model[Unsupervised.MODEL.value] = UMAP(
                min_dist=self.model[Unsupervised.PARAMETERS.value][
                    Unsupervised.MIN_DISTANCE.value
                ],
                n_neighbors=int(
                    self.model[Unsupervised.PARAMETERS.value][
                        Unsupervised.N_NEIGHBORS.value
                    ]
                ),
                spread=self.model[Unsupervised.PARAMETERS.value][
                    Unsupervised.SPREAD.value
                ],
                metric=Unsupervised.EUCLIDEAN.value,
                verbose=0,
            )
            self.model[Unsupervised.MODEL.value].fit(self.scaled_umap_data.values)
            results = {}
            results[Unsupervised.DATA.value] = self.data
            results[Unsupervised.METHODS.value] = self.methods
            results[Unsupervised.DR_MODEL.value] = self.model
            self.__save(data=results)

    def __save(self, data: dict) -> None:
        write_pickle(
            data=data,
            save_path=os.path.join(
                self.save_dir,
                f"{self.model[Unsupervised.HASHED_NAME.value]}.{Formats.PICKLE.value}",
            ),
        )
        self.model_timer.stop_timer()
        stdout_success(
            msg=f"Model {self.model_count+1}/{len(self.search_space)} ({self.model[Unsupervised.HASHED_NAME.value]}) saved...",
            elapsed_time=self.model_timer.elapsed_time,
        )

    def transform(
        self,
        data_path: Union[str, os.PathLike],
        model: Union[str, os.PathLike, dict],
        settings: Optional[Dict[str, Any]] = None,
        save_dir: Optional[os.PathLike] = None,
        save_format: Optional[Literal["csv", "pickle"]] = "csv",
    ) -> Union[None, pd.DataFrame]:
        """
        :Example I: Transform.
        >>> data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/logs/unsupervised_data_20240215143716.pickle'
        >>> model_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/clustering1704/academic_montalcini.pickle'
        >>> embedder = UmapEmbedder()
        >>> embedder.transform(save_dir=None, data_path=data_path, model=model_path, settings=None)
        >>> embedder.transform(save_dir=None, data_path=data_path, model=model_path, settings={'DATA_FORMAT': 'scaled', 'CLASSIFICATIONS': True})
        """
        print("Transforming data using UMAP model...")
        timer = SimbaTimer(start=True)
        check_file_exist_and_readable(file_path=data_path)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
            check_str(
                name=f"{UmapEmbedder.transform.__name__} save_format",
                value=save_format,
                options=(
                    "csv",
                    "pickle",
                ),
            )
        if isinstance(model, str):
            check_file_exist_and_readable(file_path=model)
            model = read_pickle(data_path=model)
        data = read_pickle(data_path=data_path)
        check_if_keys_exist_in_dict(
            data=data,
            key=[Unsupervised.BOUTS_FEATURES.value, Unsupervised.BOUTS_TARGETS.value],
            name=data_path,
        )
        check_if_keys_exist_in_dict(
            data=model, key=[Unsupervised.METHODS.value], name=data_path
        )
        mdl_name = model[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]
        umap_df = deepcopy(data[Unsupervised.BOUTS_FEATURES.value])
        umap_df = self.drop_fields(
            data=umap_df,
            fields=model[Unsupervised.METHODS.value][
                Unsupervised.LOW_VARIANCE_FIELDS.value
            ],
        )

        scaled_umap_data = TrainModelMixin.scaler_transform(
            data=umap_df,
            scaler=model[Unsupervised.METHODS.value][Unsupervised.SCALER.value],
        )
        scaled_umap_data = drop_df_fields(
            data=scaled_umap_data,
            fields=model[Unsupervised.METHODS.value][
                Unsupervised.LOW_VARIANCE_FIELDS.value
            ],
        )
        check_if_list_contains_values(
            data=list(scaled_umap_data.columns),
            values=model[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value],
            name=self.__class__.__name__,
        )
        results = pd.DataFrame(
            model[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].transform(
                scaled_umap_data
            ),
            columns=["X", "Y"],
            index=umap_df.index,
        )
        if settings is not None:
            check_instance(
                source=f"{UmapEmbedder.transform.__name__} settings",
                instance=settings,
                accepted_types=(dict,),
            )
            if "DATA_FORMAT" in settings.keys():
                if settings["DATA_FORMAT"] == Unsupervised.SCALED.value:
                    results = pd.concat([scaled_umap_data, results], axis=1)
                elif settings["DATA_FORMAT"] == Unsupervised.RAW.value:
                    results = pd.concat([umap_df, results], axis=1)
            if "CLASSIFICATIONS" in settings.keys():
                target_data = data[Unsupervised.BOUTS_TARGETS.value].set_index(
                    ["VIDEO", "START_FRAME", "END_FRAME"]
                )
                results = results.join(target_data, how="inner")
        if save_dir:
            if save_format == Formats.CSV.value:
                save_path = os.path.join(
                    save_dir,
                    f"Transformed_{mdl_name}_{self.datetime}.{Formats.CSV.value}",
                )
                results.to_csv(path_or_buf=save_path)
            else:
                save_path = os.path.join(
                    save_dir,
                    f"Transformed_{mdl_name}_{self.datetime}.{Formats.PICKLE.value}",
                )
                write_pickle(data=results, save_path=save_path)
            timer.stop_timer()
            stdout_success(
                msg=f"Transformed data saved at {save_path} (elapsed time: {timer.elapsed_time_str}s)"
            )
        else:
            timer.stop_timer()
            stdout_success(
                msg=f"Data transformed using UMAP model {mdl_name} (elapsed time: {timer.elapsed_time_str}s)"
            )
            return results


hyper_parameters = {
    "n_neighbors": [5],
    "min_distance": [0.1, 0.5, 0.0],
    "spread": [1.0],
    "scaler": "MIN-MAX",
    "variance": 0.25,
    "multicollinearity": 0.7,
}
data_path = "/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/logs/unsupervised_data_20240325092459.pickle"
save_dir = "/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/embedding_2"
config_path = (
    "/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini"
)
embedder = UmapEmbedder()
embedder.fit(data_path=data_path, save_dir=save_dir, hyper_parameters=hyper_parameters)
#

# data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/logs/unsupervised_data_20240215143716.pickle'
# model_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/clustering1704/academic_montalcini.pickle'
# embedder = UmapEmbedder()
# #embedder.transform(save_dir=None, data_path=data_path, model=model_path, settings=None)
#
# embedder.transform(save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/new_viz', data_path=data_path, model=model_path, settings={'DATA_FORMAT': 'scaled', 'CLASSIFICATIONS': True})


# data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/logs/unsupervised_data_20240215143716.pickle'
# model_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/clustering1704/academic_montalcini.pickle'
# embedder = UmapEmbedder()
# #embedder.transform(save_dir=None, data_path=data_path, model=model_path, settings=None)
#
# embedder.transform(save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/new_viz', data_path=data_path, model=model_path, settings={'DATA_FORMAT': 'scaled', 'CLASSIFICATIONS': True})

#

# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230416145821.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/transformed_umap'
# settings = {'DATA': 'RAW', 'format': 'csv'}
# embedder = UmapEmbedder(data_path=data_path, save_dir=save_dir)
# embedder.transform(model='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/boring_lederberg.pickle', settings=settings)


# model_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/funny_heisenberg.pickle'
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230222150701.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/'
# _ = UMAPTransform(model_path=model_path, data_path=data_path, save_dir=save_dir, settings=settings)

#
#
#
#
#
#
# hyper_parameters = {'n_neighbors': [10, 2], 'min_distance': [1.0], 'spread': [1.0], 'scaler': 'MIN-MAX', 'variance': 0.25}
# data_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/logs/unsupervised_data_20240214093117.pickle'
# save_dir = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/dim_reduction_mdls'
# config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini'
# embedder = UmapEmbedder()
# embedder.fit(data_path=data_path, save_dir=save_dir, hyper_parameters=hyper_parameters)
