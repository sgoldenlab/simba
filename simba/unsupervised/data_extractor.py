__author__ = "Simon Nilsson"

import json
import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict)
from simba.utils.printing import stdout_success
from simba.utils.read_write import read_pickle, write_pickle

CLUSTERER_PARAMETERS = "CLUSTERER HYPER-PARAMETERS"
DIMENSIONALITY_REDUCTION_PARAMETERS = "DIMENSIONALITY REDUCTION HYPER-PARAMETERS"
SCALER = "SCALER"
SCALED_DATA = "SCALED DATA"
LOW_VARIANCE_FIELDS = "LOW VARIANCE FIELDS"
FEATURE_NAMES = "FEATURE_NAMES"
FRAME_FEATURES = "FRAME_FEATURES"
FRAME_POSE = "FRAME_POSE"
FRAME_TARGETS = "FRAME_TARGETS"
BOUTS_FEATURES = "BOUTS_FEATURES"
BOUTS_TARGETS = "BOUTS_TARGETS"
BOUTS_DIM_CORDS = "BOUTS DIMENSIONALITY REDUCTION DATA"
BOUTS_CLUSTER_LABELS = "BOUTS CLUSTER LABELS"


class DataExtractor(UnsupervisedMixin, ConfigReader):
    """
    Extracts human-readable data from pickle holding unsupervised analyses.

    :param config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param data_type: The type of data to extract.
    :param settings: User-defined parameters for data extraction.

    :example:
    >>> extractor = DataExtractor(data_path='unsupervised/cluster_models/awesome_curran.pickle', data_type='BOUTS_TARGETS', settings=None, config_path='unsupervised/project_folder/project_config.ini')
    >>> extractor.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_path: Union[str, os.PathLike],
        data_type: str,
        settings: Optional[dict] = None,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        if os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path)
            self.data = read_pickle(data_path=data_path)
        else:
            check_file_exist_and_readable(file_path=data_path)
            self.data = {0: read_pickle(data_path=data_path)}
        self.settings, self.data_type = settings, data_type

    def run(self):
        if self.data_type == CLUSTERER_PARAMETERS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Clustering.CLUSTER_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Clustering.CLUSTER_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"cluster_parameters_{mdl_name}.json"
                )
                with open(save_path, "w") as fp:
                    json.dump(
                        v[Clustering.CLUSTER_MODEL.value][
                            Unsupervised.PARAMETERS.value
                        ],
                        fp,
                        indent=4,
                        sort_keys=True,
                    )
                print(f"Cluster parameters saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Cluster parameters for {len(list(self.data.keys()))} model(s) saved at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == DIMENSIONALITY_REDUCTION_PARAMETERS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path,
                    f"dimensionality_reduction_parameters_{mdl_name}.json",
                )
                with open(save_path, "w") as fp:
                    json.dump(
                        v[Unsupervised.DR_MODEL.value][Unsupervised.PARAMETERS.value],
                        fp,
                    )
                print(
                    f"Dimension reduction parameters for model {mdl_name} saved at ..."
                )
            self.timer.stop_timer()
            stdout_success(
                msg=f"Dimension reduction parameters for {len(list(self.data.keys()))} model(s) saved at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == SCALER:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(self.logs_path, f"scaler_{mdl_name}.pickle")
                write_pickle(
                    data=v[Unsupervised.METHODS.value][Unsupervised.SCALER.value],
                    save_path=save_path,
                )
                print(f"Scaler saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Scaler(s) saved for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == SCALED_DATA:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(self.logs_path, f"scaled_data_{mdl_name}.csv")
                v[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value].to_csv(
                    save_path
                )
                print(f"Scaled data saved for {mdl_name} model ...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Scaled data saved for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == LOW_VARIANCE_FIELDS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"low_variance_fields_{mdl_name}.csv"
                )
                out_df = pd.DataFrame(
                    data=v[Unsupervised.METHODS.value][
                        Unsupervised.LOW_VARIANCE_FIELDS.value
                    ],
                    columns=["FIELD_NAMES"],
                )
                out_df.to_csv(save_path)
                print(f"Low variance fields saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Low variance fields for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == FEATURE_NAMES:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"feature_names_{mdl_name}.csv"
                )
                out_df = pd.DataFrame(
                    data=v[Unsupervised.METHODS.value][
                        Unsupervised.FEATURE_NAMES.value
                    ],
                    columns=["FIELD_NAMES"],
                )
                out_df.to_csv(save_path)
                print(f"Feature names saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Feature names for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == FRAME_FEATURES:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"frame_wise_features_{mdl_name}.csv"
                )
                v[Unsupervised.DATA.value][Unsupervised.FRAME_FEATURES.value].to_csv(
                    save_path
                )
                print(f"Frame-wise features saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Frame-wise features for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == FRAME_POSE:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"frame_wise_pose_estimation_{mdl_name}.csv"
                )
                v[Unsupervised.DATA.value][Unsupervised.FRAME_POSE.value].to_csv(
                    save_path
                )
                print(f"Frame-wise pose saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Frame-wise pose for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == FRAME_TARGETS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"frame_wise_target_data_{mdl_name}.csv"
                )
                v[Unsupervised.DATA.value][Unsupervised.FRAME_TARGETS.value].to_csv(
                    save_path
                )
                print(f"Frame-wise target data saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Frame-wise target data for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == BOUTS_FEATURES:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"bout_features_data_{mdl_name}.csv"
                )
                v[Unsupervised.DATA.value][Unsupervised.BOUTS_FEATURES.value].to_csv(
                    save_path
                )
                print(f"Bout features data saved for model {mdl_name} ...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Bout features data for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == BOUTS_TARGETS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(self.logs_path, f"bout_targets_{mdl_name}.csv")
                v[Unsupervised.DATA.value][Unsupervised.BOUTS_TARGETS.value].to_csv(
                    save_path
                )
                print(f"Bout target data saved for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Bout target data for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == BOUTS_DIM_CORDS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"bout_dim_cords_{mdl_name}.csv"
                )
                mdl_data = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.MODEL.value
                ].embedding_.astype(np.float32)
                idx = v[Unsupervised.METHODS.value][
                    Unsupervised.SCALED_DATA.value
                ].index
                mdl_data = pd.DataFrame(data=mdl_data, index=idx, columns=["X", "Y"])
                mdl_data.to_csv(save_path)
                print(
                    f"Saved bout dimensionality reduction data for model {mdl_name}..."
                )
            self.timer.stop_timer()
            stdout_success(
                msg=f"Bout dimensionality reduction data for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )

        elif self.data_type == BOUTS_CLUSTER_LABELS:
            for k, v in self.data.items():
                check_if_keys_exist_in_dict(
                    data=v,
                    key=[Unsupervised.DR_MODEL.value],
                    name=DataExtractor.__name__,
                )
                mdl_name = v[Unsupervised.DR_MODEL.value][
                    Unsupervised.HASHED_NAME.value
                ]
                save_path = os.path.join(
                    self.logs_path, f"bout_cluster_labels_{mdl_name}.csv"
                )
                mdl_data = (
                    v[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value]
                    .labels_.astype(np.int64)
                    .reshape(-1, 1)
                )
                idx = v[Unsupervised.METHODS.value][
                    Unsupervised.SCALED_DATA.value
                ].index
                mdl_data = pd.DataFrame(data=mdl_data, index=idx, columns=["LABEL"])
                mdl_data.to_csv(save_path)
                print(f"Saved bout cluster labels for model {mdl_name}...")
            self.timer.stop_timer()
            stdout_success(
                msg=f"Bout cluster label data for {len(list(self.data.keys()))} model(s) at {self.logs_path}!",
                elapsed_time=self.timer.elapsed_time_str,
            )


# test = DataExtractor(data_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_mdls/hopeful_khorana.pickle',
#                      data_type=BOUTS_CLUSTER_LABELS,
#                      settings=None,
#                      config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
#
# test.run()
