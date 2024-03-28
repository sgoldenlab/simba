__author__ = "Simon Nilsson"

import json
import os
from typing import List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, UMLOptions, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_valid_extension)
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_pickle, write_pickle)

CLUSTERER_PARAMETERS = "CLUSTERER HYPER-PARAMETERS"
DIMENSIONALITY_REDUCTION_PARAMETERS = "DIMENSIONALITY REDUCTION HYPER-PARAMETERS"
SCALER = "SCALER"
SCALED_DATA = "SCALED DATA"
LOW_VARIANCE_FIELDS = "LOW VARIANCE FIELDS"
FEATURE_NAMES = "FEATURE NAMES"
FRAME_FEATURES = "FRAME FEATURES"
FRAME_POSE = "FRAME POSE"
FRAME_TARGETS = "FRAME TARGETS"
BOUTS_FEATURES = "BOUTS FEATURES"
BOUTS_TARGETS = "BOUTS TARGETS"
BOUTS_DIM_CORDS = "BOUTS DIMENSIONALITY REDUCTION DATA"
BOUTS_CLUSTER_LABELS = "BOUTS CLUSTER LABELS"


class DataExtractor(UnsupervisedMixin, ConfigReader):
    """
    Extracts human-readable data from directory of pickles or single pickle file that holds unsupervised analyses.

    :param config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param data_type: The type of data to extract. E.g., CLUSTERER_PARAMETERS, DIMENSIONALITY_REDUCTION_PARAMETERS, SCALER, SCALED_DATA, LOW_VARIANCE_FIELDS, FEATURE_NAMES, FRAME_FEATURES, FRAME_POSE, FRAME_TARGET, BOUTS_FEATURES, BOUTS_TARGETS, BOUTS_DIM_CORDS
    :param settings: User-defined parameters for data extraction.

    :example:
    >>> extractor = DataExtractor(data_path='unsupervised/cluster_models/awesome_curran.pickle', data_type=['BOUTS_TARGETS'], settings=None, config_path='unsupervised/project_folder/project_config.ini')
    >>> extractor.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_path: Union[str, os.PathLike],
        data_types: List[str],
        settings: Optional[dict] = None,
    ):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        if os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path)
            self.data_paths = find_files_of_filetypes_in_directory(
                directory=data_path,
                extensions=[f".{Formats.PICKLE.value}"],
                raise_error=True,
            )
        else:
            check_valid_extension(
                path=data_path, accepted_extensions=Formats.PICKLE.value
            )
            self.data_paths = [data_path]
        invalid_dtypes = list(set(data_types) - set(UMLOptions.DATA_TYPES.value))
        if len(data_types) == 0:
            raise InvalidInputError(
                msg=f"data_types is an empty list. Accepted options: {UMLOptions.DATA_TYPES.value}.",
                source=self.__class__.__name__,
            )
        if len(invalid_dtypes) > 0:
            raise InvalidInputError(
                msg=f"Found invalid data types: {invalid_dtypes}. Accepted: {UMLOptions.DATA_TYPES.value}.",
                source=self.__class__.__name__,
            )
        self.save_dir = os.path.join(
            self.logs_path, f"extracted_unsupervised_model_data_{self.datetime}"
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.settings, self.data_types = settings, data_types

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            print(f"Processing file {file_cnt+1}/{len(self.data_paths)}...")
            file_timer = SimbaTimer(start=True)
            v = read_pickle(data_path=file_path)
            mdl_name = get_fn_ext(filepath=file_path)[1]
            save_subdir = os.path.join(self.save_dir, get_fn_ext(filepath=file_path)[1])
            if not os.path.isdir(save_subdir):
                os.makedirs(save_subdir)
            for data_type in self.data_types:
                if data_type == CLUSTERER_PARAMETERS:
                    check_if_keys_exist_in_dict(
                        data=v, key=[Clustering.CLUSTER_MODEL.value], name=file_path
                    )
                    save_path = os.path.join(
                        save_subdir, f"cluster_parameters_{mdl_name}.json"
                    )
                    json.dump(
                        v[Clustering.CLUSTER_MODEL.value][
                            Unsupervised.PARAMETERS.value
                        ],
                        open(save_path, "w"),
                        indent=4,
                        sort_keys=True,
                    )
                    print(
                        f"Saved cluster parameters for model {mdl_name} at {save_path}..."
                    )

                elif data_type == DIMENSIONALITY_REDUCTION_PARAMETERS:
                    check_if_keys_exist_in_dict(
                        data=v, key=[Unsupervised.DR_MODEL.value], name=file_path
                    )
                    save_path = os.path.join(
                        save_subdir,
                        f"dimensionality_reduction_parameters_{mdl_name}.json",
                    )
                    json.dump(
                        v[Unsupervised.DR_MODEL.value][Unsupervised.PARAMETERS.value],
                        open(save_path, "w"),
                        indent=4,
                        sort_keys=True,
                    )
                    print(
                        f"Saved dimension reduction parameters for model {mdl_name} at {save_path} ..."
                    )
                #
                elif data_type == SCALER:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.METHODS.value],
                        name=file_path,
                    )
                    mdl_name = v[Unsupervised.DR_MODEL.value][
                        Unsupervised.HASHED_NAME.value
                    ]
                    save_path = os.path.join(save_subdir, f"scaler_{mdl_name}.pickle")
                    write_pickle(
                        data=v[Unsupervised.METHODS.value][Unsupervised.SCALER.value],
                        save_path=save_path,
                    )
                    print(f"Saved scaler for model {mdl_name} at {save_path}...")

                elif data_type == SCALED_DATA:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.METHODS.value],
                        name=file_path,
                    )
                    save_path = os.path.join(save_subdir, f"scaled_data_{mdl_name}.csv")
                    v[Unsupervised.METHODS.value][
                        Unsupervised.SCALED_DATA.value
                    ].to_csv(save_path)
                    print(f"Saved scaled data for {mdl_name} model at {save_path}...")

                elif data_type == LOW_VARIANCE_FIELDS:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.METHODS.value],
                        name=file_path,
                    )
                    save_path = os.path.join(
                        save_subdir, f"low_variance_fields_{mdl_name}.csv"
                    )
                    out_df = pd.DataFrame(
                        data=v[Unsupervised.METHODS.value][
                            Unsupervised.LOW_VARIANCE_FIELDS.value
                        ],
                        columns=["FIELD_NAMES"],
                    )
                    out_df.to_csv(save_path)
                    print(
                        f"Saved low variance fields for model {mdl_name} at {save_path}..."
                    )

                elif data_type == FEATURE_NAMES:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.METHODS.value],
                        name=file_path,
                    )
                    save_path = os.path.join(
                        save_subdir, f"feature_names_{mdl_name}.csv"
                    )
                    out_df = pd.DataFrame(
                        data=v[Unsupervised.METHODS.value][
                            Unsupervised.FEATURE_NAMES.value
                        ],
                        columns=["FIELD_NAMES"],
                    )
                    out_df.to_csv(save_path)
                    print(f"Feature names saved for model {mdl_name} at {save_path}...")

                elif data_type == FRAME_FEATURES:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.DATA.value],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"frame_wise_features_{mdl_name}.csv"
                    )
                    v[Unsupervised.DATA.value][
                        Unsupervised.FRAME_FEATURES.value
                    ].to_csv(save_path)
                    print(
                        f"Saved frame-wise features for model {mdl_name} at {save_path}..."
                    )

                elif data_type == FRAME_POSE:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.DATA.value],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"frame_wise_pose_estimation_{mdl_name}.csv"
                    )
                    v[Unsupervised.DATA.value][Unsupervised.FRAME_POSE.value].to_csv(
                        save_path
                    )
                    print(
                        f"Frame-wise pose saved for model {mdl_name} at {save_path}..."
                    )

                elif data_type == FRAME_TARGETS:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.DATA.value],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"frame_wise_target_data_{mdl_name}.csv"
                    )
                    v[Unsupervised.DATA.value][Unsupervised.FRAME_TARGETS.value].to_csv(
                        save_path
                    )
                    print(
                        f"Saved frame-wise target data for model {mdl_name} at {save_path}..."
                    )

                elif data_type == BOUTS_FEATURES:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"bout_features_data_{mdl_name}.csv"
                    )
                    v[Unsupervised.DATA.value][
                        Unsupervised.BOUTS_FEATURES.value
                    ].to_csv(save_path)
                    print(
                        f"Saved bout features data  for model {mdl_name} at {save_path}..."
                    )

                elif data_type == BOUTS_TARGETS:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value, Unsupervised.DATA.value],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"bout_targets_{mdl_name}.csv"
                    )
                    v[Unsupervised.DATA.value][Unsupervised.BOUTS_TARGETS.value].to_csv(
                        save_path
                    )
                    print(
                        f"Saved bout target data for model {mdl_name} at {save_path}..."
                    )

                elif data_type == BOUTS_DIM_CORDS:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[Unsupervised.DR_MODEL.value],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"bout_dim_cords_{mdl_name}.csv"
                    )
                    mdl_data = v[Unsupervised.DR_MODEL.value][
                        Unsupervised.MODEL.value
                    ].embedding_.astype(np.float32)
                    idx = v[Unsupervised.METHODS.value][
                        Unsupervised.SCALED_DATA.value
                    ].index
                    mdl_data = pd.DataFrame(
                        data=mdl_data, index=idx, columns=["X", "Y"]
                    )
                    mdl_data.to_csv(save_path)
                    print(
                        f"Saved bout dimensionality reduction data for model {mdl_name} at {save_path}..."
                    )

                elif data_type == BOUTS_CLUSTER_LABELS:
                    check_if_keys_exist_in_dict(
                        data=v,
                        key=[
                            Unsupervised.DR_MODEL.value,
                            Clustering.CLUSTER_MODEL.value,
                        ],
                        name=file_path,
                    )

                    save_path = os.path.join(
                        save_subdir, f"bout_cluster_labels_{mdl_name}.csv"
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
                    print(
                        f"Saved bout cluster labels for model {mdl_name} at {save_path}..."
                    )

                else:
                    raise InvalidInputError(
                        msg=f"Invalid datatype {data_type}.",
                        source=self.__class__.__name__,
                    )

            file_timer.stop_timer()
            stdout_success(
                msg=f"{mdl_name} model data extraction complete",
                elapsed_time=file_timer.elapsed_time,
            )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Data for {len(self.data_paths)} model(s) extracted",
            elapsed_time=self.timer.elapsed_time,
        )


# test = DataExtractor(data_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/cluster_mdls/hopeful_khorana.pickle',
#                      data_types=['CLUSTERER HYPER-PARAMETERS'],
#                      settings=None,
#                      config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini')
#
# test.run()
