__author__ = "Simon Nilsson"

import os
import json
import pandas as pd
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.mixins.config_reader import ConfigReader
from simba.unsupervised.enums import Unsupervised, Clustering
from simba.utils.checks import check_if_dir_exists, check_file_exist_and_readable
from simba.utils.printing import stdout_success


CLUSTERER_PARAMETERS = 'CLUSTERER HYPER-PARAMETERS'
DIMENSIONALITY_REDUCTION_PARAMETERS = 'DIMENSIONALITY REDUCTION HYPER-PARAMETERS'
SCALER = 'SCALER'
SCALED_DATA = 'SCALED DATA'
LOW_VARIANCE_FIELDS = 'LOW VARIANCE FIELDS'
FEATURE_NAMES = 'FEATURE_NAMES'
FRAME_FEATURES = 'FRAME_FEATURES'
FRAME_POSE = 'FRAME_POSE'
FRAME_TARGETS = 'FRAME_TARGETS'
BOUTS_FEATURES = 'BOUTS_FEATURES'
BOUTS_TARGETS = 'BOUTS_TARGETS'

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

    def __init__(self,
                 config_path: str,
                 data_path: str,
                 data_type: str,
                 settings: dict or None=None):



        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        if os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path)
            self.data = self.read_pickle(data_path=data_path)
        else:
            check_file_exist_and_readable(file_path=data_path)
            self.data = {0: self.read_pickle(data_path=data_path)}
        self.settings, self.data_type = settings, data_type

    def run(self):
        if self.data_type == CLUSTERER_PARAMETERS:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'cluster_parameters_{v[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                with open(save_path, 'w') as fp: json.dump(v[Clustering.CLUSTER_MODEL.value][Unsupervised.PARAMETERS.value], fp)
                stdout_success(msg=f'Cluster parameters saved at {save_path}')

        elif self.data_type == DIMENSIONALITY_REDUCTION_PARAMETERS:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'dimensionality_reduction_parameters_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                with open(save_path, 'w') as fp: json.dump(v[Unsupervised.DR_MODEL.value][Unsupervised.PARAMETERS.value], fp)
                stdout_success(msg=f'Dimension reduction parameters saved at {save_path}')


        elif self.data_type == SCALER:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'scaler_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.pickle')
                self.write_pickle(data=v[Unsupervised.METHODS.value][Unsupervised.SCALER.value], save_path=save_path)
                stdout_success(msg=f'Scaler saved at {save_path}')

        elif self.data_type == SCALED_DATA:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'scaled_data_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                v[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value].to_csv(save_path)
                stdout_success(msg=f'Scaled data saved at {save_path}')

        elif self.data_type == LOW_VARIANCE_FIELDS:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'low_variance_fields_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                out_df = pd.DataFrame(data=v[Unsupervised.METHODS.value][Unsupervised.LOW_VARIANCE_FIELDS.value], columns=['FIELD_NAMES'])
                out_df.to_csv(save_path)
                stdout_success(msg=f'Low variance fields saved at {save_path}')

        elif self.data_type == FEATURE_NAMES:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'feature_names_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                out_df = pd.DataFrame(data=v[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value], columns=['FIELD_NAMES'])
                out_df.to_csv(save_path)
                stdout_success(msg=f'Feature names fields saved at {save_path}')

        elif self.data_type == FRAME_FEATURES:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'frame_wise_features_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                v[Unsupervised.DATA.value][Unsupervised.FRAME_FEATURES.value].to_csv(save_path)
                stdout_success(msg=f'Frame-wise features saved at {save_path}')

        elif self.data_type == FRAME_POSE:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'frame_wise_pose_estimation_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                v[Unsupervised.DATA.value][Unsupervised.FRAME_POSE.value].to_csv(save_path)
                stdout_success(msg=f'Frame-wise pose saved at {save_path}')

        elif self.data_type == FRAME_TARGETS:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'frame_wise_target_data_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                v[Unsupervised.DATA.value][Unsupervised.FRAME_TARGETS.value].to_csv(save_path)
                stdout_success(msg=f'Frame-wise target data saved at {save_path}')

        elif self.data_type == BOUTS_FEATURES:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'bout_features_data_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                v[Unsupervised.DATA.value][Unsupervised.BOUTS_FEATURES.value].to_csv(save_path)
                stdout_success(msg=f'Bout features data saved at {save_path}')

        elif self.data_type == BOUTS_TARGETS:
            for k, v in self.data.items():
                save_path = os.path.join(self.logs_path, f'bout_targets_{v[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
                v[Unsupervised.DATA.value][Unsupervised.BOUTS_TARGETS.value].to_csv(save_path)
                stdout_success(msg=f'Bout target data saved at {save_path}')




# test = DataExtractor(data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/awesome_curran.pickle',
#                      data_type='BOUTS_TARGETS',
#                      settings=None,
#                      config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')
#
# test.run()
