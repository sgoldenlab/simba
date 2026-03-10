__author__ = "Simon Nilsson; sronilsson@gmail.com"

import argparse
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_if_keys_exist_in_dict, check_int, check_that_column_exist,
    check_valid_dict, check_valid_lst)
from simba.utils.data import plug_holes_shortest_bout
from simba.utils.enums import TagNames
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import (SimbaTimer, log_event, stdout_information,
                                  stdout_success)
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)
from simba.utils.warnings import NoFileFoundWarning

MINIMUM_BOUT_LENGTH = 'minimum_bout_length'
THRESHOLD = 'threshold'
MODEL_NAME = 'model_name'
MODEL_PATH = 'model_path'

class InferenceBatch(TrainModelMixin, ConfigReader):
    """
    Run classifier inference on all files with the ``project_folder/csv/features_extracted`` directory.
    Results are stored in the ``project_folder/csv/machine_results`` directory of the SimBA project.

    .. note::
       To compute aggregate statistics from the output of this class, see :func:`simba.data_processors.agg_clf_calculator.AggregateClfCalculator`

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format.
    :param Optional[Union[str, os.PathLike]] features_dir: Optional directory containing featurized files in CSV or parquet format. If None, then the `project_folder/csv/features_extracted` directory of the project will be used.
    :param Optional[Union[str, os.PathLike]] save_dir: Optional directory to save the data for the analyzed videos. If None, then the `project_folder/csv/machine_results` directory of the project will be used.
    :param Optional[int] minimum_bout_length: Optional minimum bout length (milliseconds) override. If None, classifier-specific minimum bout settings from project configuration are used.
    :param Optional[Dict[str, Dict[str, List[str]]]] feature_subsets_by_clf: Optional per-classifier feature subsets to use during inference. Format: ``{classifier_name: {subset_name: [feature_col_1, feature_col_2, ...]}}``. If provided, each classifier is applied once per subset and outputs are suffixed with the subset name.
    :param bool verbose: If True, print progress and status messages during inference. Default: True.

    :example I:
    >>> inferencer = InferenceBatch(config_path='MyConfigPath')
    >>> inferencer.run()

    :example II:
    >>> inferencer = InferenceBatch(config_path=r"D:/troubleshooting/mitra/project_folder/project_config.ini", features_dir=r"D:/troubleshooting/mitra/project_folder/videos/bg_removed/rotated/tail_features/APPENDED")
    >>> inferencer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 features_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 minimum_bout_length: Optional[int] = None,
                 feature_subsets_by_clf: Optional[Dict[str, Dict[str, List[str]]]] = None,
                 verbose: bool = True):

        ConfigReader.__init__(self, config_path=config_path)
        if features_dir is not None:
            check_if_dir_exists(in_dir=features_dir, source=self.__class__.__name__)
            self.features_dir = deepcopy(features_dir)
            self.feature_file_paths = find_files_of_filetypes_in_directory(directory=self.features_dir, extensions=[f'.{self.file_type}'], raise_warning=False, raise_error=False)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
            self.save_dir = deepcopy(save_dir)
        else:
            self.save_dir = self.machine_results_dir
        TrainModelMixin.__init__(self)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if len(self.feature_file_paths) == 0:
            raise NoFilesFoundError(msg=f"Zero files found in the {self.features_dir}. Create features before running classifier.", source=self.__class__.__name__,)
        if feature_subsets_by_clf is not None:
            check_valid_dict(x=feature_subsets_by_clf, valid_key_dtypes=(str,), source=f'{self.__class__.__name__} feature_subsets_by_clf')
            for cnt, (k, v) in enumerate(feature_subsets_by_clf.items()):
                if k not in self.clf_names:
                    raise InvalidInputError(msg=f'Unknown classifier "{k}" in feature_subsets_by_clf. Valid classifiers: {self.clf_names}', source=self.__class__.__name__)
                check_valid_dict(x=v, valid_key_dtypes=(str,), valid_values_dtypes=(list,), source=f'{self.__class__.__name__} feature_subsets_by_clf {cnt}')
                for subset_name, feature_names in v.items():
                    check_valid_lst(data=feature_names, source=f'{self.__class__.__name__} feature_subsets_by_clf {k} {subset_name}', valid_dtypes=(str,), min_len=1, raise_error=True)
        if minimum_bout_length is not None: check_int(name=f'{self.__class__.__name__} minimum_bout_length', value=minimum_bout_length, allow_zero=False, allow_negative=False, raise_error=True)
        self.verbose, self.feature_subsets_by_clf, self.minimum_bout_length = verbose, feature_subsets_by_clf, minimum_bout_length
        if verbose: stdout_information(msg=f"Analyzing {len(self.feature_file_paths)} file(s) with {self.clf_cnt} classifier(s)...")
        self.timer = SimbaTimer(start=True)
        self.model_dict = self.get_model_info(config=self.config, model_cnt=self.clf_cnt)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.feature_file_paths)
        for file_cnt, file_path in enumerate(self.feature_file_paths):
            video_timer = SimbaTimer(start=True)
            _, file_name, _ = get_fn_ext(file_path)
            if self.verbose: stdout_information(msg=f"Analyzing video {file_name}... (Video {file_cnt+1}/{len(self.feature_file_paths)})")
            file_save_path = os.path.join(self.save_dir, f"{file_name}.{self.file_type}")
            in_df = read_df(file_path, self.file_type)
            x_df = self.drop_bp_cords(df=in_df).astype(np.float32)
            self.check_df_dataset_integrity(df=x_df, logs_path=self.logs_path, file_name=file_name)
            _, px_per_mm, fps = self.read_video_info(video_name=file_name, raise_error=False)
            out_df = deepcopy(in_df)
            for m, m_hyp in self.model_dict.items():
                check_if_keys_exist_in_dict(data=m_hyp, key=[MODEL_PATH, MODEL_NAME, THRESHOLD, MINIMUM_BOUT_LENGTH], name=f'classifier {m}', raise_error=False)
                if not os.path.isfile(m_hyp[MODEL_PATH]):
                    NoFileFoundWarning(msg=f'SKIPPING CLASSIFIER {m} for video {file_name}. The classifier model file {m_hyp[MODEL_PATH]} could not be found.', source=self.__class__.__name__)
                    continue
                clf = self.read_pickle(file_path=m_hyp[MODEL_PATH])
                if self.feature_subsets_by_clf is None or m_hyp[MODEL_NAME] not in self.feature_subsets_by_clf:
                    probability_column = f"Probability_{m_hyp[MODEL_NAME]}"
                    out_df[probability_column] = self.clf_predict_proba(clf=clf, x_df=x_df, data_path=file_path, model_name=m_hyp[MODEL_NAME])
                    out_df[m_hyp[MODEL_NAME]] = np.where(out_df[probability_column] > m_hyp[THRESHOLD], 1, 0)
                    clf_min_bout = self.minimum_bout_length if self.minimum_bout_length is not None else m_hyp[MINIMUM_BOUT_LENGTH]
                    if int(clf_min_bout) > 0:
                        if self.verbose: stdout_information(msg=f'Correcting minimum bouts in video {file_name} and classifier {m_hyp[MODEL_NAME]} ({clf_min_bout}ms)...')
                        out_df = plug_holes_shortest_bout(data_df=out_df, clf_name=m_hyp[MODEL_NAME], fps=fps, shortest_bout=clf_min_bout)
                else:
                    for model_subset_name, model_x in self.feature_subsets_by_clf[m_hyp[MODEL_NAME]].items():
                        probability_column = f"Probability_{m_hyp[MODEL_NAME]}_{model_subset_name}"
                        check_that_column_exist(df=x_df, column_name=model_x, file_name=file_path, raise_error=True)
                        out_df[probability_column] = self.clf_predict_proba(clf=clf, x_df=x_df[model_x], data_path=file_path, model_name=model_subset_name)
                        out_df[f'{m_hyp[MODEL_NAME]}_{model_subset_name}'] = np.where(out_df[probability_column] > m_hyp[THRESHOLD], 1, 0)
                        clf_min_bout = self.minimum_bout_length if self.minimum_bout_length is not None else m_hyp[MINIMUM_BOUT_LENGTH]
                        if int(clf_min_bout) > 0:
                            if self.verbose: stdout_information(msg=f'Correcting minimum bouts in video {file_name} and classifier {m_hyp[MODEL_NAME]} ({clf_min_bout}ms)...')
                            out_df = plug_holes_shortest_bout(data_df=out_df, clf_name=f'{m_hyp[MODEL_NAME]}_{model_subset_name}', fps=fps, shortest_bout=clf_min_bout)
            write_df(df=out_df, file_type=self.file_type, save_path=file_save_path)
            video_timer.stop_timer()
            if self.verbose: stdout_information(msg=f"Predictions created for {file_name} (frame count: {len(in_df)}, elapsed time: {video_timer.elapsed_time_str}) ...")
        self.timer.stop_timer()
        stdout_success(msg=f"Machine predictions complete for {len(self.feature_file_paths)} file(s). Files saved in {self.save_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Perform classifications according to rules defined in SimBA project_config.ini.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA Project config.')
    args = parser.parse_args()
    runner = InferenceBatch(config_path=args.config_path)
    runner.run()



#
# test = InferenceBatch(config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
# test.run()

# test = InferenceBatch(config_path=r"C:\troubleshooting\Top_down_old\project_folder\project_config.ini")
# test.run()

#
# test = InferenceBatch(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                       features_dir=r'C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated\tail_features_additional\APPENDED')
# test.run()



# test = InferenceBatch(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()





# test = InferenceBatch(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
#                       features_dir=r'D:\troubleshooting\mitra\project_folder\videos\bg_removed\rotated\laying_down_features\APPENDED')
# test.run()
# if __name__ == "__main__":
#     test = InferenceBatch(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
#                           features_dir=r'D:\troubleshooting\mitra\project_folder\videos\bg_removed\rotated\tail_features\APPENDED')
#     test.run()


# test = InferenceBatch(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/levi/project_folder/project_config.ini')
# test.run()

# test = InferenceBatch(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()

# test = RunModel(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run_models()
