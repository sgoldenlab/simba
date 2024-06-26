__author__ = "Simon Nilsson"

import os
from copy import deepcopy

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import \
    check_all_file_names_are_represented_in_video_log
from simba.utils.data import plug_holes_shortest_bout
from simba.utils.enums import TagNames
from simba.utils.errors import NoFilesFoundError, ParametersFileError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df
from simba.utils.warnings import SkippingFileWarning


class InferenceBatch(TrainModelMixin, ConfigReader):
    """
    Run classifier inference on all files with the ``project_folder/csv/features_extracted`` directory.
    Results are stored in the ``project_folder/csv/machine_results`` directory of the SimBA project.

    :param str config_path: path to SimBA project config file in Configparser format

    Example
    ----------
    >>> _ = InferenceBatch(config_path='MyConfigPath').run()
    """

    def __init__(self, config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        if len(self.feature_file_paths) == 0:
            raise NoFilesFoundError(
                "Zero files found in the project_folder/csv/features_extracted directory. Create features before running classifier.",
                source=self.__class__.__name__,
            )
        print(
            f"Analyzing {len(self.feature_file_paths)} file(s) with {self.clf_cnt} classifier(s)"
        )
        self.timer = SimbaTimer(start=True)
        self.model_dict = self.get_model_info(
            config=self.config, model_cnt=self.clf_cnt
        )

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.feature_file_paths)
        for file_cnt, file_path in enumerate(self.feature_file_paths):
            video_timer = SimbaTimer(start=True)
            _, file_name, _ = get_fn_ext(file_path)
            print(f"Analyzing video {file_name}...")
            file_save_path = os.path.join(self.machine_results_dir, f"{file_name}.{self.file_type}")
            in_df = read_df(file_path, self.file_type)
            x_df = self.drop_bp_cords(df=in_df).astype("float32")
            self.check_df_dataset_integrity(df=x_df, logs_path=self.logs_path, file_name=file_name)
            _, px_per_mm, fps = self.read_video_info(video_name=file_name, raise_error=False)
            out_df = deepcopy(in_df)
            for m, m_hyp in self.model_dict.items():
                if not os.path.isfile(m_hyp["model_path"]):
                    raise NoFilesFoundError(msg=f"{m_hyp['model_path']} is not a VALID model file path", source=self.__class__.__name__)
                probability_column = f"Probability_{m_hyp['model_name']}"
                clf = self.read_pickle(file_path=m_hyp["model_path"])
                out_df[probability_column] = self.clf_predict_proba(clf=clf, x_df=x_df, data_path=file_path, model_name=m_hyp["model_name"])
                out_df[m_hyp["model_name"]] = np.where(out_df[probability_column] > m_hyp["threshold"], 1, 0)
                if int(m_hyp["minimum_bout_length"]) > 0:
                    print(f'Correcting minimum bouts in video {file_name} and classifier {m_hyp["model_name"]} ({m_hyp["minimum_bout_length"]})...')
                    out_df = plug_holes_shortest_bout(data_df=out_df, clf_name=m_hyp["model_name"], fps=fps, shortest_bout=m_hyp["minimum_bout_length"])
            write_df(out_df, self.file_type, file_save_path)
            video_timer.stop_timer()
            print(f"Predictions created for {file_name} (elapsed time: {video_timer.elapsed_time_str}) ...")
        self.timer.stop_timer()
        stdout_success(msg="Machine predictions complete. Files saved in project_folder/csv/machine_results directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# test = InferenceBatch(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/levi/project_folder/project_config.ini')
# test.run()

# test = InferenceBatch(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()

# test = RunModel(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run_models()
