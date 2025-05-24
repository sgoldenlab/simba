__author__ = "Simon Nilsson"

import glob
import os
from typing import List, Optional, Union

import pandas as pd

from simba.cue_light_tools.cue_light_tools import find_frames_when_cue_light_on
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_int, check_that_column_exist, check_valid_dataframe, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats
from simba.utils.errors import NoDataError, NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_config_entry, read_df)


class CueLightClfAnalyzer(ConfigReader):
    """
    Compute aggregate statistics when classified behaviors are occurring in relation to the cue light
    ON and OFF states.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param int pre_window: Time period (in millisecond) before the onset of each cue light to compute aggregate classification statistics within.
    :param int post_window: Time period (in millisecond) after the offset of each cue light to compute aggregate classification statistics within.
    :param List[str] cue_light_names: Names of cue lights, as defined in the SimBA ROI interface.
    :param List[str] list: Names of the classifiers we want to compute aggregate statistics for.

    .. note::
       `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.

    :example:
    >>> cue_light_clf_analyzer = CueLightClfAnalyzer(config_path='MyProjectConfig', pre_window=1000, post_window=1000, cue_light_names=['Cue_light'], clf_list=['Attack'])
    >>> cue_light_clf_analyzer.analyze_clf()
    >>> cue_light_clf_analyzer.organize_results()
    >>> cue_light_clf_analyzer.save_data()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 cue_light_names: List[str],
                 clf_names: List[str],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 pre_window: int = 0,
                 post_window: int = 0):

        ConfigReader.__init__(self, config_path=config_path)
        check_valid_lst(data=cue_light_names, source=f'{self.__class__.__name__} cue_light_names', valid_dtypes=(str,), min_len=1, raise_error=True)
        check_valid_lst(data=clf_names, source=f'{self.__class__.__name__} clf_names', valid_dtypes=(str,), min_len=1, raise_error=True, valid_values=self.clf_names)
        check_int(name=f'{self.__class__.__name__} pre_window', value=pre_window, min_value=0)
        check_int(name=f'{self.__class__.__name__} post_window', value=post_window, min_value=0)
        if data_dir is None:
            self.data_dir = self.cue_lights_data_dir
        else:
            check_if_dir_exists(in_dir=data_dir)
            self.data_dir = data_dir
        self.cue_light_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True, as_dict=True)
        self.machine_results_paths = find_files_of_filetypes_in_directory(directory=self.machine_results_dir,  extensions=[f'.{self.file_type}'], raise_error=True, as_dict=True)
        missing_ml = [x for x in self.cue_light_paths.keys() if x not in self.machine_results_paths.keys()]
        if len(missing_ml) > 0:
            raise NoDataError(msg=f'{len(missing_ml)} cue-light file(s) are missing classification files in the {self.machine_results_dir} directory: {missing_ml}', source=self.__class__.__name__)
        self.cue_light_names, self.pre_window, self.post_window, self.clf_names = cue_light_names, pre_window, post_window, clf_names
        self.save_path = os.path.join(self.logs_path, f"Cue_lights_clf_statistics_{self.datetime}.csv")


    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=list(self.cue_light_paths.values()))


        self.results = pd.DataFrame(columns=['VIDEO', 'CUE LIGHT', 'CLASSIFIER', 'CUE LIGHT BOUT START TIME', 'CUE LIGHT BOUT END TIME', 'CUE LIGHT BOUT START FRAME', 'CUE LIGHT BOUT END FRAME', ' CUE LIGHT BOUT BEHAVIOR PRESENT (S)', 'CUE LIGHT BOUT BEHAVIOR ABSENT (S)', f'PRE CUE LIGHT BOUT ({self.pre_window}s) PRESENT (S)', f'PRE CUE LIGHT BOUT ({self.pre_window}s) ABSENT (S)', f'POST CUE LIGHT BOUT ({self.pre_window}s) PRESENT (S)', f'POST CUE LIGHT BOUT ({self.pre_window}s) ABSENT (S)'])
        for file_cnt, (video_name, cue_light_data_path) in enumerate(self.cue_light_paths.items()):
            machine_results_path = self.machine_results_paths[video_name]
            ml_df = read_df(machine_results_path, self.file_type)
            cue_light_df = read_df(cue_light_data_path, self.file_type)
            check_valid_dataframe(df=ml_df, source=machine_results_path, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.clf_names)
            check_valid_dataframe(df=cue_light_df, source=cue_light_data_path, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.cue_light_names)
            data_df = pd.concat([ml_df, cue_light_df[self.cue_light_names]], axis=1)
            del cue_light_df, ml_df
            _, _, fps = self.read_video_info(video_name=video_name)
            self.prior_window_frames_cnt = int(self.pre_window * fps)
            self.post_window_frames_cnt = int(self.post_window * fps)
            cue_light_bouts = detect_bouts(data_df=data_df, target_lst=self.cue_light_names, fps=fps).reset_index(drop=True)
            for bout_cnt, bout in cue_light_bouts.iterrows():
                cue_frm_range = list(range(bout['Start_frame'], bout['End_frame']+1))
                pre_window_frms = list(range(max(0, bout['Start_frame']-self.prior_window_frames_cnt), bout['Start_frame']))
                post_window_frms = list(range(bout['End_frame']+1, min((bout['End_frame'] + self.post_window_frames_cnt), len(data_df))))
                cue_frm_range_df = data_df.loc[cue_frm_range][self.clf_names]
                pre_window_frms_df = data_df.loc[pre_window_frms][self.clf_names]
                post_window_frms_df = data_df.loc[post_window_frms][self.clf_names]
                for clf in self.clf_names:
                    cue_clf_present = round(cue_frm_range_df[clf].sum() / fps, 4)
                    cue_clf_absent = round(bout['Bout_time'] - cue_clf_present, 4)
                    pre_clf_present = round(pre_window_frms_df[clf].sum() / fps, 4)
                    pre_clf_absent = round(self.pre_window - pre_clf_present, 4)
                    post_clf_present = round(post_window_frms_df[clf].sum() / fps, 4)
                    post_clf_absent = round(self.post_window - post_clf_present, 4)
                    self.results.loc[len(self.results)] = [video_name, bout['Event'], clf, bout['Start_time'], bout['End Time'], bout['Start_frame'], bout['End_frame'], cue_clf_present, cue_clf_absent, pre_clf_present, pre_clf_absent, post_clf_present, post_clf_absent]

    def save(self):
        self.results = self.results.sort_values(by=['VIDEO', 'CUE LIGHT', 'CUE LIGHT BOUT START TIME'], ascending=True)
        if self.post_window == 0:
            self.results = self.results.drop([f'POST CUE LIGHT BOUT ({self.pre_window}s) PRESENT (S)', f'POST CUE LIGHT BOUT ({self.pre_window}s) ABSENT (S)'], axis=1)
        if self.pre_window == 0:
            self.results = self.results.drop([f'PRE CUE LIGHT BOUT ({self.pre_window}s) PRESENT (S)', f'PRE CUE LIGHT BOUT ({self.pre_window}s) ABSENT (S)'], axis=1)
        self.results.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'Cue light classifier statistics saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)


# test = CueLightClfAnalyzer(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini",
#                            pre_window=1,
#                            post_window=1,
#                            cue_light_names=['cl'],
#                            clf_names=['freeze'])
# test.run()
# test.save()
# test.organize_results()
# test.save_data()
