__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_int, check_str, check_valid_dataframe, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df)


class CueLightMovementAnalyzer(ConfigReader):
    """
    Compute aggregate statistics of animal movement in relation to the cue light
    ON and OFF states.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter int pre_window: Time period (in millisecond) before the onset of each cue light to compute aggregate classification statistics within.
    :parameter int post_window: Time period (in millisecond) after the offset of each cue light to compute aggregate classification statistics within.
    :parameter List[str] cue_light_names: Names of cue lights, as defined in the SimBA ROI interface.
    :parameter float threshold: The body-part post-estimation probability threshold. SimBA omits movement calculations for frames where the body-part probability threshold is lower than the user-specified threshold.
    :parameter bool roi_setting: If True, SimBA calculates movement statistics within non-light cue ROIs.

    .. note:
       `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.


    :examples:
    >>> cue_light_movement_analyzer = CueLightMovementAnalyzer(config_path='MyProjectConfig', cue_light_names=['Cue_light'], pre_window=1000, post_window=1000, threshold=0.0, roi_setting=True)
    >>> cue_light_movement_analyzer.calculate_whole_session_movement()
    >>> cue_light_movement_analyzer.organize_results()
    >>> cue_light_movement_analyzer.save_results()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 cue_light_names: List[str],
                 bp_name: str,
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 pre_window: int = 0,
                 post_window: int = 0):

        ConfigReader.__init__(self, config_path=config_path)
        check_valid_lst(data=cue_light_names, source=f'{self.__class__.__name__} cue_light_names', valid_dtypes=(str,), min_len=1, raise_error=True)
        check_int(name=f'{self.__class__.__name__} pre_window', value=pre_window, min_value=0)
        check_int(name=f'{self.__class__.__name__} post_window', value=post_window, min_value=0)
        if data_dir is None:
            self.data_dir = self.cue_lights_data_dir
        else:
            check_if_dir_exists(in_dir=data_dir)
            self.data_dir = data_dir
        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True, as_dict=True)
        self.data_file_count = len(list(self.data_paths.keys()))
        check_str(name=f'{self.__class__.__name__} bp_name', value=bp_name, options=self.body_parts_lst)
        self.bp_cols = [f'{bp_name}_x', f'{bp_name}_y']
        self.cue_light_names, self.pre_window, self.post_window, self.bp_name = cue_light_names, pre_window, post_window, bp_name
        self.save_path = os.path.join(self.logs_path, f"Cue_lights_movement_statistics_{self.datetime}.csv")

    def run(self):
        print(f"Analyzing cue-light movement data for {self.data_file_count} file(s)...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=list(self.data_paths.values()))
        self.results = pd.DataFrame(columns=['VIDEO', 'BODY PART', 'CUE LIGHT', 'CUE LIGHT BOUT START TIME', 'CUE LIGHT BOUT END TIME', 'CUE LIGHT BOUT START FRAME', 'CUE LIGHT BOUT END FRAME', 'CUE LIGHT BOUT MOVEMENT (CM)', 'CUE LIGHT BOUT VELOCITY (CM/S)', f'PRE CUE LIGHT BOUT ({self.pre_window}s) MOVEMENT (CM)', f'PRE CUE LIGHT BOUT ({self.pre_window}s) VELOCITY (CM/S)', f'POST CUE LIGHT BOUT ({self.post_window}s) MOVEMENT (CM)', f'POST CUE LIGHT BOUT ({self.post_window}s) VELOCITY (CM/S)'])
        for file_cnt, (video_name, data_path) in enumerate(self.data_paths.items()):
            video_timer = SimbaTimer(start=True)
            self.data_df = read_df(data_path, self.file_type).reset_index(drop=True)
            check_valid_dataframe(df=self.data_df, source=data_path, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.bp_cols + self.cue_light_names)
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)
            self.prior_window_frames_cnt = int(self.pre_window * fps)
            self.post_window_frames_cnt = int(self.post_window * fps)
            cue_light_bouts = detect_bouts(data_df=self.data_df, target_lst=self.cue_light_names, fps=fps).reset_index(drop=True)
            for bout_cnt, bout in cue_light_bouts.iterrows():
                cue_frm_range = list(range(bout['Start_frame'], bout['End_frame']+1))
                pre_window_frms = list(range(max(0, bout['Start_frame']-self.prior_window_frames_cnt), bout['Start_frame']))
                post_window_frms = list(range(bout['End_frame']+1, min((bout['End_frame'] + self.post_window_frames_cnt), len(self.data_df))))
                cue_frm_range_arr = self.data_df.loc[cue_frm_range][self.bp_cols].values
                pre_window_frms_arr = self.data_df.loc[pre_window_frms][self.bp_cols].values
                post_window_frms_arr = self.data_df.loc[post_window_frms][self.bp_cols].values
                cue_bout_movement, cue_bout_velocity = FeatureExtractionSupplemental().distance_and_velocity(x=cue_frm_range_arr, fps=fps, pixels_per_mm=px_per_mm, centimeters=True)
                pre_window_movement, pre_window_velocity = FeatureExtractionSupplemental().distance_and_velocity(x=pre_window_frms_arr, fps=fps, pixels_per_mm=px_per_mm, centimeters=True)
                post_window_movement, post_window_velocity = FeatureExtractionSupplemental().distance_and_velocity(x=post_window_frms_arr, fps=fps, pixels_per_mm=px_per_mm, centimeters=True)
                self.results.loc[len(self.results)] = [video_name, self.bp_name, bout['Event'], bout['Start_time'], bout['End Time'], bout['Start_frame'], bout['End_frame'], cue_bout_movement, cue_bout_velocity, pre_window_movement, pre_window_velocity, post_window_movement, post_window_velocity]
            video_timer.stop_timer()
            print(f'Cue light movement statistics for video {video_name} complete... ({file_cnt+1}/{self.data_file_count}, elapsed time: {video_timer.elapsed_time_str}s)')
    def save(self):
        self.results = self.results.sort_values(by=['VIDEO', 'CUE LIGHT', 'CUE LIGHT BOUT START TIME'], ascending=True)
        if self.post_window == 0:
            self.results = self.results.drop([f'POST CUE LIGHT BOUT ({self.post_window}s) MOVEMENT (CM)', f'POST CUE LIGHT BOUT ({self.post_window}s) VELOCITY (CM/S)'], axis=1)
        if self.pre_window == 0:
            self.results = self.results.drop([f'PRE CUE LIGHT BOUT ({self.pre_window}s) MOVEMENT (CM)', f'PRE CUE LIGHT BOUT ({self.pre_window}s) VELOCITY (CM/S)'], axis=1)
        self.results.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'Cue light movement data saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)




# test = CueLightMovementAnalyzer(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini",
#                                 pre_window=0,
#                                 post_window=0,
#                                 cue_light_names=['cl'],
#                                 data_dir=r"C:\troubleshooting\cue_light\t1\project_folder\csv\cue_lights",
#                                 bp_name='Nose')
#
# test.run()
# test.save()

# test = CueLightMovementAnalyzer(config_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                                 pre_window=100, post_window=100, cue_light_names=['Rectangl_1'], threshold=0.0, roi_setting=True)
#
#
# def __init__(self,
#              config_path: str,
#              pre_window: int,
#              post_window: int,
#              cue_light_names: list,
#              threshold: float,
#              roi_setting: bool):
