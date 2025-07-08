__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.roi_tools.roi_utils import get_roi_dict_from_dfs
from simba.utils.checks import (check_all_file_names_are_represented_in_video_log, check_file_exist_and_readable, check_if_dir_exists, check_valid_dataframe, check_valid_lst, check_valid_boolean)
from simba.utils.data import detect_bouts, slice_roi_dict_for_video
from simba.utils.enums import ROI_SETTINGS
from simba.utils.errors import InvalidInputError, NoROIDataError
from simba.utils.printing import SimbaTimer, stdout_success, stdout_warning
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.warnings import ROIWarning

TOTAL_TIME = 'TOTAL BEHAVIOR TIME IN ROI (S)'
START_COUNTS = 'STARTED BEHAVIOR BOUTS IN ROI (COUNT)'
ENDED_COUNTS = 'ENDED BEHAVIOR BOUTS IN ROI (COUNT)'
SHAPE_TYPE = 'Shape_type'
VERTICES = 'vertices'

BR_TAG = 'Bottom right tag'
B_TAG = 'Bottom tag'
T_TAG = 'Top tag'
C_TAG = 'Center tag'
BL_TAG = 'Bottom left tag'
TR_TAG = 'Top right tag'
TL_TAG = 'Top left tag'
R_TAG = 'Right tag'
L_TAG = 'Left tag'
BR_X = "Bottom_right_X"
BR_Y = "Bottom_right_Y"
TL_X = 'topLeftX'
TL_Y = "topLeftY"
CENTER_X, CENTER_Y = "Center_X", "Center_Y"


class ROIClfCalculator(ConfigReader):
    """
    Compute aggregate statistics of classification results within user-defined ROIs.
    Results are stored in `project_folder/logs` directory of the SimBA project.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param List[str] bp_names: List of body-parts to use as proxy for animal locations.
    :param Optional[Union[str, os.PathLike]] save_path: Optional location where to store the results in CSV format. If None, then results are stored in logs folder of SImBA project.
    :param Optional[List[Union[str, os.PathLike]]] data_paths: Optional list of data files to analyze. If None, then all file sin the ``machine_results`` directory is analyzed.
    :param Optional[List[str]] clf_names: Optional List of classifiers to analyze. If None, then all classifiers in SimBA project are analyzed.
    :param Optional[List[str]] roi_names: Optional list of ROI names to analyze. If None, then all ROI names are analyzed.
    :param bool clf_time: If True, computes aggregate time of each classifier in each ROI. Deafult True.
    :param bool started_bout_cnt: If True, computes started bout counts of each classifier in each ROI. Deafult True.
    :param bool ended_bout_cnt: If True, computes ended bout counts of each classifier in each ROI. Deafult True.

    .. note:
       'GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results`__.

    :example:
    >>> analyzer = ROIClfCalculator(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini", bp_names=('nose',), clf_names=('straub_tail',))
    >>> analyzer.run()
    >>> analyzer.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bp_names: List[str],
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 data_paths: Optional[List[Union[str, os.PathLike]]] = None,
                 clf_names: Optional[List[str]] = None,
                 roi_names: Optional[List[str]] = None,
                 clf_time: bool = True,
                 started_bout_cnt: bool = True,
                 ended_bout_cnt: bool = True,
                 bout_table: bool = False):

        check_valid_boolean(value=clf_time, source=f'{self.__class__.__name__} clf_time', raise_error=True)
        check_valid_boolean(value=started_bout_cnt, source=f'{self.__class__.__name__} started_bout_cnt', raise_error=True)
        check_valid_boolean(value=ended_bout_cnt, source=f'{self.__class__.__name__} ended_bout_cnt', raise_error=True)
        check_valid_boolean(value=bout_table, source=f'{self.__class__.__name__} bout_table', raise_error=True)
        if not any([clf_time, started_bout_cnt, ended_bout_cnt, bout_table]):
            raise InvalidInputError(msg='clf_time, started_bout_cnt, ended_bout_cnt, bout_table are all False. Set at least one measure to True', source=self.__class__.__name__)
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg=f'No ROI data found. Expected at path {self.roi_coordinates_path}. Create ROI data before computing ROI classification data stratisfied by ROI.', source=self.__class__.__name__)
        self.read_roi_data()
        check_valid_lst(data=bp_names, source=f'{self.__class__.__name__} bp_names', min_len=1, valid_dtypes=(str,), valid_values=self.body_parts_lst)
        self.bout_save_path = os.path.join(self.logs_path, f"Classification_time_by_ROI_detailed_bouts_{self.datetime}.csv")
        if save_path is None:
            self.save_path = os.path.join(self.logs_path, f"Classification_time_by_ROI_{self.datetime}.csv")
        else:
            check_if_dir_exists(os.path.dirname(save_path))
            self.save_path = save_path
        if data_paths is None:
            if len(self.machine_results_paths) == 0:
                NoROIDataError(msg=f'Cannot compute classification by ROI data. No classification data found in {self.machine_results_dir} directory', source=self.__class__.__name__)
            data_paths = self.machine_results_paths
        else:
            check_valid_lst(data=data_paths, source=f'{self.__class__.__name__} data_paths', valid_dtypes=(str,), min_len=1)
            for i in data_paths: check_file_exist_and_readable(file_path=i)
        self.data_paths = data_paths
        if clf_names is not None:
            check_valid_lst(data=clf_names, source=f'{self.__class__.__name__} clf_names', min_len=1, valid_dtypes=(str,), valid_values=self.clf_names)
        if roi_names is not None:
            check_valid_lst(data=roi_names, source=f'{self.__class__.__name__} roi_names', min_len=1, valid_dtypes=(str,), valid_values=self.shape_names)
        self.bp_names, self.clf_time, self.started_bout_cnt, self.ended_bout_cnt = bp_names, clf_time, started_bout_cnt,ended_bout_cnt
        self.bp_cols = []
        self.clf_names, self.roi_names, self.bout_table = clf_names, roi_names, bout_table
        for bp_name in self.bp_names: self.bp_cols.append([f"{bp_name}_x", f"{bp_name}_y", f"{bp_name}_p"])
        self.required_fields = [i for ii in self.bp_cols for i in ii] + list(self.clf_names)
        self.results_df = pd.DataFrame(columns=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE', 'VALUE'])

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        results = {}
        bouts_results = []
        for cnt, data_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            video_name = get_fn_ext(filepath=data_path)[1]
            print(f'Analyzing classification ROI data for video {video_name} (File {cnt+1}/{len(self.data_paths)})...')
            _, _, self.fps = self.read_video_info(video_name=video_name)
            results[video_name] = {}
            video_rois, video_roi_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=video_name)
            if len(video_roi_names) == 0:
                ROIWarning(msg=f'Skipping video {video_name}: No ROIs found for video {video_name}', source=self.__class__.__name__)
                continue
            input_video_rois = get_roi_dict_from_dfs(rectangle_df=video_rois['rectangles'], circle_df=video_rois['circleDf'], polygon_df=video_rois['polygons'])
            video_rois = {k: v for k, v in input_video_rois.items() if k in self.roi_names}
            if len(list(video_rois.keys())) == 0:
                ROIWarning(msg=f'Skipping video {video_name}: No ROIs found for video {video_name}. The video has the ROIs {list(input_video_rois.keys())} but analysis is to be performed on ROIs {self.roi_names}', source=self.__class__.__name__)
                continue
            data_df = read_df(file_path=data_path, file_type=self.file_type)
            check_valid_dataframe(df=data_df, source=f'{data_path}', required_fields=self.required_fields)
            data_df = data_df[self.required_fields]
            for (bp_x, bp_y, bp_p) in self.bp_cols:
                bp_name = bp_x[:-2]
                bp_arr = data_df[[bp_x, bp_y]].values.astype(np.int32)
                results[video_name][bp_name] = {}
                for roi_name, roi_data in video_rois.items():
                    if roi_data[SHAPE_TYPE] == ROI_SETTINGS.RECTANGLE.value:
                        roi_coords = np.array([[roi_data[TL_X], roi_data[TL_Y]], [roi_data[BR_X], roi_data[BR_Y]]]) #, roi_data[['Bottom_right_X', 'Bottom_right_Y']].values]).astype(np.int32)
                        results[video_name][bp_name][roi_name] = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_arr, roi_coords=roi_coords)
                    elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                        circle_center = np.array([roi_data[CENTER_X], roi_data[CENTER_Y]]).astype(np.int32)
                        results[video_name][bp_name][roi_name] = FeatureExtractionMixin.is_inside_circle(bp=bp_arr, roi_center=circle_center, roi_radius=roi_data['radius'])
                        pass
                    elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.POLYGON.value:
                        vertices = roi_data[VERTICES].astype(np.int32)
                        results[video_name][bp_name][roi_name] = FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_arr, roi_coords=vertices)
            for clf_name in self.clf_names:
                clf_data = data_df[clf_name].values
                for bp_name, bp_data in results[video_name].items():
                    for roi_name, roi_data in results[video_name][bp_name].items():
                        field_name = f'{clf_name}_{bp_name}_{roi_name}'
                        data_df[field_name] = 0
                        roi_clf_idx = np.where((roi_data == 1) & (clf_data == 1))[0]
                        data_df[field_name].iloc[roi_clf_idx] = 1
                        bouts = detect_bouts(data_df=data_df, target_lst=[field_name], fps=int(self.fps))
                        bouts['ROI NAME'] = roi_name
                        bouts['BODY-PART NAME'] = bp_name
                        bouts['CLASSIFIER NAME'] = clf_name
                        bouts['VIDEO NAME'] = video_name
                        bouts_results.append(bouts)
                        total_time = float(bouts['Bout_time'].sum())
                        start_frames, end_frames = list(bouts["Start_frame"]), list(bouts["End_frame"])
                        roi_clf_start_cnt = float(len([x for x in start_frames if x in roi_clf_idx]))
                        roi_clf_end_cnt = float(len([x for x in start_frames if x in roi_clf_idx]))
                        if self.clf_time:
                            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, roi_name, bp_name, TOTAL_TIME, total_time]
                        if self.started_bout_cnt:
                            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, roi_name, bp_name, START_COUNTS, roi_clf_start_cnt]
                        if self.ended_bout_cnt:
                            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, roi_name, bp_name, ENDED_COUNTS, roi_clf_end_cnt]
            video_timer.stop_timer()
            print(f'Video {video_name} classifier by ROI analysis complete {len(video_rois)} ROI(s) complete ({cnt+1}/{len(self.data_paths)}) (elapsed time {video_timer.elapsed_time_str}s) ...')

        self.bouts_results = pd.concat(bouts_results, axis=0).reset_index(drop=True) if len(bouts_results) > 0 else None

    def save(self):
        if len(self.results_df) == 0:
            raise NoROIDataError(f'No ROI drawings detected for the {len(self.data_paths)} video file(s). No data is saved.', source=self.__class__.__name__)
        else:
            self.results_df = self.results_df.sort_values(by=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE'])
            self.results_df['VALUE'] = self.results_df['VALUE'].round(4)
            self.results_df.to_csv(self.save_path)
            self.timer.stop_timer()
            stdout_success(msg=f"Classification by ROI data for {len(self.data_paths)} video(s) saved in {self.save_path}.", elapsed_time=self.timer.elapsed_time_str)
        if self.bout_table and self.bouts_results is not None:
            self.bouts_results = self.bouts_results.drop(['Event'], axis=1)
            self.bouts_results= self.bouts_results.rename(columns={'Start_time': 'START TIME (S)', 'End Time': 'END TIME (S)', 'Start_frame': 'START FRAME', 'End_frame': 'END FRAME', 'Bout_time': 'DURATION (S)'})
            self.bouts_results = self.bouts_results[['VIDEO NAME', 'BODY-PART NAME', 'ROI NAME', 'CLASSIFIER NAME', 'START TIME (S)', 'END TIME (S)', 'START FRAME', 'END FRAME', 'DURATION (S)']]
            self.bouts_results.to_csv(self.bout_save_path)
            stdout_success(msg=f"Detailed classification by ROI bout data for saved in {self.bout_save_path}.", elapsed_time=self.timer.elapsed_time_str)

# analyzer = ROIClfCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", bp_names=['Nose'], clf_names=['straub_tail'], roi_names=['Cue_light_1'], bout_table=True)
# analyzer.run()
# analyzer.save()

# analyzer = ROIClfCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", bp_names=['Nose'], clf_names=['straub_tail'], measures=['TOTAL BEHAVIOR TIME IN ROI (S)'])
# analyzer.run()
#


#clf_ROI_analyzer = clf_within_ROI(config_ini="/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini")
#clf_ROI_analyzer.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['rec'], 'Circle': ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])






#
# clf_ROI_analyzer = clf_within_ROI(config_ini="/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini")
# clf_ROI_analyzer.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['rec'], 'Circle': ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])
#

# test = ROIClfCalculator(config_ini="/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini")
# test.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['DAMN'], 'Circle': [], 'Polygon': ['YOU_SUCK_SIMON']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])
