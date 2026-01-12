__author__ = "Simon Nilsson; sronilsson@gmail.com"

import functools
import multiprocessing
import os
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.roi_tools.roi_utils import get_roi_dict_from_dfs
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists, check_int,
    check_valid_boolean, check_valid_dataframe, check_valid_lst)
from simba.utils.data import (detect_bouts, slice_roi_dict_for_video,
                              terminate_cpu_pool)
from simba.utils.enums import ROI_SETTINGS, Keys
from simba.utils.errors import InvalidInputError, NoROIDataError
from simba.utils.lookups import get_current_time
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_fn_ext, read_df,
                                    read_video_info)
from simba.utils.warnings import NotEnoughDataWarning, ROIWarning

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



def _clf_by_roi_helper(data: tuple,
                       roi_dict: dict,
                       verbose: bool,
                       required_fields: list,
                       bp_cols: list,
                       clf_time: bool,
                       start_bout_cnt: bool,
                       end_bout_cnt: bool,
                       clf_names: list,
                       roi_names: list,
                       video_info_df: pd.DataFrame):

    batch_id, data_paths = data
    batch_results, batch_bouts_results = {}, []
    batch_results_df = pd.DataFrame(columns=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE', 'VALUE'])
    for cnt, data_path in enumerate(data_paths):
        video_timer = SimbaTimer(start=True)
        video_name = get_fn_ext(filepath=data_path)[1]
        if verbose: print(f'Analyzing classification ROI data for video {video_name} (File {cnt+1}/{len(data_paths)}, core batch: {batch_id})...')
        _, _, fps = read_video_info(video_name=video_name, video_info_df=video_info_df)
        batch_results[video_name] = {}
        video_rois, video_roi_names = slice_roi_dict_for_video(data=roi_dict, video_name=video_name)
        if len(video_roi_names) == 0:
            ROIWarning(msg=f'Skipping video {video_name}: No ROIs found for video {video_name}', source=_clf_by_roi_helper.__name__)
            continue
        video_roi_names = roi_names if roi_names is not None else video_roi_names
        input_video_rois = get_roi_dict_from_dfs(rectangle_df=video_rois[Keys.ROI_RECTANGLES.value], circle_df=video_rois[Keys.ROI_CIRCLES.value], polygon_df=video_rois[Keys.ROI_POLYGONS.value])
        video_rois = {k: v for k, v in input_video_rois.items() if k in video_roi_names}
        if len(list(video_rois.keys())) == 0:
            ROIWarning(msg=f'Skipping video {video_name}: No ROIs found for video {video_name}. The video has the ROIs {list(input_video_rois.keys())} but analysis is to be performed on ROIs {video_roi_names}', source=self.__class__.__name__)
            continue
        data_df = read_df(file_path=data_path)
        check_valid_dataframe(df=data_df, source=f'{data_path}', required_fields=required_fields)
        data_df = data_df[required_fields]
        for (bp_x, bp_y, bp_p) in bp_cols:
            bp_name = bp_x[:-2]
            bp_arr = data_df[[bp_x, bp_y]].values.astype(np.int32)
            batch_results[video_name][bp_name] = {}
            for roi_name, roi_data in video_rois.items():
                if roi_data[SHAPE_TYPE] == ROI_SETTINGS.RECTANGLE.value:
                    roi_coords = np.array([[roi_data[TL_X], roi_data[TL_Y]], [roi_data[BR_X], roi_data[BR_Y]]])
                    batch_results[video_name][bp_name][roi_name] = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_arr, roi_coords=roi_coords)
                elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                    circle_center = np.array([roi_data[CENTER_X], roi_data[CENTER_Y]]).astype(np.int32)
                    batch_results[video_name][bp_name][roi_name] = FeatureExtractionMixin.is_inside_circle(bp=bp_arr, roi_center=circle_center, roi_radius=roi_data['radius'])
                elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.POLYGON.value:
                    vertices = roi_data[VERTICES].astype(np.int32)
                    batch_results[video_name][bp_name][roi_name] = FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_arr, roi_coords=vertices)
        for clf_name in clf_names:
            clf_data = data_df[clf_name].values
            for bp_name, bp_data in batch_results[video_name].items():
                for roi_name, roi_data in batch_results[video_name][bp_name].items():
                    field_name = f'{clf_name}_{bp_name}_{roi_name}'
                    data_df[field_name] = 0
                    roi_clf_idx = np.where((roi_data == 1) & (clf_data == 1))[0]
                    data_df[field_name].iloc[roi_clf_idx] = 1
                    bouts = detect_bouts(data_df=data_df, target_lst=[field_name], fps=fps)
                    bouts['ROI NAME'] = roi_name
                    bouts['BODY-PART NAME'] = bp_name
                    bouts['CLASSIFIER NAME'] = clf_name
                    bouts['VIDEO NAME'] = video_name
                    batch_bouts_results.append(bouts)
                    total_time = float(bouts['Bout_time'].sum())
                    start_frames, end_frames = list(bouts["Start_frame"]), list(bouts["End_frame"])
                    roi_clf_start_cnt = float(len([x for x in start_frames if x in roi_clf_idx]))
                    roi_clf_end_cnt = float(len([x for x in start_frames if x in roi_clf_idx]))
                    if clf_time:
                        batch_results_df.loc[len(batch_results_df)] = [video_name, clf_name, roi_name, bp_name, TOTAL_TIME, total_time]
                    if start_bout_cnt:
                        batch_results_df.loc[len(batch_results_df)] = [video_name, clf_name, roi_name, bp_name, START_COUNTS, roi_clf_start_cnt]
                    if end_bout_cnt:
                        batch_results_df.loc[len(batch_results_df)] = [video_name, clf_name, roi_name, bp_name, ENDED_COUNTS, roi_clf_end_cnt]

        video_timer.stop_timer()
        if verbose: print(f'Video {video_name} classifier by ROI analysis complete {len(video_rois)} ROI(s) complete ({cnt+1}/{len(data_paths)}) (elapsed time {video_timer.elapsed_time_str}s) ...')

    batch_bouts_results = pd.concat(batch_bouts_results, axis=0).reset_index(drop=True) if len(batch_bouts_results) > 0 else None
    return (batch_id, batch_results_df, batch_bouts_results)


class ROIClfCalculatorMultiprocess(ConfigReader):
    """
    Compute aggregate statistics of classification results within user-defined ROIs. Results are stored in `project_folder/logs` directory of the SimBA project.

    .. seealso::
       For single core process, see :func:`simba.roi_tools.roi_clf_calculator.ROIClfCalculator`

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
    >>> analyzer = ROIClfCalculatorMultiprocess(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini", bp_names=['resident_NOSE'], clf_names=['attack'], clf_time=True, started_bout_cnt=True, ended_bout_cnt=False, bout_table=True, transpose=True, core_cnt=20)
    >>> analyzer.run()
    >>> analyzer.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bp_names: List[str],
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 data_paths: Optional[List[Union[str, os.PathLike]]] = None,
                 roi_coordinates_path: Union[str, os.PathLike] = None,
                 clf_names: Optional[List[str]] = None,
                 roi_names: Optional[List[str]] = None,
                 clf_time: bool = True,
                 started_bout_cnt: bool = True,
                 ended_bout_cnt: bool = True,
                 bout_table: bool = False,
                 transpose: bool = False,
                 verbose: bool = True,
                 core_cnt: int = -1):

        check_valid_boolean(value=clf_time, source=f'{self.__class__.__name__} clf_time', raise_error=True)
        check_valid_boolean(value=started_bout_cnt, source=f'{self.__class__.__name__} started_bout_cnt', raise_error=True)
        check_valid_boolean(value=ended_bout_cnt, source=f'{self.__class__.__name__} ended_bout_cnt', raise_error=True)
        check_valid_boolean(value=bout_table, source=f'{self.__class__.__name__} bout_table', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.results_flag = True if any([clf_time, started_bout_cnt, ended_bout_cnt]) else False
        if not any([clf_time, started_bout_cnt, ended_bout_cnt, bout_table]):
            raise InvalidInputError(msg='clf_time, started_bout_cnt, ended_bout_cnt, bout_table are all False. Set at least one measure to True', source=self.__class__.__name__)
        ConfigReader.__init__(self, config_path=config_path)
        if roi_coordinates_path is not None:
            check_file_exist_and_readable(file_path=roi_coordinates_path, raise_error=True)
            self.roi_coordinates_path = deepcopy(roi_coordinates_path)
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
        self.clf_names, self.roi_names, self.bout_table, self.transpose, self.verbose = clf_names, roi_names, bout_table, transpose, verbose
        for bp_name in self.bp_names: self.bp_cols.append([f"{bp_name}_x", f"{bp_name}_y", f"{bp_name}_p"])
        self.required_fields = [i for ii in self.bp_cols for i in ii] + list(self.clf_names)
        self.results_df = pd.DataFrame(columns=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE', 'VALUE'])
        self.bouts_results = []

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        print(f'Running classifier analysis stratisfied by ROI in {len(self.data_paths)} file(s) using {self.core_cnt} cores... ({get_current_time()})')
        chunked_data_paths = [self.data_paths[i:i + ((len(self.data_paths) + self.core_cnt - 1) // self.core_cnt)] for i in range(0, len(self.data_paths), (len(self.data_paths) + self.core_cnt - 1) // self.core_cnt)]
        chunked_data_paths = [(i, x) for i, x in enumerate(chunked_data_paths)]
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_clf_by_roi_helper,
                                          verbose=self.verbose,
                                          roi_dict=self.roi_dict,
                                          required_fields=self.required_fields,
                                          bp_cols=self.bp_cols,
                                          clf_time=self.clf_time,
                                          start_bout_cnt=self.started_bout_cnt,
                                          end_bout_cnt=self.ended_bout_cnt,
                                          clf_names=self.clf_names,
                                          video_info_df=self.video_info_df,
                                          roi_names=self.roi_names)
            for cnt, (batch_id, batch_results_df, batch_bout_results) in enumerate(pool.map(constants, chunked_data_paths, chunksize=self.multiprocess_chunksize)):
                self.results_df = pd.concat([self.results_df, batch_results_df], axis=0).reset_index(drop=True)
                self.bouts_results.append(batch_bout_results)
                print(f"Data batch core {batch_id + 1} / {self.core_cnt} complete...")
        self.bouts_results = pd.concat(self.bouts_results, axis=0).reset_index(drop=True) if len(self.bouts_results) > 0 else None
        terminate_cpu_pool(pool=pool, force=False)

    def save(self):
        self.timer.stop_timer()
        if self.results_flag and len(self.results_df) == 0:
            NotEnoughDataWarning(f'No classification results in ROIs detected. No aggregates results saved.', source=self.__class__.__name__)
        elif self.results_flag:
            self.results_df = self.results_df.sort_values(by=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE']).reset_index(drop=True)
            self.results_df['VALUE'] = self.results_df['VALUE'].round(4)
            if self.transpose:
                self.results_df = (self.results_df.pivot_table(index="VIDEO", columns=["CLASSIFIER", "ROI", "BODY-PART", "MEASURE"], values="VALUE", aggfunc="first"))
            else:
                self.results_df  = self.results_df.set_index('VIDEO')
            self.results_df.to_csv(self.save_path)
            stdout_success(msg=f"Classification by ROI data for {len(self.data_paths)} video(s) saved in {self.save_path}.", elapsed_time=self.timer.elapsed_time_str)
        if self.bout_table and self.bouts_results is None:
            NotEnoughDataWarning(f'No ROI classification results detected. No detailed bout data saved.', source=self.__class__.__name__)
        elif self.bout_table:
            self.bouts_results = self.bouts_results.drop(['Event'], axis=1)
            self.bouts_results= self.bouts_results.rename(columns={'Start_time': 'START TIME (S)', 'End Time': 'END TIME (S)', 'Start_frame': 'START FRAME', 'End_frame': 'END FRAME', 'Bout_time': 'DURATION (S)'})
            self.bouts_results = self.bouts_results[['VIDEO NAME', 'BODY-PART NAME', 'ROI NAME', 'CLASSIFIER NAME', 'START TIME (S)', 'END TIME (S)', 'START FRAME', 'END FRAME', 'DURATION (S)']]
            self.bouts_results.to_csv(self.bout_save_path)
            stdout_success(msg=f"Detailed classification by ROI bout data for saved in {self.bout_save_path}.", elapsed_time=self.timer.elapsed_time_str)

# if __name__ == "__main__":
#     analyzer = ROIClfCalculator(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini",
#                                 bp_names=['resident_NOSE'],
#                                 clf_names=['attack'],
#                                 clf_time=True,
#                                 started_bout_cnt=True,
#                                 ended_bout_cnt=False,
#                                 bout_table=True,
#                                 transpose=True,
#                                 core_cnt=20)
#     analyzer.run()
#     analyzer.save()

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
