__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists, check_valid_dataframe,
    check_valid_lst, check_valid_tuple)
from simba.utils.data import detect_bouts
from simba.utils.errors import InvalidInputError, NoROIDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.warnings import ROIWarning

MEASURES = ('TOTAL BEHAVIOR TIME IN ROI (S)', 'STARTED BEHAVIOR BOUTS IN ROI (COUNT)', 'ENDED BEHAVIOR BOUTS IN ROI (COUNT)')

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
    :param Tuple[str] measures: Tuple of measures to include. Options: 'TOTAL BEHAVIOR TIME IN ROI (S)', 'STARTED BEHAVIOR BOUTS IN ROI (COUNT)', 'ENDED BEHAVIOR BOUTS IN ROI (COUNT)'.

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
                 measures: List[str] = ['TOTAL BEHAVIOR TIME IN ROI (S)', 'STARTED BEHAVIOR BOUTS IN ROI (COUNT)', 'ENDED BEHAVIOR BOUTS IN ROI (COUNT)']):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        self.read_roi_data()
        if data_paths is None:
            data_paths = self.machine_results_paths
        check_valid_lst(data=data_paths, source=f'{self.__class__.__name__} data_paths', valid_dtypes=(str,), min_len=1)
        if clf_names is not None:
            check_valid_lst(data=clf_names, source=f'{self.__class__.__name__} clf_names', min_len=1, valid_dtypes=(str,))
            self.clf_names = clf_names
        if roi_names is not None:
            check_valid_lst(data=roi_names, source=f'{self.__class__.__name__} roi_names', min_len=1, valid_dtypes=(str,))
            self.roi_names = roi_names
        else:
            self.roi_names = self.shape_names
        unaccepted_measures = [x for x in measures if x not in MEASURES]
        check_valid_lst(data=measures, source=f'{self.__class__.__name__} measures', min_len=1, valid_dtypes=(str,))
        if len(unaccepted_measures) > 0:
            raise InvalidInputError(msg=f'{unaccepted_measures} are invalid measure options. Accepted: {MEASURES}', source=self.__class__.__name__)
        check_valid_lst(data=bp_names, source=f'{self.__class__.__name__} bp_names', min_len=1, valid_dtypes=(str,))
        unaccepted_bps = [x for x in bp_names if x not in self.body_parts_lst]
        if len(unaccepted_bps) > 0:
            raise InvalidInputError(msg=f'{unaccepted_bps} are invalid body-part options. Accepted: {self.body_parts_lst}', source=self.__class__.__name__)
        if save_path is None:
            self.save_path = os.path.join(self.logs_path, f"Classification_time_by_ROI_{self.datetime}.csv")
        else:
            check_if_dir_exists(os.path.dirname(save_path))
            self.save_path = save_path
        self.bp_names, self.measures = bp_names, measures
        self.data_paths = data_paths
        self.bp_cols = []
        for bp_name in self.bp_names: self.bp_cols.append([f"{bp_name}_x", f"{bp_name}_y", f"{bp_name}_p"])
        self.required_fields = [i for ii in self.bp_cols for i in ii] + list(self.clf_names)
        self.results_df = pd.DataFrame(columns=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE', 'VALUE'])

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        results = {}
        for cnt, data_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            video_name = get_fn_ext(filepath=data_path)[1]
            print(f'Analyzing classification ROI data for video {video_name} (File {cnt+1}/{len(self.data_paths)})...')
            _, _, self.fps = self.read_video_info(video_name=video_name)
            results[video_name] = {}
            video_rectangles = self.rectangles_df[self.rectangles_df['Video'] == video_name]
            video_circles = self.circles_df[self.circles_df['Video'] == video_name]
            video_polygons = self.polygon_df[self.circles_df['Video'] == video_name]
            if len(video_rectangles) + len(video_circles) + len(video_polygons) == 0:
                ROIWarning(msg=f'Skipping video {video_name}: No drawn ROIs found for video {video_name}', source=self.__class__.__name__)
                continue
            else:
                data_df = read_df(file_path=data_path, file_type=self.file_type)
                check_valid_dataframe(df=data_df, source=f'{data_path}', required_fields=self.required_fields)
                data_df = data_df[self.required_fields]
                for (bp_x, bp_y, bp_p) in self.bp_cols:
                    bp_name = bp_x[:-2]
                    bp_arr = data_df[[bp_x, bp_y]].values.astype(np.int32)
                    results[video_name][bp_name] = {}
                    for idx, rectangle in video_rectangles.iterrows():
                        roi_coords = np.array([rectangle[['topLeftX', 'topLeftY']].values, rectangle[['Bottom_right_X', 'Bottom_right_Y']].values]).astype(np.int32)
                        results[video_name][bp_name][rectangle['Name']] = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_arr, roi_coords=roi_coords)
                    for idx, circle in video_circles.iterrows():
                        circle_center = np.array(circle[['Center_X', 'Center_Y']].values).astype(np.int32)
                        results[video_name][bp_name][circle['Name']] = FeatureExtractionMixin.is_inside_circle(bp=bp_arr, roi_center=circle_center, roi_radius=circle['radius'])
                    for idx, polygon in video_polygons.iterrows():
                        vertices = polygon['vertices'].astype(np.int32)
                        results[video_name][bp_name][polygon['Name']] = FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_arr, roi_coords=vertices)

                for clf_name in self.clf_names:
                    clf_data = data_df[clf_name].values
                    for bp_name, bp_data in results[video_name].items():
                        for roi_name, roi_data in results[video_name][bp_name].items():
                            field_name = f'{clf_name}_{bp_name}_{roi_name}'
                            data_df[field_name] = 0
                            roi_clf_idx = np.where((roi_data == 1) & (clf_data == 1))[0]
                            data_df[field_name].iloc[roi_clf_idx] = 1
                            bouts = detect_bouts(data_df=data_df, target_lst=[field_name], fps=int(self.fps))
                            total_time = bouts['Bout_time'].sum()
                            start_frames, end_frames = list(bouts["Start_frame"]), list(bouts["End_frame"])
                            roi_clf_start_cnt = len([x for x in start_frames if x in roi_clf_idx])
                            roi_clf_end_cnt = len([x for x in start_frames if x in roi_clf_idx])
                            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, roi_name, bp_name, 'TOTAL BEHAVIOR TIME IN ROI (S)', total_time]
                            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, roi_name, bp_name, 'STARTED BEHAVIOR BOUTS IN ROI (COUNT)', roi_clf_start_cnt]
                            self.results_df.loc[len(self.results_df)] = [video_name, clf_name, roi_name, bp_name, 'ENDED BEHAVIOR BOUTS IN ROI (COUNT)', roi_clf_end_cnt]
                video_timer.stop_timer()
                print(f'Video {video_name} complete (elapsed time {video_timer.elapsed_time_str}s) ...')
    def save(self):
        if len(self.results_df) == 0:
            raise NoROIDataError(f'No ROI drawings detected for the {len(self.data_paths)} video file(s). No data is saved.', source=self.__class__.__name__)
        else:
            self.results_df = self.results_df[self.results_df['MEASURE'].isin(self.measures)].set_index('VIDEO')
            self.results_df = self.results_df.sort_values(by=['VIDEO', 'CLASSIFIER', 'ROI', 'BODY-PART', 'MEASURE'])
            self.results_df['VALUE'] = self.results_df['VALUE'].round(4)
            self.results_df.to_csv(self.save_path)
            self.timer.stop_timer()
            stdout_success(msg=f"Classification by ROI data for {len(self.data_paths)} video(s) saved in {self.save_path}.", elapsed_time=self.timer.elapsed_time_str)


# analyzer = ROIClfCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", bp_names=['Nose'], clf_names=['straub_tail'], measures=['TOTAL BEHAVIOR TIME IN ROI (S)'])
# analyzer.run()
# analyzer.save()


#clf_ROI_analyzer = clf_within_ROI(config_ini="/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini")
#clf_ROI_analyzer.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['rec'], 'Circle': ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])






#
# clf_ROI_analyzer = clf_within_ROI(config_ini="/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini")
# clf_ROI_analyzer.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['rec'], 'Circle': ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])
#

# test = ROIClfCalculator(config_ini="/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini")
# test.run(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['DAMN'], 'Circle': [], 'Polygon': ['YOU_SUCK_SIMON']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])
