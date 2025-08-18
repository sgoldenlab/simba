__author__ = "Simon Nilsson"

import functools
import glob
import itertools
import multiprocessing
import os
import platform
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_if_valid_img, check_int, check_nvidea_gpu_available,
    check_valid_boolean, check_valid_lst)
from simba.utils.data import detect_bouts, slice_roi_dict_from_attribute
from simba.utils.enums import Defaults, Keys
from simba.utils.errors import NoROIDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_video_meta_data,
                                    read_df, read_frm_of_video, write_df)


def slice_rectangle_from_img(img: np.ndarray,
                             top_left_x: int,
                             top_left_y: int,
                             bottom_right_x: int,
                             bottom_right_y: int):

    check_if_valid_img(data=img, source=f'{slice_rectangle_from_img.__name__} img', raise_error=True)
    height, width = img.shape[:2]
    tl_x = max(0, min(top_left_x, width))
    br_x = max(0, min(bottom_right_x, width))
    tl_y = max(0, min(top_left_y, height))
    br_y = max(0, min(bottom_right_y, height))

    if tl_x >= br_x or tl_y >= br_y:
        raise NoROIDataError(msg='The ROI has no area', source=slice_rectangle_from_img.__name__)

    return img[tl_y:br_y, tl_x:br_x]


def _get_intensity_scores_in_rois(frm_list: List[int],
                                  video_rois: dict,
                                  video_path: str):
    results = {}
    for frm_idx in range(frm_list[0], frm_list[-1]+1):
        print(f'Analyzing frame {frm_idx}...')
        img = read_frm_of_video(video_path=video_path, frame_index=frm_idx)
        for _, rectangle in video_rois[Keys.ROI_RECTANGLES.value].iterrows():
            if rectangle["Name"] not in results.keys(): results[rectangle["Name"]] = {}
            tl_x, tl_y = rectangle["topLeftX"], rectangle["topLeftY"]
            br_x, br_y = rectangle["Bottom_right_X"], rectangle["Bottom_right_Y"]
            roi_image = slice_rectangle_from_img(img=img, top_left_x=tl_x, top_left_y=tl_y, bottom_right_x=br_x, bottom_right_y=br_y)
            if roi_image.ndim == 3:
                results[rectangle["Name"]][frm_idx] = np.average(np.linalg.norm(roi_image, axis=2)) / np.sqrt(3)
            else:
                results[rectangle["Name"]][frm_idx] = np.average(roi_image)
        for _, polygon in video_rois[Keys.ROI_POLYGONS.value].iterrows():
            if polygon["Name"] not in results.keys(): results[polygon["Name"]] = {}
            x, y, w, h = cv2.boundingRect(polygon["vertices"])
            roi_img = img[y : y + h, x : x + w].copy()
            pts = polygon["vertices"] - polygon["vertices"].min(axis=0)
            mask = np.zeros(roi_img.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(roi_img, roi_img, mask=mask)
            bg = np.ones_like(roi_img, np.uint8)
            cv2.bitwise_not(bg, bg, mask=mask)
            roi_image = bg + dst
            if roi_image.ndim == 3:
                results[polygon["Name"]][frm_idx] = np.average(np.linalg.norm(roi_image, axis=2)) / np.sqrt(3)
            else:
                results[polygon["Name"]][frm_idx] = np.average(roi_image)
        for _, circle in video_rois[Keys.ROI_CIRCLES.value].iterrows():
            if circle["Name"] not in results.keys(): results[circle["Name"]] = {}
            roi_img = img[circle["centerY"] : (circle["centerY"] + 2 * circle["radius"]), circle["centerX"] : (circle["centerX"] + 2 * circle["radius"])]
            mask = np.zeros(roi_img.shape[:2], np.uint8)
            circle_img = cv2.circle(mask, (circle["centerX"], circle["centerY"]), circle["radius"], (255, 255, 255), thickness=-1)
            dst = cv2.bitwise_and(roi_img, roi_img, mask=circle_img)
            bg = np.ones_like(roi_img, np.uint8)
            cv2.bitwise_not(bg, bg, mask=mask)
            roi_image = bg + dst
            if roi_image.ndim == 3:
                results[circle["Name"]][frm_idx] = np.average(np.linalg.norm(roi_image, axis=2)) / np.sqrt(3)
            else:
                results[circle["Name"]][frm_idx] = np.average(roi_image)
    return results


class CueLightAnalyzer(ConfigReader):
    """
    Analyze when cue lights are in ON and OFF states. Results are stored in the
    ``project_folder/csv/cue_lights`` cue lights directory.

    :param Union[str, os.PathLike], config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike], data_dir: directory holding pose-estimation data. E.g., ``project_folder/csv/outlier_corrected_movement_location``
    :param List[str] cue_light_names: Names of cue light ROIs, as defined in the SimBA ROI interface.

    .. note::
       `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.

    .. video:: _static/img/cue_light_example_2.webm
       :width: 800
       :autoplay:
       :loop:

    :example:
    >>> cue_light_analyzer = CueLightAnalyzer(config_path='MyProjectConfig', in_dir='project_folder/csv/outlier_corrected_movement_location', cue_light_names=['Cue_light'])
    >>> cue_light_analyzer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike],
                 cue_light_names: List[str],
                 save_dir: Union[str, os.PathLike] = None,
                 core_cnt: int = -1,
                 detailed_data: bool = False):


        ConfigReader.__init__(self, config_path=config_path, read_video_info=True)
        check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__, raise_error=True)
        check_valid_lst(data=cue_light_names, source=self.__class__.__name__, valid_dtypes=(str,), min_len=1, raise_error=True)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_valid_boolean(value=detailed_data, source=f'{self.__class__.__name__} detailed_data', raise_error=True)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{self.file_type}'], raise_error=True, as_dict=True)
        self.read_roi_data()
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.cue_light_names, self.detailed_data = cue_light_names, detailed_data
        if save_dir is None:
            self.save_dir = self.cue_lights_data_dir
        else:
            self.save_dir = save_dir
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.video_cnt = len(list(self.data_paths.keys()))

    def _get_kmeans(self,
                    intensities: Dict[str, Dict[int, int]]):
        results = {}
        for cue_light_name, cue_light_data in intensities.items():
            cue_light_data = dict(sorted(cue_light_data.items()))
            cue_light_data = np.array(list(cue_light_data.values())).astype(np.float64)
            centroids, labels, _ = Statistics().kmeans_1d(data=cue_light_data, k=2, max_iters=300, calc_medians=True)
            centroids = centroids.flatten()
            if centroids[0] > centroids[1]:
                labels = 1 - labels
                centroids = centroids[::-1]
            results[cue_light_name] = {'labels': labels, 'intensities': cue_light_data, 'centroids': centroids}
        return results


    def _append_light_data(self, data_df: pd.DataFrame, kmeans_data: dict):
        for shape_name in self.cue_light_names:
            data_df[f'{shape_name}'] = kmeans_data[shape_name]['labels']
            data_df[f'{shape_name}_INTENSITY'] = kmeans_data[shape_name]['intensities']
        return data_df.fillna(0)

    def _remove_outlier_events(self, data_df: pd.DataFrame, time_threshold: float = 0.03):
        self.detailed_df_lst = []
        for shape_name in self.cue_light_names:
            que_light_bouts = detect_bouts(data_df=data_df, target_lst=[f'{shape_name}'], fps=self.fps)
            que_light_negative_outliers = que_light_bouts[que_light_bouts["Bout_time"] <= time_threshold]
            for idx, r in que_light_negative_outliers.iterrows():
                data_df.loc[r["Start_frame"] - 1 : r["End_frame"] + 1, f'{shape_name}'] = 0
            detailed_df = detect_bouts(data_df=data_df, target_lst=[f'{shape_name}'], fps=self.fps)
            detailed_df = detailed_df.rename(columns={'Event': 'CUE LIGHT', 'Start_time': 'ONSET TIME','End Time': 'OFFSET TIME','Start_frame': 'ONSET FRAME','End_frame': 'OFFSET FRAME','Bout_time': 'ONSET DURATION'})
            detailed_df['VIDEO'] = self.video_name
            detailed_df = detailed_df[['VIDEO', 'CUE LIGHT', 'ONSET TIME', 'OFFSET TIME', 'ONSET FRAME', 'OFFSET FRAME', 'ONSET DURATION']]
            self.detailed_df_lst.append(detailed_df)
        return data_df

    def run(self):
        print(f"Processing {len(self.cue_light_names)} cue light(s) in {len(list(self.data_paths.keys()))} data file(s)...")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=list(self.data_paths.values()))
        video_timer = SimbaTimer(start=True)
        for file_cnt, (file_name, file_path) in enumerate(self.data_paths.items()):
            self.data_df = read_df(file_path, self.file_type)
            self.video_name = file_name
            self.save_path = os.path.join(self.save_dir, f"{file_name}.{self.file_type}")
            _, _, self.fps = self.read_video_info(video_name=file_name)
            video_roi_dict, roi_names, video_roi_cnt = slice_roi_dict_from_attribute(data=self.roi_dict, shape_names=self.cue_light_names, video_names=[file_name])
            missing_rois = [x for x in self.cue_light_names if x not in roi_names]
            if len(missing_rois) > 0:
                raise NoROIDataError(msg=f'The video {file_name} does not have cue light ROI(s) named {missing_rois}.', source=self.__class__.__name__)
            self.video_path = find_video_of_file(video_dir=self.video_dir, filename=file_name, raise_error=True)
            self.video_meta_data = get_video_meta_data(self.video_path)
            self.frm_lst = list(range(0, self.video_meta_data["frame_count"], 1))
            self.frame_chunks = np.array_split(self.frm_lst, self.core_cnt)
            self.intensities = {}
            print(f'Getting light intensities for {video_roi_cnt} cue light in video {file_name}... (frame count: {self.video_meta_data["frame_count"]}, video: {file_cnt+1}/{self.video_cnt})')
            with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial(_get_intensity_scores_in_rois,
                                              video_rois=video_roi_dict,
                                              video_path=self.video_path)
                for cnt, result in enumerate(pool.imap(constants, self.frame_chunks, chunksize=self.multiprocess_chunksize)):
                    for key, subdict in result.items():
                        if key in self.intensities:self.intensities[key].update(subdict)
                        else: self.intensities[key] = subdict
                        print(f'Batch {int(np.ceil(cnt + 1 / self.core_cnt))} complete...')
            pool.terminate()
            pool.join()
            kmeans = self._get_kmeans(intensities=self.intensities)
            self.data_df = self._append_light_data(data_df=self.data_df, kmeans_data=kmeans)
            self.data_df = self._remove_outlier_events(data_df=self.data_df)
            write_df(self.data_df, self.file_type, self.save_path)
            video_timer.stop_timer()
            print(f'Cue-light data video {file_name} complete. Saved at {self.save_path} (elapsed time: {video_timer.elapsed_time_str}s).')
        if self.detailed_data:
            details_save_path = os.path.join(self.logs_path, f'cue_light_details_{self.datetime}.csv')
            detailed_df = pd.concat(self.detailed_df_lst, axis=0).reset_index(drop=True)
            detailed_df = detailed_df.sort_values(by=['VIDEO', 'CUE LIGHT', 'ONSET TIME'], ascending=True)
            detailed_df.to_csv(details_save_path)
            print(f'Detailed cue light data saved at {details_save_path}...')
        self.timer.stop_timer()
        stdout_success(msg=f"Analysed {self.video_cnt} files. Data stored in {self.save_dir}", elapsed_time=self.timer.elapsed_time)

# if __name__ == "__main__":
#     test = CueLightAnalyzer(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini",
#                             data_dir=r'C:\troubleshooting\cue_light\t1\project_folder\csv\outlier_corrected_movement_location',
#                             cue_light_names=['cl'],
#                             save_dir=r'C:\troubleshooting\cue_light\t1\project_folder\csv\cue_lights',
#                             core_cnt=23,
#                             detailed_data=True)
#     test.run()




# test = CueLightAnalyzer(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
#                         in_dir='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/csv/outlier_corrected_movement_location',
#                         cue_light_names=['Cue_light'])
# test.run()
