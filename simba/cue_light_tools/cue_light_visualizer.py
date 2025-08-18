__author__ = "Simon Nilsson"

import functools
import itertools
import multiprocessing
import os
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.data import (create_color_palettes, detect_bouts,
                              slice_roi_dict_from_attribute)
from simba.utils.enums import Defaults, Formats, Keys, TextOptions
from simba.utils.errors import NoROIDataError, NoSpecifiedOutputError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory, find_core_cnt,
                                    get_fn_ext, get_video_meta_data, read_df,
                                    read_frm_of_video, remove_a_folder)


def _plot_cue_light_data(frm_idxs: list,
                         video_setting: bool,
                         frame_setting: bool,
                         show_pose: bool,
                         data_df: pd.DataFrame,
                         bp_names: list,
                         font_size: int,
                         x_shift: int,
                         y_shift: int,
                         frames_save_dir: str,
                         video_save_dir: str,
                         circle_size: int,
                         roi_dict: dict,
                         video_path: str):

    batch_id, frame_rng = frm_idxs[0], frm_idxs[1]
    start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
    video_writer = None
    video_meta_data = get_video_meta_data(video_path=video_path)

    clrs = create_color_palettes(no_animals=1, map_size=len(bp_names)+1, cmaps=['Set3'])[0]
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, f"{batch_id}.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data['fps'], (int(video_meta_data['width']*2), video_meta_data['height']))


    while current_frm <= end_frm:
        img = read_frm_of_video(video_path, frame_index=current_frm)
        img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data["width"]), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if show_pose:
            for bp_cnt, bp_name in enumerate(bp_names):
                col_names = [f'{bp_name}_x', f'{bp_name}_y']
                bp_data = data_df.loc[current_frm, col_names].values.astype(np.int32)
                img = cv2.circle(img, tuple(bp_data), circle_size, clrs[bp_cnt], -1)
        img = PlottingMixin().roi_dict_onto_img(img=img, roi_dict=roi_dict)

        y_shift_counts = 1
        for cue_light_type, cue_light_type_data in roi_dict.items():
            for _, cue_light_data in cue_light_type_data.iterrows():
                color, name = cue_light_data['Color BGR'], cue_light_data['Name']
                light_status = 'ON' if data_df.loc[current_frm, name] == 1 else 'OFF'
                light_color = (0, 255, 255) if light_status == 'ON' else color
                que_light_bouts = detect_bouts(data_df=data_df.loc[0:current_frm], target_lst=[name], fps=video_meta_data['fps'])
                que_light_bouts_cnt = len(que_light_bouts)
                que_light_bouts_duration = round(que_light_bouts['Bout_time'].sum(), 2)
                off_duration = round((((current_frm + 1)/ video_meta_data['fps'])) - que_light_bouts_duration, 2)
                img = PlottingMixin().put_text(img=img, text=f"{name} STATUS:", pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)
                img = PlottingMixin().put_text(img=img, text=light_status, pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value + x_shift), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=light_color)
                y_shift_counts += 1
                img = PlottingMixin().put_text(img=img, text=f"{name} ONSET COUNTS:", pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)
                img = PlottingMixin().put_text(img=img, text=str(que_light_bouts_cnt), pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value + x_shift), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)
                y_shift_counts += 1
                img = PlottingMixin().put_text(img=img, text=f"{name} TIME ON (S):", pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)
                img = PlottingMixin().put_text(img=img, text=str(que_light_bouts_duration), pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value + x_shift), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)
                y_shift_counts += 1
                img = PlottingMixin().put_text(img=img, text=f"{name} TIME OFF (S):", pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)
                img = PlottingMixin().put_text(img=img, text=str(off_duration), pos=(int(video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value + x_shift), int(y_shift*y_shift_counts)), font_size=font_size, font_thickness=2, text_color_bg=(0, 0, 0), text_color=color)

        if video_setting:
            video_writer.write(np.uint8(img))
        if frame_setting:
            frame_save_path = os.path.join(frames_save_dir,f"{current_frm}.png")
            cv2.imwrite(frame_save_path, current_frm)
        print(f"Cue light frame complete: {current_frm} / {video_meta_data['frame_count']}. Video: {video_meta_data['video_name']} ")
        current_frm += 1
    if video_setting:
        video_writer.release()
    return batch_id

class CueLightVisualizer(ConfigReader):
    """

    Visualize SimBA computed cue-light ON and OFF states and the aggregate statistics of ON and OFF
    states.

    :param str config_path: path to SimBA project config file in Configparser format.
    :param List[str] cue_light_names: Names of cue lights, as defined in the SimBA ROI interface.
    :param str video_path: Path to video which user wants to create visualizations of cue light states and aggregate statistics for.
    :param bool frame_setting: If True, creates individual frames in png format. Defaults to False.
    :param bool video_setting: If True, creates compressed videos in mp4 format. Defaults to True.

    .. notes:
       `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.


    .. video:: _static/img/cue_light_example_2.webm
       :width: 800
       :autoplay:
       :loop:

    :examples:
    >>> cue_light_visualizer = CueLightVisualizer(config_path='SimBAConfig', cue_light_names=['Cue_light'], video_path='VideoPath', video_setting=True, frame_setting=False)
    >>> cue_light_visualizer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 cue_light_names: List[str],
                 video_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike],
                 frame_setting: bool = False,
                 video_setting: bool = True,
                 core_cnt: int = -1,
                 show_pose: bool = True):

        ConfigReader.__init__(self, config_path=config_path)
        check_valid_boolean(value=[frame_setting], source=f'{self.__class__.__name__} frame_setting', raise_error=True)
        check_valid_boolean(value=[video_setting], source=f'{self.__class__.__name__} video_setting', raise_error=True)
        check_valid_boolean(value=[show_pose], source=f'{self.__class__.__name__} show_pose', raise_error=True)
        check_valid_lst(data=cue_light_names, source=self.__class__.__name__, valid_dtypes=(str,), min_len=1, raise_error=True)
        check_file_exist_and_readable(file_path=video_path)
        check_file_exist_and_readable(file_path=data_path)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.video_meta_data = get_video_meta_data(video_path)
        _, self.video_name, _ = get_fn_ext(filepath=data_path)
        if (not frame_setting) and (not video_setting):
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos.")
        self.cue_light_names, self.video_path, self.data_path = cue_light_names, video_path, data_path
        self.data_df = read_df(self.data_path, self.file_type)
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.font_size, self.x_shift, self.y_shift = PlottingMixin().get_optimal_font_scales(text='ONE LONG ARSE STRING FOR YA', accepted_px_height=int(self.video_meta_data['height']/2), accepted_px_width=int(self.video_meta_data['width']/2))
        self.circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.video_meta_data['width'], self.video_meta_data['height']), circle_frame_ratio=60)
        self.read_roi_data()
        self.video_setting, self.frame_setting, self.data_path, self.show_pose = video_setting, frame_setting, data_path, show_pose
        self.video_save_dir = os.path.join(self.frames_output_dir, 'cue_lights')
        self.frames_save_dir = os.path.join(self.frames_output_dir, 'cue_lights')
        self.video_roi_dict, roi_names, video_roi_cnt = slice_roi_dict_from_attribute(data=self.roi_dict, shape_names=self.cue_light_names, video_names=[self.video_name])
        missing_rois = [x for x in self.cue_light_names if x not in roi_names]
        if len(missing_rois) > 0:
            raise NoROIDataError(msg=f'The video {self.video_name} does not have cue light ROI(s) named {missing_rois}.', source=self.__class__.__name__)
        if show_pose:
            check_valid_dataframe(df=self.data_df, source=f'{self.__class__.__name__} {data_path}', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.bp_col_names)


    def run(self):
        print(f"Creating video for {len(self.cue_light_names)} cue light(s) in in {self.video_name}...")
        frames_dir, video_temp_dir = None, None
        if self.frame_setting:
            frames_dir = os.path.join(self.frames_save_dir, self.video_name)
            create_directory(paths=frames_dir, overwrite=True)
        if self.video_setting:
            self.save_video_path = os.path.join(self.video_save_dir, f"{self.video_name}.mp4")
            video_temp_dir = os.path.join(self.video_save_dir, 'temp')
            create_directory(paths=video_temp_dir, overwrite=True)
        self.frm_lst = list(range(0, self.video_meta_data["frame_count"], 1))
        self.frame_chunks = np.array_split(self.frm_lst, self.core_cnt)
        self.frame_chunks = [(x, j) for x, j in enumerate(self.frame_chunks)]
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(_plot_cue_light_data,
                                          frame_setting=self.frame_setting,
                                          video_setting=self.video_setting,
                                          show_pose=self.show_pose,
                                          data_df=self.data_df,
                                          frames_save_dir=frames_dir,
                                          video_save_dir=video_temp_dir,
                                          circle_size=self.circle_size,
                                          font_size=self.font_size,
                                          x_shift=self.x_shift,
                                          y_shift=self.y_shift,
                                          roi_dict=self.video_roi_dict,
                                          video_path=self.video_path,
                                          bp_names=self.body_parts_lst)
            for cnt, result in enumerate(pool.imap(constants, self.frame_chunks, chunksize=self.multiprocess_chunksize)):
                print(f'Batch {result+1/self.core_cnt} complete...')
            pool.terminate()
            pool.join()
        self.timer.stop_timer()
        if self.video_setting:
            print(f"Joining {self.video_name} multiprocessed video...")
            concatenate_videos_in_folder(in_folder=video_temp_dir, save_path=self.save_video_path)
            stdout_success(msg=f"Cue light video visualization for video {self.video_name} saved at {self.save_video_path}", elapsed_time=self.timer.elapsed_time_str)
        if self.frame_setting:
            stdout_success(msg=f"Cue light frame visualization for video {self.video_name} saved at {frames_dir}", elapsed_time=self.timer.elapsed_time_str)

# if __name__ == "__main__":
#     test = CueLightVisualizer(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini",
#                               cue_light_names=['cl'],
#                               video_path=r"C:\troubleshooting\cue_light\t1\project_folder\videos\2025-05-21 16-10-06_cropped.mp4",
#                               data_path=r"C:\troubleshooting\cue_light\t1\project_folder\csv\cue_lights\2025-05-21 16-10-06_cropped.csv",
#                               video_setting=True,
#                               frame_setting=False,
#                               core_cnt=23)
#
#     test.run()