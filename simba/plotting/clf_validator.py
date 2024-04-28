__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_float, check_if_valid_rgb_tuple,
                                check_int, check_str, check_that_column_exist,
                                check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats
from simba.utils.errors import NoFilesFoundError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_df
from simba.utils.warnings import NoDataFoundWarning


class ClassifierValidationClips(ConfigReader):
    """
    Create video clips with overlaid classified events for detection of false positive event bouts.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter int window: Number of seconds before and after the event bout that should be included in the output video.
    :parameter str clf_name: Name of the classifier to create validation videos for.
    :parameter bool clips: If True, creates individual video file clips for each validation bout.
    :parameter Tuple[int, int, int] text_clr: Color of text overlay in BGR.
    :parameter Optional[Tuple[int, int, int]] highlight_clr: Color of text when probability values are above threshold. If None, same as text_clr.
    :parameter float video_speed:  FPS rate in relation to original video. E.g., the same as original video if 1.0. Default: 1.0.
    :parameter bool concat_video:  If True, creates a single video including all events bouts for each video. Default: False.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md#classifier-validation>`_.

    .. image:: _static/img/ClassifierValidationClips_1.png
       :width: 600
       :align: center


    :examples:
    >>> _ = ClassifierValidationClips(config_path='MyProjectConfigPath', window=5, clf_name='Attack', text_clr=(255,255,0), clips=False, concat_video=True).run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 window: int,
                 clf_name: str,
                 data_paths: List[str],
                 text_clr: Optional[Tuple[int, int, int]] = (255,105,180),
                 concat_video: Optional[bool] = False,
                 clips: Optional[bool] = False,
                 video_speed: Optional[float] = 1.0,
                 highlight_clr: Optional[Tuple[int, int, int]] = None):

        ConfigReader.__init__(self, config_path=config_path)
        if (not clips) and (not concat_video):
            raise NoSpecifiedOutputError(msg="Please select to create clips and/or a concatenated video")
        check_int(name="Time window", value=window, min_value=0)
        check_if_valid_rgb_tuple(data=text_clr)
        if highlight_clr is not None: check_if_valid_rgb_tuple(data=highlight_clr)
        check_valid_lst(data=data_paths, source=f'{self.__class__.__name__} data_paths', min_len=1)
        check_str(name=f'{self.__class__.__name__} clf_name', value=clf_name, options=self.clf_names)
        check_float(name=f'{self.__class__.__name__} video_speed', value=video_speed, min_value=10e-6)
        self.window, self.clf_name = int(window), clf_name
        self.clips, self.concat_video, self.video_speed, self.highlight_clr = (clips, concat_video, video_speed, highlight_clr)
        self.p_col = f"Probability_{self.clf_name}"
        self.text_clr, self.data_paths = text_clr, data_paths
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        if not os.path.exists(self.clf_validation_dir):
            os.makedirs(self.clf_validation_dir)
        print(f"Processing {len(self.data_paths)} files...")

    def __insert_inter_frms(self, bout_count: int):
        """
        Helper to create N blank frames separating the classified event bouts.
        """
        for i in range(int(self.fps)):
            inter_frm = np.full((int(self.video_info["height"]), int(self.video_info["width"]), 3), (49, 32, 189)).astype(np.uint8)
            cv2.putText(inter_frm, f"Bout #{bout_count}", (10, (self.video_info["height"] - self.video_info["height"]) + self.spacing_scale), self.font, self.font_size, (0, 0, 0), 2)
            self.concat_writer.write(inter_frm)

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            self.data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=self.data_df, column_name=self.p_col, file_name=file_path)
            _, file_name, _ = get_fn_ext(file_path)
            self.video_path = self.find_video_of_file(video_dir=self.video_dir, filename=file_name)
            if not self.video_path:
                raise NoFilesFoundError(msg=f"Could not find a video file representing {file_name} in the {self.video_dir} directory")
            self.video_info = get_video_meta_data(video_path=self.video_path)
            self.fps = int(self.video_info["fps"])
            self.video_fps = int(self.fps * self.video_speed)
            if self.video_fps < 1: self.video_fps = 1
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = (60, 12, 1500, 1.5)
            self.max_dim = max(self.video_info["width"], self.video_info["height"])
            self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
            self.spacing_scale = int(self.space_scale / (self.res_scale / self.max_dim))
            cap = cv2.VideoCapture(self.video_path)
            clf_bouts = detect_bouts(data_df=self.data_df, target_lst=[self.clf_name], fps=self.fps).reset_index(drop=True)
            if len(clf_bouts) == 0:
                NoDataFoundWarning(msg=f"Skipping video {file_name}: No classified behavior of {self.clf_name} detected...")
                continue
            if self.concat_video:
                self.concat_video_save_path = os.path.join(self.clf_validation_dir, f"{self.clf_name}_{file_name}_all_events.mp4")
                self.concat_writer = cv2.VideoWriter(self.concat_video_save_path, self.fourcc, self.video_fps, (int(self.video_info["width"]), int(self.video_info["height"])))
                self.__insert_inter_frms(bout_count=0)
            for bout_cnt, bout in clf_bouts.iterrows():
                self.bout_cnt = bout_cnt
                event_start_frm, event_end_frm = bout["Start_frame"], bout["End_frame"]
                start_window = int(event_start_frm - (int(self.video_info["fps"]) * self.window))
                end_window = int(event_end_frm + (int(self.video_info["fps"]) * self.window))
                self.save_path = os.path.join(self.clf_validation_dir, self.clf_name + f"_{bout_cnt}_{file_name}.mp4")
                if start_window < 0: start_window = 0
                current_frm = deepcopy(start_window)
                if end_window > len(self.data_df):
                    end_window = len(self.data_df)
                if self.clips:
                    bout_writer = cv2.VideoWriter(self.save_path, self.fourcc, self.video_fps, (int(self.video_info["width"]), int(self.video_info["height"])))
                event_frm_count = end_window - start_window
                frm_cnt = 0
                cap.set(1, current_frm)
                while current_frm < end_window:
                    ret, img = cap.read()
                    p, clf_val = round(float(self.data_df.loc[current_frm, self.p_col]), 3), int(self.data_df.loc[current_frm, self.clf_name])
                    self.add_spacer = 2
                    cv2.putText(img, f"{self.clf_name} event # {self.bout_cnt + 1}", (10, (self.video_info["height"] - self.video_info["height"])+ self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, f"Total frames of event: {event_frm_count}", (10, (self.video_info["height"] - self.video_info["height"]) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, f"Frames of event {start_window} to {end_window}", (10, (self.video_info["height"] - self.video_info["height"]) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    cv2.putText(img, f"Frame number: {current_frm}", (10, (self.video_info["height"] - self.video_info["height"]) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    self.add_spacer += 1
                    if (self.highlight_clr != None) and (clf_val == 1):
                        cv2.putText(img, f"Frame {self.clf_name} probability: {p}", (10, (self.video_info["height"] - self.video_info["height"]) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.highlight_clr, 2)
                    else:
                        cv2.putText(img, f"Frame {self.clf_name} probability: {p}", (10, (self.video_info["height"] - self.video_info["height"]) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.text_clr, 2)
                    print(f"Frame {frm_cnt+1} / {event_frm_count}, Event {self.bout_cnt+1}/{len(clf_bouts)}, Video {file_cnt+1}/{len(self.machine_results_paths)}...")
                    if self.clips:
                        bout_writer.write(img)
                    if self.concat_video:
                        self.concat_writer.write(img)
                    current_frm += 1
                    frm_cnt += 1
                if self.clips:
                    bout_writer.release()
                if self.concat_video and self.bout_cnt != len(clf_bouts) - 1:
                    self.__insert_inter_frms(bout_count=self.bout_cnt+1)
            if self.concat_video:
                self.concat_writer.release()
        self.timer.stop_timer()
        stdout_success(msg=f"All validation clips complete. Files are saved in the {self.clf_validation_dir} directory of the SimBA project", elapsed_time=self.timer.elapsed_time_str)


# test = ClassifierValidationClips(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                  window=1,
#                                  clf_name='Attack',
#                                  clips=False,
#                                  concat_video=True,
#                                  highlight_clr=(255, 0, 0),
#                                  video_speed=0.5,
#                                  text_clr=(0, 0, 255),
#                                  data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.run()

# test = ClassifierValidationClips(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini',
#                                  window=1,
#                                  clf_name='Attack',
#                                  clips=False,
#                                  concat_video=True,
#                                  text_clr=(0, 0, 255))
# test.create_clips()
