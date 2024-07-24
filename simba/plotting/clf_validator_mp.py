__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_float, check_if_valid_rgb_tuple,
                                check_int, check_str, check_that_column_exist,
                                check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats, TagNames, TextOptions
from simba.utils.errors import NoFilesFoundError, NoSpecifiedOutputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df, remove_files)
from simba.utils.warnings import NoDataFoundWarning

SPACE_SCALE = 60
RADIUS_SCALE = 12
RESOLUTION_SCALE = 1500
FONT_SCALE = 1.5


def val_clip_createror_mp(data: np.ndarray,
                          video_path: str,
                          scalers: dict,
                          bout_total_cnt: int,
                          fps: float,
                          video_meta_data: dict,
                          clf_name: str,
                          p_data: pd.Series,
                          clf_data: pd.Series,
                          highlight_clr: tuple,
                          text_clr: tuple):

    def _put_text(img: np.ndarray,
                  text: str,
                  pos: Tuple[int, int],
                  font_size: int,
                  font_thickness: Optional[int] = 2,
                  font: Optional[int] = cv2.FONT_HERSHEY_DUPLEX,
                  text_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
                  text_color_bg: Optional[Tuple[int, int, int]] = (0, 0, 0),
                  text_bg_alpha: float = 0.8):

        x, y = pos
        text_size, px_buffer = cv2.getTextSize(text, font, font_size, font_thickness)
        w, h = text_size
        overlay, output = img.copy(), img.copy()
        cv2.rectangle(overlay, (x, y-h), (x + w, y + px_buffer), text_color_bg, -1)
        cv2.addWeighted(overlay, text_bg_alpha, output, 1 - text_bg_alpha, 0, output)
        cv2.putText(output, text, (x, y), font, font_size, text_color, font_thickness)
        return output



    def __insert_inter_frms(bg_color: Tuple[int, int, int] = (49, 32, 189), fg_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Helper to create N blank frames separating the classified event bouts with defined BGR colors.
        """
        for i in range(int(fps)):
            inter_frm = np.full((int(video_meta_data["height"]), int(video_meta_data["width"]), 3), bg_color).astype(np.uint8)
            inter_frm = _put_text(img=inter_frm, text=f"Bout #{bount_cnt}", pos=(TextOptions.BORDER_BUFFER_X.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"]), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value, text_color=fg_color, text_bg_alpha=0.0)
            #cv2.putText(inter_frm, f"Bout #{bount_cnt}", (10, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"]), TextOptions.FONT.value, scalers["font"], fg_color, TextOptions.TEXT_THICKNESS.value)
            writer.write(inter_frm)

    SPACER = 2
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    start_frm, end_frame, save_path, c_frm, bount_cnt = (int(data[1]), int(data[2]), data[3], int(data[1]), int(data[0]))
    bout_frm_cnt = end_frame - start_frm
    writer = cv2.VideoWriter(save_path, fourcc, fps, (int(video_meta_data["width"]), int(video_meta_data["height"])))
    __insert_inter_frms()
    frm_cnt = 0
    while c_frm < end_frame:
        p, clf_val = round(float(p_data.loc[c_frm]), 3), int(clf_data.loc[c_frm])
        cap.set(1, c_frm)
        ret, img = cap.read()
        img = _put_text(img=img, text=f"{clf_name} event # {bount_cnt}", pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_color=text_clr)
        #cv2.putText(img, f"{clf_name} event # {bount_cnt}", (TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), TextOptions.FONT.value, scalers["font"], text_clr, TextOptions.TEXT_THICKNESS.value + 1)
        SPACER += 1
        img = _put_text(img=img, text=f"Total frames of event: {end_frame-start_frm+1}", pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_color=text_clr)
        #cv2.putText(img, f"Total frames of event: {end_frame-start_frm+1}", (TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), TextOptions.FONT.value, scalers["font"], text_clr, TextOptions.TEXT_THICKNESS.value + 1)
        SPACER += 1
        img = _put_text(img=img, text=f"Frames of event {start_frm} to {end_frame}", pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_color=text_clr)
        #cv2.putText(img, f"Frames of event {start_frm} to {end_frame}", (     TextOptions.BORDER_BUFFER_Y.value,     (video_meta_data["height"] - video_meta_data["height"])     + scalers["space"] * SPACER, ), TextOptions.FONT.value, scalers["font"], text_clr, TextOptions.TEXT_THICKNESS.value + 1)
        SPACER += 1
        img = _put_text(img=img, text=f"Frame number: {c_frm}", pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_color=text_clr)
        #cv2.putText( img, f"Frame number: {c_frm}", (     TextOptions.BORDER_BUFFER_Y.value,     (video_meta_data["height"] - video_meta_data["height"])     + scalers["space"] * SPACER, ), TextOptions.FONT.value, scalers["font"], text_clr, TextOptions.TEXT_THICKNESS.value + 1)
        SPACER += 1
        if (highlight_clr != None) and (clf_val == 1):
            img = _put_text(img=img, text=f"Frame {clf_name} probability: {p}", pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_color=highlight_clr)
            #cv2.putText(img,f"Frame {clf_name} probability: {p}",(    TextOptions.BORDER_BUFFER_Y.value,    (video_meta_data["height"] - video_meta_data["height"])    + scalers["space"] * SPACER,),TextOptions.FONT.value,scalers["font"],highlight_clr,TextOptions.TEXT_THICKNESS.value + 1)
        else:
            img = _put_text(img=img, text=f"Frame {clf_name} probability: {p}", pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + scalers["space"] * SPACER), font_size=scalers["font"], font_thickness=TextOptions.TEXT_THICKNESS.value + 1, text_color=text_clr)
            #cv2.putText(img,f"Frame {clf_name} probability: {p}",(    TextOptions.BORDER_BUFFER_Y.value,    (video_meta_data["height"] - video_meta_data["height"])    + scalers["space"] * SPACER,),TextOptions.FONT.value,scalers["font"],text_clr,TextOptions.TEXT_THICKNESS.value + 1)
        writer.write(img)
        c_frm += 1
        SPACER = 2
        frm_cnt += 1
        print(f'Multiprocessing frame {frm_cnt}/{bout_frm_cnt} (Bout: {bount_cnt+1}/{bout_total_cnt}, Video: {video_meta_data["video_name"]})')
    writer.release()
    cap.release()


class ClassifierValidationClipsMultiprocess(ConfigReader):
    """
    Create video clips with overlaid classified events for detection of false positive event bouts using multiple cores for improved runtime.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter int window: Number of seconds before and after the event bout that should be included in the output video.
    :parameter str clf_name: Name of the classifier to create validation videos for.
    :parameter bool clips: If True, creates individual video file clips for each validation bout.
    :parameter List[Union[str, os.PathLike]] data_paths: List of files with classification results to create videos for.
    :parameter Tuple[int, int, int] text_clr: Color of text overlay in BGR.
    :parameter Optional[Tuple[int, int, int]] highlight_clr: Color of text when probability values are above threshold. If None, same as text_clr.
    :parameter float video_speed:  FPS rate in relation to original video. E.g., the same as original video if 1.0. If output should be half the speed relative to input, set to 0.5. Default: 1.0.
    :parameter bool concat_video:  If True, creates a single video including all events bouts for each video. Default: False.
    :parameter Optional[int] core_cnt: Integer dictating the numbers of cores to use. If -1, all available cores are used.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md#classifier-validation>`_.

    Examples
    ----------
    >>> _ = ClassifierValidationClipsMultiprocess(config_path='MyProjectConfigPath', window=5, clf_name='Attack', text_clr=(255,255,0), clips=False, concat_video=True).run()
    """

    def __init__(self,
                 config_path: str,
                 window: int,
                 clf_name: str,
                 clips: bool,
                 data_paths: List[Union[str, os.PathLike]],
                 text_clr: Tuple[int, int, int],
                 concat_video: bool = False,
                 video_speed: float = 1.0,
                 highlight_clr: Optional[Tuple[int, int, int]] = None,
                 core_cnt: Optional[int] = -1):


        ConfigReader.__init__(self, config_path=config_path)
        if (not clips) and (not concat_video):
            raise NoSpecifiedOutputError(msg="Please select to create clips and/or a concatenated video", source=self.__class__.__name__)

        check_int(name="Time window", value=window, min_value=0)
        check_if_valid_rgb_tuple(data=text_clr)
        if highlight_clr is not None: check_if_valid_rgb_tuple(data=highlight_clr)
        check_valid_lst(data=data_paths, source=f'{self.__class__.__name__} data_paths', min_len=1)
        check_str(name=f'{self.__class__.__name__} clf_name', value=clf_name, options=self.clf_names)
        check_float(name=f'{self.__class__.__name__} video_speed', value=video_speed, min_value=10e-6)
        check_int(name="CORE COUNT",value=core_cnt,min_value=-1,max_value=find_core_cnt()[0],raise_error=True,)
        if core_cnt == -1:core_cnt = find_core_cnt()[0]
        self.core_cnt = core_cnt
        self.window, self.clf_name = int(window), clf_name
        self.clips, self.concat_video, self.video_speed, self.highlight_clr = clips, concat_video, video_speed, highlight_clr
        self.p_col = f"Probability_{self.clf_name}"
        self.text_clr, self.data_paths = text_clr, data_paths
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.font = TextOptions.FONT.value
        if not os.path.exists(self.clf_validation_dir):
            os.makedirs(self.clf_validation_dir)
        print(f"Processing {len(self.data_paths)} files...")

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            self.data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=self.data_df, column_name=self.p_col, file_name=file_path)
            _, file_name, _ = get_fn_ext(file_path)
            self.video_path = self.find_video_of_file(video_dir=self.video_dir, filename=file_name, raise_error=True)
            self.video_info = get_video_meta_data(video_path=self.video_path)
            self.fps = int(self.video_info["fps"])
            self.video_fps = int(self.fps * self.video_speed)
            if self.video_fps < 1: self.video_fps = 1

            self.font_size, x_scaler, self.spacing_scale = PlottingMixin().get_optimal_font_scales(text="Total frames of event: '999999'", accepted_px_width=int(self.video_info["width"] / 2), accepted_px_height=int(self.video_info["height"] / 5), text_thickness=TextOptions.TEXT_THICKNESS.value)
            clf_bouts = detect_bouts(data_df=self.data_df, target_lst=[self.clf_name], fps=self.fps).reset_index(drop=True)
            if len(clf_bouts) == 0:
                NoDataFoundWarning(msg=f"Skipping video {file_name}: No classified behavior {self.clf_name} detected...", source=self.__class__.__name__)
                continue

            clip_data = []
            for i, (bout_cnt, bout) in enumerate(clf_bouts.iterrows()):
                self.bout_cnt = bout_cnt
                event_start_frm, event_end_frm = bout["Start_frame"], bout["End_frame"]
                start_window = int(event_start_frm - (int(self.video_info["fps"]) * self.window))
                end_window = int(event_end_frm + (int(self.video_info["fps"]) * self.window))
                if end_window > len(self.data_df):
                    end_window = len(self.data_df)
                if start_window < 0:
                    start_window = 0
                self.save_path = os.path.join(self.clf_validation_dir, self.clf_name + f"_{bout_cnt}_{file_name}.mp4")
                clip_data.append([bout_cnt, start_window, end_window, self.save_path])
            clip_data = np.array(clip_data)
            print(f"Creating validation video, multiprocessing (chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})...")
            with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                constants = functools.partial(val_clip_createror_mp,
                                              video_path=self.video_path,
                                              scalers={"font": self.font_size, "space": self.spacing_scale},
                                              fps=self.video_fps,
                                              clf_name=self.clf_name,
                                              bout_total_cnt=len(clf_bouts),
                                              video_meta_data=self.video_info,
                                              p_data=self.data_df[self.p_col].astype(np.float32),
                                              clf_data=self.data_df[self.clf_name].astype(np.float32),
                                              highlight_clr=self.highlight_clr,
                                              text_clr=self.text_clr)

                for cnt, result in enumerate(
                    pool.imap(constants, clip_data, chunksize=self.multiprocess_chunksize)):
                    print(f"Bout {cnt+1} complete...")
                pool.terminate()
                pool.join()

            if self.concat_video:
                print(f"Joining {file_name} multiprocessed video...")
                concat_video_save_path = os.path.join(self.clf_validation_dir, f"{self.clf_name}_{file_name}_all_events.mp4")
                file_paths = list(clip_data[:, 3])
                concatenate_videos_in_folder(in_folder=self.clf_validation_dir,
                                             save_path=concat_video_save_path,
                                             substring=None,
                                             file_paths=file_paths,
                                             remove_splits=False)
            if not self.clips:
                remove_files(file_paths=list(clip_data[:, 3]))
            video_timer.stop_timer()
            stdout_success(msg=f"Validation clips for video {file_name} complete!", elapsed_time=video_timer.elapsed_time_str)

        self.timer.stop_timer()
        stdout_success(msg=f"All video clips complete and saved in {self.clf_validation_dir}!", elapsed_time=self.timer.elapsed_time_str)


# test = ClassifierValidationClipsMultiprocess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                              window=1,
#                                              clf_name='Attack',
#                                              clips=True,
#                                              concat_video=True,
#                                              highlight_clr=(255, 0, 0),
#                                              video_speed=0.5,
#                                              text_clr=(0, 0, 255),
#                                              data_paths=['/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.run()

# test = ClassifierValidationClips(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini',
#                                  window=1,
#                                  clf_name='Attack',
#                                  clips=False,
#                                  concat_video=True,
#                                  text_clr=(0, 0, 255))
# test.create_clips()
