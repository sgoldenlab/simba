__author__ = "Simon Nilsson"

import glob
import os
import subprocess
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_if_dir_exists, check_int, check_str,
                                check_valid_boolean)
from simba.utils.enums import Formats, Options, TextOptions
from simba.utils.errors import CountError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    read_frm_of_video)
from simba.video_processors.roi_selector import ROISelector
from simba.video_processors.video_processing import crop_video


class MultiCropper(object):
    """
    Crop each video of a specific file format (e.g., mp4) in a directory into N smaller cropped videos.


    .. note::
       `Multi-crop tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#multi-crop-videos>`__.

       `Expected GPU timesavings <https://github.com/sgoldenlab/simba/blob/master/docs/gpu_vs_cpu_video_processing_runtimes.md>`__.

    .. image:: _static/img/multicrop.gif
       :width: 700
       :align: center

    :param Union[str, os.PathLike] input_folder: Folder path holding videos to be cropped.
    :param Literal['avi', 'mp4', 'mov', 'flv', 'm4v'] file_type: File type of input video files inside the ``input_folder`` directory.
    :param Union[str, os.PathLike] output_folder: Directory where to store the cropped videos.
    :param int crop_cnt: The number of cropped videos to create from each input video. Minimum: 2.
    :param bool gpu: If True, use GPU codecs, else CPU. Default CPU.


    :example:
    >>> cropper = MultiCropper(file_type='mp4', input_folder=r'C:\troubleshooting\mitra\test', output_folder=r'C:\troubleshooting\mitra\test\cropped', crop_cnt=2, gpu=True)
    >>> cropper.run()
    """

    def __init__(self,
                 file_type: Literal["avi", "mp4", "mov", "flv", "m4v"],
                 input_folder: Union[str, os.PathLike],
                 output_folder: Union[str, os.PathLike],
                 crop_cnt: int,
                 gpu: bool = False,
                 quality: int = 60):

        check_str(name="FILE TYPE", value=file_type.lower(), options=Options.ALL_VIDEO_FORMAT_OPTIONS_2.value)
        check_if_dir_exists(in_dir=input_folder)
        check_if_dir_exists(in_dir=output_folder)
        check_int(name=f'{self.__class__.__name__} crop_cnt', value=crop_cnt, min_value=2)
        check_int(name=f'{crop_video.__name__} quality', value=quality, min_value=1, max_value=100)
        check_valid_boolean(value=[gpu], source=f'{self.__class__.__name__} gpu')
        self.file_type, self.crop_cnt, self.gpu, self.quality = file_type.lower(), crop_cnt, gpu, quality
        self.input_folder, self.output_folder = input_folder, output_folder
        self.video_paths = find_files_of_filetypes_in_directory(directory=self.input_folder, extensions=[f'.{file_type}'], raise_error=True)
        self.font = Formats.FONT.value
        self.crop_df = pd.DataFrame(columns=["video", "height", "width", "top_left_x", "top_left_y"])

    def __test_crop_locations(self):
        for idx, row in self.crop_df.iterrows():
            lst = [row["height"], row["width"], row["top_left_x"], row["top_left_y"]]
            video_name = row["video"]
            if all(v == 0 for v in lst):
                raise CountError(msg=f"SIMBA ERROR: A crop for video {video_name} has all crop coordinates set to zero. Did you click ESC, space or enter before defining the rectangle crop coordinates?!", source=self.__class__.__name__)
            else:
                pass

    def run(self):
        for file_cnt, video_path in enumerate(self.video_paths):
            video_meta_data = get_video_meta_data(video_path=video_path)
            _, video_name, _ = get_fn_ext(filepath=video_path)
            font_size, x_space, y_space = PlottingMixin().get_optimal_font_scales(text=f"Define the ROI boundaries of {self.crop_cnt} cropped videos", accepted_px_width=video_meta_data['width'], accepted_px_height=int(video_meta_data['height']/10), text_thickness=3, font=self.font)
            img = read_frm_of_video(video_path=video_path, frame_index=0)
            original_img = np.copy(img)
            txt_pos = (10, 20)
            img = PlottingMixin().put_text(img=img, text=video_name, pos=txt_pos, font_size=font_size, font_thickness=2, font=self.font, text_color=TextOptions.COLOR.value, text_color_bg=(0, 0, 0))
            img = PlottingMixin().put_text(img=img, text=f"Define the ROI boundaries of {self.crop_cnt} cropped videos", pos=(txt_pos[0], txt_pos[1] + (y_space*1)), font_size=font_size, font_thickness=2, font=self.font, text_color=TextOptions.COLOR.value, text_color_bg=(0, 0, 0))
            img = PlottingMixin().put_text(img=img, text=f"Press ESC to continue", pos=(txt_pos[0], txt_pos[1] + (y_space*2)), font_size=font_size, font_thickness=2, font=self.font, text_color=(0, 0, 255), text_color_bg=(0, 0, 0), text_bg_alpha=1.0)
            cv2.namedWindow("VIDEO IMAGE", cv2.WINDOW_NORMAL)
            cv2.imshow("VIDEO IMAGE", img)
            while True:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break

            for box_cnt in range(self.crop_cnt):
                img = PlottingMixin().put_text(img=original_img, text=video_name, pos=txt_pos, font_size=font_size, font_thickness=2, font=self.font, text_color=(144, 238, 144), text_color_bg=(0, 0, 0))
                img = PlottingMixin().put_text(img=img, text=f"Draw crop box #{box_cnt+1} boundaries and press ESC", pos=(txt_pos[0], txt_pos[1] + (y_space * 1)), font_size=font_size, font_thickness=2, font=self.font, text_color=(144, 238, 144), text_color_bg=(0, 0, 0))
                roi_selector = ROISelector(path=img, title=f"VIDEO IMAGE")
                roi_selector.run()
                cv2.destroyAllWindows()
                if roi_selector.complete:
                    self.crop_df.loc[len(self.crop_df)] = [video_path, roi_selector.height, roi_selector.width, roi_selector.top_left[0], roi_selector.top_left[1]]
                else:
                    break

        self.__test_crop_locations()
        cv2.destroyAllWindows()
        cv2.waitKey(50)

        timer = SimbaTimer(start=True)
        crop_counter = 0
        print("Starting video cropping...")
        for video_path in self.crop_df["video"].unique():
            _, video_name, video_ext = get_fn_ext(filepath=video_path)
            video_crops = self.crop_df[self.crop_df["video"] == video_path].reset_index(drop=True)
            for cnt, (idx, row) in enumerate(video_crops.iterrows()):
                print(f"Creating {video_name} crop clip {cnt+1} (crop {crop_counter+1}/{len(self.crop_df)})...")
                crop_timer = SimbaTimer(start=True)
                height, width, top_left_x, top_left_y = row["height"], row["width"], row["top_left_x"], row["top_left_y"]
                save_path = os.path.join(self.output_folder, f"{video_name}_{cnt+1}.{self.file_type}")
                crop_video(video_path=video_path, save_path=save_path, size=(width, height), top_left=(top_left_x, top_left_y), gpu=self.gpu, verbose=False, quality=self.quality)
                crop_timer.stop_timer()
                print(f"Video {video_name} crop {cnt+1} complete (elapsed time: {crop_timer.elapsed_time_str})...")
                crop_counter += 1
        timer.stop_timer()
        stdout_success(msg=f"{str(len(self.crop_df))} new cropped videos created from {len(self.video_paths)} input videos. Cropped videos are saved in the {self.output_folder} directory", elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


# cropper = MultiCropper(file_type='mp4', input_folder=r'C:\troubleshooting\mitra\test', output_folder=r'C:\troubleshooting\mitra\test\cropped', crop_cnt=2, gpu=True)
# cropper.run()


# cropper = MultiCropper(file_type='mp4', input_folder='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos', output_folder='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/edited', crop_cnt=2)
# cropper.run()
