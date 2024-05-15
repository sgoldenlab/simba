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

from copy import deepcopy

from simba.utils.checks import (check_if_filepath_list_is_empty, check_int,
                                check_str)
from simba.utils.enums import Formats, Options, TextOptions
from simba.utils.errors import (CountError, InvalidFileTypeError,
                                InvalidVideoFileError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data
from simba.video_processors.roi_selector import ROISelector


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

    """

    def __init__(
        self,
        file_type: Literal["avi", "mp4", "mov", "flv", "m4v"],
        input_folder: Union[str, os.PathLike],
        output_folder: Union[str, os.PathLike],
        crop_cnt: int,
        gpu: Optional[bool] = False,
    ):
        self.file_type, self.crop_cnt, self.gpu = file_type, crop_cnt, gpu
        self.input_folder, self.output_folder = input_folder, output_folder
        self.crop_df = pd.DataFrame(
            columns=["Video", "height", "width", "topLeftX", "topLeftY"]
        )
        check_int(name="CROP COUNT", value=self.crop_cnt, min_value=2)
        self.crop_cnt = int(crop_cnt)
        check_str(name="FILE TYPE", value=self.file_type)
        if self.file_type.__contains__("."):
            self.file_type.replace(".", "")
        if f".{file_type}" not in Options.ALL_VIDEO_FORMAT_OPTIONS.value:
            raise InvalidFileTypeError(
                msg=f"The filetype .{file_type} is invalid. Options: {Options.ALL_VIDEO_FORMAT_OPTIONS.value}"
            )
        self.files_found = glob.glob(self.input_folder + f"/*.{self.file_type}")
        check_if_filepath_list_is_empty(
            filepaths=self.files_found,
            error_msg=f"SIMBA CROP ERROR: The input direct {self.input_folder} contains ZERO videos in {self.file_type} format",
        )
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.add_spacer = TextOptions.FIRST_LINE_SPACING.value
        self.font_thickness = TextOptions.TEXT_THICKNESS.value + 1

    def __test_crop_locations(self):
        for idx, row in self.crop_df.iterrows():
            lst = [row["height"], row["width"], row["topLeftX"], row["topLeftY"]]
            video_name = row["Video"]
            if all(v == 0 for v in lst):
                raise CountError(
                    msg=f"SIMBA ERROR: A crop for video {video_name} has all crop coordinates set to zero. Did you click ESC, space or enter before defining the rectangle crop coordinates?!",
                    source=self.__class__.__name__,
                )
            else:
                pass

    def draw_txt_w_bg(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        font_scale: float,
        font_thickness: float,
        text_color: Tuple[int, int, int],
        text_color_bg: Tuple[int, int, int],
        font=Formats.FONT.value,
    ):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w + 10, y + text_h + 10), text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, y + text_h + font_scale - 1),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

        return img

    def run(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_meta_data = get_video_meta_data(video_path=file_path)
            print(video_meta_data)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            self.font_scale, self.space_scale, self.font = 0.02, 1.1, Formats.FONT.value
            cap = cv2.VideoCapture(file_path)
            cap.set(1, 0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise InvalidVideoFileError(
                    msg=f"The first frame of video {video_name} could not be read",
                    source=self.__class__.__name__,
                )
            original_frame = deepcopy(frame)
            (height, width) = frame.shape[:2]
            font_size = int(np.ceil(min(width, height) / (25 / self.font_scale)))
            space_size = int(np.ceil(int(min(width, height) / (25 / self.space_scale))))

            frame = self.draw_txt_w_bg(
                img=frame,
                text=video_name,
                pos=(
                    TextOptions.BORDER_BUFFER_X.value,
                    int((height - height) + space_size),
                ),
                font_scale=font_size,
                font_thickness=TextOptions.TEXT_THICKNESS.value,
                text_color=TextOptions.COLOR.value,
                text_color_bg=(0, 0, 0),
            )
            frame = self.draw_txt_w_bg(
                img=frame,
                text=f"Define the rectangle boundaries of {self.crop_cnt} cropped videos",
                pos=(
                    TextOptions.BORDER_BUFFER_X.value,
                    int((height - height) + space_size * self.add_spacer),
                ),
                font_scale=font_size,
                font_thickness=TextOptions.TEXT_THICKNESS.value,
                text_color=TextOptions.COLOR.value,
                text_color_bg=(0, 0, 0),
            )
            frame = self.draw_txt_w_bg(
                img=frame,
                text=f"Press ESC to continue",
                pos=(
                    TextOptions.BORDER_BUFFER_X.value,
                    int((height - height) + space_size * (self.add_spacer + 2)),
                ),
                font_scale=font_size,
                font_thickness=TextOptions.TEXT_THICKNESS.value,
                text_color=TextOptions.COLOR.value,
                text_color_bg=(0, 0, 0),
            )

            cv2.namedWindow("VIDEO IMAGE", cv2.WINDOW_NORMAL)
            cv2.imshow("VIDEO IMAGE", frame)
            while True:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break

            for box_cnt in range(self.crop_cnt):
                frame = deepcopy(original_frame)
                frame = self.draw_txt_w_bg(
                    img=frame,
                    text=video_name,
                    pos=(
                        TextOptions.BORDER_BUFFER_X.value,
                        int((height - height) + space_size),
                    ),
                    font_scale=font_size,
                    font_thickness=TextOptions.TEXT_THICKNESS.value,
                    text_color=TextOptions.COLOR.value,
                    text_color_bg=(0, 0, 0),
                )
                frame = self.draw_txt_w_bg(
                    img=frame,
                    text=f"Draw crop #{box_cnt+1} boundaries and press ESC",
                    pos=(
                        TextOptions.BORDER_BUFFER_X.value,
                        int((height - height) + space_size * self.add_spacer),
                    ),
                    font_scale=font_size,
                    font_thickness=TextOptions.TEXT_THICKNESS.value,
                    text_color=TextOptions.COLOR.value,
                    text_color_bg=(0, 0, 0),
                )
                roi_selector = ROISelector(
                    path=frame,
                    title=f"MULTI-CROP {video_name} into {self.crop_cnt} videos - press ESC when happy with a rectangle",
                )
                roi_selector.run()
                cv2.destroyAllWindows()
                self.crop_df.loc[len(self.crop_df)] = [
                    video_name,
                    roi_selector.height,
                    roi_selector.width,
                    roi_selector.top_left[0],
                    roi_selector.top_left[1],
                ]

        self.__test_crop_locations()
        cv2.destroyAllWindows()
        cv2.waitKey(50)

        self.timer = SimbaTimer(start=True)
        print("Starting video cropping...")
        for video_name in self.crop_df["Video"].unique():
            video_crops = self.crop_df[self.crop_df["Video"] == video_name].reset_index(
                drop=True
            )
            in_video_path = os.path.join(
                self.input_folder, f"{video_name}.{self.file_type}"
            )
            for cnt, (idx, row) in enumerate(video_crops.iterrows()):
                print(f"Creating {video_name} crop clip {cnt+1}...")
                crop_timer = SimbaTimer(start=True)
                height, width = row["height"], row["width"]
                topLeftX, topLeftY = row["topLeftX"], row["topLeftY"]
                out_file_fn = os.path.join(
                    self.output_folder, video_name + f"_{cnt+1}.{self.file_type}"
                )
                if self.gpu:
                    command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{in_video_path}" -vf "crop={width}:{height}:{topLeftX}:{topLeftY}" -c:v h264_nvenc -c:a copy "{out_file_fn}" -hide_banner -loglevel error'
                else:
                    command = f'ffmpeg -y -i "{in_video_path}" -vf "crop={width}:{height}:{topLeftX}:{topLeftY}" -c:v libx264 -c:a copy "{out_file_fn}" -hide_banner -loglevel error'
                subprocess.call(command, shell=True)
                crop_timer.stop_timer()
                print(
                    f"Video {video_name} crop {cnt+1} complete (elapsed time: {crop_timer.elapsed_time_str})..."
                )
        self.timer.stop_timer()
        stdout_success(
            msg=f"{str(len(self.crop_df))} new cropped videos created from {str(len(self.files_found))} input videos. Cropped videos are saved in the {self.output_folder} directory",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# cropper = MultiCropper(file_type='mp4', input_folder='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos', output_folder='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/edited', crop_cnt=2)
# cropper.run()
