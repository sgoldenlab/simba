import os
import time
from tkinter import *
from typing import Union

import numpy as np
from shapely.geometry import Polygon

from simba.mixins.image_mixin import ImageMixin
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import FileSelect, FolderSelect
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats, Options
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    get_video_meta_data)
from simba.video_processors.roi_selector_circle import ROISelectorCircle


def crop_single_video_circle(file_path: Union[str, os.PathLike]) -> None:
    """
    Crop a video based on circular regions of interest (ROIs) selected by the user.

    :param  Union[str, os.PathLike] file_path: The path to the input video file.

    .. note::
       This function crops the input video based on circular regions of interest (ROIs) selected by the user.
       The user is prompted to select a circular ROI on the video frame, and the function then crops the video
       based on the selected ROI. The cropped video is saved with "_circle_cropped" suffix in the same directory
       as the input video file.

    :example:
    >>> crop_single_video_circle(file_path='/Users/simon/Desktop/AGGRESSIVITY_4_11_21_Trial_2_camera1_rotated_20240211143355.mp4')
    """

    dir, video_name, _ = get_fn_ext(filepath=file_path)
    save_path = os.path.join(dir, f"{video_name}_circle_cropped.mp4")
    video_meta_data = get_video_meta_data(video_path=file_path)
    check_file_exist_and_readable(file_path=file_path)
    circle_selector = ROISelectorCircle(path=file_path)
    circle_selector.run()
    timer = SimbaTimer(start=True)
    r = circle_selector.circle_radius
    x, y = circle_selector.circle_center[0], circle_selector.circle_center[1]
    polygon = Polygon(
        [
            (x + r * np.cos(angle), y + r * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 100)
        ]
    )
    polygons = [polygon for x in range(video_meta_data["frame_count"])]
    polygons = ImageMixin().slice_shapes_in_imgs(
        imgs=file_path, shapes=polygons, verbose=False
    )
    time.sleep(3)
    _ = ImageMixin.img_stack_to_video(
        imgs=polygons, save_path=save_path, fps=video_meta_data["fps"]
    )
    timer.stop_timer()
    stdout_success(
        msg=f"Circle-based cropped saved at to {save_path}",
        elapsed_time=timer.elapsed_time_str,
    )


def crop_multiple_videos_circles(
    in_dir: Union[str, os.PathLike], out_dir: Union[str, os.PathLike]
) -> None:
    """
    Crop multiple videos based on circular regions of interest (ROIs) selected by the user.

    :param  Union[str, os.PathLike] in_dir: The directory containing input video files.
    :param  Union[str, os.PathLike] out_dir: The directory to save the cropped video files.

    .. note::
       This function crops multiple videos based on circular ROIs selected by the user.
       The user is asked to define a circle manually in one video within the input directory.
       The function then crops all the video in the input directory according to the shape defined
       using the first video and saves the videos in the ``out_dir`` with the same filenames as the original videos..

    :example:
    >>> crop_multiple_videos_circles(in_dir='/Users/simon/Desktop/edited/tests', out_dir='/Users/simon/Desktop')
    """

    check_if_dir_exists(in_dir=in_dir)
    check_if_dir_exists(in_dir=out_dir)
    video_files = find_all_videos_in_directory(directory=in_dir)
    circle_selector = ROISelectorCircle(path=os.path.join(in_dir, video_files[0]))
    circle_selector.run()
    r = circle_selector.circle_radius
    x, y = circle_selector.circle_center[0], circle_selector.circle_center[1]
    polygon = Polygon(
        [
            (x + r * np.cos(angle), y + r * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 100)
        ]
    )
    timer = SimbaTimer(start=True)
    for video_cnt, video_path in enumerate(video_files):
        print(
            f"Circle cropping video {video_path} ({video_cnt+1}/{len(video_files)})..."
        )
        video_path = os.path.join(in_dir, video_path)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(out_dir, f"{video_name}.mp4")
        video_meta_data = get_video_meta_data(video_path=video_path)
        polygons = [polygon for x in range(video_meta_data["frame_count"])]
        polygons = ImageMixin().slice_shapes_in_imgs(
            imgs=video_path, shapes=polygons, verbose=False
        )
        time.sleep(1)
        _ = ImageMixin.img_stack_to_video(
            imgs=polygons, save_path=save_path, fps=video_meta_data["fps"]
        )
    timer.stop_timer()
    stdout_success(
        msg=f"Circle-based cropped {len(video_files)} files to directory {out_dir}",
        elapsed_time=timer.elapsed_time_str,
    )


class CropVideoCirclesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CROP SINGLE VIDEO (CIRCLES)")
        crop_video_lbl_frm = LabelFrame(
            self.main_frm,
            text="Crop Video (CIRCLES)",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        selected_video = FileSelect(
            crop_video_lbl_frm,
            "Video path",
            title="Select a video file",
            lblwidth=20,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        button_crop_video_single = Button(
            crop_video_lbl_frm,
            text="Crop Video",
            command=lambda: crop_single_video_circle(
                file_path=selected_video.file_path
            ),
        )
        crop_video_lbl_frm_multiple = LabelFrame(
            self.main_frm,
            text="Fixed CIRCLE coordinates crop for multiple videos",
            font="bold",
            padx=5,
            pady=5,
        )
        input_folder = FolderSelect(
            crop_video_lbl_frm_multiple,
            "Video directory:",
            title="Select Folder with videos",
            lblwidth=20,
        )
        output_folder = FolderSelect(
            crop_video_lbl_frm_multiple,
            "Output directory:",
            title="Select a folder for your output videos",
            lblwidth=20,
        )
        button_crop_video_multiple = Button(
            crop_video_lbl_frm_multiple,
            text="Crop Videos",
            command=lambda: crop_multiple_videos_circles(
                in_dir=input_folder.folder_path, out_dir=output_folder.folder_path
            ),
        )

        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        button_crop_video_single.grid(row=3, sticky=NW)
        crop_video_lbl_frm_multiple.grid(row=1, sticky=NW)
        input_folder.grid(row=0, sticky=NW)
        output_folder.grid(row=1, sticky=NW)
        button_crop_video_multiple.grid(row=3, sticky=NW)
        self.main_frm.mainloop()


#
_ = CropVideoCirclesPopUp()
