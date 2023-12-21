__author__ = "Simon Nilsson"


import glob
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tkinter import *
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageTk

import simba
from simba.mixins.config_reader import ConfigReader

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import OS, ConfigKey, Formats, Options, Paths
from simba.utils.errors import (CountError, DirectoryExistError,
                                FFMPEGCodecGPUError, FFMPEGNotFoundError,
                                FileExistError, InvalidFileTypeError,
                                InvalidInputError, NoDataError,
                                NoFilesFoundError, NotDirectoryError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video,
    find_all_videos_in_directory, find_core_cnt, get_fn_ext,
    get_video_meta_data, read_config_entry, read_config_file)
from simba.utils.warnings import (FileExistWarning, InValidUserInputWarning,
                                  SameInputAndOutputWarning)
from simba.video_processors.extract_frames import video_to_frames
from simba.video_processors.roi_selector import ROISelector

MAX_FRM_SIZE = 1080, 650


def change_img_format(
    directory: Union[str, os.PathLike], file_type_in: str, file_type_out: str
) -> None:
    """
    Convert the file type of all image files within a directory.

    :parameter Union[str, os.PathLike] directory: Path to directory holding image files
    :parameter str file_type_in: Input file type, e.g., 'bmp' or 'png.
    :parameter str file_type_out: Output file type, e.g., 'bmp' or 'png.

    :example:
    >>> _ = change_img_format(directory='MyDirectoryWImages', file_type_in='bmp', file_type_out='png')

    """
    if not os.path.isdir(directory):
        raise NotDirectoryError(
            "SIMBA ERROR: {} is not a valid directory".format(directory),
            source=change_img_format.__name__,
        )
    files_found = glob.glob(directory + "/*.{}".format(file_type_in))
    if len(files_found) < 1:
        raise NoFilesFoundError(
            "SIMBA ERROR: No {} files (with .{} file ending) found in the {} directory".format(
                file_type_in, file_type_in, directory
            ),
            source=change_img_format.__name__,
        )
    print("{} image files found in {}...".format(str(len(files_found)), directory))
    for file_path in files_found:
        im = Image.open(file_path)
        save_name = file_path.replace("." + str(file_type_in), "." + str(file_type_out))
        im.save(save_name)
        os.remove(file_path)
    stdout_success(
        msg=f"SIMBA COMPLETE: Files in {directory} directory converted to {file_type_out}",
        source=change_img_format.__name__,
    )


def clahe_enhance_video(file_path: Union[str, os.PathLike]) -> None:
    """
    Convert a single video file to clahe-enhanced greyscale .avi file. The result is saved with prefix
    ``CLAHE_`` in the same directory as in the input file.

    :parameter Union[str, os.PathLike] file_path: Path to video file.

    :example:
    >>> _ = clahe_enhance_video(file_path: 'project_folder/videos/Video_1.mp4')
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    save_path = os.path.join(dir, "CLAHE_{}.avi".format(file_name))
    video_meta_data = get_video_meta_data(file_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
    print("Applying CLAHE on video {}, this might take awhile...".format(file_name))
    cap = cv2.VideoCapture(file_path)
    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        video_meta_data["fps"],
        (video_meta_data["height"], video_meta_data["width"]),
        0,
    )
    clahe_filter = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
    try:
        frm_cnt = 0
        while True:
            ret, img = cap.read()
            if ret:
                frm_cnt += 1
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe_frm = clahe_filter.apply(img)
                writer.write(clahe_frm)
                print(
                    "CLAHE converted frame {}/{}".format(
                        str(frm_cnt), str(video_meta_data["frame_count"])
                    )
                )
            else:
                cap.release()
                writer.release()
    except Exception as se:
        print(se.args)
        print("CLAHE conversion failed for video {}".format(file_name))
        cap.release()
        writer.release()
        raise ValueError()


def extract_frame_range(
    file_path: Union[str, os.PathLike], start_frame: int, end_frame: int
) -> None:
    """
    Extract a user-defined range of frames from a video file to `png` format. Images
    are saved in a folder with the suffix `_frames` within the same directory as the video file.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int start_frame: First frame in range to extract
    :parameter int end_frame: Last frame in range to extract.

    :example:
    >>> _ = extract_frame_range(file_path='project_folder/videos/Video_1.mp4', start_frame=100, end_frame=500)
    """

    check_file_exist_and_readable(file_path=file_path)
    video_meta_data = get_video_meta_data(file_path)
    check_int(name="start frame", value=start_frame, min_value=0)
    file_dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    check_int(
        name="end frame", value=end_frame, max_value=video_meta_data["frame_count"]
    )
    frame_range = list(range(int(start_frame), int(end_frame) + 1))
    save_dir = os.path.join(file_dir, file_name + "_frames")
    cap = cv2.VideoCapture(file_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for frm_cnt, frm_number in enumerate(frame_range):
        cap.set(1, frm_number)
        ret, frame = cap.read()
        frm_save_path = os.path.join(save_dir, "{}.{}".format(str(frm_number), "png"))
        cv2.imwrite(frm_save_path, frame)
        print(
            "Frame {} saved (Frame {}/{})".format(
                str(frm_number), str(frm_cnt), str(len(frame_range))
            )
        )
    stdout_success(
        msg=f"{str(len(frame_range))} frames extracted for video {file_name}",
        source=extract_frame_range.__name__,
    )


def change_single_video_fps(
    file_path: Union[str, os.PathLike], fps: int, gpu: Optional[bool] = False
) -> None:
    """
    Change the fps of a single video file. Results are stored in the same directory as in the input file with
    the suffix ``_fps_new_fps``.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int fps: Fps of the new video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = change_single_video_fps(file_path='project_folder/videos/Video_1.mp4', fps=15)
    """

    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=change_single_video_fps.__name__,
        )
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="New fps", value=fps)
    video_meta_data = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if int(fps) == int(video_meta_data["fps"]):
        SameInputAndOutputWarning(
            msg=f"The new fps is the same as the input fps for video {file_name} ({str(fps)})",
            source=change_single_video_fps.__name__,
        )
    save_path = os.path.join(
        dir_name, file_name + "_fps_{}{}".format(str(fps), str(ext))
    )
    if os.path.isfile(save_path):
        FileExistWarning(
            msg=f"Overwriting existing file at {save_path}...",
            source=change_single_video_fps.__name__,
        )
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "fps={fps}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
    else:
        command = (
            str("ffmpeg -i ")
            + '"'
            + str(file_path)
            + '"'
            + " -filter:v fps=fps="
            + str(fps)
            + " "
            + '"'
            + save_path
            + '" -y'
        )
    subprocess.call(command, shell=True)
    timer.stop_timer()
    stdout_success(
        msg=f'SIMBA COMPLETE: FPS of video {file_name} changed from {str(video_meta_data["fps"])} to {str(fps)} and saved in directory {save_path}',
        elapsed_time=timer.elapsed_time_str,
        source=change_single_video_fps.__name__,
    )


def change_fps_of_multiple_videos(
    directory: Union[str, os.PathLike], fps: int, gpu: Optional[bool] = False
) -> None:
    """
    Change the fps of all video files in a folder. Results are stored in the same directory as in the input files with
    the suffix ``_fps_new_fps``.

    :parameter Union[str, os.PathLike] directory: Path to video file directory
    :parameter int fps: Fps of the new video files.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = change_fps_of_multiple_videos(directory='project_folder/videos/Video_1.mp4', fps=15)
    """

    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=change_fps_of_multiple_videos.__name__,
        )
    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory),
            source=change_fps_of_multiple_videos.__name__,
        )
    check_int(name="New fps", value=fps)
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + "/*") if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in [".avi", ".mp4", ".mov", ".flv"]:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory".format(
                directory
            ),
            source=change_fps_of_multiple_videos.__name__,
        )
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print("Converting FPS for {}...".format(file_name))
        save_path = os.path.join(
            dir_name, file_name + "_fps_{}{}".format(str(fps), str(ext))
        )
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "fps={fps}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
        else:
            command = (
                str("ffmpeg -i ")
                + str(file_path)
                + " -filter:v fps=fps="
                + str(fps)
                + " "
                + '"'
                + save_path
                + '" -y'
            )
        subprocess.call(command, shell=True)
        print("Video {} complete...".format(file_name))
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: FPS of {str(len(video_paths))} video(s) changed to {str(fps)}",
        elapsed_time=timer.elapsed_time_str,
        source=change_fps_of_multiple_videos.__name__,
    )


def convert_video_powerpoint_compatible_format(
    file_path: Union[str, os.PathLike], gpu: Optional[bool] = False
) -> None:
    """
    Create MS PowerPoint compatible copy of a video file. The result is stored in the same directory as the
    input file with the ``_powerpointready`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = convert_video_powerpoint_compatible_format(file_path='project_folder/videos/Video_1.mp4')
    """
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=convert_video_powerpoint_compatible_format.__name__,
        )
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_powerpointready.mp4")
    if os.path.isfile(save_name):
        raise FileExistError(
            msg="SIMBA ERROR: The outfile file already exist: {}.".format(save_name),
            source=convert_video_powerpoint_compatible_format.__name__,
        )
    if gpu:
        command = 'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{}" -c:v h264_nvenc -preset slow -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -c:a aac "{}"'.format(
            file_path, save_name
        )
    else:
        command = (
            str("ffmpeg -i ")
            + '"'
            + str(file_path)
            + '"'
            + " -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "
            + '"'
            + save_name
            + '"'
        )
    print("Creating video in powerpoint compatible format... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=convert_video_powerpoint_compatible_format.__name__,
    )


# convert_video_powerpoint_compatible_format(file_path=r"C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4", gpu=True)


def convert_to_mp4(
    file_path: Union[str, os.PathLike], gpu: Optional[bool] = False
) -> None:
    """
    Convert a video file to mp4 format. The result is stored in the same directory as the
    input file with the ``_converted.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = convert_to_mp4(file_path='project_folder/videos/Video_1.avi')
    """

    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=convert_to_mp4.__name__,
        )
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_converted.mp4")
    if os.path.isfile(save_name):
        raise FileExistError(
            msg="SIMBA ERROR: The outfile file already exist: {}.".format(save_name),
            source=convert_to_mp4.__name__,
        )
    if gpu:
        command = (
            'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{}" -c:v h264_nvenc "{}"'.format(
                file_path, save_name
            )
        )
    else:
        command = (
            str("ffmpeg -i ") + '"' + str(file_path) + '"' + " " + '"' + save_name + '"'
        )
    print("Converting to mp4... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=convert_to_mp4.__name__,
    )


# convert_to_mp4(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', gpu=True)


def video_to_greyscale(
    file_path: Union[str, os.PathLike], gpu: Optional[bool] = False
) -> None:
    """
    Convert a video file to greyscale mp4 format. The result is stored in the same directory as the
    input file with the ``_grayscale.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :raise FFMPEGCodecGPUError: If no GPU is found and ``gpu == True``.
    :raise FileExistError: If video name with ``_grayscale`` suffix already exist.

    :example:
    >>> _ = video_to_greyscale(file_path='project_folder/videos/Video_1.avi')
    """
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=video_to_greyscale.__name__,
        )
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_grayscale.mp4")
    if os.path.isfile(save_name):
        raise FileExistError(
            msg="SIMBA ERROR: The outfile file already exist: {}.".format(save_name),
            source=video_to_greyscale.__name__,
        )
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "hwupload_cuda,hwdownload,format=nv12,format=gray" -c:v h264_nvenc -c:a copy "{save_name}"'
    else:
        command = (
            str("ffmpeg -i ")
            + '"'
            + str(file_path)
            + '"'
            + " -vf format=gray "
            + '"'
            + save_name
            + '"'
        )
    print(f"Converting {file_name} to greyscale... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=video_to_greyscale.__name__,
    )


# video_to_greyscale(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', gpu=True)


def superimpose_frame_count(
    file_path: Union[str, os.PathLike], gpu: Optional[bool] = False
) -> None:
    """
    Superimpose frame count on a video file. The result is stored in the same directory as the
    input file with the ``_frame_no.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = superimpose_frame_count(file_path='project_folder/videos/Video_1.avi')
    """
    simba_cw = os.path.dirname(simba.__file__)
    simba_font_path = os.path.join(simba_cw, "assets", "UbuntuMono-Regular.ttf")
    timer = SimbaTimer(start=True)
    if not os.path.isfile(simba_font_path):
        if platform.system() == OS.WINDOWS.value:
            simba_font_path = "C:/Windows/fonts/arial.ttf"
            if not os.path.isfile(simba_font_path):
                simba_font_path = os.listdir(r"C:/Windows/fonts")[0]
            simba_font_path = simba_font_path[2:].replace("\\", "/")
        elif (platform.system() == OS.MAC.value) or (
            platform.system() == OS.LINUX.value
        ):
            simba_font_path = "Arial.ttf"
    else:
        if platform.system() == OS.WINDOWS.value:
            simba_font_path = str(simba_font_path[2:].replace("\\", "/"))
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_frame_no.mp4")
    print(f"Superimposing frame numbers using font path {simba_font_path}...... ")
    try:
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile={simba_font_path}:text=%{{n}}:x=(w-tw)/2:y=h-th-10:fontcolor=white:box=1:boxcolor=white@0.5" -c:v h264_nvenc -c:a copy "{save_name}" -y'
        else:
            command = (
                str("ffmpeg -y -i ")
                + '"'
                + file_path
                + '"'
                + ' -vf "drawtext=fontfile='
                + simba_font_path
                + ": text='%{frame_num}': start_number=0: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5\" -c:a copy "
                + '"'
                + save_name
                + '" -y'
            )
        print(f"Using font path {simba_font_path}...")
        subprocess.check_output(command, shell=True)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontsize=24:fontfile={simba_font_path}:text=%{{n}}:x=(w-tw)/2:y=h-th-10:fontcolor=white:box=1:boxcolor=white@0.5" -c:v h264_nvenc -c:a copy "{save_name}" -y'
        else:
            command = f'ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile={simba_font_path}:text=\'%{{frame_num}}\':start_number=1:x=(w-tw)/2:y=h-(2*lh):fontcolor=black:fontsize=20:box=1:boxcolor=white:boxborderw=5" -c:a copy "{save_name}" -y'
        print(f"Using font path {simba_font_path}...")
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"Superimposed video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
    )


# _ = superimpose_frame_count(file_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi', gpu=False)


def remove_beginning_of_video(
    file_path: Union[str, os.PathLike], time: int, gpu: Optional[bool] = False
) -> None:
    """
    Remove N seconds from the beginning of a video file. The result is stored in the same directory as the
    input file with the ``_shorten.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int time: Number of seconds to remove from the beginning of the video.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = remove_beginning_of_video(file_path='project_folder/videos/Video_1.avi', time=10)
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="Cut time", value=time)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_shorten.mp4")
    if os.path.isfile(save_name):
        raise FileExistError(
            msg="SIMBA ERROR: The outfile file already exist: {}.".format(save_name),
            source=remove_beginning_of_video.__name__,
        )
    if gpu:
        if not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(
                msg="No GPU found (as evaluated by nvidea-smi returning None)",
                source=remove_beginning_of_video.__name__,
            )
        command = 'ffmpeg -hwaccel auto -c:v h264_cuvid -ss {} -i "{}" -c:v h264_nvenc -c:a aac "{}"'.format(
            int(time), file_path, save_name
        )
    else:
        command = (
            str("ffmpeg -ss ")
            + str(int(time))
            + " -i "
            + '"'
            + str(file_path)
            + '"'
            + " -c:v libx264 -c:a aac "
            + '"'
            + save_name
            + '"'
        )
    print("Shortening video... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=remove_beginning_of_video.__name__,
    )


# remove_beginning_of_video(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', time=10, gpu=True)


def clip_video_in_range(
    file_path: Union[str, os.PathLike],
    start_time: str,
    end_time: str,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    overwrite: Optional[bool] = False,
    include_clip_time_in_filename: Optional[bool] = False,
    gpu: Optional[bool] = False,
) -> None:
    """
    Clip video within a specific range. The result is stored in the same directory as the
    input file with the ``_clipped.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter str start_time: Start time in HH:MM:SS format.
    :parameter str end_time: End time in HH:MM:SS format.
    :parameter Optional[Union[str, os.PathLike]] out_dir: If None, then the clip will be stored in the same dir as the input video. If directory, then the location of the output files.
    :parameter Optional[bool] include_clip_time_in_filename: If True, include the clip start and end in HH-MM-SS format as suffix in the filename. If False, then use integer suffic representing the count.
    :parameter Optional[bool] overwrite: If True, the overwrite output file if path already exist. If False, then raise FileExistError.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = clip_video_in_range(file_path='project_folder/videos/Video_1.avi', start_time='00:00:05', end_time='00:00:10')
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=clip_video_in_range.__name__,
        )
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    if out_dir is not None:
        check_if_dir_exists(in_dir=out_dir)
        dir = out_dir
    check_if_string_value_is_valid_video_timestamp(value=start_time, name="START TIME")
    check_if_string_value_is_valid_video_timestamp(value=end_time, name="END TIME")
    check_that_hhmmss_start_is_before_end(
        start_time=start_time, end_time=end_time, name=f"{file_name} timestamps"
    )
    if not include_clip_time_in_filename:
        save_name = os.path.join(dir, file_name + "_clipped.mp4")
    else:
        save_name = os.path.join(
            dir,
            file_name
            + f'_{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.mp4',
        )
    if os.path.isfile(save_name) and (not overwrite):
        raise FileExistError(
            msg=f"SIMBA ERROR: The outfile file already exist: {save_name}.",
            source=clip_video_in_range.__name__,
        )
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -ss {start_time} -to {end_time} -async 1 "{save_name}" -y'
    else:
        command = f'ffmpeg -i "{file_path}" -ss {start_time} -to {end_time} -async 1 "{save_name}" -y'
    print(f"Clipping video {file_name}... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=clip_video_in_range.__name__,
    )


def downsample_video(
    file_path: Union[str, os.PathLike],
    video_height: int,
    video_width: int,
    gpu: Optional[bool] = False,
) -> None:
    """
    Down-sample a video file. The result is stored in the same directory as the
    input file with the ``_downsampled.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter int video_height: height of new video.
    :parameter int video_width: width of new video.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = downsample_video(file_path='project_folder/videos/Video_1.avi', video_height=600, video_width=400)
    """
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=downsample_video.__name__,
        )
    timer = SimbaTimer(start=True)
    check_int(name="Video height", value=video_height)
    check_int(name="Video width", value=video_width)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_downsampled.mp4")
    if os.path.isfile(save_name):
        raise FileExistError(
            "SIMBA ERROR: The outfile file already exist: {}.".format(save_name),
            source=downsample_video.__name__,
        )
    if gpu:
        command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "scale=w={video_width}:h={video_height}:force_original_aspect_ratio=decrease:flags=bicubic" -c:v h264_nvenc "{save_name}"'
    else:
        command = (
            str("ffmpeg -i ")
            + '"'
            + str(file_path)
            + '"'
            + " -vf scale="
            + str(video_width)
            + ":"
            + str(video_height)
            + " "
            + '"'
            + save_name
            + '"'
            + " -hide_banner"
        )
    print("Down-sampling video... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=downsample_video.__name__,
    )


# downsample_video(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', video_height=600, video_width=400, gpu=True)


def gif_creator(
    file_path: str,
    start_time: int,
    duration: int,
    width: int,
    gpu: Optional[bool] = False,
) -> None:
    """
    Create a sample gif from a video file. The result is stored in the same directory as the
    input file with the ``.gif`` file-ending.

    .. note::
       The height is auto-computed to retain aspect ratio

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter int start_time: Start time of the gif in relation to the video in seconds.
    :parameter int duration: Duration of the gif.
    :parameter int width: Width of the gif.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = gif_creator(file_path='project_folder/videos/Video_1.avi', start_time=5, duration=10, width=600)
    """

    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=gif_creator.__name__,
        )
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="Start time", value=start_time)
    check_int(name="Duration", value=duration)
    check_int(name="Width", value=width)
    _ = get_video_meta_data(file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + ".gif")
    if os.path.isfile(save_name):
        raise FileExistError(
            "SIMBA ERROR: The outfile file already exist: {}.".format(save_name),
            source=gif_creator.__name__,
        )
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -ss {start_time} -i "{file_path}" -to {duration} -vf "fps=10,scale={width}:-1" -c:v gif -pix_fmt rgb24 -y "{save_name}" -y'
    else:
        command = (
            "ffmpeg -ss "
            + str(start_time)
            + " -t "
            + str(duration)
            + " -i "
            + '"'
            + str(file_path)
            + '"'
            + ' -filter_complex "[0:v] fps=15,scale=w='
            + str(width)
            + ':h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" '
            + '"'
            + str(save_name)
            + '"'
        )
    print("Creating gif sample... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=gif_creator.__name__,
    )


# gif_creator(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', start_time=5, duration=15, width=600, gpu=True)


def batch_convert_video_format(
    directory: Union[str, os.PathLike],
    input_format: str,
    output_format: str,
    gpu: Optional[bool] = False,
) -> None:
    """
    Batch convert all videos in a folder of specific format into a different video format. The results are
    stored in the same directory as the input files.

    :parameter Union[str, os.PathLike] directory: Path to video file directory.
    :parameter str input_format: Format of the input files (e.g., avi).
    :parameter str output_format: Format of the output files (e.g., mp4).
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = batch_convert_video_format(directory='project_folder/videos', input_format='avi', output_format='mp4')
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=batch_convert_video_format.__name__,
        )
    if input_format == output_format:
        raise InvalidFileTypeError(
            msg=f"The input format ({input_format}) is the same as the output format ({output_format})",
            source=batch_convert_video_format.__name__,
        )
    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory),
            source=batch_convert_video_format.__name__,
        )
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + "/*") if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() == ".{}".format(input_format.lower()):
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: No files with .{} file ending found in the {} directory".format(
                input_format, directory
            ),
            source=batch_convert_video_format.__name__,
        )
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print("Processing video {}...".format(file_name))
        save_path = os.path.join(
            dir_name, file_name + ".{}".format(output_format.lower())
        )
        if os.path.isfile(save_path):
            raise FileExistError(
                msg="SIMBA ERROR: The outfile file already exist: {}.".format(
                    save_path
                ),
                source=batch_convert_video_format.__name__,
            )
        if gpu:
            command = 'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{}" -c:v h264_nvenc -cq 23 -preset:v medium -c:a copy "{}"'.format(
                file_path, save_path
            )
        else:
            command = (
                "ffmpeg -y -i "
                + '"'
                + file_path
                + '"'
                + " -c:v libx264 -crf 5 -preset medium -c:a libmp3lame -b:a 320k "
                + '"'
                + save_path
                + '"'
            )
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        print(
            "Video {} complete, (Video {}/{})...".format(
                file_name, str(file_cnt + 1), str(len(video_paths))
            )
        )

    stdout_success(
        msg=f"SIMBA COMPLETE: {str(len(video_paths))} videos converted in {directory} directory!",
        source=batch_convert_video_format.__name__,
    )


def batch_create_frames(directory: Union[str, os.PathLike]) -> None:
    """
    Extract all frames for all videos in a directory. Results are stored within sub-directories in the input
    directory named according to the video files.

    :parameter str directory: Path to directory containing video files.

    :example:
    >>> _ = batch_create_frames(directory='project_folder/videos')
    """

    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory),
            source=batch_create_frames.__name__,
        )
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + "/*") if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in [".avi", ".mp4", ".mov", ".flv"]:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory".format(
                directory
            ),
            source=batch_create_frames.__name__,
        )
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print("Processing video {}...".format(file_name))
        save_dir = os.path.join(dir_name, file_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
        print(
            "Video {} complete, (Video {}/{})...".format(
                file_name, str(file_cnt + 1), str(len(video_paths))
            )
        )
    stdout_success(
        msg=f"{str(len(video_paths))} videos converted into frames in {directory} directory!",
        source=batch_create_frames.__name__,
    )


def extract_frames_single_video(file_path: Union[str, os.PathLike]) -> None:
    """
    Extract all frames for a single. Results are stored within a subdirectory in the same
    directory as the input file.

    :parameter str file_path: Path to video file.

    :example:
    >>> _ = extract_frames_single_video(file_path='project_folder/videos/Video_1.mp4')
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    save_dir = os.path.join(dir_name, file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Processing video {}...".format(file_name))
    video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(
        msg=f"Video {file_name} converted to images in {dir_name} directory!",
        elapsed_time=timer.elapsed_time_str,
        source=extract_frames_single_video.__name__,
    )


def multi_split_video(
    file_path: Union[str, os.PathLike],
    start_times: List[str],
    end_times: List[str],
    out_dir: Optional[Union[str, os.PathLike]] = None,
    include_clip_time_in_filename: Optional[bool] = False,
    gpu: Optional[bool] = False,
) -> None:
    """
    Divide a video file into multiple video files from specified start and stop times.

    :parameter str file_path: Path to input video file.
    :parameter List[str] start_times: Start times in HH:MM:SS format.
    :parameter List[str] end_times: End times in HH:MM:SS format.
    :parameter Optional[Union[str, os.PathLike]] out_dir: If None, then the clips will be stored in the same dir as the input video. If directory, then the location of the output files.
    :parameter Optional[bool] include_clip_time_in_filename: If True, include the clip start and end in HH-MM-SS format as suffix in the filename. If False, then use integer suffic representing the count.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = multi_split_video(file_path='project_folder/videos/Video_1.mp4', start_times=['00:00:05', '00:00:20'], end_times=['00:00:10', '00:00:25'])
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        dir_name = out_dir
    for start_time_cnt, start_time in enumerate(start_times):
        check_if_string_value_is_valid_video_timestamp(
            value=start_time, name=f"Start time for clip {start_time_cnt+1}"
        )
    for end_time_cnt, end_time in enumerate(end_times):
        check_if_string_value_is_valid_video_timestamp(
            value=end_time, name=f"End time for clip {end_time_cnt+1}"
        )
    for clip_cnt, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        check_that_hhmmss_start_is_before_end(
            start_time=start_time, end_time=end_time, name=f"Clip {str(clip_cnt+1)}"
        )
        check_if_hhmmss_timestamp_is_valid_part_of_video(
            timestamp=start_time, video_path=file_path
        )
        check_if_hhmmss_timestamp_is_valid_part_of_video(
            timestamp=end_time, video_path=file_path
        )
        if not include_clip_time_in_filename:
            save_path = os.path.join(
                dir_name, file_name + "_{}".format(str(clip_cnt + 1)) + ".mp4"
            )
        else:
            save_path = os.path.join(
                dir_name,
                file_name
                + f'_{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.mp4',
            )
        if os.path.isfile(save_path):
            raise FileExistError(
                msg=f"The outfile file already exist: {save_path}.",
                source=multi_split_video.__name__,
            )
        if gpu:
            if not check_nvidea_gpu_available():
                raise FFMPEGCodecGPUError(
                    msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
                    source=multi_split_video.__name__,
                )
            command = 'ffmpeg -hwaccel auto -i "{}" -ss {} -to {} -c:v h264_nvenc -async 1 "{}"'.format(
                file_path, start_time, end_time, save_path
            )
        else:
            command = f'ffmpeg -i "{file_path}" -ss {start_time} -to {end_time} -async 1 "{save_path}"'
        print("Processing video clip {}...".format(str(clip_cnt + 1)))
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"Video {file_name} converted into {str(len(start_times))} clips in directory {dir_name}!",
        elapsed_time=timer.elapsed_time_str,
        source=multi_split_video.__name__,
    )


# multi_split_video(file_path=r'/Users/simon/Desktop/time_s_converted.mp4', start_times=['00:00:01', '00:00:02'], end_times=['00:00:04', '00:00:05'], gpu=False)


def crop_single_video(
    file_path: Union[str, os.PathLike], gpu: Optional[bool] = False
) -> None:
    """
    Crop a single video using ``simba.video_processors.roi_selector.ROISelector`` interface. Results is saved in the same directory as input video with the
    ``_cropped.mp4`` suffix`.

    :parameter str file_path: Path to video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = crop_single_video(file_path='project_folder/videos/Video_1.mp4')
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=crop_single_video.__name__,
        )
    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    roi_selector = ROISelector(path=file_path)
    roi_selector.run()
    if (
        (roi_selector.top_left[0] < 0)
        or (roi_selector.top_left[1] < 0)
        or (roi_selector.bottom_right[0] < 0)
        or (roi_selector.bottom_right[1] < 1)
    ):
        raise CountError(
            msg="CROP FAILED: Cannot use negative crop coordinates.",
            source=crop_multiple_videos.__name__,
        )
    save_path = os.path.join(dir_name, f"{file_name}_cropped.mp4")
    if os.path.isfile(save_path):
        raise FileExistError(
            msg=f"SIMBA ERROR: The out file  already exist: {save_path}.",
            source=crop_single_video.__name__,
        )
    timer = SimbaTimer(start=True)
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "crop={roi_selector.width}:{roi_selector.height}:{roi_selector.top_left[0]}:{roi_selector.top_left[1]}" -c:v h264_nvenc -c:a copy "{save_path}"'
    else:
        command = f'ffmpeg -y -i "{file_path}" -vf "crop={roi_selector.width}:{roi_selector.height}:{roi_selector.top_left[0]}:{roi_selector.top_left[1]}" -c:v libx264 -crf 17 -c:a copy "{save_path}"'
    subprocess.call(command, shell=True)
    timer.stop_timer()
    stdout_success(
        f"Video {file_name} cropped and saved at {save_path}",
        elapsed_time=timer.elapsed_time_str,
        source=crop_single_video.__name__,
    )


# _ = crop_single_video(file_path='/Users/simon/Desktop/envs/troubleshooting/ARES_data/Termite Test 1/Termite Test 1.mp4')

# crop_single_video(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', gpu=False)


def crop_multiple_videos(
    directory_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    gpu: Optional[bool] = False,
) -> None:
    """
    Crop multiple videos in a folder according to crop-coordinates defined in the **first** video.

    :parameter str directory_path: Directory containing input videos.
    :parameter str output_path: Directory where to save the cropped videos.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = crop_multiple_videos(directory_path='project_folder/videos', output_path='project_folder/videos/my_new_folder')
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=crop_multiple_videos.__name__,
        )
    if not os.path.isdir(directory_path):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory_path),
            source=crop_multiple_videos.__name__,
        )
    video_paths = []
    file_paths_in_folder = [
        f for f in glob.glob(directory_path + "/*") if os.path.isfile(f)
    ]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in Options.ALL_VIDEO_FORMAT_OPTIONS.value:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory".format(
                directory_path
            ),
            source=crop_multiple_videos.__name__,
        )
    roi_selector = ROISelector(path=file_paths_in_folder[0])
    roi_selector.run()
    if (roi_selector.width == 0 and roi_selector.height == 0) or (
        roi_selector.width
        + roi_selector.height
        + roi_selector.top_left[0]
        + roi_selector.top_left[1]
        == 0
    ):
        raise CountError(
            msg="CROP FAILED: Cropping height and width are both 0. Please try again.",
            source=crop_multiple_videos.__name__,
        )
    if (
        (roi_selector.top_left[0] < 0)
        or (roi_selector.top_left[1] < 0)
        or (roi_selector.bottom_right[0] < 0)
        or (roi_selector.bottom_right[1] < 1)
    ):
        raise CountError(
            msg="CROP FAILED: Cannot use negative crop coordinates.",
            source=crop_multiple_videos.__name__,
        )
    timer = SimbaTimer(start=True)
    for file_cnt, file_path in enumerate(video_paths):
        video_timer = SimbaTimer(start=True)
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print(f"Cropping video {file_name}...")
        video_meta_data = get_video_meta_data(file_path)
        if (roi_selector.bottom_right[0] > video_meta_data["width"]) or (
            roi_selector.bottom_right[1] > video_meta_data["height"]
        ):
            raise InvalidInputError(
                msg=f'Cannot crop video {file_name} of size {video_meta_data["resolution_str"]} at location top left: {roi_selector.top_left}, bottom right: {roi_selector.bottom_right}',
                source=crop_multiple_videos.__name__,
            )
        save_path = os.path.join(output_path, file_name + "_cropped.mp4")
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "crop={roi_selector.width}:{roi_selector.height}:{roi_selector.top_left[0]}:{roi_selector.top_left[1]}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
        else:
            command = f'ffmpeg -i "{file_path}" -vf "crop={roi_selector.width}:{roi_selector.height}:{roi_selector.top_left[0]}:{roi_selector.top_left[1]}" -c:v libx264 -crf 17 -c:a copy "{save_path}" -y'
        subprocess.call(command, shell=True)
        video_timer.stop_timer()
        print(
            f"Video {file_name} cropped (Video {file_cnt+1}/{len(video_paths)}, elapsed time: {video_timer.elapsed_time_str})"
        )
    timer.stop_timer()
    stdout_success(
        msg=f"{str(len(video_paths))} videos cropped and saved in {directory_path} directory",
        elapsed_time=timer.elapsed_time_str,
        source=crop_multiple_videos.__name__,
    )


# _ = crop_multiple_videos(directory_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos', output_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/test/test')


def frames_to_movie(
    directory: Union[str, os.PathLike],
    fps: int,
    bitrate: int,
    img_format: str,
    gpu: Optional[bool] = False,
) -> None:
    """
    Merge all image files in a folder to a mp4 video file. Video file is stored in the same directory as the
    input directory sub-folder.

    .. note::
       The Image files have to have ordered numerical names e.g., ``1.png``, ``2.png`` etc...

    :parameter str directory: Directory containing the images.
    :parameter int fps: The frame rate of the output video.
    :parameter int bitrate: The bitrate of the output video (e.g., 32000).
    :parameter str img_format: The format of the input image files (e.g., ``png``).
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = frames_to_movie(directory_path='project_folder/video_img', fps=15, bitrate=32000, img_format='png')
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=frames_to_movie.__name__,
        )
    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory),
            source=frames_to_movie.__name__,
        )
    check_int(name="FPS", value=fps)
    check_int(name="BITRATE", value=bitrate)
    file_paths_in_folder = [f for f in glob.glob(directory + "/*") if os.path.isfile(f)]
    img_paths_in_folder = [
        x for x in file_paths_in_folder if Path(x).suffix[1:] == img_format
    ]
    if len(img_paths_in_folder) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: Zero images of file-type {} found in {} directory".format(
                img_format, directory
            ),
            source=frames_to_movie.__name__,
        )
    img = cv2.imread(img_paths_in_folder[0])
    img_h, img_w = int(img.shape[0]), int(img.shape[1])
    ffmpeg_fn = os.path.join(directory, "%d.{}".format(img_format))
    save_path = os.path.join(
        os.path.dirname(directory), os.path.basename(directory) + ".mp4"
    )
    if gpu:
        command = f'ffmpeg -y -r {fps} -f image2 -s {img_h}x{img_w} -i "{ffmpeg_fn}" -c:v h264_nvenc -b:v {bitrate}k "{save_path}" -y'
    else:
        command = (
            str(
                "ffmpeg -y -r "
                + str(fps)
                + " -f image2 -s "
                + str(img_h)
                + "x"
                + str(img_w)
                + " -i "
                + '"'
                + ffmpeg_fn
                + '"'
                + " -vcodec libx264 -b "
                + str(bitrate)
                + "k "
                + '"'
                + str(save_path)
                + '"'
            )
            + " -y"
        )
    print(
        "Creating {} from {} images...".format(
            os.path.basename(save_path), str(len(img_paths_in_folder))
        )
    )
    subprocess.call(command, shell=True)
    stdout_success(msg=f"Video created at {save_path}", source=frames_to_movie.__name__)


def video_concatenator(
    video_one_path: Union[str, os.PathLike],
    video_two_path: Union[str, os.PathLike],
    resolution: Union[int, str],
    horizontal: bool,
    gpu: Optional[bool] = False,
) -> None:
    """
    Concatenate two videos to a single video

    :param str video_one_path: Path to the first video in the concatenated video
    :param str video_two_path: Path to the second video in the concatenated video
    :param int or str resolution: If str, then the name of the video which resolution you want to retain. E.g., `Video_1`. Else int, representing the video width (if vertical concat) or height (if horizontal concat). Aspect raio will be retained.
    :param horizontal: If true, then horizontal concatenation. Else vertical concatenation.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> video_concatenator(video_one_path='project_folder/videos/Video_1.mp4', video_two_path='project_folder/videos/Video_2.mp4', resolution=800, horizontal=True)
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=video_concatenator.__name__,
        )
    if not check_ffmpeg_available():
        raise FFMPEGNotFoundError(
            msg="FFMPEG not found on the computer. Install FFMPEG to use the concatenation method.",
            source=video_concatenator.__name__,
        )
    timer = SimbaTimer(start=True)
    for file_path in [video_one_path, video_two_path]:
        check_file_exist_and_readable(file_path=file_path)
        _ = get_video_meta_data(file_path)
    if type(resolution) is int:
        video_meta_data = {}
        if horizontal:
            video_meta_data["height"] = resolution
        else:
            video_meta_data["width"] = resolution
    elif resolution is "Video 1":
        video_meta_data = get_video_meta_data(video_one_path)
    else:
        video_meta_data = get_video_meta_data(video_one_path)
    dir, file_name_1, _ = get_fn_ext(video_one_path)
    _, file_name_2, _ = get_fn_ext(video_two_path)
    print(f"Concatenating videos {file_name_1} and {file_name_2}...")
    save_path = os.path.join(dir, file_name_1 + file_name_2 + "_concat.mp4")
    if horizontal:
        if gpu:
            command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_one_path}" -hwaccel auto -c:v h264_cuvid -i "{video_two_path}" -filter_complex "[0:v]scale=-1:{video_meta_data["height"]}[v0];[v0][1:v]hstack=inputs=2" -c:v h264_nvenc "{save_path}"'
        else:
            command = 'ffmpeg -y -i "{}" -i "{}" -filter_complex "[0:v]scale=-1:{}[v0];[v0][1:v]hstack=inputs=2" "{}"'.format(
                video_one_path, video_two_path, video_meta_data["height"], save_path
            )
    else:
        if gpu:
            command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_one_path}" -hwaccel auto -c:v h264_cuvid -i "{video_two_path}" -filter_complex "[0:v]scale={video_meta_data["width"]}:-1[v0];[v0][1:v]vstack=inputs=2" -c:v h264_nvenc "{save_path}"'
        else:
            command = 'ffmpeg -y -i "{}" -i "{}" -filter_complex "[0:v]scale={}:-1[v0];[v0][1:v]vstack=inputs=2" "{}"'.format(
                video_one_path, video_two_path, video_meta_data["width"], save_path
            )

    if gpu:
        process = subprocess.Popen(command, shell=True)
        output, error = process.communicate()
        if process.returncode != 0:
            if "Unknown decoder" in str(error.split(b"\n")[-2]):
                raise FFMPEGCodecGPUError(
                    msg="GPU codec not found: reverting to CPU. Properly configure FFMpeg and ensure you have GPU available or use CPU.",
                    source=video_concatenator.__name__,
                )
            else:
                raise FFMPEGCodecGPUError(
                    msg="GPU error. Properly configure FFMpeg and ensure you have GPU available, or use CPU.",
                    source=video_concatenator.__name__,
                )
        else:
            pass
    else:
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"Videos concatenated and saved at {save_path}",
        elapsed_time=timer.elapsed_time_str,
        source=video_concatenator.__name__,
    )


# video_concatenator(video_one_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                    video_two_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_2.avi', resolution='Video_1',
#                    horizontal=False,
#                    gpu=False)


class VideoRotator(ConfigReader):
    """
    GUI Tool for rotating video. Rotated video is saved with the ``_rotated_DATETIME.mp4`` suffix.

    :parameter str input_path: Path to video to rotate.
    :parameter str output_dir: Directory where to save the rotated video.
    :parameter Optional[bool] gpu: If True, use FFMPEG NVIDEA GPU codecs. Else CPU codecs.
    :parameter Optional[bool] gpu: If True, use FFPMPEG. Else, OpenCV.

    :example:
    >>> VideoRotator(input_path='project_folder/videos/Video_1.mp4', output_dir='project_folder/videos')
    """

    def __init__(
        self,
        input_path: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        gpu: Optional[bool] = False,
        ffmpeg: Optional[bool] = False,
    ) -> None:
        if gpu and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(
                msg="No GPU found (as evaluated by nvidea-smi returning None)",
                source=self.__class__.__name__,
            )
        if ffmpeg and not check_ffmpeg_available():
            raise FFMPEGNotFoundError(
                msg='FFMPEG not found on the computer (as evaluated by "ffmpeg" returning None)',
                source=self.__class__.__name__,
            )
        _, self.cpu_cnt = find_core_cnt()
        self.gpu, self.ffmpeg = gpu, ffmpeg
        self.save_dir = output_dir
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        if os.path.isfile(input_path):
            self.video_paths = [input_path]
        else:
            self.video_paths = find_all_videos_in_directory(
                directory=input_path, as_dict=True
            ).values()
            check_if_filepath_list_is_empty(
                filepaths=self.video_paths,
                error_msg=f"No videos found in {input_path} directory",
            )

    def __insert_img(self, img: np.array):
        current_frm_pil = Image.fromarray(img)
        current_frm_pil.thumbnail(MAX_FRM_SIZE, Image.ANTIALIAS)
        current_frm_pil = ImageTk.PhotoImage(
            master=self.main_frm, image=current_frm_pil
        )
        self.video_frame = Label(self.main_frm, image=current_frm_pil)
        self.video_frame.image = current_frm_pil
        self.video_frame.grid(row=0, column=0)

    def __rotate(self, value: int, img: np.array):
        self.dif_angle += value
        rotation_matrix = cv2.getRotationMatrix2D(
            (self.video_meta_data["width"] / 2, self.video_meta_data["height"] / 2),
            self.dif_angle,
            1,
        )
        img = cv2.warpAffine(
            img,
            rotation_matrix,
            (self.video_meta_data["width"], self.video_meta_data["height"]),
        )
        self.__insert_img(img=img)

    def __run_rotation(self):
        self.main_frm.destroy()
        start = time.time()
        if self.ffmpeg or self.gpu:
            for video_cnt, (video_path, rotation) in enumerate(self.results.items()):
                _, name, _ = get_fn_ext(filepath=video_path)
                save_path = os.path.join(
                    self.save_dir, f"{name}_rotated_{self.datetime}.mp4"
                )
                if self.gpu:
                    cmd = f'ffmpeg -hwaccel auto -i {video_path} -vf "hwupload_cuda,rotate={rotation}*(PI/180),format=nv12|cuda" -c:v h264_nvenc {save_path} -y'
                else:
                    cmd = f'ffmpeg -i {video_path} -vf "rotate={rotation}*(PI/180)" {save_path} -y'
                subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        else:
            for video_cnt, (video_path, rotation) in enumerate(self.results.items()):
                cap = cv2.VideoCapture(video_path)
                _, name, _ = get_fn_ext(filepath=video_path)
                rotation_matrix = cv2.getRotationMatrix2D(
                    (
                        self.video_meta_data["width"] / 2,
                        self.video_meta_data["height"] / 2,
                    ),
                    rotation,
                    1,
                )
                save_path = os.path.join(
                    self.save_dir, f"{name}_rotated_{self.datetime}.mp4"
                )
                video_meta = get_video_meta_data(video_path=video_path)
                fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                writer = cv2.VideoWriter(
                    save_path,
                    fourcc,
                    video_meta["fps"],
                    (video_meta["width"], video_meta["height"]),
                )
                img_cnt = 0
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break
                    img = cv2.warpAffine(
                        img,
                        rotation_matrix,
                        (self.video_meta_data["width"], self.video_meta_data["height"]),
                    )
                    writer.write(img)
                    img_cnt += 1
                    print(
                        f'Rotating frame {img_cnt}/{video_meta["frame_count"]} (Video {video_cnt + 1}/{len(self.results.keys())}) '
                    )
                cap.release()
                writer.release()
        stdout_success(
            msg=f"All videos rotated and saved in {self.save_dir}",
            elapsed_time=str(round((time.time() - start), 2)),
            source=self.__class__.__name__,
        )

    def __save(self):
        process = None
        self.results[self.file_path] = self.dif_angle
        if len(self.results.keys()) == len(self.video_paths):
            process = multiprocessing.Process(target=self.__run_rotation())
            process.start()
        else:
            self.__run_interface(
                file_path=self.video_paths[len(self.results.keys()) - 1]
            )
        if process is not None:
            process.join()

    def __bind_keys(self):
        self.main_frm.bind(
            "<Left>", lambda x: self.__rotate(value=1, img=self._orig_img)
        )
        self.main_frm.bind(
            "<Right>", lambda x: self.__rotate(value=-1, img=self._orig_img)
        )
        self.main_frm.bind("<Escape>", lambda x: self.__save())

    def __run_interface(self, file_path: str):
        self.dif_angle = 0
        print(file_path)
        self.video_meta_data = get_video_meta_data(video_path=file_path)
        self.file_path = file_path
        _, self.video_name, _ = get_fn_ext(filepath=file_path)
        self.main_frm = Toplevel()
        self.main_frm.title(f"ROTATE VIDEO {self.video_name}")
        self.video_frm = Frame(self.main_frm)
        self.video_frm.grid(row=0, column=0)
        self.instruction_frm = Frame(self.main_frm, width=100, height=100)
        self.instruction_frm.grid(row=0, column=2, sticky=NW)
        self.key_lbls = Label(
            self.instruction_frm,
            text="\n\n Navigation: "
            "\n Left arrow = 1° left"
            "\n Right arrow = 1° right"
            "\n Esc = Save",
        )

        self.key_lbls.grid(sticky=NW)
        self.cap = cv2.VideoCapture(file_path)
        _, self.img = self.cap.read()
        self._orig_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        self.__insert_img(img=self._orig_img)
        self.__bind_keys()

    def run(self):
        self.results = {}
        for video_path in self.video_paths:
            self.__run_interface(video_path)
        self.main_frm.mainloop()


def extract_frames_from_all_videos_in_directory(
    config_path: Union[str, os.PathLike], directory: Union[str, os.PathLike]
) -> None:
    """
    Extract all frames from all videos in a directory. The results are saved in the project_folder/frames/input directory of the SimBA project

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter str directory: path to file or folder containing videos in mp4 and/or avi format.

    :example:
    >>> extract_frames_from_all_videos_in_directory(config_path='project_folder/project_config.ini', source='/tests/test_data/video_tests')
    """

    timer = SimbaTimer(start=True)
    video_paths, video_types = [], [".avi", ".mp4"]
    files_in_folder = glob.glob(directory + "/*")
    for file_path in files_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in video_types:
            video_paths.append(file_path)
    if len(video_paths) == 0:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: 0 video files in mp4 or avi format found in {}".format(
                directory
            ),
            source=extract_frames_from_all_videos_in_directory.__name__,
        )
    config = read_config_file(config_path)
    project_path = read_config_entry(
        config, "General settings", "project_path", data_type="folder_path"
    )

    print(
        "Extracting frames for {} videos into project_folder/frames/input directory...".format(
            len(video_paths)
        )
    )
    for video_path in video_paths:
        dir_name, video_name, ext = get_fn_ext(video_path)
        save_path = os.path.join(project_path, "frames", "input", video_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            print(
                f"Frames for video {video_name} already extracted. SimBA is overwriting prior frames..."
            )
        video_to_frames(video_path, save_path, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(
        f"Frames created for {str(len(video_paths))} videos",
        elapsed_time=timer.elapsed_time_str,
        source=extract_frames_from_all_videos_in_directory.__name__,
    )


def copy_img_folder(
    config_path: Union[str, os.PathLike], source: Union[str, os.PathLike]
) -> None:
    """
    Copy directory of png files to the SimBA project. The directory is stored in the project_folder/frames/input folder of the SimBA project

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter str source: path to image folder outside SimBA project.

    :example:
    >>> copy_img_folder(config_path='MySimBAprojectConfig', source='/DirectoryWithVideos/')

    """
    timer = SimbaTimer(start=True)
    if not os.path.isdir(source):
        raise NotDirectoryError(
            msg=f"SIMBA ERROR: source {source} is not a directory.",
            source=copy_img_folder.__name__,
        )
    if len(glob.glob(source + "/*.png")) == 0:
        raise NoFilesFoundError(
            msg=f"SIMBA ERROR: source {source} does not contain any .png files.",
            source=copy_img_folder.__name__,
        )
    input_basename = os.path.basename(source)
    config = read_config_file(config_path)
    project_path = read_config_entry(
        config,
        ConfigKey.GENERAL_SETTINGS.value,
        ConfigKey.PROJECT_PATH.value,
        data_type="folder_path",
    )
    input_frames_dir = os.path.join(project_path, Paths.INPUT_FRAMES_DIR.value)
    destination = os.path.join(input_frames_dir, input_basename)
    if os.path.isdir(destination):
        raise DirectoryExistError(
            msg=f"SIMBA ERROR: {destination} already exist in SimBA project.",
            source=copy_img_folder.__name__,
        )
    print(f"Importing image files for {input_basename}...")
    shutil.copytree(source, destination)
    timer.stop_timer()
    stdout_success(
        msg=f"{destination} imported to SimBA project",
        elapsed_time=timer.elapsed_time_str,
        source=copy_img_folder.__name__,
    )


def append_audio(
    video_path: Union[str, os.PathLike],
    audio_path: Union[str, os.PathLike],
    audio_src_type: Literal["video", "audio"] = "video",
) -> None:
    """
    Append audio track from one video to another video without an audio track.

    :param Union[str, os.PathLike] video_one_path: Path to video file without audio track.
    :param Union[str, os.PathLike] audio_path: Path to file (e.g., video) with audio track.
    :param Literal['video', 'audio'] audio_src_type: Type of audio source of "video_two_path" (e.g., video or audio file).

    :example:
    >>> append_audio(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/merged_video_20230425201637.mp4',
    >>> audio_path="/Users/simon/Documents/Zoom/ddd/video1180732233.mp4")
    """

    if not check_ffmpeg_available():
        raise FFMPEGNotFoundError(
            msg="FFMpeg not found on computer. See SimBA docs for install instructions.",
            source=append_audio.__name__,
        )
    check_file_exist_and_readable(file_path=video_path)
    check_file_exist_and_readable(file_path=audio_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    audio_src_meta_data = get_video_meta_data(video_path=audio_path)
    save_path = os.path.join(
        os.path.dirname(video_path), get_fn_ext(filepath=video_path)[1] + "_w_audio.mp4"
    )
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        track_type = (
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        raise NoDataError(
            msg=f"No audio track found in file {audio_path}",
            source=append_audio.__name__,
        )

    if video_meta_data["frame_count"] != audio_src_meta_data["frame_count"]:
        InValidUserInputWarning(
            msg=f'The video ({video_meta_data["frame_count"]}) and audio source ({audio_src_meta_data["frame_count"]}) does not have an equal number of frames.',
            source=append_audio.__name__,
        )

    cmd = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -map 0:v:0 -map 1:a:0 "{save_path}" -y'

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)

    stdout_success(
        msg=f"Audio merged successfully, file saved at {save_path}!",
        source=append_audio.__name__,
    )


# append_audio(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/merged_video_20230425201637.mp4',
# audio_path="/Users/simon/Documents/Zoom/ddd/video1180732233.mp4")


# r = VideoRotator(input_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Testing/Together_1_downsampled.mp4',
#              output_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Blah',
#              gpu=True)
# r.run()


# video_concatenator(video_one_path='/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/frames/output/gantt_plots/SI_DAY3_308_CD1_PRESENT_2.mp4',
#                    video_two_path= '/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/frames/output/gantt_plots/SI_DAY3_308_CD1_PRESENT.mp4',
#                    resolution='Video 1',
#                    horizontal=False)


# video_concatenator(video_one_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                    video_two_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                    resolution='Video_1',
#                    horizontal=True,
#                    gpu=False)
