__author__ = "Simon Nilsson"


import glob
import multiprocessing
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tkinter import *
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageTk
from shapely.geometry import Polygon

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import simba
from simba.mixins.config_reader import ConfigReader
from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_lst, check_valid_tuple)
from simba.utils.enums import OS, ConfigKey, Formats, Options, Paths
from simba.utils.errors import (CountError, DirectoryExistError,
                                FFMPEGCodecGPUError, FFMPEGNotFoundError,
                                FileExistError, FrameRangeError,
                                InvalidFileTypeError, InvalidInputError,
                                InvalidVideoFileError, NoDataError,
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
from simba.video_processors.roi_selector_circle import ROISelectorCircle
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon

MAX_FRM_SIZE = 1080, 650


def change_img_format(
    directory: Union[str, os.PathLike],
    file_type_in: str,
    file_type_out: str,
    verbose: Optional[bool] = False,
) -> None:
    """
    Convert the file type of all image files within a directory.

    :parameter Union[str, os.PathLike] directory: Path to directory holding image files
    :parameter str file_type_in: Input file type, e.g., 'bmp' or 'png.
    :parameter str file_type_out: Output file type, e.g., 'bmp' or 'png.
    :parameter Optional[bool] verbose: If True, prints progress. Default False.

    :example:
    >>> _ = change_img_format(directory='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/test_img', file_type_in='png', file_type_out='bmp', verbose=True)

    """
    check_if_dir_exists(in_dir=directory, source=change_img_format.__name__)
    files_found = glob.glob(directory + f"/*.{file_type_in}")
    if len(files_found) == 0:
        raise NoFilesFoundError(
            f"SIMBA ERROR: No {file_type_in} files (with .{file_type_in} file extension) found in the {directory} directory",
            source=change_img_format.__name__,
        )
    print(f"{len(files_found)} image files found in {directory}...")
    for file_cnt, file_path in enumerate(files_found):
        if verbose:
            print(f"Converting file {file_cnt+1}/{len(files_found)} ...")
        im = Image.open(file_path)
        save_name = file_path.replace("." + str(file_type_in), "." + str(file_type_out))
        im.save(save_name)
        os.remove(file_path)
    stdout_success(
        msg=f"SIMBA COMPLETE: Files in {directory} directory converted to {file_type_out}",
        source=change_img_format.__name__,
    )


def clahe_enhance_video(
    file_path: Union[str, os.PathLike],
    clip_limit: Optional[int] = 2,
    tile_grid_size: Optional[Tuple[int]] = (16, 16),
) -> None:
    """
    Convert a single video file to clahe-enhanced greyscale .avi file. The result is saved with prefix
    ``CLAHE_`` in the same directory as in the input file.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[int] clip_limit: CLAHE amplification limit. Inccreased clip limit reduce noise in output. Default: 2.
    :parameter Optional[Tuple[int]] tile_grid_size: The histogram kernel size.

    :example:
    >>> _ = clahe_enhance_video(file_path: 'project_folder/videos/Video_1.mp4')
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(
        name=f"{clahe_enhance_video.__name__} clip_limit", value=clip_limit, min_value=0
    )
    video_meta_data = get_video_meta_data(file_path)
    check_valid_tuple(
        x=tile_grid_size,
        source=clahe_enhance_video.__name__,
        accepted_lengths=(2,),
        valid_dtypes=(int,),
    )
    if (tile_grid_size[0] > video_meta_data["height"]) or (
        (tile_grid_size[1] > video_meta_data["width"])
    ):
        raise InvalidInputError(
            msg=f'The tile grid size ({tile_grid_size}) is larger than the video size ({video_meta_data["resolution_str"]})',
            source=clahe_enhance_video.__name__,
        )
    dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    save_path = os.path.join(dir, f"CLAHE_{file_name}.avi")
    fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
    print(f"Applying CLAHE on video {file_name}, this might take awhile...")
    cap = cv2.VideoCapture(file_path)
    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        video_meta_data["fps"],
        (video_meta_data["width"], video_meta_data["height"]),
        0,
    )
    clahe_filter = cv2.createCLAHE(
        clipLimit=int(clip_limit), tileGridSize=tile_grid_size
    )
    frm_cnt = 0
    try:
        while True:
            ret, img = cap.read()
            if ret:
                frm_cnt += 1
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe_frm = clahe_filter.apply(img)
                writer.write(clahe_frm)
                print(
                    f"CLAHE converted frame {frm_cnt}/{video_meta_data['frame_count']}"
                )
            else:
                break
        cap.release()
        writer.release()
        print(f"CLAHE video created: {save_path}.")
    except Exception as se:
        print(se.args)
        print(f"CLAHE conversion failed for video {file_name}.")
        cap.release()
        writer.release()
        raise InvalidVideoFileError(
            msg=f"Could not convert file {file_path} to CLAHE enhanced video",
            source=clahe_enhance_video.__name__,
        )


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
        frm_save_path = os.path.join(save_dir, f"{frm_number}.png")
        cv2.imwrite(frm_save_path, frame)
        print(f"Frame {frm_number} saved (Frame {frm_cnt}/{len(frame_range)-1})")
    stdout_success(
        msg=f"{len(frame_range)-1} frames extracted for video {file_name} saved in {save_dir}",
        source=extract_frame_range.__name__,
    )


def change_single_video_fps(
    file_path: Union[str, os.PathLike], fps: int, gpu: Optional[bool] = False
) -> None:
    """
    Change the fps of a single video file. Results are stored in the same directory as in the input file with
    the suffix ``_fps_new_fps``.

    .. note::
       To change the FPS of all videos in a directory, use ``simba.video_processors.video_processing.change_fps_of_multiple_videos``.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int fps: Fps of the new video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = change_single_video_fps(file_path='project_folder/videos/Video_1.mp4', fps=15)
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
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
    print(f"Converting the FPS to {fps} for video {file_name} ...")
    if os.path.isfile(save_path):
        FileExistWarning(
            msg=f"Overwriting existing file at {save_path}...",
            source=change_single_video_fps.__name__,
        )
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "fps={fps}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
    else:
        command = f'ffmpeg -i "{file_path}" -filter:v fps=fps={fps} -c:v libx264 -c:a aac "{save_path}" -y'
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
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=change_fps_of_multiple_videos.__name__,
        )
    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg=f"SIMBA ERROR: {directory} is not a valid directory",
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
        video_timer = SimbaTimer(start=True)
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print(f"Converting FPS for {file_name}...")
        save_path = os.path.join(
            dir_name, file_name + "_fps_{}{}".format(str(fps), str(ext))
        )
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "fps={fps}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
        else:
            command = f'ffmpeg -i {file_path} -filter:v fps=fps={fps} -c:v libx264 "{save_path}" -y'
        subprocess.call(command, shell=True)
        video_timer.stop_timer()
        print(
            f"Video {file_name} complete... (elapsed time: {video_timer.elapsed_time_str}s)"
        )
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: FPS of {len(video_paths)} video(s) changed to {fps}",
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

    check_ffmpeg_available(raise_error=True)
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
        command = f'ffmpeg -i "{file_path}" -c:v libx264 -preset slow -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -c:v libx264 -codec:a aac "{save_name}" -y'
    print("Creating video in powerpoint compatible format... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=convert_video_powerpoint_compatible_format.__name__,
    )


# convert_video_powerpoint_compatible_format(file_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test/SI_DAY3_308_CD1_PRESENT_fps_10_fps_15.mp4", gpu=False)


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

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=convert_to_mp4.__name__,
        )
    timer = SimbaTimer(start=True)
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
        command = f'ffmpeg -i "{file_path}" -c:v libx264 "{save_name}"'
    print("Converting to mp4... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=convert_to_mp4.__name__,
    )


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
    check_ffmpeg_available(raise_error=True)
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
        command = f'ffmpeg -i "{file_path}" -vf format=gray -c:v libx264 "{save_name}"'
    print(f"Converting {file_name} to greyscale... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=video_to_greyscale.__name__,
    )


def batch_video_to_greyscale(
    directory: Union[str, os.PathLike], gpu: Optional[bool] = False
) -> None:
    """
    Convert a directory of video file to greyscale mp4 format. The results are stored in the same directory as the
    input files with the ``_grayscale.mp4`` suffix.

    :parameter Union[str, os.PathLike] directory: Path to directory holding video files in color.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :raise FFMPEGCodecGPUError: If no GPU is found and ``gpu == True``.

    :example:
    >>> _ = batch_video_to_greyscale(directory='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_2')
    """
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="No GPU found (as evaluated by nvidea-smi returning None)",
            source=batch_video_to_greyscale.__name__,
        )
    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=directory, source=batch_video_to_greyscale.__name__)
    video_paths = find_all_videos_in_directory(
        directory=directory, as_dict=True, raise_error=True
    )
    for file_cnt, (file_name, file_path) in enumerate(video_paths.items()):
        video_timer = SimbaTimer(start=True)
        in_dir, _, _ = get_fn_ext(filepath=file_path)
        save_name = os.path.join(in_dir, f"{file_name}_grayscale.mp4")
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "hwupload_cuda,hwdownload,format=nv12,format=gray" -c:v h264_nvenc -c:a copy "{save_name}" -y'
        else:
            command = f'ffmpeg -i "{file_path}" -vf format=gray -c:v libx264 "{save_name}" -hide_banner -loglevel error -y'
        print(
            f"Converting {file_name} to greyscale (Video {file_cnt+1}/{len(list(video_paths.keys()))})... "
        )
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(
            f"Video {save_name} complete, (elapsed time: {video_timer.elapsed_time_str}s)"
        )
    timer.stop_timer()
    stdout_success(
        msg=f"{len(list(video_paths.keys()))} video(s) converted to gresyscale! Saved in {directory} with '_greyscale' suffix",
        elapsed_time=timer.elapsed_time_str,
        source=batch_video_to_greyscale.__name__,
    )


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
    check_ffmpeg_available(raise_error=True)
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
            command = f'ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile={simba_font_path}: text=\'%{{frame_num}}\': start_number=0: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy "{save_name}" -y'
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


# _ = superimpose_frame_count(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4', gpu=False)


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

    check_ffmpeg_available(raise_error=True)
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
        command = f'ffmpeg -ss {int(time)} -i "{file_path}" -c:v libx264 -c:a aac "{save_name}"'
    print("Shortening video... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=remove_beginning_of_video.__name__,
    )


# remove_beginning_of_video(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_frame_no.mp4', time=10, gpu=False)


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

    check_ffmpeg_available(raise_error=True)
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
        command = f'ffmpeg -i "{file_path}" -ss {start_time} -to {end_time} -async 1 -c:v libvpx-vp9 "{save_name}" -y'
    print(f"Clipping video {file_name} between {start_time} and {end_time}... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"Video converted! {save_name} generated!",
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
    check_ffmpeg_available(raise_error=True)
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
        command = f'ffmpeg -i "{file_path}" -vf scale={video_width}:{video_height} -c:v libx264 "{save_name}" -hide_banner'
    print("Down-sampling video... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=downsample_video.__name__,
    )


def gif_creator(
    file_path: Union[str, os.PathLike],
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

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
            source=gif_creator.__name__,
        )
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="Start time", value=start_time, min_value=0)
    check_int(name="Duration", value=duration, min_value=1)
    check_int(name="Width", value=width)
    video_meta_data = get_video_meta_data(file_path)
    if (int(start_time) + int(duration)) > video_meta_data["video_length_s"]:
        raise FrameRangeError(
            msg=f'The end of the gif (start time: {start_time} + duration: {duration}) is longer than the {file_path} video: {video_meta_data["video_length_s"]}s',
            source=gif_creator.__name__,
        )

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
        command = f'ffmpeg -ss {start_time} -t {duration} -i "{file_path}" -filter_complex "[0:v] fps=15,scale=w={width}:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" "{save_name}"'
    print("Creating gif sample... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=gif_creator.__name__,
    )


# gif_creator(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4', start_time=5, duration=15, width=600, gpu=False)


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

    check_ffmpeg_available(raise_error=True)
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
            msg="SIMBA ERROR: No files with .{} file extension found in the {} directory".format(
                input_format, directory
            ),
            source=batch_convert_video_format.__name__,
        )
    for file_cnt, file_path in enumerate(video_paths):
        video_timer = SimbaTimer(start=True)
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print(f"Processing video {file_name}...")
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
            command = f'ffmpeg -y -i "{file_path}" -c:v libx264 -crf 21 -preset medium -c:a libmp3lame -b:a 320k "{save_path}"'
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(
            f"Video {file_name} complete, (elapsed time: {video_timer.elapsed_time_str}s) (Video {file_cnt + 1}/{len(video_paths)})..."
        )
    stdout_success(
        msg=f"SIMBA COMPLETE: {str(len(video_paths))} videos converted in {directory} directory!",
        source=batch_convert_video_format.__name__,
    )


# _ = batch_convert_video_format(directory='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_2',input_format='mp4', output_format='avi')


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
    print(f"Processing video {file_name}...")
    video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(
        msg=f"Video {file_name} converted to images in {dir_name} directory!",
        elapsed_time=timer.elapsed_time_str,
        source=extract_frames_single_video.__name__,
    )


# _ = extract_frames_single_video(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4')


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

    check_ffmpeg_available(raise_error=True)
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

    check_ffmpeg_available(raise_error=True)
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

    check_ffmpeg_available(raise_error=True)
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

    check_ffmpeg_available(raise_error=True)
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
    check_int(name="FPS", value=fps, min_value=1)
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
        command = f'ffmpeg -y -r {fps} -f image2 -s {img_h}x{img_w} -i "{ffmpeg_fn}" -vcodec libx264 -b {bitrate}k "{save_path}" -y'
    print(
        f"Creating {os.path.basename(save_path)} from {len(img_paths_in_folder)} images..."
    )
    subprocess.call(command, shell=True)
    stdout_success(msg=f"Video created at {save_path}", source=frames_to_movie.__name__)


# _ = frames_to_movie(directory='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled', fps=15, bitrate=32000, img_format='png')


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
            command = f'ffmpeg -y -i "{video_one_path}" -i "{video_two_path}" -filter_complex "[0:v]scale=-1:{video_meta_data["height"]}[v0];[v0][1:v]hstack=inputs=2" "{save_path}"'

    else:
        if gpu:
            command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_one_path}" -hwaccel auto -c:v h264_cuvid -i "{video_two_path}" -filter_complex "[0:v]scale={video_meta_data["width"]}:-1[v0];[v0][1:v]vstack=inputs=2" -c:v h264_nvenc "{save_path}"'
        else:
            command = f'ffmpeg -y -i "{video_one_path}" -i "{video_two_path}" -filter_complex "[0:v]scale={video_meta_data["width"]}:-1[v0];[v0][1:v]vstack=inputs=2" "{save_path}"'
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


# video_concatenator(video_one_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4',
#                    video_two_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4', resolution='Video_1',
#                    horizontal=True,
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
                    cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "hwupload_cuda,rotate={rotation}*(PI/180),format=nv12|cuda" -c:v h264_nvenc "{save_path}" -y'
                else:
                    cmd = f'ffmpeg -i "{video_path}" -vf "rotate={rotation}*(PI/180)" "{save_path}" -y'
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
            msg=f"SIMBA ERROR: 0 video files in mp4 or avi format found in {directory}",
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
    if (platform.system() == "Darwin") and (multiprocessing.get_start_method() is None):
        multiprocessing.set_start_method("spawn", force=True)
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
    if (platform.system() == "Darwin") and (multiprocessing.get_start_method() is None):
        multiprocessing.set_start_method("spawn", force=False)
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


def crop_single_video_polygon(file_path: Union[str, os.PathLike]) -> None:
    """
    Crop a video based on polygonal regions of interest (ROIs) selected by the user.

    :param  Union[str, os.PathLike] file_path: The path to the input video file.

    .. note::
       This function crops the input video based on polygonal regions of interest (ROIs) selected by the user.
       The user is prompted to select a polygonal ROI on the video frame, and the function then crops the video
       based on the selected ROI. The cropped video is saved with "_polygon_cropped" suffix in the same directory
       as the input video file.

    :example:
    >>> crop_single_video_polygon(file_path='/Users/simon/Desktop/AGGRESSIVITY_4_11_21_Trial_2_camera1_rotated_20240211143355.mp4')
    """

    dir, video_name, _ = get_fn_ext(filepath=file_path)
    save_path = os.path.join(dir, f"{video_name}_polygon_cropped.mp4")
    video_meta_data = get_video_meta_data(video_path=file_path)
    check_file_exist_and_readable(file_path=file_path)
    polygon_selector = ROISelectorPolygon(path=file_path)
    polygon_selector.run()
    timer = SimbaTimer(start=True)
    vertices = polygon_selector.polygon_vertices
    polygon = Polygon(vertices)
    polygons = [polygon for x in range(video_meta_data["frame_count"])]
    if (platform.system() == "Darwin") and (multiprocessing.get_start_method() is None):
        multiprocessing.set_start_method("spawn", force=False)
    polygons = ImageMixin().slice_shapes_in_imgs(
        imgs=file_path, shapes=polygons, verbose=True
    )
    _ = ImageMixin.img_stack_to_video(
        imgs=polygons, save_path=save_path, fps=video_meta_data["fps"]
    )
    timer.stop_timer()
    stdout_success(
        msg=f"Polygon-based cropped saved at to {save_path}",
        elapsed_time=timer.elapsed_time_str,
    )


def crop_multiple_videos_polygons(
    in_dir: Union[str, os.PathLike], out_dir: Union[str, os.PathLike]
) -> None:
    """
    Crop multiple videos based on polygonal regions of interest (ROIs) selected by the user.

    :param  Union[str, os.PathLike] in_dir: The directory containing input video files.
    :param  Union[str, os.PathLike] out_dir: The directory to save the cropped video files.

    .. note::
       This function crops multiple videos based on polygonal ROIs selected by the user.
       The user is asked to define a polygon manually in one video within the input directory.
       The function then crops all the video in the input directory according to the shape defined
       using the first video and saves the videos in the ``out_dir`` with the same filenames as the original videos..

    :example:
    >>> crop_multiple_videos_polygons(in_dir='/Users/simon/Desktop/edited/tests', out_dir='/Users/simon/Desktop')
    """

    check_if_dir_exists(in_dir=in_dir)
    check_if_dir_exists(in_dir=out_dir)
    video_files = find_all_videos_in_directory(directory=in_dir)
    polygon_selector = ROISelectorPolygon(path=os.path.join(in_dir, video_files[0]))
    polygon_selector.run()
    vertices = polygon_selector.polygon_vertices
    polygon = Polygon(vertices)
    timer = SimbaTimer(start=True)
    for video_cnt, video_path in enumerate(video_files):
        print(
            f"Polygon cropping video {video_path} ({video_cnt+1}/{len(video_files)})..."
        )
        video_path = os.path.join(in_dir, video_path)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(out_dir, f"{video_name}.mp4")
        video_meta_data = get_video_meta_data(video_path=video_path)
        polygons = [polygon for x in range(video_meta_data["frame_count"])]
        polygons = ImageMixin().slice_shapes_in_imgs(
            imgs=video_path, shapes=polygons, verbose=False
        )
        _ = ImageMixin.img_stack_to_video(
            imgs=polygons, save_path=save_path, fps=video_meta_data["fps"]
        )
    timer.stop_timer()
    stdout_success(
        msg=f"Polygon-based cropped {len(video_files)} files to directory {out_dir}",
        elapsed_time=timer.elapsed_time_str,
    )


def resize_videos_by_height(
    video_paths: List[Union[str, os.PathLike]],
    height: Union[int, str],
    overwrite: Optional[bool] = False,
    save_dir: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> Union[None, List[Union[None, str, os.PathLike]]]:
    """
    Re-size a list of videos to a specified height while retaining their aspect ratios.

    :param List[Union[str, os.PathLike]] video_paths: List of path to videos.
    :param Union[int, str] height: The height of the output videos. If int, then the height in pixels. If str, then the index in ``video_paths`` from which to grab the height.
    :param Optional[bool] overwrite: If True, then overwrites the original videos. Default False.
    :param Optional[Union[str, os.PathLike]] save_dir: If not None, then the directory where to store the re-sized videos.
    :param Optional[bool] gpu: If True, then use FFmpeg GPU codecs. Default False.
    :param Optional[bool] verbose: If True, prints progress. Default True.
    :return Union[None, List[Union[str, os.PathLike]]]: If save_dir is not None, returns the paths of the re-sized videos. Else returns empty list.

    :example:
    >>> video_paths= ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4']
    >>> _ = resize_videos_by_height(video_paths=video_paths, height=300, overwrite=False, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new')
    """
    timer = SimbaTimer(start=True)
    if (not overwrite) and (save_dir is None):
        raise InvalidInputError(
            msg="Pass a save_dir OR set overwrite to True",
            source=resize_videos_by_height.__name__,
        )
    elif (overwrite) and (save_dir is not None):
        raise InvalidInputError(
            msg="Pass EITHER overwrite as True OR pass a save_dir",
            source=resize_videos_by_height.__name__,
        )
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    check_valid_lst(
        data=video_paths, source=resize_videos_by_height.__name__, min_len=1
    )
    _ = [check_file_exist_and_readable(x) for x in video_paths]
    new_video_paths = []
    if isinstance(height, str):
        check_int(
            name=f"{resize_videos_by_height.__name__} height",
            value=height,
            min_value=0,
            max_value=len(video_paths),
        )
        video_heights = []
        for i in video_paths:
            video_heights.append(get_video_meta_data(video_path=i)["height"])
        height = video_heights[int(height)]
    for cnt, video_path in enumerate(video_paths):
        dir_name, video_name, ext = get_fn_ext(video_path)
        if verbose:
            print(
                f"Resizing height video {video_name} (Video {cnt+1}/{len(video_paths)})..."
            )
        if overwrite:
            dt = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(dir_name, f"{video_name}_{dt}.mp4")
        else:
            save_path = os.path.join(save_dir, f"{video_name}.mp4")
            new_video_paths.append(save_path)
        if gpu:
            cmd = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_path}" -vf scale_npp=-2:{height} -c:v h264_nvenc "{save_path}" -hide_banner -loglevel error -y'
        else:
            cmd = f'ffmpeg -y -i "{video_path}" -vf scale=-2:{height} "{save_path}" -hide_banner -loglevel error -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        if overwrite:
            shutil.copy(save_path, video_path)
            os.remove(save_path)
    timer.stop_timer()
    if verbose:
        print(
            f"Resized height {len(video_paths)} video(s). Elapsed time: {timer.elapsed_time_str}s."
        )
    return new_video_paths


def resize_videos_by_width(
    video_paths: List[Union[str, os.PathLike]],
    width: Union[int, str],
    overwrite: Optional[bool] = False,
    save_dir: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> Union[None, List[Union[None, str, os.PathLike]]]:
    """
    Re-size a list of videos to a specified width while retaining their aspect ratios.

    :param List[Union[str, os.PathLike]] video_paths: List of path to videos.
    :param Union[int, str] width: The width of the output videos. If int, then the width in pixels. If str, then the index in ``video_paths`` from which to grab the width.
    :param Optional[bool] overwrite: If True, then overwrites the original videos. Default False.
    :param Optional[Union[str, os.PathLike]] save_dir: If not None, then the directory where to store the re-sized videos.
    :param Optional[bool] gpu: If True, then use FFmpeg GPU codecs. Default False.
    :param Optional[bool] verbose: If True, prints progress. Default True.
    :return Union[None, List[Union[str, os.PathLike]]]: If save_dir is not None, returns the paths of the re-sized videos. Else returns empty list.

    :example:
    >>> video_paths= ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4']
    >>> _ = resize_videos_by_width(video_paths=video_paths, width=300, overwrite=False, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new')
    """

    timer = SimbaTimer(start=True)
    if (not overwrite) and (save_dir is None):
        raise InvalidInputError(
            msg="Provide a save_dir or set overwrite to True",
            source=resize_videos_by_width.__name__,
        )
    elif (overwrite) and (save_dir is not None):
        raise InvalidInputError(
            msg="Set EITHER overwrite to True OR Provide a save_dir",
            source=resize_videos_by_width.__name__,
        )
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    check_valid_lst(data=video_paths, source=resize_videos_by_width.__name__, min_len=1)
    _ = [check_file_exist_and_readable(x) for x in video_paths]
    new_video_paths = []
    if isinstance(width, str):
        check_int(
            name=f"{resize_videos_by_width.__name__} height",
            value=width,
            min_value=0,
            max_value=len(video_paths),
        )
        video_widths = []
        for i in video_paths:
            video_widths.append(get_video_meta_data(video_path=i)["width"])
        width = video_widths[int(width)]
    for cnt, video_path in enumerate(video_paths):
        dir_name, video_name, ext = get_fn_ext(video_path)
        if verbose:
            print(
                f"Resizing width video {video_name} (Video {cnt+1}/{len(video_paths)})..."
            )
        if overwrite:
            dt = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(dir_name, f"{video_name}_{dt}.mp4")
        else:
            save_path = os.path.join(save_dir, f"{video_name}.mp4")
            new_video_paths.append(save_path)
        if gpu:
            cmd = f'ffmpeg -y -hwaccel auto -i "{video_path}" -vf scale_npp={width}:-2 -c:v h264_nvenc "{save_path}" -hide_banner -loglevel error -y'
        else:
            cmd = f'ffmpeg -y -i "{video_path}" -vf scale={width}:-2 "{save_path}" -hide_banner -loglevel error -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        if overwrite:
            shutil.copy(save_path, video_path)
            os.remove(save_path)
    timer.stop_timer()
    if verbose:
        print(
            f"Resized width {len(video_paths)} video(s). Elapsed time: {timer.elapsed_time_str}s."
        )
    return new_video_paths


def create_blank_video(
    path: Union[str, os.PathLike],
    length: int,
    width: int,
    height: int,
    color: Optional[str] = "black",
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> None:
    """
    Create a "blank" uni-colored video of specified size and length.

    :param Union[str, os.PathLike] path: Location where to store the blank video.
    :param int length: Length of the blank video in seconds.
    :param int width: Width of the blank video in pixels.
    :param int height: Height of the blank video in pixels.
    :param Optional[str] color: Color of the blank video. Default black.
    :param Optional[bool] gpu: If True, then use FFmpeg GPU codecs. Default False.
    :param Optional[bool] verbose: If True, prints progress. Default True.
    :return: None.

    :example:
    >>> _ = create_blank_video(path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4', length=5, width=300, height=400, gpu=False, verbose=True, color='orange')
    """

    check_int(name=f"{create_blank_video.__name__} length", value=length, min_value=1)
    check_int(name=f"{create_blank_video.__name__} width", value=width, min_value=1)
    check_int(name=f"{create_blank_video.__name__} height", value=height, min_value=1)
    check_if_dir_exists(
        in_dir=os.path.dirname(path),
        source=create_blank_video.__name__,
        create_if_not_exist=True,
    )
    timer = SimbaTimer(start=True)
    if verbose:
        print("Creating blank video...")
    if gpu:
        cmd = f'ffmpeg -y -t {length} -f lavfi -i color=c={color}:s={width}x{height} -c:v h264_nvenc -preset slow -tune stillimage -pix_fmt yuv420p "{path}" -hide_banner -loglevel error -y'
    else:
        cmd = f'ffmpeg -y -t {length} -f lavfi -i color=c={color}:s={width}x{height} -c:v libx264 -tune stillimage -pix_fmt yuv420p "{path}" -hide_banner -loglevel error -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    if verbose:
        print(
            f"Blank video complete. Saved at {path}. Elapsed time: {timer.elapsed_time_str}s."
        )


def horizontal_video_concatenator(
    video_paths: List[Union[str, os.PathLike]],
    save_path: Union[str, os.PathLike],
    height_px: Optional[Union[int, str]] = None,
    height_idx: Optional[Union[int, str]] = None,
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> None:
    """
    Concatenates multiple videos horizontally.

    .. image:: _static/img/horizontal_video_concatenator.gif
       :width: 600
       :align: center

    :param List[Union[str, os.PathLike]] video_paths: List of input video file paths.
    :param Union[str, os.PathLike] save_path: File path to save the concatenated video.
    :param Optional[int] height_px: Height of the output video in pixels.
    :param Optional[int] height_idx: Index of the video to use for determining Height.
    :param Optional[bool] gpu: Whether to use GPU-accelerated codec (default: False).
    :param Optional[bool] verbose:Whether to print progress messages (default: True).

    :example:
    >>> video_paths = ['video1.mp4', 'video2.mp4']
    >>> x = horizontal_video_concatenator(video_paths=video_paths, height_px=50, save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/08102021_DOT_Rat7_8(2)_.mp4', gpu=False)
    """
    check_ffmpeg_available()
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None)",
            source=horizontal_video_concatenator.__name__,
        )
    timer = SimbaTimer(start=True)
    check_valid_lst(
        data=video_paths, source=horizontal_video_concatenator.__name__, min_len=2
    )
    check_if_dir_exists(
        in_dir=os.path.dirname(save_path), source=horizontal_video_concatenator.__name__
    )
    video_meta_data = [
        get_video_meta_data(video_path=video_path) for video_path in video_paths
    ]
    if ((height_px is None) and (height_idx is None)) or (
        (height_px is not None) and (height_idx is not None)
    ):
        raise InvalidInputError(
            msg="Provide a height_px OR height_idx",
            source=horizontal_video_concatenator.__name__,
        )
    if height_idx is not None:
        check_int(
            name=f"{horizontal_video_concatenator.__name__} height",
            value=height_idx,
            min_value=0,
            max_value=len(video_paths) - 1,
        )
        height = int(video_meta_data[height_idx]["height"])
    else:
        check_int(
            name=f"{horizontal_video_concatenator.__name__} height",
            value=height_px,
            min_value=1,
        )
        height = int(height_px)
    video_path_str = " ".join([f'-i "{path}"' for path in video_paths])
    codec = "h264_nvenc" if gpu else "libvpx-vp9"
    filter_complex = ";".join(
        [f"[{idx}:v]scale=-1:{height}[v{idx}]" for idx in range(len(video_paths))]
    )
    filter_complex += f";{''.join([f'[v{idx}]' for idx in range(len(video_paths))])}hstack=inputs={len(video_paths)}[v]"
    if verbose:
        print(
            f"Concatenating {len(video_paths)} videos horizontally with a {height} pixel height... "
        )
    cmd = f'ffmpeg {video_path_str} -filter_complex "{filter_complex}" -map "[v]" -c:v {codec} -loglevel error -stats "{save_path}" -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    if verbose:
        print(
            f"Horizontal concatenation complete, saved at {save_path} (elapsed time: {timer.elapsed_time_str}s.)"
        )


def vertical_video_concatenator(
    video_paths: List[Union[str, os.PathLike]],
    save_path: Union[str, os.PathLike],
    width_px: Optional[int] = None,
    width_idx: Optional[int] = None,
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> None:
    """
    Concatenates multiple videos vertically.

    .. image:: _static/img/vertical_video_concatenator.png
       :width: 300
       :align: center

    :param List[Union[str, os.PathLike]] video_paths: List of input video file paths.
    :param Union[str, os.PathLike] save_path: File path to save the concatenated video.
    :param Optional[int] width_px: Width of the output video in pixels.
    :param Optional[int] width_idx: Index of the video to use for determining width.
    :param Optional[bool] gpu: Whether to use GPU-accelerated codec (default: False).
    :param Optional[bool] verbose:Whether to print progress messages (default: True).
    :raises FFMPEGCodecGPUError: If GPU is requested but not available.
    :raises InvalidInputError: If both or neither width_px and width_idx are provided.

    :example:
    >>> video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4',
    >>>                '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4',
    >>>                '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12_1.mp4']
    >>> _ = vertical_video_concatenator(video_paths=video_paths, width_idx=1, save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/08102021_DOT_Rat7_8(2)_.mp4', gpu=False)
    """

    check_ffmpeg_available()
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDIA GPU not available", source=vertical_video_concatenator.__name__
        )
    video_meta_data = [
        get_video_meta_data(video_path=video_path) for video_path in video_paths
    ]
    timer = SimbaTimer(start=True)
    check_valid_lst(
        data=video_paths, source=vertical_video_concatenator.__name__, min_len=2
    )
    check_if_dir_exists(
        in_dir=os.path.dirname(save_path), source=vertical_video_concatenator.__name__
    )
    if ((width_px is None) and (width_idx is None)) or (
        (width_px is not None) and (width_idx is not None)
    ):
        raise InvalidInputError(
            msg="Provide a width_px OR width_idx",
            source=vertical_video_concatenator.__name__,
        )
    if width_idx is not None:
        check_int(
            name=f"{vertical_video_concatenator.__name__} width index",
            value=width_idx,
            min_value=0,
            max_value=len(video_paths) - 1,
        )
        width = int(video_meta_data[width_idx]["width"])
    else:
        check_int(
            name=f"{vertical_video_concatenator.__name__} width",
            value=width_px,
            min_value=1,
        )
        width = int(width_px)
    video_path_str = " ".join([f'-i "{path}"' for path in video_paths])
    codec = "h264_nvenc" if gpu else "libvpx-vp9"
    filter_complex = ";".join(
        [f"[{idx}:v]scale={width}:-1[v{idx}]" for idx in range(len(video_paths))]
    )
    filter_complex += f";{''.join([f'[v{idx}]' for idx in range(len(video_paths))])}"
    filter_complex += f"vstack=inputs={len(video_paths)}[v]"
    if verbose:
        print(
            f"Concatenating {len(video_paths)} videos vertically with a {width} pixel width..."
        )
    cmd = f'ffmpeg {video_path_str} -filter_complex "{filter_complex}" -map "[v]" -c:v {codec} -loglevel error -stats "{save_path}" -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    if verbose:
        print(
            f"Vertical concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)"
        )


def mosaic_concatenator(
    video_paths: List[Union[str, os.PathLike]],
    save_path: Union[str, os.PathLike],
    width_idx: Optional[Union[int, str]] = None,
    width_px: Optional[Union[int, str]] = None,
    height_idx: Optional[Union[int, str]] = None,
    height_px: Optional[Union[int, str]] = None,
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
    uneven_fill_color: Optional[str] = "black",
) -> None:
    """
    Concatenates multiple videos into a mosaic layout.

    .. image:: _static/img/mosaic_concatenator.png
       :width: 600
       :align: center

    .. note::
       if an uneven number of videos, the last index will be filled by ``uneven_fill_color``.

    :param List[Union[str, os.PathLike]] video_paths: List of input video file paths.
    :param Union[str, os.PathLike] save_path: File path to save the concatenated video.
    :param Optional[int] width_px: Width of the output video in pixels.
    :param Optional[int] width_idx: Index of the video to use for determining width.
    :param Optional[int] height_px: Height of the output video panels in pixels.
    :param Optional[int] height_idx: Height of the video to use for determining width.
    :param Optional[bool] gpu: Whether to use GPU-accelerated codec (default: False).
    :param Optional[bool] verbose: Whether to print progress messages (default: True).

    :example:
    >>> video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/2022-06-21_NOB_IOT_23.mp4']
    >>> save_path = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4'
    >>> mosaic_concatenator(video_paths=video_paths, save_path=save_path, width_idx=1, height_idx=1, gpu=False)
    """

    check_ffmpeg_available()
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDIA GPU not available", source=mosaic_concatenator.__name__
        )
    timer = SimbaTimer(start=True)
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    check_valid_lst(
        data=video_paths,
        source=f"{mosaic_concatenator.__name__} video_paths",
        min_len=3,
    )
    video_meta_data = [
        get_video_meta_data(video_path=video_path) for video_path in video_paths
    ]
    max_video_length = max([x["video_length_s"] for x in video_meta_data])
    if ((width_px is None) and (width_idx is None)) or (
        (width_px is not None) and (width_idx is not None)
    ):
        raise InvalidInputError(
            msg="Provide a width_px OR width_idx", source=mosaic_concatenator.__name__
        )
    if ((height_px is None) and (height_idx is None)) or (
        (height_px is not None) and (height_idx is not None)
    ):
        raise InvalidInputError(
            msg="Provide a height_px OR height_idx", source=mosaic_concatenator.__name__
        )
    if width_idx is not None:
        check_int(
            name=f"{vertical_video_concatenator.__name__} width index",
            value=width_idx,
            min_value=1,
            max_value=len(video_paths) - 1,
        )
        width = int(video_meta_data[width_idx]["width"])
    else:
        width = width_px
    if height_idx is not None:
        check_int(
            name=f"{vertical_video_concatenator.__name__} height index",
            value=width_idx,
            min_value=1,
            max_value=len(video_paths) - 1,
        )
        height = int(video_meta_data[width_idx]["height"])
    else:
        height = height_px
    if verbose:
        print(f"Creating mosaic video ...")
    temp_dir = os.path.join(os.path.dirname(video_paths[0]), f"temp_{dt}")
    os.makedirs(temp_dir)
    if not (len(video_paths) % 2) == 0:
        blank_path = os.path.join(temp_dir, f"{dt}.mp4")
        create_blank_video(
            path=blank_path,
            length=max_video_length,
            width=width,
            height=height,
            gpu=gpu,
            verbose=verbose,
            color=uneven_fill_color,
        )
        video_paths.append(blank_path)
    upper_videos, lower_videos = (
        video_paths[: len(video_paths) // 2],
        video_paths[len(video_paths) // 2 :],
    )
    if verbose:
        print("Creating upper mosaic... (Step 1/3)")
    if len(upper_videos) > 1:
        upper_path = os.path.join(temp_dir, "upper.mp4")
        horizontal_video_concatenator(
            video_paths=upper_videos,
            save_path=upper_path,
            gpu=gpu,
            height_px=height,
            verbose=verbose,
        )
    else:
        upper_path = upper_videos[0]
    if verbose:
        print("Creating lower mosaic... (Step 2/3)")
    if len(lower_videos) > 1:
        lower_path = os.path.join(temp_dir, "lower.mp4")
        horizontal_video_concatenator(
            video_paths=lower_videos,
            save_path=lower_path,
            gpu=gpu,
            height_px=height,
            verbose=verbose,
        )
    else:
        lower_path = lower_videos[0]
    panels_meta = [
        get_video_meta_data(video_path=video_path)
        for video_path in [lower_path, upper_path]
    ]
    if verbose:
        print("Joining upper and lower mosaic... (Step 2/3)")
    vertical_video_concatenator(
        video_paths=[upper_path, lower_path],
        save_path=save_path,
        verbose=verbose,
        gpu=gpu,
        width_px=max([x["width"] for x in panels_meta]),
    )
    timer.stop_timer()
    shutil.rmtree(temp_dir)
    if verbose:
        print(
            f"Mosaic concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)"
        )


def mixed_mosaic_concatenator(
    video_paths: List[Union[str, os.PathLike]],
    save_path: Union[str, os.PathLike],
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
    uneven_fill_color: Optional[str] = "black",
) -> None:
    """
    Create a mixed mosaic video by concatenating multiple input videos in a mosaic layout of various sizes.

    .. image:: _static/img/mixed_mosaic_concatenator.png
       :width: 600
       :align: center

    .. note::
       The resolution of the output video is determined by the resolution of the video path at the first index.

       If an uneven number of right-panel videos ( if not (len(video_paths)-1) % 2) == 0), then the last index will be filled by ``uneven_fill_color``.

    :param List[Union[str, os.PathLike]] video_paths: List of input video file paths.
    :param Union[str, os.PathLike] save_path: File path to save the concatenated video.
    :param Optional[bool] gpu: Whether to use GPU-accelerated codec (default: False).
    :param Optional[bool] verbose: Whether to print progress messages (default: True).

    :example:
    >>> video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    >>> save_path = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4'
    >>> mixed_mosaic_concatenator(video_paths=video_paths, save_path=save_path, gpu=False, verbose=True)
    """

    check_ffmpeg_available()
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDIA GPU not available", source=mixed_mosaic_concatenator.__name__
        )
    timer = SimbaTimer(start=True)
    check_valid_lst(
        data=video_paths, source=mixed_mosaic_concatenator.__name__, min_len=2
    )
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    video_meta_data = [
        get_video_meta_data(video_path=video_path) for video_path in video_paths
    ]
    max_video_length = max([x["video_length_s"] for x in video_meta_data])
    check_if_dir_exists(
        in_dir=os.path.dirname(save_path), source=mixed_mosaic_concatenator.__name__
    )
    large_mosaic_path, video_paths = video_paths[0], video_paths[1:]
    mosaic_height = int(video_meta_data[0]["height"] / 2)
    if verbose:
        print("Creating mixed mosaic video... ")
    temp_dir = os.path.join(os.path.dirname(large_mosaic_path), f"temp_{dt}")
    os.makedirs(temp_dir)
    if not (len(video_paths) % 2) == 0:
        blank_path = os.path.join(temp_dir, f"{dt}.mp4")
        create_blank_video(
            path=blank_path,
            length=max_video_length,
            width=get_video_meta_data(video_path=video_paths[-1])["width"],
            height=get_video_meta_data(video_path=video_paths[-1])["height"],
            gpu=gpu,
            verbose=True,
            color=uneven_fill_color,
        )
        video_paths.append(blank_path)
    upper_videos, lower_videos = (
        video_paths[: len(video_paths) // 2],
        video_paths[len(video_paths) // 2 :],
    )
    if verbose:
        print("Creating upper right mosaic ... (Step 1/4)")
    if len(upper_videos) > 1:
        upper_path = os.path.join(temp_dir, "upper.mp4")
        horizontal_video_concatenator(
            video_paths=upper_videos,
            save_path=upper_path,
            gpu=gpu,
            height_px=mosaic_height,
            verbose=verbose,
        )
    else:
        upper_path = upper_videos[0]
    if verbose:
        print("Creating lower right mosaic ... (Step 2/4)")
    if len(lower_videos) > 1:
        lower_path = os.path.join(temp_dir, "lower.mp4")
        horizontal_video_concatenator(
            video_paths=lower_videos,
            save_path=lower_path,
            gpu=gpu,
            verbose=verbose,
            height_px=mosaic_height,
        )
    else:
        lower_path = lower_videos[0]
    panels_meta = [
        get_video_meta_data(video_path=video_path)
        for video_path in [lower_path, upper_path]
    ]
    mosaic_path = os.path.join(temp_dir, "mosaic.mp4")
    if verbose:
        print("Joining upper and lower right mosaic... (Step 3/4)")
    vertical_video_concatenator(
        video_paths=[upper_path, lower_path],
        width_px=max([x["width"] for x in panels_meta]),
        save_path=mosaic_path,
        gpu=gpu,
        verbose=verbose,
    )
    if verbose:
        print("Joining left and right mosaic... (Step 4/4)")
    horizontal_video_concatenator(
        video_paths=[large_mosaic_path, mosaic_path],
        height_idx=0,
        save_path=save_path,
        gpu=gpu,
    )
    timer.stop_timer()
    shutil.rmtree(temp_dir)
    if verbose:
        print(
            f"Mixed mosaic concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)"
        )


def clip_videos_by_frame_ids(
    file_paths: List[Union[str, os.PathLike]],
    frm_ids: List[List[int]],
    save_dir: Optional[Union[str, os.PathLike]] = None,
):
    """
    Clip videos specified by frame IDs (numbers).

    :param List[Union[str, os.PathLike]] file_paths: List of paths to input video files.
    :param List[List[int]] frm_ids: List of lists containing start and end frame IDs for each video.
    :param Optional[Union[str, os.PathLike]] save_dir:  Directory to save the clipped videos. If None, videos will be saved in the same directory as the input videos with frame numbers as suffix.
    :return: None.

    :example:
    >>> file_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/Trial    10.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/Trial    10_1.mp4']
    >>> frm_ids = [[0, 20], [20, 40]]
    >>> clip_videos_by_frame_ids(file_paths=file_paths, frm_ids=frm_ids, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/trial_cnt')
    """

    timer = SimbaTimer(start=True)
    check_valid_lst(
        data=file_paths,
        source=clip_videos_by_frame_ids.__name__,
        valid_dtypes=(str,),
        min_len=1,
    )
    check_valid_lst(
        data=frm_ids,
        source=clip_videos_by_frame_ids.__name__,
        valid_dtypes=(list,),
        exact_len=len(file_paths),
    )
    for cnt, i in enumerate(frm_ids):
        check_valid_lst(
            data=i,
            source=f"clip_videos_by_frame_count.__name_ frm_ids {cnt}",
            valid_dtypes=(int,),
            exact_len=2,
        )
        if i[0] >= i[1]:
            raise FrameRangeError(
                msg=f"Start frame for video {i} is after or the same as the end frame ({i[0]}, {i[1]})",
                source=clip_videos_by_frame_ids.__name__,
            )
        if (i[0] < 0) or (i[1] < 1):
            raise FrameRangeError(
                msg=f"Start frame has to be at least 0 and end frame has to be at least 1",
                source=clip_videos_by_frame_ids.__name__,
            )
    video_meta_data = [get_video_meta_data(video_path=x) for x in file_paths]
    for i in range(len(video_meta_data)):
        if (frm_ids[i][0] > video_meta_data[i]["frame_count"]) or (
            frm_ids[i][1] > video_meta_data[i]["frame_count"]
        ):
            raise FrameRangeError(
                msg=f'Video {i+1} has {video_meta_data[i]["frame_count"]} frames, cannot use start and end frame {frm_ids[i]}',
                source=clip_videos_by_frame_ids.__name__,
            )
    if save_dir is not None:
        check_if_dir_exists(
            in_dir=save_dir,
            source=clip_videos_by_frame_ids.__name__,
            create_if_not_exist=True,
        )
    for cnt, file_path in enumerate(file_paths):
        video_timer = SimbaTimer(start=True)
        dir, video_name, ext = get_fn_ext(filepath=file_path)
        s_f, e_f = frm_ids[cnt][0], frm_ids[cnt][1]
        print(f"Trimming {video_name} from frame {s_f} to frame {e_f}...")
        if save_dir is not None:
            out_path = os.path.join(save_dir, os.path.basename(file_path))
        else:
            out_path = os.path.join(dir, f"{video_name}_{s_f}_{e_f}{ext}")
        cmd = f'ffmpeg -i "{file_path}" -vf "trim={s_f}:{e_f},setpts=PTS-STARTPTS" -c:v libx264 -c:a aac -loglevel error -stats "{out_path}" -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(
            f"Video {video_name} complete (elapsed time {video_timer.elapsed_time_str}s)"
        )
    timer.stop_timer()
    if save_dir is None:
        stdout_success(
            msg=f"{len(file_paths)} video(s) clipped by frame",
            elapsed_time=timer.elapsed_time_str,
        )
    else:
        stdout_success(
            msg=f"{len(file_paths)} video(s) clipped by frame and saved in {save_dir}",
            elapsed_time=timer.elapsed_time_str,
        )


# video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial    10_clipped_gantt.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial    10_clipped.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial    10_clipped_line.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial     3_clipped.mp4']
#
# video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/Trial    10.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial    10_clipped_gantt.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial    10_clipped_line.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/Trial     3_clipped.mp4']
#
# save_path = '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/merge/out.mp4'

#
# video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/gantt_plots/Trial    10.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/Trial    10.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial    10.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial     3.mp4']
# save_path = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4'

# video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/Trial    10.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial    10.mp4',
#                '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/line_plot/Trial     3.mp4']

# mixed_mosaic_concatenator(video_paths=video_paths, save_path=save_path, gpu=False, verbose=True)
