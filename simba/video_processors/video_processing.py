__author__ = "Simon Nilsson"


import functools
import gc
import glob
import multiprocessing
import os
import platform
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from tkinter import *
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from skimage.color import label2rgb
from skimage.segmentation import slic

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_img, check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import ConfigKey, Defaults, Formats, Options, Paths
from simba.utils.errors import (CountError, DirectoryExistError,
                                FFMPEGCodecGPUError, FFMPEGNotFoundError,
                                FileExistError, FrameRangeError,
                                InvalidFileTypeError, InvalidInputError,
                                InvalidVideoFileError, NoDataError,
                                NoFilesFoundError, NotDirectoryError,
                                ResolutionError, SimBAGPUError)
from simba.utils.lookups import (get_ffmpeg_crossfade_methods, get_fonts,
                                 get_named_colors, percent_to_crf_lookup,
                                 percent_to_qv_lk,
                                 video_quality_to_preset_lookup)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video,
    concatenate_videos_in_folder, create_directory,
    find_all_videos_in_directory, find_core_cnt,
    find_files_of_filetypes_in_directory, get_fn_ext, get_video_meta_data,
    read_config_entry, read_config_file, read_frm_of_video,
    read_img_batch_from_video_gpu)
from simba.utils.warnings import (CropWarning, FFMpegCodecWarning,
                                  FileExistWarning, FrameRangeWarning,
                                  GPUToolsWarning, InValidUserInputWarning,
                                  SameInputAndOutputWarning)
from simba.video_processors.async_frame_reader import AsyncVideoFrameReader
from simba.video_processors.extract_frames import video_to_frames
from simba.video_processors.roi_selector import ROISelector
from simba.video_processors.roi_selector_circle import ROISelectorCircle
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon

MAX_FRM_SIZE = 1080, 650

def change_img_format(directory: Union[str, os.PathLike],
                      file_type_in: str,
                      file_type_out: str,
                      verbose: Optional[bool] = False) -> None:
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
        raise NoFilesFoundError(f"SIMBA ERROR: No {file_type_in} files (with .{file_type_in} file extension) found in the {directory} directory", source=change_img_format.__name__,
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

def convert_to_jpeg(path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                    quality: Optional[int] = 95,
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    verbose: Optional[bool] = False) -> None:

    """
    Convert the file type of all image files within a directory to jpeg format of passed quality.

    .. note::
       Quality above 95 should be avoided; 100 disables portions of the JPEG compression algorithm, and results in large files with hardly any gain in image quality

    .. image:: _static/img/convert_to_jpeg.jpeg
       :width: 200
       :align: center

    .. seealso::
       For BMP images, see :func:`~simba.video_processors.video_processing.convert_to_bmp`. For PNG, see :func:`~simba.video_processors.video_processing.convert_to_png`.
       For TIFF, see :func:`~simba.video_processors.video_processing.convert_to_tiff`. For WEBP, see :func:`~simba.video_processors.video_processing.convert_to_webp`.

    :param Union[str, os.PathLike] directory: Path to directory holding image files, a single image file, or a list of paths to image files.
    :param Optional[int] quality: The quality of the output images (0-100).
    :param Optional[bool] save_dir: If not None, then the directory where to store converted images. If None, then stores the images in a subdirectory within the first passed image directory.
    :param Optional[bool] verbose: If True, prints progress. Default False.
    :returns: None. New images are saved in ``save_dir``.

    :example:
    >>> convert_to_jpeg(path='/Users/simon/Desktop/imgs', verbose=False, quality=15)
    >>> convert_to_jpeg(path='/Users/simon/Desktop/my_image.png', verbose=False, quality=15)
    >>> convert_to_jpeg(path=['/Users/simon/Desktop/my_image_1.png', '/Users/simon/Desktop/my_image_2.png'], verbose=True, quality=40)
    >>> convert_to_jpeg(path='/Users/simon/Desktop/imgs', save_dir='/Users/simon/Desktop/test_jpg_/test__/HELLO', verbose=False, quality=15)
    """

    timer = SimbaTimer(start=True)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    if isinstance(path, list):
        check_valid_lst(data=path, source=f'{convert_to_jpeg.__name__} path', valid_dtypes=(str,), min_len=1)
        for i in path: check_file_exist_and_readable(file_path=i)
        file_paths = deepcopy(path)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    elif os.path.isfile(path):
        file_paths = [path]
    else:
        raise InvalidInputError(msg=f'{path} is not a valid file path, directory path, or list of file paths',  source=convert_to_jpeg.__name__)
    directory, _, _ = get_fn_ext(filepath=file_paths[0])
    if save_dir is None:
        save_dir = os.path.join(directory, f'jpeg_{datetime_}')
        os.makedirs(save_dir)
    else:
        check_if_dir_exists(in_dir=save_dir, source=f'{convert_to_jpeg.__name__} save_dir', create_if_not_exist=True)
    check_int(name=f'{convert_to_jpeg.__name__} quality', value=quality, min_value=1, max_value=100)
    print(f"Converting {len(file_paths)} image file(s) to JPEG at quality {quality} ...")
    for file_cnt, file_path in enumerate(file_paths):
        dir, file_name, _ = get_fn_ext(filepath=file_path)
        save_path = os.path.join(save_dir, f'{file_name}.jpeg')
        if verbose:
            print(f"Converting file {file_cnt+1}/{len(file_paths)} ...")
        img = Image.open(file_path)
        if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')
        img.save(save_path, format='JPEG', quality=quality)
    timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to JPEG and stored in {save_dir} directory", source=convert_to_jpeg.__name__, elapsed_time=timer.elapsed_time_str)


def convert_to_bmp(path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   verbose: Optional[bool] = False) -> None:
    """
    Convert images in a directory to BMP format.

    .. seealso::
       For JEPG, see :func:`~simba.video_processors.video_processing.convert_to_jpeg`. For PNG, see :func:`~simba.video_processors.video_processing.convert_to_png`.
       For TIFF, see :func:`~simba.video_processors.video_processing.convert_to_tiff`. For WEBP, see :func:`~simba.video_processors.video_processing.convert_to_webp`.

    :param Union[str, os.PathLike] path: path to directory containing images, or path to a single image file, or a list of paths to image files.
    :param Optional[bool] save_dir: If not None, then the directory where to store converted images. If None, then stores the images in a subdirectory within the first passed image directory.
    :param Optional[bool] verbose: If True, print conversion progress. Default is False.
    :returns: None. New images are saved in ``save_dir``.

    :example:
    >>> convert_to_bmp(path='/Users/simon/Desktop/test_jpg_/landing_40.jpeg', save_dir='/Users/simon/Desktop/test_jpg_/test__')
    """

    timer = SimbaTimer(start=True)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    if isinstance(path, list):
        check_valid_lst(data=path, source=f'{convert_to_bmp.__name__} path', valid_dtypes=(str,), min_len=1)
        for i in path: check_file_exist_and_readable(file_path=i)
        file_paths = deepcopy(path)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path,
                                                          extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value,
                                                          raise_error=True)
    elif os.path.isfile(path):
        file_paths = [path]
    else:
        raise InvalidInputError(msg=f'{path} is not a valid file path or directory path or list of file paths', source=convert_to_bmp.__name__)
    directory, _, _ = get_fn_ext(filepath=file_paths[0])
    if save_dir is None:
        save_dir = os.path.join(directory, f'bmp_{datetime_}')
        os.makedirs(save_dir)
    else:
        check_if_dir_exists(in_dir=save_dir, source=f'{convert_to_bmp.__name__} save_dir', create_if_not_exist=True)
    print(f"Converting {len(file_paths)} image file(s) to BMP ...")
    for file_cnt, file_path in enumerate(file_paths):
        dir, file_name, _ = get_fn_ext(filepath=file_path)
        save_path = os.path.join(save_dir, f'{file_name}.bmp')
        if verbose:
            print(f"Converting file {file_cnt + 1}/{len(file_paths)} ...")
        img = Image.open(file_path)
        if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')
        img.save(save_path, format='BMP')
        timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to BMP and stored in {save_dir} directory",
        source=convert_to_bmp.__name__, elapsed_time=timer.elapsed_time_str)


def convert_to_png(path: Union[str, os.PathLike],
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   verbose: Optional[bool] = False) -> None:
    """
    Convert images to PNG format.

    .. seealso::
       For BMP images, see :func:`~simba.video_processors.video_processing.convert_to_bmp`. For JEPG, see :func:`~simba.video_processors.video_processing.convert_to_jpeg`.
       For TIFF, see :func:`~simba.video_processors.video_processing.convert_to_tiff`. For WEBP, see :func:`~simba.video_processors.video_processing.convert_to_webp`.

    :param Union[str, os.PathLike] path: path to directory containing images, or path to a single image file, or a list of paths to image files.
    :param Optional[bool] save_dir: If not None, then the directory where to store converted images. If None, then stores the images in a subdirectory within the first passed image directory.
    :param Optional[bool] verbose: If True, print conversion progress. Default is False.
    :returns: None. New images are saved in ``save_dir``.

    :example:
    >>> convert_to_png(path=['/Users/simon/Desktop/test_jpg_/test__/landing_40.bmp', '/Users/simon/Desktop/test_jpg_/landing_30.jpeg'], save_dir='/Users/simon/Desktop/test_jpg_/test__/HELLO_3')
    """

    timer = SimbaTimer(start=True)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    if isinstance(path, list):
        check_valid_lst(data=path, source=f'{convert_to_png.__name__} path', valid_dtypes=(str,), min_len=1)
        for i in path: check_file_exist_and_readable(file_path=i)
        file_paths = deepcopy(path)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path,
                                                          extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value,
                                                          raise_error=True)
    elif os.path.isfile(path):
        file_paths = [path]
    else:
        raise InvalidInputError(msg=f'{path} is not a valid file path or directory path or list of file paths', source=convert_to_png.__name__)
    directory, _, _ = get_fn_ext(filepath=file_paths[0])
    if save_dir is None:
        save_dir = os.path.join(directory, f'png_{datetime_}')
        os.makedirs(save_dir)
    else:
        check_if_dir_exists(in_dir=save_dir, source=f'{convert_to_png.__name__} save_dir', create_if_not_exist=True)
    print(f"Converting {len(file_paths)} image file(s) to PNG ...")
    for file_cnt, file_path in enumerate(file_paths):
        dir, file_name, _ = get_fn_ext(filepath=file_path)
        save_path = os.path.join(save_dir, f'{file_name}.png')
        if verbose:
            print(f"Converting file {file_cnt + 1}/{len(file_paths)} ...")
        img = Image.open(file_path)
        if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')
        img.save(save_path, 'PNG')
        timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to PNG and stored in {save_dir} directory",
        source=convert_to_png.__name__, elapsed_time=timer.elapsed_time_str)

def convert_to_tiff(directory: Union[str, os.PathLike],
                    stack: Optional[bool] = False,
                    compression: Literal['raw', 'tiff_deflate', 'tiff_lzw'] = 'raw',
                    verbose: Optional[bool] = False) -> None:
    """
    Convert images in a directory to TIFF format.

     .. seealso::
       For BMP images, see :func:`~simba.video_processors.video_processing.convert_to_bmp`. For JEPG, see :func:`~simba.video_processors.video_processing.convert_to_jpeg`.
       For PNG, see :func:`~simba.video_processors.video_processing.convert_to_png`. For WEBP, see :func:`~simba.video_processors.video_processing.convert_to_webp`.

    :param Union[str, os.PathLike] directory: The directory containing the images.
    :param Optional[bool] stack: If True, create a TIFF stack from the images. Default is False.
    :param Literal['raw', 'tiff_deflate', 'tiff_lzw'] compression: Compression method for the TIFF file. Options are 'raw' (no compression), 'tiff_deflate', and 'tiff_lzw'. Default is 'raw'.
    :param Optional[bool] verbose: If True, print conversion progress. Default is False.
    :returns: None. New images are saved in ``save_dir``.

    :example:
    >>> convert_to_tiff('/Users/simon/Desktop/imgs', stack=True, compression='tiff_lzw')

    """
    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=directory, source=convert_to_tiff.__name__)
    file_paths = find_files_of_filetypes_in_directory(directory=directory, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    check_str(name=f'{convert_to_tiff} compression', value=compression, options=['raw', 'tiff_deflate', 'tiff_lzw'])
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"{len(file_paths)} image file(s) found in {directory}...")
    save_dir = os.path.join(directory, f'tiff_{datetime_}')
    os.makedirs(save_dir)
    if not stack:
        for file_cnt, file_path in enumerate(file_paths):
            dir, file_name, _ = get_fn_ext(filepath=file_path)
            save_path = os.path.join(save_dir, f'{file_name}.tiff')
            if verbose:
                print(f"Converting file {file_cnt + 1}/{len(file_paths)} ...")
            img = Image.open(file_path)
            if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')
            img.save(save_path, format='TIFF', compression=compression)
            timer.stop_timer()
        stdout_success(
            msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to TIFF and stored in {save_dir} directory",
            source=convert_to_tiff.__name__, elapsed_time=timer.elapsed_time_str)
    else:
        save_path = os.path.join(save_dir, f'{os.path.basename(directory)}.tiff')
        file_paths.sort()
        with Image.open(file_paths[0]) as first_img:
            mode, size = first_img.mode, first_img.size
            images = []
            for file_cnt, file in enumerate(file_paths):
                if verbose:
                    print(f"Converting file {file_cnt + 1}/{len(file_paths)} ...")
                with Image.open(file) as img:
                    img = img.convert(mode)
                    img = img.resize(size)
                    images.append(img)
        images[0].save(save_path, save_all=True, append_images=images[1:], format='TIFF', compression=compression)
        timer.stop_timer()
        stdout_success(
            msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to TIFF stack and stored in {save_dir} directory",
            source=convert_to_tiff.__name__, elapsed_time=timer.elapsed_time_str)


def convert_to_webp(path: Union[str, os.PathLike],
                    quality: Optional[int] = 95,
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    verbose: Optional[bool] = True) -> None:

    """
    Convert the file type of all image files within a directory to WEBP format of passed quality.

    .. image:: _static/img/convert_to_webp.webp
       :width: 300
       :align: center

    .. seealso::
       For BMP images, see :func:`~simba.video_processors.video_processing.convert_to_bmp`. For JEPG, see :func:`~simba.video_processors.video_processing.convert_to_jpeg`.
       For PNG, see :func:`~simba.video_processors.video_processing.convert_to_png`. For TIFF, see :func:`~simba.video_processors.video_processing.convert_to_tiff`.

    :param Union[str, os.PathLike] path: Path to directory holding image files
    :param Optional[int] quality: The quality of the output images (0-100).
    :param Optional[bool] save_dir: If not None, then the directory where to store converted images. If None, then stores the images in a subdirectory within the first passed image directory.
    :param Optional[bool] verbose: If True, prints progress. Default False.
    :returns: None. New images are saved in ``save_dir``.

    :example:
    >>> convert_to_webp('/Users/simon/Desktop/imgs', quality=80)
    >>> convert_to_webp(path=['/Users/simon/Desktop/test_jpg_/test__/landing_40.bmp', '/Users/simon/Desktop/test_jpg_/landing_30.jpeg'], save_dir='/Users/simon/Desktop/test_jpg_/test__/HELLO_4')
    """

    timer = SimbaTimer(start=True)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    if isinstance(path, list):
        check_valid_lst(data=path, source=f'{convert_to_png.__name__} path', valid_dtypes=(str,), min_len=1)
        for i in path: check_file_exist_and_readable(file_path=i)
        file_paths = deepcopy(path)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)
    elif os.path.isfile(path):
        file_paths = [path]
    else:
        raise InvalidInputError(msg=f'{path} is not a valid file path or directory path', source=convert_to_webp.__name__)
    directory, _, _ = get_fn_ext(filepath=file_paths[0])
    if save_dir is None:
        save_dir = os.path.join(directory, f'webp_{datetime_}')
        os.makedirs(save_dir)
    else:
        check_if_dir_exists(in_dir=save_dir, source=f'{convert_to_png.__name__} save_dir', create_if_not_exist=True)
    print(f"Converting {len(file_paths)} image file(s) to WEBP ...")
    for file_cnt, file_path in enumerate(file_paths):
        dir, file_name, _ = get_fn_ext(filepath=file_path)
        save_path = os.path.join(save_dir, f'{file_name}.webp')
        img = cv2.imread(file_path)
        if verbose:
            print(f"Converting file {file_cnt+1}/{len(file_paths)} ...")
        cv2.imwrite(save_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: {len(file_paths)} image file(s) in {directory} directory converted to WEBP and stored in {save_dir} directory", source=convert_to_webp.__name__, elapsed_time=timer.elapsed_time_str)

def clahe_enhance_video(file_path: Union[str, os.PathLike],
                         clip_limit: Optional[int] = 2,
                         tile_grid_size: Optional[Tuple[int, int]] = (16, 16),
                         out_path: Optional[Union[str, os.PathLike]] = None) -> None:

    """
    Convert a single video file to clahe-enhanced greyscale .avi file.

    .. image:: _static/img/clahe_enhance_video.gif
       :width: 800
       :align: center

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param Optional[int] clip_limit: CLAHE amplification limit. Inccreased clip limit reduce noise in output. Default: 2.
    :param Optional[Tuple[int]] tile_grid_size: The histogram kernel size.
    :param Optional[Union[str, os.PathLike]] out_path:  The result is saved with prefix``CLAHE_`` in the same directory as in the input file if out_path is not passed. Else saved at the out_path.
    :returns: None.

    :example:
    >>> _ = clahe_enhance_video(file_path: 'project_folder/videos/Video_1.mp4')
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name=f"{clahe_enhance_video.__name__} clip_limit", value=clip_limit, min_value=0)
    video_meta_data = get_video_meta_data(file_path)
    check_valid_tuple(x=tile_grid_size, source=f'{clahe_enhance_video.__name__} tile_grid_size', accepted_lengths=(2,), valid_dtypes=(int,),)
    if (tile_grid_size[0] > video_meta_data["height"]) or ((tile_grid_size[1] > video_meta_data["width"])):
        raise InvalidInputError(msg=f'The tile grid size ({tile_grid_size}) is larger than the video size ({video_meta_data["resolution_str"]})', source=clahe_enhance_video.__name__,)
    dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    if out_path is None:
        save_path = os.path.join(dir, f"CLAHE_{file_name}.avi")
    else:
        check_if_dir_exists(in_dir=os.path.dirname(out_path), source=f'{clahe_enhance_video.__name__} out_path')
        save_path = out_path
    fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
    print(f"Applying CLAHE on video {file_name}, this might take awhile...")
    cap = cv2.VideoCapture(file_path)
    writer = cv2.VideoWriter( save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]), 0)
    clahe_filter = cv2.createCLAHE(clipLimit=int(clip_limit), tileGridSize=tile_grid_size)
    frm_cnt = 0
    try:
        while True:
            ret, img = cap.read()
            if ret:
                frm_cnt += 1
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe_frm = clahe_filter.apply(img)
                writer.write(clahe_frm)
                print(f"CLAHE converted frame {frm_cnt}/{video_meta_data['frame_count']} ({file_name})...")
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
        raise InvalidVideoFileError(msg=f"Could not convert file {file_path} to CLAHE enhanced video", source=clahe_enhance_video.__name__,)


def _clahe_enhance_video_mp_helper(data: tuple,
                                   video_path: str,
                                   clip_limit: int,
                                   temp_dir: str,
                                   tile_grid_size: tuple):

    cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    batch_id, img_idxs = data[0], data[1]
    save_path = os.path.join(temp_dir, f'{batch_id}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter( save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]), 0)
    clahe_filter = cv2.createCLAHE(clipLimit=int(clip_limit), tileGridSize=tile_grid_size)
    start_frm, current_frm, end_frm = img_idxs[0], img_idxs[0], img_idxs[-1]
    cap.set(1, start_frm)

    while current_frm <= end_frm:
        ret, img = cap.read()
        if ret:
            if img.ndim > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_frm = clahe_filter.apply(img)
            writer.write(clahe_frm)
            print(f"CLAHE converted frame {current_frm}/{video_meta_data['frame_count']} (core batch: {batch_id}, video name: {video_meta_data['video_name']})...")
        else:
            FrameRangeWarning(msg=f'Could not read frame {current_frm} in video {video_meta_data["video_name"]}', source=_clahe_enhance_video_mp_helper.__name__)
            break
        current_frm += 1
    writer.release()
    return batch_id

def clahe_enhance_video_mp(file_path: Union[str, os.PathLike],
                           clip_limit: int = 2,
                           tile_grid_size: Tuple[int, int] = (16, 16),
                           out_path: Optional[Union[str, os.PathLike]] = None,
                           gpu: bool = False,
                           core_cnt: int = -1) -> None:

    """
    Convert a single video file to clahe-enhanced greyscale file using multiprocessing.

    .. image:: _static/img/clahe_enhance_video.gif
       :width: 800
       :align: center

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param Optional[int] clip_limit: CLAHE amplification limit. Inccreased clip limit reduce noise in output. Default: 2.
    :param Optional[Tuple[int]] tile_grid_size: The histogram kernel size.
    :param Optional[Union[str, os.PathLike]] out_path:  The result is saved with prefix``CLAHE_`` in the same directory as in the input file if out_path is not passed. Else saved at the out_path.
    :param Union[str, os.PathLike] gpu: If True, try using GPU for segment concatenation.
    :param int core_cnt: Number of cores to use. Pass ``-1`` for all available cores.
    :returns: None.

    :example:
    >>> _ = clahe_enhance_video_mp(file_path= 'project_folder/videos/Video_1.mp4')
    """

    video_timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    check_int(name=f"{clahe_enhance_video_mp.__name__} clip_limit", value=clip_limit, min_value=0)
    video_meta_data = get_video_meta_data(file_path)
    check_valid_tuple(x=tile_grid_size, source=f'{clahe_enhance_video_mp.__name__} tile_grid_size', accepted_lengths=(2,), valid_dtypes=(int,),)
    check_valid_boolean(value=[gpu], source=f'{clahe_enhance_video_mp.__name__} gpu', raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        GPUToolsWarning(msg='No NVIDEA GPU detected and GPU selected. Running without GPU', source=clahe_enhance_video_mp.__name__)
        gpu = False
    if (tile_grid_size[0] > video_meta_data["height"]) or ((tile_grid_size[1] > video_meta_data["width"])):
        raise InvalidInputError(msg=f'The tile grid size ({tile_grid_size}) is larger than the video size ({video_meta_data["resolution_str"]})', source=clahe_enhance_video_mp.__name__,)
    dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    if out_path is None:
        save_path = os.path.join(dir, f"CLAHE_{file_name}.mp4")
    else:
        check_if_dir_exists(in_dir=os.path.dirname(out_path), source=f'{clahe_enhance_video_mp.__name__} out_path')
        save_path = out_path
    if (platform.system() == "Darwin") and (multiprocessing.get_start_method() is None):
        multiprocessing.set_start_method("spawn", force=False)
    tempdir = os.path.join(os.path.dirname(save_path), 'temp', file_name)
    create_directory(paths=tempdir, overwrite=True)
    core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
    frm_idx = list(range(0, video_meta_data['frame_count']))
    frm_idx = np.array_split(frm_idx, core_cnt)
    frm_idx = [(i, list(j)) for i, j in enumerate(frm_idx)]
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_clahe_enhance_video_mp_helper,
                                      video_path=file_path,
                                      clip_limit=clip_limit,
                                      temp_dir=tempdir,
                                      tile_grid_size=tile_grid_size)
        for cnt, result in enumerate(pool.imap(constants, frm_idx, chunksize=1)):
            print(f'Batch {(result + 1)} / {core_cnt} complete...')
    pool.terminate()
    pool.join()
    print(f"Joining {video_meta_data['video_name']} multiprocessed video...")
    concatenate_videos_in_folder(in_folder=tempdir, save_path=save_path, remove_splits=True, gpu=gpu)
    video_timer.stop_timer()
    print(f"CLAHE video {video_meta_data['video_name']} complete (elapsed time: {video_timer.elapsed_time_str}s) ...")


#_ = clahe_enhance_video_mp(file_path= r"D:\EPM_4\original\1.mp4")

def extract_frame_range(file_path: Union[str, os.PathLike],
                        start_frame: int,
                        end_frame: int,
                        save_dir: Optional[Union[str, os.PathLike]] = None,
                        verbose: Optional[bool] = True) -> None:
    """
    Extract a user-defined range of frames from a video file to `png` format.

    :param Union[str, os.PathLike] file_path: Path to video file
    :param int start_frame: First frame in range to extract
    :param int end_frame: Last frame in range to extract.
    :param Optional[Union[str, os.PathLike]] save_dir: Optional save directory. Images are saved in a folder with the suffix `_frames` within the same directory as the video file, if None is passed.
    :param Optional[bool] verbose: Wether to print progress. Default False.
    :returns: None

    :example:
    >>> _ = extract_frame_range(file_path='project_folder/videos/Video_1.mp4', start_frame=100, end_frame=500)
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    video_meta_data = get_video_meta_data(file_path)
    check_int(name="start frame", value=start_frame, min_value=0)
    file_dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    check_int(name="end frame", value=end_frame, max_value=video_meta_data["frame_count"])
    frame_range = list(range(int(start_frame), int(end_frame) + 1))
    if save_dir is None:
        save_dir = os.path.join(file_dir, f"{file_name}_frames")
    else:
        check_if_dir_exists(in_dir=save_dir, source=extract_frame_range.__name__)
    cap = cv2.VideoCapture(file_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for frm_cnt, frm_number in enumerate(frame_range):
        frm_timer = SimbaTimer(start=True)
        frame = read_frm_of_video(video_path=cap, frame_index=frm_cnt)
        frm_save_path = os.path.join(save_dir, f"{frm_number}.png")
        cv2.imwrite(frm_save_path, frame)
        frm_timer.stop_timer()
        if verbose:
            print(f"Frame {frm_number} saved (Frame {frm_cnt}/{len(frame_range)-1}) (elapsed time: {frm_timer.elapsed_time_str}s)")
    timer.stop_timer()
    stdout_success(msg=f"{len(frame_range)-1} frames extracted for video {file_name} saved in {save_dir}", elapsed_time=timer.elapsed_time_str, source=extract_frame_range.__name__)


def change_single_video_fps(file_path: Union[str, os.PathLike],
                            fps: int,
                            gpu: Optional[bool] = False) -> None:

    """
    Change the fps of a single video file. Results are stored in the same directory as in the input file with
    the suffix ``_fps_new_fps``.

    .. note::
       To change the FPS of all videos in a directory, use ``simba.video_processors.video_processing.change_fps_of_multiple_videos``.

    .. video:: _static/img/change_single_video_fps.webm
       :loop:

    :param Union[str, os.PathLike] file_path: Path to video file
    :param int fps: FPS of the new video file.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None

    :example:
    >>> _ = change_single_video_fps(file_path='project_folder/videos/Video_1.mp4', fps=15)
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=change_single_video_fps.__name__)
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="New fps", value=fps)
    video_meta_data = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if int(fps) == int(video_meta_data["fps"]):
        SameInputAndOutputWarning(msg=f"The new fps is the same as the input fps for video {file_name} ({str(fps)})", source=change_single_video_fps.__name__)
    save_path = os.path.join(dir_name, file_name + f"_fps_{fps}{ext}")
    print(f"Converting the FPS to {fps} for video {file_name} ...")
    if ext.lower() == '.webm':
        codec = 'libvpx-vp9'
    elif ext.lower() == '.avi':
        codec = 'mpeg4'
    else:
        codec = 'libx264'
    if os.path.isfile(save_path):
        FileExistWarning(msg=f"Overwriting existing file at {save_path}...", source=change_single_video_fps.__name__,)
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "fps={fps}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
    else:
        command = f'ffmpeg -i "{file_path}" -filter:v fps=fps={fps} -c:v {codec} -c:a aac "{save_path}" -loglevel error -stats -hide_banner -y'
    subprocess.call(command, shell=True)
    timer.stop_timer()
    stdout_success(msg=f'SIMBA COMPLETE: FPS of video {file_name} changed from {str(video_meta_data["fps"])} to {str(fps)} and saved in directory {save_path}', elapsed_time=timer.elapsed_time_str, source=change_single_video_fps.__name__)


def change_fps_of_multiple_videos(path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                                  fps: int,
                                  save_dir: Optional[Union[str, os.PathLike]] = None,
                                  gpu: Optional[bool] = False) -> None:
    """
    Change the fps of all video files in a folder. Results are stored in the same directory as in the input files with
    the suffix ``_fps_new_fps``.

    :param Union[str, os.PathLike] path: Path to video file directory, or a list of video file paths.
    :param int fps: Fps of the new video files.
    :param Optional[bool] save_dir: If not None, then the directory where to store converted videos. If None, then stores the new videos in the same directory as the input video with the ``_new_fps.file_extension`` suffix.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None.

    :example:
    >>> _ = change_fps_of_multiple_videos(path='project_folder/videos/Video_1.mp4', fps=15)
    >>> change_fps_of_multiple_videos(path='/Users/simon/Desktop/imgs', save_dir='/Users/simon/Desktop/imgs_4', fps=15)
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=change_fps_of_multiple_videos.__name__)
    check_int(name="New fps", value=fps, min_value=1)
    if isinstance(path, list):
        check_valid_lst(data=path, source=f'{change_fps_of_multiple_videos.__name__} path', valid_dtypes=(str,), min_len=1)
        for i in path: check_file_exist_and_readable(file_path=i)
        video_paths = deepcopy(path)
    elif os.path.isdir(path):
        video_paths = find_all_videos_in_directory(directory=path, as_dict=True, raise_error=True).values()
    else:
        raise InvalidInputError(msg=f'{path} is not a valid file path or directory path or list of file paths', source=change_fps_of_multiple_videos.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=f'{change_fps_of_multiple_videos.__name__} save_dir', create_if_not_exist=True)
    for file_cnt, file_path in enumerate(video_paths):
        video_timer = SimbaTimer(start=True)
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        video_meta_data = get_video_meta_data(video_path=file_path)
        if int(fps) == int(video_meta_data["fps"]):
            SameInputAndOutputWarning(msg=f"The new FPS ({fps}) is the same or lower than the original FPS ({video_meta_data['fps']}) for video {file_name}", source=change_fps_of_multiple_videos.__name__)
        print(f"Converting FPS from {video_meta_data['fps']} to {fps} for {file_name}...")
        if save_dir is None:
            save_path = os.path.join(dir_name, file_name + f"_fps_{fps}{ext}")
        else:
            save_path = os.path.join(save_dir, f'{file_name}{ext}')
        if ext.lower() == '.webm': codec = 'libvpx-vp9'
        elif ext.lower() == '.avi': codec = 'mpeg4'
        else: codec = 'libx264'
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "fps={fps}" -c:v h264_nvenc -c:a copy "{save_path}" -y'
        else:
            command = f'ffmpeg -i "{file_path}" -filter:v fps=fps={fps} -c:v {codec} "{save_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(command, shell=True)
        video_timer.stop_timer()
        print(f"Video {file_name} complete (saved at {save_path})... (elapsed time: {video_timer.elapsed_time_str}s)")
    timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: FPS of {len(video_paths)} video(s) changed to {fps}", elapsed_time=timer.elapsed_time_str, source=change_fps_of_multiple_videos.__name__,)


def convert_video_powerpoint_compatible_format(file_path: Union[str, os.PathLike], gpu: Optional[bool] = False) -> None:
    """
    Create MS PowerPoint compatible copy of a video file.

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. The result is stored in the same directory as the input file with the ``_powerpointready`` suffix.

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



def video_to_greyscale(file_path: Union[str, os.PathLike], gpu: Optional[bool] = False) -> None:
    """
    Convert a video file to greyscale mp4 format.

    .. image:: _static/img/to_greyscale.gif
       :width: 700
       :align: center

    .. seealso::
       For GPU CuPY acceleration, see :func:`~simba.data_processors.cuda.image.img_stack_to_grayscale_cupy`
       For GPU numba\ CUDA acceleration, see :func:`~simba.data_processors.cuda.image.img_stack_to_grayscale_cuda`,
       For multicore CPU solutions, see :func:`~simba.mixins.image_mixin.ImageMixin.img_stack_to_greyscale` or :func:`~simba.video_processors.video_processing.batch_video_to_greyscale`
       For single-core multivideo solution, see :func:`~simba.video_processors.video_processing.batch_video_to_greyscale`

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: The result is stored in the same directory as the input file with the ``_grayscale.mp4`` suffix.
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


def batch_video_to_greyscale(path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                             save_dir: Optional[Union[str, os.PathLike]] = None,
                             gpu: Optional[bool] = False) -> None:
    """
    Convert a directory of video file to greyscale mp4 format.

    .. image:: _static/img/to_greyscale.gif
       :width: 700
       :align: center

    .. seealso::
       For GPU CuPY acceleration, see :func:`~simba.data_processors.cuda.image.img_stack_to_grayscale_cupy`
       For GPU numba\ CUDA acceleration, see :func:`~simba.data_processors.cuda.image.img_stack_to_grayscale_cuda`,
       For multicore CPU solutions, see :func:`~simba.mixins.image_mixin.ImageMixin.img_stack_to_greyscale` or :func:`~simba.video_processors.video_processing.batch_video_to_greyscale`
       For single-core single video solution, see :func:`~simba.video_processors.video_processing.video_to_greyscale`

    :param Union[str, os.PathLike] path: Path to directory holding video files in color, or a list of file paths to videos in color.
    :param Optional[bool] save_dir: If not None, then the directory where to store converted videos. If None, then stores the new videos in the same directory as the input video with the ``_grayscale`` suffix.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :raise FFMPEGCodecGPUError: If no GPU is found and ``gpu == True``.
    :returns: None.

    :example:
    >>> _ = batch_video_to_greyscale(path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_2')
    >>> batch_video_to_greyscale(path='/Users/simon/Desktop/imgs_4', save_dir='/Users/simon/Desktop/imgs_4/test')
    """
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=batch_video_to_greyscale.__name__)
    timer = SimbaTimer(start=True)
    if isinstance(path, list):
        check_valid_lst(data=path, source=f'{batch_video_to_greyscale.__name__} path', valid_dtypes=(str,), min_len=1)
        for i in path: check_file_exist_and_readable(file_path=i)
        video_paths = deepcopy(path)
    elif os.path.isdir(path):
        check_if_dir_exists(in_dir=path, source=batch_video_to_greyscale.__name__)
        video_paths = find_all_videos_in_directory(directory=path, as_dict=True, raise_error=True).values()
    else:
        raise InvalidInputError(msg=f'{path} is not a valid directory path or a valid list of video paths', source=batch_video_to_greyscale.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=f'{batch_video_to_greyscale.__name__} save_dir', create_if_not_exist=True)
    for file_cnt, file_path in enumerate(video_paths):
        video_timer = SimbaTimer(start=True)
        in_dir, file_name, file_ext = get_fn_ext(filepath=file_path)
        _ = get_video_meta_data(video_path=file_path)
        if save_dir is None:
            save_name = os.path.join(in_dir, f"{file_name}_grayscale.mp4")
        else:
            save_name = os.path.join(save_dir, f'{file_name}{file_ext}')
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "hwupload_cuda,hwdownload,format=nv12,format=gray" -c:v h264_nvenc -c:a copy "{save_name}" -y'
        else:
            command = f'ffmpeg -i "{file_path}" -vf format=gray -c:v libx264 "{save_name}" -hide_banner -loglevel error -y'
        print(f"Converting {file_name} to greyscale (Video {file_cnt+1}/{len(video_paths)})... ")
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(f"Video {save_name} complete (saved at {save_name}), (elapsed time: {video_timer.elapsed_time_str}s)")
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) converted to grayscale!", elapsed_time=timer.elapsed_time_str, source=batch_video_to_greyscale.__name__)


def superimpose_frame_count(file_path: Union[str, os.PathLike],
                            gpu: Optional[bool] = False,
                            font: Optional[str] = 'Arial',
                            font_color: Optional[str] = 'black',
                            bg_color: Optional[str] = 'white',
                            save_path: Optional[Union[str, os.PathLike]] = None,
                            loc: Optional[Literal['top_left', 'top_middle', 'top_right', 'bottom_left', 'bottom_middle', 'bottom_right']] = 'bottom_middle',
                            fontsize: Optional[int] = 20) -> None:

    """
    Superimpose frame count on a video file.

    .. image:: _static/img/superimpose_frame_count.png
       :width: 700
       :align: center

    .. image:: _static/img/superimpose_frame_count.gif
       :width: 500
       :align: center

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :parameter Optional[int] fontsize: The size of the font represetnting the current frame. Default: 20.
    :parameter Optional[Union[str, os.PathLike]] save_path: Optional save location for the video with frame numbers. If None, then the new video is saved in the same directory as the input video with the ``_frame_no`` suffix.
    :parameter Optional[str] font_color: The color of frame number text. Default: 'Black'.
    :parameter Optional[str] loc: The location of the font number text. Options: 'top_left', 'top_middle', 'top_right', 'bottom_left', 'bottom_middle', 'bottom_right'. Default: Bottom middle.
    :parameter Optional[str] bg_color: The color of the box which the frame number is printed in. Default: 'White'.
    :parameter Optional[str] font: The font to use for the frame number. Default Arial.
    :returns: None. The result is stored in the same directory as the input file with the ``_frame_no.mp4`` suffix if ``save_path`` is None.

    :example:
    >>> _ = superimpose_frame_count(file_path='/Users/simon/Downloads/1_LH_0_3.mp4', fontsize=90)
    >>> superimpose_frame_count(file_path='/Users/simon/Desktop/imgs_4/test/Screen Recording 2024-02-26 at 4.43.26 PM_frame_no.mp4', save_path='/Users/simon/Desktop/imgs_4/test/blahhhh.mp4', fontsize=90)
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_int(name=f'{superimpose_frame_count.__name__} fontsize', value=fontsize, min_value=1)
    font_color = ''.join(filter(str.isalnum, font_color)).lower()
    bg_color = ''.join(filter(str.isalnum, bg_color)).lower()
    font_dict = get_fonts()
    check_str(name='font', value=font, options=tuple(font_dict.keys()))
    check_str(name='loc', value=loc, options=('top_left', 'top_middle', 'top_right', 'bottom_left', 'bottom_middle', 'bottom_right'))
    font_path = font_dict[font]
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    if save_path is None:
        save_name = os.path.join(dir, f"{file_name}_frame_no.mp4")
    else:
        check_if_dir_exists(in_dir=os.path.isdir(os.path.dirname(save_path)), source=f'{superimpose_frame_count.__name__} save_path', create_if_not_exist=True)
        save_name = save_path
    if gpu:
        if loc == 'top_left':
            cmd = f'''ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile='{font_path}':text=%{{n}}:x=10:y=10:fontcolor={font_color}:fontsize={fontsize}:box=1:boxcolor={bg_color}@0.5" -c:v h264_nvenc -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'top_center':
            cmd = f'''ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile='{font_path}':text=%{{n}}:x=(w-tw)/2:y=10:fontcolor={font_color}:fontsize={fontsize}:box=1:boxcolor={bg_color}@0.5" -c:v h264_nvenc -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'top_right':
            cmd = f'''ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile='{font_path}':text=%{{n}}:x=w-tw-10:y=10:fontcolor={font_color}:fontsize={fontsize}:box=1:boxcolor={bg_color}@0.5" -c:v h264_nvenc -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'bottom_left':
            cmd = f'''ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile='{font_path}':text=%{{n}}:x=10:y=h-th-10:fontcolor={font_color}:fontsize={fontsize}:box=1:boxcolor={bg_color}@0.5" -c:v h264_nvenc -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'bottom_right':
            cmd = f'''ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile='{font_path}':text=%{{n}}:x=w-tw-10:y=h-th-10:fontcolor={font_color}:fontsize={fontsize}:box=1:boxcolor={bg_color}@0.5" -c:v h264_nvenc -c:a copy -loglevel error -stats "{save_name}" -y'''
        else:
            cmd = f'''ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "drawtext=fontfile='{font_path}':text=%{{n}}:x=(w-tw)/2:y=h-th-10:fontcolor={font_color}:fontsize={fontsize}:box=1:boxcolor={bg_color}@0.5" -c:v h264_nvenc -c:a copy -loglevel error -stats "{save_name}" -y'''
    else:
        if loc == 'top_left':
            cmd = f'''ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile='{font_path}': text='%{{frame_num}}': start_number=0: x=10: y=10: fontcolor={font_color}: fontsize={fontsize}: box=1: boxcolor={bg_color}: boxborderw=5" -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'top_middle':
            cmd = f'''ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile='{font_path}': text='%{{frame_num}}': start_number=0: x=(w-tw)/2: y=10: fontcolor={font_color}: fontsize={fontsize}: box=1: boxcolor={bg_color}: boxborderw=5" -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'bottom_left':
            cmd = f'''ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile='{font_path}': text='%{{frame_num}}': start_number=0: x=10: y=h-th-10: fontcolor={font_color}: fontsize={fontsize}: box=1: boxcolor={bg_color}: boxborderw=5" -c:a copy -loglevel error -stats "{save_name}" -y'''
        elif loc == 'bottom_right':
            cmd = f'''ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile='{font_path}': text='%{{frame_num}}': start_number=0: x=w-tw-10: y=h-th-10: fontcolor={font_color}: fontsize={fontsize}: box=1: boxcolor={bg_color}: boxborderw=5" -c:a copy -loglevel error -stats "{save_name}" -y'''
        else:
            cmd = f'''ffmpeg -y -i "{file_path}" -vf "drawtext=fontfile='{font_path}': text='%{{frame_num}}': start_number=0: x=(w-tw)/2: y=h-th-10: fontcolor={font_color}: fontsize={fontsize}: box=1: boxcolor={bg_color}: boxborderw=5" -c:a copy -loglevel error -stats "{save_name}" -y'''

    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"Superimposed video converted! {save_name} generated!", elapsed_time=timer.elapsed_time_str)

def remove_beginning_of_video(file_path: Union[str, os.PathLike],
                              time: int,
                              save_path: Optional[Union[str, os.PathLike]] = None,
                              gpu: Optional[bool] = False) -> None:

    """
    Remove N seconds from the beginning of a video file.

    :param Union[str, os.PathLike] file_path: Path to video file
    :param int time: Number of seconds to remove from the beginning of the video.
    :param Optional[Union[str, os.PathLike]] save_path: Optional save location for the shortened video. If None, then the new video is saved in the same directory as the input video with the ``_shortened`` suffix.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. If save_path is not passed, the result is stored in the same directory as the input file with the ``_shorten.mp4`` suffix.

    :example:
    >>> _ = remove_beginning_of_video(file_path='project_folder/videos/Video_1.avi', time=10)
    >>> remove_beginning_of_video(file_path=f'/Users/simon/Desktop/imgs_4/test/blahhhh.mp4', save_path='/Users/simon/Desktop/imgs_4/test/CUT.mp4', time=3)
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=remove_beginning_of_video.__name__)
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    video_meta_data = get_video_meta_data(video_path=file_path)
    check_int(name="Cut time", value=time, min_value=1)
    time = int(time)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    if video_meta_data['video_length_s'] <= time:
        raise InvalidInputError(msg=f"The cut time {time}s is invalid for video {file_name} with length {video_meta_data['video_length_s']}s", source=remove_beginning_of_video.__name__)
    if save_path is None:
        save_name = os.path.join(dir, f"{file_name}_shorten.mp4")
    else:
        check_if_dir_exists(in_dir=os.path.isdir(os.path.dirname(save_path)), source=f'{remove_beginning_of_video.__name__} save_path', create_if_not_exist=True)
        save_name = save_path
    if gpu:
        cmd = f'ffmpeg -hwaccel auto -c:v h264_cuvid -ss {time} -i "{file_path}" -c:v h264_nvenc -c:a aac "{save_name}"'
    else:
        cmd = f'ffmpeg -ss {time} -i "{file_path}" -c:v libx264 -c:a aac "{save_name}" -loglevel error -stats -hide_banner -y'
    print(f"Removing initial {time}s from {file_name}... ")
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!", elapsed_time=timer.elapsed_time_str, source=remove_beginning_of_video.__name__)


def clip_video_in_range(file_path: Union[str, os.PathLike],
                        start_time: str,
                        end_time: str,
                        out_dir: Optional[Union[str, os.PathLike]] = None,
                        overwrite: Optional[bool] = False,
                        include_clip_time_in_filename: Optional[bool] = False,
                        gpu: Optional[bool] = False) -> None:
    """
    Clip video within a specific range.

    :param Union[str, os.PathLike] file_path: Path to video file
    :param str start_time: Start time in HH:MM:SS format.
    :param str end_time: End time in HH:MM:SS format.
    :param Optional[Union[str, os.PathLike]] out_dir: If None, then the clip will be stored in the same dir as the input video. If directory, then the location of the output files.
    :param Optional[bool] include_clip_time_in_filename: If True, include the clip start and end in HH-MM-SS format as suffix in the filename. If False, then use integer suffic representing the count.
    :param Optional[bool] overwrite: If True, the overwrite output file if path already exist. If False, then raise FileExistError.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: If None, then the clip will be stored in the same dir as the input video with the ``_clipped.mp4`` suffix. If directory, then the location of the output files.

    :example:
    >>> _ = clip_video_in_range(file_path='project_folder/videos/Video_1.avi', start_time='00:00:05', end_time='00:00:10')
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=clip_video_in_range.__name__)
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    if out_dir is not None:
        check_if_dir_exists(in_dir=out_dir)
        dir = out_dir
    check_if_string_value_is_valid_video_timestamp(value=start_time, name="START TIME")
    check_if_string_value_is_valid_video_timestamp(value=end_time, name="END TIME")
    check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=f"{file_name} timestamps")
    if not include_clip_time_in_filename:
        save_name = os.path.join(dir, file_name + "_clipped.mp4")
    else:
        save_name = os.path.join(dir, file_name + f'_{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.mp4')
    if os.path.isfile(save_name) and (not overwrite):
        raise FileExistError(
            msg=f"SIMBA ERROR: The outfile file already exist: {save_name}.",
            source=clip_video_in_range.__name__,
        )
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -ss {start_time} -to {end_time} -async 1 "{save_name}" -loglevel error -stats -hide_banner -y'
    else:
        command = f'ffmpeg -i "{file_path}" -ss {start_time} -to {end_time} -async 1 -c:v libvpx-vp9 "{save_name}" -loglevel error -stats -hide_banner -y'
    print(f"Clipping video {file_name} between {start_time} and {end_time}... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=clip_video_in_range.__name__,
    )


def downsample_video(file_path: Union[str, os.PathLike],
                     video_height: int,
                     video_width: int,
                     gpu: Optional[bool] = False) -> None:
    """
    Down-sample a video file.

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param int video_height: height of new video.
    :param int video_width: width of new video.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. The result is stored in the same directory as the input file with the ``_downsampled.mp4`` suffix.

    :example:
    >>> _ = downsample_video(file_path='project_folder/videos/Video_1.avi', video_height=600, video_width=400)
    """
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError( msg="No GPU found (as evaluated by nvidea-smi returning None)", source=downsample_video.__name__)
    timer = SimbaTimer(start=True)
    check_int(name="Video height", value=video_height)
    check_int(name="Video width", value=video_width)
    video_height = int(video_height)
    video_width = int(video_width)
    if video_width % 2 != 0:
        video_width += 1
    if video_height % 2 != 0:
        video_height += 1
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + "_downsampled.mp4")
    if os.path.isfile(save_name):
        raise FileExistError("SIMBA ERROR: The outfile file already exist: {}.".format(save_name), source=downsample_video.__name__,)
    if gpu:
        command = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "scale_cuda=w={video_width}:h={video_height}:force_original_aspect_ratio=decrease:flags=bicubic" -c:v h264_nvenc "{save_name}"'
    else:
        command = f'ffmpeg -i "{file_path}" -vf scale={video_width}:{video_height} -c:v libx264 "{save_name}" -loglevel error -stats -hide_banner'
    print("Down-sampling video... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(
        msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!",
        elapsed_time=timer.elapsed_time_str,
        source=downsample_video.__name__,
    )



def gif_creator(file_path: Union[str, os.PathLike],
                start_time: int,
                duration: int,
                width: Optional[int] = None,
                quality: Optional[int] = 100,
                save_path: Optional[Union[str, os.PathLike]] = None,
                fps: Optional[int] = 15,
                gpu: Optional[bool] = False) -> None:

    """
    Create a sample gif from a video file. The result is stored in the same directory as the
    input file with the ``.gif`` file extension if ``save_path`` is passed as None.

    .. note::
       The height is auto-computed to retain aspect ratio.

    .. seealso:
       GIFs are typically for online use but can be costly in terms of time to create and file size. For smaller files,
       consider ``webm`` format which are for online publication and ``simba.video_processors.video_processing.convert_to_webm``.

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param int start_time: Start time of the gif in relation to the video in seconds.
    :param int duration: Duration of the gif.
    :param int width: Width of the gif. If None, then retains the width and height of the input video.
    :param Optional[Union[str, os.PathLike]] save_path: Optional save location for the gif. If None, then the new video is saved in the same directory.
    :param int fps: FPS of the gif. Defaults to 15.
    :param int quality: Quality of the GIF on scale 1-100. Smaller values gives smaller GIFs by filesize with worse quality. Default: 100.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = gif_creator(file_path='project_folder/videos/Video_1.avi', start_time=5, duration=10, width=600)
    >>> gif_creator(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT.mp4', start_time=5, duration=15, width=600, gpu=False, save_path='/Users/simon/Desktop/imgs_4/Test.gif')
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None", source=gif_creator.__name__)
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="Start time", value=start_time, min_value=0)
    check_int(name="Duration", value=duration, min_value=1)
    video_meta_data = get_video_meta_data(file_path)
    if width is None:
        width = video_meta_data['width']
    check_int(name="Width", value=width, min_value=2)
    check_int(name="FPS", value=fps, min_value=1)
    check_int(name="QUALITY", value=quality, min_value=1, max_value=100)
    quality = int((quality - 1) / (100 - 1) * (256 - 1) + 1)
    if width % 2 != 0: width -= 1
    if quality == 1: quality += 1
    if (int(start_time) + int(duration)) > video_meta_data["video_length_s"]:
        raise FrameRangeError(msg=f'The end of the gif (start time: {start_time} + duration: {duration}) is longer than the {file_path} video: {video_meta_data["video_length_s"]}s', source=gif_creator.__name__)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    if save_path is None:
        save_name = os.path.join(dir, f"{file_name}.gif")
    else:
        check_if_dir_exists(in_dir=os.path.isdir(os.path.dirname(save_path)), source=f'{gif_creator.__name__} save_path', create_if_not_exist=True)
        save_name = save_path
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -ss {start_time} -i "{file_path}" -to {duration} -vf "fps=10,scale={width}:-1" -c:v gif -pix_fmt rgb24 -y "{save_name}" -y'
    else:
        command = f'ffmpeg -ss {start_time} -t {duration} -i "{file_path}" -filter_complex "[0:v] fps={fps},scale=w={width}:h=-1:flags=lanczos,split [a][b];[a] palettegen=stats_mode=single:max_colors={quality} [p];[b][p] paletteuse=dither=bayer:bayer_scale=3" "{save_name}" -loglevel error -stats -hide_banner -y'
    print("Creating gif sample... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!", elapsed_time=timer.elapsed_time_str, source=gif_creator.__name__)


def batch_convert_video_format(directory: Union[str, os.PathLike],
                               input_format: str,
                               output_format: str,
                               gpu: Optional[bool] = False) -> None:
    """
    Batch convert all videos in a folder of specific format into a different video format.

    :parameter Union[str, os.PathLike] directory: Path to video file directory.
    :parameter str input_format: Format of the input files (e.g., avi).
    :parameter str output_format: Format of the output files (e.g., mp4).
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. The results are stored in the same directory as the input files.

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
    Extract all frames for all videos in a directory.

    :parameter str directory: Path to directory containing video files.
    :returns: None. Results are stored within sub-directories in the input directory named according to the video files.

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


def extract_frames_single_video(file_path: Union[str, os.PathLike],
                                save_dir: Optional[Union[str, os.PathLike]]) -> None:
    """
    Extract all frames for a single video.

    .. note::
       Image frames are saved as PNG files named with integers in order of appearance, i.e., ``0.png, 1.png ...``

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param Optional[Union[str, os.PathLike]] save_dir: Optional directory where to save the frames. If ``save_dir`` is not passed, results are stored within a subdirectory in the same directory as the input file.
    :returns: None


    :example:
    >>> _ = extract_frames_single_video(file_path='project_folder/videos/Video_1.mp4')
    >>> extract_frames_single_video(file_path='/Users/simon/Desktop/imgs_4/test.mp4', save_dir='/Users/simon/Desktop/imgs_4/frames')
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if save_dir is None:
        save_dir = os.path.join(dir_name, file_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    else:
        check_if_dir_exists(in_dir=save_dir, source=extract_frames_single_video.__name__, create_if_not_exist=True)
    print(f"Processing video {file_name}...")
    video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(msg=f"Video {file_name} converted to images in {dir_name} directory!", elapsed_time=timer.elapsed_time_str, source=extract_frames_single_video.__name__)


#_ = extract_frames_single_video(file_path='/Users/simon/Desktop/video_test/Screen Recording 2024-05-06 at 1.23.31 PM_clipped.mp4')


def multi_split_video(file_path: Union[str, os.PathLike],
                      start_times: List[str],
                      end_times: List[str],
                      out_dir: Optional[Union[str, os.PathLike]] = None,
                      include_clip_time_in_filename: Optional[bool] = False,
                      gpu: Optional[bool] = False) -> None:
    """
    Divide a video file into multiple video files from specified start and stop times.

    :param str file_path: Path to input video file.
    :param List[str] start_times: Start times in HH:MM:SS format.
    :param List[str] end_times: End times in HH:MM:SS format.
    :param Optional[Union[str, os.PathLike]] out_dir: If None, then the clips will be stored in the same dir as the input video. If directory, then the location of the output files.
    :param Optional[bool] include_clip_time_in_filename: If True, include the clip start and end in HH-MM-SS format as suffix in the filename. If False, then use integer suffic representing the count.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None.

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


def crop_single_video(file_path: Union[str, os.PathLike], gpu: Optional[bool] = False) -> None:
    """
    Crop a single video using :func:`~simba.video_processors.roi_selector.ROISelector` interface.

    Results is saved in the same directory as input video with the ``_cropped.mp4`` suffix`.

    .. image:: _static/img/crop_single_video.gif
       :width: 700
       :align: center

    .. seealso::
       To crop multiple videos, see :func:`~simba.video_processors.video_processing.crop_multiple_videos`

    :param str file_path: Path to video file.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. Results is saved in the same directory as input video with the ``_cropped.mp4`` suffix`.

    :example:
    >>> _ = crop_single_video(file_path='project_folder/videos/Video_1.mp4')
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None", source=crop_single_video.__name__)
    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    roi_selector = ROISelector(path=file_path)
    roi_selector.run()
    if ((roi_selector.top_left[0] < 0) or (roi_selector.top_left[1] < 0) or (roi_selector.bottom_right[0] < 0) or (roi_selector.bottom_right[1] < 1)):
        raise CountError( msg="CROP FAILED: Cannot use negative crop coordinates.", source=crop_multiple_videos.__name__)
    save_path = os.path.join(dir_name, f"{file_name}_cropped.mp4")
    if os.path.isfile(save_path):
        raise FileExistError(msg=f"SIMBA ERROR: The out file  already exist: {save_path}.", source=crop_single_video.__name__)
    timer = SimbaTimer(start=True)
    crop_video(video_path=file_path, save_path=save_path, size=(roi_selector.width, roi_selector.height), top_left=(roi_selector.top_left[0], roi_selector.top_left[1]), gpu=gpu, verbose=False)
    timer.stop_timer()
    stdout_success(msg=f"Video {file_name} cropped and saved at {save_path}", elapsed_time=timer.elapsed_time_str, source=crop_single_video.__name__)


#_ = crop_single_video(file_path=r"C:\troubleshooting\mitra\test\501_MA142_Gi_Saline_0515.mp4", gpu=False)

# crop_single_video(file_path=r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4', gpu=False)


def crop_multiple_videos(directory_path: Union[str, os.PathLike], output_path: Union[str, os.PathLike], gpu: Optional[bool] = False) -> None:
    """
    Crop multiple videos in a folder according to crop-coordinates defined in the **first** video.

    .. seealso::
       To crop one video, see :func:`simba.video_processors.video_processing.crop_single_video`

    .. note::
       Calls the :func:`~simba.video_processors.roi_selector.ROISelector` interface.

    :param Union[str, os.PathLike] directory_path: Directory containing input videos.
    :param Union[str, os.PathLike] output_path: Directory where to save the cropped videos.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. Results are stored in passed ``output_path``.

    :example:
    >>> _ = crop_multiple_videos(directory_path='project_folder/videos', output_path='project_folder/videos/my_new_folder')
    >>> _ = crop_multiple_videos(directory_path=r'C:\troubleshooting\mitra\test', output_path=r'C:\troubleshooting\mitra\test\cropped', gpu=True)
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None", source=crop_multiple_videos.__name__)
    check_if_dir_exists(in_dir=directory_path)
    check_if_dir_exists(in_dir=output_path)
    check_valid_boolean(value=[gpu], source=crop_multiple_videos.__name__)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg='Cannot crop using GPU. No GPU detected through FFMPEG', source=crop_multiple_videos.__name__)
    video_paths = find_all_videos_in_directory(directory=directory_path, as_dict=True, raise_error=True)
    video_paths = list(video_paths.values())
    roi_selector = ROISelector(path=video_paths[0])
    roi_selector.run()
    if (roi_selector.width == 0 and roi_selector.height == 0) or (roi_selector.width + roi_selector.height + roi_selector.top_left[0] + roi_selector.top_left[1] == 0):
        raise CountError(msg="CROP FAILED: Cropping height and width are both 0. Please try again.", source=crop_multiple_videos.__name__)
    if ((roi_selector.top_left[0] < 0) or (roi_selector.top_left[1] < 0) or (roi_selector.bottom_right[0] < 0) or (roi_selector.bottom_right[1] < 1)):
        raise CountError(msg=f"CROP FAILED: Cannot use negative crop coordinates. Got top_left: {roi_selector.top_left}, bottom_right: {roi_selector.bottom_right}", source=crop_multiple_videos.__name__)
    timer = SimbaTimer(start=True)
    for file_cnt, file_path in enumerate(video_paths):
        video_timer = SimbaTimer(start=True)
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print(f"Cropping video {file_name} ({file_cnt+1}/{len(video_paths)})...")
        video_meta_data = get_video_meta_data(file_path)
        if (roi_selector.bottom_right[0] > video_meta_data["width"]) or (roi_selector.bottom_right[1] > video_meta_data["height"]):
            raise InvalidInputError(msg=f'Cannot crop video {file_name} of size {video_meta_data["resolution_str"]} at location top left: {roi_selector.top_left}, bottom right: {roi_selector.bottom_right}', source=crop_multiple_videos.__name__)
        save_path = os.path.join(output_path, f"{file_name}_cropped.mp4")
        crop_video(video_path=file_path, save_path=save_path, size=(roi_selector.width, roi_selector.height), top_left=(roi_selector.top_left[0], roi_selector.top_left[1]), gpu=gpu, verbose=False)
        video_timer.stop_timer()
        print(f"Video {file_name} cropped (Video {file_cnt+1}/{len(video_paths)}, elapsed time: {video_timer.elapsed_time_str})")
    timer.stop_timer()
    stdout_success(msg=f"{str(len(video_paths))} videos cropped and saved in {directory_path} directory", elapsed_time=timer.elapsed_time_str, source=crop_multiple_videos.__name__,)


def frames_to_movie(directory: Union[str, os.PathLike],
                    fps: int,
                    quality: int = 60,
                    out_format: Optional[Literal['mp4', 'avi', 'webm']] = 'mp4',
                    gpu: Optional[bool] = False) -> None:
    """
    Merge all image files in a folder to a video file. Video is stored in the same directory as the input folder.

    Image files must be numerically named (e.g., 1.png, 2.png, ...).

    :param directory: Directory containing the images.
    :param fps: Frame rate of the output video.
    :param quality: Integer (1–100) representing output video quality. Higher = better quality + bigger size.
    :param out_format: Format of the output video. One of: 'mp4', 'avi', 'webm'. Default is 'mp4'.
    :param gpu: If True, use NVIDIA GPU codecs (if available). Default is False.
    :return: None. Video is saved to disk.
    """

    import re
    import tempfile
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(
            msg="NVIDIA GPU not available (as evaluated by nvidia-smi returning None)",
            source=frames_to_movie.__name__)

    check_if_dir_exists(in_dir=directory, source=frames_to_movie.__name__)
    check_str(name='out_format', value=out_format, options=['mp4', 'avi', 'webm'])
    check_int(name="FPS", value=fps, min_value=1)
    check_int(name="quality", value=quality, min_value=1)

    # Select codec
    if out_format == 'webm':
        codec = 'libvpx-vp9'
    elif out_format == 'avi':
        codec = 'mpeg4'
    else:
        codec = 'libx264'

    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]

    save_path = os.path.join(os.path.dirname(directory), f"{os.path.basename(directory)}.{out_format}")

    img_paths = find_files_of_filetypes_in_directory(
        directory=directory,
        extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value,
        raise_error=True
    )
    sorted_filepaths = sorted(img_paths, key=natural_sort_key)
    _, start_id, _ = get_fn_ext(filepath=sorted_filepaths[0])

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
        for path in sorted_filepaths:
            safe_path = path.replace('\\', '/')
            tf.write("file '{}'\n".format(safe_path))
        list_path = tf.name

    if gpu:
        cmd = f'ffmpeg -f concat -safe 0 -r {fps} -i "{list_path}" -c:v {codec} -crf {crf} "{save_path}" -loglevel error -stats -hide_banner -y'
    else:
        cmd = f'ffmpeg -f concat -safe 0 -r {fps} -i "{list_path}" -c:v h264_nvenc -crf {crf} "{save_path}" -loglevel error -stats -hide_banner -y'

    subprocess.call(cmd, shell=True)

    os.remove(list_path)
    timer.stop_timer()
    stdout_success(msg=f"Video created at {save_path}", source=frames_to_movie.__name__, elapsed_time=timer.elapsed_time_str)


def video_concatenator(video_one_path: Union[str, os.PathLike],
                       video_two_path: Union[str, os.PathLike],
                       resolution: Optional[Union[int, str]] = 'video 1',
                       horizontal: Optional[bool] = True,
                       gpu: Optional[bool] = False) -> None:
    """
    Concatenate two videos to a single video either horizontally or vertically

    .. image:: _static/img/horizontal_video_concatenator.gif
       :width: 1000
       :align: center

    .. video:: _static/img/vertical_concat.webm
       :width: 800
       :autoplay:
       :loop:

    :param str video_one_path: Path to the first video in the concatenated video
    :param str video_two_path: Path to the second video in the concatenated video
    :param int or str resolution: If str, then the name of the video which resolution you want to retain. E.g., `Video_1`. Else int, representing the video width (if vertical concat) or height (if horizontal concat). Aspect raio will be retained.
    :param horizontal: If true, then horizontal concatenation. Else vertical concatenation.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :returns: None. The video is stored in the same directory as the ``video_one_path`` using the video names concatenated as filename.

    :example:
    >>> video_concatenator(video_one_path='project_folder/videos/Video_1.mp4', video_two_path='project_folder/videos/Video_2.mp4', resolution=800, horizontal=True)
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None", source=video_concatenator.__name__)
    if not check_ffmpeg_available():
        raise FFMPEGNotFoundError(msg="FFMPEG not found on the computer. Install FFMPEG to use the concatenation method.", source=video_concatenator.__name__)
    timer = SimbaTimer(start=True)
    for file_path in [video_one_path, video_two_path]:
        check_file_exist_and_readable(file_path=file_path)
        _ = get_video_meta_data(file_path)
    check_instance(source=video_concatenator.__name__, instance=resolution, accepted_types=(str, int))
    if type(resolution) is int:
        check_int(name='resolution', value=resolution, min_value=1)
        video_meta_data = {}
        video_meta_data["height"] = resolution
        video_meta_data["width"] = resolution
    elif resolution.lower() == "video 1":
        video_meta_data = get_video_meta_data(video_one_path)
    else:
        video_meta_data = get_video_meta_data(video_two_path)
    if video_meta_data["height"] % 2 != 0:
        video_meta_data["height"] = video_meta_data["height"] + 1
    if video_meta_data["width"] % 2 != 0:
        video_meta_data["width"] = video_meta_data["width"] + 1
    dir, file_name_1, _ = get_fn_ext(video_one_path)
    _, file_name_2, _ = get_fn_ext(video_two_path)
    print(f"Concatenating videos {file_name_1} and {file_name_2}...")
    save_path = os.path.join(dir, f"{file_name_1}_{file_name_2}_concat.mp4")
    if horizontal:
        cpu_cmd = (f'ffmpeg -y -i "{video_one_path}" -i "{video_two_path}" '
                   f'-filter_complex "[0:v]scale=ceil(iw/2)*2:{video_meta_data["height"]}[v0];'
                   f'[1:v]scale=ceil(iw/2)*2:{video_meta_data["height"]}[v1];'
                   f'[v0][v1]hstack=inputs=2" "{save_path}" -hide_banner -loglevel error -stats -y')
        gpu_cmd = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_one_path}" -hwaccel auto -c:v h264_cuvid -i "{video_two_path}" -filter_complex "[0:v]scale=-1:{video_meta_data["height"]}[v0];[v0][1:v]hstack=inputs=2" -c:v h264_nvenc "{save_path}" -hide_banner -loglevel debug -stats -y'
    else:
        cpu_cmd = (f'ffmpeg -y -i "{video_one_path}" -i "{video_two_path}" '
                   f'-filter_complex "[0:v]scale={video_meta_data["width"]}:-1[v0];'
                   f'[1:v]scale={video_meta_data["width"]}:-1[v1];'
                   f'[v0][v1]vstack=inputs=2" "{save_path}" -hide_banner -loglevel error -stats -y')
        gpu_cmd = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_one_path}" -hwaccel auto -c:v h264_cuvid -i "{video_two_path}" -filter_complex "[0:v]scale={video_meta_data["width"]}:-1[v0];[v0][1:v]vstack=inputs=2" -c:v h264_nvenc "{save_path}" -hide_banner -loglevel error -stats -y'
    if gpu:
        try:
            subprocess.run(gpu_cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.args)
            FFMpegCodecWarning(msg=f'GPU concatenation for videos failed, reverting to CPU', source=video_concatenator.__name__)
            subprocess.call(cpu_cmd, shell=True)
    else:
        subprocess.call(cpu_cmd, shell=True)

    timer.stop_timer()
    stdout_success(msg=f"Videos concatenated and saved at {save_path}", elapsed_time=timer.elapsed_time_str, source=video_concatenator.__name__)

#video_concatenator(video_one_path=r"C:\troubleshooting\mitra\project_folder\frames\output\pose_ex\704_MA115_Gi_CNO_0521_original.mp4", video_two_path=r"C:\troubleshooting\mitra\project_folder\frames\output\pose_ex\test_1.mp4", resolution='video 1', horizontal=True, gpu=True)




# video_concatenator(video_one_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4',
#                    video_two_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4', resolution='Video_1',
#                    horizontal=True,
#                    gpu=False)


class VideoRotator(ConfigReader):
    """
    GUI Tool for rotating video.


    .. image:: _static/img/VideoRotator.gif
       :width: 700
       :align: center

    :param str input_path: Path to video to rotate.
    :param str output_dir: Directory where to save the rotated video.
    :param Optional[bool] gpu: If True, use FFMPEG NVIDEA GPU codecs. Else CPU codecs.
    :param Optional[bool] gpu: If True, use FFPMPEG. Else, OpenCV.
    :returns: None. Rotated video is saved with the ``_rotated_DATETIME.mp4`` suffix.

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


def extract_frames_from_all_videos_in_directory(config_path: Union[str, os.PathLike],
                                                directory: Union[str, os.PathLike]) -> None:

    """
    Extract all frames from all videos in a directory.

    :param str config_path: path to SimBA project config file in Configparser format.
    :param str directory: path to file or folder containing videos in mp4 and/or avi format.
    :returns: None. The results are saved in the ``project_folder/frames/input directory`` of the SimBA project

    :example:
    >>> extract_frames_from_all_videos_in_directory(config_path='project_folder/project_config.ini', source='/tests/test_data/video_tests')
    """

    timer = SimbaTimer(start=True)
    video_paths = find_all_videos_in_directory(directory=directory, as_dict=True, raise_error=True)
    video_paths = list(video_paths.values())
    config = read_config_file(config_path)
    project_path = read_config_entry(config, "General settings", "project_path", data_type="folder_path")

    print(f"Extracting frames for {len(video_paths)} video(s) into project_folder/frames/input directory...")
    for video_path in video_paths:
        dir_name, video_name, ext = get_fn_ext(video_path)
        save_path = os.path.join(project_path, "frames", "input", video_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            print(f"Frames for video {video_name} already extracted. SimBA is overwriting prior frames...")
        video_to_frames(video_path, save_path, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(f"Frames created for {str(len(video_paths))} videos",elapsed_time=timer.elapsed_time_str, source=extract_frames_from_all_videos_in_directory.__name__)


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

    .. image:: _static/img/crop_single_video_circle.gif
       :width: 600
       :align: center
    .. note::
       This function crops the input video based on circular regions of interest (ROIs) selected by the user.
       The user is prompted to select a circular ROI on the video frame, and the function then crops the video
       based on the selected ROI. The cropped video is saved with "_circle_cropped" suffix in the same directory
       as the input video file.

    :param  Union[str, os.PathLike] file_path: The path to the input video file.

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
    polygon = Polygon([(x + r * np.cos(angle), y + r * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 100)])
    polygons = [polygon for x in range(video_meta_data["frame_count"])]
    if (platform.system() == "Darwin") and (multiprocessing.get_start_method() is None):
        multiprocessing.set_start_method("spawn", force=True)
    polygons = ImageMixin().slice_shapes_in_imgs(imgs=file_path, shapes=polygons, verbose=False)
    time.sleep(3)
    _ = ImageMixin.img_stack_to_video(imgs=polygons, save_path=save_path, fps=video_meta_data["fps"])
    timer.stop_timer()
    stdout_success(msg=f"Circle-based cropped saved at to {save_path}", elapsed_time=timer.elapsed_time_str,)


def crop_multiple_videos_circles(in_dir: Union[str, os.PathLike], out_dir: Union[str, os.PathLike]) -> None:
    """
    Crop multiple videos based on circular regions of interest (ROIs) selected by the user.


    .. image:: _static/img/crop_single_video_circle.gif
       :width: 600
       :align: center

    .. note::
       This function crops multiple videos based on circular ROIs selected by the user.
       The user is asked to define a circle manually in one video within the input directory.
       The function then crops all the video in the input directory according to the shape defined
       using the first video and saves the videos in the ``out_dir`` with the same filenames as the original videos..


    .. seealso::
       :func:`simba.video_processors.video_processing.crop_single_video_circle`



    :param  Union[str, os.PathLike] in_dir: The directory containing input video files.
    :param  Union[str, os.PathLike] out_dir: The directory to save the cropped video files.


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

    .. image:: _static/img/roi_selector_polygon.gif
       :width: 400
       :align: center

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

#crop_single_video_polygon(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/videos/F1 HAB.mp4')

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


def resize_videos_by_height(video_paths: List[Union[str, os.PathLike]],
                            height: Union[int, str],
                            overwrite: Optional[bool] = False,
                            save_dir: Optional[Union[str, os.PathLike]] = None,
                            gpu: Optional[bool] = False,
                            suffix: Optional[str] = None,
                            verbose: Optional[bool] = True) -> Union[None, List[Union[None, str, os.PathLike]]]:
    """
    Re-size a list of videos to a specified height while retaining their aspect ratios.

    :param List[Union[str, os.PathLike]] video_paths: List of path to videos.
    :param Union[int, str] height: The height of the output videos. If int, then the height in pixels. If str, then the index in ``video_paths`` from which to grab the height.
    :param Optional[bool] overwrite: If True, then overwrites the original videos. Default False.
    :param Optional[Union[str, os.PathLike]] save_dir: If not None, then the directory where to store the re-sized videos.
    :param Optional[bool] gpu: If True, then use FFmpeg GPU codecs. Default False.
    :param Optional[bool] suffix: If not None, then stores the resized videos in the same directory as the input videos with the provided suffix.
    :param Optional[bool] verbose: If True, prints progress. Default True.
    :return Union[None, List[Union[str, os.PathLike]]]: If save_dir is not None, returns the paths of the re-sized videos. Else returns empty list.

    :example:
    >>> video_paths= ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4']
    >>> _ = resize_videos_by_height(video_paths=video_paths, height=300, overwrite=False, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new')
    """
    timer = SimbaTimer(start=True)
    if (not overwrite) and (save_dir is None) and (suffix is None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_height.__name__)
    elif (overwrite) and (save_dir is not None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_height.__name__)
    elif (overwrite) and (suffix is not None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_height.__name__)
    elif (save_dir is not None) and (suffix is not None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_height.__name__)
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    check_valid_lst(
        data=video_paths, source=resize_videos_by_height.__name__, min_len=1
    )
    _ = [check_file_exist_and_readable(x) for x in video_paths]
    new_video_paths = []
    if isinstance(height, str):
        check_int(name=f"{resize_videos_by_height.__name__} height",value=height,min_value=0,max_value=len(video_paths))
        video_heights = []
        for i in video_paths:
            video_heights.append(get_video_meta_data(video_path=i)["height"])
        height = video_heights[int(height)]
    for cnt, video_path in enumerate(video_paths):
        dir_name, video_name, ext = get_fn_ext(video_path)
        if verbose:
            print(f"Resizing height video {video_name} (Video {cnt+1}/{len(video_paths)})...")
        if overwrite:
            dt = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(dir_name, f"{video_name}_{dt}.mp4")
        if suffix is None:
            save_path = os.path.join(save_dir, f"{video_name}.mp4")
            new_video_paths.append(save_path)
        else:
            check_str(name=f'{resize_videos_by_height.__name__} suffix', value=suffix)
            save_path = os.path.join(dir_name, f"{video_name}_{suffix}.mp4")
            new_video_paths.append(save_path)
        if gpu:
            cmd = f'ffmpeg -y -hwaccel auto -c:v h264_cuvid -i "{video_path}" -vf scale_npp=-2:{height} -c:v h264_nvenc "{save_path}" -hide_banner -loglevel error -y'
        else:
            cmd = f'ffmpeg -y -i "{video_path}" -vf scale=-2:{height} "{save_path}" -hide_banner -loglevel error -stats -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        if overwrite:
            shutil.copy(save_path, video_path)
            os.remove(save_path)
        if verbose:
            print(f"Resized video {save_path}...")
    timer.stop_timer()
    if verbose:
        print(f"Resized height {len(video_paths)} video(s). Elapsed time: {timer.elapsed_time_str}s.")
    return new_video_paths


def resize_videos_by_width(video_paths: List[Union[str, os.PathLike]],
                           width: Union[int, str],
                           overwrite: Optional[bool] = False,
                           save_dir: Optional[Union[str, os.PathLike]] = None,
                           gpu: Optional[bool] = False,
                           suffix: Optional[str] = None,
                           verbose: Optional[bool] = True) -> Union[None, List[Union[None, str, os.PathLike]]]:
    """
    Re-size a list of videos to a specified width while retaining their aspect ratios.

    :param List[Union[str, os.PathLike]] video_paths: List of path to videos.
    :param Union[int, str] width: The width of the output videos. If int, then the width in pixels. If str, then the index in ``video_paths`` from which to grab the width.
    :param Optional[bool] overwrite: If True, then overwrites the original videos. Default False.
    :param Optional[Union[str, os.PathLike]] save_dir: If not None, then the directory where to store the re-sized videos.
    :param Optional[bool] gpu: If True, then use FFmpeg GPU codecs. Default False.
    :param Optional[bool] suffix: If not None, then stores the resized videos in the same directory as the input videos with the provided suffix.
    :param Optional[bool] verbose: If True, prints progress. Default True.
    :return Union[None, List[Union[str, os.PathLike]]]: If save_dir is not None, returns the paths of the re-sized videos. Else returns empty list.

    :example:
    >>> video_paths= ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4']
    >>> _ = resize_videos_by_width(video_paths=video_paths, width=300, overwrite=False, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new')
    >>> video_paths= [r"D:\videos\4A_Mouse_5-choice_MustTouchTrainingNEWFINAL_a8_grayscale.mp4"]
    >>> resize_videos_by_width(video_paths=video_paths, width=301, overwrite=False, save_dir=r"D:\videos\test", gpu=True)
    """

    timer = SimbaTimer(start=True)
    if (not overwrite) and (save_dir is None) and (suffix is None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_width.__name__)
    elif (overwrite) and (save_dir is not None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_width.__name__)
    elif (overwrite) and (suffix is not None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_width.__name__)
    elif (save_dir is not None) and (suffix is not None):
        raise InvalidInputError(msg="Provide a save_dir, OR set overwrite to True, OR provide a suffix", source=resize_videos_by_width.__name__)
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
        if suffix is None:
            save_path = os.path.join(save_dir, f"{video_name}.mp4")
            new_video_paths.append(save_path)
        else:
            check_str(name=f'{resize_videos_by_width.__name__} suffix', value=suffix)
            save_path = os.path.join(dir_name, f"{video_name}_{suffix}.mp4")
            new_video_paths.append(save_path)
        if gpu:
            cmd = f'ffmpeg -y -hwaccel auto -i "{video_path}" -vf "scale={width}:-2" -c:v h264_nvenc "{save_path}" -hide_banner -loglevel error -stats -y'
        else:
            cmd = f'ffmpeg -y -i "{video_path}" -vf scale={width}:-2 "{save_path}" -hide_banner -loglevel error -stats -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        if overwrite:
            shutil.copy(save_path, video_path)
            os.remove(save_path)
        if verbose:
            print(f"Resized video {save_path}...")
    timer.stop_timer()
    if verbose:
        print(f"Resized width {len(video_paths)} video(s). Elapsed time: {timer.elapsed_time_str}s.")
    return new_video_paths



def create_blank_video(path: Union[str, os.PathLike],
                       length: int,
                       width: int,
                       height: int,
                       color: Optional[str] = "black",
                       gpu: Optional[bool] = False,
                       verbose: Optional[bool] = False) -> None:
    """
    Create a "blank" uni-colored video of specified size and length.

    .. note::
       Useful for when creating image mosaics with un-even number of videos and need a "fill" video in e.g. black color.

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
       :width: 1000
       :align: center

    .. seealso::
       :func:`simba.video_processors.video_processing.vertical_video_concatenator`

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


    .. seealso::
       :func:`simba.video_processors.video_processing.horizontal_video_concatenator`

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
        raise FFMPEGCodecGPUError(msg="NVIDIA GPU not available", source=mixed_mosaic_concatenator.__name__)
    timer = SimbaTimer(start=True)
    check_valid_lst(data=video_paths, source=mixed_mosaic_concatenator.__name__, min_len=2)
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    video_meta_data = [get_video_meta_data(video_path=video_path) for video_path in video_paths]
    max_video_length = max([x["video_length_s"] for x in video_meta_data])
    check_if_dir_exists(in_dir=os.path.dirname(save_path), source=mixed_mosaic_concatenator.__name__)
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


def clip_videos_by_frame_ids(file_paths: List[Union[str, os.PathLike]],
                             frm_ids: List[List[int]],
                             save_dir: Optional[Union[str, os.PathLike]] = None,
                             gpu: Optional[bool] = False):

    """
    Clip videos specified by frame IDs (numbers).

    :param List[Union[str, os.PathLike]] file_paths: List of paths to input video files.
    :param List[List[int]] frm_ids: List of lists containing start and end frame IDs for each video.
    :param Optional[Union[str, os.PathLike]] save_dir:  Directory to save the clipped videos. If None, videos will be saved in the same directory as the input videos with frame numbers as suffix.
    :param Optional[bool] gpu: If True, uses FFMpeg GPU acceleration.
    :return: None.

    :example:
    >>> file_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/Trial    10.mp4',
    >>>               '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/Trial    10_1.mp4']
    >>> frm_ids = [[0, 20], [20, 40]]
    >>> clip_videos_by_frame_ids(file_paths=file_paths, frm_ids=frm_ids, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/trial_cnt')
    """

    timer = SimbaTimer(start=True)
    check_valid_lst(data=file_paths,source=clip_videos_by_frame_ids.__name__,valid_dtypes=(str,),min_len=1)
    check_valid_lst(data=frm_ids,source=clip_videos_by_frame_ids.__name__,valid_dtypes=(list,), exact_len=len(file_paths))
    check_valid_boolean(value=gpu, source=f'{clip_videos_by_frame_ids} gpu')
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError('GPU is passed but no GPU was found on the machine.', source=clip_videos_by_frame_ids.__name__)
    for cnt, i in enumerate(frm_ids):
        check_valid_lst(data=i, source=f"clip_videos_by_frame_count.__name_ frm_ids {cnt}", valid_dtypes=(int,), exact_len=2)
        if i[0] >= i[1]:
            raise FrameRangeError( msg=f"Start frame for video {i} is after or the same as the end frame ({i[0]}, {i[1]})", source=clip_videos_by_frame_ids.__name__)
        if (i[0] < 0) or (i[1] < 1):
            raise FrameRangeError(msg=f"Start frame has to be at least 0 and end frame has to be at least 1", source=clip_videos_by_frame_ids.__name__)
    video_meta_data = [get_video_meta_data(video_path=x) for x in file_paths]
    for i in range(len(video_meta_data)):
        if (frm_ids[i][0] > video_meta_data[i]["frame_count"]) or (frm_ids[i][1] > video_meta_data[i]["frame_count"]):
            raise FrameRangeError(msg=f'Video {i+1} has {video_meta_data[i]["frame_count"]} frames, cannot use start and end frame {frm_ids[i]}', source=clip_videos_by_frame_ids.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=clip_videos_by_frame_ids.__name__, create_if_not_exist=True)
    for cnt, file_path in enumerate(file_paths):
        video_timer = SimbaTimer(start=True)
        dir, video_name, ext = get_fn_ext(filepath=file_path)
        s_f, e_f = frm_ids[cnt][0], frm_ids[cnt][1]
        print(f"Trimming {video_name} from frame {s_f} to frame {e_f}...")
        if save_dir is not None:
            out_path = os.path.join(save_dir, os.path.basename(file_path))
        else:
            out_path = os.path.join(dir, f"{video_name}_{s_f}_{e_f}{ext}")
        if not gpu:
            cmd = f'ffmpeg -i "{file_path}" -vf trim=start_frame={s_f}:end_frame={e_f} -an "{out_path}" -loglevel error -stats -y'
        else:
            cmd = f'ffmpeg -hwaccel auto -i "{file_path}" -vf trim=start_frame={s_f}:end_frame={e_f} -c:v h264_nvenc -an "{out_path}" -loglevel error -stats -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(f"Video {video_name} complete (elapsed time {video_timer.elapsed_time_str}s)")
    timer.stop_timer()
    if save_dir is None:
        stdout_success(msg=f"{len(file_paths)} video(s) clipped by frame", elapsed_time=timer.elapsed_time_str)
    else:
        stdout_success(msg=f"{len(file_paths)} video(s) clipped by frame and saved in {save_dir}", elapsed_time=timer.elapsed_time_str)

def convert_to_mp4(path: Union[str, os.PathLike],
                   codec: Literal['libx265', 'libx264', 'vp9', 'h264_cuvid', 'powerpoint'] = 'libx265',
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   quality: Optional[int] = 60) -> None:
    """
    Convert a directory containing videos, or a single video, to MP4 format using passed quality and codec.

    .. image:: _static/img/convert_to_mp4_1.webp
       :width: 800
       :align: center

    .. image:: _static/img/convert_to_mp4_2.webp
       :width: 800
       :align: center

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['libx265', 'libx264', 'vp9', 'powerpoint'] codec:
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: If CPU codec, then integer representing the quality: 10, 20, 30.. 100. If GPU codec, then string: 'Low', 'Medium', 'High'
    :return: None.

    :example:
    >>> convert_to_mp4(path='/Users/simon/Desktop/video_test', quality="60", codec='vp9')
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_mp4.__name__} codec', value=codec, options=('libx265', 'libx264', 'powerpoint', 'vp9', 'h264_cuvid'))
    check_instance(source=f'{convert_to_mp4.__name__} path', instance=path, accepted_types=(str,))
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    preset_lookup = video_quality_to_preset_lookup()
    if codec != 'h264_cuvid':
        check_int(name=f'{convert_to_mp4.__name__} quality', value=quality)
        quality = crf_lk[str(quality)]
    else:
        if not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None). Select a different codec", source=convert_to_mp4.__name__)
        check_str(name=f'{convert_to_mp4.__name__} quality', value=quality, options=('Low', 'Medium', 'High'))
        quality = preset_lookup[quality]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_mp4.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'mp4_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'mp4_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_mp4.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to MP4 (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.mp4')
        if codec == 'powerpoint':
            cmd = f'ffmpeg -i "{file_path}" -c:v libx264 -preset slow -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf {quality} -c:v libx264 -codec:a aac "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'vp9':
            cmd = f'ffmpeg -i "{file_path}" -c:v libvpx-vp9 -crf {quality} -b:v 0 -an "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'h264_cuvid':
            cmd = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -c:v h264_nvenc -preset {quality} "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path}" -c:v {codec} -crf {quality} -c:a copy -an "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to MP4 and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_mp4.__name__,)

#convert_to_mp4(path=r'D:\water_t_maze\T-Maze\Old Q175 Females\295694f3', quality=60, codec='libx265')

def convert_to_avi(path: Union[str, os.PathLike],
                   codec: Literal['xvid', 'divx', 'mjpeg'] = 'divx',
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   quality: Optional[int] = 60) -> None:

    """
    Convert a directory containing videos, or a single video, to AVI format using passed quality and codec.

    .. image:: _static/img/convert_to_avi.webp
       :width: 800
       :align: center

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['xvid', 'divx', 'mjpeg'] codec: Method to encode the AVI format. Default: xvid.
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.

    :example:
    >>> convert_to_avi(path='/Users/simon/Desktop/video_test/Screen Recording 2024-05-06 at 5.34.34 PM_converted.mp4', quality=100, codec='xvid')
    """


    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_avi.__name__} codec', value=codec, options=('xvid', 'divx', 'mjpeg'))
    check_instance(source=f'{convert_to_avi.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_avi.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    cv_lk = percent_to_qv_lk()
    crf = crf_lk[str(quality)]
    qv = cv_lk[int(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_avi.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'avi_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'avi_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_avi.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to avi (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.avi')
        if codec == 'divx':
            cmd = f'ffmpeg -i "{file_path}" -c:v mpeg4 -crf {crf} -vtag DIVX "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'xvid':
            cmd = f'ffmpeg -i "{file_path}" -c:v libxvid -q:v {qv} "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path}" -c:v mjpeg -q:v {qv} "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to AVI and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_avi.__name__,)


def convert_to_webm(path: Union[str, os.PathLike],
                    codec: Literal['vp8', 'vp9'] = 'vp9',
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    quality: Optional[int] = 60) -> None:

    """
    Convert a directory containing videos, or a single video, to WEBM format using passed quality and codec.

    .. image:: _static/img/convert_to_webm.webp
       :width: 800
       :align: center

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['vp8', 'vp9', 'av1'] codec: Method to encode the WEBM format. Default: vp9.
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.

    :example:
    >>> convert_to_webm(path='/Users/simon/Desktop/video_test/Screen Recording 2024-05-06 at 5.34.34 PM_converted.mp4', quality=100, codec='vp8')
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_webm.__name__} codec', value=codec, options=('vp8', 'vp9'))
    check_instance(source=f'{convert_to_webm.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_webm.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_webm.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'webm_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'webm_{datetime_}')
        os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_webm.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to WEBM (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.webm')
        if codec == 'vp8':
            cmd = f'ffmpeg -i "{file_path}" -c:v libvpx -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'vp9':
            cmd = f'ffmpeg -i "{file_path}" -c:v libvpx-vp9 -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path}" -c:v libaom-av1 -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to WEBM and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_webm.__name__,)

#convert_to_webm(path='/Users/simon/Desktop/video_test/Screen Recording 2024-05-06 at 5.34.34 PM_converted.mp4', quality=100, codec='vp8')


def convert_to_mov(path: Union[str, os.PathLike],
                    codec: Literal['prores', 'animation', 'dnxhd', 'cineform'] = 'prores',
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    quality: Optional[int] = 60) -> None:
    """
    Convert a directory containing videos, or a single video, to MOV format using passed quality and codec.

    :param Union[str, os.PathLike] path: Path to directory or file.
    :param Literal['prores', 'animation'] codec: Method to encode the MOV format. Default: prores.
    :param Optional[Optional[Union[str, os.PathLike]]] save_dir: Directory where to save the converted videos. If None, then creates a directory in the same directory as the input.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :return: None.
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_str(name=f'{convert_to_mov.__name__} codec', value=codec, options=('prores', 'animation', 'cineform', 'dnxhd'))
    check_instance(source=f'{convert_to_mov.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{convert_to_mov.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=convert_to_mov.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'mov_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'mov_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Paths is not a valid file or directory path.', source=convert_to_mov.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Converting video {video_name} to MOV (Video {file_cnt + 1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}.mov')
        if codec == 'prores':
            cmd = f'ffmpeg -i "{file_path}" -c:v prores_ks -profile:v 1 "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == 'dnxhd':
            cmd = f'ffmpeg -i "{file_path}" -c:v dnxhd -profile:v dnxhr_mq "{out_path}" -loglevel error -stats -hide_banner -y'
        elif codec == "cineform":
            cmd = f'ffmpeg -i "{file_path}" -c:v cfhd -compression_level 5 -q:v 3 "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{file_path} -c:v qtrle "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) converted to MOV and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=convert_to_mov.__name__, )

#convert_to_mov(path='/Users/simon/Desktop/video_test/Screen Recording 2024-05-06 at 5.34.34 PM_converted.mp4', codec='cineform')

def superimpose_video_progressbar(video_path: Union[str, os.PathLike],
                                  bar_height: Optional[int] = 10,
                                  color: Optional[str] = 'red',
                                  position: Optional[Literal['top', 'bottom']] = 'bottom',
                                  save_dir: Optional[Union[str, os.PathLike]] = None,
                                  gpu: Optional[bool] = False) -> None:

    """
    Overlay a progress bar on a directory of videos or a single video.

    .. video:: _static/img/overlay_video_progressbar.webm
       :width: 700
       :loop:

    :param Union[str, os.PathLike] video_path: Directory containing video files or a single video file
    :param Optional[int] bar_height: The height of the progressbar in percent of the video height.
    :param Optional[str] color: The color of the progress bar. See simba.utils.lookups.get_color_dict keys for accepted names.
    :param Optional[str] position: The position of the progressbar. Options: 'top', 'bottom'.
    :param Optional[Union[str, os.PathLike]] save_dir: If not None, then saves the videos in the passed directory. Else, in the same directry with the ``_progress_bar`` suffix.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None.

    :example:
    >>> superimpose_video_progressbar(video_path='/home/plateabiosciences/Downloads/031723/3A_Mouse_5-choice_MouseTouchBasic_s9_a6.mp4', gpu=False)
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_video_progressbar.__name__)
    timer = SimbaTimer(start=True)
    color = ''.join(filter(str.isalnum, color)).lower()
    if os.path.isfile(video_path):
        check_file_exist_and_readable(file_path=video_path)
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_path = find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True)
        video_paths = list(video_path.values())
    else:
        raise InvalidInputError(msg='{} is not a valid file path or directory path.', source=superimpose_video_progressbar.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir, _, _ = get_fn_ext(filepath=video_paths[0])
    for cnt, video_path in enumerate(video_paths):
        video_meta_data = get_video_meta_data(video_path=video_path)
        video_length = video_meta_data['video_length_s']
        width, height = video_meta_data['width'], video_meta_data['height']
        bar_height = int(height * (bar_height/100))
        _, video_name, ext = get_fn_ext(filepath=video_path)
        print(f'Inserting progress bar on video {video_name}...')
        save_path = os.path.join(save_dir, f'{video_name}_progress_bar{ext}')
        check_int(name=f'{superimpose_video_progressbar} height', value=bar_height, max_value=height, min_value=1)
        if position == 'bottom':
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -filter_complex "color=c={color}:s={width}x{bar_height}[bar];[0][bar]overlay=-w+(w/{video_length})*t:H-h:shortest=1" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -filter_complex "color=c={color}:s={width}x{bar_height}[bar];[0][bar]overlay=-w+(w/{video_length})*t:H-h:shortest=1" -c:v h264_nvenc -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        else:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -filter_complex "color=c={color}:s={width}x{bar_height}[bar];[0][bar]overlay=-w+(w/{video_length})*t:{bar_height}-h:shortest=1" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{video_path}" -filter_complex "color=c={color}:s={width}x{bar_height}[bar];[0][bar]overlay=-w+(w/{video_length})*t:{bar_height}-h:shortest=1" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) saved with progressbar in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=superimpose_video_progressbar.__name__, )



def crossfade_two_videos(video_path_1: Union[str, os.PathLike],
                         video_path_2: Union[str, os.PathLike],
                         crossfade_duration: Optional[int] = 2,
                         crossfade_method: Optional[str] = 'fade',
                         crossfade_offset: Optional[int] = 2,
                         quality: Optional[int] = 60,
                         out_format: Optional[Literal['mp4', 'avi', 'webm']] = 'mp4',
                         save_path: Optional[Union[str, os.PathLike]] = None,
                         gpu: Optional[bool] = False):
    """
    Cross-fade two videos.

    .. video:: _static/img/crossfade_two_videos.webm
       :loop:

    .. note::
       See ``simba.utils.lookups.get_ffmpeg_crossfade_methods`` for named crossfade methods.
       See `https://trac.ffmpeg.org/wiki/Xfade <https://trac.ffmpeg.org/wiki/Xfade>`__. for visualizations of named crossfade methods,

    :param Union[str, os.PathLike] video_path_1: Path to the first video on disk.
    :param Union[str, os.PathLike] video_path_2: Path to the second video on disk.
    :param Optional[int] crossfade_duration: The duration of the crossfade.
    :param Optional[str] crossfade_method: The crossfade method. For accepted methods, see ``simba.utils.lookups.get_ffmpeg_crossfade_methods``.
    :param Optional[int] crossfade_offset: The time in seconds into the first video before the crossfade duration begins.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to save the crossfaded video. If None, then saves the video in the same directory as ``video_path_1`` with ``_crossfade`` suffix.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.
    :return: None.

    :example:
    >>> crossfade_two_videos(video_path_1='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/1.mp4', video_path_2='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/1.mp4', crossfade_duration=5, crossfade_method='zoomin', save_path='/Users/simon/Desktop/cross_test.mp4')
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_video_progressbar.__name__)
    timer = SimbaTimer(start=True)
    video_1_meta = get_video_meta_data(video_path=video_path_1)
    video_2_meta = get_video_meta_data(video_path=video_path_2)
    if video_1_meta['resolution_str'] != video_1_meta['resolution_str']:
        raise InvalidInputError(msg=f'Video 1 and Video 2 needs to be the same resolution, got {video_2_meta["resolution_str"]} and {video_1_meta["resolution_str"]}', source=crossfade_two_videos.__name__)
    crossfade_offset_methods = get_ffmpeg_crossfade_methods()
    check_str(name=f'{crossfade_method} crossfade_method', value=crossfade_method, options=crossfade_offset_methods)
    check_str(name=f'{crossfade_method} out_format', value=out_format, options=['mp4', 'avi', 'webm'])
    check_int(name=f'{crossfade_two_videos.__name__} crossfade_duration', value=crossfade_duration, min_value=1, max_value=video_2_meta['video_length_s'])
    check_int(name=f'{crossfade_two_videos.__name__} crossfade_offset', value=crossfade_offset, min_value=0, max_value=video_1_meta['video_length_s'])
    check_int(name=f'{crossfade_two_videos.__name__} quality', value=quality)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    for video_path in [video_path_1, video_path_2]:
        video_meta_data = get_video_meta_data(video_path=video_path)
        if video_meta_data['video_length_s'] < crossfade_duration:
            raise FrameRangeError(msg=f'Video {video_meta_data["video_name"]} is shorter {video_meta_data["video_length_s"]} than the crossfade duration {crossfade_duration}.', source=crossfade_two_videos.__name__)
        if video_meta_data['video_length_s'] < crossfade_offset:
            raise FrameRangeError(msg=f'Video {video_meta_data["video_name"]} is shorter {video_meta_data["video_length_s"]} than the crossfade offset {crossfade_offset}.', source=crossfade_two_videos.__name__)
    dir_1, video_name_1, ext_1 = get_fn_ext(filepath=video_path_1)
    dir_2, video_name_2, ext_2 = get_fn_ext(filepath=video_path_2)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
    else:
        save_path = os.path.join(dir_1, f'{video_name_1}_{video_name_2}_crossfade.{out_format}')
    if not gpu:
        cmd = f'ffmpeg -i "{video_path_1}" -i "{video_path_2}" -filter_complex "xfade=transition={crossfade_method}:offset={crossfade_offset}:duration={crossfade_duration}" -crf {crf} "{save_path}" -loglevel error -stats -hide_banner -y'
    else:
        cmd = f'ffmpeg -i "{video_path_1}" -i "{video_path_2}" -hwaccel auto -filter_complex "xfade=transition={crossfade_method}:offset={crossfade_offset}:duration={crossfade_duration}" -c:v h264_nvenc "{save_path}" -loglevel error -stats -hide_banner -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Cross-faded video saved at {save_path}', elapsed_time=timer.elapsed_time_str)

def watermark_video(video_path: Union[str, os.PathLike],
                    watermark_path: Union[str, os.PathLike],
                    position: Optional[Literal['top_left', 'bottom_right', 'top_right', 'bottom_left', 'center']] = 'top_left',
                    opacity: Optional[float] = 0.5,
                    scale: Optional[float] = 0.05,
                    save_dir: Optional[Union[str, os.PathLike]] = None,
                    gpu: Optional[bool] = False) -> None:
    """
    Watermark a video file or a directory of video files with specified watermark size, opacity, and location.

    .. video:: _static/img/watermark_video.webm
       :width: 900
       :loop:

    :param Union[str, os.PathLike] video_path: The path to the video file or path to directory with video files.
    :param Union[str, os.PathLike] watermark_path: The path to the watermark .png file.
    :param Optional[str] position: The position of the watermark. Options: 'top_left', 'bottom_right', 'top_right', 'bottom_left', 'center'
    :param Optional[float] opacity: The opacity of the watermark as a value between 0-1.0. 1.0 meaning the same as input image. Default: 0.5.
    :param Optional[float] scale: The scale of the watermark as a ratio os the image size. Default: 0.05.
    :param Optional[Union[str, os.PathLike]] save_dir: The location where to save the watermarked video. If None, then saves the video in the same directory as the first video.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None

    :example:
    >>> watermark_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/multi_animal_dlc_two_c57/project_folder/videos/watermark/Together_1_powerpointready.mp4', watermark_path='/Users/simon/Desktop/splash.png', position='top_left', opacity=1.0, scale=0.2)
    """
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=watermark_video.__name__)
    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'bottom_right', 'top_right', 'bottom_left', 'center']
    check_float(name=f'{watermark_video.__name__} opacity', value=opacity, min_value=0.001, max_value=1.0)
    check_float(name=f'{watermark_video.__name__} scale', value=scale, min_value=0.001, max_value=0.999)
    check_str(name=f'{watermark_video.__name__} position', value=position, options=POSITIONS)
    check_file_exist_and_readable(file_path=watermark_path)

    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=watermark_video.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, video_ext = get_fn_ext(video_path)
        _ = get_video_meta_data(video_path=video_path)
        print(f'Watermarking {video_name} (Video {file_cnt+1}/{len(video_paths)})...')
        out_path = os.path.join(save_dir, f'{video_name}_watermarked{video_ext}')
        if position == POSITIONS[0]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=0:0" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=0:0" -c:a copy -c:v h264_nvenc "{out_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[1]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=W-w:H-h" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=W-w:H-h" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[2]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=W-w-0:0" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=W-w-0:0" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[3]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=0:H-h-0" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=0:H-h-0" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=(W-w)/2:(H-h)/2" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{watermark_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay=(W-w)/2:(H-h)/2" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'{len(video_paths)} watermarked video(s) saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

def superimpose_overlay_video(video_path: Union[str, os.PathLike],
                              overlay_video_path: Union[str, os.PathLike],
                              position: Optional[Literal['top_left', 'bottom_right', 'top_right', 'bottom_left', 'center']] = 'top_left',
                              opacity: Optional[float] = 0.5,
                              scale: Optional[float] = 0.05,
                              save_dir: Optional[Union[str, os.PathLike]] = None,
                              gpu: Optional[bool] = False) -> None:
    """
    Inset a video overlay on a second video with specified size, opacity, and location.

    .. video:: _static/img/inset_overlay_video.webm
       :width: 700
       :loop:

    :param Union[str, os.PathLike] video_path: The path to the video file.
    :param Union[str, os.PathLike] overlay_video_path: The path to video to be inserted into the ``video_path`` video.
    :param Optional[str] position: The position of the inset overlay video. Options: 'top_left', 'bottom_right', 'top_right', 'bottom_left', 'center'
    :param Optional[float] opacity: The opacity of the inset overlay video as a value between 0-1.0. 1.0 meaning the same as input image. Default: 0.5.
    :param Optional[float] scale: The scale of the inset overlay video as a ratio os the image size. Default: 0.05.
    :param Optional[Union[str, os.PathLike]] save_dir: The location where to save the output video. If None, then saves the video in the same directory as the first video.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None

    :example:
    >>> superimpose_overlay_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/multi_animal_dlc_two_c57/project_folder/videos/watermark/Together_1_powerpointready.mp4', overlay_video_path='/Users/simon/Desktop/splash.png', position='top_left', opacity=1.0, scale=0.2)
    """

    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_overlay_video.__name__)
    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'bottom_right', 'top_right', 'bottom_left', 'center']
    check_float(name=f'{superimpose_overlay_video.__name__} opacity', value=opacity, min_value=0.001, max_value=1.0)
    check_float(name=f'{superimpose_overlay_video.__name__} scale', value=scale, min_value=0.001, max_value=0.999)
    check_str(name=f'{superimpose_overlay_video.__name__} position', value=position, options=POSITIONS)
    check_file_exist_and_readable(file_path=video_path)
    check_file_exist_and_readable(file_path=overlay_video_path)

    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=superimpose_overlay_video.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, video_ext = get_fn_ext(video_path)
        _ = get_video_meta_data(video_path=video_path)
        print(f'Inserting overlay onto {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        out_path = os.path.join(save_dir, f'{video_name}_inset_overlay{video_ext}')
        if position == POSITIONS[0]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=0:0" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=0:0" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[1]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=W-w:H-h" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=W-w:H-h" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[2]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=W-w:0" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=W-w:0" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[3]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=0:H-h" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=0:H-h" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=(W-w)/2:(H-h)/2" "{out_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=(W-w)/2:(H-h)/2" -c:v h264_nvenc -c:a copy "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'{len(video_paths)} overlay video(s) saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

def roi_blurbox(video_path: Union[str, os.PathLike],
                blur_level: Optional[float] = 0.02,
                invert: Optional[bool] = False,
                save_path: Optional[Union[str, os.PathLike]] = None) -> None:

    """
    Blurs either the selected or unselected portion of a user-specified region-of-interest according to the passed blur level.
    Higher blur levels produces more opaque regions.

    .. video:: _static/img/roi_blurbox.webm
       :loop:

    :param Union[str, os.PathLike] video_path: The path to the video on disk
    :param Optional[float] blur_level: The level of the blur as a ratio 0-1.0.
    :param Optional[bool] invert: If True, blurs the unselected region. If False, blurs the selected region.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to save the blurred video. If None, then saves the blurred video in the same directory as the input video with the ``_blurred`` suffix.
    :return: None

    :example:
    >>> roi_blurbox(video_path='/Users/simon/Downloads/1_LH_clipped_downsampled.mp4', blur_level=0.2, invert=True)
    """

    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    check_float(name=f'{roi_blurbox.__name__} blur_level', value=blur_level, min_value=0.001, max_value=1.0)
    check_file_exist_and_readable(file_path=video_path)
    dir, video_name, ext = get_fn_ext(video_path)
    _ = get_video_meta_data(video_path=video_path)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=roi_blurbox.__name__)
    else:
        save_path = os.path.join(dir, f'{video_name}_blurred{ext}')
    roi_selector = ROISelector(path=video_path)
    roi_selector.run()
    w, h = roi_selector.width, roi_selector.height
    top_left_x, top_left_y = roi_selector.top_left
    max_blur_value = int(min(w, h) / 2) / 2
    blur_level = int(max_blur_value * blur_level)
    if not invert:
        cmd = f'ffmpeg -i "{video_path}" -filter_complex "[0:v]crop={w}:{h}:{top_left_x}:{top_left_y},boxblur={int(blur_level)}:10[fg]; [0:v][fg]overlay={top_left_x}:{top_left_y}[v]" -map "[v]" "{save_path}" -loglevel error -stats -hide_banner -y'
    else:
        cmd = f'ffmpeg -i "{video_path}" -filter_complex "[0:v]boxblur={blur_level}[bg];[0:v]crop={w}:{h}:{top_left_x}:{top_left_y}[fg];[bg][fg]overlay={top_left_x}:{top_left_y}" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Blurred {video_name} video saved in {save_path}', elapsed_time=timer.elapsed_time_str)


def superimpose_elapsed_time(video_path: Union[str, os.PathLike],
                             font: Optional[str] = 'Arial',
                             font_size: Optional[int] = 30,
                             font_color: Optional[str] = 'white',
                             font_border_color: Optional[str] = 'black',
                             time_format: Optional[Literal['MM:SS', 'HH:MM:SS', 'SS.MMMMMM', 'HH:MM:SS.MMMM']] = 'HH:MM:SS.MMMM',
                             font_border_width: Optional[int] = 2,
                             position: Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']] = 'top_left',
                             save_dir: Optional[Union[str, os.PathLike]] = None,
                             gpu: Optional[bool] = False) -> None:
    """
    Superimposes elapsed time on the given video file(s) and saves the modified video(s).

    .. video:: _static/img/superimpose_elapsed_time.webm
       :width: 900
       :loop:

    :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
    :param Optional[int] font_size: Font size for the elapsed time text. Default 30.
    :param Optional[str] font_color:  Font color for the elapsed time text. Default white
    :param Optional[str] font_border_color: Font border color for the elapsed time text. Default black.
    :param Optional[int] font_border_width: Font border width for the elapsed time text in pixels. Default 2.
    :param Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle_top', 'middle_bottom']] position: Position where the elapsed time text will be superimposed. Default ``top_left``.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where the modified video(s) will be saved. If not provided, the directory of the input video(s) will be used.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None

    :example:
    >>> superimpose_elapsed_time(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_4/1.mp4', position='middle_top', font_color='black', font_border_color='pink', font_border_width=5, font_size=30)
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_overlay_video.__name__)
    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']
    check_str(name=f'{superimpose_elapsed_time.__name__} position', value=position, options=POSITIONS)
    check_str(name=f'{superimpose_elapsed_time.__name__} time_format', value=time_format, options=['MM:SS', 'HH:MM:SS', 'SS.MMMMMM', 'HH:MM:SS.MMMM'])
    check_int(name=f'{superimpose_elapsed_time.__name__} font_size', value=font_size, min_value=1)
    check_int(name=f'{superimpose_elapsed_time.__name__} font_border_width', value=font_border_width, min_value=1)
    font_color = ''.join(filter(str.isalnum, font_color)).lower()
    font_border_color = ''.join(filter(str.isalnum, font_border_color)).lower()
    font_dict = get_fonts()
    check_str(name='font', value=font, options=tuple(font_dict.keys()))
    font_path = font_dict[font]
    time_format_map = {
        'MM:SS': '%{pts\\:mks}',
        'HH:MM:SS': '%{pts\\:hms}',
        'SS.MMMMMM': '%{pts}',
        'HH:MM:SS.MMMM': '%{pts\\:hms}.%{eif\\:mod(n\\,1000)\\:d\\:4}'
    }

    position_map = {
        'top_left': 'x=5:y=5',
        'top_right': 'x=(w-tw-5):y=5',
        'bottom_left': 'x=5:y=(h-th-5)',
        'bottom_right': 'x=(w-tw-5):y=(h-th-5)',
        'top_middle': 'x=(w-tw)/2:y=10',
        'bottom_middle': 'x=(w-tw)/2:y=(h-th-10)'
    }

    time_text = time_format_map[time_format]
    pos = position_map[position]

    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(
            msg=f'{video_path} is not a valid file path or a valid directory path',
            source=superimpose_elapsed_time.__name__
        )

    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])

    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Superimposing time {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_time_superimposed{ext}')

        # Escape Windows path for FFmpeg compatibility
        font_path_escaped = font_path.replace('\\', '/') if os.name == 'nt' else font_path

        vf_filter = f"drawtext=fontfile={font_path_escaped}:text='{time_text}':{pos}:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}"

        if not gpu:
            cmd = f'ffmpeg -i "{video_path}" -vf "{vf_filter}" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "{vf_filter}" -c:v h264_nvenc -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'

        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Super-imposed time on {len(video_paths)} video(s), saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

def reverse_videos(path: Union[str, os.PathLike],
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   codec: Literal['libx265', 'libx264', 'vp9'] = 'libx265',
                   quality: Optional[int] = 60,
                   gpu: Optional[bool] = False) -> None:

    """
    Reverses one or more video files located at the specified path and saves the reversed videos in the specified
    directory.

    .. video:: _static/img/reverse_videos.webm
       :width: 800
       :loop:

    :param Union[str, os.PathLike] path: Path to the video file or directory containing video files to be reversed.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the reversed videos. If not provided, reversed videos will be saved in a subdirectory named 'reversed_<timestamp>' in the same directory as the input file(s).
    :param Optional[int] quality: Output video quality expressed as a percentage. Default is 60. Values range from 1 (low quality, high compression) to 100 (high quality, low compression).
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.
    :return: None

    :example:
    >>> reverse_videos(path='/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/videos/reverse/TheVideoName_video_name_2_frame_no.mp4')
    """

    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_overlay_video.__name__)
    check_ffmpeg_available(raise_error=True)
    check_instance(source=f'{reverse_videos.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{reverse_videos.__name__} quality', value=quality)
    check_str(name='codec', value=codec, options=('libx265', 'libx264', 'vp9'))
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=reverse_videos.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'reversed_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'mp4_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Path is not a valid file or directory path.', source=reverse_videos.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, ext = get_fn_ext(filepath=file_path)
        print(f'Reversing video {video_name} (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}{ext}')
        if not gpu:
            cmd = f'ffmpeg -i "{file_path}" -vf reverse -af areverse -c:v {codec} -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -hwaccel auto -i "{file_path}" -vf reverse -af areverse -c:v h264_nvenc -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) reversed and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=reverse_videos.__name__)

def video_to_bw(video_path: Union[str, os.PathLike],
                threshold: Optional[float] = 0.5,
                save_dir: Optional[Union[str, os.PathLike]] = None,
                gpu: Optional[bool] = False) -> None:
    """
    Convert video to black and white using passed threshold.

    .. video:: _static/img/video_to_bw.webm
       :width: 800
       :autoplay:
       :loop:

    .. seealso::
       :func:`simba.mixins.image_mixin.ImageMixin.img_to_bw`, :func:`simba.mixins.image_mixin.ImageMixin.img_stack_to_bw`,
       :func:`simba.data_processors.cuda.image.img_stack_to_bw`

    :param Union[str, os.PathLike] video_path: Path to the video
    :param Optional[float] threshold: Value between 0 and 1. Lower values gives more white and vice versa.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None.

    :example:
    >>> video_to_bw(video_path='/Users/simon/Downloads/1_LH_clipped_cropped_eq_20240515135926.mp4', threshold=0.02)
    """

    check_float(name=f'{video_to_bw.__name__} threshold', value=threshold, min_value=0, max_value=1.0)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_overlay_video.__name__)
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=video_to_bw.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
    threshold = int(255 * threshold)
    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Converting video {video_name} to black and white (Video {file_cnt + 1}/{len(video_paths)})...')
        _ = get_video_meta_data(video_path=video_path)
        _, video_name, ext = get_fn_ext(video_path)
        save_path = os.path.join(save_dir, f'{video_name}_bw{ext}')
        if not gpu:
            cmd = f"ffmpeg -i '{video_path}' -vf \"format=gray,geq=lum_expr='if(lt(lum(X,Y),{threshold}),0,255)'\" -pix_fmt yuv420p '{save_path}' -loglevel error -stats -hide_banner -y"
        else:
            cmd = f"ffmpeg -hwaccel auto -i '{video_path}' -vf \"format=gray,geq=lum_expr='if(lt(lum(X,Y),{threshold}),0,255)'\" -c:v h264_nvenc '{save_path}' -loglevel error -stats -hide_banner -y"
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        timer.stop_timer()
        stdout_success(msg=f'Video {video_name} converted to black and white.', elapsed_time=timer.elapsed_time_str)

#video_to_bw(video_path='/Users/simon/Desktop/Screen Recording 2024-05-08 at 10.57.59 AM_upsampled_time_superimposed.mov', threshold=0.5)


def create_average_frm(video_path: Union[str, os.PathLike],
                       start_frm: Optional[int] = None,
                       end_frm: Optional[int] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       save_path: Optional[Union[str, os.PathLike]] = None,
                       verbose: Optional[bool] = False) -> Union[None, np.ndarray]:

    """
    Create an image representing the average frame of a segment in a video or an entire video.

    .. video:: _static/img/create_average_frm_1.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/create_average_frm_2.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/create_average_frm_3.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/create_average_frm_4.webm
       :width: 800
       :autoplay:
       :loop:

    .. note::
       Useful helper for e.g., video background subtraction ``simba.video_processors.video_processing.video_bg_substraction()``
       Either pass ``start_frm`` and ``end_frm`` OR ``start_time`` and ``end_time`` OR pass all four arguments as None.
       If all are None, then the entire video will be used to create the average frame.

    .. seealso:
       See :func:`simba.data_processors.cuda.image.create_average_frm_cupy`, :func:`simba.data_processors.cuda.image.create_average_frm_cuda` for GPU acceleration.

    :param Union[str, os.PathLike] video_path: The path to the video to create the average frame from. Default: None.
    :param Optional[int] start_frm: The first frame in the segment to create the average frame from. Default: None.
    :param Optional[int] end_frm: The last frame in the segment to create the average frame from. Default: None.
    :param Optional[str] start_time: The start timestamp in `HH:MM:SS` format in the segment to create the average frame from. Default: None.
    :param Optional[str] end_time: The end timestamp in `HH:MM:SS` format in the segment to create the average frame from. Default: None.
    :param Optional[Union[str, os.PathLike]] save_path: The path to where to save the average image. If None, then returns the average image in np,ndarray format. Default: None.
    :return Union[None, np.ndarray]: The average image (if ``save_path`` is not None) or None if  ``save_path`` is passed.
    """

    if ((start_frm is not None) or (end_frm is not None)) and ((start_time is not None) or (end_time is not None)):
        raise InvalidInputError(msg=f'Pass start_frm and end_frm OR start_time and end_time', source=create_average_frm.__name__)
    elif type(start_frm) != type(end_frm):
        raise InvalidInputError(msg=f'Pass start frame and end frame', source=create_average_frm.__name__)
    elif type(start_time) != type(end_time):
        raise InvalidInputError(msg=f'Pass start time and end time', source=create_average_frm.__name__)
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    if verbose:
        print(f'Getting average frame from {video_path}...')
    if (start_frm is not None) and (end_frm is not None):
        check_int(name='start_frm', value=start_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name='end_frm', value=end_frm, min_value=0, max_value=video_meta_data['frame_count'])
        if start_frm > end_frm:
            raise InvalidInputError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm}).', source=create_average_frm.__name__)
        frame_ids = list(range(start_frm, end_frm + 1))
    elif (start_time is not None) and (end_time is not None):
        check_if_string_value_is_valid_video_timestamp(value=start_time, name=create_average_frm.__name__)
        check_if_string_value_is_valid_video_timestamp(value=end_time, name=create_average_frm.__name__)
        check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=create_average_frm.__name__)
        check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start_time, video_path=video_path)
        frame_ids = find_frame_numbers_from_time_stamp(start_time=start_time, end_time=end_time, fps=video_meta_data['fps'])
    else:
        frame_ids = list(range(0, video_meta_data['frame_count']))
    cap.set(0, frame_ids[0])
    bg_sum, frm_cnt, frm_len = None, 0, len(frame_ids)
    while frm_cnt < frm_len:
        if verbose:
            print(f'Reading frame {frm_cnt} / {frm_len} ({video_path})...')
        ret, frm = cap.read()
        if ret:
            if bg_sum is None:
                bg_sum = np.float32(frm)
            else:
                cv2.accumulate(frm, bg_sum)
            frm_cnt += 1
        else:
            break
    img = cv2.convertScaleAbs(bg_sum / frm_len)
    cap.release()
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=create_average_frm.__name__)
        cv2.imwrite(save_path, img)
        if verbose:
            stdout_success(msg=f'Saved average frame at {save_path}', source=create_average_frm.__name__)
    else:
        return img


def video_bg_subtraction(video_path: Union[str, os.PathLike],
                         avg_frm: Optional[np.ndarray] = None,
                         bg_video_path: Optional[Union[str, os.PathLike]] = None,
                         bg_start_frm: Optional[int] = None,
                         bg_end_frm: Optional[int] = None,
                         bg_start_time: Optional[str] = None,
                         bg_end_time: Optional[str] = None,
                         threshold: Optional[int] = 50,
                         bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                         fg_color: Optional[Tuple[int, int, int]] = None,
                         save_path: Optional[Union[str, os.PathLike]] = None,
                         verbose: Optional[bool] = True,
                         method: str = 'absolute',
                         closing_kernel_size: Optional[Tuple[int, int]] = None,
                         closing_iterations: int = 3,
                         opening_kernel_size: Optional[Tuple[int, int]] = None,
                         opening_iterations: int = 3) -> None:

    """
    Subtract the background from a video.

    .. video:: _static/img/video_bg_subtraction.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/bg_remover_example_1.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/bg_remover_example_2.webm
       :width: 800
       :autoplay:
       :loop:

    .. note::
       If  ``bg_video_path`` is passed, that video will be used to parse the background. If None, ``video_path`` will be use dto parse background.
       Either pass ``start_frm`` and ``end_frm`` OR ``start_time`` and ``end_time`` OR pass all four arguments as None.
       Those two arguments will be used to slice the background video, and the sliced part is used to parse the background.

       For example, in the scenario where there is **no** animal in the ``video_path`` video for the first 20s, then the first 20s can be used to parse the background.
       In this scenario, ``bg_video_path`` can be passed as ``None`` and bg_start_time and bg_end_time can be ``00:00:00`` and ``00:00:20``, repectively.

       In the scenario where there **is** animal(s) in the entire ``video_path`` video, pass ``bg_video_path`` as a path to a video recording the arena without the animals.

    .. seealso::
       For multicore CPU alternative, see :func:`~simba.video_processors.video_processing.video_bg_substraction_mp.
       For GPU based alternative, see :func:`simba.data_processors.cuda.image.bg_subtraction_cuda` or :func:`simba.data_processors.cuda.image.bg_subtraction_cupy`.

    :param Union[str, os.PathLike] video_path: The path to the video to remove the background from.
    :param Optional[np.ndarray] avg_frm: The average frame to use to compute the background. If None is passed, then the average frame will be computed.
    :param Optional[Union[str, os.PathLike]] bg_video_path: Only used if ``avg_frm`` is None. Path to the video which contains a segment with the background only. If None, then ``video_path`` will be used.
    :param Optional[int] bg_start_frm: The first frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[int] bg_end_frm: The last frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[str] bg_start_time: The start timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[str] bg_end_time: The end timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[int] threshold: Value between 0-255 representing the difference threshold between the average frame subtracted from each frame. Higher values and more pixels will be considered background. Default: 50.
    :param Optional[Tuple[int, int, int]] bg_color: The RGB color of the moving objects in the output video. Defaults to None, which represents the original colors of the moving objects.
    :param Optional[Tuple[int, int, int]] fg_color: The RGB color of the background output video. Defaults to black (0, 0, 0).
    :param Optional[Union[str, os.PathLike]] save_path: The patch to where to save the output video where the background is removed. If None, saves the output video in the same directory as the input video with the ``_bg_subtracted`` suffix. Default: None.
    :return: None.

    :example:
    >>> video_bg_subtraction(video_path='/Users/simon/Downloads/1_LH_cropped.mp4', bg_start_time='00:00:00', bg_end_time='00:00:10', bg_color=(0, 106, 167), fg_color=(254, 204, 2))
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=video_path)
    if bg_video_path is None:
        bg_video_path = deepcopy(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    dir, video_name, ext = get_fn_ext(filepath=video_path)
    if save_path is None:
        save_path = os.path.join(dir, f'{video_name}_bg_subtracted{ext}')
    else:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=video_bg_subtraction.__name__)
    check_int(name=f'{video_bg_subtraction.__name__} threshold', value=threshold, min_value=1, max_value=255)
    check_str(name='method', value=method, options=['absolute', 'light', 'dark'], raise_error=True)
    closing_kernel, opening_kernel = None, None
    if closing_kernel_size is not None:
        check_valid_tuple(x=closing_kernel_size, source=f'{video_bg_subtraction.__name__} closing_kernel_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_int(name=f'{video_bg_subtraction.__name__} closing iterations', value=closing_iterations, min_value=1, raise_error=True)
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel_size)
    if opening_kernel_size is not None:
        check_valid_tuple(x=opening_kernel_size, source=f'{video_bg_subtraction.__name__} opening_kernel_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_int(name=f'{video_bg_subtraction.__name__} opening iterations', value=opening_iterations, min_value=1, raise_error=True)
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel_size)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'],(video_meta_data['width'], video_meta_data['height']))
    if avg_frm is None:
        avg_frm = create_average_frm(video_path=bg_video_path, start_frm=bg_start_frm, end_frm=bg_end_frm, start_time=bg_start_time, end_time=bg_end_time)
        avg_frm = cv2.resize(avg_frm, (video_meta_data['width'], video_meta_data['height']))
    else:
        check_if_valid_img(data=avg_frm, source=f'{video_bg_subtraction_mp.__name__} avg_frm')
        avg_frm = cv2.resize(avg_frm, (video_meta_data['width'], video_meta_data['height']))
    cap = cv2.VideoCapture(video_path)
    frm_cnt = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            #FrameRangeWarning(msg=f'Could not read frame {frm_cnt} in video {video_path}', source=video_bg_subtraction.__name__)
            break
        out_frm = deepcopy(frm)
        if method == 'absolute':
            diff = cv2.absdiff(frm, avg_frm)
        elif method == 'light':  # Assumes background is black, highlights light areas
            diff = np.abs(frm.astype(np.int16) - avg_frm.astype(np.int16)).astype(np.uint8)
        elif method == 'dark':  # Assumes background is white, highlights dark areas
            diff = np.abs(avg_frm.astype(np.int16) - frm.astype(np.int16)).astype(np.uint8)
        else:
            diff = cv2.absdiff(frm, avg_frm)
        if len(diff.shape) != 2:
            gray_diff = (0.07 * diff[:, :, 2] + 0.72 * diff[:, :, 1] + 0.21 * diff[:, :, 0]).astype(np.uint8)
        else:
            gray_diff = diff
        mask = np.where(gray_diff > threshold, 1, 0).astype(np.uint8)
        out_frm[mask == 0] = bg_color
        if fg_color is not None:
            out_frm[mask == 1] = fg_color
        if opening_kernel is not None:
            out_frm = cv2.morphologyEx(out_frm, cv2.MORPH_OPEN, opening_kernel, iterations=opening_iterations)
        if closing_kernel is not None:
            out_frm = cv2.morphologyEx(out_frm, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)
        writer.write(out_frm)
        frm_cnt += 1
        if verbose:print(f'Background subtraction frame {frm_cnt}/{video_meta_data["frame_count"]} (Video: {video_name})')

    writer.release()
    cap.release()
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'Background subtracted from {video_name} and saved at {save_path}', elapsed_time=timer.elapsed_time)

def reencode_mp4_video(file_path, codec, quality):
    tmp_path = f"{file_path}.tmp"
    cmd = f'ffmpeg -i "{file_path}" -c:v {codec} -crf {quality} -c:a copy "{tmp_path}" -loglevel error -stats -hide_banner -y'
    try:
        subprocess.run(cmd, shell=True, check=True)
        os.replace(tmp_path, file_path)
    except subprocess.CalledProcessError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



def _bg_remover_mp(frm_range: Tuple[int, np.ndarray],
                   video_path: Union[str, os.PathLike],
                   bg_frm: np.ndarray,
                   bg_clr: Tuple[int, int, int],
                   fg_clr: Tuple[int, int, int],
                   video_meta_data: Dict[str, Any],
                   temp_dir: Union[str, os.PathLike],
                   verbose: bool,
                   threshold: int,
                   method: str,
                   closing_kernel: tuple,
                   closing_iterations: int,
                   opening_kernel: tuple,
                   opening_iterations: int):

    batch, frm_range = frm_range[0], frm_range[1]
    start_frm, current_frm, end_frm = frm_range[0], frm_range[0], frm_range[-1]
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    save_path = os.path.join(temp_dir, f"{batch}.mp4")
    cap.set(1, start_frm)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))
    dir, video_name, ext = get_fn_ext(filepath=video_path)
    while current_frm <= end_frm:
        ret, frm = cap.read()
        if not ret:
            FrameRangeWarning(msg=f'Could nor read frame {current_frm} in video {video_meta_data["video_name"]}', source=_bg_remover_mp.__name__)
            break
        out_frm = deepcopy(frm)
        if method == 'absolute':
            diff = cv2.absdiff(frm, bg_frm)
        elif method == 'light':
            diff = np.abs(frm.astype(np.int16) - bg_frm.astype(np.int16)).astype(np.uint8)
        elif method == 'dark':
            diff = np.abs(bg_frm.astype(np.int16) - frm.astype(np.int16)).astype(np.uint8)
        else:
            diff = cv2.absdiff(frm, bg_frm)
        if len(diff.shape) != 2:
            gray_diff = (0.07 * diff[:, :, 2] + 0.72 * diff[:, :, 1] + 0.21 * diff[:, :, 0]).astype(np.uint8)
        else:
            gray_diff = diff
        mask = np.where(gray_diff > threshold, 1, 0).astype(np.uint8)
        out_frm[mask == 0] = bg_clr
        if fg_clr is not None:
            out_frm[mask == 1] = fg_clr
        if opening_kernel is not None:
            out_frm = cv2.morphologyEx(out_frm, cv2.MORPH_OPEN, opening_kernel, iterations=opening_iterations)
        if closing_kernel is not None:
            out_frm = cv2.morphologyEx(out_frm, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)
        writer.write(out_frm)
        current_frm += 1
        if verbose:
            print(f'Background subtracted frame {current_frm}/{video_meta_data["frame_count"]} (Video: {video_name})')
    writer.release()
    cap.release()
    return batch

def video_bg_subtraction_mp(video_path: Union[str, os.PathLike],
                            bg_video_path: Union[str, os.PathLike] = None,
                            bg_start_frm: Optional[int] = None,
                            bg_end_frm: Optional[int] = None,
                            bg_start_time: Optional[str] = None,
                            bg_end_time: Optional[str] = None,
                            avg_frm: Optional[np.ndarray] = None,
                            bg_color: Tuple[int, int, int] = (0, 0, 0),
                            fg_color: Optional[Tuple[int, int, int]] = None,
                            save_path: Optional[Union[str, os.PathLike]] = None,
                            core_cnt: int = -1,
                            verbose: bool = True,
                            gpu: bool = False,
                            threshold: Optional[int] = 50,
                            method: str = 'absolute',
                            closing_kernel_size: Optional[Tuple[int, int]] = None,
                            closing_iterations: int = 3,
                            opening_kernel_size: Optional[Tuple[int, int]] = None,
                            opening_iterations: int = 3) -> None:

    """
    Subtract the background from a video using multiprocessing.

    .. video:: _static/img/video_bg_substraction_mp.webm
       :width: 900
       :autoplay:
       :loop:

    .. video:: _static/img/bg_remover_example_3.webm
       :width: 900
       :autoplay:
       :loop:

    .. video:: _static/img/bg_remover_example_4.webm
       :width: 900
       :autoplay:
       :loop:

    .. note::
       If  ``bg_video_path`` is passed, that video will be used to parse the background. If None, ``video_path`` will be use to parse background.
       Either pass ``start_frm`` and ``end_frm`` OR ``start_time`` and ``end_time`` OR pass all four arguments as None.
       Those two arguments will be used to slice the background video, and the sliced part is used to parse the background.

       For example, in the scenario where there is **no** animal in the ``video_path`` video for the first 20s, then the first 20s can be used to parse the background.
       In this scenario, ``bg_video_path`` can be passed as ``None`` and bg_start_time and bg_end_time can be ``00:00:00`` and ``00:00:20``, repectively.

       In the scenario where there **is** animal(s) in the entire ``video_path`` video, pass ``bg_video_path`` as a path to a video recording the arena without the animals.

    .. seealso::
        For single core alternative, see :func:`~simba.video_processors.video_processing.video_bg_subtraction`.
        For GPU based alternative, see :func:`~simba.data_processors.cuda.image.bg_subtraction_cuda` or :func:`~simba.data_processors.cuda.image.bg_subtraction_cupy`.

    :param Union[str, os.PathLike] video_path: The path to the video to remove the background from.
    :param Optional[np.ndarray] avg_frm: The average frame to use to compute the background. If None is passed, then the average frame will be computed.
    :param Optional[Union[str, os.PathLike]] bg_video_path: Path to the video which contains a segment with the background only. If None, then ``video_path`` will be used.
    :param Optional[int] bg_start_frm: The first frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[int] bg_end_frm: The last frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[str] bg_start_time: The start timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[str] bg_end_time: The end timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[str] bg_end_time: The end timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[Tuple[int, int, int]] bg_color: The RGB color of the moving objects in the output video. Defaults to None, which represents the original colors of the moving objects.
    :param Optional[Union[str, os.PathLike]] save_path: The patch to where to save the output video where the background is removed. If None, saves the output video in the same directory as the input video with the ``_bg_subtracted`` suffix. Default: None.
    :param Optional[int] core_cnt: The number of cores to use. Defaults to -1 representing all available cores.
    :param Optional[int] threshold: Value between 0-255 representing the difference threshold between the average frame subtracted from each frame. Higher values and more pixels will be considered background. Default: 50.
    :return: None.

    :example:
    >>> video_bg_subtraction_mp(video_path='/Users/simon/Downloads/1_LH.mp4', bg_start_time='00:00:00', bg_end_time='00:00:10', bg_color=(0, 0, 0), fg_color=(255, 255, 255))
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=video_path)
    check_int(name=f'{video_bg_subtraction_mp.__name__} threshold', value=threshold, min_value=1, max_value=255)
    check_str(name='method', value=method, options=['absolute', 'light', 'dark'], raise_error=True)
    if bg_video_path is None: bg_video_path = deepcopy(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    dir, video_name, ext = get_fn_ext(filepath=video_path)
    if save_path is None:
        save_path = os.path.join(dir, f'{video_name}_bg_subtracted{ext}')
    else:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=video_bg_subtraction_mp.__name__)
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_dir = os.path.join(os.path.dirname(save_path), f'temp_{video_name}_{dt}')
    os.makedirs(temp_dir)
    check_int(name=f'{video_bg_subtraction_mp.__name__} core_cnt', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0])
    if core_cnt == -1: core_cnt = find_core_cnt()[0]
    closing_kernel, opening_kernel = None, None
    if closing_kernel_size is not None:
        check_valid_tuple(x=closing_kernel_size, source=f'{video_bg_subtraction_mp.__name__} closing_kernel_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_int(name=f'{video_bg_subtraction_mp.__name__} closing iterations', value=closing_iterations, min_value=1, raise_error=True)
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel_size)
    if opening_kernel_size is not None:
        check_valid_tuple(x=opening_kernel_size, source=f'{video_bg_subtraction_mp.__name__} opening_kernel_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_int(name=f'{video_bg_subtraction_mp.__name__} opening iterations', value=opening_iterations, min_value=1, raise_error=True)
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel_size)
    if avg_frm is None:
        bg_frm = create_average_frm(video_path=bg_video_path, start_frm=bg_start_frm, end_frm=bg_end_frm, start_time=bg_start_time, end_time=bg_end_time)
        #bg_frm = bg_frm[:, :, ::-1]
    else:
        check_if_valid_img(data=avg_frm, source=f'{video_bg_subtraction_mp.__name__} avg_frm')
        bg_frm = np.copy(avg_frm)
    bg_frm = cv2.resize(bg_frm, (video_meta_data['width'], video_meta_data['height']))
    frm_list = np.array_split(list(range(0, video_meta_data['frame_count'])), core_cnt)
    frm_data = []

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)

    for c, i in enumerate(frm_list):
        frm_data.append((c, i))
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_bg_remover_mp,
                                      video_path=video_path,
                                      bg_frm=bg_frm,
                                      bg_clr=bg_color,
                                      fg_clr=fg_color,
                                      video_meta_data=video_meta_data,
                                      temp_dir=temp_dir,
                                      verbose=verbose,
                                      threshold=threshold,
                                      method=method,
                                      closing_kernel=closing_kernel,
                                      closing_iterations=closing_iterations,
                                      opening_kernel=opening_kernel,
                                      opening_iterations=opening_iterations)
        for cnt, result in enumerate(pool.imap(constants, frm_data, chunksize=1)):
            print(f'Frame batch {result+1} completed...')
    pool.terminate()
    pool.join()
    gc.collect()
    print(f"Joining {video_name} multi-processed video...")
    concatenate_videos_in_folder(in_folder=temp_dir, save_path=save_path, video_format=ext[1:], remove_splits=True, gpu=gpu, fps=video_meta_data['fps'])
    #reencode_mp4_video(save_path, 'libx264', 23)
    timer.stop_timer()
    stdout_success(msg=f'Video saved at {save_path}', elapsed_time=timer.elapsed_time_str)

def superimpose_video_names(video_path: Union[str, os.PathLike],
                            font: Optional[str] = 'Arial',
                            font_size: Optional[int] = 30,
                            font_color: Optional[str] = 'white',
                            font_border_color: Optional[str] = 'black',
                            font_border_width: Optional[int] = 2,
                            position: Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle_top', 'middle_bottom']] = 'top_left',
                            save_dir: Optional[Union[str, os.PathLike]] = None,
                            gpu: Optional[bool] = False) -> None:
    """
    Superimposes the video name on the given video file(s) and saves the modified video(s).

    .. video:: _static/img/superimpose_video_names.webm
       :width: 900
       :loop:

    :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
    :param Optional[int] font_size: Font size for the video name text. Default 30.
    :param Optional[str] font_color:  Font color for the video name text. Default white
    :param Optional[str] font_border_color: Font border color for the video name text. Default black.
    :param Optional[int] font_border_width: Font border width for the video name text in pixels. Default 2.
    :param Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']] position: Position where the video name will be superimposed. Default ``top_left``.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where the modified video(s) will be saved. If not provided, the directory of the input video(s) will be used.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None

    :example:
    >>> superimpose_video_names(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_4/1.mp4', position='middle_top', font_color='black', font_border_color='pink', font_border_width=5, font_size=30)
    """

    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_overlay_video.__name__)
    POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']
    check_str(name=f'{superimpose_video_names.__name__} position', value=position, options=POSITIONS)
    check_int(name=f'{superimpose_video_names.__name__} font_size', value=font_size, min_value=1)
    check_int(name=f'{superimpose_video_names.__name__} font_border_width', value=font_border_width, min_value=1)
    font_dict = get_fonts()
    check_str(name='font', value=font, options=tuple(font_dict.keys()))
    font_path = font_dict[font]
    font_color = ''.join(filter(str.isalnum, font_color)).lower()
    font_border_color = ''.join(filter(str.isalnum, font_border_color)).lower()
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=superimpose_video_names.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Superimposing video name on {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_video_name_superimposed{ext}')
        if position == POSITIONS[0]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=5:y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=5:y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:v h264_nvenc -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[1]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw-5):y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw-5):y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:v h264_nvenc -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[2]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=5:y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=5:y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[3]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw-5):y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw-5):y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'

        elif position == POSITIONS[4]:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw)/2:y=10:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw)/2:y=10:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'

        else:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw)/2:y=(h-th-10):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={video_name}:x=(w-tw)/2:y=(h-th-10):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Super-imposed video name on {len(video_paths)} video(s), saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

def superimpose_freetext(video_path: Union[str, os.PathLike],
                         text: str,
                         font: Optional[str] = 'Arial',
                         font_size: Optional[int] = 30,
                         font_color: Optional[str] = 'white',
                         font_border_color: Optional[str] = 'black',
                         font_border_width: Optional[int] = 2,
                         position: Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'middle_top', 'middle_bottom']] = 'top_left',
                         save_dir: Optional[Union[str, os.PathLike]] = None,
                         gpu: Optional[bool] = False) -> None:
    """
    Superimposes passed text on the given video file(s) and saves the modified video(s).

    .. video:: _static/img/superimpose_freetext.webm
       :width: 900
       :loop:

    :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
    :param str text: The text to overlay on the video
    :param Optional[int] font_size: Font size for the text. Default 30.
    :param Optional[str] font_color:  Font color for text. Default white
    :param Optional[str] font_border_color: Font border color for the text. Default black.
    :param Optional[int] font_border_width: Font border width for the text in pixels. Default 2.
    :param Optional[Literal['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']] position: Position where the text will be superimposed. Default ``top_left``.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where the modified video(s) will be saved. If not provided, the directory of the input video(s) will be used.
    :param Optional[bool] gpu: If True, uses GPU codecs with potentially faster runtimes. Default: False.

    :return: None
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None).", source=superimpose_overlay_video.__name__)
    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_middle', 'bottom_middle']
    check_str(name=f'{superimpose_freetext.__name__} position', value=position, options=POSITIONS)
    check_int(name=f'{superimpose_freetext.__name__} font_size', value=font_size, min_value=1)
    check_int(name=f'{superimpose_freetext.__name__} font_border_width', value=font_border_width, min_value=1)
    check_str(name=f'{superimpose_freetext.__name__} text', value=text)
    font_dict = get_fonts()
    check_str(name='font', value=font, options=tuple(font_dict.keys()))
    font_path = font_dict[font]
    font_color = ''.join(filter(str.isalnum, font_color)).lower()
    font_border_color = ''.join(filter(str.isalnum, font_border_color)).lower()
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=superimpose_video_names.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Superimposing video name on {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_text_superimposed{ext}')
        if position == POSITIONS[0]:
            cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={text}:x=5:y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[1]:
            cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={text}:x=(w-tw-5):y=5:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[2]:
            cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={text}:x=5:y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[3]:
            cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={text}:x=(w-tw-5):y=(h-th-5):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == POSITIONS[4]:
            cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={text}:x=(w-tw)/2:y=10:fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        else:
            cmd = f'ffmpeg -i "{video_path}" -vf \"drawtext=fontfile={font_path}:text={text}:x=(w-tw)/2:y=(h-th-10):fontsize={font_size}:fontcolor={font_color}:borderw={font_border_width}:bordercolor={font_border_color}\" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Super-imposed free-text on {len(video_paths)} video(s), saved in {save_dir}', elapsed_time=timer.elapsed_time_str)


def rotate_video(video_path: Union[str, os.PathLike],
                 degrees: int,
                 gpu: Optional[bool] = False,
                 quality: Optional[int] = 60,
                 fill_color: Optional[str] = 'black',
                 save_dir: Optional[Union[str, os.PathLike]] = None):

    """
    Rotate a video or a directory of videos by a specified number of degrees.

    .. video:: _static/img/rotate_video.mp4
       :width: 700
       :loop:
       :autoplay:

    :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
    :param int degrees: Number of degrees (between 1 and 359, inclusive) to rotate the video clockwise.
    :param Optional[bool] gpu: If True, attempt to use GPU acceleration for rotation (default is False).
    :param Optional[int] quality: Quality of the output video, an integer between 1 and 100 (default is 60).
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the rotated video(s). If None, the directory of the input video(s) will be used with the `rotated` suiffix.
    :return: None.

    :example:
    >>> rotate_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/rot_test.mp4', degrees=180)
    >>> rotate_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\clipped\501_MA142_Gi_CNO_0514_clipped.mp4", degrees=65)
    >>> rotate_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\clipped\501_MA142_Gi_CNO_0514_clipped.mp4", degrees=10, fill_color='deeppink', gpu=True)

    """

    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    check_int(name=f'{rotate_video.__name__} font_size', value=degrees, min_value=1, max_value=359)
    check_int(name=f'{rotate_video.__name__} quality', value=quality, min_value=1, max_value=100)
    check_str(name=f'{rotate_video.__name__} fill_color', value=fill_color.lower(), options=tuple(get_named_colors()))
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=rotate_video.__name__)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=rotate_video.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Rotating video {video_name} {degrees} degrees (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_rotated{ext}')
        print(fill_color)
        if gpu:
            cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "rotate={degrees}*(PI/180):fillcolor={fill_color},format=nv12" -c:v h264_nvenc "{save_path}" -loglevel error -stats -y'
        else:
            cmd = f'ffmpeg -i "{video_path}" -vf "rotate={degrees}*(PI/180):fillcolor={fill_color}" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) ratated {degrees} and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=rotate_video.__name__,)


#rotate_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\clipped\501_MA142_Gi_CNO_0514_clipped.mp4", degrees=20, fill_color='coral')


def flip_videos(video_path: Union[str, os.PathLike],
                horizontal_flip: Optional[bool] = False,
                vertical_flip: Optional[bool] = False,
                quality: Optional[int] = 60,
                save_dir: Optional[Union[str, os.PathLike]] = None,
                gpu: Optional[bool] = False) -> None:
    """
    Flip a video or directory of videos horizontally, vertically, or both, and save them to the specified directory.

    .. video:: _static/img/flip_videos.webm
       :width: 700
       :loop:
       :autoplay:

    :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
    :param Optional[bool] horizontal_flip: If True, flip the video(s) horizontally (default is False).
    :param Optional[bool] vertical_flip: If True, flip the video(s) vertically (default is False).
    :param Optional[int] quality: Quality of the output video, an integer between 1 and 100 (default is 60).
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the flipped video(s). If None, the directory of the input video(s) will be used.
    :param Optional[bool] gpu: If True, attempt to use GPU acceleration for rotation (default is False).
    :return: None.
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=rotate_video.__name__)
    timer = SimbaTimer(start=True)
    check_int(name=f'{rotate_video.__name__} quality', value=quality, min_value=1, max_value=100)
    if not horizontal_flip and not vertical_flip: raise InvalidInputError(msg='Flip videos vertically and/or horizontally. Got both as False', source=flip_videos.__name__)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=flip_videos.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Flipping video {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_flipped{ext}')
        if vertical_flip and not horizontal_flip:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf "vflip" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "vflip" -c:v h264_nvenc -preset medium "{save_path}" -loglevel error -stats -y'
        elif horizontal_flip and not vertical_flip:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf "hflip" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "hflip" -c:v h264_nvenc -crf medium "{save_path}" -loglevel error -stats -y'
        else:
            if not gpu:
                cmd = f'ffmpeg -i "{video_path}" -vf "hflip,vflip" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "hflip,vflip" -c:v h264_nvenc -crf medium "{save_path}" -loglevel error -stats -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) flipped and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=flip_videos.__name__)


def upsample_fps(video_path: Union[str, os.PathLike],
                 fps: int,
                 quality: Optional[int] = 60,
                 save_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """
    Upsample the frame rate of a video or all videos in a directory to a specified fps with a given quality.

    .. note::
       Uses ``ffmpeg minterpolate``

    .. warning::
       Long run-times for higher resolution videos.

    :param Union[str, os.PathLike] video_path: The path to the input video file or directory containing video files.
    :param int fps: The target frame rate to upsample the video(s) to.
    :param Optional[int] quality: The quality level of the output video(s), represented as a percentage (1-100). Lower values indicate higher quality. Default is 60.
    :param Optional[Union[str, os.PathLike]] save_dir: The directory to save the upsampled video(s). If None, the videos will be saved in the same directory as the input video(s).
    :return: None. The function saves the upsampled video(s) to the specified directory.

    :example:
    >>> upsample_fps(video_path='/Users/simon/Desktop/Box2_IF19_7_20211109T173625_4_851_873_1_cropped.mp4', fps=100, quality=100)
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_int(name=f'{upsample_fps.__name__} quality', value=quality, min_value=1, max_value=100)
    check_float(name=f'{upsample_fps.__name__} fps', value=quality, min_value=10e-16)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=rotate_video.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        input_video_meta = get_video_meta_data(video_path=video_path)
        if input_video_meta['fps'] >= fps:
            FrameRangeWarning(msg=f"The FPS of the input video named {video_name} ({input_video_meta['fps']}) is the same or larger than the upscaled target FPS ({fps})", source=upsample_fps.__name__)
        print(f'Up-sampling video {video_name} to {fps} FPS (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_upsampled{ext}')
        cmd = f"""ffmpeg -i "{video_path}" -filter:v "minterpolate='fps={fps}:scd=none:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:search_param=16:vsbmc=1'" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y"""
        subprocess.call(cmd, shell=True)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) upsampled to {fps} and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=upsample_fps.__name__, )

def temporal_concatenation(video_paths: List[Union[str, os.PathLike]],
                           save_path: Optional[Union[str, os.PathLike]] = None,
                           save_format: Optional[Literal['mp4', 'mov', 'avi', 'webm']] = 'mp4',
                           quality: Optional[int] = 60,
                           gpu: Optional[bool] = False) -> None:

    """
    Concatenates multiple video files temporally into a single video.

    :param List[Union[str, os.PathLike]] video_paths: List of paths to video files to be temporally joined. The videos will be joined in the order appearance in the list.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to save the temporally concatenated videos. If None, then the video is saved in the same directory as the first video in ``video_paths`` with the name ``temporal_concat_video``.
    :param Optional[Literal['mp4', 'mov', 'avi', 'webm']] save_format: The video format of the concatenated video. Default: ``mp4``.
    :param Optional[int] quality: Integer representing the quality: 10, 20, 30.. 100.
    :param Optional[bool] gpu: If True, attempt to use GPU acceleration for rotation (default is False).
    :return: None
    """

    timer = SimbaTimer(start=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=rotate_video.__name__)
    check_valid_lst(data=video_paths, source=temporal_concatenation.__name__, valid_dtypes=(str,), min_len=2)
    check_str(name='save_format', value=save_format.lower(), options=('mp4', 'mov', 'avi', 'webm'))
    check_int(name='quality', value=quality, max_value=100, min_value=1)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    meta = []
    for i in video_paths:
        check_file_exist_and_readable(file_path=i); video_meta = get_video_meta_data(video_path=i)
        meta.append(video_meta)
    fps, resolutions = [v['fps'] for v in meta], [v['resolution_str'] for v in meta]
    unique_fps, unique_res = list(set(fps)), list(set(resolutions))
    if len(unique_fps) > 1: raise ResolutionError(msg=f'The selected videos contain more than one unique FPS: {unique_fps}', source=temporal_concatenation.__name__)
    if len(unique_res) > 1: raise ResolutionError(msg=f'The selected videos contain more than one unique resolutions: {unique_res}', source=temporal_concatenation.__name__)
    if save_path is None:
        save_path = os.path.join(os.path.dirname(video_paths[0]), f'temporal_concat_video.{save_format}')
    else:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
    filter_complex = ""
    for i, path in enumerate(video_paths):
        filter_complex += f"[{i}:v]"
    filter_complex += f"concat=n={len(video_paths)}:v=1[v]"
    input_options = " ".join([f"-i \"{path}\"" for path in video_paths])
    if not gpu:
        cmd = f'ffmpeg {input_options} -filter_complex "{filter_complex}" -crf {crf} -map "[v]" "{save_path}" -hide_banner -loglevel error -stats -y'
    else:
        cmd = f'ffmpeg -hwaccel auto {input_options} -filter_complex "{filter_complex}" -c:v h264_nvenc -crf medium -map "[v]" "{save_path}" -hide_banner -loglevel error -stats -y'
    subprocess.call(cmd, shell=True)
    timer.stop_timer()
    stdout_success(msg=f'{len(video_paths)} videos temporally concatenated and saved at {save_path}', elapsed_time=timer.elapsed_time_str)


def get_img_slic(img: np.ndarray,
                 n_segments: Optional[int] = 50,
                 compactness: Optional[int] = 50,
                 sigma: Optional[float] = 1) -> np.ndarray:

    """
    Simplify an image into superpixels using SLIC (Simple Linear Iterative Clustering).

    :param np.ndarray img: Image to segment.
    :param n_segments: Number of segments to produce.
    :param compactness: How compact ("square") the output segments are.
    :param np.ndarray sigma: Amount of Gaussian smoothing.
    :return: Smoothened version of the input image.
    :rtype: np.ndarray

    :example:
    >>> img = read_frm_of_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0626.mp4", frame_index=0)
    >>> sliced_img = get_img_slic(img=img)
    """

    check_if_valid_img(data=img, source=f'{get_img_slic.__name__} img', raise_error=True)
    check_int(name=f'{get_img_slic.__name__} n_segments', value=n_segments, min_value=2)
    check_int(name=f'{get_img_slic.__name__} compactness', value=compactness, min_value=1)
    check_int(name=f'{get_img_slic.__name__} sigma', value=compactness, min_value=0)
    segments = slic(image=img, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0)
    segmented_image = label2rgb(segments, img, kind='avg', bg_label=0)

    return segmented_image

def _slic_helper(frm_range: np.ndarray,
                 n_segments: int,
                 sigma: float,
                 compactness: int,
                 save_dir: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike]):

    """ SLIC multiprocess helper called by slic.get_video_slic"""

    video_cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    batch, start_frm, end_frm = frm_range[0], frm_range[1][0], frm_range[1][-1]
    save_path = os.path.join(save_dir, f'{batch}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    for frm_idx in range(start_frm, end_frm):
        print(f'Frame {frm_idx}/{end_frm}, Batch {batch}...')
        img = read_frm_of_video(video_path=video_cap, frame_index=frm_idx)
        img = get_img_slic(img=img, n_segments=n_segments, compactness=compactness, sigma=sigma)
        writer.write(img)
    writer.release()
    return batch


def get_video_slic(video_path: Union[str, os.PathLike],
                   save_path: Union[str, os.PathLike],
                   n_segments: Optional[int] = 50,
                   compactness: Optional[int] = 50,
                   sigma: Optional[int] = 1,
                   core_cnt: Optional[int] = -1) -> None:

    """
    Apply SLIC superpixel segmentation to all frames of a video and save the output as a new video.

    .. video:: _static/img/get_video_slic.webm
       :width: 800
       :autoplay:
       :loop:

    .. seealso::
       To convert single image, see :func:`simba.video_processors.video_processing.get_img_slic`

    :param Union[str, os.PathLike] video_path: Path to the input video file.
    :param Union[str, os.PathLike] save_path: Path to save the processed video with SLIC superpixel segmentation.
    :param Optional[int] n_segments: Approximate number of superpixels for each frame. Defaults to 50.
    :param Optional[int] compactness: Balance of color and spatial proximity.  Higher values result in more uniformly shaped superpixels. Defaults to 50.
    :param Optional[int] sigma: Standard deviation for Gaussian smoothing applied to each frame before segmentation. Defaults to 1.
    :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. Set to -1 to use all available cores. Defaults to -1.
    :return: None. The segmented video is saved to `save_path`.

    :example:
    >>> #video_path = r"C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0626.mp4"
    """
    timer = SimbaTimer(start=True)
    check_int(name=f'{get_img_slic.__name__} n_segments', value=n_segments, min_value=2)
    check_int(name=f'{get_img_slic.__name__} compactness', value=compactness, min_value=1)
    check_int(name=f'{get_img_slic.__name__} sigma', value=sigma, min_value=1)
    check_int(name=f'{get_img_slic.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    if core_cnt == -1 or core_cnt > find_core_cnt()[0]: core_cnt = find_core_cnt()[0]
    frm_ranges = np.array_split(np.arange(0, video_meta_data['frame_count'] + 1), core_cnt)
    frm_ranges = [(y, x) for y, x in enumerate(frm_ranges)]
    out_dir, out_name, _= get_fn_ext(filepath=save_path)
    temp_folder = os.path.join(out_dir, "temp")
    if not os.path.isdir(temp_folder): os.makedirs(temp_folder)
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_slic_helper,
                                      video_path=video_path,
                                      save_dir=temp_folder,
                                      n_segments=n_segments,
                                      compactness=compactness,
                                      sigma=sigma)
        for cnt, core_batch in enumerate(pool.map(constants, frm_ranges, chunksize=1)):
            print(f'Core batch {core_batch} complete...')
    pool.join()
    pool.terminate()
    timer.stop_timer()
    concatenate_videos_in_folder(in_folder=temp_folder, save_path=save_path)
    stdout_success(msg=f'SLIC video saved at {save_path}', elapsed_time=timer.elapsed_time_str)



def is_video_seekable(data_path: Union[str, os.PathLike],
                      gpu: bool = False,
                      batch_size: Optional[int] = None,
                      verbose: bool = False,
                      raise_error: bool = True,
                      save_path: Optional[Union[str, os.PathLike]] = None) -> Union[None, bool, Tuple[Dict[str, List[int]]]]:
    """
    Determines if the given video file(s) are seekable and can be processed frame-by-frame without issues.

    This function checks if all frames in the specified video(s) can be read sequentially. It can process videos
    using either CPU or GPU, with optional batch processing to handle memory limitations. If unreadable frames are
    detected, the function can either raise an error or return a result indicating the issue.

    :param Union[str, os.PathLike] data_path: Path to the video file or a path to a directory containing video files.
    :param bool gpu: If True, then use GPU. Else, CPU.
    :param Optional[int] batch_size: Optional int representing the number of frames in each video to process sequentially. If None, all frames in a video is processed at once. Use a smaller value to avoid MemoryErrors. Default None.
    :param bool verbose: If True, prints progress. Default None.
    :param bool raise_error: If True, raises error if not all passed videos are seeakable.

    :example:
    >>> _ = is_video_seekable(data_path='/Users/simon/Desktop/unseekable/20200730_AB_7dpf_850nm_0003_fps_5.mp4', batch_size=400)
    """

    timer = SimbaTimer(start=True)
    if batch_size is not None:
        check_int(name=f'{is_video_seekable.__name__}', value=batch_size, min_value=1)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
    check_valid_boolean(value=[verbose], source=f'{is_video_seekable.__name__} verbose')
    if not check_ffmpeg_available():
        raise FFMPEGNotFoundError(msg='SimBA could not find FFMPEG on the computer.', source=is_video_seekable.__name__)
    if gpu and not check_nvidea_gpu_available():
        raise SimBAGPUError(msg='SimBA could not find a NVIDEA GPU on the computer and GPU is set to True.', source=is_video_seekable.__name__)
    if os.path.isfile(data_path):
        data_paths = [data_path]
    elif os.path.isdir(data_path):
        data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
    else:
        raise InvalidInputError(msg=f'{data_path} is not a valid in directory or file path.', source=is_video_seekable.__name__)
    _ = [get_video_meta_data(video_path=x) for x in data_paths]

    results = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Checking seekability video {video_name}...')
        video_meta_data = get_video_meta_data(video_path=file_path)
        video_frm_ranges = np.arange(0, video_meta_data['frame_count'])
        if batch_size is not None:
            video_frm_ranges = np.array_split(video_frm_ranges, max(1, int(video_frm_ranges.shape[0]/batch_size)))
        else:
            video_frm_ranges = [video_frm_ranges]
        video_error_frms = []
        for video_frm_range in video_frm_ranges:
            print(f'Processing frame {video_frm_range[0]} to {video_frm_range[-1]} (video: {video_name})...')
            if not gpu:
                imgs = ImageMixin.read_img_batch_from_video(video_path=file_path, start_frm=video_frm_range[0], end_frm=video_frm_range[-1], verbose=False)
            else:
                imgs = read_img_batch_from_video_gpu(video_path=file_path, start_frm=video_frm_range[0], end_frm=video_frm_range[-1], verbose=False)
            invalid_frms = [k for k, v in imgs.items() if v is None]
            video_error_frms.extend(invalid_frms)
        results[video_name] = video_error_frms

    timer.stop_timer()
    if all(len(v) == 0 for v in results.values()):
        stdout_success(msg=f'The {len(data_paths)} videos are valid.', source=is_video_seekable.__name__, elapsed_time=timer.elapsed_time_str)
        return True
    else:
        if save_path is not None:
            out_df = pd.DataFrame.from_dict(data=results).T
            out_df.to_csv(save_path)
            FrameRangeWarning(msg=f'Some videos have unseekable frames. See {save_path} for results. Elapsed time: {timer.elapsed_time_str}s', source=is_video_seekable.__name__)
        if raise_error:
            raise FrameRangeError(msg=f'{results} The frames in the videos listed are unreadable. Consider re-encoding these videos.', source=is_video_seekable.__name__)
        else:
            return (False, results)


def crop_video(video_path: Union[str, os.PathLike],
               save_path: Union[str, os.PathLike],
               size: Tuple[int, int],
               top_left: Tuple[int, int],
               gpu: bool = False,
               verbose: bool = True,
               quality: int = 60):
    """
    Crops a video from the given file at `video_path` and saves the result to `save_path`.
    Optionally uses GPU acceleration for faster processing, falling back to CPU if GPU fails.

    :param video_path Union[str, os.PathLike]: The path to the input video file to crop. Can be a string or path-like object.
    :param save_path Union[str, os.PathLike]: The path to save the cropped video file. Can be a string or path-like object.
    :param size Tuple[int, int]: A tuple (width, height) specifying the size of the output cropped video.
    :param top_left Tuple[int, int]: A tuple (x, y) specifying the top-left corner of the cropping area.
    :param gpu bool:  True, attempts to use GPU acceleration for the video cropping. Defaults to False.
    :param verbose bool: If True, prints progress messages and the elapsed time. Defaults to True.
    :param quality bool:  The quality of the output video, on a scale from 1 to 100. Defaults to 60 (balances encoding time vs file size).
    :return: None. The result is saved at `save_path`. If `verbose` is True, prints the elapsed time and success message.
    """

    timer = SimbaTimer(start=True)
    if not check_ffmpeg_available():
        raise FFMPEGNotFoundError(msg="FFMPEG not found on the computer. Install FFMPEG to use the crop videos.", source=crop_video.__name__)
    check_valid_boolean(value=[gpu], source=f'{crop_video.__name__} gpu')
    check_valid_boolean(value=[verbose], source=f'{crop_video.__name__} verbose')
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None. Try without GPU selection", source=crop_video.__name__)
    video_meta_data = get_video_meta_data(video_path=video_path)
    check_int(name=f'{crop_video.__name__} quality', value=quality, min_value=1, max_value=100)
    quality_lk = {int(k):v for k, v in percent_to_crf_lookup().items()}
    closest_key = min(quality_lk, key=lambda k: abs(k - quality))
    quality_code = quality_lk[closest_key]
    check_if_dir_exists(in_dir=os.path.dirname(save_path))
    check_valid_tuple(x=size, source=f'{crop_video.__name__} size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
    check_valid_tuple(x=top_left, source=f'{crop_video.__name__} top_left', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=0)
    bottom_right = (int(top_left[0] + size[0]), int(top_left[1] + size[1]))
    if top_left[0] < 0: top_left = (0, top_left[1])
    if top_left[1] < 0: top_left = (top_left[0], 0)
    if bottom_right[0] < 0: bottom_right = (0, bottom_right[1])
    if bottom_right[1] < 0: bottom_right = (bottom_right[0], 0)
    if bottom_right[0] > video_meta_data['width']: bottom_right = (video_meta_data['width'], bottom_right[1])
    if bottom_right[1] > video_meta_data['height']: bottom_right = (bottom_right[0], video_meta_data['height'])
    width, height = int(bottom_right[0] - top_left[0]), (bottom_right[1] - top_left[1])
    width, height = (width + 1) // 2 * 2, (height + 1) // 2 * 2
    top_left_x, top_left_y = top_left[0], top_left[1]
    gpu_cmd = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{video_path}" -vf "crop={width}:{height}:{top_left_x}:{top_left_y}, format=yuv420p" -c:v h264_nvenc -cq {quality_code} -c:a copy "{save_path}" -hide_banner -loglevel error -stats -y'
    cpu_cmd = f'ffmpeg -i "{video_path}" -vf "crop={width}:{height}:{top_left_x}:{top_left_y}" -c:v {Formats.BATCH_CODEC.value} -crf {quality_code} -c:a copy "{save_path}" -hide_banner -loglevel error -stats -y'
    if gpu:
        try:
            subprocess.run(gpu_cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            CropWarning(msg=f'GPU crop for video {video_meta_data["video_name"]} failed, reverting to CPU. Crop dimensions may be to small for GPU codec.', source=crop_video.__name__)
            subprocess.call(cpu_cmd, shell=True)
    else:
        subprocess.call(cpu_cmd, shell=True)
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'Cropped video saved at {save_path}', elapsed_time=timer.elapsed_time_str)


def split_mosaic(video_path: Union[str, os.PathLike],
                 tile_size: Tuple[int, int],
                 save_dir: Union[str, os.PathLike],
                 verbose: bool = True):
    """
    Helper to split a mosaic video into its constituent parts.

    :param Union[str, os.PathLike] video_path: The path to the input video file to crop. Can be a string or path-like object.
    :param Tuple[int, int] tile_size: A tuple (tile_width, tile_height) indicating the size of each tile in pixels. The width and height must evenly divide the dimensions of the video.
    :param Union[str, os.PathLike] save_dir: Directory to save the output tile videos. Will be created if it does not exist.
    :param bool verbose: If True, prints progress information and status messages during processing.

    :example:
    >>> split_mosaic(video_path=r"3.mp4", tile_size=(1280, 720), save_dir=r"3_cropped")

    """
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=video_path)
    check_valid_tuple(x=tile_size, source=f'{split_mosaic.__name__} tile_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
    check_if_dir_exists(in_dir=save_dir, source=f'{split_mosaic.__name__} save_dir', create_if_not_exist=True)
    video_meta = get_video_meta_data(video_path=video_path)
    tile_w, tile_h = tile_size
    cnt_tiles_x = video_meta['width'] // tile_w
    cnt_tiles_y = video_meta['height'] // tile_h
    for i in range(cnt_tiles_y):
        for j in range(cnt_tiles_x):
            if verbose:
                print(f'Creating tile ({i}, {j})...')
            x, y = j * tile_w, i * tile_h
            output_path = os.path.join(save_dir, f"tile_{i}_{j}.mp4")
            cmd = f'ffmpeg -i "{video_path}" -filter:v "crop={tile_w}:{tile_h}:{x}:{y}" -c:v {Formats.BATCH_CODEC.value} -crf 10 -c:a copy "{output_path}" -hide_banner -loglevel error -stats -y'
            subprocess.call(cmd, shell=True)
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'Tile data saved in {save_dir}', elapsed_time=timer.elapsed_time_str)


def get_async_frame_batch(batch_reader: AsyncVideoFrameReader, timeout: int = 10) -> Tuple[int, int, np.ndarray]:
    check_int(name=f'{get_async_frame_batch.__name__} timeout', min_value=0, raise_error=True, value=timeout)
    check_instance(source=f'{get_async_frame_batch.__name__} batch_reader', instance=batch_reader, accepted_types=(AsyncVideoFrameReader,), raise_error=True)
    x = batch_reader.frame_queue.get(timeout=timeout)
    if isinstance(x, Exception):
        raise x
    else:
        return x



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
