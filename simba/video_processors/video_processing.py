__author__ = "Simon Nilsson"

import glob, os
import cv2
from pathlib import Path
import numpy as np
import shutil
import re
import subprocess
import simba
from simba.mixins.config_reader import ConfigReader
from tkinter import *
from datetime import datetime
import time
from PIL import Image, ImageTk
import multiprocessing
from typing import List, Union

from simba.utils.checks import (check_file_exist_and_readable, check_int, check_if_filepath_list_is_empty)
from simba.utils.read_write import (get_fn_ext,
                                    get_video_meta_data,
                                    find_all_videos_in_directory,
                                    find_core_cnt,
                                    read_config_entry,
                                    read_config_file)
from simba.utils.printing import stdout_success, SimbaTimer
from simba.video_processors.extract_frames import video_to_frames
from simba.utils.enums import Formats, Paths, ConfigKey

from simba.utils.errors import (NotDirectoryError,
                                NoFilesFoundError,
                                FileExistError,
                                CountError,
                                InvalidInputError,
                                DirectoryExistError)
from simba.utils.warnings import (SameInputAndOutputWarning,
                                  FileExistWarning)


MAX_FRM_SIZE = 1080, 650

def change_img_format(directory: Union[str, os.PathLike],
                      file_type_in: str,
                      file_type_out: str) -> None:
    """
    Convert the file type of all image files within a directory.

    :parameter Union[str, os.PathLike] directory: Path to directory holding image files
    :parameter str file_type_in: Input file type, e.g., 'bmp' or 'png.
    :parameter str file_type_out: Output file type, e.g., 'bmp' or 'png.

    :example:
    >>> _ = change_img_format(directory='MyDirectoryWImages', file_type_in='bmp', file_type_out='png')

    """
    if not os.path.isdir(directory):
        raise NotDirectoryError('SIMBA ERROR: {} is not a valid directory'.format(directory))
    files_found = glob.glob(directory + '/*.{}'.format(file_type_in))
    if len(files_found) < 1:
        raise NoFilesFoundError('SIMBA ERROR: No {} files (with .{} file ending) found in the {} directory'.format(file_type_in, file_type_in, directory))
    print('{} image files found in {}...'.format(str(len(files_found)), directory))
    for file_path in files_found:
        im = Image.open(file_path)
        save_name = file_path.replace('.' + str(file_type_in), '.' + str(file_type_out))
        im.save(save_name)
        os.remove(file_path)
    stdout_success(msg=f'SIMBA COMPLETE: Files in {directory} directory converted to {file_type_out}')


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
    save_path = os.path.join(dir, 'CLAHE_{}.avi'.format(file_name))
    video_meta_data = get_video_meta_data(file_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.AVI_CODEC.value)
    print('Applying CLAHE on video {}, this might take awhile...'.format(file_name))
    cap = cv2.VideoCapture(file_path)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['height'], video_meta_data['width']), 0)
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
                print('CLAHE converted frame {}/{}'.format(str(frm_cnt), str(video_meta_data['frame_count'])))
            else:
                cap.release()
                writer.release()
    except Exception as se:
        print(se.args)
        print('CLAHE conversion failed for video {}'.format(file_name))
        cap.release()
        writer.release()
        raise ValueError()

def extract_frame_range(file_path: Union[str, os.PathLike],
                        start_frame: int,
                        end_frame: int) -> None:
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
    check_int(name='start frame', value=start_frame, min_value=0)
    file_dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    check_int(name='end frame', value=end_frame, max_value=video_meta_data['frame_count'])
    frame_range = list(range(int(start_frame), int(end_frame) + 1))
    save_dir = os.path.join(file_dir, file_name + '_frames')
    cap = cv2.VideoCapture(file_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for frm_cnt, frm_number in enumerate(frame_range):
        cap.set(1, frm_number)
        ret, frame = cap.read()
        frm_save_path = os.path.join(save_dir, '{}.{}'.format(str(frm_number), 'png'))
        cv2.imwrite(frm_save_path,frame)
        print('Frame {} saved (Frame {}/{})'.format(str(frm_number), str(frm_cnt), str(len(frame_range))))
    stdout_success(msg=f'{str(len(frame_range))} frames extracted for video {file_name}')


def change_single_video_fps(file_path: Union[str, os.PathLike],
                            fps: int) -> None:
    """
    Change the fps of a single video file. Results are stored in the same directory as in the input file with
    the suffix ``_fps_new_fps``.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int fps: Fps of the new video file.

    :example:
    >>> _ = change_single_video_fps(file_path='project_folder/videos/Video_1.mp4', fps=15)
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='New fps', value=fps)
    video_meta_data = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if int(fps) == int(video_meta_data['fps']):
        SameInputAndOutputWarning(msg=f'The new fps is the same as the input fps for video {file_name} ({str(fps)})')
    save_path = os.path.join(dir_name, file_name + '_fps_{}{}'.format(str(fps), str(ext)))
    if os.path.isfile(save_path):
        FileExistWarning(msg=f'Overwriting existing file at {save_path}')
    command = str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -filter:v fps=fps=' + str(fps) + ' ' + '"' + save_path + '"'
    subprocess.call(command, shell=True)
    stdout_success(msg=f'SIMBA COMPLETE: FPS of video {file_name} changed from {str(video_meta_data["fps"])} to {str(fps)} and saved in directory {save_path}')

def change_fps_of_multiple_videos(directory: Union[str, os.PathLike],
                                  fps: int) -> None:
    """
    Change the fps of all video files in a folder. Results are stored in the same directory as in the input files with
    the suffix ``_fps_new_fps``.

    :parameter Union[str, os.PathLike] directory: Path to video file directory
    :parameter int fps: Fps of the new video files.

    :example:
    >>> _ = change_fps_of_multiple_videos(directory='project_folder/videos/Video_1.mp4', fps=15)
    """

    if not os.path.isdir(directory):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory))
    check_int(name='New fps', value=fps)
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in ['.avi', '.mp4', '.mov', '.flv']:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory'.format(directory))
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Converting FPS for {}...'.format(file_name))
        save_path = os.path.join(dir_name, file_name + '_fps_{}{}'.format(str(fps), str(ext)))
        command = str('ffmpeg -i ') + str(file_path) + ' -filter:v fps=fps=' + str(fps) + ' ' + '"' + save_path + '"'
        subprocess.call(command, shell=True)
        print('Video {} complete...'.format(file_name))
    stdout_success(msg=f'SIMBA COMPLETE: FPS of {str(len(video_paths))} videos changed to { str(fps)}')

def convert_video_powerpoint_compatible_format(file_path: Union[str, os.PathLike]) -> None:
    """
    Create MS PowerPoint compatible copy of a video file. The result is stored in the same directory as the
    input file with the ``_powerpointready`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.

    :example:
    >>> _ = convert_video_powerpoint_compatible_format(file_path='project_folder/videos/Video_1.mp4')
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_powerpointready.mp4')
    if os.path.isfile(save_name):
        raise FileExistError(msg='SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac ' + '"' + save_name + '"')
    print('Creating video in powerpoint compatible format... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')

def convert_to_mp4(file_path: Union[str, os.PathLike]) -> None:
    """
    Convert a video file to mp4 format. The result is stored in the same directory as the
    input file with the ``_converted.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.

    :example:
    >>> _ = convert_to_mp4(file_path='project_folder/videos/Video_1.avi')
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_converted.mp4')
    if os.path.isfile(save_name):
        raise FileExistError(msg='SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' ' + '"' + save_name + '"')
    print('Converting to mp4... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')


def video_to_greyscale(file_path: Union[str, os.PathLike]) -> None:
    """
    Convert a video file to greyscale mp4 format. The result is stored in the same directory as the
    input file with the ``_grayscale.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.

    :example:
    >>> _ = video_to_greyscale(file_path='project_folder/videos/Video_1.avi')
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_grayscale.mp4')
    if os.path.isfile(save_name):
        raise FileExistError(msg='SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -vf format=gray ' + '"' + save_name + '"')
    print('Converting to greyscale... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')


def superimpose_frame_count(file_path: Union[str, os.PathLike]) -> None:
    """
    Superimpose frame count on a video file. The result is stored in the same directory as the
    input file with the ``_frame_no.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.

    :example:
    >>> _ = superimpose_frame_count(file_path='project_folder/videos/Video_1.avi')
    """


    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_frame_no.mp4')
    print('Superimposing frame numbers...... ')
    try:
        command = (str('ffmpeg -y -i ') + '"' + file_path + '"' + ' -vf "drawtext=fontfile=Arial.ttf: text=\'%{frame_num}\': start_number=0: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy ' + '"' + save_name + '"')
        subprocess.check_output(command, shell=True)
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        simba_cw = os.path.dirname(simba.__file__)
        simba_font_path = Path(simba_cw, 'assets', 'UbuntuMono-Regular.ttf')
        command = 'ffmpeg -y -i ' + file_path + ' -vf "drawtext=fontfile={}:'.format(simba_font_path) + "text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" + '" ' + "-c:a copy " + save_name
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')

def remove_beginning_of_video(file_path: Union[str, os.PathLike],
                              time: int) -> None:
    """
    Remove N seconds from the beginning of a video file. The result is stored in the same directory as the
    input file with the ``_shorten.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int time: Number of seconds to remove from the beginning of the video.

    :example:
    >>> _ = remove_beginning_of_video(file_path='project_folder/videos/Video_1.avi', time=10)
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='Cut time', value=time)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_shorten.mp4')
    if os.path.isfile(save_name):
        raise FileExistError(msg='SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = (str('ffmpeg -ss ') + str(int(time)) + ' -i ' + '"' + str(file_path) + '"' + ' -c:v libx264 -c:a aac ' + '"' + save_name + '"')
    print('Shortening video... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')


def clip_video_in_range(file_path: Union[str, os.PathLike],
                        start_time: str,
                        end_time: str) -> None:
    """
    Clip video within a specific range. The result is stored in the same directory as the
    input file with the ``_clipped.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter str start_time: Start time in HH:MM:SS format.
    :parameter str end_time: End time in HH:MM:SS format.

    :example:
    >>> _ = clip_video_in_range(file_path='project_folder/videos/Video_1.avi', start_time='00:00:05', end_time='00:00:10')
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_clipped.mp4')
    if os.path.isfile(save_name):
        raise FileExistError(msg='SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -ss ' + str(start_time) + ' -to ' + str(end_time) + ' -async 1 ' + '"' + save_name + '"')
    print('Clipping video... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')

def downsample_video(file_path: Union[str, os.PathLike],
                     video_height: int,
                     video_width: int) -> None:
    """
    Down-sample a video file. The result is stored in the same directory as the
    input file with the ``_downsampled.mp4`` suffix.

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter int video_height: height of new video.
    :parameter int video_width: width of new video.

    :example:
    >>> _ = downsample_video(file_path='project_folder/videos/Video_1.avi', video_height=600, video_width=400)
    """

    check_int(name='Video height', value=video_height)
    check_int(name='Video width', value=video_width)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_downsampled.mp4')
    if os.path.isfile(save_name):
        raise FileExistError('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -vf scale=' + str(video_width) + ':' + str(video_height) + ' ' + '"' + save_name + '"' + ' -hide_banner')
    print('Down-sampling video... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')

def gif_creator(file_path: str,
                start_time: int,
                duration: int,
                width: int) -> None:
    """
    Create a sample gif from a video file. The result is stored in the same directory as the
    input file with the ``.gif`` file-ending.

    .. note::
       The height is auto-computed to retain aspect ratio

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter int start_time: Start time of the gif in relation to the video in seconds.
    :parameter int duration: Duration of the gif.
    :parameter int width: Width of the gif.

    :example:
    >>> _ = gif_creator(file_path='project_folder/videos/Video_1.avi', start_time=5, duration=10, width=600)
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='Start time', value=start_time)
    check_int(name='Duration', value=duration)
    check_int(name='Width', value=width)
    _ = get_video_meta_data(file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '.gif')
    if os.path.isfile(save_name):
        raise FileExistError('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
    command = 'ffmpeg -ss ' + str(start_time) + ' -t ' + str(duration) + ' -i ' + '"' + str(file_path) + '"' + ' -filter_complex "[0:v] fps=15,scale=w=' + str(width) + ':h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" ' + '"' + str(save_name) + '"'
    print('Creating gif sample... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'SIMBA COMPLETE: Video converted! {save_name} generated!')


def batch_convert_video_format(directory: Union[str, os.PathLike],
                               input_format: str,
                               output_format: str) -> None:
    """
    Batch convert all videos in a folder of specific format into a different video format. The results are
    stored in the same directory as the input files.

    :parameter Union[str, os.PathLike] directory: Path to video file directory.
    :parameter str input_format: Format of the input files (e.g., avi).
    :parameter str output_format: Format of the output files (e.g., mp4).

    :example:
    >>> _ = gif_creator(directory='project_folder/videos', input_format='avi', output_format='mp4')
    """

    if not os.path.isdir(directory):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory))
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() == '.{}'.format(input_format.lower()):
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: No files with .{} file ending found in the {} directory'.format(input_format, directory))
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Processing video {}...'.format(file_name))
        save_path = os.path.join(dir_name, file_name + '.{}'.format(output_format.lower()))
        if os.path.isfile(save_path):
            raise FileExistError(msg='SIMBA ERROR: The outfile file already exist: {}.'.format(save_path))
        command = 'ffmpeg -y -i ' + '"' + file_path + '"' + ' -c:v libx264 -crf 5 -preset medium -c:a libmp3lame -b:a 320k '+'"' + save_path + '"'
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        print('Video {} complete, (Video {}/{})...'.format(file_name, str(file_cnt+1), str(len(video_paths))))

    stdout_success(msg=f'SIMBA COMPLETE: {str(len(video_paths))} videos converted in {directory} directory!')

def batch_create_frames(directory: Union[str, os.PathLike]) -> None:
    """
    Extract all frames for all videos in a directory. Results are stored within sub-directories in the input
    directory named according to the video files.

    :parameter str directory: Path to directory containing video files.

    :example:
    >>> _ = batch_create_frames(directory='project_folder/videos')
    """

    if not os.path.isdir(directory):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory))
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in ['.avi', '.mp4', '.mov', '.flv']:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory'.format(directory))
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Processing video {}...'.format(file_name))
        save_dir = os.path.join(dir_name, file_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
        print('Video {} complete, (Video {}/{})...'.format(file_name, str(file_cnt + 1), str(len(video_paths))))
    stdout_success(msg=f'{str(len(video_paths))} videos converted into frames in {directory} directory!')

def extract_frames_single_video(file_path: Union[str, os.PathLike]) -> None:
    """
    Extract all frames for a single. Results are stored within a sub-directory in the same
    directory as the input file.

    :parameter str file_path: Path to video file.

    :example:
    >>> _ = extract_frames_single_video(file_path='project_folder/videos/Video_1.mp4')
    """

    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    save_dir = os.path.join(dir_name, file_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print('Processing video {}...'.format(file_name))
    video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
    stdout_success(msg=f'Video {file_name} converted to images in {dir_name} directory!')

def multi_split_video(file_path: Union[str, os.PathLike],
                      start_times: List[str],
                      end_times: List[str]) -> None:
    """
    Divide a video file into multiple video files from specified start and stop times.

    :parameter str file_path: Path to input video file.
    :parameter List[str] start_times: Start times in HH:MM:SS format.
    :parameter List[str] end_times: End times in HH:MM:SS format.

    :example:
    >>> _ = multi_split_video(file_path='project_folder/videos/Video_1.mp4', start_times=['00:00:05', '00:00:20'], end_times=['00:00:10', '00:00:25'])
    """

    check_file_exist_and_readable(file_path=file_path)
    video_meta_data = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    r = re.compile('.{2}:.{2}:.{2}')
    for start_time_cnt, start_time in enumerate(start_times):
        if (len(start_time) != 8) or (not r.match(start_time)) or (re.search('[a-zA-Z]', start_time)):
            raise InvalidInputError(msg=f'Start time for clip {str(start_time_cnt+1)} is should be in the format XX:XX:XX where X is an integer between 0-9')
    for end_time_cnt, end_time in enumerate(end_times):
        if (len(end_time) != 8) or (not r.match(end_time)) or (re.search('[a-zA-Z]', end_time)):
            raise InvalidInputError(msg=f'End time for clip {str(end_time_cnt+1)} is should be in the format XX:XX:XX where X is an integer between 0-9')
    for clip_cnt, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        start_h, start_m, start_s = start_time.split(':')
        end_h, end_m, end_s = end_time.split(':')
        start_in_s = int(start_h) * 3600 + int(start_m) * 60 + int(start_s)
        end_in_s = int(end_h) * 3600 + int(end_m) * 60 + int(end_s)
        if end_in_s < start_in_s:
            raise InvalidInputError(f'Clip {str(clip_cnt+1)} has has an end-time which is before the start-time.')
        if end_in_s == start_in_s:
            raise InvalidInputError(msg=f'Clip {str(clip_cnt+1)} has the same start time and end time.')
        if end_in_s > video_meta_data['video_length_s']:
            raise InvalidInputError(f'Clip {str(clip_cnt + 1)} has end time at {str(end_in_s)} seconds into the video, which is greater then the lenth of the video ({str(video_meta_data["video_length_s"])}s).')
        save_path = os.path.join(dir_name, file_name + '_{}'.format(str(clip_cnt+1)) + '.mp4')
        if os.path.isfile(save_path):
            raise FileExistError(msg=f'The outfile file already exist: {save_path}.')
        command = (str('ffmpeg -i ') + '"' + file_path + '"' + ' -ss ' + start_time + ' -to ' + end_time + ' -async 1 ' + '"' + save_path + '"')
        print('Processing video clip {}...'.format(str(clip_cnt+1)))
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'Video {file_name} converted into {str(len(start_times))} clips in directory {dir_name}!')

def crop_single_video(file_path: Union[str, os.PathLike]) -> None:
    """
    Crop a single video using cv2.selectROI interface. Results is saved in the same directory as input video with the
    ``_cropped.mp4`` suffix`.

    :parameter str file_path: Path to video file.

    :example:
    >>> _ = crop_single_video(file_path='project_folder/videos/Video_1.mp4')
    """

    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    cap = cv2.VideoCapture(file_path)
    cap.set(1, 0)
    ret, frame = cap.read()
    cv2.namedWindow('Select cropping ROI', cv2.WINDOW_NORMAL)
    ROI = cv2.selectROI("Select cropping ROI", frame)
    width = int(abs(ROI[0] - (ROI[2] + ROI[0])))
    height = int(abs(ROI[2] - (ROI[3] + ROI[2])))
    top_lext_x, top_left_y = int(ROI[0]), int(ROI[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if (width == 0 and height == 0) or (width + height + top_lext_x + top_left_y == 0):
        raise CountError(msg='CROP FAILED: Cropping height and width are both 0. Please try again.')
    save_path = os.path.join(dir_name, file_name + '_cropped.mp4')
    if os.path.isfile(save_path):
        raise FileExistError(msg='SIMBA ERROR: The out file file already exist: {}.'.format(save_path))
    command = str('ffmpeg -y -i ') + '"' + str(file_path) + '"' + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(top_lext_x) + ':' + str(top_left_y) + '" ' + str('-c:v libx264 -crf 21 -c:a copy ') + '"' + str(save_path) + '"'
    subprocess.call(command, shell=True)
    stdout_success(f'Video {file_name} cropped and saved at {save_path}')

def crop_multiple_videos(directory_path: Union[str, os.PathLike],
                         output_path: Union[str, os.PathLike]) -> None:
    """
    Crop multiple videos in a folder according to crop-coordinates defines in the **first** video.

    :parameter str directory_path: Directory containing input videos.
    :parameter str output_path: Directory where to save the cropped videos.

    :example:
    >>> _ = crop_multiple_videos(directory_path='project_folder/videos', output_path='project_folder/videos/my_new_folder')
    """

    if not os.path.isdir(directory_path):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory_path))
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory_path + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in ['.avi', '.mp4', '.mov', '.flv']:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory'.format(directory_path))
    cap = cv2.VideoCapture(file_paths_in_folder[0])
    cap.set(1, 0)
    ret, frame = cap.read()
    cv2.namedWindow('Select cropping ROI', cv2.WINDOW_NORMAL)
    ROI = cv2.selectROI("Select cropping ROI", frame)
    width = int(abs(ROI[0] - (ROI[2] + ROI[0])))
    height = int(abs(ROI[2] - (ROI[3] + ROI[2])))
    top_lext_x, top_left_y = int(ROI[0]), int(ROI[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if (width == 0 and height == 0) or (width + height + top_lext_x + top_left_y == 0):
        raise CountError(msg='CROP FAILED: Cropping height and width are both 0. Please try again.')
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Cropping video {}...'.format(file_name))
        _ = get_video_meta_data(file_path)
        save_path = os.path.join(output_path, file_name + '_cropped.mp4')
        command = str('ffmpeg -i ') + '"' + file_path + '"' + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(top_lext_x) + ':' + str(top_left_y) + '" ' + str('-c:v libx264 -crf 21 -c:a copy ') + '"' + str(save_path) + '"'
        subprocess.call(command, shell=True)
        print('Video {} cropped (Video {}/{})'.format(file_name, str(file_cnt+1), str(len(video_paths))))
    stdout_success(msg=f'{str(len(video_paths))} videos cropped and saved in {directory_path} directory')

def frames_to_movie(directory: Union[str, os.PathLike],
                    fps: int,
                    bitrate: int,
                    img_format: str) -> None:

    """
    Merge all image files in a folder to a mp4 video file. Video file is stored in the same directory as the
    input directory sub-folder.

    .. note::
       The Image files have to have ordered numerical names e.g., ``1.png``, ``2.png`` etc...

    :parameter str directory: Directory containing the images.
    :parameter int fps: The frame rate of the output video.
    :parameter int bitrate: The bitrate of the output video (e.g., 32000).
    :parameter str img_format: The format of the input image files (e.g., ``png``).

    :example:
    >>> _ = crop_multiple_videos(directory_path='project_folder/video_img', fps=15, bitrate=32000, img_format='png')
    """


    if not os.path.isdir(directory):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory))
    check_int(name='FPS', value=fps)
    check_int(name='BITRATE', value=bitrate)
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    img_paths_in_folder = [x for x in file_paths_in_folder if Path(x).suffix[1:] == img_format]
    if len(img_paths_in_folder) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: Zero images of file-type {} found in {} directory'.format(img_format, directory))
    img = cv2.imread(img_paths_in_folder[0])
    img_h, img_w = int(img.shape[0]), int(img.shape[1])
    ffmpeg_fn = os.path.join(directory, '%d.{}'.format(img_format))
    save_path = os.path.join(os.path.dirname(directory), os.path.basename(directory) + '.mp4')
    command = str('ffmpeg -y -r ' + str(fps) + ' -f image2 -s ' + str(img_h) + 'x' + str(img_w) + ' -i ' + '"' + ffmpeg_fn + '"' + ' -vcodec libx264 -b ' + str(bitrate) + 'k ' + '"' + str(save_path) + '"')
    print('Creating {} from {} images...'.format(os.path.basename(save_path), str(len(img_paths_in_folder))))
    subprocess.call(command, shell=True)
    stdout_success(msg=f'Video created at {save_path}')


def video_concatenator(video_one_path: Union[str, os.PathLike],
                       video_two_path: Union[str, os.PathLike],
                       resolution: Union[int, str],
                       horizontal: bool) -> None:

    """
    Concatenate two videos to a single video

    :param str video_one_path: Path to the first video in the concatenated video
    :param str video_two_path: Path to the second video in the concatenated video
    :param int or str resolution: If str, then the name of the video which resolution you want to retain. E.g., `Video_1`. Else int, representinmg the video width (if vertical concat) or height (if horizontal concat). Aspect raio will be retained.
    :param horizontal: If true, then horizontal concatenation. Else vertical concatenation.

    :example:
    >>> video_concatenator(video_one_path='project_folder/videos/Video_1.mp4', video_two_path='project_folder/videos/Video_2.mp4', resolution=800, horizontal=True)
    """


    for file_path in [video_one_path, video_two_path]:
        check_file_exist_and_readable(file_path=file_path)
        _ = get_video_meta_data(file_path)
    if type(resolution) is int:
        video_meta_data = {}
        if horizontal:
            video_meta_data['height'] = resolution
        else:
            video_meta_data['width'] = resolution
    elif resolution is 'Video 1':
        video_meta_data = get_video_meta_data(video_one_path)
    else:
        video_meta_data = get_video_meta_data(video_one_path)
    dir, file_name_1, _ = get_fn_ext(video_one_path)
    _, file_name_2, _ = get_fn_ext(video_two_path)
    print('Concatenating videos...')
    save_path = os.path.join(dir, file_name_1 + file_name_2 + '_concat.mp4')
    if horizontal:
        command = 'ffmpeg -y -i "{}" -i "{}" -filter_complex "[0:v]scale=-1:{}[v0];[v0][1:v]hstack=inputs=2" "{}"'.format(video_one_path, video_two_path, video_meta_data['height'], save_path)
    else:
        command = 'ffmpeg -y -i "{}" -i "{}" -filter_complex "[0:v]scale={}:-1[v0];[v0][1:v]vstack=inputs=2" "{}"'.format(video_one_path, video_two_path, video_meta_data['width'], save_path)
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    stdout_success(msg=f'Videos concatenated and saved at {save_path}')


class VideoRotator(ConfigReader):
    """
    GUI Tool for rotating video. Rotated video is saved with the ``_rotated_DATETIME.mp4`` suffix.

    :parameter str input_path: Path to video to rotate.
    :parameter str output_dir: Directory where to save the rotated video.

    :example:
    >>> VideoRotator(input_path='project_folder/videos/Video_1.mp4', output_dir='project_folder/videos')
    """


    def __init__(self,
                 input_path: Union[str, os.PathLike],
                 output_dir: Union[str, os.PathLike]) -> None:

        _, self.cpu_cnt  = find_core_cnt()
        self.save_dir = output_dir
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        if os.path.isfile(input_path):
            self.video_paths = [input_path]
        else:
            self.video_paths = find_all_videos_in_directory(directory=input_path, as_dict=True).values()
            check_if_filepath_list_is_empty(filepaths=self.video_paths, error_msg=f'No videos found in {input_path} directory')

    def __insert_img(self, img: np.array):
        current_frm_pil = Image.fromarray(img)
        current_frm_pil.thumbnail(MAX_FRM_SIZE, Image.ANTIALIAS)
        current_frm_pil = ImageTk.PhotoImage(master=self.main_frm, image=current_frm_pil)
        self.video_frame = Label(self.main_frm, image=current_frm_pil)
        self.video_frame.image = current_frm_pil
        self.video_frame.grid(row=0, column=0)

    def __rotate(self, value: int, img: np.array):
        self.dif_angle += value
        rotation_matrix = cv2.getRotationMatrix2D((self.video_meta_data['width'] / 2, self.video_meta_data['height'] / 2), self.dif_angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (self.video_meta_data['width'], self.video_meta_data['height']))
        self.__insert_img(img=img)

    def __run_rotation(self):
        self.main_frm.destroy()
        start = time.time()
        for video_cnt, (video_path, rotation) in enumerate(self.results.items()):
            cap = cv2.VideoCapture(video_path)
            _, name, _ = get_fn_ext(filepath=video_path)
            rotation_matrix = cv2.getRotationMatrix2D((self.video_meta_data['width'] / 2, self.video_meta_data['height'] / 2), rotation, 1)
            save_path = os.path.join(self.save_dir, f'{name}_rotated_{self.datetime}.mp4')
            video_meta = get_video_meta_data(video_path=video_path)
            fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
            writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'],(video_meta['width'], video_meta['height']))
            img_cnt = 0
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                img = cv2.warpAffine(img, rotation_matrix,
                                     (self.video_meta_data['width'], self.video_meta_data['height']))
                writer.write(img)
                img_cnt += 1
                print(
                    f'Rotating frame {img_cnt}/{video_meta["frame_count"]} (Video {video_cnt + 1}/{len(self.results.keys())}) ')
            cap.release()
            writer.release()
        stdout_success(msg=f'All videos rotated and saved in {self.save_dir}', elapsed_time=str(round((time.time() - start), 2)))

    def __save(self):
        process = None
        self.results[self.file_path] = self.dif_angle
        if len(self.results.keys()) == len(self.video_paths):
            process = multiprocessing.Process(target=self.__run_rotation())
            process.start()
        else:
            self.__run_interface(file_path=self.video_paths[len(self.results.keys())-1])
        if process is not None:
            process.join()

    def __bind_keys(self):
        self.main_frm.bind('<Left>', lambda x: self.__rotate(value = 1, img=self._orig_img))
        self.main_frm.bind('<Right>', lambda x: self.__rotate(value = -1, img=self._orig_img))
        self.main_frm.bind('<Escape>', lambda x: self.__save())

    def __run_interface(self, file_path: str):
        self.dif_angle = 0
        print(file_path)
        self.video_meta_data = get_video_meta_data(video_path=file_path)
        self.file_path = file_path
        _, self.video_name, _ = get_fn_ext(filepath=file_path)
        self.main_frm = Toplevel()
        self.main_frm.title(f'ROTATE VIDEO {self.video_name}')
        self.video_frm = Frame(self.main_frm)
        self.video_frm.grid(row=0, column=0)
        self.instruction_frm = Frame(self.main_frm, width=100, height=100)
        self.instruction_frm.grid(row=0, column=2, sticky=NW)
        self.key_lbls = Label(self.instruction_frm,
                                    text='\n\n Navigation: '
                                         '\n Left arrow = 1° left' 
                                         '\n Right arrow = 1° right'
                                         '\n Esc = Save')

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
    Extract all frames from all videos in a directory. The results are saved in the project_folder/frames/input directory of the SimBA project

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter str directory: path to file or folder containing videos in mp4 and/or avi format.

    :example:
    >>> extract_frames_from_all_videos_in_directory(config_path='project_folder/project_config.ini', source='/tests/test_data/video_tests')
    """

    timer = SimbaTimer(start=True)
    video_paths, video_types = [], ['.avi', '.mp4']
    files_in_folder = glob.glob(directory + '/*')
    for file_path in files_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in video_types:
            video_paths.append(file_path)
    if len(video_paths) == 0:
        raise NoFilesFoundError(msg='SIMBA ERROR: 0 video files in mp4 or avi format found in {}'.format(directory))
    config = read_config_file(config_path)
    project_path = read_config_entry(config, 'General settings', 'project_path', data_type='folder_path')

    print('Extracting frames for {} videos into project_folder/frames/input directory...'.format(len(video_paths)))
    for video_path in video_paths:
        dir_name, video_name, ext = get_fn_ext(video_path)
        save_path = os.path.join(project_path, 'frames', 'input', video_name)
        if not os.path.exists(save_path): os.makedirs(save_path)
        else: print(f'Frames for video {video_name} already extracted. SimBA is overwriting prior frames...')
        video_to_frames(video_path, save_path, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(f'Frames created for {str(len(video_paths))} videos', elapsed_time=timer.elapsed_time_str)


def copy_img_folder(config_path: Union[str, os.PathLike], source: Union[str, os.PathLike]) -> None:
    """
    Copy directory of png files to the SimBA project. The directory is stored in the project_folder/frames/input folder of the SimBA project

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter str source: path to image folder outside SimBA project.

    :example:
    >>> copy_img_folder(config_path='MySimBAprojectConfig', source='/DirectoryWithVideos/')


    """
    timer = SimbaTimer(start=True)
    if not os.path.isdir(source):
        raise NotDirectoryError(msg=f'SIMBA ERROR: source {source} is not a directory.')
    if len(glob.glob(source + '/*.png')) == 0:
        raise NoFilesFoundError(msg=f'SIMBA ERROR: source {source} does not contain any .png files.')
    input_basename = os.path.basename(source)
    config = read_config_file(config_path)
    project_path = read_config_entry(config, ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value, data_type='folder_path')
    input_frames_dir = os.path.join(project_path, Paths.INPUT_FRAMES_DIR.value)
    destination = os.path.join(input_frames_dir, input_basename)
    if os.path.isdir(destination):
        raise DirectoryExistError(msg=f'SIMBA ERROR: {destination} already exist in SimBA project.')
    print(f'Importing image files for {input_basename}...')
    shutil.copytree(source, destination)
    timer.stop_timer()
    stdout_success(msg=f'{destination} imported to SimBA project', elapsed_time=timer.elapsed_time_str)


# r = VideoRotator(input_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Testing/Together_1_downsampled.mp4',
#              output_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Blah')
# r.run()


# video_concatenator(video_one_path='/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/frames/output/gantt_plots/SI_DAY3_308_CD1_PRESENT_2.mp4',
#                    video_two_path= '/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/frames/output/gantt_plots/SI_DAY3_308_CD1_PRESENT.mp4',
#                    resolution='Video 1',
#                    horizontal=False)



