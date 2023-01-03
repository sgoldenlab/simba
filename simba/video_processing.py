import glob, os
from simba.read_config_unit_tests import (check_file_exist_and_readable,
                                          check_int)
from simba.misc_tools import get_fn_ext, get_video_meta_data
import cv2
from pathlib import Path
from PIL import Image
from simba.extract_frames_fast import video_to_frames
import re
import subprocess
import simba



def change_img_format(directory: str,
                      file_type_in: str,
                      file_type_out: str):
    """
    Helper to convert file type of all image files in a folder.

    Parameters
    ----------
    directory: str
        Path to directory holding image files
    file_type_in: str
        Input file type
    file_type_out: str
        Output file type

    Returns
    -------
    None

    Example
    ----------
    >>> _ = change_img_format(directory='MyDirectoryWImages', file_type_in='bmp', file_type_out='png')

    """
    if not os.path.isdir(directory):
        print('SIMBA ERROR: {} is not a valid directory'.format(directory))
        raise NotADirectoryError
    files_found = glob.glob(directory + '/*.{}'.format(file_type_in))
    if len(files_found) < 1:
        print('SIMBA ERROR: No {} files (with .{} file ending) found in the {} directory'.format(file_type_in, file_type_in, directory))
        raise ValueError
    print('{} image files found in {}...'.format(str(len(files_found)), directory))
    for file_path in files_found:
        im = Image.open(file_path)
        save_name = file_path.replace('.' + str(file_type_in), '.' + str(file_type_out))
        im.save(save_name)
        os.remove(file_path)
    print('SIMBA COMPLETE: Files in {} directory converted to {}'.format(directory, file_type_out))


def clahe_enhance_video(file_path: str):
    """
    Helper to convert a single video file to clahe-enhanced greyscale .avi file. The result is saved with prefix
    ``CLAHE_`` in the same directory as in the input file.

    Parameters
    ----------
    file_path: str
        Path to video file.

    Returns
    -------
    None
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, file_ext = get_fn_ext(filepath=file_path)
    save_path = os.path.join(dir, 'CLAHE_{}.avi'.format(file_name))
    video_meta_data = get_video_meta_data(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
    except Exception as se:
        print(se.args)
        print('CLAHE conversion failed for video {}'.format(file_name))
        raise ValueError


def extract_frame_range(file_path: str,
                        start_frame: int,
                        end_frame: int):
    """
    Helper to extract a user-defined range of frames from a video file and save those in png format. Images
    are saved in a folder with the suffix `_frames` within the same directory as the video file.

    Parameters
    ----------
    file_path: str
        Path to video file.
    start_frame: int
        First frame in range to extract.
    end_frame: int
        Last frame in range to extract.

    Returns
    -------
    None
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
    print('SIMBA COMPLETE: {} frames extracted for video {}'.format(str(len(frame_range)), file_name))




def change_single_video_fps(file_path: str,
                            fps: int):
    """
    Helper to change the fps of a single video file. Results are stored in the same directory as in the input file with
    the suffix ``_fps_new_fps``.

    Parameters
    ----------
    file_path: str
        Path to video file.
    fps: int
        Fps of the new video file
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='New fps', value=fps)
    video_meta_data = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if int(fps) == int(video_meta_data['fps']):
        print('SIMBA WARNING: The new fps is the same as the input fps for video {}'.format(file_name))
    save_path = os.path.join(dir_name, file_name + '_fps_{}{}'.format(str(fps), str(ext)))
    if os.path.isfile(save_path):
        print('SIMBA WARNING: Overwriting existing file at {}'.format(save_path))
    command = str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -filter:v fps=fps=' + str(fps) + ' ' + '"' + save_path + '"'
    subprocess.call(command, shell=True)
    print('SIMBA COMPLETE: FPS of video {} changed from {} to {} and saved in directory {}'.format(file_name, str(video_meta_data['fps']), str(fps), save_path))


def change_fps_of_multiple_videos(directory: str,
                                  fps: int):
    """
    Helper to change the fps of video files in a folder. Results are stored in the same directory as in the input files with
    the suffix ``_fps_new_fps``.

    Parameters
    ----------
    directory: str
        Path to directory with video files
    fps: int
        Fps of the new video file
    """

    if not os.path.isdir(directory):
        print('SIMBA ERROR: {} is not a valid directory'.format(directory))
        raise NotADirectoryError()
    check_int(name='New fps', value=fps)
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in ['.avi', '.mp4', '.mov', '.flv']:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        print('SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory'.format(directory))
        raise ValueError()
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Converting FPS for {}...'.format(file_name))
        save_path = os.path.join(dir_name, file_name + '_fps_{}{}'.format(str(fps), str(ext)))
        command = str('ffmpeg -i ') + str(file_path) + ' -filter:v fps=fps=' + str(fps) + ' ' + '"' + save_path + '"'
        subprocess.call(command, shell=True)
        print('Video {} complete...'.format(file_name))
    print('SIMBA COMPLETE: FPS of {} videos changed to {}'.format(str(len(video_paths)), str(fps)))

def convert_video_powerpoint_compatible_format(file_path: str):
    """
    Helper to make a powerpoint compatible copy of a video file. The results is stored in the same directory as the
    input file with the ``_powerpointready`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.

    """


    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_powerpointready.mp4')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac ' + '"' + save_name + '"')
    print('Creating video in powerpoint compatible format... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))

def convert_to_mp4(file_path: str):
    """
    Helper to convert a video file to mp4 format. The results is stored in the same directory as the
    input file with the ``_converted.mp4`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.

    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_converted.mp4')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' ' + '"' + save_name + '"')
    print('Converting to mp4... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))


def video_to_greyscale(file_path: str):
    """
    Helper to convert a video file to greyscale mp4 format. The results is stored in the same directory as the
    input file with the ``_grayscale.mp4`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.

    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_grayscale.mp4')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -vf format=gray ' + '"' + save_name + '"')
    print('Converting to greyscale... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))


def superimpose_frame_count(file_path: str):
    """
    Helper to superimpose frame count on a video file. The results is stored in the same directory as the
    input file with the ``_frame_no.mp4`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.

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
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))



def remove_beginning_of_video(file_path: str,
                              time: int):
    """
    Helper to remove N seconds from the beginning of a video file. The results is stored in the same directory as the
    input file with the ``_shorten.mp4`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='Cut time', value=time)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_shorten.mp4')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = (str('ffmpeg -ss ') + str(int(time)) + ' -i ' + '"' + str(file_path) + '"' + ' -c:v libx264 -c:a aac ' + '"' + save_name + '"')
    print('Shortening video... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))



def clip_video_in_range(file_path: str,
                        start_time: str,
                        end_time: str):
    """
    Helper to clip video in a specific range. The results is stored in the same directory as the
    input file with the ``_clipped.mp4`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.
    start_time: str
        Start time of clip in HH:MM:SS format.
    end_time: str
        End time of clip in HH:MM:SS format.
    """

    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_clipped.mp4')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -ss ' + str(start_time) + ' -to ' + str(end_time) + ' -async 1 ' + '"' + save_name + '"')
    print('Clipping video... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))

def downsample_video(file_path: str,
                     video_height: int,
                     video_width: int):
    """
    Helper to down-sample a video file. The results is stored in the same directory as the
    input file with the ``_downsampled.mp4`` suffix.

    Parameters
    ----------
    file_path: str
        Path to video file.
    video_height: int
        The height of the output video.
    video_width: int
        The width of the output video.
    """

    check_int(name='Video height', value=video_height)
    check_int(name='Video width', value=video_width)
    check_file_exist_and_readable(file_path=file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '_downsampled.mp4')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = (str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -vf scale=' + str(video_width) + ':' + str(video_height) + ' ' + '"' + save_name + '"' + ' -hide_banner')
    print('Down-sampling video... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))


def gif_creator(file_path: str,
                start_time: int,
                duration: int,
                width: int):
    """
    Helper to create a sample gif from a video file. The results is stored in the same directory as the
    input file with the ``.mp4`` file-ending.

    Parameters
    ----------
    file_path: str
        Path to video file.
    start_time: int
        The time of the first frame in the gif in seconds.
    duration: int
        The duration of the gif in seconds.
    width: int
        The width of the gif in pixels. The aspect ratio of the gif will be the same as in the video, i.e.,
        height is automatically computed.
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='Start time', value=start_time)
    check_int(name='Duration', value=duration)
    check_int(name='Width', value=width)
    _ = get_video_meta_data(file_path)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, file_name + '.gif')
    if os.path.isfile(save_name):
        print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_name))
        raise FileExistsError()
    command = 'ffmpeg -ss ' + str(start_time) + ' -t ' + str(duration) + ' -i ' + '"' + str(file_path) + '"' + ' -filter_complex "[0:v] fps=15,scale=w=' + str(width) + ':h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" ' + '"' + str(save_name) + '"'
    print('Creating gif sample... ')
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video converted! {} generated!'.format(save_name))



def batch_convert_video_format(directory: str,
                               input_format: str,
                               output_format: str):
    """
    Helper to batch convert all videos in a folder of specific format into a different video format. The results are
    stored in the same directory as the input files.

    Parameters
    ----------
    directory: str
        Path to directory containing video files.
    input_format: str
        Format of the input files (e.g., avi)
    output_format: str
        Format of the output files (e.g., mp4)
    """

    if not os.path.isdir(directory):
        print('SIMBA ERROR: {} is not a valid directory'.format(directory))
        raise NotADirectoryError()
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() == '.{}'.format(input_format.lower()):
            video_paths.append(file_path)
    if len(video_paths) < 1:
        print('SIMBA ERROR: No files with .{} file ending found in the {} directory'.format(input_format, directory))
        raise ValueError()
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Processing video {}...'.format(file_name))
        save_path = os.path.join(dir_name, file_name + '.{}'.format(output_format.lower()))
        if os.path.isfile(save_path):
            print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_path))
            raise FileExistsError()
        command = 'ffmpeg -y -i ' + '"' + file_path + '"' + ' -c:v libx264 -crf 5 -preset medium -c:a libmp3lame -b:a 320k '+'"' + save_path + '"'
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        print('Video {} complete, (Video {}/{})...'.format(file_name, str(file_cnt+1), str(len(video_paths))))
    print('SIMBA COMPLETE: {} videos converted in {} directory!'.format(str(len(video_paths)), directory))


def batch_create_frames(directory: str):
    """
    Helper to extract all frames for all videos in a directory. Results are stored within sub-directories in the input
    directory named according to the video files.

    Parameters
    ----------
    directory: str
        Path to directory containing video files.
    """

    if not os.path.isdir(directory):
        print('SIMBA ERROR: {} is not a valid directory'.format(directory))
        raise NotADirectoryError()
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in ['.avi', '.mp4', '.mov', '.flv']:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        print('SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory'.format(directory))
        raise ValueError()
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Processing video {}...'.format(file_name))
        save_dir = os.path.join(dir_name, file_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
        print('Video {} complete, (Video {}/{})...'.format(file_name, str(file_cnt + 1), str(len(video_paths))))
    print('SIMBA COMPLETE: {} videos converted into frames in {} directory!'.format(str(len(video_paths)), directory))


def extract_frames_single_video(file_path: str):
    """
    Helper to extract all frames for a single. Results are stored within a sub-directory in the same
    directory as the input file.

    Parameters
    ----------
    file_path: str
        Path to video file.
    """

    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    save_dir = os.path.join(dir_name, file_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print('Processing video {}...'.format(file_name))
    video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
    print('SIMBA COMPLETE: Video {} converted to images in {} directory!'.format(file_name, dir_name))


def multi_split_video(file_path: str,
                      start_times: list,
                      end_times: list):
    """
    Helper divide a video file into multiple video files from specified start and stop times.

    Parameters
    ----------
    file_path: str
        Path to video file.
    start_times: list
        List of clip start times in [HH:MM:SS] format.
    end_times: list
        List of clip end times in [HH:MM:SS] format.
    """

    check_file_exist_and_readable(file_path=file_path)
    video_meta_data = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    r = re.compile('.{2}:.{2}:.{2}')
    for start_time_cnt, start_time in enumerate(start_times):
        if (len(start_time) != 8) or (not r.match(start_time)) or (re.search('[a-zA-Z]', start_time)):
            print('SIMBA ERROR: Start time for clip {} is should be in the format XX:XX:XX where X is an integer between 0-9'.format(str(start_time_cnt+1)))
            raise ValueError()
    for end_time_cnt, end_time in enumerate(end_times):
        if (len(end_time) != 8) or (not r.match(end_time)) or (re.search('[a-zA-Z]', end_time)):
            print('SIMBA ERROR: Start time for clip {} is should be in the format XX:XX:XX where X is an integer between 0-9'.format(str(end_time_cnt+1)))
            raise ValueError()

    for clip_cnt, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        start_h, start_m, start_s = start_time.split(':')
        end_h, end_m, end_s = end_time.split(':')
        start_in_s = int(start_h) * 3600 + int(start_m) * 60 + int(start_s)
        end_in_s = int(end_h) * 3600 + int(end_m) * 60 + int(end_s)
        if end_in_s < start_in_s:
            print('SIMBA ERROR: Clip {} has has an end-time which is before the start-time.'.format(str(clip_cnt+1)))
            raise ValueError()
        if end_in_s == start_in_s:
            print('SIMBA ERROR: Clip {} has the same start time and end time.'.format(str(clip_cnt+1)))
            raise ValueError()
        if end_in_s > video_meta_data['video_length_s']:
            print('SIMBA ERROR: Clip {} has end time at {} seconds into the video, which is greater then the lenth of the video ({}s) .'.format(str(clip_cnt + 1), str(end_in_s), str(video_meta_data['video_length_s'])))
            raise ValueError()
        save_path = os.path.join(dir_name, file_name + '_{}'.format(str(clip_cnt+1)) + '.mp4')
        if os.path.isfile(save_path):
            print('SIMBA ERROR: The outfile file already exist: {}.'.format(save_path))
            raise FileExistsError()
        command = (str('ffmpeg -i ') + '"' + file_path + '"' + ' -ss ' + start_time + ' -to ' + end_time + ' -async 1 ' + '"' + save_path + '"')
        print('Processing video clip {}...'.format(str(clip_cnt+1)))
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    print('SIMBA COMPLETE: Video {} converted into {} clips in directory {}!'.format(file_name, str(len(start_times)), dir_name))


def crop_single_video(file_path: str):
    """
    Helper to crop a single video,

    Parameters
    ----------
    file_path: str
        Path to video file.
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
        print('CROP FAILED: Cropping height and width are both 0. Please try again.')
        raise ValueError()
    save_path = os.path.join(dir_name, file_name + '_cropped.mp4')
    if os.path.isfile(save_path):
        print('SIMBA ERROR: The out file file already exist: {}.'.format(save_path))
        raise FileExistsError()
    command = str('ffmpeg -y -i ') + '"' + str(file_path) + '"' + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(top_lext_x) + ':' + str(top_left_y) + '" ' + str('-c:v libx264 -crf 21 -c:a copy ') + '"' + str(save_path) + '"'
    subprocess.call(command, shell=True)
    print('SIMBA COMPLETE: Video {} cropped and saved at {}'.format(file_name, save_path))


def crop_multiple_videos(directory_path: str,
                         output_path: str):
    """
    Helper to crop multiple videos in a folder according to coordinates defines in one video.

    Parameters
    ----------
    directory_path: str
        Path to directory holding video files.
    output_path:
        Directory where to store the cropped videos.
    """

    if not os.path.isdir(directory_path):
        print('SIMBA ERROR: {} is not a valid directory'.format(directory_path))
        raise NotADirectoryError()
    video_paths = []
    file_paths_in_folder = [f for f in glob.glob(directory_path + '/*') if os.path.isfile(f)]
    for file_path in file_paths_in_folder:
        _, _, ext = get_fn_ext(filepath=file_path)
        if ext.lower() in ['.avi', '.mp4', '.mov', '.flv']:
            video_paths.append(file_path)
    if len(video_paths) < 1:
        print('SIMBA ERROR: No files with .mp4, .avi, .mov, .flv file ending found in the {} directory'.format(directory))
        raise ValueError()
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
        print('CROP FAILED: Cropping height and width are both 0. Please try again.')
        raise ValueError()
    for file_cnt, file_path in enumerate(video_paths):
        dir_name, file_name, ext = get_fn_ext(filepath=file_path)
        print('Cropping video {}...'.format(file_name))
        _ = get_video_meta_data(file_path)
        save_path = os.path.join(output_path, file_name + '_cropped.mp4')
        command = str('ffmpeg -i ') + '"' + file_path + '"' + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(top_lext_x) + ':' + str(top_left_y) + '" ' + str('-c:v libx264 -crf 21 -c:a copy ') + '"' + str(save_path) + '"'
        subprocess.call(command, shell=True)
        print('Video {} cropped (Video {}/{})'.format(file_name, str(file_cnt+1), str(len(video_paths))))
    print('SIMBA COMPLETE: {} videos cropped and saved in {} directory '.format(str(len(video_paths)), directory_path))



def frames_to_movie(directory: str,
                    fps: int,
                    bitrate: int,
                    img_format: str):

    """
    Helper to merge frames in a folder to a mp4 video file. Video file is stored in the same directory as the
    input directory sub-folder.

    Parameters
    ----------
    directory_path: str
        Path to directory holding images.
    fps: int
        Frame rate of output video
    bitrate:
        Bitrate of output video
    img_format:
        Type of images inside the input directory path.
    """


    if not os.path.isdir(directory):
        print('SIMBA ERROR: {} is not a valid directory'.format(directory))
        raise NotADirectoryError()
    check_int(name='FPS', value=fps)
    check_int(name='BITRATE', value=bitrate)
    file_paths_in_folder = [f for f in glob.glob(directory + '/*') if os.path.isfile(f)]
    img_paths_in_folder = [x for x in file_paths_in_folder if Path(x).suffix[1:] == img_format]
    if len(img_paths_in_folder) < 1:
        print('SIMBA ERROR: Zero images of file-type {} found in {} directory'.format(img_format, directory))
        raise ValueError()
    img = cv2.imread(img_paths_in_folder[0])
    img_h, img_w = int(img.shape[0]), int(img.shape[1])
    ffmpeg_fn = os.path.join(directory, '%d.{}'.format(img_format))
    save_path = os.path.join(os.path.dirname(directory), os.path.basename(directory) + '.mp4')
    command = str('ffmpeg -y -r ' + str(fps) + ' -f image2 -s ' + str(img_h) + 'x' + str(img_w) + ' -i ' + '"' + ffmpeg_fn + '"' + ' -vcodec libx264 -b ' + str(bitrate) + 'k ' + '"' + str(save_path) + '"')
    print('Creating {} from {} images...'.format(os.path.basename(save_path), str(len(img_paths_in_folder))))
    subprocess.call(command, shell=True)
    print('SIMBA COMPLETE: Video created at {}'.format(save_path))


def video_concatenator(video_one_path: str,
                       video_two_path: str,
                       resolution: int or str,
                       horizontal: bool):
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
    print('SIMBA COMPLETE! Videos concatenated and saved at {}.'.format(save_path))

# video_concatenator(video_one_path='/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/frames/output/gantt_plots/SI_DAY3_308_CD1_PRESENT_2.mp4',
#                    video_two_path= '/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/frames/output/gantt_plots/SI_DAY3_308_CD1_PRESENT.mp4',
#                    resolution='Video 1',
#                    horizontal=False)



