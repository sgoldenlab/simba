import os
from typing import List, Union, Optional
import subprocess
from datetime import datetime
import shutil

from simba.utils.read_write import get_video_meta_data
from simba.utils.checks import check_valid_lst, check_if_dir_exists, check_int, check_ffmpeg_available, check_nvidea_gpu_available
from simba.utils.errors import InvalidInputError, FFMPEGCodecGPUError
from simba.utils.printing import SimbaTimer
from simba.video_processors.video_processing import create_blank_video

def horizontal_video_concatenator(video_paths: List[Union[str, os.PathLike]],
                                  save_path: Union[str, os.PathLike],
                                  height_px: Optional[Union[int, str]] = None,
                                  height_idx: Optional[Union[int, str]] = None,
                                  gpu: Optional[bool] = False,
                                  verbose: Optional[bool] = True) -> None:

    """
    Concatenates multiple videos horizontally.

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
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None)", source=horizontal_video_concatenator.__name__)
    timer = SimbaTimer(start=True)
    check_valid_lst(data=video_paths, source=horizontal_video_concatenator.__name__, min_len=2)
    check_if_dir_exists(in_dir=os.path.dirname(save_path), source=horizontal_video_concatenator.__name__)
    video_meta_data = [get_video_meta_data(video_path=video_path) for video_path in video_paths]
    if ((height_px is None) and (height_idx is None)) or ((height_px is not None) and (height_idx is not None)):
        raise InvalidInputError(msg='Provide a height_px OR height_idx', source=horizontal_video_concatenator.__name__)
    if height_idx is not None:
        check_int(name=f'{horizontal_video_concatenator.__name__} height', value=height_idx, min_value=0, max_value=len(video_paths)-1)
        height = int(video_meta_data[height_idx]['height'])
    else:
        check_int(name=f'{horizontal_video_concatenator.__name__} height', value=height_px, min_value=1)
        height = int(height_px)
    video_path_str = " ".join([f'-i "{path}"' for path in video_paths])
    codec = 'h264_nvenc' if gpu else 'libvpx-vp9'
    filter_complex = ";".join([f"[{idx}:v]scale=-1:{height}[v{idx}]" for idx in range(len(video_paths))])
    filter_complex += f";{''.join([f'[v{idx}]' for idx in range(len(video_paths))])}hstack=inputs={len(video_paths)}[v]"
    if verbose:
        print(f'Concatenating {len(video_paths)} videos horizontally with a {height} pixel height... ')
    cmd = f'ffmpeg {video_path_str} -filter_complex "{filter_complex}" -map "[v]" -c:v {codec} -loglevel error -stats "{save_path}" -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    if verbose:
        print(f'Horizontal concatenation complete, saved at {save_path} (elapsed time: {timer.elapsed_time_str}s.)')


def vertical_video_concatenator(video_paths: List[Union[str, os.PathLike]],
                                save_path: Union[str, os.PathLike],
                                width_px: Optional[int] = None,
                                width_idx: Optional[int] = None,
                                gpu: Optional[bool] = False,
                                verbose: Optional[bool] = True) -> None:
    """
    Concatenates multiple videos vertically.

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
    if gpu and not check_nvidea_gpu_available(): raise FFMPEGCodecGPUError(msg="NVIDIA GPU not available", source=vertical_video_concatenator.__name__)
    video_meta_data = [get_video_meta_data(video_path=video_path) for video_path in video_paths]
    timer = SimbaTimer(start=True)
    check_valid_lst(data=video_paths, source=vertical_video_concatenator.__name__, min_len=2)
    check_if_dir_exists(in_dir=os.path.dirname(save_path), source=vertical_video_concatenator.__name__)
    if ((width_px is None) and (width_idx is None)) or ((width_px is not None) and (width_idx is not None)):
        raise InvalidInputError(msg='Provide a width_px OR width_idx', source=vertical_video_concatenator.__name__)
    if width_idx is not None:
        check_int(name=f'{vertical_video_concatenator.__name__} width index', value=width_idx, min_value=0, max_value=len(video_paths) - 1)
        width = int(video_meta_data[width_idx]['width'])
    else:
        check_int(name=f'{vertical_video_concatenator.__name__} width', value=width_px, min_value=1)
        width = int(width_px)
    video_path_str = " ".join([f'-i "{path}"' for path in video_paths])
    codec = 'h264_nvenc' if gpu else 'libvpx-vp9'
    filter_complex = ";".join([f"[{idx}:v]scale={width}:-1[v{idx}]" for idx in range(len(video_paths))])
    filter_complex += f";{''.join([f'[v{idx}]' for idx in range(len(video_paths))])}"
    filter_complex += f"vstack=inputs={len(video_paths)}[v]"
    if verbose:
        print(f'Concatenating {len(video_paths)} videos vertically with a {width} pixel width...')
    cmd = f'ffmpeg {video_path_str} -filter_complex "{filter_complex}" -map "[v]" -c:v {codec} -loglevel error -stats "{save_path}" -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    if verbose:
        print(f'Vertical concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)')

def mosaic_concatenator(video_paths: List[Union[str, os.PathLike]],
                        save_path: Union[str, os.PathLike],
                        width_idx: Optional[Union[int, str]] = None,
                        width_px: Optional[Union[int, str]] = None,
                        height_idx: Optional[Union[int, str]] = None,
                        height_px: Optional[Union[int, str]] = None,
                        gpu: Optional[bool] = False,
                        verbose: Optional[bool] = True,
                        uneven_fill_color: Optional[str] = 'black') -> None:
    """
    Concatenates multiple videos into a mosaic layout.

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
    if gpu and not check_nvidea_gpu_available(): raise FFMPEGCodecGPUError(msg="NVIDIA GPU not available", source=mosaic_concatenator.__name__)
    timer = SimbaTimer(start=True)
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    check_valid_lst(data=video_paths, source=f'{mosaic_concatenator.__name__} video_paths', min_len=3)
    video_meta_data = [get_video_meta_data(video_path=video_path) for video_path in video_paths]
    max_video_length = max([x['video_length_s'] for x in video_meta_data])
    if ((width_px is None) and (width_idx is None)) or ((width_px is not None) and (width_idx is not None)):
        raise InvalidInputError(msg='Provide a width_px OR width_idx', source=mosaic_concatenator.__name__)
    if ((height_px is None) and (height_idx is None)) or ((height_px is not None) and (height_idx is not None)):
        raise InvalidInputError(msg='Provide a height_px OR height_idx', source=mosaic_concatenator.__name__)
    if width_idx is not None:
        check_int(name=f'{vertical_video_concatenator.__name__} width index', value=width_idx, min_value=1, max_value=len(video_paths) - 1)
        width = int(video_meta_data[width_idx]['width'])
    else:
        width = width_px
    if height_idx is not None:
        check_int(name=f'{vertical_video_concatenator.__name__} height index', value=width_idx, min_value=1, max_value=len(video_paths) - 1)
        height = int(video_meta_data[width_idx]['height'])
    else:
        height = height_px
    if verbose:
        print(f'Creating mosaic video ...')
    temp_dir = os.path.join(os.path.dirname(video_paths[0]), f'temp_{dt}')
    os.makedirs(temp_dir)
    if not (len(video_paths) % 2) == 0:
        blank_path = os.path.join(temp_dir, f'{dt}.mp4')
        create_blank_video(path=blank_path, length=max_video_length, width=width, height=height, gpu=gpu, verbose=verbose, color=uneven_fill_color)
        video_paths.append(blank_path)
    upper_videos, lower_videos = video_paths[:len(video_paths)//2], video_paths[len(video_paths)//2:]
    if verbose: print('Creating upper mosaic... (Step 1/3)')
    if len(upper_videos) > 1:
        upper_path = os.path.join(temp_dir, 'upper.mp4')
        horizontal_video_concatenator(video_paths=upper_videos, save_path=upper_path, gpu=gpu, height_px=height, verbose=verbose)
    else:
        upper_path = upper_videos[0]
    if verbose: print('Creating lower mosaic... (Step 2/3)')
    if len(lower_videos) > 1:
        lower_path = os.path.join(temp_dir, 'lower.mp4')
        horizontal_video_concatenator(video_paths=lower_videos, save_path=lower_path, gpu=gpu, height_px=height, verbose=verbose)
    else:
        lower_path = lower_videos[0]
    panels_meta = [get_video_meta_data(video_path=video_path) for video_path in [lower_path, upper_path]]
    if verbose: print('Joining upper and lower mosaic... (Step 2/3)')
    vertical_video_concatenator(video_paths=[upper_path, lower_path], save_path=save_path, verbose=verbose, gpu=gpu, width_px=max([x['width'] for x in panels_meta]))
    timer.stop_timer()
    shutil.rmtree(temp_dir)
    if verbose:
        print(f'Mosaic concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)')

def mixed_mosaic_concatenator(video_paths: List[Union[str, os.PathLike]],
                              save_path: Union[str, os.PathLike],
                              gpu: Optional[bool] = False,
                              verbose: Optional[bool] = True,
                              uneven_fill_color: Optional[str] = 'black') -> None:
    """
    Create a mixed mosaic video by concatenating multiple input videos in a mosaic layout of various sizes.

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
    if gpu and not check_nvidea_gpu_available(): raise FFMPEGCodecGPUError(msg="NVIDIA GPU not available", source=mixed_mosaic_concatenator.__name__)
    timer = SimbaTimer(start=True)
    check_valid_lst(data=video_paths, source=mixed_mosaic_concatenator.__name__, min_len=2)
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    video_meta_data = [get_video_meta_data(video_path=video_path) for video_path in video_paths]
    max_video_length = max([x['video_length_s'] for x in video_meta_data])
    check_if_dir_exists(in_dir=os.path.dirname(save_path), source=mixed_mosaic_concatenator.__name__)
    large_mosaic_path, video_paths = video_paths[0], video_paths[1:]
    mosaic_height = int(video_meta_data[0]['height'] / 2)
    if verbose: print('Creating mixed mosaic video... ')
    temp_dir = os.path.join(os.path.dirname(video_paths[0]), f'temp_{dt}')
    os.makedirs(temp_dir)
    if not (len(video_paths) % 2) == 0:
        blank_path = os.path.join(temp_dir, f'{dt}.mp4')
        create_blank_video(path=blank_path, length=max_video_length, width=video_meta_data[-1]['width'], height=mosaic_height, gpu=gpu, verbose=True, color=uneven_fill_color)
        video_paths.append(blank_path)
    upper_videos, lower_videos = video_paths[:len(video_paths) // 2], video_paths[len(video_paths) // 2:]
    if verbose: print('Creating upper right mosaic ... (Step 1/4)')
    if len(upper_videos) > 1:
        upper_path = os.path.join(temp_dir, 'upper.mp4')
        horizontal_video_concatenator(video_paths=upper_videos, save_path=upper_path, gpu=gpu, height_px=mosaic_height, verbose=verbose)
    else:
        upper_path = upper_videos[0]
    if verbose: print('Creating lower right mosaic ... (Step 2/4)')
    if len(lower_videos) > 1:
        lower_path = os.path.join(temp_dir, 'lower.mp4')
        horizontal_video_concatenator(video_paths=lower_videos, save_path=lower_path, gpu=gpu, verbose=verbose)
    else:
        lower_path = lower_videos[0]
    panels_meta = [get_video_meta_data(video_path=video_path) for video_path in [lower_path, upper_path]]
    mosaic_path = os.path.join(temp_dir, 'mosaic.mp4')
    if verbose: print('Joining upper and lower right mosaic... (Step 3/4)')
    vertical_video_concatenator(video_paths=[upper_path, lower_path], width_px=min([x['width'] for x in panels_meta]), save_path=mosaic_path, gpu=gpu, verbose=verbose)
    if verbose: print('Joining left and right mosaic... (Step 4/4)')
    horizontal_video_concatenator(video_paths=[large_mosaic_path, mosaic_path], height_idx=0, save_path=save_path, gpu=gpu)
    timer.stop_timer()
    shutil.rmtree(temp_dir)
    if verbose:
        print(f'Mixed mosaic concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)')

video_paths = [
    '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4',
    '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4',
    '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/2022-06-21_NOB_IOT_23.mp4']
save_path = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4'
mixed_mosaic_concatenator(video_paths=video_paths, save_path=save_path, gpu=False, verbose=True)



    #
    # mosaic_right_path = os.path.join(save_dir, 'mosaic.mp4')
    #
    #
    # shutil.rmtree(save_dir)
    # timer.stop_timer()
    # if verbose:
    #     print(f'Mixed mosaic video complete. Elapsed time: {timer.elapsed_time_str}s.')




    #
    # filter_complex = ""
    # for i in range(len(video_paths)):
    #     filter_complex += f"[{i}:v]scale={width}:{height}[scaled_{i}];"
    # filter_complex += f"tile={col_cnt}x2:padding=10:width=2*in_w:height=2*in_h"
    # ffmpeg_command = (
    #         f"ffmpeg "
    #         + " ".join([f'-i "{input_file}"' for input_file in video_paths])
    #         + f" -filter_complex \"{filter_complex}\" "
    #         + f"-c:v libx264 -crf 18 -preset veryfast {save_path}"
    # )
    # subprocess.run(ffmpeg_command, shell=True)

# video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/2022-06-21_NOB_IOT_23.mp4']
# save_path = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4'
# mosaic_concatenator(video_paths=video_paths, save_path=save_path, width_idx=1, height_idx=1, gpu=False)
# # #     #
# import subprocess
#
# def create_mosaic(input_files, output_file, panel_width, panel_height, num_cols):
#     num_videos = len(input_files)
#     num_rows = (num_videos + num_cols - 1) // num_cols
#
#     # Generate filter_complex string with scaling and tile
#     filter_complex = ""
#     for i in range(num_videos):
#         filter_complex += f"[{i}:v]scale={panel_width}:{panel_height}[scaled_{i}];"
#     filter_complex += "".join([f"[scaled_{i}]" for i in range(num_videos)])
#     filter_complex += f"tile={num_cols}x{num_rows}:padding=10"
#
#     # Generate FFmpeg command
#     ffmpeg_command = (
#             f"ffmpeg "
#             + " ".join([f'-i "{input_file}"' for input_file in input_files])
#             + f" -filter_complex \"{filter_complex}\" "
#             + f"-c:v libx264 -crf 18 -preset veryfast {output_file}"
#     )
#
#     # Run FFmpeg command
#     subprocess.run(ffmpeg_command, shell=True)
#
#
# # Example usage:
# input_files = ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/2022-06-21_NOB_IOT_23.mp4']
# output_file = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/blank_test.mp4'
# panel_width = 100  # Example width for each panel
# panel_height = 100  # Example height for each panel
# num_cols = 2  # Example number of columns
# create_mosaic(input_files, output_file, panel_width, panel_height, num_cols)


  #
    #
    #
    # video_paths = resize_videos_by_width_and_height(video_paths=video_paths, height=height, width=width, overwrite=False, save_dir=save_dir)
    #
    #     video_lengths = []
    #     for i in video_paths: video_lengths.append(get_video_meta_data(video_path=i)['video_length_s'])
    #
    #
    # if len(upper_videos) > 1:
    #     upper_path = os.path.join(save_dir, 'upper.mp4')
    #     horizontal_video_concatenator(video_paths=upper_videos, save_path=upper_path, gpu=gpu)
    # else:
    #     upper_path = upper_videos[0]
    # if len(lower_videos) > 1:
    #     lower_path = os.path.join(save_dir, 'lower.mp4')
    #     horizontal_video_concatenator(video_paths=lower_videos, save_path=lower_path, gpu=gpu)
    # else:
    #     lower_path = lower_videos[0]
    # vertical_video_concatenator(video_paths=[upper_path, lower_path], save_path=save_path, verbose=verbose, gpu=gpu)
    # shutil.rmtree(save_dir)
    # timer.stop_timer()
    # if verbose:
    #     print(f'Mosaic video complete. Elapsed time: {timer.elapsed_time_str}s.')
