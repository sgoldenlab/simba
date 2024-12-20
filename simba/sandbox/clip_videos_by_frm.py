import os
from typing import List, Union, Tuple, Optional
import subprocess
from simba.utils.read_write import get_video_meta_data, get_fn_ext
from simba.utils.checks import check_valid_lst, check_int, check_if_dir_exists
from simba.utils.errors import FrameRangeError
from simba.utils.printing import stdout_success, SimbaTimer


def clip_videos_by_frame_ids(file_paths: List[Union[str, os.PathLike]],
                             frm_ids: List[List[int]],
                             save_dir: Optional[Union[str, os.PathLike]] = None):

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
    check_valid_lst(data=file_paths, source=clip_videos_by_frame_ids.__name__, valid_dtypes=(str,), min_len=1)
    check_valid_lst(data=frm_ids, source=clip_videos_by_frame_ids.__name__, valid_dtypes=(list,), exact_len=len(file_paths))
    for cnt, i in enumerate(frm_ids):
        check_valid_lst(data=i, source=f'clip_videos_by_frame_count.__name_ frm_ids {cnt}', valid_dtypes=(int,), exact_len=2)
        if i[0] >= i[1]: raise FrameRangeError(msg=f'Start frame for video {i} is after or the same as the end frame ({i[0]}, {i[1]})', source=clip_videos_by_frame_ids.__name__)
        if (i[0] < 0) or (i[1] < 1): raise FrameRangeError(msg=f'Start frame has to be at least 0 and end frame has to be at least 1', source=clip_videos_by_frame_ids.__name__)
    video_meta_data = [get_video_meta_data(video_path=x) for x in file_paths]
    for i in range(len(video_meta_data)):
        if (frm_ids[i][0] > video_meta_data[i]['frame_count']) or (frm_ids[i][1] > video_meta_data[i]['frame_count']):
            raise FrameRangeError(msg=f'Video {i+1} has {video_meta_data[i]["frame_count"]} frames, cannot use start and end frame {frm_ids[i]}', source=clip_videos_by_frame_ids.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=clip_videos_by_frame_ids.__name__, create_if_not_exist=True)
    for cnt, file_path in enumerate(file_paths):
        video_timer = SimbaTimer(start=True)
        dir, video_name, ext = get_fn_ext(filepath=file_path)
        s_f, e_f = frm_ids[cnt][0], frm_ids[cnt][1]
        print(f'Trimming {video_name} from frame {s_f} to frame {e_f}...')
        if save_dir is not None:
            out_path = os.path.join(save_dir, os.path.basename(file_path))
        else:
            out_path = os.path.join(dir, f'{video_name}_{s_f}_{e_f}.{ext}')
        cmd = f'ffmpeg -i "{file_path}" -vf "trim={s_f}:{e_f},setpts=PTS-STARTPTS" -c:v libx264 -c:a aac -loglevel error -stats "{out_path}" -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(f'Video {video_name} complete (elapsed time {video_timer.elapsed_time_str}s)')
    timer.stop_timer()
    if save_dir is None:
        stdout_success(msg=f'{len(file_paths)} video(s) clipped by frame', elapsed_time=timer.elapsed_time_str)
    else:
        stdout_success(msg=f'{len(file_paths)} video(s) clipped by frame and saved in {save_dir}', elapsed_time=timer.elapsed_time_str)


# file_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/Trial    10.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/Trial    10_1.mp4']
# frm_ids = [[0, 20], [20, 40]]
# clip_videos_by_frame_ids(file_paths=file_paths, frm_ids=frm_ids, save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/frames/output/path_plots/trial_cnt')
