import functools
import multiprocessing
import os
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.data_processors.light_dark_box_analyzer import LightDarkBoxAnalyzer
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_float, check_if_dir_exists, check_str,
                                check_video_and_data_frm_count_align)
from simba.utils.enums import Defaults, Formats, Options
from simba.utils.errors import NoDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data)


def _light_dark_box_visualizer(pose_data: pd.DataFrame,
                               light_data: pd.DataFrame,
                               video_temp_dir: str,
                               video_path: str,
                               circle_size: Union[int, float],
                               txt_shift: Tuple[int, int],
                               font_size: Union[int, float]):


    group, idx = int(pose_data[0]), pose_data[1].index.tolist()
    start_frm, end_frm = idx[0], idx[-1]
    video_meta_data = get_video_meta_data(video_path=video_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    video_save_path = os.path.join(video_temp_dir, f"{group}.mp4")
    video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    cap = cv2.VideoCapture(video_path)
    light_data = light_data[light_data['EVENT'] == 'LIGHT BOX'].reset_index(drop=True)
    light_data['FRAME LIST'] = light_data.apply(lambda row: list(range(row['START FRAME'], row['END FRAME'] + 1)), axis=1)
    light_box_frm_lst = [f for s in light_data['FRAME LIST'] for f in s]
    for frm_cnt, frm_id in enumerate(range(start_frm, end_frm + 1)):
        cap.set(1, int(frm_id))
        ret, img = cap.read()
        x, y, p = pose_data[1].loc[frm_id].values.astype(np.int32)
        if frm_id in light_box_frm_lst:
            img = cv2.circle(img, (int(x), int(y)), circle_size, (0, 0, 255), -1)
            img = PlottingMixin().put_text(img=img, text='ZONE: LIGHT BOX', pos=txt_shift, font_size=font_size, text_color=(0, 255, 255), text_bg_alpha=0.8)
        else:
            img = PlottingMixin().put_text(img=img, text='ZONE: DARK BOX', pos=txt_shift, font_size=font_size, text_color=(255, 255, 255), text_bg_alpha=0.8)
        video_writer.write(img.astype(np.uint8))
        print(f"Multi-processing video frame {frm_id} on core {group}...")

    cap.release()
    video_writer.release()
    return group



class LightDarkBoxPlotter():

    """
    Generate annotated videos visualizing behavior episodes in a light/dark box setup.

    .. seealso::
       For light/dark box data analysis, see :func:`simba.data_processors.light_dark_box_analyzer.LightDarkBoxAnalyzer`.


    .. video:: _static/img/LightDarkBoxPlotter.webm
       :width: 800
       :autoplay:
       :loop:

    :param (str or os.PathLike) data_dir: Directory containing pose estimation CSV files.
    :param (str or os.PathLike) video_dir: Directory containing video files corresponding to CSVs.
    :param (str or os.PathLike) save_dir: Output directory to save the resulting annotated videos.
    :param (str) body_part: The name of the body part used to determine position and behavior.
    :param (int or float) fps : Frames per second of the videos (used for timing episodes).
    :param (float) threshold: Threshold value for light/dark classification (between 0.0 and 1.0).
    :param (float) minimum_episode_duration : Minimum duration (in seconds) for an episode to be considered valid.
    :param (int) core_cnt: Number of CPU cores to use for parallel processing. If -1, uses all available cores.
    """

    def __init__(self,
                 video_dir: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 body_part: str,
                 fps: Union[int, float],
                 threshold: float = 0.01,
                 minimum_episode_duration: float = 10e-16,
                 core_cnt: int = -1):

        check_if_dir_exists(in_dir=data_dir, source=f'{self.__class__.__name__} data_dir', raise_error=True)
        check_if_dir_exists(in_dir=video_dir, source=f'{self.__class__.__name__} video_dir', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], raise_error=True, as_dict=True)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True, as_dict=True)
        missing_videos = [x for x in self.data_paths.keys() if x not in self.video_paths.keys()]
        if len(missing_videos) > 0:
            raise NoDataError(msg=f'Video(s) {missing_videos} are missing from the video directory {video_dir}', source=self.__class__.__name__)
        self.data_dir = data_dir
        check_str(name=f'{self.__class__.__name__} body_part', value=body_part)
        check_float(name=f'{self.__class__.__name__} fps', value=fps, min_value=10e-6)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} minimum_duration', value=minimum_episode_duration, min_value=0.0)
        self.body_part, self.fps, self.threshold, self.min_dur, self.core_cnt = body_part, fps, threshold, minimum_episode_duration, core_cnt
        self.save_dir = save_dir
        self.timer = SimbaTimer(start=True)

    def run(self):
        analyzer = LightDarkBoxAnalyzer(data_dir=self.data_dir, save_path=None, body_part=self.body_part, fps=self.fps, threshold=self.threshold, minimum_episode_duration=self.min_dur)
        for file_cnt, (file_name, file_path) in enumerate(self.data_paths.items()):
            video_timer = SimbaTimer(start=True)
            analyzer.data_paths = {file_name: file_path}
            analyzer.run()
            behavior_data = analyzer.results[analyzer.results['VIDEO'] == file_name].reset_index(drop=True)
            pose_data = analyzer.data_df[analyzer.bp_cols]
            check_video_and_data_frm_count_align(video=self.video_paths[file_name], data=pose_data, name=file_name, raise_error=True)
            pose_data = np.array_split(pose_data, self.core_cnt)
            pose_data = [(c, x) for c, x in enumerate(pose_data)]
            self.temp_dir = os.path.join(self.save_dir, file_name, "temp")
            self.video_save_path = os.path.join(self.save_dir, f'{file_name}.mp4')
            if not os.path.isdir(self.temp_dir): os.makedirs(self.temp_dir)
            video_meta = get_video_meta_data(video_path=self.video_paths[file_name])
            circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(video_meta['width'], video_meta['height']), circle_frame_ratio=50)
            font_size, x_shift, y_shift = PlottingMixin().get_optimal_font_scales(text='LONG ARSE STRING', accepted_px_width=int(video_meta['width']/3), accepted_px_height=int(video_meta['height']/3))
            with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial(_light_dark_box_visualizer,
                                              light_data=behavior_data,
                                              video_path=self.video_paths[file_name],
                                              video_temp_dir=self.temp_dir,
                                              circle_size=circle_size,
                                              txt_shift=(x_shift, y_shift),
                                              font_size=font_size)
                for cnt, result in enumerate(pool.imap(constants, pose_data, chunksize=1)):
                    print(f"Section {result}/{len(pose_data)} complete...")
            pool.terminate()
            pool.join()
            concatenate_videos_in_folder(in_folder=self.temp_dir, save_path=self.video_save_path, gpu=False)
            video_timer.stop_timer()
            print(f"Video {self.video_save_path} complete (elapsed time: {video_timer.elapsed_time_str}s)...")

        self.timer.stop_timer()
        stdout_success(msg=f"{len(self.video_paths)} videos saved in {self.save_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


if __name__ == '__main__':
    d = LightDarkBoxPlotter(data_dir=r'D:\light_dark_box\project_folder\csv\input_csv\test',
                            video_dir=r'D:\light_dark_box\project_folder\videos',
                            fps=30,
                            body_part='center',
                            minimum_episode_duration=1,
                            core_cnt=23,
                            save_dir=r"D:\light_dark_box\project_folder\videos\light_dark",
                            threshold=0.2)
    d.run()









