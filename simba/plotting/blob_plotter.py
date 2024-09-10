__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import os
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import functools
import multiprocessing

from numba import jit

from simba.data_processors.blob_location_computer import BlobLocationComputer
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_nvidea_gpu_available,
                                check_str, check_valid_boolean)
from simba.utils.enums import Defaults, Options, TextOptions
from simba.utils.errors import FFMPEGCodecGPUError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_all_videos_in_directory,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data)


class BlobPlotter(PlottingMixin):
    """
     Plot the results of animal tracking based on blob.

     .. seealso::
        :func:`simba.mixins.plotting_mixin.PlottingMixin._plot_blobs`, :func:`simba.mixins.image_mixin.ImageMixin.get_blob_locations`

     .. video:: _static/img/BlobLocationComputer.webm
        :width: 800
        :autoplay:
        :loop:

     :param Union[List[str], str, os.PathLike] data_path: Path(s) to video file(s) or directory containing video files.
     :param Optional[bool] gpu: Whether to use GPU for processing. Defaults to False.
     :param Optional[int] batch_size: Number of frames to process in each batch. Defaults to 2000. Increase if your RAM allows.
     :param Optional[Tuple[int, int, int]] circle_color: Color of the blobs as an RGB tuple. Defaults to pink (255, 105, 180).
     :param Optional[Union[str, os.PathLike]] save_dir: Directory to save output files. If None, no files will be saved.
     :param Optional[int] verbose: If True, then prints msgs informing on progress.
     :param Optional[str] smoothing: Savitzky Golay, Gaussian, or None. Smooths body-part coordinate data for more accurate blob representation. Default None.
     :param Optional[int] circle_size: The circle defining the x, y location of the animal in the data. Defaults to None and SimBA will try and retrieve the optimal circle size based in the video resolution.
     :param Optional[int] core_cnt: The number of cores to use for multiprocessing. Deafults to -1 which means all available cores.

     :example:
     >>> BlobPlotter(data_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\test\2022-06-20_NOB_DOT_4_downsampled.mp4", smoothing='Savitzky Golay', circle_size=10).run()
     """

    def __init__(self,
                 data_path: Union[List[str], str, os.PathLike],
                 gpu: Optional[bool] = False,
                 batch_size: Optional[int] = 2000,
                 circle_color: Optional[Tuple[int, int, int]] = (TextOptions.COLOR.value),
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 verbose: Optional[int] = True,
                 smoothing: Optional[str] = None,
                 circle_size: Optional[int] = None,
                 core_cnt: Optional[int] = -1):



        PlottingMixin.__init__(self)
        if os.path.isdir(data_path):
            self.data_paths = find_all_videos_in_directory(directory=data_path, as_dict=True, raise_error=True).values()
        elif os.path.isfile(data_path):
            self.data_paths = [data_path]
        else:
            raise InvalidInputError(msg=f'{data_path} is not a valid directory or video file path or directory path.')
        self.video_meta_data = {}
        for i in self.data_paths:
            self.video_meta_data[get_fn_ext(filepath=i)[1]] = get_video_meta_data(video_path=i)
        if save_dir is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=self.__class__.__name__)
        check_valid_boolean(value=[verbose, gpu])
        if gpu and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        if smoothing is not None:
            check_str(name=f'{self.__class__.__name__} smoothing', value=smoothing, options=Options.SMOOTHING_OPTIONS.value)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1)
        if (core_cnt == -1) or core_cnt > find_core_cnt()[0]:
            core_cnt = find_core_cnt()[0]
        check_if_valid_rgb_tuple(data=circle_color)
        self.verbose = verbose
        self.gpu = gpu
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.smoothing = smoothing
        self.core_cnt = core_cnt
        self.circle_size = circle_size
        self.circle_color = circle_color

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array, group: int):
        group_col = np.full((data.shape[0], 1), group)
        return np.hstack((group_col, data))

    def run(self):
        timer = SimbaTimer(start=True)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            temp_dir = os.path.join(os.path.dirname(file_path), video_name, '_temp')
            check_if_dir_exists(in_dir=temp_dir, source=self.__class__.__name__, create_if_not_exist=True)
            if self.save_dir is None:
                save_path = os.path.join(os.path.dirname(file_path), video_name + '_blob_tracking.mp4')
            else:
                save_path = os.path.join(self.save_dir, video_name + '_blob_tracking.mp4')
            blob_locator = BlobLocationComputer(data_path=file_path, verbose=self.verbose, gpu=self.gpu, batch_size=self.batch_size, save_dir=None, smoothing=self.smoothing, multiprocessing=True)
            blob_locator.run()
            print(f'Creating blob location video for {video_name}...')
            if not os.path.isdir(temp_dir): os.makedirs(temp_dir)
            x_y = blob_locator.location_data[video_name]
            del blob_locator
            x_y = np.hstack([np.arange(0, x_y.shape[0]).reshape(-1, 1), x_y])
            x_y = np.array_split(x_y, self.core_cnt)
            bps = [self.__insert_group_idx_column(data=i, group=cnt) for cnt, i in enumerate(x_y)]
            if self.circle_size is None:
                self.circle_size = self.get_optimal_circle_size(frame_size=(int(self.video_meta_data[video_name]['width']), int(self.video_meta_data[video_name]['height'])), circle_frame_ratio=100)
            with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial(PlottingMixin._plot_blobs, verbose=self.verbose, circle_size=self.circle_size, circle_clr=self.circle_color, video_path=file_path, temp_dir=temp_dir)
                for cnt, result in enumerate(pool.imap(constants, bps, chunksize=1)):
                    print(f'Completed section {result+1}...')
            concatenate_videos_in_folder(in_folder=temp_dir, save_path=save_path, video_format="avi", remove_splits=True)
            video_timer.stop_timer()
            stdout_success(msg=f'Blob video {video_name} complete and saved at {save_path}', elapsed_time=video_timer.elapsed_time_str)
        timer.stop_timer()
        stdout_success(msg=f'{len(self.data_paths)} blob videos saved.', elapsed_time=video_timer.elapsed_time_str)


#BlobPlotter(data_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\test\2022-06-20_NOB_DOT_4_downsampled.mp4", smoothing='Savitzky Golay', circle_size=30, gpu=True).run()