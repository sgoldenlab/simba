__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import os
from copy import deepcopy
from typing import Optional, Union

import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd

from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.enums import Formats, Methods, Options
from simba.utils.errors import FFMPEGCodecGPUError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    get_video_meta_data, remove_files,
                                    write_df)
from simba.video_processors.video_processing import (video_bg_subtraction,
                                                     video_bg_subtraction_mp)


class BlobLocationComputer(object):

    """
    Detecting and saving blob locations from video files.

    .. video:: _static/img/BlobLocationComputer.webm
       :width: 800
       :autoplay:
       :loop:

    :param Union[str, os.PathLike] data_path: Path to a video file or a directory containing video files. The videos will be processed for blob detection.
    :param Optional[bool] verbose:  If True, prints progress and success messages to the console. Default is True.
    :param Optional[bool] gpu: If True, GPU acceleration will be used for blob detection. Default is True.
    :param Optional[int] batch_size: The number of frames to process in each batch for blob detection. Default is 2500.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where the blob location data will be saved as CSV files. If None, the results will not be saved. Default is None.
    :param Optional[bool] multiprocessing: If True, video background subtraction will be done using  multiprocessing. Default is False.

    :example:
    >>> x = BlobLocationComputer(data_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_downsampled_bg_subtracted.mp4", multiprocessing=True, gpu=True, batch_size=2000, save_dir=r"C:\blob_positions")
    >>> x.run()
    """
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 verbose: Optional[bool] = True,
                 gpu: Optional[bool] = True,
                 batch_size: Optional[int] = 2500,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 smoothing: Optional[str] = None,
                 multiprocessing: Optional[bool] = False):


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
        check_valid_boolean(value=[verbose, gpu, multiprocessing])
        if gpu and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        if smoothing is not None:
            check_str(name=f'{self.__class__.__name__} smoothing', value=smoothing, options=Options.SMOOTHING_OPTIONS.value)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.gpu = gpu
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.smoothing = smoothing

    def run(self):
        timer = SimbaTimer(start=True)
        self.location_data = {}
        for file_cnt, video_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, ext = get_fn_ext(filepath=video_path)
            temp_video_path = os.path.join(os.path.dirname(video_path), video_name + '_temp.mp4')
            if not self.multiprocessing:
                _ = video_bg_subtraction(video_path=video_path, verbose=self.verbose, bg_color=(0, 0, 0), fg_color=(255, 255, 255), save_path=temp_video_path)
            else:
                _ = video_bg_subtraction_mp(video_path=video_path, verbose=self.verbose, bg_color=(0, 0, 0), fg_color=(255, 255, 255), save_path=temp_video_path)
            self.location_data[video_name] = ImageMixin.get_blob_locations(video_path=temp_video_path, gpu=self.gpu, verbose=self.verbose, batch_size=self.batch_size).astype(np.int32)
            remove_files(file_paths=[temp_video_path])
            video_timer.stop_timer()
            print(f'Blob detection for video {video_name} ({file_cnt+1}/{len(self.data_paths)}) complete (elapsed time: {video_timer.elapsed_time_str}s)...')
        timer.stop_timer()
        if self.smoothing is not None:
            print('Smoothing data...')
            smoothened_data = {}
            if self.smoothing == Methods.SAVITZKY_GOLAY.value:
                for video_name, video_data in self.location_data.items():
                    smoothened_data[video_name] = savgol_smoother(data=video_data, fps=self.video_meta_data[video_name]['fps'], time_window=2000, source=video_name)
            if self.smoothing == Methods.GAUSSIAN.value:
                for video_name, video_data in self.location_data.items():
                    smoothened_data[video_name] = df_smoother(data=pd.DataFrame(video_data, columns=['X', 'Y']), fps=self.video_meta_data[video_name]['fps'], time_window=2000, source=video_name, method='gaussian')
            self.location_data = deepcopy(smoothened_data)
            del smoothened_data
        if self.save_dir is not None:
            for video_name, video_data in self.location_data.items():
                save_path = os.path.join(self.save_dir, f'{video_name}.csv')
                df = pd.DataFrame(video_data, columns=['X', 'Y'])
                write_df(df=df, file_type=Formats.CSV.value, save_path=save_path)
            if self.verbose:
                stdout_success(f'Video blob detection complete for {len(self.data_paths)} videos, data saved at {self.save_dir}', elapsed_time=timer.elapsed_time_str)
        else:
            if self.verbose:
                stdout_success(f'Video blob detection complete for {len(self.data_paths)} video', elapsed_time=timer.elapsed_time_str)

# x = BlobLocationComputer(data_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_downsampled.mp4", multiprocessing=True, gpu=True, batch_size=2000, save_dir=r"C:\troubleshooting\RAT_NOR\project_folder\csv\blob_positions", smoothing='Savitzky Golay')
# x.run()