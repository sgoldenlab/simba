__author__ = "Simon Nilsson"

import os
import shutil
from datetime import datetime
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_lst)
from simba.utils.enums import Paths, TagNames
from simba.utils.errors import FFMPEGCodecGPUError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import copy_files_to_directory, get_fn_ext
from simba.video_processors.video_processing import (
    horizontal_video_concatenator, mixed_mosaic_concatenator,
    mosaic_concatenator, vertical_video_concatenator)

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
MOSAIC = "mosaic"
MIXED_MOSAIC = "mixed_mosaic"
ACCEPTED_TYPES = [HORIZONTAL, VERTICAL, MOSAIC, MIXED_MOSAIC]


class FrameMergererFFmpeg(ConfigReader):
    """
    Merge separate visualizations of classifications, descriptive statistics etc., into  single video mosaic.

    .. note::
       `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-12-merge-frames>`_.

        .. image:: _static/img/mosaic_videos.gif
          :width: 600
          :align: center

    :parameter str config_path: Optional path to SimBA project config file in Configparser format.
    :parameter Literal["horizontal", "vertical", "mosaic", "mixed_mosaic"] concat_type: Type of concatenation. OPTIONS: 'horizontal', 'vertical', 'mosaic', 'mixed_mosaic'.
    :parameter List[Union[str, os.PathLike]] video_paths: List with videos to concatenate.
    :parameter Optional[int] video_height: Optional height of the canatenated videos. Required if concat concat_type is not mixed_mosaic.
    :parameter int video_width: Optional wisth of the canatenated videos. Required if concat concat_type is not mixed_mosaic.
    :parameter Optional[bool] gpu: If True, use NVIDEA FFMpeg GPU codecs. Default False.

    :example:
    >>> video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4']
    >>> merger = FrameMergererFFmpeg(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', video_paths=videos, video_height=600, video_width=600, concat_type='mosaic')
    >>> merger.run()
    """

    def __init__(
        self,
        concat_type: Literal["horizontal", "vertical", "mosaic", "mixed_mosaic"],
        video_paths: List[Union[str, os.PathLike]],
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        config_path: Optional[str] = None,
        gpu: Optional[bool] = False,
    ):

        if gpu and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(
                msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None",
                source=self.__class__.__name__,
            )
        check_ffmpeg_available()
        check_str(
            name=f"{FrameMergererFFmpeg.__name__} concat_type",
            value=concat_type,
            options=ACCEPTED_TYPES,
        )
        check_valid_lst(
            data=video_paths,
            source=f"{self.__class__.__name__} video_paths",
            valid_dtypes=(str,),
            min_len=2,
        )
        for i in video_paths:
            check_file_exist_and_readable(file_path=i)
        if concat_type != MIXED_MOSAIC:
            check_int(
                name=f"{FrameMergererFFmpeg.__name__} video_height",
                value=video_height,
                min_value=0,
            )
            check_int(
                name=f"{FrameMergererFFmpeg.__name__} video_width",
                value=video_height,
                min_value=0,
            )
        if config_path is not None:
            ConfigReader.__init__(self, config_path=config_path)
            log_event(
                logger_name=str(__class__.__name__),
                log_type=TagNames.CLASS_INIT.value,
                msg=self.create_log_msg_from_init_args(locals=locals()),
            )
            self.output_dir = os.path.join(
                self.project_path, Paths.CONCAT_VIDEOS_DIR.value
            )
            self.output_path = os.path.join(
                self.project_path,
                Paths.CONCAT_VIDEOS_DIR.value,
                f"merged_video_{self.datetime}.mp4",
            )
        else:
            self.timer = SimbaTimer(start=True)
            self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            self.output_dir, _, _ = get_fn_ext(filepath=video_paths[0])
            self.output_path = os.path.join(
                self.output_dir, f"merged_video_{self.datetime}.mp4"
            )

        self.video_height, self.video_width, self.gpu = video_height, video_width, gpu
        self.video_paths, self.concat_type = video_paths, concat_type
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self):
        if self.concat_type == HORIZONTAL:
            _ = horizontal_video_concatenator(
                video_paths=self.video_paths,
                save_path=self.output_path,
                height_px=self.video_height,
                gpu=self.gpu,
                verbose=True,
            )
        elif self.concat_type == VERTICAL:
            _ = vertical_video_concatenator(
                video_paths=self.video_paths,
                save_path=self.output_path,
                width_px=self.video_width,
                gpu=self.gpu,
                verbose=True,
            )
        elif self.concat_type == MOSAIC:
            _ = mosaic_concatenator(
                video_paths=self.video_paths,
                save_path=self.output_path,
                width_px=self.video_width,
                height_px=self.video_height,
                gpu=self.gpu,
                verbose=True,
            )
        else:
            _ = mixed_mosaic_concatenator(
                video_paths=self.video_paths,
                save_path=self.output_path,
                gpu=self.gpu,
                verbose=True,
            )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Merged video saved at {self.output_path}",
            source=self.__class__.__name__,
            elapsed_time=self.timer.elapsed_time_str,
        )


# videos = ['/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4', '/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/SI_DAY3_308_CD1_PRESENT_downsampled.mp4']
#
# merger = FrameMergererFFmpeg(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                     video_paths=videos,
#                     video_height=600,
#                     video_width=600,
#                     concat_type='vertical') #horizontal, vertical, mosaic, mixed_mosaic
# merger.run()


#
# FrameMergererFFmpeg(config_path=None,
#                     frame_types={'Video 1': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                  'Video 2': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi'},
#                     video_height=640,
#                     video_width=480,
#                     concat_type='vertical') #horizontal, vertical, mosaic, mixed_mosaic
#
#


# FrameMergererFFmpeg(config_path=None,
#                     frame_types={'Video 1': r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4',
#                                  'Video 2': r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4',
#                                  'Video 3': r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4',
#                                  'Video 4': r'C:\Users\Nape_Computer_2\Desktop\test_videos\Box1_PM2_day_5_20211104T171021.mp4'},
#                     video_height=640,
#                     video_width=480,
#                     concat_type='mixed_mosaic',
#                     gpu=False) #horizontal, vertical, mosaic, mixed_mosaic
#
