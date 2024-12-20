import math
import os
from copy import deepcopy
from typing import Optional, Tuple, Union

import pandas as pd

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_str, check_valid_boolean,
                                check_valid_dataframe, check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import (DuplicationError, NoFilesFoundError,
                                ParametersFileError)
from simba.utils.read_write import read_video_info_csv
from simba.utils.warnings import InvalidValueWarning


def read_video_info(video_name: str,
                    video_info_df: pd.DataFrame,
                    raise_error: Optional[bool] = True) -> Union[Tuple[pd.DataFrame, float, float], Tuple[None, None, None]]:
    """
    Helper to read the metadata (pixels per mm, resolution, fps etc) from the video_info.csv for a single input file/video

    :parameter pd.DataFrame vid_info_df: Parsed ``project_folder/logs/video_info.csv`` file. This file can be parsed by :meth:`simba.utils.read_write.read_video_info_csv`.
    :parameter str video_name: Name of the video as represented in the ``Video`` column of the ``project_folder/logs/video_info.csv`` file.
    :parameter Optional[bool] raise_error: If True, raises error if the video cannot be found in the ``vid_info_df`` file. If False, returns None if the video cannot be found.
    :returns: 3-part tuple: One row DataFrame representing the video in the ``project_folder/logs/video_info.csv`` file, the frame rate of the video, and the the pixels per millimeter of the video
    :rtype: Union[Tuple[pd.DataFrame, float, float], Tuple[None, None, None]]

    :example:
    >>> video_info_df = read_video_info_csv(file_path='project_folder/logs/video_info.csv')
    >>> read_video_info(vid_info_df=vid_info_df, video_name='Together_1')
    """

    check_str(name=f'{read_video_info.__name__} video_name', value=video_name, allow_blank=False)
    check_valid_boolean(value=[raise_error], source=f'{read_video_info.__name__} raise_error')
    check_valid_dataframe(df=video_info_df, source='', required_fields=["pixels/mm", "fps", "Video"])
    video_settings = video_info_df.loc[video_info_df["Video"] == video_name]
    if len(video_settings) > 1:
        raise DuplicationError(msg=f"SimBA found multiple rows in `project_folder/logs/video_info.csv` for videos named {video_name}. Please make sure that each video name is represented ONCE in the file", source='')
    elif len(video_settings) < 1:
        if raise_error:
            raise ParametersFileError(msg=f"SimBA could not find {video_name} in the `project_folder/logs/video_info.csv` file. Make sure all videos analyzed are represented in the file.", source='')
        else:
            return (None, None, None)
    else:
        px_per_mm = video_settings["pixels/mm"].values[0]
        fps = video_settings["fps"].values[0]
        if math.isnan(px_per_mm):
            raise ParametersFileError(msg=f'Pixels per millimeter for video {video_name} in the `project_folder/logs/video_info.csv` file is not a valid number. Please correct it to proceed.')
        if math.isnan(fps):
            raise ParametersFileError(msg=f'The FPS for video {video_name} in the `project_folder/logs/video_info.csv` file is not a valid number. Please correct it to proceed.')
        check_float(name=f'pixels per millimeter video {video_name}', value=px_per_mm); check_float(name=f'fps video {video_name}', value=fps)
        px_per_mm, fps = float(px_per_mm), float(fps)
        if px_per_mm <= 0:
            InvalidValueWarning(msg=f"Video {video_name} has a pixel per millimeter conversion factor of 0 or less. Correct the pixel/mm conversion factor values inside the `project_folder/logs/video_info.csv` file", source='')
        if fps <= 1:
            InvalidValueWarning(msg=f"Video {video_name} an FPS of 1 or less.  It is recommended to use videos with more than one frame per second. If inaccurate, correct the FPS values inside the `project_folder/logs/video_info.csv` file", source='')
        return video_settings, px_per_mm, fps



    #
    #         px_per_mm = float(video_settings["pixels/mm"])
    #         fps = float(video_settings["fps"])
    #         if math.isnan(px_per_mm):
    #             raise ParametersFileError(
    #                 msg=f'Pixels per millimeter for video {video_name} in the {self.video_info_path} file is not a valid number.')
    #         if math.isnan(fps):
    #             raise ParametersFileError(
    #                 msg=f'The FPS for video {video_name} in the {self.video_info_path} file is not a valid number.')
    #         return video_settings, px_per_mm, fps
    #     except TypeError:
    #         raise ParametersFileError(
    #             msg=f"Make sure the videos that are going to be analyzed are represented with APPROPRIATE VALUES inside the project_folder/logs/video_info.csv file in your SimBA project. Could not interpret the fps, pixels per millimeter and/or fps as numerical values for video {video_name}",
    #             source=self.__class__.__name__,
    #         )
    # # return info_df



read_video_info(video_name='501_MA142_Gi_CNO_0514', video_info_df_path=r"C:\troubleshooting\mitra\project_folder\logs\video_info.csv")