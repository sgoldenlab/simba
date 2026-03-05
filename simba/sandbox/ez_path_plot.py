__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import cv2
import numpy as np
import pandas as pd

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_instance, check_int, check_str,
                                check_valid_boolean)
from simba.utils.errors import (DataHeaderError, DuplicationError,
                                InvalidFileTypeError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_fn_ext,
                                    get_number_of_header_columns_in_df,
                                    get_video_meta_data, read_config_file,
                                    read_df)

H5 = '.h5'
CSV = '.csv'

class EzPathPlot(object):
    def __init__(self,
                 body_part: str,
                 bg_color: Union[Tuple[int, int, int], int] = (255, 255, 255),
                 line_color: Union[Tuple[int, int, int], Literal['time', 'velocity']] = (147, 20, 255),
                 line_thickness: Optional[float] = None,
                 svg: bool = False,
                 line_opacity: float = 1.0,
                 smoothing_time: Optional[int] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 dpi: int = 500):
        """
        Create a path plot with enhanced options.

        :param str body_part: The specific body part to plot the path for.
        :param Union[Tuple[int, int, int], int] bg_color: Background color as RGB tuple (0-255) or grayscale int (0-255). Default: (255, 255, 255).
        :param Union[Tuple[int, int, int], Literal['time', 'velocity']] line_color: Line color as RGB tuple (0-255) or 'time'/'velocity' for color mapping. Default: (147, 20, 255).
        :param Optional[float] line_thickness: Thickness of the path line. If None, uses default. Default: None.
        :param bool svg: If True, saves as SVG format. If False, saves as PNG. Default: False.
        :param float line_opacity: Line opacity (0.0-1.0). Default: 1.0.
        :param Optional[int] smoothing_time: Smoothing time window in milliseconds. If None, no smoothing. Default: None.
        :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the plot. If None, uses default location. Default: None.
        :param int dpi: Resolution for saved images. Default: 500.
        """
        # Validate body_part
        check_str(name=f'{self.__class__.__name__} body_part', value=body_part, allow_blank=False)

        # Validate bg_color
        if isinstance(bg_color, int):
            check_int(name=f'{self.__class__.__name__} bg_color', value=bg_color, min_value=0, max_value=255)
        elif isinstance(bg_color, tuple):
            check_if_valid_rgb_tuple(data=bg_color, raise_error=True, source=f'{self.__class__.__name__} bg_color')
        else:
            check_instance(source=f'{self.__class__.__name__} bg_color', instance=bg_color, accepted_types=(tuple, int))

        # Validate line_color
        if isinstance(line_color, str):
            check_str(name=f'{self.__class__.__name__} line_color', value=line_color, options=('time', 'velocity'))
        elif isinstance(line_color, tuple):
            check_if_valid_rgb_tuple(data=line_color, raise_error=True, source=f'{self.__class__.__name__} line_color')
        else:
            check_instance(source=f'{self.__class__.__name__} line_color', instance=line_color, accepted_types=(tuple, str))

        # Validate line_thickness
        if line_thickness is not None:
            check_float(name=f'{self.__class__.__name__} line_thickness', value=line_thickness, min_value=0.0, allow_zero=False, allow_negative=False)

        # Validate svg
        check_valid_boolean(value=svg, source=f'{self.__class__.__name__} svg')

        # Validate line_opacity
        check_float(name=f'{self.__class__.__name__} line_opacity', value=line_opacity, min_value=0.0, max_value=1.0, allow_zero=True, allow_negative=False)

        # Validate smoothing_time
        if smoothing_time is not None:
            check_int(name=f'{self.__class__.__name__} smoothing_time', value=smoothing_time, min_value=1, allow_zero=False, allow_negative=False)

        # Validate save_dir
        if save_dir is not None:
            check_str(name=f'{self.__class__.__name__} save_dir', value=str(save_dir))
            check_if_dir_exists(in_dir=save_dir, create_if_not_exist=True)

        # Validate dpi
        check_int(name=f'{self.__class__.__name__} dpi', value=dpi, min_value=1, allow_zero=False, allow_negative=False)
    def run(self):
        for i in range(1, self.data.shape[0]):
            line_data = self.data[:i+1]
            img = deepcopy(self.bg_img)
            for j in range(1, line_data.shape[0]):
                x1, y1 = line_data[j-1][0], line_data[j-1][1]
                x2, y2 = line_data[j][0], line_data[j][1]
                cv2.line(img, (x1, y1), (x2, y2), self.line_color, self.line_thickness)
            cv2.circle(img, (line_data[-1][0], line_data[-1][1]), self.circle_size, self.line_color, -1)
            self.writer.write(img.astype(np.uint8))
            print(f"Frame {i}/{len(self.data)} complete...")

        self.writer.release()
        self.timer.stop_timer()
        stdout_success(msg=f"Path plot saved at {self.save_name}", elapsed_time=self.timer.elapsed_time_str)
