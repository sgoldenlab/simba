__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Union, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

from simba.utils.errors import (DataHeaderError, DuplicationError, InvalidFileTypeError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_config_file, read_df, get_number_of_header_columns_in_df
from simba.utils.checks import check_file_exist_and_readable, check_if_valid_rgb_tuple, check_int

H5 = '.h5'
CSV = '.csv'

class EzPathPlot(object):
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 body_part: str,
                 bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
                 line_color: Optional[Tuple[int, int, int]] = (147, 20, 255),
                 line_thickness: Optional[int] = 10,
                 circle_size: Optional[int] = 5):
        """
        Create a simple path plot for a single path in a single video.

        .. note::
           For more complex path plots with/without multiprocessing, see ``simba.plotting.path_plotter.PathPlotterSingleCore`` and ``simba.plotting.path_plotter_mp.PathPlotterMulticore``.

           Notebook example link ->


        .. image:: _static/img/EzPathPlot.gif
          :width: 500
          :align: center

        :param Union[str, os.PathLike] data_path: The path to the data file in H5c or CSV format containing the coordinates.
        :param Union[str, os.PathLike] video_path: The path to the video file.
        :param str body_part: The specific body part to plot the path for.
        :param Optional[Tuple[int, int, int]] bg_color: The background color of the plot. Defaults to (255, 255, 255).
        :param Optional[Tuple[int, int, int]] line_color: The color of the path line. Defaults to (147, 20, 255).
        :param Optional[int] line_thickness: The thickness of the path line. Defaults to 10.
        :param Optional[int] circle_size: The size of the circle indicating each data point. Defaults to 5.

        :example:
        >>> path_plotter = EzPathPlot(data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/h5/Together_1DLC_resnet50_two_black_mice_DLC_052820May27shuffle1_150000_el.h5', video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi', body_part='Mouse_1_Nose', bg_color=(255, 255, 255), line_color=(147,20,255))
        >>> path_plotter.run()

        """
        check_file_exist_and_readable(file_path=data_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        check_if_valid_rgb_tuple(data=bg_color)
        check_if_valid_rgb_tuple(data=line_color)
        check_int(name=f'{self.__class__.__name__} line_thickness', value=line_thickness, min_value=1)
        check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        if line_color == bg_color:
            raise DuplicationError(msg=f"The line and background cannot be identical - ({line_color})", source=self.__class__.__name__)

        dir, file_name, ext = get_fn_ext(filepath=data_path)
        if ext.lower() == H5:
            self.data = pd.read_hdf(data_path)
            headers = []
            if len(self.data.columns[0]) == 4:
                for c in self.data.columns:
                    headers.append("{}_{}_{}".format(c[1], c[2], c[3]))
            elif len(self.data.columns[0]) == 3:
                for c in self.data.columns:
                    headers.append("{}_{}".format(c[2], c[3]))
            self.data.columns = headers
        elif ext.lower() == CSV:
            self.data = pd.read_csv(data_path)
        else:
            raise InvalidFileTypeError(msg=f"File type {ext} is not supported (OPTIONS: h5 or csv)")
        if len(self.data.columns[0]) == 4:
            self.data = self.data.loc[3:]
        elif len(self.data.columns[0]) == 3:
            self.data = self.data.loc[2:]
        body_parts_available = list(set([x[:-2] for x in self.data.columns]))
        if body_part not in body_parts_available:
            raise DataHeaderError(msg=f"Body-part {body_part} is not present in the data file. The body-parts available are: {body_parts_available}", source=self.__class__.__name__)
        bps = [f'{body_part}_x', f'{body_part}_y']
        if (bps[0] not in self.data.columns) or (bps[1] not in self.data.columns):
            raise DataHeaderError(msg=f"Could not finc column {bps[0]} and/or column {bps[1]} in the data file {data_path}", source=self.__class__.__name__)
        self.data = self.data[bps].fillna(method="ffill").astype(int).reset_index(drop=True).values
        self.save_name = os.path.join(dir, f"{file_name}_line_plot.mp4")
        self.writer = cv2.VideoWriter(self.save_name, 0x7634706D, int(self.video_meta_data["fps"]), (self.video_meta_data["width"], self.video_meta_data["height"]))
        self.bg_img = np.zeros([self.video_meta_data["height"], self.video_meta_data["width"], 3])
        self.bg_img[:] = [bg_color]
        self.line_color, self.line_thickness, self.circle_size  = line_color, line_thickness, circle_size
        self.timer = SimbaTimer(start=True)
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
