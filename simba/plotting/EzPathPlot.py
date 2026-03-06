
__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from typing import Optional, Tuple, Union, List
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from copy import deepcopy
import numpy as np
from simba.utils.checks import (check_file_exist_and_readable, check_that_column_exist,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_valid_tuple, check_str,
                                check_float, check_valid_boolean, check_instance)
from simba.utils.printing import stdout_success, stdout_information
from simba.utils.read_write import (get_fn_ext, find_files_of_filetypes_in_directory, read_df, find_video_of_file, read_frm_of_video)
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.errors import InvalidInputError
from simba.utils.warnings import VideoFileWarning

H5 = ".h5"
CSV = ".csv"


class EzPathPlot(object):
    """
    Create a simple path plot image for a single or several pose-estimation files.

    Draws the trajectory of one body part over time as a line on a solid or video-frame background.

    .. image:: _static/img/get_path_img.webp
       :width: 1000
       :align: center

    .. seealso::
       For batch processing with more options and optional multiprocessing, use :class:`simba.plotting.path_plotter.PathPlotterSingleCore` or :class:`simba.plotting.path_plotter_mp.PathPlotterMulticore`.

    :param Union[str, os.PathLike, List[Union[str, os.PathLike]]] data_path: Path to a single pose CSV/H5 file, a directory of CSV files, or a list of file paths. Only CSV is used when a directory is given.
    :param str body_part: Name of the body part to plot (e.g. ``"Nose"``). Must match column prefixes ``<body_part>_x``, ``<body_part>_y`` in the data.
    :param Union[Tuple[int, int, int], int] bg_color: Background as RGB tuple ``(R, G, B)`` or an integer. If integer, the first frame of the matching video is used as background (requires ``video_dir``). Default ``(255, 255, 255)``.
    :param Optional[Union[str, os.PathLike]] video_dir: Directory to search for videos when ``bg_color`` is an integer. Required when using a video frame as background.
    :param Union[Tuple[int, int, int], Literal['time', 'velocity']] line_color: Line color as RGB tuple or ``'time'`` / ``'velocity'`` for color mapping. Default ``(147, 20, 255)``.
    :param float line_thickness: Thickness of the path line. Must be > 0.
    :param bool svg: If True, save as SVG; otherwise PNG.
    :param Optional[Tuple[int, int]] size: Output image size ``(width, height)`` in pixels. If None, size is taken from the first video frame when using video background.
    :param float line_opacity: Opacity of the path line in [0, 1]. Default 1.0.
    :param Optional[int] smoothing_time: Smoothing window in frames. If given, path coordinates are smoothed before plotting.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save plots. If None, each plot is saved next to its source data file.
    :param int dpi: DPI for saved images. Default 500.
    :param bool verbose: If True, print progress per file.

    :example:
    >>> plotter = EzPathPlot(
    ...     data_path='/path/to/pose.csv',
    ...     body_part='Nose',
    ...     bg_color=(255, 255, 255),
    ...     line_color='velocity',
    ...     save_dir='/path/to/output'
    ... )
    >>> plotter.run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 body_part: str,
                 bg_color: Union[Tuple[int, int, int], int] = (255, 255, 255),
                 video_dir: Optional[Union[str, os.PathLike]] = None,
                 line_color: Union[Tuple[int, int, int], Literal['time', 'velocity']] = (147, 20, 255),
                 line_thickness: float = 1,
                 svg: bool = False,
                 size: Optional[Tuple[int, int]] = None,
                 line_opacity: float = 1.0,
                 smoothing_time: Optional[int] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 dpi: int = 500,
                 verbose: bool = True):

        check_str(name=f'{self.__class__.__name__} body_part', value=body_part, allow_blank=False)
        if isinstance(bg_color, int):
            check_int(name=f'{self.__class__.__name__} bg_color', value=bg_color, min_value=0)
            if video_dir is None:
                raise InvalidInputError(msg='If background is a video, pass the video directory', source=self.__class__.__name__)
            check_if_dir_exists(in_dir=video_dir, source=f'{self.__class__.__name__} video_dir', raise_error=True)
        else:
            check_if_valid_rgb_tuple(data=bg_color, raise_error=True, source=f'{self.__class__.__name__} bg_color')
        if isinstance(line_color, str):
            check_str(name=f'{self.__class__.__name__} line_color', value=line_color, options=('time', 'velocity'))
        else:
            check_if_valid_rgb_tuple(data=line_color, raise_error=True, source=f'{self.__class__.__name__} line_color')
        check_float(name=f'{self.__class__.__name__} line_thickness', value=line_thickness, min_value=0.0, allow_zero=False, allow_negative=False)
        check_valid_boolean(value=svg, source=f'{self.__class__.__name__} svg')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_float(name=f'{self.__class__.__name__} line_opacity', value=line_opacity, min_value=0.0, max_value=1.0, allow_zero=True, allow_negative=False)
        if size is not None:
            check_instance(source=f'{self.__class__.__name__} size', instance=size, accepted_types=(tuple,))
            check_valid_tuple(x=size, source=f'{self.__class__.__name__} size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        if smoothing_time is not None: check_int(name=f'{self.__class__.__name__} smoothing_time', value=smoothing_time, min_value=1, allow_zero=False, allow_negative=False)
        if save_dir is not None:
            check_str(name=f'{self.__class__.__name__} save_dir', value=str(save_dir))
            check_if_dir_exists(in_dir=save_dir, create_if_not_exist=True)
        check_int(name=f'{self.__class__.__name__} dpi', value=dpi, min_value=1, allow_zero=False, allow_negative=False)
        self.body_part, self.bg_color, self.line_color = body_part, bg_color, line_color
        self.line_thickness, self.svg, self.size = line_thickness, svg, size
        self.line_opacity, self.smoothing_time = line_opacity, smoothing_time
        self.save_dir, self.dpi, self.verbose, self.video_dir = save_dir, dpi, verbose, video_dir
        if os.path.isfile(data_path):
            self.data_paths = [data_path]
        elif os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.csv'], raise_error=True, sort_alphabetically=True)
        elif isinstance(data_path, list):
            self.data_paths = data_path
        for p in self.data_paths:
            check_file_exist_and_readable(file_path=p, raise_error=True)
        self.bp_cols = [f"{body_part}_x", f"{body_part}_y", f"{body_part}_p"]

    def run(self):
        for file_cnt, data_path in enumerate(self.data_paths):
            data = read_df(file_path=data_path)
            check_that_column_exist(df=data, column_name=self.bp_cols, file_name=data_path, raise_error=True)
            data = data[self.bp_cols].values.astype(np.float32)
            file_name = get_fn_ext(filepath=data_path)[1]
            save_dir = os.path.dirname(data_path) if self.save_dir is None else self.save_dir
            file_ext = 'svg' if self.svg else 'png'
            save_path = os.path.join(save_dir, f'{file_name}.{file_ext}')
            if isinstance(self.bg_color, int):
                video_path = find_video_of_file(video_dir=self.video_dir, filename=file_name, raise_error=False, warning=True, recursive=True)
                if video_path is None:
                    VideoFileWarning(msg=f'Skipping video {file_name}. Could not find a video for {file_name} in directory  {self.video_dir}', source=self.__class__.__name__)
                    continue
                else:
                    bg_clr = read_frm_of_video(video_path=video_path, frame_index=0)
            else:
                bg_clr = deepcopy(self.bg_color)
            if self.verbose: stdout_information(msg=f'Creating path plot {save_path} ({file_cnt+1}/{len(self.data_paths)})...')

            _ = PlottingMixin().get_path_img(data=data[:, :2],
                                             size=self.size,
                                             line_thickness=self.line_thickness,
                                             line_color=self.line_color,
                                             bg_clr=bg_clr,
                                             save_path=save_path,
                                             svg=True if self.svg else False,
                                             smoothing_time=self.smoothing_time,
                                             dpi=self.dpi,
                                             opacity=self.line_opacity)
        stdout_success(msg=f'Created {len(self.data_paths)} path plot(s).', source=self.__class__.__name__)






#DATA_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/2022-06-20_NOB_DOT_4.csv'
# DATA_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/2022-06-20_NOB_DOT_4.csv'
# VIDEO_DIR = '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos'
#
# BP = 'Nose'
#
#
# plotter = EzPathPlot(data_path=DATA_PATH,
#                      body_part=BP,
#                      line_color='velocity',
#                      video_dir=VIDEO_DIR,
#                      svg=False)
# plotter.run()

