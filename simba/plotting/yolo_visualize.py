import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.geometry_mixin import GeometryMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dataframe)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool
from simba.utils.errors import FrameRangeError
from simba.utils.read_write import (find_core_cnt, get_fn_ext,
                                    get_video_meta_data)

EXPECTED_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
FRAME = 'FRAME'
CLASS_ID = 'CLASS_ID'
CONFIDENCE = 'CONFIDENCE'
CLASS_NAME = 'CLASS_NAME'
CORD_FIELDS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']

class YOLOVisualizer():

    """
    Visualize YOLO bounding-box inference results on a source video.

    .. seealso::
       For bounding-box inference, see :class:`simba.model.yolo_inference.YoloInference`.

    .. video:: _static/img/YOLOVisualizer.webm
      :width: 500
      :loop:
      :autoplay:
      :muted:
      :align: center

    .. video:: _static/img/YoloInference_1.webm
       :width: 500
       :loop:
       :autoplay:
       :muted:
       :align: center

    .. video:: _static/img/YoloInference_2.webm
       :width: 500
       :loop:
       :autoplay:
       :muted:
       :align: center

    :param Union[str, os.PathLike] data_path: Path to YOLO results CSV. Expected columns: ``FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, X1..Y4``.
    :param Union[str, os.PathLike] video_path: Path to the video from which the data was produced.
    :param Union[str, os.PathLike] save_dir: Directory where to save visualization output.
    :param Optional[str] palette: Palette option (reserved for compatibility). Current implementation uses a fixed color.
    :param Optional[int] core_cnt: CPU core count for parallel processing. Use ``-1`` for all available cores.
    :param float threshold: Confidence threshold in ``[0.0, 1.0]``. Detections below threshold are masked before polygon conversion.
    :param Optional[int] padding: Polygon padding offset in pixels used during multiframe bbox-to-polygon conversion for rendering. Positive values expand polygons outward, negative values shrink polygons inward. If ``None``, no padding offset is applied. This affects visualization geometry only, not the underlying YOLO detections in the input CSV.
    :param Optional[int] thickness: Polygon line thickness. If ``None``, default geometry plotter thickness is used.
    :param bool verbose: If True, prints progress information. Default: True.
    :raises FrameRangeError: If YOLO result frame coverage does not match video frame count.

    :example:
    >>> test = YOLOVisualizer(
    ...     data_path=r"/mnt/c/troubleshooting/yolo_inference/08102021_DOT_Rat7_8(2).csv",
    ...     video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat7_8(2).mp4",
    ...     save_dir="/mnt/c/troubleshooting/yolo_videos",
    ...     threshold=0.25,
    ...     core_cnt=4
    ... )
    >>> test.run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 palette: Optional[str] = 'Set1',
                 core_cnt: Optional[int] = -1,
                 threshold: float = 0.0,
                 padding: Optional[int] = 20,
                 thickness: Optional[int] = None,
                 verbose: bool = True):

        check_file_exist_and_readable(file_path=data_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.data_path, self.video_path = data_path, video_path
        self.video_name = get_fn_ext(filepath=data_path)[1]
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        if padding is not None: check_int(name=f'{self.__class__.__name__} padding', value=padding, min_value=-1, unaccepted_vals=[0])
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        self.core_cnt = core_cnt
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]: self.core_cnt = find_core_cnt()[0]
        if thickness is not None:
            check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=0, unaccepted_vals=[0])
        check_if_dir_exists(in_dir=save_dir)
        check_valid_boolean(value=[verbose], source=self.__class__.__name__, raise_error=True)
        self.save_dir, self.verbose, self.palette, self.thickness = save_dir, verbose, palette, thickness
        self.threshold, self.padding = threshold, padding

    def run(self):
        data_df = pd.read_csv(self.data_path, index_col=0)
        check_valid_dataframe(df=data_df, source=self.__class__.__name__, required_fields=EXPECTED_COLS)
        df_frm_cnt = np.unique(data_df[FRAME].values).shape[0]
        pool = get_cpu_pool(core_cnt=self.core_cnt, source=self.__class__.__name__)
        if self.video_meta_data['frame_count'] != df_frm_cnt:
            raise FrameRangeError(
                msg=f'The bounding boxes contain data for {df_frm_cnt} frames, while the video is {self.video_meta_data["frame_count"]} frames',
                source=self.__class__.__name__)
        classes = np.unique(data_df[CLASS_NAME].values)
        geometries = []
        for cls in classes:
            cls_df = data_df[data_df[CLASS_NAME] == cls]
            class_id = cls_df[CLASS_ID].iloc[0]
            missing_frms = [x for x in np.arange(0, df_frm_cnt) if x not in cls_df[FRAME].values]
            missing_df = pd.DataFrame(missing_frms, columns=[FRAME])
            missing_df[CLASS_ID], missing_df[CLASS_NAME], missing_df[CONFIDENCE] = class_id, cls, 0
            for cord_col in CORD_FIELDS: missing_df[cord_col] = 0
            cls_df = pd.concat([cls_df, missing_df], axis=0).sort_values(by=[FRAME])
            cls_df.loc[cls_df[CONFIDENCE] < self.threshold, CORD_FIELDS] = -1
            cls_arr = cls_df[CORD_FIELDS].values
            cls_arr = cls_arr.reshape(cls_arr.shape[0], 4, 2)
            geometries.append(GeometryMixin().multiframe_bodyparts_to_polygon(data=cls_arr, video_name=self.video_name, core_cnt=self.core_cnt, verbose=self.verbose, parallel_offset=self.padding, pool=pool))
        plotter = GeometryPlotter(geometries=geometries,
                                  video_name=self.video_path,
                                  core_cnt=self.core_cnt,
                                  save_dir=self.save_dir,
                                  verbose=self.verbose,
                                  colors=[(0, 255, 255)],
                                  thickness=self.thickness,
                                  shape_opacity=0.6,
                                  pool=pool)
        plotter.run()
        terminate_cpu_pool(pool=pool, source=self.__class__.__name__)










# test = YOLOVisualizer(data_path=r"E:\litpose_yolo\bbox\out_pose\6.01.001_2026_03_11_23_25_00_000_2_cam1.csv",
#                       video_path=r"Z:\home\simon\lp_300126\videos\6.01.001_2026_03_11_23_25_00_000_2\6.01.001_2026_03_11_23_25_00_000_2_cam1.mp4",
#                       save_dir=r"E:\litpose_yolo\bbox\out_pose",
#                       threshold=0.0,
#                       core_cnt=4)
# test.run()