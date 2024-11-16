import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from simba.mixins.geometry_mixin import GeometryMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dataframe)
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
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 palette: Optional[str] = 'Set1',
                 core_cnt: Optional[int] = -1,
                 thickness: Optional[int] = None,
                 verbose: Optional[bool] = False):

        """
        Visualize the results of YOLO model.

        .. seelalso::
           :func:`simba.bounding_box_tools.yolo.model.inference_yolo`

        .. video:: _static/img/YOLOVisualizer.webm
          :width: 500
          :loop:
          :autoplay:

        :param Union[str, os.PathLike] data_path: Path to YOLO results CSV resylts. Produced by :func:`simba.bounding_box_tools.yolo.model.inference_yolo`
        :param Union[str, os.PathLike] video_path: Path to the video from which the data was produced.
        :param Union[str, os.PathLike] save_dir: Directory where to save the video results.
        :param Optional[str] palette: Color palette from where to draw the colors for the bounding polygons/boxes. Default: `Set1`.
        :param Optional[int] core_cnt: The number of CPU cores use dto produce the video. -1 for all available cores. Default: -1.
        :param Optional[bool] verbose: If True, prints progress (useful for debugging). Default: False.

        :example:
        >>> test = YOLOVisualizer(data_path=r"/mnt/c/troubleshooting/yolo_inference/08102021_DOT_Rat7_8(2).csv", video_path=r'/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat7_8(2).mp4', save_dir="/mnt/c/troubleshooting/yolo_videos")
        >>> test.run()
        """

        check_file_exist_and_readable(file_path=data_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.data_path, self.video_path = data_path, video_path
        self.video_name = get_fn_ext(filepath=data_path)[1]
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = core_cnt
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]:
            self.core_cnt = find_core_cnt()[0]
        if thickness is not None:
            check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=0, unaccepted_vals=[0])
        check_if_dir_exists(in_dir=save_dir)
        check_valid_boolean(value=[verbose], source=self.__class__.__name__, raise_error=True)
        self.save_dir, self.verbose, self.palette, self.thickness = save_dir, verbose, palette, thickness

    def run(self):
        data_df = pd.read_csv(self.data_path, index_col=0)
        check_valid_dataframe(df=data_df, source=self.__class__.__name__, required_fields=EXPECTED_COLS)
        df_frm_cnt = np.unique(data_df[FRAME].values).shape[0]
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
            for cord_col in ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']: missing_df[cord_col] = 0
            cls_df = pd.concat([cls_df, missing_df], axis=0).sort_values(by=[FRAME])
            cls_arr = cls_df[CORD_FIELDS].values
            cls_arr = cls_arr.reshape(cls_arr.shape[0], 4, 2)
            geometries.append(GeometryMixin().multiframe_bodyparts_to_polygon(data=cls_arr, video_name=self.video_name, core_cnt=self.core_cnt, verbose=self.verbose))
        plotter = GeometryPlotter(geometries=geometries, video_name=self.video_path, core_cnt=self.core_cnt,
                                  save_dir=self.save_dir, verbose=self.verbose, palette=self.palette, thickness=self.thickness)
        plotter.run()