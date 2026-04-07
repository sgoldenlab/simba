import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from simba.mixins.geometry_mixin import GeometryMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import (check_file_exist_and_readable, check_float, check_if_dir_exists, check_int, check_valid_boolean, check_valid_dataframe)
from simba.utils.data import get_cpu_pool, terminate_cpu_pool
from simba.utils.errors import FrameRangeError
from simba.utils.read_write import (find_core_cnt, get_fn_ext, get_video_meta_data)

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

    .. video:: _static/img/YoloInference_3.mp4
       :width: 500
       :loop:
       :autoplay:
       :muted:
       :align: center

    :param Union[str, os.PathLike] data_path: Path to YOLO results CSV. Expected columns: ``FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, X1..Y4``.
    :param Union[str, os.PathLike] video_path: Path to the video from which the data was produced.
    :param Union[str, os.PathLike] save_dir: Directory where to save visualization output.
    :param Optional[str] palette: Matplotlib color palette name for per-class geometry colors (e.g., ``'Set1'``, ``'tab10'``). Default: ``'Set1'``.
    :param Optional[int] core_cnt: CPU core count for parallel processing. Use ``-1`` for all available cores.
    :param float threshold: Confidence threshold in ``[0.0, 1.0]``. Detections below threshold are masked before polygon conversion.
    :param Optional[int] padding: Polygon padding offset in pixels used during multiframe bbox-to-polygon conversion for rendering. Positive values expand polygons outward, negative values shrink polygons inward. If ``None``, no padding offset is applied. This affects visualization geometry only, not the underlying YOLO detections in the input CSV.
    :param Optional[int] thickness: Polygon line thickness. If ``None``, default geometry plotter thickness is used.
    :param float opacity: Polygon fill opacity in ``[0.0, 1.0]``. Default: 0.6.
    :param Optional[Tuple[int, int, int]] outline_color: BGR color for polygon outlines. If ``None``, no outlines are drawn. Default: None.
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
                 opacity: float = 0.6,
                 outline_color: Optional[Tuple[int, int, int]] = None,
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
        check_float(name=f'{self.__class__.__name__} opacity', value=opacity, min_value=0.0, max_value=1.0)
        self.save_dir, self.verbose, self.palette, self.thickness = save_dir, verbose, palette, thickness
        self.threshold, self.padding, self.opacity, self.outline_color = threshold, padding, opacity, outline_color

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
                                  palette=self.palette,
                                  thickness=self.thickness,
                                  shape_opacity=self.opacity,
                                  outline_clr=self.outline_color,
                                  pool=pool)
        plotter.run()
        terminate_cpu_pool(pool=pool, source=self.__class__.__name__)
#
#
# if __name__ == '__main__':
#     DATA_PATH = r"E:\open_video\open_field_2\yolo_bbox_project\results\1_clip_1min.csv"
#     VIDEO_PATH = r"E:\open_video\open_field_2\sample\clips\1_clip_1min.mp4"
#     SAVE_DIR = r"E:\open_video\open_field_2\sample\clips\results"
#
#     CONFIGS = [
#         {"palette": "Set1",     "outline_color": None,            "opacity": 0.3, "thickness": None},
#         {"palette": "Set1",     "outline_color": (0, 0, 255),     "opacity": 0.5, "thickness": 2},
#         {"palette": "Set1",     "outline_color": (255, 255, 255), "opacity": 0.8, "thickness": 3},
#         {"palette": "Set1",     "outline_color": (0, 255, 0),     "opacity": 1.0, "thickness": 4},
#         {"palette": "tab10",    "outline_color": None,            "opacity": 0.3, "thickness": None},
#         {"palette": "tab10",    "outline_color": (0, 0, 0),       "opacity": 0.5, "thickness": 2},
#         {"palette": "tab10",    "outline_color": (255, 0, 0),     "opacity": 0.8, "thickness": 3},
#         {"palette": "tab10",    "outline_color": (0, 255, 255),   "opacity": 1.0, "thickness": 4},
#         {"palette": "Pastel1",  "outline_color": None,            "opacity": 0.3, "thickness": None},
#         {"palette": "Pastel1",  "outline_color": (128, 128, 128), "opacity": 0.5, "thickness": 2},
#         {"palette": "Pastel1",  "outline_color": (0, 0, 255),     "opacity": 0.8, "thickness": 3},
#         {"palette": "Pastel1",  "outline_color": (255, 255, 0),   "opacity": 1.0, "thickness": 4},
#         {"palette": "Dark2",    "outline_color": None,            "opacity": 0.3, "thickness": None},
#         {"palette": "Dark2",    "outline_color": (255, 0, 255),   "opacity": 0.5, "thickness": 2},
#         {"palette": "Dark2",    "outline_color": (0, 128, 255),   "opacity": 0.8, "thickness": 3},
#         {"palette": "Dark2",    "outline_color": (255, 255, 255), "opacity": 1.0, "thickness": 4},
#     ]
#
#     COLLECTED_DIR = os.path.join(SAVE_DIR, "collected")
#     os.makedirs(COLLECTED_DIR, exist_ok=True)
#     for idx, cfg in enumerate(CONFIGS):
#         cfg_save_dir = os.path.join(SAVE_DIR, f"config_{idx+1:02d}")
#         os.makedirs(cfg_save_dir, exist_ok=True)
#         print(f"--- Config {idx+1}/16: palette={cfg['palette']}, outline={cfg['outline_color']}, opacity={cfg['opacity']}, thickness={cfg['thickness']} ---")
#         viz = YOLOVisualizer(data_path=DATA_PATH,
#                              video_path=VIDEO_PATH,
#                              save_dir=cfg_save_dir,
#                              threshold=0.0,
#                              core_cnt=2,
#                              palette=cfg["palette"],
#                              outline_color=cfg["outline_color"],
#                              opacity=cfg["opacity"],
#                              thickness=cfg["thickness"])
#         viz.run()
#         output_path = os.path.join(cfg_save_dir, get_fn_ext(filepath=VIDEO_PATH)[1] + ".mp4")
#         shutil.copy2(output_path, os.path.join(COLLECTED_DIR, f"{idx}.mp4"))
#
#     from simba.video_processors.video_processing import clip_video_in_range, mosaic_concatenator
#     COLLECTED_DIR = r"E:\open_video\open_field_2\sample\clips\results\collected"
#     CLIP_DIR = os.path.join(COLLECTED_DIR, "clips_10s")
#     os.makedirs(CLIP_DIR, exist_ok=True)
#     clipped_paths = []
#     for idx in range(16):
#         src = os.path.join(COLLECTED_DIR, f"{idx}.mp4")
#         clip_save = os.path.join(CLIP_DIR, f"{idx}.mp4")
#         clip_video_in_range(file_path=src, start_time="00:00:00", end_time="00:00:10", save_path=clip_save, overwrite=True)
#         clipped_paths.append(clip_save)
#     mosaic_concatenator(video_paths=clipped_paths, save_path=os.path.join(COLLECTED_DIR, "mosaic_10s.mp4"), width_idx=0, height_idx=0)