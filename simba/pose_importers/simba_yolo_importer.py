import argparse
import os
import sys
from typing import Dict, Optional, Union

import pandas as pd

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_boolean,
                                check_valid_dataframe, check_valid_tuple)
from simba.utils.enums import ConfigKey
from simba.utils.errors import PermissionError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    write_df)

OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
CLASS_ID, CONFIDENCE, CLASS_NAME  = 'CLASS_ID', 'CONFIDENCE', 'CLASS_NAME'
FRAME = 'FRAME'

class SimBAYoloImporter(ConfigReader):
    """
    Import YOLO pose estimation results into SimBA project format with optional interpolation and smoothing.

    .. seealso::
       YOLO pose data can be created with :func:`simba.model.yolo_pose_inference.YOLOPoseInference` or :func:`simba.model.yolo_pose_track_inference.YOLOPoseTrackInference`.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Union[str, os.PathLike] data_dir: Directory containing YOLO results CSV files.
    :param bool verbose: If True, prints progress information. Default: False.
    :param Optional[float] px_per_mm: Pixels per millimeter for the videos. If provided, updates project video info.
    :param Optional[tuple] resolution: Video resolution as (width, height). Default: (927, 927).
    :param Optional[float] fps: Video frames per second. Default: 927.
    :param Optional[bool] add_to_video_info: If True, adds video metadata to project video_info.csv. Default: True.
    :param Optional[Dict[str, str]] interpolation_settings: Dictionary with 'method' ('linear', 'quadratic', 'nearest') and 'type' ('body-parts', 'animals'). If None, no interpolation applied.
    :param Optional[Dict[str, str]] smoothing_settings: Dictionary with 'method' ('savitzky-golay', 'gaussian') and 'time_window' (int, milliseconds). If None, no smoothing applied.

    :example:
        >>> importer = SimBAYoloImporter(data_dir='yolo_results/', config_path='project_config.ini', verbose=True, px_per_mm=1.43, fps=30)
        >>> importer.run()

    :example:
        >>> # With interpolation and smoothing
        >>> interpolation = {'method': 'linear', 'type': 'body-parts'}
        >>> smoothing = {'method': 'savitzky-golay', 'time_window': 200}
        >>> importer = SimBAYoloImporter(data_dir='yolo_results/', config_path='project_config.ini', interpolation_settings=interpolation, smoothing_settings=smoothing)
        >>> importer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike],
                 verbose: bool = False,
                 px_per_mm: Optional[float] = None,
                 resolution: Optional[tuple] = None, #WxH
                 fps: Optional[Union[float]] = None,
                 add_to_video_info: Optional[bool] = True,
                 interpolation_settings: Optional[Dict[str, str]] = None,
                 smoothing_settings: Optional[Dict[str, str]] = None):

        if px_per_mm is not None:
            check_float(name=f'{self.__class__.__name__} px_per_mm', value=px_per_mm, allow_negative=False, allow_zero=False)
        if resolution is not None:
            check_valid_tuple(x=resolution, source=f'{self.__class__.__name__} resolution', accepted_lengths=(2,), valid_dtypes=(int,))
        else:
            resolution = (927, 927)
        if fps is not None:
            check_float(name=f'{self.__class__.__name__} fps', value=fps, allow_negative=False, allow_zero=False)
        else:
            fps = 927
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=add_to_video_info, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(data=interpolation_settings, key=['method', 'type'], name=f'{self.__class__.__name__} interpolation_settings')
            check_str(name=f'{self.__class__.__name__} interpolation_settings type', value=interpolation_settings['type'], options=('body-parts', 'animals'))
            check_str(name=f'{self.__class__.__name__} interpolation_settings method', value=interpolation_settings['method'], options=('linear', 'quadratic', 'nearest'))
            self.interpolation_type, self.interpolation_method = interpolation_settings['type'], interpolation_settings['method']
        else:
            self.interpolation_type, self.interpolation_method = None, None
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(data=smoothing_settings, key=['method', 'time_window'], name=f'{self.__class__.__name__} smoothing_settings')
            check_str(name=f'{self.__class__.__name__} smoothing_settings method', value=smoothing_settings['method'], options=('savitzky-golay', 'gaussian'))
            check_int(name=f'{self.__class__.__name__} smoothing_settings time_window', value=smoothing_settings['time_window'], min_value=1)
            self.smoothing_time, self.smoothing_method = smoothing_settings['time_window'], smoothing_settings['method']
        else:
            self.smoothing_time, self.smoothing_method = None, None
        read_video_info = True if px_per_mm is not None else False
        check_if_dir_exists(in_dir=data_dir, source=f'{self.__class__.__name__} data_dir')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=read_video_info, create_logger=False)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions='.csv', as_dict=True, raise_error=True)
        self.verbose, self.px_per_mm, self.resolution, self.fps, self.add_to_video_info = verbose, px_per_mm, resolution, fps, add_to_video_info
        self.interpolation_settings, self.smoothing_settings = interpolation_settings, smoothing_settings
    def run(self):
        for video_counter, (video_name, data_path) in enumerate(self.data_paths.items()):
            video_timer = SimbaTimer(start=True)
            yolo_df = pd.read_csv(data_path, index_col=0)
            check_valid_dataframe(df=yolo_df, source=data_path, required_fields=OUT_COLS)
            class_names = yolo_df[CLASS_NAME].unique()
            yolo_df = yolo_df.drop([CLASS_ID, CONFIDENCE] + COORD_COLS, axis=1)
            if video_counter == 0:
                bp_names = [x for x in yolo_df.columns if x not in (FRAME, CONFIDENCE, CLASS_NAME)]
                bp_names = list(dict.fromkeys([x[:-2] for x in bp_names]))
                bp_names = [f"{cls}_{bp}" for cls in class_names for bp in bp_names]
                with open(self.body_parts_path, "w") as f:
                    f.writelines(f"{bp}\n" for bp in bp_names)
                self.config.set(section=ConfigKey.MULTI_ANIMAL_ID_SETTING.value, option=ConfigKey.MULTI_ANIMAL_IDS.value, value=','.join(class_names))
            pivoted = yolo_df.pivot(index=FRAME, columns=CLASS_NAME)
            pivoted.columns = [f"{cls}_{col}" for col, cls in pivoted.columns]
            out_df = pivoted.reset_index(drop=True)
            col_order = []
            for class_name in class_names:
                class_cols = [x for x in out_df.columns if class_name in x]
                col_order.extend((class_cols))
            out_df = out_df[col_order].reset_index(drop=True)
            out_df.columns = [s[:-1] + s[-1].lower() if s else s for s in list(out_df.columns)]
            data_save_path = os.path.join(self.outlier_corrected_dir, f'{video_name}.csv')
            write_df(df=out_df, file_type='csv', save_path=data_save_path)
            if self.interpolation_settings is not None:
                interpolator = Interpolate(config_path=self.config_path, data_path=data_save_path, type=self.interpolation_type, method=self.interpolation_method, multi_index_df_headers=False, copy_originals=False)
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(config_path=self.config_path, data_path=data_save_path, time_window=self.smoothing_time, method=self.smoothing_method, multi_index_df_headers=False, copy_originals=False)
                smoother.run()
            if hasattr(self, 'video_info_df') and self.add_to_video_info:
                self.video_info_df = self.video_info_df[self.video_info_df['Video'] != video_name].reset_index(drop=True)
                self.video_info_df.loc[len(self.video_info_df)] = [video_name, self.fps, self.resolution[1], self.resolution[1], 927.927, self.px_per_mm]
            video_timer.stop_timer()
            print(f'Imported video {video_name} ({video_counter+1}/{len(list(self.data_paths.keys()))}) (elapsed time: {video_timer.elapsed_time_str}s)...')
        if hasattr(self, 'video_info_df'):
            self.video_info_df = self.video_info_df.set_index("Video")
            try:
                self.video_info_df.to_csv(self.video_info_path)
            except PermissionError:
                raise PermissionError(msg=f"SimBA tried to write to {self.video_info_path}, but was not allowed. If this file is open in another program, try closing it.", source=self.__class__.__name__)
        self.timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'{len(list(self.data_paths.keys()))} data file(s) imported to SimBA project in directory {self.outlier_corrected_dir}', elapsed_time=self.timer.elapsed_time_str)

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Create a SimBA project from CLI.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory with YOLO data.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA project config.')
    parser.add_argument('--verbose', action='store_true', help='Process verbosity.')
    parser.add_argument('--px_per_mm', type=float, default=1.12, help='Pixels per millimeter for all videos imported')
    parser.add_argument('--fps', type=float, default=30.0, help='FPS of all videos imported')
    args = parser.parse_args()

    importer = SimBAYoloImporter(data_dir=args.data_dir,
                                 config_path=args.config_path,
                                 verbose=args.verbose,
                                 px_per_mm=args.px_per_mm,
                                 fps=args.fps)
    importer.run()




#
# importer = SimBAYoloImporter(data_dir=r'E:\maplight_videos\yolo_mdl\mdl\results', config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini", verbose=True, px_per_mm=1.43, fps=30)
# importer.run()