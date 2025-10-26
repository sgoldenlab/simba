import argparse
import os
import sys
from typing import Optional, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_tuple)
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
    Import YOLO pose estimation results into SimBA project format.


    .. seealso::
       YOLO pose data can be created with :func:`simba.model.yolo_pose_inference.YOLOPoseInference`.

    :param config_path: Path to SimBA project config file.
    :param data_dir: Directory containing YOLO results CSV files.
    :param verbose: If True, prints progress information.
    :param px_per_mm: Pixels per millimeter for the video. If provided, updates project video info.
    :param resolution: Video resolution as tuple (width, height). Defaults to (927, 927).
    :param fps: Video frames per second. Defaults to 927.

    :example:
    >>> importer = SimBAYoloImporter(data_dir=r'E:\maplight_videos\yolo_mdl\mdl\results', config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini", verbose=True, px_per_mm=1.43, fps=30)
    >>> importer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike],
                 verbose: bool = False,
                 px_per_mm: Optional[float] = None,
                 resolution: Optional[tuple] = None, #WxH
                 fps: Optional[Union[float]] = None):

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
        read_video_info = True if px_per_mm is not None else False
        check_if_dir_exists(in_dir=data_dir, source=f'{self.__class__.__name__} data_dir')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=read_video_info, create_logger=False)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions='.csv', as_dict=True, raise_error=True)
        self.verbose, self.px_per_mm, self.resolution, self.fps = verbose, px_per_mm, resolution, fps

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
            pivoted = yolo_df.pivot(index=FRAME, columns=CLASS_NAME)
            pivoted.columns = [f"{cls}_{col}" for col, cls in pivoted.columns]
            out_df = pivoted.reset_index(drop=True)
            col_order = []
            for class_name in class_names:
                class_cols = [x for x in out_df.columns if class_name in x]
                col_order.extend((class_cols))
            out_df = out_df[col_order].reset_index(drop=True)
            out_df.columns = [s[:-1] + s[-1].lower() if s else s for s in list(out_df.columns)]
            write_df(df=out_df, file_type='csv', save_path=os.path.join(self.outlier_corrected_dir, f'{video_name}.csv'))
            if hasattr(self, 'video_info_df'):
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





importer = SimBAYoloImporter(data_dir=r'E:\maplight_videos\yolo_mdl\mdl\results', config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini", verbose=True, px_per_mm=1.43, fps=30)
importer.run()