import os
from typing import Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_valid_dataframe)
from simba.utils.enums import Formats
from simba.utils.errors import FrameRangeError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    read_df, write_df)

CLASSIFIER_NAME = 'ATTACK'
START, END = 'START', 'END'

class SingleClfAppenderExcel(ConfigReader):
    """
    Appends binary behavior labels (ATTACK) from Excel annotations to feature CSV files.
    
    Reads an Excel file containing frame-by-frame annotations where each sheet represents a video
    and contains START and END columns indicating frame ranges of attack behavior. The class appends
    a binary ATTACK column to the corresponding feature CSV files (1 for attack present, 0 for absent).
    Results are stored in the `project_folder/csv/targets_inserted` directory of the SimBA project.

    .. note::
       Custom label appender class inhereting from ``ConfigReader`` which is not used by default in any of the standard SimBA entry-points.
       `Example expected input file <https://github.com/sgoldenlab/simba/blob/master/misc/SingleClfAppenderExcel_example.xlsx>`__.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] label_path: path to Excel file containing behavior annotations. Each sheet name should match a video name, with START and END columns indicating frame ranges.
    :param Union[str, os.PathLike] data_dir: Directory containing feature CSV files. If None, uses the project's features_dir from the config.
    :param Union[str, os.PathLike] save_dir: Directory where labeled CSV files will be saved. If None, uses the project's targets_folder from the config.

    :example:
    >>> labeler = SingleClfAppenderExcel(config_path='MyProjectConfig', label_path='annotations.xlsx')
    >>> labeler.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 label_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike] = None,
                 save_dir: Union[str, os.PathLike] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        data_dir = self.features_dir if data_dir is None else data_dir
        check_if_dir_exists(in_dir=data_dir, raise_error=True)
        check_file_exist_and_readable(file_path=label_path, raise_error=True)
        self.save_dir = self.targets_folder if save_dir is None else save_dir
        check_if_dir_exists(in_dir=self.save_dir, raise_error=True)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions='.csv', raise_error=True, as_dict=True)
        self.label_path, self.data_dir = label_path, data_dir
        self.label_dict = pd.read_excel(label_path, sheet_name=None)


    def run(self):
        frm_lbl_cnt = 0
        for video_cnt, (video_name, video_labels) in enumerate(self.label_dict.items()):
            video_timer = SimbaTimer(start=True)
            save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            print(f'Appending LABELS video {video_name} ({video_cnt + 1}/{len(list(self.label_dict.keys()))})....')
            if video_name not in self.data_paths.keys():
                raise NoFilesFoundError(msg=f'Could not find file for video {video_name} in {self.data_dir}', source=self.__class__.__name__)
            lbls = self.label_dict[video_name]
            lbls = lbls[[START, END]]
            data_df = read_df(file_path=self.data_paths[video_name], file_type='csv').reset_index(drop=True)
            data_df[CLASSIFIER_NAME] = 0
            if len(lbls) > 0:
                check_valid_dataframe(df=lbls, source=f'{self.__class__.__name__} {video_name} {self.label_path}', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=[START, END])
                lbled_frms = sorted(list(np.concatenate([np.arange(s, e + 1) for s, e in zip(lbls[START], lbls[END])])))
                frm_lbl_cnt += len(lbled_frms)
                if max(lbled_frms) > len(data_df):
                    raise FrameRangeError(msg=f'The dataframe {self.data_paths[video_name]} contains {len(data_df)} frames but labels for behavior present exist for frame {max(lbled_frms)}', source=self.__class__.__name__)
                data_df.loc[lbled_frms, CLASSIFIER_NAME] = 1
            write_df(df=data_df, file_type='csv', save_path=save_path)
            video_timer.stop_timer()
            print(f'Saved labels for {save_path} ({video_cnt+1}/{len(self.label_dict.keys())}, elapsed time: {video_timer.elapsed_time_str}s)...')

        self.timer.stop_timer()
        stdout_success(f'{CLASSIFIER_NAME} labels for {len(self.label_dict.keys())} files saved in {self.save_dir} (TOTAL FRAME LABELS: {frm_lbl_cnt})', source=self.__class__.__name__)

# r = SingleClfAppenderExcel(config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini", label_path=r"C:\Users\sroni\Downloads\maplight_aggression.xlsx")
# r.run()

