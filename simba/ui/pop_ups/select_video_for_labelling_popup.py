import os
from tkinter import filedialog
from typing import Union

from simba.labelling.standard_labeller import LabellingInterface
from simba.utils.checks import (check_file_exist_and_readable,
                                check_valid_boolean)
from simba.utils.enums import ConfigKey, Dtypes, Options
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import (get_video_meta_data, read_config_entry,
                                    read_config_file)


class SelectLabellingVideoPupUp():

    """
    Launch PopUp to select video for labelling

    :example:
    >>> SelectLabellingVideoPupUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", continuing=False)
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 continuing: bool = False):

        check_file_exist_and_readable(file_path=config_path)
        check_valid_boolean(value=[continuing], source=f'{self.__class__.__name__} config_path', raise_error=True)
        if not continuing: win_title = "SELECT VIDEO FILE - NEW VIDEO ANNOTATION"
        else: win_title = "SELECT VIDEO FILE - CONTINUING VIDEO ANNOTATION"
        config = read_config_file(config_path=config_path)
        project_path = read_config_entry(config, ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value, data_type=Dtypes.STR.value)
        file_type = read_config_entry(config, ConfigKey.GENERAL_SETTINGS.value, ConfigKey.FILE_TYPE.value, data_type=Dtypes.STR.value)
        video_dir = os.path.join(project_path, 'videos')
        targets_inserted_dir = os.path.join(project_path, 'csv', 'targets_inserted')
        video_dir = None if not os.path.isdir(video_dir) else video_dir
        video_file_path = filedialog.askopenfilename(filetypes=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], initialdir=video_dir, title=win_title)
        check_file_exist_and_readable(file_path=video_file_path)
        video_meta = get_video_meta_data(video_file_path)
        print(f"INITIATING ANNOTATION INTERFACE FOR {video_meta['video_name']} \n VIDEO INFO: {video_meta}")

        if continuing:
            self.targets_inserted_file_path = os.path.join(targets_inserted_dir, f"{video_meta['video_name']}.{file_type}")
            if not os.path.isfile(self.targets_inserted_file_path):
                raise NoFilesFoundError( msg=f'When continuing annotations, SimBA expects a file at {self.targets_inserted_file_path}. SimBA could not find this file.', source=self.__class__.__name__)

        _ = LabellingInterface(config_path=config_path,
                               file_path=video_file_path,
                               thresholds=None,
                               continuing=continuing)



#SelectLabellingVideoPupUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", continuing=False)