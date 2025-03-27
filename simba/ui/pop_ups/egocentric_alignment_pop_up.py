import os
from tkinter import *
from typing import Union

from simba.data_processors.egocentric_aligner import EgocentricalAligner
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FolderSelect, SimbaCheckbox)
from simba.utils.checks import check_if_dir_exists, check_nvidea_gpu_available
from simba.utils.enums import Keys, Links
from simba.utils.errors import InvalidInputError, NoDataError, SimBAGPUError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext)


class EgocentricAlignPopUp(ConfigReader, PopUpMixin):
    """

    :example:
    >>> _ = EgocentricAlignPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        PopUpMixin.__init__(self, title="EGOCENTRIC ALIGN VIDEO AND POSE", icon='egocentric')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        self.clr_dict = get_color_dict()
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.data_dir = FolderSelect(settings_frm, "DATA DIRECTORY:", lblwidth=45, initialdir=self.outlier_corrected_dir)
        self.videos_dir = FolderSelect(settings_frm, "VIDEO DIRECTORY:", lblwidth=45, initialdir=self.video_dir)
        self.save_dir = FolderSelect(settings_frm, "SAVE DIRECTORY:", lblwidth=45, initialdir=self.video_dir)
        self.center_anchor_dropdown = DropDownMenu(settings_frm, "CENTER ANCHOR:", self.body_parts_lst, labelwidth=45)
        self.direction_anchor_dropdown = DropDownMenu(settings_frm, "DIRECTION ANCHOR:", self.body_parts_lst, labelwidth=45)
        self.direction_dropdown = DropDownMenu(settings_frm, "DIRECTION:", list(range(0, 361)), labelwidth=45)
        self.fill_clr_dropdown = DropDownMenu(settings_frm, "ROTATION COLOR:", list(self.clr_dict.keys()), labelwidth=45)
        self.core_cnt_dropdown = DropDownMenu(settings_frm, "CPU COUNT:", list(range(1, self.cpu_cnt + 1)), labelwidth=45)
        self.gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt='USE GPU')

        self.center_anchor_dropdown.setChoices(self.body_parts_lst[0])
        self.direction_anchor_dropdown.setChoices(self.body_parts_lst[1])
        self.direction_dropdown.setChoices(0)
        self.fill_clr_dropdown.setChoices('Black')
        self.core_cnt_dropdown.setChoices(self.cpu_cnt)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir.grid(row=0, column=0, sticky=NW)
        self.videos_dir.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.center_anchor_dropdown.grid(row=3, column=0, sticky=NW)
        self.direction_anchor_dropdown.grid(row=4, column=0, sticky=NW)
        self.direction_dropdown.grid(row=5, column=0, sticky=NW)
        self.fill_clr_dropdown.grid(row=6, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=7, column=0, sticky=NW)
        self.gpu_cb.grid(row=8, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        data_dir, video_dir = self.data_dir.folder_path, self.videos_dir.folder_path
        save_dir = self.save_dir.folder_path
        check_if_dir_exists(in_dir=data_dir);
        check_if_dir_exists(in_dir=video_dir);
        check_if_dir_exists(in_dir=save_dir)
        if (save_dir == data_dir) or (save_dir == video_dir):
            raise InvalidInputError(msg='The save directory cannot be the same as the data/video directories',
                                    source=self.__class__.__name__)
        center_anchor, direction_anchor = self.center_anchor_dropdown.getChoices(), self.direction_anchor_dropdown.getChoices()
        fill_clr, core_cnt, gpu = self.fill_clr_dropdown.getChoices(), self.core_cnt_dropdown.getChoices(), self.gpu_var.get()
        direction = int(self.direction_dropdown.getChoices())
        if gpu and not check_nvidea_gpu_available():
            raise SimBAGPUError(msg='No NVIDEA GPU detected.', source=self.__class__.__name__)
        data_file_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{self.file_type}'],
                                                               raise_error=True)
        data_file_paths = [os.path.join(data_dir, x) for x in data_file_paths]
        data_file_names = [get_fn_ext(filepath=x)[1] for x in data_file_paths]
        video_file_paths = list(
            find_all_videos_in_directory(directory=video_dir, as_dict=True, raise_error=True).values())
        video_file_names = [get_fn_ext(filepath=x)[1] for x in video_file_paths]
        missing_video_files = [x for x in video_file_names if x not in data_file_names]
        if len(missing_video_files) > 0: raise NoDataError(
            msg=f'Some data files are missing a video file: {missing_video_files}', source=self.__class__.__name__)

        aligner = EgocentricalAligner(data_dir=data_dir,
                                      save_dir=save_dir,
                                      anchor_1=center_anchor,
                                      anchor_2=direction_anchor,
                                      direction=direction,
                                      anchor_location=None,
                                      fill_clr=self.clr_dict[fill_clr],
                                      verbose=True,
                                      gpu=gpu,
                                      videos_dir=video_dir)

        aligner.run()
