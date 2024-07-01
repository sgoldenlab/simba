import os
from tkinter import *
from tkinter import ttk
from typing import Optional, Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FileSelect, FolderSelect
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance, check_int)
from simba.utils.enums import Formats, Options
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import (copy_multiple_videos_to_project,
                                    copy_single_video_to_project)


class ImportVideosFrame(PopUpMixin, ConfigReader):

    """
    .. image:: _static/img/ImportVideosFrame.webp
       :width: 500
       :align: center

    :param Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] parent_frm: Parent frame to insert the Import Videos frame into. If None, one is created.
    :param Optional[Union[str, os.PathLike]] config_path:
    :param Optional[int] idx_row: The row in parent_frm to insert the Videos frame into. Default: 0.
    :param Optional[int] idx_column: The column in parent_frm to insert the Videos frame into. Default: 0.

    :example:
    >>> ImportVideosFrame(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    """

    def __init__(self,
                 parent_frm: Optional[Union[Frame, Canvas, LabelFrame, ttk.Frame]] = None,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 idx_row: Optional[int] = 0,
                 idx_column: Optional[int] = 0):

        if parent_frm is None and config_path is None:
            raise InvalidInputError(msg='If parent_frm is None, please pass config_path', source=self.__class__.__name__)

        elif parent_frm is None and config_path is not None:
            PopUpMixin.__init__(self, config_path=config_path, title='IMPORT VIDEO FILES')
            parent_frm = self.main_frm

        check_instance(source=f'{ImportVideosFrame} parent_frm', accepted_types=(Frame, Canvas, LabelFrame, ttk.Frame), instance=parent_frm)
        check_int(name=f'{ImportVideosFrame} idx_row', value=idx_row, min_value=0)
        check_int(name=f'{ImportVideosFrame} idx_column', value=idx_column, min_value=0)

        import_videos_frm = LabelFrame(parent_frm, text="IMPORT VIDEOS", fg="black", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        if config_path is None:
            Label(import_videos_frm, text="Please CREATE PROJECT CONFIG before importing VIDEOS \n").grid(row=0, column=0, sticky=NW)
            import_videos_frm.grid(row=0, column=0, sticky=NW)
        else:
            ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
            import_multiple_videos_frm = LabelFrame(import_videos_frm, text="IMPORT MULTIPLE VIDEOS")
            self.video_directory_select = FolderSelect(import_multiple_videos_frm, "VIDEO DIRECTORY: ", lblwidth=25)
            self.video_type = DropDownMenu(import_multiple_videos_frm, "VIDEO FILE FORMAT: ", Options.VIDEO_FORMAT_OPTIONS.value, "25")
            self.video_type.setChoices(Options.VIDEO_FORMAT_OPTIONS.value[0])
            import_multiple_btn = Button(import_multiple_videos_frm, text="Import MULTIPLE videos", fg="blue", command=lambda: self.__run_video_import(multiple_videos=True))
            self.multiple_videos_symlink_var = BooleanVar(value=False)
            multiple_videos_symlink_cb = Checkbutton(import_multiple_videos_frm, text="Import SYMLINKS", variable=self.multiple_videos_symlink_var)

            import_single_frm = LabelFrame(import_videos_frm, text="IMPORT SINGLE VIDEO", pady=5, padx=5)
            self.video_file_select = FileSelect(import_single_frm, "VIDEO PATH: ", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
            import_single_btn = Button(import_single_frm, text="Import SINGLE video", fg="blue", command=lambda: self.__run_video_import(multiple_videos=False))
            self.single_video_symlink_var = BooleanVar(value=False)
            single_video_symlink_cb = Checkbutton(import_single_frm, text="Import SYMLINK", variable=self.single_video_symlink_var)

            import_videos_frm.grid(row=0, column=0, sticky=NW)
            import_multiple_videos_frm.grid(row=0, sticky=W)
            self.video_directory_select.grid(row=1, sticky=W)
            self.video_type.grid(row=2, sticky=W)
            multiple_videos_symlink_cb.grid(row=3, sticky=W)
            import_multiple_btn.grid(row=4, sticky=W)

            import_single_frm.grid(row=1, column=0, sticky=NW)
            self.video_file_select.grid(row=0, sticky=W)
            single_video_symlink_cb.grid(row=1, sticky=W)
            import_single_btn.grid(row=2, sticky=W)
            import_videos_frm.grid(row=idx_row, column=idx_column, sticky=NW)

        #parent_frm.mainloop()

    def __run_video_import(self, multiple_videos: bool):
        if multiple_videos:
            check_if_dir_exists(in_dir=self.video_directory_select.folder_path)
            copy_multiple_videos_to_project(config_path=self.config_path,
                                            source=self.video_directory_select.folder_path,
                                            symlink=self.multiple_videos_symlink_var.get(),
                                            file_type=self.video_type.getChoices())

        else:
            check_file_exist_and_readable(file_path=self.video_file_select.file_path)
            copy_single_video_to_project(simba_ini_path=self.config_path,
                                         symlink=self.single_video_symlink_var.get(),
                                         source_path=self.video_file_select.file_path)