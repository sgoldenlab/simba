__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.labelling.extract_labelled_frames import AnnotationFrameExtractor
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Dtypes, Formats
from simba.utils.errors import (InvalidInputError, NoChoosenClassifierError,
                                NoDataError, NoFilesFoundError)
from simba.utils.read_write import find_video_of_file, get_fn_ext, str_2_bool

DOWNSAMPLE_OPTION = ["None", "2x", "3x", "4x", "5x"]
IMG_FORMAT_OPTIONS = ['png', 'jpg', 'webp']
ALL_VIDEOS = 'ALL VIDEOS'

class ExtractAnnotationFramesPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> ExtractAnnotationFramesPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        if len(self.target_file_paths) == 0:
            raise NoFilesFoundError(msg=f'Cannot extract annotation images: No data files found in {self.targets_folder}.')
        if len(self.clf_names) == 0:
            raise NoFilesFoundError(msg=f'The SimBA project {config_path} does not have any defined classifier names.')
        self.video_dict = {}
        for file_path in self.target_file_paths:
            self.video_dict[get_fn_ext(filepath=file_path)[1]] = file_path
        PopUpMixin.__init__(self, config_path=config_path, title="EXTRACT ANNOTATED FRAMES", icon='frames')
        video_options = [ALL_VIDEOS] + list(self.video_dict.keys())
        self.clf_frame = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE CLASSIFIERS", icon_name='forest')
        self.clf_frame.grid(row=0, column=0, sticky=NW)
        self.classifiers_cbs = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.classifiers_cbs[clf_name] = SimbaCheckbox(parent=self.clf_frame, txt=clf_name, val=True)
            self.classifiers_cbs[clf_name][0].grid(row=clf_cnt, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='style')
        self.resolution_downsample_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=DOWNSAMPLE_OPTION, label="DOWN-SAMPLE IMAGES:", label_width=35, dropdown_width=25, value=DOWNSAMPLE_OPTION[0], img='minus')
        self.img_format_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=IMG_FORMAT_OPTIONS, label="IMAGE FORMAT:", label_width=35, dropdown_width=25, value=IMG_FORMAT_OPTIONS[0], img='file_type')
        self.greyscale_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE IMAGES:", label_width=35, dropdown_width=25, value='FALSE', img='grey')

        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.resolution_downsample_dropdown.grid(row=0, column=0, sticky=NW)
        self.img_format_dropdown.grid(row=1, column=0, sticky=NW)
        self.greyscale_dropdown.grid(row=2, column=0, sticky=NW)

        self.choose_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE VIDEOS", icon_name='video')
        self.video_dropdown = SimBADropDown(parent=self.choose_video_frm, dropdown_options=video_options, label="VIDEO:", label_width=35, dropdown_width=25, value=ALL_VIDEOS, img='video_2')

        self.choose_video_frm.grid(row=2, column=0, sticky=NW)
        self.video_dropdown.grid(row=0, column=0, sticky=NW)

        self.run_btn = SimbaButton(parent=self.main_frm, txt="RUN", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run)
        self.run_btn.grid(row=self.children_cnt_main()+3, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self):
        downsample_value = self.resolution_downsample_dropdown.getChoices()
        downsample = None if downsample_value == 'None' else int(''.join(c for c in downsample_value if c.isdigit()))
        greyscale = str_2_bool(input_str=self.greyscale_dropdown.getChoices())
        img_format = self.img_format_dropdown.getChoices()
        clfs = []
        for clf_name, selections in self.classifiers_cbs.items():
            if selections[1].get():
                clfs.append(clf_name)
        if len(clfs) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)
        video_selection = self.video_dropdown.getChoices()
        if video_selection == ALL_VIDEOS:
            data_paths = list(self.video_dict.values())
        else:
            data_paths = [self.video_dict[video_selection]]
        for data_path in data_paths:
            _ = find_video_of_file(video_dir=self.video_dir, filename=get_fn_ext(filepath=data_path)[1], raise_error=True)
        frame_extractor = AnnotationFrameExtractor(config_path=self.config_path,
                                                   clfs=clfs,
                                                   downsample=downsample,
                                                   img_format=img_format,
                                                   greyscale=greyscale,
                                                   data_paths=data_paths)
        frame_extractor.run()



#ExtractAnnotationFramesPopUp(config_path=r"C:\troubleshooting\multi_animal_dlc_two_c57\project_folder\project_config.ini")