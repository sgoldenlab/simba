__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.single_run_model_validation_video import \
    ValidateModelOneVideo
from simba.plotting.single_run_model_validation_video_mp import \
    ValidateModelOneVideoMultiprocess
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, SimBADropDown
from simba.utils.checks import check_float, check_int
from simba.utils.enums import Links, Options
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.read_write import str_2_bool

AUTO = 'AUTO'
GANTT_FRAME = "Gantt chart: final frame only (slightly faster)"
GANTT_VIDEO = "Gantt chart: video"

FONT_SIZE_OPTIONS = list(range(1, 56, 1))
FONT_SIZE_OPTIONS.insert(0, AUTO)
TEXT_SPACE_OPTIONS = list(range(1, 106, 2))
TEXT_SPACE_OPTIONS.insert(0, AUTO)

OPACITY_OPTIONS = list(np.arange(0.1, 1.1, 0.1))
OPACITY_OPTIONS = [round(x, 1) for x in OPACITY_OPTIONS]
POSE_PALETTE_OPTIONS = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value

class ValidationVideoPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_path: Union[str, os.PathLike],
                 model_path: Union[str, os.PathLike],
                 discrimination_threshold: float,
                 shortest_bout: int):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if not os.path.isfile(feature_path):
            raise NoFilesFoundError(msg=f'Set DATA FEATURE FILE PATH to a valid file before creating validation video: Got {feature_path}.', source=self.__class__.__name__)
        if not os.path.isfile(model_path):
            raise NoFilesFoundError(msg=f'Set MODEL FILE PATH to a valid file before creating validation video: Got {model_path}.', source=self.__class__.__name__)
        if not check_float(name=f'{self.__class__.__name__} discrimination_threshold', min_value=0.0, max_value=1.0, value=discrimination_threshold, raise_error=False):
            raise InvalidInputError(msg=f'Set the DISCRIMINATION THRESHOLD to a value between 0.0 and 1.0 before creating validation video. Got: {discrimination_threshold}.', source=self.__class__.__name__)
        discrimination_threshold = float(discrimination_threshold)
        if not check_int(name=f'{self.__class__.__name__} shortest_bout', value=shortest_bout, min_value=0, raise_error=False):
            raise InvalidInputError(msg=f'Set the MINIMUM BOUT LENGTH (MS) to a value above 1 before creating validation video. Got {shortest_bout}.', source=self.__class__.__name__)
        shortest_bout = int(shortest_bout)
        self.mdl_path, self.feature_path = model_path, feature_path
        PopUpMixin.__init__(self, title="CREATE VALIDATION VIDEO", icon='tick')
        self.discrimination_threshold, self.shortest_bout = float(discrimination_threshold), int(shortest_bout)
        style_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='style', icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value, padx=5, pady=5, relief='solid')
        self.font_size_dropdown = SimBADropDown(parent=style_frm, dropdown_options=FONT_SIZE_OPTIONS, label='FONT SIZE: ', label_width=30, dropdown_width=40, value=AUTO)
        self.space_dropdown = SimBADropDown(parent=style_frm, dropdown_options=TEXT_SPACE_OPTIONS, label='TEXT SPACING: ', label_width=30, dropdown_width=40, value=AUTO)
        self.circle_size_dropdown = SimBADropDown(parent=style_frm, dropdown_options=TEXT_SPACE_OPTIONS, label='CIRCLE SIZE: ', label_width=30, dropdown_width=40, value=AUTO)
        self.text_opacity_dropdown = SimBADropDown(parent=style_frm, dropdown_options=OPACITY_OPTIONS, label='TEXT OPACITY: ', label_width=30, dropdown_width=40, value=0.8)
        self.text_thickness_dropdown = SimBADropDown(parent=style_frm, dropdown_options=FONT_SIZE_OPTIONS, label='TEXT THICKNESS: ', label_width=30, dropdown_width=40, value=2)
        self.bp_palette_dropdown = SimBADropDown(parent=style_frm, dropdown_options=POSE_PALETTE_OPTIONS, label='BODY-PART PALETTE: ', label_width=30, dropdown_width=40, value=POSE_PALETTE_OPTIONS[0])

        style_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
        self.font_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.space_dropdown.grid(row=1, column=0, sticky=NW)
        self.circle_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.text_opacity_dropdown.grid(row=3, column=0, sticky=NW)
        self.text_thickness_dropdown.grid(row=4, column=0, sticky=NW)
        self.bp_palette_dropdown.grid(row=5, column=0, sticky=NW)

        tracking_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TRACKING OPTIONS", icon_name='pose', icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value, padx=5, pady=5, relief='solid')
        self.show_pose_dropdown = SimBADropDown(parent=tracking_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW POSE: ', label_width=30, dropdown_width=40, value='TRUE')
        self.show_animal_names_dropdown = SimBADropDown(parent=tracking_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW ANIMAL NAMES: ', label_width=30, dropdown_width=40, value='FALSE')
        self.core_cnt_dropdown = SimBADropDown(parent=tracking_frm, dropdown_options=list(range(1, self.cpu_cnt + 1)), label='CPU COUNT: ', label_width=30, dropdown_width=40, value=int(self.cpu_cnt/2))
        self.show_bbox_dropdown = SimBADropDown(parent=tracking_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW BOUNDING BOX: ', label_width=30, dropdown_width=40, value='FALSE')
        self.show_clf_conf_dropdown = SimBADropDown(parent=tracking_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW CONFIDENCE: ', label_width=30, dropdown_width=40, value='FALSE')

        tracking_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        self.show_pose_dropdown.grid(row=0, column=0, sticky=NW)
        self.show_animal_names_dropdown.grid(row=1, column=0, sticky=NW)
        self.show_bbox_dropdown.grid(row=2, column=0, sticky=NW)
        self.show_clf_conf_dropdown.grid(row=3, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=4, column=0, sticky=NW)

        gantt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="GANTT SETTINGS", icon_name='gantt_small', icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value, padx=5, pady=5, relief='solid')
        self.gantt_dropdown = SimBADropDown(parent=gantt_frm, dropdown_options= Options.GANTT_VALIDATION_OPTIONS.value, label='GANTT TYPE:', label_width=30, dropdown_width=40, value=Options.GANTT_VALIDATION_OPTIONS.value[2])
        gantt_frm.grid(row=3, column=0, sticky=NW, padx=10, pady=10)
        self.gantt_dropdown.grid(row=0, column=0, sticky=NW)


        self.create_run_frm(run_function=self.__run)
        self.main_frm.mainloop()

    def __run(self):
        show_pose = str_2_bool(self.show_pose_dropdown.getChoices())
        show_animal_names = str_2_bool(self.show_animal_names_dropdown.getChoices())
        font_size = None if self.font_size_dropdown.getChoices() == AUTO else int(self.font_size_dropdown.getChoices())
        circle_size = None if self.circle_size_dropdown.getChoices() == AUTO else int(self.circle_size_dropdown.getChoices())
        text_space_scale = None if self.space_dropdown.getChoices() == AUTO else int(self.space_dropdown.getChoices())
        text_thickness = int(self.text_thickness_dropdown.getChoices())
        text_opacity = float(self.text_opacity_dropdown.getChoices())
        core_cnt = int(self.core_cnt_dropdown.getChoices())
        bp_palette = self.bp_palette_dropdown.getChoices()
        bbox = str_2_bool(self.show_bbox_dropdown.getChoices())
        clf_conf = str_2_bool(self.show_clf_conf_dropdown.getChoices())
        create_gantt = self.gantt_dropdown.getChoices()
        if create_gantt.strip() == GANTT_FRAME: create_gantt = 1
        elif create_gantt.strip() == GANTT_VIDEO: create_gantt = 2
        else: create_gantt = None

        if core_cnt == 1:
            validation_video_creator = ValidateModelOneVideo(config_path=self.config_path,
                                                             feature_path=self.feature_path,
                                                             model_path=self.mdl_path,
                                                             discrimination_threshold=self.discrimination_threshold,
                                                             shortest_bout=self.shortest_bout,
                                                             font_size=font_size,
                                                             create_gantt=create_gantt,
                                                             circle_size=circle_size,
                                                             text_spacing=text_space_scale,
                                                             show_pose=show_pose,
                                                             text_thickness=text_thickness,
                                                             text_opacity=text_opacity,
                                                             show_animal_names=show_animal_names,
                                                             show_clf_confidence=clf_conf)


        else:
            validation_video_creator = ValidateModelOneVideoMultiprocess(config_path=self.config_path,
                                                                         feature_path=self.feature_path,
                                                                         model_path=self.mdl_path,
                                                                         discrimination_threshold=self.discrimination_threshold,
                                                                         shortest_bout=self.shortest_bout,
                                                                         font_size=font_size,
                                                                         create_gantt=create_gantt,
                                                                         show_animal_bounding_boxes=bbox,
                                                                         circle_size=circle_size,
                                                                         text_spacing=text_space_scale,
                                                                         show_pose=show_pose,
                                                                         text_thickness=text_thickness,
                                                                         text_opacity=text_opacity,
                                                                         bp_palette=bp_palette,
                                                                         show_animal_names=show_animal_names,
                                                                         core_cnt=core_cnt,
                                                                         show_clf_confidence=clf_conf)

        threading.Thread(target=validation_video_creator.run).start()
        #self.root.destroy()





# ValidationVideoPopUp(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
#                      feature_path=r"D:\troubleshooting\mitra\project_folder\csv\features_extracted\592_MA147_CNO1_0515.csv",
#                      model_path=r"D:\troubleshooting\mitra\models\grooming_undersample_2_2000\grooming.sav",
#                      discrimination_threshold=0.4,
#                      shortest_bout=500)