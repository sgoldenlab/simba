__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ROI_feature_visualizer_mp import ROIfeatureVisualizerMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import check_float, check_nvidea_gpu_available
from simba.utils.enums import Formats, Links
from simba.utils.errors import NoFilesFoundError, NoROIDataError
from simba.utils.read_write import find_all_videos_in_directory, str_2_bool

class VisualizeROIFeaturesPopUp(PopUpMixin, ConfigReader, FeatureExtractionMixin):

    """
    :example:
    >>> _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini')
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.video_file_paths = find_all_videos_in_directory(directory=self.video_dir, as_dict=True)
        self.video_list = [k for k in self.video_file_paths.keys()]
        if len(self.video_list) == 0:
            raise NoFilesFoundError(msg=f"No videos in SimBA project {self.video_dir} directory. Import videos into you SimBA project to visualize ROI features.",source=self.__class__.__name__,)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoFilesFoundError(msg=f"No ROI data found in SimBA project (expected at path {self.roi_coordinates_path}). Draw ROIs before visualize ROI features.", source=self.__class__.__name__, )
        self.read_roi_data()
        self.video_names_w_rois = list(self.video_names_w_rois)
        self.video_file_paths = {k:v for k, v in self.video_file_paths.items() if k in self.video_names_w_rois}
        self.video_list = [k for k in self.video_file_paths.keys()]
        if len(self.video_list) == 0:
            raise NoROIDataError(msg=f'None of the imported videos in the project has ROI data associated with them. Draw ROIs on the videos in the project before visualize ROI features')
        self.max_video_name_len = len(max(self.video_list, key=len))
        self.directing_viable = NORMAL if self.check_directionality_viable()[0] else DISABLED
        self.gpu_available = NORMAL if check_nvidea_gpu_available() else DISABLED
        PopUpMixin.__init__(self, title="VISUALIZE ROI FEATURES", icon='shapes_small')
        FeatureExtractionMixin.__init__(self, config_path=self.config_path)

        self.probability_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="PROBABILITY THRESHOLD", icon_name='probability', icon_link=Links.ROI_FEATURES_PLOT.value)
        threshold_label = SimBALabel(parent=self.probability_frm, txt="NOTE: body-part locations detected with probabilities \n below this threshold are filtered.", font=Formats.FONT_REGULAR_ITALICS.value)
        self.threshold_entry_box = Entry_Box(self.probability_frm, "PROBABILITY THRESHOLD:", "25", value=0.0, entry_box_width=15)

        self.probability_frm.grid(row=0, column=0, sticky=NW)
        threshold_label.grid(row=0, column=0, sticky=NW)
        self.threshold_entry_box.grid(row=1, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.ROI_FEATURES_PLOT.value)
        self.show_directionality_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['FALSE', 'LINES', 'FUNNEL'],  label='SHOW DIRECTIONALITY:', label_width=25, dropdown_width=15, value='FALSE', state=self.directing_viable, img='direction_3')
        self.border_clr_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(self.colors_dict.keys()), label='BORDER COLOR:', label_width=25, dropdown_width=15, value='Black', img='fill', tooltip_key='BORDER_BG_COLOR')
        self.show_pose_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW POSE:', label_width=25, dropdown_width=15, value='TRUE', img='pose')
        self.show_roi_centers_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW ROI CENTERS:', label_width=25, dropdown_width=15, value='TRUE', img='bullseye')
        self.show_roi_eartags_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW ROI EAR TAGS:', label_width=25, dropdown_width=15, value='TRUE', img='ear_small')
        self.show_animal_names_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW ANIMAL NAMES:', label_width=25, dropdown_width=15, value='FALSE', img='id_card')
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='USE GPU:', label_width=25, dropdown_width=15, value='FALSE', state=self.gpu_available, img='gpu_3')
        self.cpu_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(2, self.cpu_cnt+1)), label='CPU CORES: ', label_width=25, dropdown_width=15, value=int(self.cpu_cnt/2), img='cpu_small')
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.show_directionality_dropdown.grid(row=0, column=0, sticky=NW)
        self.border_clr_dropdown.grid(row=1, column=0, sticky=NW)
        self.show_pose_dropdown.grid(row=2, column=0, sticky=NW)
        self.show_roi_centers_dropdown.grid(row=3, column=0, sticky=NW)
        self.show_roi_eartags_dropdown.grid(row=4, column=0, sticky=NW)
        self.show_animal_names_dropdown.grid(row=5, column=0, sticky=NW)
        self.cpu_cnt_dropdown.grid(row=6, column=0, sticky=NW)

        self.body_parts_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.ROI_FEATURES_PLOT.value)
        self.animal_cnt_dropdown = SimBADropDown(parent=self.body_parts_frm, dropdown_options=list(range(1, self.animal_cnt + 1)), label='NUMBER OF ANIMALS:', label_width=25, dropdown_width=15, value=1, command=self.__populate_bp_dropdown)
        self.__populate_bp_dropdown(bp_cnt=1)
        self.body_parts_frm.grid(row=2, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZE SINGLE VIDEO", icon_name='video', icon_link=Links.ROI_FEATURES_PLOT.value)
        self.single_video_dropdown = SimBADropDown(parent=self.single_video_frm, dropdown_options=self.video_list, label='SELECT VIDEO:', label_width=25, dropdown_width=self.max_video_name_len+10, value=self.video_list[0])
        self.single_video_btn = SimbaButton(parent=self.single_video_frm, txt="VISUALIZE ROI FEATURES: SINGLE VIDEO", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': False}, width=240, txt_clr='blue')

        self.single_video_frm.grid(row=3, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.single_video_btn.grid(row=1, column=0, sticky=NW)

        self.all_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZE ALL VIDEOS", icon_name='video', icon_link=Links.ROI_FEATURES_PLOT.value)
        self.all_videos_btn = SimbaButton(parent=self.all_videos_frm, txt="VISUALIZE ROI FEATURES: ALL VIDEOS", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': True}, width=240, txt_clr='red')

        self.all_videos_frm.grid(row=4, column=0, sticky=NW)
        self.all_videos_btn.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()


    def __populate_bp_dropdown(self, bp_cnt: int):
        if hasattr(self, "bp_dropdown_dict"):
            if len(self.bp_dropdown_dict.keys()) != bp_cnt:
                for k, v in self.bp_dropdown_dict.items():
                    v.destroy()

        self.bp_dropdown_dict = {}
        for cnt in range(int(self.animal_cnt_dropdown.getChoices())):
            self.bp_dropdown_dict[cnt] = SimBADropDown(parent=self.body_parts_frm, dropdown_options=self.body_parts_lst, label=self.multi_animal_id_list[cnt], label_width=25, dropdown_width=15, value=self.body_parts_lst[cnt])
            self.bp_dropdown_dict[cnt].grid(row=cnt + 1, column=0, sticky=NW)

    def run(self, multiple: bool):
        check_float(name="Body-part probability threshold", value=self.threshold_entry_box.entry_get, min_value=0.0, max_value=1.0)
        direction = None if self.show_directionality_dropdown.get_value() == 'FALSE' else self.show_directionality_dropdown.get_value().lower()
        core_cnt = int(self.cpu_cnt_dropdown.get_value())
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        show_center_tags = str_2_bool(self.show_roi_centers_dropdown.get_value())
        show_ear_tags = str_2_bool(self.show_roi_eartags_dropdown.get_value())
        border_clr = self.colors_dict[self.border_clr_dropdown.getChoices()]
        show_animal_names = str_2_bool(self.show_animal_names_dropdown.get_value())
        show_pose = str_2_bool(self.show_pose_dropdown.get_value())

        if multiple:
            video_paths = [v for k, v in self.video_file_paths.items()]
        else:
            video_paths = [self.video_file_paths[self.single_video_dropdown.getChoices()]]
        body_parts = [v.getChoices() for v in self.bp_dropdown_dict.values()]
        for video_path in video_paths:
            roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path=self.config_path,
                                                                          video_path=video_path,
                                                                          body_parts=body_parts,
                                                                          core_cnt=core_cnt,
                                                                          gpu=gpu,
                                                                          show_roi_centers=show_center_tags,
                                                                          show_animal_names=show_animal_names,
                                                                          show_roi_eartags=show_ear_tags,
                                                                          show_pose=show_pose,
                                                                          border_bg_clr=border_clr,
                                                                          direction=direction)

            threading.Thread(target=roi_feature_visualizer.run()).start()



#_ = VisualizeROIFeaturesPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")


#_ = VisualizeROIFeaturesPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")


#_ = VisualizeROIFeaturesPopUp(config_path=r"C:\troubleshooting\roi_duplicates\project_folder\project_config.ini")


# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
# ROIAnalysisPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')

# _ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
