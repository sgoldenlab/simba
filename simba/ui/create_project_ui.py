__author__ = "Simon Nilsson"

import os
import tkinter.ttk as ttk
from copy import deepcopy
from tkinter import *

import pandas as pd
import PIL.Image
from PIL import ImageTk

import simba
from simba.ui.import_pose_frame import ImportPoseFrame
from simba.ui.import_videos_frame import ImportVideosFrame
from simba.ui.pop_ups.clf_add_remove_print_pop_up import PoseResetterPopUp
from simba.ui.pop_ups.create_user_defined_pose_configuration_pop_up import \
    CreateUserDefinedPoseConfigurationPopUp
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown, hxtScrollbar)
from simba.utils.checks import check_if_dir_exists, check_int, check_str
from simba.utils.config_creator import ProjectConfigCreator
from simba.utils.enums import Formats, Keys, Links, Methods, Options, Paths
from simba.utils.errors import DuplicationError, MissingProjectConfigEntryError
from simba.utils.lookups import (get_body_part_configurations,
                                 get_bp_config_codes, get_icons_paths)
from simba.video_processors.video_processing import \
    extract_frames_from_all_videos_in_directory


class ProjectCreatorPopUp():
    """
    Mixin for GUI pop-up windows that accept user-inputs for creating a SimBA project.

    .. image:: _static/img/ProjectCreatorPopUp.webp
       :width: 800
       :align: center

    :example:
    >>> ProjectCreatorPopUp()
    """

    def __init__(self):

        self.main_frm = Toplevel()
        self.main_frm.minsize(750, 800)
        self.main_frm.wm_title("PROJECT CONFIGURATION")
        self.main_frm.columnconfigure(0, weight=1)
        self.main_frm.rowconfigure(0, weight=1)
        parent_tab = ttk.Notebook(hxtScrollbar(self.main_frm))
        self.btn_icons = get_icons_paths()
        for k in self.btn_icons.keys():
            self.btn_icons[k]["img"] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname("__file__"), self.btn_icons[k]["icon_path"])))
        self.main_frm.iconphoto(False, self.btn_icons['settings']['img'])

        self.create_project_tab = ttk.Frame(parent_tab)
        self.import_videos_tab = ttk.Frame(parent_tab)
        self.import_data_tab = ttk.Frame(parent_tab)
        #.extract_frms_tab = ttk.Frame(parent_tab)

        parent_tab.add(self.create_project_tab, text=f'{"[ Create project config ]": ^20s}', compound="left", image=self.btn_icons["create"]["img"])
        parent_tab.add(self.import_videos_tab, text=f'{"[ Import videos ]": ^20s}', compound="left", image=self.btn_icons["video"]["img"])
        parent_tab.add(self.import_data_tab, text=f'{"[ Import tracking data ]": ^20s}', compound="left", image=self.btn_icons["pose"]["img"])
        #parent_tab.add( self.extract_frms_tab, text=f'{"[ Extract frames ]": ^20s}', compound="left", image=self.btn_icons["frames"]["img"])
        parent_tab.grid(row=0, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.create_project_tab, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CREATE_PROJECT.value)
        self.general_settings_frm = CreateLabelFrameWithIcon(parent=self.settings_frm, header="GENERAL PROJECT SETTINGS", icon_name='settings', icon_link=Links.CREATE_PROJECT.value, padx=5, pady=5, relief='solid')
        self.project_dir_select = FolderSelect(self.general_settings_frm, "PROJECT DIRECTORY:", lblwidth=35, entry_width=35, font=Formats.FONT_REGULAR.value)
        self.project_name_eb = Entry_Box(self.general_settings_frm, "PROJECT NAME:", labelwidth=35, entry_box_width=35)
        self.file_type_dropdown = SimBADropDown(parent=self.general_settings_frm, dropdown_options=Options.WORKFLOW_FILE_TYPE_OPTIONS.value, label='WORKFLOW FILE TYPE:', label_width=35, dropdown_width=35, value=Options.WORKFLOW_FILE_TYPE_OPTIONS.value[0])

        self.clf_entry_boxes = []
        self.ml_settings_frm = CreateLabelFrameWithIcon(parent=self.create_project_tab, header="MACHINE LEARNING SETTINGS", icon_name='forest', icon_link=Links.CREATE_PROJECT.value, font=Formats.FONT_HEADER.value, padx=5, pady=5, relief='solid')
        self.clf_cnt_dropdown = SimBADropDown(parent=self.ml_settings_frm, dropdown_options=list(range(1, 26)), label='NUMBER OF CLASSIFIERS (BEHAVIORS)', label_width=35, dropdown_width=35, value=1, command=self.__create_entry_boxes)
        self.__create_entry_boxes(cnt=1)
        self.animal_settings_frm = CreateLabelFrameWithIcon(parent=self.create_project_tab, header="ANIMAL SETTINGS", icon_name='pose', icon_link=Links.CREATE_PROJECT.value, font=Formats.FONT_HEADER.value, padx=5, pady=5, relief='solid')
        self.tracking_type_dropdown = SimBADropDown(parent=self.animal_settings_frm, dropdown_options=Options.TRACKING_TYPE_OPTIONS.value, label='TYPE OF TRACKING', label_width=35, dropdown_width=35, value=Options.TRACKING_TYPE_OPTIONS.value[0], command=self.update_body_part_dropdown)

        project_animal_cnt_path = os.path.join(os.path.dirname(simba.__file__), Paths.SIMBA_NO_ANIMALS_PATH.value)
        self.animal_count_lst = list(pd.read_csv(project_animal_cnt_path, header=None)[0])
        self.bp_lu = get_body_part_configurations()
        self.bp_config_codes = get_bp_config_codes()
        self.classical_tracking_options = deepcopy(Options.CLASSICAL_TRACKING_OPTIONS.value)
        self.multi_tracking_options = deepcopy(Options.MULTI_ANIMAL_TRACKING_OPTIONS.value)
        self.three_dim_tracking_options = deepcopy(Options.THREE_DIM_TRACKING_OPTIONS.value)
        self.user_defined_options = [x
                                     for x in list(self.bp_lu.keys())
                                     if x not in self.classical_tracking_options
                                     and x not in self.multi_tracking_options
                                     and x not in self.three_dim_tracking_options]
        for k in self.bp_lu.keys():
            self.bp_lu[k]["img"] = ImageTk.PhotoImage(file=os.path.join(os.path.dirname("__file__"), self.bp_lu[k]["img_path"]))

        self.classical_tracking_option_dict = {k: self.bp_lu[k] for k in self.classical_tracking_options}
        self.multi_tracking_option_dict = {k: self.bp_lu[k] for k in self.multi_tracking_options}
        self.classical_tracking_options.append(Methods.CREATE_POSE_CONFIG.value)
        self.multi_tracking_options.append(Methods.CREATE_POSE_CONFIG.value)
        self.three_dim_tracking_options.append(Methods.CREATE_POSE_CONFIG.value)
        self.classical_tracking_options.extend(self.user_defined_options)
        self.multi_tracking_options.extend(self.user_defined_options)
        self.three_dim_tracking_options.extend(self.user_defined_options)

        self.selected_tracking_dropdown = SimBADropDown(parent=self.animal_settings_frm, dropdown_options=Options.CLASSICAL_TRACKING_OPTIONS.value, label='BODY-PART CONFIGURATION: ', label_width=35, dropdown_width=35, value=self.classical_tracking_options[0], command=self.update_img)
        self.img_lbl = Label(self.animal_settings_frm, image=self.bp_lu[self.classical_tracking_options[0]]["img"], font=Formats.FONT_REGULAR.value)
        reset_btn = SimbaButton(parent=self.animal_settings_frm, txt="RESET USER DEFINED POSE-CONFIGS", txt_clr='red', img='clean', cmd=PoseResetterPopUp)

        run_frm = CreateLabelFrameWithIcon(parent=self.create_project_tab, header="CREATE PROJECT CONFIG", icon_name='create', icon_link=Links.CREATE_PROJECT.value)
        create_project_btn = SimbaButton(parent=run_frm, txt="CREATE PROJECT CONFIG", txt_clr='navy', img='create', font=Formats.FONT_HEADER.value, cmd=self.run)

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.general_settings_frm.grid(row=0, column=0, sticky=NW, pady=5)
        self.project_dir_select.grid(row=0, column=0, sticky=NW)
        self.project_name_eb.grid(row=1, column=0, sticky=NW)
        self.file_type_dropdown.grid(row=2, column=0, sticky=NW)

        self.ml_settings_frm.grid(row=1, column=0, sticky=NW, pady=5)
        self.clf_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.animal_settings_frm.grid(row=2, column=0, sticky=NW, pady=5)
        self.tracking_type_dropdown.grid(row=0, column=0, sticky=NW)
        self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        self.img_lbl.grid(row=2, column=0, sticky=NW)
        reset_btn.grid(row=0, column=1, sticky=NW)
        run_frm.grid(row=3, column=0, sticky=NW)
        create_project_btn.grid(row=0, column=0, sticky=NW)

        ImportVideosFrame(parent_frm=self.import_videos_tab, config_path=None, idx_row=0, idx_column=0)
        ImportPoseFrame(parent_frm=self.import_data_tab, config_path=None, idx_row=0, idx_column=0)
        #extract_frames_frm = LabelFrame(self.extract_frms_tab, text="EXTRACT FRAMES INTO PROJECT", fg="black", font=Formats.FONT_HEADER.value, pady=5, padx=5)
        #extract_frames_note = Label(extract_frames_frm, text="Note: Frame extraction is not needed for any of the parts of the SimBA pipeline.\n Caution: This extract all frames from all videos in project. \n and is computationally expensive if there is a lot of videos at high frame rates/resolution.", font=Formats.FONT_REGULAR.value)
        #extract_frames_btn = SimbaButton(parent=extract_frames_frm, txt="EXTRACT FRAMES", txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=None)
        #extract_frames_frm.grid(row=0, column=0, sticky=NW)
        #extract_frames_note.grid(row=0, column=0, sticky=NW)
        #extract_frames_btn.grid(row=1, column=0, sticky=NW)
        self.update_body_part_dropdown(Methods.CLASSIC_TRACKING.value)
        #self.main_frm.mainloop()

    def __create_entry_boxes(self, cnt):
        existing_values = []
        for entry in self.clf_entry_boxes:
            try:
                existing_values.append(entry.entry_get.strip())
            except:
                existing_values.append("")
        for entry in self.clf_entry_boxes:
            try:
                entry.destroy()
            except:
                pass
        self.clf_entry_boxes = []
        valid_cnt, _ = check_int(name=f'{self.__class__.__name__} cnt', value=cnt, min_value=1, raise_error=False)
        count = int(cnt) if valid_cnt else 1
        for clf_cnt in range(count):
            entry = Entry_Box(parent=self.ml_settings_frm, fileDescription=f'CLASSIFIER NAME {clf_cnt + 1}: ', labelwidth=35, entry_box_width=35)
            entry.grid(row=clf_cnt + 2, column=0, sticky=NW)
            if clf_cnt < len(existing_values) and existing_values[clf_cnt]:
                try:
                    entry.entry_set(existing_values[clf_cnt])
                except AttributeError:
                    try:
                        entry.entPath.insert(0, existing_values[clf_cnt])
                    except:
                        pass
            self.clf_entry_boxes.append(entry)
        # for entry in self.clf_entry_boxes:
        #     entry.destroy()
        # self.clf_entry_boxes = []
        # for clf_cnt in range(int(cnt)):
        #     entry = Entry_Box(parent=self.ml_settings_frm, fileDescription=f'CLASSIFIER NAME {clf_cnt + 1}: ', labelwidth=35, entry_box_width=35)
        #     entry.grid(row=clf_cnt + 2, column=0, sticky=NW)
        #     self.clf_entry_boxes.append(entry)

    def update_body_part_dropdown(self, selected_value):
        self.selected_tracking_dropdown.destroy()
        if selected_value == Methods.MULTI_TRACKING.value:
            self.selected_tracking_dropdown = SimBADropDown(parent=self.animal_settings_frm, dropdown_options=self.multi_tracking_options, label='BODY-PART CONFIGURATION: ', label_width=35, dropdown_width=35, value=self.multi_tracking_options[0], command=self.update_img)
            self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        elif selected_value == Methods.CLASSIC_TRACKING.value:
            self.selected_tracking_dropdown = SimBADropDown(parent=self.animal_settings_frm, dropdown_options=self.classical_tracking_options, label='BODY-PART CONFIGURATION: ', label_width=35, dropdown_width=35, value=self.classical_tracking_options[0], command=self.update_img)
            self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)

        elif selected_value == Methods.THREE_D_TRACKING.value:
            self.selected_tracking_dropdown = SimBADropDown(parent=self.animal_settings_frm, dropdown_options=self.three_dim_tracking_options, label='BODY-PART CONFIGURATION: ', label_width=35, dropdown_width=35, value=self.three_dim_tracking_options[0], command=self.update_img)
            self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        self.update_img(self.selected_tracking_dropdown.getChoices())

    def update_img(self, selected_value):
        if selected_value != Methods.CREATE_POSE_CONFIG.value:
            self.img_lbl.config(image=self.bp_lu[selected_value]["img"])
        else:
            _ = CreateUserDefinedPoseConfigurationPopUp(master=self.main_frm, project_config_class=ProjectCreatorPopUp)

    def extract_frames(self):
        if not hasattr(self, "config_path"):
            raise MissingProjectConfigEntryError(msg="Create PROJECT CONFIG before extracting frames")
        video_dir = os.path.join(os.path.dirname(self.config_path), "videos")
        extract_frames_from_all_videos_in_directory(config_path=self.config_path, directory=video_dir)

    def run(self):
        project_dir = self.project_dir_select.folder_path
        check_if_dir_exists(in_dir=project_dir)
        project_name = self.project_name_eb.entry_get
        check_str(name="PROJECT NAME", value=project_name, allow_blank=False)
        target_list = []
        for number, entry_box in enumerate(self.clf_entry_boxes):
            target_list.append(entry_box.entry_get.strip())
        if len(list(set(target_list))) != len(self.clf_entry_boxes):
            raise DuplicationError(msg="All classifier names have to be unique")
        selected_config = self.selected_tracking_dropdown.getChoices()
        if selected_config in self.bp_config_codes.keys():
            config_code = self.bp_config_codes[selected_config]
        else:
            config_code = Methods.USER_DEFINED.value

        config_idx = None
        for cnt, k in enumerate(self.bp_lu.keys()):
            if k == selected_config:
                config_idx = cnt
        animal_cnt = self.animal_count_lst[config_idx]

        config_creator = ProjectConfigCreator(project_path=project_dir,
                                              project_name=project_name,
                                              target_list=target_list,
                                              pose_estimation_bp_cnt=config_code,
                                              body_part_config_idx=config_idx,
                                              animal_cnt=animal_cnt,
                                              file_type=self.file_type_dropdown.getChoices())

        self.config_path = config_creator.config_path
        ImportPoseFrame(parent_frm=self.import_data_tab, idx_row=0, idx_column=0, config_path=self.config_path)
        ImportVideosFrame(parent_frm=self.import_videos_tab, config_path=self.config_path, idx_row=0, idx_column=0)

#ProjectCreatorPopUp()
