__author__ = "Simon Nilsson"

import tkinter.ttk as ttk
from tkinter import *
import os
from copy import deepcopy
import pandas as pd
from simba.tkinter_functions import (hxtScrollbar,
                                     FolderSelect,
                                     Entry_Box,
                                     DropDownMenu)
from PIL import ImageTk
import PIL.Image
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.enums import Formats, Options, Methods, Paths
from simba.utils.errors import DuplicationError
from simba.utils.lookups import (get_body_part_configurations,
                                 get_bp_config_codes,
                                 get_icons_paths)
from simba.pop_up_classes import (PoseResetterPopUp,
                                  CreateUserDefinedPoseConfigurationPopUp)
from simba.read_config_unit_tests import (check_if_dir_exists, check_str)
from simba.project_config_creator import ProjectConfigCreator
from simba.misc_tools import extract_frames_from_all_videos_in_directory


class ProjectCreatorPopUp(PopUpMixin):
    """
    Creates tkinter GUI pop-up window accepting user-input for generating SimBA project.
    """

    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(750, 750)
        self.main_frm.wm_title("PROJECT CONFIGURATION")
        parent_tab = ttk.Notebook(hxtScrollbar(self.main_frm))
        self.btn_icons = get_icons_paths()
        for k in self.btn_icons.keys():
            self.btn_icons[k]['img'] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), self.btn_icons[k]['icon_path'])))
        self.create_project_tab = ttk.Frame(parent_tab)
        self.import_videos_tab = ttk.Frame(parent_tab)
        self.import_data_tab = ttk.Frame(parent_tab)
        self.extract_frms_tab = ttk.Frame(parent_tab)

        parent_tab.add(self.create_project_tab, text=f'{"[ Create project config ]": ^20s}', compound='left', image=self.btn_icons['create']['img'])
        parent_tab.add(self.import_videos_tab, text=f'{"[ Import videos ]": ^20s}', compound='left', image=self.btn_icons['video']['img'])
        parent_tab.add(self.import_data_tab, text=f'{"[ Import tracking data ]": ^20s}', compound='left', image=self.btn_icons['pose']['img'])
        parent_tab.add(self.extract_frms_tab, text=f'{"[ Extract frames ]": ^20s}', compound='left', image=self.btn_icons['frames']['img'])

        parent_tab.grid(row=0, column=0, sticky=NW)
        self.settings_frm = LabelFrame(self.create_project_tab, text='SETTINGS', fg='black', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5)
        self.general_settings_frm = LabelFrame(self.settings_frm, text='GENERAL PROJECT SETTINGS',fg='black', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5)

        self.project_dir_select = FolderSelect(self.general_settings_frm, "Project directory:", lblwidth='25')
        self.project_name_eb = Entry_Box(self.general_settings_frm, 'Project name:', labelwidth='25')
        self.file_type_dropdown = DropDownMenu(self.general_settings_frm,'Workflow file type:', Options.WORKFLOW_FILE_TYPE_OPTIONS.value, '25')
        self.file_type_dropdown.setChoices(choice=Options.WORKFLOW_FILE_TYPE_OPTIONS.value[0])


        self.clf_name_entries = []
        self.ml_settings_frm = LabelFrame(self.settings_frm, text='MACHINE LEARNING SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5)
        self.clf_cnt = Entry_Box(self.ml_settings_frm, 'Number of classifiers (behaviors): ', '25', validation='numeric')
        add_clf_btn = Button(self.ml_settings_frm, text='<Add predictive classifier(s)>', fg="blue", command=lambda: self.create_entry_boxes_from_entrybox(count=self.clf_cnt.entry_get, parent=self.ml_settings_frm, current_entries=self.clf_name_entries))

        self.animal_settings_frm = LabelFrame(self.settings_frm, text='ANIMAL SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.tracking_type_dropdown = DropDownMenu(self.animal_settings_frm, 'Type of Tracking', Options.TRACKING_TYPE_OPTIONS.value, '25', com=self.update_body_part_dropdown)
        self.tracking_type_dropdown.setChoices(Options.TRACKING_TYPE_OPTIONS.value[0])

        project_animal_cnt_path = os.path.join(os.path.dirname(__file__), Paths.SIMBA_NO_ANIMALS_PATH.value)
        self.animal_count_lst = list(pd.read_csv(project_animal_cnt_path, header=None)[0])
        self.bp_lu = get_body_part_configurations()
        self.bp_config_codes = get_bp_config_codes()
        self.classical_tracking_options = deepcopy(Options.CLASSICAL_TRACKING_OPTIONS.value)
        self.multi_tracking_options = deepcopy(Options.MULTI_ANIMAL_TRACKING_OPTIONS.value)
        self.three_dim_tracking_options = deepcopy(Options.THREE_DIM_TRACKING_OPTIONS.value)
        self.user_defined_options = [x for x in list(self.bp_lu.keys()) if x not in self.classical_tracking_options and x not in self.multi_tracking_options and x not in self.three_dim_tracking_options]
        for k in self.bp_lu.keys():
            self.bp_lu[k]['img'] = PhotoImage(file=os.path.join(os.path.dirname(__file__), self.bp_lu[k]['img_path']))
        self.classical_tracking_option_dict = {k: self.bp_lu[k] for k in self.classical_tracking_options}
        self.multi_tracking_option_dict = {k: self.bp_lu[k] for k in self.multi_tracking_options}
        self.classical_tracking_options.append(Methods.CREATE_POSE_CONFIG.value)
        self.multi_tracking_options.append(Methods.CREATE_POSE_CONFIG.value)
        self.three_dim_tracking_options.append(Methods.CREATE_POSE_CONFIG.value)
        self.classical_tracking_options.extend(self.user_defined_options)
        self.multi_tracking_options.extend(self.user_defined_options)
        self.three_dim_tracking_options.extend(self.user_defined_options)
        self.selected_tracking_dropdown = DropDownMenu(self.animal_settings_frm, 'Body-part config', Options.CLASSICAL_TRACKING_OPTIONS.value, '25', com=self.update_img)
        self.selected_tracking_dropdown.setChoices(self.classical_tracking_options[0])
        self.img_lbl = Label(self.animal_settings_frm, image=self.bp_lu[self.classical_tracking_options[0]]['img'])
        reset_btn = Button(self.animal_settings_frm, text='RESET USER DEFINED POSE-CONFIGS', fg='red', command=lambda: PoseResetterPopUp())
        run_frm = Frame(master=self.settings_frm)
        create_project_btn = Button(run_frm, text='CREATE PROJECT CONFIG', fg='navy', font=("Helvetica", 16, 'bold'), command= lambda: self.run())
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.general_settings_frm.grid(row=0, column=0, sticky=NW)
        self.project_dir_select.grid(row=0, column=0, sticky=NW)
        self.project_name_eb.grid(row=1, column=0, sticky=NW)
        self.file_type_dropdown.grid(row=2, column=0, sticky=NW)

        self.ml_settings_frm.grid(row=1, column=0, sticky=NW)
        self.clf_cnt.grid(row=0, column=0, sticky=NW)
        add_clf_btn.grid(row=1, column=0, sticky=NW)

        self.animal_settings_frm.grid(row=2, column=0, sticky=NW)
        self.tracking_type_dropdown.grid(row=0, column=0, sticky=NW)
        self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        self.img_lbl.grid(row=2, column=0, sticky=NW)
        reset_btn.grid(row=0, column=1, sticky=NW)
        run_frm.grid(row=3, column=0, sticky=NW)
        create_project_btn.grid(row=0, column=0, sticky=NW)

        self.create_import_videos_menu(parent_frm=self.import_videos_tab)
        self.create_import_pose_menu(parent_frm=self.import_data_tab)

        extract_frames_frm = LabelFrame(self.extract_frms_tab, text='Extract Frames into project folder', fg='black', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        extract_frames_note = Label(extract_frames_frm, text='Note: This is no longer needed for any of the parts of the SimBA pipeline.\n Caution: This extract all frames from all videos in project. \n and is computationally expensive if there is a lot of videos at high frame rates/resolution.')
        extract_frames_btn = Button(extract_frames_frm, text='Extract frames', fg='blue', command=lambda: None)

        extract_frames_frm.grid(row=0, column=0, sticky=NW)
        extract_frames_note.grid(row=0, column=0, sticky=NW)
        extract_frames_btn.grid(row=1, column=0, sticky=NW)
        self.update_body_part_dropdown(Methods.CLASSIC_TRACKING.value)
        self.main_frm.mainloop()

    def update_body_part_dropdown(self, selected_value):
        self.selected_tracking_dropdown.destroy()
        if selected_value == Methods.MULTI_TRACKING.value:
            self.selected_tracking_dropdown = DropDownMenu(self.animal_settings_frm, 'Body-part config', self.multi_tracking_options, '25', com=self.update_img)
            self.selected_tracking_dropdown.setChoices(self.multi_tracking_options[0])
            self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        elif selected_value == Methods.CLASSIC_TRACKING.value:
            self.selected_tracking_dropdown = DropDownMenu(self.animal_settings_frm, 'Body-part config', self.classical_tracking_options, '25', com=self.update_img)
            self.selected_tracking_dropdown.setChoices(self.classical_tracking_options[0])
            self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        elif selected_value == Methods.THREE_D_TRACKING.value:
            self.selected_tracking_dropdown = DropDownMenu(self.animal_settings_frm, 'Body-part config', self.three_dim_tracking_options, '25', com=self.update_img)
            self.selected_tracking_dropdown.setChoices(self.three_dim_tracking_options[0])
            self.selected_tracking_dropdown.grid(row=1, column=0, sticky=NW)
        self.update_img(self.selected_tracking_dropdown.getChoices())

    def update_img(self, selected_value):
        if selected_value != Methods.CREATE_POSE_CONFIG.value:
            self.img_lbl.config(image=self.bp_lu[selected_value]['img'])
        else:
            _ = CreateUserDefinedPoseConfigurationPopUp(master=self.main_frm, project_config_class=ProjectCreatorPopUp)

    def extract_frames(self):
        if not hasattr(self, 'config_path'):
            print('SIMBA ERROR: Create PROJECT CONFIG before extracting frames')
            raise FileNotFoundError('SIMBA ERROR: Create PROJECT CONFIG before extracting frames')
        video_dir = os.path.join(os.path.dirname(self.config_path), 'videos')
        extract_frames_from_all_videos_in_directory(config_path=self.config_path, directory=video_dir)

    def run(self):
        project_dir = self.project_dir_select.folder_path
        check_if_dir_exists(in_dir=project_dir)
        project_name = self.project_name_eb.entry_get
        check_str(name='PROJECT NAME', value=project_name, allow_blank=False)
        target_list = []
        for number, entry_box in enumerate(self.clf_name_entries):
            target_list.append(entry_box.entry_get.strip())
        if len(list(set(target_list))) != len(self.clf_name_entries):
            raise DuplicationError(msg='All classifier names have to be unique')

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
        self.create_import_pose_menu(parent_frm=self.import_data_tab)
        self.create_import_videos_menu(parent_frm=self.import_videos_tab)


# ProjectCreatorPopUp()