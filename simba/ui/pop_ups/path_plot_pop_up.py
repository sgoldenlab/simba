__author__ = "Simon Nilsson"

from tkinter import *
from collections import defaultdict
import os

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import Paths, Keys, Links, Formats
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon, Entry_Box
from simba.utils.read_write import get_file_name_info_in_directory
from simba.plotting.path_plotter import PathPlotterSingleCore
from simba.plotting.path_plotter_mp import PathPlotterMulticore



class PathPlotPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing path plots')

        PopUpMixin.__init__(self, title='CREATE PATH PLOTS')
        self.resolutions.insert(0, 'As input')
        self.animal_cnt_options = list(range(1, self.animal_cnt+1))

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.PATH_PLOTS.value)
        self.autocompute_var = BooleanVar(value=True)
        self.auto_compute_styles = Checkbutton(self.style_settings_frm, text='Auto-compute styles', variable=self.autocompute_var, command=self.enable_style_settings)
        self.resolution_dropdown = DropDownMenu(self.style_settings_frm, 'Resolution:', self.resolutions, '16')
        self.bg_clr_dropdown = DropDownMenu(self.style_settings_frm, 'Background color:', list(self.colors_dict.keys()), '16')

        self.line_width = Entry_Box(self.style_settings_frm, 'Line width: ', '16', validation='numeric')
        self.max_lines_entry = Entry_Box(self.style_settings_frm, 'Max prior lines (ms): ', '16', validation='numeric')
        self.font_size = Entry_Box(self.style_settings_frm, 'Font size: ', '16', validation='numeric')
        self.font_thickness = Entry_Box(self.style_settings_frm, 'Font thickness: ', '16', validation='numeric')
        self.circle_size = Entry_Box(self.style_settings_frm, 'Circle size: ', '16', validation='numeric')
        self.resolution_dropdown.setChoices(self.resolutions[0])
        self.line_width.entry_set(val=6)
        self.bg_clr_dropdown.setChoices(list(self.colors_dict.keys())[0])
        self.circle_size.entry_set(val=20)
        self.font_size.entry_set(val=3)
        self.font_thickness.entry_set(val=2)
        self.max_lines_entry.entry_set(2000)
        self.resolution_dropdown.disable()
        self.line_width.set_state("disable")
        self.bg_clr_dropdown.disable()
        self.circle_size.set_state("disable")
        self.font_size.set_state("disable")
        self.max_lines_entry.set_state("disable")
        self.font_thickness.set_state("disable")

        self.body_parts_frm = LabelFrame(self.main_frm, text='CHOOSE BODY-PARTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.number_of_animals_dropdown = DropDownMenu(self.body_parts_frm, '# Animals:', self.animal_cnt_options, '16', com=self.populate_body_parts_menu)
        self.number_of_animals_dropdown.setChoices(self.animal_cnt_options[0])

        self.clf_frm = LabelFrame(self.main_frm, text='CHOOSE CLASSIFICATION VISUALIZATION', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.include_clf_locations_var = BooleanVar(value=False)
        self.include_clf_locations_cb = Checkbutton(self.clf_frm, text='Include classification locations', variable=self.include_clf_locations_var, command=self.populate_clf_location_data)
        self.include_clf_locations_cb.grid(row=0, sticky=NW)
        self.populate_clf_location_data()


        self.populate_body_parts_menu(self.animal_cnt_options[0])
        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.path_frames_var = BooleanVar()
        self.path_videos_var = BooleanVar()
        self.path_last_frm_var = BooleanVar()
        self.multiprocessing_var = BooleanVar()
        path_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.path_frames_var)
        path_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.path_videos_var)
        path_last_frm_cb = Checkbutton(self.settings_frm, text='Create last frame', variable=self.path_last_frm_var)
        self.multiprocess_cb = Checkbutton(self.settings_frm, text='Multiprocess videos (faster)', variable=self.multiprocessing_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocessing_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command= lambda: self.__create_path_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command= lambda: self.__create_path_plots(multiple_videos=True))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.auto_compute_styles.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=1, sticky=NW)
        self.max_lines_entry.grid(row=2, sticky=NW)
        self.line_width.grid(row=3, sticky=NW)
        self.circle_size.grid(row=4, sticky=NW)
        self.font_size.grid(row=5, sticky=NW)
        self.font_thickness.grid(row=6, sticky=NW)
        self.bg_clr_dropdown.grid(row=7, sticky=NW)

        self.clf_frm.grid(row=1, sticky=NW)

        self.body_parts_frm.grid(row=2, sticky=NW)
        self.number_of_animals_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=3, sticky=NW)
        path_frames_cb.grid(row=0, sticky=NW)
        path_videos_cb.grid(row=1, sticky=NW)
        path_last_frm_cb.grid(row=2, sticky=NW)
        self.multiprocess_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=NW)

        self.run_frm.grid(row=4, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def populate_body_parts_menu(self, choice):
        if hasattr(self, 'bp_dropdowns'):
            for (k,v), (k2,v2) in zip(self.bp_dropdowns.items(), self.bp_colors.items()):
                self.bp_dropdowns[k].destroy()
                self.bp_colors[k].destroy()
        self.bp_dropdowns, self.bp_colors = {}, {}
        for animal_cnt in range(int(self.number_of_animals_dropdown.getChoices())):
            self.bp_dropdowns[animal_cnt] = DropDownMenu(self.body_parts_frm, 'Body-part {}:'.format(str(animal_cnt+1)), self.body_parts_lst, '16')
            self.bp_dropdowns[animal_cnt].setChoices(self.body_parts_lst[animal_cnt])
            self.bp_dropdowns[animal_cnt].grid(row=animal_cnt+1, column=0, sticky=NW)

            self.bp_colors[animal_cnt] = DropDownMenu(self.body_parts_frm, '', list(self.colors_dict.keys()), '2')
            self.bp_colors[animal_cnt].setChoices(list(self.colors_dict.keys())[animal_cnt])
            self.bp_colors[animal_cnt].grid(row=animal_cnt+1, column=1, sticky=NW)

    def populate_clf_location_data(self):
        self.clf_name, self.clf_clr, self.clf_size = {}, {}, {}
        size_lst = list(range(1, 51))
        size_lst = ['Size: ' + str(x) for x in size_lst]
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.clf_name[clf_cnt] = DropDownMenu(self.clf_frm, 'Classifier {}:'.format(str(clf_cnt + 1)), self.clf_names, '16')
            self.clf_name[clf_cnt].setChoices(self.clf_names[clf_cnt])
            self.clf_name[clf_cnt].grid(row=clf_cnt+1, column=0, sticky=NW)

            self.clf_clr[clf_cnt] = DropDownMenu(self.clf_frm, '', list(self.colors_dict.keys()), '2')
            self.clf_clr[clf_cnt].setChoices(list(self.colors_dict.keys())[clf_cnt])
            self.clf_clr[clf_cnt].grid(row=clf_cnt+1, column=1, sticky=NW)

            self.clf_size[clf_cnt] = DropDownMenu(self.clf_frm, '', size_lst, '2')
            self.clf_size[clf_cnt].setChoices(size_lst[15])
            self.clf_size[clf_cnt].grid(row=clf_cnt + 1, column=2, sticky=NW)

        self.enable_clf_location_settings()


    def enable_clf_location_settings(self):
        if self.include_clf_locations_var.get():
            for clf_cnt in self.clf_name.keys():
                self.clf_name[clf_cnt].enable()
                self.clf_clr[clf_cnt].enable()
                self.clf_size[clf_cnt].enable()
        else:
            for clf_cnt in self.clf_name.keys():
                self.clf_name[clf_cnt].disable()
                self.clf_clr[clf_cnt].disable()
                self.clf_size[clf_cnt].disable()

    def enable_style_settings(self):
        if not self.autocompute_var.get():
            self.resolution_dropdown.enable()
            self.max_lines_entry.set_state('normal')
            self.line_width.set_state('normal')
            self.font_thickness.set_state("normal")
            self.circle_size.set_state('normal')
            self.font_size.set_state('normal')
            self.bg_clr_dropdown.enable()
        else:
            self.resolution_dropdown.disable()
            self.max_lines_entry.set_state('disable')
            self.font_thickness.set_state("disable")
            self.line_width.set_state('disable')
            self.circle_size.set_state('disable')
            self.font_size.set_state('disable')
            self.bg_clr_dropdown.disable()

    def __create_path_plots(self,
                            multiple_videos: bool):
        if self.autocompute_var.get():
            style_attr = None
        else:
            if self.resolution_dropdown.getChoices() != 'As input':
                width = int(self.resolution_dropdown.getChoices().split('×')[0])
                height = int(self.resolution_dropdown.getChoices().split('×')[1])
            else:
                width, height = 'As input', 'As input'
            check_int(name='PATH LINE WIDTH', value=self.line_width.entry_get, min_value=1)
            check_int(name='PATH CIRCLE SIZE', value=self.circle_size.entry_get, min_value=1)
            check_int(name='PATH FONT SIZE', value=self.font_size.entry_get, min_value=1)
            check_int(name='PATH MAX LINES', value=self.max_lines_entry.entry_get, min_value=1)
            check_int(name='FONT THICKNESS', value=self.font_thickness.entry_get, min_value=1)
            style_attr = {'width': width,
                          'height': height,
                          'line width': int(self.line_width.entry_get),
                          'font size': int(self.font_size.entry_get),
                          'font thickness': int(self.font_thickness.entry_get),
                          'circle size': int(self.circle_size.entry_get),
                          'bg color': self.bg_clr_dropdown.getChoices(),
                          'max lines': int(self.max_lines_entry.entry_get),
                          'clf locations': self.include_clf_locations_var.get()}

        animal_attr = defaultdict(list)
        for attr in (self.bp_dropdowns, self.bp_colors):
            for key, value in attr.items():
                animal_attr[key].append(value.getChoices())

        clf_attr = None
        if self.include_clf_locations_var.get():
            clf_attr = defaultdict(list)
            for attr in (self.clf_name, self.clf_clr, self.clf_size):
                for key, value in attr.items():
                    clf_attr[key].append(value.getChoices())

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if not self.multiprocessing_var.get():
            path_plotter = PathPlotterSingleCore(config_path=self.config_path,
                                                 frame_setting=self.path_frames_var.get(),
                                                 video_setting=self.path_videos_var.get(),
                                                 last_frame=self.path_last_frm_var.get(),
                                                 files_found=data_paths,
                                                 input_style_attr=style_attr,
                                                 animal_attr=animal_attr,
                                                 input_clf_attr=clf_attr)
        else:
            path_plotter = PathPlotterMulticore(config_path=self.config_path,
                                                frame_setting=self.path_frames_var.get(),
                                                video_setting=self.path_videos_var.get(),
                                                last_frame=self.path_last_frm_var.get(),
                                                files_found=data_paths,
                                                input_style_attr=style_attr,
                                                animal_attr=animal_attr,
                                                input_clf_attr=clf_attr,
                                                cores=int(self.multiprocess_dropdown.getChoices()))

        path_plotter.run()

#_ = PathPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
