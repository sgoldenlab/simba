from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_int,
                                          check_float,
                                          check_file_exist_and_readable)
import multiprocessing
import webbrowser
from tkinter import *
from simba.misc_tools import (check_multi_animal_status,
                              get_video_meta_data,
                              convert_parquet_to_csv,
                              convert_csv_to_parquet,
                              tabulate_clf_info)
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.get_coordinates_tools_v2 import get_coordinates_nilsson
from simba.heat_mapper_location import HeatmapperLocation
import os, glob
from simba.classifications_per_ROI import clf_within_ROI
from simba.tkinter_functions import hxtScrollbar, DropDownMenu, CreateToolTip
from collections import defaultdict
from simba.train_model_functions import get_all_clf_names
from simba.tkinter_functions import Entry_Box, FileSelect, FolderSelect
from simba.reorganize_keypoint_in_pose import KeypointReorganizer
from simba.ez_lineplot import DrawPathPlot
from simba.FSTTC_calculator import FSTTCPerformer
from simba.Kleinberg_calculator import KleinbergCalculator
from simba.timebins_clf_analyzer import TimeBinsClf
from simba.create_clf_log import ClfLogCreator
from simba.multi_cropper import MultiCropper
from simba.extract_seqframes import extract_seq_frames
from simba.video_processing import (downsample_video,
                                    clahe_enhance_video,
                                    crop_single_video,
                                    crop_multiple_videos,
                                    clip_video_in_range,
                                    remove_beginning_of_video,
                                    multi_split_video,
                                    change_img_format,
                                    batch_convert_video_format,
                                    convert_to_mp4,
                                    convert_video_powerpoint_compatible_format,
                                    extract_frame_range,
                                    extract_frames_single_video,
                                    batch_create_frames,
                                    change_single_video_fps,
                                    change_fps_of_multiple_videos,
                                    frames_to_movie,
                                    gif_creator,
                                    video_concatenator)

class HeatmapLocationPopup(object):
    def __init__(self,
                 config_path: str):

        self.config_path = config_path
        self.config = read_config_file(ini_path=config_path)
        self.setting_main = Toplevel()
        self.setting_main.minsize(400, 400)
        self.setting_main.wm_title('HEATMAPS: LOCATION')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.project_animal_cnt = read_config_entry(config=self.config, section='General settings', option='animal_no', data_type='int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.project_animal_cnt)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.project_animal_cnt, self.x_cols, self.y_cols, [], [])
        self.all_body_parts = []
        for animal, bp_cords in self.animal_bp_dict.items():
            for bp_dim, bp_data in bp_cords.items(): self.all_body_parts.extend(([x[:-2] for x in bp_data]))

        self.settings_frm = LabelFrame(self.setting_main, text="Settings")
        self.settings_frm.grid(row=0, column=0, sticky=NW)

        self.body_part_lbl = Label(self.settings_frm, text="Body-part: ")
        self.chosen_bp_val = StringVar(value=self.all_body_parts[0])
        self.choose_bp_dropdown = OptionMenu(self.settings_frm, self.chosen_bp_val, *self.all_body_parts)
        self.body_part_lbl.grid(row=0, column=0, sticky=NW)
        self.choose_bp_dropdown.grid(row=0, column=1, sticky=NW)

        self.bin_size_lbl = Label(self.settings_frm, text="Bin size (mm): ")
        self.bin_size_var = IntVar()
        self.bin_size_entry = Entry(self.settings_frm, width=15, textvariable=self.bin_size_var)
        self.bin_size_lbl.grid(row=1, column=0, sticky=NW)
        self.bin_size_entry.grid(row=1, column=1, sticky=NW)

        self.max_scale_lbl = Label(self.settings_frm, text="Max scale (s): ")
        self.max_scale_var = StringVar()
        self.max_scale_entry = Entry(self.settings_frm, width=15, textvariable=self.max_scale_var)
        self.max_scale_lbl.grid(row=2, column=0, sticky=NW)
        self.max_scale_entry.grid(row=2, column=1, sticky=NW)

        palette_options = ['magma', 'jet', 'inferno', 'plasma', 'viridis', 'gnuplot2']
        self.palette_lbl = Label(self.settings_frm, text="Palette : ")
        self.palette_var = StringVar(value=palette_options[0])
        self.palette_dropdown = OptionMenu(self.settings_frm, self.palette_var, *palette_options)
        self.palette_lbl.grid(row=3, column=0, sticky=NW)
        self.palette_dropdown.grid(row=3, column=1, sticky=NW)

        self.final_img_var = BooleanVar(value=False)
        self.final_img_cb = Checkbutton(self.settings_frm, text='Create last image', variable=self.final_img_var)
        self.frames_var = BooleanVar(value=False)
        self.frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.frames_var)
        self.videos_var = BooleanVar(value=False)
        self.videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.videos_var)
        self.final_img_cb.grid(row=4, column=0, sticky=NW)
        self.frames_cb.grid(row=5, column=0, sticky=NW)
        self.videos_cb.grid(row=6, column=0, sticky=NW)

        run_btn = Button(self.settings_frm, text='Run', command=lambda: self.create_heatmap_location())
        run_btn.grid(row=7, column=0, sticky=NW)

    def create_heatmap_location(self):
        if str(self.max_scale_var.get()) != 'auto':
            check_int(name='Max scale (auto or int)', value=self.max_scale_var.get(), min_value=1)
        check_int(name='Bin size', value=self.bin_size_var.get(), min_value=1)
        heat_mapper = HeatmapperLocation(config_path=self.config_path,
                                         final_img_setting=self.final_img_var.get(),
                                         video_setting=self.videos_var.get(),
                                         frame_setting=self.frames_var.get(),
                                         bin_size=self.bin_size_var.get(),
                                         palette=self.palette_var.get(),
                                         bodypart=self.chosen_bp_val.get(),
                                         max_scale=self.max_scale_var.get())
        heat_mapper.create_heatmaps()

class QuickLineplotPopup(object):
    def __init__(self,
                 config_path: str):

        self.config_path = config_path
        self.config = read_config_file(ini_path=config_path)
        self.setting_main = Toplevel()
        self.setting_main.minsize(400, 400)
        self.setting_main.wm_title('SIMPLE LINE PLOT')

        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.videos_dir = os.path.join(self.project_path, 'videos')
        self.video_files = [os.path.basename(x) for x in glob.glob(self.videos_dir + '/*')]
        if len(self.video_files) == 0:
            print('SIMBA ERROR: No files detected in the project_folder/videos directory.')
            raise ValueError()

        self.project_animal_cnt = read_config_entry(config=self.config, section='General settings', option='animal_no', data_type='int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.project_animal_cnt)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, list(self.multi_animal_id_lst), self.project_animal_cnt, list(self.x_cols), list(self.y_cols), [], [])
        self.all_body_parts = []
        for animal, bp_cords in self.animal_bp_dict.items():
            for bp_dim, bp_data in bp_cords.items(): self.all_body_parts.extend(([x[:-2] for x in bp_data]))

        self.settings_frm = LabelFrame(self.setting_main, text="Settings")
        self.settings_frm.grid(row=0, column=0, sticky=NW)

        self.video_lbl = Label(self.settings_frm, text="Video: ")
        self.chosen_video_val = StringVar(value=self.video_files[0])
        self.chosen_video_dropdown = OptionMenu(self.settings_frm, self.chosen_video_val, *self.video_files)
        self.video_lbl.grid(row=0, column=0, sticky=NW)
        self.chosen_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.body_part_lbl = Label(self.settings_frm, text="Body-part: ")
        self.chosen_bp_val = StringVar(value=self.all_body_parts[0])
        self.choose_bp_dropdown = OptionMenu(self.settings_frm, self.chosen_bp_val, *self.all_body_parts)
        self.body_part_lbl.grid(row=1, column=0, sticky=NW)
        self.choose_bp_dropdown.grid(row=1, column=1, sticky=NW)

        run_btn =  Button(self.settings_frm,text='Create path plot',command=lambda: draw_line_plot(self.config_path, self.chosen_video_val.get(), self.chosen_bp_val.get()))
        run_btn.grid(row=2, column=1, pady=10)

class ClfByROI(object):
    def __init__(self,
                 config_path: str):
        ROI_clf_toplevel = Toplevel()
        ROI_clf_toplevel.minsize(400, 400)
        ROI_clf_toplevel.wm_title("Classifications by ROI settings")
        ROI_clf_toplevel.lift()
        ROI_clf_toplevel = Canvas(hxtScrollbar(ROI_clf_toplevel))
        ROI_clf_toplevel.pack(fill="both", expand=True)

        ROI_menu = LabelFrame(ROI_clf_toplevel, text='Select_ROI(s)', padx=5, pady=5)
        classifier_menu = LabelFrame(ROI_clf_toplevel, text='Select classifier(s)', padx=5, pady=5)
        body_part_menu = LabelFrame(ROI_clf_toplevel, text='Select body part', padx=5, pady=5)
        self.menu_items = clf_within_ROI(config_path)
        self.ROI_check_boxes_status_dict = {}
        self.clf_check_boxes_status_dict = {}

        for row_number, ROI in enumerate(self.menu_items.ROI_str_name_list):
            self.ROI_check_boxes_status_dict[ROI] = IntVar()
            ROI_check_button = Checkbutton(ROI_menu, text=ROI, variable=self.ROI_check_boxes_status_dict[ROI])
            ROI_check_button.grid(row=row_number, sticky=W)

        for row_number, clf_name in enumerate(self.menu_items.behavior_names):
            self.clf_check_boxes_status_dict[clf_name] = IntVar()
            clf_check_button = Checkbutton(classifier_menu, text=clf_name,
                                           variable=self.clf_check_boxes_status_dict[clf_name])
            clf_check_button.grid(row=row_number, sticky=W)

        self.choose_bp = DropDownMenu(body_part_menu, 'Body part', self.menu_items.body_part_list, '12')
        self.choose_bp.setChoices(self.menu_items.body_part_list[0])
        self.choose_bp.grid(row=0, sticky=W)
        run_analysis_button = Button(ROI_clf_toplevel, text='Analyze classifications in each ROI',
                                     command=lambda: self.run_clf_by_ROI_analysis())
        body_part_menu.grid(row=0, sticky=W, padx=10, pady=10)
        ROI_menu.grid(row=1, sticky=W, padx=10, pady=10)
        classifier_menu.grid(row=2, sticky=W, padx=10, pady=10)
        run_analysis_button.grid(row=3, sticky=W, padx=10, pady=10)

    def run_clf_by_ROI_analysis(self):
        body_part_list = [self.choose_bp.getChoices()]
        ROI_dict_lists, behavior_list = defaultdict(list), []
        for loop_val, ROI_entry in enumerate(self.ROI_check_boxes_status_dict):
            check_val = self.ROI_check_boxes_status_dict[ROI_entry]
            if check_val.get() == 1:
                shape_type = self.menu_items.ROI_str_name_list[loop_val].split(':')[0].replace(':', '')
                shape_name = self.menu_items.ROI_str_name_list[loop_val].split(':')[1].replace(' ', '')
                ROI_dict_lists[shape_type].append(shape_name)

        for loop_val, clf_entry in enumerate(self.clf_check_boxes_status_dict):
            check_val = self.clf_check_boxes_status_dict[clf_entry]
            if check_val.get() == 1:
                behavior_list.append(self.menu_items.behavior_names[loop_val])
        if len(ROI_dict_lists) == 0: print('No ROIs selected.')
        if len(behavior_list) == 0: print('No classifiers selected.')

        else:
            clf_within_ROI.perform_ROI_clf_analysis(self.menu_items, ROI_dict_lists, behavior_list, body_part_list)


class FSTTCPopUp(object):
    def __init__(self,
                 config_path: str):

        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.target_cnt = read_config_entry(config=self.config, section='SML settings', option='No_targets', data_type='int')
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        fsttc_main = Toplevel()
        fsttc_main.minsize(400,320)
        fsttc_main.wm_title('Calculate forward-spike time tiling coefficients')
        fsttc_main.lift()
        fsttc_main = Canvas(hxtScrollbar(fsttc_main))
        fsttc_main.pack(fill="both", expand=True)

        fsttc_link_label = Label(fsttc_main, text='[Click here to learn about FSTTC]',cursor='hand2', fg='blue')
        fsttc_link_label.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md'))

        fsttc_settings_frm = LabelFrame(fsttc_main,text='FSTTC Settings', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        graph_cb_var = BooleanVar()
        graph_cb = Checkbutton(fsttc_settings_frm,text='Create graph',variable=graph_cb_var)
        time_delta = Entry_Box(fsttc_settings_frm,'Time Delta','10')
        behaviors_frm = LabelFrame(fsttc_settings_frm,text="Behaviors")
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(behaviors_frm, text=clf, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, sticky=NW)

        fsttc_run_btn = Button(fsttc_main,text='Calculate FSTTC',command=lambda:self.run_fsttc(time_delta=time_delta.entry_get, graph_var= graph_cb_var.get(), behaviours_dict=clf_var_dict))

        fsttc_link_label.grid(row=0,sticky=W,pady=5)
        fsttc_settings_frm.grid(row=1,sticky=W,pady=5)
        graph_cb.grid(row=0,sticky=W,pady=5)
        time_delta.grid(row=1,sticky=W,pady=5)
        behaviors_frm.grid(row=2,sticky=W,pady=5)
        fsttc_run_btn.grid(row=3, pady=10)

    def run_fsttc(self,
                  graph_var: bool,
                  behaviours_dict: dict,
                  time_delta: int=None):

        check_int('Time delta', value=time_delta)
        targets = []
        for behaviour, behavior_val in behaviours_dict.items():
            if behavior_val.get():
                targets.append(behaviour)

        if len(targets) < 2:
            print('SIMBA ERROR: FORWARD SPIKE TIME TILING COEFFICIENTS REQUIRE 2 OR MORE BEHAVIORS.')
            raise ValueError()

        FSTCC_performer = FSTTCPerformer(config_path=self.config_path,
                                         time_window=time_delta,
                                         behavior_lst=targets,
                                         create_graphs=graph_var)
        FSTCC_performer.find_sequences()
        FSTCC_performer.calculate_FSTTC()
        FSTCC_performer.save_FSTTC()
        FSTCC_performer.plot_FSTTC()



class KleinbergPopUp(object):
    def __init__(self,
                 config_path: str):
        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        self.target_cnt = read_config_entry(config=self.config, section='SML settings', option='No_targets', data_type='int')
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        kleinberg_main_frm = Toplevel()
        kleinberg_main_frm.minsize(400,320)
        kleinberg_main_frm.wm_title('Apply Kleinberg behavior classification smoothing')
        kleinberg_main_frm.lift()
        kleinberg_main_frm = Canvas(hxtScrollbar(kleinberg_main_frm))
        kleinberg_main_frm.pack(fill="both", expand=True)

        kleinberg_link = Label(kleinberg_main_frm, text='[Click here to learn about Kleinberg Smoother]', cursor='hand2', fg='blue')
        kleinberg_link.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md'))

        kleinberg_settings_frm = LabelFrame(kleinberg_main_frm,text='Kleinberg Settings', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.k_sigma = Entry_Box(kleinberg_settings_frm,'Sigma','10')
        self.k_sigma.entry_set('2')
        self.k_gamma = Entry_Box(kleinberg_settings_frm,'Gamma','10')
        self.k_gamma.entry_set('0.3')
        self.k_hierarchy = Entry_Box(kleinberg_settings_frm,'Hierarchy','10')
        self.k_hierarchy.entry_set('1')
        self.h_search_lbl = Label(kleinberg_settings_frm, text="Hierarchical search: ")
        self.h_search_lbl_val = BooleanVar()
        self.h_search_lbl_val.set(False)
        self.h_search_lbl_val_cb = Checkbutton(kleinberg_settings_frm, variable=self.h_search_lbl_val, command=None)
        kleinberg_table_frame = LabelFrame(kleinberg_main_frm, text='Choose classifier(s) to apply Kleinberg smoothing')
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(kleinberg_table_frame, text=clf, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, sticky=NW)

        run_kleinberg_btn = Button(kleinberg_main_frm, text='Apply Kleinberg Smoother', command=lambda: self.run_kleinberg(behaviors_dict=clf_var_dict, hierarchical_search=self.h_search_lbl_val.get()))

        kleinberg_link.grid(row=0,sticky=W)
        kleinberg_settings_frm.grid(row=1,sticky=W,padx=10)
        self.k_sigma.grid(row=0,sticky=W)
        self.k_gamma.grid(row=1,sticky=W)
        self.k_hierarchy.grid(row=2,sticky=W)
        self.h_search_lbl.grid(row=3, column=0, sticky=W)
        self.h_search_lbl_val_cb.grid(row=3, column=1, sticky=W)
        kleinberg_table_frame.grid(row=2,pady=10,padx=10)
        run_kleinberg_btn.grid(row=3)

    def run_kleinberg(self,
                      behaviors_dict: dict,
                      hierarchical_search: bool):

        targets = []
        for behaviour, behavior_val in behaviors_dict.items():
            if behavior_val.get():
                targets.append(behaviour)

        if len(targets) == 0:
            print('SIMBA ERROR: Select at least one classifier to apply Kleinberg smoothing')

        check_int(name='Hierarchy', value=self.k_hierarchy.entry_get)
        check_int(name='Sigma', value=self.k_sigma.entry_get)
        check_float(name='Gamma', value=self.k_gamma.entry_get)

        try:
            print('Applying kleinberg hyperparameter Setting: Sigma: {}, Gamma: {}, Hierarchy: {}'.format(str(self.k_sigma.entry_get), str(self.k_gamma.entry_get), str(self.k_hierarchy.entry_get)))
        except:
            print('Please insert accurate values for all hyperparameters.')

        kleinberg_analyzer = KleinbergCalculator(config_path=self.config_path,
                                                 classifier_names=targets,
                                                 sigma=self.k_sigma.entry_get,
                                                 gamma=self.k_gamma.entry_get,
                                                 hierarchy=self.k_hierarchy.entry_get,
                                                 hierarchical_search=hierarchical_search)
        kleinberg_analyzer.perform_kleinberg()


class TimeBinsClfPopUp(object):
    def __init__(self,
                 config_path: str):
        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        clf_timebins_main = Toplevel()
        clf_timebins_main.minsize(250, 500)
        clf_timebins_main.wm_title("Classifications by time bins settings")
        clf_timebins_main.lift()
        clf_timebins_main = Canvas(hxtScrollbar(clf_timebins_main))
        clf_timebins_main.pack(fill="both", expand=True)
        self.target_cnt = read_config_entry(config=self.config, section='SML settings', option='No_targets', data_type='int')
        clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        cbox_titles = ['First occurance (s)', 'Event count', 'Total event duration (s)',
                       'Mean event duration (s)', 'Median event duration (s)',
                       'Mean event interval (s)', 'Median event interval (s)']
        self.timebin_entrybox = Entry_Box(clf_timebins_main, 'Set time bin size (s)', '15')
        measures_frm = LabelFrame(clf_timebins_main, text='MEASUREMENTS', font=("Helvetica", 12, 'bold'), fg='black')
        clf_frm = LabelFrame(clf_timebins_main, text='CLASSIFIERS', font=("Helvetica", 12, 'bold'), fg='black')
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(measures_frm, text=title, variable=self.measurements_var_dict[title])
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name])
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(clf_timebins_main, text='Run', command=lambda: self.run_time_bins_clf())
        measures_frm.grid(row=0, sticky=NW)
        clf_frm.grid(row=1, sticky=NW)
        self.timebin_entrybox.grid(row=2, sticky=NW)
        run_button.grid(row=3, sticky=NW)

    def run_time_bins_clf(self):
        check_int(name='Time bin', value=self.timebin_entrybox.entry_get)
        measurement_lst, clf_list = [], []
        for name, val in self.measurements_var_dict.items():
            if val.get():
                measurement_lst.append(name)
        for name, val in self.clf_var_dict.items():
            if val.get():
                clf_list.append(name)
        if len(measurement_lst) == 0:
            print('SIMBA ERROR: Select at least 1 measurement to calculate descriptive statistics for.')
            raise ValueError()
        if len(clf_list) == 0:
            print('SIMBA ERROR: Select at least 1 classifier to calculate descriptive statistics for.')
            raise ValueError()
        time_bins_clf_analyzer = TimeBinsClf(config_path=self.config_path, bin_length=int(self.timebin_entrybox.entry_get), measurements=measurement_lst, classifiers=clf_list)
        time_bins_clf_multiprocessor = multiprocessing.Process(target=time_bins_clf_analyzer.analyze_timebins_clf())
        time_bins_clf_multiprocessor.start()

class ClfDescriptiveStatsPopUp(object):
    def __init__(self,
                 config_path: str):
        self.config, self.config_path = read_config_file(ini_path=config_path), config_path
        main_frm = Toplevel()
        main_frm.minsize(400, 600)
        main_frm.wm_title("Analyze Classifications: Descriptive statistics")
        main_frm.lift()
        main_frm = Canvas(hxtScrollbar(main_frm))
        main_frm.pack(fill="both", expand=True)
        self.target_cnt = read_config_entry(config=self.config, section='SML settings', option='No_targets', data_type='int')
        clf_names = get_all_clf_names(config=self.config, target_cnt=self.target_cnt)
        measures_frm = LabelFrame(main_frm, text='MEASUREMENTS', font=("Helvetica", 12, 'bold'), fg='black')
        clf_frm = LabelFrame(main_frm, text='CLASSIFIERS', font=("Helvetica", 12, 'bold'), fg='black')
        cbox_titles = ['Bout count', 'Total event duration (s)', 'Mean event bout duration (s)', 'Median event bout duration (s)', 'First event occurrence (s)', 'Mean event bout interval duration (s)', 'Median event bout interval duration (s)']
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(measures_frm, text=title, variable=self.measurements_var_dict[title])
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name])
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(main_frm, text='Run', command=lambda: self.run_descriptive_analysis())
        measures_frm.grid(row=0, sticky=NW)
        clf_frm.grid(row=1, sticky=NW)
        run_button.grid(row=2, sticky=NW)

    def run_descriptive_analysis(self):
        measurement_lst, clf_list = [], []
        for name, val in self.measurements_var_dict.items():
            if val.get():
                measurement_lst.append(name)
        for name, val in self.clf_var_dict.items():
            if val.get():
                clf_list.append(name)
        if len(measurement_lst) == 0:
            print('SIMBA ERROR: Select at least 1 measurement to calculate descriptive statistics for.')
            raise ValueError()
        if len(clf_list) == 0:
            print('SIMBA ERROR: Select at least 1 classifier to calculate descriptive statistics for.')
            raise ValueError()
        data_log_analyzer = ClfLogCreator(config_path=self.config_path, data_measures=measurement_lst, classifiers=clf_list)
        data_log_analyzer.analyze_data()
        data_log_analyzer.save_results()

class DownsampleVideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(400, 500)
        main_frm.wm_title("Downsample Video Resolution")
        main_frm.lift()
        main_frm = Canvas(hxtScrollbar(main_frm))
        main_frm.pack(fill="both", expand=True)


        instructions = Label(main_frm, text='Choose only one of the following method (Custom or Default)')
        choose_video_frm = LabelFrame(main_frm, text='SELECT VIDEO', font=("Helvetica", 12, 'bold'), fg='black', padx=5, pady=5)
        self.video_path_selected = FileSelect(choose_video_frm, "Video path", title='Select a video file')
        custom_frm = LabelFrame(main_frm, text='Custom resolution', font=("Helvetica", 12, 'bold'), fg='black', padx=5, pady=5)
        self.entry_width = Entry_Box(custom_frm, 'Width', '10')
        self.entry_height = Entry_Box(custom_frm, 'Height', '10')

        self.custom_downsample_btn = Button(custom_frm, text='Downsample to custom resolution', font=("Helvetica", 12, 'bold'), fg='black', command=lambda: self.custom_downsample())
        default_frm = LabelFrame(main_frm, text='Default resolution', font='bold', padx=5, pady=5)
        custom_resolutions = ["1980 x 1080", "1280 x 720", "720 x 480", "640 x 480", "320 x 240"]
        self.radio_btns = {}
        self.var = StringVar()
        for custom_cnt, resolution_radiobtn in enumerate(custom_resolutions):
            self.radio_btns[resolution_radiobtn] = Radiobutton(default_frm, text=resolution_radiobtn, variable=self.var, value=resolution_radiobtn)
            self.radio_btns[resolution_radiobtn].grid(row=custom_cnt, sticky=NW)

        self.default_downsample_btn = Button(default_frm, text='Downsample to default resolution',command=lambda: self.default_downsample())
        instructions.grid(row=0,sticky=NW,pady=10)
        choose_video_frm.grid(row=1, column=0,sticky=NW)
        self.video_path_selected.grid(row=0, column=0,sticky=NW)
        custom_frm.grid(row=2, column=0,sticky=NW)
        self.entry_width.grid(row=0, column=0,sticky=NW)
        self.entry_height.grid(row=1, column=0, sticky=NW)
        self.custom_downsample_btn.grid(row=3, column=0, sticky=NW)
        default_frm.grid(row=4, column=0, sticky=NW)
        self.default_downsample_btn.grid(row=len(custom_resolutions)+1, column=0, sticky=NW)

    def custom_downsample(self):
        width = self.entry_width.entry_get
        height = self.entry_height.entry_get
        check_int(name='Video width', value=width)
        check_int(name='Video height', value=height)
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(file_path=self.video_path_selected.file_path, video_width=int(width), video_height=int(height))

    def default_downsample(self):
        resolution = self.var.get()
        width, height = resolution.split('x', 2)[0].strip(), resolution.split('x', 2)[1].strip()
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(file_path=self.video_path_selected.file_path, video_width=int(width), video_height=int(height))


class CLAHEPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CLAHE conversion")
        clahe_frm = LabelFrame(main_frm, text='Contrast Limited Adaptive Histogram Equalization', font='bold', padx=5, pady=5)
        selected_video = FileSelect(clahe_frm, "Video path ", title='Select a video file')
        button_clahe = Button(clahe_frm, text='Apply CLAHE', command=lambda: clahe_enhance_video(file_path=selected_video.file_path))
        clahe_frm.grid(row=0,sticky=W)
        selected_video.grid(row=0,sticky=W)
        button_clahe.grid(row=1,pady=5)

class CropVideoPopUp(object):
    def __init__(self):
        crop_video_main = Toplevel()
        crop_video_main.minsize(300, 300)
        crop_video_main.wm_title("Crop Single Video")
        crop_video_lbl_frm = LabelFrame(crop_video_main, text='Crop Video',font='bold',padx=5,pady=5)
        selected_video = FileSelect(crop_video_lbl_frm,"Video path",title='Select a video file')
        button_crop_video_single = Button(crop_video_lbl_frm, text='Crop Video',command=lambda: crop_single_video(file_path=selected_video.file_path))

        crop_video_lbl_frm_multiple = LabelFrame(crop_video_main, text='Fixed coordinates crop for multiple videos', font='bold',  padx=5, pady=5)
        input_folder = FolderSelect(crop_video_lbl_frm_multiple, 'Video directory:', title='Select Folder with videos')
        output_folder = FolderSelect(crop_video_lbl_frm_multiple, 'Output directory:', title='Select a folder for your output videos')
        button_crop_video_multiple = Button(crop_video_lbl_frm_multiple, text='Confirm', command=lambda: crop_multiple_videos(directory_path=input_folder.folder_path, output_path=output_folder.folder_path))

        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        button_crop_video_single.grid(row=1, sticky=NW, pady=10)
        crop_video_lbl_frm_multiple.grid(row=1, sticky=W, pady=10, padx=5)
        input_folder.grid(row=0,sticky=W,pady=5)
        output_folder.grid(row=1,sticky=W,pady=5)
        button_crop_video_multiple.grid(row=2,sticky=W,pady=5)


class ClipVideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("Clip video")
        selected_video = FileSelect(main_frm, "Video path", title='Select a video file')
        method_1_frm = LabelFrame(main_frm, text='Method 1', font='bold', padx=5, pady=5)
        label_set_time_1 = Label(method_1_frm, text='Please enter the time frame in hh:mm:ss format')
        start_time = Entry_Box(method_1_frm, 'Start at:', '8')
        end_time = Entry_Box(method_1_frm, 'End at:', '8')
        CreateToolTip(method_1_frm, 'Method 1 will retrieve the specified time input. (eg: input of Start at: 00:00:00, End at: 00:01:00, will create a new video from the chosen video from the very start till it reaches the first minute of the video)')
        method_2_frm = LabelFrame(main_frm, text='Method 2', font='bold', padx=5, pady=5)
        method_2_time = Entry_Box(method_2_frm, 'Seconds:', '8')
        label_method_2 = Label(method_2_frm, text='Method 2 will retrieve from the end of the video (e.g.,: an input of 3 seconds will get rid of the first 3 seconds of the video).')
        button_cutvideo_method_1 = Button(method_1_frm, text='Cut Video', command=lambda: clip_video_in_range(file_path=selected_video.file_path, start_time=start_time.entry_get, end_time=end_time.entry_get))
        button_cutvideo_method_2 = Button(method_2_frm, text='Cut Video', command=lambda: remove_beginning_of_video( file_path=selected_video.file_path, time=method_2_time.entry_get))
        selected_video.grid(row=0, sticky=W)
        method_1_frm.grid(row=1, sticky=NW, pady=5)
        label_set_time_1.grid(row=0, sticky=NW)
        start_time.grid(row=1, sticky=NW)
        end_time.grid(row=2, sticky=NW)
        button_cutvideo_method_1.grid(row=3, sticky=NW)
        method_2_frm.grid(row=2, sticky=NW, pady=5)
        label_method_2.grid(row=0, sticky=NW)
        method_2_time.grid(row=2, sticky=NW)
        button_cutvideo_method_2.grid(row=3, sticky=NW)


class MultiShortenPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(300, 200)
        self.main_frm.wm_title("Clip video into multiple videos")
        self.main_frm.lift()
        self.main_frm = Canvas(hxtScrollbar(self.main_frm))
        self.main_frm.pack(fill="both", expand=True)
        settings_frm = LabelFrame(self.main_frm, text='Split videos into different parts', font='bold', padx=5, pady=5)
        self.selected_video = FileSelect(settings_frm, "Video path", title='Select a video file')
        self.clip_cnt = Entry_Box(settings_frm, '# of clips', '8')
        confirm_settings_btn = Button(settings_frm, text='Confirm', command=lambda: self.show_start_stop())
        settings_frm.grid(row=0, sticky=NW)
        self.selected_video.grid(row=1, sticky=NW, columnspan=2)
        self.clip_cnt.grid(row=2, sticky=NW)
        confirm_settings_btn.grid(row=2, column=1, sticky=W)
        instructions = Label(settings_frm, text='Enter clip start and stop times in HH:MM:SS format', fg='navy')
        instructions.grid(row=3, column=0)

    def show_start_stop(self):
        check_int(name='Number of clips', value=self.clip_cnt.entry_get)
        if hasattr(self, 'table'):
            self.table.destroy()
        self.table = LabelFrame(self.main_frm)
        self.table.grid(row=2, column=0, sticky=NW)
        Label(self.table, text='Clip #').grid(row=0, column=0)
        Label(self.table, text='Start Time').grid(row=0, column=1, sticky=NW)
        Label(self.table, text='Stop Time').grid(row=0, column=2, sticky=NW)
        self.clip_names, self.start_times, self.end_times = [], [], []
        for i in range(int(self.clip_cnt.entry_get)):
            Label(self.table, text='Clip ' + str(i + 1)).grid(row=i + 2, sticky=W)
            self.start_times.append(Entry(self.table))
            self.start_times[i].grid(row=i + 2, column=1, sticky=W)
            self.end_times.append(Entry(self.table))
            self.end_times[i].grid(row=i + 2, column=2, sticky=W)

        run_button = Button(self.table, text='Clip video', command=lambda: self.run_clipping(), fg='navy', font=("Helvetica", 12, 'bold'))
        run_button.grid(row=int(self.clip_cnt.entry_get) + 2, column=2, sticky=W)

    def run_clipping(self):
        start_times, end_times = [], []
        check_file_exist_and_readable(self.selected_video.file_path)
        for start_time, end_time in zip(self.start_times, self.end_times):
            print(start_time.get())
            start_times.append(start_time.get())
            end_times.append(end_time.get())
        multi_split_video(file_path=self.selected_video.file_path, start_times=start_times, end_times=end_times)



class ChangeImageFormatPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CHANGE IMAGE FORMAT")
        self.input_folder_selected = FolderSelect(main_frm, "Image directory", title='Select folder with images:')
        set_input_format_frm = LabelFrame(main_frm, text='Original image format', font=("Helvetica", 12, 'bold'), padx=15, pady=5)
        set_output_format_frm = LabelFrame(main_frm, text='Output image format', font=("Helvetica", 12, 'bold'), padx=15, pady=5)

        self.input_file_type, self.out_file_type = StringVar(), StringVar()
        input_png_rb = Radiobutton(set_input_format_frm, text=".png", variable=self.input_file_type, value="png")
        input_jpeg_rb = Radiobutton(set_input_format_frm, text=".jpg", variable=self.input_file_type, value="jpg")
        input_bmp_rb = Radiobutton(set_input_format_frm, text=".bmp", variable=self.input_file_type, value="bmp")
        output_png_rb = Radiobutton(set_output_format_frm, text=".png", variable=self.out_file_type, value="png")
        output_jpeg_rb = Radiobutton(set_output_format_frm, text=".jpg", variable=self.out_file_type, value="jpg")
        output_bmp_rb = Radiobutton(set_output_format_frm, text=".bmp", variable=self.out_file_type, value="bmp")
        run_btn = Button(main_frm, text='Convert image file format', command= lambda: self.run_img_conversion())
        self.input_folder_selected.grid(row=0,column=0)
        set_input_format_frm.grid(row=1,column=0,pady=5)
        set_output_format_frm.grid(row=2, column=0, pady=5)
        input_png_rb.grid(row=0, column=0)
        input_jpeg_rb.grid(row=1, column=0)
        input_bmp_rb.grid(row=2, column=0)
        output_png_rb.grid(row=0, column=0)
        output_jpeg_rb.grid(row=1, column=0)
        output_bmp_rb.grid(row=2, column=0)
        run_btn.grid(row=3,pady=5)

    def run_img_conversion(self):
        if len(os.listdir(self.input_folder_selected.folder_path)) == 0:
            print('SIMBA ERROR: The input folder {} contains ZERO files.'.format(self.input_folder_selected.folder_path))
            raise ValueError('SIMBA ERROR: The input folder {} contains ZERO files.'.format(self.input_folder_selected.folder_path))
        change_img_format(directory=self.input_folder_selected.folder_path, file_type_in=self.input_file_type.get(), file_type_out=self.out_file_type.get())


class ConertVideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CONVERT VIDEO FORMAT")

        convert_multiple_videos_frm = LabelFrame(main_frm, text='Convert multiple videos', font=("Helvetica", 12, 'bold'), padx=5, pady=5)
        video_dir = FolderSelect(convert_multiple_videos_frm, 'Video directory', title='Select folder with videos')
        original_format = Entry_Box(convert_multiple_videos_frm, 'Input format', '12')
        output_format = Entry_Box(convert_multiple_videos_frm, 'Output format', '12')
        convert_multiple_btn = Button(convert_multiple_videos_frm, text='Convert multiple videos', command=lambda: batch_convert_video_format(directory=video_dir.folder_path, input_format=original_format.entry_get, output_format=output_format.entry_get))

        convert_single_video_frm = LabelFrame(main_frm,text='Convert single video',font=("Helvetica",12,'bold'),padx=5,pady=5)
        self.selected_video = FileSelect(convert_single_video_frm, "Video path", title='Select a video file')
        self.output_format = StringVar()
        checkbox_v1 = Radiobutton(convert_single_video_frm, text="Convert to .mp4", variable=self.output_format, value='mp4')
        checkbox_v2 = Radiobutton(convert_single_video_frm, text="Convert mp4 into PowerPoint supported format", variable=self.output_format, value='pptx')
        convert_single_btn = Button(convert_single_video_frm, text='Convert video format', command= lambda: self.convert_single())

        convert_multiple_videos_frm.grid(row=0,sticky=W)
        video_dir.grid(row=0,sticky=W)
        original_format.grid(row=1,sticky=W)
        output_format.grid(row=2,sticky=W)
        convert_multiple_btn.grid(row=3,pady=10)
        convert_single_video_frm.grid(row=1,sticky=W)
        self.selected_video.grid(row=0,sticky=W)
        checkbox_v1.grid(row=1,column=0,sticky=W)
        checkbox_v2.grid(row=2,column=0,sticky=W)
        convert_single_btn.grid(row=3,column=0,pady=10)

    def convert_single(self):
        if self.output_format.get() == 'mp4':
            convert_to_mp4(file_path=self.selected_video.file_path)
        if self.output_format.get() == 'pptx':
            convert_video_powerpoint_compatible_format(file_path=self.selected_video.file_path)


class ExtractSpecificFramesPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("EXTRACT DEFINED FRAMES")
        self.video_file_selected = FileSelect(main_frm, "Video path", title='Select a video file')
        select_frames_frm = LabelFrame(main_frm, text='Frames to be extracted', padx=5, pady=5)
        self.start_frm = Entry_Box(select_frames_frm, 'Start Frame:', '10')
        self.end_frm = Entry_Box(select_frames_frm, 'End Frame:', '10')
        run_btn = Button(select_frames_frm, text='Extract Frames', command= lambda: self.start_frm_extraction())

        self.video_file_selected.grid(row=0,column=0,sticky=NW,pady=10)
        select_frames_frm.grid(row=1,column=0,sticky=NW)
        self.start_frm.grid(row=2,column=0,sticky=NW)
        self.end_frm.grid(row=3,column=0,sticky=NW)
        run_btn.grid(row=4,pady=5, sticky=NW)

    def start_frm_extraction(self):
        start_frame = self.start_frm.entry_get
        end_frame = self.end_frm.entry_get
        check_int(name='Start frame', value=start_frame)
        check_int(name='End frame', value=end_frame)
        if int(end_frame) < int(start_frame):
            print('SIMBA ERROR: The end frame ({}) cannot come before the start frame ({})'.format(str(end_frame), str(start_frame)))
            raise ValueError()
        video_meta_data = get_video_meta_data(video_path=self.video_file_selected.file_path)
        if int(start_frame) > video_meta_data['frame_count']:
            print('SIMBA ERROR: The start frame ({}) is larger than the number of frames in the video ({})'.format(str(start_frame), str(video_meta_data['frame_count'])))
            raise ValueError
        if int(end_frame) > video_meta_data['frame_count']:
            print('SIMBA ERROR: The end frame ({}) is larger than the number of frames in the video ({})'.format(str(end_frame), str(video_meta_data['frame_count'])))
            raise ValueError()
        extract_frame_range(file_path=self.video_file_selected.file_path, start_frame=int(start_frame), end_frame=int(end_frame))

class ExtractAllFramesPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("EXTRACT ALL FRAMES")
        single_video_frm = LabelFrame(main_frm, text='Single video', padx=5, pady=5, font='bold')
        video_path = FileSelect(single_video_frm, "Video path", title='Select a video file')
        single_video_btn = Button(single_video_frm, text='Extract Frames (Single video)', command=lambda: extract_frames_single_video(file_path=video_path.file_path))
        multiple_videos_frm = LabelFrame(main_frm, text='Multiple videos', padx=5, pady=5, font='bold')
        folder_path = FolderSelect(multiple_videos_frm, 'Folder path', title=' Select video folder')
        multiple_video_btn = Button(multiple_videos_frm, text='Extract Frames (Multiple videos)', command=lambda: batch_create_frames(directory=folder_path.folder_path))
        single_video_frm.grid(row=0, sticky=NW, pady=10)
        video_path.grid(row=0, sticky=NW)
        single_video_btn.grid(row=1, sticky=W, pady=10)
        multiple_videos_frm.grid(row=1, sticky=W, pady=10)
        folder_path.grid(row=0, sticky=W)
        multiple_video_btn.grid(row=1, sticky=W, pady=10)

class Csv2ParquetPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 300)
        main_frm.wm_title("Convert CSV directory to parquet")
        frm = LabelFrame(main_frm, text='Select CSV directory', padx=5, pady=5, font='bold')
        folder_path = FolderSelect(frm, 'CSV folder path', title=' Select CSV folder')
        run_btn = Button(frm, text='Convert CSV to parquet', command=lambda: convert_csv_to_parquet(directory=folder_path.folder_path))
        frm.grid(row=1, sticky=W, pady=10)
        folder_path.grid(row=0, sticky=W)
        run_btn.grid(row=1, sticky=W, pady=10)

class Parquet2CsvPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 300)
        main_frm.wm_title("Convert parquet directory to CSV")
        frm = LabelFrame(main_frm, text='Select parquet directory', padx=5, pady=5, font='bold')
        folder_path = FolderSelect(frm, 'Parquet folder path', title=' Select parquet folder')
        run_btn = Button(frm, text='Convert parquet to CSV', command=lambda: convert_parquet_to_csv(directory=folder_path.folder_path))
        frm.grid(row=1, sticky=W, pady=10)
        folder_path.grid(row=0, sticky=W)
        run_btn.grid(row=1, sticky=W, pady=10)

class MultiCropPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 300)
        main_frm.wm_title("Multi Crop")
        input_folder = FolderSelect(main_frm, "Input Video Folder  ")
        output_folder = FolderSelect(main_frm, "Output Folder")
        video_type = Entry_Box(main_frm, " Video type (e.g. mp4)", "15")
        crop_cnt = Entry_Box(main_frm, "# of crops", "15")

        run_btn = Button(main_frm,text='Crop',command=lambda:MultiCropper(file_type=video_type.entry_get,
                                                                          input_folder=input_folder.folder_path,
                                                                          output_folder=output_folder.folder_path,
                                                                          crop_cnt=crop_cnt.entry_get))
        input_folder.grid(row=0,sticky=W,pady=2)
        output_folder.grid(row=1,sticky=W,pady=2)
        video_type.grid(row=2,sticky=W,pady=2)
        crop_cnt.grid(row=3,sticky=W,pady=2)
        run_btn.grid(row=4,pady=10)

class ChangeFpsSingleVideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CHANGE FRAME RATE: SINGLE VIDEO")
        video_path = FileSelect(main_frm, "Video path", title='Select a video file')
        fps_entry_box = Entry_Box(main_frm, 'Output FPS:', '10')
        run_btn = Button(main_frm, text='Convert', command=lambda: change_single_video_fps(file_path=video_path.file_path, fps=fps_entry_box.entry_get))
        video_path.grid(row=0,sticky=W)
        fps_entry_box.grid(row=1,sticky=W)
        run_btn.grid(row=2)

class ChangeFpsMultipleVideosPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(400, 200)
        main_frm.wm_title("CHANGE FRAME RATE: MULTIPLE VIDEO")
        folder_path = FolderSelect(main_frm, "Folder path", title='Select folder with videos: ')
        fps_entry = Entry_Box(main_frm, 'Output FPS: ', '10')
        run_btn = Button(main_frm, text='Convert', command=lambda: change_fps_of_multiple_videos(directory=folder_path.folder_path, fps=fps_entry.entry_get))
        folder_path.grid(row=0, sticky=W)
        fps_entry.grid(row=1, sticky=W)
        run_btn.grid(row=2)

class ExtractSEQFramesPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("EXTRACT ALL FRAMES FROM SEQ FILE")
        video_path = FileSelect(main_frm, "Video Path", title='Select a video file: ')
        run_btn = Button(main_frm, text='Extract All Frames', command=lambda: extract_seq_frames(video_path.file_path))
        video_path.grid(row=0)
        run_btn.grid(row=1)


class MergeFrames2VideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(250, 250)
        main_frm.wm_title("MERGE IMAGES TO VIDEO")
        self.folder_path = FolderSelect(main_frm, "IMAGE DIRECTORY", title='Select directory with frames: ')
        settings_frm = LabelFrame(main_frm, text='SETTINGS', padx=5, pady=5, font=("Helvetica",12,'bold'), fg='black')
        self.img_format_entry_box = Entry_Box(settings_frm, 'IMAGE FORMAT (e.g. png): ', '20')
        self.bitrate_entry_box = Entry_Box(settings_frm, 'BITRATE (e.g. 8000): ', '20', validation='numeric')
        self.fps_entry = Entry_Box(settings_frm, 'FPS: ', '20', validation='numeric')
        run_btn = Button(settings_frm, text='Create Video', command=lambda: self.run())
        settings_frm.grid(row=1,pady=10)
        self.folder_path.grid(row=0,column=0,pady=10)
        self.img_format_entry_box.grid(row=1,column=0,sticky=W)
        self.fps_entry.grid(row=2,column=0,sticky=W,pady=5)
        self.bitrate_entry_box.grid(row=3,column=0,sticky=W,pady=5)
        run_btn.grid(row=4,column=1,sticky=E,pady=10)

    def run(self):
        img_format = self.img_format_entry_box.entry_get
        bitrate = self.bitrate_entry_box.entry_get
        fps = self.fps_entry.entry_get
        _ = frames_to_movie(directory=self.folder_path.folder_path, fps=fps, bitrate=bitrate, img_format=img_format)

class PrintModelInfoPopUp(object):
    def __init__(self):
        model_info_win = Toplevel()
        model_info_win.minsize(250, 250)
        model_info_win.wm_title("PRINT MACHINE MODEL INFO")
        model_info_frame = LabelFrame(model_info_win, text='PRINT MODEL INFORMATION', padx=5, pady=5, font='bold')
        model_path_selector = FileSelect(model_info_frame, 'Model path', title='Select a video file')
        btn_print_info = Button(model_info_frame,text='PRINT MODEL INFO',command=lambda:tabulate_clf_info(clf_path=model_path_selector.file_path))
        model_info_frame.grid(row=0, sticky=W)
        model_path_selector.grid(row=0, sticky=W, pady=5)
        btn_print_info.grid(row=1, sticky=W)

class CreateGIFPopUP(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(250, 250)
        main_frm.wm_title("CREATE GIF FROM VIDEO")
        settings_frm = LabelFrame(main_frm, text='SETTINGS', padx=5, pady=5, font=("Helvetica",12,'bold'), fg='black')
        selected_video = FileSelect(settings_frm, 'Video path: ', title='Select a video file')
        start_time_entry_box = Entry_Box(settings_frm, 'Start time (s): ', '16', validation='numeric')
        duration_entry_box = Entry_Box(settings_frm, 'Duration (s): ', '16', validation='numeric')
        width_entry_box = Entry_Box(settings_frm, 'Width: ', '16', validation='numeric')
        width_instructions_1 = Label(settings_frm, text='example Width: 240, 360, 480, 720, 1080', font=("Times", 10, "italic"))
        width_instructions_2 = Label(settings_frm, text='Aspect ratio is kept (i.e., height is automatically computed)', font=("Times", 10, "italic"))
        run_btn = Button(settings_frm,text='CREATE GIF', command=lambda:gif_creator(file_path=selected_video.file_path, start_time=start_time_entry_box.entry_get, duration=duration_entry_box.entry_get, width=width_entry_box.entry_get))
        settings_frm.grid(row=0,sticky=W)
        selected_video.grid(row=0,sticky=W,pady=5)
        start_time_entry_box.grid(row=1,sticky=W)
        duration_entry_box.grid(row=2,sticky=W)
        width_entry_box.grid(row=3,sticky=W)
        width_instructions_1.grid(row=4,sticky=W)
        width_instructions_2.grid(row=5, sticky=W)
        run_btn.grid(row=6,sticky=NW, pady=10)

class CalculatePixelsPerMMInVideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title('CALCULATE PIXELS PER MILLIMETER IN VIDEO')
        self.video_path = FileSelect(main_frm, "Select a video file: ", title='Select a video file')
        self.known_distance = Entry_Box(main_frm, 'Known length in real life (mm): ', '0', validation='numeric')
        run_btn = Button(main_frm, text='GET PIXELS PER MILLIMETER', command= lambda: self.run())
        self.video_path.grid(row=0,column=0,pady=10,sticky=W)
        self.known_distance.grid(row=1,column=0,pady=10,sticky=W)
        run_btn.grid(row=2,column=0,pady=10)

    def run(self):
        file_path = self.video_path.file_path
        distance = self.known_distance.entry_get
        _ = get_video_meta_data(video_path=file_path)
        check_int(name='Distance', value=distance)
        check_file_exist_and_readable(file_path=file_path)
        if int(distance) <= 0:
            print('SIMBA ERROR: Known distance has to be greater than 0')
            raise ValueError()
        pixels = get_coordinates_nilsson(self.video_path.file_path, distance)
        print('1 PIXEL REPRESENTS {} MILLIMETERS IN VIDEO {}'.format(str(round(pixels, 4)), os.path.basename(file_path)))

class MakePathPlotPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CREATE PATH PLOT")
        settings_frm = LabelFrame(main_frm)
        video_path = FileSelect(settings_frm, 'VIDEO PATH: ', lblwidth='30')
        body_part = Entry_Box(settings_frm, 'BODY PART: ', '30')
        data_path = FileSelect(settings_frm, 'DATA PATH (e.g., DLC H5 or CSV file): ', lblwidth='30')
        color_lst = ['White',
                     'Black',
                     'Grey',
                     'Red',
                     'Dark-red',
                     'Maroon',
                     'Orange',
                     'Dark-orange',
                     'Coral',
                     'Chocolate',
                     'Yellow',
                     'Green',
                     'Dark-grey',
                     'Light-grey',
                     'Pink',
                     'Lime',
                     'Purple',
                     'Cyan']
        background_color = DropDownMenu(settings_frm,'BACKGROUND COLOR: ',color_lst,'10')
        background_color.setChoices(choice='White')
        line_color = DropDownMenu(settings_frm, 'LINE COLOR: ', color_lst, '10')
        line_color.setChoices(choice='Red')
        line_thickness = DropDownMenu(settings_frm, 'LINE THICKNESS: ', list(range(1, 11)), '10')
        line_thickness.setChoices(choice=1)
        circle_size = DropDownMenu(settings_frm, 'CIRCLE SIZE: ', list(range(1, 11)), '10')
        circle_size.setChoices(choice=5)
        run_btn = Button(settings_frm,text='CREATE PATH PLOT VIDEO',command = lambda: DrawPathPlot(data_path=data_path.file_path,
                                                                                                                video_path=video_path.file_path,
                                                                                                                body_part=body_part.entry_get,
                                                                                                                bg_color=background_color.getChoices(),
                                                                                                                line_color=line_color.getChoices(),
                                                                                                                line_thinkness=line_thickness.getChoices(),
                                                                                                                circle_size=circle_size.getChoices()))
        settings_frm.grid(row=0,sticky=W)
        video_path.grid(row=0,sticky=W)
        data_path.grid(row=1,sticky=W)
        body_part.grid(row=2,sticky=W)
        background_color.grid(row=3,sticky=W)
        line_color.grid(row=4, sticky=W)
        line_thickness.grid(row=5, sticky=W)
        circle_size.grid(row=6, sticky=W)
        run_btn.grid(row=7,pady=10)

class PoseReorganizerPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(500,800)
        self.main_frm.wm_title('RE-ORGANIZE POSE_ESTIMATION DATA')
        self.main_frm.lift()
        self.main_frm = Canvas(hxtScrollbar(self.main_frm))
        self.main_frm.pack(fill="both", expand=True)
        settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=('Helvetica', 12, 'bold'), pady=5, padx=5)
        self.data_folder = FolderSelect(settings_frm, 'DATA FOLDER: ', lblwidth='10')
        self.pose_tool_dropdown = DropDownMenu(settings_frm, 'Tracking tool', ['DLC', 'maDLC'], '10')
        self.pose_tool_dropdown.setChoices('DLC')
        self.file_format = DropDownMenu(settings_frm,'FILE TYPE: ',['csv','h5'],'10')
        self.file_format.setChoices('csv')
        confirm_btn = Button(settings_frm, text='Confirm', command=lambda: self.confirm())
        settings_frm.grid(row=0,sticky=NW)
        self.data_folder.grid(row=0,sticky=NW,columnspan=3)
        self.pose_tool_dropdown.grid(row=1,sticky=NW)
        self.file_format.grid(row=2,sticky=NW)
        confirm_btn.grid(row=2,column=1,sticky=NW)

    def confirm(self):
        if hasattr(self, 'table'):
            self.table.destroy()

        self.keypoint_reorganizer = KeypointReorganizer(data_folder=self.data_folder.folder_path, pose_tool=self.pose_tool_dropdown.getChoices(), file_format=self.file_format.getChoices())
        self.table = LabelFrame(self.main_frm, text='SET NEW ORDER', font=('Helvetica', 12, 'bold'), pady=5, padx=5)
        self.current_order = LabelFrame(self.table,text='CURRENT ORDER:')
        self.new_order = LabelFrame(self.table,text='NEW ORDER:')

        self.table.grid(row=1,sticky=W,pady=10)
        self.current_order.grid(row=0, column=0, sticky=NW, pady=5)
        self.new_order.grid(row=0,column=1,sticky=NW,padx=5,pady=5)
        idx1, idx2, oldanimallist, oldbplist, self.newanimallist, self.newbplist = ([0] * len(self.keypoint_reorganizer.bp_list) for i in range(6))

        if self.keypoint_reorganizer.animal_list:
            animal_list_reduced = list(set(self.keypoint_reorganizer.animal_list))
            self.pose_tool = 'maDLC'
            for i in range(len(self.keypoint_reorganizer.bp_list)):
                idx1[i] = Label(self.current_order,text=str(i+1) + '.')
                oldanimallist[i] = Label(self.current_order,text=str(self.keypoint_reorganizer.animal_list[i]))
                oldbplist[i] = Label(self.current_order,text=str(self.keypoint_reorganizer.bp_list[i]))
                idx1[i].grid(row=i,column=0,sticky=W)
                oldanimallist[i].grid(row=i,column=1,sticky=W, ipady=5)
                oldbplist[i].grid(row=i,column=2,sticky=W, ipady=5)
                idx2[i] = Label(self.new_order,text=str(i+1) + '.')
                self.newanimallist[i] = DropDownMenu(self.new_order, ' ', animal_list_reduced, '10')
                self.newbplist[i] = DropDownMenu(self.new_order,' ', self.keypoint_reorganizer.bp_list,'10')
                self.newanimallist[i].setChoices(self.keypoint_reorganizer.animal_list[i])
                self.newbplist[i].setChoices(self.keypoint_reorganizer.bp_list[i])
                idx2[i].grid(row=i,column=0,sticky=W)
                self.newanimallist[i].grid(row=i, column=1, sticky=W)
                self.newbplist[i].grid(row=i,column=2,sticky=W)

        else:
            self.pose_tool = 'DLC'
            for i in range(len(self.keypoint_reorganizer.bp_list)):
                idx1[i] = Label(self.current_order, text=str(i + 1) + '.')
                oldbplist[i] = Label(self.current_order, text=str(self.keypoint_reorganizer.bp_list[i]))
                idx1[i].grid(row=i, column=0, sticky=W, ipady=5)
                oldbplist[i].grid(row=i, column=2, sticky=W, ipady=5)
                idx2[i] = Label(self.new_order, text=str(i + 1) + '.')
                self.newbplist[i] = StringVar()
                oldanimallist[i] = OptionMenu(self.new_order, self.newbplist[i], *self.keypoint_reorganizer.bp_list)
                self.newbplist[i].set(self.keypoint_reorganizer.bp_list[i])
                idx2[i].grid(row=i, column=0, sticky=W)
                oldanimallist[i].grid(row=i, column=1, sticky=W)

        button_run = Button(self.table, text='Run re-organization', command= lambda: self.run_reorganization())
        button_run.grid(row=2, column=1, sticky=W)

    def run_reorganization(self):
        if self.pose_tool == 'DLC':
            new_bp_list = []
            for curr_choice in self.newbplist:
                new_bp_list.append(curr_choice.get())
            self.keypoint_reorganizer.perform_reorganization(animal_list=None, bp_lst=new_bp_list)

        if self.pose_tool == 'maDLC':
            new_bp_list, new_animal_list = [], []
            for curr_animal, curr_bp in zip(self.newanimallist, self.newbplist):
                new_bp_list.append(curr_bp.getChoices())
                new_animal_list.append(curr_animal.getChoices())
            self.keypoint_reorganizer.perform_reorganization(animal_list=new_animal_list, bp_lst=new_bp_list)

class AboutSimBAPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(896, 507)
        main_frm.wm_title("ABOUT SIMBA")
        canvas = Canvas(main_frm,width=896,height=507,bg='black')
        canvas.pack()
        scriptdir = os.path.dirname(__file__)
        img = PhotoImage(file=os.path.join(scriptdir,'About_me_050122_1.png'))
        canvas.create_image(0,0,image=img,anchor='nw')
        canvas.image = img

class ConcatenatingVideosPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 300)
        main_frm.wm_title("CONCATENATE VIDEOS")
        settings_frm = LabelFrame(main_frm, text='SETTINGS', font=('Helvetica', 12, 'bold'), pady=5, padx=5)
        video_path_1 = FileSelect(settings_frm, "First video path: ", title='Select a video file')
        video_path_2 = FileSelect(settings_frm, "Second video path: ", title='Select a video file')
        resolutions =  ['Video 1', 'Video 2', 320, 640, 720, 1280, 1980]
        resolution_dropdown = DropDownMenu(settings_frm, 'Resolution:', resolutions, '15')
        resolution_dropdown.setChoices(resolutions[0])
        horizontal = BooleanVar(value=False)
        horizontal_radio_btn = Radiobutton(settings_frm, text="Horizontal concatenation", variable=horizontal, value=True)
        vertical_radio_btn = Radiobutton(settings_frm, text="Vertical concatenation", variable=horizontal, value=False)
        run_btn = Button(main_frm, text='RUN', font=('Helvetica', 12, 'bold'), command=lambda: video_concatenator(video_one_path=video_path_1.file_path,
                                                                                      video_two_path=video_path_2.file_path,
                                                                                      resolution=resolution_dropdown.getChoices(),
                                                                                      horizontal=horizontal.get()))

        settings_frm.grid(row=0, column=0, sticky=NW)
        video_path_1.grid(row=0, column=0, sticky=NW)
        video_path_2.grid(row=1, column=0, sticky=NW)
        resolution_dropdown.grid(row=2,column=0,sticky=NW)
        horizontal_radio_btn.grid(row=3,column=0,sticky=NW)
        vertical_radio_btn.grid(row=4, column=0, sticky=NW)
        run_btn.grid(row=1, column=0, sticky=NW)

