from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_int)
import webbrowser
from tkinter import *
from simba.misc_tools import check_multi_animal_status
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.heat_mapper_location import HeatmapperLocation
import os, glob
from simba.ez_lineplot import draw_line_plot
from simba.classifications_per_ROI import clf_within_ROI
from simba.tkinter_functions import hxtScrollbar, DropDownMenu
from collections import defaultdict
from simba.train_model_functions import get_all_clf_names
from simba.tkinter_functions import Entry_Box
from simba.FSTTC_calculator import FSTTCPerformer
from simba.read_config_unit_tests import check_int, check_float
from simba.Kleinberg_calculator import KleinbergCalculator


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