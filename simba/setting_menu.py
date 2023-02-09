from tkinter import *
from simba.read_config_unit_tests import read_config_file, read_config_entry, check_float, check_int, read_project_path_and_file_type
import os
from simba.misc_tools import check_multi_animal_status
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.ROI_analyzer import ROIAnalyzer
from simba.ROI_feature_analyzer import ROIFeatureCreator
from simba.timebins_movement_analyzer import TimeBinsMovementAnalyzer
from simba.movement_processor import MovementProcessor
from simba.ROI_time_bin_calculator import ROITimebinCalculator
from simba.enums import ReadConfig, Formats, Dtypes, Paths

class SettingsMenu(object):

    def __init__(self,
                 config_path: str,
                 title: str):

        self.title, self.config_path = title, config_path
        self.config = read_config_file(ini_path=config_path)
        self.setting_main = Toplevel()
        self.setting_main.minsize(400, 400)
        self.setting_main.wm_title(self.title)
        self.project_path, _ = read_project_path_and_file_type(config=self.config)
        self.project_animal_cnt = read_config_entry(config=self.config, section=ReadConfig.GENERAL_SETTINGS.value, option=ReadConfig.ANIMAL_CNT.value, data_type=Dtypes.INT.value)
        self.animal_cnt_frm = LabelFrame(self.setting_main, text='SELECT NUMBER OF ANIMALS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.animal_cnt = IntVar(value=1)
        self.animal_option = set(range(1, self.project_animal_cnt + 1))
        self.animal_cnt_dropdown = OptionMenu(self.animal_cnt_frm, self.animal_cnt, *self.animal_option)
        self.animal_cnt_dropdown_lbl = Label(self.animal_cnt_frm, text="# of animals")
        self.animal_cnt_confirm_btn = Button(self.animal_cnt_frm, text="Confirm", command=lambda: self.display_second_choice_frm())
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.project_animal_cnt)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.roi_definitions_path = os.path.join(self.project_path, 'logs', Paths.ROI_DEFINITIONS.value)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.project_animal_cnt, self.x_cols, self.y_cols, [], [])
        self.all_body_parts = []
        for animal, bp_cords in self.animal_bp_dict.items():
            for bp_dim, bp_data in bp_cords.items():
                self.animal_bp_dict[animal][bp_dim] = [x[:-2] for x in bp_data]
                self.all_body_parts.extend(([x[:-2] for x in bp_data]))

        self.animal_cnt_frm.grid(row=0, sticky=NW)
        self.animal_cnt_dropdown_lbl.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=1, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=2, sticky=NW)

    def update_config(self):
        with open(self.config_path, 'w') as f:
            self.config.write(f)

    def create_choose_body_parts_frm(self):
        self.lbl_frm_cnt += 1
        self.animals_dict = {}
        self.choose_body_parts_frm = LabelFrame(self.second_choice_frm, text="Choose animal body-parts")
        for animal_cnt in range(self.animal_cnt.get()):
            animal_name = list(self.animal_bp_dict.keys())[animal_cnt]
            self.animals_dict[animal_name] = {}
            self.animals_dict[animal_name]['label'] = Label(self.choose_body_parts_frm, text='{} body-part: '.format(str(animal_name)))
            self.animals_dict[animal_name]['var'] = StringVar(value=self.animal_bp_dict[animal_name]['X_bps'][0])
            self.animals_dict[animal_name]['dropdown'] = OptionMenu(self.choose_body_parts_frm, self.animals_dict[animal_name]['var'], *self.animal_bp_dict[animal_name]['X_bps'])
            self.animals_dict[animal_name]['label'].grid(row=animal_cnt, column=0, sticky=NW)
            self.animals_dict[animal_name]['dropdown'].grid(row=animal_cnt, column=1, sticky=NW)
        self.choose_body_parts_frm.grid(row=self.lbl_frm_cnt, sticky=NW)

    def create_choose_single_body_parts_frm(self):
        self.lbl_frm_cnt += 1
        self.choose_body_part_frm = LabelFrame(self.second_choice_frm, text="Choose animal body-parts")
        self.body_part_lbl = Label(self.choose_body_part_frm, text="Body-part: ")
        self.chosen_bp_val = StringVar(value=self.all_body_parts[0])
        self.choose_bp_dropdown = OptionMenu(self.choose_body_part_frm, self.chosen_bp_val, *self.all_body_parts)
        self.body_part_lbl.grid(row=0, column=0, sticky=NW)
        self.choose_bp_dropdown.grid(row=0, column=1, sticky=NW)
        self.choose_body_part_frm.grid(row=self.lbl_frm_cnt, column=0, sticky=NW)

    def create_calculate_distances_checkbox(self):
        self.lbl_frm_cnt += 1
        self.calculate_distances_frm = LabelFrame(self.second_choice_frm, text="Calculate distances")
        self.calculate_distance_moved_var = BooleanVar(value=False)
        self.calculate_distance_moved_cb = Checkbutton(self.calculate_distances_frm, text='Calculate distance moved within ROI', variable=self.calculate_distance_moved_var)
        self.calculate_distance_moved_cb.grid(row=0, column=0, sticky=NW)
        self.calculate_distances_frm.grid(row=self.lbl_frm_cnt, column=0, sticky=NW, pady=5)

    def create_threshold_entry(self):
        self.lbl_frm_cnt += 1
        self.probability_frm = LabelFrame(self.second_choice_frm, text="Set probability threshold")
        self.probability_bp_val = StringVar(value='0.00')
        self.probability_bp_lbl = Label(self.probability_frm, text="Probability threshold: ")
        self.probability_bp_entry_box = Entry(self.probability_frm, width=15, textvariable=self.probability_bp_val)
        self.probability_frm.grid(row=self.lbl_frm_cnt, sticky=NW, pady=5)
        self.probability_bp_lbl.grid(row=0, column=0, sticky=NW)
        self.probability_bp_entry_box.grid(row=0, column=1, sticky=NW)

    def create_run_btn(self):
        self.lbl_frm_cnt += 1
        self.execute_frm = LabelFrame(self.setting_main, text="EXECUTE", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.run_btn = Button(self.execute_frm, text='Run', command=lambda: self.run_analysis())
        self.execute_frm.grid(row=self.lbl_frm_cnt, sticky=NW, pady=5)
        self.run_btn.grid(row=0, column=0, sticky=NW)

    def create_time_bin_entry(self):
        self.lbl_frm_cnt += 1
        self.time_bin_frm = LabelFrame(self.second_choice_frm, text="Time-bins", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.time_bin_val = IntVar()
        self.time_bin_lbl = Label(self.time_bin_frm, text="Time-bin size (s): ")
        self.time_bin_entry = Entry(self.time_bin_frm, width=15, textvariable=self.time_bin_val)
        self.time_bin_frm.grid(row=self.lbl_frm_cnt)
        self.time_bin_lbl.grid(row=0, column=0)
        self.time_bin_entry.grid(row=0, column=1)

    def display_second_choice_frm(self):
        if hasattr(self, 'second_choice_frm'):
            self.second_choice_frm.destroy()
        self.second_choice_frm = LabelFrame(self.setting_main, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.second_choice_frm.grid(row=1, pady=5)
        self.lbl_frm_cnt = 0

        if self.title == 'ROI ANALYSIS':
            self.create_choose_body_parts_frm()
            self.create_threshold_entry()
            self.create_calculate_distances_checkbox()
            self.create_run_btn()

        if self.title == 'APPEND ROI FEATURES':
            if not os.path.isfile(self.roi_definitions_path):
                print('SIMBA ERROR: No ROIs have been defined. Please define ROIs before appending ROI-based features')
                raise FileNotFoundError()
            self.create_choose_body_parts_frm()
            self.create_run_btn()

        if self.title == 'ANALYZE MOVEMENT':
            self.create_choose_body_parts_frm()
            self.create_threshold_entry()
            self.create_run_btn()

        if self.title == 'TIME BINS: DISTANCE/VELOCITY':
            self.create_choose_body_parts_frm()
            self.create_time_bin_entry()
            self.create_run_btn()

        if self.title == 'TIME BINS: ANALYZE ROI':
            self.create_choose_body_parts_frm()
            self.create_time_bin_entry()
            self.create_run_btn()

        if self.title == 'HEATMAPS':
            self.create_choose_single_body_parts_frm()

    def run_analysis(self):
        if self.title == 'ROI ANALYSIS':
            self.config.set(ReadConfig.ROI_SETTINGS.value, ReadConfig.ROI_ANIMAL_CNT.value, str(self.animal_cnt.get()))
            check_float(name='Probability threshold', value=self.probability_bp_entry_box.get())
            self.config.set(ReadConfig.ROI_SETTINGS.value, ReadConfig.PROBABILITY_THRESHOLD.value, str(self.probability_bp_entry_box.get()))
            for cnt, (name, data) in enumerate(self.animals_dict.items()):
                self.config.set('ROI settings', 'animal_{}_bp'.format(str(cnt + 1)), str(self.animals_dict[name]['var'].get()))
            self.update_config()
            roi_analyzer = ROIAnalyzer(ini_path=self.config_path, data_path='outlier_corrected_movement_location', calculate_distances=self.calculate_distance_moved_var.get())
            roi_analyzer.read_roi_dfs()
            roi_analyzer.analyze_ROIs()
            roi_analyzer.save_data()

        if self.title == 'APPEND ROI FEATURES':
            roi_feature_creator = ROIFeatureCreator(config_path=self.config_path)
            roi_feature_creator.analyze_ROI_data()
            roi_feature_creator.save_new_features_files()

        if self.title == 'ANALYZE MOVEMENT':
            self.config.set(ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, ReadConfig.ROI_ANIMAL_CNT.value, str(self.animal_cnt.get()))
            check_float(name='Probability threshold', value=self.probability_bp_entry_box.get())
            self.config.set(ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, ReadConfig.PROBABILITY_THRESHOLD.value, str(self.probability_bp_entry_box.get()))
            for cnt, (name, data) in enumerate(self.animals_dict.items()):
                self.config.set(ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'animal_{}_bp'.format(str(cnt + 1)), str(self.animals_dict[name]['var'].get()))
            self.update_config()
            movement_processor = MovementProcessor(config_path=self.config_path)
            movement_processor.process_movement()
            movement_processor.save_results()

        if self.title == 'TIME BINS: DISTANCE/VELOCITY':
            check_int(name='Time bin', value=str(self.time_bin_val.get()))
            self.config.set(ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, ReadConfig.ROI_ANIMAL_CNT.value, str(self.animal_cnt.get()))
            for cnt, (name, data) in enumerate(self.animals_dict.items()):
                self.config.set(ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'animal_{}_bp'.format(str(cnt + 1)), str(self.animals_dict[name]['var'].get()))
            self.update_config()
            time_bin_movement_analyzer = TimeBinsMovementAnalyzer(config_path=self.config_path, bin_length=int(self.time_bin_val.get()))
            time_bin_movement_analyzer.analyze_movement()

        if self.title == 'TIME BINS: ANALYZE ROI':
            check_int(name='Time bin', value=self.time_bin_val.get())
            self.config.set(ReadConfig.ROI_SETTINGS.value, ReadConfig.ROI_ANIMAL_CNT.value, str(self.animal_cnt.get()))
            for cnt, (name, data) in enumerate(self.animals_dict.items()):
                self.config.set(ReadConfig.ROI_SETTINGS.value, 'animal_{}_bp'.format(str(cnt + 1)),str(self.animals_dict[name]['var'].get()))
            self.update_config()
            roi_time_bin_calculator = ROITimebinCalculator(config_path=self.config_path, bin_length=self.time_bin_val.get())
            roi_time_bin_calculator.analyze_time_bins()
            roi_time_bin_calculator.save_results()




# test = SettingsMenu(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', title='ROI ANALYSIS')
# test.setting_main.mainloop()

# test = SettingsMenu(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', title='APPEND ROI FEATURES')
# test.setting_main.mainloop()

# test = SettingsMenu(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', title='ANALYZE MOVEMENT')
# test.setting_main.mainloop()

# test = SettingsMenu(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', title='TIME BINS: ANALYZE ROI')
# test.setting_main.mainloop()



