from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_int)
from tkinter import *
from simba.misc_tools import check_multi_animal_status
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.heat_mapper_location import HeatmapperLocation
import os, glob
from simba.ez_lineplot import draw_line_plot


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
        self.max_scale_var = IntVar()
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
        check_int(name='Max scale', value=self.max_scale_var.get(), min_value=1)
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
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.project_animal_cnt, self.x_cols, self.y_cols, [], [])
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










