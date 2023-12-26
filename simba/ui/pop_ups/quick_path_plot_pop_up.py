
import os
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import find_all_videos_in_directory
from simba.plotting.ez_lineplot import draw_line_plot

class QuickLineplotPopup(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='SIMPLE LINE PLOT')
        ConfigReader.__init__(self, config_path=config_path)
        video_filepaths = find_all_videos_in_directory(directory=os.path.join(self.project_path, 'videos'))
        self.video_files = [os.path.basename(x) for x in video_filepaths]
        if len(self.video_files) == 0:
            raise NoFilesFoundError(msg='SIMBA ERROR: No files detected in the project_folder/videos directory.', source=self.__class__.__name__)

        self.all_body_parts = []
        for animal, bp_cords in self.animal_bp_dict.items():
            for bp_dim, bp_data in bp_cords.items(): self.all_body_parts.extend(([x[:-2] for x in bp_data]))

        self.settings_frm = LabelFrame(self.main_frm, text="Settings")
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

        run_btn = Button(self.settings_frm,text='Create path plot',command=lambda: draw_line_plot(self.config_path, self.chosen_video_val.get(), self.chosen_bp_val.get()))
        run_btn.grid(row=2, column=1, pady=10)
        self.main_frm.mainloop()


#_ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
