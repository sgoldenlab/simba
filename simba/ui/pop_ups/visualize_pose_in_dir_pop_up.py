__author__ = "Simon Nilsson"

from tkinter import *

from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, Entry_Box, DropDownMenu, FolderSelect
from simba.utils.enums import Keys, Links, Formats
from simba.utils.lookups import get_color_dict
from simba.utils.errors import NotDirectoryError
from simba.plotting.plot_pose_in_dir import create_video_from_dir
from simba.mixins.pop_up_mixin import PopUpMixin

class VisualizePoseInFolderPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title='Visualize pose-estimation', size=(350, 200))
        settings_frame = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.input_folder = FolderSelect(settings_frame, 'Input directory (with csv/parquet files)', title='Select input folder', lblwidth=20)
        self.output_folder = FolderSelect(settings_frame, 'Output directory (where your videos will be saved)', title='Select output folder', lblwidth=20)
        self.circle_size = Entry_Box(settings_frame, 'Circle size', 0, validation='numeric', labelwidth=20)
        run_btn = Button(self.main_frm, text='VISUALIZE POSE', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='blue', command= lambda: self.run())
        self.advanced_settings_btn = Button(self.main_frm, text='OPEN ADVANCED SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='red', command=lambda: self.launch_adv_settings())
        settings_frame.grid(row=0, sticky=W)
        self.input_folder.grid(row=0, column=0, pady=10, sticky=W)
        self.output_folder.grid(row=1, column=0, pady=10, sticky=W)
        self.circle_size.grid(row=2, column=0, pady=10, sticky=W)
        run_btn.grid(row=3, column=0, pady=10)
        self.advanced_settings_btn.grid(row=4, column=0, pady=10)
        self.color_lookup = None


    def run(self):
        circle_size_int = self.circle_size.entry_get
        input_folder = self.input_folder.folder_path
        output_folder = self.output_folder.folder_path
        if (input_folder == '') or (input_folder == 'No folder selected'):
            raise NotDirectoryError(msg='SIMBA ERROR: Please select an input folder to continue')
        elif (output_folder == '') or (output_folder == 'No folder selected'):
            raise NotDirectoryError(msg='SimBA ERROR: Please select an output folder to continue')
        else:
            if self.color_lookup is not None:
                cleaned_color_lookup = {}
                for k, v in self.color_lookup.items():
                    cleaned_color_lookup[k] = v.getChoices()
                self.color_lookup = cleaned_color_lookup
            create_video_from_dir(in_directory=input_folder, out_directory=output_folder, circle_size=int(circle_size_int), clr_attr=self.color_lookup)

    def launch_adv_settings(self):
        if self.advanced_settings_btn['text'] == 'OPEN ADVANCED SETTINGS':
            self.advanced_settings_btn.configure(text="CLOSE ADVANCED SETTINGS")
            self.adv_settings_frm = LabelFrame(self.main_frm, text='ADVANCED SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
            self.confirm_btn = Button(self.adv_settings_frm, text='Confirm', command=lambda: self.launch_clr_menu())
            self.specify_animals_dropdown = DropDownMenu(self.adv_settings_frm, 'ANIMAL COUNT: ', list(range(1, 11)), '20')
            self.adv_settings_frm.grid(row=5, column=0, pady=10)
            self.specify_animals_dropdown.grid(row=0, column=0, sticky=NW)
            self.confirm_btn.grid(row=0, column=1)
        elif self.advanced_settings_btn['text'] == 'CLOSE ADVANCED SETTINGS':
            if hasattr(self, 'adv_settings_frm'):
                self.adv_settings_frm.destroy()
                self.color_lookup = None
            self.advanced_settings_btn.configure(text="OPEN ADVANCED SETTINGS")

    def launch_clr_menu(self):
        if hasattr(self, 'color_table_frme'):
            self.color_table_frme.destroy()
        clr_dict = get_color_dict()
        self.color_table_frme = LabelFrame(self.adv_settings_frm, text='SELECT COLORS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.color_lookup = {}
        for animal_cnt in list(range(int(self.specify_animals_dropdown.getChoices()))):
            self.color_lookup['Animal_{}'.format(str(animal_cnt+1))] = DropDownMenu(self.color_table_frme, 'Animal {} color:'.format(str(animal_cnt+1)), list(clr_dict.keys()), '20')
            self.color_lookup['Animal_{}'.format(str(animal_cnt+1))].setChoices(list(clr_dict.keys())[animal_cnt])
            self.color_lookup['Animal_{}'.format(str(animal_cnt+1))].grid(row=animal_cnt, column=0, sticky=NW)
        self.color_table_frme.grid(row=1, column=0, sticky=NW)