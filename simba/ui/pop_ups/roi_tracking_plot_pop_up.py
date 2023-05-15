__author__ = "Simon Nilsson"

from tkinter import *
import multiprocessing

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_if_filepath_list_is_empty, check_float
from simba.utils.enums import Keys, Links, Formats
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon, Entry_Box
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext
from simba.utils.printing import stdout_success
from simba.plotting.ROI_plotter import ROIPlot
from simba.plotting.ROI_plotter_mp import ROIPlotMultiprocess

class VisualizeROITrackingPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='VISUALIZE ROI TRACKING')
        ConfigReader.__init__(self, config_path=config_path)
        self.video_list = []
        video_file_paths = find_files_of_filetypes_in_directory(directory=self.video_dir, extensions=['.mp4', '.avi'])

        for file_path in video_file_paths:
            _, video_name, ext = get_fn_ext(filepath=file_path)
            self.video_list.append(video_name + ext)

        check_if_filepath_list_is_empty(filepaths=self.video_list, error_msg='No videos in SimBA project. Import videos into you SimBA project to visualize ROI tracking.')
        self.multiprocess_var = BooleanVar()
        self.show_pose_var = BooleanVar(value=True)
        self.animal_name_var = BooleanVar(value=True)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_DATA_PLOT.value)
        self.threshold_entry_box = Entry_Box(self.settings_frm, 'Body-part probability threshold', '30')
        self.threshold_entry_box.entry_set(0.0)
        threshold_label = Label(self.settings_frm, text='Note: body-part locations detected with probabilities below this threshold is removed from visualization.', font=("Helvetica", 10, 'italic'))

        self.show_pose_cb = Checkbutton(self.settings_frm, text='Show pose-estimated location', variable=self.show_pose_var)
        self.show_animal_name_cb = Checkbutton(self.settings_frm, text='Show animal names', variable=self.animal_name_var)
        self.multiprocess_cb = Checkbutton(self.settings_frm, text='Multi-process (faster)', variable=self.multiprocess_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN VISUALIZATION', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')

        self.single_video_frm = LabelFrame(self.run_frm, text='SINGLE video', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.single_video_dropdown = DropDownMenu(self.single_video_frm, 'Select video', self.video_list, '15')
        self.single_video_dropdown.setChoices(self.video_list[0])
        self.single_video_btn = Button(self.single_video_frm, text='Create SINGLE ROI video', fg='blue', command=lambda: self.run_visualize(multiple=False))

        self.all_videos_frm = LabelFrame(self.run_frm, text='ALL videos', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.all_videos_btn = Button(self.all_videos_frm, text='Create ALL ROI videos ({} videos found)'.format(str(len(self.video_list))), fg='red', command=lambda: self.run_visualize(multiple=True))

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_entry_box.grid(row=0, column=0, sticky=NW)
        threshold_label.grid(row=1, column=0, sticky=NW)
        self.show_pose_cb.grid(row=2, column=0, sticky=NW)
        self.show_animal_name_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=4, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=4, column=1, sticky=NW)
        self.run_frm.grid(row=1, column=0, sticky=NW)
        self.single_video_frm.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.single_video_btn.grid(row=1, column=0, sticky=NW)
        self.all_videos_frm.grid(row=1, column=0, sticky=NW)
        self.all_videos_btn.grid(row=0, column=0, sticky=NW)

    def run_visualize(self, multiple: bool):
        if multiple:
            videos = self.video_list
        else:
            videos = [self.single_video_dropdown.getChoices()]

        check_float(name='Body-part probability threshold', value=self.threshold_entry_box.entry_get, min_value=0.0, max_value=1.0)
        style_attr = {}
        style_attr['Show_body_part'] = False
        style_attr['Show_animal_name'] = False
        if self.show_pose_var.get(): style_attr['Show_body_part'] = True
        if self.animal_name_var.get(): style_attr['Show_animal_name'] = True
        if not self.multiprocess_var.get():
            self.config.set('ROI settings', 'probability_threshold', str(self.threshold_entry_box.entry_get))
            with open(self.config_path, 'w') as f:
                self.config.write(f)
            for video in videos:
                roi_plotter = ROIPlot(ini_path=self.config_path, video_path=video, style_attr=style_attr)
                roi_plotter.insert_data()
                roi_plotter_multiprocessor = multiprocessing.Process(target=roi_plotter.visualize_ROI_data())
                roi_plotter_multiprocessor.start()
        else:
            with open(self.config_path, 'w') as f:
                self.config.write(f)
            core_cnt = self.multiprocess_dropdown.getChoices()
            for video in videos:
                roi_plotter = ROIPlotMultiprocess(ini_path=self.config_path, video_path=video, core_cnt=int(core_cnt), style_attr=style_attr)
                roi_plotter.insert_data()
                roi_plotter.visualize_ROI_data()

        stdout_success(msg='All ROI videos created and saved in project_folder/frames/output/ROI_analysis directory')

#_ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
