__author__ = "Simon Nilsson"

from tkinter import *
import multiprocessing

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_float
from simba.utils.enums import Keys, Links, Formats
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon, Entry_Box
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext
from simba.utils.errors import NoFilesFoundError
from simba.plotting.ROI_feature_visualizer import ROIfeatureVisualizer
from simba.plotting.ROI_feature_visualizer_mp import ROIfeatureVisualizerMultiprocess

class VisualizeROIFeaturesPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='VISUALIZE ROI FEATURES')
        ConfigReader.__init__(self, config_path=config_path)
        self.video_list = []
        video_file_paths = find_files_of_filetypes_in_directory(directory=self.video_dir, extensions=['.mp4', '.avi'])
        for file_path in video_file_paths:
            _, video_name, ext = get_fn_ext(filepath=file_path)
            self.video_list.append(video_name + ext)

        if len(self.video_list) == 0:
            raise NoFilesFoundError(msg='SIMBA ERROR: No videos in SimBA project. Import videos into you SimBA project to visualize ROI features.')


        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_FEATURES_PLOT.value)
        self.threshold_entry_box = Entry_Box(self.settings_frm, 'Probability threshold', '15')
        self.threshold_entry_box.entry_set(0.0)
        threshold_label = Label(self.settings_frm, text='Note: body-part locations detected with probabilities below this threshold will be filtered out.', font=("Helvetica", 10, 'italic'))
        self.border_clr_dropdown = DropDownMenu(self.settings_frm, 'Border color:', list(self.colors_dict.keys()), '12')
        self.border_clr_dropdown.setChoices('Black')

        self.show_pose_var = BooleanVar(value=True)
        self.show_ROI_centers_var = BooleanVar(value=True)
        self.show_ROI_tags_var = BooleanVar(value=True)
        self.show_direction_var = BooleanVar(value=False)
        self.multiprocess_var = BooleanVar(value=False)
        show_pose_cb = Checkbutton(self.settings_frm, text='Show pose', variable=self.show_pose_var)
        show_roi_center_cb = Checkbutton(self.settings_frm, text='Show ROI centers', variable=self.show_ROI_centers_var)
        show_roi_tags_cb = Checkbutton(self.settings_frm, text='Show ROI ear tags', variable=self.show_ROI_tags_var)
        show_roi_directionality_cb = Checkbutton(self.settings_frm, text='Show directionality', variable=self.show_direction_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.show_direction_var, dropdown_menus=[self.directionality_type_dropdown]))

        multiprocess_cb = Checkbutton(self.settings_frm, text='Multi-process (faster)', variable=self.multiprocess_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.directionality_type_dropdown = DropDownMenu(self.settings_frm, 'Direction type:', ['Funnel', 'Lines'], '12')
        self.directionality_type_dropdown.setChoices(choice='Funnel')
        self.directionality_type_dropdown.disable()


        self.single_video_frm = LabelFrame(self.main_frm, text='Visualize ROI features on SINGLE video', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.single_video_dropdown = DropDownMenu(self.single_video_frm, 'Select video', self.video_list, '15')
        self.single_video_dropdown.setChoices(self.video_list[0])
        self.single_video_btn = Button(self.single_video_frm, text='Visualize ROI features for SINGLE video', command=lambda: self.run(multiple=False))

        self.all_videos_frm = LabelFrame(self.main_frm, text='Visualize ROI features on ALL videos', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.all_videos_btn = Button(self.all_videos_frm, text='Generate ROI visualization on ALL videos', command=lambda: self.run(multiple=True))

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_entry_box.grid(row=0, sticky=NW)
        threshold_label.grid(row=1, sticky=NW)
        self.border_clr_dropdown.grid(row=2, sticky=NW)
        show_pose_cb.grid(row=3, sticky=NW)
        show_roi_center_cb.grid(row=4, sticky=NW)
        show_roi_tags_cb.grid(row=5, sticky=NW)
        show_roi_directionality_cb.grid(row=6, sticky=NW)
        self.directionality_type_dropdown.grid(row=6, column=1, sticky=NW)
        multiprocess_cb.grid(row=7, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=7, column=1, sticky=NW)
        self.single_video_frm.grid(row=1, sticky=W)
        self.single_video_dropdown.grid(row=0, sticky=W)
        self.single_video_btn.grid(row=1, pady=12)
        self.all_videos_frm.grid(row=2,sticky=W,pady=10)
        self.all_videos_btn.grid(row=0,sticky=W)

        #self.main_frm.mainloop()

    def run(self,
            multiple: bool):

        check_float(name='Body-part probability threshold', value=self.threshold_entry_box.entry_get, min_value=0.0,max_value=1.0)
        self.config.set('ROI settings', 'probability_threshold', str(self.threshold_entry_box.entry_get))
        with open(self.config_path, 'w') as f:
            self.config.write(f)

        style_attr = {}
        style_attr['ROI_centers'] = self.show_ROI_centers_var.get()
        style_attr['ROI_ear_tags'] = self.show_ROI_tags_var.get()
        style_attr['Directionality'] = self.show_direction_var.get()
        style_attr['Border_color'] = self.colors_dict[self.border_clr_dropdown.getChoices()]
        style_attr['Pose_estimation'] = self.show_pose_var.get()
        style_attr['Directionality_style'] = self.directionality_type_dropdown.getChoices()

        if not multiple:
            if not self.multiprocess_var.get():
                roi_feature_visualizer = ROIfeatureVisualizer(config_path=self.config_path, video_name=self.single_video_dropdown.getChoices(), style_attr=style_attr)
                roi_feature_visualizer.run()
            else:
                roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path=self.config_path, video_name=self.single_video_dropdown.getChoices(), style_attr=style_attr, core_cnt=int(self.multiprocess_dropdown.getChoices()))
                roi_feature_visualizer_mp = multiprocessing.Process(target=roi_feature_visualizer.run())
                roi_feature_visualizer_mp.start()
        else:
            if not self.multiprocess_var.get():
                for video_name in self.video_list:
                    roi_feature_visualizer = ROIfeatureVisualizer(config_path=self.config_path, video_name=video_name, style_attr=style_attr)
                    roi_feature_visualizer.run()
            else:
                for video_name in self.video_list:
                    roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path=self.config_path, video_name=video_name, style_attr=style_attr, core_cnt=int(self.multiprocess_dropdown.getChoices()))
                    roi_feature_visualizer_mp = multiprocessing.Process(target=roi_feature_visualizer.run())
                    roi_feature_visualizer_mp.start()

#_ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

