__author__ = "Simon Nilsson"

"""
Tkinter LabelFrame pop-up classes used in SimBA
"""

import sys

import numpy as np
from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_int,
                                          check_float,
                                          check_str,
                                          check_file_exist_and_readable,
                                          check_if_filepath_list_is_empty)
import multiprocessing
import webbrowser
from tkinter import *
import simba
from simba.misc_tools import (get_video_meta_data,
                              convert_parquet_to_csv,
                              convert_csv_to_parquet,
                              tabulate_clf_info,
                              get_color_dict,
                              find_all_videos_in_directory,
                              get_file_name_info_in_directory,
                              SimbaTimer,
                              find_files_of_filetypes_in_directory,
                              archive_processed_files,
                              copy_img_folder,
                              str_2_bool)
from simba.drop_bp_cords import get_fn_ext
from simba.utils.lookups import get_third_party_appender_file_formats
from simba.third_party_label_appenders.third_party_appender import ThirdPartyLabelAppender
from simba.plotting.ROI_feature_visualizer import ROIfeatureVisualizer
from simba.get_coordinates_tools_v2 import get_coordinates_nilsson
from simba.roi_tools.ROI_clf_calculator import ROIClfCalculator
from simba.tkinter_functions import hxtScrollbar, DropDownMenu, CreateToolTip, CreateLabelFrameWithIcon
from simba.train_model_functions import get_all_clf_names
from simba.interpolate_smooth_post_hoc import PostHocSmooth, PostHocInterpolate
from simba.tkinter_functions import Entry_Box, FileSelect, FolderSelect
from simba.feature_extractors.feature_subsets import FeatureSubsetsCalculator
from simba.reorganize_keypoint_in_pose import KeypointReorganizer
from simba.plotting.ez_lineplot import DrawPathPlot
from simba.FSTTC_calculator import FSTTCPerformer
from simba.Kleinberg_calculator import KleinbergCalculator
from simba.timebins_clf_analyzer import TimeBinsClf
from simba.create_clf_log import ClfLogCreator
from simba.plotting.ez_lineplot import draw_line_plot
from simba.multi_cropper import MultiCropper
from simba.plotting.gantt_creator_mp import GanttCreatorMultiprocess
from simba.plotting.gantt_creator import GanttCreatorSingleProcess
from simba.plotting.probability_plot_creator import TresholdPlotCreatorSingleProcess
from simba.plotting.probability_plot_creator_mp import TresholdPlotCreatorMultiprocess
from simba.plotting.ROI_plotter_mp import ROIPlotMultiprocess
from simba.remove_keypoints_in_pose import KeypointRemover
from simba.extract_annotation_frames import AnnotationFrameExtractor
from simba.plotting.plot_pose_in_dir import create_video_from_dir
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.extract_seqframes import extract_seq_frames
from simba.plotting.frame_mergerer_ffmpeg import FrameMergererFFmpeg
from simba.plotting.ROI_feature_visualizer_mp import ROIfeatureVisualizerMultiprocess
from simba.plotting.ROI_plotter import ROIPlot
from simba.user_pose_config_creator import PoseConfigCreator
from simba.pose_reset import PoseResetter
from simba.read_config_unit_tests import check_if_dir_exists
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
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
from simba.plotting.plot_clf_results import PlotSklearnResultsSingleCore
from simba.plotting.plot_clf_results_mp import PlotSklearnResultsMultiProcess
from simba.plotting.path_plotter import PathPlotterSingleCore
from simba.plotting.path_plotter_mp import PathPlotterMulticore
from simba.batch_process_videos.batch_process_menus import BatchProcessFrame
from simba.plotting.Directing_animals_visualizer import DirectingOtherAnimalsVisualizer
from simba.plotting.Directing_animals_visualizer_mp import DirectingOtherAnimalsVisualizerMultiprocess
from simba.plotting.heat_mapper_location import HeatmapperLocationSingleCore
from simba.plotting.heat_mapper_location_mp import HeatMapperLocationMultiprocess
from simba.plotting.distance_plotter import DistancePlotterSingleCore
from simba.plotting.clf_validator import ClassifierValidationClips
from simba.plotting.single_run_model_validation_video import ValidateModelOneVideo
from simba.plotting.single_run_model_validation_video_mp import ValidateModelOneVideoMultiprocess
from simba.plotting.distance_plotter_mp import DistancePlotterMultiCore
from simba.plotting.heat_mapper_clf import HeatMapperClfSingleCore
from simba.plotting.heat_mapper_clf_mp import HeatMapperClfMultiprocess
from simba.feature_extractors.unit_tests import read_video_info_csv
from simba.plotting.data_plotter import DataPlotter
from simba.severity_processor import SeverityProcessor
from simba.enums import (ReadConfig,
                         Options,
                         Formats,
                         Paths,
                         Dtypes,
                         Links,
                         Keys)

import pandas as pd
import subprocess
import urllib
from collections import defaultdict
from datetime import datetime
from PIL import Image, ImageTk
import atexit
from simba.rw_dfs import read_df
import os, glob
from simba.pup_retrieval_protocol import PupRetrieverCalculator
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.errors import (NoChoosenClassifierError,
                                NoChoosenMeasurementError,
                                NoChoosenROIError,
                                NoFilesFoundError,
                                CountError,
                                FrameRangeError,
                                NotDirectoryError,
                                MixedMosaicError,
                                DuplicationError,
                                NoSpecifiedOutputError,
                                NoROIDataError)
sys.setrecursionlimit(10**7)

class HeatmapLocationPopup(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='HEATMAPS: LOCATION')
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()), error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/outlier_corrected_movement_location directory. ')
        max_scales = list(np.linspace(5, 600, 5))
        max_scales.insert(0, 'Auto-compute')
        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.HEATMAP_LOCATION.value)
        self.palette_dropdown = DropDownMenu(self.style_settings_frm, 'Palette:', self.palette_options, '16')
        self.shading_dropdown = DropDownMenu(self.style_settings_frm, 'Shading:', self.shading_options, '16')
        self.bp_dropdown = DropDownMenu(self.style_settings_frm, 'Body-part:', self.bp_names, '16')
        self.max_time_scale_dropdown = DropDownMenu(self.style_settings_frm, 'Max time scale (s):', max_scales, '16')
        self.bin_size_dropdown = DropDownMenu(self.style_settings_frm, 'Bin size (mm):', self.heatmap_bin_size_options, '16')

        self.palette_dropdown.setChoices(self.palette_options[0])
        self.shading_dropdown.setChoices(self.shading_options[0])
        self.bp_dropdown.setChoices(self.bp_names[0])
        self.max_time_scale_dropdown.setChoices(max_scales[0])
        self.bin_size_dropdown.setChoices('80×80')

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=("Helvetica", 14, 'bold'), pady=5, padx=5)
        self.heatmap_frames_var = BooleanVar()
        self.heatmap_videos_var = BooleanVar()
        self.heatmap_last_frm_var = BooleanVar()
        self.multiprocessing_var = BooleanVar()
        heatmap_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.heatmap_frames_var)
        heatmap_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.heatmap_videos_var)
        heatmap_last_frm_cb = Checkbutton(self.settings_frm, text='Create last frame', variable=self.heatmap_last_frm_var)
        self.multiprocess_cb = Checkbutton(self.settings_frm, text='Multiprocess videos (faster)', variable=self.multiprocessing_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocessing_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue',
                                           command=lambda: self.__create_heatmap_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command=lambda: self.__create_heatmap_plots(multiple_videos=False))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.palette_dropdown.grid(row=0, sticky=NW)
        self.shading_dropdown.grid(row=1, sticky=NW)
        self.bp_dropdown.grid(row=2, sticky=NW)
        self.max_time_scale_dropdown.grid(row=3, sticky=NW)
        self.bin_size_dropdown.grid(row=4, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        heatmap_frames_cb.grid(row=0, sticky=NW)
        heatmap_videos_cb.grid(row=1, sticky=NW)
        heatmap_last_frm_cb.grid(row=2, sticky=NW)
        self.multiprocess_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=NW)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        #self.main_frm.mainloop()

    def __create_heatmap_plots(self,
                               multiple_videos: bool):

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if self.max_time_scale_dropdown.getChoices() != 'Auto-compute':
            max_scale = int(self.max_time_scale_dropdown.getChoices().split('×')[0])
        else:
            max_scale = 'auto'

        bin_size = int(self.bin_size_dropdown.getChoices().split('×')[0])

        style_attr = {'palette': self.palette_dropdown.getChoices(),
                      'shading': self.shading_dropdown.getChoices(),
                      'max_scale': max_scale,
                      'bin_size': bin_size}

        if not self.multiprocessing_var.get():
            heatmapper_clf = HeatmapperLocationSingleCore(config_path=self.config_path,
                                                          style_attr=style_attr,
                                                          final_img_setting=self.heatmap_last_frm_var.get(),
                                                          video_setting=self.heatmap_videos_var.get(),
                                                          frame_setting=self.heatmap_frames_var.get(),
                                                          bodypart=self.bp_dropdown.getChoices(),
                                                          files_found=data_paths)

            heatmapper_clf_processor = multiprocessing.Process(heatmapper_clf.create_heatmaps())
            heatmapper_clf_processor.start()

        else:
            heatmapper_clf = HeatMapperLocationMultiprocess(config_path=self.config_path,
                                                            style_attr=style_attr,
                                                            final_img_setting=self.heatmap_last_frm_var.get(),
                                                            video_setting=self.heatmap_videos_var.get(),
                                                            frame_setting=self.heatmap_frames_var.get(),
                                                            bodypart=self.bp_dropdown.getChoices(),
                                                            files_found=data_paths,
                                                            core_cnt=int(self.multiprocess_dropdown.getChoices()))

            heatmapper_clf.create_heatmaps()

#_ = HeatmapLocationPopup(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

class QuickLineplotPopup(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='SIMPLE LINE PLOT')
        video_filepaths = find_all_videos_in_directory(directory=os.path.join(self.project_path, 'videos'))
        self.video_files = [os.path.basename(x) for x in video_filepaths]
        if len(self.video_files) == 0:
            raise NoFilesFoundError(msg='SIMBA ERROR: No files detected in the project_folder/videos directory.')

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


#_ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')



class ClfByROIPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='CLASSIFICATIONS BY ROI')


        body_part_menu = CreateLabelFrameWithIcon(parent=self.main_frm, header='Select body part', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        #body_part_menu = LabelFrame(self.main_frm, text='Select body part', padx=5, pady=5)
        ROI_menu = LabelFrame(self.main_frm, text='Select ROI(s)', padx=5, pady=5)
        classifier_menu = LabelFrame(self.main_frm, text='Select classifier(s)', padx=5, pady=5)
        measurements_menu = LabelFrame(self.main_frm, text='Select measurements', padx=5, pady=5)
        self.clf_roi_analyzer = ROIClfCalculator(config_path)
        self.total_time_var = BooleanVar()
        self.start_bouts_var = BooleanVar()
        self.end_bouts_var = BooleanVar()
        self.total_time_cb = Checkbutton(measurements_menu, text='Total time by ROI (s)', variable=self.total_time_var)
        self.start_bouts_cb = Checkbutton(measurements_menu, text='Started bouts by ROI (count)', variable=self.start_bouts_var)
        self.end_bouts_cb = Checkbutton(measurements_menu, text='Ended bouts by ROI (count)', variable=self.end_bouts_var)
        self.ROI_check_boxes_status_dict = {}
        self.clf_check_boxes_status_dict = {}

        for row_number, ROI in enumerate(self.clf_roi_analyzer.ROI_str_name_list):
            self.ROI_check_boxes_status_dict[ROI] = IntVar()
            ROI_check_button = Checkbutton(ROI_menu, text=ROI, variable=self.ROI_check_boxes_status_dict[ROI])
            ROI_check_button.grid(row=row_number, sticky=W)

        for row_number, clf_name in enumerate(self.clf_roi_analyzer.clf_names):
            self.clf_check_boxes_status_dict[clf_name] = IntVar()
            clf_check_button = Checkbutton(classifier_menu, text=clf_name,
                                           variable=self.clf_check_boxes_status_dict[clf_name])
            clf_check_button.grid(row=row_number, sticky=W)

        self.choose_bp = DropDownMenu(body_part_menu, 'Body part', self.clf_roi_analyzer.body_part_list, '12')
        self.choose_bp.setChoices(self.clf_roi_analyzer.body_part_list[0])
        self.choose_bp.grid(row=0, sticky=W)
        run_analysis_button = Button(self.main_frm, text='Analyze classifications in each ROI',command=lambda: self.run_clf_by_ROI_analysis())
        body_part_menu.grid(row=0, sticky=W, padx=10, pady=10)
        ROI_menu.grid(row=1, sticky=W, padx=10, pady=10)
        classifier_menu.grid(row=2, sticky=W, padx=10, pady=10)
        self.total_time_cb.grid(row=0, sticky=NW)
        self.start_bouts_cb.grid(row=1, sticky=NW)
        self.end_bouts_cb.grid(row=2, sticky=NW)
        measurements_menu.grid(row=3, sticky=W, padx=10, pady=10)
        run_analysis_button.grid(row=4, sticky=W, padx=10, pady=10)

    def run_clf_by_ROI_analysis(self):
        body_part_list = [self.choose_bp.getChoices()]
        ROI_dict_lists, behavior_list = defaultdict(list), []
        measurements_list = []
        for loop_val, ROI_entry in enumerate(self.ROI_check_boxes_status_dict):
            check_val = self.ROI_check_boxes_status_dict[ROI_entry]
            if check_val.get() == 1:
                shape_type = self.clf_roi_analyzer.ROI_str_name_list[loop_val].split(':')[0].replace(':', '')
                shape_name = self.clf_roi_analyzer.ROI_str_name_list[loop_val].split(':')[1][1:]
                ROI_dict_lists[shape_type].append(shape_name)

        for measurement_var, measurement_name in zip([self.total_time_var.get(), self.start_bouts_var.get(), self.end_bouts_var.get()], ['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)']):
            if measurement_var:
                measurements_list.append(measurement_name)


        for loop_val, clf_entry in enumerate(self.clf_check_boxes_status_dict):
            check_val = self.clf_check_boxes_status_dict[clf_entry]
            if check_val.get() == 1:
                behavior_list.append(self.clf_roi_analyzer.clf_names[loop_val])
        if len(ROI_dict_lists) == 0:
            raise NoChoosenROIError()
        if len(behavior_list) == 0:
            raise NoChoosenClassifierError()
        if len(measurements_list) == 0:
            raise NoChoosenMeasurementError()
        else:
            self.clf_roi_analyzer.run(ROI_dict_lists=ROI_dict_lists, behavior_list=behavior_list, body_part_list=body_part_list, measurements=measurements_list)

#_ = ClfByROIPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')


class FSTTCPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='FORWARD SPIKE TIME TILING COEFFICIENTS')
        fsttc_link_label = Label(self.main_frm, text='[Click here to learn about FSTTC]',cursor='hand2', fg='blue')
        fsttc_link_label.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md'))

        fsttc_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='FSTTC Settings', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.FSTTC.value)
        #fsttc_settings_frm = LabelFrame(self.main_frm,text='FSTTC Settings', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        graph_cb_var = BooleanVar()
        graph_cb = Checkbutton(fsttc_settings_frm,text='Create graph',variable=graph_cb_var)
        time_delta = Entry_Box(fsttc_settings_frm,'Time Delta','10', validation='numeric')
        behaviors_frm = LabelFrame(fsttc_settings_frm,text="Behaviors")
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(behaviors_frm, text=clf, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, sticky=NW)

        fsttc_run_btn = Button(self.main_frm,text='Calculate FSTTC',command=lambda:self.run_fsttc(time_delta=time_delta.entry_get, graph_var= graph_cb_var.get(), behaviours_dict=clf_var_dict))

        fsttc_settings_frm.grid(row=0,sticky=W,pady=5)
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
            raise CountError(msg='SIMBA ERROR: FORWARD SPIKE TIME TILING COEFFICIENTS REQUIRE 2 OR MORE BEHAVIORS.')

        FSTCC_performer = FSTTCPerformer(config_path=self.config_path,
                                         time_window=time_delta,
                                         behavior_lst=targets,
                                         create_graphs=graph_var)
        FSTCC_performer.find_sequences()
        FSTCC_performer.calculate_FSTTC()
        FSTCC_performer.save_FSTTC()
        FSTCC_performer.plot_FSTTC()

#_ = FSTTCPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class KleinbergPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='APPLY KLEINBERG BEHAVIOR CLASSIFICATION SMOOTHING')
        kleinberg_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='Kleinberg Settings', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.KLEINBERG.value)
        #kleinberg_settings_frm = LabelFrame(self.main_frm,text='Kleinberg Settings', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')



        self.k_sigma = Entry_Box(kleinberg_settings_frm,'Sigma','10')
        self.k_sigma.entry_set('2')
        self.k_gamma = Entry_Box(kleinberg_settings_frm,'Gamma','10')
        self.k_gamma.entry_set('0.3')
        self.k_hierarchy = Entry_Box(kleinberg_settings_frm,'Hierarchy','10')
        self.k_hierarchy.entry_set('1')
        self.h_search_lbl = Label(kleinberg_settings_frm, text="Hierarchical search: ")
        self.h_search_lbl_val = BooleanVar()
        self.h_search_lbl_val.set(False)
        self.h_search_lbl_val_cb = Checkbutton(kleinberg_settings_frm, variable=self.h_search_lbl_val)
        kleinberg_table_frame = LabelFrame(self.main_frm, text='Choose classifier(s) to apply Kleinberg smoothing')
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(kleinberg_table_frame, text=clf, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, sticky=NW)

        run_kleinberg_btn = Button(self.main_frm, text='Apply Kleinberg Smoother', command=lambda: self.run_kleinberg(behaviors_dict=clf_var_dict, hierarchical_search=self.h_search_lbl_val.get()))

        kleinberg_settings_frm.grid(row=0,sticky=W,padx=10)
        self.k_sigma.grid(row=0,sticky=W)
        self.k_gamma.grid(row=1,sticky=W)
        self.k_hierarchy.grid(row=2,sticky=W)
        self.h_search_lbl.grid(row=3, column=0, sticky=W)
        self.h_search_lbl_val_cb.grid(row=3, column=1, sticky=W)
        kleinberg_table_frame.grid(row=1,pady=10,padx=10)
        run_kleinberg_btn.grid(row=2)

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
        check_float(name='Sigma', value=self.k_sigma.entry_get)
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


#_ = KleinbergPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class TimeBinsClfPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        super().__init__(config_path=config_path, title='CLASSIFICATION BY TIME BINS')

        cbox_titles = Options.TIMEBINS_MEASURMENT_OPTIONS.value
        self.timebin_entrybox = Entry_Box(self.main_frm, 'Set time bin size (s)', '15', validation='numeric')
        measures_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='MEASUREMENTS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        #measures_frm = LabelFrame(self.main_frm, text='MEASUREMENTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        clf_frm = LabelFrame(self.main_frm, text='CLASSIFIERS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(measures_frm, text=title, variable=self.measurements_var_dict[title])
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(self.clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name])
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(self.main_frm, text='Run', command=lambda: self.run_time_bins_clf())
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
            raise NoChoosenMeasurementError()
        if len(clf_list) == 0:
            raise NoChoosenClassifierError()
        time_bins_clf_analyzer = TimeBinsClf(config_path=self.config_path, bin_length=int(self.timebin_entrybox.entry_get), measurements=measurement_lst, classifiers=clf_list)
        time_bins_clf_multiprocessor = multiprocessing.Process(target=time_bins_clf_analyzer.analyze_timebins_clf())
        time_bins_clf_multiprocessor.start()

#_ = TimeBinsClfPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class ClfDescriptiveStatsPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        super().__init__(config_path=config_path, title='ANALYZE CLASSIFICATIONS: DESCRIPTIVE STATISTICS')
        measures_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='MEASUREMENTS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
       #measures_frm = LabelFrame(self.main_frm, text='MEASUREMENTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        clf_frm = LabelFrame(self.main_frm, text='CLASSIFIERS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        cbox_titles = Options.CLF_DESCRIPTIVES_OPTIONS.value
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(measures_frm, text=title, variable=self.measurements_var_dict[title])
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(self.clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name])
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(self.main_frm, text='Run', command=lambda: self.run_descriptive_analysis())
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
            raise NoChoosenMeasurementError()
        if len(clf_list) == 0:
            raise NoChoosenClassifierError()
        data_log_analyzer = ClfLogCreator(config_path=self.config_path, data_measures=measurement_lst, classifiers=clf_list)
        data_log_analyzer.run()
        data_log_analyzer.save()

#_ = ClfDescriptiveStatsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class DownsampleVideoPopUp(PopUpMixin):
    def __init__(self):

        super().__init__(title='DOWN-SAMPLE VIDEO RESOLUTION')
        instructions = Label(self.main_frm, text='Choose only one of the following method (Custom or Default)')
        choose_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SELECT VIDEO', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DOWNSAMPLE.value)
        #choose_video_frm = LabelFrame(self.main_frm, text='SELECT VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black', padx=5, pady=5)
        self.video_path_selected = FileSelect(choose_video_frm, "Video path", title='Select a video file')
        custom_frm = LabelFrame(self.main_frm, text='Custom resolution', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black', padx=5, pady=5)
        self.entry_width = Entry_Box(custom_frm, 'Width', '10', validation='numeric')
        self.entry_height = Entry_Box(custom_frm, 'Height', '10', validation='numeric')

        self.custom_downsample_btn = Button(custom_frm, text='Downsample to custom resolution', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black', command=lambda: self.custom_downsample())
        default_frm = LabelFrame(self.main_frm, text='Default resolution', font='bold', padx=5, pady=5)
        self.radio_btns = {}
        self.var = StringVar()
        for custom_cnt, resolution_radiobtn in enumerate(self.resolutions):
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
        self.default_downsample_btn.grid(row=len(self.resolutions)+1, column=0, sticky=NW)

    def custom_downsample(self):
        width = self.entry_width.entry_get
        height = self.entry_height.entry_get
        check_int(name='Video width', value=width)
        check_int(name='Video height', value=height)
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(file_path=self.video_path_selected.file_path, video_width=int(width), video_height=int(height))

    def default_downsample(self):
        resolution = self.var.get()
        width, height = resolution.split('×', 2)[0].strip(), resolution.split('×', 2)[1].strip()
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(file_path=self.video_path_selected.file_path, video_width=int(width), video_height=int(height))

#_ = DownsampleVideoPopUp()

class CLAHEPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CLAHE VIDEO CONVERSION')
        clahe_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='Contrast Limited Adaptive Histogram Equalization', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #clahe_frm = LabelFrame(self.main_frm, text='Contrast Limited Adaptive Histogram Equalization', font='bold', padx=5, pady=5)
        selected_video = FileSelect(clahe_frm, "Video path ", title='Select a video file')
        button_clahe = Button(clahe_frm, text='Apply CLAHE', command=lambda: clahe_enhance_video(file_path=selected_video.file_path))
        clahe_frm.grid(row=0,sticky=W)
        selected_video.grid(row=0,sticky=W)
        button_clahe.grid(row=1,pady=5)
        #self.main_frm.mainloop()

#_ = CLAHEPopUp()

class CropVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CROP SINGLE VIDEO')
        crop_video_lbl_frm = LabelFrame(self.main_frm, text='Crop Video',font='bold',padx=5,pady=5)
        selected_video = FileSelect(crop_video_lbl_frm,"Video path",title='Select a video file', lblwidth=20)
        button_crop_video_single = Button(crop_video_lbl_frm, text='Crop Video',command=lambda: crop_single_video(file_path=selected_video.file_path))

        crop_video_lbl_frm_multiple = LabelFrame(self.main_frm, text='Fixed coordinates crop for multiple videos', font='bold',  padx=5, pady=5)
        input_folder = FolderSelect(crop_video_lbl_frm_multiple, 'Video directory:', title='Select Folder with videos', lblwidth=20)
        output_folder = FolderSelect(crop_video_lbl_frm_multiple, 'Output directory:', title='Select a folder for your output videos', lblwidth=20)
        button_crop_video_multiple = Button(crop_video_lbl_frm_multiple, text='Confirm', command=lambda: crop_multiple_videos(directory_path=input_folder.folder_path, output_path=output_folder.folder_path))

        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        button_crop_video_single.grid(row=1, sticky=NW, pady=10)
        crop_video_lbl_frm_multiple.grid(row=1, sticky=W, pady=10, padx=5)
        input_folder.grid(row=0,sticky=W,pady=5)
        output_folder.grid(row=1,sticky=W,pady=5)
        button_crop_video_multiple.grid(row=2,sticky=W,pady=5)

#_ = CropVideoPopUp()

class ClipVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CLIP VIDEO')
        selected_video = CreateLabelFrameWithIcon(parent=self.main_frm, header='Video path', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #selected_video = FileSelect(self.main_frm, "Video path", title='Select a video file')
        method_1_frm = LabelFrame(self.main_frm, text='Method 1', font='bold', padx=5, pady=5)
        label_set_time_1 = Label(method_1_frm, text='Please enter the time frame in hh:mm:ss format')
        start_time = Entry_Box(method_1_frm, 'Start at (s):', '8', validation='numeric')
        end_time = Entry_Box(method_1_frm, 'End at (s):', '8', validation='numeric')
        CreateToolTip(method_1_frm, 'Method 1 will retrieve the specified time input. (eg: input of Start at: 00:00:00, End at: 00:01:00, will create a new video from the chosen video from the very start till it reaches the first minute of the video)')
        method_2_frm = LabelFrame(self.main_frm, text='Method 2', font='bold', padx=5, pady=5)
        method_2_time = Entry_Box(method_2_frm, 'Seconds:', '8', validation='numeric')
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
        #self.main_frm.mainloop()
#_ = ClipVideoPopUp()

class MultiShortenPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CLIP VIDEO INTO MULTIPLE VIDEOS')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='Split videos into different parts', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #settings_frm = LabelFrame(self.main_frm, text='Split videos into different parts', font='bold', padx=5, pady=5)
        self.selected_video = FileSelect(settings_frm, "Video path", title='Select a video file', lblwidth=15)
        self.clip_cnt = Entry_Box(settings_frm, '# of clips', '15', validation='numeric')
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

        run_button = Button(self.table, text='Clip video', command=lambda: self.run_clipping(), fg='navy', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        run_button.grid(row=int(self.clip_cnt.entry_get) + 2, column=2, sticky=W)

    def run_clipping(self):
        start_times, end_times = [], []
        check_file_exist_and_readable(self.selected_video.file_path)
        for start_time, end_time in zip(self.start_times, self.end_times):
            start_times.append(start_time.get())
            end_times.append(end_time.get())
        multi_split_video(file_path=self.selected_video.file_path, start_times=start_times, end_times=end_times)

#_ = MultiShortenPopUp()

class ChangeImageFormatPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CHANGE IMAGE FORMAT')

        self.input_folder_selected = FolderSelect(self.main_frm, "Image directory", title='Select folder with images:')
        set_input_format_frm = LabelFrame(self.main_frm, text='Original image format', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=15, pady=5)
        set_output_format_frm = LabelFrame(self.main_frm, text='Output image format', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=15, pady=5)

        self.input_file_type, self.out_file_type = StringVar(), StringVar()
        input_png_rb = Radiobutton(set_input_format_frm, text=".png", variable=self.input_file_type, value="png")
        input_jpeg_rb = Radiobutton(set_input_format_frm, text=".jpg", variable=self.input_file_type, value="jpg")
        input_bmp_rb = Radiobutton(set_input_format_frm, text=".bmp", variable=self.input_file_type, value="bmp")
        output_png_rb = Radiobutton(set_output_format_frm, text=".png", variable=self.out_file_type, value="png")
        output_jpeg_rb = Radiobutton(set_output_format_frm, text=".jpg", variable=self.out_file_type, value="jpg")
        output_bmp_rb = Radiobutton(set_output_format_frm, text=".bmp", variable=self.out_file_type, value="bmp")
        run_btn = Button(self.main_frm, text='Convert image file format', command= lambda: self.run_img_conversion())
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
            raise NoFilesFoundError(msg='SIMBA ERROR: The input folder {} contains ZERO files.'.format(self.input_folder_selected.folder_path))
        change_img_format(directory=self.input_folder_selected.folder_path, file_type_in=self.input_file_type.get(), file_type_out=self.out_file_type.get())


class ConvertVideoPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CONVERT VIDEO FORMAT")

        convert_multiple_videos_frm = CreateLabelFrameWithIcon(parent=main_frm, header='Convert multiple videos', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #convert_multiple_videos_frm = LabelFrame(main_frm, text='Convert multiple videos', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5)
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
            raise FrameRangeError(msg='SIMBA ERROR: The end frame ({}) cannot come before the start frame ({})'.format(str(end_frame), str(start_frame)))
        video_meta_data = get_video_meta_data(video_path=self.video_file_selected.file_path)
        if int(start_frame) > video_meta_data['frame_count']:
            raise FrameRangeError(msg='SIMBA ERROR: The start frame ({}) is larger than the number of frames in the video ({})'.format(str(start_frame), str(video_meta_data['frame_count'])))
        if int(end_frame) > video_meta_data['frame_count']:
            raise FrameRangeError(msg='SIMBA ERROR: The end frame ({}) is larger than the number of frames in the video ({})'.format(str(end_frame), str(video_meta_data['frame_count'])))
        extract_frame_range(file_path=self.video_file_selected.file_path, start_frame=int(start_frame), end_frame=int(end_frame))

class ExtractAllFramesPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("EXTRACT ALL FRAMES")

        single_video_frm = CreateLabelFrameWithIcon(parent=main_frm, header='Single video', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #single_video_frm = LabelFrame(main_frm, text='Single video', padx=5, pady=5, font='bold')
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
        fps_entry_box = Entry_Box(main_frm, 'Output FPS:', '10', validation='numeric')
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
        fps_entry = Entry_Box(main_frm, 'Output FPS: ', '10', validation='numeric')
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
        settings_frm = LabelFrame(main_frm, text='SETTINGS', padx=5, pady=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
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
        settings_frm = CreateLabelFrameWithIcon(parent=main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #settings_frm = LabelFrame(main_frm, text='SETTINGS', padx=5, pady=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
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
        check_file_exist_and_readable(file_path=self.video_path.file_path)
        check_int(name='Distance', value=self.known_distance.entry_get, min_value=1)
        _ = get_video_meta_data(video_path=self.video_path.file_path)
        mm_cnt = get_coordinates_nilsson(self.video_path.file_path, self.known_distance.entry_get)
        print(f'1 PIXEL REPRESENTS {round(mm_cnt, 4)} MILLIMETERS IN VIDEO {os.path.basename(self.video_path.file_path)}.')

class MakePathPlotPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(200, 200)
        main_frm.wm_title("CREATE PATH PLOT")
        settings_frm = CreateLabelFrameWithIcon(parent=main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #settings_frm = LabelFrame(main_frm)
        video_path = FileSelect(settings_frm, 'VIDEO PATH: ', lblwidth='30')
        body_part = Entry_Box(settings_frm, 'BODY PART: ', '30')
        data_path = FileSelect(settings_frm, 'DATA PATH (e.g., H5 or CSV file): ', lblwidth='30')
        color_lst = list(get_color_dict().keys())
        background_color = DropDownMenu(settings_frm,'BACKGROUND COLOR: ',color_lst,'18')
        background_color.setChoices(choice='White')
        line_color = DropDownMenu(settings_frm, 'LINE COLOR: ', color_lst, '18')
        line_color.setChoices(choice='Red')
        line_thickness = DropDownMenu(settings_frm, 'LINE THICKNESS: ', list(range(1, 11)), '18')
        line_thickness.setChoices(choice=1)
        circle_size = DropDownMenu(settings_frm, 'CIRCLE SIZE: ', list(range(1, 11)), '18')
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
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
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
        self.table = LabelFrame(self.main_frm, text='SET NEW ORDER', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
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
        img = PhotoImage(file=os.path.join(scriptdir, Paths.ABOUT_ME.value))
        canvas.create_image(0,0,image=img,anchor='nw')
        canvas.image = img

class ConcatenatingVideosPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 300)
        main_frm.wm_title("CONCATENATE VIDEOS")
        settings_frm = CreateLabelFrameWithIcon(parent=main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #settings_frm = LabelFrame(main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        video_path_1 = FileSelect(settings_frm, "First video path: ", title='Select a video file')
        video_path_2 = FileSelect(settings_frm, "Second video path: ", title='Select a video file')
        resolutions = ['Video 1', 'Video 2', 320, 640, 720, 1280, 1980]
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

class VisualizePoseInFolderPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(350, 200)
        self.main_frm.wm_title('Visualize pose-estimation')
        settings_frame = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #settings_frame = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.input_folder = FolderSelect(settings_frame, 'Input directory (with csv/parquet files)', title='Select input folder')
        self.output_folder = FolderSelect(settings_frame, 'Output directory (where your videos will be saved)', title='Select output folder')
        self.circle_size = Entry_Box(settings_frame, 'Circle size', 0, validation='numeric')
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


class DropTrackingDataPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(500, 800)
        self.main_frm.wm_title('Drop body-parts in pose-estimation data')
        self.main_frm.lift()
        self.main_frm = Canvas(hxtScrollbar(self.main_frm))
        self.main_frm.pack(fill="both", expand=True)
        file_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='FILE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        #file_settings_frm = LabelFrame(self.main_frm, text='FILE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.data_folder_path = FolderSelect(file_settings_frm, 'Data Folder', lblwidth='20')
        self.file_format = DropDownMenu(file_settings_frm, 'File Type', ['csv', 'h5'], '20')
        self.pose_tool = DropDownMenu(file_settings_frm, 'Tracking tool', ['DLC', 'maDLC'], '20')
        self.pose_tool.setChoices('DLC')
        self.file_format.setChoices('csv')
        self.bp_cnt = Entry_Box(file_settings_frm, '# body-parts to remove', '20', validation='numeric')
        confirm_btn = Button(file_settings_frm, text='Confirm', command=lambda: self.confirm())
        file_settings_frm.grid(row=0,sticky=NW)
        self.data_folder_path.grid(row=0,sticky=W,columnspan=3)
        self.pose_tool.grid(row=1,sticky=NW)
        self.file_format.grid(row=2,sticky=NW)
        self.bp_cnt.grid(row=3,sticky=NW)
        confirm_btn.grid(row=3,column=1,sticky=NW)

    def confirm(self):
        if hasattr(self, 'bp_table'):
            self.bp_table.destroy()
        self.keypoint_remover = KeypointRemover(data_folder=self.data_folder_path.folder_path, pose_tool=self.pose_tool.getChoices(), file_format=self.file_format.getChoices())
        self.bp_table = LabelFrame(self.main_frm, text='REMOVE BODY-PARTS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.bp_table.grid(row=1, sticky=NW, pady=5)
        self.animal_names_lst, self.drop_down_list = [], []
        if self.pose_tool.getChoices() == 'DLC':
            for bp_number in range(int(self.bp_cnt.entry_get)):
                bp_drop_down = DropDownMenu(self.bp_table, 'Body-part {}:'.format(str(bp_number + 1)), self.keypoint_remover.body_part_names, '10')
                bp_drop_down.setChoices(self.keypoint_remover.body_part_names[bp_number])
                self.drop_down_list.append(bp_drop_down)
                bp_drop_down.grid(row=bp_number, column=0, sticky=NW)
        if self.pose_tool.getChoices()  == 'maDLC':
            for bp_number in range(int(self.bp_cnt.entry_get)):
                animal_drop_down = DropDownMenu(self.bp_table, 'Animal name:', self.keypoint_remover.animal_names, '10')
                animal_drop_down.setChoices(self.keypoint_remover.animal_names[0])
                self.animal_names_lst.append(animal_drop_down)
                bp_drop_down = DropDownMenu(self.bp_table, 'Body-part {}:'.format(str(bp_number + 1)), self.keypoint_remover.body_part_names, '10')
                bp_drop_down.setChoices(self.keypoint_remover.body_part_names[bp_number])
                self.drop_down_list.append(bp_drop_down)
                animal_drop_down.grid(row=bp_number, column=0, sticky=NW)
                bp_drop_down.grid(row=bp_number, column=1, sticky=NW)

        run_btn = Button(self.main_frm, text='RUN BODY-PART REMOVAL', command=lambda: self.run())
        run_btn.grid(row=2, column=0, sticky=NW)

    def run(self):
        bp_to_remove_list, animal_names_list = [], []
        for number, drop_down in enumerate(self.drop_down_list):
            bp_to_remove_list.append(drop_down.getChoices())
        if self.pose_tool == 'maDLC':
            for number, drop_down in enumerate(self.animal_names_lst):
                animal_names_list.append(drop_down.getChoices())
        _ = self.keypoint_remover.run_bp_removal(bp_to_remove_list=bp_to_remove_list, animal_names=animal_names_list)

class ConcatenatorPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str or None):
        super().__init__(config_path=config_path, title='MERGE (CONCATENATE) VIDEOS')
        self.config_path = config_path
        self.select_video_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='VIDEOS #', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CONCAT_VIDEOS.value)
        #self.select_video_cnt_frm = LabelFrame(self.main_frm, text='VIDEOS #', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.select_video_cnt_dropdown = DropDownMenu(self.select_video_cnt_frm, 'VIDEOS #', list(range(2,21)), '15')
        self.select_video_cnt_dropdown.setChoices(2)
        self.select_video_cnt_btn = Button(self.select_video_cnt_frm, text='SELECT', command=lambda: self.populate_table())
        self.select_video_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_btn.grid(row=0, column=1, sticky=NW)

    def populate_table(self):
        if hasattr(self, 'video_table_frm'):
            self.video_table_frm.destroy()
            self.join_type_frm.destroy()
        self.video_table_frm = LabelFrame(self.main_frm, text='VIDEO PATHS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.video_table_frm.grid(row=1, sticky=NW)
        self.join_type_frm = LabelFrame(self.main_frm, text='JOIN TYPE', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.join_type_frm.grid(row=2, sticky=NW)
        self.videos_dict = {}
        for cnt in range(int(self.select_video_cnt_dropdown.getChoices())):
            self.videos_dict[cnt] = FileSelect(self.video_table_frm, "Video {}: ".format(str(cnt+1)), title='Select a video file')
            self.videos_dict[cnt].grid(row=cnt, column=0, sticky=NW)

        self.join_type_var = StringVar()
        self.icons_dict = {}
        simba_dir = os.path.dirname(simba.__file__)
        icon_assets_dir = os.path.join(simba_dir, Paths.ICON_ASSETS.value)
        concat_icon_dir = os.path.join(icon_assets_dir, 'concat_icons')
        for file_cnt, file_path in enumerate(glob.glob(concat_icon_dir + '/*')):
            _, file_name, _ = get_fn_ext(file_path)
            self.icons_dict[file_name] = {}
            self.icons_dict[file_name]['img'] = ImageTk.PhotoImage(Image.open(file_path))
            self.icons_dict[file_name]['btn'] = Radiobutton(self.join_type_frm, text=file_name, variable=self.join_type_var, value=file_name)
            self.icons_dict[file_name]['btn'].config(image=self.icons_dict[file_name]['img'])
            self.icons_dict[file_name]['btn'].image = self.icons_dict[file_name]['img']
            self.icons_dict[file_name]['btn'].grid(row=0, column=file_cnt, sticky=NW)
        self.join_type_var.set(value='mosaic')
        self.resolution_frm = LabelFrame(self.main_frm, text='RESOLUTION', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.resolution_width = DropDownMenu(self.resolution_frm, 'Width', ['480', '640', '1280', '1920', '2560'], '15')
        self.resolution_width.setChoices('640')
        self.resolution_height = DropDownMenu(self.resolution_frm, 'Height', ['480', '640', '1280', '1920', '2560'], '15')
        self.resolution_height.setChoices('480')
        self.resolution_frm.grid(row=3, column=0, sticky=NW)
        self.resolution_width.grid(row=0, column=0, sticky=NW)
        self.resolution_height.grid(row=1, column=0, sticky=NW)

        run_btn = Button(self.main_frm, text='RUN', command=lambda: self.run())
        run_btn.grid(row=4, column=0, sticky=NW)

    def run(self):
        videos_info = {}
        for cnt, (video_name, video_data) in enumerate(self.videos_dict.items()):
            _ = get_video_meta_data(video_path=video_data.file_path)
            videos_info['Video {}'.format(str(cnt+1))] = video_data.file_path

        if (len(videos_info.keys()) < 3) & (self.join_type_var.get() == 'mixed_mosaic'):
            raise MixedMosaicError(msg='Ff using the mixed mosaic join type, please tick check-boxes for at least three video types.')
        if (len(videos_info.keys()) < 3) & (self.join_type_var.get() == 'mosaic'):
            self.join_type_var.set(value='vertical')


        _ = FrameMergererFFmpeg(config_path=self.config_path,
                                frame_types=videos_info,
                                video_height=int(self.resolution_height.getChoices()),
                                video_width=int(self.resolution_width.getChoices()),
                                concat_type=self.join_type_var.get())

class SetMachineModelParameters(PopUpMixin):

    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='SET MODEL PARAMETERS')
        self.clf_table_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.SET_RUN_ML_PARAMETERS.value)
        Label(self.clf_table_frm, text='CLASSIFIER', font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(row=0, column=0)
        Label(self.clf_table_frm, text='MODEL PATH (.SAV)', font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(row=0, column=1, sticky=NW)
        Label(self.clf_table_frm, text='DISCRIMINATION THRESHOLD', font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(row=0, column=2, sticky=NW)
        Label(self.clf_table_frm, text='MINIMUM BOUT LENGTH (MS)', font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(row=0, column=3, sticky=NW)
        self.clf_data = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.clf_data[clf_name] = {}
            Label(self.clf_table_frm, text=clf_name, font=Formats.LABELFRAME_HEADER_FORMAT.value).grid(row=clf_cnt+1, column=0, sticky=NW)
            self.clf_data[clf_name]['path'] = FileSelect(self.clf_table_frm, title='Select model (.sav) file')
            self.clf_data[clf_name]['threshold'] = Entry_Box(self.clf_table_frm, '', '0')
            self.clf_data[clf_name]['min_bout'] = Entry_Box(self.clf_table_frm, '', '0')
            self.clf_data[clf_name]['path'].grid(row=clf_cnt+1, column=1, sticky=NW)
            self.clf_data[clf_name]['threshold'].grid(row=clf_cnt+1, column=2, sticky=NW)
            self.clf_data[clf_name]['min_bout'].grid(row=clf_cnt + 1, column=3, sticky=NW)

        set_btn = Button(self.main_frm, text='SET MODEL(S)', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='red', command = lambda:self.set())
        self.clf_table_frm.grid(row=0,sticky=W,pady=5,padx=5)
        set_btn.grid(row=1,pady=10)

    def set(self):
        for model_name, model_settings in self.clf_data.items():
            check_file_exist_and_readable(model_settings['path'].file_path)
            check_float(name='Classifier {} threshhold'.format(model_name), value=model_settings['threshold'].entry_get, max_value=1.0, min_value=0.0)
            check_int(name='Classifier {} minimum bout'.format(model_name), value=model_settings['min_bout'].entry_get, min_value=0.0)

        for cnt, (model_name, model_settings) in enumerate(self.clf_data.items()):
            self.config.set('SML settings', 'model_path_{}'.format(str(cnt+1)), model_settings['path'].file_path)
            self.config.set('threshold_settings', 'threshold_{}'.format(str(cnt+1)), model_settings['threshold'].entry_get)
            self.config.set('Minimum_bout_lengths', 'min_bout_{}'.format(str(cnt+1)), model_settings['min_bout'].entry_get)

        with open(self.config_path, 'w') as f:
            self.config.write(f)

        print('SIMBA COMPLETE: Model paths/settings saved in project_config.ini')

#_ = SetMachineModelParameters(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class OutlierSettingsPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        super().__init__(config_path=config_path, title='OUTLIER SETTINGS')
        self.animal_bps = {}
        for animal_name, animal_data in self.animal_bp_dict.items(): self.animal_bps[animal_name] = [x[:-2] for x in animal_data['X_bps']]

        self.location_correction_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='LOCATION CORRECTION', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OULIERS.value)
        #self.location_correction_frm = LabelFrame(self.main_frm, text='LOCATION CORRECTION', font=('Times', 12, 'bold'), pady=5, padx=5)

        bp_entry_cnt, self.criterion_dropdowns = 0, {}
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name] = {}
            self.criterion_dropdowns[animal_name]['location_bp_1'] = DropDownMenu(self.location_correction_frm, 'Choose {} body part 1:'.format(animal_name), self.animal_bps[animal_name], '30')
            self.criterion_dropdowns[animal_name]['location_bp_2'] = DropDownMenu(self.location_correction_frm, 'Choose {} body part 2:'.format(animal_name), self.animal_bps[animal_name], '30')
            self.criterion_dropdowns[animal_name]['location_bp_1'].setChoices(self.animal_bps[animal_name][0])
            self.criterion_dropdowns[animal_name]['location_bp_2'].setChoices(self.animal_bps[animal_name][1])
            self.criterion_dropdowns[animal_name]['location_bp_1'].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt+=1
            self.criterion_dropdowns[animal_name]['location_bp_2'].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt+=1
        self.location_criterion = Entry_Box(self.location_correction_frm, 'Location criterion: ', '15')
        self.location_criterion.grid(row=bp_entry_cnt, column=0, sticky=NW)
        self.location_correction_frm.grid(row=0, column=0, sticky=NW)

        self.movement_correction_frm = LabelFrame(self.main_frm, text='MOVEMENT CORRECTION', font=('Times', 12, 'bold'), pady=5, padx=5)
        bp_entry_cnt = 0
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name]['movement_bp_1'] = DropDownMenu(self.movement_correction_frm, 'Choose {} body part 1:'.format(animal_name), self.animal_bps[animal_name], '30')
            self.criterion_dropdowns[animal_name]['movement_bp_2'] = DropDownMenu(self.movement_correction_frm, 'Choose {} body part 2:'.format(animal_name), self.animal_bps[animal_name], '30')
            self.criterion_dropdowns[animal_name]['movement_bp_1'].setChoices(self.animal_bps[animal_name][0])
            self.criterion_dropdowns[animal_name]['movement_bp_2'].setChoices(self.animal_bps[animal_name][1])
            self.criterion_dropdowns[animal_name]['movement_bp_1'].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt+=1
            self.criterion_dropdowns[animal_name]['movement_bp_2'].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt+=1
        self.movement_criterion = Entry_Box(self.movement_correction_frm, 'Location criterion: ', '15')
        self.movement_criterion.grid(row=bp_entry_cnt, column=0, sticky=NW)
        self.movement_correction_frm.grid(row=1, column=0, sticky=NW)

        agg_type_frm = LabelFrame(self.main_frm, text='AGGREGATION METHOD', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.agg_type_dropdown = DropDownMenu(agg_type_frm, 'Aggregation method:', ['mean', 'median'], '15')
        self.agg_type_dropdown.setChoices('median')
        self.agg_type_dropdown.grid(row=0, column=0, sticky=NW)
        agg_type_frm.grid(row=2, column=0, sticky=NW)

        run_btn = Button(self.main_frm, text='CONFIRM', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='red', command=lambda: self.run())
        run_btn.grid(row=3, column=0, sticky=NW)

    def run(self):
        check_float(name='LOCATION CRITERION', value=self.location_criterion.entry_get, min_value=0.0)
        check_float(name='MOVEMENT CRITERION', value=self.movement_criterion.entry_get, min_value=0.0)
        self.config.set('Outlier settings', 'movement_criterion', str(self.movement_criterion.entry_get))
        self.config.set('Outlier settings', 'location_criterion', str(self.location_criterion.entry_get))
        self.config.set('Outlier settings', 'mean_or_median', str(self.agg_type_dropdown.getChoices()))
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.config.set('Outlier settings', 'movement_bodyPart1_{}'.format(animal_name), self.criterion_dropdowns[animal_name]['movement_bp_1'].getChoices())
            self.config.set('Outlier settings', 'movement_bodyPart2_{}'.format(animal_name), self.criterion_dropdowns[animal_name]['movement_bp_2'].getChoices())
            self.config.set('Outlier settings', 'location_bodyPart1_{}'.format(animal_name), self.criterion_dropdowns[animal_name]['location_bp_1'].getChoices())
            self.config.set('Outlier settings', 'location_bodyPart2_{}'.format(animal_name), self.criterion_dropdowns[animal_name]['location_bp_2'].getChoices())
        with open(self.config_path, 'w') as f:
            self.config.write(f)

        print('SIMBA COMPLETE: Outlier correction settings updated in the project_config.ini')

# _ = OutlierSettingsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class RemoveAClassifierPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='Warning: Remove classifier(s) settings')
        self.remove_clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SELECT A CLASSIFIER TO REMOVE', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.REMOVE_CLF.value)
        #self.remove_clf_frm = LabelFrame(self.main_frm, text='SELECT A CLASSIFIER TO REMOVE')
        self.clf_dropdown = DropDownMenu(self.remove_clf_frm, 'Classifier', self.clf_names, '12')
        self.clf_dropdown.setChoices(self.clf_names[0])

        run_btn = Button(self.main_frm,text='REMOVE CLASSIFIER',command=lambda:self.run())
        self.remove_clf_frm.grid(row=0,sticky=W)
        self.clf_dropdown.grid(row=0,sticky=W)
        run_btn.grid(row=1,pady=10)

    def run(self):
        for i in range(len(self.clf_names)):
            self.config.remove_option('SML settings', 'model_path_{}'.format(str(i+1)))
            self.config.remove_option('SML settings', 'target_name_{}'.format(str(i+1)))
            self.config.remove_option('threshold_settings', 'threshold_{}'.format(str(i+1)))
            self.config.remove_option('Minimum_bout_lengths', 'min_bout_{}'.format(str(i+1)))
        self.clf_names.remove(self.clf_dropdown.getChoices())
        self.config.set('SML settings', 'no_targets', str(len(self.clf_names)))

        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.config.set('SML settings', 'model_path_{}'.format(str(clf_cnt+1)), '')
            self.config.set('SML settings', 'target_name_{}'.format(str(clf_cnt+1)), clf_name)
            self.config.set('threshold_settings', 'threshold_{}'.format(str(clf_cnt+1)), 'None')
            self.config.set('Minimum_bout_lengths', 'min_bout_{}'.format(str(clf_cnt+1)), 'None')

        with open(self.config_path, 'w') as f:
            self.config.write(f)

        print('SIMBA COMPLETE: {} classifier removed from SimBA project.'.format(self.clf_dropdown.getChoices()))

#_ = RemoveAClassifierPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class VisualizeROIFeaturesPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='VISUALIZE ROI FEATURES')
        self.video_list = []
        video_file_paths = find_files_of_filetypes_in_directory(directory=self.videos_dir, extensions=['.mp4', '.avi'])
        for file_path in video_file_paths:
            _, video_name, ext = get_fn_ext(filepath=file_path)
            self.video_list.append(video_name + ext)

        if len(self.video_list) == 0:
            raise NoFilesFoundError(msg='SIMBA ERROR: No videos in SimBA project. Import videos into you SimBA project to visualize ROI features.')


        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_FEATURES_PLOT.value)
        #self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
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
                roi_feature_visualizer.create_visualization()
            else:
                roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path=self.config_path, video_name=self.single_video_dropdown.getChoices(), style_attr=style_attr, core_cnt=int(self.multiprocess_dropdown.getChoices()))
                roi_feature_visualizer_mp = multiprocessing.Process(target=roi_feature_visualizer.create_visualization())
                roi_feature_visualizer_mp.start()
        else:
            if not self.multiprocess_var.get():
                for video_name in self.video_list:
                    roi_feature_visualizer = ROIfeatureVisualizer(config_path=self.config_path, video_name=video_name, style_attr=style_attr)
                    roi_feature_visualizer.create_visualization()
            else:
                for video_name in self.video_list:
                    roi_feature_visualizer = ROIfeatureVisualizerMultiprocess(config_path=self.config_path, video_name=video_name, style_attr=style_attr, core_cnt=int(self.multiprocess_dropdown.getChoices()))
                    roi_feature_visualizer_mp = multiprocessing.Process(target=roi_feature_visualizer.create_visualization())
                    roi_feature_visualizer_mp.start()

#_ = VisualizeROIFeaturesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')



class VisualizeROITrackingPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='VISUALIZE ROI TRACKING')
        self.video_list = []
        video_file_paths = find_files_of_filetypes_in_directory(directory=self.videos_dir, extensions=['.mp4', '.avi'])

        for file_path in video_file_paths:
            _, video_name, ext = get_fn_ext(filepath=file_path)
            self.video_list.append(video_name + ext)

        check_if_filepath_list_is_empty(filepaths=self.video_list, error_msg='SIMBA ERROR: No videos in SimBA project. Import videos into you SimBA project to visualize ROI tracking.')
        self.multiprocess_var = BooleanVar()
        self.show_pose_var = BooleanVar(value=True)
        self.animal_name_var = BooleanVar(value=True)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_DATA_PLOT.value)
        #self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=10, padx=10, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
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
        print('All ROI videos created and saved in project_folder/frames/output/ROI_analysis directory')

#_ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class CreateUserDefinedPoseConfigurationPopUp(object):
    def __init__(self,
                 master=None,
                 project_config_class=None):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("USER-DEFINED POSE CONFIGURATION")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.config_name_entry_box = Entry_Box(self.main_frm,'Pose config name','23')
        self.animal_cnt_entry_box = Entry_Box(self.main_frm, '# of Animals', '23', validation='numeric')
        self.no_body_parts_entry_box = Entry_Box(self.main_frm, '# of Body-parts (per animal)', '23', validation='numeric')
        self.img_path_file_select = FileSelect(self.main_frm, 'Image path', lblwidth=23)
        self.master, self.project_config_class = master, project_config_class
        self.confirm_btn = Button(self.main_frm, text='CONFIRM',  fg='blue', command=lambda: self.create_bodypart_table())
        self.save_btn = Button(self.main_frm, text='SAVE USER-DEFINED POSE-CONFIG', fg='blue', command=lambda: self.save_pose_config())
        self.save_btn.config(state='disabled')

        self.config_name_entry_box.grid(row=0,sticky=W)
        self.animal_cnt_entry_box.grid(row=1,sticky=W)
        self.no_body_parts_entry_box.grid(row=2,sticky=W)
        self.img_path_file_select.grid(row=3,sticky=W,pady=2)
        self.confirm_btn.grid(row=4,pady=5)
        self.save_btn.grid(row=6,pady=5)
        self.main_frm.lift()

    def create_bodypart_table(self):
        if hasattr(self, 'bp_table_frm'):
            self.bp_table_frm.destroy()
        check_int(name='ANIMAL NUMBER', value=self.animal_cnt_entry_box.entry_get)
        check_int(name='BODY-PART NUMBER', value=self.no_body_parts_entry_box.entry_get)
        self.selected_animal_cnt, self.selected_bp_cnt = int(self.animal_cnt_entry_box.entry_get), int(self.no_body_parts_entry_box.entry_get)
        check_int(name='number of animals', value=self.selected_animal_cnt)
        check_int(name='number of body-parts', value=self.selected_bp_cnt)
        self.bp_name_list = []
        self.bp_animal_list = []

        if self.selected_animal_cnt > 1:
            self.table_frame = LabelFrame(self.main_frm, text='Bodypart name                       Animal ID number')
        else:
            self.table_frame = LabelFrame(self.main_frm, text='Bodypart name')

        self.table_frame.grid(row=5, sticky=W, column=0)
        scroll_table = hxtScrollbar(self.table_frame)

        for i in range(self.selected_bp_cnt * self.selected_animal_cnt):
            bp_name_entry = Entry_Box(scroll_table, str(i+1), '2')
            bp_name_entry.grid(row=i, column=0)
            self.bp_name_list.append(bp_name_entry)
            if self.selected_animal_cnt > 1:
                animal_id_entry = Entry_Box(scroll_table, '', '2', validation='numeric')
                animal_id_entry.grid(row=i, column=1)
                self.bp_animal_list.append(animal_id_entry)
        self.save_btn.config(state='normal')

    def validate_unique_entries(self,
                                body_part_names: list,
                                animal_ids: list):
        if len(animal_ids) > 0:
            user_entries = []
            for (bp_name, animal_id) in zip(body_part_names, animal_ids):
                user_entries.append('{}_{}'.format(bp_name, animal_id))
        else:
            user_entries = body_part_names
        duplicates = list(set([x for x in user_entries if user_entries.count(x) > 1]))
        if duplicates:
            print(duplicates)
            raise DuplicationError(msg='SIMBA ERROR: SimBA found duplicate body-part names (see above). Please enter unique body-part (and animal ID) names.')
        else:
            pass

    def save_pose_config(self):

        config_name = self.config_name_entry_box.entry_get
        image_path = self.img_path_file_select.file_path
        check_file_exist_and_readable(image_path)
        check_str(name='POSE CONFIG NAME', value=config_name.strip(), allow_blank=False)
        bp_lst, animal_id_lst = [], []
        for bp_name_entry in self.bp_name_list:
            bp_lst.append(bp_name_entry.entry_get)
        for animal_id_entry in  self.bp_animal_list:
            check_int(name='Animal ID number', value=animal_id_entry.entry_get)
            animal_id_lst.append(animal_id_entry.entry_get)

        self.validate_unique_entries(body_part_names=bp_lst, animal_ids=animal_id_lst)

        pose_config_creator = PoseConfigCreator(pose_name=config_name,
                                                no_animals=int(self.selected_animal_cnt),
                                                img_path=image_path,
                                                bp_list=bp_lst,
                                                animal_id_int_list=animal_id_lst)
        pose_config_creator.launch()
        print('SIMBA COMPLETE: User-defined pose-configuration "{}" created.'.format(config_name))
        self.main_frm.winfo_toplevel().destroy()
        self.master.winfo_toplevel().destroy()
        self.project_config_class()

class PoseResetterPopUp(object):
    def __init__(self):
        popup = Tk()
        popup.minsize(300, 100)
        popup.wm_title("WARNING!")
        popupframe = LabelFrame(popup)
        label = Label(popupframe, text='Do you want to remove user-defined pose-configurations?')
        label.grid(row=0,columnspan=2)
        B1 = Button(popupframe, text='YES', fg='blue', command=lambda: PoseResetter(master=popup))
        B2 = Button(popupframe, text="NO", fg='red', command=popup.destroy)
        popupframe.grid(row=0,columnspan=2)
        B1.grid(row=1,column=0,sticky=W)
        B2.grid(row=1,column=1,sticky=W)
        popup.mainloop()


class SklearnVisualizationPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='VISUALIZE CLASSIFICATION (SKLEARN) RESULTS')
        self.video_lst = find_all_videos_in_directory(directory=self.videos_dir)
        self.use_default_font_settings_val = BooleanVar(value=True)
        self.create_videos_var = BooleanVar()
        self.create_frames_var = BooleanVar()
        self.include_timers_var = BooleanVar()
        self.rotate_img_var = BooleanVar()
        self.multiprocess_var = BooleanVar()

        bp_threshold_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='BODY-PART VISUALIZATION THRESHOLD', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.SKLEARN_PLOTS.value)
        #bp_threshold_frm = LabelFrame(self.main_frm,text='BODY-PART VISUALIZATION THRESHOLD',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        self.bp_threshold_lbl = Label(bp_threshold_frm,text='Body-parts detected below the set threshold won\'t be shown in the output videos.', font=("Helvetica", 11, 'italic'))
        self.bp_threshold_entry = Entry_Box(bp_threshold_frm,'Body-part probability threshold', '32')
        self.get_bp_probability_threshold()

        self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.auto_compute_font_cb = Checkbutton(self.style_settings_frm, text='Auto-compute font/key-point settings', variable=self.use_default_font_settings_val,
                                                command= lambda: self.enable_entrybox_from_checkbox(check_box_var=self.use_default_font_settings_val, reverse=True, entry_boxes=[self.sklearn_text_size_entry_box, self.sklearn_text_spacing_entry_box, self.sklearn_text_thickness_entry_box, self.sklearn_circle_size_entry_box]))
        self.sklearn_text_size_entry_box = Entry_Box(self.style_settings_frm, 'Text size: ', '12')
        self.sklearn_text_spacing_entry_box = Entry_Box(self.style_settings_frm, 'Text spacing: ', '12')
        self.sklearn_text_thickness_entry_box = Entry_Box(self.style_settings_frm, 'Text thickness: ', '12')
        self.sklearn_circle_size_entry_box = Entry_Box(self.style_settings_frm, 'Circle size: ', '12')
        self.sklearn_text_size_entry_box.set_state('disable')
        self.sklearn_text_spacing_entry_box.set_state('disable')
        self.sklearn_text_thickness_entry_box.set_state('disable')
        self.sklearn_circle_size_entry_box.set_state('disable')

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.create_videos_cb = Checkbutton(self.settings_frm, text='Create video', variable=self.create_videos_var)
        self.create_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.create_frames_var)
        self.timers_cb = Checkbutton(self.settings_frm, text='Include timers overlay', variable=self.include_timers_var)
        self.rotate_cb = Checkbutton(self.settings_frm, text='Rotate video 90°', variable=self.rotate_img_var)
        self.multiprocess_cb = Checkbutton(self.settings_frm, text='Multiprocess videos (faster)', variable=self.multiprocess_var, command= lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command= lambda: self.__initiate_video_creation(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', self.video_lst, '12')
        self.single_video_dropdown.setChoices(self.video_lst[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(self.video_lst))), fg='blue', command= lambda: self.__initiate_video_creation(multiple_videos=True))

        bp_threshold_frm.grid(row=0,sticky=NW)
        self.bp_threshold_lbl.grid(row=0,sticky=NW)
        self.bp_threshold_entry.grid(row=1,sticky=NW)

        self.style_settings_frm.grid(row=1,sticky=NW)
        self.auto_compute_font_cb.grid(row=0,sticky=NW)
        self.sklearn_text_size_entry_box.grid(row=1, sticky=NW)
        self.sklearn_text_spacing_entry_box.grid(row=2, sticky=NW)
        self.sklearn_text_thickness_entry_box.grid(row=3, sticky=NW)
        self.sklearn_circle_size_entry_box.grid(row=4, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        self.create_videos_cb.grid(row=0, sticky=NW)
        self.create_frames_cb.grid(row=1, sticky=NW)
        self.timers_cb.grid(row=2, sticky=NW)
        self.rotate_cb.grid(row=3, sticky=NW)
        self.multiprocess_cb.grid(row=4, sticky=NW)
        self.multiprocess_dropdown.grid(row=4, column=1, sticky=NW)
        self.multiprocess_dropdown.disable()

        self.run_frm.grid(row=3, sticky=NW)

        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=1, sticky=NW)
        self.single_video_dropdown.grid(row=1, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

    def get_bp_probability_threshold(self):
        try:
            self.bp_threshold_entry.entry_set(self.config.getfloat('threshold_settings', 'bp_threshold_sklearn'))
        except:
            self.bp_threshold_entry.entry_set(0.0)

    def __initiate_video_creation(self, multiple_videos: bool = False):
        check_float(name='BODY-PART PROBABILITY THRESHOLD', value=self.bp_threshold_entry.entry_get, min_value=0.0, max_value=1.0)
        self.config.set('threshold_settings', 'bp_threshold_sklearn', self.bp_threshold_entry.entry_get)
        with open(self.config_path, 'w') as f:
            self.config.write(f)

        if not self.use_default_font_settings_val.get():
            print_settings = {'font_size': self.sklearn_text_size_entry_box.entry_get,
                              'circle_size': self.sklearn_circle_size_entry_box.entry_get,
                              'space_size': self.sklearn_text_spacing_entry_box.entry_get,
                              'text_thickness': self.sklearn_text_thickness_entry_box.entry_get}
            for k, v in print_settings.items():
                check_float(name=v, value=v)
        else:
            print_settings = False

        if not multiple_videos:
            video_file_path = self.single_video_dropdown.getChoices()
        else:
            video_file_path = None

        if self.multiprocess_var.get():
            simba_plotter = PlotSklearnResultsMultiProcess(config_path=self.config_path,
                                                           video_setting=self.create_videos_var.get(),
                                                           rotate=self.rotate_img_var.get(),
                                                           video_file_path=video_file_path,
                                                           frame_setting=self.create_frames_var.get(),
                                                           print_timers=self.include_timers_var.get(),
                                                           cores=int(self.multiprocess_dropdown.getChoices()),
                                                           text_settings=print_settings)

        else:
            simba_plotter = PlotSklearnResultsSingleCore(config_path=self.config_path,
                                                         video_setting=self.create_videos_var.get(),
                                                         rotate=self.rotate_img_var.get(),
                                                         video_file_path=video_file_path,
                                                         frame_setting=self.create_frames_var.get(),
                                                         print_timers=self.include_timers_var.get(),
                                                         text_settings=print_settings)
        simba_plotter.initialize_visualizations()

#_ = SklearnVisualizationPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')


class GanttPlotPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        super().__init__(config_path=config_path, title='VISUALIZE GANTT PLOTS')
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing gantt charts')


        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.GANTT_PLOTS.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.use_default_style_bool = BooleanVar(value=True)
        self.auto_compute_style_cb = Checkbutton(self.style_settings_frm, text='Use default style', variable=self.use_default_style_bool, command=lambda: self.enable_text_settings())
        self.resolution_dropdown = DropDownMenu(self.style_settings_frm, 'Resolution:', self.resolutions, '16')
        self.font_size_entry = Entry_Box(self.style_settings_frm, 'Font size: ', '16', validation='numeric')
        self.font_rotation_entry = Entry_Box(self.style_settings_frm, 'Font rotation degree: ', '16', validation='numeric')
        self.font_size_entry.entry_set(val=8)
        self.font_rotation_entry.entry_set(val=45)
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.resolution_dropdown.disable()
        self.font_size_entry.set_state('disable')
        self.font_rotation_entry.set_state('disable')

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.gantt_frames_var = BooleanVar()
        self.gantt_last_frame_var = BooleanVar()
        self.gantt_videos_var = BooleanVar()
        self.gantt_multiprocess_var = BooleanVar()

        gantt_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.gantt_frames_var)
        gantt_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.gantt_videos_var)
        gantt_last_frame_cb = Checkbutton(self.settings_frm, text='Create last frame', variable=self.gantt_last_frame_var)
        gantt_multiprocess_cb = Checkbutton(self.settings_frm, text='Multi-process (faster)', variable=self.gantt_multiprocess_var, command= lambda: self.enable_dropdown_from_checkbox(check_box_var=self.gantt_multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))

        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command= lambda: self.__create_gantt_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command= lambda: self.__create_gantt_plots(multiple_videos=True))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.auto_compute_style_cb.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=1, sticky=NW)
        self.font_size_entry.grid(row=2, sticky=NW)
        self.font_rotation_entry.grid(row=3, sticky=NW)


        self.settings_frm.grid(row=1, sticky=NW)
        gantt_videos_cb.grid(row=0, sticky=NW)
        gantt_frames_cb.grid(row=1, sticky=W)
        gantt_last_frame_cb.grid(row=2, sticky=NW)
        gantt_multiprocess_cb.grid(row=3, column=0, sticky=W)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=W)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

    def enable_text_settings(self):
        if not self.use_default_style_bool.get():
            self.resolution_dropdown.enable()
            self.font_rotation_entry.set_state('normal')
            self.font_size_entry.set_state('normal')
        else:
            self.resolution_dropdown.disable()
            self.font_rotation_entry.set_state('disable')
            self.font_size_entry.set_state('disable')

    def __create_gantt_plots(self,
                             multiple_videos: bool):
        width = int(self.resolution_dropdown.getChoices().split('×')[0])
        height = int(self.resolution_dropdown.getChoices().split('×')[1])
        check_int(name='FONT SIZE', value=self.font_size_entry.entry_get, min_value=1)
        check_int(name='FONT ROTATION DEGREES', value=self.font_rotation_entry.entry_get, min_value=0, max_value=360)
        style_attr = {'width': width,
                      'height': height,
                      'font size': int(self.font_size_entry.entry_get),
                      'font rotation': int(self.font_rotation_entry.entry_get)}

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if self.gantt_multiprocess_var.get():
            gantt_creator = GanttCreatorMultiprocess(config_path=self.config_path,
                                                     frame_setting=self.gantt_frames_var.get(),
                                                     video_setting=self.gantt_videos_var.get(),
                                                     last_frm_setting=self.gantt_last_frame_var.get(),
                                                     files_found=data_paths,
                                                     cores=int(self.multiprocess_dropdown.getChoices()),
                                                     style_attr=style_attr)
        else:
            gantt_creator = GanttCreatorSingleProcess(config_path=self.config_path,
                                                      frame_setting=self.gantt_frames_var.get(),
                                                      video_setting=self.gantt_videos_var.get(),
                                                      last_frm_setting=self.gantt_last_frame_var.get(),
                                                      files_found=data_paths,
                                                      style_attr=style_attr)
        gantt_creator.create_gannt()


#_ = GanttPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')



class VisualizeClassificationProbabilityPopUp(PopUpMixin):

    def __init__(self,
                 config_path: str):
        super().__init__(config_path=config_path, title='CREATE CLASSIFICATION PROBABILITY PLOTS')
        self.max_y_lst = [x for x in range(10, 110, 10)]
        self.max_y_lst.insert(0, 'auto')
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),
                                        error_msg='SIMBA ERROR: Cant visualize probabilities, no data in project_folder/csv/machine_results directory')


        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VISUALIZE_CLF_PROBABILITIES.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=("Helvetica", 14, 'bold'), pady=5, padx=5)
        self.resolution_dropdown = DropDownMenu(self.style_settings_frm, 'Resolution:', self.resolutions, '16')
        self.max_y_dropdown = DropDownMenu(self.style_settings_frm, 'Max Y-axis:', self.max_y_lst, '16')


        self.line_clr_dropdown = DropDownMenu(self.style_settings_frm, 'Line color:', self.colors, '16')
        self.font_size_entry = Entry_Box(self.style_settings_frm, 'Font size: ', '16', validation='numeric')
        self.line_width = Entry_Box(self.style_settings_frm, 'Line width: ', '16', validation='numeric')
        self.circle_size = Entry_Box(self.style_settings_frm, 'Circle size: ', '16', validation='numeric')
        self.line_clr_dropdown.setChoices('blue')
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.font_size_entry.entry_set(val=10)
        self.line_width.entry_set(val=6)
        self.circle_size.entry_set(val=20)
        self.max_y_dropdown.setChoices(choice='auto')

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.probability_frames_var = BooleanVar()
        self.probability_videos_var = BooleanVar()
        self.probability_last_frm_var = BooleanVar()
        self.probability_multiprocess_var = BooleanVar()

        self.clf_dropdown = DropDownMenu(self.settings_frm, 'Classifier:', self.clf_names, '16')
        self.clf_dropdown.setChoices(self.clf_names[0])
        probability_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.probability_frames_var)
        probability_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.probability_videos_var)
        probability_last_frm_cb = Checkbutton(self.settings_frm, text='Create last frame', variable=self.probability_last_frm_var)
        probability_multiprocess_cb = Checkbutton(self.settings_frm, text='Multi-process (faster)', variable=self.probability_multiprocess_var, command= lambda: self.enable_dropdown_from_checkbox(check_box_var=self.probability_multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))

        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command= lambda: self.__create_probability_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command= lambda: self.__create_probability_plots(multiple_videos=True))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.line_clr_dropdown.grid(row=1, sticky=NW)
        self.font_size_entry.grid(row=2, sticky=NW)
        self.line_width.grid(row=3, sticky=NW)
        self.circle_size.grid(row=4, sticky=NW)
        self.max_y_dropdown.grid(row=5, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        self.clf_dropdown.grid(row=0, sticky=NW)
        probability_frames_cb.grid(row=1, sticky=NW)
        probability_videos_cb.grid(row=2, sticky=NW)
        probability_last_frm_cb.grid(row=3, sticky=NW)
        probability_multiprocess_cb.grid(row=4, column=0, sticky=W)
        self.multiprocess_dropdown.grid(row=4, column=1, sticky=W)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        #self.main_frm.mainloop()

    def __create_probability_plots(self,
                                   multiple_videos: bool):
        width = int(self.resolution_dropdown.getChoices().split('×')[0])
        height = int(self.resolution_dropdown.getChoices().split('×')[1])
        check_int(name='PLOT FONT SIZE', value=self.font_size_entry.entry_get, min_value=1)
        check_int(name='PLOT LINE WIDTH', value=self.line_width.entry_get, min_value=1)
        check_int(name='PLOT CIRCLE SIZE', value=self.circle_size.entry_get, min_value=1)
        style_attr = {'width': width,
                      'height': height,
                      'font size': int(self.font_size_entry.entry_get),
                      'line width': int(self.line_width.entry_get),
                      'color': self.line_clr_dropdown.getChoices(),
                      'circle size': int(self.circle_size.entry_get),
                      'y_max': self.max_y_dropdown.getChoices()}

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if not self.probability_multiprocess_var.get():
            probability_plot_creator = TresholdPlotCreatorSingleProcess(config_path=self.config_path,
                                                                        frame_setting=self.probability_frames_var.get(),
                                                                        video_setting=self.probability_videos_var.get(),
                                                                        last_image=self.probability_last_frm_var.get(),
                                                                        files_found=data_paths,
                                                                        clf_name=self.clf_dropdown.getChoices(),
                                                                        style_attr=style_attr)
        else:
            probability_plot_creator = TresholdPlotCreatorMultiprocess(config_path=self.config_path,
                                                                       frame_setting=self.probability_frames_var.get(),
                                                                       video_setting=self.probability_videos_var.get(),
                                                                       last_frame=self.probability_last_frm_var.get(),
                                                                       files_found=data_paths,
                                                                       clf_name=self.clf_dropdown.getChoices(),
                                                                       cores=int(self.multiprocess_dropdown.getChoices()),
                                                                       style_attr=style_attr)
        probability_plot_creator.create_plots()

#_ = VisualizeClassificationProbabilityPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class PathPlotPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='CREATE PATH PLOTS')
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing path plots')

        self.resolutions.insert(0, 'As input')
        self.animal_cnt_options = list(range(1, self.project_animal_cnt+1))

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.PATH_PLOTS.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=("Helvetica", 14, 'bold'), pady=5,padx=5)
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
            self.bp_dropdowns[animal_cnt] = DropDownMenu(self.body_parts_frm, 'Body-part {}:'.format(str(animal_cnt+1)), self.bp_names, '16')
            self.bp_dropdowns[animal_cnt].setChoices(self.bp_names[animal_cnt])
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
            self.include_clf_locations_cb.config(state=NORMAL)
        else:
            self.resolution_dropdown.disable()
            self.max_lines_entry.set_state('disable')
            self.font_thickness.set_state("disable")
            self.line_width.set_state('disable')
            self.circle_size.set_state('disable')
            self.font_size.set_state('disable')
            self.bg_clr_dropdown.disable()
            self.include_clf_locations_cb.config(state=DISABLED)

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
                                                 style_attr=style_attr,
                                                 animal_attr=animal_attr,
                                                 clf_attr=clf_attr)
        else:
            path_plotter = PathPlotterMulticore(config_path=self.config_path,
                                                frame_setting=self.path_frames_var.get(),
                                                video_setting=self.path_videos_var.get(),
                                                last_frame=self.path_last_frm_var.get(),
                                                files_found=data_paths,
                                                style_attr=style_attr,
                                                animal_attr=animal_attr,
                                                clf_attr=clf_attr,
                                                cores=int(self.multiprocess_dropdown.getChoices()))

        path_plotter.create_path_plots()

#_ = PathPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class DistancePlotterPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='CREATE DISTANCE PLOTS')

        self.data_path = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.max_y_lst = list(range(10, 510, 10))
        self.max_y_lst.insert(0, 'auto')
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/outlier_corrected_movement_location directory. ')

        self.number_of_distances = list(range(1, len(self.bp_names)*2))
        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DISTANCE_PLOTS.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.resolution_dropdown = DropDownMenu(self.style_settings_frm, 'Resolution:', self.resolutions, '16')
        self.font_size_entry = Entry_Box(self.style_settings_frm, 'Font size: ', '16', validation='numeric')
        self.line_width = Entry_Box(self.style_settings_frm, 'Line width: ', '16', validation='numeric')
        self.opacity_dropdown = DropDownMenu(self.style_settings_frm, 'Line opacity:', list(np.round(np.arange(0.0, 1.1, 0.1), 1)), '16')
        self.max_y_dropdown = DropDownMenu(self.style_settings_frm, 'Max Y-axis:', self.max_y_lst, '16')
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.font_size_entry.entry_set(val=8)
        self.line_width.entry_set(val=6)
        self.opacity_dropdown.setChoices(0.5)
        self.max_y_dropdown.setChoices(choice='auto')
        self.distances_frm = LabelFrame(self.main_frm, text='CHOOSE DISTANCES', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.number_of_distances_dropdown = DropDownMenu(self.distances_frm, '# Distances:', self.number_of_distances, '16', com=self.__populate_distances_menu)
        self.number_of_distances_dropdown.setChoices(self.number_of_distances[0])
        self.__populate_distances_menu(1)

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.distance_frames_var = BooleanVar()
        self.distance_videos_var = BooleanVar()
        self.distance_final_img_var = BooleanVar()
        self.multiprocess_var = BooleanVar()
        distance_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.distance_frames_var)
        distance_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.distance_videos_var)
        distance_final_img_cb = Checkbutton(self.settings_frm, text='Create last frame', variable=self.distance_final_img_var)
        self.multiprocess_cb = Checkbutton(self.settings_frm, text='Multiprocess (faster)', variable=self.multiprocess_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'Cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(choice=2)
        self.multiprocess_dropdown.disable()


        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command=lambda: self.__create_distance_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:',list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos,text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command=lambda: self.__create_distance_plots(multiple_videos=False))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.font_size_entry.grid(row=1, sticky=NW)
        self.line_width.grid(row=2, sticky=NW)
        self.opacity_dropdown.grid(row=3, sticky=NW)
        self.max_y_dropdown.grid(row=4, sticky=NW)

        self.distances_frm.grid(row=1, sticky=NW)
        self.number_of_distances_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        distance_frames_cb.grid(row=0, sticky=NW)
        distance_videos_cb.grid(row=1, sticky=NW)
        distance_final_img_cb.grid(row=2, sticky=NW)
        self.multiprocess_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=NW)

        self.run_frm.grid(row=3, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

    def __populate_distances_menu(self, choice):
        if hasattr(self, 'bp_1'):
            for k,v in self.bp_1.items():
                self.bp_1[k].destroy()
                self.bp_2[k].destroy()
                self.distance_clrs[k].destroy()

        self.bp_1, self.bp_2, self.distance_clrs = {}, {}, {}
        for distance_cnt in range(int(self.number_of_distances_dropdown.getChoices())):
            self.bp_1[distance_cnt] = DropDownMenu(self.distances_frm, 'Distance {}:'.format(str(distance_cnt+1)), self.bp_names, '16')
            self.bp_1[distance_cnt].setChoices(self.bp_names[distance_cnt])
            self.bp_1[distance_cnt].grid(row=distance_cnt+1, column=0, sticky=NW)

            self.bp_2[distance_cnt] = DropDownMenu(self.distances_frm, '', self.bp_names, '2')
            self.bp_2[distance_cnt].setChoices(self.bp_names[distance_cnt])
            self.bp_2[distance_cnt].grid(row=distance_cnt+1, column=1, sticky=NW)

            self.distance_clrs[distance_cnt] = DropDownMenu(self.distances_frm, '', self.colors_dict, '2')
            self.distance_clrs[distance_cnt].setChoices(list(self.colors_dict.keys())[distance_cnt])
            self.distance_clrs[distance_cnt].grid(row=distance_cnt + 1, column=3, sticky=NW)

    def __create_distance_plots(self,
                                multiple_videos: bool):

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        line_attr = defaultdict(list)
        for attr in (self.bp_1, self.bp_2, self.distance_clrs):
            for key, value in attr.items():
                line_attr[key].append(value.getChoices())

        width = int(self.resolution_dropdown.getChoices().split('×')[0])
        height = int(self.resolution_dropdown.getChoices().split('×')[1])
        check_int(name='DISTANCE FONT SIZE', value=self.font_size_entry.entry_get, min_value=1)
        check_int(name='DISTANCE LINE WIDTH', value=self.line_width.entry_get, min_value=1)
        style_attr = {'width': width,
                      'height': height,
                      'line width': int(self.line_width.entry_get),
                      'font size': int(self.font_size_entry.entry_get),
                      'opacity': float(self.opacity_dropdown.getChoices()),
                      'y_max': self.max_y_dropdown.getChoices()}
        if not self.multiprocess_var.get():
            distance_plotter = DistancePlotterSingleCore(config_path=self.config_path,
                                                         frame_setting=self.distance_frames_var.get(),
                                                         video_setting=self.distance_videos_var.get(),
                                                         final_img=self.distance_final_img_var.get(),
                                                         style_attr=style_attr,
                                                         files_found=data_paths,
                                                         line_attr=line_attr)
        else:
            distance_plotter = DistancePlotterMultiCore(config_path=self.config_path,
                                                        frame_setting=self.distance_frames_var.get(),
                                                        video_setting=self.distance_videos_var.get(),
                                                        final_img=self.distance_final_img_var.get(),
                                                        style_attr=style_attr,
                                                        files_found=data_paths,
                                                        line_attr=line_attr,
                                                        core_cnt=int(self.multiprocess_dropdown.getChoices()))

        distance_plotter.create_distance_plot()

# _ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')


class HeatmapClfPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='CREATE CLASSIFICATION HEATMAP PLOTS')
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. ')
        max_scales = list(np.linspace(5, 600, 5))
        max_scales.insert(0, 'Auto-compute')

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.HEATMAP_CLF.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=("Helvetica", 14, 'bold'), pady=5, padx=5)
        self.palette_dropdown = DropDownMenu(self.style_settings_frm, 'Palette:', self.palette_options, '16')
        self.shading_dropdown = DropDownMenu(self.style_settings_frm, 'Shading:', self.shading_options, '16')
        self.clf_dropdown = DropDownMenu(self.style_settings_frm, 'Classifier:', self.clf_names, '16')
        self.bp_dropdown = DropDownMenu(self.style_settings_frm, 'Body-part:', self.bp_names, '16')
        self.max_time_scale_dropdown = DropDownMenu(self.style_settings_frm, 'Max time scale (s):', max_scales, '16')
        self.bin_size_dropdown = DropDownMenu(self.style_settings_frm, 'Bin size (mm):', self.heatmap_bin_size_options, '16')

        self.palette_dropdown.setChoices(self.palette_options[0])
        self.shading_dropdown.setChoices(self.shading_options[0])
        self.clf_dropdown.setChoices(self.clf_names[0])
        self.bp_dropdown.setChoices(self.bp_names[0])
        self.max_time_scale_dropdown.setChoices(max_scales[0])
        self.bin_size_dropdown.setChoices('80×80')

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=("Helvetica", 14, 'bold'), pady=5, padx=5)
        self.heatmap_frames_var = BooleanVar()
        self.heatmap_videos_var = BooleanVar()
        self.heatmap_last_frm_var = BooleanVar()
        self.multiprocessing_var = BooleanVar()
        heatmap_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.heatmap_frames_var)
        heatmap_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.heatmap_videos_var)
        heatmap_last_frm_cb = Checkbutton(self.settings_frm, text='Create last frame', variable=self.heatmap_last_frm_var)
        self.multiprocess_cb = Checkbutton(self.settings_frm, text='Multiprocess videos (faster)', variable=self.multiprocessing_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocessing_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command=lambda: self.__create_heatmap_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command=lambda: self.__create_heatmap_plots(multiple_videos=False))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.palette_dropdown.grid(row=0, sticky=NW)
        self.shading_dropdown.grid(row=1, sticky=NW)
        self.clf_dropdown.grid(row=2, sticky=NW)
        self.bp_dropdown.grid(row=3, sticky=NW)
        self.max_time_scale_dropdown.grid(row=4, sticky=NW)
        self.bin_size_dropdown.grid(row=5, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        heatmap_frames_cb.grid(row=0, sticky=NW)
        heatmap_videos_cb.grid(row=1, sticky=NW)
        heatmap_last_frm_cb.grid(row=2, sticky=NW)
        self.multiprocess_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=NW)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        # self.main_frm.mainloop()

    def __create_heatmap_plots(self,
                               multiple_videos: bool):

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if self.max_time_scale_dropdown.getChoices() != 'Auto-compute':
            max_scale = int(self.max_time_scale_dropdown.getChoices().split('×')[0])
        else:
            max_scale = 'auto'

        bin_size = int(self.bin_size_dropdown.getChoices().split('×')[0])


        style_attr = {'palette': self.palette_dropdown.getChoices(),
                      'shading': self.shading_dropdown.getChoices(),
                      'max_scale': max_scale,
                      'bin_size': bin_size}

        if not self.multiprocessing_var.get():
            heatmapper_clf = HeatMapperClfSingleCore(config_path=self.config_path,
                                                     style_attr=style_attr,
                                                     final_img_setting=self.heatmap_last_frm_var.get(),
                                                     video_setting=self.heatmap_videos_var.get(),
                                                     frame_setting=self.heatmap_frames_var.get(),
                                                     bodypart=self.bp_dropdown.getChoices(),
                                                     files_found=data_paths,
                                                     clf_name=self.clf_dropdown.getChoices())

            heatmapper_clf_processor = multiprocessing.Process(heatmapper_clf.create_heatmaps())
            heatmapper_clf_processor.start()

        else:
            heatmapper_clf = HeatMapperClfMultiprocess(config_path=self.config_path,
                                                       style_attr=style_attr,
                                                       final_img_setting=self.heatmap_last_frm_var.get(),
                                                       video_setting=self.heatmap_videos_var.get(),
                                                       frame_setting=self.heatmap_frames_var.get(),
                                                       bodypart=self.bp_dropdown.getChoices(),
                                                       files_found=data_paths,
                                                       clf_name=self.clf_dropdown.getChoices(),
                                                       core_cnt=int(self.multiprocess_dropdown.getChoices()))

            heatmapper_clf.create_heatmaps()

#_ = HeatmapClfPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')


class DataPlotterPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='CREATE DATA PLOTS')
        self.color_lst = list(self.colors_dict.keys())
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        self.animal_cnt_options = list(range(1, self.project_animal_cnt + 1))
        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_TABLES.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.rounding_decimals_options = list(range(0, 11))
        self.font_thickness_options = list(range(1, 11))
        self.resolution_dropdown = DropDownMenu(self.style_settings_frm, 'RESOLUTION:', self.resolutions, '18')
        self.rounding_decimals_dropdown = DropDownMenu(self.style_settings_frm, 'DECIMAL ACCURACY:', self.rounding_decimals_options, '18')
        self.background_color_dropdown = DropDownMenu(self.style_settings_frm, 'BACKGROUND COLOR: ', self.color_lst, '18')
        self.font_color_dropdown = DropDownMenu(self.style_settings_frm, 'HEADER COLOR: ', self.color_lst, '18')
        self.font_thickness_dropdown = DropDownMenu(self.style_settings_frm, 'FONT THICKNESS: ', self.font_thickness_options, '18')

        self.background_color_dropdown.setChoices(choice='White')
        self.font_color_dropdown.setChoices(choice='Black')
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.rounding_decimals_dropdown.setChoices(2)
        self.font_thickness_dropdown.setChoices(1)

        self.body_parts_frm = LabelFrame(self.main_frm, text='CHOOSE BODY-PARTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.number_of_animals_dropdown = DropDownMenu(self.body_parts_frm, '# Animals:', self.animal_cnt_options, '16', com=self.__populate_body_parts_menu)
        self.number_of_animals_dropdown.setChoices(self.animal_cnt_options[0])
        self.__populate_body_parts_menu(self.animal_cnt_options[0])

        self.settings_frm = LabelFrame(self.main_frm, text='VISUALIZATION SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.data_frames_var = BooleanVar()
        self.data_videos_var = BooleanVar()
        data_frames_cb = Checkbutton(self.settings_frm, text='Create frames', variable=self.data_frames_var)
        data_videos_cb = Checkbutton(self.settings_frm, text='Create videos', variable=self.data_videos_var)

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command=lambda: self.__create_data_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command=lambda: self.__create_data_plots(multiple_videos=False))

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.rounding_decimals_dropdown.grid(row=1, sticky=NW)
        self.background_color_dropdown.grid(row=2, sticky=NW)
        self.font_color_dropdown.grid(row=3, sticky=NW)
        self.font_thickness_dropdown.grid(row=4, sticky=NW)

        self.body_parts_frm.grid(row=1, sticky=NW)
        self.number_of_animals_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        data_frames_cb.grid(row=0, sticky=NW)
        data_videos_cb.grid(row=1, sticky=NW)

        self.run_frm.grid(row=3, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

    def __populate_body_parts_menu(self, choice):
        if hasattr(self, 'bp_dropdowns'):
            for (k, v), (k2, v2) in zip(self.bp_dropdowns.items(), self.bp_colors.items()):
                self.bp_dropdowns[k].destroy()
                self.bp_colors[k].destroy()
        self.bp_dropdowns, self.bp_colors = {}, {}
        for animal_cnt in range(int(self.number_of_animals_dropdown.getChoices())):
            self.bp_dropdowns[animal_cnt] = DropDownMenu(self.body_parts_frm, 'Body-part {}:'.format(str(animal_cnt + 1)), self.bp_names, '16')
            self.bp_dropdowns[animal_cnt].setChoices(self.bp_names[animal_cnt])
            self.bp_dropdowns[animal_cnt].grid(row=animal_cnt + 1, column=0, sticky=NW)

            self.bp_colors[animal_cnt] = DropDownMenu(self.body_parts_frm, '', self.color_lst, '2')
            self.bp_colors[animal_cnt].setChoices(self.color_lst[animal_cnt])
            self.bp_colors[animal_cnt].grid(row=animal_cnt + 1, column=1, sticky=NW)

    def __create_data_plots(self,
                            multiple_videos: bool):

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        width = int(self.resolution_dropdown.getChoices().split('×')[0])
        height = int(self.resolution_dropdown.getChoices().split('×')[1])
        body_part_attr = []
        for k, v in self.bp_dropdowns.items():
            body_part_attr.append([v.getChoices(), self.bp_colors[k].getChoices()])

        style_attr = {'bg_color': self.background_color_dropdown.getChoices(),
                      'header_color': self.font_color_dropdown.getChoices(),
                      'font_thickness': int(self.font_thickness_dropdown.getChoices()),
                      'size': (int(width), int(height)),
                      'data_accuracy': int(self.rounding_decimals_dropdown.getChoices())}

        data_plotter = DataPlotter(config_path=self.config_path,
                                   body_part_attr=body_part_attr,
                                   data_paths=data_paths,
                                   style_attr=style_attr,
                                   frame_setting=self.data_frames_var.get(),
                                   video_setting=self.data_videos_var.get())

        _ = data_plotter.create_data_plots()

#_ = DataPlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')


class DirectingOtherAnimalsVisualizerPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='CREATE ANIMAL DIRECTION VIDEOS')

        self.color_lst = list(self.colors_dict.keys())
        self.color_lst.insert(0, 'Random')
        self.size_lst = list(range(1, 11))
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)

        self.show_pose_var = BooleanVar(value=True)
        self.highlight_direction_endpoints_var = BooleanVar(value=True)
        self.merge_directionality_lines_var = BooleanVar(value=False)
        self.multiprocess_var = BooleanVar(value=False)

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        #self.style_settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.show_pose_cb = Checkbutton(self.style_settings_frm, text='Show pose-estimated body-parts', variable=self.show_pose_var)
        self.highlight_direction_endpoints_cb = Checkbutton(self.style_settings_frm, text='Highlight direction end-points', variable=self.highlight_direction_endpoints_var)
        self.merge_directionality_lines_cb = Checkbutton(self.style_settings_frm, text='Polyfill direction lines', variable=self.merge_directionality_lines_var)
        self.direction_clr_dropdown = DropDownMenu(self.style_settings_frm, 'Direction color:', self.color_lst, '16')
        self.pose_size_dropdown = DropDownMenu(self.style_settings_frm, 'Pose circle size:', self.size_lst, '16')
        self.line_thickness = DropDownMenu(self.style_settings_frm, 'Line thickness:', self.size_lst, '16')
        self.line_thickness.setChoices(choice=4)
        self.pose_size_dropdown.setChoices(choice=3)
        self.direction_clr_dropdown.setChoices(choice='Random')
        multiprocess_cb = Checkbutton(self.style_settings_frm, text='Multi-process (faster)', variable=self.multiprocess_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(self.style_settings_frm, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_frm = LabelFrame(self.run_frm, text='SINGLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_single_video_btn = Button(self.run_single_video_frm, text='Create single video', fg='blue', command=lambda: self.__create_directionality_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, 'Video:', list(self.files_found_dict.keys()), '12')
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm, text='MULTIPLE VIDEO', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text='Create multiple videos ({} video(s) found)'.format(str(len(list(self.files_found_dict.keys())))), fg='blue', command=lambda: self.__create_directionality_plots(multiple_videos=False))

        self.style_settings_frm.grid(row=0, column=0, sticky=NW)
        self.show_pose_cb.grid(row=0, column=0, sticky=NW)
        self.highlight_direction_endpoints_cb.grid(row=1, column=0, sticky=NW)
        self.merge_directionality_lines_cb.grid(row=2, column=0, sticky=NW)
        self.direction_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.pose_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.line_thickness.grid(row=5, column=0, sticky=NW)
        multiprocess_cb.grid(row=6, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=6, column=1, sticky=NW)

        self.run_frm.grid(row=1, column=0, sticky=NW)
        self.run_single_video_frm.grid(row=0, column=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, column=0, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)

    def __create_directionality_plots(self,
                                      multiple_videos: bool):

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        style_attr = {'Show_pose': self.show_pose_var.get(),
                      'Pose_circle_size': int(self.pose_size_dropdown.getChoices()),
                      'Direction_color': self.direction_clr_dropdown.getChoices(),
                      'Direction_thickness': int(self.line_thickness.getChoices()),
                      'Highlight_endpoints': self.highlight_direction_endpoints_var.get(),
                      'Polyfill': self.merge_directionality_lines_var.get()}


        for data_path in data_paths:
            if not self.multiprocess_var.get():
                directing_other_animal_visualizer = DirectingOtherAnimalsVisualizer(config_path=self.config_path,
                                                                                    data_path=data_path,
                                                                                    style_attr=style_attr)
            else:
                directing_other_animal_visualizer = DirectingOtherAnimalsVisualizerMultiprocess(config_path=self.config_path,
                                                                                                data_path=data_path,
                                                                                                style_attr=style_attr,
                                                                                                core_cnt=int(self.multiprocess_dropdown.getChoices()))
            directing_other_animal_visualizer.run()

#_ = DirectingOtherAnimalsVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')


class H5CreatorPopUp(object):
    def __init__(self,
                 config_path: str):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("SIMBA PLOTLY DASHBOARD")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)


        self.config_path = config_path
        self.agg_clf_data_var = BooleanVar(value=True)
        self.timebin_clf_data_var = BooleanVar(value=True)
        self.clf_probabilities_var = BooleanVar(value=True)
        self.entire_clf_var = BooleanVar(value=True)

        self.data_settings = LabelFrame(self.main_frm, text='DATA SETTINGS', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        agg_clf_data_cb = Checkbutton(self.data_settings, text='Aggregate classification data', variable=self.agg_clf_data_var)
        timebin_clf_data_cb = Checkbutton(self.data_settings, text='Classification time-bins', variable=self.timebin_clf_data_var)
        clf_probabilities_cb = Checkbutton(self.data_settings, text='Classification probabilities', variable=self.clf_probabilities_var)
        entire_clf_cb = Checkbutton(self.data_settings, text='Entire classification data', variable=self.entire_clf_var)

        self.save_frm = LabelFrame(self.main_frm, text='SAVE', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        self.save_button = Button(self.save_frm, text='SAVE DATA (H5)', fg='green', command=lambda: self.save())

        self.load_frm = LabelFrame(self.main_frm, text='LOAD DASHBOARD', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        self.groups_file = FileSelect(self.load_frm, "EXPERIMENTAL GROUPS FILE (CSV, OPTIONAL)", lblwidth='35')
        self.dashboard_file = FileSelect(self.load_frm, "DASHBOARD FILE (H5)", lblwidth='35')
        self.load_btn = Button(self.load_frm, text='LOAD DASHBOARD', fg='blue', command=lambda: self.load())

        self.data_settings.grid(row=0, column=0, sticky='NW')
        agg_clf_data_cb.grid(row=0, column=0, sticky='NW')
        timebin_clf_data_cb.grid(row=1, column=0, sticky='NW')
        clf_probabilities_cb.grid(row=2, column=0, sticky='NW')
        entire_clf_cb.grid(row=3, column=0, sticky='NW')

        self.save_frm.grid(row=1, column=0, sticky='NW')
        self.save_button.grid(row=0, column=0, sticky='NW')

        self.load_frm.grid(row=2, column=0, sticky='NW')
        self.dashboard_file.grid(row=0, column=0, sticky='NW')
        self.groups_file.grid(row=1, column=0, sticky='NW')
        self.load_btn.grid(row=2, column=0, sticky='NW')

        self.main_frm.mainloop()

    def save(self):
        config = read_config_file(ini_path=self.config_path)
        project_path = read_config_entry(config, 'General settings', 'project_path', data_type='folder_path')
        file_type = read_config_entry(config, 'General settings', 'workflow_file_type', 'str', 'csv')
        model_cnt = read_config_entry(config=config, section='SML settings', option='no_targets', data_type='int')
        logs_path = os.path.join(project_path, 'logs')
        datetime_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.storage_path = os.path.join(logs_path, 'SimBA_dash_storage_{}.h5'.format(datetime_stamp))
        storage_file = pd.HDFStore(self.storage_path, table=True, complib='blosc:zlib', complevel=9)
        video_info_path = os.path.join(project_path, 'logs', 'video_info.csv')
        video_info_df = read_video_info_csv(file_path=video_info_path)
        machine_results_dir = os.path.join(project_path, 'csv', 'machine_results')
        clf_names = simba.train_model_functions.get_all_clf_names(config=config, target_cnt=model_cnt)
        clf_col_names = clf_names + ['Probability_' + x for x in clf_names]
        storage_file['Classifier_names'] = pd.DataFrame(data=clf_names, columns=['Classifier_names'])
        storage_file['Video_info'] = video_info_df
        timer = SimbaTimer()
        timer.start_timer()

        if self.agg_clf_data_var.get():
            data_files = glob.glob(logs_path + '/data_summary_*')
            if len(data_files) == 0:
                print('SIMBA WARNING: No aggregate classification data found in SimBA project')
            else:
                print('Compressing {} aggregate classification data files...'.format(str(len(data_files))))
                for file_path in data_files:
                    _, file_name, _ = get_fn_ext(file_path)
                    df = read_df(file_path, 'csv')
                    storage_file['SklearnData/{}'.format(file_name)] = df

        if self.timebin_clf_data_var.get():
            data_files = glob.glob(logs_path + '/Time_bins_ML_results_*')
            if len(data_files) == 0:
                print('SIMBA WARNING: No time bins data classification data found in SimBA project')
            else:
                print('Compressing {} time-bin classification data files...'.format(str(len(data_files))))
                for file_path in data_files:
                    _, file_name, _ = get_fn_ext(file_path)
                    df = read_df(file_path, 'csv')
                    storage_file['TimeBins/{}'.format(file_name)] = df

        if self.clf_probabilities_var.get():
            data_files = glob.glob(machine_results_dir + '/*.' + file_type)
            if len(data_files) == 0:
                print('SIMBA WARNING: No classification probabilities found in SimBA project')
            else:
                print('Compressing machine classification probability calculations for {} files...'.format(str(len(data_files))))
                for file_path in data_files:
                    df = read_df(file_path, file_type)[clf_col_names]
                    _, file_name, _ = get_fn_ext(file_path)
                    storage_file['VideoData/{}'.format(file_name)] = df

        if self.entire_clf_var.get():
            data_files = glob.glob(machine_results_dir + '/*.' + file_type)
            if len(data_files) == 0:
                print('SIMBA WARNING: No machine results found in SimBA project')
            else:
                print('Compressing data set for {} video files...'.format(str(len(data_files))))
                for file_path in data_files:
                    df = read_df(file_path, file_type)
                    _, file_name, _ = get_fn_ext(file_path)
                    storage_file['Entire_data/{}'.format(file_name)] = df

        storage_file.close()
        timer.stop_timer()
        print('SIMBA COMPLETE: SimBA project plotly/dash file container saved at {} (elapsed time {}s)'.format(self.storage_path, timer.elapsed_time_str))

    def wait_for_internet_connection(self, url):
        while True:
            try:
                response = urllib.request.urlopen(url, timeout=1)
                return
            except:
                pass

    def terminate_children(self, children):
        for process in children:
            process.terminate()

    def load(self):
        url = 'http://127.0.0.1:8050'
        simba_dir = os.path.dirname(simba.__file__)
        check_file_exist_and_readable(file_path=self.dashboard_file.file_path)
        dash_board_file_path = self.dashboard_file.file_path
        groups_file_path = self.groups_file.file_path

        if hasattr(self, 'process_one') or hasattr(self, 'process_two'):
            self.process_one.kill()
            self.process_two.kill()
        self.process_one = subprocess.Popen([sys.executable, os.path.join(simba_dir, 'SimBA_dash_app.py'), dash_board_file_path, groups_file_path])
        self.wait_for_internet_connection(url)
        self.process_two = subprocess.Popen([sys.executable, os.path.join(simba_dir, 'run_dash_tkinter.py'), url])
        subprocess_children = [self.process_one, self.process_two]
        atexit.register(self.terminate_children, subprocess_children)


class PupRetrievalPopUp(object):
    def __init__(self,
                 config_path: str):

        self.smoothing_options, self.config_path = ['gaussian'], config_path
        self.smooth_factor_options = list(range(1, 11))
        self.config = read_config_file(ini_path=config_path)
        self.project_path = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value,  ReadConfig.PROJECT_PATH.value, data_type=ReadConfig.FOLDER_PATH.value)
        self.ROI_path = os.path.join(self.project_path, Paths.ROI_DEFINITIONS.value)
        self.roi_analyzer = ROIAnalyzer(ini_path=config_path, data_path=None)
        self.roi_analyzer.read_roi_dfs()
        self.shape_names = self.roi_analyzer.shape_names
        self.animal_names = self.roi_analyzer.multi_animal_id_list
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.roi_analyzer.clf_cnt)

        self.distance_plots_var = BooleanVar(value=True)
        self.swarm_plot_var = BooleanVar(value=True)
        self.log_var = BooleanVar(value=True)

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("SIMBA PUP RETRIEVAL PROTOCOL 1")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)

        self.pup_track_p_entry = Entry_Box(self.main_frm, 'Tracking probability (PUP): ', '20')
        self.dam_track_p_entry = Entry_Box(self.main_frm, 'Tracking probability (DAM): ', '20')
        self.start_distance_criterion_entry = Entry_Box(self.main_frm, 'Start distance criterion (MM):', '20', validation='numeric')
        self.carry_frames_entry = Entry_Box(self.main_frm, 'Carry time (S)', '20', validation='numeric')
        self.core_nest_name_dropdown = DropDownMenu(self.main_frm, 'Core-nest name: ', self.shape_names, '20')
        self.nest_name_dropdown = DropDownMenu(self.main_frm, 'Nest name: ', self.shape_names, '20')
        self.dam_name_dropdown = DropDownMenu(self.main_frm, 'Dam name: ', self.animal_names, '20')
        self.pup_name_dropdown = DropDownMenu(self.main_frm, 'Pup name: ', self.animal_names, '20')
        self.smooth_function_dropdown = DropDownMenu(self.main_frm, 'Smooth function: ', self.smoothing_options, '20')
        self.smooth_factor_dropdown = DropDownMenu(self.main_frm, 'Smooth factor: ', self.smooth_factor_options, '20')
        self.max_time_entry = Entry_Box(self.main_frm, 'Max time (S)', '20', validation='numeric')
        self.carry_classifier_dropdown = DropDownMenu(self.main_frm, 'Carry classifier name: ', self.clf_names, '20')
        self.approach_classifier_dropdown = DropDownMenu(self.main_frm, 'Approach classifier name: ', self.clf_names, '20')
        self.dig_classifier_dropdown = DropDownMenu(self.main_frm, 'Dig classifier name: ', self.clf_names, '20')
        self.create_distance_plots_cb = Checkbutton(self.main_frm, text='Create distance plots (pre- and post tracking smoothing', variable=self.distance_plots_var)
        self.swarm_plot_cb = Checkbutton(self.main_frm, text='Create results swarm plot', variable=self.swarm_plot_var)
        self.log_cb = Checkbutton(self.main_frm, text='Create log-file', variable=self.log_var)

        self.pup_track_p_entry.entry_set(0.025)
        self.dam_track_p_entry.entry_set(0.5)
        self.start_distance_criterion_entry.entry_set(80)
        self.carry_frames_entry.entry_set(3)
        self.core_nest_name_dropdown.setChoices(choice=self.shape_names[0])
        self.nest_name_dropdown.setChoices(choice=self.shape_names[1])
        self.dam_name_dropdown.setChoices(choice=self.animal_names[0])
        self.pup_name_dropdown.setChoices(choice=self.animal_names[1])
        self.smooth_function_dropdown.setChoices(choice=self.smoothing_options[0])
        self.smooth_factor_dropdown.setChoices(choice=5)
        self.carry_frames_entry.entry_set(90)
        self.carry_classifier_dropdown.setChoices(self.clf_names[0])
        self.approach_classifier_dropdown.setChoices(self.clf_names[0])
        self.dig_classifier_dropdown.setChoices(self.clf_names[0])
        self.max_time_entry.entry_set(90)

        button_run = Button(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='red',command=lambda: self.run())

        self.pup_track_p_entry.grid(row=0,sticky=W)
        self.dam_track_p_entry.grid(row=1,sticky=W)
        self.start_distance_criterion_entry.grid(row=2,sticky=W)
        self.carry_frames_entry.grid(row=3,sticky=W)
        self.core_nest_name_dropdown.grid(row=4,sticky=W)
        self.nest_name_dropdown.grid(row=5,sticky=W)
        self.dam_name_dropdown.grid(row=6,sticky=W)
        self.pup_name_dropdown.grid(row=7,sticky=W)
        self.smooth_function_dropdown.grid(row=8,sticky=W)
        self.smooth_factor_dropdown.grid(row=9, sticky=W)
        self.max_time_entry.grid(row=10, sticky=W)
        self.carry_classifier_dropdown.grid(row=11, sticky=W)
        self.approach_classifier_dropdown.grid(row=12,sticky=W)
        self.dig_classifier_dropdown.grid(row=13,sticky=W)
        self.swarm_plot_cb.grid(row=14,sticky=W)
        self.create_distance_plots_cb.grid(row=15, sticky=W)
        self.log_cb.grid(row=16, sticky=W)

        button_run.grid(row=17,sticky=W)

        #self.main_frm.mainloop()

    def run(self):
        pup_track_p = self.pup_track_p_entry.entry_get
        dam_track_p = self.dam_track_p_entry.entry_get
        start_distance_criterion = self.start_distance_criterion_entry.entry_get
        carry_frames = self.carry_frames_entry.entry_get
        core_nest = self.core_nest_name_dropdown.getChoices()
        nest = self.nest_name_dropdown.getChoices()
        dam_name = self.dam_name_dropdown.getChoices()
        pup_name = self.pup_name_dropdown.getChoices()
        smooth_function = self.smooth_function_dropdown.getChoices()
        smooth_factor = self.smooth_factor_dropdown.getChoices()
        max_time = self.max_time_entry.entry_get
        clf_carry = self.carry_classifier_dropdown.getChoices()
        clf_approach = self.approach_classifier_dropdown.getChoices()
        clf_dig = self.dig_classifier_dropdown.getChoices()
        check_float(name='Tracking probability (PUP)', value=pup_track_p, max_value=1.0, min_value=0.0)
        check_float(name='Tracking probability (DAM)', value=dam_track_p, max_value=1.0, min_value=0.0)
        check_float(name='Start distance criterion (MM)', value=start_distance_criterion)
        check_int(name='Carry frames (S)', value=carry_frames)
        check_int(name='max_time', value=max_time)

        swarm_plot = self.swarm_plot_var.get()
        distance_plot = self.distance_plots_var.get()
        log = self.log_var.get()

        settings = {'pup_track_p': float(pup_track_p),
                    'dam_track_p': float(dam_track_p),
                    'start_distance_criterion': float(start_distance_criterion),
                    'carry_time': float(carry_frames),
                    'core_nest': core_nest,
                    'nest': nest,
                    'dam_name': dam_name,
                    'pup_name': pup_name,
                    'smooth_function': smooth_function,
                    'smooth_factor': int(smooth_factor),
                    'max_time': float(max_time),
                    'clf_carry': clf_carry,
                    'clf_approach': clf_approach,
                    'clf_dig': clf_dig,
                    'swarm_plot': swarm_plot,
                    'distance_plots': distance_plot,
                    'log': log}

        pup_calculator = PupRetrieverCalculator(config_path=self.config_path, settings=settings)
        pup_calculator.run()
        pup_calculator.save_results()

class ClassifierValidationPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='SIMBA CLASSIFIER VALIDATION CLIPS')
        color_names = list(self.colors_dict.keys())
        self.one_vid_per_bout_var = BooleanVar(value=False)
        self.one_vid_per_video_var = BooleanVar(value=True)
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CLF_VALIDATION.value)
        #self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.seconds_entry = Entry_Box(self.settings_frm, 'SECONDS: ', '15', validation='numeric')
        self.clf_dropdown = DropDownMenu(self.settings_frm, 'CLASSIFIER: ', self.clf_names, '15')
        self.clr_dropdown = DropDownMenu(self.settings_frm, 'TEXT COLOR: ', color_names, '15')
        self.clf_dropdown.setChoices(self.clf_names[0])
        self.clr_dropdown.setChoices('Cyan')
        self.seconds_entry.entry_set(val=2)

        self.individual_bout_clips_cb = Checkbutton(self.settings_frm, text='CREATE ONE CLIP PER BOUT', variable=self.one_vid_per_bout_var)
        self.individual_clip_per_video_cb = Checkbutton(self.settings_frm, text='CREATE ONE CLIP PER VIDEO', variable=self.one_vid_per_video_var)

        run_btn = Button(self.settings_frm, text='RUN VALIDATION', command= lambda: self.run())
        self.settings_frm.grid(row=0,sticky=W)
        self.seconds_entry.grid(row=0,sticky=W)
        self.clf_dropdown.grid(row=1,sticky=W)
        self.clr_dropdown.grid(row=2, sticky=W)
        self.individual_bout_clips_cb.grid(row=3, column=0, sticky=NW)
        self.individual_clip_per_video_cb.grid(row=4, column=0, sticky=NW)
        run_btn.grid(row=5,sticky=NW)

    def run(self):
        check_int(name='CLIP SECONDS', value=self.seconds_entry.entry_get)
        clf_validator = ClassifierValidationClips(config_path=self.config_path,
                                                  window=int(self.seconds_entry.entry_get),
                                                  clf_name=self.clf_dropdown.getChoices(),
                                                  clips=self.one_vid_per_bout_var.get(),
                                                  text_clr=self.colors_dict[self.clr_dropdown.getChoices()],
                                                  concat_video=self.one_vid_per_video_var.get())
        clf_validator.create_clips()


#_ = ClassifierValidationPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')

class AnalyzeSeverityPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='SIMBA SEVERITY ANALYSIS')
        if len(self.multi_animal_id_lst) > 1:
            self.multi_animal_id_lst.insert(0, 'ALL ANIMALS')
        self.frame_cnt_var = BooleanVar(value=False)
        self.seconds_cnt_var = BooleanVar(value=False)
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        #self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        self.clf_dropdown = DropDownMenu(self.settings_frm, 'CLASSIFIER:', self.clf_names, '25')
        self.clf_dropdown.setChoices(self.clf_names[0])
        self.brackets_dropdown = DropDownMenu(self.settings_frm, 'BRACKETS:', list(range(1,21)), '25')
        self.brackets_dropdown.setChoices(10)
        self.animal_dropdown = DropDownMenu(self.settings_frm, 'ANIMALS', self.multi_animal_id_lst, '25')
        self.animal_dropdown.setChoices(self.multi_animal_id_lst[0])

        frame_cnt_cb = Checkbutton(self.settings_frm, text='FRAME COUNT', variable=self.frame_cnt_var)
        seconds_cnt_cb = Checkbutton(self.settings_frm, text='SECONDS', variable=self.seconds_cnt_var)

        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        run_btn = Button(run_frm, text='RUN SEVERITY ANALYSIS', command= lambda: self.run())

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.clf_dropdown.grid(row=0, column=0, sticky=NW)
        self.brackets_dropdown.grid(row=1, column=0, sticky=NW)
        self.animal_dropdown.grid(row=2, column=0, sticky=NW)
        frame_cnt_cb.grid(row=3, column=0, sticky=NW)
        seconds_cnt_cb.grid(row=4, column=0, sticky=NW)

        run_frm.grid(row=1, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def run(self):
        if self.animal_dropdown.getChoices() == 'ALL ANIMALS':
            animals = self.multi_animal_id_lst[1:]
        else:
            animals = [self.animal_dropdown.getChoices()]
        settings = {'brackets': int(self.brackets_dropdown.getChoices()),
                    'clf': self.clf_dropdown.getChoices(),
                    'animals': animals,
                    'time': self.seconds_cnt_var.get(),
                    'frames': self.frame_cnt_var.get()}

        if (not self.seconds_cnt_var.get()) and (not self.frame_cnt_var.get()):
            raise NoSpecifiedOutputError(msg='SIMBA ERROR: Please select frames and/or time output metrics')
        severity_processor = SeverityProcessor(config_path=self.config_path,
                                               settings=settings)
        severity_processor.run()
        severity_processor.save()

class ImportFrameDirectoryPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        super().__init__(config_path=config_path, title='IMPORT FRAME DIRECTORY')
        self.frame_folder = FolderSelect(self.main_frm, 'FRAME DIRECTORY:', title='Select the main directory with frame folders')
        import_btn = Button(self.main_frm, text='IMPORT FRAMES', fg='blue', command=lambda: self.run())

        self.frame_folder.grid(row=0, column=0, sticky=NW)
        import_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.frame_folder.folder_path):
            raise NotDirectoryError(msg=f'SIMBA ERROR: {self.frame_folder.folder_path} is not a valid directory.')
        copy_img_folder(config_path=self.config_path, source=self.frame_folder.folder_path)


class AddClfPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title='ADD CLASSIFIER')
        ConfigReader.__init__(self, config_path=config_path)
        self.clf_eb = Entry_Box(self.main_frm,'CLASSIFIER NAME', '15')
        add_btn = Button(self.main_frm, text='ADD CLASSIFIER', command=lambda: self.run())
        self.clf_eb.grid(row=0, column=0, sticky=NW)
        add_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        clf_name = self.clf_eb.entry_get.strip()
        check_str(name='CLASSIFIER NAME', value=clf_name)
        self.config.set(ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, str(self.target_cnt+1))
        self.config.set(ReadConfig.SML_SETTINGS.value, f'model_path_{str(self.target_cnt + 1)}', '')
        self.config.set(ReadConfig.SML_SETTINGS.value, f'target_name_{str(self.target_cnt + 1)}', clf_name)
        self.config.set(ReadConfig.THRESHOLD_SETTINGS.value, f'threshold_{str(self.target_cnt + 1)}', 'None')
        self.config.set(ReadConfig.MIN_BOUT_LENGTH.value, f'min_bout_{str(self.target_cnt + 1)}', 'None')
        with open(self.config_path, 'w') as f:
            self.config.write(f)
        print(f'SIMBA COMPLETE: {clf_name} classifier added to SimBA project')

class ArchiveProcessedFilesPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title='ADD CLASSIFIER')
        self.archive_eb = Entry_Box(self.main_frm,'ARCHIVE DIRECTORY NAME', '25')
        archive_btn = Button(self.main_frm, text='RUN ARCHIVE', fg='blue', command=lambda: self.run())
        self.archive_eb.grid(row=0, column=0, sticky=NW)
        archive_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        archive_name = self.archive_eb.entry_get.strip()
        check_str(name='CLASSIFIER NAME', value=archive_name)
        archive_processed_files(config_path=self.config_path,
                                archive_name=archive_name)


class InterpolatePopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title='INTERPOLATE POSE')
        self.input_dir = FolderSelect(self.main_frm, 'DATA DIRECTORY:', lblwidth=25)
        self.method_dropdown = DropDownMenu(self.main_frm, 'INTERPOLATION METHOD:', Options.INTERPOLATION_OPTIONS.value, '25')
        self.method_dropdown.setChoices(Options.INTERPOLATION_OPTIONS.value[0])
        run_btn = Button(self.main_frm, text='RUN INTERPOLATION', fg='blue', command=lambda: self.run())
        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.method_dropdown.grid(row=1, column=0, sticky=NW)
        run_btn.grid(row=2, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.input_dir.folder_path):
            raise NotDirectoryError(msg=f'{self.input_dir.folder_path} is not a valid directory.')
        PostHocInterpolate(config_path=self.config_path, method=self.method_dropdown.getChoices(), input_dir=self.input_dir.folder_path)


class SmoothingPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title='SMOOTH POSE')
        self.input_dir = FolderSelect(self.main_frm, 'DATA DIRECTORY:', lblwidth=20)
        self.time_window = Entry_Box(self.main_frm, 'TIME WINDOW (MS):', '20', validation='numeric')
        self.method_dropdown = DropDownMenu(self.main_frm, 'SMOOTHING METHOD:', Options.SMOOTHING_OPTIONS.value, '20')
        self.method_dropdown.setChoices(Options.SMOOTHING_OPTIONS.value[0])
        run_btn = Button(self.main_frm, text='RUN SMOOTHING', fg='blue', command=lambda: self.run())

        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.method_dropdown.grid(row=1, column=0, sticky=NW)
        self.time_window.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=3, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.input_dir.folder_path):
            raise NotDirectoryError(msg=f'{self.input_dir.folder_path} is not a valid directory.')
        check_int(name='TIME WINDOW', value=self.time_window.entry_get, min_value=1)
        _ = PostHocSmooth(config_path=self.config_path,
                          input_dir=self.input_dir.folder_path,
                          time_window=int(self.time_window.entry_get),
                          smoothing_method=self.method_dropdown.getChoices())

class BatchPreProcessPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='BATCH PROCESS VIDEO')
        selections_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SELECTIONS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.BATCH_PREPROCESS.value)
        #selections_frm = LabelFrame(self.main_frm,text='SELECTIONS',font='bold',padx=5,pady=5)
        self.input_folder_select = FolderSelect(selections_frm,'INPUT VIDEO DIRECTORY:', title='Select Folder with Input Videos', lblwidth=20)
        self.output_folder_select = FolderSelect(selections_frm,'OUTPUT VIDEO DIRECTORY:',title='Select Folder for Output videos', lblwidth=20)
        confirm_btn = Button(selections_frm,text='CONFIRM', fg='blue', command=lambda: self.run())
        selections_frm.grid(row=0, column=0, sticky=NW)
        self.input_folder_select.grid(row=0, column=0, sticky=NW)
        self.output_folder_select.grid(row=1, column=0, sticky=NW)
        confirm_btn.grid(row=2, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.input_folder_select.folder_path):
            raise NotDirectoryError(msg=f'INPUT folder dir ({self.input_folder_select.folder_path}) is not a valid directory.')
        if not os.path.isdir(self.output_folder_select.folder_path):
            raise NotDirectoryError(msg=f'OUTPUT folder dir ({self.output_folder_select.folder_path}) is not a valid directory.')
        if self.output_folder_select.folder_path == self.input_folder_select.folder_path:
            raise DuplicationError(msg='The INPUT directory and OUTPUT directory CANNOT be the same folder')
        else:
            batch_preprocessor = BatchProcessFrame(input_dir=self.input_folder_select.folder_path, output_dir=self.output_folder_select.folder_path)
            batch_preprocessor.create_main_window()
            batch_preprocessor.create_video_table_headings()
            batch_preprocessor.create_video_rows()
            batch_preprocessor.create_execute_btn()
            batch_preprocessor.batch_process_main_frame.mainloop()

class AppendROIFeaturesByBodyPartPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, config_path=config_path, title='APPEND ROI FEATURES')
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg='SIMBA ERROR: No ROIs have been defined. Please define ROIs before appending ROI-based features')
        self.create_choose_number_of_body_parts_frm(project_body_parts=self.project_bps, run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        settings = {}
        settings['body_parts'] = {}
        settings['threshold'] = 0.00
        for bp_cnt, bp_dropdown in self.body_parts_dropdowns.items():
            settings['body_parts'][f'animal_{str(bp_cnt+1)}_bp'] = bp_dropdown.getChoices()
        roi_feature_creator = ROIFeatureCreator(config_path=self.config_path, settings=settings)
        roi_feature_creator.analyze_ROI_data()
        roi_feature_creator.save_new_features_files()


#_ = AppendROIFeaturesByBodyPartPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')


class ExtractAnnotationFramesPopUp(PopUpMixin):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title='EXTRACT ANNOTATED FRAMES')
        ConfigReader.__init__(self, config_path=config_path)
        self.create_clf_checkboxes(main_frm=self.main_frm, clfs=self.clf_names)
        self.settings_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=("Helvetica", 12, 'bold'), pady=5, padx=5)
        down_sample_resolution_options = ['None', '2x', '3x', '4x', '5x']
        self.resolution_downsample_dropdown = DropDownMenu(self.settings_frm, 'Down-sample images:', down_sample_resolution_options, '25')
        self.resolution_downsample_dropdown.setChoices(down_sample_resolution_options[0])
        self.settings_frm.grid(row=self.children_cnt_main(), column=0, sticky=NW)
        self.resolution_downsample_dropdown.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_filepath_list_is_empty(self.target_file_paths, error_msg=f'SIMBA ERROR: Zero files found in the {self.targets_folder} directory')
        downsample_setting = self.resolution_downsample_dropdown.getChoices()
        if downsample_setting != Dtypes.NONE.value:
            downsample_setting = int(''.join(filter(str.isdigit, downsample_setting)))
        clfs = []
        for clf_name, selection in self.clf_selections.items():
            if selection.get(): clfs.append(clf_name)
        if len(clfs) == 0: raise NoChoosenClassifierError()
        settings = {'downsample': downsample_setting}

        frame_extractor = AnnotationFrameExtractor(config_path=self.config_path, clfs=clfs, settings=settings)
        frame_extractor.run()


class FeatureSubsetExtractorPopUp(PopUpMixin):
    def __init__(self, config_path: str):
        super().__init__(title='EXTRACT FEATURE SUBSETS', hyperlink='https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md')
        self.feature_subset_options = ['Two-point body-part distances (mm)',
                                       'Within-animal three-point body-part angles (degrees)',
                                       'Within-animal three-point convex hull perimeters (mm)',
                                       'Within-animal four-point convex hull perimeters (mm)',
                                       'Entire animal convex hull perimeters (mm)',
                                       'Entire animal convex hull area (mm2)',
                                       'Frame-by-frame body-part movements (mm)',
                                       'Frame-by-frame distance to ROI centers (mm)']
        self.config_path = config_path
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='documentation', icon_link=Links.FEATURE_SUBSETS.value)
        self.feature_family_dropdown = DropDownMenu(self.settings_frm, 'FEATURE FAMILY:', self.feature_subset_options, '20')
        self.feature_family_dropdown.setChoices(self.feature_subset_options[0])
        self.save_dir = FolderSelect(self.settings_frm, 'SAVE DIRECTORY:', lblwidth=20)
        self.create_run_frm(run_function=self.run)

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.feature_family_dropdown.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)

        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.save_dir.folder_path)
        feature_extractor = FeatureSubsetsCalculator(config_path=self.config_path,
                                                     feature_family=self.feature_family_dropdown.getChoices(),
                                                     save_dir=self.save_dir.folder_path)
        feature_extractor.run()

class ThirdPartyAnnotatorAppenderPopUp(PopUpMixin):
    def __init__(self, config_path: str):
        self.config_path = config_path
        super().__init__(config_path=config_path, title='APPEND THIRD-PARTY ANNOTATIONS')
        apps_lst = Options.THIRD_PARTY_ANNOTATION_APPS_OPTIONS.value
        warnings_lst = Options.THIRD_PARTY_ANNOTATION_ERROR_OPTIONS.value
        app_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='THIRD-PARTY APPLICATION', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value)
        #app_frm = LabelFrame(self.main_frm, text='THIRD-PARTY APPLICATION', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.app_dropdown = DropDownMenu(app_frm, 'THIRD-PARTY APPLICATION:', apps_lst, '35')
        self.app_dropdown.setChoices(apps_lst[0])
        app_frm.grid(row=0, column=0, sticky=NW)
        self.app_dropdown.grid(row=0, column=0, sticky=NW)

        select_data_frm = LabelFrame(self.main_frm, text='SELECT DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.data_folder = FolderSelect(select_data_frm, 'DATA DIRECTORY:', lblwidth=35)
        select_data_frm.grid(row=1, column=0, sticky=NW)
        self.data_folder.grid(row=0, column=0, sticky=NW)

        self.error_dropdown_dict = self.create_dropdown_frame(main_frm=self.main_frm, drop_down_titles=warnings_lst, drop_down_options=['WARNING', 'ERROR'], frm_title='WARNINGS AND ERRORS')
        log_frm = LabelFrame(self.main_frm, text='LOGGING', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.log_var = BooleanVar(value=True)
        self.log_cb = Checkbutton(log_frm, text='CREATE IMPORT LOG', variable=self.log_var)
        log_frm.grid(row=5, column=0, sticky=NW)
        self.log_cb.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        settings = {'log': self.log_var.get()}
        settings['file_format'] = get_third_party_appender_file_formats()[self.app_dropdown.getChoices()]
        settings['errors'], app_choice = {}, self.app_dropdown.getChoices()
        for error_name, error_dropdown in self.error_dropdown_dict.items():
            settings['errors'][error_name] = error_dropdown.getChoices()
        check_if_dir_exists(in_dir=self.data_folder.folder_path)

        third_party_importer = ThirdPartyLabelAppender(app=self.app_dropdown.getChoices(),
                                                       config_path=self.config_path,
                                                       data_dir=self.data_folder.folder_path,
                                                       settings=settings)
        third_party_importer.run()


class ValidationVideoPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str,
                 simba_main_frm: object):

        self.config_path = config_path
        self.feature_file_path = simba_main_frm.csvfile.file_path
        self.model_path = simba_main_frm.modelfile.file_path
        self.discrimination_threshold = simba_main_frm.dis_threshold.entry_get
        self.shortest_bout = simba_main_frm.min_behaviorbout.entry_get
        super().__init__(config_path=config_path, title='CREATE VALIDATION VIDEO')
        style_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='STYLE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value)
        #style_frm = LabelFrame(self.main_frm, text='STYLE SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.default_style_var = BooleanVar(value=True)
        default_style_cb = Checkbutton(style_frm, text='AUTO-COMPUTE STYLES', variable=self.default_style_var, command= lambda: self.enable_entrybox_from_checkbox(check_box_var=self.default_style_var, entry_boxes=[self.font_size_eb, self.spacing_eb, self.circle_size], reverse=True))
        self.font_size_eb = Entry_Box(style_frm, 'Font size: ', '25', validation='numeric')
        self.spacing_eb = Entry_Box(style_frm, 'Text spacing: ', '25', validation='numeric')
        self.circle_size = Entry_Box(style_frm, 'Circle size: ', '25', validation='numeric')
        self.font_size_eb.entry_set(val=1)
        self.spacing_eb.entry_set(val=10)
        self.circle_size.entry_set(val=5)
        self.enable_entrybox_from_checkbox(check_box_var=self.default_style_var,
                                           entry_boxes=[self.font_size_eb,
                                                        self.spacing_eb,
                                                        self.circle_size],
                                           reverse=True)

        tracking_frm = LabelFrame(self.main_frm, text='TRACKING OPTIONS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.show_pose_dropdown = DropDownMenu(tracking_frm, 'Show pose:', Options.BOOL_STR_OPTIONS.value, '20')
        self.show_animal_names_dropdown = DropDownMenu(tracking_frm, 'Show animal names:', Options.BOOL_STR_OPTIONS.value, '20')
        self.show_pose_dropdown.setChoices(Options.BOOL_STR_OPTIONS.value[0])
        self.show_animal_names_dropdown.setChoices(Options.BOOL_STR_OPTIONS.value[1])

        multiprocess_frame = LabelFrame(self.main_frm, text='MULTI-PROCESS SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.multiprocess_var = BooleanVar(value=False)
        self.multiprocess_cb = Checkbutton(multiprocess_frame, text='Multiprocess videos (faster)', variable=self.multiprocess_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocess_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(multiprocess_frame, 'CPU cores:', list(range(2, self.cpu_cnt)), '12')
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        gantt_frm = LabelFrame(self.main_frm, text='GANTT SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.gantt_dropdown = DropDownMenu(gantt_frm, 'GANTT TYPE:', Options.GANTT_VALIDATION_OPTIONS.value, '12')
        self.gantt_dropdown.setChoices(Options.GANTT_VALIDATION_OPTIONS.value[0])


        style_frm.grid(row=0, column=0, sticky=NW)
        default_style_cb.grid(row=0, column=0, sticky=NW)
        self.font_size_eb.grid(row=1, column=0, sticky=NW)
        self.spacing_eb.grid(row=2, column=0, sticky=NW)
        self.circle_size.grid(row=3, column=0, sticky=NW)

        tracking_frm.grid(row=1, column=0, sticky=NW)
        self.show_pose_dropdown.grid(row=0, column=0, sticky=NW)
        self.show_animal_names_dropdown.grid(row=1, column=0, sticky=NW)

        multiprocess_frame.grid(row=2, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=0, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=0, column=1, sticky=NW)
        gantt_frm.grid(row=3, column=0, sticky=NW)
        self.gantt_dropdown.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):

        settings = {'pose': str_2_bool(self.show_pose_dropdown.getChoices()), 'animal_names': str_2_bool(self.show_animal_names_dropdown.getChoices())}
        settings['styles'] = None
        if not self.default_style_var.get():
            check_float(name='FONT SIZE', value=self.font_size_eb.entry_get)
            check_float(name='CIRCLE SIZE', value=self.circle_size.entry_get)
            check_float(name='SPACE SCALE', value=self.spacing_eb.entry_get)
            settings['styles']['circle size'] = int(self.circle_size.entry_get)
            settings['styles']['font size'] = self.font_size_eb.entry_get
            settings['styles']['space_scale'] = int(self.spacing_eb.entry_get)
        check_float(name='MINIMUM BOUT LENGTH', value=self.shortest_bout)
        check_float(name='DISCRIMINATION THRESHOLD', value=self.discrimination_threshold)
        check_file_exist_and_readable(file_path=self.feature_file_path)
        check_file_exist_and_readable(file_path=self.model_path)

        if not self.multiprocess_var.get():
            validation_video_creator = ValidateModelOneVideo(config_path=self.config_path,
                                                             feature_file_path=self.feature_file_path,
                                                             model_path=self.model_path,
                                                             discrimination_threshold=float(self.discrimination_threshold),
                                                             shortest_bout=int(self.shortest_bout),
                                                             settings=settings,
                                                             create_gantt=self.gantt_dropdown.getChoices())

        else:
            validation_video_creator = ValidateModelOneVideoMultiprocess(config_path=self.config_path,
                                                                         feature_file_path=self.feature_file_path,
                                                                         model_path=self.model_path,
                                                                         discrimination_threshold=float(self.discrimination_threshold),
                                                                         shortest_bout=int(self.shortest_bout),
                                                                         cores=int(self.multiprocess_dropdown.getChoices()),
                                                                         settings=settings,
                                                                         create_gantt=self.gantt_dropdown.getChoices())
        validation_video_creator.run()

#_ = ValidationVideoPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')


#_ = ThirdPartyAnnotatorAppenderPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')





#_ = FeatureSubsetExtractorPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')


#_ = ExtractAnnotationFramesPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')




#_ = AppendROIFeaturesByBodyPartPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')



#- = PupRetrievalPopUp(config_path='/Users/simon/Downloads/Automated PRT_test/project_folder/project_config.ini')
#_ = CreateUserDefinedPoseConfigurationPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = SklearnVisualizationPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = GanttPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
#_ = VisualizeClassificationProbabilityPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
#_ = PathPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
#_ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
#_ = HeatmapClfPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
#_ = DataPlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = DirectingOtherAnimalsVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini')
#_ = H5CreatorPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
