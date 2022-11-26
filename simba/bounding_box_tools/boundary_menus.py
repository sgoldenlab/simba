from simba.read_config_unit_tests import read_config_entry, read_config_file, check_int
from simba.tkinter_functions import DropDownMenu, Entry_Box
from simba.drop_bp_cords import create_body_part_dictionary, getBpNames
from simba.misc_tools import check_multi_animal_status, get_fn_ext, find_video_of_file, create_single_color_lst
import os, glob
import pickle
from tkinter import *
from simba.bounding_box_tools.find_bounderies import AnimalBoundaryFinder
from simba.bounding_box_tools.visualize_boundaries import BoundaryVisualizer
from simba.bounding_box_tools.boundary_statistics import BoundaryStatisticsCalculator
from simba.bounding_box_tools.agg_boundary_stats import AggBoundaryStatisticsCalculator

class BoundaryMenus(object):
    """
    Class creating GUI interface for extrapolating bounding boxes from pose-estimation data, and calculating
    statstics on bounding boxes and pose-estmated key-point intersections.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    `Bounding boxes tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md__.

    Examples
    ----------
    >>> _ = BoundaryMenus(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
    """

    def __init__(self,
                 config_path: str):

        self.config_path = config_path
        self.config = read_config_file(self.config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.animal_cnt = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.boundary_main_frm = Toplevel()
        self.boundary_main_frm.minsize(750, 300)
        self.boundary_main_frm.wm_title("SIMBA ANCHORED ROI (BOUNDARY BOXES ANALYSIS)")
        self.named_shape_colors = {'White': (255, 255, 255),
                                   'Grey': (220, 200, 200),
                                   'Red': (0, 0, 255),
                                   'Dark-red': (0, 0, 139),
                                   'Maroon': (0, 0, 128),
                                   'Orange': (0, 165, 255),
                                   'Dark-orange': (0, 140, 255),
                                   'Coral': (80, 127, 255),
                                   'Chocolate': (30, 105, 210),
                                   'Yellow': (0, 255, 255),
                                   'Green': (0, 128, 0),
                                   'Dark-grey': (105, 105, 105),
                                   'Light-grey': (192, 192, 192),
                                   'Pink': (178, 102, 255),
                                   'Lime': (204, 255, 229),
                                   'Purple': (255, 51, 153),
                                   'Cyan': (255, 255, 102)}
        self.settings_frm = LabelFrame(self.boundary_main_frm, text='SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, list(self.multi_animal_id_lst), self.no_animals, list(self.x_cols), list(self.y_cols), [], [])
        self.max_animal_name_char = len(max([x for x in list(self.animal_bp_dict.keys())]))
        self.find_boundaries_btn = Button(self.settings_frm, text='FIND ANIMAL BOUNDARIES', command=lambda: self.__launch_find_boundaries_pop_up())
        self.visualize_boundaries_btn = Button(self.settings_frm, text='VISUALIZE BOUNDARIES', command=lambda: self.__launch_visualize_boundaries())
        self.boundary_statistics_btn = Button(self.settings_frm, text='CALCULATE BOUNDARY STATISTICS', command=lambda: self.__launch_boundary_statistics())
        self.agg_boundary_statistics_btn = Button(self.settings_frm, text='CALCULATE AGGREGATE BOUNDARY STATISTICS', command=lambda: self.__launch_agg_boundary_statistics())
        self.settings_frm.grid(row=0, sticky=W)
        self.find_boundaries_btn.grid(row=0, column=0, sticky=NW)
        self.visualize_boundaries_btn.grid(row=1, column=0, sticky=NW)
        self.boundary_statistics_btn.grid(row=0, column=1, sticky=NW)
        self.agg_boundary_statistics_btn.grid(row=1, column=1, sticky=NW)

    def __launch_find_boundaries_pop_up(self):
        self.find_boundaries_frm = Toplevel()
        self.find_boundaries_frm.minsize(750, 300)
        self.find_boundaries_frm.wm_title("FIND ANIMAL BOUNDARIES")
        self.find_boundaries_frm.lift()
        self.select_shape_type_frm = LabelFrame(self.find_boundaries_frm, text='SELECT SHAPE TYPE', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.shape_types = ['ENTIRE ANIMAL', 'SINGLE BODY-PART SQUARE', 'SINGLE BODY-PART CIRCLE']
        self.shape_dropdown = DropDownMenu(self.select_shape_type_frm,'SELECT SHAPE TYPE',self.shape_types,'15')
        self.shape_dropdown.setChoices(self.shape_types[0])
        self.select_btn = Button(self.select_shape_type_frm, text='SELECT', command=lambda: self.__populate_find_boundaries_menu())
        self.select_shape_type_frm.grid(row=0, column=0, sticky=NW)
        self.shape_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_btn.grid(row=0, column=1, sticky=NW)

    def __populate_find_boundaries_menu(self):
        if hasattr(self, 'boundary_settings'):
            self.boundary_settings.destroy()
        self.selected_shape_type = self.shape_dropdown.getChoices()
        self.boundary_settings = LabelFrame(self.find_boundaries_frm, text='BOUNDARY SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        boundary_settings_row_cnt = 1
        if (self.selected_shape_type == 'SINGLE BODY-PART SQUARE') | (self.selected_shape_type == 'SINGLE BODY-PART CIRCLE'):
            self.animals = {}
            for animal_cnt, (name, animal_data) in enumerate(self.animal_bp_dict.items()):
                self.animals[name] = {}
                self.animals[name]['animal_name_lbl'] = Label(self.boundary_settings, text=name, width=self.max_animal_name_char + 5)
                animal_bps = [x[:-2] for x in animal_data['X_bps']]
                self.animals[name]['body_part_dropdown'] = DropDownMenu(self.boundary_settings,'BODY-PART: ', animal_bps,' 0')
                self.animals[name]['body_part_dropdown'].setChoices(animal_bps[0])
                self.animals[name]['animal_name_lbl'].grid(row=boundary_settings_row_cnt, column=0, sticky=NW)
                self.animals[name]['body_part_dropdown'].grid(row=boundary_settings_row_cnt, column=1, sticky=NW)
                boundary_settings_row_cnt += 1
        elif self.selected_shape_type == 'ENTIRE ANIMAL':
            self.force_rectangle_var = BooleanVar()
            self.force_rectangle_cb = Checkbutton(self.boundary_settings, text='FORCE RECTANGLE', variable=self.force_rectangle_var, command=None)
            self.force_rectangle_cb.grid(row=boundary_settings_row_cnt, column=0, sticky=NW)
            boundary_settings_row_cnt += 1
        self.boundary_settings.grid(row=1, column=0, sticky=W, padx=5, pady=15)
        self.parallel_offset_entry = Entry_Box(self.boundary_settings, 'PARALLEL OFFSET (MM):', labelwidth='18', width=10, validation='numeric')
        self.run_find_shapes_btn = Button(self.boundary_settings, text='RUN', command=lambda: self.__run_find_boundaries())
        self.parallel_offset_entry.entry_set('0')
        self.parallel_offset_entry.grid(row=boundary_settings_row_cnt, column=0, sticky=NW)
        self.run_find_shapes_btn.grid(row=boundary_settings_row_cnt+1, column=0, sticky=NW)

    def __run_find_boundaries(self):
        if self.selected_shape_type == 'ENTIRE ANIMAL':
            force_rectangle = self.force_rectangle_var.get()
            body_parts = None
        elif (self.selected_shape_type == 'SINGLE BODY-PART SQUARE') | (self.selected_shape_type == 'SINGLE BODY-PART CIRCLE'):
            body_parts = {}
            for animal, animal_data in self.animals.items():
                body_parts[animal] = self.animals[animal]['body_part_dropdown'].getChoices()
            force_rectangle = False
        parallel_offset = self.parallel_offset_entry.entry_get
        check_int(name='PARALLEL OFFSET', value=parallel_offset)
        boundary_finder = AnimalBoundaryFinder(config_path=self.config_path,
                                               roi_type=self.selected_shape_type,
                                               body_parts=body_parts,
                                               force_rectangle=force_rectangle,
                                               parallel_offset=int(parallel_offset))
        boundary_finder.find_boundaries()

    def __launch_visualize_boundaries(self):
        self.anchored_roi_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        if not os.path.isfile(self.anchored_roi_path):
            print('SIMBA ERROR: No anchored ROI found in {}'.format(self.anchored_roi_path))
            raise FileNotFoundError()
        with open(self.anchored_roi_path, 'rb') as fp: self.roi_data = pickle.load(fp)
        videos_in_project = glob.glob(self.video_dir + '/*')
        videos_with_data = list(self.roi_data.keys())
        if len(videos_in_project) == 0:
            print('SIMBA ERROR: Zero video files found in SimBA project')
            raise ValueError()
        video_names = []
        for file_path in videos_in_project:
            _, name, _ = get_fn_ext(filepath=file_path)
            video_names.append(name)
        sets_w_data_and_video = list(set(videos_with_data).intersection(video_names))
        if len(sets_w_data_and_video) == 0:
            print('SIMBA ERROR: Zero video files found with calculated anchored ROIs in SimBA project')
            raise ValueError()
        self.viz_boundaries_frm = Toplevel()
        self.viz_boundaries_frm.minsize(600, 150)
        self.viz_boundaries_frm.wm_title("VISUALIZE ANIMAL BOUNDARIES")
        self.video_settings_frm = LabelFrame(self.viz_boundaries_frm, text='SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.roi_attr_frm = LabelFrame(self.viz_boundaries_frm, text='ROI ATTRIBUTES', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.select_video_dropdown = DropDownMenu(self.video_settings_frm, 'SELECT VIDEO: ', sets_w_data_and_video, ' 0')
        self.select_video_dropdown.setChoices(sets_w_data_and_video[0])
        self.run_visualize_roi = Button(self.viz_boundaries_frm, text='RUN', font=('Helvetica', 15, 'bold'), fg='blue', command=lambda: self.__run_boundary_visualization())
        self.include_keypoints_var = BooleanVar()
        self.convert_to_grayscale_var = BooleanVar()
        self.highlight_intersections_var = BooleanVar()
        self.enable_attr_var = BooleanVar()
        self.include_keypoints_cb = Checkbutton(self.video_settings_frm, text='INCLUDE KEY-POINTS', variable=self.include_keypoints_var, command=None)
        self.convert_to_grayscale_cb = Checkbutton(self.video_settings_frm, text='GREYSCALE', variable=self.convert_to_grayscale_var, command=None)
        self.highlight_intersections_cb = Checkbutton(self.video_settings_frm, text='HIGHLIGHT INTERSECTIONS', variable=self.highlight_intersections_var, command=None)
        self.enable_roi_attr_cb = Checkbutton(self.video_settings_frm, text='ENABLE USER-DEFINED ROI ATTRIBUTES', variable=self.enable_attr_var, command=self.__enable_roi_attributes)
        self.video_settings_frm.grid(row=0, sticky=NW)
        self.select_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.include_keypoints_cb.grid(row=1, column=0, sticky=NW)
        self.convert_to_grayscale_cb.grid(row=2, column=0, sticky=NW)
        self.highlight_intersections_cb.grid(row=3, column=0, sticky=NW)
        self.enable_roi_attr_cb.grid(row=4, column=0, sticky=NW)
        self.run_visualize_roi.grid(row=5, column=0, sticky=NW)
        self.animal_attr_dict = {}

        Label(self.roi_attr_frm, text='ANIMAL', font=('Helvetica', 12, 'bold'), width=self.max_animal_name_char + 10).grid(row=0, column=0, sticky=N)
        Label(self.roi_attr_frm, text='ROI COLOR', font=('Helvetica', 12, 'bold'), width=self.max_animal_name_char + 10).grid(row=0, column=1, sticky=N)
        Label(self.roi_attr_frm, text='ROI THICKNESS', font=('Helvetica', 12, 'bold'), width=self.max_animal_name_char + 10).grid(row=0, column=2, sticky=N)
        Label(self.roi_attr_frm, text='KEY-POINT SIZE', font=('Helvetica', 12, 'bold'), width=self.max_animal_name_char + 10).grid(row=0, column=3, sticky=N)
        Label(self.roi_attr_frm, text='HIGHLIGHT COLOR', font=('Helvetica', 12, 'bold'), width=self.max_animal_name_char + 10).grid(row=0, column=4, sticky=N)
        Label(self.roi_attr_frm, text='HIGHLIGHT THICKNESS', font=('Helvetica', 12, 'bold'), width=self.max_animal_name_char + 10).grid(row=0, column=5, sticky=N)
        for cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.animal_attr_dict[animal_name] = {}
            self.animal_attr_dict[animal_name]['label'] = Label(self.roi_attr_frm, text=animal_name, width=self.max_animal_name_char)
            self.animal_attr_dict[animal_name]['clr_dropdown'] = DropDownMenu(self.roi_attr_frm, '', list(self.named_shape_colors.keys()), '10', command=None)
            self.animal_attr_dict[animal_name]['clr_dropdown'].setChoices(list(self.named_shape_colors.keys())[cnt])
            self.animal_attr_dict[animal_name]['thickness_dropdown'] = DropDownMenu(self.roi_attr_frm, '', list(range(1, 10)), '10',command=None)
            self.animal_attr_dict[animal_name]['thickness_dropdown'].setChoices(1)
            self.animal_attr_dict[animal_name]['circle_size_dropdown'] = DropDownMenu(self.roi_attr_frm, '', list(range(1, 10)), '10', command=None)
            self.animal_attr_dict[animal_name]['circle_size_dropdown'].setChoices(1)
            self.animal_attr_dict[animal_name]['highlight_clr_dropdown'] = DropDownMenu(self.roi_attr_frm, '', list(self.named_shape_colors.keys()), '10',command=None)
            self.animal_attr_dict[animal_name]['highlight_clr_dropdown'].setChoices('Red')
            self.animal_attr_dict[animal_name]['highlight_clr_thickness'] = DropDownMenu(self.roi_attr_frm, '',list(range(1, 10)), '10',command=None)
            self.animal_attr_dict[animal_name]['highlight_clr_thickness'].setChoices(5)
            self.animal_attr_dict[animal_name]['label'].grid(row=cnt+1, column=0, sticky=NW)
            self.animal_attr_dict[animal_name]['clr_dropdown'].grid(row=cnt+1, column=1, sticky=NW)
            self.animal_attr_dict[animal_name]['thickness_dropdown'].grid(row=cnt+1, column=2, sticky=NW)
            self.animal_attr_dict[animal_name]['circle_size_dropdown'].grid(row=cnt+1, column=3, sticky=NW)
            self.animal_attr_dict[animal_name]['highlight_clr_dropdown'].grid(row=cnt+1, column=4, sticky=NW)
            self.animal_attr_dict[animal_name]['highlight_clr_thickness'].grid(row=cnt+1, column=5, sticky=NW)
        self.roi_attr_frm.grid(row=3, column=0, sticky=NW)
        self.__enable_roi_attributes()

    def __enable_roi_attributes(self):
        if self.enable_attr_var.get():
            for animal_name in self.animal_attr_dict.keys():
                self.animal_attr_dict[animal_name]['clr_dropdown'].enable()
                self.animal_attr_dict[animal_name]['thickness_dropdown'].enable()
                self.animal_attr_dict[animal_name]['circle_size_dropdown'].enable()
                self.animal_attr_dict[animal_name]['highlight_clr_dropdown'].enable()
                self.animal_attr_dict[animal_name]['highlight_clr_thickness'].enable()
        else:
            for animal_name in self.animal_attr_dict.keys():
                self.animal_attr_dict[animal_name]['clr_dropdown'].disable()
                self.animal_attr_dict[animal_name]['thickness_dropdown'].disable()
                self.animal_attr_dict[animal_name]['circle_size_dropdown'].disable()
                self.animal_attr_dict[animal_name]['highlight_clr_dropdown'].disable()
                self.animal_attr_dict[animal_name]['highlight_clr_thickness'].disable()

    def __run_boundary_visualization(self):
        include_keypoints = self.include_keypoints_var.get()
        greyscale = self.convert_to_grayscale_var.get()
        highlight_intersections = self.highlight_intersections_var.get()
        roi_attr = {}
        if self.enable_attr_var.get():
            for cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
                roi_attr[animal_name] = {}
                roi_attr[animal_name]['bbox_clr'] = self.named_shape_colors[self.animal_attr_dict[animal_name]['clr_dropdown'].getChoices()]
                roi_attr[animal_name]['bbox_thickness'] = int(self.animal_attr_dict[animal_name]['thickness_dropdown'].getChoices())
                roi_attr[animal_name]['keypoint_size'] = int(self.animal_attr_dict[animal_name]['circle_size_dropdown'].getChoices())
                roi_attr[animal_name]['highlight_clr'] = self.named_shape_colors[self.animal_attr_dict[animal_name]['highlight_clr_dropdown'].getChoices()]
                roi_attr[animal_name]['highlight_clr_thickness'] = int(self.animal_attr_dict[animal_name]['highlight_clr_thickness'].getChoices())
        else:
            roi_attr = None

        video_visualizer = BoundaryVisualizer(config_path=self.config_path,
                                              video_name=self.select_video_dropdown.getChoices(),
                                              include_key_points=include_keypoints,
                                              greyscale=greyscale,
                                              show_intersections=highlight_intersections,
                                              roi_attributes=roi_attr)

        video_visualizer.run_visualization()

    def __launch_boundary_statistics(self):
        self.anchored_roi_path = os.path.join(self.project_path, 'logs', 'anchored_rois.pickle')
        if not os.path.isfile(self.anchored_roi_path):
            print('SIMBA ERROR: No anchored ROI data found at {}'.format(self.anchored_roi_path))
            raise FileNotFoundError()
        self.statistics_frm = Toplevel()
        self.statistics_frm.minsize(400, 150)
        self.statistics_frm.wm_title("ANIMAL ANCHORED ROI STATISTICS")
        self.settings_frm = LabelFrame(self.statistics_frm, text='SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.file_type_frm = LabelFrame(self.statistics_frm, text='OUTPUT FILE TYPE', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.roi_intersections_var = BooleanVar(value=True)
        self.roi_keypoint_intersections_var = BooleanVar(value=True)
        self.roi_intersections_cb = Checkbutton(self.settings_frm, text='ROI-ROI INTERSECTIONS', variable=self.roi_intersections_var, command=None)
        self.roi_keypoint_intersections_cb = Checkbutton(self.settings_frm, text='ROI-KEYPOINT INTERSECTIONS', variable=self.roi_keypoint_intersections_var, command=None)
        self.out_file_type = StringVar(value='CSV')
        input_csv_rb = Radiobutton(self.file_type_frm, text=".csv", variable=self.out_file_type, value="CSV")
        input_parquet_rb = Radiobutton(self.file_type_frm, text=".parquet", variable=self.out_file_type, value="PARQUET")
        input_pickle_rb = Radiobutton(self.file_type_frm, text=".pickle", variable=self.out_file_type, value="PICKLE")
        self.run_statistics = Button(self.statistics_frm, text='RUN', command=lambda: self.__run_statistics())
        self.settings_frm.grid(row=0, sticky=NW)
        self.file_type_frm.grid(row=1, sticky=NW)
        self.roi_intersections_cb.grid(row=0, column=0, sticky=NW)
        self.roi_keypoint_intersections_cb.grid(row=1, column=0, sticky=NW)
        input_csv_rb.grid(row=0, column=0, sticky=NW)
        input_parquet_rb.grid(row=1, column=0, sticky=NW)
        input_pickle_rb.grid(row=2, column=0, sticky=NW)
        self.run_statistics.grid(row=3, column=0, sticky=NW)

    def __run_statistics(self):
        save_format = self.out_file_type.get()
        roi_intersections = self.roi_intersections_var.get()
        roi_keypoint_intersections = self.roi_keypoint_intersections_var.get()
        if (not roi_intersections) and (not roi_keypoint_intersections):
            print('SIMBA ERROR: Please select at least one category of statistics')
            raise ValueError()
        statistics_calculator = BoundaryStatisticsCalculator(config_path=self.config_path,
                                                             roi_intersections=roi_intersections,
                                                             roi_keypoint_intersections=roi_keypoint_intersections,
                                                             save_format=save_format)
        statistics_calculator.save_results()

    def __launch_agg_boundary_statistics(self):
        self.data_path = os.path.join(self.project_path, 'csv', 'anchored_roi_data')
        if not os.path.isdir(self.data_path):
            print('SIMBA ERROR: No anchored roi statistics found in {}'.format(self.data_path))
            raise ValueError
        self.main_agg_statistics_frm = Toplevel()
        self.main_agg_statistics_frm.minsize(400, 175)
        self.main_agg_statistics_frm.wm_title("ANIMAL ANCHORED ROI AGGREGATE STATISTICS")
        self.agg_settings_frm = LabelFrame(self.main_agg_statistics_frm, text='SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)

        self.interaction_time = BooleanVar(value=True)
        self.interaction_bout_cnt = BooleanVar(value=True)
        self.interaction_bout_mean = BooleanVar(value=True)
        self.interaction_bout_median = BooleanVar(value=True)
        self.detailed_interaction_data_var = BooleanVar(value=True)
        self.interaction_time_cb = Checkbutton(self.agg_settings_frm, text='INTERACTION TIME (s)', variable=self.interaction_time, command=None)
        self.interaction_bout_cnt_cb = Checkbutton(self.agg_settings_frm, text='INTERACTION BOUT COUNT', variable=self.interaction_bout_cnt, command=None)
        self.interaction_bout_mean_cb = Checkbutton(self.agg_settings_frm, text='INTERACTION BOUT TIME MEAN (s)', variable=self.interaction_bout_mean, command=None)
        self.interaction_bout_median_cb = Checkbutton(self.agg_settings_frm, text='INTERACTION BOUT TIME MEDIAN (s)', variable=self.interaction_bout_median, command=None)
        self.detailed_interaction_data_cb = Checkbutton(self.agg_settings_frm, text='DETAILED INTERACTIONS TABLE', variable=self.detailed_interaction_data_var, command=None)
        self.minimum_bout_entry_box = Entry_Box(self.agg_settings_frm, 'MINIMUM BOUT LENGTH (MS):', labelwidth='25', width=10, validation='numeric')
        self.run_btn = Button(self.main_agg_statistics_frm, text='CALCULATE AGGREGATE STATISTICS', command=lambda: self._run_agg_stats())

        self.agg_settings_frm.grid(row=0, sticky=NW)
        self.interaction_time_cb.grid(row=0, column=0, sticky=NW)
        self.interaction_bout_cnt_cb.grid(row=1, column=0, sticky=NW)
        self.interaction_bout_mean_cb.grid(row=2, column=0, sticky=NW)
        self.interaction_bout_median_cb.grid(row=3, column=0, sticky=NW)
        self.detailed_interaction_data_cb.grid(row=4, column=0, sticky=NW)
        self.minimum_bout_entry_box.grid(row=5, column=0, sticky=NW)
        self.run_btn.grid(row=1, column=0, sticky=NW)

    def _run_agg_stats(self):
        measures = []
        for cb, name in zip([self.interaction_time, self.interaction_bout_cnt, self.interaction_bout_mean, self.interaction_bout_median, self.detailed_interaction_data_var], ['INTERACTION TIME (s)', 'INTERACTION BOUT COUNT', 'INTERACTION BOUT TIME MEAN (s)', 'INTERACTION BOUT TIME MEDIAN (s)', 'DETAILED INTERACTIONS TABLE']):
            if cb.get():
                measures.append(name)
        min_bout = self.minimum_bout_entry_box.entry_get
        check_int(name='MIN BOUT LENGTH', value=min_bout)
        if min_bout == '': min_bout = 0
        if len(measures) == 0:
            print('SIMBA ERROR: Select at least one descriptive statistics checkbox')
            raise ValueError()

        agg_stats_calculator = AggBoundaryStatisticsCalculator(config_path=self.config_path,
                                                               measures=measures,
                                                               shortest_allowed_interaction=int(min_bout))
        agg_stats_calculator.run()
        agg_stats_calculator.save()

# test = BoundaryMenus(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# test.boundary_main_frm.mainloop()

