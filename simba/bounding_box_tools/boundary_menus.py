__author__ = "Simon Nilsson"

import glob
import os
import pickle
from tkinter import *
from typing import Union

from simba.bounding_box_tools.agg_boundary_stats import \
    AggBoundaryStatisticsCalculator
from simba.bounding_box_tools.boundary_statistics import \
    BoundaryStatisticsCalculator
from simba.bounding_box_tools.find_boundaries import AnimalBoundaryFinder
from simba.bounding_box_tools.visualize_boundaries import BoundaryVisualizer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import NoChoosenMeasurementError, NoFilesFoundError
from simba.utils.lookups import get_named_colors
from simba.utils.read_write import find_all_videos_in_project, get_fn_ext


class BoundaryMenus(ConfigReader, PopUpMixin):
    """
    Launch GUI interface for extrapolating bounding boxes from pose-estimation data, and calculating
    statstics on bounding boxes and pose-estimated key-point intersections.

    :parameter str config_path: str path to SimBA project config file in Configparser format

    Notes
    ----------
    `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md>`_.

    Examples
    ----------
    >>> _ = BoundaryMenus(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(
            self, title="SIMBA ANCHORED ROI (BOUNDARY BOXES ANALYSIS)", size=(750, 300)
        )
        self.named_shape_colors = get_named_colors()
        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.BBOXES.value,
        )
        self.max_animal_name_char = len(
            max([x for x in list(self.animal_bp_dict.keys())])
        )
        self.find_boundaries_btn = Button(
            self.settings_frm,
            text="FIND ANIMAL BOUNDARIES",
            command=lambda: self.__launch_find_boundaries_pop_up(),
        )
        self.visualize_boundaries_btn = Button(
            self.settings_frm,
            text="VISUALIZE BOUNDARIES",
            command=lambda: self.__launch_visualize_boundaries(),
        )
        self.boundary_statistics_btn = Button(
            self.settings_frm,
            text="CALCULATE BOUNDARY STATISTICS",
            command=lambda: self.__launch_boundary_statistics(),
        )
        self.agg_boundary_statistics_btn = Button(
            self.settings_frm,
            text="CALCULATE AGGREGATE BOUNDARY STATISTICS",
            command=lambda: self.__launch_agg_boundary_statistics(),
        )
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
        self.select_shape_type_frm = LabelFrame(
            self.find_boundaries_frm,
            text="SELECT SHAPE TYPE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=15,
        )
        self.shape_types = [
            "ENTIRE ANIMAL",
            "SINGLE BODY-PART SQUARE",
            "SINGLE BODY-PART CIRCLE",
        ]
        self.shape_dropdown = DropDownMenu(
            self.select_shape_type_frm, "SELECT SHAPE TYPE", self.shape_types, "15"
        )
        self.shape_dropdown.setChoices(self.shape_types[0])
        self.select_btn = Button(
            self.select_shape_type_frm,
            text="SELECT",
            command=lambda: self.__populate_find_boundaries_menu(),
        )
        self.select_shape_type_frm.grid(row=0, column=0, sticky=NW)
        self.shape_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_btn.grid(row=0, column=1, sticky=NW)

    def __populate_find_boundaries_menu(self):
        if hasattr(self, "boundary_settings"):
            self.boundary_settings.destroy()
        self.selected_shape_type = self.shape_dropdown.getChoices()
        self.boundary_settings = LabelFrame(
            self.find_boundaries_frm,
            text="BOUNDARY SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=15,
        )
        boundary_settings_row_cnt = 1
        if (self.selected_shape_type == "SINGLE BODY-PART SQUARE") | (
            self.selected_shape_type == "SINGLE BODY-PART CIRCLE"
        ):
            self.animals = {}
            for animal_cnt, (name, animal_data) in enumerate(
                self.animal_bp_dict.items()
            ):
                self.animals[name] = {}
                self.animals[name]["animal_name_lbl"] = Label(
                    self.boundary_settings,
                    text=name,
                    width=self.max_animal_name_char + 5,
                )
                animal_bps = [x[:-2] for x in animal_data["X_bps"]]
                self.animals[name]["body_part_dropdown"] = DropDownMenu(
                    self.boundary_settings, "BODY-PART: ", animal_bps, " 0"
                )
                self.animals[name]["body_part_dropdown"].setChoices(animal_bps[0])
                self.animals[name]["animal_name_lbl"].grid(
                    row=boundary_settings_row_cnt, column=0, sticky=NW
                )
                self.animals[name]["body_part_dropdown"].grid(
                    row=boundary_settings_row_cnt, column=1, sticky=NW
                )
                boundary_settings_row_cnt += 1
        elif self.selected_shape_type == "ENTIRE ANIMAL":
            self.force_rectangle_var = BooleanVar()
            self.force_rectangle_cb = Checkbutton(
                self.boundary_settings,
                text="FORCE RECTANGLE",
                variable=self.force_rectangle_var,
                command=None,
            )
            self.force_rectangle_cb.grid(
                row=boundary_settings_row_cnt, column=0, sticky=NW
            )
            boundary_settings_row_cnt += 1
        self.boundary_settings.grid(row=1, column=0, sticky=W, padx=5, pady=15)
        self.parallel_offset_entry = Entry_Box(
            self.boundary_settings,
            "PARALLEL OFFSET (MM):",
            labelwidth="18",
            width=10,
            validation="numeric",
        )
        self.run_find_shapes_btn = Button(
            self.boundary_settings,
            text="RUN",
            command=lambda: self.__run_find_boundaries(),
        )
        self.parallel_offset_entry.entry_set("0")
        self.parallel_offset_entry.grid(
            row=boundary_settings_row_cnt, column=0, sticky=NW
        )
        self.run_find_shapes_btn.grid(
            row=boundary_settings_row_cnt + 1, column=0, sticky=NW
        )

    def __run_find_boundaries(self):
        body_parts, force_rectangle = None, False
        if self.selected_shape_type == "ENTIRE ANIMAL":
            force_rectangle = self.force_rectangle_var.get()
            body_parts = None
        elif (self.selected_shape_type == "SINGLE BODY-PART SQUARE") | (
            self.selected_shape_type == "SINGLE BODY-PART CIRCLE"
        ):
            body_parts = {}
            for animal, animal_data in self.animals.items():
                body_parts[animal] = self.animals[animal][
                    "body_part_dropdown"
                ].getChoices()
            force_rectangle = False
        parallel_offset = self.parallel_offset_entry.entry_get
        check_int(name="PARALLEL OFFSET", value=parallel_offset)
        boundary_finder = AnimalBoundaryFinder(
            config_path=self.config_path,
            roi_type=self.selected_shape_type,
            body_parts=body_parts,
            force_rectangle=force_rectangle,
            parallel_offset=int(parallel_offset),
        )
        boundary_finder.run()

    def __launch_visualize_boundaries(self):
        self.anchored_roi_path = os.path.join(
            self.project_path, "logs", "anchored_rois.pickle"
        )
        if not os.path.isfile(self.anchored_roi_path):
            raise NoFilesFoundError(
                msg="No anchored ROI found in {}.".format(self.anchored_roi_path)
            )
        with open(self.anchored_roi_path, "rb") as fp:
            self.roi_data = pickle.load(fp)
        videos_in_project = find_all_videos_in_project(videos_dir=self.video_dir)
        videos_with_data = list(self.roi_data.keys())
        if len(videos_in_project) == 0:
            raise NoFilesFoundError(msg="Zero video files found in SimBA project")
        video_names = []
        for file_path in videos_in_project:
            _, name, _ = get_fn_ext(filepath=file_path)
            video_names.append(name)
        sets_w_data_and_video = list(set(videos_with_data).intersection(video_names))
        if len(sets_w_data_and_video) == 0:
            raise NoFilesFoundError(
                msg="SIMBA ERROR: Zero video files found with calculated anchored ROIs in SimBA project"
            )
        self.viz_boundaries_frm = Toplevel()
        self.viz_boundaries_frm.minsize(600, 150)
        self.viz_boundaries_frm.wm_title("VISUALIZE ANIMAL BOUNDARIES")
        self.video_settings_frm = LabelFrame(
            self.viz_boundaries_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=15,
        )
        self.roi_attr_frm = LabelFrame(
            self.viz_boundaries_frm,
            text="ROI ATTRIBUTES",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=15,
        )
        self.select_video_dropdown = DropDownMenu(
            self.video_settings_frm, "SELECT VIDEO: ", sets_w_data_and_video, " 0"
        )
        self.select_video_dropdown.setChoices(sets_w_data_and_video[0])
        self.run_visualize_roi = Button(
            self.viz_boundaries_frm,
            text="RUN",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="blue",
            command=lambda: self.__run_boundary_visualization(),
        )
        self.include_keypoints_var = BooleanVar()
        self.convert_to_grayscale_var = BooleanVar()
        self.highlight_intersections_var = BooleanVar()
        self.enable_attr_var = BooleanVar()
        self.include_keypoints_cb = Checkbutton(
            self.video_settings_frm,
            text="INCLUDE KEY-POINTS",
            variable=self.include_keypoints_var,
            command=None,
        )
        self.convert_to_grayscale_cb = Checkbutton(
            self.video_settings_frm,
            text="GREYSCALE",
            variable=self.convert_to_grayscale_var,
            command=None,
        )
        self.highlight_intersections_cb = Checkbutton(
            self.video_settings_frm,
            text="HIGHLIGHT INTERSECTIONS",
            variable=self.highlight_intersections_var,
            command=None,
        )
        self.enable_roi_attr_cb = Checkbutton(
            self.video_settings_frm,
            text="ENABLE USER-DEFINED ROI ATTRIBUTES",
            variable=self.enable_attr_var,
            command=self.__enable_roi_attributes,
        )
        self.video_settings_frm.grid(row=0, sticky=NW)
        self.select_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.include_keypoints_cb.grid(row=1, column=0, sticky=NW)
        self.convert_to_grayscale_cb.grid(row=2, column=0, sticky=NW)
        self.highlight_intersections_cb.grid(row=3, column=0, sticky=NW)
        self.enable_roi_attr_cb.grid(row=4, column=0, sticky=NW)
        self.run_visualize_roi.grid(row=5, column=0, sticky=NW)
        self.animal_attr_dict = {}

        Label(
            self.roi_attr_frm,
            text="ANIMAL",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            width=self.max_animal_name_char + 10,
        ).grid(row=0, column=0, sticky=N)
        Label(
            self.roi_attr_frm,
            text="ROI COLOR",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            width=self.max_animal_name_char + 10,
        ).grid(row=0, column=1, sticky=N)
        Label(
            self.roi_attr_frm,
            text="ROI THICKNESS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            width=self.max_animal_name_char + 10,
        ).grid(row=0, column=2, sticky=N)
        Label(
            self.roi_attr_frm,
            text="KEY-POINT SIZE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            width=self.max_animal_name_char + 10,
        ).grid(row=0, column=3, sticky=N)
        Label(
            self.roi_attr_frm,
            text="HIGHLIGHT COLOR",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            width=self.max_animal_name_char + 10,
        ).grid(row=0, column=4, sticky=N)
        Label(
            self.roi_attr_frm,
            text="HIGHLIGHT THICKNESS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            width=self.max_animal_name_char + 10,
        ).grid(row=0, column=5, sticky=N)
        for cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.animal_attr_dict[animal_name] = {}
            self.animal_attr_dict[animal_name]["label"] = Label(
                self.roi_attr_frm, text=animal_name, width=self.max_animal_name_char
            )
            self.animal_attr_dict[animal_name]["clr_dropdown"] = DropDownMenu(
                self.roi_attr_frm, "", self.named_shape_colors, "10", command=None
            )
            self.animal_attr_dict[animal_name]["clr_dropdown"].setChoices(
                self.named_shape_colors[cnt]
            )
            self.animal_attr_dict[animal_name]["thickness_dropdown"] = DropDownMenu(
                self.roi_attr_frm, "", list(range(1, 10)), "10", command=None
            )
            self.animal_attr_dict[animal_name]["thickness_dropdown"].setChoices(1)
            self.animal_attr_dict[animal_name]["circle_size_dropdown"] = DropDownMenu(
                self.roi_attr_frm, "", list(range(1, 10)), "10", command=None
            )
            self.animal_attr_dict[animal_name]["circle_size_dropdown"].setChoices(1)
            self.animal_attr_dict[animal_name]["highlight_clr_dropdown"] = DropDownMenu(
                self.roi_attr_frm, "", self.named_shape_colors, "10", command=None
            )
            self.animal_attr_dict[animal_name]["highlight_clr_dropdown"].setChoices(
                "Red"
            )
            self.animal_attr_dict[animal_name]["highlight_clr_thickness"] = (
                DropDownMenu(
                    self.roi_attr_frm, "", list(range(1, 10)), "10", command=None
                )
            )
            self.animal_attr_dict[animal_name]["highlight_clr_thickness"].setChoices(5)
            self.animal_attr_dict[animal_name]["label"].grid(
                row=cnt + 1, column=0, sticky=NW
            )
            self.animal_attr_dict[animal_name]["clr_dropdown"].grid(
                row=cnt + 1, column=1, sticky=NW
            )
            self.animal_attr_dict[animal_name]["thickness_dropdown"].grid(
                row=cnt + 1, column=2, sticky=NW
            )
            self.animal_attr_dict[animal_name]["circle_size_dropdown"].grid(
                row=cnt + 1, column=3, sticky=NW
            )
            self.animal_attr_dict[animal_name]["highlight_clr_dropdown"].grid(
                row=cnt + 1, column=4, sticky=NW
            )
            self.animal_attr_dict[animal_name]["highlight_clr_thickness"].grid(
                row=cnt + 1, column=5, sticky=NW
            )
        self.roi_attr_frm.grid(row=3, column=0, sticky=NW)
        self.__enable_roi_attributes()

    def __enable_roi_attributes(self):
        if self.enable_attr_var.get():
            for animal_name in self.animal_attr_dict.keys():
                self.animal_attr_dict[animal_name]["clr_dropdown"].enable()
                self.animal_attr_dict[animal_name]["thickness_dropdown"].enable()
                self.animal_attr_dict[animal_name]["circle_size_dropdown"].enable()
                self.animal_attr_dict[animal_name]["highlight_clr_dropdown"].enable()
                self.animal_attr_dict[animal_name]["highlight_clr_thickness"].enable()
        else:
            for animal_name in self.animal_attr_dict.keys():
                self.animal_attr_dict[animal_name]["clr_dropdown"].disable()
                self.animal_attr_dict[animal_name]["thickness_dropdown"].disable()
                self.animal_attr_dict[animal_name]["circle_size_dropdown"].disable()
                self.animal_attr_dict[animal_name]["highlight_clr_dropdown"].disable()
                self.animal_attr_dict[animal_name]["highlight_clr_thickness"].disable()

    def __run_boundary_visualization(self):
        include_keypoints = self.include_keypoints_var.get()
        greyscale = self.convert_to_grayscale_var.get()
        highlight_intersections = self.highlight_intersections_var.get()
        roi_attr = {}
        if self.enable_attr_var.get():
            for cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
                roi_attr[animal_name] = {}
                roi_attr[animal_name]["bbox_clr"] = self.named_shape_colors[
                    self.animal_attr_dict[animal_name]["clr_dropdown"].getChoices()
                ]
                roi_attr[animal_name]["bbox_thickness"] = int(
                    self.animal_attr_dict[animal_name][
                        "thickness_dropdown"
                    ].getChoices()
                )
                roi_attr[animal_name]["keypoint_size"] = int(
                    self.animal_attr_dict[animal_name][
                        "circle_size_dropdown"
                    ].getChoices()
                )
                roi_attr[animal_name]["highlight_clr"] = self.named_shape_colors[
                    self.animal_attr_dict[animal_name][
                        "highlight_clr_dropdown"
                    ].getChoices()
                ]
                roi_attr[animal_name]["highlight_clr_thickness"] = int(
                    self.animal_attr_dict[animal_name][
                        "highlight_clr_thickness"
                    ].getChoices()
                )
        else:
            roi_attr = None

        video_visualizer = BoundaryVisualizer(
            config_path=self.config_path,
            video_name=self.select_video_dropdown.getChoices(),
            include_key_points=include_keypoints,
            greyscale=greyscale,
            show_intersections=highlight_intersections,
            roi_attributes=roi_attr,
        )

        video_visualizer.run()

    def __launch_boundary_statistics(self):
        self.anchored_roi_path = os.path.join(
            self.project_path, "logs", "anchored_rois.pickle"
        )
        if not os.path.isfile(self.anchored_roi_path):
            raise NoFilesFoundError(
                msg="SIMBA ERROR: No anchored ROI found in {}.".format(
                    self.anchored_roi_path
                )
            )
        self.statistics_frm = Toplevel()
        self.statistics_frm.minsize(400, 150)
        self.statistics_frm.wm_title("ANIMAL ANCHORED ROI STATISTICS")
        self.settings_frm = LabelFrame(
            self.statistics_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=15,
        )
        self.file_type_frm = LabelFrame(
            self.statistics_frm,
            text="OUTPUT FILE TYPE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=15,
        )
        self.roi_intersections_var = BooleanVar(value=True)
        self.roi_keypoint_intersections_var = BooleanVar(value=True)
        self.roi_intersections_cb = Checkbutton(
            self.settings_frm,
            text="ROI-ROI INTERSECTIONS",
            variable=self.roi_intersections_var,
            command=None,
        )
        self.roi_keypoint_intersections_cb = Checkbutton(
            self.settings_frm,
            text="ROI-KEYPOINT INTERSECTIONS",
            variable=self.roi_keypoint_intersections_var,
            command=None,
        )
        self.out_file_type = StringVar(value="CSV")
        input_csv_rb = Radiobutton(
            self.file_type_frm, text=".csv", variable=self.out_file_type, value="CSV"
        )
        input_parquet_rb = Radiobutton(
            self.file_type_frm,
            text=".parquet",
            variable=self.out_file_type,
            value="PARQUET",
        )
        input_pickle_rb = Radiobutton(
            self.file_type_frm,
            text=".pickle",
            variable=self.out_file_type,
            value="PICKLE",
        )
        self.run_statistics = Button(
            self.statistics_frm, text="RUN", command=lambda: self.__run_statistics()
        )
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
            raise NoChoosenMeasurementError()
        statistics_calculator = BoundaryStatisticsCalculator(
            config_path=self.config_path,
            roi_intersections=roi_intersections,
            roi_keypoint_intersections=roi_keypoint_intersections,
            save_format=save_format,
        )
        statistics_calculator.save_results()

    def __launch_agg_boundary_statistics(self):
        self.data_path = os.path.join(self.project_path, "csv", "anchored_roi_data")
        if not os.path.isdir(self.data_path):
            raise NoFilesFoundError(
                msg="SIMBA ERROR: No anchored ROI statistics found in {}.".format(
                    self.anchored_roi_path
                )
            )
        self.main_agg_statistics_frm = Toplevel()
        self.main_agg_statistics_frm.minsize(400, 175)
        self.main_agg_statistics_frm.wm_title(
            "ANIMAL ANCHORED ROI AGGREGATE STATISTICS"
        )
        self.agg_settings_frm = LabelFrame(
            self.main_agg_statistics_frm,
            text="SETTINGS",
            font=("Helvetica", 15, "bold"),
            pady=5,
            padx=15,
        )

        self.interaction_time = BooleanVar(value=True)
        self.interaction_bout_cnt = BooleanVar(value=True)
        self.interaction_bout_mean = BooleanVar(value=True)
        self.interaction_bout_median = BooleanVar(value=True)
        self.detailed_interaction_data_var = BooleanVar(value=True)
        self.interaction_time_cb = Checkbutton(
            self.agg_settings_frm,
            text="INTERACTION TIME (s)",
            variable=self.interaction_time,
            command=None,
        )
        self.interaction_bout_cnt_cb = Checkbutton(
            self.agg_settings_frm,
            text="INTERACTION BOUT COUNT",
            variable=self.interaction_bout_cnt,
            command=None,
        )
        self.interaction_bout_mean_cb = Checkbutton(
            self.agg_settings_frm,
            text="INTERACTION BOUT TIME MEAN (s)",
            variable=self.interaction_bout_mean,
            command=None,
        )
        self.interaction_bout_median_cb = Checkbutton(
            self.agg_settings_frm,
            text="INTERACTION BOUT TIME MEDIAN (s)",
            variable=self.interaction_bout_median,
            command=None,
        )
        self.detailed_interaction_data_cb = Checkbutton(
            self.agg_settings_frm,
            text="DETAILED INTERACTIONS TABLE",
            variable=self.detailed_interaction_data_var,
            command=None,
        )
        self.minimum_bout_entry_box = Entry_Box(
            self.agg_settings_frm,
            "MINIMUM BOUT LENGTH (MS):",
            labelwidth="25",
            width=10,
            validation="numeric",
        )
        self.run_btn = Button(
            self.main_agg_statistics_frm,
            text="CALCULATE AGGREGATE STATISTICS",
            command=lambda: self._run_agg_stats(),
        )

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
        for cb, name in zip(
            [
                self.interaction_time,
                self.interaction_bout_cnt,
                self.interaction_bout_mean,
                self.interaction_bout_median,
                self.detailed_interaction_data_var,
            ],
            [
                "INTERACTION TIME (s)",
                "INTERACTION BOUT COUNT",
                "INTERACTION BOUT TIME MEAN (s)",
                "INTERACTION BOUT TIME MEDIAN (s)",
                "DETAILED INTERACTIONS TABLE",
            ],
        ):
            if cb.get():
                measures.append(name)
        min_bout = self.minimum_bout_entry_box.entry_get
        check_int(name="MIN BOUT LENGTH", value=min_bout)
        if min_bout == "":
            min_bout = 0
        if len(measures) == 0:
            raise NoChoosenMeasurementError()

        agg_stats_calculator = AggBoundaryStatisticsCalculator(
            config_path=self.config_path,
            measures=measures,
            shortest_allowed_interaction=int(min_bout),
        )
        agg_stats_calculator.run()
        agg_stats_calculator.save()


# test = BoundaryMenus(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.main_frm.mainloop()
