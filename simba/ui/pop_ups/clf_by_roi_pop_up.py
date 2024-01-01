__author__ = "Simon Nilsson"

import os
from collections import defaultdict
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_clf_calculator import ROIClfCalculator
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu
from simba.utils.enums import Keys, Links
from simba.utils.errors import (NoChoosenClassifierError,
                                NoChoosenMeasurementError, NoChoosenROIError,
                                ROICoordinatesNotFoundError)


class ClfByROIPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(
                expected_file_path=self.roi_coordinates_path,
                source=self.__class__.__name__,
            )
        PopUpMixin.__init__(self, title="CLASSIFICATIONS BY ROI")
        self.read_roi_data()
        body_part_menu = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Select body part",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ANALYZE_ML_RESULTS.value,
        )
        ROI_menu = LabelFrame(self.main_frm, text="Select ROI(s)", padx=5, pady=5)
        classifier_menu = LabelFrame(
            self.main_frm, text="Select classifier(s)", padx=5, pady=5
        )
        measurements_menu = LabelFrame(
            self.main_frm, text="Select measurements", padx=5, pady=5
        )
        self.total_time_var = BooleanVar()
        self.start_bouts_var = BooleanVar()
        self.end_bouts_var = BooleanVar()
        self.total_time_cb = Checkbutton(
            measurements_menu,
            text="Total time by ROI (s)",
            variable=self.total_time_var,
        )
        self.start_bouts_cb = Checkbutton(
            measurements_menu,
            text="Started bouts by ROI (count)",
            variable=self.start_bouts_var,
        )
        self.end_bouts_cb = Checkbutton(
            measurements_menu,
            text="Ended bouts by ROI (count)",
            variable=self.end_bouts_var,
        )
        self.ROI_check_boxes_status_dict = {}
        self.clf_check_boxes_status_dict = {}

        for row_number, ROI in enumerate(self.roi_types_names_lst):
            self.ROI_check_boxes_status_dict[ROI] = IntVar()
            ROI_check_button = Checkbutton(
                ROI_menu, text=ROI, variable=self.ROI_check_boxes_status_dict[ROI]
            )
            ROI_check_button.grid(row=row_number, sticky=W)

        for row_number, clf_name in enumerate(self.clf_names):
            self.clf_check_boxes_status_dict[clf_name] = IntVar()
            clf_check_button = Checkbutton(
                classifier_menu,
                text=clf_name,
                variable=self.clf_check_boxes_status_dict[clf_name],
            )
            clf_check_button.grid(row=row_number, sticky=W)

        self.choose_bp = DropDownMenu(
            body_part_menu, "Body part", self.body_parts_lst, "12"
        )
        self.choose_bp.setChoices(self.body_parts_lst[0])
        self.choose_bp.grid(row=0, sticky=W)
        run_analysis_button = Button(
            self.main_frm,
            text="Analyze classifications in each ROI",
            command=lambda: self.run_clf_by_ROI_analysis(),
        )
        body_part_menu.grid(row=0, sticky=W, padx=10, pady=10)
        ROI_menu.grid(row=1, sticky=W, padx=10, pady=10)
        classifier_menu.grid(row=2, sticky=W, padx=10, pady=10)
        self.total_time_cb.grid(row=0, sticky=NW)
        self.start_bouts_cb.grid(row=1, sticky=NW)
        self.end_bouts_cb.grid(row=2, sticky=NW)
        measurements_menu.grid(row=3, sticky=W, padx=10, pady=10)
        run_analysis_button.grid(row=4, sticky=W, padx=10, pady=10)

        self.main_frm.mainloop()

    def run_clf_by_ROI_analysis(self):
        body_part_list = [self.choose_bp.getChoices()]
        ROI_dict_lists, behavior_list = defaultdict(list), []
        measurements_list = []
        for loop_val, ROI_entry in enumerate(self.ROI_check_boxes_status_dict):
            check_val = self.ROI_check_boxes_status_dict[ROI_entry]
            if check_val.get() == 1:
                shape_type = (
                    self.roi_types_names_lst[loop_val].split(":")[0].replace(":", "")
                )
                shape_name = self.roi_types_names_lst[loop_val].split(":")[1][1:]
                ROI_dict_lists[shape_type].append(shape_name)

        for measurement_var, measurement_name in zip(
            [
                self.total_time_var.get(),
                self.start_bouts_var.get(),
                self.end_bouts_var.get(),
            ],
            [
                "Total time by ROI (s)",
                "Started bouts by ROI (count)",
                "Ended bouts by ROI (count)",
            ],
        ):
            if measurement_var:
                measurements_list.append(measurement_name)

        for loop_val, clf_entry in enumerate(self.clf_check_boxes_status_dict):
            check_val = self.clf_check_boxes_status_dict[clf_entry]
            if check_val.get() == 1:
                behavior_list.append(self.clf_names[loop_val])
        if len(ROI_dict_lists) == 0:
            raise NoChoosenROIError(source=self.__class__.__name__)
        if len(behavior_list) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)
        if len(measurements_list) == 0:
            raise NoChoosenMeasurementError(source=self.__class__.__name__)
        else:
            self.clf_roi_analyzer = ROIClfCalculator(config_ini=self.config_path)
            self.clf_roi_analyzer.run(
                ROI_dict_lists=ROI_dict_lists,
                behavior_list=behavior_list,
                body_part_list=body_part_list,
                measurements=measurements_list,
            )


#
# _ = ClfByROIPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
