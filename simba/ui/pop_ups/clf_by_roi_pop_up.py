__author__ = "Simon Nilsson"

import os
from collections import defaultdict
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_clf_calculator import ROIClfCalculator
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        SimbaButton, SimbaCheckbox)
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import (NoChoosenClassifierError,
                                NoChoosenMeasurementError, NoChoosenROIError,
                                NoDataError, NoROIDataError,
                                ROICoordinatesNotFoundError)

MEASURES = ('TOTAL BEHAVIOR TIME IN ROI (S)', 'STARTED BEHAVIOR BOUTS IN ROI (COUNT)', 'ENDED BEHAVIOR BOUTS IN ROI (COUNT)')



class ClfByROIPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = ClfByROIPopUp(config_path=r"C:\troubleshooting\open_field_below\project_folder\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path, source=self.__class__.__name__)
        if len(self.machine_results_paths) == 0:
            raise NoDataError(f'Cannot compute ROI by classifier data: No data exist in {self.machine_results_dir} directory.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="CLASSIFICATIONS BY ROI", icon='shapes_small')
        self.read_roi_data()
        roi_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT ROIs", icon_name='roi', padx=5, pady=5, relief='solid')

        self.roi_vars = {}
        for roi_cnt, roi_name in enumerate(self.roi_names):
            roi_cb, self.roi_vars[roi_name] = SimbaCheckbox(parent=roi_frm, txt=roi_name, val=True)
            roi_cb.grid(row=roi_cnt, sticky=NW)
        roi_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)

        clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT CLASSIFIERS", icon_name='forest', padx=5, pady=5, relief='solid')
        self.clf_vars = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            clf_cb, self.clf_vars[clf_name] = SimbaCheckbox(parent=clf_frm, txt=clf_name, val=True)
            clf_cb.grid(row=clf_cnt, sticky=NW)
        clf_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)


        measurements_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT MEASUREMENTS", icon_name='ruler', padx=5, pady=5, relief='solid')
        measurements_frm.grid(row=2, column=0, sticky=NW, padx=10, pady=10)
        self.total_time_cb, self.total_time_var = SimbaCheckbox(parent=measurements_frm, txt='TOTAL BEHAVIOR TIME IN ROI (S)', txt_img='timer_2', val=True)
        self.total_time_cb.grid(row=0, column=0, sticky=NW)
        self.start_bouts_cb, self.start_bouts_var = SimbaCheckbox(parent=measurements_frm, txt='STARTED BEHAVIOR BOUTS IN ROI (COUNT)', txt_img='abacus', val=True)
        self.start_bouts_cb.grid(row=1, column=0, sticky=NW)
        self.end_bouts_cb, self.end_bouts_var = SimbaCheckbox(parent=measurements_frm, txt='ENDED BEHAVIOR BOUTS IN ROI (COUNT)', txt_img='abacus', val=True)
        self.end_bouts_cb.grid(row=2, column=0, sticky=NW)
        self.detailed_bouts_cb, self.detailed_bouts_var = SimbaCheckbox(parent=measurements_frm, txt='DETAILED BOUTS TABLE - EACH BEHAVIOR EVENT BY ROI (START/END TIME)', txt_img='abacus', val=True)
        self.detailed_bouts_cb.grid(row=3, column=0, sticky=NW)


        bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        self.bp_vars = {}
        for bp_cnt, bp_name in enumerate(self.body_parts_lst):
            bp_cb, self.bp_vars[bp_name] = SimbaCheckbox(parent=bp_frm, txt=bp_name)
            bp_cb.grid(row=bp_cnt, sticky=NW)
        bp_frm.grid(row=3, column=0, sticky=NW, padx=10, pady=10)
        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        self.selected_rois, self.selected_bps = [], []
        self.selected_clfs, self.selected_measures = [], []
        for k, v in self.roi_vars.items():
            if v.get(): self.selected_rois.append(k)
        for k, v in self.clf_vars.items():
            if v.get(): self.selected_clfs.append(k)
        for k, v in self.bp_vars.items():
            if v.get(): self.selected_bps.append(k)
        if len(self.selected_rois) == 0:
            raise NoROIDataError(msg='Please check AT LEAST ONE ROI.', source=self.__class__.__name__)
        if len(self.selected_bps) == 0:
            raise NoDataError(msg='Please check AT LEAST ONE BODY-PART.', source=self.__class__.__name__)
        if len(self.selected_clfs) == 0:
            raise NoDataError(msg='Please check AT LEAST ONE CLASSIFIER.', source=self.__class__.__name__)
        total_time = self.total_time_var.get()
        started_bouts = self.start_bouts_var.get()
        ended_bouts = self.end_bouts_var.get()
        detailed_bouts = self.detailed_bouts_var.get()

        if not any([total_time, started_bouts, ended_bouts, detailed_bouts]):
            raise NoDataError(msg='Please check AT LEAST ONE MEASUREMENT,', source=self.__class__.__name__)

        analyzer = ROIClfCalculator(config_path=self.config_path,
                                    bp_names=self.selected_bps,
                                    save_path=None,
                                    data_paths=None,
                                    clf_names=self.selected_clfs,
                                    roi_names=self.selected_rois,
                                    clf_time=total_time,
                                    started_bout_cnt=started_bouts,
                                    ended_bout_cnt=ended_bouts,
                                    bout_table=detailed_bouts)
        analyzer.run()
        analyzer.save()

# x = ClfByROIPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# x.main_frm.mainloop()