import glob
import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.statistics_mixin import Statistics
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.ui.tkinter_functions import FolderSelect
from simba.unsupervised.cluster_validation_calculator import ClusterValidators
from simba.unsupervised.dbcv_calculator import DBCVCalculator
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.enums import Formats
from simba.utils.errors import CountError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success

VALIDATION_METRICS = [
    "DENSITY-BASED CLUSTER VALIDATION (DBCV)",
    "DUNN INDEX",
    "DAVIS-BUILDIN INDEX",
    "CALINSKI-HARABASZ SCORE",
]
DBCV = "DENSITY-BASED CLUSTER VALIDATION (DBCV)"
DUNN = "DUNN INDEX"
DAVIS_BUILDIN = "DAVIS-BUILDIN INDEX"
CALINSKI_HARABASZ = "CALINSKI-HARABASZ SCORE"


class ClusterValidatorPopUp(PopUpMixin, ConfigReader, UnsupervisedMixin):
    def __init__(self, config_path: Union[str, os.PathLike]):
        check_file_exist_and_readable(file_path=config_path)
        PopUpMixin.__init__(self, title="CLUSTER VALIDATION METRICS")
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.folder_selected = FolderSelect(
            self.data_frm,
            "DATASETS (DIRECTORY WITH PICKLES):",
            lblwidth=35,
            initialdir=self.project_path,
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.folder_selected.grid(row=0, column=0, sticky=NW)
        self.cb_settings = self.create_cb_frame(
            main_frm=self.main_frm,
            cb_titles=VALIDATION_METRICS,
            frm_title="VALIDATION METHODS",
        )
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        timer = SimbaTimer(start=True)
        check_if_dir_exists(in_dir=self.folder_selected.folder_path)
        data_paths = glob.glob(
            self.folder_selected.folder_path + f"/*.{Formats.PICKLE.value}"
        )
        check_if_filepath_list_is_empty(
            filepaths=data_paths,
            error_msg=f"No pickle files found in {self.folder_selected.folder_path}",
        )
        selections = []
        for k, v in self.cb_settings.items():
            if v.get():
                selections.append(k)
        if len(selections) == 0:
            raise CountError(
                msg="Select at least 1 data-type checkbox",
                source=self.__class__.__name__,
            )
        for selection in selections:
            if selection == DBCV:
                print(f"Running {DBCV}...")
                dbcv_calculator = DBCVCalculator(
                    data_path=self.folder_selected.folder_path,
                    config_path=self.config_path,
                )
                dbcv_calculator.run()
            else:
                if selection == DUNN:
                    validator_func = Statistics.dunn_index
                elif selection == DAVIS_BUILDIN:
                    validator_func = Statistics.davis_bouldin
                elif selection == CALINSKI_HARABASZ:
                    validator_func = Statistics.calinski_harabasz
                else:
                    raise InvalidInputError(
                        msg=f"Validator algorithm {selection} is not a valid options. Options: {VALIDATION_METRICS}",
                        source=self.__class__.__name__,
                    )
                print(f"Running {selection}...")
                calculator = ClusterValidators(
                    config_path=self.config_path,
                    data_path=self.folder_selected.folder_path,
                    validator_func=validator_func,
                )
                calculator.run()
                calculator.save()
        timer.stop_timer()
        stdout_success(
            msg="Cluster validations complete!", source=timer.elapsed_time_str
        )


# _ = ClusterValidatorPopUp(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini')
