__author__ = "Simon Nilsson"

import threading
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FolderSelect
from simba.unsupervised.clusterer_comparison_calculator import \
    ClustererComparisonCalculator
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Formats, Options
from simba.utils.errors import CountError

ADJ_MUTUAL_INFO = "ADJUSTED MUTUAL INFORMATION"
FOWLKES_MALLOWS = "FOWLKES MALLOWS"
ADJ_RAND_INDEX = "ADJUSTED RAND INDEX"


class ClustererComparisonPopUp(PopUpMixin, ConfigReader):
    """
    Pop up for computing clusterer comparisons.

    :example:
    >>> _ = ClustererComparisonPopUp(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini')
    """

    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="DATA EXTRACTOR")
        ConfigReader.__init__(self, config_path=config_path)
        data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.data_dir_select = FolderSelect(
            data_frm,
            "DATA DIRECTORY (PICKLES):",
            lblwidth=25,
            initialdir=self.project_path,
        )
        data_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_select.grid(row=0, column=0, sticky=NW)
        self.cb_dict = self.create_cb_frame(
            main_frm=self.main_frm,
            cb_titles=[ADJ_MUTUAL_INFO, FOWLKES_MALLOWS, ADJ_RAND_INDEX],
            frm_title="STATISTICS",
        )
        self.create_run_frm(title="RUN", run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        stats_selections = []
        check_if_dir_exists(in_dir=self.data_dir_select.folder_path)
        for k, v in self.cb_dict.items():
            if v.get():
                stats_selections.append(k)
        if len(stats_selections) == 0:
            raise CountError(
                msg="Select at least ONE measurement.", source=self.__class__.__name__
            )
        cluster_comparer = ClustererComparisonCalculator(
            config_path=self.config_path,
            data_dir=self.data_dir_select.folder_path,
            statistics=[x.lower() for x in stats_selections],
        )
        cluster_comparer.run()


# ClustererComparisonPopUp(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini')
