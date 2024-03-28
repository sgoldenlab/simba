__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import FileSelect
from simba.unsupervised.cluster_frequentist_calculator import \
    ClusterFrequentistCalculator
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats


class ClusterFrequentistStatisticsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CLUSTER FREQUENTIST STATISTICS")
        ConfigReader.__init__(self, config_path=config_path)
        self.descriptive_stats_var = BooleanVar(value=True)
        self.kruskal_wallis_var = BooleanVar(value=True)
        self.oneway_anova_var = BooleanVar(value=True)
        self.tukey_var = BooleanVar(value=True)
        self.use_scaled_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.model_select = FileSelect(
            self.data_frm,
            "CLUSTERER PATH:",
            lblwidth=25,
            initialdir=self.project_path,
            file_types=[("Cluster model pickle", f"*.{Formats.PICKLE.value}")],
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)

        self.stats_frm = LabelFrame(
            self.main_frm,
            text="STATISTICS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.descriptive_stats_cb = Checkbutton(
            self.stats_frm,
            text="CLUSTER DESCRIPTIVE STATISTICS",
            variable=self.descriptive_stats_var,
        )
        self.oneway_anova_cb = Checkbutton(
            self.stats_frm,
            text="CLUSTER FEATURE ONE-WAY ANOVA",
            variable=self.oneway_anova_var,
        )
        self.feature_tukey_posthoc_cb = Checkbutton(
            self.stats_frm,
            text="CLUSTER FEATURE POST-HOC (TUKEY)",
            variable=self.tukey_var,
        )
        self.use_scaled_cb = Checkbutton(
            self.stats_frm,
            text="USE SCALED FEATURE VALUES",
            variable=self.use_scaled_var,
        )

        self.kruskal_wallis_cb = Checkbutton(
            self.stats_frm,
            text="KRUSKAL-WALLIS",
            variable=self.kruskal_wallis_var,
        )

        self.stats_frm.grid(row=1, column=0, sticky=NW)
        self.descriptive_stats_cb.grid(row=0, column=0, sticky=NW)
        self.oneway_anova_cb.grid(row=1, column=0, sticky=NW)
        self.kruskal_wallis_cb.grid(row=2, column=0, sticky=NW)
        self.feature_tukey_posthoc_cb.grid(row=3, column=0, sticky=NW)
        self.use_scaled_cb.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(self.model_select.file_path)
        settings = {
            "scaled": self.use_scaled_var.get(),
            "anova": self.oneway_anova_var.get(),
            "tukey_posthoc": self.tukey_var.get(),
            "kruskal_wallis": self.kruskal_wallis_var.get(),
            "descriptive_statistics": self.descriptive_stats_var.get(),
        }
        calculator = ClusterFrequentistCalculator(
            config_path=self.config_path,
            data_path=self.model_select.file_path,
            settings=settings,
        )
        calculator.run()


# _ = ClusterFrequentistStatisticsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
