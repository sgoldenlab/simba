__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import glob
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FileSelect
from simba.unsupervised.embedding_correlation_calculator import \
    EmbeddingCorrelationCalculator
from simba.unsupervised.enums import UMLOptions
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats, Options


class EmbedderCorrelationsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="EMBEDDING CORRELATIONS")
        ConfigReader.__init__(self, config_path=config_path)
        self.spearman_var = BooleanVar(value=True)
        self.pearsons_var = BooleanVar(value=True)
        self.kendall_var = BooleanVar(value=True)
        self.plots_var = BooleanVar(value=False)
        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.data_file_selected = FileSelect(
            self.data_frm,
            "DATASET (PICKLE):",
            initialdir=self.project_path,
            file_types=[("SimBA model (pickle", f"*.{Formats.PICKLE.value}")],
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.data_file_selected.grid(row=0, column=0, sticky=NW)

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.spearman_cb = Checkbutton(
            self.settings_frm, text="SPEARMAN", variable=self.spearman_var
        )
        self.pearsons_cb = Checkbutton(
            self.settings_frm, text="PEARSONS", variable=self.pearsons_var
        )
        self.kendall_cb = Checkbutton(
            self.settings_frm, text="KENDALL", variable=self.kendall_var
        )
        self.plot_correlation_dropdown = DropDownMenu(
            self.settings_frm,
            "PLOT CORRELATION:",
            UMLOptions.CORRELATION_OPTIONS.value,
            "25",
        )
        self.plot_correlation_clr_dropdown = DropDownMenu(
            self.settings_frm, "PLOT PALETTE:", Options.PALETTE_OPTIONS.value, "25"
        )
        self.plot_correlation_dropdown.setChoices(
            UMLOptions.CORRELATION_OPTIONS.value[0]
        )
        self.plot_correlation_dropdown.disable()
        self.plot_correlation_clr_dropdown.setChoices(Options.PALETTE_OPTIONS.value[0])
        self.plot_correlation_clr_dropdown.disable()
        self.plots_cb = Checkbutton(
            self.settings_frm,
            text="PLOTS",
            variable=self.plots_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.plots_var,
                dropdown_menus=[
                    self.plot_correlation_dropdown,
                    self.plot_correlation_clr_dropdown,
                ],
            ),
        )

        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.spearman_cb.grid(row=0, column=0, sticky=NW)
        self.pearsons_cb.grid(row=1, column=0, sticky=NW)
        self.kendall_cb.grid(row=2, column=0, sticky=NW)
        self.plots_cb.grid(row=3, column=0, sticky=NW)
        self.plot_correlation_dropdown.grid(row=4, column=0, sticky=NW)
        self.plot_correlation_clr_dropdown.grid(row=5, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(self.data_file_selected.file_path)
        settings = {
            "correlations": [],
            "plots": {"create": False, "correlations": None, "palette": None},
        }
        if self.spearman_var.get():
            settings["correlations"].append("spearman")
        if self.pearsons_var.get():
            settings["correlations"].append("pearson")
        if self.kendall_var.get():
            settings["correlations"].append("kendall")
        if self.plots_var.get():
            settings["plots"]["create"] = True
            settings["plots"][
                "correlations"
            ] = self.plot_correlation_dropdown.getChoices()
            settings["plots"][
                "palette"
            ] = self.plot_correlation_clr_dropdown.getChoices()

        calculator = EmbeddingCorrelationCalculator(
            config_path=self.config_path,
            data_path=self.data_file_selected.file_path,
            settings=settings,
        )
        threading.Thread(target=calculator.run()).start()


# _ = EmbedderCorrelationsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
