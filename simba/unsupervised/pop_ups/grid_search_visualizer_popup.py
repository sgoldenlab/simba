__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FolderSelect
from simba.unsupervised.enums import Clustering, UMLOptions, Unsupervised
from simba.unsupervised.grid_search_visualizers import GridSearchVisualizer
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoSpecifiedOutputError


class GridSearchVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(
            self, config_path=config_path, title="GRID SEARCH VISUALIZER"
        )
        data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            fg="black",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.data_dir_select = FolderSelect(
            data_frm, "DATA DIRECTORY:", lblwidth=25, initialdir=self.project_path
        )
        self.save_dir_select = FolderSelect(
            data_frm, "OUTPUT DIRECTORY: ", lblwidth=25, initialdir=self.project_path
        )
        data_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_select.grid(row=0, column=0, sticky=NW)
        self.save_dir_select.grid(row=1, column=0, sticky=NW)

        self.visualization_options = (
            UMLOptions.CATEGORICAL_OPTIONS.value + UMLOptions.CONTINUOUS_OPTIONS.value
        )
        settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            padx=5,
            pady=5,
            fg="black",
        )
        self.scatter_size_dropdown = DropDownMenu(
            settings_frm, "SCATTER SIZE:", UMLOptions.SCATTER_SIZE.value, "25"
        )
        self.scatter_size_dropdown.setChoices(5)
        self.categorical_palette_dropdown = DropDownMenu(
            settings_frm,
            "CATEGORICAL PALETTE:",
            Options.PALETTE_OPTIONS_CATEGORICAL.value,
            "25",
        )
        self.categorical_palette_dropdown.setChoices("Set1")
        self.continuous_palette_dropdown = DropDownMenu(
            settings_frm, "CONTINUOUS PALETTE:", Options.PALETTE_OPTIONS.value, "25"
        )
        self.continuous_palette_dropdown.setChoices("magma")

        settings_frm.grid(row=1, column=0, sticky=NW)
        self.scatter_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.categorical_palette_dropdown.grid(row=1, column=0, sticky=NW)
        self.continuous_palette_dropdown.grid(row=2, column=0, sticky=NW)

        self.define_plots_frm = LabelFrame(
            self.main_frm,
            text="DEFINE PLOTS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            padx=5,
            pady=5,
            fg="black",
        )
        self.plot_cnt_dropdown = DropDownMenu(
            self.define_plots_frm,
            "# PLOTS:",
            UMLOptions.GRAPH_CNT.value,
            "25",
            com=lambda x: self.show_plot_table(),
        )
        self.plot_cnt_dropdown.setChoices(UMLOptions.GRAPH_CNT.value[0])
        self.show_plot_table()
        self.define_plots_frm.grid(row=2, column=0, sticky=NW)
        self.plot_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(title="RUN", run_function=self.run)
        self.main_frm.mainloop()

    def show_plot_table(self):
        if hasattr(self, "plot_table"):
            self.plot_table.destroy()

        self.plot_data = {}
        self.plot_table = LabelFrame(
            self.define_plots_frm,
            text="PLOTS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            padx=5,
            pady=5,
            fg="black",
        )
        self.scatter_name_header = Label(self.plot_table, text="PLOT NAME").grid(
            row=0, column=0
        )
        self.field_name_header = Label(self.plot_table, text="COLOR VARIABLE").grid(
            row=0, column=1
        )
        for idx in range(int(self.plot_cnt_dropdown.getChoices())):
            row_name = idx
            self.plot_data[row_name] = {}
            self.plot_data[row_name]["label"] = Label(
                self.plot_table, text=f"Scatter {str(idx+1)}:"
            )
            self.plot_data[row_name]["variable"] = DropDownMenu(
                self.plot_table, " ", self.visualization_options, "10", com=None
            )
            self.plot_data[row_name]["variable"].setChoices(
                self.visualization_options[0]
            )
            self.plot_data[idx]["label"].grid(row=idx + 1, column=0, sticky=NW)
            self.plot_data[idx]["variable"].grid(row=idx + 1, column=1, sticky=NW)
        self.plot_table.grid(row=1, column=0, sticky=NW)

    def run(self):
        if len(self.plot_data.keys()) < 1:
            raise NoSpecifiedOutputError(msg="Specify at least one plot")
        settings = {}
        settings["SCATTER_SIZE"] = int(self.scatter_size_dropdown.getChoices())
        settings["CATEGORICAL_PALETTE"] = self.categorical_palette_dropdown.getChoices()
        settings["CONTINUOUS_PALETTE"] = self.continuous_palette_dropdown.getChoices()

        continuous_vars, categorical_vars = [], []
        for k, v in self.plot_data.items():
            if v["variable"].getChoices() in UMLOptions.CONTINUOUS_OPTIONS.value:
                continuous_vars.append(v["variable"].getChoices())
            else:
                categorical_vars.append(v["variable"].getChoices())
        grid_search_visualizer = GridSearchVisualizer(
            model_dir=self.data_dir_select.folder_path,
            save_dir=self.save_dir_select.folder_path,
            settings=settings,
        )
        if len(continuous_vars) > 0:
            threading.Thread(
                target=grid_search_visualizer.continuous_visualizer(
                    continuous_vars=continuous_vars
                )
            ).start()
        if len(categorical_vars) > 0:
            threading.Thread(
                target=grid_search_visualizer.categorical_visualizer(
                    categorical_vars=categorical_vars
                )
            ).start()


# _ = GridSearchVisualizerPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
