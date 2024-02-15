__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        FolderSelect)
from simba.unsupervised.enums import UMLOptions, Unsupervised
from simba.unsupervised.tsne import TSNEGridSearch
from simba.unsupervised.umap_embedder import UmapEmbedder
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoSpecifiedOutputError


class FitDimReductionPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="FIT DIMENSIONALITY REDUCTION MODELS")
        ConfigReader.__init__(self, config_path=config_path)
        self.variance_options = [
            str(x) + "%" for x in UMLOptions.VARIANCE_OPTIONS.value
        ]
        self.dataset_frm = LabelFrame(
            self.main_frm,
            text="DATASET",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.dataset_file_selected = FileSelect(
            self.dataset_frm,
            "DATASET (PICKLE):",
            lblwidth=25,
            file_types=[("Dataset pickle", "*.pickle")],
            initialdir=self.logs_path,
        )
        self.dataset_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)

        self.save_frm = LabelFrame(
            self.main_frm,
            text="SAVE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.save_dir = FolderSelect(
            self.save_frm, "SAVE DIRECTORY:", lblwidth=25, initialdir=self.project_path
        )
        self.save_frm.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)

        settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.scaling_dropdown = DropDownMenu(
            settings_frm, "SCALING:", Options.SCALER_NAMES.value, "25"
        )
        self.scaling_dropdown.setChoices(Options.SCALER_NAMES.value[0])
        self.var_threshold_dropdown = DropDownMenu(
            settings_frm, "VARIANCE THRESHOLD:", self.variance_options, "25"
        )
        self.var_threshold_dropdown.setChoices(self.variance_options[0])
        self.algo_dropdown = DropDownMenu(
            settings_frm,
            "ALGORITHM:",
            UMLOptions.DR_ALGO_OPTIONS.value,
            "25",
            com=lambda x: self.show_dr_hyperparameters(),
        )
        self.algo_dropdown.setChoices(UMLOptions.DR_ALGO_OPTIONS.value[0])
        self.show_dr_hyperparameters()

        settings_frm.grid(row=2, column=0, sticky=NW)
        self.scaling_dropdown.grid(row=0, column=0, sticky=NW)
        self.var_threshold_dropdown.grid(row=1, column=0, sticky=NW)
        self.algo_dropdown.grid(row=2, column=0, sticky=NW)
        self.main_frm.mainloop()

    def show_dr_hyperparameters(self):
        if hasattr(self, "hyperparameters_frm"):
            self.hyperparameters_frm.destroy()
            self.value_frm.destroy()
            self.run_frm.destroy()

        self.hyperparameters_frm = LabelFrame(
            self.main_frm,
            text="GRID SEARCH HYPER-PARAMETERS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.value_frm = LabelFrame(self.main_frm, fg="black")
        self.dr_value_entry_box = Entry_Box(self.value_frm, "VALUE: ", "12")
        self.value_frm.grid(row=3, column=0, sticky=NW)
        self.dr_value_entry_box.grid(row=0, column=1, sticky=NW)
        self.hyperparameters_frm.grid(row=4, column=0, sticky=NW)

        if self.algo_dropdown.getChoices() == Unsupervised.UMAP.value:
            Label(self.hyperparameters_frm, text=Unsupervised.N_NEIGHBORS.value).grid(
                row=1, column=0
            )
            Label(self.hyperparameters_frm, text=Unsupervised.MIN_DISTANCE.value).grid(
                row=1, column=1
            )
            Label(self.hyperparameters_frm, text=Unsupervised.SPREAD.value).grid(
                row=1, column=2
            )

            self.n_neighbors_estimators_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )
            self.min_distance_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )
            self.spread_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )

            neighbours_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.n_neighbors_estimators_listbox,
                    entry_box=self.dr_value_entry_box,
                ),
            )
            min_distance_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.min_distance_listbox,
                    entry_box=self.dr_value_entry_box,
                ),
            )
            spread_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.spread_listbox, entry_box=self.dr_value_entry_box
                ),
            )

            neighbours_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(
                    list_box=self.n_neighbors_estimators_listbox
                ),
            )
            min_distance_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(
                    list_box=self.min_distance_listbox
                ),
            )
            spread_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(list_box=self.spread_listbox),
            )

            self.add_values_to_several_listboxes(
                list_boxes=[
                    self.n_neighbors_estimators_listbox,
                    self.min_distance_listbox,
                    self.spread_listbox,
                ],
                values=[15, 0.1, 1],
            )

            neighbours_add_btn.grid(row=2, column=0)
            min_distance_add_btn.grid(row=2, column=1)
            spread_add_btn.grid(row=2, column=2)
            neighbours_remove_btn.grid(row=3, column=0)
            min_distance_remove_btn.grid(row=3, column=1)
            spread_remove_btn.grid(row=3, column=2)

            self.n_neighbors_estimators_listbox.grid(row=4, column=0, sticky=NW)
            self.min_distance_listbox.grid(row=4, column=1, sticky=NW)
            self.spread_listbox.grid(row=4, column=2, sticky=NW)
            self.create_run_frm(run_function=self.__run_umap_gridsearch)

        elif self.algo_dropdown.getChoices() == Unsupervised.TSNE.value:
            Label(self.hyperparameters_frm, text="PERPLEXITY").grid(row=1, column=0)
            self.perplexity_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )
            perplexity_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.perplexity_listbox, entry_box=self.dr_value_entry_box
                ),
            )
            perplexity_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(
                    list_box=self.perplexity_listbox
                ),
            )
            perplexity_add_btn.grid(row=2, column=0)
            perplexity_remove_btn.grid(row=3, column=0)
            self.perplexity_listbox.grid(row=4, column=0, sticky=NW)
            self.create_run_frm(run_function=self.__run_tsne_gridsearch)

    def __run_tsne_gridsearch(self):
        self.__get_settings()
        perplexities = [int(x) for x in self.perplexity_listbox.get(0, END)]
        if len(perplexities) == 0:
            raise NoSpecifiedOutputError("Provide value(s) for perplexity")
        hyperparameters = {
            "perplexity": perplexities,
            "scaler": self.scaling_dropdown.getChoices(),
            "variance": self.variance_selected,
        }
        tsne_searcher = TSNEGridSearch(
            data_path=self.data_path, save_dir=self.save_path
        )
        # tsne_searcher.fit(hyperparameters=hyperparameters)

    def __run_umap_gridsearch(self):
        self.__get_settings()
        n_neighbours = [
            float(x) for x in self.n_neighbors_estimators_listbox.get(0, END)
        ]
        min_distances = [float(x) for x in self.min_distance_listbox.get(0, END)]
        spreads = [float(x) for x in self.spread_listbox.get(0, END)]
        if len(min_distances) == 0 or len(n_neighbours) == 0 or len(spreads) == 0:
            raise NoSpecifiedOutputError(
                "Provide at least one hyperparameter value for neighbors, min distances, and spread"
            )
        hyper_parameters = {
            "n_neighbors": n_neighbours,
            "min_distance": min_distances,
            "spread": spreads,
            "scaler": self.scaling_dropdown.getChoices(),
            "variance": self.variance_selected,
        }
        umap_searcher = UmapEmbedder()
        threading.Thread(
            target=umap_searcher.fit(
                data_path=self.data_path,
                save_dir=self.save_path,
                hyper_parameters=hyper_parameters,
            )
        ).start()

    def __get_settings(self):
        self.variance_selected = (
            int(self.var_threshold_dropdown.getChoices()[:-1]) / 100
        )
        self.save_path = self.save_dir.folder_path
        self.data_path = self.dataset_file_selected.file_path
        self.scaler = self.scaling_dropdown.getChoices()
        check_if_dir_exists(self.save_path)
        check_file_exist_and_readable(self.data_path)


# _ = FitDimReductionPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
