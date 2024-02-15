__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import threading
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        FolderSelect)
from simba.unsupervised.enums import Clustering, UMLOptions, Unsupervised
from simba.unsupervised.hdbscan_clusterer import HDBSCANClusterer
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Formats, Options


class FitClusterModelsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CLUSTERING FIT: GRID SEARCH")
        ConfigReader.__init__(self, config_path=config_path)

        self.dataset_frm = LabelFrame(
            self.main_frm,
            text="DATASET",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.data_dir_selected = FolderSelect(
            self.dataset_frm,
            "DATA DIRECTORY (PICKLES): ",
            lblwidth=25,
            initialdir=self.project_path,
        )
        self.save_frm = LabelFrame(
            self.main_frm,
            text="SAVE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.save_dir = FolderSelect(
            self.save_frm, "SAVE DIRECTORY: ", lblwidth=25, initialdir=self.project_path
        )
        self.algo_frm = LabelFrame(
            self.main_frm,
            text="ALGORITHM",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.algo_dropdown = DropDownMenu(
            self.algo_frm,
            "ALGORITHM:",
            UMLOptions.CLUSTERING_ALGO_OPTIONS.value,
            "25",
            com=lambda x: self.show_hyperparameters(),
        )
        self.algo_dropdown.setChoices(UMLOptions.CLUSTERING_ALGO_OPTIONS.value[0])
        self.value_frm = LabelFrame(self.main_frm, fg="black")
        self.value_entry_box = Entry_Box(self.value_frm, "VALUE:", "25")

        self.dataset_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_selected.grid(row=0, column=0, sticky=NW)
        self.save_frm.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)
        self.algo_frm.grid(row=2, column=0, sticky=NW)
        self.algo_dropdown.grid(row=0, column=0, sticky=NW)
        self.value_frm.grid(row=3, column=0, sticky=NW)
        self.value_entry_box.grid(row=0, column=0, sticky=NW)
        self.show_hyperparameters()
        self.main_frm.mainloop()

    def show_hyperparameters(self):
        if hasattr(self, "hyperparameters_frm"):
            self.hyperparameters_frm.destroy()
            self.value_frm.destroy()
            self.run_frm.destroy()
        #
        if self.algo_dropdown.getChoices() == Unsupervised.HDBSCAN.value:
            self.hyperparameters_frm = LabelFrame(
                self.main_frm,
                text="GRID SEARCH CLUSTER HYPER-PARAMETERS",
                font=Formats.LABELFRAME_HEADER_FORMAT.value,
                fg="black",
            )

            Label(self.hyperparameters_frm, text=Clustering.ALPHA.value).grid(
                row=1, column=0
            )
            Label(
                self.hyperparameters_frm, text=Clustering.MIN_CLUSTER_SIZE.value
            ).grid(row=1, column=1)
            Label(self.hyperparameters_frm, text=Clustering.MIN_SAMPLES.value).grid(
                row=1, column=2
            )
            Label(self.hyperparameters_frm, text=Clustering.EPSILON.value).grid(
                row=1, column=3
            )

            self.alpha_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )
            self.min_cluster_size_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )
            self.min_samples_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )
            self.epsilon_listbox = Listbox(
                self.hyperparameters_frm, bg="lightgrey", fg="black", height=5, width=15
            )

            alpha_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.alpha_listbox, entry_box=self.value_entry_box
                ),
            )
            min_cluster_size_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.min_cluster_size_listbox,
                    entry_box=self.value_entry_box,
                ),
            )
            min_samples_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.min_samples_listbox, entry_box=self.value_entry_box
                ),
            )
            epsilon_add_btn = Button(
                self.hyperparameters_frm,
                text="ADD",
                fg="blue",
                command=lambda: self.add_to_listbox_from_entrybox(
                    list_box=self.epsilon_listbox, entry_box=self.value_entry_box
                ),
            )

            alpha_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(list_box=self.alpha_listbox),
            )
            min_cluster_size_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(
                    list_box=self.min_cluster_size_listbox
                ),
            )
            min_samples_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(
                    list_box=self.min_samples_listbox
                ),
            )
            epsilon_remove_btn = Button(
                self.hyperparameters_frm,
                text="REMOVE",
                fg="red",
                command=lambda: self.remove_from_listbox(list_box=self.epsilon_listbox),
            )

            self.add_values_to_several_listboxes(
                list_boxes=[
                    self.alpha_listbox,
                    self.min_cluster_size_listbox,
                    self.min_samples_listbox,
                    self.epsilon_listbox,
                ],
                values=[1, 15, 1, 1],
            )

            self.hyperparameters_frm.grid(row=4, column=0, sticky=NW)
            alpha_add_btn.grid(row=2, column=0)
            min_cluster_size_add_btn.grid(row=2, column=1)
            min_samples_add_btn.grid(row=2, column=2)
            epsilon_add_btn.grid(row=2, column=3)

            alpha_remove_btn.grid(row=3, column=0)
            min_cluster_size_remove_btn.grid(row=3, column=1)
            min_samples_remove_btn.grid(row=3, column=2)
            epsilon_remove_btn.grid(row=3, column=3)

            self.alpha_listbox.grid(row=4, column=0, sticky=NW)
            self.min_cluster_size_listbox.grid(row=4, column=1, sticky=NW)
            self.min_samples_listbox.grid(row=4, column=2, sticky=NW)
            self.epsilon_listbox.grid(row=4, column=3, sticky=NW)

            self.create_run_frm(run_function=self.run_hdbscan_clustering)

    def __get_settings(self):
        self.data_directory = self.data_dir_selected.folder_path
        self.save_directory = self.save_dir.folder_path
        check_if_dir_exists(self.data_dir_selected.folder_path)
        check_if_dir_exists(self.save_dir.folder_path)

    def run_hdbscan_clustering(self):
        self.__get_settings()
        alphas = [float(x) for x in self.alpha_listbox.get(0, END)]
        min_cluster_sizes = [int(x) for x in self.min_cluster_size_listbox.get(0, END)]
        min_samples = [int(x) for x in self.min_samples_listbox.get(0, END)]
        epsilons = [float(x) for x in self.epsilon_listbox.get(0, END)]
        hyper_parameters = {
            "alpha": alphas,
            "min_cluster_size": min_cluster_sizes,
            "min_samples": min_samples,
            "cluster_selection_epsilon": epsilons,
        }
        clusterer = HDBSCANClusterer()
        threading.Thread(
            target=clusterer.fit(
                data_path=self.data_directory,
                save_dir=self.save_directory,
                hyper_parameters=hyper_parameters,
            )
        ).start()


# _ = FitClusterModelsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
