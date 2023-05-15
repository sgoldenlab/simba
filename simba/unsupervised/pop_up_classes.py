__author__ = "Simon Nilsson"

""" Tkinter pop-up classes for unsupervised ML"""

import glob
from tkinter import *
import numpy as np

from simba.ui.tkinter_functions import (FolderSelect,
                                        DropDownMenu,
                                        FileSelect,
                                        Entry_Box)

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin

from simba.utils.enums import Formats, Options
from simba.unsupervised.enums import UMLOptions, Unsupervised, Clustering
from simba.utils.checks import check_file_exist_and_readable, check_if_filepath_list_is_empty, check_if_dir_exists, check_int


from simba.unsupervised.grid_search_visualizers import GridSearchVisualizer
from simba.unsupervised.data_extractor import DataExtractor
from simba.utils.errors import NoSpecifiedOutputError
from simba.unsupervised.umap_embedder import UmapEmbedder
from simba.unsupervised.tsne import TSNEGridSearch
from simba.unsupervised.hdbscan_clusterer import HDBSCANClusterer
from simba.unsupervised.cluster_visualizer import ClusterVisualizer
from simba.unsupervised.cluster_statistics import (ClusterFrequentistCalculator,
                                                   ClusterXAICalculator,
                                                   EmbeddingCorrelationCalculator)
from simba.unsupervised.dbcv_calculator import DBCVCalculator

class GridSearchVisualizerPopUp(PopUpMixin):
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path, title='GRID SEARCH VISUALIZER')

        data_frm = LabelFrame(self.main_frm, text='DATA', fg='black', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.data_dir_select = FolderSelect(data_frm, "DATA DIRECTORY:", lblwidth=25)
        self.save_dir_select = FolderSelect(data_frm, "OUTPUT DIRECTORY: ", lblwidth=25)
        data_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_select.grid(row=0, column=0, sticky=NW)
        self.save_dir_select.grid(row=1, column=0, sticky=NW)

        self.visualization_options = UMLOptions.CATEGORICAL_OPTIONS.value + UMLOptions.CONTINUOUS_OPTIONS.value
        settings_frm = LabelFrame(self.main_frm,text='SETTINGS',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        self.scatter_size_dropdown = DropDownMenu(settings_frm, 'SCATTER SIZE:', UMLOptions.SCATTER_SIZE.value, '25')
        self.scatter_size_dropdown.setChoices(5)
        self.categorical_palette_dropdown = DropDownMenu(settings_frm, 'CATEGORICAL PALETTE:', Options.PALETTE_OPTIONS_CATEGORICAL.value, '25')
        self.categorical_palette_dropdown.setChoices('Set1')
        self.continuous_palette_dropdown = DropDownMenu(settings_frm, 'CONTINUOUS PALETTE:', Options.PALETTE_OPTIONS.value, '25')
        self.continuous_palette_dropdown.setChoices('magma')

        settings_frm.grid(row=1, column=0, sticky=NW)
        self.scatter_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.categorical_palette_dropdown.grid(row=1, column=0, sticky=NW)
        self.continuous_palette_dropdown.grid(row=2, column=0, sticky=NW)

        self.define_plots_frm = LabelFrame(self.main_frm, text='DEFINE PLOTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        self.plot_cnt_dropdown = DropDownMenu(self.define_plots_frm, '# PLOTS:', UMLOptions.GRAPH_CNT.value, '25', com= lambda x:self.show_plot_table())
        self.plot_cnt_dropdown.setChoices(UMLOptions.GRAPH_CNT.value[0])
        self.show_plot_table()
        self.define_plots_frm.grid(row=2, column=0, sticky=NW)
        self.plot_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(title='RUN', run_function=self.run)
        self.main_frm.mainloop()

    def show_plot_table(self):
        if hasattr(self, 'plot_table'):
            self.plot_table.destroy()

        self.plot_data = {}
        self.plot_table = LabelFrame(self.define_plots_frm, text='PLOTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        self.scatter_name_header = Label(self.plot_table, text='PLOT NAME').grid(row=0, column=0)
        self.field_name_header = Label(self.plot_table, text='COLOR VARIABLE').grid(row=0, column=1)
        for idx in range(int(self.plot_cnt_dropdown.getChoices())):
            row_name = idx
            self.plot_data[row_name] = {}
            self.plot_data[row_name]['label'] = Label(self.plot_table, text=f'Scatter {str(idx+1)}:')
            self.plot_data[row_name]['variable'] = DropDownMenu(self.plot_table, ' ', self.visualization_options, '10', com=None)
            self.plot_data[row_name]['variable'].setChoices(self.visualization_options[0])
            self.plot_data[idx]['label'].grid(row=idx+1, column=0, sticky=NW)
            self.plot_data[idx]['variable'].grid(row=idx+1, column=1, sticky=NW)
        self.plot_table.grid(row=1, column=0, sticky=NW)

    def run(self):
        if len(self.plot_data.keys()) < 1:
            raise NoSpecifiedOutputError(msg='Specify at least one plot')
        settings = {}
        settings['SCATTER_SIZE'] = int(self.scatter_size_dropdown.getChoices())
        settings['CATEGORICAL_PALETTE'] = self.categorical_palette_dropdown.getChoices()
        settings['CONTINUOUS_PALETTE'] = self.continuous_palette_dropdown.getChoices()

        continuous_vars, categorical_vars = [], []
        for k, v in self.plot_data.items():
            if v['variable'].getChoices() in UMLOptions.CONTINUOUS_OPTIONS.value:
                continuous_vars.append(v['variable'].getChoices())
            else:
                categorical_vars.append(v['variable'].getChoices())
        grid_search_visualizer = GridSearchVisualizer(model_dir=self.data_dir_select.folder_path,
                                                      save_dir=self.save_dir_select.folder_path,
                                                      settings=settings)
        grid_search_visualizer.continuous_visualizer(continuous_vars=continuous_vars)
        grid_search_visualizer.categorical_visualizer(categoricals=categorical_vars)

# _ = GridSearchVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')


class DataExtractorPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='DATA EXTRACTOR')
        ConfigReader.__init__(self, config_path=config_path)
        data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.data_dir_select = FolderSelect(data_frm, "DATA DIRECTORY:", lblwidth=25)

        data_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_select.grid(row=0, column=0, sticky=NW)

        data_type_frm = LabelFrame(self.main_frm, text='DATA TYPE', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.data_type_dropdown = DropDownMenu(data_type_frm, 'DATA TYPE:', UMLOptions.DATA_TYPES.value, '25')
        self.data_type_dropdown.setChoices(UMLOptions.DATA_TYPES.value[0])
        data_type_frm.grid(row=1, column=0, sticky=NW)
        self.data_type_dropdown.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run, title='RUN')
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(self.data_dir_select.folder_path)
        DataExtractor(config_path=self.config_path,
                      data_path=self.data_dir_select.folder_path,
                      data_type=self.data_type_dropdown.getChoices())

#_ = DataExtractorPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')


class FitDimReductionPopUp(PopUpMixin, ConfigReader):

    def __init__(self):
        super().__init__(title='FIT DIMENSIONALITY REDUCTION MODELS')
        self.variance_options = [str(x) + '%' for x in UMLOptions.VARIANCE_OPTIONS.value]

        self.dataset_frm = LabelFrame(self.main_frm, text='DATASET', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dataset_file_selected = FileSelect(self.dataset_frm, "DATASET (PICKLE):", lblwidth=25)
        self.dataset_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)

        self.save_frm = LabelFrame(self.main_frm, text='SAVE', font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.save_dir = FolderSelect(self.save_frm, "SAVE DIRECTORY:", lblwidth=25)
        self.save_frm.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)

        settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.scaling_dropdown = DropDownMenu(settings_frm, 'SCALING:', Options.SCALER_NAMES.value, '25')
        self.scaling_dropdown.setChoices(Options.SCALER_NAMES.value[0])
        self.var_threshold_dropdown = DropDownMenu(settings_frm, 'VARIANCE THRESHOLD:', self.variance_options, '25')
        self.var_threshold_dropdown.setChoices(self.variance_options[0])
        self.algo_dropdown = DropDownMenu(settings_frm, 'ALGORITHM:', UMLOptions.DR_ALGO_OPTIONS.value, '25', com=lambda x: self.show_dr_hyperparameters())
        self.algo_dropdown.setChoices(UMLOptions.DR_ALGO_OPTIONS.value[0])
        self.show_dr_hyperparameters()

        settings_frm.grid(row=2, column=0, sticky=NW)
        self.scaling_dropdown.grid(row=0, column=0, sticky=NW)
        self.var_threshold_dropdown.grid(row=1, column=0, sticky=NW)
        self.algo_dropdown.grid(row=2, column=0, sticky=NW)
        self.main_frm.mainloop()

    def show_dr_hyperparameters(self):
        if hasattr(self, 'hyperparameters_frm'):
            self.hyperparameters_frm.destroy()
            self.value_frm.destroy()
            self.run_frm.destroy()

        self.hyperparameters_frm = LabelFrame(self.main_frm, text='GRID SEARCH HYPER-PARAMETERS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.value_frm = LabelFrame(self.main_frm, fg='black')
        self.dr_value_entry_box = Entry_Box(self.value_frm, 'VALUE: ', '12')
        self.value_frm.grid(row=3, column=0, sticky=NW)
        self.dr_value_entry_box.grid(row=0, column=1, sticky=NW)
        self.hyperparameters_frm.grid(row=4, column=0, sticky=NW)

        if self.algo_dropdown.getChoices() == Unsupervised.UMAP.value:
            Label(self.hyperparameters_frm, text=Unsupervised.N_NEIGHBORS.value).grid(row=1, column=0)
            Label(self.hyperparameters_frm, text=Unsupervised.MIN_DISTANCE.value).grid(row=1, column=1)
            Label(self.hyperparameters_frm, text=Unsupervised.SPREAD.value).grid(row=1, column=2)

            self.n_neighbors_estimators_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_distance_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.spread_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)

            neighbours_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox_from_entrybox(list_box=self.n_neighbors_estimators_listbox, entry_box=self.dr_value_entry_box))
            min_distance_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox_from_entrybox(list_box=self.min_distance_listbox, entry_box=self.dr_value_entry_box))
            spread_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox_from_entrybox(list_box=self.spread_listbox, entry_box=self.dr_value_entry_box))

            neighbours_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.n_neighbors_estimators_listbox))
            min_distance_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.min_distance_listbox))
            spread_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.spread_listbox))

            self.add_values_to_several_listboxes(list_boxes=[self.n_neighbors_estimators_listbox, self.min_distance_listbox, self.spread_listbox], values= [15, 0.1, 1])

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
            Label(self.hyperparameters_frm, text='PERPLEXITY').grid(row=1, column=0)
            self.perplexity_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            perplexity_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox_from_entrybox(list_box=self.perplexity_listbox, entry_box=self.dr_value_entry_box))
            perplexity_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.perplexity_listbox))
            perplexity_add_btn.grid(row=2, column=0)
            perplexity_remove_btn.grid(row=3, column=0)
            self.perplexity_listbox.grid(row=4, column=0, sticky=NW)
            self.create_run_frm(run_function=self.__run_tsne_gridsearch)

    def __run_tsne_gridsearch(self):
        self.__get_settings()
        perplexities = [int(x) for x in self.perplexity_listbox.get(0, END)]
        if len(perplexities) == 0:
            raise NoSpecifiedOutputError('Provide value(s) for perplexity')
        hyperparameters = {'perplexity': perplexities,
                           'scaler': self.scaling_dropdown.getChoices(),
                           'variance': self.variance_selected}
        tsne_searcher = TSNEGridSearch(data_path=self.data_path,
                                       save_dir=self.save_path)
        #tsne_searcher.fit(hyperparameters=hyperparameters)

    def __run_umap_gridsearch(self):
        self.__get_settings()
        n_neighbours = [float(x) for x in self.n_neighbors_estimators_listbox.get(0, END)]
        min_distances = [float(x) for x in self.min_distance_listbox.get(0, END)]
        spreads = [float(x) for x in self.spread_listbox.get(0, END)]
        if len(min_distances) == 0 or len(n_neighbours) == 0 or len(spreads) == 0:
            raise NoSpecifiedOutputError('Provide at least one hyperparameter value for neighbors, min distances, and spread')
        hyper_parameters = {'n_neighbors': n_neighbours,
                           'min_distance': min_distances,
                           'spread': spreads,
                           'scaler': self.scaling_dropdown.getChoices(),
                           'variance': self.variance_selected}
        umap_searcher = UmapEmbedder()
        umap_searcher.fit(data_path=self.data_path,
                          save_dir=self.save_path,
                          hyper_parameters=hyper_parameters)

    def __get_settings(self):
        self.variance_selected = int(self.var_threshold_dropdown.getChoices()[:-1]) / 100
        self.save_path = self.save_dir.folder_path
        self.data_path = self.dataset_file_selected.file_path
        self.scaler = self.scaling_dropdown.getChoices()
        check_if_dir_exists(self.save_path)
        check_file_exist_and_readable(self.data_path)


# _ = FitDimReductionPopUp()

class TransformDimReductionPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='DIMENSIONALITY REDUCTION: TRANSFORM')
        self.dim_reduction_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.model_select = FileSelect(self.dim_reduction_frm, 'MODEL (PICKLE):', lblwidth=25)
        self.dataset_select = FileSelect(self.dim_reduction_frm, 'DATASET (PICKLE):', lblwidth=25)
        self.save_dir = FolderSelect(self.dim_reduction_frm, "SAVE DIRECTORY: ", lblwidth=25)
        self.features_dropdown = DropDownMenu(self.dim_reduction_frm, 'INCLUDE FEATURES:', UMLOptions.DATA_FORMATS.value, '25')
        self.features_dropdown.setChoices(UMLOptions.DATA_FORMATS.value[0])
        self.save_format_dropdown = DropDownMenu(self.dim_reduction_frm, 'SAVE FORMATS:', UMLOptions.SAVE_FORMATS.value, '25')
        self.save_format_dropdown.setChoices(UMLOptions.SAVE_FORMATS.value[0])

        self.dim_reduction_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)
        self.dataset_select.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.features_dropdown.grid(row=3, column=0, sticky=NW)
        self.save_format_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.save_dir.folder_path)
        check_file_exist_and_readable(file_path=self.model_select.file_path)
        check_file_exist_and_readable(file_path=self.dataset_select.file_path)

        umap_searcher = UmapEmbedder()
        umap_searcher.transform(data_path=self.dataset_select.file_path,
                                model=self.model_select.file_path,
                                save_dir=self.save_dir.folder_path,
                                settings=None)

#_ = TransformDimReductionPopUp()

class FitClusterModelsPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CLUSTERING FIT: GRID SEARCH')
        self.dataset_frm = LabelFrame(self.main_frm, text='DATASET', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.data_dir_selected = FolderSelect(self.dataset_frm, "DATA DIRECTORY (PICKLES): ", lblwidth=25)
        self.save_frm = LabelFrame(self.main_frm, text='SAVE', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.save_dir = FolderSelect(self.save_frm, "SAVE DIRECTORY: ", lblwidth=25)
        self.algo_frm = LabelFrame(self.main_frm, text='ALGORITHM', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.algo_dropdown = DropDownMenu(self.algo_frm, 'ALGORITHM:', UMLOptions.CLUSTERING_ALGO_OPTIONS.value, '25', com=lambda x: self.show_hyperparameters())
        self.algo_dropdown.setChoices(UMLOptions.CLUSTERING_ALGO_OPTIONS.value[0])
        self.value_frm = LabelFrame(self.main_frm, fg='black')
        self.value_entry_box = Entry_Box(self.value_frm, 'VALUE:', '25')

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
        if hasattr(self, 'hyperparameters_frm'):
            self.hyperparameters_frm.destroy()
            self.value_frm.destroy()
            self.run_frm.destroy()
#
        if self.algo_dropdown.getChoices() == Unsupervised.HDBSCAN.value:
            self.hyperparameters_frm = LabelFrame(self.main_frm, text='GRID SEARCH CLUSTER HYPER-PARAMETERS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')

            Label(self.hyperparameters_frm, text=Clustering.ALPHA.value).grid(row=1, column=0)
            Label(self.hyperparameters_frm, text=Clustering.MIN_CLUSTER_SIZE.value).grid(row=1, column=1)
            Label(self.hyperparameters_frm, text=Clustering.MIN_SAMPLES.value).grid(row=1, column=2)
            Label(self.hyperparameters_frm, text=Clustering.EPSILON.value).grid(row=1, column=3)

            self.alpha_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_cluster_size_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_samples_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.epsilon_listbox = Listbox(self.hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)

            alpha_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox_from_entrybox(list_box=self.alpha_listbox, entry_box=self.value_entry_box))
            min_cluster_size_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox_from_entrybox(list_box=self.min_cluster_size_listbox, entry_box=self.value_entry_box))
            min_samples_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox_from_entrybox(list_box=self.min_samples_listbox, entry_box=self.value_entry_box))
            epsilon_add_btn = Button(self.hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox_from_entrybox(list_box=self.epsilon_listbox, entry_box=self.value_entry_box))

            alpha_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.alpha_listbox))
            min_cluster_size_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.min_cluster_size_listbox))
            min_samples_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.min_samples_listbox))
            epsilon_remove_btn = Button(self.hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.epsilon_listbox))

            self.add_values_to_several_listboxes(list_boxes=[self.alpha_listbox, self.min_cluster_size_listbox, self.min_samples_listbox, self.epsilon_listbox], values= [1, 15, 1, 1])

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
        hyper_parameters = {'alpha': alphas,
                            'min_cluster_size': min_cluster_sizes,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': epsilons}
        clusterer = HDBSCANClusterer()
        clusterer.fit(data_path=self.data_directory, save_dir=self.save_directory, hyper_parameters=hyper_parameters)

#_ = FitClusterModelsPopUp()

class TransformClustererPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title='CLUSTERING: TRANSFORM')
        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.model_select = FileSelect(self.settings_frm, 'CLUSTER MODEL (PICKLE):', lblwidth=25)
        self.data_select = FileSelect(self.settings_frm, 'DATASET (PICKLE):',  lblwidth=25)
        self.save_dir = FolderSelect(self.settings_frm, "SAVE DIRECTORY:",  lblwidth=25)
        self.features_dropdown = DropDownMenu(self.settings_frm, 'INCLUDE FEATURES:', UMLOptions.DATA_FORMATS.value, '25')
        self.features_dropdown.setChoices(UMLOptions.DATA_FORMATS.value[0])
        self.save_format_dropdown = DropDownMenu(self.settings_frm, 'SAVE FORMATS:', UMLOptions.SAVE_FORMATS.value, '25')
        self.save_format_dropdown.setChoices(UMLOptions.SAVE_FORMATS.value[0])
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)
        self.data_select.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.features_dropdown.grid(row=3, column=0, sticky=NW)
        self.save_format_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(self.model_select.file_path)
        check_file_exist_and_readable(self.data_select.file_path)
        check_if_dir_exists(self.save_dir.folder_path)
        settings = {Unsupervised.DATA.value: self.features_dropdown.getChoices(),
                    Unsupervised.FORMAT.value: self.save_format_dropdown.getChoices()}

        clusterer = HDBSCANClusterer()
        clusterer.transform(data_path=self.data_select.file_path,
                            model=self.model_select.file_path,
                            save_dir=self.save_dir.folder_path,
                            settings=settings)


#_ = TransformClustererPopUp()

class ClusterVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='CLUSTER VIDEO VISUALIZATIONS')
        ConfigReader.__init__(self, config_path=config_path)
        self.include_pose_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.videos_dir_select = FolderSelect(self.data_frm, "VIDEOS DIRECTORY:", lblwidth=25)
        self.dataset_file_selected = FileSelect(self.data_frm, "DATASET (PICKLE): ", lblwidth=25)
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.videos_dir_select.grid(row=0, column=0, sticky=NW)

        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.include_pose = Checkbutton(self.settings_frm, text='INCLUDE POSE-ESTIMATION', variable=self.include_pose_var, command=lambda: self.enable_entrybox_from_checkbox(check_box_var=self.include_pose_var,entry_boxes=[self.circle_size_entry]))
        self.circle_size_entry = Entry_Box(self.settings_frm, 'CIRCLE SIZE: ', '25', validation='numeric')
        self.circle_size_entry.entry_set(val=5)
        self.circle_size_entry.set_state(setstatus='disable')

        self.speed_dropdown = DropDownMenu(self.settings_frm, 'VIDEO SPEED:', UMLOptions.SPEED_OPTIONS.value, '25')
        self.speed_dropdown.setChoices(1.0)
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.include_pose.grid(row=0, column=0, sticky=NW)
        self.circle_size_entry.grid(row=1, column=0, sticky=NW)
        self.speed_dropdown.grid(row=2, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.dataset_file_selected.file_path)
        check_if_dir_exists(in_dir=self.videos_dir_select.folder_path)
        speed = float(self.speed_dropdown.getChoices())
        if self.include_pose_var.get():
            check_int(name='CIRCLE SIZE', value=self.circle_size_entry.entry_get)
            circle_size = self.circle_size_entry.entry_get
        else:
            circle_size = np.inf
        settings = {'videos_speed': speed,
                    'pose': {'include': self.include_pose_var.get(),
                             'circle_size': circle_size}}
        cluster_visualizer = ClusterVisualizer(config_path=self.config_path,
                                               settings=settings,
                                               video_dir=self.videos_dir_select.folder_path,
                                               data_path=self.dataset_file_selected.file_path)
        cluster_visualizer.run()

#_ = ClusterVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')
#
class ClusterFrequentistStatisticsPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, title='CLUSTER FREQUENTIST STATISTICS')
        ConfigReader.__init__(self, config_path=config_path)
        self.descriptive_stats_var = BooleanVar(value=True)
        self.oneway_anova_var = BooleanVar(value=True)
        self.tukey_var = BooleanVar(value=True)
        self.use_scaled_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.model_select = FileSelect(self.data_frm, 'CLUSTERER PATH:', lblwidth=25)
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)

        self.stats_frm = LabelFrame(self.main_frm, text='STATISTICS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.descriptive_stats_cb = Checkbutton(self.stats_frm, text='CLUSTER DESCRIPTIVE STATISTICS', variable=self.descriptive_stats_var)
        self.oneway_anova_cb = Checkbutton(self.stats_frm, text='CLUSTER FEATURE ONE-WAY ANOVA', variable=self.oneway_anova_var)
        self.feature_tukey_posthoc_cb = Checkbutton(self.stats_frm, text='CLUSTER FEATURE POST-HOC (TUKEY)', variable=self.tukey_var)
        self.use_scaled_cb = Checkbutton(self.stats_frm, text='USE SCALED FEATURE VALUES', variable=self.use_scaled_var)

        self.stats_frm.grid(row=1, column=0, sticky=NW)
        self.descriptive_stats_cb.grid(row=0, column=0, sticky=NW)
        self.oneway_anova_cb.grid(row=1, column=0, sticky=NW)
        self.feature_tukey_posthoc_cb.grid(row=2, column=0, sticky=NW)
        self.use_scaled_cb.grid(row=3, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()


    def run(self):
        check_file_exist_and_readable(self.model_select.file_path)
        settings = {'scaled': self.use_scaled_var.get(),
                    'anova': self.oneway_anova_var.get(),
                    'tukey_posthoc': self.tukey_var.get(),
                    'descriptive_statistics': self.descriptive_stats_var.get()}

        calculator = ClusterFrequentistCalculator(config_path=self.config_path,
                                                  data_path=self.model_select.file_path,
                                                  settings=settings)
        calculator.run()

#_ = ClusterFrequentistStatisticsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')

class ClusterXAIPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        super().__init__(title='CLUSTER XAI STATISTICS')
        ConfigReader.__init__(self, config_path=config_path)
        self.gini_importance_var = BooleanVar(value=True)
        self.permutation_importance_var = BooleanVar(value=True)
        self.shap_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.model_select = FileSelect(self.data_frm, 'MODEL PATH:', lblwidth=25)
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)

        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.gini_importance_cb = Checkbutton(self.settings_frm, text='CLUSTER RF GINI IMPORTANCE', variable=self.gini_importance_var)
        self.permutation_cb = Checkbutton(self.settings_frm, text='CLUSTER RF PERMUTATION IMPORTANCE', variable=self.permutation_importance_var)
        self.shap_method_dropdown = DropDownMenu(self.settings_frm, 'SHAP METHOD:', UMLOptions.SHAP_CLUSTER_METHODS.value, '25')
        self.shap_method_dropdown.setChoices(UMLOptions.SHAP_CLUSTER_METHODS.value[0])
        self.shap_sample_dropdown = DropDownMenu(self.settings_frm, 'SHAP SAMPLES:', UMLOptions.SHAP_SAMPLE_OPTIONS.value, '25')
        self.shap_sample_dropdown.setChoices(100)
        self.shap_method_dropdown.disable()
        self.shap_sample_dropdown.disable()
        self.shap_cb = Checkbutton(self.settings_frm, text='CLUSTER RF SHAP VALUES', variable=self.shap_var, command= lambda:self.enable_dropdown_from_checkbox(check_box_var=self.shap_var, dropdown_menus=[self.shap_method_dropdown, self.shap_sample_dropdown]))

        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.gini_importance_cb.grid(row=0, column=0, sticky=NW)
        self.permutation_cb.grid(row=1, column=0, sticky=NW)
        self.shap_cb.grid(row=2, column=0, sticky=NW)
        self.shap_method_dropdown.grid(row=3, column=0, sticky=NW)
        self.shap_sample_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.model_select.file_path)
        settings = {'gini_importance': self.gini_importance_var.get(), 'permutation_importance': self.permutation_importance_var.get(),
                    'shap': {'method': self.shap_method_dropdown.getChoices(), 'run': self.shap_var.get(), 'sample': self.shap_sample_dropdown.getChoices()}}
        xai_calculator = ClusterXAICalculator(data_path=self.model_select.file_path,
                                              settings=settings,
                                              config_path=self.config_path)
        xai_calculator.run()


#_ = ClusterXAIPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')

class EmbedderCorrelationsPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='EMBEDDING CORRELATIONS')
        ConfigReader.__init__(self, config_path=config_path)
        self.spearman_var = BooleanVar(value=True)
        self.pearsons_var = BooleanVar(value=True)
        self.kendall_var = BooleanVar(value=True)
        self.plots_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.data_file_selected = FileSelect(self.data_frm, "DATASET (PICKLE):")
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.data_file_selected.grid(row=0, column=0, sticky=NW)

        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.spearman_cb = Checkbutton(self.settings_frm, text='SPEARMAN', variable=self.spearman_var)
        self.pearsons_cb = Checkbutton(self.settings_frm, text='PEARSONS', variable=self.pearsons_var)
        self.kendall_cb = Checkbutton(self.settings_frm, text='KENDALL', variable=self.kendall_var)
        self.plot_correlation_dropdown = DropDownMenu(self.settings_frm, 'PLOT CORRELATION:', UMLOptions.CORRELATION_OPTIONS.value, '25')
        self.plot_correlation_clr_dropdown = DropDownMenu(self.settings_frm, 'PLOT CORRELATION:', Options.PALETTE_OPTIONS.value, '25')
        self.plot_correlation_dropdown.setChoices(UMLOptions.CORRELATION_OPTIONS.value[0])
        self.plot_correlation_dropdown.disable()
        self.plot_correlation_clr_dropdown.setChoices(Options.PALETTE_OPTIONS.value[0])
        self.plot_correlation_clr_dropdown.disable()
        self.plots_cb = Checkbutton(self.settings_frm, text='PLOTS', variable=self.plots_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.plots_var, dropdown_menus=[self.plot_correlation_dropdown, self.plot_correlation_clr_dropdown]))

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
        settings = {'correlations': [], 'plots': {'create': False, 'correlations': None, 'palette': None}}
        if self.spearman_var.get(): settings['correlations'].append('spearman')
        if self.pearsons_var.get(): settings['correlations'].append('pearson')
        if self.kendall_var.get(): settings['correlations'].append('kendall')

        calculator = EmbeddingCorrelationCalculator(config_path=self.config_path,
                                                   data_path=self.data_file_selected.file_path,
                                                   settings=settings)
        calculator.run()

#_ = EmbedderCorrelationsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')


class PrintEmBeddingInfoPopUp(PopUpMixin, ConfigReader, UnsupervisedMixin):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, title='PRINT EMBEDDING MODEL INFO')
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dataset_file_selected = FileSelect(self.data_frm, "DATASET (PICKLE): ", lblwidth=25)
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.dataset_file_selected.file_path)
        data = self.read_pickle(data_path=self.dataset_file_selected.file_path)
        parameters = {**data[Unsupervised.DR_MODEL.value][Unsupervised.PARAMETERS.value]}
        print(parameters)

#_ = PrintEmBeddingInfoPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')

class DBCVPopUp(PopUpMixin, ConfigReader, UnsupervisedMixin):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='DENSITY BASED CLUSTER VALIDATION')
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.folder_selected = FolderSelect(self.data_frm, "DATASETS (DIRECTORY WITH PICKLES):", lblwidth=35)
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.folder_selected.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.folder_selected.folder_path)
        data_paths = glob.glob(self.folder_selected.folder_path + '/*.pickle')
        check_if_filepath_list_is_empty(filepaths=data_paths, error_msg=f'No pickle files in {self.folder_selected.folder_path}')
        dbcv_calculator = DBCVCalculator(data_path=self.folder_selected.folder_path, config_path=self.config_path)
        dbcv_calculator.run()


#_ = DBCVPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')