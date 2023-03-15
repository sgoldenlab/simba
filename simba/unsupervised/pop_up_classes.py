import os.path
from tkinter import *

import numpy as np

from simba.tkinter_functions import (hxtScrollbar,
                                     FolderSelect,
                                     DropDownMenu,
                                     FileSelect,
                                     Entry_Box)
from simba.enums import Formats, ReadConfig, Dtypes, Options
from simba.train_model_functions import get_all_clf_names
from simba.unsupervised.visualizers import (GridSearchClusterVisualizer,
                                            ClusterVisualizer)
from simba.unsupervised.data_extractors import DataExtractorMultipleModels
from simba.unsupervised.umap_embedder import (UMAPGridSearch, UMAPTransform)
from simba.unsupervised.tsne import TSNEGridSearch
from simba.unsupervised.hdbscan_clusterer import (HDBSCANClusterer,
                                                  HDBSCANTransform)
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_float,
                                          check_int,
                                          check_file_exist_and_readable)
from simba.unsupervised.cluster_statistics import (ClusterFrequentistCalculator,
                                                   ClusterXAICalculator,
                                                   EmbeddingCorrelationCalculator)


class GridSearchClusterVisualizerPopUp(object):
    def __init__(self,
                 config_path: str):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.graph_cnt_options = list(range(1, 11))
        self.scatter_sizes_options = list(range(10, 110, 10))
        self.graph_cnt_options.insert(0, 'None')
        self.config, self.config_path = read_config_file(config_path), config_path
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.field_types_options = ['START FRAME', 'VIDEO NAMES', 'CLASSIFIER', 'CLASSIFIER PROBABILITY']
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.main_frm.wm_title("VISUALIZATION OF CLUSTER GRID SEARCH")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)

        data_frm = LabelFrame(self.main_frm,text='DATA',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        self.clusterers_dir_select = FolderSelect(data_frm, "CLUSTERERS DIRECTORY:", lblwidth=25)
        self.save_dir_select = FolderSelect(data_frm, "IMAGE SAVE DIRECTORY: ", lblwidth=25)

        settings_frm = LabelFrame(self.main_frm,text='SETTINGS',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        self.scatter_size_dropdown = DropDownMenu(settings_frm, 'SCATTER SIZE:', self.scatter_sizes_options, '25')
        self.scatter_size_dropdown.setChoices(50)
        self.categorical_palette_dropdown = DropDownMenu(settings_frm, 'CATEGORICAL PALETTE:', Options.PALETTE_OPTIONS_CATEGORICAL.value, '25')
        self.categorical_palette_dropdown.setChoices('Set1')
        self.continuous_palette_dropdown = DropDownMenu(settings_frm, 'CONTINUOUS PALETTE:', Options.PALETTE_OPTIONS.value, '25')
        self.continuous_palette_dropdown.setChoices('magma')


        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        run_btn = Button(run_frm, text='RUN', fg='blue', command=lambda: self.run())

        plot_select_cnt_frm = LabelFrame(self.main_frm, text='SELECT PLOTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        self.plot_cnt_dropdown = DropDownMenu(plot_select_cnt_frm, '# PLOTS:', self.graph_cnt_options, '25', com= lambda x:self.show_plot_table())
        self.plot_cnt_dropdown.setChoices(self.graph_cnt_options[0])

        data_frm.grid(row=0, column=0, sticky=NW)
        self.clusterers_dir_select.grid(row=0, column=0, sticky=NW)
        self.save_dir_select.grid(row=1, column=0, sticky=NW)

        settings_frm.grid(row=1, column=0, sticky=NW)
        self.scatter_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.categorical_palette_dropdown.grid(row=1, column=0, sticky=NW)
        self.continuous_palette_dropdown.grid(row=2, column=0, sticky=NW)

        run_frm.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

        plot_select_cnt_frm.grid(row=3, column=0, sticky=NW)
        self.plot_cnt_dropdown.grid(row=0, column=0, sticky=NW)


        self.main_frm.mainloop()

    def show_plot_table(self):
        if hasattr(self, 'plot_table'):
            self.plot_table.destroy()

        if self.plot_cnt_dropdown.getChoices() != 'NONE':
            self.plot_rows = {}
            self.plot_table = LabelFrame(self.main_frm, text='SELECT PLOTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
            self.plot_table.grid(row=4, column=0, sticky=NW)
            self.scatter_name_header = Label(self.plot_table, text='PLOT NAME').grid(row=0, column=0)
            self.field_type_header = Label(self.plot_table, text='FIELD TYPE').grid(row=0, column=1)
            self.field_name_header = Label(self.plot_table, text='FIELD NAME').grid(row=0, column=2)
            for idx in range(int(self.plot_cnt_dropdown.getChoices())):
                row_name = idx
                self.plot_rows[row_name] = {}
                self.plot_rows[row_name]['label'] = Label(self.plot_table, text=f'Scatter {str(idx+1)}:')
                self.plot_rows[row_name]['field_type_dropdown'] = DropDownMenu(self.plot_table, ' ', self.field_types_options, '10', com=lambda k, x=idx: self.change_field_name_state(k, x))
                self.plot_rows[row_name]['field_type_dropdown'].setChoices(self.field_types_options[0])
                self.plot_rows[row_name]['field_name_dropdown'] = DropDownMenu(self.plot_table, ' ', self.clf_names, '10', com=None)
                self.plot_rows[row_name]['field_name_dropdown'].disable()
                self.plot_rows[idx]['label'].grid(row=idx+1, column=0, sticky=NW)
                self.plot_rows[idx]['field_type_dropdown'].grid(row=idx+1, column=1, sticky=NW)
                self.plot_rows[idx]['field_name_dropdown'].grid(row=idx+1, column=2, sticky=NW)

    def change_field_name_state(self, k, x):
        if (k == 'CLASSIFIER'):
            self.plot_rows[x]['field_name_dropdown'].enable()
            self.plot_rows[x]['field_name_dropdown'].setChoices(self.clf_names[0])
        else:
            self.plot_rows[x]['field_name_dropdown'].disable()
            self.plot_rows[x]['field_name_dropdown'].setChoices(None)

    def run(self):
        clusterers_path = self.clusterers_dir_select.folder_path
        save_dir = self.save_dir_select.folder_path

        settings = {}
        settings['SCATTER_SIZE'] = int(self.scatter_size_dropdown.getChoices())
        settings['CATEGORICAL_PALETTE'] = self.categorical_palette_dropdown.getChoices()
        settings['CONTINUOUS_PALETTE'] = self.continuous_palette_dropdown.getChoices()
        hue_dict = None
        if self.plot_cnt_dropdown.getChoices() != 'NONE':
            hue_dict = {}
            for cnt, (k, v) in enumerate(self.plot_rows.items()):
                hue_dict[cnt] = {}
                hue_dict[cnt]['FIELD_TYPE'] = v['field_type_dropdown'].getChoices()
                hue_dict[cnt]['FIELD_NAME'] = v['field_name_dropdown'].getChoices()
        settings['HUE'] = hue_dict
        if not os.path.isdir(clusterers_path):
            print('SIMBA ERROR: Save path is not a valid directory.')
            raise NotADirectoryError()

        print(settings['HUE'])
        grid_search_visualizer = GridSearchClusterVisualizer(clusterers_path=clusterers_path,
                                                             save_dir=save_dir,
                                                             settings=settings)
        grid_search_visualizer.create_datasets()
        grid_search_visualizer.create_imgs()


class BatchDataExtractorPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("VISUALIZATION OF GRID SEARCH")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.hyperparameter_log_var = BooleanVar(value=True)

        data_frm = LabelFrame(self.main_frm, text='DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        self.embedding_model_select = FolderSelect(data_frm, "EMBEDDING DIRECTORY (PICKLES):", lblwidth=25)
        self.cluster_model_select = FolderSelect(data_frm, "CLUSTER DIRECTORY (PICKLES):", lblwidth=25)
        self.save_dir_select = FolderSelect(data_frm, "SAVE DIRECTORY: ", lblwidth=25)


        settings_frm = LabelFrame(self.main_frm, text='SETTINGS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        self.features_dropdown = DropDownMenu(settings_frm, 'SCATTER SIZE:', Options.UNSUPERVISED_FEATURE_OPTIONS.value, '25')
        self.include_log_cb = Checkbutton(settings_frm, text='INCLUDE HYPERPARAMETER LOG', variable=self.hyperparameter_log_var)

        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        run_btn = self.create_btn = Button(run_frm, text='RUN', fg='blue', command=lambda: self.run())

        data_frm.grid(row=0, column=0, sticky=NW)
        self.embedding_model_select.grid(row=0, column=0, sticky=NW)
        self.cluster_model_select.grid(row=1, column=0, sticky=NW)
        self.save_dir_select.grid(row=2, column=0, sticky=NW)

        settings_frm.grid(row=1, column=0, sticky=NW)
        self.features_dropdown.grid(row=1, column=0, sticky=NW)
        self.include_log_cb.grid(row=2, column=0, sticky=NW)

        run_frm.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def run(self):
        embedding_dir = self.embedding_model_select.folder_path
        cluster_dir = self.cluster_model_select.folder_path
        save_dir = self.save_dir_select.folder_path
        if not os.path.isdir(cluster_dir):
            cluster_dir = None

        settings = {'include_features': False, 'scaled_features': False, 'parameter_log': False}
        if self.features_dropdown.getChoices() == Options.UNSUPERVISED_FEATURE_OPTIONS.value[0]:
            settings['include_features'] = True
        if self.features_dropdown.getChoices() == Options.UNSUPERVISED_FEATURE_OPTIONS.value[1]:
            settings['include_features'] = True
            settings['scaled_features'] = True
        if self.hyperparameter_log_var.get():
            settings['parameter_log'] = True

        _ = DataExtractorMultipleModels(embeddings_dir=embedding_dir,
                                        clusterer_dir=cluster_dir,
                                        save_dir=save_dir)

class FitDimReductionPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("VISUALIZATION OF GRID SEARCH")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)

        self.scaler_options = Options.SCALER_NAMES.value
        self.dim_reduction_algo_options = ['UMAP',
                                           'TSNE']
        self.feature_removal_options = list(range(10, 100, 10))
        self.feature_removal_options = [str(x) + '%' for x in self.feature_removal_options]
        self.feature_removal_options.insert(0, 'NONE')


        self.dataset_frm = LabelFrame(self.main_frm, text='DATASET', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dataset_file_selected = FileSelect(self.dataset_frm, "DATASET (PICKLE): ")
        self.save_frm = LabelFrame(self.main_frm, text='SAVE', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dr_save_dir = FolderSelect(self.save_frm, "SAVE DIRECTORY: ")
        settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.scaling_dropdown = DropDownMenu(settings_frm, 'SCALING:', self.scaler_options, '20')
        self.scaling_dropdown.setChoices(self.scaler_options[0])
        self.feature_removal_dropdown = DropDownMenu(settings_frm, 'VARIANCE THRESHOLD:', self.feature_removal_options, '20')
        self.feature_removal_dropdown.setChoices(self.feature_removal_options[0])
        self.choose_algo_dropdown = DropDownMenu(settings_frm, 'ALGORITHM:', self.dim_reduction_algo_options, '20', com=lambda x: self.show_dr_algo_hyperparameters())
        self.choose_algo_dropdown.setChoices(self.dim_reduction_algo_options[0])
        self.show_dr_algo_hyperparameters()


        self.dataset_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)
        self.save_frm.grid(row=1, column=0, sticky=NW)
        self.dr_save_dir.grid(row=0, column=0, sticky=NW)
        settings_frm.grid(row=2, column=0, sticky=NW)
        self.choose_algo_dropdown.grid(row=0, column=0, sticky=NW)
        self.feature_removal_dropdown.grid(row=1, column=0, sticky=NW)
        self.scaling_dropdown.grid(row=2, column=0, sticky=NW)

        self.main_frm.mainloop()

    def add_to_listbox(self,
                       list_box: Listbox,
                       entry_box: Entry_Box):
        value = entry_box.entry_get
        check_float(name='VALUE', value=value)
        list_box_content = [float(x) for x in list_box.get(0, END)]
        if float(value) not in list_box_content:
            list_box.insert(0, value)

    def remove_from_listbox(self,
                       list_box: Listbox):
        selection = list_box.curselection()
        if selection:
            list_box.delete(selection[0])

    def show_dr_algo_hyperparameters(self):
        if hasattr(self, 'dr_hyperparameters_frm'):
            self.dr_hyperparameters_frm.destroy()
            self.dr_value_frm.destroy()
            self.run_frm.destroy()

        self.dr_hyperparameters_frm = LabelFrame(self.main_frm, text='GRID SEARCH HYPER-PARAMETERS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dr_value_frm = LabelFrame(self.main_frm, fg='black')
        self.dr_value_entry_box = Entry_Box(self.dr_value_frm, 'VALUE: ', '12')
        self.dr_value_frm.grid(row=3, column=0, sticky=NW)
        self.dr_value_entry_box.grid(row=0, column=1, sticky=NW)
        self.dr_hyperparameters_frm.grid(row=4, column=0, sticky=NW)
        if self.choose_algo_dropdown.getChoices() == 'UMAP':
            n_neighbors_estimators_lbl = Label(self.dr_hyperparameters_frm, text='N NEIGHBOURS')
            min_distance_lbl = Label(self.dr_hyperparameters_frm, text='MIN DISTANCE')
            spread_lbl = Label(self.dr_hyperparameters_frm, text='SPREAD')
            add_min_distance_btn = Button(self.dr_hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox(list_box=self.min_distance_listb, entry_box=self.dr_value_entry_box))
            add_neighbours_btn = Button(self.dr_hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox(list_box=self.n_neighbors_estimators_listb, entry_box=self.dr_value_entry_box))
            add_spread_btn = Button(self.dr_hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox(list_box=self.spread_listb, entry_box=self.dr_value_entry_box))
            remove_min_distance_btn = Button(self.dr_hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.min_distance_listb))
            remove_neighbours_btn = Button(self.dr_hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.n_neighbors_estimators_listb))
            remove_spread_btn = Button(self.dr_hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.spread_listb))
            self.n_neighbors_estimators_listb = Listbox(self.dr_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_distance_listb = Listbox(self.dr_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.spread_listb = Listbox(self.dr_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            n_neighbors_estimators_lbl.grid(row=1, column=0)
            min_distance_lbl.grid(row=1, column=1)
            spread_lbl.grid(row=1, column=2)
            add_neighbours_btn.grid(row=2, column=0)
            add_min_distance_btn.grid(row=2, column=1)
            add_spread_btn.grid(row=2, column=2)
            remove_neighbours_btn.grid(row=3, column=0)
            remove_min_distance_btn.grid(row=3, column=1)
            remove_spread_btn.grid(row=3, column=2)
            self.n_neighbors_estimators_listb.grid(row=4, column=0, sticky=NW)
            self.min_distance_listb.grid(row=4, column=1, sticky=NW)
            self.spread_listb.grid(row=4, column=2, sticky=NW)
            self.run_frm = LabelFrame(self.main_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
            run_btn = Button(self.run_frm, text='RUN', fg='blue', command= lambda: self.run_gridsearch())
            self.run_frm.grid(row=5, column=0, sticky=NW)
            run_btn.grid(row=0, column=1, sticky=NW)

        if self.choose_algo_dropdown.getChoices() == 'TSNE':
            perplexity_lbl = Label(self.dr_hyperparameters_frm, text='PERPLEXITY')
            add_perplexity_btn = Button(self.dr_hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox(list_box=self.perplexity_listb, entry_box=self.dr_value_entry_box))
            remove_perplexity_btn = Button(self.dr_hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.perplexity_listb))
            self.perplexity_listb = Listbox(self.dr_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            perplexity_lbl.grid(row=1, column=0)
            add_perplexity_btn.grid(row=2, column=0)
            remove_perplexity_btn.grid(row=3, column=0)
            self.perplexity_listb.grid(row=4, column=0, sticky=NW)
            self.run_frm = LabelFrame(self.main_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
            run_btn = Button(self.run_frm, text='RUN', fg='blue', command= lambda: self.run_gridsearch())
            self.run_frm.grid(row=5, column=0, sticky=NW)
            run_btn.grid(row=0, column=1, sticky=NW)

    def run_gridsearch(self):
        variance = None
        if self.feature_removal_dropdown.getChoices() != 'NONE':
            variance = int(self.feature_removal_dropdown.getChoices()[:-1]) / 100
        save_path = self.dr_save_dir.folder_path
        data_path = self.dataset_file_selected.file_path

        if self.choose_algo_dropdown.getChoices() == 'UMAP':
            min_distance = [float(x) for x in self.min_distance_listb.get(0, END)]
            n_neighbours = [float(x) for x in self.n_neighbors_estimators_listb.get(0, END)]
            spread = [float(x) for x in self.spread_listb.get(0, END)]
            if len(min_distance) == 0 or len(n_neighbours) == 0 or len(spread) == 0:
                print('SIMBA ERROR: Provide values for neighbors, min distances, and spread')
                raise ValueError('SIMBA ERROR: Provide at least one hyperparameter value for neighbors, min distances, and spread')
            hyperparameters = {'n_neighbors': n_neighbours,
                                'min_distance': min_distance,
                                'spread': spread,
                                'scaler': self.scaling_dropdown.getChoices(),
                                'variance': variance}
            umap_searcher = UMAPGridSearch(data_path=data_path,
                                           save_dir=save_path)

            umap_searcher.fit(hyper_parameters=hyperparameters)

        if self.choose_algo_dropdown.getChoices() == 'TSNE':
            perplexity = [int(x) for x in self.perplexity_listb.get(0, END)]
            if len(perplexity) == 0:
                print('SIMBA ERROR: Provide value(s) for perplexity')
                raise ValueError('SIMBA ERROR: Provide value(s) for perplexity')

            hyperparameters = {'perplexity': perplexity,
                               'scaler': self.scaling_dropdown.getChoices(),
                               'variance': variance}
            tsne_searcher = TSNEGridSearch(data_path=data_path,
                                           save_dir=save_path)
            tsne_searcher.fit(hyperparameters=hyperparameters)


class TransformDimReductionPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("DIMENSIONALITY REDUCTION: TRANSFORM")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.features_options = ['EXCLUDE', 'INCLUDE: ORIGINAL', 'INCLUDE: SCALED']

        self.dim_reduction_frm = LabelFrame(self.main_frm, text='DIMENSIONALITY REDUCTION: TRANSFORM', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dim_reduction_model = FileSelect(self.dim_reduction_frm, 'MODEL (PICKLE):', lblwidth=25)
        self.dim_reduction_dataset = FileSelect(self.dim_reduction_frm, 'DATASET (PICKLE):', lblwidth=25)
        self.save_dir = FolderSelect(self.dim_reduction_frm, "SAVE DIRECTORY: ", lblwidth=25)
        self.features_dropdown = DropDownMenu(self.dim_reduction_frm, 'FEATURES:', self.features_options, '12')

        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        run_btn = self.create_btn = Button(run_frm, text='RUN', fg='blue', command=lambda: self.run())

        self.dim_reduction_frm.grid(row=0, column=0, sticky=NW)
        self.dim_reduction_model.grid(row=0, column=0, sticky=NW)
        self.dim_reduction_dataset.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)

        run_frm.grid(row=1, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def run(self):
        model_path = self.dim_reduction_model.file_path
        data_path = self.dim_reduction_dataset.file_path
        save_dir = self.save_dir.folder_path
        settings = {}
        settings['features'] = self.features_dropdown.getChoices()
        settings['save_format'] = 'csv'

        _ = UMAPTransform(model_path=model_path,
                          data_path=data_path,
                          save_dir=save_dir,
                          settings=settings)

class FitClusterModelsPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("GRID SEARCH CLUSTER MODELS")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)

        self.cluster_algo_options = ['HDBSCAN']

        self.clustering_frm = LabelFrame(self.main_frm, text='CLUSTERING', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.cluster_dataset_frm = LabelFrame(self.clustering_frm, text='DATASET', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dimensionality_reduction_data_selected = FolderSelect(self.cluster_dataset_frm, "EMBEDDING DIRECTORY: ")
        self.clustering_save_dir_frm = LabelFrame(self.clustering_frm, text='SAVE', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.clustering_save_dir_folder = FolderSelect(self.clustering_save_dir_frm, "SAVE DIRECTORY: ")
        self.choose_cluster_algo_frm = LabelFrame(self.clustering_frm, text='ALGORITHM', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.choose_cluster_algo_dropdown = DropDownMenu(self.choose_cluster_algo_frm, 'ALGORITHM:', self.cluster_algo_options, '12', com=lambda x: self.show_cluster_algo_hyperparameters())
        self.choose_cluster_algo_dropdown.setChoices(self.cluster_algo_options[0])
        self.show_cluster_algo_hyperparameters()

        self.clustering_frm.grid(row=0, column=0, sticky=NW)
        self.cluster_dataset_frm.grid(row=0, column=0, sticky=NW)
        self.dimensionality_reduction_data_selected.grid(row=0, column=0, sticky=NW)
        self.clustering_save_dir_frm.grid(row=1, column=0, sticky=NW)
        self.clustering_save_dir_folder.grid(row=0, column=0, sticky=NW)
        self.choose_cluster_algo_frm.grid(row=2, column=0, sticky=NW)
        self.choose_cluster_algo_dropdown.grid(row=0, column=0, sticky=NW)

    def show_cluster_algo_hyperparameters(self):
        if hasattr(self, 'cluster_hyperparameters_frm'):
            self.cluster_hyperparameters_frm.destroy()
            self.run_frm.destroy()

        if self.choose_cluster_algo_dropdown.getChoices() == 'HDBSCAN':
            self.cluster_hyperparameters_frm = LabelFrame(self.clustering_frm, text='GRID SEARCH CLUSTER HYPER-PARAMETERS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
            self.cluster_value_frm = LabelFrame(self.clustering_frm, fg='black')
            self.cluster_value_entry_box = Entry_Box(self.cluster_value_frm, 'VALUE: ', '12')

            alpha_lbl = Label(self.cluster_hyperparameters_frm, text='ALPHA')
            min_cluster_size_lbl = Label(self.cluster_hyperparameters_frm, text='MIN CLUSTER SIZE')
            min_samples_lbl = Label(self.cluster_hyperparameters_frm, text='MIN SAMPLES')
            cluster_selection_epsilon_lbl = Label(self.cluster_hyperparameters_frm, text='EPSILON')

            add_alpha_btn = Button(self.cluster_hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox(list_box=self.alpha_listb, entry_box=self.cluster_value_entry_box))
            add_min_cluster_size_btn = Button(self.cluster_hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox(list_box=self.min_cluster_size_listb, entry_box=self.cluster_value_entry_box))
            add_min_samples_btn = Button(self.cluster_hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox(list_box=self.min_samples_listb, entry_box=self.cluster_value_entry_box))
            add_epsilon_btn = Button(self.cluster_hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox(list_box=self.epsilon_listb, entry_box=self.cluster_value_entry_box))

            remove_alpha_btn = Button(self.cluster_hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.alpha_listb))
            remove_min_cluster_size_btn = Button(self.cluster_hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.min_cluster_size_listb))
            remove_min_samples_btn = Button(self.cluster_hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.min_samples_listb))
            remove_epsilon_btn = Button(self.cluster_hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.epsilon_listb))

            self.alpha_listb = Listbox(self.cluster_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_cluster_size_listb = Listbox(self.cluster_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_samples_listb = Listbox(self.cluster_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.epsilon_listb = Listbox(self.cluster_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)

            self.cluster_value_frm.grid(row=3, column=0, sticky=NW)
            self.cluster_value_entry_box.grid(row=0, column=1, sticky=NW)
            self.cluster_hyperparameters_frm.grid(row=4, column=0, sticky=NW)

            alpha_lbl.grid(row=1, column=0)
            min_cluster_size_lbl.grid(row=1, column=1)
            min_samples_lbl.grid(row=1, column=2)
            cluster_selection_epsilon_lbl.grid(row=1, column=3)

            add_alpha_btn.grid(row=2, column=0)
            add_min_cluster_size_btn.grid(row=2, column=1)
            add_min_samples_btn.grid(row=2, column=2)
            add_epsilon_btn.grid(row=2, column=3)

            remove_alpha_btn.grid(row=3, column=0)
            remove_min_cluster_size_btn.grid(row=3, column=1)
            remove_min_samples_btn.grid(row=3, column=2)
            remove_epsilon_btn.grid(row=3, column=3)

            self.alpha_listb.grid(row=4, column=0)
            self.min_cluster_size_listb.grid(row=4, column=1)
            self.min_samples_listb.grid(row=4, column=2)
            self.epsilon_listb.grid(row=4, column=3)

            self.run_frm = LabelFrame(self.cluster_hyperparameters_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
            run_btn = Button(self.run_frm, text='RUN', fg='blue', command=lambda: self.run_hdbscan_clustering())
            self.run_frm.grid(row=5, column=0, sticky=NW)
            run_btn.grid(row=0, column=1, sticky=NW)

    def add_to_listbox(self,
                       list_box: Listbox,
                       entry_box: Entry_Box):
        value = entry_box.entry_get
        check_float(name='VALUE', value=value)
        list_box_content = [float(x) for x in list_box.get(0, END)]
        if float(value) not in list_box_content:
            list_box.insert(0, value)

    def remove_from_listbox(self,
                       list_box: Listbox):
        selection = list_box.curselection()
        if selection:
            list_box.delete(selection[0])

    def run_hdbscan_clustering(self):
        alpha = [float(x) for x in self.alpha_listb.get(0, END)]
        min_cluster_size = [int(x) for x in self.min_cluster_size_listb.get(0, END)]
        min_samples = [int(x) for x in self.min_samples_listb.get(0, END)]
        epsilon = [float(x) for x in self.epsilon_listb.get(0, END)]
        hyper_parameters = {'alpha': alpha,
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': epsilon}
        data_path = self.dimensionality_reduction_data_selected.folder_path
        save_dir = self.clustering_save_dir_folder.folder_path
        clusterer = HDBSCANClusterer(data_path=data_path, save_dir=save_dir)
        clusterer.fit(hyper_parameters=hyper_parameters)



class TransformClustererPopUp(object):
    def __init__(self):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("CLUSTERER: TRANSFORM")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.features_options = ['EXCLUDE', 'INCLUDE: ORIGINAL', 'INCLUDE: SCALED']


        self.transform_clusterer_frm = LabelFrame(self.main_frm, text='CLUSTERER: TRANSFORM', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.cluster_model_select = FileSelect(self.transform_clusterer_frm, 'CLUSTER MODEL (PICKLE):', lblwidth=25)
        self.dataset_select = FileSelect(self.transform_clusterer_frm, 'DATASET (PICKLE):',  lblwidth=25)
        self.save_dir = FolderSelect(self.transform_clusterer_frm, "SAVE DIRECTORY:",  lblwidth=25)
        self.feature_option_dropdown = DropDownMenu(self.transform_clusterer_frm, 'INCLUDE FEATURES:', self.features_options, '25')

        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        run_btn = self.create_btn = Button(run_frm, text='RUN', fg='blue', command=lambda: self.run())

        self.transform_clusterer_frm.grid(row=0, column=0, sticky=NW)
        self.cluster_model_select.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.feature_option_dropdown.grid(row=2, column=0, sticky=NW)

        run_frm.grid(row=1, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def run(self):
        clusterer_path = self.cluster_model_select.file_path
        data_path = self.dataset_select.file_path
        save_dir = self.save_dir.folder_path
        settings = {'features': self.feature_option_dropdown.getChoices()}

        _ = HDBSCANTransform(clusterer_model_path=clusterer_path,
                             data_path=data_path,
                             save_dir=save_dir,
                             settings=settings)

class ClusterVisualizerPopUp(object):
    def __init__(self,
                 config_path: str):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.config_path = config_path
        self.main_frm.wm_title("CLUSTER EXAMPLE VISUALIZATIONS")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.include_pose_var = BooleanVar(value=True)
        self.speed_options = list(np.arange(0.1, 2.1, 0.1))

        self.data_frm = LabelFrame(self.main_frm, text='DATA', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.videos_dir_select = FolderSelect(self.data_frm, "VIDEOS DIRECTORY:", lblwidth=25)
        self.dataset_file_selected = FileSelect(self.data_frm, "DATASET (PICKLE): ", lblwidth=25)
        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.include_pose = Checkbutton(self.settings_frm, text='INCLUDE POSE-ESTIMATION', variable=self.include_pose_var, command=lambda: self.toggle_pose_settings())
        self.circle_size_entry = Entry_Box(self.settings_frm, 'CIRCLE SIZE: ', '25', validation='numeric')
        self.circle_size_entry.set_state(setstatus='disable')
        self.speed_dropdown = DropDownMenu(self.settings_frm, 'VIDEO SPEED:', self.speed_options, '25')

        self.run_frm = LabelFrame(self.main_frm, text='RUN', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        run_btn = Button(self.run_frm, text='RUN', fg='blue', command=lambda: self.run())

        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.videos_dir_select.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=1, column=0, sticky=NW)
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.include_pose.grid(row=0, column=0, sticky=NW)
        self.circle_size_entry.grid(row=1, column=0, sticky=NW)
        self.speed_dropdown.grid(row=2, column=0, sticky=NW)
        self.run_frm.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def toggle_pose_settings(self):
        if self.include_pose_var.get():
            self.circle_size_entry.set_state(setstatus='normal')
        else:
            self.circle_size_entry.set_state(setstatus='disable')

    def run(self):
        circle_size = np.inf
        include_pose = self.include_pose_var.get()
        speed = float(self.speed_dropdown.getChoices())
        data_path = self.dataset_file_selected.file_path
        check_file_exist_and_readable(file_path=data_path)
        videos_dir = self.videos_dir_select.folder_path

        if include_pose:
            circle_size = self.circle_size_entry.entry_get
            check_int(name='CIRCLE SIZE', value=self.circle_size_entry.entry_get, min_value=1)

        settings = {'videos_speed': speed,
                    'pose': {'include': include_pose, 'circle_size': circle_size}}

        cluster_visualizer = ClusterVisualizer(config_path=self.config_path,
                                               settings=settings,
                                               video_dir=videos_dir,
                                               data_path=data_path)
        cluster_visualizer.create()


class ClusterFrequentistStatisticsPopUp(object):
    def __init__(self,
                 config_path: str):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.config_path = config_path
        self.main_frm.wm_title("CLUSTER FREQUENTIST STATISTICS")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.feature_descriptive_stats_var = BooleanVar(value=True)
        self.feature_oneway_anova_var = BooleanVar(value=True)
        self.feature_tukey_var = BooleanVar(value=True)
        self.use_scaled_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dim_reduction_model = FileSelect(self.data_frm, 'CLUSTERER PATH:', lblwidth=25)

        self.d_stats_frm = LabelFrame(self.main_frm, text='CLUSTER STATISTICS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.feature_means_cb = Checkbutton(self.d_stats_frm, text='CLUSTER DESCRIPTIVE STATISTICS', variable=self.feature_descriptive_stats_var)
        self.feature_anova_cb = Checkbutton(self.d_stats_frm, text='CLUSTER FEATURE ONE-WAY ANOVA', variable=self.feature_oneway_anova_var)
        self.feature_tukey_posthoc_cb = Checkbutton(self.d_stats_frm, text='CLUSTER FEATURE POST-HOC (TUKEY)', variable=self.feature_tukey_var)
        self.use_scaled_cb = Checkbutton(self.d_stats_frm, text='USE SCALED FEATURE VALUES', variable=self.use_scaled_var)


        self.run_frm = LabelFrame(self.main_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        run_btn = Button(self.run_frm, text='RUN', fg='blue', command=lambda: self.run())

        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.dim_reduction_model.grid(row=0, column=0, sticky=NW)

        self.d_stats_frm.grid(row=1, column=0, sticky=NW)
        self.feature_means_cb.grid(row=0, column=0, sticky=NW)
        self.feature_anova_cb.grid(row=1, column=0, sticky=NW)
        self.feature_tukey_posthoc_cb.grid(row=2, column=0, sticky=NW)
        self.use_scaled_cb.grid(row=3, column=0, sticky=NW)

        self.run_frm.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def run(self):
        settings = {'scaled': self.use_scaled_var.get(),
                    'anova': self.feature_oneway_anova_var.get(),
                    'tukey_posthoc': self.feature_oneway_anova_var.get(),
                    'descriptive_statistics': self.feature_descriptive_stats_var.get()}
        model_path = self.dim_reduction_model.file_path
        check_file_exist_and_readable(model_path)

        descriptive_calculator = ClusterFrequentistCalculator(config_path=self.config_path,
                                                              data_path=model_path,
                                                              settings=settings)

        descriptive_calculator.run()


class ClusterMLStatisticsPopUp(object):
    def __init__(self,
                 config_path: str):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.config_path = config_path
        self.main_frm.wm_title("CLUSTER XAI STATISTICS")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.gini_importance_var = BooleanVar(value=True)
        self.permutation_importance_var = BooleanVar(value=True)
        self.shap_var = BooleanVar(value=False)
        self.shap_method_options = ['Paired clusters']
        self.shap_sample_options = list(range(100, 1100, 100))

        self.data_frm = LabelFrame(self.main_frm, text='DATA', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.clusterer_data = FileSelect(self.data_frm, 'CLUSTERER PATH:', lblwidth=25)

        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.rf_gini_importance_cb = Checkbutton(self.settings_frm, text='CLUSTER RF GINI IMPORTANCE', variable=self.gini_importance_var)
        self.rf_permutation_cb = Checkbutton(self.settings_frm, text='CLUSTER RF PERMUTATION IMPORTANCE', variable=self.permutation_importance_var)
        self.cluster_shap_cb = Checkbutton(self.settings_frm, text='CLUSTER RF SHAPLEY VALUES', variable=self.shap_var, command= lambda: self.activate_shap_menu)
        self.shap_method_dropdown = DropDownMenu(self.settings_frm, 'SHAP METHOD:', self.shap_method_options, '25')
        self.shap_sample_dropdown = DropDownMenu(self.settings_frm, 'SHAP SAMPLES:', self.shap_sample_options, '25')
        self.shap_method_dropdown.disable()
        self.shap_sample_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        run_btn = Button(self.run_frm, text='RUN', fg='blue', command=lambda: self.run())

        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.clusterer_data.grid(row=0, column=0, sticky=NW)

        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.rf_gini_importance_cb.grid(row=0, column=0, sticky=NW)
        self.rf_permutation_cb.grid(row=1, column=0, sticky=NW)
        self.cluster_shap_cb.grid(row=2, column=0, sticky=NW)
        self.shap_method_dropdown.grid(row=3, column=0, sticky=NW)
        self.shap_sample_dropdown.grid(row=4, column=0, sticky=NW)
        self.run_frm.grid(row=5, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def activate_shap_menu(self):
        if self.shap_var.get():
            self.shap_method_dropdown.enable()
            self.shap_sample_dropdown.enable()
        else:
            self.shap_method_dropdown.disable()
            self.shap_sample_dropdown.disable()

    def run(self):
        clusterer_path = self.clusterer_data.file_path
        gini_importance = self.gini_importance_var.get()
        permutation_importance = self.permutation_importance_var.get()
        shap = self.shap_var.get()
        shap_method = self.shap_method_dropdown.getChoices()
        shap_sample = self.shap_sample_dropdown.getChoices()


        settings = {'gini_importance': gini_importance, 'permutation_importance': permutation_importance,
                    'shap': {'method': shap_method, 'run': shap, 'sample': shap_sample}}

        _ = ClusterXAICalculator(data_path=clusterer_path,
                                 settings=settings,
                                 config_path=self.config_path)


class EmbedderCorrelationsPopUp(object):
    def __init__(self,
                 config_path: str):
        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.config_path = config_path
        self.main_frm.wm_title("EMBEDDING CORRELATIONS")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)
        self.config_path = config_path
        self.correlation_options = ['SPEARMAN', 'PEARSONS', 'KENDALL']

        self.spearman_var = BooleanVar(value=True)
        self.pearsons_var = BooleanVar(value=True)
        self.kendall_var = BooleanVar(value=True)
        self.plots_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(self.main_frm, text='DATA', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dataset_file_selected = FileSelect(self.data_frm, "DATASET (PICKLE): ")

        self.settings_frm = LabelFrame(self.main_frm, text='SETTINGS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.spearman_cb = Checkbutton(self.settings_frm, text='SPEARMAN', variable=self.spearman_var)
        self.pearsons_cb = Checkbutton(self.settings_frm, text='PEARSONS', variable=self.pearsons_var)
        self.kendall_cb = Checkbutton(self.settings_frm, text='KENDALL', variable=self.kendall_var)
        self.plots_cb = Checkbutton(self.settings_frm, text='PLOTS', variable=self.plots_var, command=lambda: self.activate_plots())
        self.plot_correlation_dropdown = DropDownMenu(self.settings_frm, 'PLOT CORRELATION:', self.correlation_options, '25')
        self.plot_correlation_clr_dropdown = DropDownMenu(self.settings_frm, 'PLOT CORRELATION:', Options.PALETTE_OPTIONS.value, '25')
        self.plot_correlation_dropdown.setChoices(self.correlation_options[0])
        self.plot_correlation_dropdown.disable()
        self.plot_correlation_clr_dropdown.setChoices(Options.PALETTE_OPTIONS.value[0])
        self.plot_correlation_clr_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        run_btn = Button(self.run_frm, text='RUN', fg='blue', command=lambda: self.run())

        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.spearman_cb.grid(row=0, column=0, sticky=NW)
        self.pearsons_cb.grid(row=1, column=0, sticky=NW)
        self.kendall_cb.grid(row=2, column=0, sticky=NW)
        self.plots_cb.grid(row=3, column=0, sticky=NW)
        self.plot_correlation_dropdown.grid(row=4, column=0, sticky=NW)
        self.plot_correlation_clr_dropdown.grid(row=5, column=0, sticky=NW)
        self.run_frm.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)


    def activate_plots(self):
        if self.plots_var.get():
            self.plot_correlation_dropdown.enable()
            self.plot_correlation_clr_dropdown.enable()
        else:
            self.plot_correlation_dropdown.disable()
            self.plot_correlation_clr_dropdown.disable()

    def run(self):
        settings = {'correlations': [], 'plots': {'create': False, 'correlations': None, 'palette': None}}
        if self.spearman_var.get(): settings['correlations'].append('spearman')
        if self.pearsons_var.get(): settings['correlations'].append('pearson')
        if self.kendall_var.get(): settings['correlations'].append('kendall')

        if self.plots_var.get():
            settings['plots']['create'] = True
            settings['plots']['correlations'] = self.plot_correlation_dropdown.getChoices()
            settings['plots']['palette'] = self.plot_correlation_clr_dropdown.getChoices()

        data_path = self.dataset_file_selected.file_path

        _ = EmbeddingCorrelationCalculator(config_path=self.config_path,
                                           data_path=data_path,
                                           settings=settings)




#_ = ClusterFrequentistStatisticsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')


# settings = {'video_speed': 0.01, 'pose': {'include': True, 'circle_size': 5}}
# test = ClusterVisualizer(video_dir='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/videos',
#                          data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/dreamy_spence_awesome_elion.pickle',
#                          settings=settings,
#                          config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')
# test.create()






#_ = BatchDataExtractorPopUp()
#_ = GridSearchClusterVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')