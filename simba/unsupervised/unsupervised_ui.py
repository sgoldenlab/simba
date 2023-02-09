from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          read_project_path_and_file_type,
                                          check_float)
from tkinter import *
from simba.tkinter_functions import (hxtScrollbar,
                                     Entry_Box,
                                     DropDownMenu,
                                     FileSelect,
                                     FolderSelect)
import tkinter.ttk as ttk
from simba.enums import Formats
from simba.unsupervised.umap_embedder import UMAPEmbedder
from simba.unsupervised.hdbscan_clusterer import HDBSCANClusterer
from simba.enums import ReadConfig, Dtypes
from simba.train_model_functions import get_all_clf_names
from simba.unsupervised.dataset_creator import DatasetCreator
from simba.unsupervised.pop_up_classes import GridSearchVisualizationPopUp

class UnsupervisedGUI(object):
    def __init__(self,
                 config_path: str):

        self.config, self.config_path = read_config_file(config_path), config_path
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.main = Toplevel()
        self.main.minsize(900, 800)
        self.main.wm_title("UNSUPERVISED ANALYSIS")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(0, weight=1)
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.data_slice_options = ['ALL FEATURES (EXCLUDING POSE)',
                                   'ALL FEATURES (INCLUDING POSE)',
                                   'USER-DEFINED FEATURE SET']
        self.normalization_options = ['STANDARD',
                                      'QUANTILE',
                                      'MIN-MAX']
        self.dim_reduction_algo_options = ['UMAP',
                                           'PCA',
                                           'TSNE']

        self.cluster_algo_options = ['HDBSCAN']
        self.clf_slice_options = ['ALL FRAMES']
        for clf_name in self.clf_names:
            self.clf_slice_options.append(f'ALL {clf_name} PRESENT')
            self.clf_slice_options.append(f'ALL {clf_name} ABSENT')

        self.feature_removal_options = list(range(10, 100, 10))
        self.feature_removal_options = [str(x) + '%' for x in self.feature_removal_options]
        self.feature_removal_options.insert(0, 'NONE')
        self.main = ttk.Notebook(hxtScrollbar(self.main))
        self.create_dataset_tab = ttk.Frame(self.main)
        self.dimensionality_reduction_tab = ttk.Frame(self.main)
        self.clustering_tab = ttk.Frame(self.main)
        self.visualization_tab = ttk.Frame(self.main)
        self.metrics_tab = ttk.Frame(self.main)
        self.main.add(self.create_dataset_tab, text=f'{"[CREATE DATASET]": ^20s}')
        self.main.add(self.dimensionality_reduction_tab, text=f'{"[DIMENSIONALITY REDUCTION]": ^20s}')
        self.main.add(self.clustering_tab, text=f'{"[CLUSTERING]": ^20s}')
        self.main.add(self.visualization_tab, text=f'{"[VISUALIZATION]": ^20s}')
        self.main.add(self.metrics_tab, text=f'{"[METRICS]": ^20s}')
        self.main.grid(row=0)

        create_dataset_frm = LabelFrame(self.create_dataset_tab, text='CREATE DATASET', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.feature_file_selected = FileSelect(create_dataset_frm, "FEATURE FILE (CSV)", lblwidth=15)

        self.data_slice_dropdown = DropDownMenu(create_dataset_frm, 'DATA SLICE:', self.data_slice_options, '15', com= lambda x: self.change_status_of_file_select())
        self.data_slice_dropdown.setChoices(self.data_slice_options[0])
        self.clf_slice_dropdown = DropDownMenu(create_dataset_frm, 'CLASSIFIER SLICE:', self.clf_slice_options, '15')
        self.clf_slice_dropdown.setChoices(self.clf_slice_options[0])
        self.create_btn = Button(create_dataset_frm, text='CREATE DATASET', fg='blue', command= lambda: self.create_dataset())

        create_dataset_frm.grid(row=0, column=0, sticky=NW)
        self.data_slice_dropdown.grid(row=0, column=0, sticky=NW)
        self.clf_slice_dropdown.grid(row=1, column=0, sticky=NW)
        self.feature_file_selected.grid(row=2, column=0, sticky=NW)
        self.create_btn.grid(row=3, column=0, sticky=NW)

        self.dim_reduction_frm = LabelFrame(self.dimensionality_reduction_tab, text='DIMENSIONALITY REDUCTION', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dataset_frm = LabelFrame(self.dim_reduction_frm, text='DATASET', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dataset_file_selected = FileSelect(self.dataset_frm, "DATASET (PICKLE): ")
        self.save_frm = LabelFrame(self.dim_reduction_frm, text='SAVE', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dr_save_dir = FolderSelect(self.save_frm, "SAVE DIRECTORY: ")
        settings_frm = LabelFrame(self.dim_reduction_frm, text='SETTINGS', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.scaling_dropdown = DropDownMenu(settings_frm, 'SCALING:', self.normalization_options, '20')
        self.scaling_dropdown.setChoices(self.normalization_options[0])
        self.feature_removal_dropdown = DropDownMenu(settings_frm, 'VARIANCE THRESHOLD:', self.feature_removal_options, '20')
        self.feature_removal_dropdown.setChoices(self.feature_removal_options[0])
        self.choose_algo_dropdown = DropDownMenu(settings_frm, 'ALGORITHM:', self.dim_reduction_algo_options, '20', com=lambda x: self.show_dr_algo_hyperparameters())
        self.choose_algo_dropdown.setChoices(self.dim_reduction_algo_options[0])
        self.show_dr_algo_hyperparameters()

        self.dim_reduction_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)
        self.save_frm.grid(row=1, column=0, sticky=NW)
        self.dr_save_dir.grid(row=0, column=0, sticky=NW)
        settings_frm.grid(row=2, column=0, sticky=NW)
        self.choose_algo_dropdown.grid(row=0, column=0, sticky=NW)
        self.feature_removal_dropdown.grid(row=1, column=0, sticky=NW)
        self.scaling_dropdown.grid(row=2, column=0, sticky=NW)

        self.clustering_frm = LabelFrame(self.clustering_tab, text='CLUSTERING', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.cluster_dataset_frm = LabelFrame(self.clustering_frm, text='DATASET', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dimensionality_reduction_data_selected = FolderSelect(self.cluster_dataset_frm, "DATASET DIR: ")
        self.clustering_save_dir_frm = LabelFrame(self.clustering_frm, text='SAVE', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.clustering_save_dir_folder = FolderSelect(self.clustering_save_dir_frm, "SAVE DIR: ")
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

        self.visualization_frm = LabelFrame(self.visualization_tab, text='VISUALIZATIONS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.grid_search_visualization_btn = Button(self.visualization_frm, text='GRID-SEARCH VISUALIZATION', fg='blue', command= lambda: self.launch_grid_search_visualization_pop_up())
        self.visualization_frm.grid(row=0, column=0, sticky='NW')
        self.grid_search_visualization_btn.grid(row=0, column=0, sticky='NW')

        self.metrics_frm = LabelFrame(self.metrics_tab, text='METRICS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dbcv_btn = Button(self.metrics_frm, text='DENSITY-BASED CLUSTER VALIDATION', fg='blue', command=lambda: None)

        self.metrics_frm.grid(row=0, column=0, sticky='NW')
        self.dbcv_btn.grid(row=0, column=0, sticky='NW')

        self.main.mainloop()

    def run_create_dataset(self):
        slice_type = self.data_slice_dropdown.getChoices()
        clf_slice = self.clf_slice_dropdown.getChoices()
        feature_path = None
        if slice_type == 'USER-DEFINED FEATURE SET':
            feature_path = self.feature_file_selected.file_path
        settings = {'data_type': slice_type, 'feature_path': feature_path, 'clf_slice': clf_slice}
        _ = DatasetCreator(settings=settings, config_path=self.config_path)

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

    def show_dr_algo_hyperparameters(self):
        if hasattr(self, 'dr_hyperparameters_frm'):
            self.dr_hyperparameters_frm.destroy()
            self.dr_value_frm.destroy()
            self.run_frm.destroy()

        if self.choose_algo_dropdown.getChoices() == 'UMAP':
            self.dr_hyperparameters_frm = LabelFrame(self.dim_reduction_frm, text='GRID SEARCH HYPER-PARAMETERS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
            self.dr_value_frm = LabelFrame(self.dim_reduction_frm, fg='black')
            self.dr_value_entry_box = Entry_Box(self.dr_value_frm, 'VALUE: ', '12')
            n_neighbors_estimators_lbl = Label(self.dr_hyperparameters_frm, text='N NEIGHBOURS')
            min_distance_lbl = Label(self.dr_hyperparameters_frm, text='MIN DISTANCE')
            add_min_distance_btn = self.create_btn = Button(self.dr_hyperparameters_frm, text='ADD', fg='blue', command= lambda: self.add_to_listbox(list_box=self.min_distance_listb, entry_box=self.dr_value_entry_box))
            add_neighbours_btn = Button(self.dr_hyperparameters_frm, text='ADD', fg='blue', command=lambda: self.add_to_listbox(list_box=self.n_neighbors_estimators_listb, entry_box=self.dr_value_entry_box))
            remove_min_distance_btn = Button(self.dr_hyperparameters_frm, text='REMOVE', fg='red', command= lambda: self.remove_from_listbox(list_box=self.min_distance_listb))
            remove_neighbours_btn = Button(self.dr_hyperparameters_frm, text='REMOVE', fg='red', command=lambda: self.remove_from_listbox(list_box=self.n_neighbors_estimators_listb))
            self.n_neighbors_estimators_listb = Listbox(self.dr_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.min_distance_listb = Listbox(self.dr_hyperparameters_frm, bg='lightgrey', fg='black', height=5, width=15)
            self.dr_value_frm.grid(row=3, column=0, sticky=NW)
            self.dr_value_entry_box.grid(row=0, column=1, sticky=NW)
            self.dr_hyperparameters_frm.grid(row=4, column=0, sticky=NW)
            n_neighbors_estimators_lbl.grid(row=1, column=0)
            min_distance_lbl.grid(row=1, column=1)
            add_neighbours_btn.grid(row=2, column=0)
            add_min_distance_btn.grid(row=2, column=1)
            remove_neighbours_btn.grid(row=3, column=0)
            remove_min_distance_btn.grid(row=3, column=1)
            self.n_neighbors_estimators_listb.grid(row=4, column=0, sticky=NW)
            self.min_distance_listb.grid(row=4, column=1, sticky=NW)
            self.run_frm = LabelFrame(self.dim_reduction_frm, text='RUN', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
            run_btn = Button(self.run_frm, text='RUN', fg='blue', command= lambda: self.run_umap_gridsearch())
            self.run_frm.grid(row=5, column=0, sticky=NW)
            run_btn.grid(row=0, column=1, sticky=NW)

    def change_status_of_file_select(self):
        if self.data_slice_dropdown.getChoices() == 'USER-DEFINED FEATURE SET':
            self.feature_file_selected.set_state(setstatus=NORMAL)
        else:
            self.feature_file_selected.set_state(setstatus=DISABLED)

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

    def create_dataset(self):
        data_slice_type = self.data_slice_dropdown.getChoices()
        classifier_slice_type = self.clf_slice_dropdown.getChoices()
        feature_file_path = None
        if data_slice_type is 'USER-DEFINED FEATURE SET':
            feature_file_path = self.feature_file_selected.file_path
        settings = {'data_slice': data_slice_type,
                    'clf_slice': classifier_slice_type,
                    'feature_path': feature_file_path}
        _ = DatasetCreator(settings=settings, config_path=self.config_path)

    def run_umap_gridsearch(self):
        min_distance = [float(x) for x in self.min_distance_listb.get(0, END)]
        n_neighbours = [float(x) for x in self.n_neighbors_estimators_listb.get(0, END)]
        variance = None
        if self.feature_removal_dropdown.getChoices() != 'NONE':
            variance = int(self.feature_removal_dropdown.getChoices()[:-1]) / 100
        save_path = self.dr_save_dir.folder_path
        data_path = self.dataset_file_selected.file_path

        if len(min_distance) == 0 or len(n_neighbours) == 0:
            print('SIMBA ERROR: Provide values for neighbors and min distances')
            raise ValueError('SIMBA ERROR: Provide values for neighbors and min distances')
        else:
            hyperparameters = {'n_neighbors': n_neighbours,
                               'min_distance': min_distance,
                               'scaler': self.scaling_dropdown.getChoices(),
                               'variance': variance}
            umap_embedder = UMAPEmbedder(config_path=self.config_path,
                                         data_path=data_path,
                                         save_dir=save_path)

            umap_embedder.fit(hyper_parameters=hyperparameters)

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

        clusterer = HDBSCANClusterer(config_path=self.config_path, data_path=data_path, save_dir=save_dir)
        clusterer.fit(hyper_parameters=hyper_parameters)

    def launch_grid_search_visualization_pop_up(self):
        _ = GridSearchVisualizationPopUp(config_path=self.config_path)

#_ = UnsupervisedGUI(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')