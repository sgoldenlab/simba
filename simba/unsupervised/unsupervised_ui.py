from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          read_project_path_and_file_type)
from tkinter import *
from simba.tkinter_functions import (hxtScrollbar,
                                     Entry_Box,
                                     DropDownMenu,
                                     FileSelect)
import tkinter.ttk as ttk
from simba.enums import Formats, Options
from simba.enums import ReadConfig, Dtypes
from simba.train_model_functions import get_all_clf_names
from simba.unsupervised.dataset_creator import DatasetCreator
from simba.unsupervised.pop_up_classes import (GridSearchClusterVisualizerPopUp,
                                               BatchDataExtractorPopUp,
                                               FitDimReductionPopUp,
                                               FitClusterModelsPopUp,
                                               TransformDimReductionPopUp,
                                               TransformClustererPopUp,
                                               ClusterVisualizerPopUp,
                                               ClusterFrequentistStatisticsPopUp,
                                               ClusterMLStatisticsPopUp,
                                               EmbedderCorrelationsPopUp)

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

        self.clf_slice_options = [f'ALL CLASSIFIERS ({str(len(self.clf_names))})']
        for clf_name in self.clf_names:
            self.clf_slice_options.append(f'{clf_name}')

        self.bout_aggregation_options = ['MEAN', 'MEDIAN']
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
        self.feature_file_selected = FileSelect(create_dataset_frm, "FEATURE FILE (CSV)", lblwidth=25)

        self.data_slice_dropdown = DropDownMenu(create_dataset_frm, 'FEATURE SLICE:', self.data_slice_options, '25', com= lambda x: self.change_status_of_file_select())
        self.data_slice_dropdown.setChoices(self.data_slice_options[0])
        self.clf_slice_dropdown = DropDownMenu(create_dataset_frm, 'CLASSIFIER SLICE:', self.clf_slice_options, '25')
        self.clf_slice_dropdown.setChoices(self.clf_slice_options[0])
        self.change_status_of_file_select()
        self.bout_dropdown = DropDownMenu(create_dataset_frm, 'BOUT AGGREGATION:', self.bout_aggregation_options, '25')
        self.bout_dropdown.setChoices(choice='MEAN')
        self.min_bout_length = Entry_Box(create_dataset_frm, 'MINIMUM BOUT LENGTH (MS): ', '25', validation='numeric')
        self.min_bout_length.entry_set(val=0)
        self.create_btn = Button(create_dataset_frm, text='CREATE DATASET', fg='blue', command= lambda: self.create_dataset())

        create_dataset_frm.grid(row=0, column=0, sticky=NW)
        self.data_slice_dropdown.grid(row=0, column=0, sticky=NW)
        self.clf_slice_dropdown.grid(row=1, column=0, sticky=NW)
        self.bout_dropdown.grid(row=2, column=0, sticky=NW)
        self.feature_file_selected.grid(row=3, column=0, sticky=NW)
        self.min_bout_length.grid(row=4, column=0, sticky=NW)
        self.create_btn.grid(row=5, column=0, sticky=NW)

        self.dim_reduction_frm = LabelFrame(self.dimensionality_reduction_tab, text='DIMENSIONALITY REDUCTION', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.dim_reduction_fit_btn = Button(self.dim_reduction_frm, text='DIMENSIONALITY REDUCTION MODELS: FIT', fg='blue', command= lambda: FitDimReductionPopUp())
        self.dim_reduction_transform_btn = Button(self.dim_reduction_frm, text='DIMENSIONALITY REDUCTION MODELS: TRANSFORM', fg='red', command= lambda: TransformDimReductionPopUp())

        self.dim_reduction_frm.grid(row=0, column=0, sticky=NW)
        self.dim_reduction_fit_btn.grid(row=1, column=0, sticky=NW)
        self.dim_reduction_transform_btn.grid(row=2, column=0, sticky=NW)

        self.clustering_frm = LabelFrame(self.clustering_tab, text='CLUSTERING', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.cluster_fit_btn = Button(self.clustering_frm, text='CLUSTERING MODELS: FIT ', fg='blue', command= lambda: FitClusterModelsPopUp())
        self.cluster_transform_btn = Button(self.clustering_frm, text='CLUSTERING MODELS: TRANSFORM ', fg='red', command=lambda: TransformClustererPopUp())
        self.clustering_frm.grid(row=0, column=0, sticky=NW)
        self.cluster_fit_btn.grid(row=1, column=0, sticky=NW)
        self.cluster_transform_btn.grid(row=2, column=0, sticky=NW)

        self.visualization_frm = LabelFrame(self.visualization_tab, text='VISUALIZATIONS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.grid_search_visualization_btn = Button(self.visualization_frm, text='GRID-SEARCH VISUALIZATION', fg='blue', command= lambda: self.launch_grid_search_visualization_pop_up())
        self.cluster_visualizer = Button(self.visualization_frm, text='CLUSTER VISUALIZER', fg='red', command= lambda: ClusterVisualizerPopUp(config_path=self.config_path))
        self.visualization_frm.grid(row=0, column=0, sticky='NW')
        self.grid_search_visualization_btn.grid(row=0, column=0, sticky='NW')
        self.cluster_visualizer.grid(row=1, column=0, sticky='NW')

        self.metrics_frm = LabelFrame(self.metrics_tab, text='METRICS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dbcv_btn = Button(self.metrics_frm, text='DENSITY-BASED CLUSTER VALIDATION', fg='blue', command=lambda: None)
        self.extract_single_metrics_btn = Button(self.metrics_frm, text='EXTRACT UNSUPERVISED RESULTS (MULTIPLE MODELS)', fg='red', command=lambda: BatchDataExtractorPopUp())
        self.cluster_descriptives_btn = Button(self.metrics_frm, text='CLUSTER FREQUENTIST STATISTICS', fg='green', command=lambda: ClusterFrequentistStatisticsPopUp(config_path=self.config_path))
        self.cluster_xai_btn = Button(self.metrics_frm, text='CLUSTER XAI STATISTICS', fg='blue', command=lambda: ClusterMLStatisticsPopUp(config_path=self.config_path))
        self.embedding_corr_btn = Button(self.metrics_frm, text='EMBEDDING CORRELATIONS', fg='orange', command=lambda: EmbedderCorrelationsPopUp(config_path=self.config_path))

        self.metrics_frm.grid(row=0, column=0, sticky='NW')
        self.dbcv_btn.grid(row=0, column=0, sticky='NW')
        self.extract_single_metrics_btn.grid(row=1, column=0, sticky='NW')
        self.cluster_descriptives_btn.grid(row=2, column=0, sticky='NW')
        self.cluster_xai_btn.grid(row=3, column=0, sticky='NW')
        self.embedding_corr_btn.grid(row=4, column=0, sticky='NW')

        self.main.mainloop()

    def change_status_of_file_select(self):
        if self.data_slice_dropdown.getChoices() == 'USER-DEFINED FEATURE SET':
            self.feature_file_selected.set_state(setstatus=NORMAL)
        else:
            self.feature_file_selected.set_state(setstatus=DISABLED)

    def create_dataset(self):
        data_slice_type = self.data_slice_dropdown.getChoices()
        classifier_slice_type = self.clf_slice_dropdown.getChoices()
        bout_selection = self.bout_dropdown.getChoices()
        bout_length = self.min_bout_length.entry_get
        feature_file_path = None
        if data_slice_type is 'USER-DEFINED FEATURE SET':
            feature_file_path = self.feature_file_selected.file_path
        settings = {'data_slice': data_slice_type,
                    'clf_slice': classifier_slice_type,
                    'bout_aggregation': bout_selection,
                    'min_bout_length': bout_length,
                    'feature_path': feature_file_path}
        _ = DatasetCreator(settings=settings, config_path=self.config_path)

    def launch_grid_search_visualization_pop_up(self):
        _ = GridSearchClusterVisualizerPopUp(config_path=self.config_path)

_ = UnsupervisedGUI(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')