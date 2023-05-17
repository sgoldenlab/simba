__author__ = "Simon Nilsson"

from tkinter import *
from simba.ui.tkinter_functions import (hxtScrollbar,
                                        Entry_Box,
                                        DropDownMenu,
                                        FileSelect)
import tkinter.ttk as ttk
from simba.utils.enums import Formats
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.unsupervised.enums import UMLOptions, Unsupervised
from simba.unsupervised.dataset_creator import DatasetCreator
from simba.unsupervised.pop_up_classes import (GridSearchVisualizerPopUp,
                                               DataExtractorPopUp,
                                               FitDimReductionPopUp,
                                               FitClusterModelsPopUp,
                                               TransformDimReductionPopUp,
                                               TransformClustererPopUp,
                                               ClusterVisualizerPopUp,
                                               ClusterFrequentistStatisticsPopUp,
                                               ClusterXAIPopUp,
                                               EmbedderCorrelationsPopUp,
                                               PrintEmBeddingInfoPopUp,
                                               DBCVPopUp)

class UnsupervisedGUI(ConfigReader, PopUpMixin):
    """
    Main access to unsupervised interface
    """
    def __init__(self,
                 config_path: str):



        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title="UNSUPERVISED ANALYSIS", config_path=config_path, size=(1000, 800))
        self.main_frm = Toplevel()
        self.main_frm.minsize(1000, 800)
        self.main_frm.wm_title("UNSUPERVISED ANALYSIS")
        self.main_frm.columnconfigure(0, weight=1)
        self.main_frm.rowconfigure(0, weight=1)
        self.main_frm = ttk.Notebook(hxtScrollbar(self.main_frm))
        self.create_dataset_tab = ttk.Frame(self.main_frm)
        self.dimensionality_reduction_tab = ttk.Frame(self.main_frm)
        self.clustering_tab = ttk.Frame(self.main_frm)
        self.visualization_tab = ttk.Frame(self.main_frm)
        self.metrics_tab = ttk.Frame(self.main_frm)

        self.main_frm.add(self.create_dataset_tab, text=f'{"[CREATE DATASET]": ^20s}', compound='left', image=self.menu_icons['features']['img'])
        self.main_frm.add(self.dimensionality_reduction_tab, text=f'{"[DIMENSIONALITY REDUCTION]": ^20s}', compound='left', image=self.menu_icons['dimensionality_reduction']['img'])
        self.main_frm.add(self.clustering_tab, text=f'{"[CLUSTERING]": ^20s}', compound='left', image=self.menu_icons['cluster']['img'])
        self.main_frm.add(self.visualization_tab, text=f'{"[VISUALIZATION]": ^20s}', compound='left', image=self.menu_icons['visualize']['img'])
        self.main_frm.add(self.metrics_tab, text=f'{"[METRICS]": ^20s}', compound='left', image=self.menu_icons['metrics']['img'])
        self.main_frm.grid(row=0)

        self.clf_slice_options = [f'ALL CLASSIFIERS ({len(self.clf_names)})']
        for clf_name in self.clf_names: self.clf_slice_options.append(f'{clf_name}')
        create_dataset_frm = LabelFrame(self.create_dataset_tab, text='CREATE DATASET', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.feature_file_selected = FileSelect(create_dataset_frm, "FEATURE FILE (CSV)", lblwidth=25)

        self.data_slice_dropdown = DropDownMenu(create_dataset_frm, 'FEATURE SLICE:', UMLOptions.FEATURE_SLICE_OPTIONS.value, '25', com= lambda x: self.change_status_of_file_select())
        self.data_slice_dropdown.setChoices(UMLOptions.FEATURE_SLICE_OPTIONS.value[0])
        self.clf_slice_dropdown = DropDownMenu(create_dataset_frm, 'CLASSIFIER SLICE:', self.clf_slice_options, '25')
        self.clf_slice_dropdown.setChoices(self.clf_slice_options[0])
        self.change_status_of_file_select()
        self.bout_dropdown = DropDownMenu(create_dataset_frm, 'BOUT AGGREGATION METHOD:', UMLOptions.BOUT_AGGREGATION_METHODS.value, '25')
        self.bout_dropdown.setChoices(choice=UMLOptions.BOUT_AGGREGATION_METHODS.value[0])
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
        self.dim_reduction_fit_btn = Button(self.dim_reduction_frm, text='DIMENSIONALITY REDUCTION: FIT', fg='blue', command= lambda: FitDimReductionPopUp())
        self.dim_reduction_transform_btn = Button(self.dim_reduction_frm, text='DIMENSIONALITY REDUCTION: TRANSFORM', fg='red', command= lambda: TransformDimReductionPopUp())
        self.dim_reduction_frm.grid(row=0, column=0, sticky=NW)
        self.dim_reduction_fit_btn.grid(row=1, column=0, sticky=NW)
        self.dim_reduction_transform_btn.grid(row=2, column=0, sticky=NW)

        self.clustering_frm = LabelFrame(self.clustering_tab, text='CLUSTERING', pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        self.cluster_fit_btn = Button(self.clustering_frm, text='CLUSTERING: FIT ', fg='blue', command= lambda: FitClusterModelsPopUp())
        self.cluster_transform_btn = Button(self.clustering_frm, text='CLUSTERING: TRANSFORM ', fg='red', command=lambda: TransformClustererPopUp())
        self.clustering_frm.grid(row=0, column=0, sticky=NW)
        self.cluster_fit_btn.grid(row=1, column=0, sticky=NW)
        self.cluster_transform_btn.grid(row=2, column=0, sticky=NW)

        self.visualization_frm = LabelFrame(self.visualization_tab, text='VISUALIZATIONS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.grid_search_visualization_btn = Button(self.visualization_frm, text='GRID-SEARCH VISUALIZATION', fg='blue', command= lambda: GridSearchVisualizerPopUp(config_path=self.config_path))
        self.cluster_visualizer = Button(self.visualization_frm, text='DATA VISUALIZERS', fg='red', command= lambda: ClusterVisualizerPopUp(config_path=self.config_path))
        self.visualization_frm.grid(row=0, column=0, sticky='NW')
        self.grid_search_visualization_btn.grid(row=0, column=0, sticky='NW')
        self.cluster_visualizer.grid(row=1, column=0, sticky='NW')

        self.metrics_frm = LabelFrame(self.metrics_tab, text='METRICS', pady=5, padx=5, font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.dbcv_btn = Button(self.metrics_frm, text='DENSITY-BASED CLUSTER VALIDATION', fg='blue', command=lambda: DBCVPopUp(config_path=self.config_path))
        self.extract_single_metrics_btn = Button(self.metrics_frm, text='EXTRACT DATA', fg='red', command=lambda: DataExtractorPopUp(config_path=self.config_path))
        self.cluster_descriptives_btn = Button(self.metrics_frm, text='CLUSTER FREQUENTIST STATISTICS', fg='green', command=lambda: ClusterFrequentistStatisticsPopUp(config_path=self.config_path))
        self.cluster_xai_btn = Button(self.metrics_frm, text='CLUSTER XAI STATISTICS', fg='blue', command=lambda: ClusterXAIPopUp(config_path=self.config_path))
        self.embedding_corr_btn = Button(self.metrics_frm, text='EMBEDDING CORRELATIONS', fg='orange', command=lambda: EmbedderCorrelationsPopUp(config_path=self.config_path))
        self.print_embedding_info_btn = Button(self.metrics_frm, text='PRINT MODEL INFO', fg='black', command=lambda: PrintEmBeddingInfoPopUp(config_path=self.config_path))

        self.metrics_frm.grid(row=0, column=0, sticky='NW')
        self.dbcv_btn.grid(row=0, column=0, sticky='NW')
        self.extract_single_metrics_btn.grid(row=1, column=0, sticky='NW')
        self.cluster_descriptives_btn.grid(row=2, column=0, sticky='NW')
        self.cluster_xai_btn.grid(row=3, column=0, sticky='NW')
        self.embedding_corr_btn.grid(row=4, column=0, sticky='NW')
        self.print_embedding_info_btn.grid(row=5, column=0, sticky='NW')

        #self.main_frm.mainloop()

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
        if data_slice_type is Unsupervised.USER_DEFINED_SET.value:
            feature_file_path = self.feature_file_selected.file_path

        settings = {'data_slice': data_slice_type,
                    'clf_slice': classifier_slice_type,
                    'bout_aggregation_type': bout_selection,
                    'min_bout_length': bout_length,
                    'feature_file_path': feature_file_path}
        _ = DatasetCreator(settings=settings, config_path=self.config_path)


