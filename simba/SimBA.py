__author__ = "Simon Nilsson", "JJ Choong"

import os.path
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
from tkinter.filedialog import askopenfilename,askdirectory
from PIL import ImageTk
import PIL.Image
import tkinter.ttk as ttk
from tkinter.messagebox import askyesno
import webbrowser
from simba.tkinter_functions import (DropDownMenu,
                                     Entry_Box,
                                     FileSelect)
from simba.plotting.interactive_probability_grapher import InteractiveProbabilityGrapher
from simba.Validate_model_one_video_run_clf import ValidateModelRunClf
from simba.cue_light_tools.cue_light_menues import CueLightAnalyzerMenu
from simba.machine_model_settings_pop_up import MachineModelSettingsPopUp
from simba.outlier_tools.outlier_corrector_movement import OutlierCorrecterMovement
from simba.outlier_tools.outlier_corrector_location import OutlierCorrecterLocation
from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper
from simba.third_party_label_appenders.BENTO_appender import BentoAppender
from simba.third_party_label_appenders.BORIS_appender import BorisAppender
from simba.third_party_label_appenders.solomon_importer import SolomonImporter
from simba.third_party_label_appenders.ethovision_import import ImportEthovision
from simba.third_party_label_appenders.deepethogram_importer import DeepEthogramImporter
from simba.third_party_label_appenders.observer_importer import NoldusObserverImporter
from simba.Directing_animals_analyzer import DirectingOtherAnimalsAnalyzer
from simba.pose_importers.import_trk import *
from simba.train_single_model import TrainSingleModel
from simba.train_mutiple_models import TrainMultipleModelsFromMeta
from simba.run_model_new import RunModel
import urllib.request
import threading
import atexit
from simba.video_info_table import VideoInfoTable
from simba.read_config_unit_tests import check_int
from simba.roi_tools.ROI_define import *
from simba.roi_tools.ROI_menus import *
from simba.roi_tools.ROI_reset import *
from simba.misc_tools import (run_user_defined_feature_extraction_class,
                              extract_frames_from_all_videos_in_directory)
from simba.video_processing import (video_to_greyscale,
                                    superimpose_frame_count)
from simba.setting_menu import SettingsMenu
from simba.pop_up_classes import (HeatmapLocationPopup,
                                  QuickLineplotPopup,
                                  ClfByROIPopUp,
                                  FSTTCPopUp,
                                  KleinbergPopUp,
                                  TimeBinsClfPopUp,
                                  ClfDescriptiveStatsPopUp,
                                  DownsampleVideoPopUp,
                                  CLAHEPopUp,
                                  CropVideoPopUp,
                                  ClipVideoPopUp,
                                  MultiShortenPopUp,
                                  ChangeImageFormatPopUp,
                                  ConvertVideoPopUp,
                                  ExtractSpecificFramesPopUp,
                                  ExtractAllFramesPopUp,
                                  Csv2ParquetPopUp,
                                  Parquet2CsvPopUp,
                                  MultiCropPopUp,
                                  ChangeFpsSingleVideoPopUp,
                                  ChangeFpsMultipleVideosPopUp,
                                  ExtractSEQFramesPopUp,
                                  MergeFrames2VideoPopUp,
                                  PrintModelInfoPopUp,
                                  CreateGIFPopUP,
                                  CalculatePixelsPerMMInVideoPopUp,
                                  PoseReorganizerPopUp,
                                  MakePathPlotPopUp,
                                  AboutSimBAPopUp,
                                  ConcatenatingVideosPopUp,
                                  VisualizePoseInFolderPopUp,
                                  DropTrackingDataPopUp,
                                  ConcatenatorPopUp,
                                  SetMachineModelParameters,
                                  OutlierSettingsPopUp,
                                  RemoveAClassifierPopUp,
                                  VisualizeROIFeaturesPopUp,
                                  VisualizeROITrackingPopUp,
                                  SklearnVisualizationPopUp,
                                  GanttPlotPopUp,
                                  VisualizeClassificationProbabilityPopUp,
                                  PathPlotPopUp,
                                  DistancePlotterPopUp,
                                  HeatmapClfPopUp,
                                  DataPlotterPopUp,
                                  DirectingOtherAnimalsVisualizerPopUp,
                                  PupRetrievalPopUp,
                                  ClassifierValidationPopUp,
                                  AnalyzeSeverityPopUp,
                                  ImportFrameDirectoryPopUp,
                                  AddClfPopUp,
                                  ArchiveProcessedFilesPopUp,
                                  InterpolatePopUp,
                                  SmoothingPopUp,
                                  BatchPreProcessPopUp,
                                  AppendROIFeaturesByBodyPartPopUp,
                                  ExtractAnnotationFramesPopUp,
                                  FeatureSubsetExtractorPopUp,
                                  ThirdPartyAnnotatorAppenderPopUp,
                                  ValidationVideoPopUp)
from simba.bounding_box_tools.boundary_menus import BoundaryMenus
from simba.labelling_interface import select_labelling_video
from simba.labelling_advanced_interface import select_labelling_video_advanced
from simba.enums import (Formats,
                         OS,
                         Defaults)
from simba.utils.lookups import get_bp_config_code_class_pairs, get_icons_paths
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.create_project_pop_up import ProjectCreatorPopUp
from simba.plotly_create_h5 import create_plotly_container
#from simba.unsupervised.unsupervised_ui import UnsupervisedGUI
import sys
import subprocess

sys.setrecursionlimit(10 ** 6)
currentPlatform = platform.system()

class LoadProjectPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 200)
        main_frm.wm_title("Load SimBA project (project_config.ini file)")
        load_project_frm = LabelFrame(main_frm, text='LOAD SIMBA PROJECT_CONFIG.INI', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        self.selected_file = FileSelect(load_project_frm,'Select file: ', title='Select project_config.ini file')
        load_project_btn = Button(load_project_frm, text='LOAD PROJECT', font=("Helvetica", 12, 'bold'), command=lambda: self.launch_project())

        load_project_frm.grid(row=0)
        self.selected_file.grid(row=0,sticky=NW)
        load_project_btn.grid(row=1,pady=10, sticky=NW)

    def launch_project(self):
        print('Loading {}...'.format(self.selected_file.file_path))
        check_file_exist_and_readable(file_path=self.selected_file.file_path)
        _ = SimbaProjectPopUp(config_path=self.selected_file.file_path)

def wait_for_internet_connection(url):
    while True:
        try:
            response = urllib.request.urlopen(url, timeout=1)
            return
        except:
            pass

class SimbaProjectPopUp(ConfigReader, PopUpMixin):
    def __init__(self,
                 config_path: str):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        simongui = Toplevel()
        simongui.minsize(1300, 800)
        simongui.wm_title("LOAD PROJECT")
        simongui.columnconfigure(0, weight=1)
        simongui.rowconfigure(0, weight=1)

        self.btn_icons = get_icons_paths()
        for k in self.btn_icons.keys():
            self.btn_icons[k]['img'] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), self.btn_icons[k]['icon_path'])))

        tab_parent = ttk.Notebook(hxtScrollbar(simongui))

        tab2 = ttk.Frame(tab_parent)
        tab3 = ttk.Frame(tab_parent)
        tab4 = ttk.Frame(tab_parent)
        tab5 = ttk.Frame(tab_parent)
        tab6 = ttk.Frame(tab_parent)
        tab7 = ttk.Frame(tab_parent)
        tab8 = ttk.Frame(tab_parent)
        tab9 = ttk.Frame(tab_parent)
        tab10 = ttk.Frame(tab_parent)
        tab11 = ttk.Frame(tab_parent)


        tab_parent.add(tab2, text= f"{'[ Further imports (data/video/frames) ]':20s}", compound='left', image=self.btn_icons['pose']['img'])
        tab_parent.add(tab3, text=f"{'[ Video parameters ]':20s}", compound='left', image=self.btn_icons['calipher']['img'])
        tab_parent.add(tab4, text=f"{'[ Outlier correction ]':20s}", compound='left', image=self.btn_icons['outlier']['img'])
        tab_parent.add(tab6, text=f"{'[ ROI ]':10s}", compound='left', image=self.btn_icons['roi']['img'])
        tab_parent.add(tab5, text=f"{'[ Extract features ]':20s}", compound='left', image=self.btn_icons['features']['img'])
        tab_parent.add(tab7, text=f"{'[ Label behavior] ':20s}", compound='left', image=self.btn_icons['label']['img'])
        tab_parent.add(tab8, text=f"{'[ Train machine model ]':20s}", compound='left', image=self.btn_icons['clf']['img'])
        tab_parent.add(tab9, text=f"{'[ Run machine model ]':20s}", compound='left', image=self.btn_icons['clf_2']['img'])
        tab_parent.add(tab10, text=f"{'[ Visualizations ]':20s}", compound='left', image=self.btn_icons['visualize']['img'])
        tab_parent.add(tab11,text=f"{'[ Add-ons ]':20s}", compound='left', image=self.btn_icons['add_on']['img'])

        tab_parent.grid(row=0)
        tab_parent.enable_traversal()

        import_frm = LabelFrame(tab2)
        import_frm.grid(row=0, column=0, sticky=NW)

        further_methods_frm = LabelFrame(import_frm, text='FURTHER METHODS', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        extract_frm_btn = Button(further_methods_frm, text='EXTRACT FRAMES FOR ALL VIDEOS IN SIMBA PROJECT', fg='blue', command= lambda: extract_frames_from_all_videos_in_directory(config_path=self.config_path, directory=self.video_dir))
        import_frm_dir_btn = Button(further_methods_frm,text='IMPORT FRAMES DIRECTORY TO SIMBA PROJECT', fg='blue', command=lambda: ImportFrameDirectoryPopUp(config_path=self.config_path))
        add_clf_btn = Button(further_methods_frm, text='ADD CLASSIFIER TO SIMBA PROJECT', fg='blue', command=lambda: AddClfPopUp(config_path=self.config_path))
        remove_clf_btn = Button(further_methods_frm, text='REMOVE CLASSIFIER FROM SIMBA PROJECT', fg='blue', command=lambda: RemoveAClassifierPopUp(config_path=self.config_path))
        archive_files_btn = Button(further_methods_frm, text='ARCHIVE PROCESSED FILES IN SIMBA PROJECT', fg='blue', command=lambda: ArchiveProcessedFilesPopUp(config_path=self.config_path))
        reverse_btn = Button(further_methods_frm, text='REVERSE TRACKING IDENTITIES IN SIMBA PROJECT',fg='blue', command=lambda: self.reverseid())
        interpolate_btn = Button(further_methods_frm, text='INTERPOLATE POSE IN SIMBA PROJECT', fg='blue', command=lambda: InterpolatePopUp(config_path=self.config_path))
        smooth_btn = Button(further_methods_frm, text='SMOOTH POSE IN SIMBA PROJECT', fg='blue', command=lambda: SmoothingPopUp(config_path=self.config_path))

        label_setscale = CreateLabelFrameWithIcon(parent=tab3, header='VIDEO PARAMETERS (FPS, RESOLUTION, PPX/MM ....)', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_PARAMETERS.value)
        #label_setscale = LabelFrame(tab3,text='VIDEO PARAMETERS (FPS, RESOLUTION, PPX/MM ....)', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5,padx=5,fg='black')
        self.distance_in_mm_eb = Entry_Box(label_setscale, 'KNOWN DISTANCE (MILLIMETERS)', '25', validation='numeric')
        button_setdistanceinmm = Button(label_setscale, text='AUTO-POPULATE', fg='green', command=lambda: self.set_distance_mm())

        button_setscale = Button(label_setscale, text='CONFIGURE VIDEO PARAMETERS', compound='left', image=self.btn_icons['calipher']['img'], relief=RAISED, fg='blue', command=lambda: self.create_video_info_table())
        button_setscale.image = self.btn_icons['calipher']['img']

        self.new_ROI_frm = LabelFrame(tab6, text='SIMBA ROI INTERFACE', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.start_new_ROI = Button(self.new_ROI_frm, text='DEFINE ROIs', fg='green', compound='left', image=self.btn_icons['roi']['img'], relief=RAISED, command= lambda: ROI_menu(self.config_path))
        self.start_new_ROI.image = self.btn_icons['roi']['img']

        self.delete_all_ROIs = Button(self.new_ROI_frm, text='DELETE ALL ROI DEFINITIONS', fg='red', compound='left', image=self.btn_icons['trash']['img'], command=lambda: delete_all_ROIs(self.config_path))
        self.delete_all_ROIs.image = self.btn_icons['trash']['img']

        self.tutorial_link = Label(self.new_ROI_frm, text='[Link to ROI user-guide]', cursor='hand2',  font='Verdana 8 underline')
        self.tutorial_link.bind("<Button-1>", lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md'))

        self.new_ROI_frm.grid(row=0, sticky=NW)
        self.start_new_ROI.grid(row=0, sticky=NW)
        self.delete_all_ROIs.grid(row=1, column=0, sticky=NW)
        self.tutorial_link.grid(row=2, sticky=W)

        self.roi_draw = LabelFrame(tab6, text='ANALYZE ROI DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        analyze_roi_btn = Button(self.roi_draw, text='ANALYZE ROI DATA: AGGREGATES', fg='green', command=lambda: SettingsMenu(config_path=self.config_path, title='ROI ANALYSIS'))
        analyze_roi_time_bins_btn = Button(self.roi_draw, text='ANALYZE ROI DATA: TIME-BINS',  fg='blue', command=lambda: SettingsMenu(config_path=self.config_path, title='TIME BINS: ANALYZE ROI'))

        self.roi_draw.grid(row=0, column=1, sticky=N)
        analyze_roi_btn.grid(row=0, sticky='NW')
        analyze_roi_time_bins_btn.grid(row=1, sticky='NW')

        ###plot roi
        self.roi_draw1 = LabelFrame(tab6, text='VISUALIZE ROI DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value)

        # button
        visualizeROI = Button(self.roi_draw1, text='VISUALIZE ROI TRACKING', fg='green', command= lambda: VisualizeROITrackingPopUp(config_path=self.config_path))
        visualizeROIfeature = Button(self.roi_draw1, text='VISUALIZE ROI FEATURES', fg='blue', command= lambda: VisualizeROIFeaturesPopUp(config_path=self.config_path))

        ##organize
        self.roi_draw1.grid(row=0, column=2, sticky=N)
        visualizeROI.grid(row=0, sticky='NW')
        visualizeROIfeature.grid(row=1, sticky='NW')

        processmovementdupLabel = LabelFrame(tab6,text='OTHER ANALYSES / VISUALIZATIONS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        analyze_distances_velocity_btn = Button(processmovementdupLabel, text='ANALYZE DISTANCES / VELOCITY: AGGREGATES', fg='green', command=lambda: SettingsMenu(config_path=self.config_path, title='ANALYZE MOVEMENT'))
        analyze_distances_velocity_timebins_btn = Button(processmovementdupLabel, text='ANALYZE DISTANCES / VELOCITY: TIME-BINS', fg='blue', command=lambda: SettingsMenu(config_path=self.config_path, title='TIME BINS: DISTANCE/VELOCITY'))

        heatmaps_location_button = Button(processmovementdupLabel,text='CREATE LOCATION HEATMAPS', fg='red', command=lambda: HeatmapLocationPopup(config_path=self.config_path))

        button_lineplot = Button(processmovementdupLabel, text='CREATE PATH PLOTS', fg='orange', command=lambda: QuickLineplotPopup(config_path=self.config_path))
        button_analyzeDirection = Button(processmovementdupLabel,text='ANALYZE DIRECTIONALITY BETWEEN ANIMALS', fg='pink', command =lambda: self.directing_other_animals_analysis())
        button_visualizeDirection = Button(processmovementdupLabel,text='VISUALIZE DIRECTIONALITY BETWEEN ANIMALS', fg='brown', command=lambda:self.directing_other_animals_visualizer())

        #organize
        processmovementdupLabel.grid(row=0,column=3,sticky=NW)
        analyze_distances_velocity_btn.grid(row=0, sticky=NW)
        heatmaps_location_button.grid(row=1, sticky=NW)
        analyze_distances_velocity_timebins_btn.grid(row=2, sticky=NW)
        button_lineplot.grid(row=3, sticky=NW)
        button_analyzeDirection.grid(row=4, sticky=NW)
        button_visualizeDirection.grid(row=5, sticky=NW)


        label_outliercorrection = CreateLabelFrameWithIcon(parent=tab4, header='OUTLIER CORRECTION', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OUTLIERS_DOC.value)
        #label_outliercorrection = LabelFrame(tab4,text='OUTLIER CORRECTION',font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg='black')
        button_settings_outlier = Button(label_outliercorrection,text='SETTINGS', fg='blue', command = lambda: OutlierSettingsPopUp(config_path=self.config_path))
        button_outliercorrection = Button(label_outliercorrection,text='RUN OUTLIER CORRECTION', fg='green', command=lambda:self.correct_outlier())
        button_skipOC = Button(label_outliercorrection,text='SKIP OUTLIER CORRECTION (CAUTION)',fg='red', command=lambda: self.initiate_skip_outlier_correction())


        #extract features
        label_extractfeatures = CreateLabelFrameWithIcon(parent=tab5, header='EXTRACT FEATURES', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.EXTRACT_FEATURES.value)
        #label_extractfeatures = LabelFrame(tab5,text='EXTRACT FEATURES',font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg='black')
        button_extractfeatures = Button(label_extractfeatures,text='EXTRACT FEATURES', fg='blue', command=lambda: threading.Thread(target=self.run_feature_extraction).start())

        def activate(box, *args):
            for entry in args:
                if box.get() == 0:
                    entry.configure(state='disabled')
                elif box.get() == 1:
                    entry.configure(state='normal')

        labelframe_usrdef = LabelFrame(label_extractfeatures)
        self.scriptfile = FileSelect(labelframe_usrdef, 'Script path')
        self.scriptfile.btnFind.configure(state='disabled')
        self.user_defined_var = BooleanVar(value=False)
        userscript = Checkbutton(labelframe_usrdef,text='Apply user-defined feature extraction script',variable=self.user_defined_var, command=lambda: activate(self.user_defined_var, self.scriptfile.btnFind))

        roi_feature_frm = CreateLabelFrameWithIcon(parent=tab5, header='APPEND ROI FEATURES', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.APPEND_ROI_FEATURES.value)
        #roi_feature_frm = LabelFrame(tab5, text='APPEND ROI FEATURES', pady=5, font=Formats.LABELFRAME_HEADER_FORMAT.value)
        append_roi_features_by_animal = Button(roi_feature_frm, text='APPEND ROI DATA TO FEATURES: BY ANIMAL (CAUTION)', fg='red',command=lambda: SettingsMenu(config_path=self.config_path, title='APPEND ROI FEATURES'))
        append_roi_features_by_body_part = Button(roi_feature_frm, text='APPEND ROI DATA TO FEATURES: BY BODY-PARTS (CAUTION)', fg='orange',command=lambda: AppendROIFeaturesByBodyPartPopUp(config_path=self.config_path))

        feature_tools_frm = LabelFrame(tab5, text='FEATURE TOOLS', pady=5, font=Formats.LABELFRAME_HEADER_FORMAT.value)
        compute_feature_subset_btn = Button(feature_tools_frm, text='CALCULATE FEATURE SUBSETS', fg='blue',command=lambda: FeatureSubsetExtractorPopUp(config_path=self.config_path))

        label_behavior_frm = CreateLabelFrameWithIcon(parent=tab7, header='LABEL BEHAVIOR', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.LABEL_BEHAVIOR.value)
        #label_behavior_frm = LabelFrame(tab7,text='LABEL BEHAVIOR',font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg='black')
        select_video_btn_new = Button(label_behavior_frm, text='Select video (create new video annotation)',command= lambda:select_labelling_video(config_path=self.config_path,
                                                                                                                                                   threshold_dict=None,
                                                                                                                                                   setting='from_scratch',
                                                                                                                                                   continuing=False))
        select_video_btn_continue = Button(label_behavior_frm, text='Select video (continue existing video annotation)', command= lambda:select_labelling_video(config_path=self.config_path,
                                                                                                                                                   threshold_dict=None,                                                                                                                             setting=None,
                                                                                                                                                   continuing=True))
        label_thirdpartyann = CreateLabelFrameWithIcon(parent=tab7, header='IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.THIRD_PARTY_ANNOTATION.value)
        #label_thirdpartyann = LabelFrame(tab7,text='IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS',font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg='black')
        button_importmars = Button(label_thirdpartyann,text='Import MARS Annotation (select folder with .annot files)', fg='blue', command=self.importMARS)
        button_importboris = Button(label_thirdpartyann,text='Import BORIS Annotation (select folder with .csv files)', fg='green', command=self.importBoris)
        button_importsolomon = Button(label_thirdpartyann,text='Import SOLOMON Annotation (select folder with .csv files', fg='purple', command=self.importSolomon)
        button_importethovision = Button(label_thirdpartyann, text='Import ETHOVISION Annotation (select folder with .xls/xlsx files)', fg='blue', command=self.import_ethovision)
        button_importdeepethogram = Button(label_thirdpartyann,text='Import DEEPETHOGRAM Annotation (select folder with .csv files)', fg='green', command=self.import_deepethogram)
        import_observer_btn = Button(label_thirdpartyann, text='Import NOLDUS OBSERVER Annotation (select folder with .xls/xlsx files)', fg='purple', command=self.import_noldus_observer)



        label_pseudo = CreateLabelFrameWithIcon(parent=tab7, header='PSEUDO-LABELLING', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.PSEUDO_LBL.value)
        #label_pseudo = LabelFrame(tab7,text='PSEUDO-LABELLING',font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        pseudo_intructions_lbl_1 = Label(label_pseudo,text='Note that SimBA pseudo-labelling require initial machine predictions.')
        pseudo_intructions_lbl_2 = Label(label_pseudo, text='Click here more information on how to use the SimBA pseudo-labelling interface.', cursor="hand2", fg="blue")
        pseudo_intructions_lbl_2.bind("<Button-1>", lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md'))
        pLabel_framedir = FileSelect(label_pseudo,'Video Path',lblwidth='10')
        plabelframe_threshold = LabelFrame(label_pseudo,text='Threshold',pady=5,padx=5)
        plabel_threshold =[0]*len(self.clf_names)
        for count, i in enumerate(list(self.clf_names)):
            plabel_threshold[count] = Entry_Box(plabelframe_threshold,str(i),'20')
            plabel_threshold[count].grid(row=count+2,sticky=W)

        pseudo_lbl_btn = Button(label_pseudo,text='Correct labels',command = lambda:select_labelling_video(config_path=self.config_path,
                                                                                                           threshold_dict=dict(zip(self.clf_names, plabel_threshold)),
                                                                                                           setting='pseudo',
                                                                                                           continuing=False,
                                                                                                           video_file_path=pLabel_framedir.file_path))

        label_adv_label = CreateLabelFrameWithIcon(parent=tab7, header='ADVANCED LABEL BEHAVIOR', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ADVANCED_LBL.value)
        #label_adv_label = LabelFrame(tab7,text='ADVANCED LABEL BEHAVIOR',font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        label_adv_note_1 = Label(label_adv_label,text='Note that you will have to specify the presence of *both* behavior and non-behavior on your own.')
        label_adv_note_2 = Label(label_adv_label, text='Click here more information on how to use the SimBA labelling interface.', cursor="hand2", fg="blue")
        label_adv_note_2.bind("<Button-1>", lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md'))
        adv_label_btn_new = Button(label_adv_label, text='Select video (create new video annotation)',command= lambda: select_labelling_video_advanced(config_path=self.config_path, continuing=False))
        adv_label_btn_continue = Button(label_adv_label, text='Select video (continue existing video annotation)',command=lambda: select_labelling_video_advanced(config_path=self.config_path, continuing=True))

        lbl_tools_frm = LabelFrame(tab7, text='LABELLING TOOLS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        visualize_annotation_img_btn = Button(lbl_tools_frm, text='Visualize annotations', fg='blue', command=lambda: ExtractAnnotationFramesPopUp(config_path=self.config_path))
        third_party_annotations_btn = Button(lbl_tools_frm, text='Append third-party annotations', fg='purple', command=lambda: ThirdPartyAnnotatorAppenderPopUp(config_path=self.config_path))


        label_trainmachinemodel = CreateLabelFrameWithIcon(parent=tab8, header='TRAIN MACHINE MODELS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.TRAIN_ML_MODEL.value)
        #label_trainmachinemodel = LabelFrame(tab8,text='TRAIN MACHINE MODELS',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        button_trainmachinesettings = Button(label_trainmachinemodel,text='SETTINGS',command=self.trainmachinemodelsetting)
        button_trainmachinemodel = Button(label_trainmachinemodel,text='TRAIN SINGLE MODEL (GLOBAL ENVIRONMENT)',fg='blue',command = lambda: threading.Thread(target=self.train_single_model(config_path=self.config_path)).start())
        button_train_multimodel = Button(label_trainmachinemodel, text='TRAIN MULTIPLE MODELS (ONE FOR EACH SAVED SETTING)',fg='green',command = lambda: threading.Thread(target=self.train_multiple_models_from_meta(config_path=self.config_path)).start())


        label_model_validation = CreateLabelFrameWithIcon(parent=tab9, header='VALIDATE MODEL ON SINGLE VIDEO', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value)
        #label_model_validation = LabelFrame(tab9, text='VALIDATE MODEL ON SINGLE VIDEO', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='blue')
        self.csvfile = FileSelect(label_model_validation, 'SELECT DATA FEATURE FILE', color='blue', lblwidth=30)
        self.modelfile = FileSelect(label_model_validation, 'SELECT MODEL FILE', color='blue', lblwidth=30)
        button_runvalidmodel = Button(label_model_validation, text='RUN MODEL', fg='blue', command=lambda: self.validate_model_first_step())

        button_generateplot = Button(label_model_validation, text="INTERACTIVE PROBABILITY PLOT", fg='blue', command= lambda: self.launch_interactive_plot())
        self.dis_threshold = Entry_Box(label_model_validation, 'DISCRIMINATION THRESHOLD (0.0-1.0):', '30')
        self.min_behaviorbout = Entry_Box(label_model_validation, 'MINIMUM BOUT LENGTH (MS):', '30', validation='numeric')
        button_validate_model = Button(label_model_validation, text='CREATE VALIDATION VIDEO', fg='blue', command= lambda: ValidationVideoPopUp(config_path=config_path, simba_main_frm=self))

        label_runmachinemodel = LabelFrame(tab9,text='RUN MACHINE MODEL',font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        button_run_rfmodelsettings = Button(label_runmachinemodel,text='MODEL SETTINGS', fg='green', compound='left', image=self.btn_icons['settings']['img'], command= lambda: SetMachineModelParameters(config_path=self.config_path))
        button_run_rfmodelsettings.image = self.btn_icons['settings']['img']

        button_runmachinemodel = Button(label_runmachinemodel,text='RUN MODELS', fg='green', compound='left', image=self.btn_icons['clf']['img'], command=self.runrfmodel)
        button_runmachinemodel.image = self.btn_icons['clf']['img']

        kleinberg_button = Button(label_runmachinemodel,text='KLEINBERG SMOOTHING', fg='green', command=lambda: KleinbergPopUp(config_path=self.config_path))
        fsttc_button = Button(label_runmachinemodel,text='FSTTC', fg='green', command=lambda:FSTTCPopUp(config_path=self.config_path))

        label_machineresults = CreateLabelFrameWithIcon(parent=tab9, header='ANALYZE MACHINE RESULTS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        #label_machineresults = LabelFrame(tab9,text='ANALYZE MACHINE RESULTS',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        button_process_datalog = Button(label_machineresults, text='ANALYZE MACHINE PREDICTIONS: AGGREGATES', fg='blue', command=lambda: ClfDescriptiveStatsPopUp(config_path=self.config_path))
        button_process_movement = Button(label_machineresults, text='ANALYZE DISTANCES/VELOCITY: AGGREGATES', fg='blue', command=lambda: SettingsMenu(config_path=self.config_path, title='ANALYZE MOVEMENT'))
        button_movebins = Button(label_machineresults, text='ANALYZE DISTANCES/VELOCITY: TIME BINS', fg='blue', command=lambda: SettingsMenu(config_path=self.config_path, title='TIME BINS: DISTANCE/VELOCITY'))
        button_classifierbins = Button(label_machineresults,text='ANALYZE MACHINE PREDICTIONS: TIME-BINS', fg='blue', command=lambda: TimeBinsClfPopUp(config_path=self.config_path))
        button_classifier_ROI = Button(label_machineresults, text='ANALYZE MACHINE PREDICTION: BY ROI', fg='blue', command=lambda: ClfByROIPopUp(config_path=self.config_path))
        button_severity = Button(label_machineresults, text='ANALYZE MACHINE PREDICTION: BY SEVERITY', fg='blue', command=lambda: AnalyzeSeverityPopUp(config_path=self.config_path))


        visualization_frm = CreateLabelFrameWithIcon(parent=tab10, header='DATA VISUALIZATIONS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VISUALIZATION.value)
        #visualization_frm = LabelFrame(tab10,text='DATA VISUALIZATIONS',font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg='black')
        sklearn_visualization_btn = Button(visualization_frm,text='VISUALIZE CLASSIFICATIONS', fg='black',command= lambda:SklearnVisualizationPopUp(config_path=self.config_path))
        sklearn_visualization_btn.grid(row=0, column=0, sticky=NW)

        gantt_visualization_btn = Button(visualization_frm,text='VISUALIZE GANTT', fg='blue', command=lambda:GanttPlotPopUp(config_path=self.config_path))
        gantt_visualization_btn.grid(row=1, column=0, sticky=NW)

        probability_visualization_btn = Button(visualization_frm,text='VISUALIZE PROBABILITIES', fg='green', command=lambda:VisualizeClassificationProbabilityPopUp(config_path=self.config_path))
        probability_visualization_btn.grid(row=2, column=0, sticky=NW)

        path_visualization_btn = Button(visualization_frm, text='VISUALIZE PATHS', fg='orange', command=lambda: PathPlotPopUp(config_path=self.config_path))
        path_visualization_btn.grid(row=3, column=0, sticky=NW)

        distance_visualization_btn = Button(visualization_frm, text='VISUALIZE DISTANCES', fg='red', command=lambda: DistancePlotterPopUp(config_path=self.config_path))
        distance_visualization_btn.grid(row=4, column=0, sticky=NW)

        heatmap_clf_visualization_btn = Button(visualization_frm, text='VISUALIZE CLASSIFICATION HEATMAPS', fg='pink', command=lambda: HeatmapClfPopUp(config_path=self.config_path))
        heatmap_clf_visualization_btn.grid(row=5, column=0, sticky=NW)

        data_plot_visualization_btn = Button(visualization_frm, text='VISUALIZE DATA PLOTS', fg='purple',command=lambda: DataPlotterPopUp(config_path=self.config_path))
        data_plot_visualization_btn.grid(row=6, column=0, sticky=NW)

        clf_validation_btn = Button(visualization_frm, text='CLASSIFIER VALIDATION CLIPS', fg='blue', command=lambda: ClassifierValidationPopUp(config_path=self.config_path))
        clf_validation_btn.grid(row=7, column=0, sticky=NW)

        merge_frm = CreateLabelFrameWithIcon(parent=tab10, header='MERGE FRAMES', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CONCAT_VIDEOS.value)
        #merge_frm = LabelFrame(tab10, text='MERGE FRAMES', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='black')
        merge_frm_btn = Button(merge_frm, text='MERGE FRAMES', fg='black', command=lambda:ConcatenatorPopUp(config_path=self.config_path))

        plotlyInterface = CreateLabelFrameWithIcon(parent=tab10, header='PLOTLY / DASH', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.PLOTLY.value)
        #plotlyInterface = LabelFrame(tab10, text= 'PLOTLY / DASH', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        plotlyInterfaceTitles = ['Sklearn results', 'Time bin analyses', 'Probabilities', 'Severity analysis']
        toIncludeVar = []
        for i in range(len(plotlyInterfaceTitles)+1):
            toIncludeVar.append(IntVar())
        plotlyCheckbox = [0] * (len(plotlyInterfaceTitles)+1)
        for i in range(len(plotlyInterfaceTitles)):
            plotlyCheckbox[i] = Checkbutton(plotlyInterface, text=plotlyInterfaceTitles[i], variable=toIncludeVar[i])
            plotlyCheckbox[i].grid(row=i, sticky=W)

        button_save_plotly_file = Button(plotlyInterface, text='Save SimBA / Plotly dataset', command=lambda: self.generateSimBPlotlyFile(toIncludeVar))
        self.plotly_file = FileSelect(plotlyInterface, 'SimBA Dashboard file (H5)', title='Select SimBA/Plotly dataset (h5)')
        self.groups_file = FileSelect(plotlyInterface, 'SimBA Groups file (CSV)', title='Select groups file (csv')
        button_open_plotly_interface = Button(plotlyInterface, text='Open SimBA / Plotly dataset', fg='black', command=lambda: [self.open_plotly_interface('http://127.0.0.1:8050')])

        #addons
        lbl_addon = LabelFrame(tab11,text='SimBA Expansions',pady=5, padx=5,font=Formats.LABELFRAME_HEADER_FORMAT.value,fg='black')
        button_bel = Button(lbl_addon,text='Pup retrieval - Analysis Protocol 1', fg='blue',command=lambda: PupRetrievalPopUp(config_path=self.config_path))
        cue_light_analyser_btn = Button(lbl_addon, text='Cue light analysis', fg='red', command=lambda: CueLightAnalyzerMenu(config_path=self.config_path))
        anchored_roi_analysis_btn = Button(lbl_addon, text='Animal-anchored ROI analysis', fg='orange', command=lambda: BoundaryMenus(config_path=self.config_path))
        #unsupervised_btn = Button(lbl_addon,text='Unsupervised analysis', fg='purple', command=lambda: UnsupervisedGUI(config_path=self.config_path))



        self.create_import_videos_menu(parent_frm=import_frm, idx_row=0, idx_column=0)
        self.create_import_pose_menu(parent_frm=import_frm, idx_row=1, idx_column=0)
        further_methods_frm.grid(row=0, column=1, sticky=NW,pady=5,padx=5)
        extract_frm_btn.grid(row=1, column=0, sticky=NW)
        import_frm_dir_btn.grid(row=2, column=0, sticky=NW)
        add_clf_btn.grid(row=3, column=0, sticky=NW)
        remove_clf_btn.grid(row=4,column=0, sticky=NW)
        archive_files_btn.grid(row=5,column=0, sticky=NW)
        reverse_btn.grid(row=6, column=0, sticky=NW)
        interpolate_btn.grid(row=7, column=0, sticky=NW)
        smooth_btn.grid(row=8, column=0, sticky=NW)

        label_setscale.grid(row=0, sticky=NW,pady=5,padx=5)
        self.distance_in_mm_eb.grid(row=0,column=0,sticky=NW)
        button_setdistanceinmm.grid(row=0, column=1, sticky=NW)
        button_setscale.grid(row=1,column=0,sticky=NW)

        label_outliercorrection.grid(row=0,sticky=W)
        button_settings_outlier.grid(row=0,sticky=W)
        button_outliercorrection.grid(row=1,sticky=W)
        button_skipOC.grid(row=2,sticky=W,pady=5)

        label_extractfeatures.grid(row=0, column=0, sticky=NW)
        button_extractfeatures.grid(row=0,column=0, sticky=NW)
        labelframe_usrdef.grid(row=1, column=0, sticky=NW, pady=5)
        userscript.grid(row=1, column=0, sticky=NW)
        self.scriptfile.grid(row=2, column=0, sticky=NW)

        roi_feature_frm.grid(row=1, column=0, sticky=NW)
        append_roi_features_by_animal.grid(row=0, column=0, sticky=NW)
        append_roi_features_by_body_part.grid(row=1, column=0, sticky=NW)

        feature_tools_frm.grid(row=2, column=0, sticky=NW)
        compute_feature_subset_btn.grid(row=0, column=0, sticky=NW)


        label_behavior_frm.grid(row=5,sticky=W)
        select_video_btn_new.grid(row=0,sticky=W)
        select_video_btn_continue.grid(row=1,sticky=W)

        label_pseudo.grid(row=6,sticky=W,pady=10)

        pLabel_framedir.grid(row=0,sticky=W)
        pseudo_intructions_lbl_1.grid(row=1,sticky=W)
        pseudo_intructions_lbl_2.grid(row=2, sticky=W)
        plabelframe_threshold.grid(row=4,sticky=W)
        pseudo_lbl_btn.grid(row=5,sticky=W)

        label_adv_label.grid(row=7, column=0, sticky=NW)
        label_adv_note_1.grid(row=0, column=0, sticky=NW)
        label_adv_note_2.grid(row=1, column=0, sticky=NW)
        adv_label_btn_new.grid(row=3, column=0, sticky=NW)
        adv_label_btn_continue.grid(row=4, column=0, sticky=NW)

        label_thirdpartyann.grid(row=8, sticky=W)
        button_importmars.grid(row=0, column=0, sticky=NW)
        button_importboris.grid(row=1, column=0, sticky=NW)
        button_importsolomon.grid(row=2, column=0, sticky=NW)
        button_importethovision.grid(row=0, column=1, sticky=NW)
        button_importdeepethogram.grid(row=1, column=1, sticky=NW)
        import_observer_btn.grid(row=2, column=1, sticky=NW)

        lbl_tools_frm.grid(row=9, column=0, sticky=NW)
        visualize_annotation_img_btn.grid(row=0, column=0, sticky=NW)
        third_party_annotations_btn.grid(row=1, column=0, sticky=NW)

        label_trainmachinemodel.grid(row=6,sticky=W)
        button_trainmachinesettings.grid(row=0,column=0,sticky=NW,padx=5)
        button_trainmachinemodel.grid(row=1,column=0,sticky=NW,padx=5)
        button_train_multimodel.grid(row=2,column=0,sticky=NW,padx=5)

        label_model_validation.grid(row=7, sticky=W, pady=5)
        self.csvfile.grid(row=0, sticky=W)
        self.modelfile.grid(row=1, sticky=W)
        button_runvalidmodel.grid(row=2, sticky=W)
        button_generateplot.grid(row=3, sticky=W)
        self.dis_threshold.grid(row=4, sticky=W)
        self.min_behaviorbout.grid(row=5, sticky=W)
        button_validate_model.grid(row=6, sticky=W)

        label_runmachinemodel.grid(row=8,sticky=NW)
        button_run_rfmodelsettings.grid(row=0,sticky=NW)
        button_runmachinemodel.grid(row=1,sticky=NW)
        kleinberg_button.grid(row=2,sticky=NW)
        fsttc_button.grid(row=3,sticky=NW)

        label_machineresults.grid(row=9,sticky=W,pady=5)
        button_process_datalog.grid(row=2,column=0,sticky=W,padx=3)
        button_process_movement.grid(row=2,column=1,sticky=W,padx=3)
        button_movebins.grid(row=3,column=1,sticky=W,padx=3)
        button_classifierbins.grid(row=3,column=0,sticky=W,padx=3)
        button_classifier_ROI.grid(row=4, column=0, sticky=W, padx=3)
        button_severity.grid(row=4, column=1, sticky=W, padx=3)

        visualization_frm.grid(row=11,column=0,sticky=W+N,padx=5)
        merge_frm.grid(row=11,column=2,sticky=W+N,padx=5)
        merge_frm_btn.grid(row=0, sticky=NW, padx=5)

        plotlyInterface.grid(row=11, column=3, sticky=W + N, padx=5)
        button_save_plotly_file.grid(row=10, sticky=W)
        self.plotly_file.grid(row=11, sticky=W)
        self.groups_file.grid(row=12, sticky=W)
        button_open_plotly_interface.grid(row=13, sticky=W)

        lbl_addon.grid(row=15,sticky=W)
        button_bel.grid(row=0,sticky=W)
        cue_light_analyser_btn.grid(row=1, sticky=NW)
        anchored_roi_analysis_btn.grid(row=2, sticky=NW)
        #unsupervised_btn.grid(row=3, sticky=NW)

    def create_video_info_table(self):
        video_info_tabler = VideoInfoTable(config_path=self.config_path)
        video_info_tabler.create_window()

    def initiate_skip_outlier_correction(self):
        outlier_correction_skipper = OutlierCorrectionSkipper(config_path=self.config_path)
        outlier_correction_skipper.skip_outlier_correction()


    def validate_model_first_step(self):
        _ = ValidateModelRunClf(config_path=self.config_path, input_file_path=self.csvfile.file_path, clf_path=self.modelfile.file_path)


    def directing_other_animals_analysis(self):
        directing_animals_analyzer = DirectingOtherAnimalsAnalyzer(config_path=self.config_path)
        directing_animals_analyzer.process_directionality()
        directing_animals_analyzer.create_directionality_dfs()
        directing_animals_analyzer.save_directionality_dfs()
        directing_animals_analyzer.summary_statistics()

    def directing_other_animals_visualizer(self):
        _ = DirectingOtherAnimalsVisualizerPopUp(config_path=self.config_path)


    def reverseid(self):
        config = ConfigParser()
        configFile = str(self.config_path)
        config.read(configFile)
        noanimal = int(config.get('General settings','animal_no'))

        if noanimal ==2:
            reverse_dlc_input_files(self.config_path)
        else:
            print('This only works if you have exactly 2 animals in your tracking data and video.')

    def train_single_model(self, config_path=None):
        model_trainer = TrainSingleModel(config_path=config_path)
        model_trainer.perform_sampling()
        model_trainer.train_model()
        model_trainer.save_model()

    def train_multiple_models_from_meta(self, config_path=None):
        model_trainer = TrainMultipleModelsFromMeta(config_path=config_path)
        model_trainer.train_models_from_meta()


    def importBoris(self):
        ann_folder = askdirectory()
        boris_appender = BorisAppender(config_path=self.config_path, boris_folder=ann_folder)
        boris_appender.create_boris_master_file()
        boris_appender.run()

    def importSolomon(self):
        ann_folder = askdirectory()
        solomon_importer = SolomonImporter(config_path=self.config_path,
                                           solomon_dir=ann_folder)
        solomon_importer.import_solomon()

    def import_ethovision(self):
        ann_folder = askdirectory()
        ImportEthovision(config_path=self.config_path, folder_path=ann_folder)

    def import_deepethogram(self):
        ann_folder = askdirectory()
        deepethogram_importer = DeepEthogramImporter(config_path=self.config_path, deep_ethogram_dir=ann_folder)
        deepethogram_importer.import_deepethogram()

    def import_noldus_observer(self):
        directory = askdirectory()
        noldus_observer_importer = NoldusObserverImporter(config_path=self.config_path, data_dir=directory)
        noldus_observer_importer.run()

    def importMARS(self):
        bento_dir = askdirectory()
        bento_appender = BentoAppender(config_path=self.config_path,
                                       bento_dir=bento_dir)
        bento_appender.run()

    def choose_animal_bps(self):
        if hasattr(self, 'path_plot_animal_bp_frm'):
            self.path_plot_animal_bp_frm.destroy()
        self.path_plot_animal_bp_frm = LabelFrame(self.path_plot_frm, text='CHOOSE ANIMAL BODY-PARTS',font=('Helvetica',10,'bold'), pady=5, padx=5)
        self.path_plot_bp_dict = {}
        for animal_cnt in range(int(self.path_animal_cnt_dropdown.getChoices())):
            self.path_plot_bp_dict[animal_cnt] = DropDownMenu(self.path_plot_animal_bp_frm, 'Animal {} bodypart'.format(str(animal_cnt + 1)), self.bp_set, '15')
            self.path_plot_bp_dict[animal_cnt].setChoices(self.bp_set[animal_cnt])
            self.path_plot_bp_dict[animal_cnt].grid(row=animal_cnt, sticky=NW)
        self.path_plot_animal_bp_frm.grid(row=2, column=0, sticky=NW)

    def launch_interactive_plot(self):
        interactive_grapher = InteractiveProbabilityGrapher(config_path=self.config_path,
                                                            file_path=self.csvfile.file_path,
                                                            model_path=self.modelfile.file_path)
        interactive_grapher.create_plots()


    def generateSimBPlotlyFile(self,var):
        inputList = []
        for i in var:
            inputList.append(i.get())

        create_plotly_container(self.path_plot_frm, inputList)

    def open_plotly_interface(self, url):

        try:
            self.p.kill()
            self.p2.kill()
        except:
            print('Starting plotly')
        #get h5 file path and csv file path
        filePath, groupPath = self.plotly_file.file_path, self.groups_file.file_path

        #print file read
        if filePath.endswith('.h5'):
            print('Reading in',os.path.basename(filePath))
        elif groupPath.endswith('.csv'):
            print('Reading in',os.path.basename(groupPath))

        self.p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__),'SimBA_dash_app.py'), filePath, groupPath])
        # csvPath = os.path.join(os.path.dirname(self.config_path),'csv')
        # p = subprocess.Popen([sys.executable, r'simba\SimBA_dash_app.py', filePath, groupPath, csvPath])
        wait_for_internet_connection(url)
        self.p2 = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__),'run_dash_tkinter.py'), url])
        subprocess_children = [self.p, self.p2]
        atexit.register(terminate_children, subprocess_children)

    def runrfmodel(self):
        rf_model_runner = RunModel(config_path=self.config_path)
        rf_model_runner.run_models()

    def trainmachinemodelsetting(self):
        _ = MachineModelSettingsPopUp(config_path=self.config_path)

    def run_feature_extraction(self):
        print(f'Pose-estimation body part setting for feature extraction: {str(self.animal_cnt)} animals {str(self.pose_setting)} body-parts')
        feature_extractor_classes = get_bp_config_code_class_pairs()
        if self.user_defined_var.get():
            _ = run_user_defined_feature_extraction_class(file_path=self.scriptfile.file_path, config_path=self.config_path)
        else:
            if (self.pose_setting == '8'):
                feature_extractor = feature_extractor_classes[self.pose_setting][self.animal_cnt](config_path=self.config_path)
            else:
                feature_extractor = feature_extractor_classes[self.pose_setting](config_path=self.config_path)
            feature_extractor.extract_features()

    def set_distance_mm(self):
        check_int(name='DISTANCE IN MILLIMETER', value=self.distance_in_mm_eb.entry_get, min_value=1)
        self.config.set('Frame settings', 'distance_mm', self.distance_in_mm_eb.entry_get)
        with open(self.config_path, 'w') as f:
            self.config.write(f)

    def correct_outlier(self):
        outlier_correcter_movement = OutlierCorrecterMovement(config_path=self.config_path)
        outlier_correcter_movement.correct_movement_outliers()
        outlier_correcter_location = OutlierCorrecterLocation(config_path=self.config_path)
        outlier_correcter_location.correct_location_outliers()
        print('SIMBA COMPLETE: Outlier correction complete. Outlier corrected files located in "project_folder/csv/outlier_corrected_movement_location" directory')

    def callback(self,url):
        webbrowser.open_new(url)



class App(object):
    def __init__(self):
        bg_path = os.path.join(os.path.dirname(__file__), Paths.BG_IMG_PATH.value)
        icon_path_windows = os.path.join(os.path.dirname(__file__), Paths.LOGO_ICON_WINDOWS_PATH.value)
        icon_path_darwin = os.path.join(os.path.dirname(__file__), Paths.LOGO_ICON_DARWIN_PATH.value)
        self.menu_icons = get_icons_paths()
        self.root = Tk()
        self.root.title('SimBA')
        self.root.minsize(750,750)
        self.root.geometry(Formats.ROOT_WINDOW_SIZE.value)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        if currentPlatform == OS.WINDOWS.value:
            self.root.iconbitmap(icon_path_windows)
        if currentPlatform == OS.MAC.value:
            self.root.iconphoto(False, ImageTk.PhotoImage(PIL.Image.open(icon_path_darwin)))
        for k in self.menu_icons.keys():
            self.menu_icons[k]['img'] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), self.menu_icons[k]['icon_path'])))
        bg_img = PhotoImage(file=bg_path)
        background = Label(self.root, image=bg_img, bd=0)
        background.pack(fill='both', expand=True)
        background.image = bg_img

        menu = Menu(self.root)
        self.root.config(menu=menu)

        file_menu = Menu(menu)
        menu.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Create a new project', compound='left', image=self.menu_icons['create']['img'], command=lambda: ProjectCreatorPopUp())
        file_menu.add_command(label='Load project', compound='left', image=self.menu_icons['load']['img'], command=lambda: LoadProjectPopUp())
        file_menu.add_separator()
        file_menu.add_command(label='Restart', compound='left', image=self.menu_icons['restart']['img'], command=lambda: self.restart())
        file_menu.add_separator()
        file_menu.add_command(label='Exit', compound='left', image=self.menu_icons['exit']['img'], command=self.root.destroy)

        batch_process_menu = Menu(menu)
        menu.add_cascade(label='Process Videos', menu=batch_process_menu)
        batch_process_menu.add_command(label='Batch pre-process videos', compound='left', image=self.menu_icons['factory']['img'], command=lambda: BatchPreProcessPopUp())

        video_process_menu = Menu(menu)
        fps_menu = Menu(video_process_menu)
        fps_menu.add_command(label='Change fps for single video', command=ChangeFpsSingleVideoPopUp)
        fps_menu.add_command(label='Change fps for multiple videos',command=ChangeFpsMultipleVideosPopUp)
        menu.add_cascade(label='Tools',menu=video_process_menu)
        video_process_menu.add_command(label='Clip videos', compound='left', image=self.menu_icons['clip']['img'], command=ClipVideoPopUp)
        video_process_menu.add_command(label='Clip video into multiple videos', compound='left', image=self.menu_icons['clip']['img'], command=MultiShortenPopUp)
        video_process_menu.add_command(label='Crop videos', compound='left', image=self.menu_icons['crop']['img'], command=CropVideoPopUp)
        video_process_menu.add_command(label='Multi-crop', compound='left', image=self.menu_icons['crop']['img'], command=MultiCropPopUp)
        video_process_menu.add_command(label='Down-sample videos', compound='left', image=self.menu_icons['sample']['img'], command=DownsampleVideoPopUp)
        video_process_menu.add_command(label='Get mm/ppx',  compound='left', image=self.menu_icons['calipher']['img'], command = CalculatePixelsPerMMInVideoPopUp)
        video_process_menu.add_command(label='Create path plot', compound='left', image=self.menu_icons['path']['img'], command=MakePathPlotPopUp)
        video_process_menu.add_cascade(label='Change fps...', compound='left', image=self.menu_icons['fps']['img'], menu =fps_menu)
        video_process_menu.add_cascade(label='Concatenate two videos', compound='left', image=self.menu_icons['concat']['img'], command=ConcatenatingVideosPopUp)
        video_process_menu.add_cascade(label='Concatenate multiple videos', compound='left', image=self.menu_icons['concat']['img'], command=lambda:ConcatenatorPopUp(config_path=None))
        video_process_menu.add_cascade(label='Visualize pose-estimation in folder...', compound='left', image=self.menu_icons['visualize']['img'], command=VisualizePoseInFolderPopUp)
        video_process_menu.add_cascade(label='Reorganize Tracking Data', compound='left', image=self.menu_icons['reorganize']['img'], command= PoseReorganizerPopUp)
        video_process_menu.add_cascade(label='Drop body-parts from tracking data', compound='left', image=self.menu_icons['trash']['img'], command=DropTrackingDataPopUp)

        format_menu = Menu(video_process_menu)
        format_menu.add_command(label='Change image file formats',command=ChangeImageFormatPopUp)
        format_menu.add_command(label='Change video file formats',command=ConvertVideoPopUp)
        video_process_menu.add_cascade(label='Change formats...', compound='left', image=self.menu_icons['convert']['img'], menu=format_menu)

        video_process_menu.add_command(label='CLAHE enhance video', compound='left', image=self.menu_icons['clahe']['img'], command=CLAHEPopUp)
        video_process_menu.add_command(label='Superimpose frame numbers on video', compound='left', image=self.menu_icons['trash']['img'], command=lambda:superimpose_frame_count(file_path=askopenfilename()))
        video_process_menu.add_command(label='Convert to grayscale', compound='left', image=self.menu_icons['grey']['img'], command=lambda:video_to_greyscale(file_path=askopenfilename()))
        video_process_menu.add_command(label='Merge frames to video', compound='left', image=self.menu_icons['merge']['img'], command=MergeFrames2VideoPopUp)
        video_process_menu.add_command(label='Generate gifs', compound='left', image=self.menu_icons['gif']['img'], command=CreateGIFPopUP)
        video_process_menu.add_command(label='Print classifier info...', compound='left', image=self.menu_icons['print']['img'], command=PrintModelInfoPopUp)

        extract_frames_menu = Menu(video_process_menu)
        extract_frames_menu.add_command(label='Extract defined frames',command=ExtractSpecificFramesPopUp)
        extract_frames_menu.add_command(label='Extract frames',command=ExtractAllFramesPopUp)
        extract_frames_menu.add_command(label='Extract frames from seq files', command=ExtractSEQFramesPopUp)
        video_process_menu.add_cascade(label='Extract frames...', compound='left', image=self.menu_icons['frames']['img'], menu=extract_frames_menu)

        convert_data_menu = Menu(video_process_menu)
        convert_data_menu.add_command(label='Convert CSV to parquet', command=Csv2ParquetPopUp)
        convert_data_menu.add_command(label='Convert parquet o CSV', command=Parquet2CsvPopUp)
        video_process_menu.add_cascade(label='Convert working file type...', compound='left', image=self.menu_icons['change']['img'], menu=convert_data_menu)

        help_menu = Menu(menu)
        menu.add_cascade(label='Help',menu=help_menu)

        links_menu = Menu(help_menu)
        links_menu.add_command(label='Download weights',command = lambda:webbrowser.open_new(str(r'https://osf.io/sr3ck/')))
        links_menu.add_command(label='Download classifiers', command=lambda: webbrowser.open_new(str(r'https://osf.io/kwge8/')))
        links_menu.add_command(label='Ex. feature list',command=lambda: webbrowser.open_new(str(r'https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv')))
        links_menu.add_command(label='SimBA github', command=lambda: webbrowser.open_new(str(r'https://github.com/sgoldenlab/simba')))
        links_menu.add_command(label='Gitter Chatroom', command=lambda: webbrowser.open_new(str(r'https://gitter.im/SimBA-Resource/community')))
        links_menu.add_command(label='Install FFmpeg',command =lambda: webbrowser.open_new(str(r'https://m.wikihow.com/Install-FFmpeg-on-Windows')))
        links_menu.add_command(label='Install graphviz', command=lambda: webbrowser.open_new(str(r'https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software')))
        help_menu.add_cascade(label="Links",menu=links_menu, compound='left', image=self.menu_icons['link']['img'])
        help_menu.add_command(label='About', compound='left', image=self.menu_icons['about']['img'], command=AboutSimBAPopUp)

        self.frame = Frame(background, bd=2, relief=SUNKEN, width=300, height=300)
        self.frame.pack(expand=True)
        self.txt = Text(self.frame, bg='white')
        self.txt.config(state=DISABLED, font=Formats.TKINTER_FONT.value)
        self.txt.pack(expand=True, fill='both')
        sys.stdout = StdRedirector(self.txt)

    def restart(self):
        confirm_restart = askyesno(title='RESTART', message='Are you sure that you want restart SimBA?')
        if confirm_restart:
            self.root.destroy()
            python = sys.executable
            os.execl(python, python, *sys.argv)



# writes text out in GUI
class StdRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.config(state=NORMAL)
        self.text_space.insert("end", string)
        self.text_space.update()
        self.text_space.see("end")
        self.text_space.config(state=DISABLED)

    def flush(self):
        pass

class SplashScreen:
    def __init__(self, parent):
        self.parent = parent
        self.Splash()
        self.Window()

    def Splash(self):
        splash_path_win = os.path.join(os.path.dirname(__file__), Paths.SPLASH_PATH_WINDOWS.value)
        splash_path_linux = os.path.join(os.path.dirname(__file__), Paths.SPLASH_PATH_LINUX.value)
        if currentPlatform == OS.WINDOWS.value:
            self.splash_img = PIL.Image.open(splash_path_win)
        else:
            if os.path.isfile(splash_path_linux):
                self.splash_img = PIL.Image.open(splash_path_linux)
            else:
                self.splash_img = PIL.Image.open(splash_path_win)
        self.splash_img_tk = ImageTk.PhotoImage(self.splash_img)


    def Window(self):
        width, height = self.splash_img.size
        half_width = int((self.parent.winfo_screenwidth()-width)//2)
        half_height = int((self.parent.winfo_screenheight()-height)//2)
        self.parent.geometry("%ix%i+%i+%i" %(width, height, half_width, half_height))
        Label(self.parent, image=self.splash_img_tk).pack()

def terminate_children(children):
    for process in children:
        process.terminate()

def main():
    if currentPlatform == OS.WINDOWS.value:
        import ctypes
        myappid = 'SimBA development wheel'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    root = Tk()
    root.overrideredirect(True)
    _ = SplashScreen(root)
    root.after(Defaults.SPLASH_TIME.value, root.destroy)
    root.mainloop()
    app = App()
    print(Defaults.WELCOME_MSG.value)
    app.root.mainloop()