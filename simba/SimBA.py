__author__ = "Simon Nilsson"

import os.path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import atexit
import subprocess
import sys
import threading
import tkinter.ttk as ttk
import urllib.request
import webbrowser
from tkinter.filedialog import askdirectory
from tkinter.messagebox import askyesno

import PIL.Image
from PIL import ImageTk

from simba.bounding_box_tools.boundary_menus import BoundaryMenus
from simba.cue_light_tools.cue_light_menues import CueLightAnalyzerMenu
from simba.labelling.labelling_advanced_interface import \
    select_labelling_video_advanced
from simba.labelling.labelling_interface import select_labelling_video
from simba.labelling.targeted_annotations_clips import \
    select_labelling_video_targeted_clips
from simba.model.grid_search_rf import GridSearchRandomForestClassifier
from simba.model.inference_batch import InferenceBatch
from simba.model.inference_validation import InferenceValidation
from simba.model.train_rf import TrainRandomForestClassifier
from simba.outlier_tools.outlier_corrector_location import \
    OutlierCorrecterLocation
from simba.outlier_tools.outlier_corrector_movement import \
    OutlierCorrecterMovement
from simba.outlier_tools.skip_outlier_correction import \
    OutlierCorrectionSkipper
from simba.plotting.interactive_probability_grapher import \
    InteractiveProbabilityGrapher
from simba.roi_tools.ROI_define import *
from simba.roi_tools.ROI_menus import *
from simba.roi_tools.ROI_reset import *
from simba.third_party_label_appenders.BENTO_appender import BentoAppender
from simba.third_party_label_appenders.BORIS_appender import BorisAppender
from simba.third_party_label_appenders.deepethogram_importer import \
    DeepEthogramImporter
from simba.third_party_label_appenders.ethovision_import import \
    ImportEthovision
from simba.third_party_label_appenders.observer_importer import \
    NoldusObserverImporter
from simba.third_party_label_appenders.solomon_importer import SolomonImporter
from simba.ui.create_project_ui import ProjectCreatorPopUp
from simba.ui.machine_model_settings_ui import MachineModelSettingsPopUp
from simba.ui.pop_ups.about_simba_pop_up import AboutSimBAPopUp
from simba.ui.pop_ups.animal_directing_other_animals_pop_up import \
    AnimalDirectingAnimalPopUp
from simba.ui.pop_ups.append_roi_features_animals_pop_up import \
    AppendROIFeaturesByAnimalPopUp
from simba.ui.pop_ups.append_roi_features_bodypart_pop_up import \
    AppendROIFeaturesByBodyPartPopUp
from simba.ui.pop_ups.archive_files_pop_up import ArchiveProcessedFilesPopUp
from simba.ui.pop_ups.batch_preprocess_pop_up import BatchPreProcessPopUp
from simba.ui.pop_ups.boolean_conditional_slicer_pup_up import \
    BooleanConditionalSlicerPopUp
from simba.ui.pop_ups.clf_add_remove_print_pop_up import (
    AddClfPopUp, PrintModelInfoPopUp, RemoveAClassifierPopUp)
from simba.ui.pop_ups.clf_by_roi_pop_up import ClfByROIPopUp
from simba.ui.pop_ups.clf_by_timebins_pop_up import TimeBinsClfPopUp
from simba.ui.pop_ups.clf_descriptive_statistics_pop_up import \
    ClfDescriptiveStatsPopUp
from simba.ui.pop_ups.clf_plot_pop_up import SklearnVisualizationPopUp
from simba.ui.pop_ups.clf_probability_plot_pop_up import \
    VisualizeClassificationProbabilityPopUp
from simba.ui.pop_ups.clf_validation_plot_pop_up import \
    ClassifierValidationPopUp
from simba.ui.pop_ups.csv_2_parquet_pop_up import (Csv2ParquetPopUp,
                                                   Parquet2CsvPopUp)
from simba.ui.pop_ups.data_plot_pop_up import DataPlotterPopUp
from simba.ui.pop_ups.directing_animal_to_bodypart_plot_pop_up import \
    DirectingAnimalToBodyPartVisualizerPopUp
from simba.ui.pop_ups.directing_other_animals_plot_pop_up import \
    DirectingOtherAnimalsVisualizerPopUp
from simba.ui.pop_ups.direction_animal_to_bodypart_settings_pop_up import \
    DirectionAnimalToBodyPartSettingsPopUp
from simba.ui.pop_ups.distance_plot_pop_up import DistancePlotterPopUp
from simba.ui.pop_ups.fsttc_pop_up import FSTTCPopUp
from simba.ui.pop_ups.gantt_pop_up import GanttPlotPopUp
from simba.ui.pop_ups.heatmap_clf_pop_up import HeatmapClfPopUp
from simba.ui.pop_ups.heatmap_location_pop_up import HeatmapLocationPopup
from simba.ui.pop_ups.kleinberg_pop_up import KleinbergPopUp
from simba.ui.pop_ups.make_path_plot_pop_up import MakePathPlotPopUp
from simba.ui.pop_ups.movement_analysis_pop_up import MovementAnalysisPopUp
from simba.ui.pop_ups.movement_analysis_time_bins_pop_up import \
    MovementAnalysisTimeBinsPopUp
from simba.ui.pop_ups.mutual_exclusivity_pop_up import MutualExclusivityPupUp
from simba.ui.pop_ups.outlier_settings_pop_up import OutlierSettingsPopUp
from simba.ui.pop_ups.path_plot_pop_up import PathPlotPopUp
from simba.ui.pop_ups.pose_bp_drop_pop_up import DropTrackingDataPopUp
from simba.ui.pop_ups.pose_reorganizer_pop_up import PoseReorganizerPopUp
from simba.ui.pop_ups.pup_retrieval_pop_up import PupRetrievalPopUp
from simba.ui.pop_ups.quick_path_plot_pop_up import QuickLineplotPopup
from simba.ui.pop_ups.remove_roi_features_pop_up import RemoveROIFeaturesPopUp
from simba.ui.pop_ups.roi_analysis_pop_up import ROIAnalysisPopUp
from simba.ui.pop_ups.roi_analysis_time_bins_pop_up import \
    ROIAnalysisTimeBinsPopUp
from simba.ui.pop_ups.roi_features_plot_pop_up import VisualizeROIFeaturesPopUp
from simba.ui.pop_ups.roi_size_standardizer_popup import \
    ROISizeStandardizerPopUp
from simba.ui.pop_ups.roi_tracking_plot_pop_up import VisualizeROITrackingPopUp
from simba.ui.pop_ups.set_machine_model_parameters_pop_up import \
    SetMachineModelParameters
from simba.ui.pop_ups.severity_analysis_pop_up import AnalyzeSeverityPopUp
from simba.ui.pop_ups.smoothing_interpolation_pop_up import (InterpolatePopUp,
                                                             SmoothingPopUp)
from simba.ui.pop_ups.spontaneous_alternation_pop_up import \
    SpontaneousAlternationPopUp
from simba.ui.pop_ups.subset_feature_extractor_pop_up import \
    FeatureSubsetExtractorPopUp
from simba.ui.pop_ups.third_party_annotator_appender_pop_up import \
    ThirdPartyAnnotatorAppenderPopUp
from simba.ui.pop_ups.validation_plot_pop_up import ValidationVideoPopUp
from simba.ui.pop_ups.video_processing_pop_up import (
    BrightnessContrastPopUp, CalculatePixelsPerMMInVideoPopUp,
    ChangeFpsMultipleVideosPopUp, ChangeFpsSingleVideoPopUp, CLAHEPopUp,
    ClipSingleVideoByFrameNumbers, ClipVideoPopUp, ConcatenatingVideosPopUp,
    ConcatenatorPopUp, Convert2AVIPopUp, Convert2bmpPopUp, Convert2jpegPopUp,
    Convert2MOVPopUp, Convert2MP4PopUp, Convert2PNGPopUp, Convert2TIFFPopUp,
    Convert2WEBMPopUp, Convert2WEBPPopUp, ConvertROIDefinitionsPopUp,
    CreateGIFPopUP, CropVideoCirclesPopUp, CropVideoPolygonsPopUp,
    CropVideoPopUp, DownsampleMultipleVideosPopUp, DownsampleSingleVideoPopUp,
    DownsampleVideoPopUp, ExtractAllFramesPopUp, ExtractAnnotationFramesPopUp,
    ExtractSEQFramesPopUp, ExtractSpecificFramesPopUp,
    GreyscaleSingleVideoPopUp, ImportFrameDirectoryPopUp,
    InitiateClipMultipleVideosByFrameNumbersPopUp,
    InitiateClipMultipleVideosByTimestampsPopUp, InteractiveClahePopUp,
    MergeFrames2VideoPopUp, MultiCropPopUp, MultiShortenPopUp,
    SuperImposeFrameCountPopUp, VideoRotatorPopUp, VideoTemporalJoinPopUp,
    SuperimposeTimerPopUp, SuperimposeWatermarkPopUp, SuperimposeProgressBarPopUp)
from simba.ui.pop_ups.visualize_pose_in_dir_pop_up import \
    VisualizePoseInFolderPopUp
from simba.ui.tkinter_functions import DropDownMenu, Entry_Box, FileSelect
from simba.ui.video_info_ui import VideoInfoTable
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_int)
from simba.utils.custom_feature_extractor import CustomFeatureExtractor
from simba.utils.enums import OS, Defaults, Formats, Paths, TagNames
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import (get_bp_config_code_class_pairs, get_emojis,
                                 get_icons_paths)
from simba.utils.printing import stdout_success, stdout_warning
from simba.utils.read_write import get_video_meta_data
from simba.utils.warnings import FFMpegNotFoundWarning, PythonVersionWarning
from simba.video_processors.video_processing import \
    extract_frames_from_all_videos_in_directory

sys.setrecursionlimit(10**6)
currentPlatform = platform.system()

UNSUPERVISED = False


class LoadProjectPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(300, 200)
        self.main_frm.wm_title("Load SimBA project (project_config.ini file)")
        self.load_project_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="LOAD SIMBA PROJECT_CONFIG.INI",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.LOAD_PROJECT.value,
        )
        self.selected_file = FileSelect(
            self.load_project_frm,
            "Select file: ",
            title="Select project_config.ini file",
            file_types=[("SimBA Project .ini", "*.ini")],
        )
        load_project_btn = Button(
            self.load_project_frm,
            text="LOAD PROJECT",
            fg="blue",
            command=lambda: self.launch_project(),
        )

        self.load_project_frm.grid(row=0, sticky=NW)
        self.selected_file.grid(row=0, sticky=NW)
        load_project_btn.grid(row=1, pady=10, sticky=NW)

    def launch_project(self):
        check_file_exist_and_readable(file_path=self.selected_file.file_path)
        _ = SimbaProjectPopUp(config_path=self.selected_file.file_path)
        self.load_project_frm.destroy()
        self.main_frm.destroy()


def wait_for_internet_connection(url):
    while True:
        try:
            response = urllib.request.urlopen(url, timeout=1)
            return
        except:
            pass


class SimbaProjectPopUp(ConfigReader, PopUpMixin):
    """
    Main entry to the SimBA loaded project pop-up.
    """

    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        stdout_success(f"Loaded project {config_path}", source=self.__class__.__name__)
        simongui = Toplevel()
        simongui.minsize(1300, 800)
        simongui.wm_title("LOAD PROJECT")
        simongui.columnconfigure(0, weight=1)
        simongui.rowconfigure(0, weight=1)

        self.btn_icons = get_icons_paths()
        for k in self.btn_icons.keys():
            self.btn_icons[k]["img"] = ImageTk.PhotoImage(
                image=PIL.Image.open(
                    os.path.join(
                        os.path.dirname(__file__), self.btn_icons[k]["icon_path"]
                    )
                )
            )

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

        tab_parent.add(
            tab2,
            text=f"{'[ Further imports (data/video/frames) ]':20s}",
            compound="left",
            image=self.btn_icons["pose"]["img"],
        )
        tab_parent.add(
            tab3,
            text=f"{'[ Video parameters ]':20s}",
            compound="left",
            image=self.btn_icons["calipher"]["img"],
        )
        tab_parent.add(
            tab4,
            text=f"{'[ Outlier correction ]':20s}",
            compound="left",
            image=self.btn_icons["outlier"]["img"],
        )
        tab_parent.add(
            tab6,
            text=f"{'[ ROI ]':10s}",
            compound="left",
            image=self.btn_icons["roi"]["img"],
        )
        tab_parent.add(
            tab5,
            text=f"{'[ Extract features ]':20s}",
            compound="left",
            image=self.btn_icons["features"]["img"],
        )
        tab_parent.add(
            tab7,
            text=f"{'[ Label behavior] ':20s}",
            compound="left",
            image=self.btn_icons["label"]["img"],
        )
        tab_parent.add(
            tab8,
            text=f"{'[ Train machine model ]':20s}",
            compound="left",
            image=self.btn_icons["clf"]["img"],
        )
        tab_parent.add(
            tab9,
            text=f"{'[ Run machine model ]':20s}",
            compound="left",
            image=self.btn_icons["clf_2"]["img"],
        )
        tab_parent.add(
            tab10,
            text=f"{'[ Visualizations ]':20s}",
            compound="left",
            image=self.btn_icons["visualize"]["img"],
        )
        tab_parent.add(
            tab11,
            text=f"{'[ Add-ons ]':20s}",
            compound="left",
            image=self.btn_icons["add_on"]["img"],
        )

        tab_parent.grid(row=0)
        tab_parent.enable_traversal()

        import_frm = LabelFrame(tab2)
        import_frm.grid(row=0, column=0, sticky=NW)

        further_methods_frm = CreateLabelFrameWithIcon(
            parent=import_frm,
            header="FURTHER METHODS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ADDITIONAL_IMPORTS.value,
        )
        extract_frm_btn = Button(
            further_methods_frm,
            text="EXTRACT FRAMES FOR ALL VIDEOS IN SIMBA PROJECT",
            fg="blue",
            command=lambda: extract_frames_from_all_videos_in_directory(
                config_path=self.config_path, directory=self.video_dir
            ),
        )
        import_frm_dir_btn = Button(
            further_methods_frm,
            text="IMPORT FRAMES DIRECTORY TO SIMBA PROJECT",
            fg="blue",
            command=lambda: ImportFrameDirectoryPopUp(config_path=self.config_path),
        )
        add_clf_btn = Button(
            further_methods_frm,
            text="ADD CLASSIFIER TO SIMBA PROJECT",
            fg="blue",
            command=lambda: AddClfPopUp(config_path=self.config_path),
        )
        remove_clf_btn = Button(
            further_methods_frm,
            text="REMOVE CLASSIFIER FROM SIMBA PROJECT",
            fg="blue",
            command=lambda: RemoveAClassifierPopUp(config_path=self.config_path),
        )
        archive_files_btn = Button(
            further_methods_frm,
            text="ARCHIVE PROCESSED FILES IN SIMBA PROJECT",
            fg="blue",
            command=lambda: ArchiveProcessedFilesPopUp(config_path=self.config_path),
        )
        reverse_btn = Button(
            further_methods_frm,
            text="REVERSE TRACKING IDENTITIES IN SIMBA PROJECT",
            fg="blue",
            command=lambda: None,
        )
        interpolate_btn = Button(
            further_methods_frm,
            text="INTERPOLATE POSE IN SIMBA PROJECT",
            fg="blue",
            command=lambda: InterpolatePopUp(config_path=self.config_path),
        )
        smooth_btn = Button(
            further_methods_frm,
            text="SMOOTH POSE IN SIMBA PROJECT",
            fg="blue",
            command=lambda: SmoothingPopUp(config_path=self.config_path),
        )

        label_setscale = CreateLabelFrameWithIcon(
            parent=tab3,
            header="VIDEO PARAMETERS (FPS, RESOLUTION, PPX/MM ....)",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_PARAMETERS.value,
        )
        # label_setscale = LabelFrame(tab3,text='VIDEO PARAMETERS (FPS, RESOLUTION, PPX/MM ....)', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5,padx=5,fg='black')
        self.distance_in_mm_eb = Entry_Box(
            label_setscale, "KNOWN DISTANCE (MILLIMETERS)", "25", validation="numeric"
        )
        button_setdistanceinmm = Button(
            label_setscale,
            text="AUTO-POPULATE",
            fg="green",
            command=lambda: self.set_distance_mm(),
        )

        button_setscale = Button(
            label_setscale,
            text="CONFIGURE VIDEO PARAMETERS",
            compound="left",
            image=self.btn_icons["calipher"]["img"],
            relief=RAISED,
            fg="blue",
            command=lambda: self.create_video_info_table(),
        )
        button_setscale.image = self.btn_icons["calipher"]["img"]

        self.new_ROI_frm = CreateLabelFrameWithIcon(
            parent=tab6,
            header="SIMBA ROI INTERFACE",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI.value,
        )
        self.start_new_ROI = Button(
            self.new_ROI_frm,
            text="DEFINE ROIs",
            fg="green",
            compound="left",
            image=self.btn_icons["roi"]["img"],
            relief=RAISED,
            command=lambda: ROI_menu(self.config_path),
        )
        self.start_new_ROI.image = self.btn_icons["roi"]["img"]

        self.delete_all_ROIs = Button(
            self.new_ROI_frm,
            text="DELETE ALL ROI DEFINITIONS",
            fg="red",
            compound="left",
            image=self.btn_icons["trash"]["img"],
            command=lambda: delete_all_ROIs(self.config_path),
        )
        self.delete_all_ROIs.image = self.btn_icons["trash"]["img"]

        self.standardize_roi_size_popup_btn = Button(
            self.new_ROI_frm,
            text="STANDARDIZE ROI SIZES",
            fg="blue",
            compound="left",
            image=self.btn_icons["calipher"]["img"],
            command=lambda: ROISizeStandardizerPopUp(config_path=self.config_path),
        )
        self.standardize_roi_size_popup_btn.image = self.btn_icons["calipher"]["img"]

        self.new_ROI_frm.grid(row=0, sticky=NW)
        self.start_new_ROI.grid(row=0, sticky=NW)
        self.delete_all_ROIs.grid(row=1, column=0, sticky=NW)
        self.standardize_roi_size_popup_btn.grid(row=2, column=0, sticky=NW)

        self.roi_draw = LabelFrame(
            tab6, text="ANALYZE ROI DATA", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        analyze_roi_btn = Button(
            self.roi_draw,
            text="ANALYZE ROI DATA: AGGREGATES",
            fg="green",
            command=lambda: ROIAnalysisPopUp(config_path=self.config_path),
        )
        analyze_roi_time_bins_btn = Button(
            self.roi_draw,
            text="ANALYZE ROI DATA: TIME-BINS",
            fg="blue",
            command=lambda: ROIAnalysisTimeBinsPopUp(config_path=self.config_path),
        )

        self.roi_draw.grid(row=0, column=1, sticky=N)
        analyze_roi_btn.grid(row=0, sticky="NW")
        analyze_roi_time_bins_btn.grid(row=1, sticky="NW")

        ###plot roi
        self.roi_draw1 = LabelFrame(
            tab6, text="VISUALIZE ROI DATA", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )

        # button
        visualizeROI = Button(
            self.roi_draw1,
            text="VISUALIZE ROI TRACKING",
            fg="green",
            command=lambda: VisualizeROITrackingPopUp(config_path=self.config_path),
        )
        visualizeROIfeature = Button(
            self.roi_draw1,
            text="VISUALIZE ROI FEATURES",
            fg="blue",
            command=lambda: VisualizeROIFeaturesPopUp(config_path=self.config_path),
        )

        ##organize
        self.roi_draw1.grid(row=0, column=2, sticky=N)
        visualizeROI.grid(row=0, sticky="NW")
        visualizeROIfeature.grid(row=1, sticky="NW")

        processmovementdupLabel = LabelFrame(
            tab6,
            text="OTHER ANALYSES / VISUALIZATIONS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        analyze_distances_velocity_btn = Button(
            processmovementdupLabel,
            text="ANALYZE DISTANCES / VELOCITY: AGGREGATES",
            fg="green",
            command=lambda: MovementAnalysisPopUp(config_path=self.config_path),
        )
        analyze_distances_velocity_timebins_btn = Button(
            processmovementdupLabel,
            text="ANALYZE DISTANCES / VELOCITY: TIME-BINS",
            fg="blue",
            command=lambda: MovementAnalysisTimeBinsPopUp(config_path=self.config_path),
        )

        heatmaps_location_button = Button(
            processmovementdupLabel,
            text="CREATE LOCATION HEATMAPS",
            fg="red",
            command=lambda: HeatmapLocationPopup(config_path=self.config_path),
        )

        button_lineplot = Button(
            processmovementdupLabel,
            text="CREATE PATH PLOTS",
            fg="orange",
            command=lambda: QuickLineplotPopup(config_path=self.config_path),
        )
        button_analyzeDirection = Button(
            processmovementdupLabel,
            text="ANALYZE DIRECTIONALITY BETWEEN ANIMALS",
            fg="pink",
            command=lambda: AnimalDirectingAnimalPopUp(config_path=self.config_path),
        )
        button_visualizeDirection = Button(
            processmovementdupLabel,
            text="VISUALIZE DIRECTIONALITY BETWEEN ANIMALS",
            fg="brown",
            command=lambda: DirectingOtherAnimalsVisualizerPopUp(
                config_path=self.config_path
            ),
        )
        button_analyzeDirection_bp = Button(
            processmovementdupLabel,
            text="ANALYZE DIRECTIONALITY BETWEEN BODY PARTS",
            fg="purple",
            command=lambda: DirectionAnimalToBodyPartSettingsPopUp(
                config_path=self.config_path
            ),
        )
        button_visualizeDirection_bp = Button(
            processmovementdupLabel,
            text="VISUALIZE DIRECTIONALITY BETWEEN BODY PARTS",
            fg="black",
            command=lambda: DirectingAnimalToBodyPartVisualizerPopUp(
                config_path=self.config_path
            ),
        )
        btn_agg_boolean_conditional_statistics = Button(
            processmovementdupLabel,
            text="AGGREGATE BOOLEAN CONDITIONAL STATISTICS",
            fg="grey",
            command=lambda: BooleanConditionalSlicerPopUp(config_path=self.config_path),
        )
        spontaneous_alternation_pop_up_btn = Button(
            processmovementdupLabel,
            text="SPONTANEOUS ALTERNATION",
            fg="navy",
            command=lambda: threading.Thread(
                target=SpontaneousAlternationPopUp(config_path=self.config_path)
            ).start(),
        )

        # organize
        processmovementdupLabel.grid(row=0, column=3, sticky=NW)
        analyze_distances_velocity_btn.grid(row=0, sticky=NW)
        heatmaps_location_button.grid(row=1, sticky=NW)
        analyze_distances_velocity_timebins_btn.grid(row=2, sticky=NW)
        button_lineplot.grid(row=3, sticky=NW)
        button_analyzeDirection.grid(row=4, sticky=NW)
        button_visualizeDirection.grid(row=5, sticky=NW)
        button_analyzeDirection_bp.grid(row=6, sticky=NW)
        button_visualizeDirection_bp.grid(row=7, sticky=NW)
        btn_agg_boolean_conditional_statistics.grid(row=8, sticky=NW)
        spontaneous_alternation_pop_up_btn.grid(row=9, sticky=NW)

        label_outliercorrection = CreateLabelFrameWithIcon(
            parent=tab4,
            header="OUTLIER CORRECTION",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.OUTLIERS_DOC.value,
        )
        # label_outliercorrection = LabelFrame(tab4,text='OUTLIER CORRECTION',font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg='black')
        button_settings_outlier = Button(
            label_outliercorrection,
            text="SETTINGS",
            fg="blue",
            command=lambda: OutlierSettingsPopUp(config_path=self.config_path),
        )
        button_outliercorrection = Button(
            label_outliercorrection,
            text="RUN OUTLIER CORRECTION",
            fg="green",
            command=lambda: threading.Thread(target=self.correct_outlier).start(),
        )
        button_skipOC = Button(
            label_outliercorrection,
            text="SKIP OUTLIER CORRECTION (CAUTION)",
            fg="red",
            command=lambda: threading.Thread(
                target=self.initiate_skip_outlier_correction
            ).start(),
        )

        # extract features
        label_extractfeatures = CreateLabelFrameWithIcon(
            parent=tab5,
            header="EXTRACT FEATURES",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.EXTRACT_FEATURES.value,
        )
        button_extractfeatures = Button(
            label_extractfeatures,
            text="EXTRACT FEATURES",
            fg="blue",
            command=lambda: threading.Thread(
                target=self.run_feature_extraction
            ).start(),
        )

        def activate(box, *args):
            for entry in args:
                if box.get() == 0:
                    entry.configure(state="disabled")
                elif box.get() == 1:
                    entry.configure(state="normal")

        labelframe_usrdef = LabelFrame(label_extractfeatures)
        self.scriptfile = FileSelect(
            labelframe_usrdef, "Script path", file_types=[("Python .py file", "*.py")]
        )
        self.scriptfile.btnFind.configure(state="disabled")
        self.user_defined_var = BooleanVar(value=False)
        userscript = Checkbutton(
            labelframe_usrdef,
            text="Apply user-defined feature extraction script",
            variable=self.user_defined_var,
            command=lambda: activate(self.user_defined_var, self.scriptfile.btnFind),
        )

        roi_feature_frm = CreateLabelFrameWithIcon(
            parent=tab5,
            header="APPEND ROI FEATURES",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.APPEND_ROI_FEATURES.value,
        )
        append_roi_features_by_animal = Button(
            roi_feature_frm,
            text="APPEND ROI DATA TO FEATURES: BY ANIMAL (CAUTION)",
            fg="red",
            command=lambda: AppendROIFeaturesByAnimalPopUp(
                config_path=self.config_path
            ),
        )
        append_roi_features_by_body_part = Button(
            roi_feature_frm,
            text="APPEND ROI DATA TO FEATURES: BY BODY-PARTS (CAUTION)",
            fg="orange",
            command=lambda: AppendROIFeaturesByBodyPartPopUp(
                config_path=self.config_path
            ),
        )
        remove_roi_features_from_feature_set = Button(
            roi_feature_frm,
            text="REMOVE ROI FEATURES FROM FEATURE SET",
            fg="darkred",
            command=lambda: RemoveROIFeaturesPopUp(
                config_path=self.config_path, dataset="features_extracted"
            ),
        )

        feature_tools_frm = LabelFrame(
            tab5,
            text="FEATURE TOOLS",
            pady=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        compute_feature_subset_btn = Button(
            feature_tools_frm,
            text="CALCULATE FEATURE SUBSETS",
            fg="blue",
            command=lambda: FeatureSubsetExtractorPopUp(config_path=self.config_path),
        )

        label_behavior_frm = CreateLabelFrameWithIcon(
            parent=tab7,
            header="LABEL BEHAVIOR",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.LABEL_BEHAVIOR.value,
        )
        select_video_btn_new = Button(
            label_behavior_frm,
            text="Select video (create new video annotation)",
            command=lambda: select_labelling_video(
                config_path=self.config_path,
                threshold_dict=None,
                setting="from_scratch",
                continuing=False,
            ),
        )
        select_video_btn_continue = Button(
            label_behavior_frm,
            text="Select video (continue existing video annotation)",
            command=lambda: select_labelling_video(
                config_path=self.config_path,
                threshold_dict=None,
                setting=None,
                continuing=True,
            ),
        )
        label_thirdpartyann = CreateLabelFrameWithIcon(
            parent=tab7,
            header="IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.THIRD_PARTY_ANNOTATION.value,
        )
        button_importmars = Button(
            label_thirdpartyann,
            text="Import MARS Annotation (select folder with .annot files)",
            fg="blue",
            command=self.importMARS,
        )
        button_importboris = Button(
            label_thirdpartyann,
            text="Import BORIS Annotation (select folder with .csv files)",
            fg="green",
            command=self.importBoris,
        )
        button_importsolomon = Button(
            label_thirdpartyann,
            text="Import SOLOMON Annotation (select folder with .csv files",
            fg="purple",
            command=self.importSolomon,
        )
        button_importethovision = Button(
            label_thirdpartyann,
            text="Import ETHOVISION Annotation (select folder with .xls/xlsx files)",
            fg="blue",
            command=self.import_ethovision,
        )
        button_importdeepethogram = Button(
            label_thirdpartyann,
            text="Import DEEPETHOGRAM Annotation (select folder with .csv files)",
            fg="green",
            command=self.import_deepethogram,
        )
        import_observer_btn = Button(
            label_thirdpartyann,
            text="Import NOLDUS OBSERVER Annotation (select folder with .xls/xlsx files)",
            fg="purple",
            command=self.import_noldus_observer,
        )

        label_pseudo = CreateLabelFrameWithIcon(
            parent=tab7,
            header="PSEUDO-LABELLING",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.PSEUDO_LBL.value,
        )
        pseudo_intructions_lbl_1 = Label(
            label_pseudo,
            text="Note that SimBA pseudo-labelling require initial machine predictions.",
        )
        pseudo_intructions_lbl_2 = Label(
            label_pseudo,
            text="Click here more information on how to use the SimBA pseudo-labelling interface.",
            cursor="hand2",
            fg="blue",
        )
        pseudo_intructions_lbl_2.bind(
            "<Button-1>",
            lambda e: self.callback(
                "https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md"
            ),
        )
        pLabel_framedir = FileSelect(label_pseudo, "Video Path", lblwidth="10")
        plabelframe_threshold = LabelFrame(
            label_pseudo, text="Threshold", pady=5, padx=5
        )
        plabel_threshold = [0] * len(self.clf_names)
        for count, i in enumerate(list(self.clf_names)):
            plabel_threshold[count] = Entry_Box(plabelframe_threshold, str(i), "20")
            plabel_threshold[count].grid(row=count + 2, sticky=W)

        pseudo_lbl_btn = Button(
            label_pseudo,
            text="Correct labels",
            command=lambda: select_labelling_video(
                config_path=self.config_path,
                threshold_dict=dict(zip(self.clf_names, plabel_threshold)),
                setting="pseudo",
                continuing=False,
                video_file_path=pLabel_framedir.file_path,
            ),
        )

        label_adv_label = CreateLabelFrameWithIcon(
            parent=tab7,
            header="ADVANCED LABEL BEHAVIOR",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ADVANCED_LBL.value,
        )
        label_adv_note_1 = Label(
            label_adv_label,
            text="Note that you will have to specify the presence of *both* behavior and non-behavior on your own.",
        )
        label_adv_note_2 = Label(
            label_adv_label,
            text="Click here more information on how to use the SimBA labelling interface.",
            cursor="hand2",
            fg="blue",
        )
        label_adv_note_2.bind(
            "<Button-1>",
            lambda e: self.callback(
                "https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md"
            ),
        )
        adv_label_btn_new = Button(
            label_adv_label,
            text="Select video (create new video annotation)",
            command=lambda: select_labelling_video_advanced(
                config_path=self.config_path, continuing=False
            ),
        )
        adv_label_btn_continue = Button(
            label_adv_label,
            text="Select video (continue existing video annotation)",
            command=lambda: select_labelling_video_advanced(
                config_path=self.config_path, continuing=True
            ),
        )

        targeted_clip_annotator_frm = CreateLabelFrameWithIcon(
            parent=tab7,
            header="TARGETED CLIP ANNOTATOR",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ADVANCED_LBL.value,
        )
        targeted_clip_annotator_note = Label(
            targeted_clip_annotator_frm,
            text="A bout annotator that creates annotated clips from videos associated with ML results.",
        )
        targeted_clip_annotator_btn = Button(
            targeted_clip_annotator_frm,
            text="Select video",
            command=lambda: select_labelling_video_targeted_clips(
                config_path=self.config_path
            ),
        )

        lbl_tools_frm = LabelFrame(
            tab7,
            text="LABELLING TOOLS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        visualize_annotation_img_btn = Button(
            lbl_tools_frm,
            text="Visualize annotations",
            fg="blue",
            command=lambda: ExtractAnnotationFramesPopUp(config_path=self.config_path),
        )
        third_party_annotations_btn = Button(
            lbl_tools_frm,
            text="Append third-party annotations",
            fg="purple",
            command=lambda: ThirdPartyAnnotatorAppenderPopUp(
                config_path=self.config_path
            ),
        )
        remove_roi_features_from_annotation_set = Button(
            lbl_tools_frm,
            text="Remove ROI features from label set",
            fg="darkred",
            command=lambda: RemoveROIFeaturesPopUp(
                config_path=self.config_path, dataset="targets_inserted"
            ),
        )

        label_trainmachinemodel = CreateLabelFrameWithIcon(
            parent=tab8,
            header="TRAIN MACHINE MODELS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.TRAIN_ML_MODEL.value,
        )
        button_trainmachinesettings = Button(
            label_trainmachinemodel,
            text="SETTINGS",
            fg="darkorange",
            command=self.trainmachinemodelsetting,
        )
        button_trainmachinemodel = Button(
            label_trainmachinemodel,
            text="TRAIN SINGLE MODEL (GLOBAL ENVIRONMENT)",
            fg="blue",
            command=lambda: self.train_single_model(config_path=self.config_path),
        )

        button_train_multimodel = Button(
            label_trainmachinemodel,
            text="TRAIN MULTIPLE MODELS (ONE FOR EACH SAVED SETTING)",
            fg="green",
            command=lambda: self.train_multiple_models_from_meta(
                config_path=self.config_path
            ),
        )

        label_model_validation = CreateLabelFrameWithIcon(
            parent=tab9,
            header="VALIDATE MODEL ON SINGLE VIDEO",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value,
        )
        # label_model_validation = LabelFrame(tab9, text='VALIDATE MODEL ON SINGLE VIDEO', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='blue')
        self.csvfile = FileSelect(
            label_model_validation,
            "SELECT DATA FEATURE FILE",
            color="blue",
            lblwidth=30,
            file_types=[("SimBA CSV", "*.csv"), ("SimBA PARQUET", "*.parquet")],
            initialdir=os.path.join(
                self.project_path, Paths.FEATURES_EXTRACTED_DIR.value
            ),
        )
        self.modelfile = FileSelect(
            label_model_validation,
            "SELECT MODEL FILE",
            color="blue",
            lblwidth=30,
            initialdir=self.project_path,
        )
        button_runvalidmodel = Button(
            label_model_validation,
            text="RUN MODEL",
            fg="blue",
            command=lambda: threading.Thread(
                target=self.validate_model_first_step
            ).start(),
        )

        button_generateplot = Button(
            label_model_validation,
            text="INTERACTIVE PROBABILITY PLOT",
            fg="blue",
            command=lambda: self.launch_interactive_plot(),
        )
        self.dis_threshold = Entry_Box(
            label_model_validation, "DISCRIMINATION THRESHOLD (0.0-1.0):", "30"
        )
        self.min_behaviorbout = Entry_Box(
            label_model_validation,
            "MINIMUM BOUT LENGTH (MS):",
            "30",
            validation="numeric",
        )
        button_validate_model = Button(
            label_model_validation,
            text="CREATE VALIDATION VIDEO",
            fg="blue",
            command=lambda: ValidationVideoPopUp(
                config_path=config_path, simba_main_frm=self
            ),
        )

        label_runmachinemodel = CreateLabelFrameWithIcon(
            parent=tab9,
            header="RUN MACHINE MODEL",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.SCENARIO_2.value,
        )
        button_run_rfmodelsettings = Button(
            label_runmachinemodel,
            text="MODEL SETTINGS",
            fg="green",
            compound="left",
            image=self.btn_icons["settings"]["img"],
            command=lambda: SetMachineModelParameters(config_path=self.config_path),
        )
        button_run_rfmodelsettings.image = self.btn_icons["settings"]["img"]

        button_runmachinemodel = Button(
            label_runmachinemodel,
            text="RUN MODELS",
            fg="green",
            compound="left",
            image=self.btn_icons["clf"]["img"],
            command=lambda: threading.Thread(target=self.runrfmodel).start(),
        )
        button_runmachinemodel.image = self.btn_icons["clf"]["img"]

        kleinberg_button = Button(
            label_runmachinemodel,
            text="KLEINBERG SMOOTHING",
            fg="green",
            command=lambda: KleinbergPopUp(config_path=self.config_path),
        )
        fsttc_button = Button(
            label_runmachinemodel,
            text="FSTTC",
            fg="green",
            command=lambda: FSTTCPopUp(config_path=self.config_path),
        )
        mutual_exclusivity = Button(
            label_runmachinemodel,
            text="MUTUAL EXCLUSIVITY CORRECTION",
            fg="green",
            command=lambda: MutualExclusivityPupUp(config_path=self.config_path),
        )

        label_machineresults = CreateLabelFrameWithIcon(
            parent=tab9,
            header="ANALYZE MACHINE RESULTS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ANALYZE_ML_RESULTS.value,
        )
        button_process_datalog = Button(
            label_machineresults,
            text="ANALYZE MACHINE PREDICTIONS: AGGREGATES",
            fg="blue",
            command=lambda: ClfDescriptiveStatsPopUp(config_path=self.config_path),
        )
        button_process_movement = Button(
            label_machineresults,
            text="ANALYZE DISTANCES/VELOCITY: AGGREGATES",
            fg="blue",
            command=lambda: MovementAnalysisPopUp(config_path=self.config_path),
        )
        button_movebins = Button(
            label_machineresults,
            text="ANALYZE DISTANCES/VELOCITY: TIME BINS",
            fg="blue",
            command=lambda: MovementAnalysisTimeBinsPopUp(config_path=self.config_path),
        )
        button_classifierbins = Button(
            label_machineresults,
            text="ANALYZE MACHINE PREDICTIONS: TIME-BINS",
            fg="blue",
            command=lambda: TimeBinsClfPopUp(config_path=self.config_path),
        )
        button_classifier_ROI = Button(
            label_machineresults,
            text="ANALYZE MACHINE PREDICTION: BY ROI",
            fg="blue",
            command=lambda: ClfByROIPopUp(config_path=self.config_path),
        )
        button_severity = Button(
            label_machineresults,
            text="ANALYZE MACHINE PREDICTION: BY SEVERITY",
            fg="blue",
            command=lambda: AnalyzeSeverityPopUp(config_path=self.config_path),
        )

        visualization_frm = CreateLabelFrameWithIcon(
            parent=tab10,
            header="DATA VISUALIZATIONS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VISUALIZATION.value,
        )
        sklearn_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE CLASSIFICATIONS",
            fg="black",
            command=lambda: SklearnVisualizationPopUp(config_path=self.config_path),
        )
        sklearn_visualization_btn.grid(row=0, column=0, sticky=NW)
        gantt_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE GANTT",
            fg="blue",
            command=lambda: GanttPlotPopUp(config_path=self.config_path),
        )
        gantt_visualization_btn.grid(row=1, column=0, sticky=NW)
        probability_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE PROBABILITIES",
            fg="green",
            command=lambda: VisualizeClassificationProbabilityPopUp(
                config_path=self.config_path
            ),
        )
        probability_visualization_btn.grid(row=2, column=0, sticky=NW)
        path_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE PATHS",
            fg="orange",
            command=lambda: PathPlotPopUp(config_path=self.config_path),
        )
        path_visualization_btn.grid(row=3, column=0, sticky=NW)
        distance_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE DISTANCES",
            fg="red",
            command=lambda: DistancePlotterPopUp(config_path=self.config_path),
        )
        distance_visualization_btn.grid(row=4, column=0, sticky=NW)
        heatmap_clf_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE CLASSIFICATION HEATMAPS",
            fg="pink",
            command=lambda: HeatmapClfPopUp(config_path=self.config_path),
        )
        heatmap_clf_visualization_btn.grid(row=5, column=0, sticky=NW)
        data_plot_visualization_btn = Button(
            visualization_frm,
            text="VISUALIZE DATA PLOTS",
            fg="purple",
            command=lambda: DataPlotterPopUp(config_path=self.config_path),
        )
        data_plot_visualization_btn.grid(row=6, column=0, sticky=NW)
        clf_validation_btn = Button(
            visualization_frm,
            text="CLASSIFIER VALIDATION CLIPS",
            fg="blue",
            command=lambda: ClassifierValidationPopUp(config_path=self.config_path),
        )
        clf_validation_btn.grid(row=7, column=0, sticky=NW)
        merge_frm = CreateLabelFrameWithIcon(
            parent=tab10,
            header="MERGE FRAMES",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.CONCAT_VIDEOS.value,
        )
        merge_frm_btn = Button(
            merge_frm,
            text="MERGE FRAMES",
            fg="black",
            command=lambda: ConcatenatorPopUp(config_path=self.config_path),
        )
        plotlyInterface = CreateLabelFrameWithIcon(
            parent=tab10,
            header="PLOTLY / DASH",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.PLOTLY.value,
        )
        plotlyInterfaceTitles = [
            "Sklearn results",
            "Time bin analyses",
            "Probabilities",
            "Severity analysis",
        ]
        toIncludeVar = []
        for i in range(len(plotlyInterfaceTitles) + 1):
            toIncludeVar.append(IntVar())
        plotlyCheckbox = [0] * (len(plotlyInterfaceTitles) + 1)
        for i in range(len(plotlyInterfaceTitles)):
            plotlyCheckbox[i] = Checkbutton(
                plotlyInterface, text=plotlyInterfaceTitles[i], variable=toIncludeVar[i]
            )
            plotlyCheckbox[i].grid(row=i, sticky=W)

        button_save_plotly_file = Button(
            plotlyInterface,
            text="Save SimBA / Plotly dataset",
            command=lambda: self.generateSimBPlotlyFile(toIncludeVar),
        )
        self.plotly_file = FileSelect(
            plotlyInterface,
            "SimBA Dashboard file (H5)",
            title="Select SimBA/Plotly dataset (h5)",
        )
        self.groups_file = FileSelect(
            plotlyInterface, "SimBA Groups file (CSV)", title="Select groups file (csv"
        )
        button_open_plotly_interface = Button(
            plotlyInterface,
            text="Open SimBA / Plotly dataset",
            fg="black",
            command=lambda: [self.open_plotly_interface("http://127.0.0.1:8050")],
        )

        # addons
        lbl_addon = LabelFrame(
            tab11,
            text="SimBA Expansions",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        button_bel = Button(
            lbl_addon,
            text="Pup retrieval - Analysis Protocol 1",
            fg="blue",
            command=lambda: PupRetrievalPopUp(config_path=self.config_path),
        )
        cue_light_analyser_btn = Button(
            lbl_addon,
            text="Cue light analysis",
            fg="red",
            command=lambda: CueLightAnalyzerMenu(config_path=self.config_path),
        )
        anchored_roi_analysis_btn = Button(
            lbl_addon,
            text="Animal-anchored ROI analysis",
            fg="orange",
            command=lambda: BoundaryMenus(config_path=self.config_path),
        )

        self.create_import_videos_menu(parent_frm=import_frm, idx_row=0, idx_column=0)
        self.create_import_pose_menu(parent_frm=import_frm, idx_row=1, idx_column=0)
        further_methods_frm.grid(row=0, column=1, sticky=NW, pady=5, padx=5)
        extract_frm_btn.grid(row=1, column=0, sticky=NW)
        import_frm_dir_btn.grid(row=2, column=0, sticky=NW)
        add_clf_btn.grid(row=3, column=0, sticky=NW)
        remove_clf_btn.grid(row=4, column=0, sticky=NW)
        archive_files_btn.grid(row=5, column=0, sticky=NW)
        reverse_btn.grid(row=6, column=0, sticky=NW)
        interpolate_btn.grid(row=7, column=0, sticky=NW)
        smooth_btn.grid(row=8, column=0, sticky=NW)

        label_setscale.grid(row=0, sticky=NW, pady=5, padx=5)
        self.distance_in_mm_eb.grid(row=0, column=0, sticky=NW)
        button_setdistanceinmm.grid(row=0, column=1, sticky=NW)
        button_setscale.grid(row=1, column=0, sticky=NW)

        label_outliercorrection.grid(row=0, sticky=W)
        button_settings_outlier.grid(row=0, sticky=W)
        button_outliercorrection.grid(row=1, sticky=W)
        button_skipOC.grid(row=2, sticky=W, pady=5)

        label_extractfeatures.grid(row=0, column=0, sticky=NW)
        button_extractfeatures.grid(row=1, column=0, sticky=NW)
        labelframe_usrdef.grid(row=0, column=0, sticky=NW, pady=15)
        userscript.grid(row=1, column=0, sticky=NW)
        self.scriptfile.grid(row=2, column=0, sticky=NW)

        roi_feature_frm.grid(row=1, column=0, sticky=NW)
        append_roi_features_by_animal.grid(row=0, column=0, sticky=NW)
        append_roi_features_by_body_part.grid(row=1, column=0, sticky=NW)
        remove_roi_features_from_feature_set.grid(row=2, column=0, sticky=NW)

        feature_tools_frm.grid(row=2, column=0, sticky=NW)
        compute_feature_subset_btn.grid(row=0, column=0, sticky=NW)

        label_behavior_frm.grid(row=5, sticky=W)
        select_video_btn_new.grid(row=0, sticky=W)
        select_video_btn_continue.grid(row=1, sticky=W)

        label_pseudo.grid(row=6, sticky=W, pady=10)

        pLabel_framedir.grid(row=0, sticky=W)
        pseudo_intructions_lbl_1.grid(row=1, sticky=W)
        pseudo_intructions_lbl_2.grid(row=2, sticky=W)
        plabelframe_threshold.grid(row=4, sticky=W)
        pseudo_lbl_btn.grid(row=5, sticky=W)

        label_adv_label.grid(row=7, column=0, sticky=NW)
        label_adv_note_1.grid(row=0, column=0, sticky=NW)
        label_adv_note_2.grid(row=1, column=0, sticky=NW)
        adv_label_btn_new.grid(row=3, column=0, sticky=NW)
        adv_label_btn_continue.grid(row=4, column=0, sticky=NW)

        targeted_clip_annotator_frm.grid(row=8, column=0, sticky=NW)
        targeted_clip_annotator_note.grid(row=0, column=0, sticky=NW)
        targeted_clip_annotator_btn.grid(row=1, column=0, sticky=NW)

        label_thirdpartyann.grid(row=9, sticky=W)
        button_importmars.grid(row=0, column=0, sticky=NW)
        button_importboris.grid(row=1, column=0, sticky=NW)
        button_importsolomon.grid(row=2, column=0, sticky=NW)
        button_importethovision.grid(row=0, column=1, sticky=NW)
        button_importdeepethogram.grid(row=1, column=1, sticky=NW)
        import_observer_btn.grid(row=2, column=1, sticky=NW)

        lbl_tools_frm.grid(row=10, column=0, sticky=NW)
        visualize_annotation_img_btn.grid(row=0, column=0, sticky=NW)
        third_party_annotations_btn.grid(row=1, column=0, sticky=NW)
        remove_roi_features_from_annotation_set.grid(row=2, column=0, sticky=NW)

        label_trainmachinemodel.grid(row=6, sticky=W)
        button_trainmachinesettings.grid(row=0, column=0, sticky=NW, padx=5)
        button_trainmachinemodel.grid(row=1, column=0, sticky=NW, padx=5)
        button_train_multimodel.grid(row=2, column=0, sticky=NW, padx=5)

        label_model_validation.grid(row=7, sticky=W, pady=5)
        self.csvfile.grid(row=0, sticky=W)
        self.modelfile.grid(row=1, sticky=W)
        button_runvalidmodel.grid(row=2, sticky=W)
        button_generateplot.grid(row=3, sticky=W)
        self.dis_threshold.grid(row=4, sticky=W)
        self.min_behaviorbout.grid(row=5, sticky=W)
        button_validate_model.grid(row=6, sticky=W)

        label_runmachinemodel.grid(row=8, sticky=NW)
        button_run_rfmodelsettings.grid(row=0, sticky=NW)
        button_runmachinemodel.grid(row=1, sticky=NW)
        kleinberg_button.grid(row=2, sticky=NW)
        fsttc_button.grid(row=3, sticky=NW)
        mutual_exclusivity.grid(row=4, sticky=NW)

        label_machineresults.grid(row=9, sticky=W, pady=5)
        button_process_datalog.grid(row=2, column=0, sticky=W, padx=3)
        button_process_movement.grid(row=2, column=1, sticky=W, padx=3)
        button_movebins.grid(row=3, column=1, sticky=W, padx=3)
        button_classifierbins.grid(row=3, column=0, sticky=W, padx=3)
        button_classifier_ROI.grid(row=4, column=0, sticky=W, padx=3)
        button_severity.grid(row=4, column=1, sticky=W, padx=3)

        visualization_frm.grid(row=11, column=0, sticky=W + N, padx=5)
        merge_frm.grid(row=11, column=2, sticky=W + N, padx=5)
        merge_frm_btn.grid(row=0, sticky=NW, padx=5)

        plotlyInterface.grid(row=11, column=3, sticky=W + N, padx=5)
        button_save_plotly_file.grid(row=10, sticky=W)
        self.plotly_file.grid(row=11, sticky=W)
        self.groups_file.grid(row=12, sticky=W)
        button_open_plotly_interface.grid(row=13, sticky=W)

        lbl_addon.grid(row=15, sticky=W)
        button_bel.grid(row=0, sticky=W)
        cue_light_analyser_btn.grid(row=1, sticky=NW)
        anchored_roi_analysis_btn.grid(row=2, sticky=NW)

        if UNSUPERVISED:
            from simba.unsupervised.unsupervised_main import UnsupervisedGUI

            unsupervised_btn = Button(
                lbl_addon,
                text="Unsupervised analysis",
                fg="purple",
                command=lambda: UnsupervisedGUI(config_path=self.config_path),
            )
            unsupervised_btn.grid(row=3, sticky=NW)

    def create_video_info_table(self):
        video_info_tabler = VideoInfoTable(config_path=self.config_path)
        video_info_tabler.create_window()

    def initiate_skip_outlier_correction(self):
        outlier_correction_skipper = OutlierCorrectionSkipper(
            config_path=self.config_path
        )
        outlier_correction_skipper.run()

    def validate_model_first_step(self):
        _ = InferenceValidation(
            config_path=self.config_path,
            input_file_path=self.csvfile.file_path,
            clf_path=self.modelfile.file_path,
        )

    def train_single_model(self, config_path=None):
        model_trainer = TrainRandomForestClassifier(config_path=config_path)
        model_trainer.run()
        model_trainer.save()

    def train_multiple_models_from_meta(self, config_path=None):
        model_trainer = GridSearchRandomForestClassifier(config_path=config_path)
        model_trainer.run()

    def importBoris(self):
        ann_folder = askdirectory()
        boris_appender = BorisAppender(
            config_path=self.config_path, data_dir=ann_folder
        )
        boris_appender.create_boris_master_file()
        threading.Thread(target=boris_appender.run).start()

    def importSolomon(self):
        ann_folder = askdirectory()
        solomon_importer = SolomonImporter(
            config_path=self.config_path, data_dir=ann_folder
        )
        threading.Thread(target=solomon_importer.run).start()

    def import_ethovision(self):
        ann_folder = askdirectory()
        ethovision_importer = ImportEthovision(
            config_path=self.config_path, data_dir=ann_folder
        )
        threading.Thread(target=ethovision_importer.run).start()

    def import_deepethogram(self):
        ann_folder = askdirectory()
        deepethogram_importer = DeepEthogramImporter(
            config_path=self.config_path, data_dir=ann_folder
        )
        threading.Thread(target=deepethogram_importer.run).start()

    def import_noldus_observer(self):
        directory = askdirectory()
        noldus_observer_importer = NoldusObserverImporter(
            config_path=self.config_path, data_dir=directory
        )
        threading.Thread(target=noldus_observer_importer.run).start()

    def importMARS(self):
        bento_dir = askdirectory()
        bento_appender = BentoAppender(config_path=self.config_path, data_dir=bento_dir)
        threading.Thread(target=bento_appender.run).start()

    def choose_animal_bps(self):
        if hasattr(self, "path_plot_animal_bp_frm"):
            self.path_plot_animal_bp_frm.destroy()
        self.path_plot_animal_bp_frm = LabelFrame(
            self.path_plot_frm,
            text="CHOOSE ANIMAL BODY-PARTS",
            font=("Helvetica", 10, "bold"),
            pady=5,
            padx=5,
        )
        self.path_plot_bp_dict = {}
        for animal_cnt in range(int(self.path_animal_cnt_dropdown.getChoices())):
            self.path_plot_bp_dict[animal_cnt] = DropDownMenu(
                self.path_plot_animal_bp_frm,
                "Animal {} bodypart".format(str(animal_cnt + 1)),
                self.bp_set,
                "15",
            )
            self.path_plot_bp_dict[animal_cnt].setChoices(self.bp_set[animal_cnt])
            self.path_plot_bp_dict[animal_cnt].grid(row=animal_cnt, sticky=NW)
        self.path_plot_animal_bp_frm.grid(row=2, column=0, sticky=NW)

    def launch_interactive_plot(self):
        interactive_grapher = InteractiveProbabilityGrapher(
            config_path=self.config_path,
            file_path=self.csvfile.file_path,
            model_path=self.modelfile.file_path,
        )
        interactive_grapher.run()

    def generateSimBPlotlyFile(self, var):
        inputList = []
        for i in var:
            inputList.append(i.get())
        stdout_warning(msg="SimBA plotly interface is not available.")
        pass

    def open_plotly_interface(self, url):
        try:
            self.p.kill()
            self.p2.kill()
        except:
            print("Starting plotly")
        # get h5 file path and csv file path
        filePath, groupPath = self.plotly_file.file_path, self.groups_file.file_path

        # print file read
        if filePath.endswith(".h5"):
            print("Reading in", os.path.basename(filePath))
        elif groupPath.endswith(".csv"):
            print("Reading in", os.path.basename(groupPath))

        self.p = subprocess.Popen(
            [
                sys.executable,
                os.path.join(
                    os.path.dirname(__file__), "dash_app", "SimBA_dash_app.py"
                ),
                filePath,
                groupPath,
            ]
        )
        # csvPath = os.path.join(os.path.dirname(self.config_path),'csv')
        # p = subprocess.Popen([sys.executable, r'simba\SimBA_dash_app.py', filePath, groupPath, csvPath])
        wait_for_internet_connection(url)
        self.p2 = subprocess.Popen(
            [
                sys.executable,
                os.path.join(
                    os.path.dirname(__file__), "dash_app", "run_dash_tkinter.py"
                ),
                url,
            ]
        )
        subprocess_children = [self.p, self.p2]
        atexit.register(terminate_children, subprocess_children)

    def runrfmodel(self):
        rf_model_runner = InferenceBatch(config_path=self.config_path)
        rf_model_runner.run()

    def trainmachinemodelsetting(self):
        _ = MachineModelSettingsPopUp(config_path=self.config_path)

    def run_feature_extraction(self):
        print("Running feature extraction...")
        print(
            f"Pose-estimation body part setting for feature extraction: {str(self.animal_cnt)} animals {str(self.pose_setting)} body-parts..."
        )
        feature_extractor_classes = get_bp_config_code_class_pairs()
        if self.user_defined_var.get():
            custom_feature_extractor = CustomFeatureExtractor(
                extractor_file_path=self.scriptfile.file_path,
                config_path=self.config_path,
            )
            custom_feature_extractor.run()
            stdout_success(
                msg="Custom feature extraction complete!",
                source=self.__class__.__name__,
            )
        else:
            if self.pose_setting not in feature_extractor_classes.keys():
                raise InvalidInputError(
                    msg=f"The project pose-configuration key is set to {self.pose_setting} which is invalid. OPTIONS: {list(feature_extractor_classes.keys())}. Check the pose-estimation setting in the project_config.ini",
                    source=self.__class__.__name__,
                )
            if self.pose_setting == "8":
                feature_extractor = feature_extractor_classes[self.pose_setting][
                    self.animal_cnt
                ](config_path=self.config_path)
            else:
                feature_extractor = feature_extractor_classes[self.pose_setting](
                    config_path=self.config_path
                )
            feature_extractor.run()

    def set_distance_mm(self):
        check_int(
            name="DISTANCE IN MILLIMETER",
            value=self.distance_in_mm_eb.entry_get,
            min_value=1,
        )
        self.config.set(
            "Frame settings", "distance_mm", self.distance_in_mm_eb.entry_get
        )
        with open(self.config_path, "w") as f:
            self.config.write(f)

    def correct_outlier(self):
        outlier_correcter_movement = OutlierCorrecterMovement(
            config_path=self.config_path
        )
        outlier_correcter_movement.run()
        outlier_correcter_location = OutlierCorrecterLocation(
            config_path=self.config_path
        )
        outlier_correcter_location.run()
        stdout_success(
            msg='Outlier corrected files located in "project_folder/csv/outlier_corrected_movement_location" directory',
            source=self.__class__.__name__,
        )

    def callback(self, url):
        webbrowser.open_new(url)


class App(object):
    def __init__(self):
        bg_path = os.path.join(os.path.dirname(__file__), Paths.BG_IMG_PATH.value)
        emojis = get_emojis()
        icon_path_windows = os.path.join(
            os.path.dirname(__file__), Paths.LOGO_ICON_WINDOWS_PATH.value
        )
        icon_path_darwin = os.path.join(
            os.path.dirname(__file__), Paths.LOGO_ICON_DARWIN_PATH.value
        )
        self.menu_icons = get_icons_paths()
        self.root = Tk()
        self.root.title("SimBA")
        self.root.minsize(750, 750)
        self.root.geometry(Formats.ROOT_WINDOW_SIZE.value)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        if currentPlatform == OS.WINDOWS.value:
            self.root.iconbitmap(icon_path_windows)
        if currentPlatform == OS.MAC.value:
            self.root.iconphoto(
                False, ImageTk.PhotoImage(PIL.Image.open(icon_path_darwin))
            )
        for k in self.menu_icons.keys():
            self.menu_icons[k]["img"] = ImageTk.PhotoImage(
                image=PIL.Image.open(
                    os.path.join(
                        os.path.dirname(__file__), self.menu_icons[k]["icon_path"]
                    )
                )
            )
        bg_img = ImageTk.PhotoImage(file=bg_path)
        background = Label(self.root, image=bg_img, bd=0, bg="white")
        background.pack(fill="both", expand=True)
        background.image = bg_img

        menu = Menu(self.root)
        self.root.config(menu=menu)

        file_menu = Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Create a new project",
            compound="left",
            image=self.menu_icons["create"]["img"],
            command=lambda: ProjectCreatorPopUp(),
        )
        file_menu.add_command(
            label="Load project",
            compound="left",
            image=self.menu_icons["load"]["img"],
            command=lambda: LoadProjectPopUp(),
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Restart",
            compound="left",
            image=self.menu_icons["restart"]["img"],
            command=lambda: self.restart(),
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
            compound="left",
            image=self.menu_icons["exit"]["img"],
            command=self.root.destroy,
        )

        batch_process_menu = Menu(menu)
        menu.add_cascade(label="Process Videos", menu=batch_process_menu)
        batch_process_menu.add_command(
            label="Batch pre-process videos",
            compound="left",
            image=self.menu_icons["factory"]["img"],
            command=lambda: BatchPreProcessPopUp(),
        )

        video_process_menu = Menu(menu)
        fps_menu = Menu(video_process_menu)
        fps_menu.add_command(
            label="Change fps for single video", command=ChangeFpsSingleVideoPopUp
        )
        fps_menu.add_command(
            label="Change fps for multiple videos", command=ChangeFpsMultipleVideosPopUp
        )

        menu.add_cascade(label="Tools", menu=video_process_menu)
        video_process_menu.add_cascade(label="Change fps...", compound="left", image=self.menu_icons["fps"]["img"], menu=fps_menu)

        clip_video_menu = Menu(menu)
        clip_video_menu.add_command(label="Clip single video", command=ClipVideoPopUp)
        clip_video_menu.add_command(
            label="Clip multiple videos",
            command=InitiateClipMultipleVideosByTimestampsPopUp,
        )

        clip_video_menu.add_command(
            label="Clip video into multiple videos", command=MultiShortenPopUp
        )
        clip_video_menu.add_command(
            label="Clip single video by frame numbers",
            command=ClipSingleVideoByFrameNumbers,
        )
        clip_video_menu.add_command(
            label="Clip multiple videos by frame numbers",
            command=InitiateClipMultipleVideosByFrameNumbersPopUp,
        )

        video_process_menu.add_cascade(
            label="Clip videos...",
            compound="left",
            image=self.menu_icons["clip"]["img"],
            menu=clip_video_menu,
        )

        crop_video_menu = Menu(menu)
        crop_video_menu.add_command(label="Crop videos", compound="left", image=self.menu_icons["crop"]["img"], command=CropVideoPopUp)
        crop_video_menu.add_command(label="Crop videos (circles)", compound="left", image=self.menu_icons["circle"]["img"], command=CropVideoCirclesPopUp)
        crop_video_menu.add_command(label="Crop videos (polygons)", compound="left", image=self.menu_icons["polygon"]["img"], command=CropVideoPolygonsPopUp)
        crop_video_menu.add_command(label="Multi-crop", compound="left", image=self.menu_icons["crop"]["img"], command=MultiCropPopUp)
        video_process_menu.add_cascade(label="Crop videos...", compound="left", image=self.menu_icons["crop"]["img"], menu=crop_video_menu)

        format_menu = Menu(video_process_menu)
        img_format_menu = Menu(format_menu)
        video_format_menu = Menu(format_menu)
        img_format_menu.add_command(label="Convert image directory to PNG", command=Convert2PNGPopUp)
        img_format_menu.add_command(label="Convert image directory to JPEG", command=Convert2jpegPopUp)
        img_format_menu.add_command(label="Convert image directory to BMP", command=Convert2bmpPopUp)
        img_format_menu.add_command(label="Convert image directory to TIFF", command=Convert2TIFFPopUp)
        img_format_menu.add_command(label="Convert image directory to WEBP", command=Convert2WEBPPopUp)
        video_format_menu.add_command(label="Convert videos to MP4", command=Convert2MP4PopUp)
        video_format_menu.add_command(label="Convert videos to AVI", command=Convert2AVIPopUp)
        video_format_menu.add_command(label="Convert videos to WEBM", command=Convert2WEBMPopUp)
        video_format_menu.add_command(label="Convert videos to MOV", command=Convert2MOVPopUp)
        format_menu.add_cascade(label="Change image formats...", compound="left", menu=img_format_menu)
        format_menu.add_cascade(label="Change video formats...", compound="left", menu=video_format_menu)
        video_process_menu.add_cascade(label="Change formats...", compound="left", image=self.menu_icons["convert"]["img"], menu=format_menu)
        clahe_menu = Menu(video_process_menu)
        clahe_menu.add_command(label="CLAHE enhance videos", command=CLAHEPopUp)
        clahe_menu.add_command(label="Interactively CLAHE enhance videos", command=InteractiveClahePopUp)
        video_process_menu.add_cascade(label="CLAHE enhance videos...", compound="left", image=self.menu_icons["clahe"]["img"], menu=clahe_menu)

        video_process_menu.add_cascade(label="Concatenate multiple videos", compound="left", image=self.menu_icons["concat"]["img"], command=lambda: ConcatenatorPopUp(config_path=None))
        video_process_menu.add_cascade(
            label="Concatenate two videos",
            compound="left",
            image=self.menu_icons["concat"]["img"],
            command=ConcatenatingVideosPopUp,
        )
        video_process_menu.add_command(
            label="Convert to grayscale",
            compound="left",
            image=self.menu_icons["grey"]["img"],
            command=lambda: GreyscaleSingleVideoPopUp(),
        )
        video_process_menu.add_command(
            label="Convert ROI definitions",
            compound="left",
            image=self.menu_icons["roi"]["img"],
            command=lambda: ConvertROIDefinitionsPopUp(),
        )
        convert_data_menu = Menu(video_process_menu)
        convert_data_menu.add_command(label="Convert CSV to parquet", command=Csv2ParquetPopUp)
        convert_data_menu.add_command(label="Convert parquet o CSV", command=Parquet2CsvPopUp)

        video_process_menu.add_cascade(
            label="Convert working file type...",
            compound="left",
            image=self.menu_icons["change"]["img"],
            menu=convert_data_menu,
        )

        video_process_menu.add_command(
            label="Create path plot",
            compound="left",
            image=self.menu_icons["path"]["img"],
            command=MakePathPlotPopUp,
        )

        downsample_video_menu = Menu(video_process_menu)
        downsample_video_menu.add_command(label="Down-sample single video", command=DownsampleSingleVideoPopUp)
        downsample_video_menu.add_command(label="Down-sample multiple videos", command=DownsampleMultipleVideosPopUp)
        video_process_menu.add_cascade(label="Down-sample video...", compound="left", image=self.menu_icons["sample"]["img"], menu=downsample_video_menu)
        video_process_menu.add_cascade(
            label="Drop body-parts from tracking data",
            compound="left",
            image=self.menu_icons["trash"]["img"],
            command=DropTrackingDataPopUp,
        )
        extract_frames_menu = Menu(video_process_menu)
        extract_frames_menu.add_command(
            label="Extract defined frames", command=ExtractSpecificFramesPopUp
        )
        extract_frames_menu.add_command(
            label="Extract frames", command=ExtractAllFramesPopUp
        )
        extract_frames_menu.add_command(
            label="Extract frames from seq files", command=ExtractSEQFramesPopUp
        )
        video_process_menu.add_cascade(
            label="Extract frames...",
            compound="left",
            image=self.menu_icons["frames"]["img"],
            menu=extract_frames_menu,
        )

        video_process_menu.add_command(
            label="Generate gifs",
            compound="left",
            image=self.menu_icons["gif"]["img"],
            command=CreateGIFPopUP,
        )

        video_process_menu.add_command(
            label="Get mm/ppx",
            compound="left",
            image=self.menu_icons["calipher"]["img"],
            command=CalculatePixelsPerMMInVideoPopUp,
        )

        video_process_menu.add_command(label="Change video brightness / contrast", compound="left", image=self.menu_icons["brightness"]["img"], command=BrightnessContrastPopUp)


        video_process_menu.add_command(
            label="Merge frames to video",
            compound="left",
            image=self.menu_icons["merge"]["img"],
            command=MergeFrames2VideoPopUp,
        )
        video_process_menu.add_command(
            label="Print classifier info...",
            compound="left",
            image=self.menu_icons["print"]["img"],
            command=PrintModelInfoPopUp,
        )

        video_process_menu.add_cascade(
            label="Reorganize Tracking Data",
            compound="left",
            image=self.menu_icons["reorganize"]["img"],
            command=PoseReorganizerPopUp,
        )
        video_process_menu.add_command(
            label="Rotate videos",
            compound="left",
            image=self.menu_icons["rotate"]["img"],
            command=VideoRotatorPopUp,
        )

        superimpose_menu = Menu(menu)
        superimpose_menu.add_command(label="Superimpose frame numbers on videos", command=SuperImposeFrameCountPopUp)
        superimpose_menu.add_command(label="Superimpose watermark on videos", command=SuperimposeWatermarkPopUp)
        superimpose_menu.add_command(label="Superimpose timer on videos", command=SuperimposeTimerPopUp)
        superimpose_menu.add_command(label="Superimpose progress-bar on videos", command=SuperimposeProgressBarPopUp)
        video_process_menu.add_cascade(label="Superimpose on videos...", compound="left", image=self.menu_icons["superimpose"]["img"], menu=superimpose_menu)

        video_process_menu.add_command(label="Temporal join videos", compound="left", image=self.menu_icons["stopwatch"]["img"], command=VideoTemporalJoinPopUp)

        video_process_menu.add_cascade(
            label="Visualize pose-estimation in folder...",
            compound="left",
            image=self.menu_icons["visualize"]["img"],
            command=VisualizePoseInFolderPopUp,
        )

        help_menu = Menu(menu)
        menu.add_cascade(label="Help", menu=help_menu)

        links_menu = Menu(help_menu)
        links_menu.add_command(
            label="Download weights",
            command=lambda: webbrowser.open_new(str(r"https://osf.io/sr3ck/")),
        )
        links_menu.add_command(
            label="Download classifiers",
            command=lambda: webbrowser.open_new(str(r"https://osf.io/kwge8/")),
        )
        links_menu.add_command(
            label="Ex. feature list",
            command=lambda: webbrowser.open_new(
                str(
                    r"https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv"
                )
            ),
        )
        links_menu.add_command(
            label="SimBA github",
            command=lambda: webbrowser.open_new(
                str(r"https://github.com/sgoldenlab/simba")
            ),
        )
        links_menu.add_command(
            label="Gitter Chatroom",
            command=lambda: webbrowser.open_new(
                str(r"https://gitter.im/SimBA-Resource/community")
            ),
        )
        links_menu.add_command(
            label="Install FFmpeg",
            command=lambda: webbrowser.open_new(
                str(r"https://m.wikihow.com/Install-FFmpeg-on-Windows")
            ),
        )
        links_menu.add_command(
            label="Install graphviz",
            command=lambda: webbrowser.open_new(
                str(
                    r"https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software"
                )
            ),
        )
        help_menu.add_cascade(
            label="Links",
            menu=links_menu,
            compound="left",
            image=self.menu_icons["link"]["img"],
        )
        help_menu.add_command(
            label="About",
            compound="left",
            image=self.menu_icons["about"]["img"],
            command=AboutSimBAPopUp,
        )

        self.frame = Frame(background, bd=2, relief=SUNKEN, width=750, height=300)
        self.r_click_menu = Menu(self.root, tearoff=0)
        self.r_click_menu.add_command(
            label="Copy selection", command=lambda: self.copy_selection_to_clipboard()
        )
        self.r_click_menu.add_command(
            label="Copy all", command=lambda: self.copy_all_to_clipboard()
        )
        self.r_click_menu.add_command(
            label="Paste", command=lambda: self.paste_to_txt()
        )
        self.r_click_menu.add_separator()
        self.r_click_menu.add_command(label="Clear", command=lambda: self.clean_txt())
        y_sb = Scrollbar(self.frame, orient=VERTICAL)
        self.frame.pack(expand=True)
        self.txt = Text(
            self.frame,
            bg="white",
            insertborderwidth=2,
            height=30,
            width=100,
            yscrollcommand=y_sb,
        )
        if currentPlatform == OS.WINDOWS.value:
            self.txt.bind("<Button-3>", self.show_right_click_pop_up)
        if currentPlatform == OS.MAC.value:
            self.txt.bind("<Button-2>", self.show_right_click_pop_up)
        self.txt.tag_configure(
            TagNames.GREETING.value,
            justify="center",
            foreground="blue",
            font=("Rockwell", 16, "bold"),
        )
        self.txt.tag_configure(
            TagNames.ERROR.value,
            justify="left",
            foreground="red",
            font=Formats.TKINTER_FONT.value,
        )
        self.txt.tag_configure(
            TagNames.STANDARD.value,
            justify="left",
            foreground="black",
            font=Formats.TKINTER_FONT.value,
        )
        self.txt.tag_configure(
            TagNames.COMPLETE.value,
            justify="left",
            foreground="darkgreen",
            font=Formats.TKINTER_FONT.value,
        )
        self.txt.tag_configure(
            TagNames.WARNING.value,
            justify="left",
            foreground="darkorange",
            font=Formats.TKINTER_FONT.value,
        )
        self.txt.tag_configure(
            "TABLE",
            foreground="darkorange",
            font=("Consolas", 10),
            wrap="none",
            borderwidth=0,
        )
        self.txt.insert(INSERT, Defaults.WELCOME_MSG.value + emojis["relaxed"] + "\n" * 2)
        self.txt.tag_add(TagNames.GREETING.value, "1.0", "3.25")
        y_sb.pack(side=RIGHT, fill=Y)
        self.txt.pack(expand=True, fill="both")
        y_sb.config(command=self.txt.yview)
        self.txt.config(state=DISABLED, font=Formats.TKINTER_FONT.value)

        clear_txt_btn = Button(
            self.frame,
            text=" CLEAR",
            compound=LEFT,
            image=self.menu_icons["clean"]["img"],
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            command=lambda: self.clean_txt(),
        )
        clear_txt_btn.pack(side=BOTTOM, fill=X)
        sys.stdout = StdRedirector(self.txt)

        if OS.PYTHON_VER.value != "3.6":
            PythonVersionWarning(msg=f"SimBA is not extensively tested beyond python 3.6. You are using python {OS.PYTHON_VER.value}. If you encounter errors in python>3.6, please report them on GitHub or Gitter (links in the help toolbar) and we will work together to fix the issues!", source=self.__class__.__name__)

        if not check_ffmpeg_available():
            FFMpegNotFoundWarning(msg='SimBA could not find a FFMPEG installation on computer (as evaluated by "ffmpeg" returning None). SimBA works best with FFMPEG and it is recommended to install it on your computer',
                source=self.__class__.__name__,
            )

    def restart(self):
        confirm_restart = askyesno(
            title="RESTART", message="Are you sure that you want restart SimBA?"
        )
        if confirm_restart:
            self.root.destroy()
            os.execl(sys.executable, sys.executable, *sys.argv)

    def clean_txt(self):
        self.txt.config(state=NORMAL)
        self.txt.delete("1.0", END)

    def show_right_click_pop_up(self, event):
        try:
            self.r_click_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.r_click_menu.grab_release()

    def copy_selection_to_clipboard(self):
        self.root.clipboard_clear()
        text = self.txt.get("sel.first", "sel.last")
        text = text.encode("ascii", "ignore").decode()
        self.root.clipboard_append(text)

    def copy_all_to_clipboard(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.txt.get("1.0", "end-1c"))

    def paste_to_txt(self):
        try:
            print(self.root.clipboard_get())
        except UnicodeDecodeError:
            raise InvalidInputError(
                msg="Can only paste utf-8 compatible text",
                source=self.__class__.__name__,
            )


class StdRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget
        self.emojis = get_emojis()

    def write(self, s: str):
        tag_name = TagNames.STANDARD.value
        try:
            s, tag_name = s.split(Defaults.STR_SPLIT_DELIMITER.value, 2)
        except ValueError:
            pass
        if (tag_name != TagNames.STANDARD.value) and (tag_name != "TABLE"):
            s = s + " " + self.emojis[tag_name]
        self.text_space.config(state=NORMAL)
        self.text_space.insert("end", s, (tag_name))
        self.text_space.update()
        self.text_space.see("end")
        self.text_space.config(state=DISABLED)

    def flush(self):
        pass


class SplashStatic:
    def __init__(self, parent):
        self.parent = parent
        self.load_splash_img()
        self.display_splash()

    def load_splash_img(self):
        splash_path_win = os.path.join(
            os.path.dirname(__file__), Paths.SPLASH_PATH_WINDOWS.value
        )
        splash_path_linux = os.path.join(
            os.path.dirname(__file__), Paths.SPLASH_PATH_LINUX.value
        )
        if currentPlatform == OS.WINDOWS.value:
            self.splash_img = PIL.Image.open(splash_path_win)
        else:
            if os.path.isfile(splash_path_linux):
                self.splash_img = PIL.Image.open(splash_path_linux)
            else:
                self.splash_img = PIL.Image.open(splash_path_win)
        self.splash_img_tk = ImageTk.PhotoImage(self.splash_img)

    def display_splash(self):
        width, height = self.splash_img.size
        half_width = int((self.parent.winfo_screenwidth() - width) // 2)
        half_height = int((self.parent.winfo_screenheight() - height) // 2)
        self.parent.geometry("%ix%i+%i+%i" % (width, height, half_width, half_height))
        Label(self.parent, image=self.splash_img_tk).pack()


class SplashMovie:
    def __init__(self):
        self.parent, self.img_cnt = Tk(), 0
        self.parent.overrideredirect(True)
        self.parent.configure(bg="white")
        splash_path = os.path.join(
            os.path.dirname(__file__), Paths.SPLASH_PATH_MOVIE.value
        )
        self.meta_ = get_video_meta_data(splash_path)
        self.cap = cv2.VideoCapture(splash_path)
        width, height = self.meta_["width"], self.meta_["height"]
        half_width = int((self.parent.winfo_screenwidth() - width) // 2)
        half_height = int((self.parent.winfo_screenheight() - height) // 2)
        self.parent.geometry("%ix%i+%i+%i" % (width, height, half_width, half_height))
        self.img_lbl = Label(self.parent, bg="white", image="")
        self.img_lbl.pack()
        self.show_animation()

    def show_animation(self):
        for frm_cnt in range(self.meta_["frame_count"] - 1):
            self.cap.set(1, frm_cnt)
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.img_lbl.configure(image=frame)
            self.img_lbl.imgtk = frame
            self.parent.update()
            cv2.waitKey(max(33, int(self.meta_["fps"] / 1000)))
        self.parent.destroy()


def terminate_children(children):
    for process in children:
        process.terminate()


def main():
    if currentPlatform == OS.WINDOWS.value:
        import ctypes

        myappid = "SimBA development wheel"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    SplashMovie()
    app = App()
    app.root.mainloop()
