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
from simba.ui.import_pose_frame import ImportPoseFrame
from simba.ui.import_videos_frame import ImportVideosFrame
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
from simba.ui.pop_ups.clf_annotation_counts_pop_up import \
    ClfAnnotationCountPopUp
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
from simba.ui.pop_ups.interpolate_pop_up import InterpolatePopUp
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
from simba.ui.pop_ups.smoothing_popup import SmoothingPopUp
from simba.ui.pop_ups.spontaneous_alternation_pop_up import \
    SpontaneousAlternationPopUp
from simba.ui.pop_ups.subset_feature_extractor_pop_up import \
    FeatureSubsetExtractorPopUp
from simba.ui.pop_ups.third_party_annotator_appender_pop_up import \
    ThirdPartyAnnotatorAppenderPopUp
from simba.ui.pop_ups.validation_plot_pop_up import ValidationVideoPopUp
from simba.ui.pop_ups.video_processing_pop_up import (
    BackgroundRemoverPopUp, BoxBlurPopUp, BrightnessContrastPopUp,
    CalculatePixelsPerMMInVideoPopUp, ChangeFpsMultipleVideosPopUp,
    ChangeFpsSingleVideoPopUp, CLAHEPopUp, ClipSingleVideoByFrameNumbers,
    ClipVideoPopUp, ConcatenatingVideosPopUp, ConcatenatorPopUp,
    Convert2AVIPopUp, Convert2BlackWhitePopUp, Convert2bmpPopUp,
    Convert2jpegPopUp, Convert2MOVPopUp, Convert2MP4PopUp, Convert2PNGPopUp,
    Convert2TIFFPopUp, Convert2WEBMPopUp, Convert2WEBPPopUp,
    ConvertROIDefinitionsPopUp, CreateAverageFramePopUp, CreateGIFPopUP,
    CropVideoCirclesPopUp, CropVideoPolygonsPopUp, CropVideoPopUp,
    CrossfadeVideosPopUp, DownsampleMultipleVideosPopUp,
    DownsampleSingleVideoPopUp, ExtractAllFramesPopUp,
    ExtractAnnotationFramesPopUp, ExtractSEQFramesPopUp,
    ExtractSpecificFramesPopUp, FlipVideosPopUp, GreyscaleSingleVideoPopUp,
    ImportFrameDirectoryPopUp, InitiateClipMultipleVideosByFrameNumbersPopUp,
    InitiateClipMultipleVideosByTimestampsPopUp, InteractiveClahePopUp,
    ManualTemporalJoinPopUp, MergeFrames2VideoPopUp, MultiCropPopUp,
    MultiShortenPopUp, ReverseVideoPopUp, RotateVideoSetDegreesPopUp,
    SuperImposeFrameCountPopUp, SuperimposeProgressBarPopUp,
    SuperimposeTextPopUp, SuperimposeTimerPopUp, SuperimposeVideoNamesPopUp,
    SuperimposeVideoPopUp, SuperimposeWatermarkPopUp, UpsampleVideosPopUp,
    VideoRotatorPopUp, VideoTemporalJoinPopUp)
from simba.ui.pop_ups.visualize_pose_in_dir_pop_up import \
    VisualizePoseInFolderPopUp
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        SimbaButton)
from simba.ui.video_info_ui import VideoInfoTable
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_int)
from simba.utils.custom_feature_extractor import CustomFeatureExtractor
from simba.utils.enums import OS, Defaults, Formats, Paths, TagNames
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import (get_bp_config_code_class_pairs, get_emojis,
                                 get_icons_paths, load_simba_fonts)
from simba.utils.printing import stdout_success, stdout_warning
from simba.utils.read_write import find_core_cnt, get_video_meta_data
from simba.utils.warnings import FFMpegNotFoundWarning, PythonVersionWarning
from simba.video_processors.video_processing import \
    extract_frames_from_all_videos_in_directory

sys.setrecursionlimit(10**6)
currentPlatform = platform.system()

PRINT_EMOJIS = True
UNSUPERVISED_INTERFACE = False

class LoadProjectPopUp(object):
    def __init__(self):
        self.main_frm = Toplevel()
        self.main_frm.minsize(300, 200)
        self.main_frm.wm_title("Load SimBA project (project_config.ini file)")
        self.load_project_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="LOAD SIMBA PROJECT_CONFIG.INI", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.LOAD_PROJECT.value)
        self.selected_file = FileSelect(self.load_project_frm, "Select file: ", title="Select project_config.ini file", file_types=[("SimBA Project .ini", "*.ini")])

        load_project_btn = SimbaButton(parent=self.load_project_frm, txt="LOAD PROJECT", txt_clr='blue', img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.launch_project)
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
        self.core_cnt = find_core_cnt()[0]
        self.btn_icons = get_icons_paths()

        for k in self.btn_icons.keys():
            self.btn_icons[k]["img"] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), self.btn_icons[k]["icon_path"])))

        tab_style = ttk.Style()
        tab_style.configure(style='Custom.TNotebook.Tab', font=Formats.FONT_REGULAR.value)
        tab_style.map('Custom.TNotebook.Tab', foreground=[('selected', 'navy')], font=[('selected', Formats.FONT_REGULAR_BOLD.value)])

        tab_parent = ttk.Notebook(hxtScrollbar(simongui), style='Custom.TNotebook')
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

        tab_parent.add(tab2, text=f"{'[ Further imports (data/video/frames) ]':20s}", compound="left", image=self.btn_icons["pose"]["img"])
        tab_parent.add(tab3, text=f"{'[ Video parameters ]':15s}", compound="left", image=self.btn_icons["calipher"]["img"])
        tab_parent.add(tab4, text=f"{'[ Outlier correction ]':15s}", compound="left", image=self.btn_icons["outlier"]["img"])
        tab_parent.add(tab6, text=f"{'[ ROI ]':10s}", compound="left", image=self.btn_icons["roi"]["img"])
        tab_parent.add(tab5, text=f"{'[ Extract features ]':15s}", compound="left", image=self.btn_icons["features"]["img"])
        tab_parent.add(tab7, text=f"{'[ Label behavior] ':15s}", compound="left", image=self.btn_icons["label"]["img"])
        tab_parent.add(tab8, text=f"{'[ Train machine model ]':15s}", compound="left", image=self.btn_icons["clf"]["img"])
        tab_parent.add(tab9, text=f"{'[ Run machine model ]':15s}", compound="left", image=self.btn_icons["clf_2"]["img"])
        tab_parent.add(tab10, text=f"{'[ Visualizations ]':10s}", compound="left", image=self.btn_icons["visualize"]["img"])
        tab_parent.add(tab11, text=f"{'[ Add-ons ]':10s}", compound="left", image=self.btn_icons["add_on"]["img"])

        tab_parent.grid(row=0)
        tab_parent.enable_traversal()

        import_frm = LabelFrame(tab2)
        import_frm.grid(row=0, column=0, sticky=NW)

        further_methods_frm = CreateLabelFrameWithIcon(parent=import_frm, header="FURTHER METHODS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ADDITIONAL_IMPORTS.value)
        extract_frm_btn = SimbaButton(parent=further_methods_frm, txt="EXTRACT FRAMES FOR ALL VIDEOS IN SIMBA PROJECT", txt_clr='blue', compound='right', img='image', font=Formats.FONT_REGULAR.value, cmd=extract_frames_from_all_videos_in_directory, cmd_kwargs={'config_path': lambda:self.config_path, 'directory': lambda:self.video_dir})
        import_frm_dir_btn = SimbaButton(parent=further_methods_frm, txt="IMPORT FRAMES DIRECTORY TO SIMBA PROJECT", txt_clr='blue', compound='right', img='import', font=Formats.FONT_REGULAR.value, cmd=ImportFrameDirectoryPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        add_clf_btn = SimbaButton(parent=further_methods_frm, txt="ADD CLASSIFIER TO SIMBA PROJECT", txt_clr='blue', compound='right', img='plus', font=Formats.FONT_REGULAR.value, cmd=AddClfPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        remove_clf_btn = SimbaButton(parent=further_methods_frm, txt="REMOVE CLASSIFIER FROM SIMBA PROJECT", txt_clr='blue', compound='right', img='trash', font=Formats.FONT_REGULAR.value, cmd=RemoveAClassifierPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        archive_files_btn = SimbaButton(parent=further_methods_frm, txt="ARCHIVE PROCESSED FILES IN SIMBA PROJECT", txt_clr='blue', compound='right', img='archive', font=Formats.FONT_REGULAR.value, cmd=ArchiveProcessedFilesPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        reverse_btn = SimbaButton(parent=further_methods_frm, txt="REVERSE TRACKING IDENTITIES IN SIMBA PROJECT", txt_clr='blue', compound='right', img='reverse_blue', font=Formats.FONT_REGULAR.value, cmd=None)
        interpolate_btn = SimbaButton(parent=further_methods_frm, txt="INTERPOLATE POSE IN SIMBA PROJECT", txt_clr='blue', compound='right', img='line_chart_blue', font=Formats.FONT_REGULAR.value, cmd=InterpolatePopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        smooth_btn = SimbaButton(parent=further_methods_frm, txt="SMOOTH POSE IN SIMBA PROJECT", txt_clr='blue', compound='right', img='wand_blue', font=Formats.FONT_REGULAR.value, cmd=SmoothingPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        label_setscale = CreateLabelFrameWithIcon(parent=tab3, header="VIDEO PARAMETERS (FPS, RESOLUTION, PPX/MM ....)", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_PARAMETERS.value)
        self.distance_in_mm_eb = Entry_Box(label_setscale, "KNOWN DISTANCE (MILLIMETERS)", "35", validation="numeric")
        button_setdistanceinmm = SimbaButton(parent=label_setscale, txt="AUTO-POPULATE", txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.set_distance_mm)

        button_setscale = SimbaButton(parent=label_setscale, txt="CONFIGURE VIDEO PARAMETERS", txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.create_video_info_table, img='calipher')
        self.new_ROI_frm = CreateLabelFrameWithIcon(parent=tab6, header="SIMBA ROI INTERFACE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI.value)
        self.start_new_ROI = SimbaButton(parent=self.new_ROI_frm, txt="DEFINE ROIs", txt_clr='green', font=Formats.FONT_REGULAR.value, img='roi', cmd=ROI_menu, cmd_kwargs={'config_path': lambda:self.config_path})

        self.delete_all_ROIs = SimbaButton(parent=self.new_ROI_frm, txt="DELETE ALL ROI DEFINITIONS", txt_clr='red', font=Formats.FONT_REGULAR.value, img='trash', cmd=delete_all_ROIs, cmd_kwargs={'config_path': lambda:self.config_path})
        self.standardize_roi_size_popup_btn = SimbaButton(parent=self.new_ROI_frm, txt="STANDARDIZE ROI SIZES", txt_clr='blue', font=Formats.FONT_REGULAR.value, img='calipher', cmd=ROISizeStandardizerPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        self.new_ROI_frm.grid(row=0, sticky=NW)
        self.start_new_ROI.grid(row=0, sticky=NW)
        self.delete_all_ROIs.grid(row=1, column=0, sticky=NW)
        self.standardize_roi_size_popup_btn.grid(row=2, column=0, sticky=NW)

        self.roi_draw = LabelFrame(tab6, text="ANALYZE ROI DATA", font=Formats.FONT_HEADER.value)
        analyze_roi_btn = SimbaButton(parent=self.roi_draw, txt="ANALYZE ROI DATA: AGGREGATES", txt_clr='green', img='analyze_green', font=Formats.FONT_REGULAR.value, cmd=ROIAnalysisPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        analyze_roi_time_bins_btn = SimbaButton(parent=self.roi_draw, txt="ANALYZE ROI DATA: TIME-BINS", txt_clr='blue', img='analyze_blue', font=Formats.FONT_REGULAR.value, cmd=ROIAnalysisTimeBinsPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        self.roi_draw.grid(row=0, column=1, sticky=N)
        analyze_roi_btn.grid(row=0, sticky="NW")
        analyze_roi_time_bins_btn.grid(row=1, sticky="NW")

        self.roi_draw1 = LabelFrame(tab6, text="VISUALIZE ROI DATA", font=Formats.FONT_HEADER.value)


        visualizeROI = SimbaButton(parent=self.roi_draw1, txt="VISUALIZE ROI TRACKING", txt_clr='green', img='visualize_green', font=Formats.FONT_REGULAR.value, cmd=VisualizeROITrackingPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        visualizeROIfeature = SimbaButton(parent=self.roi_draw1, txt="VISUALIZE ROI FEATURES", txt_clr='blue', img='visualize_blue', font=Formats.FONT_REGULAR.value, cmd=VisualizeROIFeaturesPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        ##organize
        self.roi_draw1.grid(row=0, column=2, sticky=N)
        visualizeROI.grid(row=0, sticky="NW")
        visualizeROIfeature.grid(row=1, sticky="NW")

        processmovementdupLabel = LabelFrame( tab6, text="OTHER ANALYSES / VISUALIZATIONS", font=Formats.FONT_HEADER.value)
        analyze_distances_velocity_btn = SimbaButton(parent=processmovementdupLabel, txt="ANALYZE DISTANCES / VELOCITY: AGGREGATES", img='metrics_green', txt_clr='green', font=Formats.FONT_REGULAR.value, cmd=MovementAnalysisPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        analyze_distances_velocity_timebins_btn = SimbaButton(parent=processmovementdupLabel, txt="ANALYZE DISTANCES / VELOCITY: TIME-BINS", img='metrics_blue',  txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=MovementAnalysisTimeBinsPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        heatmaps_location_button = SimbaButton(parent=processmovementdupLabel, txt="CREATE LOCATION HEATMAPS", txt_clr='red', img='heatmap', font=Formats.FONT_REGULAR.value, cmd=HeatmapLocationPopup, cmd_kwargs={'config_path': lambda:self.config_path})
        button_lineplot = SimbaButton(parent=processmovementdupLabel, txt="CREATE PATH PLOTS", txt_clr='orange', img='path', font=Formats.FONT_REGULAR.value, cmd=QuickLineplotPopup, cmd_kwargs={'config_path': lambda:self.config_path})

        button_analyzeDirection = SimbaButton(parent=processmovementdupLabel, txt="ANALYZE DIRECTIONALITY BETWEEN ANIMALS", img='direction', txt_clr='pink', font=Formats.FONT_REGULAR.value, cmd=AnimalDirectingAnimalPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_visualizeDirection = SimbaButton(parent=processmovementdupLabel, txt="VISUALIZE DIRECTIONALITY BETWEEN ANIMALS", img='direction', txt_clr='brown', font=Formats.FONT_REGULAR.value, cmd=DirectingOtherAnimalsVisualizerPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_analyzeDirection_bp = SimbaButton(parent=processmovementdupLabel, txt="ANALYZE DIRECTIONALITY BETWEEN BODY PARTS", img='direction', txt_clr='purple', font=Formats.FONT_REGULAR.value, cmd=DirectionAnimalToBodyPartSettingsPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_visualizeDirection_bp = SimbaButton(parent=processmovementdupLabel, txt="VISUALIZE DIRECTIONALITY BETWEEN BODY PARTS", img='direction', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=DirectingAnimalToBodyPartVisualizerPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        btn_agg_boolean_conditional_statistics = SimbaButton(parent=processmovementdupLabel, txt="AGGREGATE BOOLEAN CONDITIONAL STATISTICS", img='details', txt_clr='grey', font=Formats.FONT_REGULAR.value, cmd=BooleanConditionalSlicerPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        spontaneous_alternation_pop_up_btn = SimbaButton(parent=processmovementdupLabel, txt="SPONTANEOUS ALTERNATION", img='t', txt_clr='navy', font=Formats.FONT_REGULAR.value, cmd=SpontaneousAlternationPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

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

        label_outliercorrection = CreateLabelFrameWithIcon(parent=tab4, header="OUTLIER CORRECTION", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OUTLIERS_DOC.value)
        button_settings_outlier = SimbaButton(parent=label_outliercorrection, txt="SETTINGS", txt_clr='blue', img='settings', font=Formats.FONT_REGULAR.value, cmd=OutlierSettingsPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        button_outliercorrection = SimbaButton(parent=label_outliercorrection, txt="RUN OUTLIER CORRECTION", txt_clr='green', img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.correct_outlier, thread=False)
        button_skipOC = SimbaButton(parent=label_outliercorrection, txt="SKIP OUTLIER CORRECTION (CAUTION)", txt_clr='red', img='skip_2', font=Formats.FONT_REGULAR.value, cmd=self.initiate_skip_outlier_correction, thread=True)

        label_extractfeatures = CreateLabelFrameWithIcon(parent=tab5, header="EXTRACT FEATURES", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.EXTRACT_FEATURES.value)

        button_extractfeatures = SimbaButton(parent=label_extractfeatures, txt="EXTRACT FEATURES", txt_clr='blue', img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.run_feature_extraction, thread=True)

        def activate(box, *args):
            for entry in args:
                if box.get() == 0:
                    entry.configure(state="disabled")
                elif box.get() == 1:
                    entry.configure(state="normal")

        labelframe_usrdef = LabelFrame(label_extractfeatures)
        self.scriptfile = FileSelect(labelframe_usrdef, "SCRIPT PATH:", file_types=[("Python .py file", "*.py")])
        self.scriptfile.btnFind.configure(state="disabled")
        self.user_defined_var = BooleanVar(value=False)
        userscript = Checkbutton(labelframe_usrdef, text="Apply user-defined feature extraction script", font=Formats.FONT_REGULAR.value, variable=self.user_defined_var, command=lambda: activate(self.user_defined_var, self.scriptfile.btnFind))

        roi_feature_frm = CreateLabelFrameWithIcon(parent=tab5, header="APPEND ROI FEATURES", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.APPEND_ROI_FEATURES.value)

        append_roi_features_by_animal = SimbaButton(parent=roi_feature_frm, txt="APPEND ROI DATA TO FEATURES: BY ANIMAL (CAUTION)", txt_clr='red', img='join_red', font=Formats.FONT_REGULAR.value, cmd=AppendROIFeaturesByAnimalPopUp, cmd_kwargs={'config_path': lambda:self.config_path}, thread=True)
        append_roi_features_by_body_part = SimbaButton(parent=roi_feature_frm, txt="APPEND ROI DATA TO FEATURES: BY BODY-PARTS (CAUTION)", img='join_yellow', txt_clr='orange', font=Formats.FONT_REGULAR.value, cmd=AppendROIFeaturesByBodyPartPopUp, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)
        remove_roi_features_from_feature_set = SimbaButton(parent=roi_feature_frm, txt="REMOVE ROI FEATURES FROM FEATURE SET", txt_clr='darkred', img='trash', font=Formats.FONT_REGULAR.value, cmd=RemoveROIFeaturesPopUp, cmd_kwargs={'config_path': lambda:self.config_path, 'dataset': lambda:'features_extracted'}, thread=False)


        feature_tools_frm = LabelFrame(tab5, text="FEATURE TOOLS", pady=5, font=Formats.FONT_HEADER.value)
        compute_feature_subset_btn = SimbaButton(parent=feature_tools_frm, txt="CALCULATE FEATURE SUBSETS", txt_clr='blue', img='subset_blue', font=Formats.FONT_REGULAR.value, cmd=FeatureSubsetExtractorPopUp, cmd_kwargs={'config_path': lambda: self.config_path}, thread=False)

        label_behavior_frm = CreateLabelFrameWithIcon(parent=tab7, header="LABEL BEHAVIOR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.LABEL_BEHAVIOR.value)
        select_video_btn_new = SimbaButton(parent=label_behavior_frm, txt="Select video (create new video annotation)", img='label_blue', txt_clr='navy', cmd=select_labelling_video, cmd_kwargs={'config_path': lambda :self.config_path, 'threshold_dict': lambda: None, 'setting': lambda: "from_scratch", 'continuing': lambda: False}, thread=False)
        select_video_btn_continue = SimbaButton(parent=label_behavior_frm, txt="Select video (continue existing video annotation)", img='label_yellow', txt_clr='darkgoldenrod', cmd=select_labelling_video, cmd_kwargs={'config_path': lambda: self.config_path, 'threshold_dict': lambda:None, 'setting': lambda: None, 'continuing': lambda:True}, thread=False)

        label_thirdpartyann = CreateLabelFrameWithIcon(parent=tab7, header="IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.THIRD_PARTY_ANNOTATION.value)
        button_importmars = SimbaButton(parent=label_thirdpartyann, txt="Import MARS Annotation (select folder with .annot files)", txt_clr="blue", cmd=self.importMARS, thread=False)
        button_importboris = SimbaButton(parent=label_thirdpartyann, txt="Import BORIS Annotation (select folder with .csv files)", txt_clr="green", cmd=self.importBoris, thread=False)
        button_importsolomon = SimbaButton(parent=label_thirdpartyann, txt="Import SOLOMON Annotation (select folder with .csv files", txt_clr="purple", cmd=self.importSolomon, thread=False)

        button_importethovision = SimbaButton(parent=label_thirdpartyann, txt="Import ETHOVISION Annotation (select folder with .xls/xlsx files)", txt_clr="blue", cmd=self.import_ethovision, thread=False)
        button_importdeepethogram = SimbaButton(parent=label_thirdpartyann, txt="Import DEEPETHOGRAM Annotation (select folder with .csv files)", txt_clr="green", cmd=self.import_deepethogram, thread=False)
        import_observer_btn = SimbaButton(parent=label_thirdpartyann, txt="Import NOLDUS OBSERVER Annotation (select folder with .xls/xlsx files)", txt_clr="purple", cmd=self.import_noldus_observer, thread=False)


        label_pseudo = CreateLabelFrameWithIcon(parent=tab7,header="PSEUDO-LABELLING",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.PSEUDO_LBL.value)
        pseudo_intructions_lbl_1 = Label(label_pseudo,text="Note: SimBA pseudo-labelling require initial machine predictions.", font=Formats.FONT_REGULAR.value)
        pseudo_intructions_lbl_2 = Label( label_pseudo, text="Click here more information on how to use the SimBA pseudo-labelling interface.", font=Formats.FONT_REGULAR.value, cursor="hand2", fg="blue")
        pseudo_intructions_lbl_2.bind("<Button-1>", lambda e: self.callback("https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md"))
        pLabel_framedir = FileSelect(label_pseudo, "Video Path", lblwidth="10")
        plabelframe_threshold = LabelFrame(label_pseudo, text="Threshold", pady=5, padx=5, font=Formats.FONT_HEADER.value)
        plabel_threshold = [0] * len(self.clf_names)
        for count, i in enumerate(list(self.clf_names)):
            plabel_threshold[count] = Entry_Box(plabelframe_threshold, str(i), "20")
            plabel_threshold[count].grid(row=count + 2, sticky=W)

        pseudo_lbl_btn = SimbaButton(parent=label_pseudo, txt="Correct labels", cmd=select_labelling_video, cmd_kwargs={'config_path': lambda:self.config_path, 'threshold_dict': lambda:dict(zip(self.clf_names, plabel_threshold)), 'setting': lambda:'pseudo', 'continuing': lambda:False, 'video_file_path': lambda:pLabel_framedir.file_path}, thread=False)

        label_adv_label = CreateLabelFrameWithIcon(parent=tab7, header="ADVANCED LABEL BEHAVIOR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ADVANCED_LBL.value)
        label_adv_note_1 = Label(label_adv_label, text="Note that you will have to specify the presence of *both* behavior and non-behavior on your own.", font=Formats.FONT_REGULAR.value)
        label_adv_note_2 = Label(label_adv_label, text="Click here more information on how to use the SimBA labelling interface.", font=Formats.FONT_REGULAR.value, cursor="hand2", fg="blue")
        label_adv_note_2.bind("<Button-1>", lambda e: self.callback("https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md"))

        adv_label_btn_new = SimbaButton(parent=label_adv_label, txt="Select video (create new video annotation)", cmd=select_labelling_video_advanced, cmd_kwargs={'config_path': lambda:self.config_path, 'continuing': lambda:False}, thread=False)
        adv_label_btn_continue = SimbaButton(parent=label_adv_label, txt="Select video (continue existing video annotation)", cmd=select_labelling_video_advanced, cmd_kwargs={'config_path': lambda:self.config_path, 'continuing': lambda:True}, thread=False)

        targeted_clip_annotator_frm = CreateLabelFrameWithIcon(parent=tab7,header="TARGETED CLIP ANNOTATOR",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.ADVANCED_LBL.value)
        targeted_clip_annotator_note = Label(targeted_clip_annotator_frm, font=Formats.FONT_REGULAR.value, text="A bout annotator that creates annotated clips from videos associated with ML results.")
        targeted_clip_annotator_btn = SimbaButton(parent=targeted_clip_annotator_frm, txt="Select video", cmd=select_labelling_video_targeted_clips, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)

        lbl_tools_frm = LabelFrame(tab7, text="LABELLING TOOLS", font=Formats.FONT_HEADER.value, fg="black")
        visualize_annotation_img_btn = SimbaButton(parent=lbl_tools_frm, txt="Visualize annotations", cmd=ExtractAnnotationFramesPopUp, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)
        third_party_annotations_btn = SimbaButton(parent=lbl_tools_frm, txt="Append third-party annotations", txt_clr='purple', cmd=ThirdPartyAnnotatorAppenderPopUp, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)
        remove_roi_features_from_annotation_set = SimbaButton(parent=lbl_tools_frm, txt="Remove ROI features from label set", txt_clr='darkred', cmd=RemoveROIFeaturesPopUp, cmd_kwargs={'config_path': lambda:self.config_path, 'dataset': lambda:'targets_inserted'}, thread=False)
        compute_annotation_statistics = SimbaButton(parent=lbl_tools_frm, txt="Count annotations in project", txt_clr='orange', cmd=ClfAnnotationCountPopUp, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)



        label_trainmachinemodel = CreateLabelFrameWithIcon(parent=tab8, header="TRAIN MACHINE MODELS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.TRAIN_ML_MODEL.value)


        button_trainmachinesettings = SimbaButton(parent=label_trainmachinemodel, txt="SETTINGS", img='settings', txt_clr='darkorange', cmd=self.trainmachinemodelsetting, thread=False)
        button_trainmachinemodel = SimbaButton(parent=label_trainmachinemodel, txt="TRAIN SINGLE MODEL (GLOBAL ENVIRONMENT)", img='one_blue', txt_clr='blue', cmd=self.train_single_model, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)

        button_train_multimodel = SimbaButton(parent=label_trainmachinemodel, txt="TRAIN MULTIPLE MODELS (ONE FOR EACH SAVED SETTING)", img='multiple_green', txt_clr='green', cmd=self.train_multiple_models_from_meta, cmd_kwargs={'config_path': lambda:self.config_path}, thread=False)

        label_model_validation = CreateLabelFrameWithIcon( parent=tab9, header="VALIDATE MODEL ON SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value)
        self.csvfile = FileSelect(label_model_validation,"SELECT DATA FEATURE FILE",color="blue",lblwidth=30,file_types=[("SimBA CSV", "*.csv"), ("SimBA PARQUET", "*.parquet")],initialdir=os.path.join(    self.project_path, Paths.FEATURES_EXTRACTED_DIR.value))
        self.modelfile = FileSelect(label_model_validation,"SELECT MODEL FILE",color="blue",lblwidth=30,initialdir=self.project_path)

        button_runvalidmodel = SimbaButton(parent=label_model_validation, txt="RUN MODEL", txt_clr='blue', img='rocket', cmd=self.validate_model_first_step, thread=True)
        button_generateplot = SimbaButton(parent=label_model_validation, txt="INTERACTIVE PROBABILITY PLOT",  img='interactive_blue', txt_clr='blue', cmd=self.launch_interactive_plot, thread=False)

        self.dis_threshold = Entry_Box(label_model_validation, "DISCRIMINATION THRESHOLD (0.0-1.0):", "35")
        self.min_behaviorbout = Entry_Box(label_model_validation,"MINIMUM BOUT LENGTH (MS):","35",validation="numeric")
        button_validate_model = SimbaButton(parent=label_model_validation, txt="CREATE VALIDATION VIDEO", txt_clr='blue', img='visualize_blue', cmd=ValidationVideoPopUp, cmd_kwargs={'config_path': lambda:config_path, 'simba_main_frm': lambda:self})

        label_runmachinemodel = CreateLabelFrameWithIcon(parent=tab9,header="RUN MACHINE MODEL",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.SCENARIO_2.value)

        button_run_rfmodelsettings = SimbaButton(parent=label_runmachinemodel, txt="MODEL SETTINGS", txt_clr='green', img='settings', cmd=SetMachineModelParameters, cmd_kwargs={'config_path': lambda:config_path})
        button_runmachinemodel = SimbaButton(parent=label_runmachinemodel, txt="RUN MODELS", txt_clr='green', img='clf', cmd=self.runrfmodel, thread=True)

        kleinberg_button = SimbaButton(parent=label_runmachinemodel, txt="KLEINBERG SMOOTHING", txt_clr='green', img='feather_green', cmd=KleinbergPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        fsttc_button = SimbaButton(parent=label_runmachinemodel, txt="FSTTC", txt_clr='green', img='tile_green', cmd=FSTTCPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        mutual_exclusivity = SimbaButton(parent=label_runmachinemodel, txt="MUTUAL EXCLUSIVITY CORRECTION", img='seperate_green', txt_clr='green', cmd=MutualExclusivityPupUp, cmd_kwargs={'config_path': lambda:self.config_path})

        label_machineresults = CreateLabelFrameWithIcon( parent=tab9, header="ANALYZE MACHINE RESULTS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)


        button_process_datalog = SimbaButton(parent=label_machineresults, txt="ANALYZE MACHINE PREDICTIONS: AGGREGATES", img='metrics_blue', txt_clr='blue', cmd=ClfDescriptiveStatsPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_process_movement = SimbaButton(parent=label_machineresults, txt="ANALYZE DISTANCES/VELOCITY: AGGREGATES", img='metrics_blue', txt_clr='blue', cmd=MovementAnalysisPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_movebins = SimbaButton(parent=label_machineresults, txt="ANALYZE DISTANCES/VELOCITY: TIME BINS", txt_clr='blue', img='metrics_blue', cmd=MovementAnalysisTimeBinsPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_classifierbins = SimbaButton(parent=label_machineresults, txt="ANALYZE MACHINE PREDICTIONS: TIME-BINS", txt_clr='blue', img='metrics_blue', cmd=TimeBinsClfPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_classifier_ROI = SimbaButton(parent=label_machineresults, txt="ANALYZE MACHINE PREDICTION: BY ROI", txt_clr='blue', img='metrics_blue', cmd=ClfByROIPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        button_severity = SimbaButton(parent=label_machineresults, txt="ANALYZE MACHINE PREDICTION: BY SEVERITY", txt_clr='blue', img='metrics_blue', cmd=AnalyzeSeverityPopUp, cmd_kwargs={'config_path': lambda:self.config_path})

        visualization_frm = CreateLabelFrameWithIcon(parent=tab10, header="DATA VISUALIZATIONS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VISUALIZATION.value)
        sklearn_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE CLASSIFICATIONS", img='split', txt_clr='black', cmd=SklearnVisualizationPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        sklearn_visualization_btn.grid(row=0, column=0, sticky=NW)
        gantt_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE GANTT", img='bar_graph_blue', txt_clr='blue', cmd=GanttPlotPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        gantt_visualization_btn.grid(row=1, column=0, sticky=NW)
        probability_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE PROBABILITIES", img='dice_green', txt_clr='green', cmd=VisualizeClassificationProbabilityPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        probability_visualization_btn.grid(row=2, column=0, sticky=NW)
        path_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE PATHS", img='path', txt_clr='orange', cmd=PathPlotPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        path_visualization_btn.grid(row=3, column=0, sticky=NW)
        distance_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE DISTANCES", img='distance_red', txt_clr='red', cmd=DistancePlotterPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        distance_visualization_btn.grid(row=4, column=0, sticky=NW)
        heatmap_clf_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE CLASSIFICATION HEATMAPS", img='heatmap', txt_clr='pink', cmd=HeatmapClfPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        heatmap_clf_visualization_btn.grid(row=5, column=0, sticky=NW)
        data_plot_visualization_btn = SimbaButton(parent=visualization_frm, txt="VISUALIZE DATA PLOTS", img='metrics', txt_clr='purple', cmd=DataPlotterPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        data_plot_visualization_btn.grid(row=6, column=0, sticky=NW)
        clf_validation_btn = SimbaButton(parent=visualization_frm, txt="CLASSIFIER VALIDATION CLIPS", txt_clr='blue', img='check_blue', cmd=ClassifierValidationPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        clf_validation_btn.grid(row=7, column=0, sticky=NW)
        merge_frm = CreateLabelFrameWithIcon(parent=tab10, header="MERGE FRAMES", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CONCAT_VIDEOS.value)
        merge_frm_btn = SimbaButton(parent=merge_frm, txt="MERGE FRAMES", img='merge', txt_clr='black', cmd=ConcatenatorPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        plotlyInterface = CreateLabelFrameWithIcon(parent=tab10, header="PLOTLY / DASH", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.PLOTLY.value)
        plotlyInterfaceTitles = ["Sklearn results", "Time bin analyses", "Probabilities", "Severity analysis"]
        toIncludeVar = []
        for i in range(len(plotlyInterfaceTitles) + 1):
            toIncludeVar.append(IntVar())
        plotlyCheckbox = [0] * (len(plotlyInterfaceTitles) + 1)
        for i in range(len(plotlyInterfaceTitles)):
            plotlyCheckbox[i] = Checkbutton(plotlyInterface, text=plotlyInterfaceTitles[i], variable=toIncludeVar[i])
            plotlyCheckbox[i].grid(row=i, sticky=W)

        button_save_plotly_file = Button(plotlyInterface, text="Save SimBA / Plotly dataset", command=lambda: self.generateSimBPlotlyFile(toIncludeVar))
        self.plotly_file = FileSelect( plotlyInterface, "SimBA Dashboard file (H5)", title="Select SimBA/Plotly dataset (h5)")
        self.groups_file = FileSelect(plotlyInterface, "SimBA Groups file (CSV)", title="Select groups file (csv)")
        button_open_plotly_interface = Button(plotlyInterface, text="Open SimBA / Plotly dataset", font=Formats.FONT_REGULAR.value, fg="black", command=lambda: [self.open_plotly_interface("http://127.0.0.1:8050")])

        lbl_addon = LabelFrame(tab11, text="SimBA Expansions", pady=5, padx=5, font=Formats.FONT_HEADER.value, fg="black")
        button_bel = SimbaButton(parent=lbl_addon, txt="Pup retrieval - Analysis Protocol 1", txt_clr='blue', cmd=PupRetrievalPopUp, cmd_kwargs={'config_path': lambda:self.config_path})
        cue_light_analyser_btn = SimbaButton(parent=lbl_addon, txt="Cue light analysis", txt_clr='red', cmd=CueLightAnalyzerMenu, cmd_kwargs={'config_path': lambda:self.config_path})

        anchored_roi_analysis_btn = SimbaButton(parent=lbl_addon, txt="Animal-anchored ROI analysis", txt_clr='orange', cmd=BoundaryMenus, cmd_kwargs={'config_path': lambda:self.config_path})
        ImportVideosFrame(parent_frm=import_frm, config_path=config_path, idx_row=0, idx_column=0)
        ImportPoseFrame(parent_frm=import_frm, idx_row=1, idx_column=0, config_path=config_path)
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
        third_party_annotations_btn.grid(row=0, column=1, sticky=NW)
        remove_roi_features_from_annotation_set.grid(row=1, column=0, sticky=NW)
        compute_annotation_statistics.grid(row=1, column=1, sticky=NW)

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

        if UNSUPERVISED_INTERFACE:
            from simba.unsupervised.unsupervised_main import UnsupervisedGUI
            unsupervised_btn = Button(lbl_addon, text="Unsupervised analysis", fg="purple", font=Formats.FONT_REGULAR.value, command=lambda: UnsupervisedGUI(config_path=self.config_path))
            unsupervised_btn.grid(row=3, sticky=NW)

    def create_video_info_table(self):
        video_info_tabler = VideoInfoTable(config_path=self.config_path)
        video_info_tabler.create_window()

    def initiate_skip_outlier_correction(self):
        outlier_correction_skipper = OutlierCorrectionSkipper(config_path=self.config_path)
        outlier_correction_skipper.run()

    def validate_model_first_step(self):
        _ = InferenceValidation(config_path=self.config_path, input_file_path=self.csvfile.file_path, clf_path=self.modelfile.file_path)

    def train_single_model(self, config_path=None):
        model_trainer = TrainRandomForestClassifier(config_path=config_path)
        model_trainer.run()
        model_trainer.save()

    def train_multiple_models_from_meta(self, config_path=None):
        model_trainer = GridSearchRandomForestClassifier(config_path=config_path)
        model_trainer.run()

    def importBoris(self):
        ann_folder = askdirectory()
        boris_appender = BorisAppender(config_path=self.config_path, data_dir=ann_folder)
        threading.Thread(target=boris_appender.run).start()

    def importSolomon(self):
        ann_folder = askdirectory()
        solomon_importer = SolomonImporter(config_path=self.config_path, data_dir=ann_folder)
        threading.Thread(target=solomon_importer.run).start()

    def import_ethovision(self):
        ann_folder = askdirectory()
        ethovision_importer = ImportEthovision(config_path=self.config_path, data_dir=ann_folder)
        threading.Thread(target=ethovision_importer.run).start()

    def import_deepethogram(self):
        ann_folder = askdirectory()
        deepethogram_importer = DeepEthogramImporter(config_path=self.config_path, data_dir=ann_folder)
        if self.core_cnt > Defaults.THREADSAFE_CORE_COUNT.value:
            deepethogram_importer.run()
        else:
            threading.Thread(target=deepethogram_importer.run).start()

    def import_noldus_observer(self):
        directory = askdirectory()
        noldus_observer_importer = NoldusObserverImporter(config_path=self.config_path, data_dir=directory)
        threading.Thread(target=noldus_observer_importer.run).start()

    def importMARS(self):
        bento_dir = askdirectory()
        bento_appender = BentoAppender(config_path=self.config_path, data_dir=bento_dir)
        threading.Thread(target=bento_appender.run).start()

    def launch_interactive_plot(self):
        interactive_grapher = InteractiveProbabilityGrapher(config_path=self.config_path,file_path=self.csvfile.file_path,model_path=self.modelfile.file_path)
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
        print(f"Pose-estimation body part setting for feature extraction: {str(self.animal_cnt)} animals {str(self.pose_setting)} body-parts...")
        feature_extractor_classes = get_bp_config_code_class_pairs()
        if self.user_defined_var.get():
            custom_feature_extractor = CustomFeatureExtractor(extractor_file_path=self.scriptfile.file_path,config_path=self.config_path)
            custom_feature_extractor.run()
            stdout_success(msg="Custom feature extraction complete!",source=self.__class__.__name__)
        else:
            if self.pose_setting not in feature_extractor_classes.keys():
                raise InvalidInputError(msg=f"The project pose-configuration key is set to {self.pose_setting} which is invalid. OPTIONS: {list(feature_extractor_classes.keys())}. Check the pose-estimation setting in the project_config.ini", source=self.__class__.__name__)
            if self.pose_setting == "8":
                feature_extractor = feature_extractor_classes[self.pose_setting][self.animal_cnt](config_path=self.config_path)
            else:
                feature_extractor = feature_extractor_classes[self.pose_setting](config_path=self.config_path)
            feature_extractor.run()

    def set_distance_mm(self):
        check_int(name="DISTANCE IN MILLIMETER",value=self.distance_in_mm_eb.entry_get,min_value=1)
        self.config.set("Frame settings", "distance_mm", self.distance_in_mm_eb.entry_get)
        with open(self.config_path, "w") as f:
            self.config.write(f)

    def correct_outlier(self):
        outlier_correcter_movement = OutlierCorrecterMovement(config_path=self.config_path)
        outlier_correcter_movement.run()
        outlier_correcter_location = OutlierCorrecterLocation(config_path=self.config_path)
        outlier_correcter_location.run()
        stdout_success(msg='Outlier corrected files located in "project_folder/csv/outlier_corrected_movement_location" directory',source=self.__class__.__name__)

    def callback(self, url):
        webbrowser.open_new(url)


class App(object):
    def __init__(self):
        bg_path = os.path.join(os.path.dirname(__file__), Paths.BG_IMG_PATH.value)
        emojis = get_emojis()
        icon_path_windows = os.path.join(os.path.dirname(__file__), Paths.LOGO_ICON_WINDOWS_PATH.value)
        icon_path_darwin = os.path.join(os.path.dirname(__file__), Paths.LOGO_ICON_DARWIN_PATH.value)
        self.menu_icons = get_icons_paths()
        self.root = Tk()
        self.root.title("SimBA")
        self.root.minsize(750, 750)
        self.root.geometry(Formats.ROOT_WINDOW_SIZE.value)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        if currentPlatform == OS.WINDOWS.value:
            load_simba_fonts()
            self.root.iconbitmap(icon_path_windows)
        if currentPlatform == OS.MAC.value:
            load_simba_fonts()
            self.root.iconphoto(False, ImageTk.PhotoImage(PIL.Image.open(icon_path_darwin)))
        for k in self.menu_icons.keys():
            self.menu_icons[k]["img"] = ImageTk.PhotoImage(image=PIL.Image.open(os.path.join(os.path.dirname(__file__), self.menu_icons[k]["icon_path"])))
        bg_img = ImageTk.PhotoImage(file=bg_path)
        background = Label(self.root, image=bg_img, bd=0, bg="white")
        background.pack(fill="both", expand=True)
        background.image = bg_img

        menu = Menu(self.root)
        self.root.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Create a new project", compound="left", image=self.menu_icons["create"]["img"], command=lambda: ProjectCreatorPopUp(), font=Formats.FONT_REGULAR.value)
        file_menu.add_command(label="Load project", compound="left", image=self.menu_icons["load"]["img"], command=lambda: LoadProjectPopUp(), font=Formats.FONT_REGULAR.value)
        file_menu.add_separator()
        file_menu.add_command(label="Restart", compound="left", image=self.menu_icons["restart"]["img"], command=lambda: self.restart(), font=Formats.FONT_REGULAR.value)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", compound="left", image=self.menu_icons["exit"]["img"], command=self.root.destroy, font=Formats.FONT_REGULAR.value)

        batch_process_menu = Menu(menu)
        menu.add_cascade(label="Process Videos", menu=batch_process_menu)
        batch_process_menu.add_command(label="Batch pre-process videos", compound="left", image=self.menu_icons["factory"]["img"], command=lambda: BatchPreProcessPopUp(), font=Formats.FONT_REGULAR.value)

        video_process_menu = Menu(menu)
        fps_menu = Menu(video_process_menu)
        fps_menu.add_command(label="Change FPS for single video", compound="left", image=self.menu_icons["single_blue"]["img"], command=ChangeFpsSingleVideoPopUp, font=Formats.FONT_REGULAR.value)
        fps_menu.add_command(label="Change FPS for multiple videos", compound="left", image=self.menu_icons["multiple_blue"]["img"], command=ChangeFpsMultipleVideosPopUp, font=Formats.FONT_REGULAR.value)
        fps_menu.add_command(label="Up-sample fps with interpolation", command=UpsampleVideosPopUp, font=Formats.FONT_REGULAR.value)

        menu.add_cascade(label="Tools", menu=video_process_menu)
        video_process_menu.add_cascade(label="Change FPS...", compound="left", image=self.menu_icons["fps"]["img"], menu=fps_menu, font=Formats.FONT_REGULAR.value)

        clip_video_menu = Menu(menu)
        clip_video_menu.add_command(label="Clip single video", command=ClipVideoPopUp, font=Formats.FONT_REGULAR.value)
        clip_video_menu.add_command(label="Clip multiple videos", command=InitiateClipMultipleVideosByTimestampsPopUp, font=Formats.FONT_REGULAR.value)

        clip_video_menu.add_command(label="Clip video into multiple videos", command=MultiShortenPopUp, font=Formats.FONT_REGULAR.value)
        clip_video_menu.add_command(label="Clip single video by frame numbers", command=ClipSingleVideoByFrameNumbers, font=Formats.FONT_REGULAR.value)
        clip_video_menu.add_command(label="Clip multiple videos by frame numbers", command=InitiateClipMultipleVideosByFrameNumbersPopUp, font=Formats.FONT_REGULAR.value)

        video_process_menu.add_cascade(label="Clip videos...", compound="left", image=self.menu_icons["clip"]["img"], menu=clip_video_menu, font=Formats.FONT_REGULAR.value)

        crop_video_menu = Menu(menu)
        crop_video_menu.add_command(label="Crop videos", compound="left", image=self.menu_icons["crop"]["img"], command=CropVideoPopUp, font=Formats.FONT_REGULAR.value)
        crop_video_menu.add_command(label="Crop videos (circles)", compound="left", image=self.menu_icons["circle"]["img"], command=CropVideoCirclesPopUp, font=Formats.FONT_REGULAR.value)
        crop_video_menu.add_command(label="Crop videos (polygons)", compound="left", image=self.menu_icons["polygon"]["img"], command=CropVideoPolygonsPopUp, font=Formats.FONT_REGULAR.value)
        crop_video_menu.add_command(label="Multi-crop", compound="left", image=self.menu_icons["crop"]["img"], command=MultiCropPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Crop videos...", compound="left", image=self.menu_icons["crop"]["img"], menu=crop_video_menu, font=Formats.FONT_REGULAR.value)

        format_menu = Menu(video_process_menu)
        img_format_menu = Menu(format_menu)
        video_format_menu = Menu(format_menu)

        img_format_menu.add_command(label="Convert images to PNG", compound="left",  image=self.menu_icons["png"]["img"], command=Convert2PNGPopUp, font=Formats.FONT_REGULAR.value)
        img_format_menu.add_command(label="Convert images  to JPEG", compound="left", image=self.menu_icons["jpeg"]["img"], command=Convert2jpegPopUp, font=Formats.FONT_REGULAR.value)
        img_format_menu.add_command(label="Convert images to BMP", compound="left",  image=self.menu_icons["bmp"]["img"], command=Convert2bmpPopUp, font=Formats.FONT_REGULAR.value)
        img_format_menu.add_command(label="Convert images  to TIFF", compound="left",  image=self.menu_icons["tiff"]["img"], command=Convert2TIFFPopUp, font=Formats.FONT_REGULAR.value)
        img_format_menu.add_command(label="Convert images  to WEBP", compound="left",  image=self.menu_icons["webp"]["img"], command=Convert2WEBPPopUp, font=Formats.FONT_REGULAR.value)
        video_format_menu.add_command(label="Convert videos to MP4", compound="left",  image=self.menu_icons["mp4"]["img"], command=Convert2MP4PopUp, font=Formats.FONT_REGULAR.value)
        video_format_menu.add_command(label="Convert videos to AVI", compound="left",  image=self.menu_icons["avi"]["img"], command=Convert2AVIPopUp, font=Formats.FONT_REGULAR.value)
        video_format_menu.add_command(label="Convert videos to WEBM", compound="left",  image=self.menu_icons["webm"]["img"], command=Convert2WEBMPopUp, font=Formats.FONT_REGULAR.value)
        video_format_menu.add_command(label="Convert videos to MOV", compound="left",  image=self.menu_icons["mov"]["img"], command=Convert2MOVPopUp, font=Formats.FONT_REGULAR.value)
        format_menu.add_cascade(label="Convert image file formats...", compound="left",  image=self.menu_icons["image"]["img"], menu=img_format_menu, font=Formats.FONT_REGULAR.value)
        format_menu.add_cascade(label="Change video file formats...", compound="left", image=self.menu_icons["video_2"]["img"], menu=video_format_menu, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Convert file formats...", compound="left", image=self.menu_icons["convert"]["img"], menu=format_menu, font=Formats.FONT_REGULAR.value)

        rm_clr_menu = Menu(video_process_menu)
        rm_clr_menu.add_command(label="Convert to grayscale", compound="left", image=self.menu_icons["greyscale"]["img"], command=lambda: GreyscaleSingleVideoPopUp(), font=Formats.FONT_REGULAR.value)
        rm_clr_menu.add_command(label="Convert to black and white", compound="left", image=self.menu_icons["bw"]["img"], command=Convert2BlackWhitePopUp, font=Formats.FONT_REGULAR.value)
        rm_clr_menu.add_command(label="CLAHE enhance videos", compound="left", image=self.menu_icons["clahe"]["img"], command=CLAHEPopUp, font=Formats.FONT_REGULAR.value)
        rm_clr_menu.add_command(label="Interactively CLAHE enhance videos", compound="left", image=self.menu_icons["clahe"]["img"], command=InteractiveClahePopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Remove color from videos...", compound="left", image=self.menu_icons["clahe"]["img"], menu=rm_clr_menu, font=Formats.FONT_REGULAR.value)

        concatenate_menu = Menu(video_process_menu)
        concatenate_menu.add_command(label="Concatenate two videos", compound="left", image=self.menu_icons["concat"]["img"], command=ConcatenatingVideosPopUp, font=Formats.FONT_REGULAR.value)
        concatenate_menu.add_command(label="Concatenate multiple videos", compound="left", image=self.menu_icons["concat_videos"]["img"], command=lambda: ConcatenatorPopUp(config_path=None), font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Concatenate (stack) videos...", compound="left", image=self.menu_icons["concat"]["img"], menu=concatenate_menu, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Convert ROI definitions", compound="left", image=self.menu_icons["roi"]["img"], command=lambda: ConvertROIDefinitionsPopUp(), font=Formats.FONT_REGULAR.value)
        convert_data_menu = Menu(video_process_menu)
        convert_data_menu.add_command(label="Convert CSV to parquet", compound="left", image=self.menu_icons["parquet"]["img"], command=Csv2ParquetPopUp, font=Formats.FONT_REGULAR.value)
        convert_data_menu.add_command(label="Convert parquet o CSV", compound="left", image=self.menu_icons["csv_grey"]["img"], command=Parquet2CsvPopUp, font=Formats.FONT_REGULAR.value)

        video_process_menu.add_cascade(label="Convert working file type...", compound="left", image=self.menu_icons["change"]["img"], menu=convert_data_menu, font=Formats.FONT_REGULAR.value)

        video_process_menu.add_command(label="Create path plot", compound="left", image=self.menu_icons["path"]["img"], command=MakePathPlotPopUp, font=Formats.FONT_REGULAR.value)

        downsample_video_menu = Menu(video_process_menu)
        downsample_video_menu.add_command(label="Down-sample single video", compound="left", image=self.menu_icons["single_green"]["img"], command=DownsampleSingleVideoPopUp, font=Formats.FONT_REGULAR.value)
        downsample_video_menu.add_command(label="Down-sample multiple videos", compound="left", image=self.menu_icons["multiple_green"]["img"], command=DownsampleMultipleVideosPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Down-sample video...", compound="left", image=self.menu_icons["sample"]["img"], menu=downsample_video_menu, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Drop body-parts from tracking data", compound="left", image=self.menu_icons["trash"]["img"], command=DropTrackingDataPopUp, font=Formats.FONT_REGULAR.value)
        extract_frames_menu = Menu(video_process_menu, font=Formats.FONT_REGULAR.value)
        extract_frames_menu.add_command(label="Extract defined frames", command=ExtractSpecificFramesPopUp, font=Formats.FONT_REGULAR.value)
        extract_frames_menu.add_command(label="Extract frames", command=ExtractAllFramesPopUp, font=Formats.FONT_REGULAR.value)
        extract_frames_menu.add_command(label="Extract frames from seq files", command=ExtractSEQFramesPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Extract frames...", compound="left", image=self.menu_icons["frames"]["img"], menu=extract_frames_menu, font=Formats.FONT_REGULAR.value)

        video_process_menu.add_command(label="Create GIFs", compound="left", image=self.menu_icons["gif"]["img"], command=CreateGIFPopUP, font=Formats.FONT_REGULAR.value)

        video_process_menu.add_command(label="Get metric conversion factor (pixels/millimeter)", compound="left", image=self.menu_icons["calipher"]["img"], command=CalculatePixelsPerMMInVideoPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Change video brightness / contrast", compound="left", image=self.menu_icons["brightness"]["img"], command=BrightnessContrastPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Merge frames to video", compound="left", image=self.menu_icons["merge"]["img"], command=MergeFrames2VideoPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Print classifier info...", compound="left", image=self.menu_icons["print"]["img"], command=PrintModelInfoPopUp, font=Formats.FONT_REGULAR.value)

        video_process_menu.add_cascade(label="Reorganize Tracking Data", compound="left", image=self.menu_icons["reorganize"]["img"], command=PoseReorganizerPopUp, font=Formats.FONT_REGULAR.value)

        rotate_menu = Menu(menu)
        rotate_menu.add_command(label="Rotate videos", compound="left", image=self.menu_icons["flip_red"]["img"], command=RotateVideoSetDegreesPopUp, font=Formats.FONT_REGULAR.value)
        rotate_menu.add_command(label="Interactively rotate videos",  compound="left", image=self.menu_icons["flip_red"]["img"], command=VideoRotatorPopUp, font=Formats.FONT_REGULAR.value)
        rotate_menu.add_command(label="Flip videos",  compound="left", image=self.menu_icons["flip_green"]["img"], command=FlipVideosPopUp, font=Formats.FONT_REGULAR.value)
        rotate_menu.add_command(label="Reverse videos",  compound="left", image=self.menu_icons["reverse_blue"]["img"], command=ReverseVideoPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Rotate / flip / reverse videos...", compound="left", image=self.menu_icons["rotate"]["img"], menu=rotate_menu, font=Formats.FONT_REGULAR.value)

        superimpose_menu = Menu(menu)
        superimpose_menu.add_command(label="Superimpose frame numbers", compound="left", image=self.menu_icons["number_black"]["img"], command=SuperImposeFrameCountPopUp, font=Formats.FONT_REGULAR.value)
        superimpose_menu.add_command(label="Superimpose watermark", compound="left", image=self.menu_icons["watermark_green"]["img"], command=SuperimposeWatermarkPopUp, font=Formats.FONT_REGULAR.value)
        superimpose_menu.add_command(label="Superimpose timer", compound="left", image=self.menu_icons["timer"]["img"], command=SuperimposeTimerPopUp, font=Formats.FONT_REGULAR.value)
        superimpose_menu.add_command(label="Superimpose progress-bar", compound="left", image=self.menu_icons["progressbar_black"]["img"], command=SuperimposeProgressBarPopUp, font=Formats.FONT_REGULAR.value)
        superimpose_menu.add_command(label="Superimpose video on video", compound="left", image=self.menu_icons["video"]["img"], command=SuperimposeVideoPopUp, font=Formats.FONT_REGULAR.value)
        superimpose_menu.add_command(label="Superimpose video names", compound="left", image=self.menu_icons["id_card"]["img"], command=SuperimposeVideoNamesPopUp, font=Formats.FONT_REGULAR.value)
        superimpose_menu.add_command(label="Superimpose free-text", compound="left", image=self.menu_icons["text_black"]["img"], command=SuperimposeTextPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Superimpose on videos...", compound="left", image=self.menu_icons["superimpose"]["img"], menu=superimpose_menu, font=Formats.FONT_REGULAR.value)

        temporal_join_videos = Menu(menu)
        temporal_join_videos.add_command(label="Temporal join all videos in directory", command=VideoTemporalJoinPopUp, font=Formats.FONT_REGULAR.value)
        temporal_join_videos.add_command(label="Temporal join selected videos", command=ManualTemporalJoinPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_cascade(label="Temporal join videos...", compound="left", image=self.menu_icons["stopwatch"]["img"], menu=temporal_join_videos, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Box blur videos", compound="left", image=self.menu_icons["blur"]["img"], command=BoxBlurPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Cross-fade videos", compound="left", image=self.menu_icons["crossfade"]["img"], command=CrossfadeVideosPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Create average frames from videos", compound="left", image=self.menu_icons["average"]["img"], command=CreateAverageFramePopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Video background remover...", compound="left", image=self.menu_icons["remove_bg"]["img"], command=BackgroundRemoverPopUp, font=Formats.FONT_REGULAR.value)
        video_process_menu.add_command(label="Visualize pose-estimation in folder...", compound="left", image=self.menu_icons["visualize"]["img"], command=VisualizePoseInFolderPopUp, font=Formats.FONT_REGULAR.value)
        help_menu = Menu(menu)
        menu.add_cascade(label="Help", menu=help_menu)
        links_menu = Menu(help_menu)
        links_menu.add_command(label="Download weights", compound="left", image=self.menu_icons["dumbbell"]["img"], command=lambda: webbrowser.open_new(str(r"https://osf.io/sr3ck/")), font=Formats.FONT_REGULAR.value)
        links_menu.add_command(label="Download data/classifiers", compound="left", image=self.menu_icons["osf"]["img"], command=lambda: webbrowser.open_new(str(r"https://osf.io/kwge8/")), font=Formats.FONT_REGULAR.value)
        links_menu.add_command(label="Ex. feature list", compound='left', image=self.menu_icons["documentation"]["img"], command=lambda: webbrowser.open_new(str(r"https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv")), font=Formats.FONT_REGULAR.value)
        links_menu.add_command(label="SimBA github", compound="left", image=self.menu_icons["github"]["img"], command=lambda: webbrowser.open_new(str(r"https://github.com/sgoldenlab/simba")), font=Formats.FONT_REGULAR.value)
        links_menu.add_command(label="Gitter Chatroom", compound="left", image=self.menu_icons["gitter"]["img"], command=lambda: webbrowser.open_new(str(r"https://gitter.im/SimBA-Resource/community")), font=Formats.FONT_REGULAR.value)
        links_menu.add_command(label="Install FFmpeg", compound="left", image=self.menu_icons["ffmpeg"]["img"],  command=lambda: webbrowser.open_new(str(r"https://m.wikihow.com/Install-FFmpeg-on-Windows")), font=Formats.FONT_REGULAR.value)
        links_menu.add_command(label="SimBA API", compound="left", image=self.menu_icons["api"]["img"], command=lambda: webbrowser.open_new(str(r"https://simba-uw-tf-dev.readthedocs.io/")), font=Formats.FONT_REGULAR.value)
        help_menu.add_cascade(label="Links", menu=links_menu, compound="left", image=self.menu_icons["link"]["img"], font=Formats.FONT_REGULAR.value)
        help_menu.add_command(label="About", compound="left", image=self.menu_icons["about"]["img"], command=AboutSimBAPopUp, font=Formats.FONT_REGULAR.value)

        self.frame = Frame(background, bd=2, relief=SUNKEN, width=750, height=300)
        self.r_click_menu = Menu(self.root, tearoff=0)
        self.r_click_menu.add_command(label="Copy selection", command=lambda: self.copy_selection_to_clipboard(), font=Formats.FONT_REGULAR.value)
        self.r_click_menu.add_command(label="Copy all", command=lambda: self.copy_all_to_clipboard(), font=Formats.FONT_REGULAR.value)
        self.r_click_menu.add_command(label="Paste", command=lambda: self.paste_to_txt(), font=Formats.FONT_REGULAR.value)
        self.r_click_menu.add_separator()
        self.r_click_menu.add_command(label="Clear", command=lambda: self.clean_txt(), font=Formats.FONT_REGULAR.value)
        y_sb = Scrollbar(self.frame, orient=VERTICAL)
        self.frame.pack(expand=True)
        self.txt = Text(self.frame, bg="white", insertborderwidth=2, height=30, width=100, yscrollcommand=y_sb)
        if currentPlatform == OS.WINDOWS.value: self.txt.bind("<Button-3>", self.show_right_click_pop_up)
        elif currentPlatform == OS.MAC.value: self.txt.bind("<Button-2>", self.show_right_click_pop_up)
        self.txt.tag_configure(TagNames.GREETING.value, justify="center", foreground="blue", font=Formats.FONT_LARGE.value)
        self.txt.tag_configure(TagNames.ERROR.value, justify="left", foreground="red", font=Formats.FONT_REGULAR.value)
        self.txt.tag_configure(TagNames.STANDARD.value, justify="left", foreground="black", font=Formats.FONT_REGULAR.value)
        self.txt.tag_configure(TagNames.COMPLETE.value, justify="left", foreground="darkgreen", font=Formats.FONT_REGULAR.value)
        self.txt.tag_configure(TagNames.WARNING.value, justify="left", foreground="darkorange", font=Formats.FONT_REGULAR.value)
        self.txt.tag_configure("TABLE", foreground="darkorange", font=Formats.FONT_REGULAR.value, wrap="none", borderwidth=0)
        if PRINT_EMOJIS:
            self.txt.insert(INSERT, Defaults.WELCOME_MSG.value + emojis["relaxed"] + "\n" * 2)
        else:
            self.txt.insert(INSERT, Defaults.WELCOME_MSG.value + "\n" * 2)
        self.txt.tag_add(TagNames.GREETING.value, "1.0", "3.25")
        y_sb.pack(side=RIGHT, fill=Y)
        self.txt.pack(expand=True, fill="both")
        y_sb.config(command=self.txt.yview)
        self.txt.config(state=DISABLED, font=Formats.FONT_REGULAR.value)

        clear_txt_btn = SimbaButton(parent=self.frame, txt=" CLEAR", txt_clr='blue', img='clean', cmd=self.clean_txt, font=Formats.FONT_HEADER.value)
        clear_txt_btn.pack(side=BOTTOM, fill=X)
        sys.stdout = StdRedirector(self.txt)

        if OS.PYTHON_VER.value != "3.6":
            self.txt['width'], self.txt['height'] = 200, 38
            PythonVersionWarning(msg=f"SimBA is not extensively tested beyond python 3.6. You are using python {OS.PYTHON_VER.value}. If you encounter errors in python>3.6, please report them on GitHub or Gitter (links in the help toolbar) and we will work together to fix the issues!", source=self.__class__.__name__)

        if not check_ffmpeg_available():
            FFMpegNotFoundWarning(msg='SimBA could not find a FFMPEG installation on computer (as evaluated by "ffmpeg" returning None). SimBA works best with FFMPEG and it is recommended to install it on your computer', source=self.__class__.__name__)

    def restart(self):
        confirm_restart = askyesno(title="RESTART", message="Are you sure that you want restart SimBA?")
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
            if PRINT_EMOJIS:
                s = s + " " + self.emojis[tag_name]
            else:
                pass
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
