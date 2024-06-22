__author__ = "Simon Nilsson"

import os
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pkg_resources

import simba


class ConfigKey(Enum):
    GENERAL_SETTINGS = "General settings"
    PROJECT_PATH = "project_path"
    SML_SETTINGS = "SML settings"
    VIDEO_INFO_CSV = "video_info.csv"
    FOLDER_PATH = "folder_path"
    FILE_TYPE = "workflow_file_type"
    TARGET_CNT = "No_targets"
    ANIMAL_CNT = "animal_no"
    PROJECT_NAME = "project_name"
    OS = "OS_system"
    MODEL_DIR = "model_dir"
    THRESHOLD_SETTINGS = "threshold_settings"
    MIN_BOUT_LENGTH = "Minimum_bout_lengths"
    FRAME_SETTINGS = "Frame settings"
    LINE_PLOT_SETTINGS = "Line plot settings"
    PATH_PLOT_SETTINGS = "Path plot settings"
    DISTANCE_PLOT_SETTINGS = "Distance plot"
    HEATMAP_SETTINGS = "Heatmap settings"
    ROI_SETTINGS = "ROI settings"
    DIRECTIONALITY_SETTINGS = "Directionality settings"
    BODYPART_DIRECTION_VALUE = "bodypart_direction"
    PROBABILITY_THRESHOLD = "probability_threshold"
    PROCESS_MOVEMENT_SETTINGS = "process movements"
    CREATE_ENSEMBLE_SETTINGS = "create ensemble settings"
    VALIDATION_SETTINGS = "validation/run model"
    MULTI_ANIMAL_ID_SETTING = "Multi animal IDs"
    MULTI_ANIMAL_IDS = "ID_list"
    OUTLIER_SETTINGS = "Outlier settings"
    POSE_SETTING = "pose_estimation_body_parts"
    RF_JOBS = "RF_n_jobs"
    VALIDATION_VIDEO = "generate_validation_video"
    MOVEMENT_CRITERION = "movement_criterion"
    LOCATION_CRITERION = "location_criterion"
    ROI_ANIMAL_CNT = "no_of_animals"
    DISTANCE_MM = "distance_mm"
    SKLEARN_BP_PROB_THRESH = "bp_threshold_sklearn"


class Paths(Enum):
    INPUT_CSV = Path("csv/input_csv/")
    LINE_PLOT_DIR = Path("frames/output/line_plot/")
    VIDEO_INFO = Path("logs/video_info.csv")
    OUTLIER_CORRECTED = Path("csv/outlier_corrected_movement_location/")
    OUTLIER_CORRECTED_MOVEMENT = Path("csv/outlier_corrected_movement/")
    MACHINE_RESULTS_DIR = Path("csv/machine_results/")
    SKLEARN_RESULTS = Path("frames/output/sklearn_results/")
    CLUSTER_EXAMPLES = Path("frames/output/cluster_examples/")
    CLF_VALIDATION_DIR = Path("frames/output/classifier_validation/")
    CLF_DATA_VALIDATION_DIR = Path("csv/validation/")
    TEST_PATH = "/Users/simon/Desktop/envs/simba_dev/simba/"
    SINGLE_CLF_VALIDATION = Path("frames/output/validation/")
    INPUT_FRAMES_DIR = Path("frames/input/")
    ICON_ASSETS = Path("assets/icons/")
    DATA_TABLE = Path("frames/output/live_data_table/")
    ROI_FEATURES = Path("frames/output/ROI_features/")
    ROI_ANALYSIS = Path("frames/output/ROI_analysis/")
    ANNOTATED_FRAMES_DIR = Path("frames/output/annotated_frames/")
    SPONTANEOUS_ALTERNATION_VIDEOS_DIR = Path("frames/output/spontanous_alternation")
    DIRECTIONALITY_DF_DIR = Path("logs/directionality_dataframes/")
    BODY_PART_DIRECTIONALITY_DF_DIR = Path("logs/body_part_directionality_dataframes/")
    DIRECTING_ANIMALS_OUTPUT_PATH = Path("frames/output/ROI_directionality_visualize/")
    DIRECTING_BETWEEN_ANIMALS_OUTPUT_PATH = Path("frames/output/Directing_animals/")
    DIRECTING_BETWEEN_ANIMAL_BODY_PART_OUTPUT_PATH = Path(
        "frames/output/Body_part_directing_animals/"
    )
    BP_NAMES = Path("logs/measures/pose_configs/bp_names/project_bp_names.csv")
    SIMBA_BP_CONFIG_PATH = Path("pose_configurations/bp_names/bp_names.csv")
    SIMBA_SHAP_CATEGORIES_PATH = Path(
        "assets/shap/feature_categories/shap_feature_categories.csv"
    )
    SIMBA_FEATURE_EXTRACTION_COL_NAMES_PATH = Path(
        "assets/lookups/feature_extraction_headers.csv"
    )
    SIMBA_NO_ANIMALS_PATH = Path("pose_configurations/no_animals/no_animals.csv")
    SIMBA_SHAP_IMG_PATH = Path("assets/shap/")
    SCHEMATICS = Path("pose_configurations/schematics/")
    PROJECT_POSE_CONFIG_NAMES = Path(
        "pose_configurations/configuration_names/pose_config_names.csv"
    )
    CONCAT_VIDEOS_DIR = Path("frames/output/merged/")
    GANTT_PLOT_DIR = Path("frames/output/gantt_plots/")
    HEATMAP_CLF_LOCATION_DIR = Path("frames/output/heatmaps_classifier_locations/")
    HEATMAP_LOCATION_DIR = Path("frames/output/heatmaps_locations/")
    FRAMES_OUTPUT_DIR = Path("frames/output/")
    FEATURES_EXTRACTED_DIR = Path("csv/features_extracted/")
    TARGETS_INSERTED_DIR = Path("csv/targets_inserted/")
    PATH_PLOT_DIR = Path("frames/output/path_plots")
    ABOUT_ME = Path("assets/img/about_me.png")
    PROBABILITY_PLOTS_DIR = Path("frames/output/probability_plots/")
    ROI_DEFINITIONS = Path("measures/ROI_definitions.h5")
    DETAILED_ROI_DATA_DIR = Path("logs/Detailed_ROI_data/")
    SHAP_LOGS = Path("logs/shap/")
    SPLASH_PATH_WINDOWS = Path("assets/img/splash.png")
    SPLASH_PATH_LINUX = Path("assets/img/splash.PNG")
    SPLASH_PATH_MOVIE = Path("assets/img/splash_2024.mp4")
    BG_IMG_PATH = Path("assets/img/bg_2024.png")
    LOGO_ICON_WINDOWS_PATH = Path("assets/icons/SimBA_logo.ico")
    LOGO_ICON_DARWIN_PATH = Path("assets/icons/SimBA_logo.png")
    UNSUPERVISED_MODEL_NAMES = Path("assets/lookups/model_names.parquet")
    CRITICAL_VALUES = Path("simba/assets/lookups/critical_values_05.pickle")


class Formats(Enum):
    MP4_CODEC = "mp4v"
    AVI_CODEC = "XVID"
    BATCH_CODEC = "libx264"
    NUMERIC_DTYPES = (np.float32, np.float64, np.int64, np.int32, np.int8, int, float)
    LABELFRAME_HEADER_FORMAT = ("Helvetica", 12, "bold")
    LABELFRAME_HEADER_CLICKABLE_FORMAT = ("Helvetica", 12, "bold", "underline")
    LABELFRAME_HEADER_CLICKABLE_COLOR = f"#{5:02x}{99:02x}{193:02x}"
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    XLXS = "xlsx"
    PERIMETER = "perimeter"
    AREA = "area"
    H5 = "h5"
    ROOT_WINDOW_SIZE = "750x750"
    FONT = cv2.FONT_HERSHEY_TRIPLEX
    TKINTER_FONT = ("Rockwell", 11)
    DLC_NETWORK_FILE_NAMES = [
        "dlc_resnet50",
        "dlc_resnet_50",
        "dlc_dlcrnetms5",
        "dlc_effnet_b0",
        "dlc_resnet101",
    ]
    DLC_FILETYPES = {
        "skeleton": ["sk.h5", "sk_filtered.h5"],
        "box": ["bx.h5", "bx_filtered.h5"],
        "ellipse": ["el.h5", "el_filtered.h5"],
    }


class Options(Enum):
    ROLLING_WINDOW_DIVISORS = [2, 5, 6, 7.5, 15]
    CLF_MODELS = ["RF", "GBC", "XGBoost"]
    CLF_MAX_FEATURES = ["sqrt", "log", "None"]
    CLF_CRITERION = ["gini", "entropy"]
    UNDERSAMPLE_OPTIONS = ["None", "random undersample"]
    OVERSAMPLE_OPTIONS = ["None", "SMOTE", "SMOTEENN"]
    CLASS_WEIGHT_OPTIONS = ["None", "balanced", "balanced_subsample", "custom"]
    BUCKET_METHODS = [
        "fd",
        "doane",
        "auto",
        "scott",
        "stone",
        "rice",
        "sturges",
        "sqrt",
    ]
    CLF_TEST_SIZE_OPTIONS = [
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
    ]
    PALETTE_OPTIONS = [
        "magma",
        "jet",
        "inferno",
        "plasma",
        "viridis",
        "gnuplot2",
        "RdBu",
        "winter",
    ]
    PALETTE_OPTIONS_CATEGORICAL = [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
    ]
    RESOLUTION_OPTIONS = [
        "320×240",
        "640×480",
        "720×480",
        "800×640",
        "960×800",
        "1120×960",
        "1280×720",
        "1980×1080",
    ]
    DPI_OPTIONS = [100, 200, 400, 800, 1600, 3200]
    RESOLUTION_OPTIONS_2 = ["AUTO", 240, 320, 480, 640, 720, 800, 960, 1120, 1080, 1980]
    SPEED_OPTIONS = [round(x, 1) for x in list(np.arange(0.1, 2.1, 0.1))]
    PERFORM_FLAGS = ["yes", True, "True"]
    RUN_OPTIONS_FLAGS = ["yes", True, "True", "False", "no", False, "true", "false"]
    SCALER_NAMES = ["MIN-MAX", "STANDARD", "QUANTILE"]
    HEATMAP_SHADING_OPTIONS = ["gouraud", "flat"]
    HEATMAP_BIN_SIZE_OPTIONS = [
        "10×10",
        "20×20",
        "40×40",
        "80×80",
        "100×100",
        "160×160",
        "320×320",
        "640×640",
        "1280×1280",
    ]
    INTERPOLATION_OPTIONS = [
        "Animal(s): Nearest",
        "Animal(s): Linear",
        "Animal(s): Quadratic",
        "Body-parts: Nearest",
        "Body-parts: Linear",
        "Body-parts: Quadratic",
    ]
    INTERPOLATION_OPTIONS_W_NONE = [
        "None",
        "Animal(s): Nearest",
        "Animal(s): Linear",
        "Animal(s): Quadratic",
        "Body-parts: Nearest",
        "Body-parts: Linear",
        "Body-parts: Quadratic",
    ]
    IMPORT_TYPE_OPTIONS = [
        "CSV (DLC/DeepPoseKit)",
        "JSON (BENTO)",
        "H5 (multi-animal DLC)",
        "SLP (SLEAP)",
        "CSV (SLEAP)",
        "H5 (SLEAP)",
        "TRK (multi-animal APT)",
        "MAT (DANNCE 3D)",
    ]
    SMOOTHING_OPTIONS = ["Gaussian", "Savitzky Golay"]
    MULTI_DLC_TYPE_IMPORT_OPTION = ["skeleton", "box", "ellipse"]
    FEATURE_SUBSET_OPTIONS = [
        "Two-point body-part distances (mm)",
        "Within-animal three-point body-part angles (degrees)",
        "Within-animal three-point convex hull perimeters (mm)",
        "Within-animal four-point convex hull perimeters (mm)",
        "Entire animal convex hull perimeters (mm)",
        "Entire animal convex hull area (mm2)",
        "Frame-by-frame body-part movements (mm)",
        "Frame-by-frame body-part distances to ROI centers (mm)",
        "Frame-by-frame body-parts inside ROIs (Boolean)",
    ]
    SMOOTHING_OPTIONS_W_NONE = ["None", "Gaussian", "Savitzky Golay"]
    VIDEO_FORMAT_OPTIONS = ["mp4", "avi"]
    ALL_VIDEO_FORMAT_OPTIONS = (".avi", ".mp4", ".mov", ".flv", ".m4v", ".webm")
    ALL_IMAGE_FORMAT_OPTIONS = (".bmp", ".png", ".jpeg", ".jpg", ".webp")
    ALL_IMAGE_FORMAT_STR_OPTIONS = ".bmp .png .jpeg .jpg"
    ALL_VIDEO_FORMAT_STR_OPTIONS = ".avi .mp4 .mov .flv .m4v .webm"
    WORKFLOW_FILE_TYPE_OPTIONS = ["csv", "parquet"]
    WORKFLOW_FILE_TYPE_STR_OPTIONS = ".csv .parquet"
    TRACKING_TYPE_OPTIONS = ["Classic tracking", "Multi tracking", "3D tracking"]
    UNSUPERVISED_FEATURE_OPTIONS = [
        "INCLUDE FEATURE DATA (ORIGINAL)",
        "INCLUDE FEATURES (SCALED)",
        "EXCLUDE FEATURE DATA",
    ]
    TIMEBINS_MEASURMENT_OPTIONS = [
        "First occurrence (s)",
        "Event count",
        "Total event duration (s)",
        "Mean event duration (s)",
        "Median event duration (s)",
        "Mean event interval (s)",
        "Median event interval (s)",
    ]
    CLF_DESCRIPTIVES_OPTIONS = [
        "Bout count",
        "Total event duration (s)",
        "Mean event bout duration (s)",
        "Median event bout duration (s)",
        "First event occurrence (s)",
        "Mean event bout interval duration (s)",
        "Median event bout interval duration (s)",
    ]
    CLASSICAL_TRACKING_OPTIONS = [
        "1 animal; 4 body-parts",
        "1 animal; 7 body-parts",
        "1 animal; 8 body-parts",
        "1 animal; 9 body-parts",
        "2 animals; 8 body-parts",
        "2 animals; 14 body-parts",
        "2 animals; 16 body-parts",
        "MARS",
    ]
    MULTI_ANIMAL_TRACKING_OPTIONS = [
        "Multi-animals; 4 body-parts",
        "Multi-animals; 7 body-parts",
        "Multi-animals; 8 body-parts",
        "AMBER",
    ]
    THREE_DIM_TRACKING_OPTIONS = ["3D tracking"]
    TRAIN_TEST_SPLIT = ["FRAMES", "BOUTS"]
    BOOL_STR_OPTIONS = ["TRUE", "FALSE"]
    GANTT_VALIDATION_OPTIONS = [
        "None",
        "Gantt chart: final frame only (slightly faster)",
        "Gantt chart: video",
    ]
    THIRD_PARTY_ANNOTATION_APPS_OPTIONS = [
        "BORIS",
        "ETHOVISION",
        "OBSERVER",
        "SOLOMON",
        "DEEPETHOGRAM",
        "BENTO",
    ]
    THIRD_PARTY_ANNOTATION_ERROR_OPTIONS = [
        "INVALID annotations file data format",
        "ADDITIONAL third-party behavior detected",
        "Annotations OVERLAP conflict",
        "ZERO third-party video behavior annotations found",
        "Annotations and pose FRAME COUNT conflict",
        "Annotations EVENT COUNT conflict",
        "Annotations data file NOT FOUND",
    ]
    SCALER_OPTIONS = ["MIN-MAX", "STANDARD", "QUANTILE"]
    MIN_MAX_SCALER = "MIN-MAX"
    STANDARD_SCALER = "STANDARD"
    QUANTILE_SCALER = "QUANTILE"


class TextOptions(Enum):
    FIRST_LINE_SPACING = (
        2  # DISTANCE MULTIPLIER BETWEEN FIRST PRINTED ROW AND THE TOP OF THE IMAGE.
    )
    LINE_SPACING = 1  # DISTANCE MULTIPLIER BETWEEN 2nd AND THIRD, THIRD AND FOURTH ETC. PRINTED ROWS.
    BORDER_BUFFER_X = 5  # ADDITIONAL PIXEL BUFFER DISTANCE BETWEEN THE START OF THE IMAGE AND FIRST PRINTED CHARACTER ON X AXIS
    BORDER_BUFFER_Y = 10  # ADDITIONAL PIXEL BUFFER DISTANCE BETWEEN THE START OF THE IMAGE AND FIRST PRINTED CHARACTER ON Y AXIS
    FONT_SCALER = 0.8  # CONSTANT USED TO SCALE FONT ACCORDING TO RESOLUTION. INCREASING VALUE WILL RESULT IN LARGER TEXT.
    RESOLUTION_SCALER = 1500  # CONSTANT USED TO SCALE SPACINGS, FONT, AND RADIUS OF CIRCLES. LARGER NUMBER WILL RESULT IN SMALLER SPACINGS, FONT, AND RADIUS OF CIRCLES.
    RADIUS_SCALER = 10  # CONSTANT USED TO SCALE CIRCLES. INCREASING VALUE WILL RESULT IN LARGER CIRCLES.
    SPACE_SCALER = 25  # CONSTANT USED TO SCALE SPACE BETWEEN PRINTED ROWS. INCREASING VALUE WILL RESULT IN LARGER PIXEL DISTANCES BETWEEN SEQUENTIAL ROWS.
    TEXT_THICKNESS = 1  # THE THICKNESS OR "BOLDNESS" OF THE FONT IN PIXELS.
    LINE_THICKNESS = (
        2  # THICKNESS OF LINES IN CIRCLES, BOUNDING BOXES AND OTHER GEOMETRIES
    )
    COLOR = (
        147,
        20,
        255,
    )  # THE COLOR OF THE TEXT IN BGR. (147, 20, 255)  REPRESENT DEEP PINK: TYPICALLY NOT A COLOR IN ANIMAL ARENA.
    FONT = cv2.FONT_HERSHEY_SIMPLEX


class Defaults(Enum):
    MAX_TASK_PER_CHILD = 10
    LARGE_MAX_TASK_PER_CHILD = 1000
    MAXIMUM_MAX_TASK_PER_CHILD = 8000
    CHUNK_SIZE = 1
    SPLASH_TIME = 2500
    try:
        WELCOME_MSG = f'Welcome fellow scientists! \n SimBA v.{pkg_resources.get_distribution("simba-uw-tf-dev").version} \n '
    except pkg_resources.DistributionNotFound:
        WELCOME_MSG = f'Welcome fellow scientists! \n SimBA v. "dev" \n '
    BROWSE_FOLDER_BTN_TEXT = "Browse Folder"
    BROWSE_FILE_BTN_TEXT = "Browse File"
    NO_FILE_SELECTED_TEXT = "No file selected"
    STR_SPLIT_DELIMITER = "\t"


class TagNames(Enum):
    GREETING = "greeting"
    COMPLETE = "complete"
    WARNING = "warning"
    ERROR = "error"
    TRASH = "trash"
    STANDARD = "standard"
    CLASS_INIT = "CLASS_INIT"


class DirNames(Enum):
    PROJECT = "project_folder"
    MODEL = "models"
    CONFIGS = "configs"
    CSV = "csv"
    FRAMES = "frames"
    LOGS = "logs"
    MEASURES = "measures"
    POSE_CONFIGS = "pose_configs"
    BP_NAMES = "bp_names"
    VIDEOS = "videos"
    FEATURES_EXTRACTED = "features_extracted"
    INPUT_CSV = "input_csv"
    MACHINE_RESULTS = "machine_results"
    OUTLIER_MOVEMENT = "outlier_corrected_movement"
    OUTLIER_MOVEMENT_LOCATION = "outlier_corrected_movement_location"
    TARGETS_INSERTED = "targets_inserted"
    INPUT = "input"
    OUTPUT = "output"


class Keys(Enum):
    ROI_RECTANGLES = "rectangles"
    ROI_CIRCLES = "circleDf"
    ROI_POLYGONS = "polygons"
    DOCUMENTATION = "documentation"
    FRAME_COUNT = "frame_count"


class UMAPParam(Enum):
    N_NEIGHBORS = "n_neighbors"
    MIN_DISTANCE = "min_distance"
    SPREAD = "spread"
    VARIANCE = "variance"
    SCALER = "scaler"
    HYPERPARAMETERS = [N_NEIGHBORS, MIN_DISTANCE, SPREAD, SCALER, VARIANCE]


class Dtypes(Enum):
    NAN = "NaN"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    FOLDER = "folder_path"
    NONE = "None"
    SQRT = "sqrt"
    ENTROPY = "entropy"


class Methods(Enum):
    USER_DEFINED = "user_defined"
    CLASSIC_TRACKING = "Classic tracking"
    THREE_D_TRACKING = "3D tracking"
    MULTI_TRACKING = "Multi tracking"
    CREATE_POSE_CONFIG = "Create pose config..."
    RANDOM_UNDERSAMPLE = "random undersample"
    SMOTE = "SMOTE"
    SMOTEENN = "SMOTEENN"
    GAUSSIAN = "Gaussian"
    SAVITZKY_GOLAY = "Savitzky Golay"
    SPLIT_TYPE_FRAMES = "FRAMES"
    SPLIT_TYPE_BOUTS = "BOUTS"
    BORIS = "BORIS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    ANOVA = "ANOVA"
    INVALID_THIRD_PARTY_APPENDER_FILE = "INVALID annotations file data format"
    ADDITIONAL_THIRD_PARTY_CLFS = "ADDITIONAL third-party behavior detected"
    ZERO_THIRD_PARTY_VIDEO_ANNOTATIONS = "ZERO third-party video annotations found"
    THIRD_PARTY_FPS_CONFLICT = "Annotations and pose FPS conflict"
    THIRD_PARTY_EVENT_COUNT_CONFLICT = "Annotations EVENT COUNT conflict"
    THIRD_PARTY_EVENT_OVERLAP = "Annotations OVERLAP inaccuracy"
    ZERO_THIRD_PARTY_VIDEO_BEHAVIOR_ANNOTATIONS = (
        "ZERO third-party video behavior annotations found"
    )
    THIRD_PARTY_FRAME_COUNT_CONFLICT = "Annotations and pose FRAME COUNT conflict"
    THIRD_PARTY_ANNOTATION_FILE_NOT_FOUND = "Annotations data file NOT FOUND"


class OS(Enum):
    WINDOWS = "Windows"
    LINUX = "Linux"
    MAC = "Darwin"
    PYTHON_VER = str(f"{sys.version_info.major}.{sys.version_info.minor}")


class Links(Enum):
    FEATURE_SUBSETS = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/feature_subsets.md"
    )
    HEATMAP_LOCATION = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#heatmaps"
    )
    HEATMAP_CLF = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-classification-heatmaps"
    DATA_TABLES = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-data-tables"
    CONCAT_VIDEOS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#merging-concatenating-videos"
    DISTANCE_PLOTS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-distance-plots"
    PATH_PLOTS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-path-plots"
    VISUALIZE_CLF_PROBABILITIES = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-classification-probabilities"
    GANTT_PLOTS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-gantt-charts"
    SKLEARN_PLOTS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#visualizing-classifications"
    ANALYZE_ML_RESULTS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results"
    FSTTC = "https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md"
    KLEINBERG = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md"
    )
    DOWNSAMPLE = "https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#downsample-video"
    VIDEO_TOOLS = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md"
    )
    SET_RUN_ML_PARAMETERS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-3-run-the-classifier-on-new-data"
    OULIERS = (
        "https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf"
    )
    REMOVE_CLF = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-optional-step--import-more-dlc-tracking-data-or-videos"
    ROI_FEATURES_PLOT = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-5-visualizing-roi-features"
    ROI_DATA_PLOT = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-4-visualizing-roi-data"
    DIRECTING_ANIMALS_PLOTS = "https://github.com/sgoldenlab/simba/blob/master/docs/directionality_between_animals.md"
    CLF_VALIDATION = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/classifier_validation.md"
    )
    BATCH_PREPROCESS = "https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md"
    THIRD_PARTY_ANNOTATION_NEW = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot_new.md"
    )
    OUT_OF_SAMPLE_VALIDATION = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-8-evaluating-the-model-on-new-out-of-sample-data"
    ROI = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md"
    ROI_FEATURES = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data"
    ROI_DATA_ANALYSIS = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-2-analyzing-roi-data"
    DATA_ANALYSIS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results"
    ANALYZE_ROI = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-2-analyzing-roi-data"
    EXTRACT_FEATURES = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-5-extract-features"
    USER_DEFINED_FEATURE_EXTRACTION = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/extractFeatures.md"
    )
    APPEND_ROI_FEATURES = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data"
    LABEL_BEHAVIOR = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md"
    )
    THIRD_PARTY_ANNOTATION = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md"
    )
    PSEUDO_LBL = "https://github.com/sgoldenlab/simba/blob/master/docs/pseudoLabel.md"
    ADVANCED_LBL = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md"
    )
    TRAIN_ML_MODEL = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model"
    VISUALIZATION = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-5--visualizing-results"
    PLOTLY = "https://github.com/sgoldenlab/simba/blob/master/docs/plotly_dash.md"
    VIDEO_PARAMETERS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-3-set-video-parameters"
    OUTLIERS_DOC = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-4-outlier-correction"
    CREATE_PROJECT = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-1-generate-project-config"
    LOAD_PROJECT = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#part-2-load-project-1"
    SCENARIO_2 = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md"
    BBOXES = "https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md"
    CUE_LIGHTS = (
        "https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md"
    )
    ADDITIONAL_IMPORTS = "https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-2-optional-step--import-more-dlc-tracking-data-or-videos"
    AGGREGATE_BOOL_STATS = "https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#compute-aggregate-conditional-statistics-from-boolean-fields"
    GITHUB_REPO = "https://github.com/sgoldenlab/simba"
    OSF_REPO = "https://osf.io/tmu6y/"
    GITTER = "https://gitter.im/SimBA-Resource/community"
    COUNT_ANNOTATIONS_IN_PROJECT = "https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md#count-annotations-in-simba-project"
    COUNT_ANNOTATIONS_OUTSIDE_PROJECT = "https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#extract-project-annotation-counts"
    CIRCLE_CROP = "https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_tools.md#circle-crop"


class Labelling(Enum):
    PADDING = 5
    MAX_FRM_SIZE = (1280, 650)
    VIDEO_FRAME_SIZE = (700, 500)
    PLAY_VIDEO_SCRIPT_PATH = os.path.join(
        os.path.dirname(simba.__file__), "labelling", "play_annotation_video.py"
    )
    VALID_ANNOTATIONS_ADVANCED = [0, 1, 2]


class GeometryEnum(Enum):
    CAP_STYLE_MAP = {"round": 1, "square": 2, "flat": 3}
    HISTOGRAM_COMPARISON_MAP = {
        "correlation": 0,
        "chi_square": 1,
        "intersection": 2,
        "bhattacharyya": 3,
        "hellinger": 4,
        "chi_square_alternative": 5,
    }
    CONTOURS_MODE_MAP = {"exterior": 0, "all": 1, "interior": 3}
    CONTOURS_RETRIEVAL_MAP = {"simple": 2, "none": 0, "l1": 3, "kcos": 4}
    RANKING_METHODS = [
        "area",
        "min_distance",
        "max_distance",
        "mean_distance",
        "left_to_right",
        "top_to_bottom",
    ]


class MLParamKeys(Enum):
    CLASSIFIER = "classifier"
    RF_META_DATA = 'RF_meta_data'
    CLASSIFIER_NAME = "classifier_name"
    RF_ESTIMATORS = "rf_n_estimators"
    RF_CRITERION = "rf_criterion"
    TT_SIZE = "train_test_size"
    MIN_LEAF = "rf_min_sample_leaf"
    RF_METADATA = "generate_rf_model_meta_data_file"
    EX_DECISION_TREE = "generate_example_decision_tree"
    CLF_REPORT = "generate_classification_report"
    IMPORTANCE_LOG = "generate_features_importance_log"
    IMPORTANCE_BAR_CHART = "generate_features_importance_bar_graph"
    PERMUTATION_IMPORTANCE = "compute_feature_permutation_importance"
    LEARNING_CURVE = "generate_sklearn_learning_curves"
    PRECISION_RECALL = "generate_precision_recall_curves"
    RF_MAX_FEATURES = "rf_max_features"
    RF_MAX_DEPTH = "rf_max_depth"
    LEARNING_CURVE_K_SPLITS = "learning_curve_k_splits"
    LEARNING_CURVE_DATA_SPLITS = "learning_curve_data_splits"
    N_FEATURE_IMPORTANCE_BARS = "n_feature_importance_bars"
    SHAP_SCORES = "generate_shap_scores"
    SHAP_PRESENT = "shap_target_present_no"
    SHAP_ABSENT = "shap_target_absent_no"
    SHAP_SAVE_ITERATION = "shap_save_iteration"
    PARTIAL_DEPENDENCY = "partial_dependency"
    TRAIN_TEST_SPLIT_TYPE = "train_test_split_type"
    UNDERSAMPLE_SETTING = "under_sample_setting"
    UNDERSAMPLE_RATIO = "under_sample_ratio"
    OVERSAMPLE_SETTING = "over_sample_setting"
    OVERSAMPLE_RATIO = "over_sample_ratio"
    CLASS_WEIGHTS = "class_weights"
    CLASS_CUSTOM_WEIGHTS = "class_custom_weights"
    EX_DECISION_TREE_FANCY = "generate_example_decision_tree_fancy"
    IMPORTANCE_BARS_N = "N_feature_importance_bars"
    LEARNING_DATA_SPLITS = "LearningCurve_shuffle_data_splits"
    MODEL_TO_RUN = "model_to_run"
    SAVE_TRAIN_TEST_FRM_IDX = "save_train_test_frm_idx"
    SHAP_MULTIPROCESS = "shap_multiprocess"
    CLASSIFIER_MAP = "classifier_map"

class TestPaths(Enum):
    CRITICAL_VALUES = "../simba/assets/lookups/critical_values_05.pickle"
