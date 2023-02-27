from enum import Enum
from pathlib import Path
import cv2

class ReadConfig(Enum):
    GENERAL_SETTINGS = 'General settings'
    PROJECT_PATH = 'project_path'
    SML_SETTINGS = 'SML settings'
    VIDEO_INFO_CSV = 'video_info.csv'
    FOLDER_PATH = 'folder_path'
    FILE_TYPE = 'workflow_file_type'
    TARGET_CNT = 'No_targets'
    ANIMAL_CNT = 'animal_no'
    PROJECT_NAME = 'project_name'
    OS = 'OS_system'
    MODEL_DIR = 'model_dir'
    THRESHOLD_SETTINGS = 'threshold_settings'
    MIN_BOUT_LENGTH = 'Minimum_bout_lengths'
    FRAME_SETTINGS = 'Frame settings'
    LINE_PLOT_SETTINGS = 'Line plot settings'
    PATH_PLOT_SETTINGS = 'Path plot settings'
    DISTANCE_PLOT_SETTINGS = 'Distance plot'
    HEATMAP_SETTINGS = 'Heatmap settings'
    ROI_SETTINGS = 'ROI settings'
    PROBABILITY_THRESHOLD = 'probability_threshold'
    PROCESS_MOVEMENT_SETTINGS = 'process movements'
    CREATE_ENSEMBLE_SETTINGS = 'create ensemble settings'
    VALIDATION_SETTINGS = 'validation/run model'
    MULTI_ANIMAL_ID_SETTING = 'Multi animal IDs'
    MULTI_ANIMAL_IDS = 'ID_list'
    OUTLIER_SETTINGS = 'Outlier settings'
    CLASS_WEIGHTS = 'class_weights'
    CUSTOM_WEIGHTS = 'custom_weights'
    CLASSIFIER = 'classifier'
    TT_SIZE = 'train_test_size'
    MODEL_TO_RUN = 'model_to_run'
    UNDERSAMPLE_SETTING = 'under_sample_setting'
    OVERSAMPLE_SETTING = 'over_sample_setting'
    UNDERSAMPLE_RATIO = 'under_sample_ratio'
    OVERSAMPLE_RATIO = 'over_sample_ratio'
    RF_ESTIMATORS = 'RF_n_estimators'
    RF_MAX_FEATURES = 'RF_max_features'
    RF_CRITERION = 'RF_criterion'
    MIN_LEAF = 'RF_min_sample_leaf'
    PERMUTATION_IMPORTANCE = 'compute_permutation_importance'
    LEARNING_CURVE = 'generate_learning_curve'
    PRECISION_RECALL = 'generate_precision_recall_curve'
    EX_DECISION_TREE = 'generate_example_decision_tree'
    EX_DECISION_TREE_FANCY = 'generate_example_decision_tree_fancy'
    CLF_REPORT = 'generate_classification_report'
    IMPORTANCE_LOG = 'generate_features_importance_log'
    IMPORTANCE_BAR_CHART = 'generate_features_importance_bar_graph'
    SHAP_SCORES = 'generate_shap_scores'
    RF_METADATA = 'RF_meta_data'
    LEARNING_CURVE_K_SPLITS = 'LearningCurve_shuffle_k_splits'
    LEARNING_DATA_SPLITS = 'LearningCurve_shuffle_data_splits'
    IMPORTANCE_BARS_N = 'N_feature_importance_bars'
    SHAP_PRESENT = 'shap_target_present_no'
    SHAP_ABSENT = 'shap_target_absent_no'
    POSE_SETTING = 'pose_estimation_body_parts'
    RF_JOBS = 'RF_n_jobs'
    VALIDATION_VIDEO = 'generate_validation_video'
    MOVEMENT_CRITERION = 'movement_criterion'
    LOCATION_CRITERION = 'location_criterion'
    ROI_ANIMAL_CNT = 'no_of_animals'
    DISTANCE_MM = 'distance_mm'
    SKLEARN_BP_PROB_THRESH = 'bp_threshold_sklearn'
    SPLIT_TYPE = 'train_test_split_type'

class Paths(Enum):
    INPUT_CSV = Path('csv/input_csv/')
    LINE_PLOT_DIR = Path('frames/output/line_plot/')
    VIDEO_INFO = Path('logs/video_info.csv')
    OUTLIER_CORRECTED = Path('csv/outlier_corrected_movement_location/')
    MACHINE_RESULTS_DIR = Path('csv/machine_results/')
    SKLEARN_RESULTS = Path('frames/output/sklearn_results/')
    CLUSTER_EXAMPLES = Path('frames/output/cluster_examples/')
    CLF_VALIDATION_DIR = Path('frames/output/classifier_validation/')
    SINGLE_CLF_VALIDATION = Path('frames/output/validation/')
    ICON_ASSETS = Path('assets/icons/')
    DATA_TABLE = Path('frames/output/live_data_table/')
    ROI_FEATURES = Path('frames/output/ROI_features/')
    ROI_ANALYSIS = Path('frames/output/ROI_analysis/')
    DIRECTIONALITY_DF_DIR = Path('logs/directionality_dataframes/')
    DIRECTING_ANIMALS_OUTPUT_PATH = Path('frames/output/ROI_directionality_visualize/')
    DIRECTING_BETWEEN_ANIMALS_OUTPUT_PATH = Path('frames/output/Directing_animals/')
    BP_NAMES = Path('logs/measures/pose_configs/bp_names/project_bp_names.csv')
    SIMBA_BP_CONFIG_PATH = Path('pose_configurations/bp_names/bp_names.csv')
    SIMBA_SHAP_CATEGORIES_PATH = Path('assets/shap/feature_categories/shap_feature_categories.csv')
    SIMBA_NO_ANIMALS_PATH = Path('pose_configurations/no_animals/no_animals.csv')
    SIMBA_SHAP_IMG_PATH = Path('assets/shap/')
    SCHEMATICS = Path('pose_configurations/schematics/')
    PROJECT_POSE_CONFIG_NAMES = Path('pose_configurations/configuration_names/pose_config_names.csv')
    CONCAT_VIDEOS_DIR = Path('frames/output/merged/')
    GANTT_PLOT_DIR = Path('frames/output/gantt_plots/')
    HEATMAP_CLF_LOCATION_DIR = Path('frames/output/heatmaps_classifier_locations/')
    HEATMAP_LOCATION_DIR = Path('frames/output/heatmaps_locations/')
    FEATURES_EXTRACTED_DIR = Path('csv/features_extracted/')
    TARGETS_INSERTED_DIR = Path('csv/targets_inserted/')
    PATH_PLOT_DIR = Path('frames/output/path_plots')
    ABOUT_ME = 'About_me_050122_1.png'
    PROBABILITY_PLOTS_DIR = Path('frames/output/probability_plots/')
    ROI_DEFINITIONS = Path('measures/ROI_definitions.h5')
    DETAILED_ROI_DATA_DIR = Path('logs/Detailed_ROI_data/')
    SHAP_LOGS = Path('logs/shap/')
    UNSUPERVISED_MODEL_NAMES = Path('assets/unsupervised/model_names.parquet')

class Formats(Enum):
    MP4_CODEC = 'mp4v'
    AVI_CODEC = 'XVID'
    LABELFRAME_HEADER_FORMAT = ('Helvetica', 12, 'bold')
    CSV = 'csv'
    FONT = cv2.FONT_HERSHEY_TRIPLEX

class Options(Enum):
    CLF_MODELS = ['RF', 'GBC', 'XGBoost']
    CLF_MAX_FEATURES = ['sqrt', 'log', 'None']
    CLF_CRITERION = ['gini', 'entropy']
    UNDERSAMPLE_OPTIONS = ['None', 'random undersample']
    OVERSAMPLE_OPTIONS = ['None', 'SMOTE', 'SMOTEENN']
    CLASS_WEIGHT_OPTIONS = ['None', 'balanced', 'balanced_subsample', 'custom']
    CLF_TEST_SIZE_OPTIONS = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    PALETTE_OPTIONS = ['magma', 'jet', 'inferno', 'plasma', 'viridis', 'gnuplot2']
    PALETTE_OPTIONS_CATEGORICAL = ['Pastel1', 'Paired', 'Accent', 'Dark', 'Set1', 'Set2', 'tab10']
    RESOLUTION_OPTIONS = ['640×480', '800×640', '960×800', '1120×960']
    PERFORM_FLAGS = ['yes', True, 'True']
    RUN_OPTIONS_FLAGS = ['yes', True, 'True', 'False', 'no', False]
    SCALER_NAMES = ['MIN-MAX', 'STANDARD', 'QUANTILE']
    UNSUPERVISED_FEATURE_OPTIONS = ['INCLUDE FEATURE DATA (ORIGINAL)', 'INCLUDE FEATURES (SCALED)', 'EXCLUDE FEATURE DATA']
    TRAIN_TEST_SPLIT = ['FRAMES', 'BOUTS']

class Defaults(Enum):
    MAX_TASK_PER_CHILD = 10
    CHUNK_SIZE = 1
    BROWSE_FOLDER_BTN_TEXT = 'Browse Folder'
    BROWSE_FILE_BTN_TEXT = 'Browse File'
    NO_FILE_SELECTED_TEXT = 'No file selected'

class DirNames(Enum):
    PROJECT = 'project_folder'
    MODEL = 'models'
    CONFIGS = 'configs'
    CSV = 'csv'
    FRAMES = 'frames'
    LOGS = 'logs'
    MEASURES = 'measures'
    POSE_CONFIGS = 'pose_configs'
    BP_NAMES = 'bp_names'
    VIDEOS = 'videos'
    FEATURES_EXTRACTED = 'features_extracted'
    INPUT_CSV = 'input_csv'
    MACHINE_RESULTS = 'machine_results'
    OUTLIER_MOVEMENT = 'outlier_corrected_movement'
    OUTLIER_MOVEMENT_LOCATION = 'outlier_corrected_movement_location'
    TARGETS_INSERTED = 'targets_inserted'
    INPUT = 'input'
    OUTPUT = 'output'

class Keys(Enum):
    ROI_RECTANGLES = 'rectangles'
    ROI_CIRCLES = 'circleDf'
    ROI_POLYGONS = 'polygons'

class Dtypes(Enum):
    NAN = 'NaN'
    STR = 'str'
    INT = 'int'
    FLOAT = 'float'
    NONE = 'None'
    SQRT = 'sqrt'
    ENTROPY = 'entropy'

class Methods(Enum):
    RANDOM_UNDERSAMPLE = 'random undersample'
    SMOTE = 'SMOTE'
    SMOTEENN = 'SMOTEENN'
    GAUSSIAN = 'Gaussian'
    SAVITZKY_GOLAY = 'Savitzky Golay'
    USER_DEFINED = 'user_defined'
    SPLIT_TYPE_FRAMES = 'FRAMES'
    SPLIT_TYPE_BOUTS = 'BOUTS'

class MetaKeys(Enum):
    CLF_NAME = 'classifier_name'
    RF_ESTIMATORS = 'rf_n_estimators'
    CRITERION = 'rf_criterion'
    TT_SIZE = 'train_test_size'
    MIN_LEAF = 'rf_min_sample_leaf'
    META_FILE = 'generate_rf_model_meta_data_file'
    EX_DECISION_TREE = 'generate_example_decision_tree'
    CLF_REPORT = 'generate_classification_report'
    IMPORTANCE_LOG = 'generate_features_importance_log'
    IMPORTANCE_BAR_CHART = 'generate_features_importance_bar_graph'
    PERMUTATION_IMPORTANCE = 'compute_feature_permutation_importance'
    LEARNING_CURVE = 'generate_sklearn_learning_curves'
    PRECISION_RECALL = 'generate_precision_recall_curves'
    RF_MAX_FEATURES = 'rf_max_features'
    LEARNING_CURVE_K_SPLITS = 'learning_curve_k_splits'
    LEARNING_CURVE_DATA_SPLITS = 'learning_curve_data_splits'
    N_FEATURE_IMPORTANCE_BARS = 'n_feature_importance_bars'
    SHAP_SCORES = 'generate_shap_scores'
    SHAP_PRESENT = 'shap_target_present_no'
    SHAP_ABSENT = 'shap_target_absent_no'
    TRAIN_TEST_SPLIT_TYPE = 'train_test_split_type'

