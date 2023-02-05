from enum import Enum
from pathlib import Path

class ReadConfig(Enum):
    GENERAL_SETTINGS = 'General settings'
    PROJECT_PATH = 'project_path'
    SML_SETTINGS = 'SML settings'
    VIDEO_INFO_CSV = 'video_info.csv'
    FOLDER_PATH = 'folder_path'
    FILE_TYPE = 'workflow_file_type'
    TARGET_CNT = 'No_targets'
    ANIMAL_CNT = 'animal_no'

class Paths(Enum):
    LINE_PLOT_DIR = Path('frames/output/line_plot/')
    VIDEO_INFO = Path('logs/video_info.csv')
    OUTLIER_CORRECTED = Path('csv/outlier_corrected_movement_location/')
    MACHINE_RESULTS_DIR = Path('csv/machine_results/')
    CLF_VALIDATION_DIR = Path('frames/output/classifier_validation/')
    ICON_ASSETS = Path('assets/icons/')
    DATA_TABLE = Path('frames/output/live_data_table/')
    DIRECTIONALITY_DF_DIR = Path('logs/directionality_dataframes/')
    DIRECTING_ANIMALS_OUTPUT_PATH = Path('frames/output/ROI_directionality_visualize/')
    BP_NAMES = Path('logs/measures/pose_configs/bp_names/project_bp_names.csv')
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

class Formats(Enum):
    MP4_CODEC = 'mp4v'
    LABELFRAME_HEADER_FORMAT = ('Helvetica', 12, 'bold')

class Options(Enum):
    CLF_MODELS = ['RF', 'GBC', 'XGBoost']
    CLF_MAX_FEATURES = ['sqrt', 'log', 'None']
    CLF_CRITERION = ['gini', 'entropy']
    UNDERSAMPLE_OPTIONS = ['None', 'random undersample']
    OVERSAMPLE_OPTIONS = ['None', 'SMOTE', 'SMOTEENN']
    CLASS_WEIGHT_OPTIONS = ['None', 'balanced', 'balanced_subsample', 'custom']
    CLF_TEST_SIZE_OPTIONS = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    PALETTE_OPTIONS = ['magma', 'jet', 'inferno', 'plasma', 'viridis', 'gnuplot2']
    RESOLUTION_OPTIONS = ['640×480', '800×640', '960×800', '1120×960']