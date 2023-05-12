import simba
from simba.pose_importers.dlc_importer_csv import import_multiple_dlc_tracking_csv_file
from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper
from simba.utils.cli.cli_tools import feature_extraction_runner, set_video_parameters
from simba.model.inference_batch import InferenceBatch
from simba.data_processors.agg_clf_calculator import AggregateClfCalculator
from simba.data_processors.movement_calculator import MovementCalculator

# DEFINITIONS
# DEFINE THE PATH TO YOUR SIMBA PROJECT CONFIG INI FILE
CONFIG_PATH = '/Users/simon/Desktop/envs/troubleshooting/notebook_example/project_folder/project_config.ini'

# DEFINE THE PATH TO DIRECTORY CONTAINING YOUR NEW DLC CSV TRACKING FILES
DATA_DIR = '/Users/simon/Desktop/envs/troubleshooting/notebook_example/data'

# DEFINE IF / HOW YOU WANT TO INTERPOLATE MISSING POSE-ESTIMATION DATA,
# AND IF/HOW YOU WANT TO SMOOTH THE NEW POSE ESTIMATION DATA: HERE WE DO NEITHER.
INTERPOLATION_SETTING = 'None' # OPTIONS: 'None', Animal(s): Nearest', 'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear', 'Body-parts: Quadratic'
SMOOTHING_SETTING = None # OPTIONS: 'Gaussian', 'Savitzky Golay'
SMOOTHING_TIME = None # TIME IN MILLISECOND

# DEFINE THE FPS AND THE PIXELS PER MILLIMETER OF THE INCOMING DATA: HAS TO BE THE SAME FOR ALL NEW VIDEOS.
# IF YOU HAVE VARYING FPS / PX PER MILLIMETER / RESOLUTIONS, THEN USE GUI (2023/05)
FPS = 15
PX_PER_MM = 4.6
RESOLUTION = (600, 400) # WIDTH X HEIGHT


# RUN THE DATA IMPORTER FOR A DIRECTORY OF FILES
import_multiple_dlc_tracking_csv_file(config_path=CONFIG_PATH,
                                        interpolation_setting=INTERPOLATION_SETTING,
                                        smoothing_setting=SMOOTHING_SETTING,
                                        smoothing_time=SMOOTHING_TIME,
                                        data_dir=DATA_DIR)


# RUN THE OUTLIER CORRECTION SKIPPER
OutlierCorrectionSkipper(config_path=CONFIG_PATH).run()

# SET THE VIDEO PARAMETERS FOR THE NEW VIDEOS
set_video_parameters(config_path=CONFIG_PATH, px_per_mm=PX_PER_MM, fps=FPS, resolution=RESOLUTION)

# COMPUTE FEATURES
feature_extraction_runner(config_path=CONFIG_PATH)

# RUN CLASSIFIERS
# > NOTE: THE MINIMUM BOUT LENGTH, CLASSIFIER PATHS, AND CLASSIFICATION THRESHOLD ARE SPECIFIED IN YOUR PROJECT CONFIG
# IF YOU WANT TO CHANGE THESE PARAMETERS, THEN CHANGE THEM IN THE PROJECT_CONFIG.INI PRIOR TO RUNNING THE BATCH INFERENCE COMMAND
InferenceBatch(config_path=CONFIG_PATH).run()

# CLASSIFIER DESCRIPTIVE STATISTICS
## SPECIFY WHICH AGGREGATE STATISTICS AND WHICH CLASSIFIERS
DATA_MEASURES = [['Bout count',
                  'Total event duration (s)',
                  'Mean event bout duration (s)',
                  'Median event bout duration (s)',
                  'First event occurrence (s)',
                  'Mean event bout interval duration (s)',
                  'Median event bout interval duration (s)']]
CLASSIFIERS = ['freeze', 'groom', 'escape']

## RUN THE CLASSIFIER AGGREGATE STATISTIC CALCULATOR AND SAVE THE RESULTS TO DISK
agg_clf_results = AggregateClfCalculator(config_path=CONFIG_PATH, data_measures=DATA_MEASURES, classifiers=CLASSIFIERS)
agg_clf_results.run()
agg_clf_results.save()

# MOVEMENT DESCRIPTIVE STATISTICS
## SPECIFY WHICH BODY-PARTS AND WHICH POSE-ESTIMATION CONFIDENCE THRESHOLD
MOVEMENT_BODY_PARTS = ['Animal_1 CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
MOVEMENT_THRESHOLD = 0.00
## RUN THE MOVEMENT CALCULATOR AND SAVE THE RESULTS TO DISK
movement_results = MovementCalculator(config_path=CONFIG_PATH, body_parts=MOVEMENT_BODY_PARTS, threshold=MOVEMENT_THRESHOLD)
movement_results.run()
movement_results.save()