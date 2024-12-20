import glob

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.read_write import read_config_file, read_df

# DEFINITIONS
CONFIG_PATH = r"C:\troubleshooting\mitra\project_folder\project_config.ini"
CLASSIFIER_PATH = r"C:\troubleshooting\mitra\models\generated_models\grooming.sav"
CLASSIFIER_NAME = 'grooming'
COUNT_PRESENT = 250
COUNT_ABSENT = 250


# READ IN THE CONFIG AND THE CLASSIFIER
config = read_config_file(config_path=CONFIG_PATH)
config_object = ConfigReader(config_path=CONFIG_PATH)
clf = read_df(file_path=CLASSIFIER_PATH, file_type='pickle')


# READ IN THE DATA

#Read in the path to all files inside the project_folder/csv/targets_inserted directory
file_paths = glob.glob(config_object.targets_folder + '/*' + config_object.file_type)

#Reads in the data held in all files in ``file_paths`` defined above
data, _ = TrainModelMixin().read_all_files_in_folder_mp(file_paths=file_paths, file_type=config.get('General settings', 'workflow_file_type').strip())

#We find all behavior annotations that are NOT the targets. I.e., if SHAP values for Attack is going to be calculated, bit we need to find which other annotations exist in the data e.g., Escape and Defensive.
non_target_annotations = TrainModelMixin().read_in_all_model_names_to_remove(config=config, model_cnt=config_object.clf_cnt, clf_name=CLASSIFIER_NAME)

# We remove the body-part coordinate columns and the annotations which are not the target from the data
data = data.drop(non_target_annotations + config_object.bp_headers, axis=1)

# We place the target data in its own variable
target_df = data.pop(CLASSIFIER_NAME)


TrainModelMixin().create_shap_log_mp(ini_file_path=CONFIG_PATH,
                                     rf_clf=clf,
                                     x_df=data,
                                     y_df=target_df,
                                     x_names=data.columns,
                                     clf_name=CLASSIFIER_NAME,
                                     cnt_present=COUNT_PRESENT,
                                     cnt_absent=COUNT_ABSENT,
                                     save_path=config_object.logs_path)

