# Shapley calculations: Example II (GPU)

# >NOTE I: The SHAP library has to be built from got rather than pip: ``pip install git+https://github.com/slundberg/shap.git``
# >NOTE II: The scikit model can not be built using max_depth > 31 for it to work with this code.

# In this example, we have previously created a classifier. We have the data used to create this classifier, and now we want to compute SHAP explainability scores
# for this classifier using GPU (to speed things up a MASSIVELY).

from simba.sandbox.create_shap_log import create_shap_log
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import read_df, read_config_file
import glob


# DEFINITIONS
CONFIG_PATH = r"/mnt/c/troubleshooting/mitra/project_folder/project_config.ini"
CLASSIFIER_PATH = r"/mnt/c/troubleshooting/mitra/models/generated_models/grooming.sav"
CLASSIFIER_NAME = 'grooming'
SAVE_DIR = r'/mnt/c/troubleshooting/mitra/models/generated_models'
COUNT_PRESENT = 2000
COUNT_ABSENT = 2000


# READ IN THE CONFIG AND THE CLASSIFIER
config = read_config_file(config_path=CONFIG_PATH)
config_object = ConfigReader(config_path=CONFIG_PATH, create_logger=False)
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

shap_values, raw_values, expected_value = create_shap_log(rf_clf=clf,
                                                          x=data,
                                                          y=target_df,
                                                          cnt_present=COUNT_PRESENT,
                                                          cnt_absent=COUNT_ABSENT,
                                                          x_names=list(data.columns),
                                                          clf_name='grooming',
                                                          save_dir=None,
                                                          verbose=True)