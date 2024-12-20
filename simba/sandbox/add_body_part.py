import os.path
import numpy as np
import pandas as pd

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import read_df, write_df, get_fn_ext, read_frm_of_video
from simba.utils.checks import check_if_dir_exists

#CHANGE THIS TO THE PATH OF YOUR SIMBA PROJECT CONFIG
CONFIG_PATH = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini'

#CHANGE THIS PATH TO A NEW DIRECTORY ON YOUR COMPUTER
SAVE_DIRECTORY = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/new_data'

BP_1 = 'Ear_left_1' # THE NAME OF YOUR LEFT EAR BODY-PART
BP_2 = 'Ear_right_1' # THE NAME OF YOUR RIGHT EAR BODY-PART
NEW_BP_NAME = 'Head_1' # THE NAME OF YOUR NEW BODY-PART

########################

config = ConfigReader(config_path=CONFIG_PATH) # READS IN YOUR PROJECT CONFIG
check_if_dir_exists(in_dir=SAVE_DIRECTORY) # CHECKS THAT YOUR SPECIFIED SAVE_DIRECTORY ACTUALLY EXIST

for file_path in config.outlier_corrected_paths: # LOOPS OVER EACH CSV FILE IN THE "project_folder/csv/outlier_corrected_movement_location" directory
    df = read_df(file_path=file_path, file_type=config.file_type) #READS THE FILE
    file_name = get_fn_ext(filepath=file_path)[1] #GET THE FILENAME OF THE FILE BEING READ
    bp_1, bp_2 = df[[f'{BP_1}_x', f'{BP_1}_y']].values.astype(int), df[[f'{BP_2}_x', f'{BP_2}_y']].values.astype(int) #GET THE COLUMNS OF THE BODY-PARTS SPECIFIED
    results = FeatureExtractionMixin.find_midpoints(bp_1=bp_1, bp_2=bp_2, percentile=0.5) #FIND THE MIDPOINT IN BETWEEN THE TWO BODY-PARTS
    results = np.hstack((results, np.ones(results.shape[0]).reshape(-1, 1))) # THE NEW BODY-PART WILL NOT HAVE A PROBABILITY VALUE, SO WE SET THEM ALL TO 1
    df = pd.concat([df, pd.DataFrame(results, columns=[f'{NEW_BP_NAME}_x', f'{NEW_BP_NAME}_y', f'{NEW_BP_NAME}_p'])], axis=1) # ADD THE NEW BODY-PART TO THE DATA
    save_path = os.path.join(SAVE_DIRECTORY, f'{file_name}.{config.file_type}') #CREATE A PATH FOR THE DATA WITH THE NEW BODY-PART FILE
    write_df(df=df, file_type=config.file_type, save_path=save_path) #WRITE THE DATA TO DISK
    print(f'File saved {save_path}...')
print(f'SimBA complete: {len(config.outlier_corrected_paths)} files saved in {SAVE_DIRECTORY}')