import os

from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)

SIMBA_PROJECT_CONFIG_PATH = r"C:\troubleshooting\mitra\project_folder\project_config.ini" #PATH TO THE SIMBA PROJECT CONFIG (USED TO FIND THE BODY PART NAMES AND CLASSIFIER NAMES)
DATA_DIRECTORY = r'C:\troubleshooting\mitra\project_folder\csv\targets_inserted' #PATH TO A DIRECTORY CONTAINING SIMBA CSV FILES
SAVE_DIRECTORY = r'C:\troubleshooting\mitra\project_folder\csv\targets_inserted\temp\new_targets_inserted' #PATH TO AN EMTY DIRECTORY USE DTO SAVED THE NEW CSV FILES.

FIELD_TO_KEEP = ["MOVEMENT_SUM_2.0_NOSE",
                 "GEOMETRY_MEAN_BODY_AREA_0.5",
                 "GEOMETRY_MEAN_BODY_AREA_2.0",
                 "GEOMETRY_SUM_HULL_WIDTH_2.0",
                 "GEOMETRY_VAR_HULL_LENGTH_2.0",
                 "GEOMETRY_SUM_HULL_LENGTH_0.25",
                 "GEOMETRY_MEAN_HULL_LENGTH_0.25"] #LIST OF FEATURE NAMES TO KEEP (ALL FEATURES NOT IN THIS LIST WILL BE REMOVED).

data_paths = find_files_of_filetypes_in_directory(directory=DATA_DIRECTORY, extensions=['.csv'], raise_error=True)
config = ConfigReader(config_path=SIMBA_PROJECT_CONFIG_PATH, read_video_info=False, create_logger=False)
fields_to_keep = config.bp_col_names + FIELD_TO_KEEP + config.clf_names

for file_cnt, file_path in enumerate(data_paths):
    df = read_df(file_path=file_path, file_type='csv', usecols=fields_to_keep)
    video_name = get_fn_ext(filepath=file_path)[1]
    print(f'Processing {video_name}...')
    save_path = os.path.join(SAVE_DIRECTORY, f'{video_name}.csv')
    write_df(df=df, file_type='csv', save_path=save_path)
print(f'COMPLETE: New files stored in {SAVE_DIRECTORY}.')
