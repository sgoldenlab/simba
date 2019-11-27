import os
from configparser import ConfigParser
import pandas as pd

config = ConfigParser()
configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"
config.read(configFile)
csv_dir = config.get('General settings', 'csv_path')
csv_dir_in = os.path.join(csv_dir, 'features_extracted')
use_master = config.get('General settings', 'use_master_config')
csv_dir_out = os.path.join(csv_dir, 'features_extracted_scaled')

if not os.path.exists(csv_dir_out):
    os.makedirs(csv_dir_out)
filesFound = []
configFilelist = []
loop = 0

########### FIND CSV FILES ###########
if use_master == 'yes':
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            fname = os.path.join(csv_dir_in, i)
            filesFound.append(fname)
if use_master == 'no':
    config_folder_path = config.get('General settings', 'config_folder')
    for i in os.listdir(config_folder_path):
        if i.__contains__(".ini"):
            configFilelist.append(os.path.join(config_folder_path, i))
            iniVidName = i.split(".")[0]
            csv_fn = iniVidName + '.csv'
            file = os.path.join(csv_dir_in, csv_fn)
            filesFound.append(file)

for i in filesFound:
    if use_master == 'no':
        configFile = configFilelist[loop]
        config = ConfigParser()
        config.read(configFile)
        fps = config.getint('Frame settings', 'fps')
        resWidth = config.getint('Frame settings', 'resolution_width')
        resHeight = config.getint('Frame settings', 'resolution_height')
    inputDf = pd.read_csv(i)
    currDf = inputDf.drop(['Ear_left_1_x', 'Ear_left_1_y', 'Ear_left_1_p', 'Ear_right_1_x', 'Ear_right_1_y', 'Ear_right_1_p', 'Nose_1_x', 'Nose_1_y', 'Nose_1_p', 'Center_1_x', \
                           'Center_1_y', 'Center_1_p', 'Lat_left_1_x', 'Lat_left_1_y', 'Lat_left_1_p', 'Lat_right_1_x', 'Lat_right_1_y', 'Lat_right_1_p', 'Tail_base_1_x', \
                           'Tail_base_1_y', 'Tail_base_1_p', 'Tail_end_1_x', 'Tail_end_1_y', 'Tail_end_1_p', 'Ear_left_2_x', 'Ear_left_2_y', 'Ear_left_2_p', 'Ear_right_2_x', \
                           'Ear_right_2_y', 'Ear_right_2_p', 'Nose_2_x', 'Nose_2_y', 'Nose_2_p', 'Center_2_x', 'Center_2_y', 'Center_2_p', 'Lat_left_2_x', 'Lat_left_2_y', 'Lat_left_2_p' \
                           'Lat_right_2_x', 'Lat_right_2_y', 'Lat_right_2_p', 'Tail_base_2_x', 'Tail_base_2_y', 'Tail_base_2_p', 'Tail_end_2_x', 'Tail_end_2_y', 'Tail_end_2_p', 'video_no', \
                           'frames', 'Total_angle_both_mice', 'Total_angle_1.75', 'Total_angle_2', 'Total_angle_4', 'Total_angle_8' 'Total_angle_16', 'Total_angle_32', 'Total_angle_both_mice_1.75',
                           'Total_angle_both_mice_2', 'Total_angle_both_mice_4', 'Total_angle_both_mice_8', 'Total_angle_both_mice_16', 'Total_angle_both_mice_32' 
'


                           ], axis = 1)


































