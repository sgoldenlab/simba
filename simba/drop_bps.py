import pandas as pd
import os, glob
from datetime import datetime



def drop_bps_tracking_input_bps(DATA_FOLDER, POSE_TOOL, FILE_FORMAT, NO_BP_TO_DROP):
    files_found = glob.glob(DATA_FOLDER + '/*.' + FILE_FORMAT)
    if FILE_FORMAT == 'h5':
        first_df = pd.read_hdf(files_found[0])
    if FILE_FORMAT == 'csv':
        first_df = pd.read_csv(files_found[0], header=[0,1,2])
    header_list = list(first_df.columns)[1:]
    body_part_names, animal_names = [], []

    if POSE_TOOL == 'DLC':
        for header_entry in header_list:
            if header_entry[1] not in body_part_names:
                body_part_names.append(header_entry[1])

    if POSE_TOOL == 'maDLC':
        for header_entry in header_list:
            if header_entry[1] not in body_part_names:
                animal_names.append(header_entry[1])
                body_part_names.append(header_entry[2])

    return animal_names, body_part_names

def run_bp_removal(pose_tool, animal_names, bp_names, folder_path, file_format):
    date_time = datetime.now().strftime('%Y%m%d%H%M%S')
    new_folder_name = 'Reorganized_bp_' + str(date_time)
    new_directory = os.path.join(folder_path, new_folder_name)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    print('Saving new pose-estimation files in ' + str(os.path.basename(new_directory)) + '...')
    found_files = glob.glob(folder_path + '/*.' + file_format)
    if (pose_tool == 'DLC') or (pose_tool == 'maDLC'):
        for file in found_files:
            if file_format == 'csv':
                df = pd.read_csv(file, header=[0,1,2], index_col=0)
            if file_format == 'h5':
                df = pd.read_hdf(file)
            for body_part in bp_names:
                if pose_tool == 'DLC':
                    df = df.drop(body_part, axis=1, level=1)
                if pose_tool == 'maDLC':
                    df = df.drop(body_part, axis=1, level=2)
            save_path = os.path.join(new_directory, os.path.basename(file))
            if file_format == 'csv':
                df.to_csv(save_path)
            if file_format == 'h5':
                df.to_hdf(save_path, key='re-organized', format='table', mode='w')
            print('Saved ' + str(os.path.basename(file)) + '...')
    print('All data with dropped body-parts saved in ' + new_directory)