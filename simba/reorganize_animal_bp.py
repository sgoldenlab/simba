import pandas as pd
import glob, os
from datetime import datetime
from collections import OrderedDict

def display_original_bp_list(IN_FOLDER, POSE_TOOL, FILE_FORMAT):
    if POSE_TOOL == 'DLC':
        found_files = glob.glob(IN_FOLDER + '/*.' + FILE_FORMAT)
        # for h5
        if FILE_FORMAT == 'h5':
            try:
                first_df = pd.read_hdf(found_files[0])
            except IndexError:
                print('No H5 files found in directory.')
            header_list = list(first_df.columns)
            animal_bp_tuple_list = []
            for header_val in header_list:
                new_value = (header_val[1], header_val[2])
                if new_value not in animal_bp_tuple_list:
                    animal_bp_tuple_list.append(new_value)
            animal_bp_tuple_list = list(set(animal_bp_tuple_list))

            in_animal_list = [x[0] for x in animal_bp_tuple_list]
            in_bp_list = [x[1] for x in animal_bp_tuple_list]

            return in_animal_list, in_bp_list, header_list

        # for csv
        if FILE_FORMAT == 'csv':
            try:
                first_df = pd.read_csv(found_files[0])
            except IndexError:
                print('No CSV files found in directory.')
            # find if it is madlc or original if it is equal to '0' it is original else is madlc
            #if is madlc multianimal tracking
            if first_df.scorer.loc[2] == 'coords':
                scorer = list(first_df.columns)
                scorer.remove('scorer')
                individuals = list(first_df.loc[0,:])
                individuals.remove('individuals')
                bodyparts = list(first_df.loc[1,:])
                bodyparts.remove('bodyparts')
                coords = list(first_df.loc[2,:])
                coords.remove('coords')

                header_list = list(zip(scorer,individuals,bodyparts,coords))

                animallist = list(zip(individuals,bodyparts))
                animal_bp_tuple_list = list(OrderedDict.fromkeys(animallist))
                in_animal_list = [x[0] for x in animal_bp_tuple_list]
                in_bp_list = [x[1] for x in animal_bp_tuple_list]

                return in_animal_list, in_bp_list, header_list

            # if is normal dlc
            elif int(first_df.scorer.loc[2]) == 0:
                scorer = list(first_df.columns)
                scorer.remove('scorer')
                bodyparts = list(first_df.loc[0,:])
                bodyparts.remove('bodyparts')
                new_body_part_list = []
                for bp in bodyparts:
                    if bp not in new_body_part_list:
                        new_body_part_list.append(bp)
                coords = list(first_df.loc[1,:])
                coords.remove('coords')

                header_list = list(zip(scorer, bodyparts, coords))

                return None, new_body_part_list, header_list







def reorganize_bp_order(IN_FOLDER, POSE_TOOL, FILE_FORMAT, HEADER_LIST, NEW_ANIMAL_LIST, NEW_BP_LIST, DLC_VERSION):
    date_time = datetime.now().strftime('%Y%m%d%H%M%S')
    new_folder_name = 'Reorganized_bp_' + str(date_time)
    new_directory = os.path.join(IN_FOLDER, new_folder_name)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    print('Saving re-organized pose=estimation files in ' + str(os.path.basename(new_directory)) + '...')
    if POSE_TOOL == 'DLC':
        found_files = glob.glob(IN_FOLDER + '/*.' + FILE_FORMAT)
        new_header_tuples = []
        if DLC_VERSION == 'maDLC':
            for animal_name, animal_bp in zip(NEW_ANIMAL_LIST, NEW_BP_LIST):
                new_header_tuples.append((HEADER_LIST[0][0], animal_name, animal_bp, 'x'))
                new_header_tuples.append((HEADER_LIST[0][0], animal_name, animal_bp, 'y'))
                new_header_tuples.append((HEADER_LIST[0][0], animal_name, animal_bp, 'likelihood'))
            new_df_ordered_cols = pd.MultiIndex.from_tuples(new_header_tuples, names=['scorer', 'individuals', 'bodyparts', 'coords'])
        if DLC_VERSION == 'DLC':
            for animal_bp in NEW_BP_LIST:
                new_header_tuples.append((HEADER_LIST[0][0], animal_bp, 'x'))
                new_header_tuples.append((HEADER_LIST[0][0], animal_bp, 'y'))
                new_header_tuples.append((HEADER_LIST[0][0], animal_bp, 'likelihood'))
            new_df_ordered_cols = pd.MultiIndex.from_tuples(new_header_tuples, names=['scorer', 'bodyparts', 'coords'])
        for file in found_files:
            if FILE_FORMAT == 'h5':
                curr_df = pd.read_hdf(file)
            if FILE_FORMAT == 'csv':
                curr_df = pd.read_csv(file, header=[0, 1, 2])
            curr_df_reorganized = pd.DataFrame(curr_df, columns=new_df_ordered_cols)
            df_save_path = os.path.join(new_directory, os.path.basename(file))
            if FILE_FORMAT == 'h5':
                curr_df_reorganized.to_hdf(df_save_path, key='re-organized', format='table', mode='w')
            if FILE_FORMAT == 'csv':
                curr_df_reorganized.to_csv(df_save_path)
            print(str(os.path.basename(df_save_path)) + ' saved...')
        print('All pose-estimation files with re-organized body-part order saved in ' + new_directory)




