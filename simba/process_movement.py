import pandas as pd
import os
from configparser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime
import statistics
import numpy as np
import glob
from simba.drop_bp_cords import define_movement_cols, get_fn_ext
from simba.drop_bp_cords import get_workflow_file_format
from simba.rw_dfs import *
from simba.features_scripts.unit_tests import read_video_info
import itertools
from collections import defaultdict
from simba.misc_tools import check_multi_animal_status
from collections.abc import Iterable

def analyze_process_movement(configini):



    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el

    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidLogFilePath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    wfileType = get_workflow_file_format(config)

    noAnimals = config.getint('process movements', 'no_of_animals')
    animal_bp_dict = {}
    for i in range(noAnimals):
        animal_bp_dict[i] = {}
        bp_name = config.get('process movements', 'animal_{}_bp'.format(str(i+1)))
        animal_bp_dict[i]['X'] = bp_name + '_x'
        animal_bp_dict[i]['Y'] = bp_name + '_y'
        animal_bp_dict[i]['x_shifted'] = bp_name + '_x_shifted'
        animal_bp_dict[i]['y_shifted'] = bp_name + '_y_shifted'


    log_fn = os.path.join(projectPath, 'logs', 'Movement_log_' + dateTime + '.csv')
    multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)

    ########### logfile path ###########

    columnNames = define_movement_cols(multiAnimalIDList)
    log_df = pd.DataFrame(columns=columnNames)

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')

    animal_combs = list(itertools.combinations(animal_bp_dict, 2))

    for file_counter, currentFile in enumerate(filesFound):
        dir_name, currVideoName, ext = get_fn_ext(currentFile)
        currVideoSettings, currPixPerMM, fps = read_video_info(vidinfDf, currVideoName)
        fps = int(fps)
        csv_df = read_df(currentFile, wfileType)
        csv_df_shifted = csv_df.shift(-1, axis=0)
        csv_df_shifted = csv_df_shifted.add_suffix('_shifted')
        csv_df = pd.concat([csv_df, csv_df_shifted], axis=1)

        for a in animal_bp_dict.keys():
            csv_df['Movement_animal_{}'.format(str(a+1))] = (np.sqrt((csv_df[animal_bp_dict[a]['X']] - csv_df[animal_bp_dict[a]['x_shifted']]) ** 2 + (csv_df[animal_bp_dict[a]['Y']] - csv_df[animal_bp_dict[a]['y_shifted']]) ** 2)) / currPixPerMM

        if noAnimals > 1:
            for c in animal_combs:
                csv_df['Animal_distance_{}_{}'.format(str(c[0]), str(c[1]))] = (np.sqrt((csv_df[animal_bp_dict[c[0]]['X']] - csv_df[animal_bp_dict[c[1]]['x_shifted']]) ** 2 + (csv_df[animal_bp_dict[c[0]]['Y']] - csv_df[animal_bp_dict[c[1]]['y_shifted']]) ** 2)) / currPixPerMM

        df_lists = [csv_df[i:i + fps] for i in range(0, csv_df.shape[0], fps)]
        agg_dict = {}
        distance_dict = defaultdict(list)
        frameCounter = 0
        for animal in range(noAnimals):
            agg_dict[animal] = defaultdict(list)
        for cnt, currentDf in enumerate(df_lists):
            for animal in range(noAnimals):
                agg_dict[animal]['Movement'].append(currentDf['Movement_animal_{}'.format(str(animal + 1))].mean())
                agg_dict[animal]['Velocity'].append(currentDf['Movement_animal_{}'.format(str(animal + 1))].mean() / 1)

            if noAnimals > 1:
                for c in animal_combs:
                    col_name = 'Animal_distance_{}_{}'.format(str(c[0]), str(c[1]))
                    distance_dict[col_name].append(currentDf[col_name].mean() / 1)

            frameCounter += fps

        move_list = []
        vel_list_mean = []
        vel_list_median = []
        for animal in range(noAnimals):
            move_list.append(sum(agg_dict[animal]['Movement']))
            vel_list_mean.append(statistics.mean(agg_dict[animal]['Velocity']))
            vel_list_median.append(statistics.median(agg_dict[animal]['Velocity']))
        if noAnimals > 1:
            mean_dist_lst, median_dist_lst = [], []
            for c in animal_combs:
                col_name = 'Animal_distance_{}_{}'.format(str(c[0]), str(c[1]))
                mean_dist_lst.append(statistics.mean(distance_dict[col_name]) / 10)
                median_dist_lst.append(statistics.median(distance_dict[col_name]) / 10)

        currentVidList = [currVideoName, frameCounter, move_list, vel_list_mean, vel_list_median]
        if noAnimals > 1:
            currentVidList = currentVidList + mean_dist_lst + median_dist_lst

        currentVidList = list(flatten(currentVidList))
        log_df.loc[file_counter] = currentVidList
        print('Files # processed for movement data: ' + str(file_counter + 1) + '/' + str(len(filesFound)) + '...')
    log_df = np.round(log_df,decimals=4)
    log_df.to_csv(log_fn, index=False)
    print('All files processed for movement data. ' + 'Data saved @ project_folder\logs')



