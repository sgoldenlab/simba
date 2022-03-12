import os
import pandas as pd
import statistics
import numpy as np
import cv2
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError, NoSectionError
import glob
from simba.drop_bp_cords import *

def data_plot_config(configini, SelectedBp):
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    noAnimals = config.getint('General settings', 'animal_no')
    projectPath = config.get('General settings', 'project_path')
    poseConfigSetting = config.get('create ensemble settings', 'pose_estimation_body_parts')
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'live_data_table')
    bplist = define_bp_drop_down(configini)
    print(bplist)
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    csv_dir_in = os.path.join(projectPath, 'csv', 'features_extracted')
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    videoCounter = 0

    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            multiAnimalIDList = []
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
            print('Applying settings for classical tracking...')
    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    Xcols, Ycols, Pcols = getBpNames(configini)
    bpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, Pcols, [])

    ##### FIND RELEVANT COLUMN
    if poseConfigSetting != 'user_defined':
        if noAnimals == 1:
            move1ColName = "Movement_mouse_centroid"
        if noAnimals == 2:
            move1ColName = "Movement_mouse_1_centroid"
    if poseConfigSetting == 'user_defined':
        if noAnimals == 1:
            move1ColName = "movement_" + SelectedBp
        if noAnimals == 2:
            move1ColName = "movement_" + SelectedBp + '_1'


    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + "/*." + wfileType)

    print('Generating data plots for ' + str(len(filesFound)) + ' video(s)...')

    for currentFile in filesFound:
        frameCounter = 0
        list_nose_movement_M1, list_nose_movement_M2 = [], []
        loop = 0
        CurrentVideoName = os.path.basename(currentFile.replace('.csv', ''))
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName)]
        try:
            fps = int(videoSettings['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        videoCounter += 1
        csv_df = pd.read_csv(currentFile)
        savePath = os.path.join(frames_dir_out, CurrentVideoName)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        df_lists = [csv_df[i:i + fps] for i in range(0, csv_df.shape[0], fps)]

        for currentDf in df_lists:
            try:
                mmMove_nose_M1 = currentDf[move1ColName].mean()
            except (KeyError, UnboundLocalError):
                move1ColName = bplist[0][0]
                move1ColName = 'Movement_' + str(move1ColName)
                mmMove_nose_M1 = currentDf[move1ColName].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
            current_velocity_M1_cm_sec = round(mmMove_nose_M1, 2)
            meanVelocity_M1 = statistics.mean(list_nose_movement_M1)
            meanVelocity_M1 = round(meanVelocity_M1, 2)
            total_Movement_M1 = sum(list_nose_movement_M1)
            total_Movement_M1 = round(total_Movement_M1, 2)
            if noAnimals == 2:
                if poseConfigSetting != 'user_defined':
                    mmMove_nose_M2 = currentDf["Movement_mouse_2_centroid"].mean()
                if poseConfigSetting == 'user_defined':
                    mmMove_nose_M2 = currentDf["movement_" + SelectedBp].mean()
                list_nose_movement_M2.append(mmMove_nose_M2)
                current_velocity_M2_cm_sec = round(mmMove_nose_M2, 2)
                meanVelocity_M2 = statistics.mean(list_nose_movement_M2)
                meanVelocity_M2 = round(meanVelocity_M2, 2)
                total_Movement_M2 = sum(list_nose_movement_M2)
                total_Movement_M2 = round(total_Movement_M2, 2)

            # save images
            for index, row in currentDf.iterrows():
                img_size = (400, 600, 3)
                img = np.ones(img_size) * 255
                cv2.putText(img, str('Mean velocity animal 1: '), (5, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, str('Total movement animal 1: '), (5, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str('Current velocity animal 1: '), (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                cv2.putText(img, str(meanVelocity_M1) + str(' cm/s'), (275, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, str(total_Movement_M1) + str(' cm'), (275, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str(current_velocity_M1_cm_sec) + str(' cm/s'), (275, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                if noAnimals == 2:
                    cv2.putText(img, str('Mean velocity animal 2: '), (5, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(img, str('Total movement animal 2: '), (5, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(img, str('Current velocity animal 2: '), (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                    cv2.putText(img, str(meanVelocity_M2) + str(' cm/s'), (275, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(img, str(total_Movement_M2) + str(' cm'), (275, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(img, str(current_velocity_M2_cm_sec) + str(' cm/s'), (275, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)

                    if poseConfigSetting != 'user_defined':
                        centroid_distance_cm = (int(row["Centroid_distance"])) / 10
                        centroid_distance_cm = round(centroid_distance_cm, 2)
                        nose_2_nose_dist_cm = (int(row["Nose_to_nose_distance"])) / 10
                        nose_2_nose_dist_cm = round(nose_2_nose_dist_cm, 2)
                        cv2.putText(img, str('Centroid distance: '), (5, 140), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                        cv2.putText(img, str('Nose to nose distance: '), (5, 160), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                        cv2.putText(img, str(centroid_distance_cm) + str(' cm'), (275, 140), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                        cv2.putText(img, str(nose_2_nose_dist_cm) + str(' cm'), (275, 160), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                imageSaveName = os.path.join(savePath, str(loop) + '.png')
                cv2.imwrite(imageSaveName, img)
                print('Live plot ' + str(loop) + '/' + str(len(csv_df)) + ' for video ' + str(videoCounter) + '/' + str(len(filesFound)))
                loop += 1
                frameCounter += 1
    print('Finished generating data plots.')