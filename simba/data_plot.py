import os
import pandas as pd
import statistics
import numpy as np
import cv2
from configparser import ConfigParser, MissingSectionHeaderError

def data_plot_config(configini):
    config = ConfigParser()
    configFile = str(configini)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'live_data_table')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    filesFound = []
    loopy = 0

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
    print('Generating data plots for ' + str(len(filesFound)) + ' video(s)...')

    for i in filesFound:
        frameCounter = 0
        list_nose_movement_M1 = []
        list_nose_movement_M2 = []
        loop = 0
        currentFile = i
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        try:
            fps = int(videoSettings['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        loopy += 1
        csv_df = pd.read_csv(currentFile)
        rowCount = csv_df.shape[0]
        VideoName = os.path.basename(currentFile).replace('.csv', '')
        imagesDirOut = VideoName + str('_data_plots')
        savePath = os.path.join(frames_dir_out, imagesDirOut)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        df_lists = [csv_df[i:i + fps] for i in range(0, csv_df.shape[0], fps)]

        for i in df_lists:
            currentDf = i
            mmMove_nose_M1 = currentDf["Movement_mouse_1_centroid"].mean()
            mmMove_nose_M2 = currentDf["Movement_mouse_2_centroid"].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
            list_nose_movement_M2.append(mmMove_nose_M2)
            current_velocity_M1_cm_sec = round(mmMove_nose_M1, 2)
            current_velocity_M2_cm_sec = round(mmMove_nose_M2, 2)
            meanVelocity_M1 = statistics.mean(list_nose_movement_M1)
            meanVelocity_M2 = statistics.mean(list_nose_movement_M2)
            meanVelocity_M1 = round(meanVelocity_M1, 2)
            meanVelocity_M2 = round(meanVelocity_M2, 2)
            total_Movement_M1 = sum(list_nose_movement_M1)
            total_Movement_M2 = sum(list_nose_movement_M2)
            total_Movement_M1 = round(total_Movement_M1, 2)
            total_Movement_M2 = round(total_Movement_M2, 2)

            # save images
            for index, row in currentDf.iterrows():
                img_size = (400, 600, 3)
                img = np.ones(img_size) * 255
                cv2.putText(img, str('Mean velocity animal 1: '), (5, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, str('Mean velocity animal 2: '), (5, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, str('Total movement animal 1: '), (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str('Total movement animal 2: '), (5, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str('Current velocity animal 1: '), (5, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                cv2.putText(img, str('Current velocity animal 2: '), (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                cv2.putText(img, str(meanVelocity_M1) + str(' cm/s'), (275, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, str(meanVelocity_M2) + str(' cm/s'), (275, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, str(total_Movement_M1) + str(' cm'), (275, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str(total_Movement_M2) + str(' cm'), (275, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str(current_velocity_M1_cm_sec) + str(' cm/s'), (275, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                cv2.putText(img, str(current_velocity_M2_cm_sec) + str(' cm/s'), (275, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 100, 0), 1)
                centroid_distance_cm = (int(row["Centroid_distance"])) / 10
                centroid_distance_cm = round(centroid_distance_cm, 2)
                nose_2_nose_dist_cm = (int(row["Nose_to_nose_distance"])) / 10
                nose_2_nose_dist_cm = round(nose_2_nose_dist_cm, 2)
                cv2.putText(img, str('Centroid distance: '), (5, 140), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                cv2.putText(img, str('Nose to nose distance: '), (5, 160), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                cv2.putText(img, str(centroid_distance_cm) + str(' cm'), (275, 140), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                cv2.putText(img, str(nose_2_nose_dist_cm) + str(' cm'), (275, 160), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (153, 50, 204), 1)
                imageName = str(loop) + str('.png')
                imageSaveName = os.path.join(savePath, imageName)
                cv2.imwrite(imageSaveName, img)
                print('Live plot ' + str(loop) + '/' + str(rowCount) + ' for video ' + str(loopy) + '/' + str(len(filesFound)))
                loop += 1
                frameCounter += 1
    print('Finished generating data plots.')