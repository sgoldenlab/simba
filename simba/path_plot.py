import os
import cv2
import pandas as pd
from collections import deque
import numpy as np
import re
from configparser import ConfigParser
import seaborn as sns
import imutils

def path_plot_config(configini):
    pd.options.mode.chained_assignment = None
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'path_plots')
    try:
        maxDequeLines = config.getint('Path plot settings', 'deque_points')
    except ValueError:
        print('ERROR: "Max lines" not set.')
    fileFormat = config.get('Path plot settings', 'file_format')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    try:
        severity_brackets = config.getint('Path plot settings', 'severity_brackets')
    except ValueError:
        print('ERROR: "Severity scale" not set.')
    filesFound = []
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)

    severityGrades = list(np.arange(0, 1.0, ((10 / severity_brackets) / 10)))
    severityGrades.append(10)
    severityColourRGB = []
    severityColour = []
    clrs = sns.color_palette('Reds', n_colors=severity_brackets)
    for color in clrs:
        for value in color:
            value *= 255
            value = int(value)
            severityColourRGB.append(value)
    severityColourList = [severityColourRGB[i:i + 3] for i in range(0, len(severityColourRGB), 3)]
    for color in severityColourList:
        r = color[0]
        g = color[1]
        b = color[2]
        severityColour.append((b, g, r))

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
    print('Generating path plots for ' + str(len(filesFound)) + ' video(s)...')
    loopy = 0

    for i in filesFound:
        loopy+=1
        trackedBodyPart = config.get('Line plot settings', 'Bodyparts')
        severityBool = config.get('Path plot settings', 'plot_severity')
        listPaths_mouse1 = deque(maxlen=maxDequeLines)
        listPaths_mouse2 = deque(maxlen=maxDequeLines)
        severityCircles = []
        loop = 0
        currentFile = i
        csv_df = pd.read_csv(currentFile, index_col=[0])
        rowCount = csv_df.shape[0]
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        try:
            resWidth = int(videoSettings['Resolution_width'])
            resHeight = int(videoSettings['Resolution_height'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        filter_col = [col for col in csv_df if col.startswith(trackedBodyPart)][0:6]
        filter_col = [x for x in filter_col if "_p" not in x]
        shifted_headings = [x + '_shifted' for x in filter_col]
        filter_col.append('attack')
        filter_col.append('Scaled_movement_M1_M2')
        csv_df = csv_df[filter_col]
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(
            columns={filter_col[0]: shifted_headings[0], filter_col[1]: shifted_headings[1],
                     filter_col[2]: shifted_headings[2], filter_col[3]: shifted_headings[3]})
        csv_df_shifted = csv_df_shifted.drop('attack', axis=1)
        csv_df_shifted = csv_df_shifted.drop('Scaled_movement_M1_M2', axis=1)
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        columnNames = list(csv_df_combined)
        maxImageSizeColumn_x = resWidth
        maxImageSizeColumn_y = resHeight
        VideoNo = os.path.basename(currentFile)
        VideoNo = 'Video' + str(re.sub("[^0-9]", "", VideoNo))
        imagesDirOut = VideoNo + str('_tracked_path_dot')
        savePath = os.path.join(frames_dir_out, imagesDirOut)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        img_size = (maxImageSizeColumn_y, maxImageSizeColumn_x, 3)

        for index, row in csv_df_combined.iterrows():
            img = np.ones(img_size) * 255
            overlay = img.copy()
            m1tuple = (int(row[columnNames[0]]), int(row[columnNames[1]]), int(row[columnNames[6]]), (int(row[columnNames[7]])))
            m2tuple = (int(row[columnNames[2]]), int(row[columnNames[3]]), int(row[columnNames[8]]), (int(row[columnNames[9]])))
            if index == 0:
                m1tuple = (0, 0, 0, 0)
                m2tuple = (0, 0, 0, 0)
            listPaths_mouse1.appendleft(m1tuple)
            listPaths_mouse2.appendleft(m2tuple)
            for i in range(len(listPaths_mouse1)):
                tupleM1 = listPaths_mouse1[i]
                tupleM2 = listPaths_mouse2[i]
                cv2.line(img, (tupleM1[2], tupleM1[3]), (tupleM1[0], tupleM1[1]), (255, 191, 0), 2)
                cv2.line(img, (tupleM2[2], tupleM2[3]), (tupleM2[0], tupleM2[1]), (0, 255, 0), 2)
            attackPrediction = int(row[columnNames[4]])
            severityScore = float(row[columnNames[5]])
            if severityBool == 'yes':
                if attackPrediction == 1:
                    midpoints = (
                        list(zip(np.linspace(m1tuple[0], m2tuple[0], 3), np.linspace(m1tuple[1], m2tuple[1], 3))))
                    locationEventX, locationEventY = midpoints[1]
                    for i in range(severity_brackets):
                        lowerBound = severityGrades[i]
                        upperBound = severityGrades[i + 1]
                        if (severityScore > lowerBound) and (severityScore <= upperBound):
                            severityCircles.append((locationEventX, locationEventY, severityColour[i]))
                for y in range(len(severityCircles)):
                    currEventX, currEventY, colour = severityCircles[y]
                    cv2.circle(overlay, (int(currEventX), int(currEventY)), 20, colour, -1)
            image_new = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)
            m1tuple = (int(row[columnNames[0]]), int(row[columnNames[1]]))
            m2tuple = (int(row[columnNames[2]]), int(row[columnNames[3]]))
            cv2.circle(image_new, (m1tuple[0], m1tuple[1]), 20, (255, 0, 0), -1)
            cv2.circle(image_new, (m2tuple[0], m2tuple[1]), 20, (0, 128, 0), -1)
            saveName = str(loop) + str(fileFormat)
            imageSaveName = os.path.join(savePath, saveName)
            image_new = imutils.resize(image_new, width=400)
            if img_size[0] < img_size[1]:
                image_new = imutils.rotate_bound(image_new, 90.0000)
            cv2.imwrite(imageSaveName, image_new)
            loop += 1
            print('Path plot ' + str(loop) + '/' + str(rowCount) + ' for video ' + str(loopy) + '/' + str(len(filesFound)))
    print('Finished generating path plots. Plots are saved @ project_folder/frames/output/path_plots')