import os
import cv2
import pandas as pd
from collections import deque
import numpy as np
from configparser import ConfigParser, NoSectionError, NoOptionError
import seaborn as sns
import imutils
import glob
from simba.rw_dfs import *
from simba.misc_tools import check_multi_animal_status
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary, createColorListofList


def path_plot_config(configini):
    pd.options.mode.chained_assignment = None
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    noAnimals = config.getint('Path plot settings', 'no_animal_pathplot')
    projectPath = config.get('General settings', 'project_path')
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'path_plots')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)

    try:
        maxDequeLines = config.getint('Path plot settings', 'deque_points')
    except ValueError:
        print('ERROR: "Max lines" not set.')

    csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
    severityBool = config.get('Path plot settings', 'plot_severity')
    severityTarget = config.get('Path plot settings','severity_target')

    Xcols, Ycols, _ = getBpNames(configini)
    colorListofList = createColorListofList(noAnimals, cMapSize=int(len(Xcols) + 1))
    multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, [], colorListofList)

    try:
        trackedBodyPart1 = config.get('Path plot settings', 'animal_1_bp')
        trackedBodyPart2 = config.get('Path plot settings', 'animal_2_bp')
    except:
        trackedBodyPart1 = animalBpDict[multiAnimalIDList[0]]['X_bps'][0][:-2]
        trackedBodyPart2 = animalBpDict[multiAnimalIDList[1]]['X_bps'][0][:-2]


    try:
        severity_brackets = config.getint('Path plot settings', 'severity_brackets')
    except ValueError:
        print('"Severity scale" not set.')
        severity_brackets = 1

    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    severityGrades = list(np.arange(0, 1.0, ((10 / severity_brackets) / 10)))
    severityGrades.append(10)

    severityColourRGB, severityColour  = [], []
    clrs = sns.color_palette('Reds', n_colors=severity_brackets)

    for color in clrs:
        for value in color:
            value *= 255
            value = int(value)
            severityColourRGB.append(value)
    severityColourList = [severityColourRGB[i:i + 3] for i in range(0, len(severityColourRGB), 3)]

    for color in severityColourList:
        r, g, b = color[0], color[1], color[2]
        severityColour.append((b, g, r))

    filesFound = glob.glob(csv_dir_in + "/*." + wfileType)
    print('Generating path plots for ' + str(len(filesFound)) + ' video(s)...')
    fileCounter = 0

    for currentFile in filesFound:
        fileCounter+=1
        loop = 0
        listPaths_mouse1, listPaths_mouse2 = deque(maxlen=maxDequeLines), deque(maxlen=maxDequeLines)
        severityCircles = []
        csv_df = read_df(currentFile, wfileType)
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]

        try:
            resWidth = int(videoSettings['Resolution_width'])
            resHeight = int(videoSettings['Resolution_height'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')

        if noAnimals == 1:
            trackedBodyPartHeadings = [trackedBodyPart1 + '_x', trackedBodyPart1 + '_y']

        elif noAnimals == 2:
            trackedBodyPartHeadings = [trackedBodyPart1 + '_x', trackedBodyPart1 + '_y', trackedBodyPart2 + '_x', trackedBodyPart2 + '_y']

        filter_col = csv_df[trackedBodyPartHeadings]
        shifted_headings = [x + '_shifted' for x in filter_col]
        csv_df_shifted = filter_col.shift(periods=1)
        csv_df_shifted.columns = shifted_headings
        csv_df_combined = pd.concat([filter_col, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        columnNames = list(csv_df_combined)
        maxImageSizeColumn_x, maxImageSizeColumn_y = resWidth, resHeight
        savePath = os.path.join(frames_dir_out, CurrentVideoName.replace('.csv', ''))

        if not os.path.exists(savePath):
            os.makedirs(savePath)

        img_size = (maxImageSizeColumn_y, maxImageSizeColumn_x, 3)

        if severityBool == 'yes':
            csv_df_combined[severityTarget] = csv_df[severityTarget].values
            if noAnimals == 2:
                csv_df_combined['Scaled_movement_M1_M2'] = csv_df['Scaled_movement_M1_M2'].values
            else:
                csv_df_combined['Scaled_movement_M1'] = csv_df['Scaled_movement_M1'].values
            columnNames = list(csv_df_combined)

        for index, row in csv_df_combined.iterrows():
            img = np.ones(img_size) * 255
            overlay = img.copy()
            if noAnimals == 1:
                m1tuple = (int(row[columnNames[0]]), int(row[columnNames[1]]), int(row[columnNames[2]]), (int(row[columnNames[3]])))
            if noAnimals == 2:
                m1tuple = (int(row[columnNames[0]]), int(row[columnNames[1]]), int(row[columnNames[4]]), (int(row[columnNames[5]])))
                m2tuple = (int(row[columnNames[2]]), int(row[columnNames[3]]), int(row[columnNames[6]]), (int(row[columnNames[7]])))
            if index == 0:
                m1tuple, m2tuple = (0, 0, 0, 0), (0, 0, 0, 0)
            listPaths_mouse1.appendleft(m1tuple)
            if noAnimals == 2:
                listPaths_mouse2.appendleft(m2tuple)
            for i in range(len(listPaths_mouse1)):
                tupleM1 = listPaths_mouse1[i]
                cv2.line(img, (tupleM1[2], tupleM1[3]), (tupleM1[0], tupleM1[1]), (255, 191, 0), 2)
                if noAnimals == 2:
                    tupleM2 = listPaths_mouse2[i]
                    cv2.line(img, (tupleM2[2], tupleM2[3]), (tupleM2[0], tupleM2[1]), (0, 255, 0), 2)

            if severityBool == 'yes':
                attackPrediction = int(row[columnNames[8]])
                severityScore = float(row[columnNames[9]])
                if attackPrediction == 1:
                    if noAnimals == 2:
                        midpoints = (list(zip(np.linspace(m1tuple[0], m2tuple[0], 3), np.linspace(m1tuple[1], m2tuple[1], 3))))
                        locationEventX, locationEventY = midpoints[1]
                    if noAnimals == 1:
                        locationEventX, locationEventY = m1tuple
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
            cv2.circle(image_new, (m1tuple[0], m1tuple[1]), 20, (255, 0, 0), -1)
            if noAnimals == 2:
                m2tuple = (int(row[columnNames[2]]), int(row[columnNames[3]]))
                cv2.circle(image_new, (m2tuple[0], m2tuple[1]), 20, (0, 128, 0), -1)
            imageSaveName = os.path.join(savePath, str(loop) + '.png')
            image_new = imutils.resize(image_new, width=400)
            if img_size[0] < img_size[1]:
                image_new = imutils.rotate_bound(image_new, 270.0000)
            cv2.imwrite(imageSaveName, image_new)
            loop += 1
            print('Path plot ' + str(loop) + '/' + str(len(csv_df_combined)) + ' for video ' + str(fileCounter) + '/' + str(len(filesFound)))
    print('Finished generating path plots. Plots are saved @ project_folder/frames/output/path_plots')



#path_plot_config(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini")