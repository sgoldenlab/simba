import os
import cv2
import pandas as pd
from collections import deque
import numpy as np
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
import seaborn as sns
import imutils
import glob
from pylab import *
from simba.rw_dfs import *
from simba.drop_bp_cords import getBpHeaders
#pd.options.mode.chained_assignment = None

def path_plot_config(configini):
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    noAnimals = config.getint('Path plot settings', 'no_animal_pathplot')
    projectPath = config.get('General settings', 'project_path')
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'path_plots')
    bplist = getBpHeaders(configini)
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    try:
        maxDequeLines = config.getint('Path plot settings', 'deque_points')
    except ValueError:
        print('ERROR: "Max lines" not set - applying default (100)')
        maxDequeLines = 100

    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    severityBool = config.get('Path plot settings', 'plot_severity')
    severityTarget = config.get('Path plot settings','severity_target')
    animalBodypartList = []
    for animal in range(noAnimals):
        currBpName = 'animal' + '_' + str(animal+1) + '_bp'
        animalBpName = config.get('Path plot settings', currBpName)
        animalBpNameX, animalBpNameY = animalBpName + '_x', animalBpName + '_y'
        animalBodypartList.append([animalBpNameX, animalBpNameY])
    columns2grab = [item[0:2] for item in animalBodypartList]
    columns2grab = [item for sublist in columns2grab for item in sublist]
    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal+1) + '_')
            print('Applying settings for classical tracking...')
    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    try:
        severity_brackets = config.getint('Path plot settings', 'severity_brackets')
    except ValueError:
        print('"Severity scale" not set.')
        severity_brackets = 1

    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    severityGrades = list(np.arange(0, 1.0, ((10 / severity_brackets) / 10)))
    severityGrades.append(1)

    severityColourRGB, severityColour  = [], []
    clrs = sns.color_palette('magma', n_colors=severity_brackets)
    for color in clrs:
        for value in color:
            value *= 255
            value = int(value)
            severityColourRGB.append(value)
    severityColourList = [severityColourRGB[i:i + 3] for i in range(0, len(severityColourRGB), 3)]
    for color in severityColourList:
        r, g, b = color[0], color[1], color[2]
        severityColour.append((b, g, r))

    animalColors, cmap = [], cm.get_cmap('Set1', noAnimals)
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        animalColors.append(rgb)

    filesFound = glob.glob(csv_dir_in + "/*." + wfileType)
    print('Generating path plots for ' + str(len(filesFound)) + ' video(s)...')
    fileCounter = 0

    for currentFile in filesFound:
        fileCounter+=1
        dequePaths, severityCircles = [], []
        loop = 0
        for animal in range(noAnimals):
            dequePaths.append(deque(maxlen=maxDequeLines))

        csv_df = read_df(currentFile, wfileType)
        csv_df = csv_df.loc[:, ~csv_df.columns.str.contains('^Unnamed')]
        csv_df.columns = bplist
        csv_df = csv_df[columns2grab]
        CurrentVideoName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.' + wfileType, ''))]
        try:
            resWidth, resHeight = int(videoSettings['Resolution_width']), int(videoSettings['Resolution_height'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        mySpaceScale, myRadius, myResolution, myFontScale = 25, 10, 1500, 0.8
        maxResDimension = max(resWidth, resHeight)
        DrawScale = int(myRadius / (myResolution / maxResDimension))
        textScale = float(myFontScale / (myResolution / maxResDimension))
        shifted_headings = [x + '_shifted' for x in csv_df]
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = shifted_headings
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        columnNames = list(csv_df_combined)
        savePath = os.path.join(frames_dir_out, CurrentVideoName.replace('.' + wfileType, ''))
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        img_size = (resHeight, resWidth, 3)

        if severityBool == 'yes':
            csv_df_combined[severityTarget] = csv_df[severityTarget].values
            csv_df_combined['Scaled_movement_M1_M2'] = csv_df['Scaled_movement_M1_M2'].values
            columnNames = list(csv_df_combined)
        rowCounter = 0
        for index, row in csv_df_combined.iterrows():
            img = np.ones(img_size) * 255
            overlay = img.copy()
            animalTuples = []
            currentPoints = np.empty((noAnimals, 4), dtype=int)
            for animal in range(len(currentPoints)):
                currentPoints[animal][0], currentPoints[animal][1], currentPoints[animal][2], currentPoints[animal][3] = csv_df_combined.at[rowCounter, animalBodypartList[animal][0]], csv_df_combined.at[rowCounter, animalBodypartList[animal][1]], csv_df_combined.at[rowCounter, str(animalBodypartList[animal][0]) + '_shifted'], csv_df_combined.at[rowCounter, str(animalBodypartList[animal][1]) + '_shifted']
                animalTuples.append((currentPoints[animal][0], currentPoints[animal][1], currentPoints[animal][2], currentPoints[animal][3]))
            if index == 0:
                currentPoints = np.zeros((noAnimals, 4), dtype=int)
            for currPoint in range(len(currentPoints)):
                dequePaths[currPoint].append(currentPoints[currPoint])
            for path in range(len(dequePaths)):
                currColor = animalColors[path]
                for point in dequePaths[path]:
                    cv2.line(img, (point[2], point[3]), (point[0], point[1]), currColor, 2)
            rowCounter += 1


            if severityBool == 'yes':
                attackPrediction = int(row[columnNames[8]])
                severityScore = float(row[columnNames[9]])
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
            colorCounter = 0
            for currTuple, color, animalID in zip(animalTuples, animalColors, multiAnimalIDList):
                cv2.circle(image_new, (currTuple[0], currTuple[1]), DrawScale, color, -1)
                cv2.putText(image_new, str(animalID),(currTuple[0], currTuple[1]), cv2.FONT_HERSHEY_TRIPLEX, textScale, color, 2)
            imageSaveName = os.path.join(savePath, str(loop) + '.png')
            #image_new = imutils.resize(image_new, width=400)
            if img_size[0] < img_size[1]:
                image_new = imutils.rotate_bound(image_new, 270.0000)
            cv2.imwrite(imageSaveName, image_new)
            rowCounter+=1
            loop += 1
            print('Path plot ' + str(loop) + '/' + str(len(csv_df_combined)) + ' for video ' + str(fileCounter) + '/' + str(len(filesFound)))
    print('Finished generating path plots. Plots are saved @ project_folder/frames/output/path_plots')