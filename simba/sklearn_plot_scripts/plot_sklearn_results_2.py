import numpy as np
import cv2
import os
import pandas as pd
from scipy import ndimage
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError
import glob
from simba.drop_bp_cords import getBpNames
from pylab import *
import random

def plotsklearnresult(configini,videoSetting, frameSetting):
    config = ConfigParser()
    configFile = str(configini)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, "machine_results")
    animalsNo = config.getint('General settings', 'animal_no')
    projectPath = config.get('General settings', 'project_path')
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'sklearn_results')
    poseConfSetting = config.get('create ensemble settings', 'pose_estimation_body_parts')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    counters_no = config.getint('SML settings', 'No_targets')
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    mulltiAnimalIDList= config.get('Multi animal IDs', 'id_list')
    mulltiAnimalIDList = mulltiAnimalIDList.split(",")
    if mulltiAnimalIDList[0] != '':
        mulltiAnimalStatus = True
    if mulltiAnimalIDList[0] == '':
        mulltiAnimalStatus = False
    print(mulltiAnimalStatus)
    vidinfDf = pd.read_csv(vidInfPath)
    target_names, colorList_animal_1, colorList_animal_2, loopy = [], [], [], 0
    Xcols, Ycols, Pcols = getBpNames(configini)
    cmap = cm.get_cmap('hot', len(Xcols) + 1)
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        colorList_animal_1.append(rgb)
    if animalsNo >= 2:
        cmap = cm.get_cmap('winter', len(Xcols) + 1)
        for i in range(cmap.N):
            rgb = list((cmap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            colorList_animal_2.append(rgb)
    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Processing ' + str(len(filesFound)) + ' videos ...')

    ########### GET MODEL NAMES ###########
    for i in range(counters_no):
        currentModelNames = 'target_name_' + str(i + 1)
        currentModelNames = config.get('SML settings', currentModelNames)
        target_names.append(currentModelNames)

    cmap = cm.get_cmap('Set1', counters_no + 3)
    colors = []
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        colors.append(rgb)

    ########### FIND PREDICTION COLUMNS ###########
    for i in filesFound:
        target_counters, target_timers = ([0] * counters_no, [0] * counters_no)
        currentVideo = i
        loopy += 1
        CurrentVideoName = os.path.basename(currentVideo)
        if frameSetting == 1:
            videoFrameDir = os.path.join(frames_dir_out, CurrentVideoName.replace('.csv', ''))
            if not os.path.exists(videoFrameDir):
                os.makedirs(videoFrameDir)
        CurrentVideoRow = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName.replace('.csv', ''))]
        try:
            fps = int(CurrentVideoRow['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        currentDf = pd.read_csv(currentVideo)
        currentDf = currentDf.fillna(0)
        currentDf = currentDf.astype(int)
        currentDf = currentDf.loc[:, ~currentDf.columns.str.contains('^Unnamed')]
        currentDf = currentDf.reset_index()
        animalBpHeaderList, animalBpHeaderListY, animalBpHeaderListX = ([], [], [])
        animal1_BPsX, animal1_BPsY = (currentDf[Xcols], currentDf[Ycols])
        for i in range(len(animal1_BPsX.columns)):
            animalBpHeaderListX.append(animal1_BPsX.columns[i])
            animalBpHeaderListY.append(animal1_BPsY.columns[i])
            animalBpHeaderList.append(animal1_BPsX.columns[i])
            animalBpHeaderList.append(animal1_BPsY.columns[i])
        animalBpHeaderListX, animalBpHeaderListY, animalBpHeaderList = ([x for x in animalBpHeaderListX if "Tail_end" not in x], [x for x in animalBpHeaderListY if "Tail_end" not in x], [x for x in animalBpHeaderList if "Tail_end" not in x])

        if (animalsNo == 2) and (poseConfSetting == 'user_defined'):
            animal_1_BpHeaderList = [s for s in animalBpHeaderList if mulltiAnimalIDList[0] in s]
            animal_2_BpHeaderList = [s for s in animalBpHeaderList if mulltiAnimalIDList[1] in s]
        elif (animalsNo == 2) and (poseConfSetting != 'user_defined'):
            animal_1_BpHeaderList = [s for s in animalBpHeaderList if '_1_' in s]
            animal_2_BpHeaderList = [s for s in animalBpHeaderList if '_2_' in s]
        else:
            animal_1_BpHeaderList = animalBpHeaderList.copy()
        if os.path.exists(os.path.join(projectPath,'videos', CurrentVideoName.replace('.csv', '.mp4'))):
            videoPathName = os.path.join(projectPath,'videos', CurrentVideoName.replace('.csv', '.mp4'))
        elif os.path.exists(os.path.join(projectPath,'videos', CurrentVideoName.replace('.csv', '.avi'))):
            videoPathName = os.path.join(projectPath,'videos', CurrentVideoName.replace('.csv', '.avi'))
        else:
            print('Cannot locate video ' + str(CurrentVideoName.replace('.csv', '')) + ' in mp4 or avi format')
            break
        cap = cv2.VideoCapture(videoPathName)
        width, height, frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outputFileName = os.path.join(frames_dir_out, CurrentVideoName)
        if height < width:
            videoHeight, videoWidth = width, height
        if height >= width:
            videoHeight, videoWidth = height, width
        writer = cv2.VideoWriter(outputFileName.replace('.csv', '.mp4'), fourcc, fps, (videoWidth, videoHeight))
        mySpaceScale, myRadius, myResolution, myFontScale = 60, 12, 1500, 1.5
        maxResDimension = max(width, height)
        circleScale, fontScale, spacingScale = int(myRadius / (myResolution / maxResDimension)), float(myFontScale / (myResolution / maxResDimension)), int(mySpaceScale / (myResolution / maxResDimension))
        currRow = 0
        a = np.deg2rad(90)
        while (cap.isOpened()):
            ret, frame = cap.read()
            IDlabelLoc, rotationFlag = [], False
            if ret == True:
                if (animalsNo == 1) and (poseConfSetting != 'user_defined'):
                    currAnimal1 = currentDf.loc[currentDf.index[currRow], animalBpHeaderList]
                    currAnimal1 = np.array(currAnimal1).astype(int)
                    currAnimal1 = np.reshape(currAnimal1, (-1, 2))
                    M1polyglon_array_hull = cv2.convexHull((currAnimal1.astype(int)))
                    cv2.drawContours(frame, [M1polyglon_array_hull.astype(int)], 0, (255, 255, 255), 2)
                if (animalsNo == 2) and (poseConfSetting != 'user_defined'):
                    currAnimal1, currAnimal2 = (currentDf.loc[currentDf.index[currRow], animal_1_BpHeaderList],currentDf.loc[currentDf.index[currRow], animal_2_BpHeaderList])
                    currAnimal1, currAnimal2  = (np.array(currAnimal1).astype(int), np.array(currAnimal2).astype(int))
                    currAnimal1, currAnimal2 = (np.reshape(currAnimal1, (-1, 2)), np.reshape(currAnimal2, (-1, 2)))
                    M1polyglon_array_hull, M2polyglon_array_hull = (cv2.convexHull((currAnimal1.astype(int))), cv2.convexHull((currAnimal2.astype(int))))
                    cv2.drawContours(frame, [M1polyglon_array_hull.astype(int)], 0, (255, 255, 255), 2)
                    cv2.drawContours(frame, [M2polyglon_array_hull.astype(int)], 0, (255, 255, 255), 2)
                for cords in range(len(animalBpHeaderListX)):
                    currXval = animal1_BPsX.loc[animal1_BPsX.index[currRow], animalBpHeaderListX[cords]]
                    currYval = animal1_BPsY.loc[animal1_BPsY.index[currRow], animalBpHeaderListY[cords]]
                    if animalBpHeaderListX[cords] in animal_1_BpHeaderList:
                        color = colorList_animal_1[cords]
                    elif animalBpHeaderListX[cords] in animal_2_BpHeaderList:
                        color = colorList_animal_2[cords]
                    cv2.circle(frame, (int(currXval), int(currYval)), circleScale, color, -1, lineType=cv2.LINE_AA)
                    if (mulltiAnimalStatus == True) and ('Center' in animalBpHeaderListX[cords]) and (animalBpHeaderListX[cords] in animal_1_BpHeaderList):
                        IDlabelLoc.append([currXval, currYval])
                    if (mulltiAnimalStatus == True) and ('Center' in animalBpHeaderListX[cords]) and (animalBpHeaderListX[cords] in animal_2_BpHeaderList):
                        IDlabelLoc.append([currXval, currYval])
                if (not IDlabelLoc) and (mulltiAnimalStatus == True):
                    animal1_x, animal1_y = currentDf.at[currRow, animal_1_BpHeaderList[0]], currentDf.at[currRow, animal_1_BpHeaderList[1]]
                    animal2_x, animal2_y = currentDf.at[currRow, animal_2_BpHeaderList[0]], currentDf.at[currRow, animal_2_BpHeaderList[1]]
                    IDlabelLoc.append([animal1_x, animal1_y])
                    IDlabelLoc.append([animal2_x, animal2_y])
                if height < width:
                    frame = ndimage.rotate(frame, 90)
                    rotationFlag = True
                if (mulltiAnimalStatus == True):
                    if rotationFlag == False:
                        cv2.putText(frame, str(mulltiAnimalIDList[0]), (IDlabelLoc[0][0], IDlabelLoc[0][1]), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0, 255, 0), 4)
                        cv2.putText(frame, str(mulltiAnimalIDList[1]), (IDlabelLoc[1][0], IDlabelLoc[1][1]), cv2.FONT_HERSHEY_COMPLEX,fontScale, (0, 255, 0), 4)
                    if rotationFlag == True:
                        newX1, newY1 = abs(int(IDlabelLoc[0][0]*cos(a) + IDlabelLoc[0][1]*sin(a))), int(frame.shape[0] - int(((-IDlabelLoc[0][1])*cos(a) + IDlabelLoc[0][0]*sin(a))))
                        newX2, newY2 = abs(int(IDlabelLoc[1][0] * cos(a) + IDlabelLoc[1][1] * sin(a))), int(frame.shape[0] - int(((-IDlabelLoc[1][1]) * cos(a) + IDlabelLoc[1][0] * sin(a))))
                        cv2.putText(frame, str(mulltiAnimalIDList[0]), (newX1, newY1), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0,0,255), 4)
                        cv2.putText(frame, str(mulltiAnimalIDList[1]), (newX2, newY2), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255,0,0), 4)

                # draw event timers
                for b in range(counters_no):
                    target_timers[b] = (1 / fps) * target_counters[b]
                    target_timers[b] = round(target_timers[b], 2)

                cv2.putText(frame, str('Timers'), (10, ((height - height) + spacingScale)), cv2.FONT_HERSHEY_COMPLEX, fontScale, (0, 255, 0), 4)
                addSpacer = 2
                for k in range(counters_no):
                    cv2.putText(frame, (str(target_names[k]) + ' ' + str(target_timers[k]) + str('s')), (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 4)
                    addSpacer += 1
                cv2.putText(frame, str('ensemble prediction'), (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 4)

                addSpacer += 1
                for p in range(counters_no):
                    TargetVal = int(currentDf.loc[currRow, [target_names[p]]])
                    if TargetVal == 1:
                        cv2.putText(frame, str(target_names[p]), (10, (height - height) + spacingScale * addSpacer), cv2.FONT_HERSHEY_TRIPLEX, int(fontScale*1.8), colors[p], 4)
                        target_counters[p] += 1
                        addSpacer += 1
                if videoSetting == 1:
                    writer.write(frame)
                if frameSetting == 1:
                    frameName = os.path.join(videoFrameDir, str(currRow) + '.png')
                    cv2.imwrite(frameName, frame)
                if (videoSetting == 0) and (frameSetting == 0):
                    print('Error: Please choose video and/or frames.')
                    break
                currRow+=1
                print('Frame ' + str(currRow) + '/' + str(frames) + '. Video ' + str(loopy) + '/' + str(len(filesFound)))
            if frame is None:
                print('Video ' + str(os.path.basename(CurrentVideoName.replace('.csv', '.mp4'))) + ' saved.')
                cap.release()
                break


