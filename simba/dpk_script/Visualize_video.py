import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import cv2
import pandas as pd
import os
from configparser import ConfigParser
import glob
import time
import random

def visualizeDPK(dpkini):
    config = ConfigParser()
    configFile = str(dpkini)
    config.read(configFile)
    project_folder = config.get('general DPK settings', 'project_folder')
    predictionsFolder = os.path.join(project_folder, 'predictions')
    videoInputFolder = os.path.join(project_folder, 'videos', 'input')
    videoOutputFolder = os.path.join(project_folder, 'videos', 'output')

    if not os.path.exists(videoOutputFolder):
        os.makedirs(videoOutputFolder)
    filesFound = glob.glob(predictionsFolder + '/*.csv')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fileCounter = 0

    for predictions in filesFound:
        fileCounter += 1
        currRow = 0
        vidFileName = os.path.basename(predictions.replace('.csv', '.mp4'))
        predictionsDf = pd.read_csv(predictions, index_col=0)
        #predictionsDf = predictionsDf.loc[:, ~predictionsDf.columns.str.endswith('_p')]
        Xpredictions, Ypredictions, Ppredictions = (predictionsDf.filter(like='_x', axis=1), predictionsDf.filter(like='_y', axis=1), predictionsDf.filter(like='_p', axis=1))
        Xpredictions = Xpredictions.rename(columns=lambda x: x.strip('_x'))
        Ypredictions = Ypredictions.rename(columns=lambda x: x.strip('_y'))
        Ppredictions = Ppredictions.rename(columns=lambda x: x.strip('_p'))
        print(Ppredictions)
        bodypartColNames = list(Xpredictions.columns)
        vidOutputFile = os.path.join(videoOutputFolder, vidFileName.replace('.mp4', '.avi'))
        vidinputFile = os.path.join(videoInputFolder, vidFileName)
        cap = cv2.VideoCapture(vidinputFile)

        ## find vid size and fps
        colorList = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(vidOutputFile, fourcc, fps, (width, height))
        for color in range(len(bodypartColNames)):
            r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colorTuple = (r, g, b)
            colorList.append(colorTuple)
        while (cap.isOpened()):
            loop = 0
            ret, frame = cap.read()
            if ret == True:
                for bodyParts in bodypartColNames:
                    currPval = Ppredictions.loc[Ppredictions.index[currRow], bodyParts]
                    if currPval > 0.0001:
                        currXval = Xpredictions.loc[Xpredictions.index[currRow], bodyParts]
                        currYval = Ypredictions.loc[Ypredictions.index[currRow], bodyParts]
                        cv2.circle(frame, (int(currXval), int(currYval)), 10, colorList[loop], -1, lineType=cv2.LINE_AA)
                    loop+=1
                writer.write(frame)
                currRow+=1
                print('Frame: ' + str(currRow) + '/' + str(frames) + '. Video: ' + str(fileCounter) + '/' + str(len(filesFound)) + '.')
            if frame is None:
                print('Video ' + str(vidOutputFile) + ' saved.')
                cap.release()
                time.sleep(2)
                break
        print('All videos saved in ' + videoOutputFolder)