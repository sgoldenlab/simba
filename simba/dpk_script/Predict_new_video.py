import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import cv2
import numpy as np
from deepposekit.models import load_model
from deepposekit.io import DataGenerator, VideoReader, VideoWriter
import os
from configparser import ConfigParser
import pandas as pd
import glob

def predictnewvideoDPK(dpkini,videofolder):
    configFile = str(dpkini)
    config = ConfigParser()
    config.read(configFile)
    project_folder = config.get('general DPK settings', 'project_folder')
    modelPath = config.get('predict settings', 'modelPath')
    videoFolderPath = videofolder
    print(videoFolderPath)
    batchSize = config.getint('predict settings', 'batch_size')
    outputfolder = os.path.join(project_folder, 'predictions')
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    bodyPartColumnNames = []

    skeletonPath = os.path.join(project_folder, 'skeleton.csv')
    skeletonDf = pd.read_csv(skeletonPath)
    skeletonList = list(skeletonDf['name'])

    for i in skeletonList:
        x_col, y_col, p_col = (str(i) + '_x', str(i) + '_y', str(i) + '_p')
        bodyPartColumnNames.append(x_col)
        bodyPartColumnNames.append(y_col)
        bodyPartColumnNames.append(p_col)

    filesFound = glob.glob(videoFolderPath + '/*.mp4')


    #Check if videos are greyscale
    cap = cv2.VideoCapture(filesFound[0])
    cap.set(1, 0)
    ret, frame = cap.read()
    fileName = str(0) + str('.bmp')
    filePath = os.path.join(videoFolderPath, fileName)
    cv2.imwrite(filePath, frame)
    img = cv2.imread(filePath)
    imgDepth = img.shape[2]
    if imgDepth == 3:
        greyscaleStatus = False
    else:
        greyscaleStatus = True
    os.remove(filePath)

    # This loads the trained model into memory for making predictions
    model = load_model(modelPath)
    for video in filesFound:
        print('Analyzing file: ' + str(os.path.basename(video)))
        reader = VideoReader(video, batch_size=batchSize, gray=greyscaleStatus)
        predictions = model.predict(reader, verbose=1)
        reader.close()
        outputFilename = os.path.join(outputfolder, os.path.basename(video).replace('.mp4', '.csv'))
        x, y, confidence = np.split(predictions, 3, -1)
        outputDataFrame = pd.DataFrame(columns=bodyPartColumnNames)
        for i in range(len(x)):
            currX, currY, currConf = (x[i], y[i], confidence[i])
            currCordList = []
            for ii in range(len(currX)):
                cords = [float(currX[ii]), float(currY[ii]), float(currConf[ii])]
                currCordList.extend(cords)
            outputDataFrame = outputDataFrame.append(pd.Series(dict(zip(outputDataFrame.columns, currCordList))), ignore_index=True)
            outputDataFrame.reset_index(inplace=True, drop=True)
        outputDataFrame.reset_index(inplace=True, drop=True)
        outputDataFrame.to_csv(outputFilename)
        print('Saved predictions: ' + outputFilename)
    print('All files analyzed.')