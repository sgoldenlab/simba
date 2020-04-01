import cv2
import pandas as pd
from scipy import ndimage
import os
import numpy as np
from configparser import ConfigParser
import math

def mergeframesPlot(configini,inputList):
    dirStatusList = inputList
    dirFolders = ["sklearn_results", "gantt_plots", "path_plots", "live_data_table", "line_plot", "probability_plots"]
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    frameDirIn = os.path.join(projectPath, 'frames', 'output')
    framesDir = os.path.join(projectPath, 'frames', 'output', 'merged')
    vidLogsPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidLogsDf = pd.read_csv(vidLogsPath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def define_writer(outputImage, fps, fourcc, mergedFilePath, largePanelFlag):
        print(outputImage.shape[0], outputImage.shape[1])
        if largePanelFlag == True:
            writer = cv2.VideoWriter(mergedFilePath, fourcc, fps, (outputImage.shape[1], outputImage.shape[0]))
        if largePanelFlag == False:
            writer = cv2.VideoWriter(mergedFilePath, fourcc, fps, (outputImage.shape[1], outputImage.shape[0]))
        return writer


    if not os.path.exists(framesDir):
        os.makedirs(framesDir)
    dirsList, toDelList = [], []
    total=0

    for i in inputList:
        total += i
    totalImages = total
    if not totalImages % 2 == 0:
        totalImages += 1

    for status, foldername in zip(dirStatusList, dirFolders):
        if status == 1:
            folderPath = os.path.join(frameDirIn, foldername)
            foldersInFolder = [f.path for f in os.scandir(folderPath) if f.is_dir()]
            dirsList.append(foldersInFolder)

    for video in range(len(dirsList[0])):
        currentVidFolders = [item[video] for item in dirsList]
        vidBaseName = os.path.basename(currentVidFolders[0])
        currVidInfo = vidLogsDf.loc[vidLogsDf['Video'] == str(vidBaseName)]
        fps = int(currVidInfo['fps'])
        img = cv2.imread(os.path.join(currentVidFolders[video], '0.png'))
        imgHeight, imgWidth = img.shape[0], img.shape[1]
        y_offsets = [0, int((imgHeight / 2))]
        outputImage = np.zeros((imgHeight, imgWidth*(totalImages-1), 3))
        mergedFilePath = os.path.join(framesDir, vidBaseName + '.mp4')
        imageLen = len(os.listdir(currentVidFolders[0]))
        largePanelFlag, rotationFlag = False, False
        for images in range(imageLen):
            y_offset, x_offset, imageNumber, panelCounter = 0, 0, 0, 0
            for panel in currentVidFolders:
                panelCounter+=1
                imagePath = os.path.join(panel, str(images) + '.png')
                img = cv2.imread(imagePath)
                panelDirectory = os.path.basename(os.path.dirname(panel))
                if (panel == currentVidFolders[0]) and (panelDirectory == 'sklearn_results'):
                    if totalImages == 2:
                        outputImage = np.zeros((imgHeight, imgWidth * (totalImages), 3))
                    else:
                        outputImage = np.zeros((imgHeight, imgWidth * (totalImages - 1), 3))
                    largePanelHeight, largePanelWidth = img.shape[0], img.shape[1]
                    if imgHeight < imgWidth:
                        rotationFlag = True
                        img = ndimage.rotate(img, 90)
                    outputImage[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
                    largePanelFlag = True
                else:
                    if largePanelFlag == True:
                        if (rotationFlag == True) and (panelDirectory == 'path_plots'):
                            img = ndimage.rotate(img, 90)
                        img = cv2.resize(img, (int(largePanelWidth), int(largePanelHeight / 2)))
                        if (panelCounter % 2 == 0):
                            y_offset = y_offsets[0]
                            x_offset = x_offset + img.shape[1]
                        if not panelCounter % 2 == 0 and (panelCounter != 1):
                            y_offset = y_offsets[1]
                    if largePanelFlag == False:
                        y_offsets = [0, int((imgHeight))]
                        if (panelCounter == 1):
                            if (totalImages < 3):
                                outputImage = np.zeros((int(imgHeight*2), imgWidth, 3))
                            else:
                                outputImage = np.zeros((int(imgHeight * 2), int(imgWidth * int(math.ceil(totalImages/2))), 3))
                        if (not panelCounter % 2 == 0) and (panelCounter != 1):
                            y_offset = y_offsets[0]
                            x_offset = x_offset + img.shape[1]
                            img = cv2.resize(img, (int(imgWidth), int(imgHeight)))
                        if (panelCounter % 2 == 0) and (panelCounter != 1):
                            y_offset = y_offsets[1]
                            img = cv2.resize(img, (int(imgWidth), int(imgHeight)))
                    outputImage[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
            if (images == 0):
                writer = define_writer(outputImage, fps, fourcc, mergedFilePath, largePanelFlag)
            outputImage = np.uint8(outputImage)
            #outputImage = cv2.normalize(outputImage, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #cv2.imshow('test', outputImage)
            #cv2.waitKey(0)
            writer.write(outputImage)
            print('Image ' + str(images) + '/' + str(imageLen) + '. Video ' + str(video + 1) + '/' + str(len(dirsList[0])))
        print('All movies generated')
        cv2.destroyAllWindows()
        writer.release()

































































