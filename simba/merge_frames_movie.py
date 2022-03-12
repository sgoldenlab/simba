import cv2
import pandas as pd
from PIL import Image
import os
import numpy as np
from configparser import ConfigParser
import glob
import time

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
    vidLogsDf["Video"] = vidLogsDf["Video"].astype(str)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def define_writer(outputImage, fps, fourcc, mergedFilePath, largePanelFlag):
        writer = cv2.VideoWriter(mergedFilePath, fourcc, fps, (outputImage.shape[1], outputImage.shape[0]))
        return writer

    def outPutFrameSize(totalImages, sklearnframeStatus, imgHeight, imgWidth):
        if sklearnframeStatus == 1:
            if totalImages == 1:
                if imgWidth > imgHeight:
                    outputImageHeight, outputImageWidh = imgWidth, imgHeight
                else:
                    outputImageHeight, outputImageWidh = imgHeight, imgWidth
            else:
                if (totalImages-1) == 1 or (totalImages-1) == 2:
                    columns = 2
                if (totalImages - 1) == 3 or (totalImages - 1) == 4:
                    columns = 3
                if (totalImages - 1) == 5 or (totalImages - 1) == 6:
                    columns = 4
                if (totalImages-1) == 2 or (totalImages % 2 == 0):
                    if imgWidth >= imgHeight:
                        outputImageHeight, outputImageWidh = imgWidth, imgHeight * (columns)
                    if imgWidth <= imgHeight:
                        outputImageHeight, outputImageWidh = imgHeight, imgWidth * (columns)
                if ((totalImages-1) == 1) or (not totalImages % 2 == 0):
                    if imgWidth >= imgHeight:
                        outputImageHeight, outputImageWidh = imgWidth, imgHeight * (columns)
                    if imgWidth <= imgHeight:
                        outputImageHeight, outputImageWidh = imgHeight, imgWidth * (columns)
        if sklearnframeStatus == 0:
            if totalImages == 1:
                outputImageHeight, outputImageWidh = imgHeight, imgWidth
            if totalImages == 2:
                outputImageHeight, outputImageWidh = imgHeight*2, imgWidth
            if (totalImages == 3) or (totalImages == 4):
                outputImageHeight, outputImageWidh = imgHeight*2, imgWidth*2
            if (totalImages == 5) or (totalImages == 6):
                outputImageHeight, outputImageWidh = imgHeight * 2, imgWidth * 3
            if (totalImages == 7) or (totalImages == 8):
                outputImageHeight, outputImageWidh = imgHeight * 2, imgWidth * 4

        return int(outputImageHeight), int(outputImageWidh)

    if not os.path.exists(framesDir):
        os.makedirs(framesDir)

    dirsList, fakePathList = [], []
    totalImages = sum(inputList)
    for status, foldername in zip(dirStatusList, dirFolders):
        if status == 1:
            folderPath = os.path.join(frameDirIn, foldername)
            foldersInFolder = [f.path for f in os.scandir(folderPath) if f.is_dir()]
            dirsList.append(foldersInFolder)

    ### check that all folders contain an equal number of frames and they are not zero ###
    toDelList = []
    for folder in range(len(dirsList)):
        imageCounts = []
        try:
            currFolders = [item[folder] for item in dirsList]
        except IndexError:
            currFolders = dirsList[folder]
        for category in currFolders:
            imageCounts.append(len(glob.glob(category + '/*.png')))
        checkIfSame = all(x == imageCounts[0] for x in imageCounts)
        if (checkIfSame == False) or (0 in imageCounts) or (None in imageCounts):
            if os.path.basename(category) not in toDelList:
                toDelList.append(os.path.basename(category))
                print(os.path.basename(category) + str(' has zero frames or an unequal number of frames in one or several frame categories.'))

    # ### check that all folders contain each video
    videoPathList = []
    flat_list = [item for sublist in dirsList for item in sublist]
    for item in flat_list:
        videoPathList.append(os.path.basename(item))
    uniqVideo = set(videoPathList)
    for status, foldername in zip(dirStatusList, dirFolders):
        if status == 1:
            for basename in uniqVideo:
                folderPath = os.path.join(frameDirIn, foldername, basename)
                isdir = os.path.isdir(folderPath)
                if isdir == False:
                    toDelList.append(os.path.basename(folderPath))
                    print(os.path.basename(folderPath) + str(' do not have frames for each user-specified frame category'))
    if toDelList:
        print('Videos ' + str(toDelList) + ' have missing frame folders, frame folders that are empty, or frame folder categories where the number of frames do not match. SimBA will not generate merged videos from these frames.')
        time.sleep(6)
    toDel2 = []
    for list in dirsList:
        for path in list:
            currBase = os.path.basename(path)
            for deleteWords in toDelList:
                if deleteWords == currBase:
                    toDel2.append(path)
    for i in toDel2:
        dirsList = [[ele for ele in sub if ele != i] for sub in dirsList]
    for list in dirsList:
        sorted(list, key=str.lower)

    for video in range(len(dirsList[0])):
        try:
            currentVidFolders = [item[video] for item in dirsList]
        except IndexError:
            print('Error: all the selected frame categories have not been created for the videos.')
        vidBaseName = os.path.basename(currentVidFolders[0])
        print(vidBaseName)
        currVidInfo = vidLogsDf.loc[vidLogsDf['Video'] == str(vidBaseName)]
        try:
            fps = int(currVidInfo['fps'])
        except TypeError:
            print('Error: make sure your videos you are merging are represented in your video info log file.')
            break
        img = cv2.imread(os.path.join(currentVidFolders[0], '0.png'))
        imgHeight, imgWidth = img.shape[0], img.shape[1]
        y_offsets = [0, int((imgHeight / 2))]
        mergedFilePath = os.path.join(framesDir, vidBaseName + '.mp4')
        imageLen = len(os.listdir(currentVidFolders[0]))
        largePanelFlag, rotationFlag = False, False
        outputImageHeight, outputImageWidh = outPutFrameSize(totalImages, inputList[0], imgHeight, imgWidth)
        outputImage = np.zeros((outputImageHeight, outputImageWidh, 3))
        for images in range(imageLen):
            y_offset, x_offset, imageNumber, panelCounter = 0, 0, 0, 0
            for panel in currentVidFolders:
                panelCounter+=1
                imagePath = os.path.join(panel, str(images) + '.png')
                try:
                    img = cv2.imread(imagePath)
                except cv2.error as e:
                    print('Could not find frames in  ' + str(os.path.basename(panel)) + '. Make sure they have first been created before merging video')
                panelDirectory = os.path.basename(os.path.dirname(panel))
                if (panel == currentVidFolders[0]) and (panelDirectory == 'sklearn_results'):
                    largePanelHeight, largePanelWidth = img.shape[0], img.shape[1]
                    if imgHeight < imgWidth:
                        rotationFlag = True
                        img = np.array(Image.fromarray(img).rotate(90,Image.BICUBIC, expand=True))
                    outputImage[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
                    largePanelFlag = True
                else:
                    if largePanelFlag == True:
                        if (rotationFlag == True) and (panelDirectory == 'path_plots'):
                            img = np.array(Image.fromarray(img).rotate(90,Image.BICUBIC, expand=True))
                        try:
                            img = cv2.resize(img, (int(largePanelWidth), int(largePanelHeight / 2)))
                        except cv2.error as e:
                            print('Could not find frames in  ' + str(os.path.basename(panel)) + '. Make sure they have first been created before merging video')
                            break
                        if (panelCounter % 2 == 0):
                            y_offset = y_offsets[0]
                            x_offset = x_offset + img.shape[1]
                        if not panelCounter % 2 == 0 and (panelCounter != 1):
                            y_offset = y_offsets[1]
                    if largePanelFlag == False:
                        y_offsets = [0, int((imgHeight))]
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
            writer.write(outputImage)
            print('Image ' + str(images + 1) + '/' + str(imageLen) + '. Video ' + str(video + 1) + '/' + str(len(dirsList[0])))
    print('All movies generated')
    cv2.destroyAllWindows()
    writer.release()


















































