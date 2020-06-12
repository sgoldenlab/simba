import cv2
import os
import subprocess
import pandas as pd
import glob


def multicrop(vtype,inputf,outputf,noOfCrop):
    outputFileEnding = '.' + vtype
    videoFtype = "." + '/*.' + vtype
    videoFolder = str(inputf)
    addSpacer = 2
    croppingDf = pd.DataFrame(columns=['Video', "height", "width", "topLeftX", "topLeftY"])
    outputFolder = str(outputf)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    filesFound = glob.glob(videoFolder + videoFtype)
    print('Video files found: ' + str(len(filesFound)))

    for videoName in filesFound:
        videoBaseName = os.path.basename(videoName)
        cap = cv2.VideoCapture(videoName)
        cap.set(1, 0)
        ret, frame = cap.read()
        fileName = str(0) + str('.bmp')
        filePath = os.path.join(videoFolder, fileName)
        cv2.imwrite(filePath,frame)
        img = cv2.imread(filePath)
        (height, width) = img.shape[:2]
        fscale = 0.02
        space_scale = 1.1
        fontScale = min(width, height) / (25 / fscale)
        spacingScale = int(min(width, height) / (25 / space_scale))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.putText(img, str(videoBaseName), (10, ((height - height) + spacingScale)), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 2)
        cv2.putText(img, str('Define the number of videos you want to create in the console'), (10, ((height - height) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 2)
        while (1):
            cv2.imshow('image', img)
            k = cv2.waitKey(20) & 0xFF
            currBoundingBoxes = int(noOfCrop)
            if currBoundingBoxes != 0:
                cv2.destroyAllWindows()
                break

        loopy = 0
        for boxes in range(currBoundingBoxes):
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            loopy+=1
            img = cv2.imread(filePath)
            print(filePath)
            cv2.putText(img, str(videoBaseName), (10, ((height - height) + spacingScale)), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 2)
            cv2.putText(img, str('Define video ') + str(loopy) + ' coordinates and press enter', (10, ((height - height) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 2)

            ROI = cv2.selectROI('image', img)
            width = (abs(ROI[0] - (ROI[2] + ROI[0])))
            height = (abs(ROI[2] - (ROI[3] + ROI[2])))
            topLeftX = ROI[0]
            topLeftY = ROI[1]
            boxList = [videoBaseName, height, width, topLeftX, topLeftY]
            k = cv2.waitKey(20) & 0xFF
            cv2.destroyAllWindows()
            croppingDf = croppingDf.append(pd.Series(dict(zip(croppingDf.columns, boxList))), ignore_index=True)
            print(croppingDf)
    os.remove(filePath)

    for videoName in filesFound:
        videoBaseName = os.path.basename(videoName)
        currentCropDf = croppingDf.loc[croppingDf['Video'] == videoBaseName]
        loop = 0
        for index, row in currentCropDf.iterrows():
            loop+=1
            height = row['height']
            width = row['width']
            topLeftX = row['topLeftX']
            topLeftY = row['topLeftY']
            print(width, height, topLeftX, topLeftY)
            fileOutName = videoBaseName.replace(outputFileEnding, '')
            fileOutName = fileOutName + '_' + str(loop) + str(outputFileEnding)
            fileOutPathName = os.path.join(outputFolder, fileOutName)
            command = str('ffmpeg -i ') + str(videoName) + str(' -vf ') + str('"crop=') + str(width) + ':' + str(height) + ':' + str(topLeftX) + ':' + str(topLeftY) + '" ' + str('-c:v libx264 -c:a copy ') + str(fileOutPathName)
            print(command)
            subprocess.call(command, shell=True)