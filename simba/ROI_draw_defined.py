from configparser import ConfigParser
import os
import cv2
import numpy as np
import pandas as pd
import warnings
import glob


def roiByDefinition(inifile):
    global ix, iy
    global topLeftStatus
    global overlay
    global topLeftX, topLeftY, bottomRightX, bottomRightY
    global ix, iy
    global centerStatus
    global overlay
    global currCircleRadius
    global centerX, centerY, radius
    global centroids, toRemoveShapeName, removeStatus, toRemoveShape
    global recWidth, recHeight, firstLoop
    config = ConfigParser()
    configFile = str(inifile)
    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
    pd.options.mode.chained_assignment = None
    config.read(configFile)
    vidInfPath = config.get('General settings', 'project_path')
    videofilesFolder = os.path.join(vidInfPath, "videos")
    logFolderPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    rectangularDf = pd.DataFrame(columns=['Video', "Shape_type", "Name", "width", "height", "topLeftX", "topLeftY"])
    circleDf = pd.DataFrame(columns=['Video', "Shape_type", "Name", "centerX", "centerY", "radius"])
    fscale = 0.02
    space_scale = 1.1
    outPutPath = os.path.join(logFolderPath, 'measures')
    ROIcoordinatesPath = os.path.join(outPutPath, 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')

    # mouse callback function
    def draw_rectangle(event,x,y,flags,param):
        global ix,iy
        global topLeftStatus
        global overlay
        global topLeftX, topLeftY, bottomRightX, bottomRightY
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            print(topLeftStatus)
            topLeftX, topLeftY, bottomRightX, bottomRightY = (x, y, x+currRecWidth, y+currRecHeight)
            print(topLeftX, topLeftY, bottomRightX, bottomRightY)
            if topLeftStatus == True:
                overlay = img.copy()
            cv2.rectangle(overlay, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (255, 0, 0), 3)
            cv2.imshow('Define shape', overlay)
            topLeftStatus = True

    def draw_circle(event,x,y,flags,param):
        global ix,iy
        global centerStatus
        global overlay
        global currCircleRadius
        global centerX, centerY, radius
        global overlay
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            centerX, centerY, radius = (int(x), int(y), int(currCircleRadius))
            if centerStatus == True:
                overlay = img.copy()
            cv2.circle(overlay, (centerX, centerY), radius, (144, 0, 255), 2)
            cv2.imshow('Define shape', overlay)
            centerStatus = True


    def select_shape_to_change(event, x, y, flags, param):
        global centroids, toRemoveShapeName, removeStatus, toRemoveShape
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            toRemove = centroids[(centroids['CenterX'] <= x+20) & (centroids['CenterX'] >= x-20) & (centroids['CenterY'] <= y+20) & (centroids['CenterY'] >= y-20)]
            toRemoveShapeName, toRemoveShape = (toRemove["Name"].iloc[0], toRemove["Shape"].iloc[0])
            updateImage(centroids, toRemoveShapeName)
            removeStatus = False

    def select_new_shape_centroid_loc(event, x, y, flags, param):
        global toRemoveShapeName, removeStatus
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            toInsertShapeName, toInsertShape = toRemoveShapeName, toRemoveShape
            toRemoveShapeName = ''
            if toInsertShape == 'rectangle':
                newTopLeftX, newTopLeftY = (int(x -(recWidth/2)), int((y -(recHeight/2))))
                currRectangleDf.loc[(currRectangleDf['Name'] == toInsertShapeName), 'topLeftX'] = newTopLeftX
                currRectangleDf.loc[(currRectangleDf['Name'] == toInsertShapeName), 'topLeftY'] = newTopLeftY
            if toInsertShape == 'circle':
                newcenterX, newcenterY = (int(x), int(y))
                currCirclesDf.loc[(currCirclesDf['Name'] == toInsertShapeName), 'centerX'] = newcenterX
                currCirclesDf.loc[(currCirclesDf['Name'] == toInsertShapeName), 'centerY'] = newcenterY
            removeStatus = True
        updateImage(centroids, toRemoveShapeName)

    videofilesFound = glob.glob(videofilesFolder + '/*.mp4') + glob.glob(videofilesFolder + '/*.avi')
    cap = cv2.VideoCapture(videofilesFound[0])
    cap.set(1, 0)
    ret, frame = cap.read()
    fileName = str(0) + str('.bmp')
    filePath = os.path.join(videofilesFolder, fileName)
    cv2.imwrite(filePath,frame)
    img = cv2.imread(filePath)
    CurrVidName = os.path.splitext(os.path.basename(videofilesFound[0]))[0]
    CurrVidSet = vidinfDf.loc[vidinfDf['Video'] == CurrVidName]
    try:
        videoHeight = int(CurrVidSet['Resolution_height'])
        videoWidth = int(CurrVidSet['Resolution_width'])
        currPixPerMM = float(CurrVidSet['pixels/mm'])
    except TypeError:
        print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
    fontScale = min(videoWidth, videoHeight) / (25 / fscale)
    spacingScale = int(min(videoWidth, videoHeight) / (25 / space_scale))

    ##### RECTANGLES ####
    currRectangleDf = rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrVidName)]
    shapeType = 'rectangle'
    addSpacer = 2
    for index, row in currRectangleDf.iterrows():
        currRecName, currRecWidth,currRecHeight = (row['Name'], int(row['width']*currPixPerMM), int(row['height']*currPixPerMM))
        im = np.zeros((videoHeight, videoWidth, 3))
        cv2.putText(im, str(CurrVidName), (10, (videoHeight - videoHeight) + spacingScale), cv2.FONT_HERSHEY_SIMPLEX, fontScale,(0, 255, 0), 2)
        cv2.putText(im, 'Draw rectangle: ' + str(currRecName), (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
        addSpacer+=1
        cv2.putText(im, 'Pre-specified size: ' + str(row['width']) + 'x' + str(row['height']) + 'mm', (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
        addSpacer+=1
        cv2.putText(im, 'Instructions', (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
        addSpacer+=1
        cv2.putText(im, str('Double left click at the top right corner of the rectangle'), (10, (videoHeight - videoHeight )+ spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
        addSpacer+=1
        cv2.putText(im, str('Press ESC to start or to continue'), (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
        while (1):
            cv2.imshow('Instructions', im)
            k = cv2.waitKey(0)
            addSpacer = 2
            if k==27:    # Esc key to stop
                break

        cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
        ix, iy = -1, -1
        overlay = img.copy()
        topLeftStatus = False
        while (1):
            cv2.setMouseCallback('Define shape', draw_rectangle)
            cv2.imshow('Define shape', overlay)
            k = cv2.waitKey(20)
            if k == 27:
                boxList = [CurrVidName, shapeType, currRecName, currRecWidth, currRecHeight, topLeftX, topLeftY]
                rectangularDf = rectangularDf.append(pd.Series(dict(zip(rectangularDf.columns, boxList))), ignore_index=True)
                img = overlay.copy()
                cv2.destroyWindow('Define shape')
                break

    ##### CIRCLES ####
    currCirclesDf = circleInfo.loc[circleInfo['Video'] == str(CurrVidName)]
    shapeType = 'circle'
    for index, row in currCirclesDf.iterrows():
        currCircleName, currCircleRadius = (row['Name'], row['Radius']*currPixPerMM)
        im = np.zeros((videoHeight, videoWidth, 3))
        addSpacer += 1
        cv2.putText(im, str(CurrVidName), (10, (videoHeight - videoHeight) + spacingScale), cv2.FONT_HERSHEY_SIMPLEX, fontScale + 0.15,(144, 0, 255), 2)
        addSpacer += 1
        cv2.putText(im, 'Draw circle: ' + str(currCircleName), (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale + 0.2, (144, 0, 255), 2)
        addSpacer += 1
        cv2.putText(im, 'Instructions', (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 2)
        addSpacer += 1
        cv2.putText(im, str('Double left click to specify the center of the circle'), (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale + 0.025, (255, 255, 255), 2)
        addSpacer += 1
        cv2.putText(im, str('Press ESC or Enter to start or to continue'), (10, (videoHeight - videoHeight) + spacingScale * addSpacer), cv2.FONT_HERSHEY_SIMPLEX, fontScale + 0.05, (255, 255, 255), 2)
        while (1):
            cv2.imshow('Instructions', im)
            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop
                break
        overlay = img.copy()
        ix, iy = -1, -1
        cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
        centerStatus = False
        while (1):
            cv2.setMouseCallback('Define shape',draw_circle)
            cv2.imshow('Define shape', overlay)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                circList = [CurrVidName, shapeType, currCircleName, centerX, centerY, radius]
                circleDf = circleDf.append(pd.Series(dict(zip(circleDf.columns, circList))),ignore_index=True)
                img = overlay.copy()
                cv2.destroyWindow('Define shape')
                break

    duplicatedRec, duplicatedCirc = (rectangularDf.copy(), circleDf.copy())

    for othervids in videofilesFound[1:]:
        currVidName = os.path.splitext(os.path.basename(othervids))[0]
        duplicatedRec['Video'], duplicatedCirc['Video'] = (currVidName, currVidName)
        rectangularDf = rectangularDf.append(duplicatedRec, ignore_index=True)
        circleDf = circleDf.append(duplicatedCirc, ignore_index=True)

    def updateImage(centroids, toRemoveShapeName):
        global recWidth, recHeight, firstLoop
        overlay = img.copy()
        centroids = centroids.drop_duplicates()
        for rectangle in range(len(currRectangleDf)):
            videoName, recName, topLeftX, topLeftY = (currRectangleDf['Video'].iloc[rectangle], currRectangleDf['Name'].iloc[rectangle], currRectangleDf['topLeftX'].iloc[rectangle],currRectangleDf['topLeftY'].iloc[rectangle])
            if recName == toRemoveShapeName:
                recWidth, recHeight = (currRectangleDf['width'].iloc[rectangle], currRectangleDf['height'].iloc[rectangle])
                continue
            else:
                bottomRightX, bottomRightY = (topLeftX + currRectangleDf['width'].iloc[rectangle], topLeftY + currRectangleDf['height'].iloc[rectangle])
                cv2.rectangle(overlay, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (255, 0, 0), 10)
                centerOfShape = [(topLeftX + bottomRightX) / 2, (topLeftY + bottomRightY) / 2]
                cv2.circle(overlay, (int(centerOfShape[0]), int(centerOfShape[1])), 12, (255, 0, 0), -1)
                if firstLoop == True:
                    CentroidList = [CurrVidName, 'rectangle', recName, centerOfShape[0], centerOfShape[1]]
                    centroids = centroids.append(pd.Series(dict(zip(centroids.columns, CentroidList))), ignore_index=True)
                else:
                    centroids.loc[(centroids['Name'] == recName), 'CenterX'] = centerOfShape[0]
                    centroids.loc[(centroids['Name'] == recName), 'CenterY'] = centerOfShape[1]
                    correctionMask = (rectangularDf['Name'] == recName) & (rectangularDf['Video'] == videoName)
                    rectangularDf['topLeftX'][correctionMask], rectangularDf['topLeftY'][correctionMask] = topLeftX, topLeftY
        for circle in range(len(currCirclesDf)):
            videoName, circleName, centerX, centerY, radius = (currCirclesDf['Video'].iloc[circle], currCirclesDf['Name'].iloc[circle], currCirclesDf['centerX'].iloc[circle], currCirclesDf['centerY'].iloc[circle], currCirclesDf['radius'].iloc[circle])
            if circleName == toRemoveShapeName:
                continue
            else:
                cv2.circle(overlay, (centerX, centerY), radius, (144, 0, 255), 2)
                cv2.circle(overlay, (centerX, centerY), 12, (144, 0, 255), -1)
                if firstLoop == True:
                    CentroidList = [CurrVidName, 'circle', circleName, centerX, centerY]
                    centroids = centroids.append(pd.Series(dict(zip(centroids.columns, CentroidList))), ignore_index=True)
                else:
                    centroids.loc[(centroids['Name'] == circleName), 'CenterX'] = centerX
                    centroids.loc[(centroids['Name'] == circleName), 'CenterY'] = centerY
                    correctionMask = (circleDf['Name'] == circleName) & (circleDf['Video'] == videoName)
                    circleDf['centerX'][correctionMask], circleDf['centerY'][correctionMask] = centerX, centerY
        firstLoop = False
        cv2.imshow('Define shape', overlay)
        return centroids
    cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
    for videos in videofilesFound[1:]:
        firstLoop = True
        cap = cv2.VideoCapture(videos)
        cap.set(1, 0)
        ret, frame = cap.read()
        fileName = str(0) + str('.bmp')
        filePath = os.path.join(videofilesFolder, fileName)
        cv2.imwrite(filePath, frame)
        img = cv2.imread(filePath)
        overlay = img.copy()
        CurrVidName = os.path.splitext(os.path.basename(videos))[0]
        currRectangleDf = rectangularDf.loc[rectanglesInfo['Video'] == str(CurrVidName)]
        currCirclesDf = circleDf.loc[circleInfo['Video'] == str(CurrVidName)]
        centroids = pd.DataFrame(columns=['Video', "Shape", "Name", "CenterX", "CenterY"])
        toRemoveShapeName = ''
        removeStatus = True
        ix, iy = -1, -1
        while (1):
            centroids = updateImage(centroids, toRemoveShapeName)
            if removeStatus == True:
                cv2.setMouseCallback('Define shape', select_shape_to_change)
            if removeStatus == False:
                cv2.setMouseCallback('Define shape', select_new_shape_centroid_loc)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                cv2.destroyWindow('Define shape')
                cv2.destroyWindow('Instructions')
                break
            break

    storePath = os.path.join(outPutPath, 'ROI_definitions.h5')
    store = pd.HDFStore(storePath)
    store['rectangles'] = rectangularDf
    store['circleDf'] = circleDf
    polygonDf = pd.DataFrame(columns=['Video', "Shape_type", "Name", "vertices"])
    store['polygons'] = polygonDf
    print('ROI definitions saved in ' + str('storePath'))
    store.close()
    print(rectangularDf)
    print(circleDf)
    print(polygonDf)