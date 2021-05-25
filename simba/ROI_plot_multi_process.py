from configparser import ConfigParser, NoOptionError
import os
import pandas as pd
import cv2
import numpy as np
from shapely.geometry import Point
from shapely import geometry
import glob
import subprocess as sp
import multiprocessing as mp

def roiPlot(inifile):
    config = ConfigParser()
    config.read(inifile)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    ## get dataframe column name
    bpcsv = (os.path.join(os.path.dirname(inifile), 'logs', 'measures', 'pose_configs', 'bp_names',
                          'project_bp_names.csv'))
    bplist = []
    with open(bpcsv) as f:
        for row in f:
            bplist.append(row)
    bplist = list(map(lambda x: x.replace('\n', ''), bplist))
    bpcolname = ['scorer']
    for i in bplist:
        bpcolname.append(i + '_x')
        bpcolname.append(i + '_y')
        bpcolname.append(i + '_p')
    noAnimals = config.getint('ROI settings', 'no_of_animals')
    animalBodypartList = []
    if noAnimals == 2:
        arrayIndex = 2
        bodyPartAnimal_1 = config.get('ROI settings', 'animal_1_bp')
        animalBodypartList.append(bodyPartAnimal_1)
        bodyPartAnimal_2 = config.get('ROI settings', 'animal_2_bp')
        animalBodypartList.append(bodyPartAnimal_2)
        trackedBodyPartNames = ['Animal_1', 'Animal_2']
    else:
        arrayIndex = 1
        bodyPartAnimal_1 = config.get('ROI settings', 'animal_1_bp')
        animalBodypartList.append(bodyPartAnimal_1)
        trackedBodyPartNames = ['Animal_1']
    trackedBodyParts = []
    for i in range(len(animalBodypartList)):
        bps = [str(animalBodypartList[i]) + '_x', str(animalBodypartList[i]) + '_y']
        trackedBodyParts.append(bps)
    vidInfPath = config.get('General settings', 'project_path')
    videoDirIn = os.path.join(vidInfPath, 'videos')
    logFolderPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    frames_dir_out = config.get('Frame settings', 'frames_dir_out')
    frames_dir_out = os.path.join(frames_dir_out, 'ROI_analysis')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    fileCounter = 0
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)


    for i in filesFound:
        fileCounter +=1
        CurrVidFn = os.path.basename(i)
        CurrentVideoName = os.path.basename(i).replace('.' + wfileType, '')
        if os.path.isfile(os.path.join(videoDirIn, CurrentVideoName + '.mp4')):
            currentVideo = os.path.join(videoDirIn, CurrentVideoName + '.mp4')
        else:
            currentVideo = os.path.join(videoDirIn, CurrentVideoName + '.avi')
        cap = cv2.VideoCapture(currentVideo)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mySpaceScale, myRadius, myResolution, myFontScale = 60, 20, 1500, 1.5
        maxResDimension = max(width, height)
        DrawScale = int(myRadius / (myResolution / maxResDimension))
        textScale = float(myFontScale / (myResolution / maxResDimension))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName)]
        currFps = int(videoSettings['fps'])
        noRectangles = len(rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        noCircles = len(circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        noPolygons = len(polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        rectangleTimes, rectangleEntries = ([[0] * len(trackedBodyParts) for i in range(noRectangles)] , [[0] * len(trackedBodyParts) for i in range(noRectangles)])
        circleTimes, circleEntries = ([[0] * len(trackedBodyParts) for i in range(noCircles)], [[0] * len(trackedBodyParts) for i in range(noCircles)])
        polygonTime, polyGonEntries = ([[0] * len(trackedBodyParts) for i in range(noPolygons)], [[0] * len(trackedBodyParts) for i in range(noPolygons)])
        currFrameFolderOut = os.path.join(frames_dir_out, CurrentVideoName + '.mp4')
        Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        Circles = (circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        rectangleEntryCheck = [[True] * len(trackedBodyParts) for i in range(noRectangles)]
        circleEntryCheck = [[True] * len(trackedBodyParts) for i in range(noCircles)]
        polygonEntryCheck = [[True] * len(trackedBodyParts) for i in range(noPolygons)]
        currDfPath = os.path.join(csv_dir_in, CurrVidFn)
        if wfileType == 'csv':
            currDf = pd.read_csv(currDfPath)
        if wfileType == 'parquet':
            currDf = pd.read_parquet(currDfPath)
        currDf.columns = bpcolname
        currRow = 0
        writer = cv2.VideoWriter(currFrameFolderOut, fourcc, fps, (width*2, height))
        RectangleColors = [(255, 191, 0), (255, 248, 240), (255,144,30), (230,224,176), (160, 158, 95), (208,224,63), (240, 207,137), (245,147,245), (204,142,0), (229,223,176), (208,216,129)]
        CircleColors = [(122, 160, 255), (0, 69, 255), (34,34,178), (0,0,255), (128, 128, 240), (2, 56, 121), (21, 113, 239), (5, 150, 235), (2, 106, 253), (0, 191, 255), (98, 152, 247)]
        polygonColor = [(0, 255, 0), (87, 139, 46), (152,241,152), (127,255,0), (47, 107, 85), (91, 154, 91), (70, 234, 199), (20, 255, 57), (135, 171, 41), (192, 240, 208), (131,193, 157)]

        while (cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                addSpacer = 2
                spacingScale = int(mySpaceScale / (myResolution / maxResDimension))
                borderImage = cv2.copyMakeBorder(img, 0,0,0,int(width), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                borderImageHeight, borderImageWidth = borderImage.shape[0], borderImage.shape[1]
                if noAnimals == 2:
                    currentPoints = (int(currDf.loc[currDf.index[currRow], trackedBodyParts[0][0]]), int(currDf.loc[currDf.index[currRow], trackedBodyParts[1][0]]), int(currDf.loc[currDf.index[currRow], trackedBodyParts[0][1]]), int(currDf.loc[currDf.index[currRow], trackedBodyParts[1][1]]))
                    cv2.circle(borderImage, (currentPoints[0], currentPoints[2]), DrawScale, (0, 255, 0), -1)
                    cv2.circle(borderImage, (currentPoints[1], currentPoints[3]), DrawScale, (0, 140, 255), -1)
                if noAnimals == 1:
                    currentPoints = (int(currDf.loc[currDf.index[currRow], trackedBodyParts[0][0]]), int(currDf.loc[currDf.index[currRow], trackedBodyParts[0][1]]))
                    cv2.circle(borderImage, (currentPoints[0], currentPoints[1]), DrawScale, (0, 255, 0), -1)
                addSpacer += 1
                for rectangle in range(noRectangles):
                    topLeftX, topLeftY = (Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle])
                    bottomRightX, bottomRightY = (topLeftX + Rectangles['width'].iloc[rectangle], topLeftY + Rectangles['height'].iloc[rectangle])
                    rectangleName = Rectangles['Name'].iloc[rectangle]
                    cv2.rectangle(borderImage, (topLeftX, topLeftY), (bottomRightX, bottomRightY), RectangleColors[rectangle], DrawScale)
                    for bodyparts in range(len(trackedBodyParts)):
                        cv2.putText(borderImage, str(rectangleName) + ' ' + str(trackedBodyPartNames[bodyparts]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 2)
                        if (((topLeftX-10) <= currentPoints[bodyparts] <= (bottomRightX+10)) and ((topLeftY-10) <= currentPoints[bodyparts+arrayIndex] <= (bottomRightY+10))):
                            rectangleTimes[rectangle][bodyparts] = round((rectangleTimes[rectangle][bodyparts] + (1 / currFps)), 2)
                            if rectangleEntryCheck[rectangle][bodyparts] == True:
                                rectangleEntries[rectangle][bodyparts] += 1
                                rectangleEntryCheck[rectangle][bodyparts] = False
                        else:
                            rectangleEntryCheck[rectangle][bodyparts] = True
                        cv2.putText(borderImage, str(rectangleTimes[rectangle][bodyparts]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 2)
                        addSpacer += 1
                        cv2.putText(borderImage, str(rectangleName) + ' ' + str(trackedBodyPartNames[bodyparts]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 2)
                        cv2.putText(borderImage, str(rectangleEntries[rectangle][bodyparts]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 2)
                        addSpacer += 1

                for circle in range(noCircles):
                    circleName, centerX, centerY, radius = (Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle])
                    cv2.circle(borderImage, (centerX, centerY), radius,  CircleColors[circle], DrawScale)
                    for bodyparts in range(len(trackedBodyParts)):
                        cv2.putText(borderImage, str(circleName) + ' ' +str(trackedBodyPartNames[bodyparts]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale,  CircleColors[circle], 2)
                        euclidPxDistance = int(np.sqrt((currentPoints[bodyparts] - centerX) ** 2 + (currentPoints[bodyparts+arrayIndex] - centerY) ** 2))
                        if euclidPxDistance <= radius:
                            circleTimes[circle][bodyparts] = round((circleTimes[circle][bodyparts] + (1 / currFps)),2)
                            if circleEntryCheck[circle][bodyparts] == True:
                                circleEntries[circle][bodyparts] += 1
                                circleEntryCheck[circle][bodyparts] = False
                        else:
                            circleEntryCheck[circle][bodyparts] = True
                        cv2.putText(borderImage, str(circleTimes[circle][bodyparts]), ((int(borderImageWidth-(borderImageWidth/6))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, CircleColors[circle], 2)
                        addSpacer += 1
                        cv2.putText(borderImage, str(circleName) + ' ' + str(trackedBodyPartNames[bodyparts]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, CircleColors[circle], 2)
                        cv2.putText(borderImage, str(circleEntries[circle][bodyparts]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, CircleColors[circle], 2)
                        addSpacer += 1

                for polygon in range(noPolygons):
                    PolygonName, vertices = (Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon])
                    vertices = np.array(vertices, np.int32)
                    cv2.polylines(borderImage, [vertices], True, polygonColor[polygon], thickness=DrawScale)
                    for bodyparts in range(len(trackedBodyParts)):
                        pointList = []
                        cv2.putText(borderImage, str(PolygonName) + ' ' + str(trackedBodyPartNames[bodyparts]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 2)
                        for i in vertices:
                            point = geometry.Point(i)
                            pointList.append(point)
                        polyGon = geometry.Polygon([[p.x, p.y] for p in pointList])
                        CurrPoint = Point(int(currentPoints[bodyparts]), int(currentPoints[bodyparts+arrayIndex]))
                        polyGonStatus = (polyGon.contains(CurrPoint))
                        if polyGonStatus == True:
                            polygonTime[polygon][bodyparts] = round((polygonTime[polygon][bodyparts] + (1 / currFps)), 2)
                            if polygonEntryCheck[polygon][bodyparts] == True:
                                polyGonEntries[polygon][bodyparts] += 1
                                polygonEntryCheck[polygon][bodyparts] = False
                        else:
                            polygonEntryCheck[polygon][bodyparts] = True
                        cv2.putText(borderImage, str(polygonTime[polygon][bodyparts]), ((int(borderImageWidth-(borderImageWidth/6))), (height - (height + 10) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 2)
                        addSpacer += 1
                        cv2.putText(borderImage, str(PolygonName) + ' ' + str(trackedBodyPartNames[bodyparts]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 2)
                        cv2.putText(borderImage, str(polyGonEntries[polygon][bodyparts]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 2)
                        addSpacer += 1
                borderImage = np.uint8(borderImage)
                writer.write(borderImage)
                currRow += 1
                print('Frame: ' + str(currRow) + '/' + str(frames) + '. Video ' + str(fileCounter) + '/' + str(len(filesFound)))
            if img is None:
                print('Video ' + str(CurrentVideoName) + ' saved.')
                cap.release()
                break
    print('All ROI videos generated in "project_folder/frames/ROI_analysis"')