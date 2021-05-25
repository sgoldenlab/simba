from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
import os
import pandas as pd
import cv2
import numpy as np
from shapely.geometry import Point
from shapely import geometry
import glob
from pylab import cm
from simba.rw_dfs import *
from simba.drop_bp_cords import *

def roiPlot(inifile, CurrentVideo):
    config = ConfigParser()
    config.read(inifile)

    ## get dataframe column name
    bplist = getBpHeaders(inifile)
    try:
        noAnimals = config.getint('ROI settings', 'no_of_animals')
    except NoOptionError:
        noAnimals = config.getint('General settings', 'animal_no')
    projectPath = config.get('General settings', 'project_path')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    try:
        probability_threshold = config.getfloat('ROI settings', 'probability_threshold')
    except NoOptionError:
        probability_threshold = 0.000


    animalBodypartList = []
    for bp in range(noAnimals):
        animalName = 'animal_' + str(bp + 1) + '_bp'
        animalBpName = config.get('ROI settings', animalName)
        animalBpNameX, animalBpNameY, animalBpNameP  = animalBpName + '_x', animalBpName + '_y', animalBpName + '_p'
        animalBodypartList.append([animalBpNameX, animalBpNameY, animalBpNameP])
    columns2grab = [item[0:3] for item in animalBodypartList]
    columns2grab = [item for sublist in columns2grab for item in sublist]

    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            multiAnimalIDList = []
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal ' + str(animal+1) + ' ')
            print('Applying settings for classical tracking...')

    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal ' + str(animal + 1) + ' ')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    logFolderPath = os.path.join(projectPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'ROI_analysis')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    try:
        rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    except FileNotFoundError:
        print('No ROIs found: please define ROIs using the left-most menu in teh SIMBA ROI tab.')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')

    CurrentVideoPath = os.path.join(projectPath, 'videos', CurrentVideo)
    cap = cv2.VideoCapture(CurrentVideoPath)
    CurrentVideoName, videoFileType = os.path.splitext(CurrentVideo)[0],  os.path.splitext(CurrentVideo)[1]
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height, frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mySpaceScale, myRadius, myResolution, myFontScale = 25, 10, 1500, 0.8
    maxResDimension = max(width, height)
    DrawScale = int(myRadius / (myResolution / maxResDimension))
    textScale = float(myFontScale / (myResolution / maxResDimension))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName)]
    currFps = int(videoSettings['fps'])
    noRectangles = len(rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
    noCircles = len(circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
    noPolygons = len(polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
    rectangleTimes, rectangleEntries = ([[0] * len(animalBodypartList) for i in range(noRectangles)] , [[0] * len(animalBodypartList) for i in range(noRectangles)])
    circleTimes, circleEntries = ([[0] * len(animalBodypartList) for i in range(noCircles)], [[0] * len(animalBodypartList) for i in range(noCircles)])
    polygonTime, polyGonEntries = ([[0] * len(animalBodypartList) for i in range(noPolygons)], [[0] * len(animalBodypartList) for i in range(noPolygons)])
    currFrameFolderOut = os.path.join(frames_dir_out, CurrentVideoName + '.avi')
    Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
    Circles = (circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
    Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
    rectangleEntryCheck = [[True] * len(animalBodypartList) for i in range(noRectangles)]
    circleEntryCheck = [[True] * len(animalBodypartList) for i in range(noCircles)]
    polygonEntryCheck = [[True] * len(animalBodypartList) for i in range(noPolygons)]
    currDfPath = os.path.join(csv_dir_in, CurrentVideoName + '.' + wfileType)
    currDf = read_df(currDfPath, wfileType)
    try:
        currDf = currDf.set_index('scorer')
    except KeyError:
        pass
    currDf = currDf.loc[:, ~currDf.columns.str.contains('^Unnamed')]
    currDf.columns = bplist
    try:
        currDf = currDf[columns2grab]
    except KeyError:
        print('ERROR: Make sure you have analyzed the videos before visualizing them. Click on the "Analyze ROI data" buttom first')
    writer = cv2.VideoWriter(currFrameFolderOut, fourcc, fps, (width*2, height))
    RectangleColors = [(255, 191, 0), (255, 248, 240), (255,144,30), (230,224,176), (160, 158, 95), (208,224,63), (240, 207,137), (245,147,245), (204,142,0), (229,223,176), (208,216,129)]
    CircleColors = [(122, 160, 255), (0, 69, 255), (34,34,178), (0,0,255), (128, 128, 240), (2, 56, 121), (21, 113, 239), (5, 150, 235), (2, 106, 253), (0, 191, 255), (98, 152, 247)]
    polygonColor = [(0, 255, 0), (87, 139, 46), (152,241,152), (127,255,0), (47, 107, 85), (91, 154, 91), (70, 234, 199), (20, 255, 57), (135, 171, 41), (192, 240, 208), (131,193, 157)]

    animalColors = []
    cmap = cm.get_cmap('Set1', noAnimals)
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        animalColors.append(rgb)
    currRow = 0

    currentPoints = np.empty((noAnimals, 2), dtype=int)


    while (cap.isOpened()):
        try:
            ret, img = cap.read()
            if ret == True:
                addSpacer = 2
                spacingScale = int(mySpaceScale / (myResolution / maxResDimension))
                borderImage = cv2.copyMakeBorder(img, 0,0,0,int(width), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                borderImageHeight, borderImageWidth = borderImage.shape[0], borderImage.shape[1]
                current_probability_list = []
                for animal in range(len(currentPoints)):
                    currentPoints[animal][0], currentPoints[animal][1] = currDf.at[currRow, animalBodypartList[animal][0]], currDf.at[currRow, animalBodypartList[animal][1]]
                    current_probability_list.append(currDf.at[currRow, animalBodypartList[animal][2]])
                    if current_probability_list[animal] > probability_threshold:
                        cv2.circle(borderImage, (currentPoints[animal][0], currentPoints[animal][1]), DrawScale, animalColors[animal], -1)
                        cv2.putText(borderImage, str(multiAnimalIDList[animal]), (currentPoints[animal][0], currentPoints[animal][1]), cv2.FONT_HERSHEY_TRIPLEX, textScale, animalColors[animal], 1)

                addSpacer += 1
                for rectangle in range(noRectangles):
                    topLeftX, topLeftY = (Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle])
                    bottomRightX, bottomRightY = (topLeftX + Rectangles['width'].iloc[rectangle], topLeftY + Rectangles['height'].iloc[rectangle])
                    rectangleName = Rectangles['Name'].iloc[rectangle]
                    cv2.rectangle(borderImage, (topLeftX, topLeftY), (bottomRightX, bottomRightY), RectangleColors[rectangle], DrawScale)
                    for animal in range(len(currentPoints)):
                        cv2.putText(borderImage, str(rectangleName) + ' ' + str(multiAnimalIDList[animal]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 1)
                        if ((((topLeftX-10) <= currentPoints[animal][0] <= (bottomRightX+10)) and ((topLeftY-10) <= currentPoints[animal][1] <= (bottomRightY+10)))) and (current_probability_list[animal] > probability_threshold):
                            rectangleTimes[rectangle][animal] = round((rectangleTimes[rectangle][animal] + (1 / currFps)), 2)
                            if rectangleEntryCheck[rectangle][animal] == True:
                                rectangleEntries[rectangle][animal] += 1
                                rectangleEntryCheck[rectangle][animal] = False
                        else:
                            rectangleEntryCheck[rectangle][animal] = True
                        cv2.putText(borderImage, str(rectangleTimes[rectangle][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 1)
                        addSpacer += 1
                        cv2.putText(borderImage, str(rectangleName) + ' ' + str(multiAnimalIDList[animal]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 1)
                        cv2.putText(borderImage, str(rectangleEntries[rectangle][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, RectangleColors[rectangle], 1)
                        addSpacer += 1

                for circle in range(noCircles):
                    circleName, centerX, centerY, radius = (Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle])
                    cv2.circle(borderImage, (centerX, centerY), radius,  CircleColors[circle], DrawScale)
                    for animal in range(len(currentPoints)):
                        cv2.putText(borderImage, str(circleName) + ' ' + str(multiAnimalIDList[animal]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale,  CircleColors[circle], 1)
                        euclidPxDistance = int(np.sqrt((currentPoints[animal][0] - centerX) ** 2 + (currentPoints[animal][1] - centerY) ** 2))
                        if (euclidPxDistance <= radius) and (current_probability_list[animal] > probability_threshold):
                            circleTimes[circle][animal] = round((circleTimes[circle][animal] + (1 / currFps)),2)
                            if circleEntryCheck[circle][animal] == True:
                                circleEntries[circle][animal] += 1
                                circleEntryCheck[circle][animal] = False
                        else:
                            circleEntryCheck[circle][animal] = True
                        cv2.putText(borderImage, str(circleTimes[circle][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, CircleColors[circle], 1)
                        addSpacer += 1
                        cv2.putText(borderImage, str(circleName) + ' ' + str(multiAnimalIDList[animal]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, CircleColors[circle], 1)
                        cv2.putText(borderImage, str(circleEntries[circle][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, CircleColors[circle], 1)
                        addSpacer += 1

                for polygon in range(noPolygons):
                    PolygonName, vertices = (Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon])
                    vertices = np.array(vertices, np.int32)
                    cv2.polylines(borderImage, [vertices], True, polygonColor[polygon], thickness=DrawScale)
                    for animal in range(len(currentPoints)):
                        pointList = []
                        cv2.putText(borderImage, str(PolygonName) + ' ' + str(multiAnimalIDList[animal]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 1)
                        for i in vertices:
                            point = geometry.Point(i)
                            pointList.append(point)
                        polyGon = geometry.Polygon([[p.x, p.y] for p in pointList])
                        CurrPoint = Point(int(currentPoints[animal][0]), int(currentPoints[animal][1]))
                        polyGonStatus = (polyGon.contains(CurrPoint))
                        if (polyGonStatus == True) and (current_probability_list[animal] > probability_threshold):
                            polygonTime[polygon][animal] = round((polygonTime[polygon][animal] + (1 / currFps)), 2)
                            if polygonEntryCheck[polygon][animal] == True:
                                polyGonEntries[polygon][animal] += 1
                                polygonEntryCheck[polygon][animal] = False
                        else:
                            polygonEntryCheck[polygon][animal] = True
                        cv2.putText(borderImage, str(polygonTime[polygon][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 1)
                        addSpacer += 1
                        cv2.putText(borderImage, str(PolygonName) + ' ' + str(multiAnimalIDList[animal]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 1)
                        cv2.putText(borderImage, str(polyGonEntries[polygon][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, polygonColor[polygon], 1)
                        addSpacer += 1
                borderImage = np.uint8(borderImage)
                writer.write(borderImage)
                # cv2.imshow('Window', borderImage)
                # key = cv2.waitKey(3000)
                # if key == 27:
                #     cv2.destroyAllWindows()

                # break
                currRow += 1
                print('Frame: ' + str(currRow) + '/' + str(frames))
            if img is None:
                print('Video ' + str(CurrentVideoName) + ' saved.')
                cap.release()
                break

        except IndexError:
            writer.release()
            print('NOTE: index error. Last frames of the video may be missing.')
            break

    print('ROI videos generated in "project_folder/frames/ROI_analysis"')