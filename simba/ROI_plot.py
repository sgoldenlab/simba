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
from simba.drop_bp_cords import get_fn_ext
from simba.features_scripts.unit_tests import *
from simba.misc_tools import check_multi_animal_status

# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\project_config.ini"
# CurrentVideo = "Video10.mp4"

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

    multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)

    logFolderPath = os.path.join(projectPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'ROI_analysis')
    if not os.path.exists(frames_dir_out):
        os.makedirs(frames_dir_out)
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    try:
        rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
        circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
        polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
    except FileNotFoundError:
        print('No ROIs found: please define ROIs using the left-most menu in the SIMBA ROI tab.')

    CurrentVideoPath = os.path.join(projectPath, 'videos', CurrentVideo)
    cap = cv2.VideoCapture(CurrentVideoPath)
    _, CurrentVideoName, videoFileType = get_fn_ext(CurrentVideoPath)
    width, height, frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mySpaceScale, myRadius, myResolution, myFontScale = 25, 10, 1500, 0.8
    maxResDimension = max(width, height)
    DrawScale = int(myRadius / (myResolution / maxResDimension))
    textScale = float(myFontScale / (myResolution / maxResDimension))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoSettings, pix_per_mm, currFps = read_video_info(vidinfDf, CurrentVideoName)
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
        currDf = currDf.set_index('scorer').reset_index(drop=True)
    except KeyError:
        currDf.reset_index(drop=True, inplace=True)
    currDf = currDf.loc[:, ~currDf.columns.str.contains('^Unnamed')]
    currDf.columns = bplist
    try:
        currDf = currDf[columns2grab]
    except KeyError:
        print('ERROR: Make sure you have analyzed the videos before visualizing them. Click on the "Analyze ROI data" buttom first')
    writer = cv2.VideoWriter(currFrameFolderOut, fourcc, currFps, (width*2, height))
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
                    cv2.rectangle(borderImage, (topLeftX, topLeftY), (bottomRightX, bottomRightY), Rectangles['Color BGR'].iloc[rectangle], Rectangles['Thickness'].iloc[rectangle])
                    for animal in range(len(currentPoints)):
                        cv2.putText(borderImage, str(rectangleName) + ' ' + str(multiAnimalIDList[animal]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles['Color BGR'].iloc[rectangle], 1)
                        if ((((topLeftX) <= currentPoints[animal][0] <= (bottomRightX)) and ((topLeftY) <= currentPoints[animal][1] <= (bottomRightY)))) and (current_probability_list[animal] > probability_threshold):
                            rectangleTimes[rectangle][animal] = rectangleTimes[rectangle][animal] + (1 / currFps)
                            if rectangleEntryCheck[rectangle][animal] == True:
                                rectangleEntries[rectangle][animal] += 1
                                rectangleEntryCheck[rectangle][animal] = False
                        else:
                            rectangleEntryCheck[rectangle][animal] = True
                        rounded_rec_time = round(rectangleTimes[rectangle][animal], 2)
                        cv2.putText(borderImage, str(rounded_rec_time), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles['Color BGR'].iloc[rectangle], 1)
                        addSpacer += 1
                        cv2.putText(borderImage, str(rectangleName) + ' ' + str(multiAnimalIDList[animal]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles['Color BGR'].iloc[rectangle], 1)
                        cv2.putText(borderImage, str(rectangleEntries[rectangle][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles['Color BGR'].iloc[rectangle], 1)
                        addSpacer += 1

                for circle in range(noCircles):
                    circleName, centerX, centerY, radius = (Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle])
                    cv2.circle(borderImage, (centerX, centerY), radius,  Circles['Color BGR'].iloc[circle], Circles['Thickness'].iloc[circle])
                    for animal in range(len(currentPoints)):
                        cv2.putText(borderImage, str(circleName) + ' ' + str(multiAnimalIDList[animal]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale,  Circles['Color BGR'].iloc[circle], 1)
                        euclidPxDistance = int(np.sqrt((currentPoints[animal][0] - centerX) ** 2 + (currentPoints[animal][1] - centerY) ** 2))
                        if (euclidPxDistance <= radius) and (current_probability_list[animal] > probability_threshold):
                            circleTimes[circle][animal] = circleTimes[circle][animal] + (1 / currFps)
                            if circleEntryCheck[circle][animal] == True:
                                circleEntries[circle][animal] += 1
                                circleEntryCheck[circle][animal] = False
                        else:
                            circleEntryCheck[circle][animal] = True
                        rounded_circ_time = round(circleTimes[circle][animal], 2)
                        cv2.putText(borderImage, str(rounded_circ_time), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles['Color BGR'].iloc[circle], 1)
                        addSpacer += 1
                        cv2.putText(borderImage, str(circleName) + ' ' + str(multiAnimalIDList[animal]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles['Color BGR'].iloc[circle], 1)
                        cv2.putText(borderImage, str(circleEntries[circle][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles['Color BGR'].iloc[circle], 1)
                        addSpacer += 1

                for polygon in range(noPolygons):
                    PolygonName, vertices = (Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon])
                    vertices = np.array(vertices, np.int32)
                    cv2.polylines(borderImage, [vertices], True, Polygons['Color BGR'].iloc[polygon], thickness=Polygons['Thickness'].iloc[polygon])
                    for animal in range(len(currentPoints)):
                        pointList = []
                        cv2.putText(borderImage, str(PolygonName) + ' ' + str(multiAnimalIDList[animal]) + ' timer:', ((width + 5), (height - (height + 10) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons['Color BGR'].iloc[polygon], 1)
                        for i in vertices:
                            point = geometry.Point(i)
                            pointList.append(point)
                        polyGon = geometry.Polygon([[p.x, p.y] for p in pointList])
                        CurrPoint = Point(int(currentPoints[animal][0]), int(currentPoints[animal][1]))
                        polyGonStatus = (polyGon.contains(CurrPoint))
                        if (polyGonStatus == True) and (current_probability_list[animal] > probability_threshold):
                            polygonTime[polygon][animal] = polygonTime[polygon][animal] + (1 / currFps)
                            if polygonEntryCheck[polygon][animal] == True:
                                polyGonEntries[polygon][animal] += 1
                                polygonEntryCheck[polygon][animal] = False
                        else:
                            polygonEntryCheck[polygon][animal] = True
                        rounded_poly_time = round(polygonTime[polygon][animal], 2)
                        cv2.putText(borderImage, str(rounded_poly_time), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale * addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons['Color BGR'].iloc[polygon], 1)
                        addSpacer += 1
                        cv2.putText(borderImage, str(PolygonName) + ' ' + str(multiAnimalIDList[animal]) + ' entries:', ((width + 5), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons['Color BGR'].iloc[polygon], 1)
                        cv2.putText(borderImage, str(polyGonEntries[polygon][animal]), ((int(borderImageWidth-(borderImageWidth/8))), (height - (height + 10) + spacingScale*addSpacer)), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons['Color BGR'].iloc[polygon], 1)
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

        except Exception as e:
            writer.release()
            print(e.args)
            print('NOTE: index error / keyerror. Some frames of the video may be missing. Make sure you are running latest version of SimBA with pip install simba-uw-tf-dev')
            break

    print('ROI videos generated in "project_folder/frames/ROI_analysis"')