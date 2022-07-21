from __future__ import division
import os
import pandas as pd
import numpy as np
from configparser import ConfigParser, NoOptionError
import glob
import cv2
from pylab import *
from shapely.geometry import Point
from shapely import geometry
from simba.rw_dfs import *
from simba.drop_bp_cords import get_fn_ext
from copy import deepcopy
from simba.misc_tools import check_directionality_viable, check_multi_animal_status, line_length, add_missing_ROI_cols
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary, createColorListofList
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info



# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini"
# videoFileName = "Together_1.avi"

def ROItoFeaturesViz(inifile, videoFileName):

    config = ConfigParser()
    config.read(inifile)

    try:
        noAnimals = config.getint('ROI settings', 'no_of_animals')
    except NoOptionError:
        noAnimals = config.getint('General settings', 'animal_no')

    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    logFolderPath = os.path.join(projectPath, 'logs')
    vidinfDf = read_video_info_csv(os.path.join(projectPath, 'logs', 'video_info.csv'))
    multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)
    Xcols, Ycols, Pcols = getBpNames(inifile)
    cMapSize = int(len(Xcols) + 1)
    colorListofList = createColorListofList(noAnimals, cMapSize)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, [], colorListofList)
    animal_names = list(animalBpDict.keys())

    tracked_animal_bps = []
    for animal in range(noAnimals):
        bp = config.get('ROI settings', 'animal_{}_bp'.format(str(animal+1)))
        if len(bp) == 0:
            print('ERROR: Please analyze ROI data before visualizing it. no Body-part found in config file [ROI settings][animal_N_bp]')
        tracked_animal_bps.append([bp + '_x', bp + '_y'])

    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')

    rectanglesInfo = add_missing_ROI_cols(rectanglesInfo)
    circleInfo = add_missing_ROI_cols(circleInfo)
    polygonInfo = add_missing_ROI_cols(polygonInfo)

    videoFilePath = os.path.join(projectPath, 'videos', videoFileName)
    _, videoBaseName, videoFileType = get_fn_ext(videoFilePath)
    print('Analyzing ROI features for ' + videoBaseName + '...')
    Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(videoBaseName)])
    Circles = (circleInfo.loc[circleInfo['Video'] == str(videoBaseName)])
    Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(videoBaseName)])

    currVideoSettings, currPixPerMM, fps = read_video_info(vidinfDf, videoBaseName)
    currDfPath = os.path.join(csv_dir_in, videoBaseName + '.' + wfileType)
    currDf = read_df(currDfPath, wfileType)
    currDf = currDf.apply(pd.to_numeric).reset_index(drop=True).fillna(0)

    directionalitySetting, NoseCoords, EarLeftCoords, EarRightCoords = check_directionality_viable(animalBpDict)

    print('Using directionality : ' + str(directionalitySetting))
    print('Calculating measurements - hang tight this can take a while - we are real scienstist not the "data" kind - say no to big-O! :)!')

    #### FEATURES COLUMNS AND NUMPY ARRAYS WITH COORDINATES######
    added_facing_coordinate_columns = []
    rectangleFeatures = np.array([0]*5)
    Rectangle_col_inside_value, Rectangle_col_distance, Rectangle_col_facing = [], [], []
    for rectangle in range(len(Rectangles)):
        for animal in range(len(animal_names)):
            ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + animal_names[animal] + '_in_zone')
            Rectangle_col_inside_value.append(ROI_col_name)
            currDf[ROI_col_name] = 0
            ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + animal_names[animal] + '_distance')
            currDf[ROI_col_name] = 0
            Rectangle_col_distance.append(ROI_col_name)
            if directionalitySetting:
                ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + animal_names[animal] + '_facing')
                currDf[ROI_col_name] = 0
                Rectangle_col_facing.append(ROI_col_name)
        rectangleArray = np.array([Rectangles['Name'].iloc[rectangle], Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle], Rectangles['topLeftX'].iloc[rectangle] + Rectangles['width'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle] + Rectangles['height'].iloc[rectangle]])
        rectangleFeatures = np.vstack((rectangleFeatures, rectangleArray))
    rectangleFeatures = np.delete(rectangleFeatures, 0, 0)

    circleFeatures = np.array([0] * 4)
    circle_col_inside_value, circle_col_distance, circle_col_facing = [], [], []
    for circle in range(len(Circles)):
        for animal in range(len(animal_names)):
            ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + animal_names[animal] + '_in_zone')
            circle_col_inside_value.append(ROI_col_name)
            currDf[ROI_col_name] = 0
            ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + animal_names[animal] + '_distance')
            currDf[ROI_col_name] = 0
            circle_col_distance.append(ROI_col_name)
            if directionalitySetting:
                ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + animal_names[animal] + '_facing')
                currDf[ROI_col_name] = 0
                circle_col_facing.append(ROI_col_name)
        circleArray = np.array([Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle]])
        circleFeatures = np.vstack((circleFeatures, circleArray))
    circleFeatures = np.delete(circleFeatures, 0, 0)

    polygonFeatures = np.array([0] * 2)
    polygon_col_inside_value, polygon_col_distance, polygon_col_facing = [], [], []
    for polygon in range(len(Polygons)):
        for animal in range(len(animal_names)):
            ROI_col_name = str(Polygons['Name'].iloc[polygon] + '_' + animal_names[animal] + '_in_zone')
            polygon_col_inside_value.append(ROI_col_name)
            currDf[ROI_col_name] = 0
            ROI_col_name = str(Polygons['Name'].iloc[polygon] + '_' + animal_names[animal] + '_distance')
            currDf[ROI_col_name] = 0
            polygon_col_distance.append(ROI_col_name)
            if directionalitySetting:
                ROI_col_name = str(Polygons['Name'].iloc[polygon] + '_' + animal_names[animal] + '_facing')
                currDf[ROI_col_name] = 0
                polygon_col_facing.append(ROI_col_name)
        polygonArray = np.array([Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon]])
        polygonFeatures = np.vstack((polygonFeatures, polygonArray))
    polygonFeatures = np.delete(polygonFeatures, 0, 0)


    ### CALUCLATE BOOLEAN, IF ANIMAL IS IN RECTANGLES AND CIRCLES AND POLYGONS
    for index, row in currDf.iterrows():
        loop = 0
        for rectangle in range(len(Rectangles)):
            for animal in range(len(tracked_animal_bps)):
                currROIColName = Rectangle_col_inside_value[loop]
                loop+=1
                if ((((int(rectangleFeatures[rectangle, 1]) - 10) <= row[tracked_animal_bps[animal][0]] <= (int(rectangleFeatures[rectangle, 3]) + 10))) and (((int(rectangleFeatures[rectangle, 2]) - 10) <= row[tracked_animal_bps[animal][1]] <= (int(rectangleFeatures[rectangle, 4]) + 10)))):
                    currDf.loc[index, currROIColName] = 1
        for column in Rectangle_col_inside_value:
            colName1 = str(column) + '_cumulative_time'
            currDf[colName1] = currDf[column].cumsum() * float(1/fps)
            colName2 = str(column) + '_cumulative_percent'
            currDf[colName2] = currDf[colName1]/currDf.index
        loop = 0

        for circle in range(len(Circles)):
            for animal in range(len(tracked_animal_bps)):
                currROIColName = circle_col_inside_value[loop]
                loop+=1
                euclidPxDistance = np.sqrt((int(row[tracked_animal_bps[animal][0]]) - int(circleFeatures[circle, 1])) ** 2 + ((int(row[tracked_animal_bps[animal][1]]) - int(circleFeatures[circle, 2])) ** 2))
                if euclidPxDistance <= int(circleFeatures[circle, 3]):
                    currDf.loc[index, currROIColName] = 1
        for column in circle_col_inside_value:
            colName1 = str(column) + '_cumulative_time'
            currDf[colName1] = currDf[column].cumsum() * float(1 / fps)
            colName2 = str(column) + '_cumulative_percent'
            currDf[colName2] = currDf[colName1] / currDf.index
        loop = 0

        for polygon in range(len(Polygons)):
            CurrVertices = polygonFeatures[polygon, 1]
            CurrVertices = np.array(CurrVertices, np.int32)
            pointList = []
            for i in CurrVertices:
                point = geometry.Point(i)
                pointList.append(point)
            polyGon = geometry.Polygon([[p.x, p.y] for p in pointList])
            for animal in range(len(tracked_animal_bps)):
                CurrPoint = Point(int(row[tracked_animal_bps[animal][0]]), int(row[tracked_animal_bps[animal][1]]))
                currROIColName = polygon_col_inside_value[loop]
                polyGonStatus = (polyGon.contains(CurrPoint))
                if polyGonStatus == True:
                    currDf.loc[index, currROIColName] = 1


    ### CALUCLATE DISTANCE TO CENTER OF EACH RECTANGLE
    for index, row in currDf.iterrows():
        loop = 0
        for rectangle in range(len(Rectangles)):
            currRecCenter = [(int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ]
            for animal in range(len(tracked_animal_bps)):
                currROIColName = Rectangle_col_distance[loop]
                currDf.loc[index, currROIColName] = (np.sqrt((row[tracked_animal_bps[animal][0]] - currRecCenter[0]) ** 2 + (row[tracked_animal_bps[animal][1]] - currRecCenter[1]) ** 2)) / currPixPerMM
                loop += 1
        loop = 0
        for circle in range(len(Circles)):
            currCircleCenterX, currCircleCenterY = (int(circleFeatures[circle, 1]), int(circleFeatures[circle, 2]))
            for animal in range(len(tracked_animal_bps)):
                currROIColName = circle_col_distance[loop]
                currDf.loc[index, currROIColName] = (np.sqrt((int(row[tracked_animal_bps[animal][0]]) - currCircleCenterX) ** 2 + (int(row[tracked_animal_bps[animal][1]]) - currCircleCenterY) ** 2)) / currPixPerMM
                loop += 1
        loop = 0

        polygonCenterCord = np.array([0] * 2)
        for polygon in range(len(Polygons)):
            CurrVertices = polygonFeatures[polygon, 1]
            CurrVertices = np.array(CurrVertices, np.int32)
            pointList = []
            for i in CurrVertices:
                point = geometry.Point(i)
                pointList.append(point)
            polyGon = geometry.Polygon([[p.x, p.y] for p in pointList])
            polyGonCenter = polyGon.centroid.wkt
            polyGonCenter = polyGonCenter.replace("POINT", '')
            polyGonCenter = polyGonCenter.replace("(", '')
            polyGonCenter = polyGonCenter.replace(")", '')
            polyGonCenter = polyGonCenter.split(" ", 3)
            polyGonCenter = polyGonCenter[1:3]
            polyGonCenter = [float(i) for i in polyGonCenter]
            polyGonCenterX, polyGonCenterY = polyGonCenter[0], polyGonCenter[1]
            polyArray = np.array([polyGonCenterX, polyGonCenterY])
            polygonCenterCord = np.vstack((polygonCenterCord, polyArray))
            for animal in range(len(tracked_animal_bps)):
                currROIColName = polygon_col_distance[loop]
                currDf.loc[index, currROIColName] = (np.sqrt((int(row[tracked_animal_bps[animal][0]]) - polyGonCenterX) ** 2 + (int(row[tracked_animal_bps[animal][1]]) - polyGonCenterY) ** 2)) / currPixPerMM
                loop += 1
        polygonCenterCord = np.delete(polygonCenterCord, 0, 0)


    ### CALCULATE IF ANIMAL IS DIRECTING TOWARDS THE CENTER OF THE RECTANGLES AND CIRCLES
    if directionalitySetting:
        for index, row in currDf.iterrows():
            loop = 0
            for rectangle in range(len(Rectangles)):
                for animal in range(len(tracked_animal_bps)):
                    p, q, n, m, coord = ([] for i in range(5))
                    currROIColName = Rectangle_col_facing[loop]
                    p.extend((row[EarLeftCoords[animal][0]], row[EarLeftCoords[animal][1]]))
                    q.extend((row[EarRightCoords[animal][0]], row[EarRightCoords[animal][1]]))
                    n.extend((row[NoseCoords[animal][0]], row[NoseCoords[animal][1]]))
                    m.extend(((int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ))
                    center_facing_check = line_length(p, q, n, m, coord)
                    if center_facing_check[0] == True:
                        currDf.loc[index, currROIColName] = 1
                        x0 = min(center_facing_check[1][0], row[NoseCoords[animal][0]])
                        y0 = min(center_facing_check[1][1], row[NoseCoords[animal][1]])
                        deltaX = abs((center_facing_check[1][0] - row[NoseCoords[animal][0]]) / 2)
                        deltaY = abs((center_facing_check[1][1] - row[NoseCoords[animal][1]]) / 2)
                        Xmid, Ymid = int(x0 + deltaX), int(y0 + deltaY)
                        currDf.loc[index, currROIColName + '_x'] = Xmid
                        currDf.loc[index, currROIColName + '_y'] = Ymid
                    else:
                        currDf.loc[index, currROIColName + '_x'] = 0
                        currDf.loc[index, currROIColName + '_y'] = 0
                    loop += 1
            loop = 0

            for circle in range(len(Circles)):
                for animal in range(len(tracked_animal_bps)):
                    p, q, n, m, coord = ([] for i in range(5))
                    currROIColName = circle_col_facing[loop]
                    p.extend((row[EarLeftCoords[animal][0]], row[EarLeftCoords[animal][1]]))
                    q.extend((row[EarRightCoords[animal][0]], row[EarRightCoords[animal][1]]))
                    n.extend((row[NoseCoords[animal][0]], row[NoseCoords[animal][1]]))
                    m.extend((int(circleFeatures[circle, 1]), int(circleFeatures[circle, 2])))
                    center_facing_check = line_length(p, q, n, m, coord)
                    if center_facing_check[0] == True:
                        currDf.loc[index, currROIColName] = 1
                        x0 = min(center_facing_check[1][0], row[NoseCoords[animal][0]])
                        y0 = min(center_facing_check[1][1], row[NoseCoords[animal][1]])
                        deltaX = abs((center_facing_check[1][0] - row[NoseCoords[animal][0]]) / 2)
                        deltaY = abs((center_facing_check[1][1] - row[NoseCoords[animal][1]]) / 2)
                        Xmid, Ymid = int(x0 + deltaX), int(y0 + deltaY)
                        currDf.loc[index, currROIColName + '_x'] = Xmid
                        currDf.loc[index, currROIColName + '_y'] = Ymid
                    else:
                        currDf.loc[index, currROIColName + '_x'] = 0
                        currDf.loc[index, currROIColName + '_y'] = 0
                    loop += 1
            loop = 0

            for polygon in range(len(Polygons)):
                for animal in range(len(tracked_animal_bps)):
                    p, q, n, m, coord = ([] for i in range(5))
                    currROIColName = polygon_col_facing[loop]
                    p.extend((row[EarLeftCoords[animal][0]], row[EarLeftCoords[animal][0]]))
                    q.extend((row[EarRightCoords[animal][0]], row[EarRightCoords[animal][1]]))
                    n.extend((row[NoseCoords[animal][0]], row[NoseCoords[animal][1]]))
                    m.extend((int(polygonCenterCord[polygon][0]), int(polygonCenterCord[polygon][1])))
                    center_facing_check = line_length(p, q, n, m, coord)
                    if center_facing_check[0] == True:
                        currDf.loc[index, currROIColName] = 1
                        x0 = min(center_facing_check[1][0], row[NoseCoords[animal][0]])
                        y0 = min(center_facing_check[1][1], row[NoseCoords[animal][1]])
                        deltaX = abs((center_facing_check[1][0] - row[NoseCoords[animal][0]]) / 2)
                        deltaY = abs((center_facing_check[1][1] - row[NoseCoords[animal][1]]) / 2)
                        Xmid, Ymid = int(x0 + deltaX), int(y0 + deltaY)
                        currDf.loc[index, currROIColName + '_x'] = Xmid
                        currDf.loc[index, currROIColName + '_y'] = Ymid
                    else:
                        currDf.loc[index, currROIColName + '_x'] = 0
                        currDf.loc[index, currROIColName + '_y'] = 0
                    loop += 1


    currDf = currDf.fillna(0).replace(np.inf, 0)
    print('ROI features calculated for ' + videoBaseName + '.')
    print('Visualizing ROI features for ' + videoBaseName + '...')
    outputFolderName = os.path.join(projectPath, 'frames', 'output', 'ROI_features')
    currVideoPath = videoFilePath
    if not os.path.exists(outputFolderName):
        os.makedirs(outputFolderName)
    outputfilename = os.path.join(outputFolderName, videoBaseName + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(currVideoPath)
    vid_input_width, vid_input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### small image correction
    smallImageCorrectionValue = 1
    if vid_input_width < 400:
        smallImageCorrectionValue = 2

    sideImage = np.zeros((vid_input_height, vid_input_width*smallImageCorrectionValue,3), np.uint8)
    writer = cv2.VideoWriter(outputfilename, fourcc, int(fps), (int(vid_input_width + sideImage.shape[1]), vid_input_height))
    mySpaceScaleY, mySpaceScaleX, myRadius, myResolution, myFontScale = 40, 700, 20, 1500, 1
    maxResDimension = max(vid_input_width, vid_input_height)
    textScale = float(myFontScale / (myResolution / maxResDimension))
    DrawScale = int(myRadius / (myResolution / maxResDimension))
    YspacingScale = int(mySpaceScaleY / (myResolution / maxResDimension))
    XspacingScale = int(mySpaceScaleX / (myResolution / maxResDimension))
    loop, colorList = 0, []


    variableList = ['in_zone', 'distance']
    if directionalitySetting:
        variableList.append('facing')

    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            try:
                sideImage = np.zeros((vid_input_height, vid_input_width*smallImageCorrectionValue, 3), np.uint8)
                for animal in range(noAnimals):
                    bp_x_name, bp_y_name = tracked_animal_bps[animal][0], tracked_animal_bps[animal][1]
                    point = (int(currDf.loc[currDf.index[loop], bp_x_name]), int(currDf.loc[currDf.index[loop], bp_y_name]))
                    color = tuple(animalBpDict[animal_names[animal]]['colors'][0])
                    cv2.circle(img, (point[0], point[1]), DrawScale, color, -1)
                for rectangle in range(len(Rectangles)):
                    topLeftX, topLeftY, bottomRightX, bottomRightY = (int(rectangleFeatures[rectangle, 1]), int(rectangleFeatures[rectangle, 2]), int(rectangleFeatures[rectangle, 3]), int(rectangleFeatures[rectangle, 4]))
                    cv2.rectangle(img, (topLeftX, topLeftY), (bottomRightX, bottomRightY), Rectangles.at[rectangle, 'Color BGR'], Rectangles.at[rectangle, 'Thickness'])
                for circles in range(len(Circles)):
                    centerX, centerY, radius = (int(circleFeatures[circles, 1]), int(circleFeatures[circles, 2]), int(circleFeatures[circles, 3]))
                    cv2.circle(img, (centerX, centerY), radius, Circles.at[circles, 'Color BGR'], Circles.at[circles, 'Thickness'])
                for polygons in range(len(Polygons)):
                    inputVertices = (polygonFeatures[polygons, 1])
                    vertices = np.array(inputVertices, np.int32)
                    cv2.polylines(img, [vertices], True, Polygons.at[polygons, 'Color BGR'], thickness=Polygons.at[polygons, 'Thickness'])
                startX, startY, xprintAdd, yprintAdd = 10, 30, 0, 0

                for rec in range(len(Rectangles)):
                    CurrRecName = rectangleFeatures[rec, 0]
                    topLeftX, topLeftY, bottomRightX, bottomRightY = (int(rectangleFeatures[rec, 1]), int(rectangleFeatures[rec, 2]), int(rectangleFeatures[rec, 3]), int(rectangleFeatures[rec, 4]))
                    rectangleCentroid_x, rectangleCentroid_y = int((bottomRightX - topLeftX)/2 + topLeftX), int((bottomRightY - topLeftY) / 2 + topLeftY)
                    for CurrAnimalName in animal_names:
                        for variable in variableList:
                            columnName = CurrRecName + '_' + CurrAnimalName + '_' + variable
                            rectangleStatus = currDf.loc[loop, columnName]
                            cv2.putText(sideImage, str(columnName), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles.at[rec, 'Color BGR'], 2)
                            xprintAdd += XspacingScale
                            if (variable == 'in_zone'):
                                if rectangleStatus == 1:
                                    cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles.at[rec, 'Color BGR'], 2)
                                if rectangleStatus == 0:
                                    cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles.at[rec, 'Color BGR'], 2)
                            if (variable == 'facing'):
                                rectangleStatus_x, rectangleStatus_y = currDf.loc[loop, columnName + '_x'], currDf.loc[loop, columnName + '_y']
                                if rectangleStatus == 1:
                                    cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles.at[rec, 'Color BGR'], 2)
                                    cv2.circle(img, (int(rectangleCentroid_x), int(rectangleCentroid_y)), DrawScale, (0, 255, 0), -1)
                                    cv2.line(img, (int(rectangleStatus_x), int(rectangleStatus_y)), (int(rectangleCentroid_x), int(rectangleCentroid_y)), (0, 255, 0), 2)
                                if rectangleStatus == 0:
                                    cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles.at[rec, 'Color BGR'], 2)
                            if (variable == 'distance'):
                                rectangleStatus = round(rectangleStatus / 10, 2)
                                cv2.putText(sideImage, str(rectangleStatus), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Rectangles.at[rec, 'Color BGR'], 2)
                            yprintAdd = yprintAdd + YspacingScale
                            xprintAdd = 0

                for circ in range(len(Circles)):
                    CurrCircName = circleFeatures[circ, 0]
                    centerX, centerY, radius = (int(circleFeatures[circ, 1]), int(circleFeatures[circ, 2]), int(circleFeatures[circ, 3]))
                    for CurrAnimalName in animal_names:
                        for variable in variableList:
                            columnName = CurrCircName + '_' + CurrAnimalName + '_' + variable
                            circStatus = currDf.loc[loop, columnName]
                            cv2.putText(sideImage, str(columnName), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles.at[circ, 'Color BGR'], 2)
                            xprintAdd += XspacingScale
                            if (variable == 'in_zone'):
                                if circStatus == 1:
                                    cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles.at[circ, 'Color BGR'], 2)
                                if circStatus == 0:
                                    cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles.at[circ, 'Color BGR'], 2)
                            if (variable == 'facing'):
                                circleStatus_x, circleStatus_y = currDf.loc[loop, columnName + '_x'], currDf.loc[loop, columnName + '_y']
                                if circStatus == 1:
                                    cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles.at[circ, 'Color BGR'], 2)
                                    cv2.circle(img, (int(centerX), int(centerY)), DrawScale, (0, 255, 0), -1)
                                    cv2.line(img, (int(circleStatus_x), int(circleStatus_y)),(int(centerX), int(centerY)), (0, 255, 0), 2)
                                if circStatus == 0:
                                    cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles.at[circ, 'Color BGR'], 2)
                            if (variable == 'distance'):
                                circStatus = round(circStatus / 10, 2)
                                cv2.putText(sideImage, str(circStatus), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Circles.at[circ, 'Color BGR'], 2)
                            yprintAdd = yprintAdd + YspacingScale
                            xprintAdd = 0

                for poly in range(len(Polygons)):
                    CurrPolyName = polygonFeatures[poly, 0]
                    for CurrAnimalName in animal_names:
                        for variable in variableList:
                            columnName = CurrPolyName + '_' + CurrAnimalName + '_' + variable
                            PolyStatus = currDf.loc[loop, columnName]
                            cv2.putText(sideImage, str(columnName), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons.at[poly, 'Color BGR'], 2)
                            xprintAdd += XspacingScale
                            if (variable == 'in_zone'):
                                if PolyStatus == 1:
                                    cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons.at[poly, 'Color BGR'], 2)
                                if PolyStatus == 0:
                                    cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons.at[poly, 'Color BGR'], 2)
                            if (variable == 'distance'):
                                PolyStatus = round(PolyStatus / 10, 2)
                                cv2.putText(sideImage, str(PolyStatus), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons.at[poly, 'Color BGR'], 2)
                            if (variable == 'facing'):
                                polyStatus_x, polyStatus_y = currDf.loc[loop, columnName + '_x'], currDf.loc[loop, columnName + '_y']
                                if PolyStatus == 1:
                                    cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons.at[poly, 'Color BGR'], 2)
                                    cv2.circle(img, (int(polygonCenterCord[poly][0]), int(polygonCenterCord[poly][1])), DrawScale, (0, 255, 0), -1)
                                    cv2.line(img, (int(polyStatus_x), int(polyStatus_y)), (int(polygonCenterCord[poly][0]), int(polygonCenterCord[poly][1])), (0, 255, 0), 2)
                                if PolyStatus == 0:
                                    cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, Polygons.at[poly, 'Color BGR'], 2)

                            yprintAdd = yprintAdd + YspacingScale
                            xprintAdd = 0


                imageConcat = np.concatenate((img, sideImage), axis=1)
                imageConcat = np.uint8(imageConcat)
                # cv2.imshow('image', imageConcat)
                # key = cv2.waitKey(3000)  # pauses for 3 seconds before fetching next image
                # if key == 27:  # if ESC is pressed, exit loop
                #     cv2.destroyAllWindows()
                #     break
                writer.write(imageConcat)
                print('Image ' + str(loop + 1) + ' / ' + str(len(currDf)))
                loop += 1

            except IndexError as e:
                print('IndexError: Video terminated after {} frames'.format(str(loop)))
                print('Video ' + str(videoBaseName) + ' saved in project_folder/frames/output/ROI_features')
                cap.release()
                writer.release()
                print(e.args)
                break

        if img is None:
            print('Video ' + str(videoBaseName) + ' saved in project_folder/frames/output/ROI_features')
            cap.release()
            writer.release()
            break

    print('All ROI videos generated in "project_folder/frames/output/ROI_features"')


# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini"
# videoFileName = "Together_1.avi"
#
# ROItoFeaturesViz(inifile, videoFileName)
