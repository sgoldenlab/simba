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


def ROItoFeaturesViz(inifile, videoFileName):
    config = ConfigParser()
    config.read(inifile)
    noAnimals = config.getint('ROI settings', 'no_of_animals')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    logFolderPath = os.path.join(projectPath, 'logs')
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)

    if noAnimals == 2:
        arrayIndex = 2
        bodyPartAnimal_1 = config.get('ROI settings', 'animal_1_bp')
        bodyPartAnimal_2 = config.get('ROI settings', 'animal_2_bp')
        trackedBodyParts = [bodyPartAnimal_1 + '_x', bodyPartAnimal_2 + '_x', bodyPartAnimal_1 + '_y', bodyPartAnimal_2 + '_y']
        trackedBodyPartNames = ['Animal_1_', 'Animal_2_']
    else:
        arrayIndex = 1
        bodyPartAnimal_1 = config.get('ROI settings', 'animal_1_bp')
        trackedBodyParts = [bodyPartAnimal_1 + '_x', bodyPartAnimal_1 + '_y']
        trackedBodyPartNames = ['Animal_1_']
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')


    def line_length(p, q, n, M, coord):
        Px = np.abs(p[0] - M[0])
        Py = np.abs(p[1] - M[1])
        Qx = np.abs(q[0] - M[0])
        Qy = np.abs(q[1] - M[1])
        Nx = np.abs(n[0] - M[0])
        Ny = np.abs(n[1] - M[1])
        Ph = np.sqrt(Px*Px + Py*Py)
        Qh = np.sqrt(Qx*Qx + Qy*Qy)
        Nh = np.sqrt(Nx*Nx + Ny*Ny)
        if (Nh < Ph and Nh < Qh and Qh < Ph):
            coord.extend((q[0], q[1]))
            return True, coord
        elif (Nh < Ph and Nh < Qh and Ph < Qh):
            coord.extend((p[0], p[1]))
            return True, coord
        else:
            return False, coord

    videoFilePath = os.path.join(projectPath, 'videos', videoFileName)
    _, videoBaseName, videoFileType = get_fn_ext(videoFilePath)
    print('Analyzing ROI features for ' + videoBaseName + '...')
    Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(videoBaseName)])
    Circles = (circleInfo.loc[circleInfo['Video'] == str(videoBaseName)])
    Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(videoBaseName)])
    currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == videoBaseName]
    currPixPerMM = float(currVideoSettings['pixels/mm'])
    fps = float(currVideoSettings['fps'])
    currDfPath = os.path.join(csv_dir_in, videoBaseName + '.' + wfileType)
    currDf = read_df(currDfPath, wfileType)
    currDf = currDf.fillna(0)
    currDf = currDf.apply(pd.to_numeric)
    currDf = currDf.reset_index(drop=True)
    currDf = currDf.loc[:, ~currDf.columns.str.contains('^Unnamed')]
    if arrayIndex == 2:
        NoseCoords = ['Nose_1_x', 'Nose_2_x' , 'Nose_1_y', 'Nose_2_y']
        EarLeftCoords = ['Ear_left_1_x', 'Ear_left_2_x' , 'Ear_left_1_y', 'Ear_left_2_y']
        EarRightCoords = ['Ear_right_1_x', 'Ear_right_2_x' , 'Ear_right_1_y', 'Ear_right_2_y']
        directionalityCordHeaders = NoseCoords + EarLeftCoords + EarRightCoords
        if set(directionalityCordHeaders).issubset(currDf.columns):
            directionalitySetting = 'yes'
        else:
            directionalitySetting = 'no'
    if arrayIndex == 1:
        NoseCoords = ['Nose_x', 'Nose_y']
        EarLeftCoords = ['Ear_left_x', 'Ear_left_y']
        EarRightCoords = ['Ear_right_x', 'Ear_right_y']
        directionalityCordHeaders = NoseCoords + EarLeftCoords + EarRightCoords
        if set(directionalityCordHeaders).issubset(currDf.columns):
            directionalitySetting = 'yes'
        else:
            directionalitySetting = 'no'
    print('Using directionality : ' + str(directionalitySetting))
    print('Calculating measurements - hang tight this can take a while - we are real scienstist not the "data" kind - say no to big-O! :)!')

    #### FEATURES COLUMNS AND NUMPY ARRAYS WITH COORDINATES######
    rectangleFeatures = np.array([0]*5)
    Rectangle_col_inside_value, Rectangle_col_distance, Rectangle_col_facing = [], [], []
    for rectangle in range(len(Rectangles)):
        for bodypart in range(len(trackedBodyPartNames)):
            ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + trackedBodyPartNames[bodypart] + 'in_zone')
            Rectangle_col_inside_value.append(ROI_col_name)
            currDf[ROI_col_name] = 0
            ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + trackedBodyPartNames[bodypart] + 'distance')
            currDf[ROI_col_name] = 0
            Rectangle_col_distance.append(ROI_col_name)
            if directionalitySetting == 'yes':
                ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + trackedBodyPartNames[bodypart] + 'facing')
                currDf[ROI_col_name] = 0
                Rectangle_col_facing.append(ROI_col_name)
        rectangleArray = np.array([Rectangles['Name'].iloc[rectangle], Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle], Rectangles['topLeftX'].iloc[rectangle] + Rectangles['width'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle] + Rectangles['height'].iloc[rectangle]])
        rectangleFeatures = np.vstack((rectangleFeatures, rectangleArray))
    rectangleFeatures = np.delete(rectangleFeatures, 0, 0)

    circleFeatures = np.array([0] * 4)
    circle_col_inside_value, circle_col_distance, circle_col_facing = [], [], []
    for circle in range(len(Circles)):
        for bodypart in range(len(trackedBodyPartNames)):
            ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + trackedBodyPartNames[bodypart] + 'in_zone')
            circle_col_inside_value.append(ROI_col_name)
            currDf[ROI_col_name] = 0
            ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + trackedBodyPartNames[bodypart] + 'distance')
            currDf[ROI_col_name] = 0
            circle_col_distance.append(ROI_col_name)
            if directionalitySetting == 'yes':
                ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + trackedBodyPartNames[bodypart] + 'facing')
                currDf[ROI_col_name] = 0
                circle_col_facing.append(ROI_col_name)
        circleArray = np.array([Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle]])
        circleFeatures = np.vstack((circleFeatures, circleArray))
    circleFeatures = np.delete(circleFeatures, 0, 0)

    polygonFeatures = np.array([0] * 2)
    polygon_col_inside_value, polygon_col_distance, polygon_col_facing = [], [], []
    for polygon in range(len(Polygons)):
        for bodypart in range(len(trackedBodyPartNames)):
            ROI_col_name = str(Polygons['Name'].iloc[polygon] + '_' + trackedBodyPartNames[bodypart] + 'in_zone')
            polygon_col_inside_value.append(ROI_col_name)
            currDf[ROI_col_name] = 0
            ROI_col_name = str(Polygons['Name'].iloc[polygon] + '_' + trackedBodyPartNames[bodypart] + 'distance')
            currDf[ROI_col_name] = 0
            polygon_col_distance.append(ROI_col_name)
            if directionalitySetting == 'yes':
                ROI_col_name = str(Polygons['Name'].iloc[polygon] + '_' + trackedBodyPartNames[bodypart] + 'facing')
                currDf[ROI_col_name] = 0
                polygon_col_facing.append(ROI_col_name)
        polygonArray = np.array([Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon]])
        polygonFeatures = np.vstack((polygonFeatures, polygonArray))
    polygonFeatures = np.delete(polygonFeatures, 0, 0)


    ### CALUCLATE BOOLEAN, IF ANIMAL IS IN RECTANGLES AND CIRCLES AND POLYGONS
    for index, row in currDf.iterrows():
        loop = 0
        for rectangle in range(len(Rectangles)):
            for bodyparts in range(len(trackedBodyPartNames)):
                currROIColName = Rectangle_col_inside_value[loop]
                loop+=1
                if ((((int(rectangleFeatures[rectangle, 1]) - 10) <= row[trackedBodyParts[bodyparts]] <= (int(rectangleFeatures[rectangle, 3]) + 10))) and (((int(rectangleFeatures[rectangle, 2]) - 10) <= row[trackedBodyParts[bodyparts + arrayIndex]] <= (int(rectangleFeatures[rectangle, 4]) + 10)))):
                    currDf.loc[index, currROIColName] = 1
        for column in Rectangle_col_inside_value:
            colName1 = str(column) + '_cumulative_time'
            currDf[colName1] = currDf[column].cumsum() * float(1/fps)
            colName2 = str(column) + '_cumulative_percent'
            currDf[colName2] = currDf[colName1]/currDf.index
        loop = 0
        for circle in range(len(Circles)):
            for bodyparts in range(len(trackedBodyPartNames)):
                currROIColName = circle_col_inside_value[loop]
                loop+=1
                euclidPxDistance = np.sqrt((int(row[trackedBodyParts[bodyparts]]) - int(circleFeatures[circle, 1])) ** 2 + ((int(row[trackedBodyParts[bodyparts+ arrayIndex]]) - int(circleFeatures[circle, 2])) ** 2))
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
            for bodyparts in range(len(trackedBodyPartNames)):
                CurrPoint = Point(int(row[trackedBodyParts[bodyparts]]), int(row[trackedBodyParts[bodyparts + arrayIndex]]))
                currROIColName = polygon_col_inside_value[loop]
                polyGonStatus = (polyGon.contains(CurrPoint))
                if polyGonStatus == True:
                    currDf.loc[index, currROIColName] = 1


    ### CALUCLATE DISTANCE TO CENTER OF EACH RECTANGLE
    for index, row in currDf.iterrows():
        loop = 0
        for rectangle in range(len(Rectangles)):
            currRecCenter = [(int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ]
            for bodyparts in range(len(trackedBodyPartNames)):
                currROIColName = Rectangle_col_distance[loop]
                currDf.loc[index, currROIColName] = (np.sqrt((row[trackedBodyParts[bodyparts]] - currRecCenter[0]) ** 2 + (row[trackedBodyParts[bodyparts+arrayIndex]] - currRecCenter[1]) ** 2)) / currPixPerMM
                loop += 1
        loop = 0
        for circle in range(len(Circles)):
            currCircleCenterX, currCircleCenterY = (int(circleFeatures[circle, 1]), int(circleFeatures[circle, 2]))
            for bodyparts in range(len(trackedBodyPartNames)):
                currROIColName = circle_col_distance[loop]
                currDf.loc[index, currROIColName] = (np.sqrt((int(row[trackedBodyParts[bodyparts]]) - currCircleCenterX) ** 2 + (int(row[trackedBodyParts[bodyparts+arrayIndex]]) - currCircleCenterY) ** 2)) / currPixPerMM
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
            for bodyparts in range(len(trackedBodyPartNames)):
                currROIColName = polygon_col_distance[loop]
                currDf.loc[index, currROIColName] = (np.sqrt((int(row[trackedBodyParts[bodyparts]]) - polyGonCenterX) ** 2 + (int(row[trackedBodyParts[bodyparts+arrayIndex]]) - polyGonCenterY) ** 2)) / currPixPerMM
                loop += 1
        polygonCenterCord = np.delete(polygonCenterCord, 0, 0)
    loop = 0

    ### CALCULATE IF ANIMAL IS DIRECTING TOWARDS THE CENTER OF THE RECTANGLES AND CIRCLES
    if directionalitySetting == 'yes':
        for index, row in currDf.iterrows():
            loop = 0
            for rectangle in range(len(Rectangles)):
                for bodyparts in range(len(trackedBodyPartNames)):
                    p, q, n, m, coord = ([] for i in range(5))
                    currROIColName = Rectangle_col_facing[loop]
                    currROIColName
                    p.extend((row[EarLeftCoords[bodyparts]], row[EarLeftCoords[bodyparts+arrayIndex]]))
                    q.extend((row[EarRightCoords[bodyparts]], row[EarRightCoords[bodyparts + arrayIndex]]))
                    n.extend((row[NoseCoords[bodyparts]], row[NoseCoords[bodyparts + arrayIndex]]))
                    m.extend(((int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ))
                    center_facing_check = line_length(p, q, n, m, coord)
                    if center_facing_check[0] == True:
                        currDf.loc[index, currROIColName] = 1
                        x0 = min(center_facing_check[1][0], row[NoseCoords[bodyparts]])
                        y0 = min(center_facing_check[1][1], row[NoseCoords[bodyparts + arrayIndex]])
                        deltaX = abs((center_facing_check[1][0] - row[NoseCoords[bodyparts]]) / 2)
                        deltaY = abs((center_facing_check[1][1] - row[NoseCoords[bodyparts + arrayIndex]]) / 2)
                        Xmid, Ymid  = int(x0 + deltaX), int(y0 + deltaY)
                        currDf.loc[index, currROIColName + '_x'] = Xmid
                        currDf.loc[index, currROIColName + '_y'] = Ymid
                    loop += 1
            loop = 0
            for circle in range(len(Circles)):
                for bodyparts in range(len(trackedBodyPartNames)):
                    p, q, n, m, coord = ([] for i in range(5))
                    currROIColName = circle_col_facing[loop]
                    p.extend((row[EarLeftCoords[bodyparts]], row[EarLeftCoords[bodyparts+arrayIndex]]))
                    q.extend((row[EarRightCoords[bodyparts]], row[EarRightCoords[bodyparts + arrayIndex]]))
                    n.extend((row[NoseCoords[bodyparts]], row[NoseCoords[bodyparts + arrayIndex]]))
                    m.extend((int(circleFeatures[circle, 1]), int(circleFeatures[circle, arrayIndex])))
                    center_facing_check = line_length(p, q, n, m, coord)
                    if center_facing_check[0] == True:
                        currDf.loc[index, currROIColName] = 1
                        x0 = min(center_facing_check[1][0], row[NoseCoords[bodyparts]])
                        y0 = min(center_facing_check[1][1], row[NoseCoords[bodyparts + arrayIndex]])
                        deltaX = abs((center_facing_check[1][0] - row[NoseCoords[bodyparts]]) / 2)
                        deltaY = abs((center_facing_check[1][1] - row[NoseCoords[bodyparts + arrayIndex]]) / 2)
                        Xmid, Ymid = int(x0 + deltaX), int(y0 + deltaY)
                        currDf.loc[index, currROIColName + '_x'] = Xmid
                        currDf.loc[index, currROIColName + '_y'] = Ymid
                    loop += 1
            loop = 0
            for polygon in range(len(Polygons)):
                for bodyparts in range(len(trackedBodyPartNames)):
                    p, q, n, m, coord = ([] for i in range(5))
                    currROIColName = polygon_col_facing[loop]
                    p.extend((row[EarLeftCoords[bodyparts]], row[EarLeftCoords[bodyparts+arrayIndex]]))
                    q.extend((row[EarRightCoords[bodyparts]], row[EarRightCoords[bodyparts + arrayIndex]]))
                    n.extend((row[NoseCoords[bodyparts]], row[NoseCoords[bodyparts + arrayIndex]]))
                    m.extend((int(polygonCenterCord[polygon, 1]), int(polygonCenterCord[polygon, arrayIndex-1])))
                    center_facing_check = line_length(p, q, n, m, coord)
                    if center_facing_check[0] == True:
                        currDf.loc[index, currROIColName] = 1
                        x0 = min(center_facing_check[1][0], row[NoseCoords[bodyparts]])
                        y0 = min(center_facing_check[1][1], row[NoseCoords[bodyparts + arrayIndex]])
                        deltaX = abs((center_facing_check[1][0] - row[NoseCoords[bodyparts]]) / 2)
                        deltaY = abs((center_facing_check[1][1] - row[NoseCoords[bodyparts + arrayIndex]]) / 2)
                        Xmid, Ymid = int(x0 + deltaX), int(y0 + deltaY)
                        currDf.loc[index, currROIColName + '_x'] = Xmid
                        currDf.loc[index, currROIColName + '_y'] = Ymid
                    loop += 1
    currDf = currDf.fillna(0)
    currDf = currDf.replace(np.inf, 0)
    print('ROI features calculated for ' + videoBaseName + '.')
    print('Visualizing ROI features for ' + videoBaseName + '...')
    outputFolderName = os.path.join(projectPath, 'frames', 'output', 'ROI_features')
    currVideoPath = videoFilePath
    if not os.path.exists(outputFolderName):
        os.makedirs(outputFolderName)
    outputfilename = os.path.join(outputFolderName, videoBaseName + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(currVideoPath)
    cap = cv2.VideoCapture(currVideoPath)
    vid_input_width, vid_input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### small image correction
    smallImageCorrectionValue = 1
    if vid_input_width < 400:
        smallImageCorrectionValue = 2

    sideImage = np.zeros((vid_input_height, vid_input_width*smallImageCorrectionValue,3), np.uint8)
    writer = cv2.VideoWriter(outputfilename, fourcc, int(fps), (int(vid_input_width + sideImage.shape[1]), vid_input_height))
    mySpaceScaleY, mySpaceScaleX, myRadius, myResolution, myFontScale = 40, 800, 20, 1500, 1
    #maxResDimension = max(int(vid_input_width + sideImage.shape[1]), vid_input_height)
    maxResDimension = max(vid_input_width, vid_input_height)
    textScale = float(myFontScale / (myResolution / maxResDimension))
    DrawScale = int(myRadius / (myResolution / maxResDimension))
    YspacingScale = int(mySpaceScaleY / (myResolution / maxResDimension))
    XspacingScale = int(mySpaceScaleX / (myResolution / maxResDimension))
    loop, colorList = 0, []

    ### GET COLOURS
    cmap = cm.get_cmap('Set2', len(Rectangles) + len(Circles) + len(Polygons) + 1)
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        colorList.append(rgb)


    variableList = ['in_zone', 'distance', 'facing']
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            sideImage = np.zeros((vid_input_height, vid_input_width*smallImageCorrectionValue, 3), np.uint8)
            if noAnimals == 2:
                currentPoints = (int(currDf.loc[currDf.index[loop], trackedBodyParts[0]]), int(currDf.loc[currDf.index[loop], trackedBodyParts[1]]), int(currDf.loc[currDf.index[loop], trackedBodyParts[2]]), int(currDf.loc[currDf.index[loop], trackedBodyParts[3]]))
                cv2.circle(img, (currentPoints[0], currentPoints[2]), DrawScale, (0, 255, 0), -1)
                cv2.circle(img, (currentPoints[1], currentPoints[3]), DrawScale, (0, 140, 255), -1)
                animalList = ['_Animal_1_', '_Animal_2_']
            if noAnimals == 1:
                currentPoints = (int(currDf.loc[currDf.index[loop], trackedBodyParts[0]]), int(currDf.loc[currDf.index[loop], trackedBodyParts[1]]))
                cv2.circle(img, (currentPoints[0], currentPoints[1]), DrawScale, (0, 255, 0), -1)
                animalList = ['_Animal_1_']
            for rectangle in range(len(Rectangles)):
                topLeftX, topLeftY, bottomRightX, bottomRightY = (int(rectangleFeatures[rectangle, 1]), int(rectangleFeatures[rectangle, 2]), int(rectangleFeatures[rectangle, 3]), int(rectangleFeatures[rectangle, 4]))
                cv2.rectangle(img, (topLeftX, topLeftY), (bottomRightX, bottomRightY), colorList[rectangle+ 1], DrawScale)
            for circles in range(len(Circles)):
                centerX, centerY, radius = (int(circleFeatures[circles, 1]), int(circleFeatures[circles, 2]), int(circleFeatures[circles, 3]))
                cv2.circle(img, (centerX, centerY), radius, colorList[len(Rectangles )+ circle + 1], DrawScale)
            for polygons in range(len(Polygons)):
                inputVertices = (polygonFeatures[polygons, 1])
                vertices = np.array(inputVertices, np.int32)
                cv2.polylines(img, [vertices], True, colorList[len(Rectangles) + len(Circles) + polygons + 1], thickness=DrawScale)
            startX, startY, xprintAdd, yprintAdd = 10, 30, 0, 0
            for rec in range(len(Rectangles)):
                CurrRecName = rectangleFeatures[rec, 0]
                topLeftX, topLeftY, bottomRightX, bottomRightY = (int(rectangleFeatures[rec, 1]), int(rectangleFeatures[rec, 2]), int(rectangleFeatures[rec, 3]), int(rectangleFeatures[rec, 4]))
                rectangleCentroid_x, rectangleCentroid_y = int((bottomRightX - topLeftX)/2 + topLeftX), int((bottomRightY - topLeftY) / 2 + topLeftY)
                for CurrAnimalName in animalList:
                    for variable in variableList:
                        columnName = CurrRecName + CurrAnimalName + variable
                        rectangleStatus = currDf.loc[loop, columnName]
                        cv2.putText(sideImage, str(columnName), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[rec], 2)
                        xprintAdd += XspacingScale
                        if (variable == 'in_zone'):
                            if rectangleStatus == 1:
                                cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[rec], 2)
                            if rectangleStatus == 0:
                                cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[rec], 2)
                        if (variable == 'facing'):
                            rectangleStatus_x, rectangleStatus_y = currDf.loc[loop, columnName + '_x'], currDf.loc[loop, columnName + '_y']
                            if rectangleStatus == 1:
                                cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[rec], 2)
                                cv2.circle(img, (int(rectangleCentroid_x), int(rectangleCentroid_y)), DrawScale, (0, 255, 0), -1)
                                cv2.line(img, (int(rectangleStatus_x), int(rectangleStatus_y)), (int(rectangleCentroid_x), int(rectangleCentroid_y)), (0, 255, 0), 2)
                            if rectangleStatus == 0:
                                cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[rec], 2)
                        if (variable == 'distance'):
                            rectangleStatus = round(rectangleStatus / 10, 2)
                            cv2.putText(sideImage, str(rectangleStatus), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[rec], 2)
                        yprintAdd = yprintAdd + YspacingScale
                        xprintAdd = 0
            for circ in range(len(Circles)):
                CurrCircName = circleFeatures[circ, 0]
                centerX, centerY, radius = (int(circleFeatures[circ, 1]), int(circleFeatures[circ, 2]), int(circleFeatures[circ, 3]))
                for CurrAnimalName in animalList:
                    for variable in variableList:
                        columnName = CurrCircName + CurrAnimalName + variable
                        circStatus = currDf.loc[loop, columnName]
                        cv2.putText(sideImage, str(columnName), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+circ+1], 2)
                        xprintAdd += XspacingScale
                        if (variable == 'in_zone'):
                            if circStatus == 1:
                                cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+circ+1], 2)
                            if circStatus == 0:
                                cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+circ+1], 2)
                        if (variable == 'facing'):
                            circleStatus_x, circleStatus_y = currDf.loc[loop, columnName + '_x'], currDf.loc[loop, columnName + '_y']
                            if circStatus == 1:
                                cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+circ+1], 2)
                                cv2.circle(img, (int(centerX), int(centerY)), DrawScale, (0, 255, 0), -1)
                                cv2.line(img, (int(circleStatus_x), int(circleStatus_y)),(int(centerX), int(centerY)), (0, 255, 0), 2)
                            if circStatus == 0:
                                cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles) + circ + 1], 2)
                        if (variable == 'distance'):
                            circStatus = round(circStatus / 10, 2)
                            cv2.putText(sideImage, str(circStatus), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+circ+1], 2)
                        yprintAdd = yprintAdd + YspacingScale
                        xprintAdd = 0
            for poly in range(len(Polygons)):
                CurrPolyName = polygonFeatures[poly, 0]
                for CurrAnimalName in animalList:
                    for variable in variableList:
                        columnName = CurrPolyName + CurrAnimalName + variable
                        PolyStatus = currDf.loc[loop, columnName]
                        cv2.putText(sideImage, str(columnName), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+ len(Circles) + poly + 1], 2)
                        xprintAdd += XspacingScale
                        if (variable == 'in_zone') or (variable == 'facing'):
                            if PolyStatus == 1:
                                cv2.putText(sideImage, str('True'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+ len(Circles) + poly + 1], 2)
                            if PolyStatus == 0:
                                cv2.putText(sideImage, str('False'), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+ len(Circles) + poly + 1], 2)
                        if (variable == 'distance'):
                            PolyStatus = round(PolyStatus / 10, 2)
                            cv2.putText(sideImage, str(PolyStatus), (startX + xprintAdd, startY + yprintAdd), cv2.FONT_HERSHEY_TRIPLEX, textScale, colorList[len(Rectangles)+ len(Circles) + poly + 1], 2)
                        yprintAdd = yprintAdd + YspacingScale
                        xprintAdd = 0
            imageConcat = np.concatenate((img, sideImage), axis=1)
            imageConcat = np.uint8(imageConcat)
            # cv2.imshow('image', imageConcat)
            # key = cv2.waitKey(500)  # pauses for 3 seconds before fetching next image
            # if key == 27:  # if ESC is pressed, exit loop
            #     cv2.destroyAllWindows()
            #     break
            writer.write(imageConcat)
            print('Image ' + str(loop + 1) + ' / ' + str(len(currDf)))
            loop += 1
        if img is None:
            print('Video ' + str(videoBaseName) + ' saved in project_folder/frames/output/ROI_features')
            cap.release()
            break
    print('All ROI videos generated in "project_folder/frames/ROI_features"')