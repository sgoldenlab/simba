from __future__ import division
import os
import numpy as np
from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
from simba.rw_dfs import *
from shapely.geometry import Point, Polygon


def ROItoFeatures(inifile):
    config = ConfigParser()
    config.read(inifile)
    noAnimals = config.getint('ROI settings', 'no_of_animals')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'features_extracted')
    vidInfPath = config.get('General settings', 'project_path')
    logFolderPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
            print('Applying settings for classical tracking...')
    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    trackedBodyParts = []
    for currAnimal in range(1,noAnimals+1):
        trackedBodyParts.append(config.get('ROI settings', 'animal_' + str(currAnimal) + '_bp'))

    trackedBodyParts = [(x + '_x', x + '_y') for x in trackedBodyParts]
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')

    def line_length(p, q, n, M, coord):
        Px, Py = np.abs(p[0] - M[0]), np.abs(p[1] - M[1])
        Qx, Qy = np.abs(q[0] - M[0]), np.abs(q[1] - M[1])
        Nx, Ny = np.abs(n[0] - M[0]), np.abs(n[1] - M[1])
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

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Extracting ROI features from ' + str(len(filesFound)) + ' files...')
    print('Please be patient, code is not optimized...')
    for currFile in filesFound:
        CurrVidFn = os.path.basename(currFile)
        CurrentVideoName = os.path.basename(currFile).replace('.csv', '')
        print('Analyzing ROI features for ' + CurrentVideoName + '...')
        Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        Circles = (circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == CurrentVideoName]
        currPixPerMM = float(currVideoSettings['pixels/mm'])
        fps = float(currVideoSettings['fps'])
        currDfPath = os.path.join(csv_dir_in, CurrVidFn)
        currDf = read_df(currDfPath, wfileType)
        currDf = currDf.fillna(0)
        currDf = currDf.apply(pd.to_numeric)
        currDf = currDf.reset_index(drop=True)
        currDf = currDf.loc[:, ~currDf.columns.str.contains('^Unnamed')]
        directionalityCordHeaders = []
        EarLeftCoords, EarRightCoords, NoseCords = [], [], []
        for i in range(1, noAnimals + 1):
            EarLeftTuple, EarRightTuple, NoseTuple = ('Ear_left_' + str(i) + '_x', 'Ear_left_' + str(i) + '_y'), ('Ear_right_' + str(i) + '_x', 'Ear_right_' + str(i) + '_y'), ('Nose_' + str(i) + '_x', 'Nose_' + str(i) + '_y')
            EarLeftCoords.append(EarLeftTuple)
            EarRightCoords.append(EarRightTuple)
            NoseCords.append(NoseTuple)
            directionalityCordHeaders.extend((['Nose_' + str(i) + '_x', 'Nose_' + str(i) + '_y']))
            directionalityCordHeaders.extend(('Ear_left_' + str(i) + '_x', 'Ear_left_' + str(i) + '_y'))
            directionalityCordHeaders.extend(('Ear_right_' + str(i) + '_x', 'Ear_right_' + str(i) + '_y'))
        if set(directionalityCordHeaders).issubset(currDf.columns):
            directionalitySetting = 'yes'
        else:
            directionalitySetting = 'no'
        print('Using directionality : ' + str(directionalitySetting) + '...')

        #### FEATURES COLUMNS AND NUMPY ARRAYS WITH COORDINATES######
        rectangleFeatures = np.array([0]*5)
        Rectangle_col_inside_value, Rectangle_col_distance, Rectangle_col_facing = [], [], []
        for rectangle in range(len(Rectangles)):
            for bodypart in range(len(trackedBodyParts)):
                ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + multiAnimalIDList[bodypart] + '_in_zone')
                Rectangle_col_inside_value.append(ROI_col_name)
                currDf[ROI_col_name] = 0
                ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + multiAnimalIDList[bodypart] + '_distance')
                currDf[ROI_col_name] = 0
                Rectangle_col_distance.append(ROI_col_name)
                if directionalitySetting == 'yes':
                    ROI_col_name = str(Rectangles['Name'].iloc[rectangle] + '_' + multiAnimalIDList[bodypart] + '_facing')
                    currDf[ROI_col_name] = 0
                    Rectangle_col_facing.append(ROI_col_name)
            rectangleArray = np.array([Rectangles['Name'].iloc[rectangle], Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle], Rectangles['topLeftX'].iloc[rectangle] + Rectangles['width'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle] + Rectangles['height'].iloc[rectangle]])
            rectangleFeatures = np.vstack((rectangleFeatures, rectangleArray))
        rectangleFeatures = np.delete(rectangleFeatures, 0, 0)

        circleFeatures = np.array([0] * 4)
        circle_col_inside_value, circle_col_distance, circle_col_facing = [], [], []
        for circle in range(len(Circles)):
            for bodypart in range(len(trackedBodyParts)):
                ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + multiAnimalIDList[bodypart] + '_in_zone')
                circle_col_inside_value.append(ROI_col_name)
                currDf[ROI_col_name] = 0
                ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + multiAnimalIDList[bodypart] + '_distance')
                currDf[ROI_col_name] = 0
                circle_col_distance.append(ROI_col_name)
                if directionalitySetting == 'yes':
                    ROI_col_name = str(Circles['Name'].iloc[circle] + '_' + multiAnimalIDList[bodypart] + '_facing')
                    currDf[ROI_col_name] = 0
                    circle_col_facing.append(ROI_col_name)
            circleArray = np.array([Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle]])
            circleFeatures = np.vstack((circleFeatures, circleArray))
        circleFeatures = np.delete(circleFeatures, 0, 0)

        polyFeatures = np.array([0] * 2)
        poly_col_inside_value = []
        for polyGon in range(len(Polygons)):
            for bodypart in range(len(trackedBodyParts)):
                ROI_col_name = str(Polygons['Name'].iloc[polyGon] + '_' + multiAnimalIDList[bodypart] + '_in_zone')
                poly_col_inside_value.append(ROI_col_name)
                currDf[ROI_col_name] = 0
            polyArray = np.array([Polygons['Name'].iloc[polyGon], Polygons['vertices'].iloc[polyGon]])
            polyFeatures = np.vstack((polyFeatures, polyArray))
        polyFeatures = np.delete(polyFeatures, 0, 0)

        ### CALUCLATE BOOLEAN, IF ANIMAL IS IN RECTANGLES, CIRCLES, OR POLYGONS
        for index, row in currDf.iterrows():
            loop = 0
            for rectangle in range(len(Rectangles)):
                for bodyparts in range(len(trackedBodyParts)):
                    currROIColName = Rectangle_col_inside_value[loop]
                    loop+=1
                    if ((((int(rectangleFeatures[rectangle, 1]) - 10) <= row[trackedBodyParts[bodyparts][0]] <= (int(rectangleFeatures[rectangle, 3]) + 10))) and (((int(rectangleFeatures[rectangle, 2]) - 10) <= row[trackedBodyParts[bodyparts][1]] <= (int(rectangleFeatures[rectangle, 4]) + 10)))):
                        currDf.loc[index, currROIColName] = 1
            for column in Rectangle_col_inside_value:
                colName1 = str(column) + '_cumulative_time'
                currDf[colName1] = currDf[column].cumsum() * float(1/fps)
                colName2 = str(column) + '_cumulative_percent'
                currDf[colName2] = currDf[colName1]/currDf.index

            loop = 0
            for circles in range(len(Circles)):
                for bodyparts in range(len(trackedBodyParts)):
                    currROIColName = circle_col_inside_value[loop]
                    loop+=1
                    euclidPxDistance = np.sqrt((int(row[trackedBodyParts[bodyparts][0]]) - int(circleFeatures[circle, 1])) ** 2 + ((int(row[trackedBodyParts[bodyparts][1]]) - int(circleFeatures[circle, 2])) ** 2))
                    if euclidPxDistance <= int(circleFeatures[circle, 3]):
                        currDf.loc[index, currROIColName] = 1
            for column in circle_col_inside_value:
                colName1 = str(column) + '_cumulative_time'
                currDf[colName1] = currDf[column].cumsum() * float(1 / fps)
                colName2 = str(column) + '_cumulative_percent'
                currDf[colName2] = currDf[colName1] / currDf.index

            loop = 0
            for polyGon in range(len(Polygons)):
                currPolyGon = Polygon(polyFeatures[0, 1])
                for bodyparts in range(len(trackedBodyParts)):
                    currROIColName = poly_col_inside_value[loop]
                    loop += 1
                    currBpLoc = Point(row[trackedBodyParts[bodyparts][0]], row[trackedBodyParts[bodyparts][1]])
                    PolyGonCheck = (currPolyGon.contains(currBpLoc))
                    if PolyGonCheck == True:
                        currDf.loc[index, currROIColName] = 1
            for column in poly_col_inside_value:
                colName1 = str(column) + '_cumulative_time'
                currDf[colName1] = currDf[column].cumsum() * float(1 / fps)
                colName2 = str(column) + '_cumulative_percent'
                currDf[colName2] = currDf[colName1] / currDf.index

        ### CALUCLATE DISTANCE TO CENTER OF EACH RECTANGLE AND CIRCLES
        for index, row in currDf.iterrows():
            loop = 0
            for rectangle in range(len(Rectangles)):
                currRecCenter = [(int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ]
                for bodyparts in range(len(trackedBodyParts)):
                    currROIColName = Rectangle_col_distance[loop]
                    currDf.loc[index, currROIColName] = (np.sqrt((row[trackedBodyParts[bodyparts][0]] - currRecCenter[0]) ** 2 + (row[trackedBodyParts[bodyparts][1]] - currRecCenter[1]) ** 2)) / currPixPerMM
                    loop += 1
            loop = 0
            for circle in range(len(Circles)):
                currCircleCenterX, currCircleCenterY = (int(circleFeatures[circle, 1]), int(circleFeatures[circle, 2]))
                for bodyparts in range(len(trackedBodyParts)):
                    currROIColName = circle_col_distance[loop]
                    currDf.loc[index, currROIColName] = (np.sqrt((int(row[trackedBodyParts[bodyparts][0]]) - currCircleCenterX) ** 2 + (int(row[trackedBodyParts[bodyparts][1]]) - currCircleCenterY) ** 2)) / currPixPerMM
                    loop += 1

        ### CALCULATE IF ANIMAL IS DIRECTING TOWARDS THE CENTER OF THE RECTANGLES AND CIRCLES
        if directionalitySetting == 'yes':
            for index, row in currDf.iterrows():
                loop = 0
                for rectangle in range(len(Rectangles)):
                    for bodyparts in range(len(trackedBodyParts)):
                        p, q, n, m, coord = ([] for i in range(5))
                        currROIColName = Rectangle_col_facing[loop]
                        p.extend((row[EarLeftCoords[bodyparts][0]], row[EarLeftCoords[bodyparts][1]]))
                        q.extend((row[EarRightCoords[bodyparts][0]], row[EarRightCoords[bodyparts ][1]]))
                        n.extend((row[NoseCords[bodyparts][0]], row[NoseCords[bodyparts][1]]))
                        m.extend(((int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ))
                        center_facing_check = line_length(p, q, n, m, coord)
                        if center_facing_check[0] == True:
                            currDf.loc[index, currROIColName] = 1
                        loop += 1
                loop = 0
                for circle in range(len(Circles)):
                    for bodyparts in range(len(trackedBodyParts)):
                        p, q, n, m, coord = ([] for i in range(5))
                        currROIColName = circle_col_facing[loop]
                        p.extend((row[EarLeftCoords[bodyparts][0]], row[EarLeftCoords[bodyparts][1]]))
                        q.extend((row[EarRightCoords[bodyparts][0]], row[EarRightCoords[bodyparts][1]]))
                        n.extend((row[NoseCords[bodyparts][0]], row[NoseCords[bodyparts][1]]))
                        m.extend((int(circleFeatures[circle, 1]), int(circleFeatures[circle, 2])))
                        center_facing_check = line_length(p, q, n, m, coord)
                        if center_facing_check[0] == True:
                            currDf.loc[index, currROIColName] = 1
                        loop += 1
        currDf = currDf.fillna(0)
        currDf = currDf.replace(np.inf, 0)
        save_df(currDf, wfileType, currFile)
        print('New feature file with ROI data saved: ' + str(r'project_folder\csv\features_extracted') + str(r'\\') + str(CurrVidFn))
    print('COMPLETE: All ROI feature data appended to feature files. The new features can be found as the last columns in the CSV (or parquet) files inside the project-folder/csv/features_extracted directory.')