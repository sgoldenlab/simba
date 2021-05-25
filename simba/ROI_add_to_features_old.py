from __future__ import division
import os
import pandas as pd
import numpy as np
from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
from simba.rw_dfs import *

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

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    for i in filesFound:
        CurrVidFn = os.path.basename(i)
        CurrentVideoName = os.path.basename(i).replace('.csv', '')
        print('Analyzing ROI features for ' + CurrentVideoName + '...')
        Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        Circles = (circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
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
        if arrayIndex == 2:
            NoseCoords = ['Nose_1_x', 'Nose_2_x' , 'Nose_1_y', 'Nose_2_y']
            EarLeftCoords = ['Ear_left_1_x', 'Ear_left_2_x' , 'Ear_left_1_y', 'Ear_left_2_y']
            EarRightCoords = ['Ear_right_1_x', 'Ear_right_2_x' , 'Ear_right_1_y', 'Ear_right_2_y']
            directionalityCordHeaders.extend(NoseCoords)
            directionalityCordHeaders.extend(EarLeftCoords)
            directionalityCordHeaders.extend(EarRightCoords)
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


        #### FEATURES COLUMNS AND NUMPY ARRAYS WITH COORDINATES######
        rectangleFeatures = np.array([0]*5)
        Rectangle_col_inside_value = []
        Rectangle_col_distance = []
        Rectangle_col_facing = []
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
        circle_col_inside_value = []
        circle_col_distance = []
        circle_col_facing = []
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

        ### CALUCLATE BOOLEAN, IF ANIMAL IS IN RECTANGLES AND CIRCLES
        for index, row in currDf.iterrows():
            loop = 0
            for rectangle in range(len(Rectangles)):
                for bodyparts in range(len(trackedBodyPartNames)):
                    currROIColName = Rectangle_col_inside_value[loop]
                    loop+=1
                    try:
                        if ((((int(rectangleFeatures[rectangle, 1]) - 10) <= row[trackedBodyParts[bodyparts]] <= (int(rectangleFeatures[rectangle, 3]) + 10))) and (((int(rectangleFeatures[rectangle, 2]) - 10) <= row[trackedBodyParts[bodyparts + arrayIndex]] <= (int(rectangleFeatures[rectangle, 4]) + 10)))):
                            currDf.loc[index, currROIColName] = 1
                    except KeyError:
                        if noAnimals == 1:
                            newX, newY = ''.join([i for i in trackedBodyParts[0] if not i.isdigit()]), ''.join([i for i in trackedBodyParts[1] if not i.isdigit()])
                            trackedBodyParts = [newX, newY]
                            if ((((int(rectangleFeatures[rectangle, 1]) - 10) <= row[trackedBodyParts[bodyparts]] <= (int(rectangleFeatures[rectangle, 3]) + 10))) and (((int(rectangleFeatures[rectangle, 2]) - 10) <= row[trackedBodyParts[bodyparts + arrayIndex]] <= (int(rectangleFeatures[rectangle, 4]) + 10)))):
                                currDf.loc[index, currROIColName] = 1
            for column in Rectangle_col_inside_value:
                colName1 = str(column) + '_cumulative_time'
                currDf[colName1] = currDf[column].cumsum() * float(1/fps)
                colName2 = str(column) + '_cumulative_percent'
                currDf[colName2] = currDf[colName1]/currDf.index

            loop = 0
            for circles in range(len(Circles)):
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

        ### CALCULATE IF ANIMAL IS DIRECTING TOWARDS THE CENTER OF THE RECTANGLES AND CIRCLES
        if directionalitySetting == 'yes':
            for index, row in currDf.iterrows():
                loop = 0
                for rectangle in range(len(Rectangles)):
                    for bodyparts in range(len(trackedBodyPartNames)):
                        p, q, n, m, coord = ([] for i in range(5))
                        currROIColName = Rectangle_col_facing[loop]
                        p.extend((row[EarLeftCoords[bodyparts]], row[EarLeftCoords[bodyparts+arrayIndex]]))
                        q.extend((row[EarRightCoords[bodyparts]], row[EarRightCoords[bodyparts + arrayIndex]]))
                        n.extend((row[NoseCoords[bodyparts]], row[NoseCoords[bodyparts + arrayIndex]]))
                        m.extend(((int(rectangleFeatures[rectangle, 1]) + int(rectangleFeatures[rectangle, 3])) / 2, (int(rectangleFeatures[rectangle, 2]) + int(rectangleFeatures[rectangle, 4])) / 2 ))
                        center_facing_check = line_length(p, q, n, m, coord)
                        if center_facing_check[0] == True:
                            currDf.loc[index, currROIColName] = 1
                        loop += 1
                # for column in Rectangle_col_facing:
                #     colName1 = str(column) + '_cumulative_time'
                #     currDf[colName1] = currDf[column].cumsum() * (1 / int(fps))
                #     colName2 = str(column) + '_cumulative_percent'
                #     currDf[colName2] = 100 * currDf[colName1] / currDf[column].sum()
                #     print(currDf[colName2])
                #     currDf[colName1, colName2] = currDf[colName1, colName2].round(2)
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
                        loop += 1
                # for column in circle_col_facing:
                #     colName1 = str(column) + '_cumulative_time'
                #     currDf[colName1] = currDf[column].cumsum * (1 / fps)
                #     colName2 = str(column) + '_cumulative_percent'
                #     currDf[colName2] = 100 * currDf[colName1] / currDf['index'].sum()
                #     currDf[colName1, colName2] = currDf[colName1, colName2].round(2)
        currDf = currDf.fillna(0)
        currDf = currDf.replace(np.inf, 0)
        save_df(currDf, wfileType, i)
        print('New feature file with ROI data saved: ' + str(r'project_folder\csv\features_extracted') + str(r'\\') + str(CurrVidFn))
    print('All ROI feature data appended to feature files.')