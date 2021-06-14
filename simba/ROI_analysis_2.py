import glob
import itertools
from collections import defaultdict
import os
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
from datetime import datetime
import numpy as np
import pandas as pd
from shapely import geometry
from shapely.geometry import Point
from simba.drop_bp_cords import getBpHeaders
from simba.rw_dfs import *

def roiAnalysis(inifile, inputcsv, calculate_dist):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    config.read(inifile)
    ## get dataframe column name
    noAnimals = config.getint('ROI settings', 'no_of_animals')
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
        animalBpNameX, animalBpNameY, animalBpNameP = animalBpName + '_x', animalBpName + '_y', animalBpName + '_p'
        animalBodypartList.append([animalBpNameX, animalBpNameY, animalBpNameP])
    print(animalBodypartList)
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
                multiAnimalIDList.append('Animal_' + str(animal+1) + '_')
            print('Applying settings for classical tracking...')
    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    logFolderPath = os.path.join(projectPath, 'logs')
    detailed_ROI_data_path = os.path.join(projectPath, 'logs', 'Detailed_ROI_data')
    if not os.path.exists(detailed_ROI_data_path): os.makedirs(detailed_ROI_data_path)

    if calculate_dist:
        out_df_list = []

    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    csv_dir_in = os.path.join(projectPath, 'csv', inputcsv)
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
    outputDfTime = pd.DataFrame(columns=['Video'])
    rectangleNames, circleNames, polygonNames = (list(rectanglesInfo['Name'].unique()), list(circleInfo['Name'].unique()), list(polygonInfo['Name'].unique()))
    shapeList = list(itertools.chain(rectangleNames, circleNames, polygonNames))
    for newcol in range(len(shapeList)):
        for bp in multiAnimalIDList:
            colName = str(bp) + ' ' + shapeList[newcol] + ' (s)'
            outputDfTime[colName] = 0
    for newcol in range(len(shapeList)):
        for bp in multiAnimalIDList:
            colName = str(bp) + ' ' + shapeList[newcol] + ' (% of session)'
            outputDfTime[colName] = 0
    outputDfEntries = outputDfTime.copy()
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    movement_in_ROIs_dict = {}

    for i in filesFound:
        CurrVidFn = os.path.basename(i)
        CurrentVideoName = CurrVidFn.replace('.' + wfileType, '')
        print('Analysing ' + str(CurrentVideoName) + '...')
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName)]
        try:
            currFps = int(videoSettings['fps'])
            pix_per_mm = float(videoSettings['pixels/mm'])
        except TypeError:
            print('The FPS / pixels per millimeter for ' + CurrentVideoName + ' could not be found in the project/logs/video_info.csv file, or multiple entries for ' + CurrentVideoName + ' exist in your  project/logs/video_info.csv file. Make sure each video in your project is represented once in your project/logs/video_info.csv file')
        noRectangles = len(rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        noCircles = len(circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        noPolygons = len(polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        rectangleTimes, rectangleEntries = ([[0] * len(multiAnimalIDList) for i in range(noRectangles)] , [[0] * len(multiAnimalIDList) for i in range(noRectangles)])
        circleTimes, circleEntries = ([[0] * len(multiAnimalIDList) for i in range(noCircles)], [[0] * len(multiAnimalIDList) for i in range(noCircles)])
        polygonTime, polyGonEntries = ([[0] * len(multiAnimalIDList) for i in range(noPolygons)], [[0] * len(multiAnimalIDList) for i in range(noPolygons)])
        Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        Circles = (circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        rectangleEntryCheck = [[True] * len(multiAnimalIDList) for i in range(noRectangles)]
        circleEntryCheck = [[True] * len(multiAnimalIDList) for i in range(noCircles)]
        polygonEntryCheck = [[True] * len(multiAnimalIDList) for i in range(noPolygons)]
        currDfPath = os.path.join(csv_dir_in, CurrVidFn)
        csv_df = read_df(currDfPath, wfileType)
        csv_df = csv_df.loc[:, ~csv_df.columns.str.contains('^Unnamed')]
        try:
            csv_df = csv_df.set_index('scorer')
        except KeyError:
            pass
        bpHeaders = getBpHeaders(inifile)
        csv_df.columns = bpHeaders
        currDf = csv_df[columns2grab]
        totalSecInSession = currDf.shape[0] / currFps

        rect_ee_dict, circle_ee_dict, polygon_ee_dict = {}, {}, {}
        for animal in range(noAnimals):
            rect_ee_dict[multiAnimalIDList[animal]] = {}
            for rectangle in range(noRectangles):
                rect_ee_dict[multiAnimalIDList[animal]][Rectangles['Name'].iloc[rectangle]] = {'Entry_times': [], 'Exit_times': []}
        for animal in range(noAnimals):
            circle_ee_dict[multiAnimalIDList[animal]] = {}
            for circle in range(noCircles):
                circle_ee_dict[multiAnimalIDList[animal]][Circles['Name'].iloc[circle]] = {'Entry_times': [], 'Exit_times': []}
        for animal in range(noAnimals):
            polygon_ee_dict[multiAnimalIDList[animal]] = {}
            for poly in range(noPolygons):
                polygon_ee_dict[multiAnimalIDList[animal]][Polygons['Name'].iloc[poly]] = {'Entry_times': [], 'Exit_times': []}

        for index, row in currDf.iterrows():
            currentPoints = np.empty((noAnimals, 2), dtype=int)
            current_probability_list = []
            for animal in range(noAnimals):
                currentPoints[animal][0], currentPoints[animal][1] = int(row[animalBodypartList[animal][0]]), int(row[animalBodypartList[animal][1]])
                current_probability_list.append(row[animalBodypartList[animal][2]])
            for rectangle in range(noRectangles):
                topLeftX, topLeftY = (Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle])
                bottomRightX, bottomRightY = (topLeftX + Rectangles['width'].iloc[rectangle], topLeftY + Rectangles['height'].iloc[rectangle])
                rectName = Rectangles['Name'].iloc[rectangle]
                for bodyparts in range(len(currentPoints)):
                    if ((((topLeftX-10) <= currentPoints[bodyparts][0] <= (bottomRightX+10)) and ((topLeftY-10) <= currentPoints[bodyparts][1] <= (bottomRightY+10)))) and (current_probability_list[bodyparts] > probability_threshold):
                        rectangleTimes[rectangle][bodyparts] = round((rectangleTimes[rectangle][bodyparts] + (1 / currFps)), 2)
                        if rectangleEntryCheck[rectangle][bodyparts] == True:
                            rect_ee_dict[multiAnimalIDList[bodyparts]][rectName]['Entry_times'].append(index)
                            rectangleEntries[rectangle][bodyparts] += 1
                            rectangleEntryCheck[rectangle][bodyparts] = False
                    else:
                        if rectangleEntryCheck[rectangle][bodyparts] == False:
                            rect_ee_dict[multiAnimalIDList[bodyparts]][rectName]['Exit_times'].append(index)
                        rectangleEntryCheck[rectangle][bodyparts] = True

            for circle in range(noCircles):
                circleName, centerX, centerY, radius = (Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle])
                for bodyparts in range(len(currentPoints)):
                    euclidPxDistance = int(np.sqrt((currentPoints[bodyparts][0] - centerX) ** 2 + (currentPoints[bodyparts][1] - centerY) ** 2))
                    if (euclidPxDistance <= radius) and (current_probability_list[bodyparts] > probability_threshold):
                        circleTimes[circle][bodyparts] = round((circleTimes[circle][bodyparts] + (1 / currFps)),2)
                        if circleEntryCheck[circle][bodyparts] == True:
                            circle_ee_dict[multiAnimalIDList[bodyparts]][circleName]['Entry_times'].append(index)
                            circleEntries[circle][bodyparts] += 1
                            circleEntryCheck[circle][bodyparts] = False
                    else:
                        if circleEntryCheck[circle][bodyparts] == False:
                            circle_ee_dict[multiAnimalIDList[bodyparts]][circleName]['Exit_times'].append(index)
                        circleEntryCheck[circle][bodyparts] = True

            for polygon in range(noPolygons):
                PolygonName, vertices = (Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon])
                vertices = np.array(vertices, np.int32)
                for bodyparts in range(len(currentPoints)):
                    pointList = []
                    for i in vertices:
                        point = geometry.Point(i)
                        pointList.append(point)
                    polyGon = geometry.Polygon([[p.x, p.y] for p in pointList])
                    CurrPoint = Point(int(currentPoints[bodyparts][0]), int(currentPoints[bodyparts][1]))
                    polyGonStatus = (polyGon.contains(CurrPoint))
                    if (polyGonStatus == True) and (current_probability_list[bodyparts] > probability_threshold):
                        polygonTime[polygon][bodyparts] = round((polygonTime[polygon][bodyparts] + (1 / currFps)), 2)
                        if polygonEntryCheck[polygon][bodyparts] == True:
                            polygon_ee_dict[multiAnimalIDList[bodyparts]][PolygonName]['Entry_times'].append(index)
                            polyGonEntries[polygon][bodyparts] += 1
                            polygonEntryCheck[polygon][bodyparts] = False
                    else:
                        if polygonEntryCheck[polygon][bodyparts] == False:
                            polygon_ee_dict[multiAnimalIDList[bodyparts]][PolygonName]['Exit_times'].append(index)
                        polygonEntryCheck[polygon][bodyparts] = True

        video_ee_df = pd.DataFrame(columns=['Animal_name', 'Shape_name', 'Entry_frame', 'Exit_frame'])
        for animal_dict in rect_ee_dict:
            for shape_dict in rect_ee_dict[animal_dict]:
                for entry, exit in itertools.zip_longest(rect_ee_dict[animal_dict][shape_dict]['Entry_times'], rect_ee_dict[animal_dict][shape_dict]['Exit_times'], fillvalue=-1):
                    video_ee_df.loc[len(video_ee_df)] = [animal_dict, shape_dict, entry, exit]
        for animal_dict in circle_ee_dict:
            for shape_dict in circle_ee_dict[animal_dict]:
                for entry, exit in itertools.zip_longest(circle_ee_dict[animal_dict][shape_dict]['Entry_times'], circle_ee_dict[animal_dict][shape_dict]['Exit_times'], fillvalue=-1):
                    video_ee_df.loc[len(video_ee_df)] = [animal_dict, shape_dict, entry, exit]
        for animal_dict in polygon_ee_dict:
            for shape_dict in polygon_ee_dict[animal_dict]:
                for entry, exit in itertools.zip_longest(polygon_ee_dict[animal_dict][shape_dict]['Entry_times'], polygon_ee_dict[animal_dict][shape_dict]['Exit_times'], fillvalue=-1):
                    video_ee_df.loc[len(video_ee_df)] = [animal_dict, shape_dict, entry, exit]
        video_ee_df = video_ee_df.sort_values(by='Entry_frame')
        save_path = os.path.join(detailed_ROI_data_path, CurrentVideoName + '_' + dateTime + '.csv')
        video_ee_df.to_csv(save_path, index=False)

        rectangleTimes, circleTimes, polygonTime = (list(itertools.chain(*rectangleTimes)), list(itertools.chain(*circleTimes)), list(itertools.chain(*polygonTime)))
        rectangleEntries, circleEntries, polyGonEntries = (list(itertools.chain(*rectangleEntries)), list(itertools.chain(*circleEntries)), list(itertools.chain(*polyGonEntries)))
        collapsedListTime = [CurrentVideoName, rectangleTimes, circleTimes, polygonTime]
        collapsedListTime = list(itertools.chain.from_iterable(itertools.repeat(x, 1) if isinstance(x, str) else x for x in collapsedListTime))
        timesInCollTime = collapsedListTime[1:]
        timesInCollTime = [x / totalSecInSession for x in timesInCollTime]
        timesInCollTime = ['%.3f' % elem for elem in timesInCollTime]
        collapsedListTime.extend(timesInCollTime)
        collapsedListEntry = [CurrentVideoName, rectangleEntries, circleEntries, polyGonEntries]
        collapsedListEntry = list(itertools.chain.from_iterable(itertools.repeat(x, 1) if isinstance(x, str) else x for x in collapsedListEntry))
        EntrieInEntryList = collapsedListEntry[1:]
        sumEntries = sum(EntrieInEntryList)
        entry_props = []
        for entry_count in EntrieInEntryList:
            if entry_count > 0:
                entry_props.append(round((entry_count / sumEntries), 3))
            else:
                entry_props.append(0)
        collapsedListEntry.extend(entry_props)
        outputDfTime = outputDfTime.append(pd.Series(dict(zip(outputDfTime.columns, collapsedListTime))),ignore_index=True)
        outputDfEntries = outputDfEntries.append(pd.Series(dict(zip(outputDfEntries.columns, collapsedListEntry))),ignore_index=True)

        if calculate_dist:
            shape_list_of_dicts = [rect_ee_dict, circle_ee_dict, polygon_ee_dict]
            movement_in_ROIs_dict = {}
            for curr_animal in multiAnimalIDList: movement_in_ROIs_dict[curr_animal] = {}
            for shape_type in range(len(shape_list_of_dicts)):
                curr_shape_dict = shape_list_of_dicts[shape_type]
                for animal_counter, curr_animal in enumerate(curr_shape_dict):
                    curr_animal_dict = curr_shape_dict[curr_animal]
                    current_animal_df = csv_df[animalBodypartList[animal_counter]]
                    for shape in curr_animal_dict:
                        movement_in_ROIs_dict[curr_animal][shape] = {}
                        entry_list, exit_list = curr_animal_dict[shape]['Entry_times'], curr_animal_dict[shape]['Exit_times']
                        movement_list_in_shape = []
                        all_df_movements_list = []
                        try:
                            for entry_frame, exit_frame in zip(entry_list, exit_list):
                                entry_and_exit = current_animal_df.loc[entry_frame:exit_frame]
                                entry_and_exit = entry_and_exit.reset_index(drop=True)
                                entry_and_exit_shifted = entry_and_exit.shift(1)
                                entry_and_exit_shifted = entry_and_exit_shifted.combine_first(entry_and_exit).add_prefix('Shifted_')
                                entry_and_exit = pd.concat([entry_and_exit, entry_and_exit_shifted], axis=1)
                                entry_and_exit['Movement'] = (np.sqrt((entry_and_exit.iloc[:, 0] - entry_and_exit.iloc[:, 3]) ** 2 + (entry_and_exit.iloc[:, 1] - entry_and_exit.iloc[:, 4]) ** 2)) / pix_per_mm
                                #entry_and_exit.at[0, 'Movement'] = 0
                                #entry_and_exit.at[1, 'Movement'] = 0
                                all_df_movements_list.append(entry_and_exit)
                            all_movements = pd.concat(all_df_movements_list, axis=0).reset_index(drop=True)
                            inside_shape_df_list = [all_movements[i:i + currFps] for i in range(0, all_movements.shape[0], currFps)]
                            for s_inside_shape in inside_shape_df_list:
                                movement_list_in_shape.append(s_inside_shape['Movement'].mean())
                            total_movement_of_animal_in_shape = sum(movement_list_in_shape)
                            movement_in_ROIs_dict[curr_animal][shape] = total_movement_of_animal_in_shape
                        except ValueError:
                            movement_in_ROIs_dict[curr_animal][shape] = 0


            video_list, out_list_headers = [CurrentVideoName], ['Video']
            for animal in movement_in_ROIs_dict.keys():
                animal_dict = movement_in_ROIs_dict[animal]
                for shape in animal_dict.keys():
                    time = movement_in_ROIs_dict[animal][shape]
                    out_list_headers.append(animal + ' ' + shape + ' (movement inside shape (cm))')
                    video_list.append(time)
            video_df = pd.DataFrame([video_list], columns=out_list_headers)
            out_df_list.append(video_df)

    if len(filesFound) < 1:
        print('No files found. Have you corrected outliers or clicked to skip outlier correction?')

    else:
        outputDfTimeFilePath , outputDfEntryFilePath = (os.path.join(logFolderPath, 'ROI_time_data_' + dateTime + '.csv'), os.path.join(logFolderPath, 'ROI_entry_data_' + dateTime + '.csv'))
        outputDfTime.to_csv(outputDfTimeFilePath, index=False)
        outputDfEntries.to_csv(outputDfEntryFilePath, index=False)
        print('Summery ROI data saved in ' + 'project_folder\logs.')
        print('Detailed per video ROI data saved in ' + 'project_folder\logs\Detailed_ROI_data.')

        if calculate_dist:
            out_df = pd.concat(out_df_list, axis=0).set_index('Video').astype(int)
            out_movement_df_path = os.path.join(logFolderPath, 'Movement_within_ROIs_' + dateTime + '.csv')
            out_df.to_csv(out_movement_df_path)
            print('Movement-in-ROI data saved @ ' + str(out_movement_df_path))