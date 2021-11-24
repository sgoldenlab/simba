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
from simba.features_scripts.unit_tests import *


def roi_time_bins(inifile, inputcsv, binLength=int):

    if binLength < 1:
        print('WARNING: We recommend time-bin lengths above 1s.')

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
                multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
            print('Applying settings for classical tracking...')
    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    logFolderPath = os.path.join(projectPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    csv_dir_in = os.path.join(projectPath, 'csv', inputcsv)
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    try:
        rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    except FileNotFoundError:
        print('Could not find user-defined ROI definitions.')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)

    out_df_list_time = []
    out_df_list_entries = []

    for filecounter, i in enumerate(filesFound):
        CurrVidFn = os.path.basename(i)
        CurrentVideoName = CurrVidFn.replace('.' + wfileType, '')
        print('Analysing ' + str(CurrentVideoName) + '...')
        videoSettings, pix_per_mm, currFps = read_video_info(vidinfDf, CurrentVideoName)
        noRectangles = len(rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        noCircles = len(circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        noPolygons = len(polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        if (noRectangles == 0) and (noCircles == 0) and (noPolygons == 0):
            print('WARNING: No user-defined ROIs detected for video ' + str(CurrentVideoName))
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

        rect_ee_dict, circle_ee_dict, polygon_ee_dict = {}, {}, {}
        for animal in range(noAnimals):
            rect_ee_dict[multiAnimalIDList[animal]] = {}
            for rectangle in range(noRectangles):
                rect_ee_dict[multiAnimalIDList[animal]][Rectangles['Name'].iloc[rectangle]] = {'Entry_times': [],
                                                                                               'Exit_times': []}
        for animal in range(noAnimals):
            circle_ee_dict[multiAnimalIDList[animal]] = {}
            for circle in range(noCircles):
                circle_ee_dict[multiAnimalIDList[animal]][Circles['Name'].iloc[circle]] = {'Entry_times': [],
                                                                                           'Exit_times': []}
        for animal in range(noAnimals):
            polygon_ee_dict[multiAnimalIDList[animal]] = {}
            for poly in range(noPolygons):
                polygon_ee_dict[multiAnimalIDList[animal]][Polygons['Name'].iloc[poly]] = {'Entry_times': [],
                                                                                           'Exit_times': []}

        currDf = csv_df[columns2grab]
        binFrameLength = int(binLength * currFps)
        totalSecInSession = currDf.shape[0] / currFps
        currListDf = [currDf[i:i + binFrameLength] for i in range(0, currDf.shape[0], binFrameLength)]

        if filecounter == 0:
            output_time_headers, output_entry_headers = ['Video', 'Time bin #', 'Time bin start (s)',
                                                         'Time bin end (s)', 'Animal', 'ROI', 'Time'], ['Video',
                                                                                                        'Time bin #',
                                                                                                        'Time bin start (s)',
                                                                                                        'Time bin end (s)',
                                                                                                        'Animal', 'ROI',
                                                                                                        'Entries']
            setBins = len(currListDf)

        out_row_time = []
        out_row_entries = []

        video_output_list_times, video_output_list_entries = [CurrentVideoName], [CurrentVideoName]
        for bincounter, current_bin in enumerate(currListDf[:setBins]):
            if bincounter == 0:
                start_time = 0
            else:
                start_time = int(current_bin.index[0] / currFps)
            end_time = int(current_bin.index[-1] / currFps)
            rectangleTimes, rectangleEntries = ([[0] * len(multiAnimalIDList) for i in range(noRectangles)],
                                                [[0] * len(multiAnimalIDList) for i in range(noRectangles)])
            circleTimes, circleEntries = ([[0] * len(multiAnimalIDList) for i in range(noCircles)],
                                          [[0] * len(multiAnimalIDList) for i in range(noCircles)])
            polygonTime, polyGonEntries = ([[0] * len(multiAnimalIDList) for i in range(noPolygons)],
                                           [[0] * len(multiAnimalIDList) for i in range(noPolygons)])
            bin_size_s = len(current_bin / currFps)
            for index, row in current_bin.iterrows():
                currentPoints = np.empty((noAnimals, 2), dtype=int)
                current_probability_list = []
                for animal in range(noAnimals):
                    currentPoints[animal][0], currentPoints[animal][1] = int(row[animalBodypartList[animal][0]]), int(
                        row[animalBodypartList[animal][1]])
                    current_probability_list.append(row[animalBodypartList[animal][2]])
                for rectangle in range(noRectangles):
                    topLeftX, topLeftY = (
                    Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle])
                    bottomRightX, bottomRightY = (
                    topLeftX + Rectangles['width'].iloc[rectangle], topLeftY + Rectangles['height'].iloc[rectangle])
                    rectName = Rectangles['Name'].iloc[rectangle]
                    for bodyparts in range(len(currentPoints)):
                        if ((((topLeftX - 10) <= currentPoints[bodyparts][0] <= (bottomRightX + 10)) and (
                                (topLeftY - 10) <= currentPoints[bodyparts][1] <= (bottomRightY + 10)))) and (
                                current_probability_list[bodyparts] > probability_threshold):
                            rectangleTimes[rectangle][bodyparts] = rectangleTimes[rectangle][bodyparts] + (1 / currFps)
                            if rectangleEntryCheck[rectangle][bodyparts] == True:
                                rect_ee_dict[multiAnimalIDList[bodyparts]][rectName]['Entry_times'].append(index)
                                rectangleEntries[rectangle][bodyparts] += 1
                                rectangleEntryCheck[rectangle][bodyparts] = False
                        else:
                            if rectangleEntryCheck[rectangle][bodyparts] == False:
                                rect_ee_dict[multiAnimalIDList[bodyparts]][rectName]['Exit_times'].append(index)
                            rectangleEntryCheck[rectangle][bodyparts] = True

                for circle in range(noCircles):
                    circleName, centerX, centerY, radius = (
                    Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle],
                    Circles['radius'].iloc[circle])
                    for bodyparts in range(len(currentPoints)):
                        euclidPxDistance = int(np.sqrt((currentPoints[bodyparts][0] - centerX) ** 2 + (
                                    currentPoints[bodyparts][1] - centerY) ** 2))
                        if (euclidPxDistance <= radius) and (
                                current_probability_list[bodyparts] > probability_threshold):
                            circleTimes[circle][bodyparts] = circleTimes[circle][bodyparts] + (1 / currFps)
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
                            polygonTime[polygon][bodyparts] = polygonTime[polygon][bodyparts] + (1 / currFps)
                            if polygonEntryCheck[polygon][bodyparts] == True:
                                polygon_ee_dict[multiAnimalIDList[bodyparts]][PolygonName]['Entry_times'].append(index)
                                polyGonEntries[polygon][bodyparts] += 1
                                polygonEntryCheck[polygon][bodyparts] = False
                        else:
                            if polygonEntryCheck[polygon][bodyparts] == False:
                                polygon_ee_dict[multiAnimalIDList[bodyparts]][PolygonName]['Exit_times'].append(index)
                            polygonEntryCheck[polygon][bodyparts] = True

            Rectangles = Rectangles.reset_index(drop=True)
            for value, rectangle_counter in enumerate(zip(rectangleTimes, rectangleEntries)):
                for animal in range(len(multiAnimalIDList)):
                    out_row_time.append([CurrentVideoName, bincounter, start_time, end_time, multiAnimalIDList[animal], Rectangles.loc[value]['Name'], rectangleTimes[value][animal]])
                    out_row_entries.append([CurrentVideoName, bincounter, start_time, end_time, multiAnimalIDList[animal], Rectangles.loc[value]['Name'], rectangleEntries[value][animal]])

            Circles = Circles.reset_index(drop=True)
            for value, circle_counter in enumerate(zip(circleTimes, circleEntries)):
                for animal in range(len(multiAnimalIDList)):
                    out_row_time.append([CurrentVideoName, bincounter, start_time, end_time, multiAnimalIDList[animal], Circles.loc[value]['Name'], circleTimes[value][animal]])
                    out_row_entries.append([CurrentVideoName, bincounter, start_time, end_time, multiAnimalIDList[animal], Circles.loc[value]['Name'], circleEntries[value][animal]])

            Polygons = Polygons.reset_index(drop=True)
            for value, polygon_counter in enumerate(zip(polygonTime, polyGonEntries)):
                for animal in range(len(multiAnimalIDList)):
                    out_row_time.append([CurrentVideoName, bincounter, start_time, end_time, multiAnimalIDList[animal], Polygons.loc[value]['Name'], polygonTime[value][animal]])
                    out_row_entries.append([CurrentVideoName, bincounter, start_time, end_time, multiAnimalIDList[animal], Polygons.loc[value]['Name'], polyGonEntries[value][animal]])

        curr_out_df_time = pd.DataFrame(out_row_time, columns=output_time_headers)
        curr_out_df_entries = pd.DataFrame(out_row_entries, columns=output_entry_headers)
        out_df_list_time.append(curr_out_df_time)
        out_df_list_entries.append(curr_out_df_entries)

    if len(filesFound) < 1:
        print('No files found. Have you corrected outliers or clicked to skip outlier correction?')

    else:
        out_df_time = pd.concat(out_df_list_time, axis=0)
        out_df_time = out_df_time.sort_values(by=['Video', 'Time bin start (s)', 'Animal'])
        out_df_entries = pd.concat(out_df_list_entries, axis=0)
        out_df_entries = out_df_entries.sort_values(by=['Video', 'Time bin start (s)', 'Animal'])
        outputDfTimeFilePath, outputDfEntryFilePath = (
        os.path.join(logFolderPath, 'ROI_time_bins_' + str(binLength) + 's_time_data_' + dateTime + '.csv'),
        os.path.join(logFolderPath, 'ROI_time_bins_' + str(binLength) + 's_entry_data_' + dateTime + '.csv'))
        out_df_time.to_csv(outputDfTimeFilePath, index=False)
        out_df_entries.to_csv(outputDfEntryFilePath, index=False)
        print('Saved data at: ' + str(outputDfTimeFilePath))
        print('Saved data at: ' + str(outputDfEntryFilePath))
        print('ROI time-bin data for ' + str(binLength) + ' second time bins saved in project_folder/logs folder')