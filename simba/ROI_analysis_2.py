from configparser import ConfigParser
import os
import pandas as pd
import itertools
import numpy as np
from shapely.geometry import Point
from shapely import geometry
import glob
from datetime import datetime
import sys


def roiAnalysis(inifile,inputcsv):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    config.read(inifile)

    ## get dataframe column name
    bpcsv = (os.path.join(os.path.dirname(inifile), 'logs', 'measures', 'pose_configs', 'bp_names',
                          'project_bp_names.csv'))
    bplist = []
    with open(bpcsv) as f:
        for row in f:
            bplist.append(row)
    bplist = list(map(lambda x: x.replace('\n', ''), bplist))
    bpcolname =['scorer']
    for i in bplist:
        bpcolname.append(i+'_x')
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
        trackedBodyPartNames = ['Animal_1_', 'Animal_2_']
    else:
        arrayIndex = 1
        bodyPartAnimal_1 = config.get('ROI settings', 'animal_1_bp')
        animalBodypartList.append(bodyPartAnimal_1)
        trackedBodyPartNames = ['Animal_1_']
    trackedBodyParts = []
    for i in range(len(animalBodypartList)):
        bps = [str(animalBodypartList[i]) + '_x', str(animalBodypartList[i]) + '_y']
        trackedBodyParts.append(bps)

    vidInfPath = config.get('General settings', 'project_path')
    logFolderPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logFolderPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, str(inputcsv)) ### read in outlier correction movement csv
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
    circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
    polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
    outputDfTime = pd.DataFrame(columns=['Video'])
    rectangleNames, circleNames, polygonNames = (list(rectanglesInfo['Name'].unique()), list(circleInfo['Name'].unique()), list(polygonInfo['Name'].unique()))
    shapeList = list(itertools.chain(rectangleNames, circleNames, polygonNames))
    for newcol in range(len(shapeList)):
        for bp in trackedBodyPartNames:
            colName = str(bp) + shapeList[newcol]
            outputDfTime[colName] = 0
    for newcol in range(len(shapeList)):
        for bp in trackedBodyPartNames:
            colName = str(bp) + shapeList[newcol] + '_%_of_session'
            outputDfTime[colName] = 0
    outputDfEntries = outputDfTime.copy()
    filesFound = glob.glob(csv_dir_in + '/*.csv')


    for i in filesFound:
        CurrVidFn = os.path.basename(i)
        CurrentVideoName = os.path.basename(i).replace('.csv', '')
        print('Analysing ' + str(CurrentVideoName) + '...')
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(CurrentVideoName)]
        currFps = int(videoSettings['fps'])
        noRectangles = len(rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        noCircles = len(circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        noPolygons = len(polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        rectangleTimes, rectangleEntries = ([[0] * len(trackedBodyParts) for i in range(noRectangles)] , [[0] * len(trackedBodyParts) for i in range(noRectangles)])
        circleTimes, circleEntries = ([[0] * len(trackedBodyParts) for i in range(noCircles)], [[0] * len(trackedBodyParts) for i in range(noCircles)])
        polygonTime, polyGonEntries = ([[0] * len(trackedBodyParts) for i in range(noPolygons)], [[0] * len(trackedBodyParts) for i in range(noPolygons)])
        Rectangles = (rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrentVideoName)])
        Circles = (circleInfo.loc[circleInfo['Video'] == str(CurrentVideoName)])
        Polygons = (polygonInfo.loc[polygonInfo['Video'] == str(CurrentVideoName)])
        rectangleEntryCheck = [[True] * len(trackedBodyParts) for i in range(noRectangles)]
        circleEntryCheck = [[True] * len(trackedBodyParts) for i in range(noCircles)]
        polygonEntryCheck = [[True] * len(trackedBodyParts) for i in range(noPolygons)]
        currDfPath = os.path.join(csv_dir_in, CurrVidFn)
        currDf = pd.read_csv(currDfPath)
        currDf = currDf.drop(['video_no', 'frames'], axis=1, errors='ignore')

        currDf.columns = bpcolname
        totalSecInSession = currDf.shape[0] / currFps

        for index, row in currDf.iterrows():
            if noAnimals == 2:
                currentPoints = [int(row[trackedBodyParts[0][0]]), int(row[trackedBodyParts[1][0]]), int(row[trackedBodyParts[0][1]]), int(row[trackedBodyParts[1][1]])]
            if noAnimals == 1:
                currentPoints = [int(row[trackedBodyParts[0][0]]), int(row[trackedBodyParts[0][1]])]
            for rectangle in range(noRectangles):
                topLeftX, topLeftY = (Rectangles['topLeftX'].iloc[rectangle], Rectangles['topLeftY'].iloc[rectangle])
                bottomRightX, bottomRightY = (topLeftX + Rectangles['width'].iloc[rectangle], topLeftY + Rectangles['height'].iloc[rectangle])
                for bodyparts in range(len(trackedBodyParts)):
                    if (((topLeftX-10) <= currentPoints[bodyparts] <= (bottomRightX+10)) and ((topLeftY-10) <= currentPoints[bodyparts+arrayIndex] <= (bottomRightY+10))):
                        rectangleTimes[rectangle][bodyparts] = round((rectangleTimes[rectangle][bodyparts] + (1 / currFps)), 2)
                        if rectangleEntryCheck[rectangle][bodyparts] == True:
                            rectangleEntries[rectangle][bodyparts] += 1
                            rectangleEntryCheck[rectangle][bodyparts] = False
                    else:
                        rectangleEntryCheck[rectangle][bodyparts] = True
            for circle in range(noCircles):
                circleName, centerX, centerY, radius = (Circles['Name'].iloc[circle], Circles['centerX'].iloc[circle], Circles['centerY'].iloc[circle], Circles['radius'].iloc[circle])
                for bodyparts in range(len(trackedBodyParts)):
                    euclidPxDistance = int(np.sqrt((currentPoints[bodyparts] - centerX) ** 2 + (currentPoints[bodyparts+arrayIndex] - centerY) ** 2))
                    if euclidPxDistance <= radius:
                        circleTimes[circle][bodyparts] = round((circleTimes[circle][bodyparts] + (1 / currFps)),2)
                        if circleEntryCheck[circle][bodyparts] == True:
                            circleEntries[circle][bodyparts] += 1
                            circleEntryCheck[circle][bodyparts] = False
                    else:
                        circleEntryCheck[circle][bodyparts] = True
            for polygon in range(noPolygons):
                PolygonName, vertices = (Polygons['Name'].iloc[polygon], Polygons['vertices'].iloc[polygon])
                vertices = np.array(vertices, np.int32)
                for bodyparts in range(len(trackedBodyParts)):
                    pointList = []
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
        EntrieInEntryList = [x / sumEntries for x in EntrieInEntryList]
        EntrieInEntryList = ['%.3f' % elem for elem in EntrieInEntryList]
        collapsedListEntry.extend(EntrieInEntryList)
        outputDfTime = outputDfTime.append(pd.Series(dict(zip(outputDfTime.columns, collapsedListTime))),ignore_index=True)
        outputDfEntries = outputDfEntries.append(pd.Series(dict(zip(outputDfEntries.columns, collapsedListEntry))),ignore_index=True)

    if len(filesFound) < 1:
        print('No files found. Have you corrected outliers?')
    else:
        outputDfTimeFilePath , outputDfEntryFilePath = (os.path.join(logFolderPath, 'ROI_time_data_' + dateTime + '.csv'), os.path.join(logFolderPath, 'ROI_entry_data_' + dateTime + '.csv'))
        outputDfTime.to_csv(outputDfTimeFilePath, index=False)
        outputDfEntries.to_csv(outputDfEntryFilePath, index=False)
        print('ROI data saved in ' + 'project_folder\logs.')


