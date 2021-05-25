from configparser import ConfigParser
import os
import pandas as pd

def ROI_reset(inifile, currVid):
    CurrVidName, CurrVidExtention = os.path.basename(currVid), os.path.splitext(currVid)[1]
    CurrVidName = CurrVidName.replace(CurrVidExtention, '')
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    vidInfPath = config.get('General settings', 'project_path')
    logFolderPath = os.path.join(vidInfPath, 'logs')
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')
    try:
        rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
        circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
        polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
        rectangularDf = rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrVidName)]
        circleDf = circleInfo.loc[circleInfo['Video'] == str(CurrVidName)]
        polygonDf = polygonInfo.loc[polygonInfo['Video'] == str(CurrVidName)]
        ROIdefExist = True
    except FileNotFoundError:
        ROIdefExist = False
        print('Cannot delete ROI definitions: no definitions exist to delete')

    if ROIdefExist is True:
        if (len(rectangularDf) == 0 and len(circleDf) == 0 and len(polygonDf) == 0):
            print('Cannot delete ROI definitions: no records for ' + str(CurrVidName))
        else:
            rectanglesInfo = rectanglesInfo[rectanglesInfo.Video != CurrVidName]
            circleInfo = circleInfo[circleInfo['Video'] != CurrVidName]
            polygonInfo = polygonInfo[polygonInfo['Video'] != CurrVidName]
            store = pd.HDFStore(ROIcoordinatesPath, mode='w')
            store['rectangles'] = rectanglesInfo
            store['circleDf'] = circleInfo
            store['polygons'] = polygonInfo
            print('Deleted ROI record: ' + str(CurrVidName))
            store.close()







