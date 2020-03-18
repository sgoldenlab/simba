import glob
import pandas as pd
from configparser import ConfigParser
import os

def multiplyFreeHand(inifile, currVid):
    CurrVidName = os.path.splitext(os.path.basename(currVid))[0]
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    videoPath = os.path.join(projectPath, 'videos')
    ROIcoordinatesPath = os.path.join(projectPath, 'logs', 'measures', 'ROI_definitions.h5')
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
        print('Cannot apply to all: no ROI definitions exists')

    if ROIdefExist is True:
        if (len(rectangularDf) == 0 and len(circleDf) == 0 and len(polygonDf) == 0):
            print('Cannot apply ROIs to all: no records exist for ' + str(CurrVidName))
        else:
            videofilesFound = glob.glob(videoPath + '/*.mp4') + glob.glob(videoPath + '/*.avi')
            duplicatedRec, duplicatedCirc, duplicatedPoly = (rectangularDf.copy(), circleDf.copy(), polygonDf.copy())
            for vids in videofilesFound:
                currVidName = os.path.splitext(os.path.basename(vids))[0]
                duplicatedRec['Video'], duplicatedCirc['Video'], duplicatedPoly['Video'] = (currVidName, currVidName, currVidName)
                rectangularDf = rectangularDf.append(duplicatedRec, ignore_index=True)
                circleDf = circleDf.append(duplicatedCirc, ignore_index=True)
                polygonDf = polygonDf.append(duplicatedPoly, ignore_index=True)
            rectangularDf = rectangularDf.drop_duplicates(subset=['Video', 'Name'], keep="first")
            circleDf = circleDf.drop_duplicates(subset=['Video', 'Name'], keep="first")
            polygonDf = polygonDf.drop_duplicates(subset=['Video', 'Name'], keep="first")
            store = pd.HDFStore(ROIcoordinatesPath, mode='w')
            store['rectangles'] = rectangularDf
            store['circleDf'] = circleDf
            store['polygons'] = polygonDf
            store.close()
            print('ROI(s) for ' + CurrVidName + ' applied to all videos')
            print('Next, click on "draw" to modify ROI location(s) or click on "reset" to remove ROI drawing(s)')
