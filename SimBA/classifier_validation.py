import os
import pandas as pd
from operator import itemgetter
from itertools import *
import cv2
from configparser import ConfigParser

def classifier_validation_command(configini,seconds):

    ###get data from ini file
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    csv_path = config.get('General settings','csv_path')
    no_target = config.getint('SML settings', 'No_targets')
    framesdir_in = config.get('Frame folder','frame_folder')
    ##get predictions from ini file
    predictions =[]
    for i in range(no_target):
        predictions.append(config.get('SML settings', 'target_name_' + str(i + 1)))

    inputcsvfolder = os.path.join(csv_path,'machine_results') ##get csv folder
    videoFrames = os.path.join(framesdir_in,'input')  ###get video frames folder

    ##get all the csv in the folder
    FilesFound = []
    for i in os.listdir(inputcsvfolder):
        if i.endswith('.csv'):
            FilesFound.append(i)

    ## get fps from videoinfo csv
    videoinfocsv = os.path.dirname(csv_path) + '\\logs\\video_info.csv'
    df2 = pd.read_csv(videoinfocsv)

    for i in FilesFound:
        inputcsv = os.path.join(inputcsvfolder, i)
        df = pd.read_csv(inputcsv)
        csvtitle = i.split('.')[0]
        outputparentfolder = os.path.join(str(os.path.join(framesdir_in,'validation')), csvtitle)
        if not os.path.exists(outputparentfolder):
            os.makedirs(outputparentfolder)
        fps = int(df2['fps'].loc[(df2['Video'] == csvtitle)])
        for p in predictions:
            currentpred = df[p]
            currList = [i for i, e in enumerate(currentpred) if e != 0] ##get none zero from df into list
            print(currList)
            bouts = []
            behaviorFolder = os.path.join(outputparentfolder, p)
            if not os.path.exists(behaviorFolder):
                os.makedirs(behaviorFolder)
            for k, g in groupby(enumerate(currList), lambda x: x[0]-x[1]): ##group by and split
                bouts.append(list(map(itemgetter(1),g)))
                for t in bouts:
                    min, max = t[0], t[-1]
                    if (min-(seconds*fps) > 0):
                        threeSbefore = min-(seconds*fps) #three frames before
                    else:
                        threeSbefore = 0
                    if (max+(seconds*fps) < ((len(os.listdir(os.path.join(videoFrames,csvtitle)))))):
                        threeSafter = max+(seconds*fps)
                    else:
                        threeSafter = len(os.listdir(os.path.join(videoFrames,csvtitle)))-1
                    boutPlussix = list(range(threeSbefore, (threeSafter+1), 1))
                    boutpath = os.path.join(behaviorFolder, str(t[0]))
                    if not os.path.exists(boutpath):
                        os.makedirs(boutpath)
                    boutpathlist = []
                    boutpathlist.append(boutpath)
                    for k in boutPlussix:
                        currFrame = k
                        frameName = str(currFrame) + '.png'
                        framefolder = os.path.join(videoFrames,csvtitle)
                        listframepaths = []
                        framepath = os.path.join(framefolder, frameName)
                        listframepaths.append(framepath)
                        dest = os.path.join(boutpath, frameName)
                        print(dest)
                        for z in listframepaths:
                            img = cv2.imread(z, cv2.IMREAD_UNCHANGED)
                            scale_percent = 40  # percent of original size
                            width = int(img.shape[1] * scale_percent / 100)
                            height = int(img.shape[0] * scale_percent / 100)
                            dim = (width, height)
                            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                            frametext = os.path.basename(dest)
                            cv2.putText(resized, frametext, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.imwrite(dest, resized)
                            cv2.waitKey(0)
    print('done')