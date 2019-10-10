import os
import pandas as pd
from operator import itemgetter
from itertools import *
import cv2

def classifier_validation_command(csv_folder,folder_with_videoframefolders,outputfolder,vid_fps,video_length,seconds):

    FilesFound = []
    fps = int(vid_fps)
    videolength = int(video_length)

    inputcsvfolder = str(csv_folder)
    predictions = ['pursuit_prediction', 'tail_rattle_prediction', 'lateral_threat_prediction', 'attack_prediction', 'anogenital_prediction']
    videoFrames = str(folder_with_videoframefolders)

    for i in os.listdir(inputcsvfolder):
        if i.__contains__('.csv'):
            FilesFound.append(i)

    for i in FilesFound:
        currFile = i
        inputcsv = os.path.join(inputcsvfolder, currFile)
        df = pd.read_csv(inputcsv)

        csvtitle = i.split('.')[0]
        outputparentfolder = os.path.join(str(outputfolder), csvtitle)
        if not os.path.exists(outputparentfolder):
            os.makedirs(outputparentfolder)

        for p in predictions:
            currentpred = df[p]
            currList = [i for i, e in enumerate(currentpred) if e != 0]
            bouts = []
            behaviorFolder = os.path.join(outputparentfolder, p)
            if not os.path.exists(behaviorFolder):
                os.makedirs(behaviorFolder)
            for k, g in groupby(enumerate(currList), lambda x: x[0]-x[1]):
                bouts.append(list(map(itemgetter(1),g)))
                for t in bouts:
                    bout = t
                    min, max = bout[0], bout[-1]
                    if (min-(seconds*fps) > 0):
                        threeSbefore = min-(seconds*fps)
                    else:
                        threeSbefore = 0
                    if (max+(seconds*fps) < ((fps*videolength)+1)):
                        threeSafter = max+(seconds*fps)
                    else:
                        threeSafter = fps*videolength
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
                        print(framepath)
                        listframepaths.append(framepath)
                        dest = os.path.join(boutpath, frameName)
                        for z in listframepaths:
                            img = cv2.imread(z, cv2.IMREAD_UNCHANGED)
                            scale_percent = 40  # percent of original size
                            width = int(img.shape[1] * scale_percent / 100)
                            height = int(img.shape[0] * scale_percent / 100)
                            dim = (width, height)
                            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                            frametext = dest.split('\\', 12)[-1]
                            cv2.putText(resized, frametext, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.imwrite(dest, resized)
                            cv2.waitKey(0)
