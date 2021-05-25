import os
import pandas as pd
from operator import itemgetter
import cv2
from configparser import ConfigParser, NoSectionError, NoOptionError
import glob
from itertools import groupby, chain
from simba.rw_dfs import *



def validate_classifier(configini,seconds,target):
    seconds = int(seconds)
    ###get data from ini file
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    inputcsvfolder = os.path.join(projectPath, 'csv', 'machine_results')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    videoFolder = os.path.join(projectPath, 'videos')

    ##get all the csv in the folder
    csvList = glob.glob(inputcsvfolder + '/*.' + wfileType)

    ## get fps from videoinfo csv
    videoinfocsv = os.path.join(projectPath, 'logs', 'video_info.csv')
    fpsdf = pd.read_csv(videoinfocsv)
    fileCounter = 0

    #main loop
    for i in csvList:
        print('Processing file: ' + str(i) + ' File ' + str(fileCounter) + ' / ' + str(len(csvList)))
        df = read_df(i, wfileType)
        csvname = os.path.basename(i).split('.')[0]
        fps = int(fpsdf['fps'].loc[(fpsdf['Video'] == csvname)])
        targetcolumn = df[target]
        probabilitycolumn = df['Probability_' + target]
        framesBehavior = [i for i, e in enumerate(targetcolumn) if e != 0]  ##get frame numbers with behavior into a list( df column = 1)


        #get all bouts in a list
        bouts = []
        for k, g in groupby(enumerate(framesBehavior), lambda x: x[0] - x[1]):  ##group by and split # get the bouts in a list of list# groupby splits it base on continuous bout
            bouts.append(list(map(itemgetter(1), g)))
        no_bouts = (len(bouts))

        #get the seconds before bouts about to happen
        finalbout_list =[]
        for bout in bouts:
            minn = bout[0] # starting bout frame
            maxx = bout[-1] #ending bout frame

            if (minn - (seconds * fps) > 0):
                threeSbefore = minn - (seconds * fps)  # three frames before
            else:
                threeSbefore = 0
            if (maxx + (seconds * fps) < len(df)):
                threeSafter = maxx + (seconds * fps)  #three frames after
            else:
                threeSafter = len(df) - 1

            boutPlusSeconds = list(range(threeSbefore, (threeSafter + 1), 1))

            finalbout_list.append(boutPlusSeconds)

        #unlist list of list
        totalframes = list(chain(*finalbout_list))


        if os.path.exists(os.path.join(videoFolder, str(csvname) + '.mp4')):
            currVideo = os.path.join(videoFolder, str(csvname) + '.mp4')
        elif os.path.exists(os.path.join(videoFolder, str(csvname) + '.avi')):
            currVideo = os.path.join(videoFolder, str(csvname) + '.avi')
        else:
            print('Cannot locate video ' + str(csvname.replace('.csv', '')) + 'in mp4 or avi format')
            break

        cap = cv2.VideoCapture(currVideo)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        currFrameFolderOut = os.path.join(os.path.dirname(configini),'frames','output','classifier_validation')

        #make folder if not exist
        if not os.path.exists(currFrameFolderOut):
            os.makedirs(currFrameFolderOut)
        writer = cv2.VideoWriter(os.path.join(currFrameFolderOut, (csvname + '_'+ str(target) + '_' + str(no_bouts) + '.mp4')), fourcc, fps, (frame_width, frame_height))
        framecounter= 1
        # create video
        for i in range(no_bouts):

            rowCounter = finalbout_list[i][0] ## get the first bout frames from the list
            cap = cv2.VideoCapture(os.path.join(os.path.dirname(configini), 'videos', currVideo)) #
            cap.set(1,rowCounter) #set starting point at first about

            while(cap.isOpened()):
                ret, img = cap.read()
                if ret == True:
                    if rowCounter in finalbout_list[i]:
                        text1 = target + ' Event ' + str(i)
                        text2 = 'Total frames of event ' + str(i) + ' = ' + str(len(bouts[i]))
                        text3 = 'Frames of event ' + str(i) + ' from ' + str((bouts[i][0])) + ' to ' + str((bouts[i][-1]))
                        text4 = 'Frame no ' + str(rowCounter)
                        text5 = 'Probability = ' + str(round(probabilitycolumn[rowCounter],4))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = min(frame_width,frame_height)/(25/0.03)
                        cv2.putText(img, text1, (50, 50), font, fontScale, (255, 255, 0), 2)
                        cv2.putText(img, text2, (50, 100), font, fontScale, (255, 255, 0), 2)
                        cv2.putText(img, text3, (50, 150), font, fontScale, (255, 255, 0), 2)
                        cv2.putText(img, text4, (50, 200), font, fontScale, (255, 255, 0), 2)
                        cv2.putText(img, text5, (50, 250), font, fontScale, (255, 255, 0), 2)
                        writer.write(img)
                        print('Writing frame',str(framecounter),'/',str(len(totalframes)))
                        framecounter+=1

                    elif rowCounter >= (finalbout_list[i][-1]): #break if it matches the last bout frame
                        break
                    else:
                        pass

                rowCounter += 1

                if img is None:
                    break

        cap.release()



