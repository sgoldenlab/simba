import os
import cv2
import numpy as np
import pandas as pd
from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError, NoOptionError
from simba.rw_dfs import *
from simba.drop_bp_cords import get_fn_ext

def draw_line_plot(configini,video,bodypart):
    configFile = str(configini)
    config = ConfigParser()
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    configdir = os.path.dirname(configini)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    dir_path, vid_name, ext = get_fn_ext(video)
    csvname = vid_name + '.' + wfileType
    tracking_csv = os.path.join(configdir, 'csv', 'outlier_corrected_movement_location', csvname)
    inputDf = read_df(tracking_csv, wfileType)
    videopath = os.path.join(configdir,'videos',video)
    outputvideopath = os.path.join(configdir, 'frames', 'output', 'simple_path_plots')

    if not os.path.exists(outputvideopath):
        os.mkdir(outputvideopath)

    #datacleaning
    colHeads = [bodypart + '_x', bodypart + '_y', bodypart + '_p']
    df = inputDf[colHeads].copy()

    widthlist = df[colHeads[0]].astype(float).astype(int)
    heightlist = df[colHeads[1]].astype(float).astype(int)
    circletup = tuple(zip(widthlist,heightlist))

    # get resolution of video
    vcap = cv2.VideoCapture(videopath)
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        totalFrameCount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make white background
    img = np.zeros([height, width, 3])
    img.fill(255)
    img = np.uint8(img)


    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(outputvideopath,video), 0x7634706d, fps, (width,height))
    counter=0
    while (vcap.isOpened()):
        ret,frame = vcap.read()
        if ret == True:
            if counter !=0:
                cv2.line(img,circletup[counter-1],circletup[counter],5)

            lineWithCircle = img.copy()
            cv2.circle(lineWithCircle, circletup[counter],5,[0,0,255],-1)



            out.write(lineWithCircle)
            counter+=1
            print('Frame ' + str(counter) + '/' + str(totalFrameCount))

        else:
            break

    vcap.release()
    cv2.destroyAllWindows()
    print('Video generated.')


def draw_line_plot_tools(videopath,csvfile,bodypart):

    #read csv
    inputDf = pd.read_csv(csvfile)
    #restructure
    col1 = inputDf.loc[0].to_list()
    col2 = inputDf.loc[1].to_list()
    finalcol = [m+'_'+n for m,n in zip(col1,col2)]
    inputDf.columns = finalcol
    inputDf = inputDf.loc[2:]
    inputDf = inputDf.reset_index(drop=True)
    #datacleaning
    colHeads = [bodypart + '_x', bodypart + '_y', bodypart + '_likelihood']
    df = inputDf[colHeads].copy()

    widthlist = df[colHeads[0]].astype(float).astype(int)
    heightlist = df[colHeads[1]].astype(float).astype(int)
    circletup = tuple(zip(widthlist,heightlist))

    # get resolution of video
    vcap = cv2.VideoCapture(videopath)
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        totalFrameCount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make white background
    img = np.zeros([height, width, 3])
    img.fill(255)
    img = np.uint8(img)

    outputvideoname = os.path.join(os.path.dirname(videopath),'line_plot'+os.path.basename(videopath))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outputvideoname, 0x7634706d, fps, (width,height))
    counter=0
    while (vcap.isOpened()):
        ret,frame = vcap.read()
        if ret == True:
            if counter !=0:
                cv2.line(img,circletup[counter-1],circletup[counter],5)

            lineWithCircle = img.copy()
            cv2.circle(lineWithCircle, circletup[counter],5,[0,0,255],-1)



            out.write(lineWithCircle)
            counter+=1
            print('Frame ' + str(counter) + '/' + str(totalFrameCount))

        else:
            break

    vcap.release()
    cv2.destroyAllWindows()
    print('Video generated.')