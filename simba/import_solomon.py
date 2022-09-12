"""
All credit to Filipa Torrao, EMBL Rome GitHub @2909ft

"""

import os
import pandas as pd
import numpy as np
from configparser import ConfigParser
import glob
from simba.drop_bp_cords import get_fn_ext
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv

def solomonToSimba(configinifile,solomonfolder):
    print('Importing...')
    config = ConfigParser()
    config.read(configinifile)
    notarget = config.getint('SML settings','No_targets')

    #get all target
    target_list = []
    for i in range(notarget):
        target_list.append(config.get('SML settings','target_name_'+ str(i+1)))

    featurefolder = os.path.join(os.path.dirname(configinifile),'csv','features_extracted')
    solomonfiles = glob.glob(solomonfolder+'/*.csv')

    fpscsv = read_video_info_csv(os.path.join(os.path.dirname(configinifile),'logs','video_info.csv'))

    #make target inserted folder if it doesnt exits
    targetfolder = os.path.join(os.path.dirname(configinifile),'csv','targets_inserted')
    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)

    for i in solomonfiles:
        dir_name, file_name, ext = get_fn_ext(i)
        videoname = os.path.basename(i)

        if videoname in os.listdir(featurefolder):
            df = pd.read_csv(i)
            df = df.dropna()
            firstcol = df.iloc[:, 0]

            finaldf = pd.read_csv(os.path.join(featurefolder,videoname))

            #get fps
            fps = int(fpscsv.fps[fpscsv['Video'] == file_name][0])

            for j in target_list:
                col_w_behavior = df.columns[df.isin([j]).any()] #find the column with the target
                behavior = np.array(firstcol[df[col_w_behavior[0]] == j]*fps,dtype='int32') # multiple time by fps to get frames, this will get index of behavior
                finaldf[j] = 0
                for k in behavior:
                    finaldf[j][k] = 1

            finaldf.to_csv(os.path.join(targetfolder,videoname),index=False)
            print(videoname,'annotations imported')

        else:
            print('No features could be found for solomon coder annotation file: ' + str(videoname))

    print('Process completed.')