from configparser import ConfigParser, NoSectionError, NoOptionError
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from simba.labelling_aggression import *
import threading
from simba.rw_dfs import *


def updateThreshold_graph(inifile,csv,model):
    configFile = str(inifile) ## get ini file
    config = ConfigParser()
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    project_path = config.get('General settings', 'project_path')
    videos_path = os.path.join(project_path, 'videos')

    currFile = os.path.join(csv_dir,"validation",(os.path.basename(csv)))
    classifierName = str(os.path.basename(model)).split(".")[0]
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    currDf = read_df(currFile, wfileType)
    probabilityColumnName = 'Probability_' + classifierName
    probs = currDf[[probabilityColumnName]].to_numpy()

    # get frame dir
    frames_dir_in = config.get('Frame settings', 'frames_dir_in')
    currFrameFolder = os.path.basename(currFile).replace('.' + wfileType, '')
    currFramesDir = os.path.join(frames_dir_in, currFrameFolder) ### get current video frames folder

    # get video file
    currentVideoFileName = os.path.basename(currFile).replace('.' + wfileType, '.mp4')
    current_video_file_path = os.path.join(videos_path, currentVideoFileName)
    cap = cv2.VideoCapture(current_video_file_path)
    master = choose_folder2(current_video_file_path,inifile) ## open up label gui

    ### gets mouse click on graph to open up the frames
    def onclick(event):
        if event.dblclick:
            if event.button ==1: ##get point 1 on double left click
                probability = probs[int(event.xdata)].astype(str)
                load_frame2(int(event.xdata),master.guimaster(),master.Rfbox())
                print("Selected frame has a probability of",probability, ", please enter the threshold into the entry box and validate.")
                a = plt.axvline(x=int(event.xdata),color='r')
                fig.canvas.draw()
                fig.canvas.flush_events()
                a.remove()
    #plot graphs
    import matplotlib
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    ax.plot(probs)
    plt.xlabel('frame #', fontsize=16)
    plt.ylabel(str(classifierName) + ' probability', fontsize=16)
    plt.title('Click on the points of the graph to display the corresponding frames.')
    plt.grid()
    cid = fig.canvas.mpl_connect('button_press_event', onclick) ##incoporate mouse click event
    plt.ion()
    threading.Thread(plt.show()).start()
    plt.pause(.001)











