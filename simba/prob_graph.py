from configparser import ConfigParser
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from simba.labelling_aggression import *
import threading
def updateThreshold_graph(inifile,csv,model):
    ## find the csv for df
    configFile = str(inifile) ## get ini file
    config = ConfigParser()
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    currFile = os.path.join(csv_dir,"validation",(os.path.basename(csv)))
    classifierName = str(os.path.basename(model)).split(".")[0]
    currDf = pd.read_csv(currFile)
    probabilityColumnName = 'Probability_' + classifierName
    probs = currDf[[probabilityColumnName]].to_numpy()

    # get frame dir
    frames_dir_in = config.get('Frame settings', 'frames_dir_in')
    currFrameFolder = os.path.basename(currFile).replace('.csv', '')
    currFramesDir = os.path.join(frames_dir_in, currFrameFolder) ### get current video frames folder

    master = choose_folder2(currFramesDir) ## open up label gui

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











