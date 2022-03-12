from configparser import ConfigParser, NoSectionError, NoOptionError
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from simba.labelling_aggression import *
import threading
from simba.rw_dfs import *
from simba.drop_bp_cords import get_fn_ext
import os



click_counter = 0
vertical_line = plt.axvline(x=0, color='r')

def updateThreshold_graph(inifile,csv,model):
    global click_counter
    global vertical_line
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

    # get video file
    dir_name, file_name, ext = get_fn_ext(currFile)
    mp4_fp = os.path.join(videos_path, file_name + '.mp4')
    avi_fp = os.path.join(videos_path, file_name + '.avi')
    if os.path.isfile(mp4_fp):
        current_video_file_path = mp4_fp
    elif os.path.isfile(avi_fp):
        current_video_file_path = avi_fp
    else:
        print('ERROR: could not locate video in either mp4 or avi format: ' + str(mp4_fp) + ' and ' + str(avi_fp) + ' could not be found in the project_folder/videos directory')

    cap = cv2.VideoCapture(current_video_file_path)
    master = choose_folder2(current_video_file_path, inifile) ## open up label gui


    ### gets mouse click on graph to open up the frames
    def onclick(event):
        global click_counter
        global vertical_line
        if event.dblclick:
            if event.button == 1: ##get point 1 on double left click
                if click_counter == 0:
                    vertical_line = plt.axvline(x=0, color='r')
                click_counter += 1
                probability = probs[int(event.xdata)].astype(str)
                load_frame2(int(event.xdata),master.guimaster(),master.Rfbox())
                print("Selected frame has a probability of",probability, ", please enter the threshold into the entry box and validate.")
                vertical_line.set_xdata(x=int(event.xdata))
                fig.canvas.draw()
                fig.canvas.flush_events()
                #a.remove()


    #plot graphs
    import matplotlib
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    ax.plot(probs)
    plt.xlabel('frame #', fontsize=16)
    plt.ylabel(str(classifierName) + ' probability', fontsize=16)
    plt.title('Click on the points of the graph to display the corresponding frames.')
    plt.grid()
    print('b')
    cid = fig.canvas.mpl_connect('button_press_event', onclick) ##incoporate mouse click event
    plt.ion()
    threading.Thread(plt.show()).start()
    plt.pause(.001)











