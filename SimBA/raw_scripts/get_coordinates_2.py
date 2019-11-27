from configparser import ConfigParser
import os
import cv2
import numpy as np
import pandas as pd

config = ConfigParser()
configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"
config.read(configFile)
frames_dir_in = config.get('Frame settings', 'frames_dir_in')
use_master = config.get('General settings', 'use_master_config')
fps = config.getint('Frame settings', 'fps')
resWidth = config.getint('Frame settings', 'resolution_width')
resHeight = config.getint('Frame settings', 'resolution_height')
log_df = pd.DataFrame(columns=['Video','fps', 'pixels/mm', 'Resolution_width', 'Resolution_height'])
videoList = []
ppm_list = []
loopy = 0
newCordList = []
filesFound = []
configFilelist = []
framePaths = []
resHeight_list = []
resWidth_list = []
fps_list = []
csv_dir = config.get('General settings', 'csv_path')
csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
log_path = config.get('General settings', 'project_path')
log_path = os.path.join(log_path, 'project_folder', 'logs')
outFile = 'video_info.csv'
outFilePath = os.path.join(log_path, outFile)
cordStatus = False
moveStatus = False
insertStatus = False
changeLoop = False

########### FIND CSV FILES ###########
if use_master == 'yes':
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
if use_master == 'no':
    config_folder_path = config.get('General settings', 'config_folder')
    for i in os.listdir(config_folder_path):
        if i.__contains__(".ini"):
            configFilelist.append(os.path.join(config_folder_path, i))
            iniVidName = i.split(".")[0]
            csv_fn = iniVidName + '.csv'
            file = os.path.join(csv_dir_in, csv_fn)
            filesFound.append(file)

########### FIND FRAME PATHS ###########
for i in filesFound:
    currFile = i
    vidName = os.path.basename(currFile)
    vidName = vidName.replace('.csv','')
    currFramepath = os.path.join(frames_dir_in, vidName)
    framePaths.append(currFramepath)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    global cordStatus
    if (event == cv2.EVENT_LBUTTONDBLCLK) and len(cordList) < 4:
        cv2.circle(overlay,(x,y),16,(144,0,255),-1)
        cordList.append(x)
        cordList.append(y)
        if len(cordList) == 4:
            cordStatus = True
            cv2.line(overlay, (cordList[0], cordList[1]), (cordList[2], cordList[3]), (144, 0, 255), 6)

def select_cord_to_change(event,x,y,flags,param):
    global moveStatus
    global coordChange
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        if (x>=(cordList[0]-20)) and (x<=(cordList[0]+20)) and (y>=(cordList[1]-20)) and (y<=(cordList[1]+20)): #change point1
            coordChange = [1, cordList[0], cordList[1]]
            moveStatus = True
        if (x >= (cordList[2] - 20)) and (x <= (cordList[2] + 20)) and (y >= (cordList[3] - 20)) and (y <= (cordList[3] + 20)):  # change point2
            coordChange = [2, cordList[2], cordList[3]]
            moveStatus = True

def select_new_dot_location(event,x,y,flags,param):
    global insertStatus
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        newCordList.append(x)
        newCordList.append(y)
        insertStatus = True

for i in framePaths:
    currFrameDir = i
    if use_master == 'no':
        configFile = configFilelist[loopy]
        config = ConfigParser()
        config.read(configFile)
        fps = config.getint('Frame settings', 'fps')
        resWidth = config.getint('Frame settings', 'resolution_width')
        resHeight = config.getint('Frame settings', 'resolution_height')
    vidName = os.path.basename(currFrameDir)
    imgPath = os.path.join(currFrameDir, '0.png')
    print(imgPath)
    img = cv2.imread(imgPath)
    origImage = img.copy()
    overlay = img.copy()
    ix,iy = -1,-1
    cordList = []
    cv2.namedWindow('Select coordinates: double left mouse click at two locations. Press ESC when done', cv2.WINDOW_NORMAL)

    while(1):
        if cordStatus == False and (moveStatus == False) and (insertStatus == False):
            cv2.setMouseCallback('Select coordinates: double left mouse click at two locations. Press ESC when done',draw_circle)
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done',overlay)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        if (cordStatus == True) and (moveStatus == False) and (insertStatus == False):
            if changeLoop == True:
                overlay = origImage.copy()
                cv2.circle(overlay, (cordList[0], cordList[1]), 16, (144, 0, 255), -1)
                cv2.circle(overlay, (cordList[2], cordList[3]), 16, (144, 0, 255), -1)
                cv2.line(overlay, (cordList[0], cordList[1]), (cordList[2], cordList[3]), (144, 0, 255), 6)
            cv2.putText(overlay, 'Click on dot to move', (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)
            cv2.putText(overlay, 'Press ESC to exit', (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done', overlay)
            cv2.setMouseCallback('Select coordinates: double left mouse click at two locations. Press ESC when done', select_cord_to_change)
        if (moveStatus == True) and (insertStatus == False):
            if changeLoop == True:
                img = origImage.copy()
                changeLoop = False
            if coordChange[0] == 1:
                cv2.circle(img, (cordList[2], cordList[3]), 16, (144, 0, 255), -1)
            if coordChange[0] == 2:
                cv2.circle(img, (cordList[0], cordList[1]), 16, (144, 0, 255), -1)
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done', img)
            cv2.putText(img, 'Click on new dot location', (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)
            cv2.setMouseCallback('Select coordinates: double left mouse click at two locations. Press ESC when done', select_new_dot_location)
        if (insertStatus == True):
            if coordChange[0] == 1:
                cv2.circle(img, (cordList[2], cordList[3]), 16, (144, 0, 255), -1)
                cv2.circle(img, (newCordList[-2], newCordList[-1]), 16, (144, 0, 255), -1)
                cv2.line(img, (cordList[2], cordList[3]), (newCordList[-2], newCordList[-1]), (144, 0, 255), 6)
                cordList = [newCordList[-2], newCordList[-1], cordList[2], cordList[3]]
                cordStatus = True
                moveStatus = False
                insertStatus = False
                changeLoop = True
            if coordChange[0] == 2:
                cv2.circle(img, (cordList[0], cordList[1]), 16, (144, 0, 255), -1)
                cv2.circle(img, (newCordList[-2], newCordList[-1]), 16, (144, 0, 255), -1)
                cv2.line(img, (cordList[0], cordList[1]), (newCordList[-2], newCordList[-1]), (144, 0, 255), 6)
                cordList = [cordList[0], cordList[1], newCordList[-2], newCordList[-1]]
                cordStatus = True
                moveStatus = False
                insertStatus = False
                changeLoop = True
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break





    cv2.destroyAllWindows()
    euclidPixelDist = np.sqrt((cordList[0] - cordList[2]) ** 2 + (cordList[1] - cordList[3]) ** 2)
    mm_dist = int(input("Length in mm: "))
    cordStatus = False
    moveStatus = False
    insertStatus = False
    changeLoop = False
    ppm = euclidPixelDist / mm_dist
    print(ppm)
    videoList.append(vidName)
    ppm_list.append(ppm)
    fps_list.append(fps)
    resHeight_list.append(resHeight)
    resWidth_list.append(resWidth)

log_df['Video'] = videoList
log_df['fps'] = fps_list
log_df['pixels/mm'] = ppm_list
log_df['Resolution_width'] = resWidth_list
log_df['Resolution_height'] = resHeight_list
log_df.to_csv(outFilePath, index=False)




