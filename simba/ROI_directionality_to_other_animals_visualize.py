from __future__ import division
import os
import datetime
import pandas as pd
import numpy as np
from configparser import ConfigParser, NoOptionError, NoSectionError
import glob
import cv2
from pylab import cm
from simba.rw_dfs import *
from simba.drop_bp_cords import *


def ROI_directionality_other_animals_visualize(inifile):
    config = ConfigParser()
    config.read(inifile)
    noAnimals = config.getint('General settings', 'animal_no')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'directionality_dataframes')
    # frames_dir_out = os.path.join(projectPath, 'frames', 'output', 'Directionality_between_animals')
    # if not os.path.exists(frames_dir_out): os.makedirs(frames_dir_out)
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)

    animalIDlist = config.get('Multi animal IDs', 'id_list')

    if not animalIDlist:
        animalIDlist = []
        for animal in range(noAnimals):
            animalIDlist.append('Animal_' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    else:
        animalIDlist = animalIDlist.split(",")
        multiAnimalStatus = True
        print('Applying settings for multi-animal tracking...')


    def line_length(p, q, n, M, coord):
        Px = np.abs(p[0] - M[0])
        Py = np.abs(p[1] - M[1])
        Qx = np.abs(q[0] - M[0])
        Qy = np.abs(q[1] - M[1])
        Nx = np.abs(n[0] - M[0])
        Ny = np.abs(n[1] - M[1])
        Ph = np.sqrt(Px*Px + Py*Py)
        Qh = np.sqrt(Qx*Qx + Qy*Qy)
        Nh = np.sqrt(Nx*Nx + Ny*Ny)
        if (Nh < Ph and Nh < Qh and Qh < Ph):
            coord.extend((q[0], q[1]))
            return True, coord
        elif (Nh < Ph and Nh < Qh and Ph < Qh):
            coord.extend((p[0], p[1]))
            return True, coord
        else:
            return False, coord

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    if not filesFound:
        print('No directionality calculations found. Please run the calculations before running the visualization creation.')
    videoCounter = 1
    x_cols, y_cols, p_cols = getBpNames(inifile)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, animalIDlist, noAnimals, x_cols, y_cols, p_cols, [])

    for filePath in filesFound:
        fileBaseName = os.path.basename(filePath)
        filename, fileType = os.path.splitext(fileBaseName)[0],  os.path.splitext(fileBaseName)[1]
        print('Analyzing ROI features for ' + filename + '...')
        currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == filename]
        fps = float(currVideoSettings['fps'])
        currDf = read_df(filePath, wfileType)
        currDf = currDf.fillna(0)
        currDf = currDf.apply(pd.to_numeric)
        currDf = currDf.reset_index(drop=True)
        currDf = currDf.loc[:, ~currDf.columns.str.contains('^Unnamed')]
        directionalityDict = checkDirectionalityCords(animalBpDict)
        facingDfcols, directionColheaders, directionColEyeXHeads, directionColEyeYHeads, directionColBpXHeads, directionColBpYHeads = [],[],[],[],[],[]
        listofListColHeaders = []
        listofListEyeXHeaders, listofListEyeYHeaders, listofListBpXHeaders, listofListBpYHeaders = [], [], [],[]


        ####### CREATE DESTINATION DATAFRAME #############
        for animal in directionalityDict.keys():
            otherAnimals = animalIDlist.copy()
            otherAnimals.remove(animal)
            for otherAnimal in otherAnimals:
                otherAnimalDictX = animalBpDict[otherAnimal]['X_bps']
                currColHeaders, currXEyeHeaders, currYEyeHeaders, currBpXHeaders, currBpYHeaders = [], [], [], [], []
                for otherAnimalBp in otherAnimalDictX:
                    currBp = otherAnimal + '_' + otherAnimalBp
                    currBp = currBp.replace('_x', '')
                    directionColheaders.append(str(animal) + '_directing_' + currBp)
                    currColHeaders.append(directionColheaders[-1])
                    directionColEyeXHeads.append(str(animal) + '_directing_' + currBp + '_eye_x')
                    currXEyeHeaders.append(directionColEyeXHeads[-1])
                    directionColEyeYHeads.append(str(animal) + '_directing_' + currBp + '_eye_y')
                    currYEyeHeaders.append(directionColEyeYHeads[-1])
                    directionColBpXHeads.append(str(animal) + '_directing_' + currBp + '_bp_x')
                    currBpXHeaders.append(directionColBpXHeads[-1])
                    directionColBpYHeads.append(str(animal) + '_directing_' + currBp + '_bp_y')
                    currBpYHeaders.append(directionColBpYHeads[-1])
                listofListColHeaders.append(currColHeaders)
                listofListEyeXHeaders.append(currXEyeHeaders)
                listofListEyeYHeaders.append(currYEyeHeaders)
                listofListBpXHeaders.append(currBpXHeaders)
                listofListBpYHeaders.append(currBpYHeaders)

        outputFolderName = os.path.join(projectPath, 'frames', 'output', 'ROI_directionality_visualize')
        if not os.path.exists(outputFolderName):
            os.makedirs(outputFolderName)
        currVideoPath = os.path.join(projectPath, 'videos', filename + '.mp4')
        outputfilename = os.path.join(outputFolderName, filename + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        cap = cv2.VideoCapture(currVideoPath)

        vid_input_width, vid_input_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(outputfilename, fourcc, int(fps), (vid_input_width, vid_input_height))
        mySpaceScaleY, mySpaceScaleX, myRadius, myResolution, myFontScale = 40, 800, 20, 1500, 1
        maxResDimension = max(vid_input_width, vid_input_height)
        DrawScale = int(myRadius / (myResolution / maxResDimension))
        colorList = []

        cmaps = ['spring', 'summer', 'autumn', 'cool', 'Wistia', 'Pastel1', 'Set1', 'winter', 'gnuplot', 'gnuplot2', 'cubehelix', 'brg', 'jet', 'terrain', 'ocean', 'rainbow', 'gist_earth', 'gist_stern', 'gist_ncar', 'Spectral', 'coolwarm']
        cMapSize = int(len(x_cols) * noAnimals) + 1
        colorListofList = []
        for colormap in range(len(cmaps)):
            currColorMap = cm.get_cmap(cmaps[colormap], cMapSize)
            currColorList = []
            for i in range(currColorMap.N):
                rgb = list((currColorMap(i)[:3]))
                rgb = [i * 255 for i in rgb]
                rgb.reverse()
                currColorList.append(rgb)
            colorListofList.append(currColorList)

        currRow = 0
        while (cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                overlay = img.copy()
                for currentGaze in range(len(listofListColHeaders)):
                    directingAnimalList = listofListColHeaders[currentGaze]
                    for directed2bp in range(len(directingAnimalList)):
                        directing2bodyPart = directingAnimalList[directed2bp]
                        lookedStatus = int(currDf.loc[currRow, directing2bodyPart])
                        color = colorListofList[currentGaze][directed2bp]
                        if lookedStatus == 1:
                            eye_x_col = listofListEyeXHeaders[currentGaze][directed2bp]
                            eye_y_col = listofListEyeYHeaders[currentGaze][directed2bp]
                            eye_x_cord = currDf.loc[currRow, eye_x_col]
                            eye_y_cord = currDf.loc[currRow, eye_y_col]
                            bp_x_col = listofListBpXHeaders[currentGaze][directed2bp]
                            bp_y_col = listofListBpYHeaders[currentGaze][directed2bp]
                            bp_x_cord = currDf.loc[currRow, bp_x_col]
                            bp_y_cord = currDf.loc[currRow, bp_y_col]
                            if (bp_x_cord != 0 and bp_y_cord != 0):
                                if (eye_x_cord != 0 and eye_y_cord != 0):
                                    cv2.line(overlay, (int(eye_x_cord), int(eye_y_cord)), (int(bp_x_cord), int(bp_y_cord)), color, 4)
                overlay = np.uint8(overlay)
                image_new = cv2.addWeighted(overlay, 0.6, img, 1 - 0.4, 0)
                writer.write(image_new)
                print('Image ' + str(currRow + 1) + ' / ' + str(len(currDf)))

            if img is None:
                print('Video ' + str(outputfilename) + ' saved in project_folder/frames/output/ROI_directionality_visualize')
                cap.release()
            currRow += 1






