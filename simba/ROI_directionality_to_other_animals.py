from __future__ import division
import os
from _datetime import datetime
import pandas as pd
import numpy as np
from configparser import ConfigParser, NoOptionError, NoSectionError
import glob
import cv2
from pylab import cm
from simba.rw_dfs import *
from simba.drop_bp_cords import *
from simba.drop_bp_cords import get_fn_ext

def directing_to_other_animals(inifile):
    dateTimes = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    config.read(inifile)
    noAnimals = config.getint('General settings', 'animal_no')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(projectPath, 'csv', 'directionality_dataframes')
    if not os.path.exists(csv_dir_out): os.makedirs(csv_dir_out)
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)

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



    x_cols, y_cols, p_cols = getBpNames(inifile)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, animalIDlist, noAnimals, x_cols, y_cols, p_cols, [])


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
    videoCounter = 1

    for filePath in filesFound:
        _, filename, fileType = get_fn_ext(filePath)
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

        ####### CREATE DESTINATION DATAFRAME #############
        for animal in directionalityDict.keys():
            otherAnimals = animalIDlist.copy()
            otherAnimals.remove(animal)
            for otherAnimal in otherAnimals:
                otherAnimalDictX = animalBpDict[otherAnimal]['X_bps']
                currColHeaders = []
                for otherAnimalBp in otherAnimalDictX:
                    currBp = otherAnimal + '_' + otherAnimalBp
                    currBp = currBp.replace('_x', '')
                    directionColheaders.append(str(animal) + '_directing_' + currBp)
                    currColHeaders.append(directionColheaders[-1])
                    directionColEyeXHeads.append(str(animal) + '_directing_' + currBp + '_eye_x')
                    directionColEyeYHeads.append(str(animal) + '_directing_' + currBp + '_eye_y')
                    directionColBpXHeads.append(str(animal) + '_directing_' + currBp + '_bp_x')
                    directionColBpYHeads.append(str(animal) + '_directing_' + currBp + '_bp_y')
                listofListColHeaders.append(currColHeaders)
        for col1, col2, col3, col4, col5 in zip(directionColheaders, directionColEyeXHeads, directionColEyeYHeads, directionColBpXHeads, directionColBpYHeads):
            facingDfcols.extend((col1,col2,col3,col4,col5))
        emptyNumpy = np.zeros(shape=(currDf.shape[0],len(facingDfcols)))
        facingDf = pd.DataFrame(emptyNumpy, columns=facingDfcols)

        print('Calculating measurements.... say NO to BIG-O! :)!')
        #### FEATURES COLUMNS AND NUMPY ARRAYS WITH COORDINATES######
        frameCounter = 0
        for index, row in currDf.iterrows():
            for animal in directionalityDict.keys():
                otherAnimals = animalIDlist.copy()
                otherAnimals.remove(animal)
                p, q, n = ([] for i in range(3))
                earLeftXcol, earLeftYcol = directionalityDict[animal]['Ear_left']['X_bps'], directionalityDict[animal]['Ear_left']['Y_bps']
                earRightXcol, earRightYcol = directionalityDict[animal]['Ear_right']['X_bps'], directionalityDict[animal]['Ear_right']['Y_bps']
                noseXcol, noseYcol = directionalityDict[animal]['Nose']['X_bps'], directionalityDict[animal]['Nose']['Y_bps']
                p.extend((row[earLeftXcol], row[earLeftYcol]))
                q.extend((row[earRightXcol], row[earRightYcol]))
                n.extend((row[noseXcol], row[noseYcol]))
                for otherAnimal in otherAnimals:
                    otherAnimalDictX = animalBpDict[otherAnimal]['X_bps']
                    otherAnimalDictY = animalBpDict[otherAnimal]['Y_bps']
                    for otherAnimalBpX, otherAnimalBpY in zip(otherAnimalDictX, otherAnimalDictY):
                        currBp = otherAnimal + '_' + otherAnimalBpX
                        currBp = currBp.replace('_x', '')
                        currCol = str(animal) + '_directing_' + str(currBp)
                        m, coord = ([] for i in range(2))
                        m.extend((row[otherAnimalBpX], row[otherAnimalBpY]))
                        center_facing_check = line_length(p, q, n, m, coord)
                        if center_facing_check[0] == True:
                            x0, y0 = min(center_facing_check[1][0], row[noseXcol]), min(center_facing_check[1][1], row[noseYcol])
                            deltaX, deltaY = abs((center_facing_check[1][0] - row[noseXcol]) / 2), abs((center_facing_check[1][1] - row[noseYcol]) / 2)
                            Xmid, Ymid = int(x0 + deltaX), int(y0 + deltaY)
                            facingDf.loc[index, currCol + '_eye_x'] = Xmid
                            facingDf.loc[index, currCol + '_eye_y'] = Ymid
                            facingDf.loc[index, currCol + '_bp_x'] = row[otherAnimalBpX]
                            facingDf.loc[index, currCol + '_bp_y'] = row[otherAnimalBpY]
                            if (int(Xmid) != 0 and int(Ymid) != 0):
                                if (int(row[otherAnimalBpX]) != 0 and int(row[otherAnimalBpY]) != 0):
                                    facingDf.loc[index, currCol] = 1
                        if center_facing_check[0] == False:
                            pass
            frameCounter += 1
            print('Analysing frame ' + str(frameCounter) + '/' + str(len(currDf)) + '. Video ' + str(videoCounter) + '/' + str(len(filesFound)))

        save_df(facingDf, wfileType, os.path.join(csv_dir_out, filename + '.' + wfileType))
        videoCounter +=1
        print('Summary dataframe statistics saved in project_folder/csv/directionality_dataframes subdirectory.')

        print('Calculating summary statistics for ' + str(filename) + '...')
        columnCounter = 0
        outputRow, outPutHeaders = [], ['Video']
        for animal in directionalityDict.keys():
            otherAnimals = animalIDlist.copy()
            otherAnimals.remove(animal)
            for otherAnimal in otherAnimals:
                who2who = listofListColHeaders[columnCounter]
                outPutHeaders.append(animal + ' directing towards ' + otherAnimal + ' (s)')
                currDf = facingDf[who2who]
                summedSeries = currDf.sum(axis=1)
                frameCount = len(summedSeries[summedSeries > 0])
                outputRow.append(round((frameCount/ fps), 2))
                columnCounter += 1
        outputRow.insert(0, filename)
        try:
            outputDf.loc[len(outputDf)] = outputRow
        except (ValueError, UnboundLocalError):
            outputDf = pd.DataFrame(columns=[outPutHeaders])
            outputDf.loc[len(outputDf)] = outputRow


    outputDf.to_csv(os.path.join(projectPath, 'logs', 'Direction_data_' + str(dateTimes) + '.csv'), index=False)
    print('Summary directionality statistics saved in project_folder/logs/' + str('Direction_data_' + str(dateTimes) + '.csv'))