__author__ = "Simon Nilsson", "JJ Choong"

import os
import glob
import numpy as np
import pandas as pd
import cv2
import random
import imutils

def define_new_pose_configuration(configName, noAnimals, noBps, Imagepath, BpNameList, animalNumber, animal_id_number_list):
    global ix, iy
    global centerCordStatus

    def draw_circle(event,x,y,flags,param):
        global ix,iy
        global centerCordStatus
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            if centerCordStatus == False:
                cv2.circle(overlay,(x,y-sideImageHeight),10,colorList[-i],-1)
                cv2.putText(overlay,str(bpNumber+1), (x+4,y-sideImageHeight), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorList[i], 2)
                cv2.imshow('Define pose', overlay)
                centerCordStatus = True
    im = cv2.imread(Imagepath)
    imHeight, imWidth = im.shape[0], im.shape[1]
    if imWidth < 300:
        im = imutils.resize(im, width=800)
        imHeight, imWidth = im.shape[0], im.shape[1]
        im = np.uint8(im)
    fontScale = max(imWidth, imHeight) / (max(imWidth, imHeight) * 1.2)
    cv2.namedWindow('Define pose', cv2.WINDOW_NORMAL)
    overlay = im.copy()
    colorList = []
    if int(noAnimals) > 1:
        for value, bp_name_animal_number in enumerate(zip(BpNameList, animal_id_number_list)):
            BpNameList[value] = bp_name_animal_number[0] + '_' + str(bp_name_animal_number[1])


    for color in range(len(BpNameList)):
        r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colorTuple = (r, g, b)
        colorList.append(colorTuple)
    for i in range(len(BpNameList)):
        cv2.namedWindow('Define pose', cv2.WINDOW_NORMAL)
        centerCordStatus = False
        bpNumber = i
        sideImage = np.zeros((100, imWidth, 3), np.uint8)
        sideImageHeight, sideImageWidth = sideImage.shape[0], sideImage.shape[1]
        cv2.putText(sideImage, 'Double left click ' + BpNameList[i] + '. Press ESC to continue.', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale, colorList[i], 2)
        ix, iy = -1, -1
        while (1):
            cv2.setMouseCallback('Define pose', draw_circle)
            imageConcat = cv2.vconcat([sideImage, overlay])
            cv2.imshow('Define pose', imageConcat)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                cv2.destroyWindow('Define pose')
                break

    overlay = cv2.resize(overlay, (250,300))
    scriptdir = os.path.dirname(__file__)
    imagePath = os.path.join(scriptdir, 'pose_configurations', 'schematics')
    namePath = os.path.join(scriptdir, 'pose_configurations', 'configuration_names', 'pose_config_names.csv')
    bpPath = os.path.join(scriptdir, 'pose_configurations', 'bp_names', 'bp_names.csv')
    noAnimalsPath = os.path.join(scriptdir, 'pose_configurations', 'no_animals', 'no_animals.csv')
    imageNos = len(glob.glob(imagePath + '/*.png'))
    newImageName = 'Picture' + str(imageNos+1) + '.png'
    imageOutPath = os.path.join(imagePath, newImageName)
    print(imageOutPath)
    BpNameList = ",".join(BpNameList)

    # body_part_df = pd.DataFrame(BpNameList)
    # body_part_df.to_csv(bpPath, index=False, header=False)

    with open(namePath, 'a') as fd:
        fd.write(configName + '\n')
    with open(bpPath, 'a') as fd:
        fd.write(BpNameList + '\n')
    with open(noAnimalsPath, 'a') as fd:
        fd.write(str(animalNumber) + '\n')
    cv2.imwrite(imageOutPath, overlay)















