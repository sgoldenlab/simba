import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from deepposekit import Annotator
import cv2
import numpy as np
import warnings
from configparser import ConfigParser
import os
warnings.filterwarnings('ignore')

def dpkAnnotator(dpkini,annotationfile):

    config = ConfigParser()
    configFile = str(dpkini)
    config.read(configFile)
    project_path = config.get('general DPK settings', 'project_folder')
    annotationsPath = annotationfile
    bodyPartsListPath = os.path.join(project_path, 'skeleton.csv')

    app = Annotator(datapath=annotationsPath, dataset='images', skeleton=bodyPartsListPath, shuffle_colors=False, text_scale=1)
    im = np.zeros((300, 600, 3))
    cv2.putText(im, 'Instructions', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
    cv2.putText(im, '+- = rescale image by +/- 10%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'left mouse button = move active keypoint to cursor location', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'WASD = move active keypoint 1px or 10px', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'JL = next or previous image', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, '<> = jump 10 images forward or backward', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'I,K or tab, shift+tab = switch active keypoint', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'R = mark image as unannotated ("reset")', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'F = mark image as annotated ("finished")', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'esc or Q = quit', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
    cv2.putText(im, 'Tap tab to begin', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Instructions', im)
    k = cv2.waitKey(0)
    while (1):
        cv2.imshow('Instructions', im)
        k = cv2.waitKey(0)
        app.run()
        if k == 27:  # Esc key to stop
            print('Annotatations saved in: ' + str(annotationfile))
            break