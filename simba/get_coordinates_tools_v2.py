import os
import cv2
import numpy as np

def get_coordinates_nilsson(filenames,knownmm):
    global cordStatus
    global moveStatus
    global insertStatus
    global changeLoop
    cordStatus = False
    moveStatus = False
    insertStatus = False
    changeLoop = False

    newCordList = []

    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        global ix, iy,cordStatus
        if (event == cv2.EVENT_LBUTTONDBLCLK) and len(cordList) < 4:
            cv2.circle(overlay, (x, y), circleScale, (144, 0, 255), -1)
            cordList.append(x)
            cordList.append(y)
            if len(cordList) == 4:
                cordStatus = True
                cv2.line(overlay, (cordList[0], cordList[1]), (cordList[2], cordList[3]), (144, 0, 255), 6)

    def select_cord_to_change(event, x, y, flags, param):
        global moveStatus
        global coordChange
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            if (x >= (cordList[0] - 20)) and (x <= (cordList[0] + 20)) and (y >= (cordList[1] - 20)) and (
                    y <= (cordList[1] + 20)):  # change point1
                coordChange = [1, cordList[0], cordList[1]]
                moveStatus = True
            if (x >= (cordList[2] - 20)) and (x <= (cordList[2] + 20)) and (y >= (cordList[3] - 20)) and (
                    y <= (cordList[3] + 20)):  # change point2
                coordChange = [2, cordList[2], cordList[3]]
                moveStatus = True

    def select_new_dot_location(event, x, y, flags, param):
        global insertStatus
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            newCordList.append(x)
            newCordList.append(y)
            insertStatus = True

    # extract one frame
    currentDir = str(os.path.dirname(filenames))
    videoName = str(os.path.basename(filenames))
    os.chdir(currentDir)
    try:
        cap = cv2.VideoCapture(videoName)
        cap.set(1, 0)
        ret, frame = cap.read()
        fileName = str(0) + str('.bmp')
        filePath = os.path.join(currentDir, fileName)
        cv2.imwrite(filePath, frame)
        img = cv2.imread(filePath)
        (imageHeight, imageWidth) = img.shape[:2]
    except AttributeError:
        print('ERROR: Make sure the video file ' + str(videoName) + ' is located in your project_folder/videos directory')
    maxResDimension = max(imageWidth, imageHeight)
    mySpaceScale, myRadius, myResolution, myFontScale = 80, 20, 1500, 1.5
    circleScale = int(myRadius / (myResolution / maxResDimension))
    fontScale = float(myFontScale / (myResolution / maxResDimension))
    spacingScale = int(mySpaceScale / (myResolution / maxResDimension))
    origImage = img.copy()
    overlay = img.copy()
    ix,iy = -1,-1
    cordList = []
    cv2.namedWindow('Select coordinates: double left mouse click at two locations. Press ESC when done',cv2.WINDOW_NORMAL)

    while(1):

        if cordStatus == False and (moveStatus == False) and (insertStatus == False):
            cv2.setMouseCallback('Select coordinates: double left mouse click at two locations. Press ESC when done', draw_circle)
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done', overlay)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        if (cordStatus == True) and (moveStatus == False) and (insertStatus == False):
            if changeLoop == True:
                overlay = origImage.copy()
                cv2.circle(overlay, (cordList[0], cordList[1]), circleScale, (144, 0, 255), -1)
                cv2.circle(overlay, (cordList[2], cordList[3]), circleScale, (144, 0, 255), -1)
                cv2.line(overlay, (cordList[0], cordList[1]), (cordList[2], cordList[3]), (144, 0, 255), int(circleScale/5))
            cv2.putText(overlay, 'Click on circle to move', (20, 30), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 0, 255), 2)
            cv2.putText(overlay, 'Press ESC to save and exit', (20, 50+spacingScale), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 0, 255), 2)
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done', overlay)
            cv2.setMouseCallback(
                'Select coordinates: double left mouse click at two locations. Press ESC when done',
                select_cord_to_change)
        if (moveStatus == True) and (insertStatus == False):
            if changeLoop == True:
                img = origImage.copy()
                changeLoop = False
            if coordChange[0] == 1:
                cv2.circle(img, (cordList[2], cordList[3]), circleScale, (144, 0, 255), -1)
            if coordChange[0] == 2:
                cv2.circle(img, (cordList[0], cordList[1]), circleScale, (144, 0, 255), -1)
            cv2.imshow('Select coordinates: double left mouse click at two locations. Press ESC when done', img)
            cv2.putText(img, 'Click on new circle location', (20, 30), cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 0, 255), 2)
            cv2.setMouseCallback(
                'Select coordinates: double left mouse click at two locations. Press ESC when done',
                select_new_dot_location)
        if (insertStatus == True):

            if coordChange[0] == 1:
                cv2.circle(img, (cordList[2], cordList[3]), circleScale, (144, 0, 255), -1)
                cv2.circle(img, (newCordList[-2], newCordList[-1]), circleScale, (144, 0, 255), -1)
                cv2.line(img, (cordList[2], cordList[3]), (newCordList[-2], newCordList[-1]), (144, 0, 255), int(circleScale/4))
                cordList = [newCordList[-2], newCordList[-1], cordList[2], cordList[3]]
                cordStatus = True
                moveStatus = False
                insertStatus = False
                changeLoop = True
            if coordChange[0] == 2:
                cv2.circle(img, (cordList[0], cordList[1]), circleScale, (144, 0, 255), -1)
                cv2.circle(img, (newCordList[-2], newCordList[-1]), circleScale, (144, 0, 255), -1)
                cv2.line(img, (cordList[0], cordList[1]), (newCordList[-2], newCordList[-1]), (144, 0, 255), int(circleScale/4))
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
    os.remove(fileName)
    euclidPixelDist = np.sqrt((cordList[0] - cordList[2]) ** 2 + (cordList[1] - cordList[3]) ** 2)
    mm_dist = int(knownmm)
    cordStatus = False
    moveStatus = False
    insertStatus = False
    changeLoop = False
    ppm = euclidPixelDist / mm_dist
    print('pixel per mm for video ' + '"' + str(videoName) + '" = ' + str(round(ppm,4)))
    return ppm