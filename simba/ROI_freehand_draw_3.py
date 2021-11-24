from configparser import ConfigParser
import os
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import warnings
from simba.drop_bp_cords import get_fn_ext

def roiFreehand(inifile, currVid):
    global centroids
    global moveStatus
    global ix, iy
    global centerCordStatus
    global coordChange
    global insertStatus
    global euclidPxDistance
    global centroids, toRemoveShapeName, removeStatus, toRemoveShape
    global toRemoveShapeName, removeStatus
    global firstLoop
    global recWidth, recHeight, firstLoop, polygonDf
    global rectangularDf, circleDf, polygonDf

    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
    pd.options.mode.chained_assignment = None

    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    vidInfPath = config.get('General settings', 'project_path')
    videofilesFolder = os.path.join(vidInfPath, "videos")
    logFolderPath = os.path.join(vidInfPath, 'logs')
    rectangularDf = pd.DataFrame(columns=['Video', "Shape_type", "Name", "width", "height", "topLeftX", "topLeftY"])
    circleDf = pd.DataFrame(columns=['Video', "Shape_type", "Name", "centerX", "centerY", "radius"])
    polygonDf = pd.DataFrame(columns=['Video', "Shape_type", "Name", "vertices"])
    outPutPath = os.path.join(logFolderPath, 'measures')
    storePath = os.path.join(outPutPath, 'ROI_definitions.h5')
    circleCordList = []
    polygonVertList = []

    def draw_circle(event,x,y,flags,param):
        global ix,iy
        global centerCordStatus
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            cv2.circle(overlay,(x,y),DrawScale,(144,0,255),-1)
            cv2.imshow('Define shape', overlay)
            circleCordList.append(x)
            circleCordList.append(y)
            if len(circleCordList) >= 3:
                euclidPxDistance = int((np.sqrt((circleCordList[2] - circleCordList[0]) ** 2 + (circleCordList[3] - circleCordList[1]) ** 2)))
                cv2.circle(overlay, (circleCordList[0], circleCordList[1]), int(euclidPxDistance), (144, 0, 255), DrawScale)
                cv2.imshow('Define shape', overlay)
                centerCordStatus = True

    def draw_polygon_vertices(event,x,y,flags,param):
        global ix,iy
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            cv2.circle(overlay,(x,y),DrawScale,(0,255,255),-1)
            cv2.imshow('Define shape', overlay)
            verticeTuple = (x,y)
            polygonVertList.append(verticeTuple)

    def select_cord_to_change(event,x,y,flags,param):
        global moveStatus
        global coordChange
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            if (x>=(circleCordList[0]-20)) and (x<=(circleCordList[0]+20)) and (y>=(circleCordList[1]-20)) and (y<=(circleCordList[1]+20)): #change point1
                coordChange = [1, circleCordList[0], circleCordList[1]]
                moveStatus = True
            if (x >= (circleCordList[2] - 20)) and (x <= (circleCordList[2] + 20)) and (y >= (circleCordList[3] - 20)) and (y <= (circleCordList[3] + 20)):  # change point2
                coordChange = [2, circleCordList[2], circleCordList[3]]
                moveStatus = True

    def select_new_dot_location(event, x, y, flags, param):
        global insertStatus
        global euclidPxDistance
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            newCordList.append(x)
            newCordList.append(y)
            insertStatus = True

    def select_shape_to_change(event, x, y, flags, param):
        global centroids, toRemoveShapeName, removeStatus, toRemoveShape
        global rectangularDf, circleDf, polygonDf
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            toRemove = centroids[(centroids['CenterX'] <= x+20) & (centroids['CenterX'] >= x-20) & (centroids['CenterY'] <= y+20) & (centroids['CenterY'] >= y-20)]
            toRemoveShapeName, toRemoveShape = (toRemove["Name"].iloc[0], toRemove["Shape"].iloc[0])
            updateImage(centroids, toRemoveShapeName, rectangularDf, circleDf, polygonDf)
            removeStatus = False

    def select_new_shape_centroid_loc(event, x, y, flags, param):
        global toRemoveShapeName, removeStatus
        global rectangularDf, circleDf, polygonDf
        newVertices = []
        if (event == cv2.EVENT_LBUTTONDBLCLK):
            toInsertShapeName, toInsertShape = toRemoveShapeName, toRemoveShape
            toRemoveShapeName = ''
            if toInsertShape == 'rectangle':
                newTopLeftX, newTopLeftY = (int(x -(recWidth/2)), int((y -(recHeight/2))))
                rectangularDf.loc[(rectangularDf['Name'] == toInsertShapeName), 'topLeftX'] = newTopLeftX
                rectangularDf.loc[(rectangularDf['Name'] == toInsertShapeName), 'topLeftY'] = newTopLeftY
            if toInsertShape == 'circle':
                newcenterX, newcenterY = (int(x), int(y))
                circleDf.loc[(circleDf['Name'] == toInsertShapeName), 'centerX'] = newcenterX
                circleDf.loc[(circleDf['Name'] == toInsertShapeName), 'centerY'] = newcenterY
            if toInsertShape == 'polygon':
                newPolyCentroidX, newPolyCentroidY = (int(x), int(y))
                OldCentroidX = int(centroids.loc[(centroids['Name'] == toInsertShapeName), 'CenterX'].unique())
                OldCentroidY = int(centroids.loc[(centroids['Name'] == toInsertShapeName), 'CenterY'].unique())
                verticeDifferenceX = OldCentroidX - (newPolyCentroidX)
                verticeDifferenceY = OldCentroidY - (newPolyCentroidY)
                oldVertices = polygonDf.loc[(polygonDf['Name'] == toInsertShapeName), 'vertices']
                oldVertices = [num for elem in oldVertices for num in elem]
                oldVerticeX = [item[0] for item in oldVertices]
                oldVerticeY = [item[1] for item in oldVertices]
                newX = [ppp - verticeDifferenceX for ppp in oldVerticeX]
                newY = [ppp - verticeDifferenceY for ppp in oldVerticeY]
                for i in range(len(newX)):
                    coord = [newX[i], newY[i]]
                    newVertices.append(coord)
                polygonDf.loc[(polygonDf['Name'] == toInsertShapeName), 'vertices'] = [newVertices]
            removeStatus = True
        updateImage(centroids, toRemoveShapeName, rectangularDf, circleDf, polygonDf)


    cap = cv2.VideoCapture(currVid)
    cap.set(1, 0)
    ret, frame = cap.read()
    fileName = str(0) + str('.bmp')
    filePath = os.path.join(videofilesFolder, fileName)
    cv2.imwrite(filePath,frame)
    img = cv2.imread(filePath)
    DrawScale = int(max(img.shape[0], img.shape[1]) / 120)
    _, CurrVidName, ext = get_fn_ext(currVid)
    instructionHeight, instructionWidth = (400, 1000)
    ROIcoordinatesPath = os.path.join(logFolderPath, 'measures', 'ROI_definitions.h5')

    ### CHECK IF ROI DEFINITIONS EXIST
    try:
        rectanglesInfo = pd.read_hdf(ROIcoordinatesPath, key='rectangles')
        circleInfo = pd.read_hdf(ROIcoordinatesPath, key='circleDf')
        polygonInfo = pd.read_hdf(ROIcoordinatesPath, key='polygons')
        rectangularDf = rectanglesInfo.loc[rectanglesInfo['Video'] == str(CurrVidName)]
        circleDf = circleInfo.loc[circleInfo['Video'] == str(CurrVidName)]
        polygonDf = polygonInfo.loc[polygonInfo['Video'] == str(CurrVidName)]
        inputRect, inputCirc, inputPoly = (rectanglesInfo.copy(), circleInfo.copy(), polygonInfo.copy())
        inputRect, inputCirc, inputPoly = (inputRect[inputRect["Video"] != CurrVidName], inputCirc[inputCirc["Video"] != CurrVidName], inputPoly[inputPoly["Video"] != CurrVidName])
        ROIdefExist = True
    except FileNotFoundError:
        ROIdefExist = False
        vidROIDefs = False
        inputRect = pd.DataFrame(columns=['Video', "Shape_type", "Name", "width", "height", "topLeftX", "topLeftY"])
        inputCirc = pd.DataFrame(columns=['Video', "Shape_type", "Name", "centerX", "centerY", "radius"])
        inputPoly = pd.DataFrame(columns=['Video', "Shape_type", "Name", "vertices"])

    ### CHECK IF CURRENT VIDEO DEFINITIONS EXIST
    if ROIdefExist is True:
        if (len(rectangularDf) == 0 and len(circleDf) == 0 and len(polygonDf) == 0):
            vidROIDefs = False
        else:
            vidROIDefs = True

    if (ROIdefExist is False) or (vidROIDefs is False):
        ROIindexPath = os.path.join(logFolderPath, 'measures', 'ROI_index.h5')
        rect2Draw = pd.read_hdf(ROIindexPath, key='rectangles')
        circ2Draw = pd.read_hdf(ROIindexPath, key='circleDf')
        pol2draw = pd.read_hdf(ROIindexPath, key='polygons')
        ##### RECTANGLES ####
        for index, row in rect2Draw.iterrows():
            shapeType = 'rectangle'
            currRectangleName = row['Name']
            im = np.zeros((instructionHeight, instructionWidth, 3))
            cv2.putText(im, str(CurrVidName), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
            cv2.putText(im, 'Draw rectangle: ' + str(currRectangleName), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(im, 'Instructions', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(im, str('Press and hold left mouse button at the top right corner of the rectangle ROI'), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Drag the mouse to the bottom right corner of the rectangle ROI'), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, 'Repeat this to redo the rectangle "' + str(currRectangleName) + '" ROI', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Press ESC to start, and press ESC twice when happy with rectangle "' + str(currRectangleName) + '" ROI'), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            while (1):
                cv2.imshow('Instructions', im)
                k = cv2.waitKey(0)
                if k==27:    # Esc key to stop
                    break

            cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
            while (1):
                ROI = cv2.selectROI('Define shape', img)
                width = (abs(ROI[0] - (ROI[2] + ROI[0])))
                height = (abs(ROI[2] - (ROI[3] + ROI[2])))
                topLeftX = ROI[0]
                topLeftY = ROI[1]
                cv2.rectangle(img, (topLeftX, topLeftY), (topLeftX+width, topLeftY+height), (255, 0, 0), DrawScale)
                k = cv2.waitKey(0)
                boxList = [CurrVidName, shapeType, currRectangleName, width, height, topLeftX, topLeftY]
                if (k == 27) or (k==47):  # Esc key to stop
                    rectangularDf = rectangularDf.append(pd.Series(dict(zip(rectangularDf.columns, boxList))), ignore_index=True)
                    cv2.destroyWindow('Define shape')
                    break

        ##### CIRCLES ####
        for index, row in circ2Draw.iterrows():
            shapeType = 'circle'
            centerCordStatus = False
            moveStatus = False
            insertStatus = False
            changeLoop = False
            centerCordStatus = False
            currCircleName = row['Name']
            im = np.zeros((550, 1000, 3))
            cv2.putText(im, str(CurrVidName), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(144, 0, 255), 2)
            cv2.putText(im, 'Draw circle: ' + str(currCircleName), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (144, 0, 255), 2)
            cv2.putText(im, 'Instructions', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Double left click to specify the center of the circle'), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Next, double left click to specify the outer bounds of the circle'), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('To change circle center location, first double left click on the circle center,'), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,255), 2)
            cv2.putText(im, str('and then double left click in the new center location.'), (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('To change circle radius, first double left click on the circle outer bound,'), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('and then double left click at the new circle outer bound.'), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Press ESC or Enter to start or to continue'), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            while (1):
                cv2.imshow('Instructions', im)
                k = cv2.waitKey(33)
                if k==27:
                    break
            overlay = img.copy()
            origImage = img.copy()
            circleCordList = newCordList = []
            ix, iy = -1, -1

            cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
            while (1):
                if centerCordStatus == False and (moveStatus == False) and (insertStatus == False):
                    cv2.setMouseCallback('Define shape',draw_circle)
                    cv2.imshow('Define shape', overlay)
                    k = cv2.waitKey(20) & 0xFF
                    if k == 27:
                        break
                if centerCordStatus == True and (moveStatus == False) and (insertStatus == False):
                    if changeLoop == True:
                        overlay = origImage.copy()
                        cv2.circle(overlay, (circleCordList[0], circleCordList[1]), DrawScale, (144, 0, 255), -1)
                        cv2.circle(overlay, (circleCordList[2], circleCordList[3]), DrawScale, (144, 0, 255), -1)
                        cv2.circle(overlay, (circleCordList[0], circleCordList[1]), euclidPxDistance, (144, 0, 255), DrawScale)
                    euclidPxDistance = int((np.sqrt((circleCordList[2] - circleCordList[0]) ** 2 + (circleCordList[3] - circleCordList[1]) ** 2)))
                    circleCordAppend = [CurrVidName, shapeType, currCircleName, circleCordList[0], circleCordList[1], euclidPxDistance]
                    cv2.imshow('Define shape', overlay)
                    cv2.setMouseCallback('Define shape', select_cord_to_change)
                if (moveStatus == True) and (insertStatus == False):
                    if changeLoop == True:
                        img = origImage.copy()
                        changeLoop = False
                    if coordChange[0] == 1:
                        cv2.circle(img, (circleCordList[2], circleCordList[3]), DrawScale, (144, 0, 255), -1)
                    if coordChange[0] == 2:
                        cv2.circle(img, (circleCordList[0], circleCordList[1]), DrawScale, (144, 0, 255), -1)
                    euclidPxDistance = int((np.sqrt((circleCordList[2] - circleCordList[0]) ** 2 + (circleCordList[3] - circleCordList[1]) ** 2)))
                    cv2.circle(img, (circleCordList[0], circleCordList[1]), euclidPxDistance, (144, 0, 255), DrawScale)
                    circleCordAppend = [CurrVidName, shapeType, currCircleName, circleCordList[0], circleCordList[1], euclidPxDistance]
                    cv2.imshow('Define shape', img)
                    cv2.setMouseCallback('Define shape', select_new_dot_location)
                if (insertStatus == True):
                    if coordChange[0] == 1:
                        cv2.circle(img, (circleCordList[2], circleCordList[3]), DrawScale, (144, 0, 255), -1)
                        cv2.circle(img, (newCordList[-2], newCordList[-1]), DrawScale, (144, 0, 255), -1)
                        cv2.circle(img, (circleCordList[0], circleCordList[1]), euclidPxDistance, (144, 0, 255), DrawScale)
                        circleCordList = [newCordList[-2], newCordList[-1], circleCordList[2], circleCordList[3]]
                        circleCordAppend = [CurrVidName, shapeType, currCircleName, circleCordList[0], circleCordList[1],euclidPxDistance]
                        centerCordStatus = True
                        moveStatus = False
                        insertStatus = False
                        changeLoop = True
                    if coordChange[0] == 2:
                        cv2.circle(img, (circleCordList[0], circleCordList[1]), DrawScale, (144, 0, 255), -1)
                        cv2.circle(img, (newCordList[-2], newCordList[-1]), DrawScale, (144, 0, 255), -1)
                        euclidPxDistance = int((np.sqrt((circleCordList[1] - newCordList[-1]) ** 2 + (circleCordList[0] - newCordList[-2]) ** 2)))
                        cv2.circle(img, (circleCordList[0], circleCordList[1]), euclidPxDistance, (144, 0, 255), DrawScale)
                        circleCordList = [circleCordList[0], circleCordList[1], newCordList[-2], newCordList[-1]]
                        circleCordAppend = [CurrVidName, shapeType, currCircleName, circleCordList[0], circleCordList[1], euclidPxDistance]
                        centerCordStatus = True
                        moveStatus = False
                        insertStatus = False
                        changeLoop = True
                    cv2.imshow('Define shape', img)
                img = overlay.copy()
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    circleDf = circleDf.append(pd.Series(dict(zip(circleDf.columns, circleCordAppend))), ignore_index=True)
                    cv2.destroyWindow('Define shape')
                    break

        ##### POLYGONS ####
        for index, row in pol2draw.iterrows():
            currPolygonName = row['Name']
            shapeType = 'polygon'
            im = np.zeros((550, 1000, 3))
            cv2.putText(im, str(CurrVidName), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(im, 'Draw polygon: ' + str(currPolygonName), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7 + 0.2, (0, 255, 255), 2)
            cv2.putText(im, 'Instructions', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Double left click at at least 3 outer bounds of the polygon'), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im, str('Press ESC start or to continue'), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            while (1):
                cv2.imshow('Instructions', im)
                k = cv2.waitKey(0)
                if k==27:
                    break
            overlay = img.copy()
            polyGonListOfLists = []
            polygonVertList = []
            cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
            while (1):
                cv2.setMouseCallback('Define shape', draw_polygon_vertices)
                cv2.imshow('Define shape', overlay)
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    if len(polygonVertList) >= 3:
                        break
                    else:
                        pass
                #    break
                #break
            polyGon = Polygon(polygonVertList)
            polyX, polyY = polyGon.exterior.coords.xy
            for i in range(len(polyX)):
                polyGonListOfLists.append([polyX[i], polyY[i]])
            polyGonListtoDf = [CurrVidName, shapeType, currPolygonName, polyGonListOfLists]
            polyGonListOfLists = np.array(polyGonListOfLists, np.int32)
            polyGonListOfLists = polyGonListOfLists.reshape((-1, 1, 2))
            polyLine = cv2.convexHull(polyGonListOfLists)
            cv2.drawContours(overlay, [polyLine.astype(int)], 0, (0, 255, 255), DrawScale)
            cv2.imshow('Define shape', overlay)
            polygonDf = polygonDf.append(pd.Series(dict(zip(polygonDf.columns, polyGonListtoDf))), ignore_index=True)
            img = overlay.copy()
            cv2.imshow('Define shape', overlay)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # Esc key to stop
                cv2.destroyWindow('Define shape')
                cv2.destroyWindow('Instructions')
              #  break
          # break
        cv2.destroyWindow('Define shape')
        cv2.destroyWindow('Instructions')
        storePath = os.path.join(outPutPath, 'ROI_definitions.h5')
        store = pd.HDFStore(storePath, mode='w')
        rec = pd.concat([rectangularDf, inputRect], ignore_index=True, sort=False)
        circ = pd.concat([circleDf, inputCirc], ignore_index=True, sort=False)
        polyg = pd.concat([polygonDf, inputPoly], ignore_index=True, sort=False)
        store['rectangles'] = rec
        store['circleDf'] = circ
        store['polygons'] = polyg
        print('ROI definitions saved in ' + str(storePath))
        print('ROI definitions saved in ' + 'project_folder\logs\measures\ROI_definitions.h5')
        store.close()

        delImagePath = os.path.join(videofilesFolder, '0.bmp')
        os.remove(delImagePath)

    def updateImage(centroids, toRemoveShapeName, rectangularDf, circleDf, polygonDf):
        global recWidth, recHeight, firstLoop
        overlay = img.copy()
        centroids = centroids.drop_duplicates()
        for rectangle in range(len(rectangularDf)):
            videoName, recName, topLeftX, topLeftY = (rectangularDf['Video'].iloc[rectangle], rectangularDf['Name'].iloc[rectangle], rectangularDf['topLeftX'].iloc[rectangle],rectangularDf['topLeftY'].iloc[rectangle])
            if recName == toRemoveShapeName:
                recWidth, recHeight = (rectangularDf['width'].iloc[rectangle], rectangularDf['height'].iloc[rectangle])
                continue
            else:
                bottomRightX, bottomRightY = (topLeftX + rectangularDf['width'].iloc[rectangle], topLeftY + rectangularDf['height'].iloc[rectangle])
                cv2.rectangle(overlay, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (255, 0, 0), DrawScale)
                centerOfShape = [(topLeftX + bottomRightX) / 2, (topLeftY + bottomRightY) / 2]
                cv2.circle(overlay, (int(centerOfShape[0]), int(centerOfShape[1])),DrawScale, (255, 0, 0), -1)
                if firstLoop == True:
                    CentroidList = [CurrVidName, 'rectangle', recName, centerOfShape[0], centerOfShape[1]]
                    centroids = centroids.append(pd.Series(dict(zip(centroids.columns, CentroidList))), ignore_index=True)
                else:
                    centroids.loc[(centroids['Name'] == recName), 'CenterX'] = centerOfShape[0]
                    centroids.loc[(centroids['Name'] == recName), 'CenterY'] = centerOfShape[1]
                    correctionMask = (rectangularDf['Name'] == recName) & (rectangularDf['Video'] == videoName)
                    rectangularDf['topLeftX'][correctionMask], rectangularDf['topLeftY'][correctionMask] = topLeftX, topLeftY
                    #recListtoDf = [videoName, "ractangle", recName, recWidth, recHeight, topLeftX, topLeftY]
                    #rectangularDf = rectangularDf.append(pd.Series(dict(zip(rectangularDf.columns, recListtoDf))), ignore_index=True)
        for circle in range(len(circleDf)):
            videoName, circleName, centerX, centerY, radius = (circleDf['Video'].iloc[circle], circleDf['Name'].iloc[circle], circleDf['centerX'].iloc[circle], circleDf['centerY'].iloc[circle], circleDf['radius'].iloc[circle])
            if circleName == toRemoveShapeName:
                continue
            else:
                cv2.circle(overlay, (centerX, centerY), radius, (144, 0, 255), DrawScale)
                cv2.circle(overlay, (centerX, centerY), DrawScale, (144, 0, 255), -1)
                if firstLoop == True:
                    CentroidList = [CurrVidName, 'circle', circleName, centerX, centerY]
                    centroids = centroids.append(pd.Series(dict(zip(centroids.columns, CentroidList))), ignore_index=True)
                else:
                    centroids.loc[(centroids['Name'] == circleName), 'CenterX'] = centerX
                    centroids.loc[(centroids['Name'] == circleName), 'CenterY'] = centerY
                    correctionMask = (circleDf['Name'] == circleName) & (circleDf['Video'] == videoName)
                    circleDf['centerX'][correctionMask], circleDf['centerY'][correctionMask] = centerX, centerY
        #for polygon in range(len(polygonDf)):
        for index, row in polygonDf.iterrows():
            videoName, polyName, inputVertices = (row['Video'], row['Name'], row['vertices'])
            if polyName == toRemoveShapeName:
                continue
            else:
                sumofX, sumofY = (int(sum([item[0] for item in inputVertices])), int(sum([item[1] for item in inputVertices])))
                PolyCentroidX, PolyCentroidY = (int(sumofX / len(inputVertices)), int(sumofY / len(inputVertices)))
                cv2.circle(overlay, (PolyCentroidX, PolyCentroidY), DrawScale, (0,255,255), -1)
                vertices = np.array(inputVertices, np.int32)
                cv2.polylines(overlay, [vertices], True, (0, 255, 255), thickness=DrawScale)
                if firstLoop == True:
                    CentroidList = [CurrVidName, 'polygon', polyName, PolyCentroidX, PolyCentroidY]
                    centroids = centroids.append(pd.Series(dict(zip(centroids.columns, CentroidList))), ignore_index=True)
                else:
                    centroids.loc[(centroids['Name'] == polyName), 'CenterX'] = PolyCentroidX
                    centroids.loc[(centroids['Name'] == polyName), 'CenterY'] = PolyCentroidY
                    polygonDf = polygonDf.drop(polygonDf[(polygonDf['Video'] == videoName) & (polygonDf['Name'] == polyName)].index)
                    polyGonListtoDf = [videoName, 'polygon', polyName, inputVertices]
                    polygonDf = polygonDf.append(pd.Series(dict(zip(polygonDf.columns, polyGonListtoDf))), ignore_index=True)
        firstLoop = False
        cv2.imshow('Define shape', overlay)
        return centroids

    if (ROIdefExist is True) and (vidROIDefs is True):
        firstLoop = True
        cap = cv2.VideoCapture(currVid)
        cap.set(1, 0)
        ret, frame = cap.read()
        fileName = str(0) + str('.bmp')
        filePath = os.path.join(videofilesFolder, fileName)
        cv2.imwrite(filePath, frame)
        img = cv2.imread(filePath)
        overlay = img.copy()
        _, CurrVidName, ext = get_fn_ext(currVid)
        centroids = pd.DataFrame(columns=['Video', "Shape", "Name", "CenterX", "CenterY"])
        toRemoveShapeName = ''
        removeStatus = True
        ix, iy = -1, -1
        im = np.zeros((400, 1000, 3))
        cv2.putText(im, 'Move shapes for ' + str(CurrVidName), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(im, 'Instructions', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(im, str('Double left click on the centroid of the shape you wish to move'), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(im, str('Then double click in the new centroid location'), (10, 160), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)
        cv2.putText(im, str('Press ESC to start and when finished'), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        while (1):
            cv2.imshow('Instructions', im)
            k = cv2.waitKey(0)
            if k == 27:  # Esc key to stop
                break
        while (1):
            cv2.namedWindow('Define shape', cv2.WINDOW_NORMAL)
            centroids = updateImage(centroids, toRemoveShapeName, rectangularDf, circleDf, polygonDf)
            if removeStatus == True:
                cv2.setMouseCallback('Define shape', select_shape_to_change)
            if removeStatus == False:
                cv2.setMouseCallback('Define shape', select_new_shape_centroid_loc)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                cv2.destroyWindow('Define shape')
                cv2.destroyWindow('Instructions')
                break
        cv2.destroyWindow('Define shape')
        cv2.destroyWindow('Instructions')
        store = pd.HDFStore(storePath, mode='w')
        rec = pd.concat([rectangularDf, inputRect], ignore_index=True, sort=False)
        circ = pd.concat([circleDf, inputCirc], ignore_index=True, sort=False)
        polyg = pd.concat([polygonDf, inputPoly], ignore_index=True, sort=False)
        store['rectangles'] = rec
        store['circleDf'] = circ
        store['polygons'] = polyg
        print('ROI definitions saved in ' + 'project_folder\logs\measures\ROI_definitions.h5')
        store.close()

        delImagePath = os.path.join(videofilesFolder, '0.bmp')
        os.remove(delImagePath)





