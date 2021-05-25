import pandas as pd
from tkinter import *
import pandas as pd
from PIL import Image, ImageTk, ImageOps
import os
from tkinter import filedialog
from configparser import ConfigParser, NoSectionError, NoOptionError
from subprocess import *
import re
from simba.rw_dfs import *
from simba.drop_bp_cords import *
import numpy as np
from pylab import cm
import cv2


'''
parameters: inifile should be a .ini file, framedir should be a directory with a type(str), targets=list  
'''

def semisuperviseLabel(inifile,framedir,targets,threshold_list):
    print(targets, threshold_list)
    frameDirBaseName = os.path.basename(framedir).split('.')[0]
    print(frameDirBaseName)
    projectpath = os.path.dirname(inifile)
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    csvfile = os.path.join(projectpath,'csv','machine_results',(frameDirBaseName)+'.' + wfileType)
    nameofvid = os.path.basename(csvfile).split('.' + wfileType)[0]
    animalsNo = config.getint('General settings', 'animal_no')
    poseEstimationBps = config.get('create ensemble settings', 'pose_estimation_body_parts')
    currDf = read_df(csvfile, wfileType)

    # Retrieves behavior name and preexisting values from the machine generated csv file
    ##read in csv file
    df = read_df(csvfile, wfileType)
    frames_dir = framedir
    threshold ={}
    for i in range(len(threshold_list)):
        name = threshold_list[i].labelname
        threshold[name]=float(threshold_list[i].entry_get)

    ## get dataframe of probability target columns
    df_prob_targets = []
    probabilityColumnNames = []
    df_targets = pd.DataFrame(columns=targets)
    for target in targets:
        probabilityColumnName = 'Probability_' + target
        df_prob_targets.append(df[[probabilityColumnName]].to_numpy())
        probabilityColumnNames.append(probabilityColumnName)
        df_targets[target] = [1 if x > threshold[target] else 0 for x in df[[probabilityColumnName]].to_numpy()] #change new df according to threshold

    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if (multiAnimalIDList[0] != '') and (poseEstimationBps == 'user_defined'):
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            print('Applying settings for classical tracking...')

    except NoSectionError:
        multiAnimalIDList = ['']
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    cmaps = ['spring', 'summer', 'autumn', 'cool', 'Wistia', 'Pastel1', 'Set1', 'winter']
    Xcols, Ycols, Pcols = getBpNames(inifile)
    cMapSize = int(len(Xcols)/animalsNo) + 1
    colorListofList = []
    for colormap in range(animalsNo):
        currColorMap = cm.get_cmap(cmaps[colormap], cMapSize)
        currColorList = []
        for i in range(currColorMap.N):
            rgb = list((currColorMap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            currColorList.append(rgb)
        colorListofList.append(currColorList)

    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, animalsNo, Xcols, Ycols, [], colorListofList)


    class MainInterface:
        def __init__(self,videopath,targets,df_targets,video_name,inifile,maindf):
            padding = 5
            #instances
            self.targets = targets
            self.df_targets = df_targets
            self.video_name =video_name
            self.inifile = inifile
            self.maindf = maindf
            ## defining starting frames x parameters
            self.currentframeNo = 0
            self.videopath = videopath
            ## getting all the frames froom the directory
            cap = cv2.VideoCapture(videopath)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            del cap
            self.framesin = list(range(0, length))

            #sort the frame
            # self.framesin.sort(key=lambda f: int(re.sub('\D','',f)))

            ## making new top level
            self.window = Toplevel()
            folder = Frame(self.window)

            #Advancing Buttons ( Frames and button to navigate)
            self.button_frame = Frame(self.window, bd=2, width=700, height=300)
            #entry box to show frame number
            n = StringVar(self.window, value=self.currentframeNo)
            self.fbox = Entry(self.button_frame, width=7, textvariable=n)

            self.frameNumber = Label(self.button_frame, text="Frame number")
            self.forward = Button(self.button_frame, text=">", command = lambda:self.advance_frame(self.currentframeNo + 1))
            self.back = Button(self.button_frame, text="<", command= lambda: self.advance_frame(self.currentframeNo -1))
            self.forwardmax = Button(self.button_frame,text=">>", command = lambda :self.advance_frame(len(self.framesin)-1))
            self.backmax = Button(self.button_frame,text="<<", command = lambda :self.advance_frame(0))
            self.select = Button(self.button_frame, text="Jump to selected frame", command=lambda:self.advance_frame(int(self.fbox.get())))


            ## define first frames
            max_size = 1080, 650
            cap = cv2.VideoCapture(videopath)
            cap.set(1, 0)
            res, imageIn = cap.read()
            cap.release()

            current_image = Image.fromarray(imageIn)
            current_image.thumbnail(max_size, Image.ANTIALIAS)
            current_frame = ImageTk.PhotoImage(master=self.window, image=current_image)

            self.video_frame = Label(self.window, image=current_frame)
            self.video_frame.image = current_frame
            self.video_frame.grid(row=0, column=0)

            ## Jump widget
            self.jump_frame = Frame(self.window)
            self.jump = Label(self.jump_frame, text="Jump Size:")
            self.jump_size = Scale(self.jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
            self.jump_size.set(0)
            self.jump_back = Button(self.jump_frame, text="<<", command=lambda:self.advance_frame(int(self.fbox.get()) - self.jump_size.get()))
            self.jump_forward = Button(self.jump_frame, text=">>",command=lambda: self.advance_frame(int(self.fbox.get()) + self.jump_size.get()))

            ## organize
            #advancing button
            folder.grid(row=0, column=1, sticky=N)
            self.button_frame.grid(row=1, column=0)
            self.frameNumber.grid(row=0, column=1)
            self.forward.grid(row=1, column=3, sticky=E, padx = padding)
            self.back.grid(row=1, column=1, sticky=W, padx = padding)
            self.fbox.grid(row=1, column=1)
            self.forwardmax.grid(row=1,column=4,sticky=W, padx = padding)
            self.backmax.grid(row=1,column=0,sticky=W, padx = padding)
            self.select.grid(row=2, column=1, sticky=N)
            #jump
            self.jump_frame.grid(row=2, column=0)
            self.jump.grid(row=0, column=0, sticky=W)
            self.jump_size.grid(row=0, column=1, sticky=W)
            self.jump_back.grid(row=0, column=2, sticky=E)
            self.jump_forward.grid(row=0, column=3, sticky=W)

            ## Behavior Checkbox
            self.check_frame = Frame(self.window, bd=2, width=300, height=500)
            self.check_frame.grid(row=0, column=1)

            self.blabel = Label(self.check_frame, text="Check Behavior:")
            self.blabel.config(font=("Calibri", 16))
            self.blabel.grid(sticky=N)

            # Generates corresponding checkboxes according to config file
            #dictionary way
            self.checkVar ={}
            self.checkbox ={}
            count =0
            for i in self.targets:
                self.checkVar[i] = IntVar()
                self.checkbox[i] = Checkbutton(self.check_frame,text=i,variable=self.checkVar[i],command =lambda: self.saveBehavior(self.currentframeNo,self.targets))
                self.checkbox[i].grid(row=count + 1, sticky=W)
                self.checkVar[i].set(self.df_targets[i].iloc[self.currentframeNo])
                count+=1


            ## Saving a Range of Frames
            self.rangeOn = IntVar(value=0)
            self.rangeFrames = Frame(self.window)
            self.rangeFrames.grid(row=1, column=1, sticky=S)
            self.select_range = Checkbutton(self.rangeFrames, text='Frame range ', variable=self.rangeOn)
            self.select_range.grid(row=0, column=0, sticky=W)
            self.firstFrame = Entry(self.rangeFrames, width=7)
            self.firstFrame.grid(row=0, column=1, sticky=E)
            self.to_label = Label(self.rangeFrames, text=" to ")
            self.to_label.grid(row=0, column=2, sticky=E)
            self.lastFrame = Entry(self.rangeFrames, width=7)
            self.lastFrame.grid(row=0, column=3, sticky=E)

            ## Save Button
            save = Button(self.window, text="Save Range", command=lambda: self.saveRangeOfBehavior(int(self.firstFrame.get()),
                                                                                                   int(self.lastFrame.get())))
            save.grid(row=2, column=1, sticky=N)

            self.generate = Button(self.window, text="Save csv", command=lambda: self.save_video(self.maindf))
            self.generate.config(font=("Calibri", 16))
            self.generate.grid(row=10, column=1, sticky=N)

            ##video player
            video_player = Frame(self.window, width=100, height=100)
            video_player.grid(row=0, column=2, sticky=N)
            video = Button(video_player, text='Open Video', command=self.play_video)
            video.grid(sticky=N, pady=10)
            video_key = Label(video_player, text='\n\n  Keyboard shortcuts for video navigation: \n p = Pause/Play'
                                                 '\n\n After pressing pause:'
                                                 '\n o = +2 frames \n e = +10 frames \n w = +1 second'
                                                 '\n\n t = -2 frames \n s = -10 frames \n x = -1 second'
                                                 '\n\n q = Close video window \n\n')
            video_key.grid(sticky=W)
            update = Button(video_player, text='Show current video frame',
                            command= self.update_frame_from_video)
            update.grid(sticky=N)
            self.bind_keys()
            key_presses = Label(video_player,
                                text='\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame'
                                     '\n Left Arrow = -1 frame'
                                     '\n Ctrl + s = Save csv file'
                                     '\n Ctrl + l = Last frame'
                                     '\n Ctrl + o = First frame')
            key_presses.grid(sticky=S)

            # Detects user key presses

        def bind_keys(self):
            self.window.bind('<Control-s>', lambda x: self.save_video(self.maindf))
            self.window.bind('<Right>', lambda x: self.advance_frame(self.currentframeNo + 1))
            self.window.bind('<Left>', lambda x: self.advance_frame(self.currentframeNo - 1))
            self.window.bind('<Control-l>', lambda x: self.advance_frame(len(self.framesin)-1))
            self.window.bind('<Control-o>', lambda x: self.advance_frame(0))


        def play_video(self):
            script_directory = os.path.dirname(os.path.realpath(__file__))
            p = Popen('python ' + str(script_directory) + r"/play_video_pseudo.py", stdin=PIPE, stdout=PIPE, shell=True)
            main_project_dir = os.path.dirname(self.inifile)
            video_dir = os.path.join(main_project_dir, 'videos')
            video_list = os.listdir(video_dir)
            current_full_video_name = [i for i in video_list if self.video_name in i]
            try:
                current_full_video_name = current_full_video_name[0]
            except IndexError:
                print(
                    "Video not found in project_folder/videos, please make sure you have the video in the video folder")
            data = bytes(os.path.join(video_dir, str(current_full_video_name)), 'utf-8')
            p.stdin.write(data)
            p.stdin.close()
            path = os.path.join(main_project_dir,'subprocess.txt')
            with open(path, "w") as text_file:
                text_file.write(str(p.pid))

        # Updates and loads the frame corresponding to the frame the video is paused on
        def update_frame_from_video(self):
            f = open(os.path.join(os.path.dirname(self.inifile),'labelling_info.txt'),
                     'r+')
            os.fsync(f.fileno())
            vid_frame_no = int(f.readline())
            print(vid_frame_no)
            self.advance_frame(vid_frame_no)
            f.close()


        def save_video(self, dataframe):
            for i in self.targets:
                dataframe[i] = self.df_targets[i]
            dataframe = dataframe.drop(['Scaled_movement_M1','Scaled_movement_M2','Scaled_movement_M1_M2'], axis=1, errors='ignore')
            for i in self.targets:
                targetColName = 'Probability_' + i
                dataframe = dataframe.drop([targetColName], axis=1, errors='ignore')


            try:
                save_df(dataframe, wfileType, os.path.join(projectpath,'csv','targets_inserted',str(self.video_name) + '.csv'))
            except PermissionError:
                print('You don not have permission to save the annotation file - check that the file is not open in a different application. If you are working of a server make sure the file is not open on a different computer.')
            print('Files saved in',(os.path.join(projectpath,'csv','targets_inserted')))

        def saveBehavior(self, frame,targets):
            for target in targets:
                df_targets[target].loc[frame] = self.checkVar[target].get()

            # df_targets[target].loc[frame] = self.checkVar[target].get() ##get the frame numbre from the fbox and set it to equals to row in the df, then get 1 or 0

        def saveBehaviorMulti(self, frame,targets):
            df_targets[targets].loc[frame] = self.checkVar[targets].get()

        def saveRangeOfBehavior(self, start, end):
            if self.rangeOn.get():
                for i in range(start, end+1):
                    for j in targets:
                        self.saveBehaviorMulti(i,j)

        def advance_frame(self,frame):
            ## define max frames ( last frames)
            max_index_frames_in = len(self.framesin) - 1
            try:
                self.video_frame.destroy()
            except:
                pass

            try:
                if frame > max_index_frames_in:
                    print("Reached End of Frames")
                    self.currentframeNo = max_index_frames_in
                elif frame < 0:
                    self.currentframeNo = 0
                else:
                    self.currentframeNo = frame

                ## remove entry box (frame number) and refresh
                self.fbox.delete(0, END)
                self.fbox.insert(0, self.currentframeNo)

                for i in self.targets:
                    self.checkVar[i].set(self.df_targets[i].loc[self.currentframeNo])

                ## refresh the frames showed in tkinter
                max_size = 1080, 650
                IDlabelLoc = []

                cap = cv2.VideoCapture(self.videopath)
                cap.set(1, self.currentframeNo)
                res, imageIn = cap.read()
                cap.release()
                try:
                    maxResDimension = max(imageIn.shape[1], imageIn.shape[0])
                    mySpaceScale, myRadius, myResolution, myFontScale = 60, 12, 1500, 1.5
                    circleScale, fontScale, spacingScale = int(myRadius / (myResolution / maxResDimension)), float(myFontScale / (myResolution / maxResDimension)), int(mySpaceScale / (myResolution / maxResDimension))
                    if animalsNo > 1:
                        for animal in range(animalsNo):
                            currentDictID = list(animalBpDict.keys())[animal]
                            currentDict = animalBpDict[currentDictID]
                            currNoBps = len(currentDict['X_bps'])
                            IDappendFlag = False
                            for bp in range(currNoBps):
                                currXheader, currYheader, currColor = currentDict['X_bps'][bp], currentDict['Y_bps'][bp], currentDict['colors'][bp]
                                currAnimal = currDf.loc[currDf.index[self.currentframeNo], [currXheader, currYheader]]
                                if ('Centroid' in currXheader) or ('Center' in currXheader) or ('centroid' in currXheader) or ('center' in currXheader):
                                        IDlabelLoc.append([currentDictID, currAnimal[0], currAnimal[1], currentDict['colors'][bp]])
                                        IDappendFlag = True
                            if IDappendFlag == False:
                                IDlabelLoc.append([currentDictID, currAnimal[0], currAnimal[1], currentDict['colors'][bp]])
                        for currAnimal in IDlabelLoc:
                            cv2.putText(imageIn, str(currAnimal[0]), (int(currAnimal[1]), int(currAnimal[2])), cv2.FONT_HERSHEY_COMPLEX, fontScale, currAnimal[3], 4)
                except NameError:
                    pass

                for i in self.targets:
                    checkStatus = self.checkVar[i].get()
                    if checkStatus == 1:
                        imageIn = cv2.copyMakeBorder(imageIn, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,255])
                    break

                imageIn = cv2.cvtColor(imageIn, cv2.COLOR_BGR2RGB)
                current_image = Image.fromarray(imageIn)
                current_image.thumbnail(max_size, Image.ANTIALIAS)
                current_frame = ImageTk.PhotoImage(master=self.window, image=current_image)

                self.video_frame = Label(self.window, image='')
                self.video_frame.grid(row=0, column=0)

                self.video_frame.image = current_frame
                self.video_frame.config(image=current_frame)
                current_image.close()


            except IndexError:
                pass

    MainInterface(frames_dir,targets,df_targets,nameofvid,inifile,df)


