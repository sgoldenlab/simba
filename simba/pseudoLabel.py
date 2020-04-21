import pandas as pd
from tkinter import *
import pandas as pd
from PIL import Image, ImageTk
import os
from tkinter import filedialog
from configparser import ConfigParser
from subprocess import *
import numpy as np
'''
parameters: inifile should be a .ini file, framedir should be a directory with a type(str), targets=list  
'''

def semisuperviseLabel(inifile,framedir,targets,threshold_list):
    print(threshold_list)
    projectpath = os.path.dirname(inifile)
    csvfile = os.path.join(projectpath,'csv','machine_results',os.path.basename(framedir)+'.csv')
    nameofvid = os.path.basename(csvfile).split('.csv')[0]
    # Retrieves behavior name and preexisting values from the machine generated csv file
    ##read in csv file
    df = pd.read_csv(csvfile)
    frames_dir = framedir
    threshold ={}
    for i in range(len(threshold_list)):
        name = threshold_list[i].labelname.split('_')[0]
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


    class MainInterface:
        def __init__(self,framesfolder):

            ## defining starting frames x parameters
            self.currentframeNo = 0
            self.framesin = []
            ## getting all the frames froom the directory
            for i in os.listdir(framesfolder):
                self.framesin.append(os.path.join((framesfolder),i))
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
            current_image = Image.open(self.framesin[self.currentframeNo])
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
            self.forward.grid(row=1, column=3, sticky=E)
            self.back.grid(row=1, column=1, sticky=W)
            self.fbox.grid(row=1, column=1)
            self.forwardmax.grid(row=1,column=4,sticky=W)
            self.backmax.grid(row=1,column=0,sticky=W)
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
            for i in targets:
                self.checkVar[i] = IntVar()
                self.checkbox[i] = Checkbutton(self.check_frame,text=i,variable=self.checkVar[i],command =lambda: self.saveBehavior(self.currentframeNo,i))
                self.checkbox[i].grid(row=count + 1, sticky=W)
                self.checkVar[i].set(df_targets[i].iloc[self.currentframeNo])
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

            self.generate = Button(self.window, text="Save csv", command=lambda: self.save_video(df))
            self.generate.config(font=("Calibri", 16))
            self.generate.grid(row=10, column=1, sticky=N)

        def save_video(self, dataframe):
            for i in targets:
                dataframe[i] = df_targets[i]

            dataframe.to_csv(os.path.join(projectpath,'csv','targets_inserted',nameofvid + '.csv'), index=FALSE)

        def saveBehavior(self, frame,target):
            df_targets[target].loc[frame] = self.checkVar[target].get() ##get the frame numbre from the fbox and set it to equals to row in the df, then get 1 or 0

        def saveRangeOfBehavior(self, start, end):
            if self.rangeOn.get():
                for i in range(start, end+1):
                    for j in targets:
                        self.saveBehavior(i,j)

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
                # self.checkVar.set(df_userinput.iloc[self.currentframeNo]) ##set the checkbox based on df
                for i in targets:
                    self.checkVar[i].set(df_targets[i].iloc[self.currentframeNo])

                ## refresh the frames showed in tkinter
                max_size = 1080, 650
                current_image = Image.open(self.framesin[self.currentframeNo])
                current_image.thumbnail(max_size, Image.ANTIALIAS)
                current_frame = ImageTk.PhotoImage(master=self.window, image=current_image)

                self.video_frame = Label(self.window, image='')
                self.video_frame.grid(row=0, column=0)

                self.video_frame.image = current_frame
                self.video_frame.config(image=current_frame)
                current_image.close()


            except IndexError:
                pass

    MainInterface(frames_dir)


