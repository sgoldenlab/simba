import pandas as pd
from tkinter import *
import pandas as pd
from PIL import Image, ImageTk
import os
from tkinter import filedialog
from configparser import ConfigParser
from subprocess import *
import matplotlib.pyplot as plt
import numpy as np


def semisuperviseLabel(inifile,framedir,target):
    projectpath = os.path.dirname(inifile)
    csvfile = os.path.join(projectpath,'csv','machine_results',os.path.basename(framedir)+'.csv')
    nameofvid = os.path.basename(csvfile).split('.csv')[0]
    # Retrieves behavior name and preexisting values from the machine generated csv file
    ##read in csv file
    df = pd.read_csv(csvfile)
    frames_dir = framedir

    df_userinput = df[target]

    probabilityColumnName = 'Probability_' + target
    print(df[[probabilityColumnName]])
    probs = df[[probabilityColumnName]].to_numpy()

    class plotgraphx:
        def __init__(self,probabilityColumnName,currFramesDir,master):
            self.probcolname = probabilityColumnName
            self.framedir = currFramesDir
            self.master = master

            self.fig, self.ax = plt.subplots()
            self.ax.plot(probs)
            plt.xlabel('frame #', fontsize=16)
            plt.ylabel(str(probabilityColumnName) + ' probability', fontsize=16)
            plt.title('Click on the points of the graph to display the corresponding frames.')
            plt.grid()
            cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)  ##incoporate mouse click event
            plt.show()


        def onclick(self,event):
            if event.dblclick:
                if event.button == 1:  ##get point 1 on double left click
                    probability = probs[int(event.xdata)].astype(str)
                    self.master.advance_frame(int(event.xdata))
                    print("Selected frame has a probability of", probability)
                    a = plt.axvline(x=int(event.xdata), color='r')
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    a.remove()

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

            self.checkVar = IntVar()
            self.checkVar.set(df_userinput.iloc[self.currentframeNo])
            self.checkbox = Checkbutton(self.check_frame, text=target, variable = self.checkVar,
                                   command=lambda: self.saveBehavior(self.currentframeNo))
            self.checkbox.grid(sticky=W)

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
            dataframe.to_csv(os.path.join(projectpath,'csv','targets_inserted',nameofvid+'.csv'), index=FALSE)

        def saveBehavior(self, frame):
            df_userinput[frame] = self.checkVar.get() ##get the frame numbre from the fbox and set it to equals to row in the df, then get 1 or 0

        def saveRangeOfBehavior(self, start, end):
            if self.rangeOn.get():
                for i in range(start, end+1):
                    self.saveBehavior(i)

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
                self.checkVar.set(df_userinput.iloc[self.currentframeNo]) ##set the checkbox based on df

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

                #print
                if self.checkVar.get()==1:
                    print('The machine thinks it is a(n)', target)
                else:
                    print('The machine thinks it is NOT a(n)',target)

            except IndexError:
                pass

    a = MainInterface(frames_dir)
    plotgraphx(probabilityColumnName,frames_dir,a)


