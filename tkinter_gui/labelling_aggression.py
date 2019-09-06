from tkinter import *
import pandas as pd
from PIL import Image, ImageTk
import os
from tkinter import filedialog
import re
from configparser import ConfigParser
from subprocess import *
import signal

currentVideo = 0
frames_in = []

currentFrameNumber = 0
cur_dir = os.getcwd()
behaviors = []

newIndex = []
columns = []
df = pd.DataFrame(index=newIndex, columns=columns)

jumpSize = 0

# Resets variables and dataFrame
def reset():
    global newIndex
    global columns
    global df
    global behaviors

    behaviors = []
    newIndex = []
    columns = []
    df = pd.DataFrame(index=newIndex, columns=columns)

# Retrieves behavior names as indicated by user
def configure():

    configFile = str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"/project_config.ini"
    config = ConfigParser()
    config.read(configFile)

    numberOfTargets = config.get('SML settings', 'No_targets')

    for i in range(1, int(numberOfTargets)+1):
        target = config.get('SML settings', 'target_name_' + str(i))
        columns.append(target)
        behaviors.append(0)
        # print(columns)
        # print(behaviors)
        df[columns[i-1]] = 0

class main_interface():
    def __init__(self):
        window = Toplevel()
        folder = Frame(window)
        folder.grid(row=0, column=1, sticky=N)


        # Dictionary of behaviors and their variables
        self.cbDic = {}

        for i in range(len(columns)):
            self.cbDic[columns[i]] = IntVar(value = behaviors[i])
        # print(self.cbDic)

        # Navigate button frames
        self.button_frame = Frame(window, bd=2, width=700, height=300)
        self.button_frame.grid(row=1, column=0)

        self.frameNumber = Label(self.button_frame, text="Frame Number")
        self.frameNumber.grid(row=0, column=1)
        self.forward = Button(self.button_frame, text=">", command=lambda: load_frame('>', currentFrameNumber, window, self.fbox))
        self.forward.grid(row=1, column=3, sticky=E)
        self.forward = Button(self.button_frame, text=">>",
                              command=lambda: load_frame('jump', len(frames_in) - 1, window, self.fbox))
        self.forward.grid(row=1, column=4, sticky=E)
        self.back = Button(self.button_frame, text="<", command=lambda: load_frame('<', currentFrameNumber, window, self.fbox))
        self.back.grid(row=1, column=1, sticky=W)
        self.back = Button(self.button_frame, text="<<", command=lambda: load_frame('jump', 0, window, self.fbox))
        self.back.grid(row=1, column=0, sticky=W)

        n = StringVar(window, value=currentFrameNumber)
        self.fbox = Entry(self.button_frame, width=4, textvariable=n)
        self.fbox.grid(row=1, column=1)
        self.select = Button(self.button_frame, text="Jump to selected frame", command=lambda:
            load_frame('jump',int(self.fbox.get()),window, self.fbox))
        self.select.grid(row=2, column=1, sticky=N)

        # Jump a certain number of frames
        self.jump_frame = Frame(window)
        self.jump_frame.grid(row=2, column=0)
        self.jump = Label(self.jump_frame, text="Jump Size:")
        self.jump.grid(row=0, column=0, sticky=W)
        self.jump_size = Scale(self.jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
        self.jump_size.set(jumpSize)
        self.jump_size.grid(row=0, column=1, sticky=W)
        self.jump_forward = Button(self.jump_frame, text="<<", command=lambda:
            load_frame('jump', int(self.fbox.get()) - self.jump_size.get(),window, self.fbox))

        self.jump_forward.grid(row=0, column=2, sticky=E)
        self.jump_back = Button(self.jump_frame, text=">>", command=lambda:
            load_frame('jump', int(self.fbox.get()) + self.jump_size.get(),window, self.fbox))
        self.jump_back.grid(row=0, column=3, sticky=W)

        # Check behaviors
        self.check_frame = Frame(window, bd=2, width=300, height=500)
        self.check_frame.grid(row=0, column=1)

        self.blabel = Label(self.check_frame, text="Check Behaviors:")
        self.blabel.config(font=("Calibri", 20))
        self.blabel.grid(sticky=N)

        # Generates checkboxes based on user-specified behaviors
        for key in self.cbDic:
            checkbox = Checkbutton(self.check_frame, text=key, variable=self.cbDic[key],
                                   command=lambda: setValues(self.cbDic))
            checkbox.grid(sticky=W)

        # Save
        save = Button(window, text="Save and Advance to the next frame", command=lambda: self.save_checkBoxes(window))
        save.config(font=("Calibri", 16))
        save.grid(row=1, column=1, sticky=N)

        # Select a range of frames
        self.rangeOn = IntVar(value=0)

        self.rangeFrames = Frame(window)
        self.rangeFrames.grid(row=1, column=1, sticky=S)
        # self.selectrangedescription = Label(self.rangeFrames,text='To set multiple frames that contains behavior(s), select range below.')
        # self.selectrangedescription.grid(row=0)
        self.select_range = Checkbutton(self.rangeFrames, text='Frame Range ', variable=self.rangeOn)
        self.select_range.grid(row=0, column=0, sticky=W)
        self.firstFrame = Entry(self.rangeFrames, width=4)
        self.firstFrame.grid(row=0, column=1, sticky=E)
        self.to_label = Label(self.rangeFrames, text=" to ")
        self.to_label.grid(row=0, column=2, sticky=E)
        self.lastFrame = Entry(self.rangeFrames, width=4)
        self.lastFrame.grid(row=0, column=3, sticky=E)

        self.generate = Button(window, text="Generate and Quit", command=lambda: saveVideo(window))
        self.generate.grid(row=2, column=1, sticky=N)

        load_frame('jump', 0, window, self.fbox)

        videoPlayer = Frame(window, width = 100, height = 100)
        videoPlayer.grid(row = 0, column = 2, sticky = N)


        video = Button(videoPlayer, text = 'Open Current Video', command = lambda: playVideo())
        video.grid(sticky = W)
        videoKey = Label(videoPlayer, text = 'Press: \n p = Pause/Play'
                                             '\n\n After pressing pause:'
                                             '\n o = Forward 2 Frames \n e = Forward 10 Frames \n w = Forward 1 Second'
                                             '\n\n t = Back up 2 Frames \n s = Back up 10 Frames \n x = Back up 1 Second'
                                             '\n\n q = Quit')
        videoKey.grid(sticky = W)
        update = Button(videoPlayer, text = 'Show current video frame', command = lambda: updateFrameFromVideo(window, self.fbox))
        update.grid(sticky = N)

    def save_checkBoxes(self, master):
        if self.rangeOn.get():
            s = int(self.firstFrame.get())
            e = int(self.lastFrame.get())
            saveValues(s, e)
            if e < len(frames_in) - 1:
                load_frame('jump', e + 1, master, self.fbox)
            else:
                load_frame('jump', e, master, self.fbox)
        else:
            s = currentFrameNumber
            e = s
            saveValues(s, e)
            load_frame('>', e, master, self.fbox)

    # def setJumpSize(self, scale):
    #     global jumpSize
    #     jumpSize = scale.get()

# Current frame widgets
class currentFrame():

    def __init__(self, master):

        # Import and resize image
        max_size = 1080, 650
        self.current_image = Image.open(frames_in[currentFrameNumber])
        # self.current_image = self.current_image.resize((1080, 650), Image.ANTIALIAS)
        self.current_image.thumbnail(max_size, Image.ANTIALIAS)
        self.current_frame = ImageTk.PhotoImage(master=master, image=self.current_image)

        self.video_frame = Label(master, image=self.current_frame)
        self.video_frame.image = self.current_frame
        self.video_frame.grid(row=0, column=0)

# Video
def playVideo():
    scriptDirectory = os.path.dirname(os.path.realpath(__file__))
    #print(scriptDirectory + "/play_video.py")
    p = Popen('python ' + str(scriptDirectory) +r"/play_video.py", stdin=PIPE, stdout=PIPE, shell=True)
    print(p)
    print(type(p))
    data = bytes(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2] + '/videos/Video' + str(currentVideo) + '.mp4', 'utf-8')
    p.stdin.write(data)
    p.stdin.close()
    path = (str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"/subprocess.txt")
    with open(path, "w") as text_file:
        text_file.write(str(p.pid))

def updateFrameFromVideo(master, entryBox):
    f = open(str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"/labelling_info.txt", 'r+')
    #print(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2] + "/labelling_info.txt")
    os.fsync(f.fileno())
    vidFrameNo = int(f.readline())
    print(vidFrameNo)
    load_frame('jump', vidFrameNo, master, entryBox)
    f.close()

# Opening screen
class chooseFolder():
    def __init__(self):
        # Locates user specified folder
        global currentVideo
        img_dir = filedialog.askdirectory()
        os.chdir(img_dir)
        print("Working directory is %s" % os.getcwd())
        # folder_label = Label(window, text="Folder selected: %s" % os.getcwd())
        # folder_label.grid(column=1, sticky=S)
        dirpath = os.path.basename(os.getcwd())
        currentVideo = int((re.search(r'\d+', dirpath)).group())
        print('Current Video: ' + str(currentVideo))

        # Creates new data frame and loads frame
        reset()
        global frames_in
        frames_in = []
        for i in os.listdir(os.curdir):
            if i.__contains__(".png"):
                frames_in.append(i)

        frames_in = sorted(frames_in, key=lambda x: int(x.split('.')[0]))
        # print(frames_in)
        numberOfFrames = len(frames_in)
        print ("Number of Frames: " + str(numberOfFrames))

        configure()
        main_interface()
        createDataFrame(numberOfFrames)

# Loads a new frame
def load_frame(string, number, master, entry):
    global currentFrameNumber
    if number > 0 and string == '<':
        currentFrameNumber -= 1
    elif number < len(frames_in)-1 and string == '>':
        currentFrameNumber += 1
    elif string == 'jump':
        currentFrameNumber = number

    entry.delete(0, END)
    entry.insert(0, currentFrameNumber)
    currentFrame(master)

# Create a new Pandas DataFrame for current Video
def createDataFrame(number):
    df.insert(0, 'frames.', list(range(0, number)))
    print(df)

# Changes the global variables for each frame
def setValues(dictionary):
    global behaviors
    for key, item in dictionary.items():
        # print(item.get())
        for i in range(len(columns)):
            # print(columns[i])
            if key == columns[i]:
                behaviors[i] = item.get()
    # print(behaviors)

# Saves the values in dataFrame
def saveValues(start, end):
    if start == end:
        for i in range(len(behaviors)):
            df.at[currentFrameNumber, columns[i]] = int(behaviors[i])
    if start != end:
        for i in range(start, end+1):
            for b in range(len(behaviors)):
                df.at[i, columns[b]] = int(behaviors[b])
    print(df)

# Appends data to corresponding features_extracted csv and exports as new csv
def saveVideo(master):
    input = str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"\csv\features_extracted\Video" + \
            str(currentVideo) + '.csv'
    output = str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"\csv\targets_inserted\Video" + \
            str(currentVideo) + '.csv'
    data = pd.read_csv(input)
    # print(data.drop([0,1], axis = 0).head(3))
    # dropped_data = pd.DataFrame(data.drop([0,1], axis = 0)).reset_index(drop = TRUE)

    df.insert(0, 'video_no', currentVideo)
    new_data = pd.concat([data, df], axis = 1)
    new_data = new_data.fillna(0)
    new_data.rename(columns={'Unnamed: 0': 'scorer'}, inplace=True)

    new_data.to_csv(output, index = FALSE)
    print(output)
    print('Video_' + str(currentVideo) + '.csv targets inserted created')
    master.destroy()

