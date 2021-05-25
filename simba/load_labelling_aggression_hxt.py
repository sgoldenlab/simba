from tkinter import *
import pandas as pd
from PIL import Image, ImageTk
import os
from tkinter import filedialog
from configparser import ConfigParser, NoSectionError, NoOptionError
from subprocess import *
from simba.rw_dfs import *


current_video = ""
frames_in = []
current_frame_number = 0
cur_dir = os.getcwd()
behaviors = []
new_index = []
columns = []
df = pd.DataFrame(index=new_index, columns=columns)
jump_size = 0


# Resets variables and dataFrame
def reset():
    global new_index
    global columns
    global df
    global behaviors

    behaviors = []
    new_index = []
    columns = []
    df = pd.DataFrame(index=new_index, columns=columns)


# Retrieves behavior names from the config file
def configure(file_name):
    global wfileType
    config = ConfigParser()
    config.read(file_name)
    number_of_targets = config.get('SML settings', 'No_targets')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'

    for i in range(1, int(number_of_targets)+1):
        target = config.get('SML settings', 'target_name_' + str(i))
        columns.append(target)
        behaviors.append(0)
        df[columns[i-1]] = 0


# Initializes all GUI widgets
class MainInterface:
    def __init__(self):
        self.window = Toplevel()
        folder = Frame(self.window)
        folder.grid(row=0, column=1, sticky=N)

        # Advancing Buttons
        self.button_frame = Frame(self.window, bd=2, width=700, height=300)
        self.button_frame.grid(row=1, column=0)

        self.frameNumber = Label(self.button_frame, text="Frame number")
        self.frameNumber.grid(row=0, column=1)
        self.forward = Button(self.button_frame, text=">",
                              command=lambda: load_frame(current_frame_number+1, self.window, self.fbox))
        self.forward.grid(row=1, column=3, sticky=E)
        self.forward = Button(self.button_frame, text=">>",
                              command=lambda: load_frame(len(frames_in) - 1, self.window, self.fbox))
        self.forward.grid(row=1, column=4, sticky=E)
        self.back = Button(self.button_frame, text="<",
                           command=lambda: load_frame(current_frame_number-1, self.window, self.fbox))
        self.back.grid(row=1, column=1, sticky=W)
        self.back = Button(self.button_frame, text="<<", command=lambda: load_frame(0, self.window, self.fbox))
        self.back.grid(row=1, column=0, sticky=W)

        n = StringVar(self.window, value=current_frame_number)
        self.fbox = Entry(self.button_frame, width=7, textvariable=n)
        self.fbox.grid(row=1, column=1)
        self.select = Button(self.button_frame, text="Jump to selected frame", command=lambda:
            load_frame(int(self.fbox.get()), self.window, self.fbox))
        self.select.grid(row=2, column=1, sticky=N)

        # Jump Buttons
        self.jump_frame = Frame(self.window)
        self.jump_frame.grid(row=2, column=0)
        self.jump = Label(self.jump_frame, text="Jump Size:")
        self.jump.grid(row=0, column=0, sticky=W)
        self.jump_size = Scale(self.jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
        self.jump_size.set(jump_size)
        self.jump_size.grid(row=0, column=1, sticky=W)
        self.jump_forward = Button(self.jump_frame, text="<<", command=lambda:
            load_frame(int(self.fbox.get()) - self.jump_size.get(), self.window, self.fbox))
        self.jump_forward.grid(row=0, column=2, sticky=E)
        self.jump_back = Button(self.jump_frame, text=">>", command=lambda:
            load_frame(int(self.fbox.get()) + self.jump_size.get(), self.window, self.fbox))
        self.jump_back.grid(row=0, column=3, sticky=W)

        # Behavior Checkboxes
        self.check_frame = Frame(self.window, bd=2, width=300, height=500)
        self.check_frame.grid(row=0, column=1)

        self.blabel = Label(self.check_frame, text="Check Behaviors:")
        self.blabel.config(font=("Calibri", 20))
        self.blabel.grid(sticky=N)

        # Generates corresponding checkboxes according to config file
        self.cbDic = {}

        for i in range(len(columns)):
            self.cbDic[columns[i]] = IntVar(value = behaviors[i])

        for key in self.cbDic:
            checkbox = Checkbutton(self.check_frame, text=key, variable=self.cbDic[key],
                                   command=lambda: set_values(self.cbDic))
            checkbox.grid(sticky=W)

        # Save Button
        save = Button(self.window, text="Save and advance to the next frame",
                      command=lambda: self.save_checkboxes(self.window))
        save.config(font=("Calibri", 16))
        save.grid(row=1, column=1, sticky=N)

        # Saving a Range of Frames
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

        # Quit Button
        self.generate = Button(self.window, text="Generate / Save csv", command=lambda: save_video(self.window))
        self.generate.grid(row=2, column=1, sticky=N)

        # Loads the first frame
        load_frame(0, self.window, self.fbox)

        # Video Player
        video_player = Frame(self.window, width=100, height=100)
        video_player.grid(row=0, column=2, sticky=N)
        video = Button(video_player, text='Open Video', command=lambda: play_video())
        video.grid(sticky=N, pady = 10)
        video_key = Label(video_player, text='\n\n  Keyboard shortcuts for video navigation: \n p = Pause/Play'
                                             '\n\n After pressing pause:'
                                             '\n o = +2 frames \n e = +10 frames \n w = +1 second'
                                             '\n\n t = -2 frames \n s = -10 frames \n x = -1 second'
                                             '\n\n q = Close video window \n\n')
        video_key.grid(sticky=W)
        update = Button(video_player, text='Show current video frame',
                        command=lambda: update_frame_from_video(self.window, self.fbox))
        update.grid(sticky=N)
        self.bind_keys()
        key_presses = Label(video_player, text='\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame'
                                             '\n Left Arrow = -1 frame'
                                             '\n Ctrl + s = Save and +1 frame'
                                             '\n Ctrl + l = Last frame'
                                             '\n Ctrl + o = First frame')
        key_presses.grid(sticky=S)

    def Rfbox(self):
        return self.fbox

    def guimaster(self):
        return self.window

    # Detects user key presses
    def bind_keys(self):
        self.window.bind('<Control-s>', lambda x: self.save_checkboxes(self.window))
        self.window.bind('<Right>', lambda x: load_frame(current_frame_number+1, self.window, self.fbox))
        self.window.bind('<Left>', lambda x: load_frame(current_frame_number-1, self.window, self.fbox))
        self.window.bind('<Control-q>', lambda x: save_video(self.window))
        self.window.bind('<Control-l>', lambda x: load_frame(len(frames_in) - 1, self.window, self.fbox))
        self.window.bind('<Control-o>', lambda x: load_frame(0, self.window, self.fbox))

    # Saves the values of each behavior and advances either to the next or by a set number of frames
    def save_checkboxes(self, master):
        if self.rangeOn.get():
            s = int(self.firstFrame.get())
            e = int(self.lastFrame.get())
            print(s, e)
            save_values(s, e)
            if e < len(frames_in) - 1:
                load_frame(e + 1, master, self.fbox)
            else:
                load_frame(e, master, self.fbox)
        else:
            s = current_frame_number
            e = s
            save_values(s, e)
            load_frame(e+1, master, self.fbox)

        print(df)

class MainInterface2:
    def __init__(self):
        self.window = Toplevel()
        folder = Frame(self.window)
        folder.grid(row=0, column=1, sticky=N)

        # Advancing Buttons
        self.button_frame = Frame(self.window, bd=2, width=700, height=300)
        self.button_frame.grid(row=1, column=0)

        self.frameNumber = Label(self.button_frame, text="Frame number")
        self.frameNumber.grid(row=0, column=1)
        self.forward = Button(self.button_frame, text=">",
                              command=lambda: load_frame(current_frame_number+1, self.window, self.fbox))
        self.forward.grid(row=1, column=3, sticky=E)
        self.forward = Button(self.button_frame, text=">>",
                              command=lambda: load_frame(len(frames_in) - 1, self.window, self.fbox))
        self.forward.grid(row=1, column=4, sticky=E)
        self.back = Button(self.button_frame, text="<",
                           command=lambda: load_frame(current_frame_number-1, self.window, self.fbox))
        self.back.grid(row=1, column=1, sticky=W)
        self.back = Button(self.button_frame, text="<<", command=lambda: load_frame(0, self.window, self.fbox))
        self.back.grid(row=1, column=0, sticky=W)

        n = StringVar(self.window, value=current_frame_number)
        self.fbox = Entry(self.button_frame, width=7, textvariable=n)
        self.fbox.grid(row=1, column=1)
        self.select = Button(self.button_frame, text="Jump to selected frame", command=lambda:
            load_frame(int(self.fbox.get()), self.window, self.fbox))
        self.select.grid(row=2, column=1, sticky=N)

        # Jump Buttons
        self.jump_frame = Frame(self.window)
        self.jump_frame.grid(row=2, column=0)
        self.jump = Label(self.jump_frame, text="Jump Size:")
        self.jump.grid(row=0, column=0, sticky=W)
        self.jump_size = Scale(self.jump_frame, from_=0, to=100, orient=HORIZONTAL, length=200)
        self.jump_size.set(jump_size)
        self.jump_size.grid(row=0, column=1, sticky=W)
        self.jump_forward = Button(self.jump_frame, text="<<", command=lambda:
            load_frame(int(self.fbox.get()) - self.jump_size.get(), self.window, self.fbox))
        self.jump_forward.grid(row=0, column=2, sticky=E)
        self.jump_back = Button(self.jump_frame, text=">>", command=lambda:
            load_frame(int(self.fbox.get()) + self.jump_size.get(), self.window, self.fbox))
        self.jump_back.grid(row=0, column=3, sticky=W)


        # Loads the first frame
        load_frame(0, self.window, self.fbox)

        # Video Player
        video_player = Frame(self.window, width=100, height=100)
        video_player.grid(row=0, column=2, sticky=N)
        key_presses = Label(video_player, text='\n\n Keyboard shortcuts for frame navigation: \n Right Arrow = +1 frame'
                                             '\n Left Arrow = -1 frame'
                                             '\n Ctrl + s = Save and +1 frame'
                                             '\n Ctrl + l = Last frame'
                                             '\n Ctrl + o = First frame')
        key_presses.grid(sticky=S)
        self.bind_keys()
    def Rfbox(self):
        return self.fbox

    def guimaster(self):
        return self.window

    # Detects user key presses
    def bind_keys(self):
        self.window.bind('<Control-s>', lambda x: self.save_checkboxes(self.window))
        self.window.bind('<Right>', lambda x: load_frame(current_frame_number+1, self.window, self.fbox))
        self.window.bind('<Left>', lambda x: load_frame(current_frame_number-1, self.window, self.fbox))
        self.window.bind('<Control-q>', lambda x: save_video(self.window))
        self.window.bind('<Control-l>', lambda x: load_frame(len(frames_in) - 1, self.window, self.fbox))
        self.window.bind('<Control-o>', lambda x: load_frame(0, self.window, self.fbox))



# Opens the video that corresponds to the matching labelling folder
def play_video():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    p = Popen('python ' + str(script_directory) + r"/play_video.py", stdin=PIPE, stdout=PIPE, shell=True)
    main_project_dir = str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2])
    video_dir = os.path.join(main_project_dir, videos)
    video_list = os.listdir(video_dir)
    current_video_name = os.path.basename(os.getcwd())
    current_full_video_name = [i for i in video_list if current_video_name in i]
    current_full_video_name = current_full_video_name[0]
    print(current_full_video_name)
    data = bytes(video_dir+str(current_full_video_name), 'utf-8')
    p.stdin.write(data)
    p.stdin.close()
    path = (str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"/subprocess.txt")
    with open(path, "w") as text_file:
        text_file.write(str(p.pid))


# Updates and loads the frame corresponding to the frame the video is paused on
def update_frame_from_video(master, entrybox):
    f = open(str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2]) + r"/labelling_info.txt", 'r+')
    os.fsync(f.fileno())
    vid_frame_no = int(f.readline())
    print(vid_frame_no)
    load_frame(vid_frame_no, master, entrybox)
    f.close()


# Opens a dialogue box where user chooses folder with video frames for labelling, then
# loads the config file, creates an empty dataframe and loads the interface,
# Prints working directory, current video name, and number of frames in folder.
def load_folder(project_name):
    global current_video
    img_dir = filedialog.askdirectory()
    os.chdir(img_dir)
    print("Working directory is %s" % os.getcwd())
    dirpath = os.path.basename(os.getcwd())
    current_video = dirpath
    print('Current Video: ' + current_video)
    ## get the frames
    global frames_in
    frames_in = []
    for i in os.listdir(os.curdir):
        if i.__contains__(".png"):
            frames_in.append(i)
        reset()

    frames_in = sorted(frames_in, key=lambda x: int(x.split('.')[0]))
    # print(frames_in)
    number_of_frames = len(frames_in)
    print("Number of Frames: " + str(number_of_frames))

    configure(project_name)
    curr_target_csv = os.path.join(os.path.dirname(project_name),'csv','targets_inserted',os.path.basename(img_dir)+'.csv')
    df = read_df(curr_target_csv, wfileType)
    MainInterface()
    #create_data_frame(number_of_frames)

def choose_folder2(framedir):
    global current_video, guiwindow
    img_dir = framedir
    os.chdir(img_dir)

    global frames_in
    frames_in = []
    for i in os.listdir(os.curdir):
        if i.__contains__(".png"):
            frames_in.append(i)
        reset()

    frames_in = sorted(frames_in, key=lambda x: int(x.split('.')[0]))
    # print(frames_in)
    number_of_frames = len(frames_in)
    print("Number of Frames: " + str(number_of_frames))

    guiwindow = MainInterface2()

    return guiwindow



# Loads a new image frame
def load_frame(number, master, entry):
    global current_frame_number
    max_index_frames_in = len(frames_in) - 1
    try:
        if number > max_index_frames_in:
            print("Reached End of Frames")
            current_frame_number = max_index_frames_in
        elif number < 0:
            current_frame_number = 0
        else:
            current_frame_number = number

        entry.delete(0, END)
        entry.insert(0, current_frame_number)

        max_size = 1080, 650
        current_image = Image.open(frames_in[current_frame_number])
        current_image.thumbnail(max_size, Image.ANTIALIAS)
        current_frame = ImageTk.PhotoImage(master=master, image=current_image)

        video_frame = Label(master, image=current_frame)
        video_frame.image = current_frame
        video_frame.grid(row=0, column=0)

    except IndexError:
        pass

# Loads a new image frame
def load_frame2(number, master,entry):
    global current_frame_number
    max_index_frames_in = len(frames_in) - 1
    try:
        if number > max_index_frames_in:
            print("Reached End of Frames")
            current_frame_number = max_index_frames_in
        elif number < 0:
            current_frame_number = 0
        else:
            current_frame_number = number

        entry.delete(0, END)
        entry.insert(0, current_frame_number)

        max_size = 1080, 650
        current_image = Image.open(frames_in[current_frame_number])
        current_image.thumbnail(max_size, Image.ANTIALIAS)
        current_frame = ImageTk.PhotoImage(master=master, image=current_image)

        video_frame = Label(master, image=current_frame)
        video_frame.image = current_frame
        video_frame.grid(row=0, column=0)

    except IndexError:
        pass


# Creates a new Pandas DataFrame for current Video
# def create_data_frame(number):
#     df.insert(0, 'frames.', list(range(0, number)))


# Temporarily updates the value of each variable for each frame to either 0 or 1
def set_values(dictionary):
    global behaviors
    for key, item in dictionary.items():
        # print(item.get())
        for i in range(len(columns)):
            # print(columns[i])
            if key == columns[i]:
                behaviors[i] = item.get()
    # print(behaviors)


# Saves the values of each behavior in the DataFrame and prints out the updated data frame
def save_values(start, end):
    if start == end:
        for i in range(len(behaviors)):
            df.at[current_frame_number, columns[i]] = int(behaviors[i])
    if start != end:
        for i in range(start, end+1):
            for b in range(len(behaviors)):
                df.at[i, columns[b]] = int(behaviors[b])
    print(df.ix[current_frame_number])


# Appends data to corresponding features_extracted csv and exports as new csv
def save_video(master):
    global wfileType
    input_file = str(os.path.join(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2], 'csv', 'features_extracted', current_video + '.' + wfileType))
    output_file = str(os.path.join(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2], 'csv', 'targets_inserted', current_video + '.' + wfileType))
    data = read_df(input_file, wfileType)
    new_data = pd.concat([data, df], axis=1)
    new_data = new_data.fillna(0)
    new_data.rename(columns={'Unnamed: 0': 'scorer'}, inplace=True)
    save_df(new_data, wfileType, output_file)
    print(current_video + 'Annotation file for "' + str(current_video) + '"' + ' created.')
    # master.destroy()
