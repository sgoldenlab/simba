__author__ = "Simon Nilsson", "JJ Choong"

import subprocess
from os import listdir
from os.path import isfile, join
import pathlib
from simba.extract_frames_fast import *
from tkinter import *
import platform
from simba.enums import Defaults
from tkinter.filedialog import askopenfilename, askdirectory


def onMousewheel(event, canvas):
    try:
        scrollSpeed = event.delta
        if platform.system() == 'Darwin':
            scrollSpeed = event.delta
        elif platform.system() == 'Windows':
            scrollSpeed = int(event.delta / 120)
        canvas.yview_scroll(-1 * (scrollSpeed), "units")
    except:
        pass


def bindToMousewheel(event, canvas):
    canvas.bind_all("<MouseWheel>", lambda event: onMousewheel(event, canvas))


def unbindToMousewheel(event, canvas):
    canvas.unbind_all("<MouseWheel>")


def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))


def hxtScrollbar(master):
    '''
        Create canvas.
        Create a frame and put it in the canvas.
        Create two scrollbar and insert command of canvas x and y view
        Use canvas to create a window, where window = frame
        Bind the frame to the canvas
        '''
    bg = master.cget("background")
    acanvas = Canvas(master, borderwidth=0, background=bg)
    frame = Frame(acanvas, background=bg)
    vsb = Scrollbar(master, orient="vertical", command=acanvas.yview)
    vsb2 = Scrollbar(master, orient='horizontal', command=acanvas.xview)
    acanvas.configure(yscrollcommand=vsb.set)
    acanvas.configure(xscrollcommand=vsb2.set)
    vsb.pack(side="right", fill="y")
    vsb2.pack(side="bottom", fill="x")
    acanvas.pack(side="left", fill="both", expand=True)

    acanvas.create_window((10, 10), window=frame, anchor="nw")

    # bind the frame to the canvas
    acanvas.bind("<Configure>", lambda event, canvas=acanvas: onFrameConfigure(acanvas))
    acanvas.bind('<Enter>', lambda event: bindToMousewheel(event, acanvas))
    acanvas.bind('<Leave>', lambda event: unbindToMousewheel(event, acanvas))
    return frame


def form_validator_is_numeric(inStr, acttyp):
    if acttyp == '1':  # insert
        if not inStr.isdigit():
            return False
    return True


class DropDownMenu(Frame):
    def __init__(self, parent=None, dropdownLabel='', choice_dict=None, labelwidth='', com=None, **kw):
        Frame.__init__(self, master=parent, **kw)
        self.dropdownvar = StringVar()
        self.lblName = Label(self, text=dropdownLabel, width=labelwidth, anchor=W)
        self.lblName.grid(row=0, column=0)
        self.choices = choice_dict
        self.popupMenu = OptionMenu(self, self.dropdownvar, *self.choices, command=com)
        self.popupMenu.grid(row=0, column=1)

    def getChoices(self):
        return self.dropdownvar.get()

    def setChoices(self, choice):
        self.dropdownvar.set(choice)

    def enable(self):
        self.popupMenu.configure(state="normal")

    def disable(self):
        self.popupMenu.configure(state="disable")


class FileSelect(Frame):
    def __init__(self, parent=None, fileDescription="", color=None, title=None, lblwidth=None, **kw):
        self.title = title
        self.color = color if color is not None else 'black'
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, fg=str(self.color), width=str(self.lblwidth), anchor=W)
        self.lblName.grid(row=0, column=0, sticky=W)
        self.entPath = Label(self, textvariable=self.filePath, relief=SUNKEN)
        self.entPath.grid(row=0, column=1)
        self.btnFind = Button(self, text=Defaults.BROWSE_FILE_BTN_TEXT.value, command=self.setFilePath)
        self.btnFind.grid(row=0, column=2)
        self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)

    def setFilePath(self):
        file_selected = askopenfilename(title=self.title, parent=self.parent)
        if file_selected:
            self.filePath.set(file_selected)
        else:
            self.filePath.set(Defaults.NO_FILE_SELECTED_TEXT.value)

    @property
    def file_path(self):
        return self.filePath.get()

    def set_state(self, setstatus):
        self.entPath.config(state=setstatus)
        self.btnFind["state"] = setstatus


class Entry_Box(Frame):
    def __init__(self, parent=None, fileDescription="", labelwidth='', status=None, validation=None, **kw):
        super(Entry_Box, self).__init__(master=parent)
        self.validation_methods = {
            'numeric': (self.register(form_validator_is_numeric), '%P', '%d'),
        }
        self.status = status if status is not None else NORMAL
        self.labelname = fileDescription
        Frame.__init__(self, master=parent, **kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription, width=labelwidth, anchor=W)
        self.lblName.grid(row=0, column=0)
        self.entPath = Entry(self, textvariable=self.filePath, state=self.status,
                             validate='key',
                             validatecommand=self.validation_methods.get(validation, None))
        self.entPath.grid(row=0, column=1)

    @property
    def entry_get(self):
        self.entPath.get()
        return self.entPath.get()

    def entry_set(self, val):
        self.filePath.set(val)

    def set_state(self, setstatus):
        self.entPath.config(state=setstatus)

    def destroy(self):
        self.lblName.destroy()
        self.entPath.destroy()


# def colorized(filename):
#
#     def execute(command):
#         print(command)
#         subprocess.call(command, shell=True, stdout = subprocess.PIPE)
#
#     ########### DEFINE COMMAND ###########
#
#     currentFile = filename
#     outFile = currentFile.replace('.mp4', '')
#     outFile = str(outFile) + '_colorized.mp4'
#     command = (str('python bw2color_video3.py --prototxt colorization_deploy_v2.prototxt --model colorization_release_v2.caffemodel --points pts_in_hull.npy --input ' )+ str(currentFile))
#     execute(command)

def shortenvideos1(filename, starttime, endtime):
    if starttime == '' or endtime == '':
        print('Please enter the time')

    elif filename != '' and filename != 'No file selected':

        def execute(command):
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)

        ########### DEFINE COMMAND ###########

        currentFile = filename
        outFile, fileformat = currentFile.split('.')
        outFile = str(outFile) + '_clipped.mp4'
        output = os.path.basename(outFile)

        command = (str('ffmpeg -i ') + '"' + str(
            currentFile) + '"' + ' -ss ' + starttime + ' -to ' + endtime + ' -async 1 ' + '"' + outFile + '"')

        file = pathlib.Path(outFile)
        if file.exists():
            print(output, 'already exist')
        else:
            print('Clipping video....')
            execute(command)
            print(output, ' generated!')
        return output


    else:
        print('Please select a video to trim')


def mergemovebatch(dir, framespersec, vidformat, bit, imgformat):
    currDir = os.listdir(dir)
    fps = str(framespersec)
    fileformat = str('.' + vidformat)
    bitrate = str(bit)
    imageformat = str(imgformat)

    for i in currDir:
        directory = os.path.join(dir, i)
        fileOut = str(directory) + str(fileformat)
        currentDirPath = directory
        currentFileList = [f for f in listdir(currentDirPath) if isfile(join(currentDirPath, f))]
        imgPath = os.path.join(currentDirPath, currentFileList[0])
        img = cv2.imread(imgPath)
        print(imgPath)
        ffmpegFileName = os.path.join(currentDirPath, '%d.' + str(imageformat))
        imgShape = img.shape
        height = imgShape[0]
        width = imgShape[1]
        command = str(
            'ffmpeg -r ' + str(fps) + str(' -f image2 -s ') + str(height) + 'x' + str(width) + ' -i ' + '"' + str(
                ffmpegFileName) + '"' + ' -vcodec libx264 -b ' + str(bitrate) + 'k ' + '"' + str(fileOut) + '"')
        print(command)
        subprocess.call(command, shell=True)


class FolderSelect(Frame):
    def __init__(self, parent=None, folderDescription="", color=None, title=None, lblwidth=None, **kw):
        self.title = title
        self.color = color if color is not None else 'black'
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self, master=parent, **kw)
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription, fg=str(self.color), width=str(self.lblwidth), anchor=W)
        self.lblName.grid(row=0, column=0, sticky=W)
        self.entPath = Label(self, textvariable=self.folderPath, relief=SUNKEN)
        self.entPath.grid(row=0, column=1)
        self.btnFind = Button(self, text="Browse Folder", command=self.setFolderPath)
        self.btnFind.grid(row=0, column=2)
        self.folderPath.set('No folder selected')

    def setFolderPath(self):
        folder_selected = askdirectory(title=str(self.title), parent=self.parent)
        if folder_selected:
            self.folderPath.set(folder_selected)
        else:
            self.folderPath.set('No folder selected')

    @property
    def folder_path(self):
        return self.folderPath.get()


class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)
