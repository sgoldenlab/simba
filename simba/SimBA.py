import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
from collections import defaultdict
#import umap
import seaborn as sns
import os
import time
import subprocess
import itertools
#import deeplabcut
import csv
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename,askdirectory
from tkinter import tix, messagebox
import subprocess
import platform
import shutil
from tabulate import tabulate
from configparser import ConfigParser, NoSectionError, NoOptionError, MissingSectionHeaderError
from PIL import ImageTk
import PIL.Image
import tkinter.ttk as ttk
import webbrowser
import cv2
import simba.adv_label as adv
import simba.adv_load_label as advl
from simba.plotly_create_h5 import *
from simba.dlc_change_yamlfile import *
from simba.tkinter_functions import *
from simba.create_project_ini import write_inifile,write_unsupervisedini
from simba.json2csv import json2csv_file, json2csv_folder
from simba.run_RF_model import rfmodel
from simba.prob_graph import *
from simba.runmodel_1st import *
from simba.path_plot import path_plot_config
from simba.gantt import ganntplot_config
from simba.data_plot import data_plot_config
from simba.line_plot import line_plot_config
from simba.import_videos_csv_project_ini import *
from simba.labelling_aggression import *
from simba.pseudoLabel import semisuperviseLabel
from simba.load_labelling_aggression import load_folder
from simba.get_coordinates_tools_v2 import get_coordinates_nilsson
from simba.process_data_log import analyze_process_data_log
from simba.process_severity import analyze_process_severity
from simba.extract_seqframes import *
from simba.classifierValidation import validate_classifier
from simba.train_multiple_models_from_meta import *
from simba.train_model_2 import *
from simba.multiplecrop import *
from simba.extract_frames_fast import *
from simba.rw_dfs import *
from simba.validate_model_on_single_video import *
from simba.ROI_freehand_draw_3 import roiFreehand
from simba.ROI_analysis_2 import *
from simba.ROI_plot import *
from simba.ROI_draw_defined import *
from simba.ROI_multiply import *
from simba.ROI_reset import *
from simba.ROI_add_to_features import *
from simba.ROI_process_movement import *
from simba.ROI_visualize_features import ROItoFeaturesViz
from simba.plot_heatmap import plotHeatMap
from simba.append_boris import *
from simba.outlier_scripts.movement.correct_devs_mov_16bp import dev_move_16
from simba.outlier_scripts.location.correct_devs_loc_16bp import dev_loc_16
from simba.outlier_scripts.movement.correct_devs_mov_user_defined import dev_move_user_defined
from simba.outlier_scripts.location.correct_devs_loc_user_defined import dev_loc_user_defined
from simba.outlier_scripts.movement.correct_devs_mov_14bp import dev_move_14
from simba.outlier_scripts.location.correct_devs_loc_14bp import dev_loc_14
from simba.outlier_scripts.skip_outlierCorrection import skip_outlier_c
from simba.features_scripts.extract_features_16bp import extract_features_wotarget_16
from simba.features_scripts.extract_features_14bp import extract_features_wotarget_14
from simba.features_scripts.extract_features_14bp_from_16bp import extract_features_wotarget_14_from_16
from simba.features_scripts.extract_features_9bp import extract_features_wotarget_9
from simba.features_scripts.extract_features_8bp import extract_features_wotarget_8
from simba.features_scripts.extract_features_7bp import extract_features_wotarget_7
from simba.features_scripts.extract_features_4bp import extract_features_wotarget_4
from simba.features_scripts.extract_features_user_defined import extract_features_wotarget_user_defined
from simba.sklearn_plot_scripts.plot_sklearn_results_2 import plotsklearnresult
from simba.sklearn_plot_scripts.plot_sklearn_results_2_single import plotsklearnresultsingle
from simba.drop_bp_cords import define_bp_drop_down,reverse_dlc_input_files
from simba.drop_bp_cords import bodypartConfSchematic
from simba.define_new_pose_config import define_new_pose_configuration
# from simba.dpk_create_project_ini import write_dpkfile
# from simba.dpk_script.create_annotation_set import createAnnotationSet
# from simba.dpk_script.annotator import dpkAnnotator
# from simba.dpk_script.train_model import trainDPKmodel
# from simba.dpk_script.Predict_new_video import predictnewvideoDPK
# from simba.dpk_script.Visualize_video import visualizeDPK
from simba.reset_poseConfig import reset_DiagramSettings
from simba.plot_threshold import plot_threshold
from simba.merge_frames_movie import mergeframesPlot
from simba.plot_heatmap_location_new import plotHeatMapLocation
from simba.appendMars import append_dot_ANNOTT
from simba.read_DLCmulti_h5_function import importMultiDLCpose
from simba.sleap_bottom_up_convert import importSLEAPbottomUP
from simba.timeBins_movement import time_bins_movement
from simba.timeBins_classifiers import time_bins_classifier
from simba.ez_lineplot import draw_line_plot,draw_line_plot_tools
from simba.append_boris import append_Boris_annot
from simba.ROI_directionality_to_other_animals import *
from simba.ROI_directionality_to_other_animals_visualize import *
from simba.ROI_time_bins import roi_time_bins
from simba.reorganize_animal_bp import *
from simba.import_solomon import solomonToSimba
from simba.reverse_tracking_order import reverse_tracking_2_animals
from simba.Kleinberg_burst_analysis import run_kleinberg
from simba.FSTCC import FSTCC_perform
from simba.pup_retrieval_1 import pup_retrieval_1
from simba.interpolate_pose import *
from simba.import_trk import *
from simba.drop_bps import *
from simba.classifications_per_ROI import *
from simba.read_DANNCE_mat import import_DANNCE_file, import_DANNCE_folder
from simba.drop_bp_cords import get_fn_ext
from simba.misc_tools import smooth_data_gaussian, smooth_data_savitzky_golay
from simba.ethovision_import import ImportEthovision
from simba.plot_pose_in_dir import create_video_from_dir


import urllib.request
from cefpython3 import cefpython as cef
import threading
import datetime
import platform
import atexit

from simba.roi_tools.ROI_move_shape import *
from simba.roi_tools.ROI_image import *
from simba.roi_tools.ROI_zoom import *
from simba.roi_tools.ROI_define import *
from simba.roi_tools.ROI_menus import *
from simba.roi_tools.ROI_reset import *

simBA_version = 1.2
currentPlatform = platform.system()
if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
    from simba.process_videos_automation_linux import *
if currentPlatform == 'Windows':
    from simba.process_videos_automation_win import *

class roitableRow(Frame):
    def __init__(self, parent =None ,dirname='',filename = '',widths = "" ,indexs='',projectini=''):
        self.projectini = projectini
        self.filename = os.path.join(dirname,filename)
        Frame.__init__(self,master=parent)
        var=StringVar()
        self.index = Entry(self,textvariable=var,width=4)
        var.set(indexs)
        self.index.grid(row=0,column=0)
        self.lblName = Label(self,text=filename,width =widths,anchor=W)
        self.lblName.grid(row=0,column=1,sticky=W)
        self.btnset = Button(self,text='Draw',command=self.draw)
        self.btnset.grid(row=0,column=2)
        self.btnreset = Button(self,text='Reset',command =self.reset)
        self.btnreset.grid(row=0,column=3)
        self.btnapplyall = Button(self, text='Apply to all', command=self.applyall)
        self.btnapplyall.grid(row=0, column=4)

    def draw(self):
        roiFreehand(self.projectini,self.filename)

    def reset(self):
        ROI_reset(self.projectini, self.filename)

    def applyall(self):
        multiplyFreeHand(self.projectini, self.filename)

class roitableMenu:
    def __init__(self,videofolder,inifile):
        self.filesFound = []
        self.row = []
        self.videofolder = videofolder

        ### FIND FILES #####
        for i in os.listdir(videofolder):
            if i.endswith(('.avi', '.mp4', '.mov', 'flv')):
                self.filesFound.append(i)

        ### longest string in list
        maxname = max(self.filesFound, key=len)

        ## popup window
        roimenu = Toplevel()
        roimenu.minsize(500, 400)
        roimenu.wm_title("ROI Table")

        scroll = hxtScrollbar(roimenu)

        tableframe = LabelFrame(scroll,text='Video Name',labelanchor=NW)

        #### loop for tables######
        for i in range(len(self.filesFound)):
            self.row.append(roitableRow(tableframe, self.videofolder,str(self.filesFound[i]), str(len(maxname)),str(i+1) + '.', projectini=inifile))
            self.row[i].grid(row=i + 1, sticky=W)

        tableframe.grid(row=0)




class processvid_title(Frame):
    def __init__(self,parent=None,widths="",color=None,shortenbox =None,downsambox =None,graybox=None,framebox=None,clahebox=None,fpsbox=None,**kw):
        self.color = color if color is not None else 'black'
        Frame.__init__(self,master=parent,**kw)
        self.lblName = Label(self, text= 'Video Name',fg=str(self.color),width=int(widths)+5,font=("Helvetica",10,'bold'))
        self.lblName.grid(row=0,column=0,sticky=W)
        self.lblName3 = Label(self, text='Start Time',width = 13,font=("Helvetica",10,'bold'))
        self.lblName3.grid(row=0, column=1,columnspan=2)
        self.lblName4 = Label(self, text='End Time',width = 15,font=("Helvetica",10,'bold'))
        self.lblName4.grid(row=0, column=3)
        self.shorten = IntVar()
        self.lblName5 = Checkbutton(self,text='Select All',variable= self.shorten,command=shortenbox)
        self.lblName5.grid(row=0, column=4)
        self.lblName6 = Label(self, text='Width',width =13,font=("Helvetica",10,'bold'))
        self.lblName6.grid(row=0, column=5)
        self.lblName7 = Label(self, text='Height',width = 15,font=("Helvetica",10,'bold'))
        self.lblName7.grid(row=0, column=6)
        self.downsample = IntVar()
        self.lblName8 = Checkbutton(self,text='Select All',variable = self.downsample,command =downsambox)
        self.lblName8.grid(row=0,column=7,padx=5)
        self.fps = IntVar()
        self.lblName12 = Label(self,text='FPS',width = 15,font=("Helvetica",10,'bold'))
        self.lblName12.grid(row=0,column=8,padx=5)
        self.lblName13 = Checkbutton(self,text='Select All',variable =self.fps,command = fpsbox)
        self.lblName13.grid(row=0, column=9,padx=5)
        self.grayscale = IntVar()
        self.lblName9 = Checkbutton(self,text='Select All',variable =self.grayscale,command = graybox)
        self.lblName9.grid(row=0,column=10,padx=5)
        self.frameno = IntVar()
        self.lblName10 = Checkbutton(self,text='Select All',variable = self.frameno,command = framebox)
        self.lblName10.grid(row=0,column=11,padx=5)
        self.clahe = IntVar()
        self.lblName11 = Checkbutton(self,text='Select All',variable =self.clahe,command =clahebox)
        self.lblName11.grid(row=0,column=12,padx=5)

class processvideotable(Frame):
    def __init__(self,parent=None,fileDescription="",widths = "" ,dirname ="",outputdir='',color=None,**kw):
        self.color = color if color is not None else 'black'
        self.croplist = []
        self.filename = os.path.join(dirname,fileDescription)
        self.outputdir = outputdir
        Frame.__init__(self,master=parent,**kw)
        self.lblName = Label(self, text=fileDescription,fg=str(self.color),width= widths,anchor=W)
        self.lblName.grid(row=0,column=0,sticky=W)
        self.btnFind = Button(self, text="Crop",command=self.cropvid)
        self.btnFind.grid(row=0,column=1)
        self.trimstart = Entry(self)
        self.trimstart.grid(row=0,column=2)
        self.trimend = Entry(self)
        self.trimend.grid(row=0, column=3)
        self.shortenvar = IntVar()
        self.downsamplevar = IntVar()
        self.grayscalevar = IntVar()
        self.superimposevar = IntVar()
        self.clahevar = IntVar()
        self.fpsvar = IntVar()
        self.shortenvid = Checkbutton(self,text='Shorten',variable = self.shortenvar)
        self.shortenvid.grid(row=0,column=4)
        self.width = Entry(self)
        self.width.grid(row=0,column=5)
        self.height = Entry(self)
        self.height.grid(row=0,column=6)
        self.fps = Entry(self)
        self.fps.grid(row=0,column=8)
        self.changefps = Checkbutton(self,text='Change fps',variable=self.fpsvar)
        self.changefps.grid(row=0,column=9)
        self.downsamplevid = Checkbutton(self,text='Downsample',variable = self.downsamplevar)
        self.downsamplevid.grid(row=0,column=7)
        self.grayscalevid = Checkbutton(self,text='Grayscale',variable= self.grayscalevar)
        self.grayscalevid.grid(row=0,column =10)
        self.superimposevid = Checkbutton(self,text='Add Frame #',variable =self.superimposevar)
        self.superimposevid.grid(row=0,column=11)
        self.clahevid = Checkbutton(self,text='CLAHE',variable = self.clahevar)
        self.clahevid.grid(row=0,column =12)

    def cropvid(self):
        self.croplist = []
        print(self.filename)
        command = cropvid_queue(self.filename,self.outputdir)
        self.croplist.append(command)
        self.btnFind.configure(bg='red')

    def get_crop_list(self):
        return self.croplist

    def getstarttime(self):
         return self.trimstart.get()

    def getendtime(self):
        return self.trimend.get()

    def shorten(self):
        return self.shortenvar.get()

    def downsample(self):
        return self.downsamplevar.get()
    def getwidth(self):
        return self.width.get()
    def getheight(self):
        return self.height.get()
    def grayscale(self):
        return self.grayscalevar.get()
    def addframe(self):
        return self.superimposevar.get()
    def get_clahe(self):
        return self.clahevar.get()

    def getfpsvar(self):
        return self.fpsvar.get()

    def getfps(self):
        return self.fps.get()

class processvid_menu:

    def __init__(self, videofolder, outputdir):
        self.filesFound = []
        self.row = []
        self.videofolder = videofolder
        self.outputdir = outputdir

        ########### FIND FILES ###########
        for i in os.listdir(videofolder):
            lower_str_name = i.lower()
            if lower_str_name.endswith(('.avi','.mp4','.mov','.flv','.m4v')):
                self.filesFound.append(i)

        ### longest string in list
        try:
            maxname = max(self.filesFound,key=len)
        except ValueError:
            print("No videos detected: Please select folder that contain videos.")

        # Popup window
        vidprocessmenu = Toplevel()
        vidprocessmenu.minsize(1300, 400)
        vidprocessmenu.wm_title("Batch process video table")
        vidprocessmenu.lift()

        scroll = Canvas(hxtScrollbar(vidprocessmenu))
        scroll.pack(fill="both",expand=True)

        #shortcut for filling up parameters
        shortcutframe = LabelFrame(scroll,text='Quick settings',pady=5,padx=15)
        #shorten
        shortenshortcut = LabelFrame(shortcutframe,text='Shorten settings',padx=5)
        starttime = Entry_Box(shortenshortcut,'Start time','15')
        endtime = Entry_Box(shortenshortcut,'End time','15')
        shortenbutton = Button(shortenshortcut,text='Save settings',command=lambda:self.saveShortensettings(starttime.entry_get,endtime.entry_get))
        #add sec
        secondshortcut = LabelFrame(shortcutframe,text='Add seconds to start time',padx=5)
        secondstoadd = Entry_Box(secondshortcut,'seconds','15')
        addsecbutton = Button(secondshortcut,text='Add seconds',command=lambda:self.addsec(secondstoadd.entry_get))

        #downsample
        downsampleshortcut = LabelFrame(shortcutframe,text='Downsample settings',padx=5)
        width = Entry_Box(downsampleshortcut,'Width','15')
        height = Entry_Box(downsampleshortcut,'Height','15')
        downsamplebutton = Button(downsampleshortcut,text='Save settings',command=lambda:self.saveDownsamplesettings(width.entry_get,height.entry_get))

        #fps
        fpsshortcut = LabelFrame(shortcutframe,text='FPS settings', padx=5)
        fps = Entry_Box(fpsshortcut,'fps','15')
        fpsbutton = Button(fpsshortcut,text='Save settings',command=lambda:self.saveFpssettings(fps.entry_get))

        #organize
        shortenshortcut.grid(row=0,sticky=W,padx=10)
        starttime.grid(row=0,sticky=W)
        endtime.grid(row=1,sticky=W)
        shortenbutton.grid(row=2,sticky=W)

        secondshortcut.grid(row=1,sticky=W,padx=10)
        secondstoadd.grid(row=0,sticky=W)
        addsecbutton.grid(row=1,sticky=W)

        downsampleshortcut.grid(row=0,column=1,sticky=W)
        width.grid(row=0,sticky=W)
        height.grid(row=1,sticky=W)
        downsamplebutton.grid(row=2,sticky=W)

        fpsshortcut.grid(row=1,column=1,sticky=W)
        fps.grid(row=0,sticky=W)
        fpsbutton.grid(row=1,sticky=W)

        ## starting of the real table
        tableframe = LabelFrame(scroll,padx=10)

        # table title
        self.title = processvid_title(tableframe, str(len(maxname)), shortenbox=self.selectall_shorten,
                                      downsambox=self.selectall_downsample, graybox=self.selectall_grayscale,
                                      framebox=self.selectall_addframe, clahebox=self.selectall_clahe,fpsbox=self.selectall_fps)

        #### loop for tables######
        for i in range(len(self.filesFound)):
            self.row.append(processvideotable(tableframe,str(self.filesFound[i]), str(len(maxname)),self.videofolder,self.outputdir))
            self.row[i].grid(row=i+1, sticky=W)

        #organize table title
        self.title.grid(row=0, sticky=W)
        #button to trigger process video
        but = Button(scroll,text='Execute',command = lambda: threading.Thread(target=self.execute_processvideo).start(),font=('Times',12,'bold'),fg='navy')
        but.grid(row=2)

        #organize
        shortcutframe.grid(row=0,sticky=W)
        tableframe.grid(row=1,sticky=W,)


    def addsec(self,secondToAdd):
        outtimelist = []
        # loop through and get final time
        for i in range(len(self.filesFound)):
            starttime = self.row[i].trimstart.get() #get the user's input
            starttimelist = starttime.split(':') # split it into hours: minutes : seconds
            for i in range(len(starttimelist)): # remove the 0 so it is compatible with the datetimeformat
                if starttimelist[i][0] == '0':
                    starttimelist[i] = starttimelist[i][1:]

            hr,min,sec = starttimelist #get inputs for datetime
            starttime = datetime.time(int(hr),int(min),int(sec))
            out_time = str(self.addSecs(starttime,int(secondToAdd))) #call addSecs func: starttime + secondToAdd = out_time
            outtimelist.append(out_time)

        # add the final time into the table
        for i in range(len(self.filesFound)):
            self.row[i].trimend.delete(0, END)
            self.row[i].trimend.insert(0,outtimelist[i])

    def addSecs(self,tm, secs):
        fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
        fulldate = fulldate + datetime.timedelta(seconds=secs)
        return fulldate.time()

    def saveFpssettings(self, fps):
        for i in range(len(self.filesFound)):
            self.row[i].fps.delete(0, END)
            self.row[i].fps.insert(0,fps)

    def saveDownsamplesettings(self,width,height):
        for i in range(len(self.filesFound)):
            self.row[i].width.delete(0, END)
            self.row[i].height.delete(0, END)
            self.row[i].width.insert(0,width)
            self.row[i].height.insert(0,height)

    def saveShortensettings(self,startime,endtime):
        for i in range(len(self.filesFound)):
            self.row[i].trimstart.delete(0, END)
            self.row[i].trimend.delete(0, END)
            self.row[i].trimstart.insert(0,startime)
            self.row[i].trimend.insert(0,endtime)

    def selectall_clahe(self):
        for i in range(len(self.filesFound)):
            if self.title.clahe.get() == 1:
                self.row[i].clahevid.select()
            else:
                self.row[i].clahevid.deselect()

    def selectall_addframe(self):
        for i in range(len(self.filesFound)):
            if self.title.frameno.get() == 1:
                self.row[i].superimposevid.select()
            else:
                self.row[i].superimposevid.deselect()

    def selectall_grayscale(self):
        for i in range(len(self.filesFound)):
            if self.title.grayscale.get()==1:
                self.row[i].grayscalevid.select()
            else:
                self.row[i].grayscalevid.deselect()

    def selectall_downsample(self):
        for i in range(len(self.filesFound)):
            if self.title.downsample.get()==1:
                self.row[i].downsamplevid.select()
            else:
                self.row[i].downsamplevid.deselect()

    def selectall_fps(self):
        for i in range(len(self.filesFound)):
            if self.title.fps.get()==1:
                self.row[i].changefps.select()
            else:
                self.row[i].changefps.deselect()

    def selectall_shorten(self):
        for i in range(len(self.filesFound)):
            if self.title.shorten.get()==1:
                self.row[i].shortenvid.select()
            else:
                self.row[i].shortenvid.deselect()

    def get_thecroplist(self):
        self.croplistt = []

        for i in range(len(self.filesFound)):
           self.croplistt.append((self.row[i].get_crop_list()))
        self.croplistt = list(itertools.chain(*self.croplistt))

        return self.croplistt

    def get_shortenlist(self):
        self.shortenlistt = []
        for i in range(len(self.filesFound)):
            if (self.row[i].shorten()) == 1:
                self.shortenlistt.append(shortenvideos1_queue(self.outputdir,self.filesFound[i],self.row[i].getstarttime(),self.row[i].getendtime()))

        return self.shortenlistt

    def get_downsamplelist(self):
        self.downsamplelistt = []
        for i in range(len(self.filesFound)):
            if (self.row[i].downsample()) == 1:
                self.downsamplelistt.append(downsamplevideo_queue(self.row[i].getwidth(),self.row[i].getheight(),self.filesFound[i],self.outputdir))

        return self.downsamplelistt

    def get_fpslist(self):
        self.fpslist = []
        for i in range(len(self.filesFound)):
            if (self.row[i].fpsvar.get()) == 1:
                self.fpslist.append(changefps_queue(self.row[i].getfps(),self.filesFound[i],self.outputdir))
        return self.fpslist

    def get_grayscalelist(self):
        self.grayscalelistt = []
        for i in range(len(self.filesFound)):
            if (self.row[i].grayscale()) == 1:
                self.grayscalelistt.append(greyscale_queue(self.outputdir,self.filesFound[i]))

        return self.grayscalelistt

    def get_superimposeframelist(self):
        self.superimposeframelistt = []
        for i in range(len(self.filesFound)):
            if (self.row[i].addframe()) == 1:
                self.superimposeframelistt.append(superimposeframe_queue(self.outputdir, self.filesFound[i]))

        return self.superimposeframelistt

    def execute_processvideo(self):
        # create a temp folder in the output dir
        tmp_folder = os.path.join(self.outputdir,'tmp')
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder) #make temp folder

        # remove process txt file if process were killed half way
        try:
            os.remove(os.path.join(self.outputdir,'process_video_define.txt'))
        except:
            print('Executing...')

        # compiling the list of commands
        try:
            crop = self.get_thecroplist()
            crop = [i for i in crop if i] ### remove any none in crop
        except:
            crop = []
        try:
            shorten = self.get_shortenlist()
        except:
            shorten = []
        try:
            downsample = self.get_downsamplelist()
        except:
            downsample = []

        try:
            chgfps = self.get_fpslist()

        except:
            chgfps = []
        try:
            grayscale = self.get_grayscalelist()
        except:
            grayscale = []
        try:
            superimpose = self.get_superimposeframelist()
        except:
            superimpose = []
        ## copy video and move it to output dir
        copyvideos = []
        for i in self.filesFound:
            if currentPlatform == 'Windows':
                command = 'copy \"' + str(self.videofolder) + '\\' + str(os.path.basename(i)) + '\" \"' + self.outputdir + '\"'
            if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
                command = 'cp "' + str(self.videofolder) + '/' + str(os.path.basename(i)) + '" "' + self.outputdir + '/"'
            copyvideos.append(command)

        #compiling all the commands into list
        all_list = copyvideos + crop + shorten + downsample + chgfps + grayscale + superimpose
        print(len(all_list))

        #creating text file
        filepath = os.path.join(self.outputdir, 'process_video_define.txt')

        if os.path.exists(filepath):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        ## writing into the txt file
        highscore = open(filepath, append_write)
        for i in all_list:
            highscore.write(i + '\n')
        highscore.close()
        ## running it using subprocess
        with open(filepath) as fp:
            for cnt, line in enumerate(fp):
                # add probably if ffmpeg then this if not then other subprocess
                subprocess.call(line, shell=True, stdout=subprocess.PIPE)
        ##clahe
        for i in range(len(self.filesFound)):
            if self.row[i].get_clahe() == 1:
                clahe_queue(os.path.join(self.outputdir, os.path.basename(self.filesFound[i])))
            else:
                print('Clahe not applied to',str(self.filesFound[i]))
        ##rename the txt file ran
        file = os.path.join(os.path.dirname(filepath), 'Processes_ran.txt')
        os.rename(filepath, file)
        dir = os.path.join(self.outputdir, 'process_archive')
        try:
            os.makedirs(dir)
            print("Directory", dir, "created ")
        except FileExistsError:
            print("Directory", dir, "already exists")

        currentDT = datetime.datetime.now()
        currentDT = str(currentDT.month) + '_' + str(currentDT.day) + '_' + str(currentDT.year) + '_' + str(currentDT.hour) + 'hour' + '_' + str(currentDT.minute) + 'min' + '_' + str(currentDT.second) + 'sec'
        try:
            shutil.move(file, dir)
        except shutil.Error:
            os.rename(file, file[:-4] + str(currentDT) + '.txt')
            shutil.move(file[:-4] + str(currentDT) + '.txt', dir)

        print('Process video completed.')

class batch_processvideo: ##pre process video first menu (ask for input and output folder)

    def __init__(self):
        # Popup window
        batchprocess = Toplevel()
        batchprocess.minsize(400, 200)
        batchprocess.wm_title("Batch process video")

        #Video Selection Tab
        label_videoselection = LabelFrame(batchprocess,text='Folder selection',font='bold',padx=5,pady=5)
        self.folder1Select = FolderSelect(label_videoselection,'Video directory:',title='Select Folder with videos')

        #output video
        self.outputfolder = FolderSelect(label_videoselection,'Output directory:',title='Select a folder for your output videos')

        #create list of all videos in the videos folder
        button_cL = Button(label_videoselection,text='Confirm',command=self.confirmtable)

        #organize
        label_videoselection.grid(row=0,sticky=W)
        self.folder1Select.grid(row=0,sticky=W)
        self.outputfolder.grid(row=1,sticky=W)
        button_cL.grid(row=2,sticky=W)

    def confirmtable(self):

        if (self.outputfolder.folder_path != 'No folder selected') and (self.folder1Select.folder_path!='No folder selected'):
            processvid_menu(self.folder1Select.folder_path, self.outputfolder.folder_path)
        elif (self.outputfolder.folder_path=='No folder selected'):
            print('Please select an "Output directory"')
        elif (self.folder1Select.folder_path == 'No folder selected'):
            print('Please select a "Video directory"')
        else:
            print('Please select a "Video directory" and an "Output directory"')

class outlier_settings:
    def __init__(self,configini):
        self.configini = configini
        #get the no of animals
        config = ConfigParser()
        configFile = str(configini)
        config.read(configFile)
        animalno = config.getint('General settings','animal_no')

        # get list from csv
        animalBpNames = define_bp_drop_down(configini)

        # Popup window
        outlier_set = Toplevel()
        outlier_set.minsize(400, 400)
        outlier_set.wm_title("Outlier Settings")

        scroll = LabelFrame(hxtScrollbar(outlier_set))
        scroll.grid()

        IDList = config.get('Multi animal IDs', 'id_list')
        if not IDList:
            IDList = []
            for i in range(animalno):
                IDList.append('Animal ' + str(i+1))
        else:
            IDList = config.get('Multi animal IDs', 'id_list')
            IDList = IDList.split(",")


        # location correction menu bar
        self.label_location_correction = LabelFrame(scroll, text='Location correction',font=('Times',12,'bold'),pady=5,padx=5)
        self.choosebp1List, self.choosebp2List, self.var1List, self.var2List, self.dropDownBp1List, self.dropDownBp2List  = [], [], [], [], [], []
        for animal in range(len(IDList)):
            animalIDname = IDList[animal]
            currTextBp1 = 'Choose ' + str(animalIDname) + ' body part 1:'
            currTextBp2 = 'Choose ' + str(animalIDname) + ' body part 2:'
            self.choosebp1List.append(Label(self.label_location_correction, text=currTextBp1))
            self.choosebp2List.append(Label(self.label_location_correction, text=currTextBp2))
            self.var1List.append(StringVar())
            self.var2List.append(StringVar())
            self.var1List[animal].set(animalBpNames[animal][0])
            self.var2List[animal].set(animalBpNames[animal][1])
            self.dropDownBp1List.append(OptionMenu(self.label_location_correction, self.var1List[animal], *animalBpNames[animal]))
            self.dropDownBp2List.append(OptionMenu(self.label_location_correction, self.var2List[animal], *animalBpNames[animal]))
        self.location_criterion = Entry_Box(self.label_location_correction, 'Location criterion', '15')

        # movement
        self.label_movement_correction = LabelFrame(scroll, text='Movement correction', font=('Times', 12, 'bold'),pady=5, padx=5)
        self.choosebp1ListMov, self.choosebp2ListMov, self.var1ListMov, self.var2ListMov, self.dropDownBp1ListMov, self.dropDownBp2ListMov = [], [], [], [], [], []
        for animal in range(len(IDList)):
            animalIDname = IDList[animal]
            currTextBp1 = 'Choose ' + str(animalIDname) + ' body part 1:'
            currTextBp2 = 'Choose ' + str(animalIDname) + ' body part 2:'
            self.choosebp1ListMov.append(Label(self.label_movement_correction, text=currTextBp1))
            self.choosebp2ListMov.append(Label(self.label_movement_correction, text=currTextBp2))
            self.var1ListMov.append(StringVar())
            self.var2ListMov.append(StringVar())
            self.var1ListMov[animal].set(animalBpNames[animal][0])
            self.var2ListMov[animal].set(animalBpNames[animal][1])
            self.dropDownBp1ListMov.append(OptionMenu(self.label_movement_correction, self.var1ListMov[animal], *animalBpNames[animal]))
            self.dropDownBp2ListMov.append(OptionMenu(self.label_movement_correction, self.var2ListMov[animal], *animalBpNames[animal]))
        self.movement_criterion = Entry_Box(self.label_movement_correction, 'Movement criterion', '15')

        # mean or median
        medianlist = ['mean', 'median']
        self.medianvar = StringVar()
        self.medianvar.set(medianlist[0])
        label_median = LabelFrame(scroll, text='Median or Mean', font=('Times', 12, 'bold'), pady=5, padx=5)
        mediandropdown = OptionMenu(label_median, self.medianvar, *medianlist)

        button_setvalues = Button(scroll, text='Confirm', command=self.set_outliersettings, font=('Arial', 12, 'bold'),fg='red')

        self.label_location_correction.grid(row=0, sticky=W)
        self.location_criterion.grid(row=100, column=0, sticky=W)
        for row, dropdown in zip(range(0, len(IDList)+4, 2), range(len(IDList)+1)):
            try:
                self.choosebp1List[dropdown].grid(row=row, column=0, sticky=W)
                self.dropDownBp1List[dropdown].grid(row=row, column=1, sticky=W)
                self.choosebp2List[dropdown].grid(row=row+1, column=0, sticky=W)
                self.dropDownBp2List[dropdown].grid(row=row+1, column=1, sticky=W)
            except IndexError:
                pass

        self.label_movement_correction.grid(row=1, sticky=W)
        self.movement_criterion.grid(row=100, sticky=W)
        for row, dropdown in zip(range(0, len(IDList) + 2, 2), range(len(IDList) + 1)):
            try:
                self.choosebp1ListMov[dropdown].grid(row=row, column=0, sticky=W)
                self.dropDownBp1ListMov[dropdown].grid(row=row, column=1, sticky=W)
                self.choosebp2ListMov[dropdown].grid(row=row + 1, column=0, sticky=W)
                self.dropDownBp2ListMov[dropdown].grid(row=row + 1, column=1, sticky=W)
            except IndexError:
                pass

        label_median.grid(row=2,column=0,sticky=W)
        mediandropdown.grid(row=2,sticky=W)
        button_setvalues.grid(row=3,pady=10)

    def set_outliersettings(self):
        # export settings to config ini file
        configini = self.configini
        config = ConfigParser()
        config.read(configini)
        animalno = config.getint('General settings', 'animal_no')
        animalNameList = []
        try:
            multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
            multiAnimalIDList = multiAnimalIDList.split(",")

        except NoSectionError:
            multiAnimalIDList = ['']

        if multiAnimalIDList[0] == '':
            for animal in range(animalno):
                animalNameList.append('Animal_' + str(animal+1))
        else:
            animalNameList = multiAnimalIDList


        try:
            for animal in range(len(animalNameList)):
                locBp1 = self.var1List[animal].get()
                locBp2 = self.var2List[animal].get()
                movBp1 = self.var1ListMov[animal].get()
                movBp2 = self.var2ListMov[animal].get()
                config.set('Outlier settings', 'movement_bodyPart1_' + str(animalNameList[animal]), str(movBp1))
                config.set('Outlier settings', 'movement_bodyPart2_' + str(animalNameList[animal]), str(movBp2))
                config.set('Outlier settings', 'location_bodyPart1_' + str(animalNameList[animal]), str(locBp1))
                config.set('Outlier settings', 'location_bodyPart2_' + str(animalNameList[animal]), str(locBp2))
            movementcriterion = self.movement_criterion.entry_get
            locationcriterion = self.location_criterion.entry_get
            mean_or_median = self.medianvar.get()
            config.set('Outlier settings', 'movement_criterion', str(movementcriterion))
            config.set('Outlier settings', 'location_criterion', str(locationcriterion))
            config.set('Outlier settings', 'mean_or_median', str(mean_or_median))

            with open(configini, 'w') as configfile:
                config.write(configfile)
            print('Outlier correction settings updated in project_config.ini')
        except:
            print('Please make sure all fields are filled in correctly.')

class FolderSelect(Frame):
    def __init__(self,parent=None,folderDescription="",color=None,title=None,lblwidth =None,**kw):
        self.title=title
        self.color = color if color is not None else 'black'
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self,master=parent,**kw)
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription,fg=str(self.color),width=str(self.lblwidth),anchor=W)
        self.lblName.grid(row=0,column=0,sticky=W)
        self.entPath = Label(self, textvariable=self.folderPath,relief=SUNKEN)
        self.entPath.grid(row=0,column=1)
        self.btnFind = Button(self, text="Browse Folder",command=self.setFolderPath)
        self.btnFind.grid(row=0,column=2)
        self.folderPath.set('No folder selected')
    def setFolderPath(self):
        folder_selected = askdirectory(title=str(self.title),parent=self.parent)
        if folder_selected:
            self.folderPath.set(folder_selected)
        else:
            self.folderPath.set('No folder selected')

    @property
    def folder_path(self):
        return self.folderPath.get()

class DropDownMenu(Frame):
    def __init__(self,parent=None,dropdownLabel='',choice_dict=None,labelwidth='',com=None,**kw):
        Frame.__init__(self,master=parent,**kw)
        self.dropdownvar = StringVar()
        self.lblName = Label(self,text=dropdownLabel,width=labelwidth,anchor=W)
        self.lblName.grid(row=0,column=0)
        self.choices = choice_dict
        self.popupMenu = OptionMenu(self,self.dropdownvar,*self.choices,command=com)
        self.popupMenu.grid(row=0,column=1)
    def getChoices(self):
        return self.dropdownvar.get()
    def setChoices(self,choice):
        self.dropdownvar.set(choice)

class FileSelect(Frame):
    def __init__(self,parent=None,fileDescription="",color=None,title=None,lblwidth=None,**kw):
        self.title=title
        self.color = color if color is not None else 'black'
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent=parent
        Frame.__init__(self,master=parent,**kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription,fg=str(self.color),width=str(self.lblwidth),anchor=W)
        self.lblName.grid(row=0,column=0,sticky=W)
        self.entPath = Label(self, textvariable=self.filePath,relief=SUNKEN)
        self.entPath.grid(row=0,column=1)
        self.btnFind = Button(self, text="Browse File",command=self.setFilePath)
        self.btnFind.grid(row=0,column=2)
        self.filePath.set('No file selected')
    def setFilePath(self):
        file_selected = askopenfilename(title=self.title,parent=self.parent)
        if file_selected:
            self.filePath.set(file_selected)
        else:
            self.filePath.set('No file selected')
    @property
    def file_path(self):
        return self.filePath.get()

class Entry_Box(Frame):
    def __init__(self, parent=None, fileDescription="", labelwidth='',status=None, validation=None, **kw):
        super(Entry_Box, self).__init__(master=parent)
        self.validation_methods = {
            'numeric': (self.register(form_validator_is_numeric), '%P', '%d'),
        }
        self.status = status if status is not None else NORMAL
        self.labelname = fileDescription
        Frame.__init__(self,master=parent,**kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription,width=labelwidth,anchor=W)
        self.lblName.grid(row=0,column=0)
        self.entPath = Entry(self, textvariable=self.filePath, state=self.status,
                             validate='key',
                             validatecommand=self.validation_methods.get(validation, None))
        self.entPath.grid(row=0,column=1)

    @property
    def entry_get(self):
        self.entPath.get()
        return self.entPath.get()

    def entry_set(self, val):
        self.filePath.set(val)

    def set_state(self,setstatus):
        self.entPath.config(state=setstatus)

    def destroy(self):
        self.lblName.destroy()
        self.entPath.destroy()

class newcolumn(Frame):
    def __init__(self,parent=None,lengthoflist=[],width='',**kw):
        Frame.__init__(self,master=parent,**kw)
        self.entPath = []
        self.entPathvars =[]
        for i in range(len(lengthoflist)):
            self.entPathvar = IntVar()
            self.entPathvars.append(self.entPathvar)
            self.entPath.append(Entry(self, textvariable=self.entPathvars[i],width=width))
            self.entPath[i].grid(row=i,pady=3)

    def entry_get(self,row):
        return self.entPath[row].get()

    def setvariable(self,row,vars):
        return self.entPathvars[row].set(vars)

class Button_getcoord(Frame):

    def __init__(self,parent=None,filename=[],knownmm=[],ppmlist = None,**kw): #set to list and use range i in list to call each elements in list

        Frame.__init__(self, master=parent, **kw)
        self.entPath = []
        self.ppm_list = []
        self.ppmvar = []
        self.filename = filename
        labelgetcoord =Label(self,text='Get coord')
        labelgetcoord.grid(row=0,pady=6)

        labelppm =Label(self,text='Pixels/mm')
        labelppm.grid(row=0,column=1)

        for i in range(len(filename)-1):
            self.entPath.append(Button(self,text='Video'+str(i+1),command =lambda i=i :self.getcoord_forbutton(filename[i+1],knownmm[i+1],i)))
            self.entPath[i].grid(row=i+1)
            self.ppmvars=IntVar()
            self.ppmvar.append(self.ppmvars)
            self.ppm_list.append(Entry(self,textvariable=self.ppmvar[i]))
            self.ppm_list[i].grid(row=i+1,column=1)
        if ppmlist != None:
            try:
                for i in range(len(self.ppmvar)):
                    self.ppmvar[i].set(ppmlist[i])
            except IndexError:
                pass

    def getcoord_forbutton(self,filename,knownmm,count):

        ppm = get_coordinates_nilsson(filename,knownmm)

        if ppm == float('inf'):
            print('Divide by zero error. Please make sure the values in [Distance_in_mm] are updated')
        else:
            self.ppmvar[count].set(ppm)

    def getppm(self,count):
        ppms = self.ppm_list[count].get()
        return ppms

    def set_allppm(self,value):
        for i in range(len(self.filename) - 1):
            self.ppmvar[i].set(value)

def Exit():
    app.root.destroy()


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
        acanvas.bind('<Leave>', lambda event: unbindToMousewheel(event,acanvas))
        return frame



class video_info_table:
    def __init__(self,configini):
        self.filesFound = [0]  #initiate files found
        config = ConfigParser()
        self.configFile = str(configini)
        config.read(self.configFile)
        projectPath = config.get('General settings', 'project_path')
        video_info_csv = os.path.join(projectPath, 'logs', 'video_info.csv')
        self.config_videofolders = os.path.join(projectPath, 'videos')
        config_distancemm = config.get('Frame settings', 'distance_mm')

        ########### FIND FILES ###########
        for i in os.listdir(self.config_videofolders):
            if i.endswith(('.avi','.mp4','.mov','flv','m4v')):
                self.filesFound.append(i)

        # if csv exist, find the difference and append
        if os.path.exists(video_info_csv):
            df = pd.read_csv(video_info_csv)
            videodf = df['Video'].to_list()
            videodf = [s +'.mp4' for s in videodf]
            videodf = list(set(videodf) - set(self.filesFound))
            self.filesFound += videodf

        ##GUI
        self.tkintertable = Toplevel()
        self.tkintertable.minsize(1000, 500)
        self.tkintertable.wm_title("Video Info")

        self.xscrollbar = Canvas(hxtScrollbar(self.tkintertable))
        self.xscrollbar.pack(expand=True,fill=BOTH)
        self.myframe = LabelFrame(self.xscrollbar,text='Table')
        self.myframe.grid(row=6)

        self.new_col_list = ['index','Video','fps','Resolution_width','Resolution_height','Distance_in_mm']
        self.table_col=[]
        self.col_width= ['6','35','20','20','20','20']

        #### loop for tables######
        for i in range(len(self.new_col_list)):
            self.table_col.append(newcolumn(self.myframe,self.filesFound,self.col_width[i]))
            self.table_col[i].grid(row=0,column=i, sticky=W)

        ###set values for base####
        count = 0
        for i in self.filesFound:
            currvid= os.path.join(str(self.config_videofolders),str(i))
            if os.path.exists(currvid) or i==0:
                vid= cv2.VideoCapture(currvid)
                self.table_col[0].setvariable(count,str(count)+'.')
                self.table_col[1].setvariable(count,i)
                self.table_col[2].setvariable(count, int(vid.get(cv2.CAP_PROP_FPS)))
                self.table_col[3].setvariable(count, int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)))
                self.table_col[4].setvariable(count, int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                self.table_col[5].setvariable(count,config_distancemm)
            else:
                self.table_col[0].setvariable(count, str(count) + '.')
                self.table_col[1].setvariable(count, i)
                self.table_col[2].setvariable(count, int(df.loc[df['Video']==i.split('.')[0]].values.tolist()[0][1]))
                self.table_col[3].setvariable(count, int(df.loc[df['Video']==i.split('.')[0]].values.tolist()[0][2]))
                self.table_col[4].setvariable(count, int(df.loc[df['Video']==i.split('.')[0]].values.tolist()[0][3]))
                self.table_col[5].setvariable(count, config_distancemm)
            count+=1

        #set title
        count=0
        for i in self.new_col_list:
            self.table_col[count].setvariable(0,i)
            count+=1

        ###set up get coord button on table####
        self.data_lists = []
        for i in range(len(self.table_col)):
            self.data_lists.append([])

        for i in self.filesFound:
            self.data_lists[1].append(os.path.join(self.config_videofolders, str(i)))
            self.data_lists[5].append(int(0))

        if os.path.exists(video_info_csv):
            df = pd.read_csv(video_info_csv)
            videoincsv = df['Video'].tolist()
            pixelsincsv = df['pixels/mm'].tolist()
            df_dict = dict(zip(videoincsv,pixelsincsv))
            videoslist = self.filesFound.copy()
            videoslist = videoslist[1:]

            for i in range(len(videoslist)):
                videoslist[i] = str(videoslist[i]).split('.')[0]

            videosdict = dict(zip(videoslist,range(len(videoslist))))

            for keys in videosdict.keys():
                if keys in df_dict:
                    videosdict[keys] = df_dict[keys]
                else:
                    videosdict[keys] = 0
            pixellist = list(videosdict.values())
            self.pixel_list = pixellist
            self.getdata()
            self.button = Button_getcoord(self.xscrollbar, self.data_lists[1], self.data_lists[5],pixellist)
            self.button.grid(row=6, column=1)

        else:
            self.pixel_list= None
            self.getdata()
            self.button = Button_getcoord(self.xscrollbar,self.data_lists[1],self.data_lists[5])
            self.button.grid(row=6,column=1)


        #instructions
        label_getdatadesc1 = Label(self.xscrollbar,text='1. Enter the known distance (mm) in the "Distance_in_mm" column. Consider using the "autopopulate" entry box if you have a lot of videos.')
        label_getdatadesc2 = Label(self.xscrollbar,text='2. Click on <Update distance_in_mm> button before clicking on the "Get coord" button(s) to calculate pixels/mm.')
        label_getdatadesc3 = Label(self.xscrollbar,text='3. Click <Save Data> when all the data are filled in. Use the <Add Columns> button to add infmation on each video, e.g., animal ID or experimental group.')
        label_getdatadesc1.grid(row=0,sticky=W)
        label_getdatadesc2.grid(row=1, sticky=W)
        label_getdatadesc3.grid(row=2, sticky=W)

        get_data_button = Button(self.xscrollbar, text='Update distance_in_mm', command=self.getdata)
        get_data_button.grid(row=3,sticky=W)

        add_column_button = Button(self.xscrollbar,text='<Add Column>',command=self.addBox,fg='red')
        add_column_button.grid(row=4,sticky=W)

        generate_csv_button = Button(self.xscrollbar,text='Save Data',command=self.generate_video_info_csv,font='bold',fg='red')
        generate_csv_button.grid(row=5)

        setppmbutton = Button(self.xscrollbar,text='Duplicate index 1 pixel/mm (CAUTION!)',command= self.setAll_ppm, fg='red')
        setppmbutton.grid(row =5,column=1)


    def addBox(self):
        self.new_col_list.append(0)
        self.next_column = len(self.new_col_list)
        #print(self.next_column)
        self.table_col.append(newcolumn(self.myframe,self.filesFound,'20'))
        self.table_col[(self.next_column)-1].grid(row=0,column=self.next_column)

    def getdata(self):
        self.data_lists =[]
        #get all data from tables
        for i in range(len(self.table_col)):
            self.data_lists.append([])
            for j in range(len(self.filesFound)):
                self.data_lists[i].append(self.table_col[i].entry_get(j))

        # add path to videos for get coord
        if currentPlatform == 'Windows':
            self.data_lists[1] = [str(self.config_videofolders)+'\\'+s for s in self.data_lists[1]]
        if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
            self.data_lists[1] = [str(self.config_videofolders)+'/'+s for s in self.data_lists[1]]



        if self.pixel_list!= None:
            self.button = Button_getcoord(self.xscrollbar, self.data_lists[1], self.data_lists[5],self.pixel_list)
            self.button.grid(row=6, column=1)
        else:
            #update get coord with data
            self.button = Button_getcoord(self.xscrollbar,self.data_lists[1],self.data_lists[5])
            self.button.grid(row=6,column=1)

        print("Table updated.")

    def generate_video_info_csv(self):
        #get latest data from table
        self.data_lists = []
        # get all data from tables
        for i in range(len(self.table_col)):
            self.data_lists.append([])
            for j in range(len(self.filesFound)):
                self.data_lists[i].append(self.table_col[i].entry_get(j))
        #get the ppm from table
        self.ppm=['pixels/mm']
        for i in range(len(self.filesFound)-1):
            self.ppm.append((self.button.getppm(i)))
        self.data_lists.append(self.ppm)

        #remove .mp4 from first column
        self.data_lists[1] = [i.replace(i[-4:],'') for i in self.data_lists[1]]
        self.data_lists[1][0] ='Video'

        data=self.data_lists
        df=pd.DataFrame(data=data)
        df=df.transpose()
        df=df.rename(columns=df.iloc[0])
        df=df.drop(df.index[0])
        df=df.reset_index()
        df=df.drop(['index'],axis=1)
        df=df.drop(['level_0'],axis=1)


        logFolder = os.path.join(os.path.dirname(self.configFile), 'logs')
        csv_filename = 'video_info.csv'
        output = os.path.join(logFolder, csv_filename)


        df.to_csv(str(output),index=False)
        print(os.path.dirname(output),'generated.')

    def setAll_ppm(self):
        firstvalue = self.button.getppm(0)
        self.button.set_allppm(firstvalue)


class video_downsample:

    def __init__(self):
        # Popup window
        videosdownsample = Toplevel()
        videosdownsample.minsize(200, 200)
        videosdownsample.wm_title("Downsample Video Resolution")


        # Video Path
        self.videopath1selected = FileSelect(videosdownsample, "Video path",title='Select a video file')
        label_choiceq = Label(videosdownsample, text='Choose only one of the following method to downsample videos (Custom/Default)')

        #custom reso
        label_downsamplevidcustom = LabelFrame(videosdownsample,text='Custom resolution',font='bold',padx=5,pady=5)
        # width
        self.label_width = Entry_Box(label_downsamplevidcustom,'Width','10')
        # height
        self.label_height = Entry_Box(label_downsamplevidcustom,'Height','10')
        # confirm custom resolution
        self.button_downsamplevideo1 = Button(label_downsamplevidcustom, text='Downsample to custom resolution',command=self.downsample_customreso)
        #Default reso
        # Checkbox
        label_downsampleviddefault = LabelFrame(videosdownsample,text='Default resolution',font ='bold',padx=5,pady=5)
        self.var1 = IntVar()
        self.checkbox1 = Radiobutton(label_downsampleviddefault, text="1980 x 1080", variable=self.var1,value=1)
        self.checkbox2 = Radiobutton(label_downsampleviddefault, text="1280 x 720", variable=self.var1,value=2)
        self.checkbox3 = Radiobutton(label_downsampleviddefault, text="720 x 480", variable=self.var1, value=3)
        self.checkbox4 = Radiobutton(label_downsampleviddefault, text="640 x 480", variable=self.var1, value=4)
        self.checkbox5 = Radiobutton(label_downsampleviddefault, text="320 x 240", variable=self.var1, value=5)
        # Downsample video
        self.button_downsamplevideo2 = Button(label_downsampleviddefault, text='Downsample to default resolution',command=self.downsample_defaultreso)

        # Organize the window
        self.videopath1selected.grid(row=0,sticky=W)
        label_choiceq.grid(row=1, sticky=W,pady=10)

        label_downsamplevidcustom.grid(row=2,sticky=W,pady=10)
        self.label_width.grid(row=0, column=0,sticky=W)
        self.label_height.grid(row=1, column=0,sticky=W)
        self.button_downsamplevideo1.grid(row=3)

        label_downsampleviddefault.grid(row=3,sticky=W,pady=10)
        self.checkbox1.grid(row=0,stick=W)
        self.checkbox2.grid(row=1,sticky=W)
        self.checkbox3.grid(row=2, sticky=W)
        self.checkbox4.grid(row=3, sticky=W)
        self.checkbox5.grid(row=4, sticky=W)
        self.button_downsamplevideo2.grid(row=5)

    def downsample_customreso(self):
        self.width1 = self.label_width.entry_get
        self.height1 = self.label_height.entry_get

        ds = downsamplevideo(self.width1, self.height1, self.videopath1selected.file_path)

    def downsample_defaultreso(self):

        if self.var1.get()==1:
            self.width2 = str(1980)
            self.height2 = str(1080)
            print('The width selected is ' + str(self.width2) + ', the height is ' + str(self.height2))

        elif self.var1.get()==2:
            self.width2 = str(1280)
            self.height2 = str(720)
            print('The width selected is ' + str(self.width2) + ', the height is ' + str(self.height2))

        elif self.var1.get()==3:
            self.width2 = str(720)
            self.height2 = str(480)
            print('The width selected is ' + str(self.width2) + ', the height is ' + str(self.height2))

        elif self.var1.get()==4:
            self.width2 = str(640)
            self.height2 = str(480)
            print('The width selected is ' + str(self.width2) + ', the height is ' + str(self.height2))

        elif self.var1.get()==5:
            self.width2 = str(320)
            self.height2 = str(240)
            print('The width selected is ' + str(self.width2) + ', the height is ' + str(self.height2))

        ds = downsamplevideo(self.width2, self.height2, self.videopath1selected.file_path)

class Red_light_Convertion:

    def __init__(self):
        # Popup window
        redlightconversion = Toplevel()
        redlightconversion.minsize(200, 200)
        redlightconversion.wm_title("CLAHE")


        #CLAHE
        label_clahe = LabelFrame(redlightconversion,text='Contrast Limited Adaptive Histogram Equalization',font='bold',padx=5,pady=5)
        # Video Path
        self.videopath1selected = FileSelect(label_clahe, "Video path ",title='Select a video file')
        button_clahe = Button(label_clahe,text='Apply CLAHE',command=lambda:clahe(self.videopath1selected.file_path))

        #organize the window
        label_clahe.grid(row=0,sticky=W)
        self.videopath1selected.grid(row=0,sticky=W)
        button_clahe.grid(row=1,pady=5)

class crop_video:

    def __init__(self):
        # Popup window
        cropvideo = Toplevel()
        cropvideo.minsize(300, 300)
        cropvideo.wm_title("Crop Video")

        # Normal crop
        label_cropvideo = LabelFrame(cropvideo,text='Crop Video',font='bold',padx=5,pady=5)
        self.videopath1selected = FileSelect(label_cropvideo,"Video path",title='Select a video file')
        # CropVideo
        button_cropvid = Button(label_cropvideo, text='Crop Video', command=lambda:cropvid(self.videopath1selected.file_path))

        # fixed crop
        label_videoselection = LabelFrame(cropvideo, text='Fixed coordinates crop for multiple videos', font='bold', padx=5, pady=5)
        self.folder1Select = FolderSelect(label_videoselection, 'Video directory:', title='Select Folder with videos')
        # output video
        self.outputfolder = FolderSelect(label_videoselection, 'Output directory:',
                                         title='Select a folder for your output videos')

        # create list of all videos in the videos folder
        button_cL = Button(label_videoselection, text='Confirm', command=lambda: youOnlyCropOnce(self.folder1Select.folder_path,self.outputfolder.folder_path))


        #organize
        label_cropvideo.grid(row=0,sticky=W)
        self.videopath1selected.grid(row=0,sticky=W)
        button_cropvid.grid(row=1,sticky=W,pady=10)
        #fixedcrop
        label_videoselection.grid(row=1,sticky=W,pady=10,padx=5)
        self.folder1Select.grid(row=0,sticky=W,pady=5)
        self.outputfolder.grid(row=1,sticky=W,pady=5)
        button_cL.grid(row=2,sticky=W,pady=5)

class create_project_DLC:

    def __init__(self):

        # Popup window
        createproject = Toplevel()
        createproject.minsize(400, 250)
        createproject.wm_title("Create Project")


        self.label_dlc_createproject = LabelFrame(createproject,text='Create Project',font =("Helvetica",12,'bold'))
        #project name
        self.label_projectname = Entry_Box(self.label_dlc_createproject,'Project name','16')

        #Experimenter name
        self.label_experimentername = Entry_Box(self.label_dlc_createproject,'Experimenter name','16')

        #button 1
        button_videofol = Button(self.label_dlc_createproject,text='Import Single Video',command=self.changetovideo2,fg='blue')
        #button 2
        button_videofo2 = Button(self.label_dlc_createproject,text='Import Multiple Videos',command=self.changetovideo,fg='green4')

        # Video Path
        self.videopath1selected = FolderSelect(self.label_dlc_createproject, 'Video Folder          ',title='Select folder with videos',color='green4')
        self.videopath1selected.grid(row=4, sticky=W)

        #video folder
        self.folderpath1selected = FolderSelect(self.label_dlc_createproject,'Project directory   ',title='Select main directory')

        #bodypart configuration file
        self.bodypartconfigfile = FileSelect(self.label_dlc_createproject, 'Bp config file ', title='Select a csv file')

        # # statusbar
        # self.projectcreated = IntVar()
        # Label(createproject, textvariable=self.projectcreated, bd=1, relief=SUNKEN).grid(row=7,sticky=W)
        # self.projectcreated.set('Status: Waiting for input...')

        #checkbox_apply golden aggresion config yaml settings
        self.var_changeyaml = IntVar()
        checkbox2 = Checkbutton(self.label_dlc_createproject,text='Apply Golden Lab 16-body part config',variable=self.var_changeyaml)

        #checkbox for copy videos true or false
        self.var_copyvid = IntVar()
        checkbox1 = Checkbutton(self.label_dlc_createproject,text='Copy videos (If unchecked, shortcuts are created)',variable=self.var_copyvid)

        #run create project
        button_createproject = Button(self.label_dlc_createproject,text='Create Project',fg='red',command=self.createprojectcommand)

        #organize
        self.label_dlc_createproject.grid(row=0)
        self.label_projectname.grid(row=0,column=0,sticky=W)
        self.label_experimentername.grid(row=1,column=0,sticky=W)
        button_videofol.grid(row=2,sticky=W,pady=5)
        button_videofo2.grid(row=3,sticky=W,pady=5)
        self.folderpath1selected.grid(row=5,sticky=W)
        self.bodypartconfigfile.grid(row=6, sticky=W)
        checkbox2.grid(row=7,column=0,sticky=W)
        checkbox1.grid(row=8,sticky=W)
        button_createproject.grid(row=9,column=3,pady=10,padx=5)

    def changetovideo(self):
        self.videopath1selected.grid_remove()
        self.videopath1selected = FolderSelect(self.label_dlc_createproject, 'Video Folder          ',title='Select folder with videos',color='green4')
        self.videopath1selected.grid(row=4, sticky=W)


    def changetovideo2(self):
        self.videopath1selected.grid_remove()
        self.videopath1selected = FileSelect(self.label_dlc_createproject, 'Video path             ',color='blue',title='Select a video file')
        self.videopath1selected.grid(row=4, sticky=W)

    def createprojectcommand(self):
        projectname = self.label_projectname.entry_get
        experimentalname = self.label_experimentername.entry_get
        if self.var_copyvid.get()==1:
            copyvid = True
        elif self.var_copyvid.get()==0:
            copyvid = False

        if 'FileSelect' in str(type(self.videopath1selected)):
            videolist = [self.videopath1selected.file_path]
        else:
            try:
                videolist = []

                for i in os.listdir(self.videopath1selected.folder_path):
                    if ('.avi' or '.mp4') in i:
                        i = os.path.join(self.videopath1selected.folder_path, i)
                        videolist.append(i)
            except:
                print('Please select a video folder to import videos')

        if 'FileSelect' in str(type(self.bodypartconfigfile)):
            bodyPartConfigFile = (self.bodypartconfigfile.file_path)


        if self.var_changeyaml.get()==1:
            if (projectname !='') and (experimentalname !='') and ('No'and'selected' not in videolist) and (self.folderpath1selected.folder_path!='No folder selected'):
                config_path = deeplabcut.create_new_project(str(projectname), str(experimentalname), videolist,working_directory=str(self.folderpath1selected.folder_path), copy_videos=copyvid)
                changedlc_config(config_path, 0)
            else:
                print('Please make sure all the information are filled in')
        else:
            if (projectname != '') and (experimentalname != '') and ('No' and 'selected' not in videolist) and (self.folderpath1selected.folder_path != 'No folder selected'):
                config_path = deeplabcut.create_new_project(str(projectname), str(experimentalname), videolist,working_directory=str(self.folderpath1selected.folder_path), copy_videos=copyvid)
            else:
                print('Please make sure all the information are filled in')
        if bodyPartConfigFile != 'No file selected':
            changedlc_config(config_path, bodyPartConfigFile)

class createDPK_project:
    def __init__(self):
        cdpkmenu = Toplevel()
        cdpkmenu.minsize(300,200)
        cdpkmenu.wm_title('Create DeepPoseKit project')

        cdpk_label = LabelFrame(cdpkmenu,pady=5,padx=5)
        self.inifile = FileSelect(cdpk_label,'SimBA project.ini file',lblwidth='18')
        self.projectName = Entry_Box(cdpk_label,'Project Name','18')

        #button
        generate_Button = Button(cdpk_label,text='Generate project',command=self.generateProject)

        #organize
        cdpk_label.grid(row=0,sticky=W)
        self.inifile.grid(row=0,sticky=W)
        self.projectName.grid(row=1,sticky=W)
        generate_Button.grid(row=2,sticky=W)

    def generateProject(self):
        projectdir = os.path.dirname(self.inifile.file_path)
        write_dpkfile(projectdir,self.projectName.entry_get)
        print('DeepPoseKit project generated.')


class deepPoseKitMenu:
    def __init__(self,inifile):
        self.configini = inifile
        # Popup window
        dpkMenu = Toplevel()
        dpkMenu.minsize(400, 400)
        dpkMenu.wm_title("DeepPoseKit")

        #parent tab
        tab_parent = ttk.Notebook(dpkMenu)
        #children tab
        tab0 = ttk.Frame(tab_parent)
        tab1 = ttk.Frame(tab_parent)
        tab2 = ttk.Frame(tab_parent)
        tab3 = ttk.Frame(tab_parent)
        tab4 = ttk.Frame(tab_parent)
        tab5 = ttk.Frame(tab_parent)

        tab_parent.add(tab0, text=f'{"[ Import Videos ]": ^20s}')
        tab_parent.add(tab1, text=f'{"[ Create annotation set ]": ^20s}')
        tab_parent.add(tab2, text=f'{"[ Annotater ]": ^20s}')
        tab_parent.add(tab3, text=f'{"[ Train model ]": ^20s}')
        tab_parent.add(tab4, text=f'{"[ Predict new video ]": ^20s}')
        tab_parent.add(tab5, text=f'{"[ Visualize video ]": ^20s}')

        tab_parent.grid(row=0)

        #import videos
        label_importvideo = LabelFrame(tab0, text='Import videos into DPK project', font=("Helvetica",12,'bold'), padx=15,pady=5,fg='black')
        # multi video
        label_multivideoimport = LabelFrame(label_importvideo, text='Import multiple videos', pady=5, padx=5)
        self.multivideofolderpath = FolderSelect(label_multivideoimport, 'Folder path',title='Select Folder with videos')
        self.video_type = Entry_Box(label_multivideoimport, 'File format (i.e., mp4 or avi):', '20')
        button_multivideoimport = Button(label_multivideoimport, text='Import multiple videos',command=self.importvideo_multi, fg='black')
        # singlevideo
        label_singlevideoimport = LabelFrame(label_importvideo, text='Import single video', pady=5, padx=5)
        self.singlevideopath = FileSelect(label_singlevideoimport, "Video Path",title='Select a video file')
        button_importsinglevideo = Button(label_singlevideoimport, text='Import a video',command= self.importvideo_single,fg='black')

        #Create annotation
        createann_label = LabelFrame(tab1,pady=5,padx=5)
        annOutputName = Entry_Box(createann_label,'Annotation Output Name','21')
        readBatchSize = Entry_Box(createann_label,'Read Batch Size','21')
        kmeanbSize = Entry_Box(createann_label, 'k means batch size', '21')
        kmeanNcluster = Entry_Box(createann_label, 'k means n clusters', '21')
        kmeanmaxIter = Entry_Box(createann_label, 'k means max iterations', '21')
        kmeanNinit = Entry_Box(createann_label, 'k means n init', '21')
        crateaAnnButton = Button(createann_label,text='Create Annotation',command=lambda:self.dpkStep1(annOutputName.entry_get,readBatchSize.entry_get,kmeanbSize.entry_get,kmeanNcluster.entry_get,kmeanmaxIter.entry_get,kmeanNinit.entry_get))

        ## set
        readBatchSize.entry_set('100')
        kmeanbSize.entry_set('100')
        kmeanNcluster.entry_set('10')
        kmeanmaxIter.entry_set('1000')
        kmeanNinit.entry_set('10')

        #Annotator
        annfile_label = LabelFrame(tab2,pady=5,padx=5)
        annfile = FileSelect(annfile_label,'Annotation file',lblwidth='18')
        #button
        runann_button = Button(annfile_label,text='Run',command=lambda:dpkAnnotator(self.configini,annfile.file_path))

        def activate(box, *args):
            for entry in args:
                if box.get() == 0:
                    entry.set_state(DISABLED)
                elif box.get() == 1:
                    entry.set_state(NORMAL)

        #train model deeposekit
        trainmodel_label = LabelFrame(tab3,pady=5,padx=5)
        annpath = FileSelect(trainmodel_label,'Annotations path',lblwidth='18')
        savemodelPath = Entry_Box(trainmodel_label,'Output model name','18')
        epochs = Entry_Box(trainmodel_label,'epochs','18')
        dsFactor = Entry_Box(trainmodel_label,'Downsample factor','18')
        validationSplit = Entry_Box(trainmodel_label,'Validation split','18')
        sigma = Entry_Box(trainmodel_label,'Sigma','18')
        graphscale = Entry_Box(trainmodel_label,'Graph scale','18')
        TorFoptions = {'True','False'}
        augmenter = DropDownMenu(trainmodel_label,'Augmenter',TorFoptions,'18')
        validationbatchSize = Entry_Box(trainmodel_label,'Validation batch size','18')
        modelgrowthrate = Entry_Box(trainmodel_label,'Model growth rate','18')
        modelbatchsize = Entry_Box(trainmodel_label,'Model Batch Size','18')

        #logger
        loggervar = IntVar(value=1)
        loggervalidBatchSize = Entry_Box(trainmodel_label, 'Logger validation batch size', '18')
        loggercheckbox = Checkbutton(trainmodel_label, text='Logger', variable=loggervar,
                                     command=lambda: activate(loggervar, loggervalidBatchSize))

        # lr factor
        reducevar = IntVar(value=1)
        reducelrfactor = Entry_Box(trainmodel_label, 'Reduce LR factor', '18')
        reducelrcheckbox = Checkbutton(trainmodel_label, text='Reduce_Lr', variable=reducevar,
                                       command=lambda: activate(reducevar, reducelrfactor))

        checkpointscheckbox = DropDownMenu(trainmodel_label,'Model Checkpoints',TorFoptions,'18')
        earlyStopcheckbox = DropDownMenu(trainmodel_label,'Early stop',TorFoptions,'18')

        ## choose model
        modelsoption = {'DeepLabCut','LEAP','StackedDenseNet','StackedHourglass'}
        self.nnArchitecture = DropDownMenu(trainmodel_label,'NN_architecture',modelsoption,'18')
        modelsettingsbutton = Button(trainmodel_label,text='Model Settings',command=self.modelsettingschoice)

        #button
        trainmodelbutton = Button(trainmodel_label,text='Train model',command = lambda:self.traindpk(annpath.file_path,savemodelPath.entry_get,epochs.entry_get,dsFactor.entry_get,validationSplit.entry_get,sigma.entry_get,
                                                                                                     graphscale.entry_get,augmenter.getChoices(),validationbatchSize.entry_get,modelgrowthrate.entry_get,modelbatchsize.entry_get,
                                                                                                     loggervar.get(),loggervalidBatchSize.entry_get,reducevar.get(),reducelrfactor.entry_get,checkpointscheckbox.getChoices(),earlyStopcheckbox.getChoices(),self.nnArchitecture.getChoices()))

        #set
        epochs.entry_set('200')
        dsFactor.entry_set('2')
        validationSplit.entry_set('0.05')
        sigma.entry_set('5')
        graphscale.entry_set('1')
        augmenter.setChoices('True')
        validationbatchSize.entry_set('2')
        modelgrowthrate.entry_set('32')
        modelbatchsize.entry_set('2')
        loggervalidBatchSize.entry_set('2')
        reducelrfactor.entry_set('0.2')
        checkpointscheckbox.setChoices('True')
        earlyStopcheckbox.setChoices('True')
        self.nnArchitecture.setChoices('StackedDenseNet')

        #predict new video
        predictvid_label = LabelFrame(tab4,pady=5,padx=5)
        modelPath = FileSelect(predictvid_label,'Model path', lblwidth='12')
        vidfolderpredict = FolderSelect(predictvid_label,'Video Folder',lblwidth='12')
        #button
        predictbutton = Button(predictvid_label,text='Predict',command=lambda:self.runprediction(modelPath.file_path,vidfolderpredict.folder_path))

        #Visualize video
        visualize_label = LabelFrame(tab5,pady=5,padx=5)
        #button
        visualizeButton = Button(visualize_label,text='Run',command=lambda:visualizeDPK(self.configini))

        #organize
        #import videos
        label_importvideo.grid(row=0)
        # multi video
        label_multivideoimport.grid(row=0,sticky=W)
        self.multivideofolderpath.grid(row=0,sticky=W)
        self.video_type.grid(row=1,sticky=W)
        button_multivideoimport.grid(row=2,sticky=W)
        # singlevideo
        label_singlevideoimport.grid(row=1,sticky=W)
        self.singlevideopath.grid(row=0,sticky=W)
        button_importsinglevideo.grid(row=1,sticky=W)

        ##create annotation
        createann_label.grid(row=0,sticky=W,pady=1)
        annOutputName.grid(row=1, sticky=W, pady=1)
        readBatchSize.grid(row=2, sticky=W, pady=1)
        kmeanbSize.grid(row=3, sticky=W, pady=1)
        kmeanNcluster.grid(row=4, sticky=W, pady=1)
        kmeanmaxIter.grid(row=5, sticky=W, pady=1)
        kmeanNinit.grid(row=6, sticky=W, pady=1)
        crateaAnnButton.grid(row=7,pady=5)

        ##annotator
        annfile_label.grid(row=1,sticky=W,pady=1)
        annfile.grid(row=0,sticky=W,pady=1)
        runann_button.grid(row=2,pady=5)
        ##trainmodel
        trainmodel_label.grid(row=2,sticky=W,pady=1)
        annpath.grid(row=0,sticky=W,pady=1)
        savemodelPath.grid(row=1,sticky=W,pady=1)
        epochs.grid(row=2, sticky=W, pady=1)
        dsFactor.grid(row=3, sticky=W, pady=1)
        validationSplit.grid(row=4, sticky=W, pady=1)
        sigma.grid(row=5, sticky=W, pady=1)
        graphscale.grid(row=6, sticky=W, pady=1)
        augmenter.grid(row=7, sticky=W, pady=1)
        validationbatchSize.grid(row=8, sticky=W, pady=1)
        modelgrowthrate.grid(row=9, sticky=W, pady=1)
        modelbatchsize.grid(row=10, sticky=W, pady=1)
        loggercheckbox.grid(row=11, sticky=W, pady=1)
        loggervalidBatchSize.grid(row=12, sticky=W, pady=1)
        reducelrcheckbox.grid(row=13, sticky=W, pady=1)
        reducelrfactor.grid(row=14, sticky=W, pady=1)
        checkpointscheckbox.grid(row=15, sticky=W, pady=1)
        earlyStopcheckbox.grid(row=16, sticky=W, pady=1)
        self.nnArchitecture.grid(row=17, sticky=W, pady=1)
        modelsettingsbutton.grid(row=18, sticky=W, pady=5)
        trainmodelbutton.grid(row=19, pady=5)


        ##predictnewvideo
        predictvid_label.grid(row=3,sticky=W,pady=1)
        modelPath.grid(row=0,sticky=W,pady=1)
        vidfolderpredict.grid(row=1,sticky=W,pady=1)
        predictbutton.grid(row=4,pady=5)
        ##visualize
        visualize_label.grid(row=4,sticky=W,pady=1)
        visualizeButton.grid(row=2,pady=5)

    def importvideo_single(self):
        if (self.configini != 'No file selected') and (self.singlevideopath.file_path != 'No file selected'):
            copy_singlevideo_DPKini(self.configini, self.singlevideopath.file_path)
        else:
            print('Fail to import video, please select a video to import')

    def importvideo_multi(self):
        if (self.configini != 'No file selected') and (self.multivideofolderpath.folder_path != 'No folder selected') and (self.video_type.entry_get != ''):
            copy_multivideo_DPKini(self.configini, self.multivideofolderpath.folder_path, self.video_type.entry_get)
        else:
            print('Fail to import videos, please select folder with videos and enter the file format')

    def runprediction(self,modelpath,videofolder):
        configini = self.configini
        config = ConfigParser()
        config.read(configini)
        config.set('predict settings', 'modelPath',str(modelpath))
        # write
        with open(configini, 'w') as configfile:
            config.write(configfile)

        predictnewvideoDPK(configini,videofolder)

    def traindpk(self,annotationfilepath,outputmodelName,epochs,dsfactor,validationsplit,sigma,graphscale,augmenter,validationbatchsize,modelgrowthrate,modelbatchsize,logger,loggerentry,reduce,reducefactor,checkpoint,earlystop,architure):
        #get the config
        configini = self.configini
        config = ConfigParser()
        config.read(configini)
        ## main training settings
        config.set('train model settings', 'epochs', str(epochs))
        config.set('train model settings', 'downsampleFactor', str(dsfactor))
        config.set('train model settings', 'validation_split', str(validationsplit))
        config.set('train model settings', 'sigma', str(sigma))
        config.set('train model settings', 'graph_scale', str(graphscale))
        config.set('train model settings', 'augmenterCheck', str(augmenter))
        config.set('train model settings', 'validation_batch_size', str(validationbatchsize))
        config.set('train model settings', 'modelGrowthRate', str(modelgrowthrate))
        config.set('train model settings', 'model_batch_size', str(modelbatchsize))
        config.set('train model settings', 'loggerCheck', str(logger))
        config.set('train model settings', 'logger_validation_batch_size', str(loggerentry))
        config.set('train model settings', 'reducelrCheck', str(reduce))
        config.set('train model settings', 'reduce_lr_factor', str(reducefactor))
        config.set('train model settings', 'earlyStopCheck', str(earlystop))
        config.set('train model settings', 'modelcheckPointCheck', str(checkpoint))
        config.set('train model settings', 'NN_architecture', str(architure))

        #write
        with open(configini, 'w') as configfile:
            config.write(configfile)

        trainDPKmodel(configini,outputmodelName,annotationfilepath)


    def dpkStep1(self,annOutputName,readBatchSize,kmeansize,kmeancluster,kmeanmaxiter,kmeanNinit):
        print('Creating annotation set.')
        configini = self.configini

        config = ConfigParser()
        config.read(configini)

        config.set('create annotation settings', 'annotation_output_name', str(annOutputName))
        config.set('create annotation settings', 'read_batch_size', str(readBatchSize))
        config.set('create annotation settings', 'k_means_batch_size', str(kmeansize))
        config.set('create annotation settings', 'k_means_n_custers', str(kmeancluster))
        config.set('create annotation settings', 'k_means_max_iterations', str(kmeanmaxiter))
        config.set('create annotation settings', 'k_means_n_init', str(kmeanNinit))
        with open(configini, 'w') as configfile:
            config.write(configfile)

        createAnnotationSet(configini)
        print('Annotation set created')

    def modelsettingschoice(self):
        print(self.nnArchitecture.getChoices())
        if (self.nnArchitecture.getChoices() == 'StackedDenseNet') or (self.nnArchitecture.getChoices() == 'StackedHourglass'):
            self.stackMenu()

        elif (self.nnArchitecture.getChoices() == 'DeepLabCut'):
            self.deeplabcutMenu()

        elif(self.nnArchitecture.getChoices() =='LEAP'):
            self.leapmenu()

    def stackMenu(self):
        stacktoplevel = Toplevel()
        stacktoplevel.minsize(300, 200)
        stacktoplevel.wm_title("StackedDenseNet / StackHourglass")

        stacklabel = LabelFrame(stacktoplevel,pady=5,padx=5)
        n_stacks = Entry_Box(stacklabel,'N_stacks','15')
        n_transitions = Entry_Box(stacklabel,'N_transitions','15')
        growthrate = Entry_Box(stacklabel,'Growth rate','15')
        bottleneckFactor = Entry_Box(stacklabel,'Bottleneck factor','15')
        compressionfactor = Entry_Box(stacklabel,'Compression factor','15')
        TorFalseOptions = {'True','False'}
        pretrained = DropDownMenu(stacklabel,'Pretrained',TorFalseOptions,'15')
        subpixel = DropDownMenu(stacklabel, 'Subpixel', TorFalseOptions, '15')

        #set initial dropdown
        n_stacks.entry_set('1')
        n_transitions.entry_set('1')
        growthrate.entry_set('48')
        bottleneckFactor.entry_set('1')
        compressionfactor.entry_set('0.5')
        pretrained.setChoices('False')
        subpixel.setChoices('True')

        # button
        stackmenubutton = Button(stacklabel,text='Save settings',command= lambda:self.savestacksettings(n_stacks.entry_get, n_transitions.entry_get,growthrate.entry_get,bottleneckFactor.entry_get,compressionfactor.entry_get,pretrained.getChoices(),subpixel.getChoices()))

        #organize
        stacklabel.grid(row=0,sticky=W)
        n_stacks.grid(row=0,sticky=W,pady=1)
        n_transitions.grid(row=1,sticky=W,pady=1)
        growthrate.grid(row=2,sticky=W,pady=1)
        bottleneckFactor.grid(row=3,sticky=W,pady=1)
        compressionfactor.grid(row=4,sticky=W,pady=1)
        pretrained.grid(row=5,sticky=W,pady=1)
        subpixel.grid(row=6,sticky=W,pady=1)
        stackmenubutton.grid(row=7,pady=5)
    def savestacksettings(self,nstack,ntransitions,growthrate,bneckfactor,compressfactor,pretrain,subpixel):
        configini = self.configini
        config = ConfigParser()
        config.read(configini)

        config.set('StackedDenseNet/StackedHourglass settings', 'n_stacks', str(nstack))
        config.set('StackedDenseNet/StackedHourglass settings', 'n_transitions', str(ntransitions))
        config.set('StackedDenseNet/StackedHourglass settings', 'growth_rate', str(growthrate))
        config.set('StackedDenseNet/StackedHourglass settings', 'bottleneckfactor', str(bneckfactor))
        config.set('StackedDenseNet/StackedHourglass settings', 'compression_factor', str(compressfactor))
        config.set('StackedDenseNet/StackedHourglass settings', 'pretrained', str(pretrain))
        config.set('StackedDenseNet/StackedHourglass settings', 'subpixel', str(subpixel))

        with open(configini, 'w') as configfile:
            config.write(configfile)

        print('Settings saved.')

    def deeplabcutMenu(self):
        dlctoplevel = Toplevel()
        dlctoplevel.minsize(300, 200)
        dlctoplevel.wm_title("DeepLabCut")

        dlclabel = LabelFrame(dlctoplevel,pady=5,padx=5)
        weightsoptions = {'imagenet','none'}
        backboneoptions = {'resnet50','resnet101','resnet152','mobilenetv2','densenet121','densenet169','densenet201','xception'}
        torFalseoptions = {'True','False'}
        ##dropdowns
        weight = DropDownMenu(dlclabel,'Weights',weightsoptions,'10')
        backbone = DropDownMenu(dlclabel,'Backbone',backboneoptions,'10')
        alpha = Entry_Box(dlclabel,'Alpha','10')
        subpixel = DropDownMenu(dlclabel,'Subpixel',torFalseoptions,'10')

        ##sets
        #alpha.set(1)
        weight.setChoices('imagenet')
        backbone.setChoices('resnet50')
        subpixel.setChoices('True')

        #button
        dlcsettingsbutton = Button(dlclabel,text='Save Settings',command=lambda:self.savedlcsettings(weight.getChoices(),backbone.getChoices(),alpha.entry_get,subpixel.getChoices()))

        #organize
        dlclabel.grid(row=0,sticky=W)
        weight.grid(row=0,sticky=W,pady=1)
        backbone.grid(row=1, sticky=W, pady=1)
        alpha.grid(row=2, sticky=W, pady=1)
        subpixel.grid(row=3, sticky=W, pady=1)
        dlcsettingsbutton.grid(row=4,pady=5)
    def savedlcsettings(self,weight,backbone,alpha,subpixel):
        configini = self.configini
        config = ConfigParser()
        config.read(configini)

        config.set('DeepLabCut settings', 'weights', str(weight))
        config.set('DeepLabCut settings', 'backbone', str(backbone))
        config.set('DeepLabCut settings', 'alpha', str(alpha))
        config.set('DeepLabCut settings', 'subpixel', str(subpixel))

        with open(configini, 'w') as configfile:
            config.write(configfile)

        print('Settings saved.')

    def leapmenu(self):
        leaptoplevel = Toplevel()
        leaptoplevel.minsize(300, 200)
        leaptoplevel.wm_title("LEAP")

        leaplabel = LabelFrame(leaptoplevel,pady=5,padx=5)
        filters = Entry_Box(leaplabel,'Filters','10')
        torFalseoption = {'True','False'}
        upsamplinglayers = DropDownMenu(leaplabel,'Upsampling layers',torFalseoption,'15')
        batchnorm = DropDownMenu(leaplabel,'Batchnorm',torFalseoption,'15')
        poolingoption = {'max','average'}
        pooling = DropDownMenu(leaplabel,'Pooling',poolingoption,'15')
        interpolationoption = {'nearest','bilinear','bicubic'}
        interpolation = DropDownMenu(leaplabel,'Interpolation',interpolationoption,'15')
        initializeroption = {'glorot_uniform','lecun_normal'}
        initializer = DropDownMenu(leaplabel,'Initializer',initializeroption,'15')
        subpixel = DropDownMenu(leaplabel,'Subpixel',torFalseoption,'15')

        #sets
        filters.set('64')
        upsamplinglayers.setChoices('False')
        batchnorm.setChoices('False')
        pooling.setChoices('max')
        interpolation.setChoices('nearest')
        initializer.setChoices('glorot_uniform')
        subpixel.setChoices('True')
        #button
        leapbutton = Button(leaplabel,text='Save settings',command=lambda:self.saveleapsettings(filters.entry_get,upsamplinglayers.getChoices(),batchnorm.getChoices(),pooling.getChoices(),interpolation.getChoices(),initializer.getChoices(),subpixel.getChoices()))


        #organize
        leaplabel.grid(row=0,sticky=W)
        filters.grid(row=0,sticky=W)
        upsamplinglayers.grid(row=1,sticky=W)
        batchnorm.grid(row=2,sticky=W)
        pooling.grid(row=3,sticky=W)
        interpolation.grid(row=4,sticky=W)
        initializer.grid(row=5,sticky=W)
        subpixel.grid(row=6,sticky=W)
        leapbutton.grid(row=7,pady=5)
    def saveleapsettings(self,filters,upsampling,batchnorm,pooling,interpolation,initializer,subpixel):
        configini = self.configini
        config = ConfigParser()
        config.read(configini)

        config.set('LEAP settings', 'filters', str(filters))
        config.set('LEAP settings', 'upsampling_layers', str(upsampling))
        config.set('LEAP settings', 'batchnorm', str(batchnorm))
        config.set('LEAP settings', 'pooling', str(pooling))
        config.set('LEAP settings', 'interpolation', str(interpolation))
        config.set('LEAP settings', 'subpixel', str(subpixel))
        config.set('LEAP settings', 'initializer', str(initializer))

        with open(configini, 'w') as configfile:
            config.write(configfile)

        print('Settings saved.')

class Load_DLC_Model:

    def __init__(self):
        # Popup window
        loadmodel = Toplevel()
        loadmodel.minsize(200, 200)
        loadmodel.wm_title("Load DLC Model")


        tab_parent = ttk.Notebook(loadmodel)

        tab1 = ttk.Frame(tab_parent)
        tab2 = ttk.Frame(tab_parent)
        tab3 = ttk.Frame(tab_parent)
        tab4 = ttk.Frame(tab_parent)
        tab7 = ttk.Frame(tab_parent)
        tab8 = ttk.Frame(tab_parent)
        tab9 = ttk.Frame(tab_parent)
        tab10 = ttk.Frame(tab_parent)
        tab11 = ttk.Frame(tab_parent)
        tab12 = ttk.Frame(tab_parent)
        tab13 = ttk.Frame(tab_parent)
        tab14 = ttk.Frame(tab_parent)
        tab15 = ttk.Frame(tab_parent)

        tab_parent.add(tab1, text=f'{"[ Load model ]": ^20s}')
        tab_parent.add(tab2,text=f'{"[ Generate temp yaml ]": ^20s}')
        tab_parent.add(tab3, text=f'{"[ Add videos into project ]": ^20s}')
        tab_parent.add(tab4, text=f'{"[ Extract / label frames ]": ^20s}')
        tab_parent.add(tab7, text=f'{"[ Generate training set ]": ^20s}')
        tab_parent.add(tab10, text=f'{"[ Video analysis ]": ^20s}')
        tab_parent.add(tab13, text=f'{"[ Extract outliers ]": ^20s}')
        tab_parent.grid(row=0)

        #Load Model : configpath
        labelframe_loadmodel = LabelFrame(tab1, text='Load Model', font=("Helvetica",12,'bold'),padx=5,pady=5)
        self.label_set_configpath = FileSelect(labelframe_loadmodel, 'DLC config path (.yaml): ',title='Select a .yaml file')

        # generate yaml file
        label_generatetempyaml = LabelFrame(tab2,text='Generate Temp yaml (for extracting frames from subset of videos)', font=("Helvetica",12,'bold') ,padx=5,pady=5)
        label_tempyamlsingle = LabelFrame(label_generatetempyaml,text='Single video',padx=5,pady=5)
        self.label_genyamlsinglevideo = FileSelect(label_tempyamlsingle,'Select video:',title='Select a video file')
        button_generatetempyaml_single = Button(label_tempyamlsingle,text='Add single video',command=lambda:generatetempyaml(self.label_set_configpath.file_path,self.label_genyamlsinglevideo.file_path))
        label_tempyamlmulti =LabelFrame(label_generatetempyaml,text='Multiple videos',padx=5,pady=5)
        self.label_genyamlmultivideo = FolderSelect(label_tempyamlmulti,'Select video folder:',title='Select video folder')
        button_generatetempyaml_multi = Button(label_tempyamlmulti,text='Add multiple videos',command=self.generateyamlmulti)
        label_tempyml = Label(label_generatetempyaml,text='Note: After creating the temp yaml with the selected videos, load the temp.yaml file in "Load Model".',font=('Times',10,'italic'))
        label_tempyml2 = Label(label_generatetempyaml,text='       Then, you can proceed to extract frames.',font=('Times',10,'italic'))

        #singlevid multivid
        labelframe_singlemultivid = LabelFrame(tab3,text='Add Videos into project',font=("Helvetica",12,'bold'),padx=5,pady=5)
        labelframe_singlevid = LabelFrame(labelframe_singlemultivid,text='Single Video',padx=5,pady=5)
        labelframe_multivid = LabelFrame(labelframe_singlemultivid,text='Multiple Videos',padx=5,pady=5)
        self.label_set_singlevid = FileSelect(labelframe_singlevid, 'Select Single Video: ',title='Select a video file')
        self.label_video_folder = FolderSelect(labelframe_multivid, 'Select Folder with videos:',title='Select video folder')
        button_add_single_video = Button(labelframe_singlevid,text='Add single video',command = self.dlc_addsinglevideo,fg='red')
        button_add_multi_video = Button(labelframe_multivid,text='Add multiple videos',command = self.dlc_addmultivideo_command,fg='red')

        ###########extract frames########
        label_extractframes = LabelFrame(tab4, text='Extract Frames DLC', font=("Helvetica",12,'bold'),padx=15,pady=5)
        # mode
        self.label_numframes2pick = Entry_Box(label_extractframes,'numframes2pick:','26')
        label_mode = Label(label_extractframes, text='Mode', font="Verdana 10 underline")
        self.mode = IntVar()
        checkbox_auto = Radiobutton(label_extractframes, text="Automatic", variable=self.mode, value=1)
        checkbox_manual = Radiobutton(label_extractframes, text="Manual", variable=self.mode, value=2)
        # algorithm
        label_algo = Label(label_extractframes, text='Algorithm ', font="Verdana 10 underline")
        self.algo = IntVar()
        checkbox_uniform = Radiobutton(label_extractframes, text="Uniform", variable=self.algo, value=1)
        checkbox_kmean = Radiobutton(label_extractframes, text="KMeans", variable=self.algo, value=2)
        # cluster resize width
        self.label_clusterresize = Entry_Box(label_extractframes, 'Cluster Resize Width (Default = 30)', '26')
        # cluster step
        self.label_clusterstep = Entry_Box(label_extractframes, 'Cluster Step (Default = 1)', '26')
        # cluster color
        label_clustercolor = Label(label_extractframes, text='Cluster color', font="Verdana 10 underline")
        # checkbox cluster color
        self.var_clustercolor = IntVar()
        checkbox_clustercolor = Checkbutton(label_extractframes, text='True', variable=self.var_clustercolor)
        # use opencv
        label_useopencv = Label(label_extractframes, text='Use OpenCV', font="Verdana 10 underline")
        # checkbox use opencv or not
        self.var_useopencv = IntVar()
        checkbox_useopencv = Checkbutton(label_extractframes, text='True', variable=self.var_useopencv)
        # extractframecommand
        button_extractframe = Button(label_extractframes, text='Extract Frames', command=self.dlc_extractframes_command)

        ##########label Frames#####
        label_labelframes = LabelFrame(tab4, text='Label Frames', font=("Helvetica",12,'bold'),padx=15,pady=5)
        self.button_label_frames = Button(label_labelframes, text='Label Frames', command=self.dlc_label_frames_command)

        ##########Check Labels#####
        label_checklabels = LabelFrame(tab4, text='Check Labels', font=("Helvetica",12,'bold'),padx=15,pady=5)
        self.button_check_labels = Button(label_checklabels, text='Check Labelled Frames', command=self.dlc_check_labels_command)

        ####generate training sets#####
        label_generate_trainingsets = LabelFrame(tab7,text='Generate Training Set',font =("Helvetica",12,'bold'),padx=15,pady=5)
        self.button_generate_trainingsets = Button(label_generate_trainingsets, text='Generate training set',command=self.dlc_generate_trainingsets_command)

        #####train network####
        label_train_network = LabelFrame(tab7,text= 'Train Network',font =("Helvetica",12,'bold'),padx=15,pady=5)
        self.label_iteration = Entry_Box(label_train_network,'iteration','10')
        self.button_update_iteration = Button(label_train_network,text='Update iteration',command =lambda:updateiteration(self.label_set_configpath.file_path,self.label_iteration.entry_get))
        self.init_weight = FileSelect(label_train_network,'init_weight     ',title='Select training weight, eg: .DATA-00000-OF-00001 File')
        self.update_init_weight = Button(label_train_network,text='Update init_weight',command=lambda:update_init_weight(self.label_set_configpath.file_path,self.init_weight.file_path))
        self.button_train_network = Button(label_train_network, text='Train Network',command=self.dlc_train_network_command)

        #######evaluate network####
        label_eva_network = LabelFrame(tab7,text='Evaluate Network',font = ("Helvetica",12,'bold'),padx=15,pady=5)
        self.button_evaluate_network = Button(label_eva_network, text='Evaluate Network',command=self.dlc_evaluate_network_command)

        #####video analysis####
        label_video_analysis = LabelFrame(tab10,text='Video Analysis',font=("Helvetica",12,'bold'),padx=15,pady=5)
        #singlevideoanalysis
        label_singlevideoanalysis = LabelFrame(label_video_analysis,text='Single Video Analysis',pady=5,padx=5)
        self.videoanalysispath = FileSelect(label_singlevideoanalysis, "Video path",title='Select a video file')
        button_vidanalysis = Button(label_singlevideoanalysis, text='Single Video Analysis', command=self.dlc_video_analysis_command1)
        #multi video analysis
        label_multivideoanalysis = LabelFrame(label_video_analysis,text='Multiple Videos Analysis',pady=5,padx=5)
        self.videofolderpath = FolderSelect(label_multivideoanalysis,'Folder Path',title='Select video folder')
        self.video_type = Entry_Box(label_multivideoanalysis,'Video type(eg:mp4,avi):','18')
        button_multivideoanalysis = Button(label_multivideoanalysis,text='Multi Videos Analysis',command=self.dlc_video_analysis_command2)

        #### plot####
        label_plot = LabelFrame(tab10,text='Plot Video Graph',font=("Helvetica",12,'bold'),padx=15,pady=5)
        # videopath
        self.videoplotpath = FileSelect(label_plot, "Video path",title='Select a video file')
        # plot button
        button_plot = Button(label_plot, text='Plot Results', command=self.dlc_plot_videoresults_command)

        #####create video####
        label_createvideo = LabelFrame(tab10,text='Create Video',font=("Helvetica",12,'bold'),padx=15,pady=5)
        # videopath
        self.createvidpath = FileSelect(label_createvideo, "Video path",title='Select a video file')
        # save frames
        self.var_saveframes = IntVar()
        checkbox_saveframes = Checkbutton(label_createvideo, text='Save Frames', variable=self.var_saveframes)
        # create video button
        button_createvideo = Button(label_createvideo, text='Create Video', command=self.dlc_create_video_command)

        ######Extract Outliers####
        label_extractoutlier = LabelFrame(tab13,text='Extract Outliers',font=("Helvetica",12,'bold'),pady=5,padx=5)
        self.label_extractoutliersvideo = FileSelect(label_extractoutlier,'Videos to correct:',title='Select a video file')
        button_extractoutliers = Button(label_extractoutlier,text='Extract Outliers',command =lambda:deeplabcut.extract_outlier_frames(self.label_set_configpath.file_path, [str(self.label_extractoutliersvideo.file_path)],automatic=True) )

        ####label outliers###
        label_labeloutliers = LabelFrame(tab13,text='Label Outliers',font =("Helvetica",12,'bold'),pady=5,padx=5)
        button_refinelabels = Button(label_labeloutliers,text='Refine Outliers',command=lambda:deeplabcut.refine_labels(self.label_set_configpath.file_path))

        ####merge labeled outliers ###
        label_mergeoutliers = LabelFrame(tab13,text='Merge Labelled Outliers',font=("Helvetica",12,'bold'),pady=5,padx=5)
        button_mergelabeledoutlier = Button(label_mergeoutliers,text='Merge Labelled Outliers',command=lambda:deeplabcut.merge_datasets(self.label_set_configpath.file_path))

        #organize
        labelframe_loadmodel.grid(row=0,sticky=W,pady=5)
        self.label_set_configpath.grid(row=0,sticky=W)

        label_generatetempyaml.grid(row=1,sticky=W)
        label_tempyamlsingle.grid(row=0,sticky=W)
        self.label_genyamlsinglevideo.grid(row=0,sticky=W)
        button_generatetempyaml_single.grid(row=1,sticky=W)
        label_tempyamlmulti.grid(row=1,sticky=W)
        self.label_genyamlmultivideo.grid(row=0,sticky=W)
        button_generatetempyaml_multi.grid(row=1,sticky=W)
        label_tempyml.grid(row=2,sticky=W)
        label_tempyml2.grid(row=3, sticky=W)

        labelframe_singlemultivid.grid(row=2,sticky=W,pady=5)
        labelframe_singlevid.grid(row=0,sticky=W)
        self.label_set_singlevid.grid(row=0,sticky=W)
        button_add_single_video.grid(row=1,sticky=W)
        labelframe_multivid.grid(row=1,sticky=W)
        self.label_video_folder.grid(row=0,sticky=W)
        button_add_multi_video.grid(row=1,sticky=W)

        label_extractframes.grid(row=3,column=0, sticky=W,pady=5,padx=5)
        self.label_numframes2pick.grid(row=0,sticky=W)
        label_mode.grid(row=1, sticky=W)
        checkbox_auto.grid(row=2, sticky=W)
        checkbox_manual.grid(row=3, sticky=W)
        label_algo.grid(row=4, sticky=W)
        checkbox_uniform.grid(row=5, sticky=W)
        checkbox_kmean.grid(row=6, sticky=W)
        self.label_clusterresize.grid(row=7, sticky=W)
        self.label_clusterstep.grid(row=8, sticky=W)
        label_clustercolor.grid(row=9, sticky=W)
        checkbox_clustercolor.grid(row=10, sticky=W)
        label_useopencv.grid(row=11, sticky=W)
        checkbox_useopencv.grid(row=12, sticky=W)
        button_extractframe.grid(row=13,sticky=W)

        label_labelframes.grid(row=3,column=1,sticky=W+N,pady=5,padx=5)
        self.button_label_frames.grid(row=0,sticky=W)

        label_checklabels.grid(row=3,column=2,sticky=W+N,pady=5,padx=5)
        self.button_check_labels.grid(row=0,sticky=W)

        label_generate_trainingsets.grid(row=6,sticky=W,pady=5)
        self.button_generate_trainingsets.grid(row=0,sticky=W)

        label_train_network.grid(row=7,sticky=W,pady=5)
        self.label_iteration.grid(row=0,column=0,sticky=W)
        self.button_update_iteration.grid(row=0,column=1,sticky=W)
        self.init_weight.grid(row=1,column=0,sticky=W)
        self.update_init_weight.grid(row=1,column=1,sticky=W)
        self.button_train_network.grid(row=2,sticky=W)

        label_eva_network.grid(row=8,sticky=W,pady=5)
        self.button_evaluate_network.grid(row=0,sticky=W)
        #video analysis
        label_video_analysis.grid(row=9,sticky=W,pady=5)
        label_singlevideoanalysis.grid(row=0,sticky=W,pady=5)
        label_multivideoanalysis.grid(row=1,sticky=W,pady=5)
        self.videoanalysispath.grid(row=0,sticky=W)
        button_vidanalysis.grid(row=2,sticky=W)
        self.videofolderpath.grid(row=3,sticky=W,pady=5)
        self.video_type.grid(row=4,sticky=W)
        button_multivideoanalysis.grid(row=5,sticky=W)

        label_plot.grid(row=10,sticky=W,pady=5)
        self.videoplotpath.grid(row=0,sticky=W)
        button_plot.grid(row=1,sticky=W)

        label_createvideo.grid(row=11,sticky=W,pady=5)
        self.createvidpath.grid(row=0,sticky=W)
        checkbox_saveframes.grid(row=1,sticky=W)
        button_createvideo.grid(row=2,sticky=W)

        label_extractoutlier.grid(row=12,sticky=W,pady=5)
        self.label_extractoutliersvideo.grid(row=0,sticky=W)
        button_extractoutliers.grid(row=1,sticky=W)

        label_labeloutliers.grid(row=13,sticky=W,pady=5)
        button_refinelabels.grid(row=0,sticky=W)

        label_mergeoutliers.grid(row=14,sticky=W,pady=5)
        button_mergelabeledoutlier.grid(row=0,sticky=W)


    def dlc_addsinglevideo(self):
        try:
            deeplabcut.add_new_videos(self.label_set_configpath.file_path, [str(self.label_set_singlevid.file_path)],copy_videos=True)
        except FileNotFoundError:
            print('...')
            print('Fail to add video, please load .yaml file and select video file')

    def generateyamlmulti(self):
        try:

            config_path = self.label_set_configpath.file_path
            directory = self.label_genyamlmultivideo.folder_path
            filesFound = []

            ########### FIND FILES ###########
            for i in os.listdir(directory):
                if '.avi' or '.mp4' in i:
                    a = os.path.join(directory, i)
                    filesFound.append(a)
                    print(a)
            print(filesFound)

            generatetempyaml_multi(config_path,filesFound)
        except FileNotFoundError:
            print('Fail to add videos, please load .yaml file and select video folder')

    def dlc_addmultivideo_command(self):
        try:
            config_path = self.label_set_configpath.file_path
            directory = self.label_video_folder.folder_path
            filesFound = []

            ########### FIND FILES ###########
            for i in os.listdir(directory):
                if 'avi' or '.mp4' in i:
                    a = os.path.join(directory, i)
                    deeplabcut.add_new_videos(config_path, [str(a)], copy_videos=True)

            print("Videos added.")
        except FileNotFoundError:
            print('Fail to add videos, please load .yaml file and select video folder')

    def dlc_extractframes_command(self):

        config_path = self.label_set_configpath.file_path
        select_numfram2pick(config_path,self.label_numframes2pick.entry_get)

        if self.mode.get()==1:
            modes = str('automatic')
        elif self.mode.get()==2:
            modes = str('manual')

        if self.algo.get()==1:
            algorithm = str('uniform')
        elif self.algo.get()==2:
            algorithm = str('kmeans')

        if len(self.label_clusterresize.entry_get)==0:
            clusterresizewidth = int(30)
        else:
            clusterresizewidth = int(self.label_clusterresize.entry_get)

        if len(self.label_clusterstep.entry_get)==0:
            clusterstep = int(1)
        else:
            clusterstep = int(self.label_clusterstep.entry_get)

        if self.var_clustercolor.get()==1:
            clustercolor = True
        else:
            clustercolor = False

        if self.var_useopencv.get()==1:
            useopencv = True
        else:
            useopencv = False
        try:
            print(config_path,modes,algorithm,clusterstep,clusterresizewidth,clustercolor,useopencv)
            deeplabcut.extract_frames(config_path,mode=modes,algo=algorithm,crop=False,userfeedback=False,cluster_step=clusterstep,cluster_resizewidth=clusterresizewidth,cluster_color=clustercolor,opencv=useopencv)
        except:
            print('Fail to extract frames, please make sure all the information is filled in')

    def dlc_label_frames_command(self):
        config_path = self.label_set_configpath.file_path
        deeplabcut.label_frames(config_path)

    def dlc_check_labels_command(self):
        try:
            config_path = self.label_set_configpath.file_path
            deeplabcut.check_labels(config_path)
        except FileNotFoundError:
            print('Please load .yaml file to continue')

    def dlc_generate_trainingsets_command(self):
        try:
            config_path = self.label_set_configpath.file_path
            deeplabcut.create_training_dataset(config_path, num_shuffles=1)
        except FileNotFoundError:
            print('Please load .yaml file to continue')

    def dlc_train_network_command(self):
        try:
            config_path = self.label_set_configpath.file_path
            deeplabcut.train_network(config_path, shuffle=1, gputouse=0)
        except FileNotFoundError:
            print('Please load .yaml file to continue')

    def dlc_evaluate_network_command(self):
        try:
            config_path = self.label_set_configpath.file_path
            deeplabcut.evaluate_network(config_path, plotting=True)
        except FileNotFoundError:
            print('Please load .yaml file to continue')

    def dlc_video_analysis_command1(self):
        try:
            config_path = self.label_set_configpath.file_path

            vid_name = os.path.basename(self.videoanalysispath.file_path)
            vid_type = vid_name[-4:]

            deeplabcut.analyze_videos(config_path, [str(self.videoanalysispath.file_path)], shuffle=1,save_as_csv=True, videotype=vid_type)
        except FileNotFoundError:
            print('Please load .yaml file and select video path to continue')

    def dlc_video_analysis_command2(self):
        try:
            config_path = self.label_set_configpath.file_path

            folder_path = self.videofolderpath.folder_path
            vid_type = self.video_type.entry_get

            deeplabcut.analyze_videos(config_path, [str(folder_path)], shuffle=1,save_as_csv=True, videotype=vid_type)
        except FileNotFoundError:
            print('Please load .yaml file and select folder with videos to continue')


    def dlc_plot_videoresults_command(self):
        try:

            config_path = self.label_set_configpath.file_path
            deeplabcut.plot_trajectories(config_path, [str(self.videoplotpath.file_path)])
        except FileNotFoundError:
            print('Please load .yaml file and select a video file to plot graph')

    def dlc_create_video_command(self):
        try:
            config_path = self.label_set_configpath.file_path

            if self.var_saveframes==1:
                saveframes=True
            else:
                saveframes=False
            vid_name = os.path.basename(self.createvidpath.file_path)
            vid_type = vid_name[-4:]

            deeplabcut.create_labeled_video(config_path, [str(self.createvidpath.file_path)],save_frames=saveframes, videotype=vid_type)
        except FileNotFoundError:
            print('Please select .yaml file and select a video to continue.')

class shorten_video:

    def __init__(self):
        # Popup window
        shortenvid = Toplevel()
        shortenvid.minsize(200, 200)
        shortenvid.wm_title("Clip video")

        # videopath
        self.videopath1selected = FileSelect(shortenvid, "Video path",title='Select a video file')

        #timeframe for start and end cut
        label_cutvideomethod1 = LabelFrame(shortenvid,text='Method 1',font='bold',padx=5,pady=5)
        label_timeframe = Label(label_cutvideomethod1, text='Please enter the time frame in hh:mm:ss format')
        self.label_starttime = Entry_Box(label_cutvideomethod1,'Start at:','8')
        self.label_endtime = Entry_Box(label_cutvideomethod1, 'End at:',8)
        CreateToolTip(label_cutvideomethod1,
                      'Method 1 will retrieve the specified time input.(eg: input of Start at: 00:00:00, End at: 00:01:00, will create a new video from the chosen video from the very start till it reaches the first minute of the video)')

        #express time frame
        label_cutvideomethod2 = LabelFrame(shortenvid,text='Method 2',font='bold',padx=5,pady=5)
        label_method2 = Label(label_cutvideomethod2,text='Method 2 will retrieve from the end of the video (e.g.,: an input of 3 seconds will get rid of the first 3 seconds of the video).')
        # self.var_express = IntVar()
        # checkbox_express = Checkbutton(label_cutvideomethod2, text='Check this box to use Method 2', variable=self.var_express)
        self.label_time = Entry_Box(label_cutvideomethod2,'Seconds:','8')
        CreateToolTip(label_cutvideomethod2,'Method 2 will retrieve from the end of the video.(eg: an input of 3 seconds will get rid of the first 3 seconds of the video)')

        #button to cut video
        button_cutvideo1 = Button(label_cutvideomethod1, text='Cut Video', command=lambda:shortenvideos1(self.videopath1selected.file_path,self.label_starttime.entry_get,self.label_endtime.entry_get))
        button_cutvideo2 = Button(label_cutvideomethod2,text='Cut Video',command =lambda:shortenvideos2(self.videopath1selected.file_path,self.label_time.entry_get))
        #organize
        self.videopath1selected.grid(row=0,sticky=W)

        label_cutvideomethod1.grid(row=1,sticky=W,pady=5)
        label_timeframe.grid(row=0,sticky=W)
        self.label_starttime.grid(row=1,sticky=W)
        self.label_endtime.grid(row=2,sticky=W)
        button_cutvideo1.grid(row=3)


        label_cutvideomethod2.grid(row=2,sticky=W,pady=5)
        label_method2.grid(row=0,sticky=W)
        # checkbox_express.grid(row=1,sticky=W)
        self.label_time.grid(row=2,sticky=W)
        button_cutvideo2.grid(row=3)

class multi_shorten_video:
    def __init__(self):
        # Popup window
        self.multishort = Toplevel()
        self.multishort.minsize(200, 200)
        self.multishort.wm_title("Clip video into multiple videos")

        self.lblmultishort = LabelFrame(self.multishort, text='Split videos into different parts', font='bold', padx=5, pady=5)
        # videopath
        self.videopath1selected = FileSelect(self.lblmultishort, "Video path", title='Select a video file')
        self.noclips = Entry_Box(self.lblmultishort,'# of clips','8')
        confirmclip = Button(self.lblmultishort,text='Confirm',command=lambda:self.expand(self.noclips.entry_get))

        runbutton = Button(self.multishort,text='Clip video', command= lambda:splitvideos(self.videopath1selected.file_path,self.allentries),fg='navy',font=("Helvetica",12,'bold'))

        #organize
        self.lblmultishort.grid(row=0,sticky=W)
        self.videopath1selected.grid(row=1,sticky=W,columnspan=2)
        self.noclips.grid(row=2,sticky=W)
        confirmclip.grid(row=2,column=1,sticky=W)

        runbutton.grid(row=5)

    def expand(self,noclips):

        try:
            self.table.destroy()
        except:
            pass

        noclips = int(noclips)

        self.table = LabelFrame(self.multishort)

        lbl_clip = Label(self.table,text='Clip #')
        lbl_start = Label(self.table,text='Start Time')
        lbl_stop = Label(self.table,text='Stop Time')

        #organize table
        self.table.grid(row=2,sticky=W)
        lbl_clip.grid(row=0,column=0,sticky=W)
        lbl_start.grid(row=0,column=1)
        lbl_stop.grid(row=0,column=2)

        #list
        self.ent1 = [0] * noclips
        self.ent2 = [0] * noclips
        self.ent3 = [0] * noclips

        for i in range(noclips):
            self.ent1[i] = Label(self.table,text='Clip '+str(i+1))
            self.ent1[i].grid(row=i+2,sticky=W)
            self.ent2[i] = Entry(self.table)
            self.ent2[i].grid(row=i+2,column=1,sticky=W)
            self.ent3[i] = Entry(self.table)
            self.ent3[i].grid(row=i+2,column=2,sticky=W)

        self.allentries = [self.ent2,self.ent3]


class change_imageformat:

    def __init__(self):

        # Popup window
        chgimgformat = Toplevel()
        chgimgformat.minsize(200, 200)
        chgimgformat.wm_title("Change image format")

        #select directory
        self.folderpath1selected = FolderSelect(chgimgformat,"Image directory",title='Select folder with images')

        #change image format
        label_filetypein = LabelFrame(chgimgformat,text= 'Original image format',font=("Helvetica",12,'bold'),padx=15,pady=5)
        # Checkbox input
        self.varfiletypein = IntVar()
        checkbox_c1 = Radiobutton(label_filetypein, text=".png", variable=self.varfiletypein, value=1)
        checkbox_c2 = Radiobutton(label_filetypein, text=".jpeg", variable=self.varfiletypein, value=2)
        checkbox_c3 = Radiobutton(label_filetypein, text=".bmp", variable=self.varfiletypein, value=3)

        #ouput image format
        label_filetypeout = LabelFrame(chgimgformat,text='Output image format',font=("Helvetica",12,'bold'),padx=15,pady=5)
        #checkbox output
        self.varfiletypeout = IntVar()
        checkbox_co1 = Radiobutton(label_filetypeout, text=".png", variable=self.varfiletypeout, value=1)
        checkbox_co2 = Radiobutton(label_filetypeout, text=".jpeg", variable=self.varfiletypeout, value=2)
        checkbox_co3 = Radiobutton(label_filetypeout, text=".bmp", variable=self.varfiletypeout, value=3)

        #button
        button_changeimgformat = Button(chgimgformat, text='Convert image file format', command=self.changeimgformatcommand)

        #organized
        self.folderpath1selected.grid(row=0,column=0)
        label_filetypein.grid(row=1,column=0,pady=5)
        checkbox_c1.grid(row=2,column=0)
        checkbox_c2.grid(row=3,column=0)
        checkbox_c3.grid(row=4,column=0)
        label_filetypeout.grid(row=5,column=0,pady=5)
        checkbox_co1.grid(row=6,column=0)
        checkbox_co2.grid(row=7,column=0)
        checkbox_co3.grid(row=8, column=0)
        button_changeimgformat.grid(row=9,pady=5)

    def changeimgformatcommand(self):

        if self.varfiletypein.get()==1:
            filetypein = str('png')
        elif self.varfiletypein.get()==2:
            filetypein = str('jpeg')
        elif self.varfiletypein.get()==3:
            filetypein = str('bmp')

        if self.varfiletypeout.get()==1:
            filetypeout = str('png')
        elif self.varfiletypeout.get()==2:
            filetypeout = str('jpeg')
        elif self.varfiletypeout.get() == 3:
            filetypeout = str('bmp')

        cif=changeimageformat(self.folderpath1selected.folder_path, filetypein, filetypeout)

        print('Images converted to '+ str(cif) + ' format')

class convert_video:

    def __init__(self):
        # Popup window
        convertvid = Toplevel()
        convertvid.minsize(400, 400)
        convertvid.wm_title("Convert video format")

        #multi video
        label_multivideo = LabelFrame(convertvid, text='Convert multiple videos',font=("Helvetica",12,'bold'),padx=5,pady=5)
        vid_dir = FolderSelect(label_multivideo,'Video directory',title='Select folder with videos')
        ori_format = Entry_Box(label_multivideo,'Input format','12')
        final_format = Entry_Box(label_multivideo,'Output format','12')
        button_convertmultivid = Button(label_multivideo,text='Convert multiple videos',command = lambda: batch_convert_videoformat(vid_dir.folder_path,ori_format.entry_get,final_format.entry_get))

        #single video
        label_convert = LabelFrame(convertvid,text='Convert single video',font=("Helvetica",12,'bold'),padx=5,pady=5)
        self.videopath1selected = FileSelect(label_convert, "Video path",title='Select a video file')
        self.vvformat = IntVar()
        checkbox_v1 = Radiobutton(label_convert, text="Convert .avi to .mp4", variable=self.vvformat, value=1)
        checkbox_v2 = Radiobutton(label_convert, text="Convert mp4 into Powerpoint supported format", variable=self.vvformat, value=2)

        #button
        button_convertvid= Button(label_convert, text='Convert video format', command=self.convertavitomp)

        #organize
        label_multivideo.grid(row=0,sticky=W)
        vid_dir.grid(row=0,sticky=W)
        ori_format.grid(row=1,sticky=W)
        final_format.grid(row=2,sticky=W)
        button_convertmultivid.grid(row=3,pady=10)

        label_convert.grid(row=1,sticky=W)
        self.videopath1selected.grid(row=0,sticky=W)
        checkbox_v1.grid(row=1,column=0,sticky=W)
        checkbox_v2.grid(row=2,column=0,sticky=W)
        button_convertvid.grid(row=3,column=0,pady=10)

    def convertavitomp(self):

        if self.vvformat.get()== 1:
            cavi = convertavitomp4(self.videopath1selected.file_path)
            print('Video converted to ' + cavi)
        elif self.vvformat.get()== 2:
            cavi = convertpowerpoint(self.videopath1selected.file_path)
            print('Video converted to ' + cavi)

class extract_specificframes:

    def __init__(self):
        # Popup window
        extractsf = Toplevel()
        extractsf.minsize(200, 200)
        extractsf.wm_title("Extract defined Frames")

        # videopath
        self.videopath1selected = FileSelect(extractsf, "Video path",title='Select a video file')

        #entry boxes for frames to extract
        label_frametitle = LabelFrame(extractsf, text='Frames to be extracted',padx=5,pady=5)
        self.label_startframe1 = Entry_Box(label_frametitle,'Start Frame:','10')

        self.label_endframe1 = Entry_Box(label_frametitle, 'End Frame:','10')


        #button
        button_extractsf = Button(label_frametitle, text='Extract Frames', command=self.extractsfcommand)

        #organize
        self.videopath1selected.grid(row=0,column=0,sticky=W,pady=10)
        label_frametitle.grid(row=1,column=0,sticky=W)
        self.label_startframe1.grid(row=2,column=0,sticky=W)

        self.label_endframe1.grid(row=3,column=0,sticky=W)

        button_extractsf.grid(row=4,pady=5)

    def extractsfcommand(self):

        startframe1 = self.label_startframe1.entry_get
        endframe1 = self.label_endframe1.entry_get

        extractspecificframe(self.videopath1selected.file_path, startframe1, endframe1)
        print('Frames were extracted from '+ str(startframe1)+' to ' + endframe1)

def extract_allframes():

    # Popup window
    extractaf = Toplevel()
    extractaf.minsize(300, 300)
    extractaf.wm_title("Extract all frames")

    #single video
    singlelabel = LabelFrame(extractaf,text='Single video',padx=5,pady=5,font='bold')

    # videopath
    videopath = FileSelect(singlelabel, "Video path",title='Select a video file')

    #button
    button_extractaf = Button(singlelabel, text='Extract Frames (Single video)', command= lambda:extract_allframescommand(videopath.file_path))

    #multivideo
    multilabel = LabelFrame(extractaf,text='Multiple videos',padx=5,pady=5,font='bold')
    folderpath = FolderSelect(multilabel,'Folder path',title=' Select video folder')
    button_extractmulti = Button(multilabel,text='Extract Frames (Multiple videos)',command=lambda:batch_extract_allframes(folderpath.folder_path))

    #organize
    singlelabel.grid(row=0,sticky=W,pady=10)
    videopath.grid(row=0,sticky=W)
    button_extractaf.grid(row=1,sticky=W,pady=10)

    multilabel.grid(row=1,sticky=W,pady=10)
    folderpath.grid(row=0,sticky=W)
    button_extractmulti.grid(row=1,sticky=W,pady=10)

def CSV2parquet():
    csv2parq = Toplevel()
    csv2parq.minsize(300, 300)
    csv2parq.wm_title("Convert CSV directory to parquet")
    multilabel = LabelFrame(csv2parq, text='Select CSV directory', padx=5, pady=5, font='bold')
    folderpath = FolderSelect(multilabel, 'CSV folder path', title=' Select CSV folder')
    button_extractmulti = Button(multilabel, text='Convert CSV to parquet', command=lambda: convert_csv_to_parquet(folderpath.folder_path))
    multilabel.grid(row=1, sticky=W, pady=10)
    folderpath.grid(row=0, sticky=W)
    button_extractmulti.grid(row=1, sticky=W, pady=10)

def parquet2CSV():
    parq2csv = Toplevel()
    parq2csv.minsize(300, 300)
    parq2csv.wm_title("Convert parquet directory to CSV")
    multilabel = LabelFrame(parq2csv, text='Select parquet directory', padx=5, pady=5, font='bold')
    folderpath = FolderSelect(multilabel, 'Parquet folder path', title=' Select parquet folder')
    button_extractmulti = Button(multilabel, text='Convert parquet to CSV', command=lambda: convert_parquet_to_csv(folderpath.folder_path))
    multilabel.grid(row=1, sticky=W, pady=10)
    folderpath.grid(row=0, sticky=W)
    button_extractmulti.grid(row=1, sticky=W, pady=10)

class multicropmenu:
    def __init__(self):
        multimenu = Toplevel()
        multimenu.minsize(300, 300)
        multimenu.wm_title("Multi Crop")

        self.inputfolder = FolderSelect(multimenu,"Video Folder  ")
        self.outputfolder = FolderSelect(multimenu,"Output Folder")

        self.videotype = Entry_Box(multimenu,"Video type","10")
        self.croptimes = Entry_Box(multimenu,"# of crops","10")

        button_multicrop = Button(multimenu,text='Crop',command=lambda:multicrop(self.videotype.entry_get,self.inputfolder.folder_path,self.outputfolder.folder_path,self.croptimes.entry_get))

        #organize
        self.inputfolder.grid(row=0,sticky=W,pady=2)
        self.outputfolder.grid(row=1,sticky=W,pady=2)
        self.videotype.grid(row=2,sticky=W,pady=2)
        self.croptimes.grid(row=3,sticky=W,pady=2)
        button_multicrop.grid(row=4,pady=10)

class changefps:
    def __init__(self):
        fpsmenu = Toplevel()
        fpsmenu.minsize(200, 200)
        fpsmenu.wm_title("Change frame rate of video")

        # videopath
        videopath = FileSelect(fpsmenu, "Video path",title='Select a video file')

        #fps
        label_fps= Entry_Box(fpsmenu,'Output fps','10')

        #button
        button_fps = Button(fpsmenu,text='Convert',command=lambda:changefps_singlevideo(videopath.file_path,label_fps.entry_get))

        #organize
        videopath.grid(row=0,sticky=W)
        label_fps.grid(row=1,sticky=W)
        button_fps.grid(row=2)

class changefpsmulti:
    def __init__(self):
        multifpsmenu = Toplevel()
        multifpsmenu.minsize(400, 200)
        multifpsmenu.wm_title("Change frame rate of videos in a folder")

        # videopath
        videopath = FolderSelect(multifpsmenu, "Folder path", title='Select folder with videos')

        # fps
        label_fps = Entry_Box(multifpsmenu, 'Output fps', '10')

        # button
        button_fps = Button(multifpsmenu, text='Convert',
                            command=lambda: changefps_multivideo(videopath.folder_path, label_fps.entry_get))
        # organize
        videopath.grid(row=0, sticky=W)
        label_fps.grid(row=1, sticky=W)
        button_fps.grid(row=2)


class extract_seqframe:

    def __init__(self):
        extractseqtoplevel = Toplevel()
        extractseqtoplevel.minsize(200, 200)
        extractseqtoplevel.wm_title("Extract All Frames from Seq files")

        # videopath
        videopath = FileSelect(extractseqtoplevel, "Video Path",title='Select a video file')

        # button
        button_extractseqframe = Button(extractseqtoplevel, text='Extract All Frames', command=lambda: extract_seqframescommand(videopath.file_path))

        #organize
        videopath.grid(row=0)
        button_extractseqframe.grid(row=1)

class mergeframeffmpeg:

    def __init__(self):
        # Popup window
        mergeffmpeg = Toplevel()
        mergeffmpeg.minsize(250, 250)
        mergeffmpeg.wm_title("Merge images to video")

        # select directory
        self.folderpath1selected = FolderSelect(mergeffmpeg,"Working Directory",title='Select folder with frames')

        # settings
        label_settings = LabelFrame(mergeffmpeg,text='Settings',padx=5,pady=5)
        self.label_imgformat = Entry_Box(label_settings, 'Image format','10')
        label_to = Label(label_settings,text=' to ',width=0)
        self.label_vidformat = Entry_Box(label_settings,'Video format','10')
        self.label_bitrate = Entry_Box(label_settings,'Bitrate','10')
        self.label_fps = Entry_Box(label_settings,'fps','10')

        #button
        button_mergeimg = Button(label_settings, text='Merge Images', command=self.mergeframeffmpegcommand)

        #organize
        label_settings.grid(row=1,pady=10)
        self.folderpath1selected.grid(row=0,column=0,pady=10)
        self.label_imgformat.grid(row=1,column=0,sticky=W)

        label_to.grid(row=1,column=1)
        self.label_vidformat.grid(row=1,column=2,sticky=W)
        self.label_fps.grid(row=2,column=0,sticky=W,pady=5)
        self.label_bitrate.grid(row=3,column=0,sticky=W,pady=5)
        button_mergeimg.grid(row=4,column=1,sticky=E,pady=10)


    def mergeframeffmpegcommand(self):

        imgformat = self.label_imgformat.entry_get
        vidformat = self.label_vidformat.entry_get
        bitrate = self.label_bitrate.entry_get
        fps = self.label_fps.entry_get

        mergemovieffmpeg(self.folderpath1selected.folder_path,fps,vidformat,bitrate,imgformat)
        print('Video created.')

class creategif:

    def __init__(self):
        # Popup window
        create_gif = Toplevel()
        create_gif.minsize(250, 250)
        create_gif.wm_title("Generate Gif from video")


        #Create Gif
        label_creategif = LabelFrame(create_gif,text='Video to Gif',padx=5,pady=5,font='bold')
        # select video to convert to gif
        videoselected = FileSelect(label_creategif, 'Video path',title='Select a video file')
        label_starttime = Entry_Box(label_creategif,'Start time(s)','12')
        label_duration = Entry_Box(label_creategif,'Duration(s)','12')
        label_size = Entry_Box(label_creategif,'Width','12')
        size_description = Label(label_creategif,text='example width: 240,360,480,720,1080',font=("Times", 10, "italic"))
        #convert
        button_creategif = Button(label_creategif,text='Generate Gif',command=lambda :generategif(videoselected.file_path,label_starttime.entry_get,label_duration.entry_get,label_size.entry_get))

        #organize
        label_creategif.grid(row=0,sticky=W)
        videoselected.grid(row=0,sticky=W,pady=5)
        label_starttime.grid(row=1,sticky=W)
        label_duration.grid(row=2,sticky=W)
        label_size.grid(row=3,sticky=W)
        size_description.grid(row=4,sticky=W)
        button_creategif.grid(row=5,sticky=E,pady=10)

class new_window:

    def open_Folder(self):
        print("Current directory is %s" % os.getcwd())
        folder = askdirectory(title='Select Frame Folder')
        os.chdir(folder)
        print("Working directory is %s" % os.getcwd())

    def __init__(self, script_name):
        window = Tk()

        folder = Frame(window)
        folder.grid(row = 0, column = 1, sticky = N)

        choose_folder = Label(folder, text="Choose Folder")
        choose_folder.grid(row = 0, column = 0, sticky = W)
        select_folder = Button(folder, text="Choose Folder...", command=self.open_Folder)
        select_folder.grid(row = 0, column = 0, sticky = W)
        run_script = Button(folder, text="Run", command=script_name.main)
        run_script.grid(columnspan = 2, sticky = S)

def createWindow(scriptname):
    new_window(scriptname)


class visualize_pose:
    def __init__(self):
        viz_pose = Toplevel()
        viz_pose.minsize(350, 200)
        viz_pose.wm_title('Visualize pose-estimation')

        settings_frame = LabelFrame(viz_pose, text='File Settings', font=('Helvetica', 10, 'bold'), pady=5, padx=5)

        self.input_folder = FolderSelect(settings_frame, 'Input directory (with csv/parquet files)', title='Select input folder')
        self.output_folder = FolderSelect(settings_frame, 'Output directory (where your videos will be saved)', title='Select output folder')
        self.circle_size = Entry_Box(settings_frame, 'Circle size', '0')

        button_run_visualization = Button(settings_frame, text='Visualize pose', command=self.run_visualization)
        settings_frame.grid(row=0, sticky=W)
        self.input_folder.grid(row=0, column=0, pady=10, sticky=W)
        self.output_folder.grid(row=1, column=0, pady=10, sticky=W)
        self.circle_size.grid(row=2, column=0, pady=10, sticky=W)
        button_run_visualization.grid(row=3, column=0, pady=10)

    def run_visualization(self):
        circle_size_int = self.circle_size.entry_get
        input_folder = self.input_folder.folder_path
        output_folder = self.output_folder.folder_path
        print(input_folder, output_folder)
        if circle_size_int == '':
            print('Please enter a circle size to continue')

        if not circle_size_int.isdigit():
            print('Circle size must be a integer')

        elif (input_folder == '') or (input_folder == 'No folder selected'):
            print('Please select an input folder to continue')

        elif (output_folder == '') or (output_folder == 'No folder selected'):
            print('Please select an output folder to continue')

        else:
            create_video_from_dir(in_directory=input_folder, out_directory=output_folder, circle_size=int(circle_size_int))

class get_coordinates_from_video:

    def __init__(self):
        # Popup window
        getcoord = Toplevel()
        getcoord.minsize(200, 200)
        getcoord.wm_title('Get Coordinates in Video')

        # settings files selected
        self.videopath1selected = FileSelect(getcoord, "Video selected",title='Select a video file')

        # label for known mm
        self.label_knownmm = Entry_Box(getcoord,'Known length in real life(mm)','0')

        #button
        button_getcoord = Button(getcoord, text='Get Distance', command=self.getcoord)

        #organize
        self.videopath1selected.grid(row=0,column=0,pady=10,sticky=W)
        self.label_knownmm.grid(row=1,column=0,pady=10,sticky=W)
        button_getcoord.grid(row=2,column=0,pady=10)

    def getcoord(self):
        filename= self.videopath1selected.file_path
        knownmm_value = self.label_knownmm.entry_get
        if knownmm_value == '':
            print('Please enter the known millimeters to continue')

        elif filename != '' and filename != 'No file selected':

            getco = get_coordinates_nilsson(self.videopath1selected.file_path,knownmm_value)
            print('The distance between the two set points is ', str(getco))

        else:
            print('Please select a video')

class project_config:

    def __init__(self):
        self.all_entries = []
        self.allmodels = []
        # Popup window
        self.toplevel = Toplevel()
        self.toplevel.minsize(750, 750)
        self.toplevel.wm_title("Project Configuration")

        #tab
        tab_parent = ttk.Notebook(hxtScrollbar(self.toplevel))
        tab1 = ttk.Frame(tab_parent)
        tab2 = ttk.Frame(tab_parent)
        tab3 = ttk.Frame(tab_parent)
        tab4 = ttk.Frame(tab_parent)
        #tab title
        tab_parent.add(tab1,text=f'{"[ Generate project config ]": ^20s}')
        tab_parent.add(tab2, text=f'{"[ Import videos into project folder ]": ^20s}')
        tab_parent.add(tab3, text=f'{"[ Import tracking data ]": ^20s}')
        tab_parent.add(tab4, text=f'{"[ Extract frames into project folder ]": ^20s}')
        #initiate tab
        tab_parent.grid(row=0)

        # General Settings
        self.label_generalsettings = LabelFrame(tab1, text='General Settings',fg='black',font =("Helvetica",12,'bold'),padx=5,pady=5)
        self.directory1Select = FolderSelect(self.label_generalsettings, "Project Path:", title='Select Main Directory',lblwidth='12')
        self.label_project_name = Entry_Box(self.label_generalsettings, 'Project Name:',labelwidth='12')
        label_project_namedescrip = Label(self.label_generalsettings, text='(project name cannot contain spaces)')
        self.csvORparquet = DropDownMenu(self.label_generalsettings,'Workflow file type',['csv','parquet'],'15')
        self.csvORparquet.setChoices('csv')

        #SML Settings
        self.label_smlsettings = LabelFrame(self.label_generalsettings, text='SML Settings',padx=5,pady=5)
        self.label_notarget = Entry_Box(self.label_smlsettings,'Number of predictive classifiers (behaviors):','33',
                                        validation='numeric')
        addboxButton = Button(self.label_smlsettings, text='<Add predictive classifier>', fg="navy",
                              command=lambda:self.addBox(self.label_notarget.entry_get))

        ##dropdown for # of mice
        self.dropdownbox = LabelFrame(self.label_generalsettings, text='Animal Settings')

        ## choose multi animal or not
        self.singleORmulti = DropDownMenu(self.dropdownbox,'Type of Tracking',['Classic tracking','Multi tracking', '3D tracking'],'15', com=self.trackingselect)
        self.singleORmulti.setChoices('Classic tracking')

        #choice
        self.frame2 = Frame(self.dropdownbox)

        label_dropdownmice = Label(self.frame2, text='# config')
        self.option_mice, optionsBasePhotosList = bodypartConfSchematic()

        # del multi animal
        del self.option_mice[9:13]
        del optionsBasePhotosList[9:13]

        self.var = StringVar()
        self.var.set(self.option_mice[6])
        micedropdown = OptionMenu(self.frame2, self.var, *self.option_mice)

        self.var.trace("w", self.change_image)

        self.photos = []
        for i in range(len(optionsBasePhotosList)):
            self.photos.append(PhotoImage(file=os.path.join(os.path.dirname(__file__),(optionsBasePhotosList[i]))))

        self.label = Label(self.frame2, image=self.photos[6])
        self.label.grid(row=10,sticky=W,columnspan=2)
        #reset button
        resetbutton = Button(self.frame2,text='Reset user-defined pose configs',command=self.resetSettings)
        #organize
        self.singleORmulti.grid(row=0,sticky=W)
        self.frame2.grid(row=1,sticky=W)
        label_dropdownmice.grid(row=0,column=0,sticky=W)
        micedropdown.grid(row=0,column=1,sticky=W)
        self.label.grid(row=1,sticky=W,columnspan=2)
        resetbutton.grid(row=0,sticky=W,column=2)

        #generate project ini
        button_generateprojectini = Button(self.label_generalsettings, text='Generate Project Config ', command=self.make_projectini, font=("Helvetica",10,'bold'),fg='navy')

        #####import videos
        label_importvideo = LabelFrame(tab2, text='Import Videos into project folder',fg='black', font=("Helvetica",12,'bold'), padx=15, pady=5)
        # multi video
        label_multivideoimport = LabelFrame(label_importvideo, text='Import multiple videos', pady=5, padx=5)
        self.multivideofolderpath = FolderSelect(label_multivideoimport, 'Folder path',title='Select Folder with videos')
        self.video_type = Entry_Box(label_multivideoimport, 'Format (i.e., mp4, avi):', '18')
        button_multivideoimport = Button(label_multivideoimport, text='Import multiple videos',command= self.import_multivid,fg='navy')
        # singlevideo
        label_singlevideoimport = LabelFrame(label_importvideo, text='Import single video', pady=5, padx=5)
        self.singlevideopath = FileSelect(label_singlevideoimport, "Video path",title='Select a video file')
        button_importsinglevideo = Button(label_singlevideoimport, text='Import a video',command= self.import_singlevid,fg='navy')


        #import all csv file into project folder
        self.label_import_csv = LabelFrame(tab3,text='Import Tracking Data',fg='black',font=("Helvetica",12,'bold'),pady=5,padx=5)
        self.filetype = DropDownMenu(self.label_import_csv,'File type',['CSV (DLC/DeepPoseKit)','JSON (BENTO)','H5 (multi-animal DLC)','SLP (SLEAP)', 'TRK (multi-animal APT)', 'MAT (DANNCE 3D)'],'12',com=self.fileselected)
        self.filetype.setChoices('CSV (DLC/DeepPoseKit)')
        self.frame = Frame(self.label_import_csv)

        #method
        self.labelmethod = LabelFrame(self.frame,text='Interpolate missing pose-estimation data',pady=5, padx=5)
        self.interpolation = DropDownMenu(self.labelmethod,'Interpolation',['None','Animal(s): Nearest', 'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear', 'Body-parts: Quadratic'],'12')
        self.interpolation.setChoices('None')

        # smoothing
        self.smooth_pose_win_lbl = LabelFrame(self.frame, text='Smooth pose-estimation data', pady=5, padx=5)
        self.smooth_dropdown = DropDownMenu(self.smooth_pose_win_lbl, 'Smoothing', ['None', 'Gaussian', 'Savitzky Golay'], '12', com=self.smoothing_selected)
        self.smooth_dropdown.setChoices('None')
        self.smoothing_time_label = Entry_Box(self.smooth_pose_win_lbl, 'Period (ms):', labelwidth='12', width=10)

        #multicsv
        label_multicsvimport = LabelFrame(self.frame, text='Import multiple csv files', pady=5, padx=5)
        self.folder_csv = FolderSelect(label_multicsvimport,'Folder Select:',title='Select Folder with .csv(s)')
        button_import_csv = Button(label_multicsvimport,text='Import csv to project folder',command = self.import_multicsv,fg='navy')
        #singlecsv
        label_singlecsvimport = LabelFrame(self.frame, text='Import single csv file', pady=5, padx=5)
        self.file_csv = FileSelect(label_singlecsvimport,'File Select',title='Select a .csv file')
        button_importsinglecsv = Button(label_singlecsvimport,text='Import single csv to project folder',command=self.import_singlecsv,fg='navy')


        #extract videos in projects
        label_extractframes = LabelFrame(tab4,text='Extract Frames into project folder',fg='black',font=("Helvetica",12,'bold'),pady=5,padx=5)
        label_note = Label(label_extractframes,text='Note: This is no longer needed for labelling videos. Instead, extract video frames are used for visualizations')
        label_caution = Label(label_extractframes,text='Caution: This extract all frames from all videos in project,')
        label_caution2 = Label(label_extractframes,text='and is computationally expensive if there is a lot of videos at high frame rates/resolution.')
        button_extractframes = Button(label_extractframes,text='Extract frames',command=self.extract_frames,fg='navy')

        #organize
        self.label_generalsettings.grid(row=0,sticky=W)
        self.directory1Select.grid(row=1,column=0,sticky=W)
        self.label_project_name.grid(row=2,column=0,sticky=W)
        label_project_namedescrip.grid(row=3,sticky=W)
        self.csvORparquet.grid(row=4,sticky=W)

        self.label_smlsettings.grid(row=5,column=0,sticky=W,pady=5,columnspan=2)
        self.label_notarget.grid(row=0,column=0,sticky=W,pady=5,columnspan=2)
        addboxButton.grid(row=1,column=0,sticky=W,pady=6)

        self.dropdownbox.grid(row=6,column=0,sticky=W)

        button_generateprojectini.grid(row=20, pady=5, ipadx=5, ipady=5)

        label_importvideo.grid(row=4,sticky=W,pady=5)
        label_multivideoimport.grid(row=0, sticky=W)
        self.multivideofolderpath.grid(row=0, sticky=W)
        self.video_type.grid(row=1, sticky=W)
        button_multivideoimport.grid(row=2, sticky=W)
        label_singlevideoimport.grid(row=1,sticky=W)
        self.singlevideopath.grid(row=0,sticky=W)
        button_importsinglevideo.grid(row=1,sticky=W)

        self.label_import_csv.grid(row=5,sticky=W,pady=5)
        self.filetype.grid(row=0,sticky=W)
        self.frame.grid(row=1,sticky=W)
        self.labelmethod.grid(row=0,sticky=W)
        self.interpolation.grid(row=0,sticky=W)
        self.smooth_pose_win_lbl.grid(row=1, sticky=W)
        self.smooth_dropdown.grid(row=1,sticky=W)

        label_multicsvimport.grid(row=2, sticky=W)
        self.folder_csv.grid(row=0,sticky=W)
        button_import_csv.grid(row=1,sticky=W)
        label_singlecsvimport.grid(row=3,sticky=W)
        self.file_csv.grid(row=0,sticky=W)
        button_importsinglecsv.grid(row=1,sticky=W)

        label_extractframes.grid(row=7,sticky=W)
        label_note.grid(row=0,sticky=W)
        label_caution.grid(row=1,sticky=W)
        label_caution2.grid(row=2,sticky=W)
        button_extractframes.grid(row=3,sticky=W)

    def smoothing_selected(self, choice):
        if choice == 'None':
            self.smoothing_time_label.grid_forget()
        if (choice == 'Gaussian') or (choice == 'Savitzky Golay'):
            self.smoothing_time_label.grid(row=1, column=1, sticky=E)

    def trackingselect(self,val):
        try:
            self.frame2.destroy()
        except:
            pass
        # choice
        self.frame2 = Frame(self.dropdownbox)
        if val == 'Classic tracking':
            label_dropdownmice = Label(self.frame2, text='# config')
            self.option_mice, optionsBasePhotosList = bodypartConfSchematic()
            # del multi animal
            del self.option_mice[9:13]
            del optionsBasePhotosList[9:13]

            self.var = StringVar()
            self.var.set(self.option_mice[6])
            micedropdown = OptionMenu(self.frame2, self.var, *self.option_mice)

            self.var.trace("w", self.change_image)

            self.photos = []
            for i in range(len(optionsBasePhotosList)):
                self.photos.append(PhotoImage(file=os.path.join(os.path.dirname(__file__), (optionsBasePhotosList[i]))))

            self.label = Label(self.frame2, image=self.photos[6])
            self.label.grid(row=10, sticky=W, columnspan=2)
            # reset button
            resetbutton = Button(self.frame2, text='Reset user-defined pose configs', command=self.resetSettings)
            # organize
            self.frame2.grid(row=1, sticky=W)
            label_dropdownmice.grid(row=0, column=0, sticky=W)
            micedropdown.grid(row=0, column=1, sticky=W)
            self.label.grid(row=1, sticky=W, columnspan=2)
            resetbutton.grid(row=0, sticky=W, column=2)

        else:
            if val == 'Multi tracking':
                label_dropdownmice = Label(self.frame2, text='# config')
                self.option_mice, optionsBasePhotosList = bodypartConfSchematic()
                # del single animal
                del self.option_mice[0:9]
                self.option_mice.remove('3D')
                del optionsBasePhotosList[0:9]
                optionsBasePhotosList = [x for x in optionsBasePhotosList if "Picture13.png" not in x]

                self.var = StringVar()
                self.var.set(self.option_mice[2])
                micedropdown = OptionMenu(self.frame2, self.var, *self.option_mice)

                self.var.trace("w", self.change_image)

                self.photos = []
                for i in range(len(optionsBasePhotosList)):
                    self.photos.append(
                        PhotoImage(file=os.path.join(os.path.dirname(__file__), (optionsBasePhotosList[i]))))

                self.label = Label(self.frame2, image=self.photos[2])
                self.label.grid(row=10, sticky=W, columnspan=2)
                # reset button
                resetbutton = Button(self.frame2, text='Reset user-defined pose configs', command=self.resetSettings)
                # organize
                self.frame2.grid(row=1, sticky=W)
                label_dropdownmice.grid(row=0, column=0, sticky=W)
                micedropdown.grid(row=0, column=1, sticky=W)
                self.label.grid(row=1, sticky=W, columnspan=2)
                resetbutton.grid(row=0, sticky=W, column=2)

            if val == '3D tracking':
                label_dropdownmice = Label(self.frame2, text='# config')
                self.option_mice, optionsBasePhotosList = bodypartConfSchematic()

                # del single animal and multi-animal
                del self.option_mice[0:12]
                del optionsBasePhotosList[0:12]
                self.var = StringVar()
                self.var.set(self.option_mice[0])
                micedropdown = OptionMenu(self.frame2, self.var, *self.option_mice)

                self.var.trace("w", self.change_image)

                self.photos = []
                for i in range(len(optionsBasePhotosList)):
                    self.photos.append(
                        PhotoImage(file=os.path.join(os.path.dirname(__file__), (optionsBasePhotosList[i]))))

                self.label = Label(self.frame2, image=self.photos[0])
                self.label.grid(row=10, sticky=W, columnspan=2)
                # reset button
                resetbutton = Button(self.frame2, text='Reset user-defined pose configs', command=self.resetSettings)
                # organize
                self.frame2.grid(row=1, sticky=W)
                label_dropdownmice.grid(row=0, column=0, sticky=W)
                micedropdown.grid(row=0, column=1, sticky=W)
                self.label.grid(row=1, sticky=W, columnspan=2)
                resetbutton.grid(row=0, sticky=W, column=2)

    def fileselected(self,val):
        try:
            self.frame.destroy()
        except:
            pass

        self.frame = Frame(self.label_import_csv)
        self.labelmethod = LabelFrame(self.frame, text='Interpolate missing pose-estimation data', pady=5, padx=5)
        self.interpolation = DropDownMenu(self.labelmethod, 'Interpolation', ['None','Animal(s): Nearest', 'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear', 'Body-parts: Quadratic'], '12')
        self.interpolation.setChoices('None')
        self.interpolation.grid(row=0, sticky=W)

        self.smooth_pose_win_lbl = LabelFrame(self.frame, text='Smooth pose-estimation data', pady=5, padx=5)
        self.smooth_dropdown = DropDownMenu(self.smooth_pose_win_lbl, 'Smoothing', ['None', 'Gaussian', 'Savitzky Golay'], '12', com=self.smoothing_selected)
        self.smooth_dropdown.setChoices('None')
        self.smooth_dropdown.grid(row=1, sticky=E)
        self.smoothing_time_label = Entry_Box(self.smooth_pose_win_lbl, 'Period (ms):', labelwidth='12', width=10)


        if self.filetype.getChoices()=='CSV (DLC/DeepPoseKit)':
            # multicsv
            label_multicsvimport = LabelFrame(self.frame, text='Import multiple csv files', pady=5, padx=5)
            self.folder_csv = FolderSelect(label_multicsvimport, 'Folder Select:', title='Select Folder with .csv(s)')
            button_import_csv = Button(label_multicsvimport, text='Import csv to project folder', command=self.import_multicsv, fg='navy')

            # singlecsv
            label_singlecsvimport = LabelFrame(self.frame, text='Import single csv file', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlecsvimport, 'File Select', title='Select a .csv file')
            button_importsinglecsv = Button(label_singlecsvimport, text='Import single csv to project folder',
                                            command=self.import_singlecsv, fg='navy')
            self.frame.grid(row=1,sticky=W)
            self.labelmethod.grid(row=0,sticky=W)
            self.smooth_pose_win_lbl.grid(row=1,sticky=W)
            label_multicsvimport.grid(row=2, sticky=W)
            self.folder_csv.grid(row=0, sticky=W)
            button_import_csv.grid(row=1, sticky=W)
            label_singlecsvimport.grid(row=3, sticky=W)
            self.file_csv.grid(row=0, sticky=W)
            button_importsinglecsv.grid(row=1, sticky=W)

        elif val == 'MAT (DANNCE 3D)':
            # multicsv
            label_multicsvimport = LabelFrame(self.frame, text='Import multiple DANNCE files', pady=5, padx=5)
            self.folder_csv = FolderSelect(label_multicsvimport, 'Folder selected:', title='Select Folder with MAT files')
            button_import_csv = Button(label_multicsvimport, text='Import DANNCE files to project folder', command = lambda: import_DANNCE_folder(self.configinifile, self.folder_csv.folder_path, self.interpolation.getChoices()), fg='navy')

            # singlecsv
            label_singlecsvimport = LabelFrame(self.frame, text='Import single DANNCE files', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlecsvimport, 'File selected', title='Select a .csv file')
            button_importsinglecsv = Button(label_singlecsvimport, text='Import single DANNCE to project folder', command = lambda: import_DANNCE_file(self.configinifile, self.file_csv.file_path, self.interpolation.getChoices()), fg='navy')
            self.frame.grid(row=1, sticky=W)

            # method
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            label_multicsvimport.grid(row=2, sticky=W)
            self.folder_csv.grid(row=0, sticky=W)
            button_import_csv.grid(row=1, sticky=W)
            label_singlecsvimport.grid(row=3, sticky=W)
            self.file_csv.grid(row=0, sticky=W)
            button_importsinglecsv.grid(row=1, sticky=W)

        elif self.filetype.getChoices()=='JSON (BENTO)':

            # multijson
            label_multijsonimport = LabelFrame(self.frame, text='Import multiple json files', pady=5, padx=5)
            self.folder_json = FolderSelect(label_multijsonimport, 'Folder Select:',
                                            title='Select Folder with .json(s)')
            button_import_json = Button(label_multijsonimport, text='Import json to project folder', command=lambda: json2csv_folder(self.configinifile, self.folder_json.folder_path, self.interpolation.getChoices(), self.smooth_dropdown.getChoices()), fg='navy')

            # singlecsv
            label_singlejsonimport = LabelFrame(self.frame, text='Import single json file', pady=5, padx=5)
            self.file_json = FileSelect(label_singlejsonimport, 'File Select', title='Select a .csv file')
            button_importsinglejson = Button(label_singlejsonimport, text='Import single .json to project folder', command=lambda: json2csv_file(self.configinifile, self.file_json.file_path, self.interpolation.getChoices(), self.smooth_dropdown.getChoices()), fg='navy')
            # import json into projectfolder
            self.frame.grid(row=1, sticky=W)
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            label_multijsonimport.grid(row=2, sticky=W)
            self.folder_json.grid(row=0, sticky=W)
            button_import_json.grid(row=1, sticky=W)
            label_singlejsonimport.grid(row=3, sticky=W)
            self.file_json.grid(row=0, sticky=W)
            button_importsinglejson.grid(row=1, sticky=W)

        elif self.filetype.getChoices() in ('H5 (multi-animal DLC)','SLP (SLEAP)', 'TRK (multi-animal APT)'):
            animalsettings = LabelFrame(self.frame,text='Animal settings',pady=5,padx=5)
            noofanimals = Entry_Box(animalsettings,'No of animals','15')
            animalnamebutton = Button(animalsettings,text='Confirm',command=lambda:self.animalnames(noofanimals.entry_get,animalsettings))

            if self.filetype.getChoices() == 'H5 (multi-animal DLC)':
                options =['skeleton','box','ellipse']
                self.dropdowndlc = DropDownMenu(self.frame,'Tracking type',options,'15')
                self.dropdowndlc.setChoices(options[1])
                self.h5path = FolderSelect(self.frame,'Path to h5 files',lblwidth=15)
                labelinstruction = Label(self.frame,text='Please import videos before importing the multi animal DLC tracking data')
                runsettings = Button(self.frame,text='Import h5',command=self.importh5)
                #organize
                self.dropdowndlc.grid(row=3, sticky=W)

            elif self.filetype.getChoices() == 'SLP (SLEAP)':
                self.h5path = FolderSelect(self.frame, 'Path to .slp files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the multi animal SLEAP tracking data')
                runsettings = Button(self.frame, text='Import .slp', command=self.importh5)

            elif self.filetype.getChoices() == 'TRK (multi-animal APT)':
                self.h5path = FolderSelect(self.frame, 'Path to .trk files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the multi animal trk tracking data')
                runsettings = Button(self.frame, text='Import .trk', command=self.importh5)

            #organize
            self.frame.grid(row=1,sticky=W)
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            animalsettings.grid(row=2,sticky=W)
            noofanimals.grid(row=0,sticky=W)
            animalnamebutton.grid(row=0,column=1,sticky=W)

            self.h5path.grid(row=4,sticky=W)
            labelinstruction.grid(row=5,pady=10,sticky=W)
            runsettings.grid(row=6,pady=10)

    def importh5(self):
        idlist = []
        try:
            for i in self.animalnamelist:
                idlist.append(i.entry_get)
        except AttributeError:
            print('Please fill in the animal identity names appropriately.')

        id_ini = idlist.copy()
        id_ini = str(id_ini)
        if currentPlatform == 'Windows':
            id_ini = id_ini.replace('\'','')
        if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
            id_ini = id_ini.replace('/', '')
        id_ini = id_ini.replace('[', '')
        id_ini = id_ini.replace(']', '')
        id_ini = id_ini.replace(' ', '')
        config = ConfigParser()
        configFile = str(self.configinifile)
        config.read(configFile)

        # write the new values into ini file
        config.set('Multi animal IDs', 'ID_list',id_ini)

        with open(self.configinifile, 'w') as configfile:
            config.write(configfile)

        smooth_settings_dict = {}
        if self.smooth_dropdown.getChoices() == 'Gaussian':
            smooth_settings_dict['Method'] = 'Gaussian'
            smooth_settings_dict['Parameters'] = {}
            smooth_settings_dict['Parameters']['Time_window'] = self.smoothing_time_label.entry_get

        if self.smooth_dropdown.getChoices() == 'Savitzky Golay':
            smooth_settings_dict['Method'] = 'Savitzky Golay'
            smooth_settings_dict['Parameters'] = {}
            smooth_settings_dict['Parameters']['Time_window'] = self.smoothing_time_label.entry_get


        if self.smooth_dropdown.getChoices() == 'None':
            smooth_settings_dict['Method'] = 'None'

        if self.filetype.getChoices() == 'H5 (multi-animal DLC)':
            importMultiDLCpose(self.configinifile,self.h5path.folder_path,self.dropdowndlc.getChoices(),idlist, self.interpolation.getChoices(), smooth_settings_dict)

        if self.filetype.getChoices() == 'SLP (SLEAP)':
            try:
                importSLEAPbottomUP(self.configinifile,self.h5path.folder_path,idlist, self.interpolation.getChoices(), smooth_settings_dict)
            except Exception as e:
                messagebox.showerror("Error", str(e))

        if self.filetype.getChoices() == 'TRK (multi-animal APT)':
            try:
                import_trk(self.configinifile,self.h5path.folder_path,idlist, self.interpolation.getChoices(), smooth_settings_dict)
            except Exception as e:
                messagebox.showerror("Error", str(e))


    def animalnames(self,noofanimal,master):
        try:
            self.frame2.destroy()
        except:
            pass

        no_animal = int(noofanimal)
        self.animalnamelist =[0]*no_animal

        self.frame2 = Frame(master)
        self.frame2.grid(row=1,sticky=W)

        for i in range(no_animal):
            self.animalnamelist[i] = Entry_Box(self.frame2,'Animal ' + str(i+1) + ' name','15')
            self.animalnamelist[i].grid(row=i,sticky=W)

    def resetSettings(self):
        popup = Tk()
        popup.minsize(300, 100)
        popup.wm_title("Warning!")
        popupframe = LabelFrame(popup)
        label = Label(popupframe, text='Do you want to reset user-defined pose-configs?')
        label.grid(row=0,columnspan=2)
        B1 = Button(popupframe,text='Yes', command= lambda:reset_DiagramSettings(popup))
        B2 = Button(popupframe, text="No", command=popup.destroy)
        popupframe.grid(row=0,columnspan=2)
        B1.grid(row=1,column=0,sticky=W)
        B2.grid(row=1,column=1,sticky=W)

        popup.mainloop()

    def poseconfigSettings(self):
        # Popup window
        poseconfig = Toplevel()
        poseconfig.minsize(400, 400)
        poseconfig.wm_title("Pose Configuration")


        self.scroller = hxtScrollbar(poseconfig)
        self.scroller.pack(expand=True, fill=BOTH)

        # define name for poseconfig settings
        self.configname = Entry_Box(self.scroller,'Pose config name','15')

        # no of animals
        self.noOfAnimals = Entry_Box(self.scroller,'# of Animals','15')

        # no of bodyparts
        self.noOfBp = Entry_Box(self.scroller,'# of Bodyparts','15')

        # path to image
        self.imgPath = FileSelect(self.scroller,'Image Path')

        # button for bodypart table
        tablebutton = Button(self.scroller,text='Confirm',command= lambda :self.bpTable(self.scroller))
        #button for saving poseconfig
        self.saveposeConfigbutton = Button(self.scroller,text='Save Pose Config',command=lambda:self.savePoseConfig(self.scroller))
        self.saveposeConfigbutton.config(state='disabled')

        #organize
        self.configname.grid(row=0,sticky=W)
        self.noOfAnimals.grid(row=1,sticky=W)
        self.noOfBp.grid(row=2,sticky=W)
        self.imgPath.grid(row=3,sticky=W,pady=2)
        tablebutton.grid(row=4,pady=5)
        self.saveposeConfigbutton.grid(row=6,pady=5)

    def savePoseConfig(self, master):
        configName = self.configname.entry_get
        self.noAnimals = int(self.noOfAnimals.entry_get)
        noBps = self.noOfBp.entry_get
        Imagepath = self.imgPath.file_path
        BpNameList = []
        animal_id_list = []
        for entry in zip(self.bpnamelist, self.bp_animal_list):
            BpNameList.append(entry[0].entry_get)
            if self.noAnimals > 1:
                animal_id_list.append(entry[1].entry_get)
        define_new_pose_configuration(configName, self.noAnimals, noBps, Imagepath, BpNameList, self.noAnimals, animal_id_list)
        master.destroy()
        self.toplevel.destroy()
        project_config()

    def bpTable(self,master):
        try:
            self.table_frame.destroy()
        except:
            pass

        self.noAnimals = int(self.noOfAnimals.entry_get)
        noofbp = int(self.noOfBp.entry_get)
        self.bpnamelist = [0]*noofbp*self.noAnimals
        self.bp_animal_list = [0]*noofbp*self.noAnimals

        if currentPlatform == 'Windows':
            if self.noAnimals > 1:
                self.table_frame = LabelFrame(master,text='Bodyparts\' name                       Animal ID number')
            else:
                self.table_frame = LabelFrame(master, text='Bodyparts\' name')

        if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
            if self.noAnimals > 1:
                self.table_frame = LabelFrame(master,text='Bodyparts/' 'name                      Animal ID number')
            else:
                self.table_frame = LabelFrame(master, text='Bodyparts/' 'name')

        self.table_frame.grid(row=5, sticky=W, column=0)
        scroll_table = hxtScrollbar(self.table_frame)

        for i in range(noofbp * self.noAnimals):
            self.bpnamelist[i] = Entry_Box(scroll_table,str(i+1),'2')
            self.bpnamelist[i].grid(row=i, column=0)
            if self.noAnimals > 1:
                self.bp_animal_list[i] = Entry_Box(scroll_table, '', '2', validation='numeric')
                self.bp_animal_list[i].grid(row=i, column=1)


        self.saveposeConfigbutton.config(state='normal')

    def change_image(self,*args):
        if (self.var.get() != 'Create pose config...'):
            self.label.config(image=self.photos[self.option_mice.index(str(self.var.get()))])
        else:
            self.poseconfigSettings()

    def import_singlecsv(self):

        copy_singlecsv_ini(self.configinifile, self.file_csv.file_path)

        # read in configini
        config = ConfigParser()
        config.read(str(self.configinifile))
        animalIDlist = config.get('Multi animal IDs', 'id_list')

        #name filtering
        dir_name, filename, extension = get_fn_ext(self.file_csv.file_path)


        if ((('DeepCut') in filename) and (extension == '.csv')):
            newFname = filename.split('DeepCut')[0] + extension
        elif ((('DLC_') in filename) and (extension == '.csv')):
            newFname = filename.split('DLC_')[0] + extension
        else:
            newFname = filename + extension

        csvfile = os.path.join(os.path.dirname(self.configinifile), 'csv', 'input_csv', newFname)

        if not animalIDlist:
            df = pd.read_csv(csvfile)

            tmplist = []
            for i in df.loc[0]:
                tmplist.append(i)

            if 'individuals' in tmplist:
                tmplist.remove('individuals')

                if len(set(tmplist)) == 1:
                    print('single animal using maDLC detected. Removing "individuals" row...')
                    df = df.iloc[1:]
                    df.to_csv(csvfile, index=False)
                    print('Row removed for', os.path.basename(i))

            else:
                pass


        if self.interpolation.getChoices() != 'None':
            print('Interpolating missing values (Method: ' + str(self.interpolation.getChoices()) + ') ...')
            interpolate_body_parts = Interpolate(self.configinifile,pd.read_csv(csvfile, index_col=0))
            interpolate_body_parts.detect_headers()
            interpolate_body_parts.fix_missing_values(self.interpolation.getChoices())
            interpolate_body_parts.reorganize_headers()
            interpolate_body_parts.new_df.to_csv(csvfile)

        if self.smooth_dropdown.getChoices() == 'Gaussian':
            time_window = self.smoothing_time_label.entry_get
            smooth_data_gaussian(config=config, file_path=csvfile, time_window_parameter=time_window)

        if self.smooth_dropdown.getChoices() == 'Savitzky Golay':
            time_window = self.smoothing_time_label.entry_get
            smooth_data_savitzky_golay(config=config, file_path=csvfile, time_window_parameter=time_window)

    def import_multicsv(self):
        try:
            copy_allcsv_ini(self.configinifile, self.folder_csv.folder_path)
            #read in configini
            config = ConfigParser()
            config.read(str(self.configinifile))
            animalIDlist = config.get('Multi animal IDs', 'id_list')

            if not animalIDlist:
                # get all csv in project folder input csv
                csvfolder = os.path.join(os.path.dirname(self.configinifile), 'csv', 'input_csv')
                allcsvs = []
                for i in os.listdir(csvfolder):
                    if i.endswith('.csv'):
                        csvfile = os.path.join(csvfolder, i)
                        allcsvs.append(csvfile)

                # screen for madlc format but single animal
                for i in allcsvs:
                    df = pd.read_csv(i)
                    tmplist = []

                    for j in df.loc[0]:
                        tmplist.append(j)

                    #if it is madlc
                    if 'individuals' in tmplist:
                        tmplist.remove('individuals')
                        #if only single animal in madlc
                        if len(set(tmplist)) == 1:
                            print('single animal using maDLC detected. Removing "individuals" row...')
                            df = df.iloc[1:]
                            df.to_csv(i, index=False)
                            print('Row removed for',os.path.basename(i))
                    else:
                        pass

            csvfilepath = os.path.join(os.path.dirname(self.configinifile),'csv','input_csv')
            csvfile = glob.glob(csvfilepath + '/*.csv')

            if self.interpolation.getChoices() != 'None':
                print('Interpolating missing values (Method: ' + str(self.interpolation.getChoices()) + ') ...')
                for file in csvfile:
                    print(str(os.path.basename(file).replace('.csv', '')) + '...')
                    csv_df = pd.read_csv(file, index_col=0)
                    interpolate_body_parts = Interpolate(self.configinifile, csv_df)
                    interpolate_body_parts.detect_headers()
                    interpolate_body_parts.fix_missing_values(self.interpolation.getChoices())
                    interpolate_body_parts.reorganize_headers()
                    interpolate_body_parts.new_df.to_csv(file)

            if self.smooth_dropdown.getChoices() == 'Gaussian':
                time_window = self.smoothing_time_label.entry_get
                print('Smoothing pose-estimation data using Gaussian smoothing (Time window: ' + str(time_window))
                for file in csvfile:
                    smooth_data_gaussian(config=config, file_path=file, time_window_parameter=time_window)

            if self.smooth_dropdown.getChoices() == 'Savitzky Golay':
                time_window = self.smoothing_time_label.entry_get
                print('Smoothing pose-estimation data using Savitzky Golay smoothing (Time window: ' + str(time_window))
                for file in csvfile:
                    smooth_data_savitzky_golay(config=config, file_path=file, time_window_parameter=time_window)

            print('Finished importing tracking data')

        except Exception as e:
            print(e)
            print('Please select folder with csv to proceed')

    def import_multivid(self):
        try:
            copy_multivideo_ini(self.configinifile, self.multivideofolderpath.folder_path, self.video_type.entry_get)
        except:
            print('Please select a folder containing the videos and enter the correct video format to proceed')

    def import_singlevid(self):
        try:
            copy_singlevideo_ini(self.configinifile, self.singlevideopath.file_path)
        except:
            print('Please select a video to proceed')

    def addBox(self, noTargetStr):
        try:
            for i in self.lab:
                i.destroy()
            for i in self.ent1:
                i.destroy()
        except:
            pass
        try:
            noTarget = int(noTargetStr)
        except ValueError:
            assert False, 'Invalid number of predictive classifiers'

        self.all_entries = []
        self.lab=[0]*noTarget
        self.ent1=[0]*noTarget
        for i in range(noTarget):
            self.lab[i]= Label(self.label_smlsettings, text=str('Classifier ') + str(i + 1))
            self.lab[i].grid(row=i+2, column=0, sticky=W)
            self.ent1[i] = Entry(self.label_smlsettings)
            self.ent1[i].grid(row=i+2, column=1, sticky=W)

        self.all_entries = self.ent1

    def setFilepath(self):
        file_selected = askopenfilename()
        self.modelPath.set(file_selected)

    def make_projectini(self):
        #get input from user
        #general settings input

        project_path = self.directory1Select.folder_path
        project_name = self.label_project_name.entry_get

        msconfig= str('yes')

        #sml settings
        no_targets = self.label_notarget.entry_get
        target_list = []
        for number, ent1 in enumerate(self.all_entries):
            target_list.append(ent1.get())

        ### animal settings
        listindex = self.option_mice.index(str(self.var.get()))
        print(listindex)

        if self.singleORmulti.getChoices()=='Classic tracking':
            if listindex == 0:
                bp = '4'
            elif listindex == 1:
                bp = '7'
            elif listindex == 2:
                bp = '8'
            elif listindex == 3:
                bp = '9'
            elif (listindex == 4):
                bp = '8'
            elif (listindex == 5):
                bp='14'
            elif (listindex == 6):
                bp = '16'
            elif listindex == 7:
                bp = '987'
            elif listindex == 8:
                bp = 'user_defined'
            else:
                bp = 'user_defined'

        elif self.singleORmulti.getChoices() =='Multi tracking':
            if listindex == 0:
                bp = '8'
            elif listindex == 1:
                bp = '14'
            elif listindex == 2:
                bp = '16'
            else:
                bp = 'user_defined'

        elif self.singleORmulti.getChoices() == '3D tracking':
            bp = '3D_user_defined'


        if (self.singleORmulti.getChoices() =='Classic tracking') and (bp=='user_defined') and (listindex >8):
            listindex = listindex + 4
        elif (self.singleORmulti.getChoices()=='Multi tracking') and (bp=='user_defined') and (listindex > 2):
            listindex = listindex + 1

        noAnimalsPath = os.path.join(os.path.dirname(__file__), 'pose_configurations', 'no_animals', 'no_animals.csv')
        with open(noAnimalsPath, "r", encoding='utf8') as f:
            cr = csv.reader(f, delimiter=",")  # , is default
            rows = list(cr)  # create a list of rows for instance

        if (self.singleORmulti.getChoices()=='Multi tracking'):
            listindex += 9

        if (self.singleORmulti.getChoices()=='3D tracking'):
            listindex += 12

        animalNo = str(rows[listindex])
        self.configinifile = write_inifile(msconfig,project_path,project_name,no_targets,target_list,bp, listindex, animalNo, self.csvORparquet.getChoices())

        print('Project ' + '"' + str(project_name) + '"' + " created in folder " + '"' + str(os.path.basename(project_path)) + '"')



    def extract_frames(self):
        try:
            videopath = os.path.join(os.path.dirname(self.configinifile), 'videos')
            print(videopath)
            extract_frames_ini(videopath, self.configinifile)
        except Exception as e:
            print(e.args)
            print('Please make sure videos are imported and located in /project_folder/videos')


class loadprojectMenu:
    def __init__(self,inputcommand):
        lpMenu = Toplevel()
        lpMenu.minsize(300, 200)
        lpMenu.wm_title("Load Project .ini file")

        # load project ini
        label_loadprojectini = LabelFrame(lpMenu, text='Load Project .ini', font=("Helvetica", 12, 'bold'), pady=5,
                                          padx=5, fg='black')
        self.projectconfigini = FileSelect(label_loadprojectini,'File Select:', title='Select config.ini file')

        #button
        launchloadprojectButton = Button(lpMenu,text='Load Project',command=lambda:self.launch(lpMenu,inputcommand))

        #organize
        label_loadprojectini.grid(row=0)
        self.projectconfigini.grid(row=0,sticky=W)
        launchloadprojectButton.grid(row=1,pady=10)

    def launch(self,master,command):
        if (self.projectconfigini.file_path.endswith('.ini')):
            master.destroy()
            print(self.projectconfigini.file_path)
            command(self.projectconfigini.file_path)
        else:
            print('Please select the project_config.ini file')


def open_web_link(url):
    sys.excepthook = cef.ExceptHook  # To shutdown all CEF processes on error
    cef.Initialize()
    cef.CreateBrowserSync(url=url,
                          window_title=url)
    cef.MessageLoop()


def wait_for_internet_connection(url):
    while True:
        try:
            response = urllib.request.urlopen(url, timeout=1)
            return
        except:
            pass

class loadprojectini:
    def __init__(self,configini):

        #save project ini as attribute
        self.projectconfigini = configini

        #bodyparts
        bodypartscsv= os.path.join((os.path.dirname(self.projectconfigini)),'logs','measures','pose_configs','bp_names','project_bp_names.csv')
        bp_set = pd.read_csv(bodypartscsv,header=None)[0].to_list()

        # get target
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        notarget = config.getint('SML settings','no_targets')
        pose_config_setting = config.get('create ensemble settings','pose_estimation_body_parts')
        animalNumber = config.getint('General settings','animal_no')
        if pose_config_setting == 'user_defined':
            bpSet_2 = bp_set.copy()
            if animalNumber > 1:
                bpSet_2 = [x[:-2] for x in bpSet_2]
                bpSet_2 = list(set(bpSet_2))
        targetlist = {}
        for i in range(notarget):
            targetlist[(config.get('SML settings','target_name_'+str(i+1)))]=(config.get('SML settings','target_name_'+str(i+1)))
        #starting of gui
        simongui = Toplevel()
        simongui.minsize(1300, 800)
        simongui.wm_title("Load project")
        simongui.columnconfigure(0, weight=1)
        simongui.rowconfigure(0, weight=1)

        tab_parent = ttk.Notebook(hxtScrollbar(simongui))

        tab2 = ttk.Frame(tab_parent)
        tab3 = ttk.Frame(tab_parent)
        tab4 = ttk.Frame(tab_parent)
        tab5 = ttk.Frame(tab_parent)
        tab6 = ttk.Frame(tab_parent)
        tab7 = ttk.Frame(tab_parent)
        tab8 = ttk.Frame(tab_parent)
        tab9 = ttk.Frame(tab_parent)
        tab10 = ttk.Frame(tab_parent)
        tab11 = ttk.Frame(tab_parent)
        tab12 = ttk.Frame(tab_parent)


        tab_parent.add(tab2, text= f"{'[ Further imports (data/video/frames) ]':20s}")
        tab_parent.add(tab3, text=f"{'[ Video parameters ]':20s}")
        tab_parent.add(tab4, text=f"{'[ Outlier correction ]':20s}")
        tab_parent.add(tab6, text=f"{'[ ROI ]':10s}")
        tab_parent.add(tab5, text=f"{'[ Extract features ]':20s}")
        tab_parent.add(tab7, text=f"{'[ Label behavior] ':20s}")
        tab_parent.add(tab8, text=f"{'[ Train machine model ]':20s}")
        tab_parent.add(tab9, text=f"{'[ Run machine model ]':20s}")
        tab_parent.add(tab10, text=f"{'[ Visualizations ]':20s}")
        tab_parent.add(tab11, text=f"{'[ Classifier validation ]':20s}")
        tab_parent.add(tab12,text=f"{'[ Add-ons ]':20s}")

        tab_parent.grid(row=0)
        tab_parent.enable_traversal()

        #label import
        label_import = LabelFrame(tab2)

        #import all csv file into project folder
        self.label_import_csv = LabelFrame(label_import, text='Import further tracking data', font=("Helvetica",12,'bold'), pady=5, padx=5,fg='black')
        filetype = DropDownMenu(self.label_import_csv,'File type',['CSV (DLC/DeepPoseKit)','JSON (BENTO)','H5 (multi-animal DLC)','SLP (SLEAP)', 'TRK (multi-animal APT)', 'MAT (DANNCE 3D)'],'15',com=self.fileselected)
        filetype.setChoices('CSV (DLC/DeepPoseKit)')
        self.frame = Frame(self.label_import_csv)

        #method
        self.labelmethod = LabelFrame(self.frame,text='Method',pady=5, padx=5)
        self.interpolation = DropDownMenu(self.labelmethod,'Interpolation',['None','Animal(s): Nearest', 'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear', 'Body-parts: Quadratic'],'12')
        self.interpolation.setChoices('None')

        # smoothing inside project
        self.smooth_pose_win_lbl = LabelFrame(self.frame, text='Smooth pose-estimation data', pady=5, padx=5)
        self.smooth_dropdown = DropDownMenu(self.smooth_pose_win_lbl, 'Smoothing', ['None', 'Gaussian', 'Savitzky Golay'], '12', com=self.smoothing_selected)
        self.smooth_dropdown.setChoices('None')
        self.smoothing_time_label = Entry_Box(self.smooth_pose_win_lbl, 'Period (ms):', labelwidth='12', width=10)


        # multicsv
        label_multicsvimport = LabelFrame(self.frame, text='Import multiple csv files', pady=5, padx=5)
        self.folder_csv = FolderSelect(label_multicsvimport, 'Folder selected:',title='Select Folder with .csv(s)')
        button_import_csv = Button(label_multicsvimport, text='Import csv to project folder',command= self.importdlctracking_multi,fg='navy')

        # singlecsv
        label_singlecsvimport = LabelFrame(self.frame, text='Import single csv files', pady=5, padx=5)
        self.file_csv = FileSelect(label_singlecsvimport, 'File selected',title='Select a .csv file')
        button_importsinglecsv = Button(label_singlecsvimport, text='Import single csv to project folder',command= self.importdlctracking_single,fg='navy')


        #import videos
        label_importvideo = LabelFrame(label_import, text='Import further videos into project folder', font=("Helvetica",12,'bold'), padx=15,pady=5,fg='black')
        # multi video
        label_multivideoimport = LabelFrame(label_importvideo, text='Import multiple videos', pady=5, padx=5)
        self.multivideofolderpath = FolderSelect(label_multivideoimport, 'Folder path',title='Select Folder with videos')
        self.video_type = Entry_Box(label_multivideoimport, 'File format (i.e., mp4/avi):', '20')
        button_multivideoimport = Button(label_multivideoimport, text='Import multiple videos',command=self.importvideo_multi, fg='black')

        # singlevideo
        label_singlevideoimport = LabelFrame(label_importvideo, text='Import single video', pady=5, padx=5)
        self.singlevideopath = FileSelect(label_singlevideoimport, "Video Path",title='Select a video file')
        button_importsinglevideo = Button(label_singlevideoimport, text='Import a video',command= self.importvideo_single,fg='black')

        #extract frames in project folder
        label_extractframes = LabelFrame(label_import, text='Extract further frames into project folder', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        button_extractframes = Button(label_extractframes, text='Extract frames', command=self.extract_frames_loadini)

        #import frames
        label_importframefolder = LabelFrame(label_import, text='Import frame folders', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        self.frame_folder = FolderSelect(label_importframefolder,'Main frame directory',title='Select the main directory with frame folders')
        button_importframefolder = Button(label_importframefolder,text='Import frames',command = self.importframefolder )

        #import new classifier
        label_newclassifier = LabelFrame(label_import,text='Add new classifier(s)', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        self.classifierentry = Entry_Box(label_newclassifier,'Classifier','8')
        button_addclassifier = Button(label_newclassifier,text='Add classifier',command=lambda:self.addclassifier(self.classifierentry.entry_get))

        #remove classifier
        label_removeclassifier = LabelFrame(label_import,text='Remove existing classifier(s)', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        button_removeclassifier = Button(label_removeclassifier,text='Choose a classifier to remove',command=self.removeclassifiermenu)

        ## archive all csvs
        label_archivecsv =  LabelFrame(label_import,text='Archive processed files', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        archiveentrybox = Entry_Box(label_archivecsv,'Archive folder name', '16')
        button_archivecsv = Button(label_archivecsv,text='Archive',command = lambda: archive_all_csvs(self.projectconfigini,archiveentrybox.entry_get))

        #reverse identity
        label_reverseID = LabelFrame(label_import,text='Reverse Tracking Identity',font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        label_reverse_info = Label(label_reverseID,text='Note: This only works for 2 animals tracking')
        label_git_reverse = Label(label_reverseID, text='[Click here to learn more about the reverse identity process]', cursor='hand2', fg='blue')
        label_git_reverse.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/reverse_annotations.md'))
        reverse_button = Button(label_reverseID,text='Reverse ID',command=self.reverseid)

        #get coordinates
        label_setscale = LabelFrame(tab3,text='Video parameters (fps, resolution, ppx/mm, etc.)', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        self.distanceinmm = Entry_Box(label_setscale, 'Known distance (mm)', '18')
        button_setdistanceinmm = Button(label_setscale, text='Autopopulate table',command=lambda: self.set_distancemm(self.distanceinmm.entry_get))
        button_setscale = Button(label_setscale,text='Set video parameters',command=lambda:video_info_table(self.projectconfigini))

        #ROI
        ### define roi
        self.roi_define = LabelFrame(tab6, text='Define ROI')
        self.rec_entry = Entry_Box(self.roi_define, "# of rectangles", "12")
        self.cir_entry = Entry_Box(self.roi_define, "# of circles", "12")
        self.pol_entry = Entry_Box(self.roi_define, "# of polygons", "12")
        showname = Button(self.roi_define, text="Show Shape Definitions Table", command= lambda:self.table(self.roi_define,self.rec_entry.entry_get,self.cir_entry.entry_get,self.pol_entry.entry_get))

        # organize
        self.roi_define.grid(row=0, sticky=N)
        self.rec_entry.grid(row=1, sticky=W)
        self.cir_entry.grid(row=2, sticky=W)
        self.pol_entry.grid(row=3, sticky=W)
        showname.grid(row=4,pady=10)

        self.new_ROI_frm = LabelFrame(tab6, text='New SimBA ROI interface (experimental)')
        self.start_new_ROI = Button(self.new_ROI_frm, text='Define ROIs', command= lambda: ROI_menu(self.projectconfigini))
        self.delete_all_ROIs = Button(self.new_ROI_frm, text='Delete all ROI definitions', command=lambda: delete_all_ROIs(self.projectconfigini))

        self.new_ROI_frm.grid(row=1, sticky=N, pady=10)
        self.start_new_ROI.grid(row=1, sticky=W, pady=10)
        self.delete_all_ROIs.grid(row=1, column=2, sticky=W, pady=10, padx=10)



        #load roi
        self.loadroi = LabelFrame(tab6,text='Load ROI')
        self.getentrybutton = Button(self.loadroi, text="Load defined ROI table", command=self.loaddefinedroi)

        #organize
        self.loadroi.grid(row=0,column=1,sticky=N)
        self.getentrybutton.grid(row=0)

        ###analyze roi
        self.roi_draw = LabelFrame(tab6, text='Analyze ROI')
        # button
        analyzeROI = Button(self.roi_draw, text='Analyze ROI data',command= lambda:self.roi_settings('Analyze ROI Data','not append'))
        analyzeROITime = Button(self.roi_draw, text='Time bins: Analyze ROI', command=lambda: self.timebin_ml("Time bins: Analyze ROI",text='Analyze'))

        ##organize
        self.roi_draw.grid(row=0, column=2, sticky=N)
        analyzeROI.grid(row=0)
        analyzeROITime.grid(row=1,pady=55)

        ###plot roi
        self.roi_draw1 = LabelFrame(tab6, text='Visualize ROI')

        # button
        visualizeROI = Button(self.roi_draw1, text='Visualize ROI tracking', command=self.visualizeRoiTracking)
        visualizeROIfeature = Button(self.roi_draw1, text='Visualize ROI features', command=self.visualizeROifeatures)
        ##organize
        self.roi_draw1.grid(row=0, column=3, sticky=N)
        visualizeROI.grid(row=0)
        visualizeROIfeature.grid(row=1,pady=10)

        #processmovementinroi (duplicate)
        processmovementdupLabel = LabelFrame(tab6,text='Analyze distances/velocity')
        button_process_movement1 = Button(processmovementdupLabel, text='Analyze distances/velocity',command=lambda: self.roi_settings('Analyze distances/velocity','processmovement'))
        self.hmlvar = IntVar()
        self.hmlvar.set(1)
        button_hmlocation = Button(processmovementdupLabel,text='Create heat maps',command=lambda:self.run_roiAnalysisSettings(Toplevel(),self.hmlvar,'locationheatmap'))

        button_timebins_M = Button(processmovementdupLabel,text='Time bins: Distance/velocity',command = lambda: self.timebin_ml("Time bins: Distance/Velocity"))
        button_lineplot = Button(processmovementdupLabel, text='Generate path plot', command=self.quicklineplot)
        button_analyzeDirection = Button(processmovementdupLabel,text='Analyze directionality between animals',command =lambda:directing_to_other_animals(self.projectconfigini) )
        button_visualizeDirection = Button(processmovementdupLabel,text='Visualize directionality between animals',command=lambda:ROI_directionality_other_animals_visualize(self.projectconfigini))

        #organize
        processmovementdupLabel.grid(row=0,column=4,sticky=N)
        button_process_movement1.grid(row=0)
        button_hmlocation.grid(row=1)
        button_timebins_M.grid(row=2)
        button_lineplot.grid(row=3)
        button_analyzeDirection.grid(row=4)
        button_visualizeDirection.grid(row=5)

        #outlier correction
        label_outliercorrection = LabelFrame(tab4,text='Outlier correction',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        label_link = Label(label_outliercorrection,text='[link to description]',cursor='hand2',font='Verdana 10 underline')
        button_settings_outlier = Button(label_outliercorrection,text='Settings',command = lambda: outlier_settings(self.projectconfigini))
        button_outliercorrection = Button(label_outliercorrection,text='Run outlier correction',command=self.correct_outlier)
        button_skipOC = Button(label_outliercorrection,text='Skip outlier correction (CAUTION)',fg='red', command=lambda:skip_outlier_c(self.projectconfigini))

        label_link.bind("<Button-1>",lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf'))

        #extract features
        label_extractfeatures = LabelFrame(tab5,text='Extract Features',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        button_extractfeatures = Button(label_extractfeatures,text='Extract Features',command = lambda: threading.Thread(target=self.extractfeatures).start())

        def activate(box, *args):
            for entry in args:
                if box.get() == 0:
                    entry.configure(state='disabled')
                elif box.get() == 1:
                    entry.configure(state='normal')

        labelframe_usrdef = LabelFrame(label_extractfeatures)
        self.scriptfile = FileSelect(labelframe_usrdef, 'Script path')
        self.scriptfile.btnFind.configure(state='disabled')
        self.usVar = IntVar()

        userscript = Checkbutton(labelframe_usrdef,text='Apply user defined feature extraction script',variable=self.usVar,command=lambda:activate(self.usVar,self.scriptfile.btnFind))


        #roiappend
        appendDf = Button(label_extractfeatures, text='Append ROI data to features (CAUTION)', fg='red', command=self.appendroisettings)
        appendDf.grid(row=10,pady=10)

        #label Behavior
        label_labelaggression = LabelFrame(tab7,text='Label Behavior',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        button_labelaggression = Button(label_labelaggression, text='Select video (create new video annotation)',command= lambda:choose_folder(self.projectconfigini))
        button_load_labelaggression = Button(label_labelaggression,text='Select video (continue existing video annotation)',command= lambda: load_folder(self.projectconfigini))

        #third party annotation
        label_thirdpartyann = LabelFrame(tab7,text='Import Third-Party behavior labels',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        button_importmars = Button(label_thirdpartyann,text='Import MARS Annotation (select folder with .annot files)',command=self.importMARS)
        button_importboris = Button(label_thirdpartyann,text='Import Boris Annotation (select folder with .csv files)',command = self.importBoris)
        button_importsolomon = Button(label_thirdpartyann,text='Import Solomon Annotation (select folder with .csv files',command = self.importSolomon)
        button_importethovision = Button(label_thirdpartyann, text='Import Ethovision Annotation (select folder with .xls/xlsx files)', command=self.import_ethovision)

        #pseudolabel
        label_pseudo = LabelFrame(tab7,text='Pseudo Labelling',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        pLabel_framedir = FileSelect(label_pseudo,'Video Path',lblwidth='10')
        plabelframe_threshold = LabelFrame(label_pseudo,text='Threshold',pady=5,padx=5)
        plabel_threshold =[0]*len(targetlist)
        count=0
        for i in list(targetlist):
            plabel_threshold[count] = Entry_Box(plabelframe_threshold,str(i),'20')
            plabel_threshold[count].grid(row=count+2,sticky=W)
            count+=1
        pLabel_button = Button(label_pseudo,text='Correct label',command = lambda:semisuperviseLabel(self.projectconfigini,pLabel_framedir.file_path,list(targetlist),plabel_threshold))

        #Advance Label Behavior
        label_adv_label = LabelFrame(tab7,text='Advance Label Behavior',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        label_adv_note = Label(label_adv_label,text='Note that you will have to specify behavior and non-behavior on your own.')
        button_adv_label = Button(label_adv_label, text='Select video (create new video annotation)',command= lambda:adv.choose_folder(self.projectconfigini))
        button_adv_load_label = Button(label_adv_label,text='Select video (continue existing video annotation)',command= lambda: advl.load_folder(self.projectconfigini))

        #train machine model
        label_trainmachinemodel = LabelFrame(tab8,text='Train Machine Models',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        button_trainmachinesettings = Button(label_trainmachinemodel,text='Settings',command=self.trainmachinemodelsetting)
        button_trainmachinemodel = Button(label_trainmachinemodel,text='Train single model from global environment',fg='blue',command = lambda: threading.Thread(target=trainmodel2(self.projectconfigini)).start())
        button_train_multimodel = Button(label_trainmachinemodel, text='Train multiple models, one for each saved settings',fg='green',command = lambda: threading.Thread(target=self.trainmultimodel).start())

        ##Single classifier valid
        label_model_validation = LabelFrame(tab9, text='Validate Model on Single Video', pady=5, padx=5,
                                            font=("Helvetica", 12, 'bold'), fg='black')
        self.csvfile = FileSelect(label_model_validation, 'Select features file',
                                  title='Select .csv file in /project_folder/csv/features_extracted')
        self.modelfile = FileSelect(label_model_validation, 'Select model file  ', title='Select the model (.sav) file')
        button_runvalidmodel = Button(label_model_validation, text='Run Model',command=lambda: validate_model_one_vid_1stStep(self.projectconfigini,self.csvfile.file_path,self.modelfile.file_path))
        button_generateplot = Button(label_model_validation, text="Generate plot", command=self.updateThreshold)
        self.dis_threshold = Entry_Box(label_model_validation, 'Discrimination threshold', '28')
        self.min_behaviorbout = Entry_Box(label_model_validation, 'Minimum behavior bout length (ms)', '28')
        self.generategantt_dropdown = DropDownMenu(label_model_validation, 'Create Gantt plot', ['None', 'Gantt chart: video', 'Gantt chart: final frame only (slightly faster)'], '15')
        self.generategantt_dropdown.setChoices('None')


        #self.ganttvar = IntVar()
        #self.generategantt = Checkbutton(label_model_validation,text='Generate Gantt plot',variable=self.ganttvar)
        button_validate_model = Button(label_model_validation, text='Validate', command=self.validatemodelsinglevid)

        #run machine model
        label_runmachinemodel = LabelFrame(tab9,text='Run Machine Model',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        button_run_rfmodelsettings = Button(label_runmachinemodel,text='Model Settings',command=self.modelselection)
        # self.descrimination_threshold = Entry_Box(label_runmachinemodel,'Discrimination threshold','28')
        # self.shortest_bout = Entry_Box(label_runmachinemodel,'Minimum behavior bout length (ms)','28')
        button_runmachinemodel = Button(label_runmachinemodel,text='Run RF Model',command = self.runrfmodel)

        #kleinberg smoothing
        kleinberg_button = Button(label_runmachinemodel,text='Kleinberg Smoothing',command = self.kleinbergMenu)

        #FSTTC
        fsttc_button = Button(label_runmachinemodel,text='FSTTC',command=self.fsttcmenu)

        # machine results
        label_machineresults = LabelFrame(tab9,text='Analyze Machine Results',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        button_process_datalog = Button(label_machineresults,text='Analyze machine predictions',command =self.analyzedatalog)

        button_process_movement = Button(label_machineresults,text='Analyze distances/velocity',command=lambda:self.roi_settings('Analyze distances/velocity',
                                                                           'processmovement'))
        button_movebins = Button(label_machineresults,text='Time bins: Distance/velocity',command=lambda:self.timebinmove('mov'))
        button_classifierbins = Button(label_machineresults,text='Time bins: Machine predictions',command=lambda:self.timebinmove('classifier'))
        button_classifier_ROI = Button(label_machineresults, text='Classifications by ROI', command=lambda: self.ROI_classifier())



        label_severity = LabelFrame(tab9,text='Analyze Severity',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        self.severityscale = Entry_Box(label_severity,'Severity scale 0 -',15)
        try:
            self.severityTarget = DropDownMenu(label_severity,'Target',targetlist,'15')
        except TypeError:
            print('ERROR: No classifier names detected in your project_config.ini - please make sure you named your classifier(s) when creating your project')
        self.severityTarget.setChoices(targetlist[(config.get('SML settings', 'target_name_' + str(1)))])
        button_process_severity = Button(label_severity,text='Analyze target severity',command=self.analyzseverity)

        #plot sklearn res
        label_plotsklearnr = LabelFrame(tab10,text='Sklearn visualization',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')

        #lbl prob threshold
        lbl_probthreshold = LabelFrame(label_plotsklearnr,text='Body-part probability threshold',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        lbl_thresexplain = Label(lbl_probthreshold,text='Bodyparts below the set threshold won\'t be shown in the output.')
        self.bpthres = Entry_Box(lbl_probthreshold,'Body-part probability threshold','32')
        bpthresbutton = Button(lbl_probthreshold,text='Save threshold',command= self.savethres)

        #set bp threshold
        try:
            thres = config.get('threshold_settings', 'bp_threshold_sklearn')
            self.bpthres.entry_set(str(thres))
        except:
            self.bpthres.entry_set(0.0)

        #all videos
        label_skv_all = LabelFrame(label_plotsklearnr,text='Apply to all videos',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        self.videovar = IntVar()
        self.genframevar = IntVar()
        videocheck = Checkbutton(label_skv_all,text='Generate video',variable=self.videovar)
        framecheck = Checkbutton(label_skv_all,text='Generate frame',variable=self.genframevar)

        button_plotsklearnr = Button(label_skv_all,text='Visualize classification results',command =self.plotsklearn_result)
        #single video
        label_skv_single = LabelFrame(label_plotsklearnr, text='Apply to single video', font=("Helvetica", 12, 'bold'),pady=5, padx=5, fg='black')
        videodir_ = os.path.join(os.path.dirname(self.projectconfigini), 'videos')
        vid_list_ =[]
        for i in os.listdir(videodir_):
            if i.endswith(('.avi','.mp4','.mov','flv','m4v')):
                vid_list_.append(i)
        if not vid_list_:
            vid_list_.append('No videos found')
        self.video_entry = DropDownMenu(label_skv_single, 'Select video', vid_list_, '15')
        self.video_entry.setChoices(vid_list_[0])
        self.videovar2 = IntVar()
        self.genframevar2 = IntVar()
        videocheck2 = Checkbutton(label_skv_single, text='Generate video', variable=self.videovar2)
        framecheck2 = Checkbutton(label_skv_single, text='Generate frame', variable=self.genframevar2)
        button_plotsklearnr2 = Button(label_skv_single, text='Visualize classification results',command=lambda: plotsklearnresultsingle(self.projectconfigini,self.videovar2.get(),self.genframevar2.get(),self.video_entry.getChoices()))

        #plotpathing
        label_plotall = LabelFrame(tab10,text='Visualizations',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        #ganttplot
        label_ganttplot = LabelFrame(label_plotall,text='Gantt plot',pady=5,padx=5)
        button_ganttplot = Button(label_ganttplot,text='Generate gantt plot',command= lambda: ganntplot_config(self.projectconfigini))

        #dataplot
        label_dataplot = LabelFrame(label_plotall, text='Data plot', pady=5, padx=5)
        if pose_config_setting == 'user_defined':
            self.SelectedBp = DropDownMenu(label_dataplot, 'Select body part', bp_set, '15')
            self.SelectedBp.setChoices((bp_set)[0])
        button_dataplot = Button(label_dataplot, text='Generate data plot', command=self.plotdataplot)

        #path plot
        label_pathplot = LabelFrame(label_plotall,text='Path plot',pady=5,padx=5)
        self.Deque_points = Entry_Box(label_pathplot,'Max lines','15')
        self.severity_brackets = Entry_Box(label_pathplot,'Severity Scale: 0 - ','15')
        self.noofAnimal = DropDownMenu(label_pathplot,'Number of animals',[1,2],'15')
        self.noofAnimal.setChoices(1)
        confirmAnimals = Button(label_pathplot,text='Confirm',command=lambda:self.tracknoofanimal(label_pathplot,bp_set))
        self.plotsvvar = IntVar()
        checkboxplotseverity = Checkbutton(label_pathplot,text='plot_severity',variable=self.plotsvvar)
        self.severityTargetpp = DropDownMenu(label_pathplot, 'Target', targetlist, '15')
        self.severityTargetpp.setChoices(targetlist[(config.get('SML settings', 'target_name_' + str(1)))])
        button_pathplot = Button(label_pathplot,text='Generate Path plot',command=self.pathplotcommand)

        CreateToolTip(self.Deque_points,'Maximum number of path lines in deque list')
        # CreateToolTip(self.Bodyparts, 'If golden aggression config: Nose, Left ear, Right ear, Centroid, Left lateral, Right lateral, Tail base, Tail end')
        CreateToolTip(self.severity_brackets,'Set number of brackets to severity measures')

        #distanceplot
        label_distanceplot = LabelFrame(label_plotall,text='Distance plot',pady=5,padx=5)
        self.poi1 = DropDownMenu(label_distanceplot,'Body part 1',bp_set,'15')
        self.poi2 = DropDownMenu(label_distanceplot,'Body part 2',bp_set,'15')
        #set choice
        self.poi1.setChoices((bp_set)[0])
        self.poi2.setChoices((bp_set)[len(bp_set)//2])
        button_distanceplot= Button(label_distanceplot,text='Generate Distance plot',command=self.distanceplotcommand)

        CreateToolTip(self.poi1,'The bodyparts from config yaml. eg: Ear_left_1,Ear_right_1,Nose_1,Center_1,Lateral_left_1,Lateral_right_1,Tail_base_1,Tail_end_1,Ear_left_2,Ear_right_2,Nose_2,Center_2,Lateral_left_2,Lateral_right_2,Tail_base_2,Tail_end_2')

        #Heatplot
        label_heatmap = LabelFrame(label_plotall, text='Heatmap', pady=5, padx=5)
        self.BinSize = Entry_Box(label_heatmap, 'Bin size (mm)', '15')
        self.MaxScale = Entry_Box(label_heatmap, 'max', '15')

        hmchoices = {'viridis','plasma','inferno','magma','jet','gnuplot2'}
        self.hmMenu = DropDownMenu(label_heatmap,'Color Palette',hmchoices,'15')
        self.hmMenu.setChoices('jet')

        #get target called on top
        self.targetMenu = DropDownMenu(label_heatmap,'Target',targetlist,'15')
        self.targetMenu.setChoices(targetlist[(config.get('SML settings','target_name_'+str(1)))])

        #bodyparts
        bpOptionListofList = define_bp_drop_down(configini)
        bpoptions = [val for sublist in bpOptionListofList for val in sublist]
        try:
            self.bp1 = DropDownMenu(label_heatmap,'Bodypart',bpoptions,'15')
            self.bp1.setChoices(bpoptions[0])
        except TypeError:
            self.bp1 = DropDownMenu(label_heatmap, 'Bodypart', bp_set, '15')
            self.bp1.setChoices(bp_set[0])
        self.intimgvar = IntVar()
        lstimg = Checkbutton(label_heatmap,text='Save last image only (if unticked heatmap videos are created)',variable=self.intimgvar)
        button_heatmap = Button(label_heatmap, text='Generate heatmap', command=self.heatmapcommand)

        #plot threshold
        label_plotThreshold = LabelFrame(label_plotall,text='Plot Threshold',pady=5,padx=5)
        self.behaviorMenu = DropDownMenu(label_plotThreshold,'Target',targetlist,'15')
        self.behaviorMenu.setChoices(targetlist[(config.get('SML settings','target_name_'+str(1)))])
        plotThresholdButton = Button(label_plotThreshold,text='Plot threshold',command=lambda:plot_threshold(self.projectconfigini,self.behaviorMenu.getChoices()))


        #Merge frames
        label_mergeframes = LabelFrame(tab10, text='Merge frames', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='black')
        # use for loop to create intvar
        mergeFramesvar = []
        for i in range(7):
            mergeFramesvar.append(IntVar())
        # use loop to create checkbox?
        mfCheckbox = [0] * 7
        mfTitlebox = ['Sklearn', 'Gantt', 'Path', "'Live' data", 'Distance','Probability']
        for i in range(6):
            mfCheckbox[i] = Checkbutton(label_mergeframes, text=mfTitlebox[i], variable=mergeFramesvar[i])
            mfCheckbox[i].grid(row=i, sticky=W)

        button_mergeframe = Button(label_mergeframes,text='Merge frames',command= lambda:self.mergeframesofplot(mergeFramesvar))

        #Plotly
        plotlyInterface = LabelFrame(tab10, text= 'Plotly / Dash', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        plotlyInterfaceTitles = ['Sklearn results', 'Time bin analyses', 'Probabilities', 'Severity analysis']
        toIncludeVar = []
        for i in range(len(plotlyInterfaceTitles)+1):
            toIncludeVar.append(IntVar())
        plotlyCheckbox = [0] * (len(plotlyInterfaceTitles)+1)
        for i in range(len(plotlyInterfaceTitles)):
            plotlyCheckbox[i] = Checkbutton(plotlyInterface, text=plotlyInterfaceTitles[i], variable=toIncludeVar[i])
            plotlyCheckbox[i].grid(row=i, sticky=W)

        button_save_plotly_file = Button(plotlyInterface, text='Save SimBA / Plotly dataset', command=lambda: self.generateSimBPlotlyFile(toIncludeVar))
        self.plotly_file = FileSelect(plotlyInterface, 'SimBA Dashboard file (H5)', title='Select SimBA/Plotly dataset (h5)')
        self.groups_file = FileSelect(plotlyInterface, 'SimBA Groups file (CSV)', title='Select groups file (csv')
        button_open_plotly_interface = Button(plotlyInterface, text='Open SimBA / Plotly dataset', fg='black', command=lambda: [self.open_plotly_interface('http://127.0.0.1:8050')])


        ## classifier validation
        label_classifier_validation = LabelFrame(tab11, text='Classifier Validation', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.seconds = Entry_Box(label_classifier_validation,'Seconds','8')
        self.cvTarget = DropDownMenu(label_classifier_validation,'Target',targetlist,'15')
        self.cvTarget.setChoices(targetlist[(config.get('SML settings', 'target_name_' + str(1)))])
        button_validate_classifier = Button(label_classifier_validation,text='Validate',command =self.classifiervalidation)

        # ## Live classification
        # label_deeplabstream = LabelFrame(tab12, text='DeepLabStream', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        # self.label_settingsini = FolderSelect(label_deeplabstream, 'Select DLS folder',title='Select DeepLabStream folder')
        # button_dlsconfirm = Button(label_deeplabstream,text='Confirm', command = self.DLSsettings)

        #addons
        lbl_addon = LabelFrame(tab12,text='SimBA Expansions',pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        button_bel = Button(lbl_addon,text='Pup retrieval - Analysis Protocol 1',command = self.pupMenu)

        button_unsupervised = Button(lbl_addon,text='Unsupervised',command = lambda :unsupervisedInterface(self.projectconfigini))

        #organize
        label_import.grid(row=0,column=0,sticky=W,pady=5)
        self.label_import_csv.grid(row=2, sticky=N + W, pady=5)
        filetype.grid(row=0,sticky=W)
        self.frame.grid(row=3,sticky=W)

        #method
        self.labelmethod.grid(row=0,sticky=W)
        self.interpolation.grid(row=0,sticky=W)
        self.smooth_pose_win_lbl.grid(row=1, sticky=W)
        self.smooth_dropdown.grid(row=0, sticky=W)
        label_multicsvimport.grid(row=2, sticky=W)
        self.folder_csv.grid(row=0, sticky=W)
        button_import_csv.grid(row=1, sticky=W)
        label_singlecsvimport.grid(row=3, sticky=W)
        self.file_csv.grid(row=0, sticky=W)
        button_importsinglecsv.grid(row=1, sticky=W)

        label_importvideo.grid(row=0,column=0, sticky=N+W, pady=5,padx=5,rowspan=2)
        label_multivideoimport.grid(row=0, sticky=W)
        self.multivideofolderpath.grid(row=0, sticky=W)
        self.video_type.grid(row=1, sticky=W)
        button_multivideoimport.grid(row=2, sticky=W)
        label_singlevideoimport.grid(row=1, sticky=W)
        self.singlevideopath.grid(row=0, sticky=W)
        button_importsinglevideo.grid(row=1, sticky=W)

        label_extractframes.grid(row=0,column=1,sticky=N+W,pady=5,padx=5)
        button_extractframes.grid(row=0,sticky=W)

        label_importframefolder.grid(row=1,column=1,sticky=N+W,pady=5,padx=5,rowspan=2)
        self.frame_folder.grid(row=0,sticky=W)
        button_importframefolder.grid(row=1,sticky=W)
        #
        label_reverseID.grid(row=2,column=1,sticky=N+W,pady=5,padx=5)
        label_reverse_info.grid(row=0,sticky=W)
        label_git_reverse.grid(row=1,sticky=W)
        reverse_button.grid(row=2,sticky=W,pady=5)

        label_newclassifier.grid(row=0,column=2,sticky=N+W,pady=5,padx=5)
        self.classifierentry.grid(row=0,column=0,sticky=W)
        button_addclassifier.grid(row=1,column=0,sticky=W)

        label_removeclassifier.grid(row=1,column=2,sticky=N+W,pady=5,padx=5)
        button_removeclassifier.grid(row=0)

        label_archivecsv.grid(row=2,column=2,sticky=W,pady=5,padx=5)
        archiveentrybox.grid(row=0)
        button_archivecsv.grid(row=1,pady=10)


        label_setscale.grid(row=2,sticky=W,pady=5,padx=5)
        self.distanceinmm.grid(row=0,column=0,sticky=W)
        button_setdistanceinmm.grid(row=0,column=1)
        button_setscale.grid(row=1,column=0,sticky=W)

        label_outliercorrection.grid(row=0,sticky=W)
        label_link.grid(row=0,sticky=W)
        button_settings_outlier.grid(row=1,sticky=W)
        button_outliercorrection.grid(row=3,sticky=W)
        button_skipOC.grid(row=4,sticky=W,pady=5)

        label_extractfeatures.grid(row=4,sticky=W)
        button_extractfeatures.grid(row=0,sticky=W)
        labelframe_usrdef.grid(row=1,sticky=W,pady=5)
        userscript.grid(row=1,sticky=W)
        self.scriptfile.grid(row=2,sticky=W)

        label_labelaggression.grid(row=5,sticky=W)
        button_labelaggression.grid(row=0,sticky=W)
        button_load_labelaggression.grid(row=1,sticky=W,pady=10)

        label_pseudo.grid(row=6,sticky=W,pady=10)
        pLabel_framedir.grid(row=0,sticky=W)
        plabelframe_threshold.grid(row=2,sticky=W)
        pLabel_button.grid(row=3,sticky=W)

        label_thirdpartyann.grid(row=7, sticky=W)
        button_importmars.grid(row=0, sticky=W)
        button_importboris.grid(row=1,sticky=W,pady=10)
        button_importsolomon.grid(row=2,sticky=W,pady=10)
        button_importethovision.grid(row=3,sticky=W,pady=10)

        label_adv_label.grid(row=8,sticky=W)
        label_adv_note.grid(row=0,pady=10,sticky=W)
        button_adv_label.grid(row=1, sticky=W)
        button_adv_load_label.grid(row=2,sticky=W,pady=10)

        label_trainmachinemodel.grid(row=6,sticky=W)
        button_trainmachinesettings.grid(row=0,column=0,sticky=W,padx=5)
        button_trainmachinemodel.grid(row=0,column=1,sticky=W,padx=5)
        button_train_multimodel.grid(row=0,column=2,sticky=W,padx=5)

        label_model_validation.grid(row=7, sticky=W, pady=5)
        self.csvfile.grid(row=0, sticky=W)
        self.modelfile.grid(row=1, sticky=W)
        button_runvalidmodel.grid(row=2, sticky=W)
        button_generateplot.grid(row=3, sticky=W)
        self.dis_threshold.grid(row=4, sticky=W)
        self.min_behaviorbout.grid(row=5, sticky=W)
        self.generategantt_dropdown.grid(row=6,sticky=W)
        button_validate_model.grid(row=7, sticky=W)

        label_runmachinemodel.grid(row=8,sticky=W,pady=5)
        button_run_rfmodelsettings.grid(row=0,sticky=W)
        # self.descrimination_threshold.grid(row=1,sticky=W)
        # button_set_d_t.grid(row=1,column=1,sticky=W)
        # self.shortest_bout.grid(row=2,column=0,sticky=W)
        # button_set_shortbout.grid(row=2,column=1,sticky=W)
        button_runmachinemodel.grid(row=1,sticky=W,pady=5)
        kleinberg_button.grid(row=2,sticky=W,pady=10)
        fsttc_button.grid(row=3,sticky=W,pady=10)

        label_machineresults.grid(row=9,sticky=W,pady=5)
        button_process_datalog.grid(row=2,column=0,sticky=W,padx=3)
        button_process_movement.grid(row=2,column=1,sticky=W,padx=3)
        button_movebins.grid(row=3,column=1,sticky=W,padx=3)
        button_classifierbins.grid(row=3,column=0,sticky=W,padx=3)
        button_classifier_ROI.grid(row=4, column=0, sticky=W, padx=3)

        #severity
        label_severity.grid(row=10,sticky=W,pady=5)
        self.severityscale.grid(row=0,sticky=W)
        self.severityTarget.grid(row=1,sticky=W)
        button_process_severity.grid(row=2,sticky=W,pady=8)

        label_plotsklearnr.grid(row=11,column=0,sticky=W+N,padx=5)
        lbl_probthreshold.grid(row=0,sticky=W,padx=5,pady=10)
        lbl_thresexplain.grid(row=0,sticky=W)
        self.bpthres.grid(row=1,sticky=E)
        bpthresbutton.grid(row=2,padx=5,pady=10)
        label_skv_all.grid(row=1,sticky=W,padx=5,pady=10)
        videocheck.grid(row=0,sticky=W)
        framecheck.grid(row=1,sticky=W)
        button_plotsklearnr.grid(row=2,sticky=W)
        label_skv_single.grid(row=2,sticky=W,pady=10,padx=5)
        self.video_entry.grid(row=0,sticky=W)
        videocheck2.grid(row=1,sticky=W)
        framecheck2.grid(row=2,sticky=W)
        button_plotsklearnr2.grid(row=3,sticky=W)

        label_plotall.grid(row=11,column=1,sticky=W+N,padx=5)
        #gantt
        label_ganttplot.grid(row=0,sticky=W)
        button_ganttplot.grid(row=0,sticky=W)
        #data
        label_dataplot.grid(row=1,sticky=W)
        if pose_config_setting == 'user_defined':
            self.SelectedBp.grid(row=1, sticky=W)
        button_dataplot.grid(row=2, sticky=W)
        #path
        label_pathplot.grid(row=2,sticky=W)
        self.Deque_points.grid(row=0,sticky=W)
        self.severity_brackets.grid(row=2,sticky=W)
        self.noofAnimal.grid(row=3,sticky=W)
        confirmAnimals.grid(row=3,column=1,sticky=W)
        checkboxplotseverity.grid(row=7,sticky=W)
        self.severityTargetpp.grid(row=8,sticky=W)
        button_pathplot.grid(row=9,sticky=W)
        #distance
        label_distanceplot.grid(row=3,sticky=W)
        self.poi1.grid(row=1,sticky=W)
        self.poi2.grid(row=2,sticky=W)
        button_distanceplot.grid(row=3,sticky=W)
        #heat
        label_heatmap.grid(row=4, sticky=W)
        self.BinSize.grid(row=0, sticky=W)
        self.MaxScale.grid(row=1, sticky=W)
        self.hmMenu.grid(row=3,sticky=W)
        self.targetMenu.grid(row=4,sticky=W)
        self.bp1.grid(row=5,sticky=W)
        lstimg.grid(row=6,sticky=W)
        button_heatmap.grid(row=7, sticky=W)

        #threshold
        label_plotThreshold.grid(row=5, sticky=W)
        self.behaviorMenu.grid(row=0, sticky=W)
        self.behaviorMenu.grid(row=1, sticky=W)
        plotThresholdButton.grid(row=2, sticky=W)

        label_mergeframes.grid(row=11,column=2,sticky=W+N,padx=5)
        button_mergeframe.grid(row=10,sticky=W)

        plotlyInterface.grid(row=11, column=3, sticky=W + N, padx=5)
        button_save_plotly_file.grid(row=10, sticky=W)
        self.plotly_file.grid(row=11, sticky=W)
        self.groups_file.grid(row=12, sticky=W)
        button_open_plotly_interface.grid(row=13, sticky=W)

        label_classifier_validation.grid(row=14,sticky=W)
        self.seconds.grid(row=0,sticky=W)
        self.cvTarget.grid(row=1,sticky=W)
        button_validate_classifier.grid(row=2,sticky=W)

        lbl_addon.grid(row=15,sticky=W)
        button_bel.grid(row=0,sticky=W)
        button_unsupervised.grid(row=1,sticky=W,pady=10)

    def smoothing_selected(self, choice):
        if choice == 'None':
            self.smoothing_time_label.grid_forget()
        if (choice == 'Gaussian') or (choice == 'Savitzky Golay'):
            self.smoothing_time_label.grid(row=0, column=1, sticky=E)

    def reverseid(self):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        noanimal = int(config.get('General settings','animal_no'))

        if noanimal ==2:
            reverse_dlc_input_files(self.projectconfigini)
        else:
            print('This only works if you have exactly 2 animals in your tracking data and video.')

    def savethres(self):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        config.set('threshold_settings', 'bp_threshold_sklearn',str(self.bpthres.entry_get))
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

        print('Threshold saved >>', str(self.bpthres.entry_get))


    def pupMenu(self):
        #top lvl
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        try:
            multi_animal_IDs = config.get('Multi animal IDs', 'id_list').split(',')
        except (NoSectionError, NoOptionError):
            multi_animal_IDs = ['1_mother', ' 2_pup']
        if len(multi_animal_IDs) != 2:
            multi_animal_IDs = ['1_mother', ' 2_pup']


        puptoplvl = Toplevel()
        puptoplvl.minsize(400,320)
        puptoplvl.wm_title('Pup retrieval - Analysis Protocol 1')

        #lblframe for input
        lbl_pup = LabelFrame(puptoplvl,text='Pup retrieval - Analysis Protocol 1', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        prob_pup = Entry_Box(lbl_pup,'Tracking probability (pup)','20')
        prob_pup.entry_set(0.025)
        prob_mom = Entry_Box(lbl_pup,'Tracking probability (dam)','20')
        prob_mom.entry_set(0.5)
        dist_start = Entry_Box(lbl_pup, 'Start distance criterion (mm)', '20')
        dist_start.entry_set(80)
        carry_frames_seconds = Entry_Box(lbl_pup, 'Carry frames (s)', '20')
        carry_frames_seconds.entry_set(3)
        corenest_name = Entry_Box(lbl_pup, 'Core-nest name', '20')
        corenest_name.entry_set('corenest')
        nest_name = Entry_Box(lbl_pup, 'Nest name', '20')
        nest_name.entry_set('nest')
        dam_name = Entry_Box(lbl_pup, 'Dam name', '20')
        dam_name.entry_set(multi_animal_IDs[0])
        pup_name = Entry_Box(lbl_pup, 'Pup name', '20')
        pup_name.entry_set(multi_animal_IDs[1])
        smooth_function = Entry_Box(lbl_pup, 'Smooth function', '20')
        smooth_function.entry_set('gaussian')
        smooth_factor = Entry_Box(lbl_pup, 'Smooth factor', '20')
        smooth_factor.entry_set(5)
        max_time = Entry_Box(lbl_pup, 'Max time (s)', '20')
        max_time.entry_set(90)
        carry_classifier_name = Entry_Box(lbl_pup, 'Carry classifier name', '20')
        carry_classifier_name.entry_set('carry')
        approach_classifier_name = Entry_Box(lbl_pup, 'Approach classifier name', '20')
        approach_classifier_name.entry_set('approach')
        dig_classifier_name = Entry_Box(lbl_pup, 'Dig classifier name', '20')
        dig_classifier_name.entry_set('doggy')

        #button
        button_run = Button(puptoplvl,text='RUN',font=("Helvetica",12,'bold'),fg='red',command= lambda: pup_retrieval_1(self.projectconfigini, float(prob_pup.entry_get), float(prob_mom.entry_get), float(dist_start.entry_get), float(carry_frames_seconds.entry_get), float(smooth_factor.entry_get), corenest_name.entry_get,  nest_name.entry_get, dam_name.entry_get, pup_name.entry_get, smooth_function.entry_get, int(max_time), carry_classifier_name.entry_get, approach_classifier_name.entry_get, dig_classifier_name.entry_get))

        #organize
        lbl_pup.grid(row=0,sticky=W)
        prob_pup.grid(row=0,sticky=W)
        prob_mom.grid(row=1,sticky=W)
        dist_start.grid(row=2,sticky=W)
        carry_frames_seconds.grid(row=4,sticky=W)
        corenest_name.grid(row=5,sticky=W)
        nest_name.grid(row=6,sticky=W)
        dam_name.grid(row=7,sticky=W)
        pup_name.grid(row=8,sticky=W)
        smooth_function.grid(row=9, sticky=W)
        smooth_factor.grid(row=10, sticky=W)
        max_time.grid(row=11, sticky=W)
        carry_classifier_name.grid(row=12,sticky=W)
        approach_classifier_name.grid(row=13,sticky=W)
        dig_classifier_name.grid(row=14, sticky=W)

        button_run.grid(row=1,padx=10,pady=10)

    def fsttcmenu(self):
        #get data
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        # get current no of target
        notarget = config.getint('SML settings', 'no_targets')
        targetlist = [0] * notarget
        varlist = [0] * notarget
        for i in range(notarget):
            varlist[i] = IntVar()
            targetlist[i] = (config.get('SML settings', 'target_name_' + str(i + 1)))

        #toplvl
        fstoplvl = Toplevel()
        fstoplvl.minsize(400,320)
        fstoplvl.wm_title('Calculate forward-spike time tiling coefficents')

        #git
        lbl_git_fsttc = Label(fstoplvl, text='[Click here to learn about FSTTC]',cursor='hand2', fg='blue')
        lbl_git_fsttc.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md'))

        #fsttc settings
        lbl_fsttc_settings = LabelFrame(fstoplvl,text='FSTTC Settings', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        cvar = IntVar()
        cr8_graph = Checkbutton(lbl_fsttc_settings,text='Create graph',variable=cvar)
        time_delta = Entry_Box(lbl_fsttc_settings,'Time Delta','10')
        lbl_behavior = LabelFrame(lbl_fsttc_settings,text="Behaviors")
        #behaviors
        behaviorlist = [0]*notarget
        for i in range(len(targetlist)):
            behaviorlist[i] = Checkbutton(lbl_behavior,text=str(targetlist[i]),variable=varlist[i])
            behaviorlist[i].grid(row=str(i),sticky=W)

        ftsccbutton = Button(fstoplvl,text='Calculate FSTTC',command=lambda:self.runfsttc(time_delta.entry_get,cvar.get(),targetlist,varlist))

        #organize
        lbl_git_fsttc.grid(row=0,sticky=W,pady=5)
        lbl_fsttc_settings.grid(row=1,sticky=W)
        cr8_graph.grid(row=0,sticky=W)
        time_delta.grid(row=1,sticky=W,pady=5)
        lbl_behavior.grid(row=2,sticky=W,pady=5)

        ftsccbutton.grid(row=3,pady=10)

    def runfsttc(self,timedelta,creategraph,targetlist,varlist):
        if creategraph == 1:
            creategraph = True
        else:
            creategraph = False

        target = []

        for i in range(len(varlist)):
            if varlist[i].get()==1:
                target.append(targetlist[i])

        FSTCC_performer = FSTCC_perform(self.projectconfigini, timedelta, target, creategraph)
        FSTCC_performer.calculate_sequence_data()
        FSTCC_performer.calculate_FSTCC()
        FSTCC_performer.save_results()
        FSTCC_performer.plot_results()

    def kleinbergMenu(self):
        kleintoplvl = Toplevel()
        kleintoplvl.minsize(400,320)
        kleintoplvl.wm_title('Apply Kleinberg behavior classification smoothing')

        #git
        label_git_kleinberg = Label(kleintoplvl, text='[Click here to learn about Kleinberg Smoother]',cursor='hand2', fg='blue')
        label_git_kleinberg.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/kleinberg_filter.md'))

        #kleinberg settings
        lbl_kleinberg_settings = LabelFrame(kleintoplvl,text='Kleinberg Settings', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.k_sigma = Entry_Box(lbl_kleinberg_settings,'Sigma','10')
        self.k_sigma.entry_set('2')
        self.k_gamma = Entry_Box(lbl_kleinberg_settings,'Gamma','10')
        self.k_gamma.entry_set('0.3')
        self.k_hierarchy = Entry_Box(lbl_kleinberg_settings,'Hierarchy','10')
        self.k_hierarchy.entry_set('1')


        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        # get current no of target
        notarget = config.getint('SML settings', 'no_targets')
        targetlist = [0]*notarget
        varlist = [0]*notarget
        for i in range(notarget):
            varlist[i] = IntVar()
            targetlist[i] = (config.get('SML settings', 'target_name_' + str(i + 1)))

        #make table for classifier to apply filter
        tablelabel = LabelFrame(kleintoplvl,text='Choose classifier(s) to apply Kleinberg smoothing')
        for i in range(notarget):
            Checkbutton(tablelabel, text=str(targetlist[i]), variable=varlist[i]).grid(row=i, sticky=W)

        run_kleinberg_button = Button(kleintoplvl,text='Apply Kleinberg Smoother',command= lambda: self.runkleinberg(targetlist,varlist))

        #organize
        label_git_kleinberg.grid(row=0,sticky=W)
        lbl_kleinberg_settings.grid(row=1,sticky=W,padx=10)
        self.k_sigma.grid(row=0,sticky=W)
        self.k_gamma.grid(row=1,sticky=W)
        self.k_hierarchy.grid(row=2,sticky=W)

        tablelabel.grid(row=2,pady=10,padx=10)

        run_kleinberg_button.grid(row=3)

    def runkleinberg(self,targetlist,varlist):
        classifier_list =[]
        for i in range(len(varlist)):
            if varlist[i].get() == 1:
                classifier_list.append(targetlist[i])

        print(classifier_list,'selected')
        run_kleinberg(self.projectconfigini,classifier_list,self.k_sigma.entry_get,self.k_gamma.entry_get,self.k_hierarchy.entry_get)
        print('Kleinberg filter applied, process completed.')


    def DLSsettings(self):

        config = ConfigParser()
        configFile = os.path.join(str(self.label_settingsini.folder_path),'settings.ini')
        config.read(configFile)

        # get current settings
        #streaming
        resolution = config.get('Streaming', 'RESOLUTION')
        framerate = config.get('Streaming', 'FRAMERATE')
        streams = config.get('Streaming','STREAMS')
        outputdir = config.get('Streaming','OUTPUT_DIRECTORY')
        multi_devices = config.get('Streaming','MULTIPLE_DEVICES')
        video_source = config.get('Streaming','VIDEO_SOURCE')
        video_path = config.get('Streaming','VIDEO_PATH')
        video = config.get('Streaming','VIDEO')
        animal_no = config.get('Streaming','ANIMALS_NUMBER')
        port = config.get('Streaming','PORT')
        ipwebcam = config.get('Streaming','IPWEBCAM')
        #deeplabcut
        dlcpath = config.get('DeepLabCut','DLC_PATH')
        model = config.get('DeepLabCut','MODEL')
        #classification
        classifier_path = config.get('Classification','PATH_TO_CLASSIFIER')
        allbp = config.get('Classification','ALL_BODYPARTS')
        ppm = config.get('Classification','PIXPERMM')
        threshold = config.get('Classification','THRESHOLD')
        poolsize = config.get('Classification','POOL_SIZE')
        #experiment
        exp_no = config.get('Experiment','EXP_NUMBER')
        record_exp = config.get('Experiment','RECORD_EXP')
        #deeplabstream
        dls_path = config.get('DeepLabStream','DLS_PATH')

        #top level design
        dlstoplevel = Toplevel()
        dlstoplevel.minsize(300, 400)
        dlstoplevel.wm_title('DeepLabStream Settings')
        #streaming
        labelStreamimg = LabelFrame(dlstoplevel,text='Streaming', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.e_reso = Entry_Box(labelStreamimg,'Resolution','15')
        self.e_fps = Entry_Box(labelStreamimg,'Frame rate','15')
        self.e_streams = Entry_Box(labelStreamimg, 'Streams', '15')
        self.e_outputdir = FolderSelect(labelStreamimg,'Output directory',title='Select output directory',lblwidth='15')
        self.e_multiDevice = Entry_Box(labelStreamimg, 'Multiple devices', '15')
        self.e_videosource = Entry_Box(labelStreamimg, 'Video source', '15')
        self.e_videopath = FileSelect(labelStreamimg,'Video path',title='Select video',lblwidth='15')
        self.e_video = Entry_Box(labelStreamimg, 'Video', '15')
        self.e_animalNo = Entry_Box(labelStreamimg, 'Animals #', '15')
        self.e_port = Entry_Box(labelStreamimg, 'Port', '15')
        self.e_ipwebcam = Entry_Box(labelStreamimg, 'IP webcam', '15')
        #deeplabcut
        label_dlc = LabelFrame(dlstoplevel,text='DeepLabCut', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.e_dlcpath = FolderSelect(label_dlc,'DLC path',title='Select deeplabcut package in python site packages',lblwidth='15')
        self.e_model = FolderSelect(label_dlc,'Model folder path',title='Select DeepLabCut tracking model folder path',lblwidth='15')
        #classification
        label_classification = LabelFrame(dlstoplevel,text='Classification', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.e_classifierpath = FileSelect(label_classification,'Classifier path',title='Select Simba Classifier (.sav) file',lblwidth='15')
        self.e_allBP = Entry_Box(label_classification,'All bodyparts',labelwidth='15')
        self.e_ppm = Entry_Box(label_classification,'Pixel / mm','15')
        self.e_threshold = Entry_Box(label_classification,'Threshold','15')
        self.e_poolsize = Entry_Box(label_classification,'Pool size','15')
        #experiment
        label_exp = LabelFrame(dlstoplevel,text='Experiment', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.e_expNo = Entry_Box(label_exp,'Experiment #','15')
        self.e_recordExp = Entry_Box(label_exp,'Record Experiment','15')
        # deeplabstream
        label_dls = LabelFrame(dlstoplevel,text='DeepLabStream', pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        self.e_dls_path = FolderSelect(label_dls,'DLS path',title='Please select DeepLabStream folder in python packages')

        #feed in config parser
        #streaming
        self.e_reso.entPath.insert(0,resolution)
        self.e_fps.entPath.insert(0,framerate)
        self.e_streams.entPath.insert(0,streams)
        self.e_outputdir.folderPath.set(outputdir)
        self.e_multiDevice.entPath.insert(0,multi_devices)
        self.e_videosource.entPath.insert(0,video_source)
        self.e_videopath.filePath.set(video_path)
        self.e_video.entPath.insert(0,video)
        self.e_animalNo.entPath.insert(0,animal_no)
        self.e_port.entPath.insert(0,port)
        self.e_ipwebcam.entPath.insert(0,ipwebcam)
        #deeplabcut
        self.e_dlcpath.folderPath.set(dlcpath)
        self.e_model.folderPath.set(model)
        #classification
        self.e_classifierpath.filePath.set(classifier_path)
        self.e_allBP.entPath.insert(0,allbp)
        self.e_ppm.entPath.insert(0,ppm)
        self.e_threshold.entPath.insert(0,threshold)
        self.e_poolsize.entPath.insert(0,poolsize)
        #experiment
        self.e_expNo.entPath.insert(0,exp_no)
        self.e_recordExp.entPath.insert(0,record_exp)
        #dls
        self.e_dls_path.folderPath.set(dls_path)

        #organize
        #streaming
        labelStreamimg.grid(row=0,sticky=W)
        self.e_reso.grid(row=0,sticky=W)
        self.e_fps.grid(row=1,sticky=W)
        self.e_streams.grid(row=2,sticky=W)
        self.e_outputdir.grid(row=3,sticky=W)
        self.e_multiDevice.grid(row=4,sticky=W)
        self.e_videosource.grid(row=5,sticky=W)
        self.e_videopath.grid(row=6,sticky=W)
        self.e_video.grid(row=7,sticky=W)
        self.e_animalNo.grid(row=8,sticky=W)
        self.e_port.grid(row=9,sticky=W)
        self.e_ipwebcam.grid(row=10,sticky=W)
        #deeplabcut
        label_dlc.grid(row=1,sticky=W)
        self.e_dlcpath.grid(row=0,sticky=W)
        self.e_model.grid(row=1,sticky=W)
        #classification
        label_classification.grid(row=2,sticky=W)
        self.e_classifierpath.grid(row=0,sticky=W)
        self.e_allBP.grid(row=1,sticky=W)
        self.e_ppm.grid(row=2,sticky=W)
        self.e_threshold.grid(row=3,sticky=W)
        self.e_poolsize.grid(row=4,sticky=W)
        #experiment
        label_exp.grid(row=3,sticky=W)
        self.e_expNo.grid(row=0,sticky=W)
        self.e_recordExp.grid(row=1,sticky=W)
        #dls
        label_dls.grid(row=4,sticky=W)
        self.e_dls_path.grid(row=0,sticky=W)

        #confirm button
        button = Button(dlstoplevel,text='Save settings',command=self.saveDLSsettings)
        button.grid(row=5,pady=10,padx=10)

        #run dls
        button2 = Button(dlstoplevel,text='Run DeepLabStream',command=lambda: threading.Thread(target=self.rundls).start())
        button2.grid(row=6,pady=10,padx=10)

    def rundls(self):
        path = str(os.path.join(self.e_dls_path.folder_path,'app.py'))
        print(path)
        call(['python',path])

    def saveDLSsettings(self):

        config = ConfigParser()
        configFile = os.path.join(str(self.label_settingsini.folder_path),'settings.ini')
        config.read(configFile)

        # write the new values into ini file
        config.set('Streaming', 'RESOLUTION', str(self.e_reso.entry_get))
        config.set('Streaming', 'FRAMERATE', str(self.e_fps.entry_get))
        config.set('Streaming', 'STREAMS', str(self.e_streams.entry_get))
        config.set('Streaming', 'OUTPUT_DIRECTORY', str(self.e_outputdir.folder_path))
        config.set('Streaming', 'MULTIPLE_DEVICES', str(self.e_multiDevice.entry_get))
        config.set('Streaming', 'VIDEO_SOURCE', str(self.e_videosource.entry_get))
        config.set('Streaming', 'VIDEO_PATH', str(self.e_videopath.file_path))
        config.set('Streaming', 'VIDEO', str(self.e_video.entry_get))
        config.set('Streaming', 'ANIMALS_NUMBER', str(self.e_animalNo.entry_get))
        config.set('Streaming', 'PORT', str(self.e_port.entry_get))
        config.set('Streaming', 'IPWEBCAM', str(self.e_ipwebcam.entry_get))

        config.set('DeepLabCut', 'DLC_PATH', str(self.e_dlcpath.folder_path))
        config.set('DeepLabCut', 'MODEL', str(self.e_model.folder_path))

        config.set('Classification', 'PATH_TO_CLASSIFIER', str(self.e_classifierpath.file_path))
        config.set('Classification', 'ALL_BODYPARTS', str(self.e_allBP.entry_get))
        config.set('Classification', 'PIXPERMM', str(self.e_ppm.entry_get))
        config.set('Classification', 'THRESHOLD', str(self.e_threshold.entry_get))
        config.set('Classification', 'POOL_SIZE', str(self.e_poolsize.entry_get))

        config.set('Experiment', 'EXP_NUMBER', str(self.e_expNo.entry_get))
        config.set('Experiment', 'RECORD_EXP', str(self.e_recordExp.entry_get))

        config.set('DeepLabStream','dls_path', str(self.e_dls_path.folder_path))

        with open(os.path.join(str(self.label_settingsini.folder_path),'settings.ini'), 'w') as configfile:
            config.write(configfile)

        print('Settings saved in',os.path.basename(configFile))

    def quicklineplot(self):
        lptoplevel = Toplevel()
        lptoplevel.minsize(300,200)
        lptoplevel.wm_title('Plot path plot')

        videodir = os.path.join(os.path.dirname(self.projectconfigini),'videos')
        vid_list = os.listdir(videodir)

        bpdir = os.path.join(os.path.dirname(self.projectconfigini),'logs','measures','pose_configs','bp_names','project_bp_names.csv')
        bplist = pd.read_csv(bpdir,header=None)[0].to_list()

        lplabelframe = LabelFrame(lptoplevel)
        videoSelected = DropDownMenu(lplabelframe, 'Video', vid_list, '15')
        videoSelected.setChoices(vid_list[0])
        bpSelected = DropDownMenu(lplabelframe, 'Body part', bplist, '15')
        bpSelected.setChoices(bplist[0])

        lpbutton = Button(lplabelframe,text='Generate path plot',command= lambda:draw_line_plot(self.projectconfigini,videoSelected.getChoices(),bpSelected.getChoices()))

        #organize
        lplabelframe.grid(row=0,sticky=W)
        videoSelected.grid(row=1, sticky=W)
        bpSelected.grid(row=2, sticky=W)
        lpbutton.grid(row=3, pady=12)

    def visualizeRoiTracking(self):
        vrtoplevel = Toplevel()
        vrtoplevel.minsize(350, 300)
        vrtoplevel.wm_title('Visualize ROI tracking')

        videodir = os.path.join(os.path.dirname(self.projectconfigini), 'videos')
        vid_list = os.listdir(videodir)

        vrlabelframe = LabelFrame(vrtoplevel,text='Visualize ROI tracking on single video',pady=10,padx=10,font=("Helvetica",12,'bold'),fg='black')
        videoSelected = DropDownMenu(vrlabelframe, 'Select video', vid_list, '15')
        videoSelected.setChoices(vid_list[0])

        vrbutton = Button(vrlabelframe, text='Generate ROI visualization', command=lambda: self.roiPlot_singlevideo(videoSelected.getChoices()))

        #multi
        multilabelframe = LabelFrame(vrtoplevel,text='Visualize ROI tracking on all videos',pady=10,padx=10,font=("Helvetica",12,'bold'),fg='black')
        multiButton = Button(multilabelframe,text='Generate ROI visualization on all videos',command = self.roiPlot_allvideos)

        #threshold
        self.p_threshold = Entry_Box(vrtoplevel,'Bp probability threshold','20')
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        #set threshdhol if exist
        try:
            thres = config.get('ROI settings', 'probability_threshold')
            self.p_threshold.entry_set(thres)
        except:
            self.p_threshold.entry_set(0.0)

        thresholdlabel = Label(vrtoplevel,text='Note: body-part locations detected with probabilities below this threshold will be filtered out.')


        # organize
        vrlabelframe.grid(row=0, sticky=W)
        videoSelected.grid(row=1, sticky=W)
        vrbutton.grid(row=2, pady=12)

        multilabelframe.grid(row=1,sticky=W,pady=10)
        multiButton.grid(row=0,sticky=W)

        self.p_threshold.grid(row=2,sticky=W,pady=10)
        thresholdlabel.grid(row=3,sticky=W,pady=10)

    def roiPlot_singlevideo(self,video):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        # write the new values into ini file

        try:
            config.set('ROI settings', 'probability_threshold', str(self.p_threshold.entry_get))
            with open(self.projectconfigini, 'w') as configfile:
                config.write(configfile)
        except:
            pass

        roiPlot(self.projectconfigini,video)


    def roiPlot_allvideos(self):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        # write the new values into ini file
        config.set('ROI settings', 'probability_threshold',str(self.p_threshold.entry_get))

        try:
            with open(self.projectconfigini, 'w') as configfile:
                config.write(configfile)
        except:
            pass

        videolist = os.listdir(os.path.join(os.path.dirname(self.projectconfigini),'videos'))

        for i in videolist:
            roiPlot(self.projectconfigini,i)

    def visualizeROifeatures(self):
        vrfeattoplevel = Toplevel()
        vrfeattoplevel.minsize(350, 300)
        vrfeattoplevel.wm_title('Visualize ROI features')

        videodir = os.path.join(os.path.dirname(self.projectconfigini), 'videos')
        vid_list = os.listdir(videodir)

        vrlabelframe = LabelFrame(vrfeattoplevel, text='Visualize ROI features on single video', pady=10, padx=10,
                                     font=("Helvetica", 12, 'bold'), fg='black')
        videoSelected = DropDownMenu(vrlabelframe, 'Select video', vid_list, '15')
        videoSelected.setChoices(vid_list[0])

        vrbutton = Button(vrlabelframe, text='Generate ROI visualization', command=lambda: self.roifeatures_singlevid(videoSelected.getChoices()))

        # multi
        multifealabelframe = LabelFrame(vrfeattoplevel, text='Visualize ROI features on all videos', pady=10, padx=10,
                                     font=("Helvetica", 12, 'bold'), fg='black')

        multifeaButton = Button(multifealabelframe, text='Generate ROI visualization on all videos',
                             command=self.roifeatures_allvid)

        # threshold
        self.p_threshold_b = Entry_Box(vrfeattoplevel, 'Bp probability threshold', '20')

        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        # set threshdhol if exist
        try:
            thres = config.get('ROI settings', 'probability_threshold')
            self.p_threshold_b.entry_set(thres)
        except:
            self.p_threshold_b.entry_set(0.0)

        thresholdlabel = Label(vrfeattoplevel,
                               text='Note: body-part locations detected with probabilities below this threshold will be filtered out.')

        # organize
        vrlabelframe.grid(row=0, sticky=W)
        videoSelected.grid(row=1, sticky=W)
        vrbutton.grid(row=2, pady=12)

        multifealabelframe.grid(row=1,sticky=W,pady=10)
        multifeaButton.grid(row=0,sticky=W)

        self.p_threshold_b.grid(row=3,sticky=W,pady=10)
        thresholdlabel.grid(row=4,sticky=W,pady=10)

    def roifeatures_singlevid(self,video):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        # write the new values into ini file
        config.set('ROI settings', 'probability_threshold', str(self.p_threshold_b.entry_get))
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

        ROItoFeaturesViz(self.projectconfigini, video)

    def roifeatures_allvid(self):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        # write the new values into ini file
        config.set('ROI settings', 'probability_threshold', str(self.p_threshold_b.entry_get))
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

        videolist = os.listdir(os.path.join(os.path.dirname(self.projectconfigini),'videos'))

        for i in videolist:
            ROItoFeaturesViz(self.projectconfigini,i)

    def timebinmove(self,var):
        timebintoplevel = Toplevel()
        timebintoplevel.minsize(200, 80)
        timebintoplevel.wm_title("Time bins settings")

        tb_labelframe = LabelFrame(timebintoplevel)
        tb_entry = Entry_Box(tb_labelframe,'Set time bin size (s)','15')
        if var == 'mov':
            tb_button = Button(tb_labelframe,text='Run',command=lambda:time_bins_movement(self.projectconfigini,int(tb_entry.entry_get)))
        else:
            tb_button = Button(tb_labelframe, text='Run',
                               command=lambda: time_bins_classifier(self.projectconfigini, int(tb_entry.entry_get)))
        ##organize
        tb_labelframe.grid(row=0,sticky=W)
        tb_entry.grid(row=1,sticky=W)
        tb_button.grid(row=2,pady=10)

    def ROI_classifier(self):
        ROI_clf_toplevel = Toplevel()
        ROI_clf_toplevel.minsize(400, 100)
        ROI_clf_toplevel.wm_title("Classifications by ROI settings")
        ROI_menu = LabelFrame(ROI_clf_toplevel, text='Select_ROI(s)', padx=5, pady=5)
        classifier_menu = LabelFrame(ROI_clf_toplevel, text='Select classifier(s)', padx=5, pady=5)
        body_part_menu = LabelFrame(ROI_clf_toplevel, text='Select body part', padx=5, pady=5)
        self.menu_items = clf_within_ROI(self.projectconfigini)
        self.ROI_check_boxes_status_dict = {}
        self.clf_check_boxes_status_dict = {}

        for row_number, ROI in enumerate(self.menu_items.ROI_str_name_list):
            self.ROI_check_boxes_status_dict[ROI] = IntVar()
            ROI_check_button = Checkbutton(ROI_menu, text=ROI, variable=self.ROI_check_boxes_status_dict[ROI])
            ROI_check_button.grid(row=row_number, sticky=W)

        for row_number, clf_name in enumerate(self.menu_items.behavior_names):
            self.clf_check_boxes_status_dict[clf_name] = IntVar()
            clf_check_button = Checkbutton(classifier_menu, text=clf_name, variable=self.clf_check_boxes_status_dict[clf_name])
            clf_check_button.grid(row=row_number, sticky=W)

        self.choose_bp = DropDownMenu(body_part_menu, 'Body part', self.menu_items.body_part_list, '12')
        self.choose_bp.setChoices(self.menu_items.body_part_list[0])
        self.choose_bp.grid(row=0, sticky=W)


        run_analysis_button = Button(ROI_clf_toplevel, text='Analyze classifications in each ROI', command=lambda: self.run_clf_by_ROI_analysis())

        body_part_menu.grid(row=0, sticky=W, padx=10, pady=10)
        ROI_menu.grid(row=1, sticky=W, padx=10, pady=10)
        classifier_menu.grid(row=2, sticky=W, padx=10, pady=10)
        run_analysis_button.grid(row=3, sticky=W, padx=10, pady=10)

    def run_clf_by_ROI_analysis(self):
        body_part_list = [self.choose_bp.getChoices()]
        ROI_dict_lists, behavior_list = defaultdict(list), []
        for loop_val, ROI_entry in enumerate(self.ROI_check_boxes_status_dict):
            check_val = self.ROI_check_boxes_status_dict[ROI_entry]
            if check_val.get() == 1:
                shape_type = self.menu_items.ROI_str_name_list[loop_val].split(':')[0].replace(':', '')
                shape_name = self.menu_items.ROI_str_name_list[loop_val].split(':')[1].replace(' ', '')
                ROI_dict_lists[shape_type].append(shape_name)

        for loop_val, clf_entry in enumerate(self.clf_check_boxes_status_dict):
            check_val = self.clf_check_boxes_status_dict[clf_entry]
            if check_val.get() == 1:
                behavior_list.append(self.menu_items.behavior_names[loop_val])
        if len(ROI_dict_lists) == 0: print('No ROIs selected.')
        if len(behavior_list) == 0: print('No classifiers selected.')

        else:
            clf_within_ROI.perform_ROI_clf_analysis(self.menu_items, ROI_dict_lists, behavior_list, body_part_list)



    def importBoris(self):
        ann_folder = askdirectory()
        append_Boris_annot(self.projectconfigini,ann_folder)

    def importSolomon(self):
        ann_folder = askdirectory()
        solomonToSimba(self.projectconfigini,ann_folder)

    def import_ethovision(self):
        ann_folder = askdirectory()
        ImportEthovision(config_path=self.projectconfigini, folder_path=ann_folder)

    def importMARS(self):
        ann_folder = askdirectory()
        append_dot_ANNOTT(self.projectconfigini, ann_folder)

    def fileselected(self,val):
        try:
            self.frame.destroy()
        except:
            pass

        self.frame = Frame(self.label_import_csv)

        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        no_animals_int = config.getint('General settings', 'animal_no')

        try:
            self.animal_ID_list = config.get('Multi animal IDs', 'id_list').split(',')
        except MissingSectionHeaderError:
            self.animal_ID_list = []
            for animal in range(no_animals_int): self.animal_ID_list.append('Animal_' + str(animal + 1))

        # method
        self.labelmethod = LabelFrame(self.frame, text='Method', pady=5, padx=5)
        self.interpolation = DropDownMenu(self.labelmethod, 'Interpolation', ['None','Animal(s): Nearest', 'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear', 'Body-parts: Quadratic'], '12')
        self.interpolation.setChoices('None')
        self.interpolation.grid(row=0, sticky=W)

        # smoothing
        self.smooth_pose_win_lbl = LabelFrame(self.frame, text='Smooth pose-estimation data', pady=5, padx=5)
        self.smooth_dropdown = DropDownMenu(self.smooth_pose_win_lbl, 'Smoothing', ['None', 'Gaussian', 'Savitzky Golay'], '12', com=self.smoothing_selected)
        self.smooth_dropdown.setChoices('None')
        self.smoothing_time_label = Entry_Box(self.smooth_pose_win_lbl, 'Period (ms):', labelwidth='12', width=10)
        self.smooth_dropdown.grid(row=0, sticky=W)

        if val == 'CSV (DLC/DeepPoseKit)':

            # multicsv
            label_multicsvimport = LabelFrame(self.frame, text='Import multiple csv files', pady=5, padx=5)
            self.folder_csv = FolderSelect(label_multicsvimport, 'Folder selected:', title='Select Folder with .csv(s)')
            button_import_csv = Button(label_multicsvimport, text='Import csv to project folder', command=self.importdlctracking_multi, fg='navy')

            # singlecsv
            label_singlecsvimport = LabelFrame(self.frame, text='Import single csv files', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlecsvimport, 'File selected', title='Select a .csv file')
            button_importsinglecsv = Button(label_singlecsvimport, text='Import single csv to project folder', command=self.importdlctracking_single, fg='navy')
            self.frame.grid(row=1, sticky=W)

            # method
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            label_multicsvimport.grid(row=2, sticky=W)
            self.folder_csv.grid(row=0, sticky=W)
            button_import_csv.grid(row=1, sticky=W)
            label_singlecsvimport.grid(row=3, sticky=W)
            self.file_csv.grid(row=0, sticky=W)
            button_importsinglecsv.grid(row=1, sticky=W)

        elif val == 'MAT (DANNCE 3D)':
            # multicsv
            label_multicsvimport = LabelFrame(self.frame, text='Import multiple DANNCE files', pady=5, padx=5)
            self.folder_csv = FolderSelect(label_multicsvimport, 'Folder selected:', title= 'Select Folder with MAT files')
            button_import_csv = Button(label_multicsvimport, text='Import DANNCE files to project folder', command = lambda: import_DANNCE_folder(self.projectconfigini, self.folder_csv.folder_path, self.interpolation.getChoices()), fg='navy')

            # singlecsv
            label_singlecsvimport = LabelFrame(self.frame, text='Import single DANNCE files', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlecsvimport, 'File selected', title='Select a .MAT file')
            button_importsinglecsv = Button(label_singlecsvimport, text='Import single DANNCE to project folder', command= lambda: import_DANNCE_file(self.projectconfigini, self.file_csv, self.interpolation.getChoices()), fg='navy')
            self.frame.grid(row=1, sticky=W)

            # method
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            label_multicsvimport.grid(row=2, sticky=W)
            self.folder_csv.grid(row=0, sticky=W)
            button_import_csv.grid(row=1, sticky=W)
            label_singlecsvimport.grid(row=3, sticky=W)
            self.file_csv.grid(row=0, sticky=W)
            button_importsinglecsv.grid(row=1, sticky=W)

        elif val =='JSON (BENTO)':
            # import json into projectfolder
            label_multijsonimport = LabelFrame(self.frame, text='Import multiple json files', pady=5, padx=5)
            self.folder_json = FolderSelect(label_multijsonimport, 'Folder Select:', title='Select Folder with .json(s)')
            button_import_json = Button(label_multijsonimport, text='Import json to project folder', command=lambda: json2csv_folder(self.projectconfigini,self.folder_json.folder_path, self.interpolation.getChoices(), self.smooth_dropdown.getChoices()), fg='navy')

            # singlejson
            label_singlejsonimport = LabelFrame(self.frame, text='Import single json file', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlejsonimport, 'File Select', title='Select a .csv file')
            button_importsinglejson = Button(label_singlejsonimport, text='Import single .json to project folder', command=lambda: json2csv_file(self.projectconfigini, self.file_csv.file_path, self.interpolation.getChoices(), self.smooth_dropdown.getChoices()), fg='navy')

            # import json into projectfolder
            self.frame.grid(row=1, sticky=W)
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            label_multijsonimport.grid(row=2, sticky=W)
            self.folder_json.grid(row=0, sticky=W)
            button_import_json.grid(row=1, sticky=W)
            label_singlejsonimport.grid(row=3, sticky=W)
            self.file_csv.grid(row=0, sticky=W)
            button_importsinglejson.grid(row=1, sticky=W)

        elif val in ('SLP (SLEAP)','H5 (multi-animal DLC)','TRK (multi-animal APT)'):
            animalsettings = LabelFrame(self.frame, text='Animal settings', pady=5, padx=5)
            noofanimals = Entry_Box(animalsettings, 'No of animals', '15')
            noofanimals.entry_set(no_animals_int)
            animalnamebutton = Button(animalsettings, text='Confirm', command=lambda: self.animalnames(noofanimals.entry_get, animalsettings))

            if val == 'H5 (multi-animal DLC)':
                options = ['skeleton', 'box','ellipse']
                self.dropdowndlc = DropDownMenu(self.frame, 'Tracking type', options, '15')
                self.dropdowndlc.setChoices(options[1])

                self.h5path = FolderSelect(self.frame, 'Path to h5 files', lblwidth=15)
                labelinstruction = Label(self.frame,
                                         text='Please import videos before importing the \n'
                                              ' multi animal DLC tracking data')
                runsettings = Button(self.frame, text='Import h5', command=self.importh5)
                #organize
                self.dropdowndlc.grid(row=3, sticky=W)

            elif val == 'SLP (SLEAP)':
                self.h5path = FolderSelect(self.frame, 'Path to .slp files', lblwidth=15)
                labelinstruction = Label(self.frame,
                                         text='Please import videos before importing the \n'
                                              ' multi animal SLEAP tracking data')
                runsettings = Button(self.frame, text='Import .slp', command=self.importh5)

            elif val == 'TRK (multi-animal APT)':
                self.h5path = FolderSelect(self.frame, 'Path to .trk files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the multi animal trk tracking data')
                runsettings = Button(self.frame, text='Import .trk', command=self.importh5)

            # organize
            self.frame.grid(row=1, sticky=W)
            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            animalsettings.grid(row=2, sticky=W)
            noofanimals.grid(row=0, sticky=W)


            animalnamebutton.grid(row=0, column=1, sticky=W)

            #save val into memory for dlc or sleap
            self.val = val

            self.h5path.grid(row=4, sticky=W)
            labelinstruction.grid(row=5, pady=10, sticky=W)
            runsettings.grid(row=6, pady=10)

    def importh5(self):
        idlist = []

        smooth_settings_dict = {}
        if self.smooth_dropdown.getChoices() == 'Gaussian':
            smooth_settings_dict['Method'] = 'Gaussian'
            smooth_settings_dict['Parameters'] = {}
            smooth_settings_dict['Parameters']['Time_window'] = self.smoothing_time_label.entry_get

        if self.smooth_dropdown.getChoices() == 'Savitzky Golay':
            smooth_settings_dict['Method'] = 'Savitzky Golay'
            smooth_settings_dict['Parameters'] = {}
            smooth_settings_dict['Parameters']['Time_window'] = self.smoothing_time_label.entry_get

        if self.smooth_dropdown.getChoices() == 'None':
            smooth_settings_dict['Method'] = 'None'

        try:
            for i in self.animalnamelist:
                idlist.append(i.entry_get)
            if self.val =='H5 (multi-animal DLC)':
                importMultiDLCpose(self.projectconfigini, self.h5path.folder_path, self.dropdowndlc.getChoices(), idlist, self.interpolation.getChoices(), smooth_settings_dict)
            if self.val == 'SLP (SLEAP)':
                importSLEAPbottomUP(self.projectconfigini,self.h5path.folder_path,idlist, self.interpolation.getChoices(), smooth_settings_dict)
            if self.val == 'TRK (multi-animal APT)':
                import_trk(self.projectconfigini,self.h5path.folder_path, idlist, self.interpolation.getChoices(), smooth_settings_dict)

        except Exception as error_str:
            print(error_str)
            print('Check that you have: ')
            print('1. Confirmed the number of animals and named your animals in the SimBA interface')
            print('2. Imported the videos for the tracking data to your SimBA project')

    def animalnames(self, noofanimal, master):
        try:
            self.frame2.destroy()
        except:
            pass

        no_animal = int(noofanimal)
        self.animalnamelist = [0] * no_animal

        self.frame2 = Frame(master)
        self.frame2.grid(row=1, sticky=W)

        for i in range(no_animal):
            self.animalnamelist[i] = Entry_Box(self.frame2, 'Animal ' + str(i + 1) + ' name', '15')
            self.animalnamelist[i].grid(row=i, sticky=W)
            try:
                self.animalnamelist[i].entry_set(self.animal_ID_list[i])
            except IndexError:
                pass


    def addclassifier(self,newclassifier):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        #get current no of target
        notarget = config.getint('SML settings', 'no_targets')
        ## increase the size of the latest no of target and create a new modelpath and target name
        notarget+=1
        modelpath = 'model_path_' + str(notarget)
        targetname = 'target_name_' + str(notarget)
        #write the new values into ini file
        config.set('SML settings', modelpath, '')
        config.set('SML settings', targetname, str(newclassifier))
        config.set('SML settings', 'no_targets', str(notarget))
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

        print(str(newclassifier),'added.')

    def removeclassifiermenu(self):
        rcMenu = Toplevel()
        rcMenu.minsize(200, 200)
        rcMenu.wm_title("Warning: Remove classifier(s) settings")

        # get target
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        notarget = config.getint('SML settings', 'no_targets')
        targetlist = []
        for i in range(notarget):
            targetlist.append(config.get('SML settings', 'target_name_' + str(i + 1)))

        labelcr = LabelFrame(rcMenu,text='Select a classifier to remove')
        classifiertoremove = DropDownMenu(labelcr,'Classifier',targetlist,'8')
        classifiertoremove.setChoices(targetlist[0])

        button = Button(labelcr,text='Remove classifier',command=lambda:self.removeclassifier(classifiertoremove.getChoices(),targetlist))

        #organize
        labelcr.grid(row=0,sticky=W)
        classifiertoremove.grid(row=0,sticky=W)
        button.grid(row=1,pady=10)

    def removeclassifier(self,choice,targetlist):
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)

        ## try to remove the selected classifier
        try:
            targetlist.remove(choice)
            print(str(choice), 'is removed.')
        except ValueError:
            print(choice,'no longer exist in the project_config.ini')

        config.remove_option('SML settings','no_targets')
        for i in range(len(targetlist)+1):
            config.remove_option('SML settings','model_path_'+str(i+1))
            config.remove_option('SML settings', 'target_name_' + str(i + 1))


        config.set('SML settings', 'no_targets', str(len(targetlist)))
        for i in range(len(targetlist)):
            config.set('SML settings','model_path_'+str(i+1),'')
            config.set('SML settings','target_name_'+str(i+1),str(targetlist[i]))

        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

    def tracknoofanimal(self,master,bplist):
        try:
            self.Bodyparts1.destroy()
        except:
            pass
        try:
            self.Bodyparts2.destroy()
        except:
            pass

        if self.noofAnimal.getChoices()=='1':
            self.Bodyparts1 = DropDownMenu(master, 'Animal 1 bodypart', bplist, '15')
            self.Bodyparts1.setChoices((bplist)[0])
            self.Bodyparts1.grid(row=4,sticky=W)
            self.Bodyparts2 = DropDownMenu(master, 'Animal 2 bodypart', bplist, '15')
            self.Bodyparts2.setChoices(None)

        elif self.noofAnimal.getChoices() == '2':
            self.Bodyparts1 = DropDownMenu(master, 'Animal 1 bodypart', bplist, '15')
            self.Bodyparts1.setChoices((bplist)[0])
            self.Bodyparts1.grid(row=4, sticky=W)
            self.Bodyparts2 = DropDownMenu(master, 'Animal 2 bodypart', bplist, '15')
            self.Bodyparts2.setChoices((bplist)[len(bplist) // 2])
            self.Bodyparts2.grid(row=5, sticky=W)

    def loaddefinedroi(self):

        h5dir = os.path.join(os.path.dirname(self.projectconfigini), 'logs', 'measures')
        h5list = os.listdir(h5dir)
        result = [i for i in h5list if '.h5' in i]
        if result == []:
            print('Please define ROI Shapes')
        else:
            videodir = os.path.join(os.path.dirname(self.projectconfigini), 'videos')

            roitableMenu(videodir, self.projectconfigini)

    def appendroisettings(self):
        apdroisettings = Toplevel()
        apdroisettings.minsize(400, 400)
        apdroisettings.wm_title("Append Roi Settings")
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        projectNoAnimals = config.getint('General settings', 'animal_no')

        # first choice frame
        firstMenu = LabelFrame(apdroisettings, text='Select number of animals')

        ## set up drop down for animals
        noOfAnimalVar = IntVar()
        animalOptions = set(range(1, projectNoAnimals + 1))
        noOfAnimalVar.set(1)

        animalMenu = OptionMenu(firstMenu, noOfAnimalVar, *animalOptions)
        animalLabel = Label(firstMenu, text="# of animals")
        setAnimalButton = Button(firstMenu, text="Confirm",
                                 command=lambda: self.run_roiAnalysisSettings(apdroisettings, noOfAnimalVar,'append'))

        # organize
        firstMenu.grid(row=0, sticky=W)
        animalLabel.grid(row=0, column=0, sticky=W)
        animalMenu.grid(row=0, column=1, sticky=W)
        setAnimalButton.grid(row=0, column=2, sticky=W)

    def timebin_ml(self,title,text='Run'):
        roisettings = Toplevel()
        roisettings.minsize(400, 400)
        roisettings.wm_title(title)
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        projectNoAnimals = config.getint('General settings', 'animal_no')

        #first choice frame
        firstMenu = LabelFrame(roisettings,text='Select number of animals')

        ## set up drop down for animals
        noOfAnimalVar = IntVar()
        animalOptions = set(range(1, projectNoAnimals+1))
        noOfAnimalVar.set(1)

        animalMenu = OptionMenu(firstMenu,noOfAnimalVar,*animalOptions)
        animalLabel = Label(firstMenu,text="# of animals")

        setAnimalButton = Button(firstMenu,text="Confirm",command=lambda:self.timebin_ml2(master=roisettings,noofanimal= noOfAnimalVar, text=text, title=title))

        #organize
        firstMenu.grid(row=0,sticky=W)
        animalLabel.grid(row=0,column=0,sticky=W)
        animalMenu.grid(row=0,column=1,sticky=W)
        setAnimalButton.grid(row=0,column=2,sticky=W)

    def timebin_ml2(self, master, noofanimal, text='Run',title =None ):
        try:
            self.secondMenu.destroy()
        except:
            pass

        self.secondMenu = LabelFrame(master, text="Choose bodyparts")

        # try to see if it exist or not
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)

        if title != "Time bins: Distance/Velocity":
            runButton = Button(self.secondMenu, text=text,
                           command=lambda: self.timebin_ml3(noofanimal.get(),self.animalVarList[animal].get(),self.binlen.entry_get,'analyze'))
        else:
            runButton = Button(self.secondMenu, text=text,
                               command=lambda: self.timebin_ml3(noofanimal.get(), self.animalVarList[animal].get(),
                                                                self.binlen.entry_get, 'Notanalyze'))

        animals2analyze = noofanimal.get()
        labelFrameList, labelList, self.animalVarList, AnimalStringVarList, optionsVarList, animalBodyMenuList = [], [], [], [], [], []
        options = define_bp_drop_down(configini)
        for animal in range(animals2analyze):
            animalName = str(animal + 1)
            labelFrameList.append(LabelFrame(self.secondMenu, text='Animal ' + animalName))
            labelList.append(Label(labelFrameList[animal], text='Bodypart'))
            self.animalVarList.append(StringVar())
            self.animalVarList[animal].set(options[animal][0])
            animalBodyMenuList.append(OptionMenu(labelFrameList[animal], self.animalVarList[animal], *options[animal]))

        #binlen
        self.binlen = Entry_Box(self.secondMenu,'Set time bin size (s)',"16")

        # organize
        self.secondMenu.grid(row=1, sticky=W)
        for animal in range(animals2analyze):
            labelFrameList[animal].grid(row=animal, column=0, sticky=W)
            labelList[animal].grid(row=0, column=0, sticky=W)
            animalBodyMenuList[animal].grid(row=animal, column=0, sticky=W)

        self.binlen.grid(row=animals2analyze+2,sticky=W)
        runButton.grid(row=animals2analyze + 3, padx=10, pady=10)

    def timebin_ml3(self,noofanimal,animalBp,binlen,text='Notanalyze'):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)

        if text == 'Notanalyze':
            config.set('process movements', 'no_of_animals', str(noofanimal))
            bp_vars_dist = self.animalVarList

            for animal in range(noofanimal):
                animalBp = str(bp_vars_dist[animal].get())
                config.set('process movements', 'animal_' + str(animal + 1) + '_bp', animalBp)

            with open(configini, 'w') as configfile:
                config.write(configfile)

            time_bins_movement(configini,int(binlen))
        else:
            config.set('ROI settings', 'no_of_animals', str(noofanimal))
            bp_vars_dist = self.animalVarList

            for animal in range(noofanimal):
                animalBp = str(bp_vars_dist[animal].get())
                config.set('ROI settings', 'animal_' + str(animal + 1) + '_bp', animalBp)

            with open(configini, 'w') as configfile:
                config.write(configfile)

            roi_time_bins(self.projectconfigini, 'outlier_corrected_movement_location', int(binlen))

    def roi_settings(self,title,selection,text='Run'):
        roisettings = Toplevel()
        roisettings.minsize(400, 400)
        roisettings.wm_title(title)
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        projectNoAnimals = config.getint('General settings', 'animal_no')

        #first choice frame
        firstMenu = LabelFrame(roisettings,text='Select number of animals')

        ## set up drop down for animals
        noOfAnimalVar = IntVar()
        animalOptions = set(range(1, projectNoAnimals+1))
        noOfAnimalVar.set(1)

        animalMenu = OptionMenu(firstMenu,noOfAnimalVar,*animalOptions)
        animalLabel = Label(firstMenu,text="# of animals")

        setAnimalButton = Button(firstMenu,text="Confirm",command=lambda:self.run_roiAnalysisSettings(roisettings,noOfAnimalVar,selection,text=text))

        #organize
        firstMenu.grid(row=0,sticky=W)
        animalLabel.grid(row=0,column=0,sticky=W)
        animalMenu.grid(row=0,column=1,sticky=W)
        setAnimalButton.grid(row=0,column=2,sticky=W)

    def run_roiAnalysisSettings(self,master,noofanimal,appendornot,text='Run'):
        try:
            self.secondMenu.destroy()
        except:
            pass

        self.secondMenu = LabelFrame(master,text="Choose bodyparts")
        self.p_threshold_a = Entry_Box(self.secondMenu,'Bp probability threshold','20')

        #try to see if it exist or not
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        try:
            pthresh = config.get('ROI settings', 'probability_threshold')
            self.p_threshold_a.entry_set(pthresh)
        except:
            self.p_threshold_a.entry_set(0.00)

        self.disvar = IntVar()
        discheckbox = Checkbutton(self.secondMenu,text='Calculate distance moved within ROI',variable=self.disvar)
        runButton = Button(self.secondMenu,text=text,command =lambda:self.run_analyze_roi(noofanimal.get(), self.animalVarList[animal], appendornot))
        animals2analyze = noofanimal.get()
        labelFrameList, labelList, self.animalVarList, AnimalStringVarList, optionsVarList, animalBodyMenuList = [],[],[],[],[],[]
        options = define_bp_drop_down(configini)
        if appendornot != 'locationheatmap':
            for animal in range(animals2analyze):
                animalName = str(animal + 1)
                labelFrameList.append(LabelFrame(self.secondMenu,text='Animal ' + animalName))
                labelList.append(Label(labelFrameList[animal],text='Bodypart'))
                self.animalVarList.append(StringVar())
                self.animalVarList[animal].set(options[animal][0])
                animalBodyMenuList.append(OptionMenu(labelFrameList[animal], self.animalVarList[animal], *options[animal]))
        if appendornot == 'locationheatmap':
            options = [item for sublist in options for item in sublist]
            labelFrameList.append(LabelFrame(self.secondMenu,text='Animal bodypart'))
            labelList.append(Label(labelFrameList[0], text='Bodypart'))
            self.animalVarList.append(StringVar())
            self.animalVarList[0].set(options[0])
            animalBodyMenuList.append(OptionMenu(labelFrameList[0], self.animalVarList[0], *options))

        #organize
        self.secondMenu.grid(row=1, sticky=W)
        for animal in range(animals2analyze):
            labelFrameList[animal].grid(row=animal,column=0, sticky=W)
            labelList[animal].grid(row=0, column=0, sticky=W)
            animalBodyMenuList[animal].grid(row=animal, column=0, sticky=W)
        if appendornot != 'locationheatmap':
            self.p_threshold_a.grid(row=animals2analyze+1, sticky=W)
            discheckbox.grid(row=animals2analyze+2, sticky=W)
            runButton.grid(row=animals2analyze+3, padx=10, pady=10)

        if appendornot == 'locationheatmap':
            heatmapframe = Frame(self.secondMenu)
            self.binsizepixels = Entry_Box(heatmapframe,'Bin size (mm)','21')
            self.scalemaxsec = Entry_Box(heatmapframe,'max','21')
            self.pal_var = StringVar()
            paloptions = ['magma','jet','inferno','plasma','viridis','gnuplot2']
            palette = OptionMenu(heatmapframe,self.pal_var,*paloptions)
            self.pal_var.set(paloptions[0])
            self.lastimgvar =IntVar()
            lastimg = Checkbutton(heatmapframe,text='Save last image only (if unticked heatmap videos are created)',variable=self.lastimgvar)
            newoptions = [item for sublist in options for item in sublist]
            self.animalbody1var = StringVar()
            # self.animalbody1var.set(newoptions[0])
            animalbodymenu1 = OptionMenu(heatmapframe, self.animalbody1var, *newoptions)


            #organize
            heatmapframe.grid(row=5,sticky=W)
            # animalbodymenu1.grid(row=0, column=0, sticky=W)
            self.binsizepixels.grid(row=1,sticky=W)
            self.scalemaxsec.grid(row=2,sticky=W)
            palette.grid(row=4,sticky=W)
            lastimg.grid(row=5,sticky=W)
            runButton.grid(row=6, padx=10, pady=10)


    def run_analyze_roi(self,noofanimal,animalVarList,appendornot):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)

        if appendornot == 'processmovement':
            config.set('process movements', 'no_of_animals', str(noofanimal))
            bp_vars_dist = self.animalVarList
            for animal in range(noofanimal):
                animalBp = str(bp_vars_dist[animal].get())
                config.set('process movements', 'animal_' + str(animal+1) + '_bp', animalBp)
            with open(configini, 'w') as configfile:
                config.write(configfile)

        elif appendornot == 'locationheatmap':
            animalBp = str(animalVarList.get())
            config.set('Heatmap location', 'body_part', animalBp)
            config.set('Heatmap location', 'Palette', str(self.pal_var.get()))
            config.set('Heatmap location', 'Scale_max_seconds', str(self.scalemaxsec.entry_get))
            config.set('Heatmap location', 'bin_size_pixels', str(self.binsizepixels.entry_get))
            with open(configini, 'w') as configfile:
                config.write(configfile)

        else:
            config.set('ROI settings', 'no_of_animals', str(noofanimal))
            bp_vars_ROI = self.animalVarList
            for animal in range(noofanimal):
                currStr = 'animal_' + str(animal+1) + '_bp'
                config.set('ROI settings', currStr, str(bp_vars_ROI[animal].get()))
                with open(configini, 'w') as configfile:
                    config.write(configfile)

        if appendornot == 'append':
            ROItoFeatures(configini)
        elif appendornot =='not append':
            if self.disvar.get()==1:
                caldist = True
            else:
                caldist = False
            #write into configini
            config.set('ROI settings', 'probability_threshold', str(self.p_threshold_a.entry_get))
            with open(configini, 'w') as configfile:
                config.write(configfile)

            roiAnalysis(configini,'outlier_corrected_movement_location',caldist)

        elif appendornot == 'processmovement':
            ROI_process_movement(configini)
        elif appendornot == 'locationheatmap':
            plotHeatMapLocation(configini,animalBp,int(self.binsizepixels.entry_get),str(self.scalemaxsec.entry_get),self.pal_var.get(),self.lastimgvar.get())
        elif appendornot == 'direction':
            print('ROI settings saved.')
        else:
            roiAnalysis(configini,'features_extracted')

    def updateThreshold(self):
        updateThreshold_graph(self.projectconfigini, self.csvfile.file_path, self.modelfile.file_path)

    def getEntry(self):
        # rectangle
        recName = []
        recWidth = []
        recHeight = []
        rec_shape = []
        try:
            for i in range(len(self.rect_list)):
                rec_shape.append("Rectangle")
                recName.append(self.rect_list[i].get())
                recWidth.append(int(self.rec_width_list[i].get()))
                recHeight.append(int(self.rec_height_list[i].get()))

            rec_df = pd.DataFrame(list(zip(rec_shape, recName, recWidth, recHeight)), columns=['Shape_type', 'Name', 'width', 'height'])

        except:
            rec_df = pd.DataFrame()

        # circle
        cirName = []
        cirRadius = []
        cir_shape = []
        try:
            for i in range(len(self.cir_entry_list)):
                cir_shape.append("Circle")
                cirName.append(self.cir_entry_list[i].get())
                cirRadius.append(int(self.cir_radius_list[i].get()))
            cir_df = pd.DataFrame(list(zip(cir_shape, cirName, cirRadius)), columns=['Shape_type', 'Name', 'Radius'])
        except:
            cir_df = pd.DataFrame()

        ## polygon
        polName = []
        polShape = []

        try:
            for i in range(len(self.pol_list)):
                polShape.append("Polygon")
                polName.append(self.pol_list[i].get())
            pol_df = pd.DataFrame(list(zip(polShape, polName)), columns=['Shape_type', 'Name'])
        except:
            pol_df = pd.DataFrame()

        if not os.path.exists(os.path.join(os.path.dirname(self.projectconfigini),"logs","measures")):
            os.makedirs(os.path.join(os.path.dirname(self.projectconfigini),"logs","measures"))

        ### to h5
        storePath = os.path.join(os.path.dirname(self.projectconfigini),"logs","measures",'ROI_index.h5')
        store = pd.HDFStore(storePath, mode='w')
        store['rectangles'] = rec_df
        store['circleDf'] = cir_df
        store['polygons'] = pol_df
        store.close()

        videodir = os.path.join(os.path.dirname(self.projectconfigini),'videos')

        roitableMenu(videodir,self.projectconfigini)

    def table(self,master,rectangle,circle,polygon):
        # refresh the frames
        try:
            self.rectbox.destroy()
            self.cirbox.destroy()
            self.polbox.destroy()
        except:
            pass

        try:
            ## make rectangle table
            self.rectbox = LabelFrame(master)
            self.rectbox.grid(row=6, sticky=W, pady=10)
            self.rect_list = list(range(int(rectangle)))
            self.rec_width_list = list(range(int(rectangle)))
            self.rec_height_list = list(range(int(rectangle)))
            rect_var = list(range(int(rectangle)))
            rec_name_list = list(range(int(rectangle)))

            #
            rectangleName = Label(self.rectbox, text="Rectangle Name")
            rectangleName.grid(row=0, column=1)

            rectangleWidth = Label(self.rectbox, text="Width")
            rectangleWidth.grid(row=0, column=2)

            rectangleHeight = Label(self.rectbox, text="Height")
            rectangleHeight.grid(row=0, column=3)

            for i in range(len(self.rect_list)):
                rec_name_list[i] = Label(self.rectbox, text="Rectangle " + str(i + 1))
                rec_name_list[i].grid(row=int(i) + 1, padx=5, sticky=W)
                self.rect_list[i] = Entry(self.rectbox)
                self.rect_list[i].grid(row=int(i) + 1, column=1, padx=5)
                rect_var[i]=StringVar()
                self.rec_width_list[i] = Entry(self.rectbox,textvariable=rect_var[i])

                self.rec_width_list[i].grid(row=int(i) + 1, column=2, padx=5)
                self.rec_height_list[i] = Entry(self.rectbox,textvariable=rect_var[i])
                self.rec_height_list[i].grid(row=int(i) + 1, column=3, padx=5)

                rect_var[i].set('0')
        except:
            pass

        try:
            ## make circle table
            self.cirbox = LabelFrame(master)
            self.cirbox.grid(row=7, sticky=W)
            cir_list = list(range(int(circle)))
            self.cir_entry_list = list(range(int(circle)))
            self.cir_radius_list = list(range(int(circle)))
            cir_var = list(range(int(circle)))

            circleName = Label(self.cirbox, text="Circle Name")
            circleName.grid(row=0, column=1)

            radiusName = Label(self.cirbox, text='Radius')
            radiusName.grid(row=0, column=2)

            for i in range(len(cir_list)):
                cir_list[i] = Label(self.cirbox, text=("Circle " + str(i + 1)))
                cir_list[i].grid(row=int(i) + 1, padx=5, sticky=W)
                self.cir_entry_list[i] = Entry(self.cirbox)
                self.cir_entry_list[i].grid(row=int(i) + 1, column=1, padx=5)
                cir_var[i] =  StringVar()
                self.cir_radius_list[i] = Entry(self.cirbox,textvariable=cir_var[i])
                self.cir_radius_list[i].grid(row=int(i) + 1, column=2, padx=5)
                cir_var[i].set('0')
        except:
            pass

        try:
            ## make polygon table/.;p
            self.polbox = LabelFrame(master)
            self.polbox.grid(row=8, sticky=W)
            self.pol_list = list(range(int(polygon)))
            pol_name = list(range(int(polygon)))
            polygonName = Label(self.polbox, text="Polygon Name")
            polygonName.grid(row=0, column=1)
            for i in range(len(pol_name)):
                pol_name[i] = Label(self.polbox, text="Polygon " + str(i + 1))
                pol_name[i].grid(row=int(i) + 1, column=0, sticky=W)
                self.pol_list[i] = Entry(self.polbox)
                self.pol_list[i].grid(row=int(i) + 1, column=1, padx=5)
        except:
            pass


        setbutton = Button(master,text='Set Shape Definitions',command=self.setvariables)
        setbutton.grid(row=10)

    def setvariables(self):
        measuresdir = os.path.join(os.path.dirname(self.projectconfigini),'logs','measures')
        try:
            os.remove(os.path.join(measuresdir,'ROI_definitions.h5'))
            os.remove(os.path.join(measuresdir, 'ROI_index.h5'))
        except:
            pass

        try:
            self.getEntry()

        except ImportError:
            print('ERROR: Please install or reinstall pytables. `Use pip install tables --upgrade` or `conda install pytables`')


    def classifiervalidation(self):
        print('Generating video...')
        validate_classifier(self.projectconfigini, self.seconds.entry_get,self.cvTarget.getChoices())
        print('Videos generated')

    def mergeframesofplot(self,var):
        inputList = []
        for i in var:
            inputList.append(i.get())

        mergeframesPlot(self.projectconfigini,inputList)


    def generateSimBPlotlyFile(self,var):
        inputList = []
        for i in var:
            inputList.append(i.get())

        create_plotly_container(self.projectconfigini, inputList)

    def open_web_link(self, url):
        sys.excepthook = cef.ExceptHook  # To shutdown all CEF processes on error
        cef.Initialize()
        cef.CreateBrowserSync(url=url,
                              window_title=url)
        cef.MessageLoop()

    def open_plotly_interface(self, url):

        # kill any existing plotly cache
        try:
            self.p.kill()
            self.p2.kill()
        except:
            print('Starting plotly')
        #get h5 file path and csv file path
        filePath, groupPath = self.plotly_file.file_path, self.groups_file.file_path

        #print file read
        if filePath.endswith('.h5'):
            print('Reading in',os.path.basename(filePath))
        elif groupPath.endswith('.csv'):
            print('Reading in',os.path.basename(groupPath))

        self.p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__),'SimBA_dash_app.py'), filePath, groupPath])
        # csvPath = os.path.join(os.path.dirname(self.projectconfigini),'csv')
        # p = subprocess.Popen([sys.executable, r'simba\SimBA_dash_app.py', filePath, groupPath, csvPath])
        wait_for_internet_connection(url)
        self.p2 = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__),'run_dash_tkinter.py'), url])
        subprocess_children = [self.p, self.p2]
        atexit.register(terminate_children, subprocess_children)



    def plotdataplot(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        pose_config_setting = config.get('create ensemble settings', 'pose_estimation_body_parts')
        if pose_config_setting == 'user_defined':
            data_plot_config(self.projectconfigini, self.SelectedBp.getChoices())
        else:
            data_plot_config(self.projectconfigini, 'Centroid')


    def plotsklearn_result(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        plotsklearnresult(self.projectconfigini,self.videovar.get(),self.genframevar.get())

    def analyzseverity(self):
        analyze_process_severity(self.projectconfigini,self.severityscale.entry_get,self.severityTarget.getChoices())

    def analyzedatalog(self):
        # Popup window
        datalogmenu = Toplevel()
        datalogmenu.minsize(400, 400)
        datalogmenu.wm_title("Analyze process log settings")

        dlmlabel = LabelFrame(datalogmenu)

        #use for loop to create intvar
        var=[]
        for i in range(7):
            var.append(IntVar())

        #use loop to create checkbox?
        checkbox = [0]*7
        titlebox =['# bout events', 'total events duration (s)','mean bout duration (s)', 'median bout duration (s)', 'first occurance (s)', 'mean interval (s)', 'median interval (s)']
        for i in range(7):
            checkbox[i] = Checkbutton(dlmlabel,text=titlebox[i],variable=var[i])
            checkbox[i].grid(row=i,sticky=W)
        #organize
        dlmlabel.grid(row=0)
        button1 = Button(dlmlabel,text='Analyze',command=lambda:self.findDatalogList(titlebox,var))
        button1.grid(row=10)

    def findDatalogList(self,titleBox,Var):
        finallist = []
        for index,i in enumerate(Var):
            if i.get()==0:
                finallist.append(titleBox[index])

        #run analyze
        analyze_process_data_log(self.projectconfigini,finallist)


    def runrfmodel(self):
        rfmodel(self.projectconfigini)

    def modelselection(self):
        runmachinemodelsettings(self.projectconfigini)

    def validatemodelsinglevid(self):
        validate_model_one_vid(self.projectconfigini, self.csvfile.file_path, self.modelfile.file_path, self.dis_threshold.entry_get, self.min_behaviorbout.entry_get, self.generategantt_dropdown.getChoices())

    def trainmultimodel(self):
        train_multimodel(self.projectconfigini)


    def trainmachinemodelsetting(self):
        trainmachinemodel_settings(self.projectconfigini)

    def extractfeatures(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
        print('Pose-estimation body part setting for feature extraction: ' + str(pose_estimation_body_parts))
        userFeatureScriptStatus = self.usVar.get()
        print(userFeatureScriptStatus)


        if userFeatureScriptStatus == 1:
            pose_estimation_body_parts == 'user_defined_script'
            import sys
            script = self.scriptfile.file_path
            print(script)
            dir = os.path.dirname(script)
            fscript = os.path.basename(script).split('.')[0]
            sys.path.insert(0,dir)
            import importlib
            mymodule = importlib.import_module(fscript)
            mymodule.extract_features_userdef(self.projectconfigini)

        if userFeatureScriptStatus == 0:
            if pose_estimation_body_parts == '16':
                extract_features_wotarget_16(self.projectconfigini)
            if (pose_estimation_body_parts == '14'):
                extract_features_wotarget_14(self.projectconfigini)
            if (pose_estimation_body_parts == '987'):
                extract_features_wotarget_14_from_16(self.projectconfigini)
            if pose_estimation_body_parts == '9':
                extract_features_wotarget_9(self.projectconfigini)
            if pose_estimation_body_parts == '8':
                extract_features_wotarget_8(self.projectconfigini)
            if pose_estimation_body_parts == '7':
                extract_features_wotarget_7(self.projectconfigini)
            if pose_estimation_body_parts == '4':
                extract_features_wotarget_4(self.projectconfigini)
            if pose_estimation_body_parts == 'user_defined':
                extract_features_wotarget_user_defined(self.projectconfigini)


    def importframefolder(self):
        if (self.projectconfigini!='No file selected') and (self.frame_folder.folder_path != 'No folder selected'):
            copy_frame_folders(self.frame_folder.folder_path, self.projectconfigini)
        else:
            print('Fail to import frame folder, please select a main directory containing all the frame folders')

    def importvideo_single(self):
        if (self.projectconfigini != 'No file selected') and (self.singlevideopath.file_path != 'No file selected'):
            copy_singlevideo_ini(self.projectconfigini, self.singlevideopath.file_path)
        else:
            print('Failed to import video, please select a video to import')

    def importvideo_multi(self):
        if (self.projectconfigini != 'No file selected') and (self.multivideofolderpath.folder_path != 'No folder selected') and (self.video_type.entry_get != ''):
            copy_multivideo_ini(self.projectconfigini, self.multivideofolderpath.folder_path,self.video_type.entry_get)
        else:
            print('Fail to import videos, please select folder with videos and enter the file format')

    def importdlctracking_single(self):
        if (self.projectconfigini != 'No file selected') and (self.file_csv.file_path != 'No file selected'):

            copy_singlecsv_ini(self.projectconfigini, self.file_csv.file_path)

            # read in configini
            configFile = str(self.projectconfigini)
            config = ConfigParser()
            config.read(configFile)
            animalIDlist = config.get('Multi animal IDs', 'id_list')

            #name filter
            filebasename = os.path.basename(self.file_csv.file_path)
            if (('.csv') and ('DeepCut') in filebasename) or (('.csv') and ('DLC_') in filebasename):
                if (('.csv') and ('DeepCut') in filebasename):
                    newFname = str(filebasename.split('DeepCut')[0]) + '.csv'
                if (('.csv') and ('DLC_') in filebasename):
                    newFname = str(filebasename.split('DLC_')[0]) + '.csv'
            else:
                newFname = str(filebasename.split('.')[0]) + '.csv'

            csvfile = os.path.join(os.path.dirname(self.projectconfigini), 'csv', 'input_csv',newFname)

            if not animalIDlist:
                df = pd.read_csv(csvfile)

                tmplist = []
                for i in df.loc[0]:
                    tmplist.append(i)

                if 'individuals' in tmplist:
                    tmplist.remove('individuals')

                    if len(set(tmplist)) == 1:
                        print('single animal using maDLC detected. Removing "individuals" row...')
                        df = df.iloc[1:]
                        df.to_csv(csvfile, index=False)
                        print('Row removed for',os.path.basename(i))

                else:
                    pass

            csv_df = pd.read_csv(csvfile, index_col=0)
            if self.interpolation.getChoices() != 'None':
                interpolate_body_parts = Interpolate(self.projectconfigini, csv_df)
                interpolate_body_parts.detect_headers()
                interpolate_body_parts.fix_missing_values(self.interpolation.getChoices())
                interpolate_body_parts.reorganize_headers()
                interpolate_body_parts.new_df.to_csv(csvfile)

            if self.smooth_dropdown.getChoices() == 'Gaussian':
                time_window = self.smoothing_time_label.entry_get
                smooth_data_gaussian(config=config, file_path=csvfile, time_window_parameter=time_window)

            if self.smooth_dropdown.getChoices() == 'Savitzky Golay':
                time_window = self.smoothing_time_label.entry_get
                smooth_data_savitzky_golay(config=config, file_path=csvfile, time_window_parameter=time_window)

            print('Finished importing tracking data.')

        else:
            print('Fail to import csv file , please select a csv file to import and load config.ini file')

    def importdlctracking_multi(self):
        if (self.projectconfigini !='No file selected') and (self.folder_csv.folder_path!= 'No folder selected'):
            copy_allcsv_ini(self.projectconfigini, self.folder_csv.folder_path)

            # read in configini
            configFile = str(self.projectconfigini)
            config = ConfigParser()
            config.read(configFile)
            animalIDlist = config.get('Multi animal IDs', 'id_list')

            if not animalIDlist:
                # get all csv in project folder input csv
                csvfolder = os.path.join(os.path.dirname(self.projectconfigini), 'csv', 'input_csv')
                allcsvs = []
                for i in os.listdir(csvfolder):
                    if i.endswith('.csv'):
                        csvfile = os.path.join(csvfolder, i)
                        allcsvs.append(csvfile)

                # screen for madlc format but single animal
                for i in allcsvs:
                    df = pd.read_csv(i)
                    tmplist = []

                    for j in df.loc[0]:
                        tmplist.append(j)

                    # if it is madlc
                    if 'individuals' in tmplist:
                        tmplist.remove('individuals')
                        # if only single animal in madlc
                        if len(set(tmplist)) == 1:
                            print('Single animal using maDLC detected. Removing "individuals" row...')
                            df = df.iloc[1:]
                            df.to_csv(i, index=False)
                            print('Row removed for',os.path.basename(i))
                    else:
                        pass

            csvfilepath = os.path.join(os.path.dirname(self.projectconfigini), 'csv', 'input_csv')
            csvfile = glob.glob(csvfilepath + '/*.csv')

            if self.interpolation.getChoices() != 'None':
                print('Interpolating missing values (Method: ' + str(self.interpolation.getChoices()) + ') ...')
                for file in csvfile:
                    csv_df = read_df(file, 'csv')
                    interpolate_body_parts = Interpolate(self.projectconfigini, csv_df)
                    interpolate_body_parts.detect_headers()
                    interpolate_body_parts.fix_missing_values(self.interpolation.getChoices())
                    interpolate_body_parts.reorganize_headers()
                    interpolate_body_parts.new_df.to_csv(file)

            if self.smooth_dropdown.getChoices() == 'Gaussian':
                time_window = self.smoothing_time_label.entry_get
                print('Smoothing data using Gaussian method and {} ms time window ...'.format(str(time_window)))
                for file in csvfile:
                    smooth_data_gaussian(config=config, file_path=file, time_window_parameter=time_window)

            if self.smooth_dropdown.getChoices() == 'Savitzky Golay':
                time_window = self.smoothing_time_label.entry_get
                print('Smoothing data using Savitzky Golay method and {} ms time window ...'.format(str(time_window)))
                for file in csvfile:
                    smooth_data_savitzky_golay(config=config, file_path=file, time_window_parameter=time_window)

            print('Finished importing tracking data.')


        else:
            print('Fail to import csv file, please select folder with the .csv files and load config.ini file')

    def set_distancemm(self, distancemm):

        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)

        config.set('Frame settings', 'distance_mm', distancemm)
        with open(configini, 'w') as configfile:
            config.write(configfile)

    def extract_frames_loadini(self):
        configini = self.projectconfigini
        videopath = os.path.join(os.path.dirname(configini), 'videos')
        extract_frames_ini(videopath, configini)

    def correct_outlier(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
        print('Pose-estimation body part setting for outlier correction: ' + str(pose_estimation_body_parts))
        if (pose_estimation_body_parts == '16') or (pose_estimation_body_parts == '987'):
            dev_move_16(configini)
            dev_loc_16(configini)
        if pose_estimation_body_parts == '14':
            dev_move_14(configini)
            dev_loc_14(configini)
        if (pose_estimation_body_parts == 'user_defined') or (pose_estimation_body_parts == '4') or (pose_estimation_body_parts == '7') or (pose_estimation_body_parts == '8') or (pose_estimation_body_parts == '9'):
            dev_move_user_defined(configini)
            dev_loc_user_defined(configini)
        print('Outlier correction complete.')

    def distanceplotcommand(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)

        config.set('Distance plot', 'POI_1', self.poi1.getChoices())
        config.set('Distance plot', 'POI_2', self.poi2.getChoices())
        with open(configini, 'w') as configfile:
            config.write(configfile)

        line_plot_config(configini)
        print('Distance plot complete.')

    def pathplotcommand(self):

        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)

        config.set('Path plot settings', 'no_animal_pathplot', self.noofAnimal.getChoices())
        config.set('Path plot settings', 'deque_points', self.Deque_points.entry_get)
        config.set('Path plot settings', 'severity_brackets', self.severity_brackets.entry_get)
        try:
            config.set('Path plot settings', 'animal_1_bp', self.Bodyparts1.getChoices())
            config.set('Path plot settings', 'animal_2_bp', self.Bodyparts2.getChoices())
        except:
            pass

        config.set('Path plot settings','severity_target',self.severityTargetpp.getChoices())

        if self.plotsvvar.get()==1:
            config.set('Path plot settings', 'plot_severity', 'yes')
        else:
            config.set('Path plot settings', 'plot_severity', 'no')

        with open(configini, 'w') as configfile:
            config.write(configfile)

        path_plot_config(configini)
        print('Path plot complete.')

    def heatmapcommand(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        config.set('Heatmap settings', 'bin_size_pixels', self.BinSize.entry_get)
        config.set('Heatmap settings', 'Scale_max_seconds', self.MaxScale.entry_get)
        config.set('Heatmap settings', 'Palette', self.hmMenu.getChoices())
        config.set('Heatmap settings', 'Target', self.targetMenu.getChoices())
        config.set('Heatmap settings', 'body_part', self.bp1.getChoices())
        with open(configini, 'w') as configfile:
            config.write(configfile)
        plotHeatMap(configini,self.bp1.getChoices(),int(self.BinSize.entry_get),str(self.MaxScale.entry_get)
                    ,self.hmMenu.getChoices(), self.targetMenu.getChoices(),self.intimgvar.get() )

    def callback(self,url):
        webbrowser.open_new(url)

class unsupervisedInterface:
    def __init__(self,inifile):
        self.unsupervisedfolderpath = os.path.join(os.path.dirname(inifile), 'unsupervised')
        #get data from yaml
        self.configini = str(inifile)
        config = ConfigParser()
        configFile = str(self.configini)
        config.read(configFile)

        classifierlist= []

        for i in range(config.getint('SML settings','no_targets')):
            classifierlist.append(config.get('SML settings','target_name_'+str(i+1)))


        window = Toplevel()
        window.wm_title("SimBA Unsupervised GUI")
        window.minsize(800, 400)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        tabControl = ttk.Notebook(window)

        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)
        tab5 = ttk.Frame(tabControl)
        tab6 = ttk.Frame(tabControl)

        tabControl.add(tab1, text=' [Create Project] ')
        tabControl.add(tab3, text=' [Perform Dimensionality Reduction] ')
        tabControl.add(tab4, text=' [Perform Clustering] ')
        tabControl.add(tab5, text=' [Train Model] ')
        tabControl.add(tab6, text=' [Visualize Clusters] ')

        tabControl.grid(row=0,sticky=NW)
        tabControl.enable_traversal()

        # save project folder frame
        saveProjectFolder = LabelFrame(tab1, text="1.Create Project Folder", font=("Helvetica", 12, 'bold'), padx=5, pady=5, fg='black')
        btnCreateFolder = Button(saveProjectFolder, text="Create folder", command= self.unsupervisedCreate)

        # create dataset
        createDataset = LabelFrame(tab1, text="2.Create Dataset", font=("Helvetica", 12, 'bold'), padx=5, pady=5, fg='black')
        self.classifier = DropDownMenu(createDataset, 'Classifier name', classifierlist, '12')
        self.classifier.setChoices(classifierlist[0])
        btnGenerateDataset = Button(createDataset, text="Generate/save dataset", command=self.generate_dataset)

        # perform dimensionality reduction
        # managing dropdown menus
        drOptions = ['UMAP', 't-SNE', 'PCA']
        self.performDR = LabelFrame(tab3, text="3.Perform Dimensionality Reduction", font=("Helvetica", 12, 'bold'), padx=10,
                               pady=10, fg='black')
        self.importFeatureDataset = FileSelect(self.performDR, "Import feature dataset .pkl")
        self.algorithm = DropDownMenu(self.performDR,'Dimensionality Algorithm',drOptions,'15',com=self.selectAlgo)
        self.algorithm.setChoices(drOptions[0])

        self.frame = Frame(self.performDR)

        self.UMAPhyperparameters = LabelFrame(self.frame, text="UMAP Hyperparameters", padx=5, pady=5)
        self.distance1 = Entry_Box(self.UMAPhyperparameters, "Distance", '10')
        self.neighbors1 = Entry_Box(self.UMAPhyperparameters, "Neighbors", '10')
        self.spread1 = Entry_Box(self.UMAPhyperparameters, "Spread", '10')
        self.dimensionsUMAP = Entry_Box(self.UMAPhyperparameters, "Dimensions", '10')

        btnSaveDR = Button(self.performDR, text="Save dimensionality reduction .npy", command=self.perform_DR)

        # perform clustering
        performHDBSCAN = LabelFrame(tab4, text="4.Perform Clustering", font=("Helvetica", 12, 'bold'), padx=5, pady=10,fg='black')
        self.importDR = FileSelect(performHDBSCAN, "Select dimensionality reduction .npy")
        btnVisualizeScatter = Button(performHDBSCAN, text="Visualize/save HDBSCAN scatter plot",command=self.visualize_scatter)
        btnVisualizeTree = Button(performHDBSCAN, text="Visualize/save HDBSCAN tree plot", command=self.visualize_tree)
        btnSaveClusters = Button(performHDBSCAN, text="Save clusters .csv & .pkl", command=self.save_clusters)

        # train model
        trainModel = LabelFrame(tab5, text="5.Train Model", font=("Helvetica", 12, 'bold'), padx=10, pady=10, fg='black')
        self.importFeatureFile = FileSelect(trainModel, 'Import condensed dataset')
        self.importClusterFile = FileSelect(trainModel, 'Import cluster .pkl file')
        btnSavePermutation = Button(trainModel, text="Save permutational importances", command=self.save_permimportances)  # command setFolderPath
        btnSaveFeatures = Button(trainModel, text="Save feature correlations", command=self.save_featcorrelations)  # command setFolderPath

        # visualize clusters
        visualizeClusters = LabelFrame(tab6, text='6.Visualize Clusters', font=("Helvetica", 12, 'bold'), padx=10, pady=10, fg='black')
        self.importDataFile = FileSelect(visualizeClusters, 'Import clusters .csv file')
        self.importVideoName = Entry_Box(visualizeClusters, "Video name", '10')
        self.importVideoFolder = FolderSelect(visualizeClusters, "Import videos")
        self.importDatasetFolder = FolderSelect(visualizeClusters, "Import initial datasets")
        self.importAnimalHeaders = FileSelect(visualizeClusters, "Import animal headers .csv")
        btnSaveSkeletons = Button(visualizeClusters, text='Save skeleton clips', command=lambda: self.visualize_skeletons_videos('skeleton'))
        btnSaveVideoClips = Button(visualizeClusters, text='Save original video clips', command=lambda: self.visualize_skeletons_videos(video_type='original'))


        #organize
        saveProjectFolder.grid(row=0,sticky=W)
        btnCreateFolder.grid(row=0)

        createDataset.grid(row=1,pady=5,sticky=W)
        self.classifier.grid(row=1,sticky=W)
        btnGenerateDataset.grid(row=2,sticky=W)

        self.performDR.grid(row=0,sticky=W,pady=10)
        self.importFeatureDataset.grid(row=0,sticky=W)
        self.algorithm.grid(row=1,sticky=W)
        self.frame.grid(row=2,sticky=W)
        self.UMAPhyperparameters.grid(row=0,sticky=W)
        self.distance1.grid(row=1,sticky=W)
        self.neighbors1.grid(row=2,sticky=W)
        self.spread1.grid(row=3,sticky=W)
        self.dimensionsUMAP.grid(row=4,sticky=W)
        btnSaveDR.grid(row=3,pady=10)

        performHDBSCAN.grid(row=0,sticky=W)
        self.importDR.grid(row=0,sticky=W)
        btnVisualizeScatter.grid(row=1,pady=5,sticky=W)
        btnVisualizeTree.grid(row=2,pady=5,sticky=W)
        btnSaveClusters.grid(row=3,pady=5,sticky=W)

        trainModel.grid(row=0,sticky=W)
        self.importFeatureFile.grid(row=0,sticky=W)
        self.importClusterFile.grid(row=1,sticky=W)
        btnSavePermutation.grid(row=2,sticky=W)
        btnSaveFeatures.grid(row=3,sticky=W)

        visualizeClusters.grid(row=0,sticky=W)
        self.importDataFile.grid(row=0,sticky=W)
        self.importVideoName.grid(row=1,sticky=W)
        self.importVideoFolder.grid(row=2,sticky=W)
        self.importDatasetFolder.grid(row=3,sticky=W)
        self.importAnimalHeaders.grid(row=4,sticky=W)
        btnSaveSkeletons.grid(row=5,sticky=W)
        btnSaveVideoClips.grid(row=6,sticky=W)

    def selectAlgo(self,val):
        try:
            self.frame.destroy()
        except:
            pass

        if val == 'UMAP':
            self.frame = Frame(self.performDR)
            self.UMAPhyperparameters = LabelFrame(self.frame, text="UMAP Hyperparameters", padx=5, pady=5)
            self.distance1 = Entry_Box(self.UMAPhyperparameters, "Distance", '10')
            self.neighbors1 = Entry_Box(self.UMAPhyperparameters, "Neighbors", '10')
            self.spread1 = Entry_Box(self.UMAPhyperparameters, "Spread", '10')
            self.dimensionsUMAP = Entry_Box(self.UMAPhyperparameters, "Dimensions", '10')
            #organize
            self.frame.grid(row=2, sticky=W)
            self.UMAPhyperparameters.grid(row=0, sticky=W)
            self.distance1.grid(row=1, sticky=W)
            self.neighbors1.grid(row=2, sticky=W)
            self.spread1.grid(row=3, sticky=W)
            self.dimensionsUMAP.grid(row=4, sticky=W)

        elif val == 't-SNE':
            self.frame = Frame(self.performDR)
            self.tSNEhyperparameters = LabelFrame(self.frame, text="t-SNE Hyperparameters", padx=5, pady=5)
            self.perplexity1 = Entry_Box(self.tSNEhyperparameters, "Perplexity", '10')
            self.iterations1 = Entry_Box(self.tSNEhyperparameters, "Iterations", '10')
            self.dimensionstSNE = Entry_Box(self.tSNEhyperparameters, "Dimensions", '10')
            # organize
            self.frame.grid(row=2, sticky=W)
            self.tSNEhyperparameters.grid(row=0, sticky=W)
            self.perplexity1.grid(row=1, sticky=W)
            self.iterations1.grid(row=2, sticky=W)
            self.dimensionstSNE.grid(row=3, sticky=W)

        elif val == 'PCA':
            self.frame = Frame(self.performDR)
            self.PCAhyperparameters = LabelFrame(self.frame, text=" PCA Hyperparameters", padx=5, pady=5)
            self.nComponents1 = Entry_Box(self.PCAhyperparameters, 'n-components', '12')
            #organize
            self.frame.grid(row=2, sticky=W)
            self.PCAhyperparameters.grid(row=0, sticky=W)
            self.nComponents1.grid(row=1, sticky=W)

    def unsupervisedCreate(self):
        folder_list = ['dimensionality_reduction','dataset','clustering','train_model','visualize_clusters']

        if not os.path.exists(self.unsupervisedfolderpath):
            for i in folder_list:
                try:
                    os.makedirs(os.path.join(self.unsupervisedfolderpath,i))
                except FileExistsError:
                    pass
            write_unsupervisedini(self.unsupervisedfolderpath)
            print('Unsupervised project folder created.')
        else:
            print('Old unsupervised project detected, delete to create new project.')

    def generate_dataset(self):
        filesFolder = os.path.join(os.path.dirname(self.configini),'csv','machine_results')
        features2removeFile = askopenfilename(title="Select csv file that includes feature to remove",filetypes=(("csv files","*.csv"),("all files","*.*")))
        classifierName = self.classifier.getChoices()
        outputPath = os.path.join(self.unsupervisedfolderpath, 'dataset')

        filesFound = glob.glob(filesFolder + '/*.csv')
        counter = 0
        concatDf = pd.DataFrame()
        features2remove = pd.read_csv(features2removeFile)
        features2removeList = list(features2remove['Column_name'])

        # loops over all of the input files
        for file in filesFound:
            # extracts the file name
            currDf = pd.read_csv(file, index_col=0)
            groupDf = pd.DataFrame()
            # finds all of the bouts of attack where you have multiple 1s in a row, continuous behavior
            v = (currDf[classifierName] != currDf[classifierName].shift()).cumsum()
            u = currDf.groupby(v)[classifierName].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)
            groupDf['groups'] = currDf.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
            differenceList = []
            # extracting beginning and end of behavior sequence
            for row in groupDf.itertuples():
                difference = row[1][1] - row[1][0]
                differenceList.append(difference)
            groupDf['boutLength'] = differenceList
            frameWindowListStart, frameWindowListEnd = [], []
            for row in groupDf.itertuples():
                frameWindowListStart.append(int(row[1][0]))
                frameWindowListEnd.append(int(row[1][1]))
            currDf = currDf.drop(features2removeList, axis=1) # why did we comment out?
            for startFrame, endFrame in zip(frameWindowListStart, frameWindowListEnd):
                relRows = currDf.loc[startFrame:endFrame]
                meanVals = relRows.mean(axis=0)
                meanVals = pd.DataFrame(meanVals).transpose()
                meanVals['frame_no_start'] = startFrame
                meanVals['frame_no_end'] = endFrame
                meanVals['video'] = os.path.basename(file).replace('.csv', '')
                concatDf = pd.concat([concatDf, meanVals], axis=0)
                concatDf.reset_index(drop=True, inplace=True)
                counter += 1
                print('Processed ' + str(os.path.basename(file)) + ' ' + str(counter))
        # saving everything at pkl file (condensed csv)

        concatDf = concatDf.drop([classifierName], axis=1, errors='ignore')
        concatDf[classifierName] = classifierName
        concatDf.to_pickle(os.path.join(outputPath, classifierName + '.pkl'))
        print('Annotations for dimensionality reduction saved.')

    def perform_DR(self):

        file = self.importFeatureDataset.file_path
        s_behavior = os.path.basename(file).split('.pkl')[0]
        outputDirectory = os.path.join(self.unsupervisedfolderpath,'dimensionality_reduction')

        print("Reading pickled file...")
        concatDf = pd.read_pickle(file)

        concatDf = concatDf.loc[:, ~concatDf.columns.str.contains('^Unnamed')]
        # drop more columns here related to probability features to cluster, user should specify beforehand
        concatDf = concatDf.drop(
            ['Scaled_movement_M1', 'Scaled_movement_M2', 'Scaled_movement_M1_M2', 'Probability', 'Low_prob_detections_0.1',
             'Low_prob_detections_0.5', 'Low_prob_detections_0.75', 'Sum_probabilities', 'Sum_probabilities_deviation',
             'Sum_probabilities_deviation_percentile_rank', 'Sum_probabilities_deviation_rank',
             'Sum_probabilities_percentile_rank'], axis=1, errors='ignore')

        videoArray = concatDf.pop('video').values
        frameNoStart = concatDf.pop('frame_no_start').values
        frameNoEnd = concatDf.pop('frame_no_end').values
        behaviour = concatDf.pop(s_behavior).values

        featureArray = (concatDf - concatDf.min()) / (concatDf.max() - concatDf.min())

        print('Performing dimensionality reduction...')
        #print('UMAP distances =', distances)
        if (self.algorithm.getChoices() == 'UMAP'):
            # UMAP
            dist_string = self.distance1.entry_get
            distances = dist_string.split()

            neigh_string = self.neighbors1.entry_get
            neighbors = neigh_string.split()

            spread_string = self.spread1.entry_get
            spread = spread_string.split()

            UMAP_dim_string = self.dimensionsUMAP.entry_get
            UMAP_dimensions = UMAP_dim_string.split()

            for dist in distances:
                for neigh in neighbors:
                    for spr in spread:
                        for index, comps in enumerate(UMAP_dimensions):
                            print('Spread: ' + str(spr) + ' Distance: ' + str(dist) + ' Neighbours: ' + str(
                                neigh) + ' Dimensions: ' + str(comps))
                            umap_reducer = umap.UMAP(n_components=int(comps), n_neighbors=int(neigh), spread=float(spr), min_dist=float(dist),
                                                     metric='euclidean', verbose=True)
                            umap_embedding = umap_reducer.fit_transform(featureArray)
                            ############### SAVE DATA #####################
                            outputNp = np.c_[
                                umap_embedding, videoArray, frameNoStart, frameNoEnd, behaviour]  # took out group and sex
                            npyName = os.path.join(outputDirectory, 'UMAP_reduced_features_spread_' + s_behavior + '_'+ str(
                                spr) + '_neighbors_' + str(neigh) + '_dist_' + str(dist) + '_dimensions_' + str(
                                comps) + '.npy')
                            np.save(npyName, outputNp)
                            print('Saved ' + str(npyName))
                            print('All UMAPs saved.')
                            currDf = pd.DataFrame(data=outputNp, columns=['X', 'Y', 'Video', 'Frame_Start', 'Frame_End',
                                                                          'Behavior'])  # took out Sex and Group
                            ax = sns.scatterplot(x="X", y="Y", data=currDf, s=10, palette='Set1')
                            ax.set_title('Spread: ' + str(spr) + ' Distance: ' + str(dist) + ' Neighbours: ' + str(
                                neigh) + ' Dimensions: ' + str(comps), fontsize=10)
                            # umap.plot.points(reducer) # UMAPs default visualization, doesn't detail
                            #plt.show()
                            pngName = os.path.join(outputDirectory, 'UMAP_reduced_features_spread_' +s_behavior + '_'+ str(
                                spr) + '_neighbors_' + str(neigh) + '_dist_' + str(dist) + '_dimensions_' + str(
                                comps) + '.png')
                            plt.savefig(pngName)

        elif (self.algorithm.getChoices() == 't-SNE'):
            # tSNE
            perp_string = self.perplexity1.entry_get
            perplexity = perp_string.split()
            iter_string = self.iterations1.entry_get
            iterations = iter_string.split()

            TSNE_dim_string = self.dimensionstSNE.entry_get
            TSNE_dimensions = TSNE_dim_string.split()
            for perp in perplexity:
                for iter in iterations:
                    for index, comps in enumerate(TSNE_dimensions):
                        print('Perplexity: ' + str(perp) + ' Iterations: ' + str(iter) + ' Dimensions: ' + str(comps))
                        tsne_reducer = TSNE(n_components=int(comps), perplexity=int(perp), n_iter=int(iter))
                        tsne_embedding = tsne_reducer.fit_transform(featureArray)
                        outputNp = np.c_[
                            tsne_embedding, videoArray, frameNoStart, frameNoEnd, behaviour]  # took out group and sex
                        npyName = os.path.join(outputDirectory, 't-SNE_reduced_features_perplexity_' + s_behavior + '_'+str(
                            perp) + '_iterations_' + str(iter) + '_dimensions_' + str(comps) + '.npy')
                        np.save(npyName, outputNp)
                        print('Saved ' + str(npyName))
                        print('All t-SNEs saved.')
                        currDf = pd.DataFrame(data=outputNp, columns=['X', 'Y', 'Video', 'Frame_Start', 'Frame_End',
                                                                      'Behavior'])  # took out Sex and Group
                        plt.figure(figsize=(16, 10))
                        ax = sns.scatterplot(x="X", y="Y", palette=sns.color_palette("Set2"), data=currDf, legend="full",
                                             alpha=0.3)
                        ax.set_title(
                            'Perplexity: ' + str(perp) + ' Iterations: ' + str(iter) + ' Dimensions: ' + str(comps))
                        pngName = os.path.join(outputDirectory, 't-SNE_reduced_features_perplexity_' +s_behavior + '_'+ str(
                            perp) + '_iterations_' + str(iter) + '_dimensions_' + str(comps) + '.png')
                        plt.savefig(pngName)

        elif (self.algorithm.getChoices()  == 'PCA'):
            pca_reducer = PCA(n_components=int(self.nComponents1.entry_get))
            pca_embedding = pca_reducer.fit_transform(featureArray)
            outputNp = np.c_[pca_embedding, videoArray, frameNoStart, frameNoEnd, behaviour]  # took out group and sex
            npyName = os.path.join(outputDirectory, 'PCA_reduced_features' + s_behavior + '_' + '.npy')
            np.save(npyName, outputNp)
            print('Saved ' + str(npyName))
            print('All PCAs saved.')
            currDf = pd.DataFrame(data=outputNp, columns=['X', 'Y', 'Video', 'Frame_Start', 'Frame_End', 'Behavior'])
            plt.figure(figsize=(16, 10))
            ax = sns.scatterplot(x="X", y="Y", palette=sns.color_palette("hls", 10), data=currDf, legend="full", alpha=0.3)
            ax.set_title('PCA_reduced_features_after_normalization')
            #plt.show()
            pngName = os.path.join(outputDirectory, 'PCA_reduced_features' + '.png')
            plt.savefig(pngName)

    def visualize_scatter(self):
        fileName = self.importDR.file_path
        outputFolder = os.path.join(self.unsupervisedfolderpath, 'clustering')

        dataNp = np.load(fileName, allow_pickle=True)

        drFeatures = np.delete(dataNp, np.s_[-1], axis=1)
        drFeatures = np.delete(drFeatures, np.s_[2], axis=1)

        # videoArray = np.delete(dataNp, [0,1,3,4,5,6], axis=1)
        # frameArray = np.delete(dataNp, [0,1,2,4,5,6], axis=1).astype('float32')
        # sexArray = np.delete(dataNp, [0,1,2,3,5,6], axis=1).astype('str')
        # groupArray = np.delete(dataNp, [0,1,2,3,4,6], axis=1).astype('str')
        # behavior = np.delete(dataNp, [0,1,2,3,4,5], axis=1).astype('float32')

        min_cluster_size = int(0.10 * len(drFeatures))  # set it as percentage of entire dataset, 10% length

        HDBSCAN_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                            gen_min_span_tree=True,
                                            leaf_size=10, metric='euclidean', min_cluster_size=min_cluster_size,
                                            min_samples=2,
                                            p=None)
        HDBscanResults = HDBSCAN_clusterer.fit_predict(drFeatures)
        np.unique(HDBscanResults)
        np.count_nonzero(HDBscanResults == -1)

        ###PLOT 2DHDBSCAN SCATTER

        # for every attack bout you have x,y location
        color_palette = sns.color_palette('Paired', len(np.unique(HDBscanResults)))
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in HDBSCAN_clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, HDBSCAN_clusterer.probabilities_)]
        X_dim = drFeatures[:, [0]].astype('float32')
        Y_dim = drFeatures[:, [1]].astype('float32')
        plt.title('Spread_1_Neigh_8_dist_0.1')  # take in combo name, currently hard-coded
        plt.scatter(X_dim, Y_dim, s=10, linewidth=0, c=cluster_member_colors, alpha=1)
        #plt.show()
        pngName = os.path.join(outputFolder, 'HDBSCAN_scatter.png')
        plt.savefig(pngName)
        # saves another column in the dataset
        print(np.unique(HDBscanResults))

    def visualize_tree(self):
        fileName = self.importDR.file_path
        outputFolder = os.path.join(self.unsupervisedfolderpath, 'clustering')

        dataNp = np.load(fileName, allow_pickle=True)

        drFeatures = np.delete(dataNp, np.s_[-1], axis=1)
        drFeatures = np.delete(drFeatures, np.s_[2], axis=1)

        # videoArray = np.delete(dataNp, [0,1,3,4,5,6], axis=1)
        # frameArray = np.delete(dataNp, [0,1,2,4,5,6], axis=1).astype('float32')
        # sexArray = np.delete(dataNp, [0,1,2,3,5,6], axis=1).astype('str')
        # groupArray = np.delete(dataNp, [0,1,2,3,4,6], axis=1).astype('str')
        # behavior = np.delete(dataNp, [0,1,2,3,4,5], axis=1).astype('float32')

        min_cluster_size = int(0.10 * len(drFeatures))  # set it as percentage of entire dataset, 10% length

        HDBSCAN_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                            gen_min_span_tree=True,
                                            leaf_size=10, metric='euclidean', min_cluster_size=min_cluster_size,
                                            min_samples=2,
                                            p=None)
        HDBscanResults = HDBSCAN_clusterer.fit_predict(drFeatures)
        np.unique(HDBscanResults)
        np.count_nonzero(HDBscanResults == -1)

        ###PLOT TREE HDBSCAN
        HDBSCAN_clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        #plt.show()
        pngName = os.path.join(outputFolder, 'HDBSCAN_tree.png')
        plt.savefig(pngName)

        # #
        # ###HDBSCAN TO PANDAS
        # clusturerDf = clusterer.condensed_tree_.to_pandas()
        # clusturerDf.to_csv('test.csv')

    def save_clusters(self):
        fileName = self.importDR.file_path
        outputFolder = os.path.join(self.unsupervisedfolderpath, 'clustering')

        dataNp = np.load(fileName, allow_pickle=True)

        drFeatures = np.delete(dataNp, np.s_[-1], axis=1)
        drFeatures = np.delete(drFeatures, np.s_[2], axis=1)

        # videoArray = np.delete(dataNp, [0,1,3,4,5,6], axis=1)
        # frameArray = np.delete(dataNp, [0,1,2,4,5,6], axis=1).astype('float32')
        # sexArray = np.delete(dataNp, [0,1,2,3,5,6], axis=1).astype('str')
        # groupArray = np.delete(dataNp, [0,1,2,3,4,6], axis=1).astype('str')
        # behavior = np.delete(dataNp, [0,1,2,3,4,5], axis=1).astype('float32')

        min_cluster_size = int(0.10 * len(drFeatures))  # set it as percentage of entire dataset, 10% length

        HDBSCAN_clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                            gen_min_span_tree=True,
                                            leaf_size=10, metric='euclidean', min_cluster_size=min_cluster_size,
                                            min_samples=2,
                                            p=None)
        HDBscanResults = HDBSCAN_clusterer.fit_predict(drFeatures)
        np.unique(HDBscanResults)
        np.count_nonzero(HDBscanResults == -1)

        # saves output as a pkl
        ###HDBSCAN TO PICKLE
        outputNp = np.c_[dataNp, HDBscanResults]
        outputDf = pd.DataFrame(
            {'X_dim': outputNp[:, 0], 'Y_dim': outputNp[:, 1], 'Video': outputNp[:, 2], 'Frame_start': outputNp[:, 3],
             'Frame_end': outputNp[:, 4], 'Behavior': outputNp[:, 5], 'Cluster': outputNp[:, 6]})
        outputDf[["X_dim", "Y_dim"]] = outputDf[["X_dim", "Y_dim"]].apply(pd.to_numeric)
        outputPath1 = os.path.join(outputFolder, 'HDBscan_bouts.pkl')
        outputPath2 = os.path.join(outputFolder, 'HDBscan_bouts.csv')
        outputDf.to_pickle(outputPath1)
        outputDf.to_csv(outputPath2)

    def save_permimportances(self):
        featureFile = self.importFeatureFile.file_path
        clusterFile = self.importClusterFile.file_path
        outputPath = os.path.join(self.unsupervisedfolderpath, 'train_model')

        clusterFile = np.load(clusterFile, allow_pickle=True)
        featureFile = pd.read_pickle(featureFile)
        # clusterFile = pd.read_pickle(clusterFile)
        clusterFile_2 = clusterFile[clusterFile['Cluster'] == 0]

        # figure out why the algorithm clustered it like this, difference between feature values
        clusterCol = clusterFile['Cluster']
        # featureFile = featureFile[featureFile['group'] == 'Female']
        featureFile = featureFile.reset_index(drop=True)
        X_dim = clusterFile_2.pop('X_dim').values
        Y_dim = clusterFile_2.pop('Y_dim').values
        FrameStart = clusterFile.pop('Frame_start').values
        FrameEnd = clusterFile.pop('Frame_end').values
        video = clusterFile_2.pop('Video').values

        featureFile = featureFile.drop(
            ['Attack', 'frame_no_start', 'frame_no_end', 'video', 'group', 'sex', 'Scaled_movement_M1',
             'Scaled_movement_M2', 'Scaled_movement_M1_M2', 'Probability_Attack', 'Low_prob_detections_0.1',
             'Low_prob_detections_0.5', 'Low_prob_detections_0.75', 'Sum_probabilities', 'Sum_probabilities_deviation',
             'Sum_probabilities_deviation_percentile_rank', 'Sum_probabilities_deviation_rank',
             'Sum_probabilities_percentile_rank'], axis=1, errors='ignore')
        featureFile['Cluster'] = clusterCol

        featureFile = featureFile[featureFile['Cluster'] != -1]

        clustersList = list(featureFile.Cluster.unique())

        def pearson_filter(featuresDf, del_corr_threshold, del_corr_plot_status):
            print('Reducing features. Correlation threshold: ' + str(del_corr_threshold))
            col_corr = set()
            corr_matrix = featuresDf.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= del_corr_threshold) and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)
                        if colname in featuresDf.columns:
                            del featuresDf[colname]
                            print(colname)
            # if del_corr_plot_status == 'yes':
            #     print('Creating feature correlation heatmap...')
            #     dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
            #     plt.matshow(featuresDf.corr())
            #     plt.tight_layout()
            #     plt.savefig(os.path.join(outputPath, 'Feature_correlations_' + dateTime + '.png'), dpi=300)
            #     plt.close('all')
            #     print('Feature correlation heatmap .png saved in project_folder/logs directory')

            return featuresDf


        for cluster in range(len(clustersList)):
            currentTargetDf = featureFile[featureFile['Cluster'] == clustersList[cluster]]
            currentTargetDf['Cluster'] = 1
            currentTargetDf = currentTargetDf.reset_index(drop=True)
            # currentNonTargetDf = featureFile[featureFile['Cluster'] == clustervalue2[cluster]]
            currentNonTargetDf = featureFile.loc[featureFile['Cluster'].isin(clustersList)]
            currentNonTargetDf = currentNonTargetDf.drop(['Cluster'], axis=1, errors='ignore')
            currentNonTargetDf['Cluster'] = 0
            currentNonTargetDf = currentNonTargetDf.reset_index(drop=True)
            features = pd.concat([currentTargetDf, currentNonTargetDf])
            targetFrameRows = features.loc[features['Cluster'] == 1]
            targetFrame = features.pop('Cluster').values

            # for cluster in clustersList:
            # currentTargetDf = featureFile[featureFile['Cluster'] == clustersList[cluster]]

            features = pearson_filter(features, 0.70, 'yes')

            # somehow, random forest models generate tables that calculate permutational importances for each feature

            # features = (features-features.min())/(features.max()-features.min())
            feature_list = list(features)
            data_train, data_test, target_train, target_test = train_test_split(features, targetFrame, test_size=0.20)

            clf = RandomForestClassifier(n_estimators=2000, max_features='sqrt', n_jobs=-1, criterion='entropy',
                                         min_samples_leaf=1, bootstrap=True, verbose=1)
            clf.fit(data_train, target_train)


        # PERFORM GINI IMPORTANCES
        importances = list(clf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feature_importance_list = [('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        feature_importance_list_varNm = [i.split(':' " ", 3)[1] for i in feature_importance_list]
        feature_importance_list_importance = [i.split(':' " ", 3)[2] for i in feature_importance_list]
        log_df = pd.DataFrame()
        log_df['Feature_name'] = feature_importance_list_varNm
        log_df['Feature_importance'] = feature_importance_list_importance
        savePath = os.path.join(outputPath, 'feature_importance_log_2.csv')
        log_df.to_csv(savePath)

    def save_featcorrelations(self):

        featureFile = self.importFeatureFile.file_path
        clusterFile = self.importClusterFile.file_path
        outputPath = os.path.join(self.unsupervisedfolderpath, 'train_model')

        clusterFile = np.load(clusterFile, allow_pickle=True)
        featureFile = pd.read_pickle(featureFile)
        # clusterFile = pd.read_pickle(clusterFile)
        clusterFile_2 = clusterFile[clusterFile['Cluster'] == 0]

        # figure out why the algorithm clustered it like this, difference between feature values
        clusterCol = clusterFile['Cluster']
        # featureFile = featureFile[featureFile['group'] == 'Female']
        featureFile = featureFile.reset_index(drop=True)
        X_dim = clusterFile_2.pop('X_dim').values
        Y_dim = clusterFile_2.pop('Y_dim').values
        FrameStart = clusterFile.pop('Frame_start').values
        FrameEnd = clusterFile.pop('Frame_end').values
        video = clusterFile_2.pop('Video').values

        featureFile = featureFile.drop(
            ['Attack', 'frame_no_start', 'frame_no_end', 'video', 'group', 'sex', 'Scaled_movement_M1',
             'Scaled_movement_M2', 'Scaled_movement_M1_M2', 'Probability_Attack', 'Low_prob_detections_0.1',
             'Low_prob_detections_0.5', 'Low_prob_detections_0.75', 'Sum_probabilities', 'Sum_probabilities_deviation',
             'Sum_probabilities_deviation_percentile_rank', 'Sum_probabilities_deviation_rank',
             'Sum_probabilities_percentile_rank'], axis=1, errors='ignore')
        featureFile['Cluster'] = clusterCol

        featureFile = featureFile[featureFile['Cluster'] != -1]

        clustersList = list(featureFile.Cluster.unique())

        def pearson_filter(featuresDf, del_corr_threshold, del_corr_plot_status):
            print('Reducing features. Correlation threshold: ' + str(del_corr_threshold))
            col_corr = set()
            corr_matrix = featuresDf.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= del_corr_threshold) and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)
                        if colname in featuresDf.columns:
                            del featuresDf[colname]
                            print(colname)
            if del_corr_plot_status == 'yes':
                print('Creating feature correlation heatmap...')
                dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
                plt.matshow(featuresDf.corr())
                plt.tight_layout()
                plt.savefig(os.path.join(outputPath, 'Feature_correlations_' + dateTime + '.png'), dpi=300)
                plt.close('all')
                print('Feature correlation heatmap .png saved in project_folder/logs directory')

            return featuresDf


        for cluster in range(len(clustersList)):
            currentTargetDf = featureFile[featureFile['Cluster'] == clustersList[cluster]]
            currentTargetDf['Cluster'] = 1
            currentTargetDf = currentTargetDf.reset_index(drop=True)
            # currentNonTargetDf = featureFile[featureFile['Cluster'] == clustervalue2[cluster]]
            currentNonTargetDf = featureFile.loc[featureFile['Cluster'].isin(clustersList)]
            currentNonTargetDf = currentNonTargetDf.drop(['Cluster'], axis=1, errors='ignore')
            currentNonTargetDf['Cluster'] = 0
            currentNonTargetDf = currentNonTargetDf.reset_index(drop=True)
            features = pd.concat([currentTargetDf, currentNonTargetDf])
            targetFrameRows = features.loc[features['Cluster'] == 1]
            targetFrame = features.pop('Cluster').values

            # for cluster in clustersList:
            # currentTargetDf = featureFile[featureFile['Cluster'] == clustersList[cluster]]

            features = pearson_filter(features, 0.70, 'yes')

            # somehow, random forest models generate tables that calculate permutational importances for each feature

            # features = (features-features.min())/(features.max()-features.min())
            feature_list = list(features)
            data_train, data_test, target_train, target_test = train_test_split(features, targetFrame, test_size=0.20)

            clf = RandomForestClassifier(n_estimators=2000, max_features='sqrt', n_jobs=-1, criterion='entropy',
                                         min_samples_leaf=1, bootstrap=True, verbose=1)
            clf.fit(data_train, target_train)

    def visualize_skeletons_videos(self,video_type):
        plot_skeleton = 'yes'
        save_frames = 'yes'
        videoFn = self.importVideoName.entry_get  # main example of all 3 clusters
        video = video_type

        # can't we just import the video file?
        dataFilePath = self.importDataFile.file_path
        videoFile = os.path.join(self.importVideoFolder.folder_path, videoFn + '.mp4')
        outputFolder = os.path.join(self.unsupervisedfolderpath, 'visualize_clusters')
        # skeletonDfpath = r"Z:\DeepLabCut\misc\UMAP\2_UMAP_091020\1_pkl_data_files\Bouts_attack_no_size_more_SA.pkl"
        animalHeaders = self.importAnimalHeaders.file_path
        animal1HeadersDf = pd.read_csv(animalHeaders)
        animal1Headers = list(animal1HeadersDf['Animal_1'])
        animal2Headers = list(animal1HeadersDf['Animal_2'])

        dataFile = pd.read_csv(dataFilePath, index_col=0)
        dataFile = dataFile[dataFile['Video'] == os.path.basename(videoFile).replace('.mp4', '')]
        cap = cv2.VideoCapture(videoFile)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height, noFrames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clusterVals = list(dataFile['Cluster'].unique())
        if plot_skeleton == 'yes':
            skeletonDf = pd.read_csv(os.path.join(self.importDatasetFolder.folder_path, videoFn + '.csv'), index_col=0)

        if save_frames == 'yes':
            dirPath = os.path.join(outputFolder, videoFn)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

        for currCluster in clusterVals:
            if save_frames == 'yes':
                dirPathCluster = os.path.join(outputFolder, videoFn, 'Cluster_' + str(currCluster))
                if not os.path.exists(dirPathCluster):
                    os.makedirs(dirPathCluster)

            cap = cv2.VideoCapture(videoFile)
            vidBasename = os.path.basename(videoFile).replace('.mp4', '.avi')
            outputDf = pd.DataFrame(columns=['Cluster'])
            outputDf['Cluster'] = [0] * noFrames
            currDf = dataFile[dataFile['Cluster'] == currCluster]
            currDf = currDf[['Frame_start', 'Frame_end']]
            for index, row in currDf.iterrows():
                frameList = list(range(row['Frame_start'], row['Frame_end']))
                for frame in frameList:
                    outputDf['Cluster'][frame] = 1

            outDf = pd.DataFrame(columns=['Start', 'End'])
            groupDf = pd.DataFrame()
            v = (outputDf['Cluster'] != outputDf['Cluster'].shift()).cumsum()
            u = outputDf.groupby(v)['Cluster'].agg(['all', 'count'])
            m = u['all'] & u['count'].ge(1)
            groupDf['groups'] = outputDf.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
            for row in groupDf.itertuples():
                start, end = row[1][0] - 30, row[1][1] + 30
                if start < 0: start = 0
                if end > len(outputDf): end = len(outputDf)
                appendList = [start, end]
                outDf.loc[len(outDf)] = appendList

            clusterCounter = 0
            for index, row in outDf.iterrows():
                clusterCounter += 1
                if save_frames == 'yes':
                    dirPathClusterExample = os.path.join(dirPathCluster, 'Example_' + str(clusterCounter))
                    if not os.path.exists(dirPathClusterExample):
                        os.makedirs(dirPathClusterExample)
                frames = list(range(row['Start'], row['End'] + 1))
                behaviorStart, behaviorEnd = frames[0] + 30, frames[-1] - 30
                frameCounter = 0
                if video == 'original':
                    outputPath1 = os.path.join(outputFolder,
                                               'Cluster_' + str(currCluster) + '_OriginalClip_#' + str(
                                                   index) + '_' + vidBasename)
                    writer1 = cv2.VideoWriter(outputPath1, fourcc, 30, (width, height))
                    cap = cv2.VideoCapture(videoFile)
                    for frameNo in frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
                        ret, frame = cap.read()
                        cv2.imshow('frame', frame)  # need to fix so width and height are 0
                        cv2.waitKey(33)
                        writer1.write(frame)
                    cv2.destroyAllWindows()
                    writer1.release()
                    print('Video ' + str(outputPath1) + ' created.')

                else:
                    for frameNo in frames:
                        outputPath2 = os.path.join(outputFolder,
                                                   'Cluster_' + str(currCluster) + '_Skeletons_' + vidBasename)
                        writer2 = cv2.VideoWriter(outputPath2, fourcc, 30, (width, height))
                        frameCounter += 1
                        currAnimal1 = list(skeletonDf.loc[skeletonDf.index[frameNo], animal1Headers])
                        currAnimal1 = np.array([currAnimal1[i:i + 2] for i in range(0, len(currAnimal1), 2)]).astype(int)
                        currAnimal2 = list(skeletonDf.loc[skeletonDf.index[frameNo], animal2Headers])
                        currAnimal2 = np.array([currAnimal2[i:i + 2] for i in range(0, len(currAnimal2), 2)]).astype(int)
                        currFrame = np.zeros((height, width, 3), dtype="uint8")
                        currAnimal1_hull = cv2.convexHull((currAnimal1.astype(int)))
                        currAnimal2_hull = cv2.convexHull((currAnimal2.astype(int)))

                        cv2.drawContours(currFrame, [currAnimal1_hull.astype(int)], 0, (255, 255, 255), 1)
                        cv2.drawContours(currFrame, [currAnimal2_hull.astype(int)], 0, (0, 255, 0), 1)
                        for anim1, anim2 in zip(currAnimal1, currAnimal2):
                            cv2.circle(currFrame, (int(anim1[0]), int(anim1[1])), 4, (147, 20, 255), thickness=-1,
                                       lineType=8, shift=0)
                            cv2.circle(currFrame, (int(anim2[0]), int(anim2[1])), 4, (0, 255, 255), thickness=-1,
                                       lineType=8, shift=0)

                        cv2.line(currFrame, (currAnimal1[0][0], currAnimal1[0][1]), (currAnimal1[1][0], currAnimal1[1][1]),
                                 (0, 0, 255), 2)
                        cv2.line(currFrame, (currAnimal2[0][0], currAnimal2[0][1]), (currAnimal2[1][0], currAnimal2[1][1]),
                                 (0, 0, 255), 2)
                        cv2.line(currFrame, (currAnimal1[4][0], currAnimal1[4][1]), (currAnimal1[5][0], currAnimal1[5][1]),
                                 (0, 0, 255), 2)
                        cv2.line(currFrame, (currAnimal2[4][0], currAnimal2[4][1]), (currAnimal2[5][0], currAnimal2[5][1]),
                                 (0, 0, 255), 2)
                        cv2.line(currFrame, (currAnimal1[2][0], currAnimal1[2][1]), (currAnimal1[6][0], currAnimal1[6][1]),
                                 (0, 0, 255), 2)
                        cv2.line(currFrame, (currAnimal2[2][0], currAnimal2[2][1]), (currAnimal2[6][0], currAnimal2[6][1]),
                                 (0, 0, 255), 2)

                        if (frameNo < behaviorStart):
                            cv2.putText(currFrame, str('Cluster ' + str(currCluster)) + ' coming up...', (30, 30),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
                        elif (frameNo > behaviorEnd):
                            cv2.putText(currFrame, str('Cluster ' + str(currCluster)) + ' happened.', (30, 30),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
                        else:
                            cv2.putText(currFrame, str(currCluster) + '!!', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                                        (0, 255, 0), 1)

                        cv2.imshow('frame', currFrame)
                        cv2.waitKey(3)  # pauses for 3 seconds before fetching next image

                        if save_frames == 'yes':
                            filename = os.path.join(dirPathClusterExample, str(frameCounter) + '.png')
                            cv2.imwrite(filename, currFrame)

                        writer2.write(currFrame)

                    for i in range(30):
                        blueFrame = np.zeros((currFrame.shape[0], currFrame.shape[1], 3))
                        blueFrame[:] = (255, 0, 0)
                        blueFrame = blueFrame.astype(np.uint8)
                        writer2.write(blueFrame)

            print('saved')
            cap.release()

        currRow = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                currVals = clusterFramesDf.loc[clusterFramesDf['Frame'] == currRow]
                if currVals.empty:
                    cv2.putText(frame, str('Cluster: None'), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    clustervalue = list(currVals['Cluster'])
                    cv2.putText(frame, str(clustervalue[0]), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

                frame = np.uint8(frame)
                cv2.imshow('image', frame)
                cv2.waitKey(30)
                # cv2.destroyAllWindows()
                writer.write(frame)
            if frame is None:
                print('Video saved.')
                cap.release()
                break
            currRow += 1
            print(currRow)

class trainmachinemodel_settings:
    def __init__(self,inifile):
        self.configini = str(inifile)
        # Popup window
        trainmmsettings = Toplevel()
        trainmmsettings.minsize(400, 400)
        trainmmsettings.wm_title("Machine model settings")

        trainmms = Canvas(hxtScrollbar(trainmmsettings))
        trainmms.pack(expand=True,fill=BOTH)

        #load metadata
        load_data_frame = LabelFrame(trainmms, text='Load Metadata',font=('Helvetica',10,'bold'), pady=5, padx=5)
        self.load_choosedata = FileSelect(load_data_frame,'File Select',title='Select a meta (.csv) file')
        load_data = Button(load_data_frame, text = 'Load', command = self.load_RFvalues,fg='blue')

        #link to github
        label_git_hyperparameters = Label(trainmms,text='[Click here to learn about the Hyperparameters]',cursor='hand2',fg='blue')
        label_git_hyperparameters.bind('<Button-1>',lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-7-train-machine-model'))

        #setting drop downs
        label_mm = LabelFrame(trainmms, text='Machine model',font=('Helvetica',10,'bold'), pady=5, padx=5)
        label_choosemm = Label(label_mm, text='Choose machine model:')
        options =['RF','GBC','Xboost']

        self.var = StringVar()
        self.var.set(options[0]) #set as default value

        modeldropdown = OptionMenu(label_mm,self.var,*options)

        self.meta_dict = {}
        ## hyperparameter settings
        label_settings = LabelFrame(trainmms, text='Hyperparameters',font=('Helvetica',10,'bold'),pady=5,padx=5)
        self.settings = []
        self.label_nestimators = Entry_Box(label_settings,'RF N Estimators','25')
        self.label_maxfeatures = Entry_Box(label_settings,'RF Max features','25')
        self.label_criterion = Entry_Box(label_settings,'RF Criterion','25')
        self.label_testsize = Entry_Box(label_settings,'Train Test Size','25')
        self.label_minsampleleaf = Entry_Box(label_settings,'RF Min sample leaf','25')
        self.label_under_s_settings = Entry_Box(label_settings, 'Under sample setting', '25')
        self.label_under_s_correctionvalue = Entry_Box(label_settings,'Under sample ratio','25')
        self.label_over_s_settings = Entry_Box(label_settings, 'Over sample setting', '25')
        self.label_over_s_ratio = Entry_Box(label_settings,'Over sample ratio','25')

        self.settings = [self.label_nestimators, self.label_maxfeatures, self.label_criterion, self.label_testsize,
                    self.label_minsampleleaf, self.label_under_s_correctionvalue,self.label_under_s_settings,
                         self.label_over_s_ratio,self.label_over_s_settings]
        ## model evaluation settings for checkboxes
        self.label_settings_box = LabelFrame(trainmms,pady=5,padx=5,text='Model Evaluations Settings',font=('Helvetica',10,'bold'))
        self.box1 = IntVar()
        self.box2 = IntVar()
        self.box3 = IntVar()
        self.box4 = IntVar()
        self.box5 = IntVar()
        self.box6 = IntVar()
        self.box7 = IntVar()
        self.box8 = IntVar()
        self.box9 = IntVar()
        self.box10 = IntVar()

        # model evaluations for entrybox
        self.LC_ksplit = Entry_Box(self.label_settings_box, 'LearningCurve shuffle K splits', '25',status=DISABLED)
        self.LC_datasplit = Entry_Box(self.label_settings_box, 'LearningCurve shuffle Data splits', '25',status=DISABLED)
        self.label_n_feature_importance_bars = Entry_Box(self.label_settings_box, 'N feature importance bars', '25',status=DISABLED)
        self.shap_present = Entry_Box(self.label_settings_box,'# target present', '25',status=DISABLED)
        self.shap_absent = Entry_Box(self.label_settings_box, '# target absent', '25', status=DISABLED)
        self.settings.extend([self.LC_ksplit, self.LC_datasplit, self.label_n_feature_importance_bars,self.shap_present,self.shap_absent])

        def activate(box, *args):
            for entry in args:
                if box.get() == 0:
                    entry.set_state(DISABLED)
                elif box.get() == 1:
                    entry.set_state(NORMAL)

        checkbutton1 = Checkbutton(self.label_settings_box,text='Generate RF model meta data file',variable = self.box1)
        checkbutton2 = Checkbutton(self.label_settings_box, text='Generate Example Decision Tree (requires "graphviz")', variable=self.box2)
        checkbutton3 = Checkbutton(self.label_settings_box, text='Generate Fancy Example Decision Tree ("dtreeviz")', variable=self.box3)
        checkbutton4 = Checkbutton(self.label_settings_box, text='Generate Classification Report', variable=self.box4)
        # checkbutton5 = Checkbutton(self.label_settings_box, text='Generate Features Importance Log', variable=self.box5)
        checkbutton6 = Checkbutton(self.label_settings_box, text='Generate Features Importance Bar Graph', variable=self.box6,
                                   command = lambda:activate(self.box6, self.label_n_feature_importance_bars))
        checkbutton7 = Checkbutton(self.label_settings_box, text='Compute Feature Permutation Importances (Note: CPU intensive)', variable=self.box7)
        checkbutton8 = Checkbutton(self.label_settings_box, text='Generate Sklearn Learning Curves (Note: CPU intensive)', variable=self.box8,
                                   command = lambda:activate(self.box8, self.LC_datasplit, self.LC_ksplit))
        checkbutton9 = Checkbutton(self.label_settings_box, text='Generate Precision Recall Curves', variable=self.box9)
        checkbutton10 = Checkbutton(self.label_settings_box,text='Calculate SHAP scores',variable=self.box10,command= lambda:activate(self.box10,self.shap_present,self.shap_absent))

        self.check_settings = [checkbutton1, checkbutton2, checkbutton3, checkbutton4, checkbutton6,
                               checkbutton7, checkbutton8, checkbutton9, checkbutton10]


        # setting drop downs for modelname
        configini = self.configini
        config = ConfigParser()
        config.read(configini)

        number_of_model = config['SML settings'].getint('No_targets')

        model_list = []
        count = 1
        for i in range(number_of_model):
            a = str('target_name_' + str(count))
            model_list.append(config['SML settings'].get(a))
            count += 1

        labelf_modelname = LabelFrame(trainmms,text='Model',font=('Helvetica',10,'bold'),pady=5,padx=5)
        label_modelname = Label(labelf_modelname,text='Model name')

        self.varmodel = StringVar()
        self.varmodel.set(model_list[0])  # set as default value

        model_name_dropdown = OptionMenu(labelf_modelname, self.varmodel, *model_list)

        # button
        button_settings_to_ini = Button(trainmms, text='Save settings into global environment', font=('Helvetica', 10, 'bold'),fg='blue', command=self.set_values)
        button_save_meta = Button(trainmms, text='Save settings for specific model', font=('Helvetica', 10, 'bold'),fg='green' ,command=self.save_new)
        button_remove_meta = Button(trainmms,text='Clear cache',font=('Helvetica', 10, 'bold'),fg='red',command = self.clearcache)

        # organize
        load_data_frame.grid(row=0, sticky=W, pady=5, padx=5)
        self.load_choosedata.grid(row=0, column=0, sticky=W)
        load_data.grid(row=1, column=0)

        label_mm.grid(row=1, sticky=W, pady=5)
        label_choosemm.grid(row=0, column=0, sticky=W)
        modeldropdown.grid(row=0, column=1, sticky=W)

        labelf_modelname.grid(row=2, sticky=W, pady=5)
        label_modelname.grid(row=0, column=0, sticky=W)
        model_name_dropdown.grid(row=0, column=1, sticky=W)

        label_git_hyperparameters.grid(row=3,sticky=W)

        label_settings.grid(row=4, sticky=W, pady=5)
        self.label_nestimators.grid(row=1, sticky=W)
        self.label_maxfeatures.grid(row=2, sticky=W)
        self.label_criterion.grid(row=3, sticky=W)
        self.label_testsize.grid(row=4, sticky=W)
        self.label_minsampleleaf.grid(row=7, sticky=W)
        self.label_under_s_settings.grid(row=8, sticky=W)
        self.label_under_s_correctionvalue.grid(row=9, sticky=W)
        self.label_over_s_settings.grid(row=10, sticky=W)
        self.label_over_s_ratio.grid(row=11,sticky=W)

        self.label_settings_box.grid(row=5,sticky=W)
        checkbutton1.grid(row=0,sticky=W)
        checkbutton2.grid(row=1,sticky=W)
        checkbutton3.grid(row=2,sticky=W)
        checkbutton4.grid(row=3,sticky=W)
        # checkbutton5.grid(row=4,sticky=W)
        checkbutton6.grid(row=5,sticky=W)
        self.label_n_feature_importance_bars.grid(row=6, sticky=W)
        checkbutton7.grid(row=7,sticky=W)
        checkbutton8.grid(row=8, sticky=W)
        self.LC_ksplit.grid(row=9, sticky=W)
        self.LC_datasplit.grid(row=10, sticky=W)
        checkbutton9.grid(row=11, sticky=W)
        checkbutton10.grid(row=12,sticky=W)
        self.shap_present.grid(row=13,sticky=W)
        self.shap_absent.grid(row=14,sticky=W)

        button_settings_to_ini.grid(row=6,pady=5)
        button_save_meta.grid(row=7)
        button_remove_meta.grid(row=8,pady=5)


    def clearcache(self):
        configs_dir = os.path.join(os.path.dirname(self.configini),'configs')
        filelist = [f for f in os.listdir(configs_dir) if f.endswith('.csv')]
        for f in filelist:
            os.remove(os.path.join(configs_dir,f))
            print(f,'deleted')

    def load_RFvalues(self):

        metadata = pd.read_csv(str(self.load_choosedata.file_path), index_col=False)
        # metadata = metadata.drop(['Feature_list'], axis=1)
        for m in metadata.columns:
            self.meta_dict[m] = metadata[m][0]
        print('Meta data file loaded')

        for key in self.meta_dict:
            cur_list = key.lower().split(sep='_')
            # print(cur_list)
            for i in self.settings:
                string = i.lblName.cget('text').lower()
                if all(map(lambda w: w in string, cur_list)):
                    i.entry_set(self.meta_dict[key])
            for k in self.check_settings:
                string = k.cget('text').lower()
                if all(map(lambda w: w in string, cur_list)):
                    if self.meta_dict[key] == 'yes':
                        k.select()
                    elif self.meta_dict[key] == 'no':
                        k.deselect()

    def get_checkbox(self):
        ### check box settings
        if self.box1.get() == 1:
            self.rfmetadata = 'yes'
        else:
            self.rfmetadata = 'no'

        if self.box2.get() == 1:
            self.generate_example_d_tree = 'yes'
        else:
            self.generate_example_d_tree = 'no'

        if self.box3.get() == 1:
            self.generate_example_decision_tree_fancy = 'yes'
        else:
            self.generate_example_decision_tree_fancy  = 'no'

        if self.box4.get() == 1:
            self.generate_classification_report = 'yes'
        else:
            self.generate_classification_report = 'no'

        if self.box5.get() == 1:
            self.generate_features_imp_log = 'yes'
        else:
            self.generate_features_imp_log = 'no'

        if self.box6.get() == 1:
            self.generate_features_bar_graph = 'yes'
        else:
            self.generate_features_bar_graph = 'no'
        self.n_importance = self.label_n_feature_importance_bars.entry_get

        if self.box7.get() == 1:
            self.compute_permutation_imp = 'yes'
        else:
            self.compute_permutation_imp = 'no'

        if self.box8.get() == 1:
            self.generate_learning_c = 'yes'
        else:
            self.generate_learning_c = 'no'
        self.learningcurveksplit = self.LC_ksplit.entry_get
        self.learningcurvedatasplit = self.LC_datasplit.entry_get

        if self.box9.get() == 1:
            self.generate_precision_recall_c = 'yes'
        else:
            self.generate_precision_recall_c = 'no'

        if self.box10.get() == 1:
            self.getshapscores = 'yes'
        else:
            self.getshapscores = 'no'

        self.shappresent = self.shap_present.entry_get
        self.shapabsent = self.shap_absent.entry_get


    def save_new(self):
        self.get_checkbox()
        meta_number = 0
        for f in os.listdir(os.path.join(os.path.dirname(self.configini), 'configs')):
            if f.__contains__('_meta') and f.__contains__(str(self.varmodel.get())):
                meta_number += 1

        # for s in self.settings:
        #     meta_df[s.lblName.cget('text')] = [s.entry_get]
        new_meta_dict = {'RF_n_estimators': self.label_nestimators.entry_get,
                         'RF_max_features': self.label_maxfeatures.entry_get, 'RF_criterion': self.label_criterion.entry_get,
                         'train_test_size': self.label_testsize.entry_get, 'RF_min_sample_leaf': self.label_minsampleleaf.entry_get,
                         'under_sample_ratio': self.label_under_s_correctionvalue.entry_get, 'under_sample_setting': self.label_under_s_settings.entry_get,
                         'over_sample_ratio': self.label_over_s_ratio.entry_get, 'over_sample_setting': self.label_over_s_settings.entry_get,
                         'generate_rf_model_meta_data_file': self.rfmetadata,
                         'generate_example_decision_tree': self.generate_example_d_tree,'generate_classification_report':self.generate_classification_report,
                         'generate_features_importance_log': self.generate_features_imp_log,'generate_features_importance_bar_graph':self.generate_features_bar_graph,
                         'n_feature_importance_bars': self.n_importance,'compute_feature_permutation_importance':self.compute_permutation_imp,
                         'generate_sklearn_learning_curves': self.generate_learning_c,
                         'generate_precision_recall_curves':self.generate_precision_recall_c, 'learning_curve_k_splits':self.learningcurveksplit,
                         'learning_curve_data_splits': self.learningcurvedatasplit,
                         'generate_shap_scores':self.getshapscores,
                         'shap_target_present_no':self.shappresent,
                         'shap_target_absetn_no':self.shapabsent}
        meta_df = pd.DataFrame(new_meta_dict, index=[0])
        meta_df.insert(0, 'Classifier_name', str(self.varmodel.get()))

        if currentPlatform == 'Windows':
            output_path = os.path.dirname(self.configini) + "\\configs\\" + \
                        str(self.varmodel.get())+ '_meta_' + str(meta_number) + '.csv'

        if currentPlatform == 'Linux'or (currentPlatform == 'Darwin'):
            output_path = os.path.dirname(self.configini) + "/configs/" + \
                        str(self.varmodel.get())+ '_meta_' + str(meta_number) + '.csv'


        print(os.path.basename(str(output_path)),'saved')

        meta_df.to_csv(output_path, index=FALSE)

    def set_values(self):
        self.get_checkbox()
        #### settings
        model = self.var.get()
        n_estimators = self.label_nestimators.entry_get
        max_features = self.label_maxfeatures.entry_get
        criterion = self.label_criterion.entry_get
        test_size = self.label_testsize.entry_get
        min_sample_leaf = self.label_minsampleleaf.entry_get
        under_s_c_v = self.label_under_s_correctionvalue.entry_get
        under_s_settings = self.label_under_s_settings.entry_get
        over_s_ratio = self.label_over_s_ratio.entry_get
        over_s_settings = self.label_over_s_settings.entry_get
        classifier_settings = self.varmodel.get()


        #export settings to config ini file
        configini = self.configini
        config = ConfigParser()
        config.read(configini)

        config.set('create ensemble settings', 'model_to_run', str(model))
        config.set('create ensemble settings', 'RF_n_estimators', str(n_estimators))
        config.set('create ensemble settings', 'RF_max_features', str(max_features))
        config.set('create ensemble settings', 'RF_criterion', str(criterion))
        config.set('create ensemble settings', 'train_test_size', str(test_size))
        config.set('create ensemble settings', 'RF_min_sample_leaf', str(min_sample_leaf))
        config.set('create ensemble settings', 'under_sample_ratio', str(under_s_c_v))
        config.set('create ensemble settings', 'under_sample_setting', str(under_s_settings))
        config.set('create ensemble settings', 'over_sample_ratio', str(over_s_ratio))
        config.set('create ensemble settings', 'over_sample_setting', str(over_s_settings))
        config.set('create ensemble settings', 'classifier',str(classifier_settings))
        config.set('create ensemble settings', 'RF_meta_data', str(self.rfmetadata))
        config.set('create ensemble settings', 'generate_example_decision_tree', str(self.generate_example_d_tree))
        config.set('create ensemble settings', 'generate_classification_report', str(self.generate_classification_report))
        config.set('create ensemble settings', 'generate_features_importance_log', str(self.generate_features_imp_log))
        config.set('create ensemble settings', 'generate_features_importance_bar_graph', str(self.generate_features_bar_graph))
        config.set('create ensemble settings', 'N_feature_importance_bars', str(self.n_importance))
        config.set('create ensemble settings', 'compute_permutation_importance', str(self.compute_permutation_imp))
        config.set('create ensemble settings', 'generate_learning_curve', str(self.generate_learning_c))
        config.set('create ensemble settings', 'generate_precision_recall_curve', str(self.generate_precision_recall_c))
        config.set('create ensemble settings', 'LearningCurve_shuffle_k_splits',str(self.learningcurveksplit))
        config.set('create ensemble settings', 'LearningCurve_shuffle_data_splits',str(self.learningcurvedatasplit))
        config.set('create ensemble settings', 'generate_example_decision_tree_fancy',str(self.generate_example_decision_tree_fancy))
        config.set('create ensemble settings', 'generate_shap_scores',str(self.getshapscores))
        config.set('create ensemble settings', 'shap_target_present_no', str(self.shappresent))
        config.set('create ensemble settings', 'shap_target_absent_no', str(self.shapabsent))

        with open(configini, 'w') as configfile:
            config.write(configfile)

        print('Settings exported to project_config.ini')

class makelineplot:
    def __init__(self):
        # Popup window
        lpToplevel = Toplevel()
        lpToplevel.minsize(200, 200)
        lpToplevel.wm_title("Make line plot")

        lpLabelframe = LabelFrame(lpToplevel)
        lpvideo = FileSelect(lpLabelframe,'Video',lblwidth='10')
        lpcsv = FileSelect(lpLabelframe,'csv file',lblwidth='10')
        bpentry = Entry_Box(lpLabelframe,'Bodypart','10')

        lpbutton = Button(lpLabelframe,text='Generate plot',command =lambda: draw_line_plot_tools(lpvideo.file_path,lpcsv.file_path,bpentry.entry_get))
        #organize
        lpLabelframe.grid(row=0,sticky=W)
        lpvideo.grid(row=0,sticky=W)
        lpcsv.grid(row=1,sticky=W)
        bpentry.grid(row=2,sticky=W)
        lpbutton.grid(row=3,pady=10)

class droptrackingdata:
    def __init__(self):
        self.dtToplevel = Toplevel()
        self.dtToplevel.minsize(500, 800)
        self.dtToplevel.wm_title('Drop bodyparts in Tracking Data')

        self.scroll = hxtScrollbar(self.dtToplevel)

        dtLabelframe = LabelFrame(self.scroll, text='File Settings', font=('Helvetica', 10, 'bold'), pady=5, padx=5)
        self.fFolder = FolderSelect(dtLabelframe, 'Data Folder', lblwidth='13')
        self.poseTool = DropDownMenu(dtLabelframe, 'Tracking tool', ['DLC','maDLC'], '13')
        self.poseTool.setChoices('DLC')
        self.fileFormat = DropDownMenu(dtLabelframe, 'File Type', ['csv', 'h5'], '13')
        self.fileFormat.setChoices('csv')
        self.noofbp = Entry_Box(dtLabelframe,'# Bp to remove','13')

        btnConfirm = Button(dtLabelframe, text='Confirm',
                            command=lambda: self.confirm(self.fFolder.folder_path, self.poseTool.getChoices(),
                                                         self.fileFormat.getChoices(), self.noofbp.entry_get))

        #organize
        dtLabelframe.grid(row=0,sticky=W)
        self.fFolder.grid(row=0,sticky=W,columnspan=3)
        self.poseTool.grid(row=1,sticky=W)
        self.fileFormat.grid(row=2,sticky=W)
        self.noofbp.grid(row=3,sticky=W)
        btnConfirm.grid(row=3,column=1,sticky=W)

    def confirm(self,folder,posetool,fileformat, nobps):
        try:
            self.frame.destroy()
        except:
            pass

        animal_names, bodypart_names = drop_bps_tracking_input_bps(folder, posetool, fileformat, nobps)

        animal_names_set, body_part_names_set = list(set(animal_names)), list(set(bodypart_names))

        self.frame = Frame(self.scroll)
        self.table1 = LabelFrame(self.frame,text='Remove body parts:')

        self.frame.grid(row=1, sticky=W, pady=10)
        self.table1.grid(row=0, sticky=N, pady=5)
        self.animal_names = []
        self.drop_down_list = []

        if posetool == 'DLC':
            for bp_number in range(int(nobps)):
                bp_drop_down = DropDownMenu(self.table1, 'Body-part ' + str(bp_number+1), bodypart_names, '10')
                bp_drop_down.setChoices(bodypart_names[bp_number])
                self.drop_down_list.append(bp_drop_down)
            for number, drop_down in enumerate(self.drop_down_list):
                drop_down.grid(row=number, column=0, sticky=W)

        if posetool == 'maDLC':
            for bp_number in range(int(nobps)):
                animal_drop_down = DropDownMenu(self.table1, 'Animal name', animal_names_set, '10')
                animal_drop_down.setChoices(animal_names[0])
                self.animal_names.append(animal_drop_down)
                bp_drop_down = DropDownMenu(self.table1, 'Body-part ' + str(bp_number+1), body_part_names_set, '10')
                bp_drop_down.setChoices(bodypart_names[bp_number])
                self.drop_down_list.append(bp_drop_down)
            for number, drop_down in enumerate(self.drop_down_list):
                self.animal_names[number].grid(row=number, column=0, sticky=W)
                drop_down.grid(row=number, column=1, sticky=W)

        button_run = Button(self.frame, text='Run removal', command=lambda: self.run_removal())
        button_run.grid(row=int(nobps) + 2, column=0, sticky=W)

    def run_removal(self):
        bp_to_remove_list = []
        for number, drop_down in enumerate(self.drop_down_list):
            bp_to_remove_list.append(drop_down.getChoices())
        run_bp_removal(self.poseTool.getChoices(), self.animal_names, bp_to_remove_list, self.fFolder.folder_path, self.fileFormat.getChoices())


class reorganizeData:
    def __init__(self):
        self.roToplevel = Toplevel()
        self.roToplevel.minsize(500,800)
        self.roToplevel.wm_title('Reorganize Tracking Data')

        self.scroll = hxtScrollbar(self.roToplevel)

        roLabelframe = LabelFrame(self.scroll,text='File Settings',font=('Helvetica',10,'bold'),pady=5,padx=5)
        self.fFolder = FolderSelect(roLabelframe,'Data Folder',lblwidth='10')
        self.poseTool = DropDownMenu(roLabelframe,'Tracking tool',['DLC'],'10')
        self.poseTool.setChoices('DLC')
        self.fileFormat = DropDownMenu(roLabelframe,'File Type',['csv','h5'],'10')
        self.fileFormat.setChoices('csv')

        btnConfirm = Button(roLabelframe,text='Confirm',command= lambda: self.confirm(self.fFolder.folder_path,self.poseTool.getChoices(),self.fileFormat.getChoices()))
        # btnreorganize = Button(self.roToplevel,text='Reorganize Data',command = )

        #organize
        roLabelframe.grid(row=0,sticky=W)
        self.fFolder.grid(row=0,sticky=W,columnspan=3)
        self.poseTool.grid(row=1,sticky=W)
        self.fileFormat.grid(row=2,sticky=W)
        btnConfirm.grid(row=2,column=1,sticky=W)

    def confirm(self,folder,posetool,fileformat):
        try:
            self.frame.destroy()
        except:
            pass

        animallist, bplist, self.headerlist = display_original_bp_list(folder,posetool,fileformat)
        self.frame = Frame(self.scroll)
        self.table1 = LabelFrame(self.frame,text='Current Order:')
        self.table2 = LabelFrame(self.frame,text='New Order')

        #organize
        self.frame.grid(row=1,sticky=W,pady=10)
        self.table1.grid(row=0,sticky=N, pady=5)
        self.table2.grid(row=0,column=1,sticky=N,padx=5,pady=5)

        idx1, idx2, oldanimallist, oldbplist, self.newanimallist, self.newbplist = ([0]*len(bplist) for i in range(6)) #create lists

        if animallist:
            animal_list_reduced = list(set(animallist))
            self.pose_tool = 'maDLC'
            #if ma dlc or h5
            for i in range(len(bplist)):

                #current order
                idx1[i] = Label(self.table1,text=str(i+1) + '.')
                oldanimallist[i] = Label(self.table1,text=str(animallist[i]))
                oldbplist[i] = Label(self.table1,text=str(bplist[i]))

                idx1[i].grid(row=i,column=0,sticky=W)
                oldanimallist[i].grid(row=i,column=1,sticky=W, ipady=5)
                oldbplist[i].grid(row=i,column=2,sticky=W, ipady=5)

                #new order
                idx2[i] = Label(self.table2,text=str(i+1) + '.')
                self.newanimallist[i] = DropDownMenu(self.table2, ' ', animal_list_reduced, '10')
                self.newbplist[i] = DropDownMenu(self.table2,' ', bplist,'10')
                self.newanimallist[i].setChoices(animallist[i])
                self.newbplist[i].setChoices(bplist[i])

                idx2[i].grid(row=i,column=0,sticky=W)
                self.newanimallist[i].grid(row=i, column=1, sticky=W)
                self.newbplist[i].grid(row=i,column=2,sticky=W)

        else:
            self.pose_tool = 'DLC'
            for i in range(len(bplist)):
                # current order
                idx1[i] = Label(self.table1, text=str(i + 1) + '.')
                oldbplist[i] = Label(self.table1, text=str(bplist[i]))

                idx1[i].grid(row=i, column=0, sticky=W, ipady=5)
                oldbplist[i].grid(row=i, column=2, sticky=W, ipady=5)

                # new order
                idx2[i] = Label(self.table2, text=str(i + 1) + '.')
                self.newbplist[i] = StringVar()
                oldanimallist[i] = OptionMenu(self.table2, self.newbplist[i], *bplist)
                self.newbplist[i].set(bplist[i])

                idx2[i].grid(row=i, column=0, sticky=W)
                oldanimallist[i].grid(row=i, column=1, sticky=W)

        button_run = Button(self.frame, text='Run re-organization', command= lambda: self.run_reorganization())
        button_run.grid(row=2, column=1, sticky=W)

    def run_reorganization(self):
        if self.pose_tool == 'DLC':
            new_bp_list = []
            for curr_choice in self.newbplist:
                new_bp_list.append(curr_choice.get())
            reorganize_bp_order(self.fFolder.folder_path, self.poseTool.getChoices(), self.fileFormat.getChoices(), self.headerlist, [], new_bp_list, self.pose_tool)
        if self.pose_tool == 'maDLC':
            new_bp_list, new_animal_list = [], []
            for curr_animal, curr_bp in zip(self.newanimallist, self.newbplist):
                new_bp_list.append(curr_bp.getChoices())
                new_animal_list.append(curr_animal.getChoices())
            reorganize_bp_order(self.fFolder.folder_path, self.poseTool.getChoices(), self.fileFormat.getChoices(), self.headerlist, new_animal_list, new_bp_list, self.pose_tool)


class runmachinemodelsettings:
    def __init__(self,inifile):
        self.row1 = []
        self.row2 = []
        self.row3 = []
        self.row4 = []
        self.targetname = []
        # Popup window
        runmms = Toplevel()
        runmms.minsize(200, 200)
        runmms.wm_title("Select model to run")

        ### read inifile and get the model
        config = ConfigParser()
        configini = str(inifile)
        config.read(configini)
        no_targets = config.get('SML settings','no_targets')

        ###get all target from ini
        for i in range(int(no_targets)):
            currentModelNames = 'target_name_' + str(i+1)
            currentModelNames = config.get('SML settings', currentModelNames)
            self.targetname.append(currentModelNames)

        ###loop for table
        table = LabelFrame(runmms)
        #set title
        tn = Label(table,text='Classifier',font=("Helvetica",10,'bold'))
        tn.grid(row=0,column=0,sticky=W,pady=5)

        selectmodel = Label(table,text='Model path (.sav)',font=("Helvetica",10,'bold'))
        selectmodel.grid(row=0,column=1)

        thresholdtitle = Label(table,text='Threshold',font=("Helvetica",10,'bold') )
        thresholdtitle.grid(row=0,column=2,sticky=W)

        minbouttitle = Label(table,text='Minimum Bout',font=("Helvetica",10,'bold'))
        minbouttitle.grid(row=0,column=3,sticky=W)

        # main loop for content of table
        for i in range(len(self.targetname)):
            self.row1.append(Label(table,text=str(self.targetname[i])))
            self.row1[i].grid(row=i+2,column=0,sticky=W)

            self.row2.append(FileSelect(table,title='Select model (.sav) file'))
            self.row2[i].grid(row=i+2,column=1,sticky=W)

            self.row3.append(Entry(table))
            self.row3[i].grid(row=i+2,column=2,sticky=W,padx=5)

            self.row4.append(Entry(table))
            self.row4[i].grid(row=i+2,column=3,sticky=W,padx=5)

        button_set = Button(runmms,text='Set model(s)',command =lambda:self.set_modelpath_to_ini(inifile),font=("Helvetica",10,'bold'),fg='red')

        table.grid(row=0,sticky=W,pady=5,padx=5)
        button_set.grid(row=1,pady=10)

    def set_modelpath_to_ini(self,inifile):
        config = ConfigParser()
        configini = str(inifile)
        config.read(configini)

        for i in range(len(self.targetname)):
            config.set('SML settings','model_path_'+(str(i+1)),str(self.row2[i].file_path))
            config.set('threshold_settings', 'threshold_' + (str(i + 1)), str(self.row3[i].get()))
            config.set('Minimum_bout_lengths', 'min_bout_' + (str(i + 1)), str(self.row4[i].get()))

        with open(configini, 'w') as configfile:
            config.write(configfile)

        print('Model paths saved in project_config.ini')

def get_frame(self):
    '''
      Return the "frame" useful to place inner controls.
    '''
    return self.canvas

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
        y = y + cy + self.widget.winfo_rooty() +27
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

def form_validator_is_numeric(inStr, acttyp):
    if acttyp == '1':  #insert
        if not inStr.isdigit():
            return False
    return True

class aboutgui:
    def __init__(self):
        about = Toplevel()
        about.minsize(896, 507)
        about.wm_title("About")

        canvas = Canvas(about,width=896,height=507,bg='black')
        canvas.pack()
        scriptdir = os.path.dirname(__file__)
        img = PhotoImage(file=os.path.join(scriptdir,'About_me_050122_1.png'))

        canvas.create_image(0,0,image=img,anchor='nw')
        canvas.image = img

class App(object):
    def __init__(self):
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        self.root = Tk()
        self.root.title('SimBA')
        self.root.minsize(750,750)
        self.root.geometry("750x750")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        if currentPlatform == 'Windows':
            self.root.iconbitmap(os.path.join(scriptdir,'SimBA_logo.ico'))

        img = PhotoImage(file=os.path.join(scriptdir,'golden.png'))
        background = Label(self.root, image=img, bd=0)
        background.pack(fill='both', expand=True)
        background.image = img

        ### drop down menu###f
        menu = Menu(self.root)
        self.root.config(menu=menu)

        #first menu
        fileMenu = Menu(menu)
        menu.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label='Create a new project',command=project_config)
        fileMenu.add_command(label='Load project', command=lambda:loadprojectMenu(loadprojectini))
        fileMenu.add_separator()
        fileMenu.add_command(label='Exit', command=self.root.destroy)

        # Process video
        pvMenu = Menu(menu)
        menu.add_cascade(label='Process Videos', menu=pvMenu)
        pvMenu.add_command(label='Batch pre-process videos', command=batch_processvideo)

        #third menu
        thirdMenu = Menu(menu)
        menu.add_cascade(label='Tracking',menu=thirdMenu)
        #dlc
        dlcmenu = Menu(thirdMenu)
        dlcmenu.add_command(label='Create DLC Model',command=create_project_DLC)
        dlcmenu.add_command(label='Load DLC Model',command=Load_DLC_Model)
        #deepPoseKit
        dpkmenu = Menu(thirdMenu)
        dpkmenu.add_command(label='Create DeepPoseKit project', command=createDPK_project)
        dpkmenu.add_command(label='Load DeepPoseKit project', command=lambda:loadprojectMenu(deepPoseKitMenu))
        #labelling tool
        labellingtoolmenu = Menu(thirdMenu)
        labellingtoolmenu.add_command(label='labelImg', command=lambda: subprocess.call(["labelImg"]))
        labellingtoolmenu.add_command(label='labelme', command=lambda: subprocess.call(["labelme"]))
        #third menu organize
        thirdMenu.add_cascade(label='DeepLabCut', menu=dlcmenu)
        thirdMenu.add_cascade(label='DeepPoseKit', menu=dpkmenu)
        thirdMenu.add_command(label='LEAP', command=lambda: print('coming soon'))
        thirdMenu.add_cascade(label='Labelling tools', menu=labellingtoolmenu)

        #fifth menu
        fifthMenu = Menu(menu)
        #changefpsmenu
        fpsMenu = Menu(fifthMenu)
        fpsMenu.add_command(label='Change fps for single video', command=changefps)
        fpsMenu.add_command(label='Change fps for multiple videos',command=changefpsmulti)
        menu.add_cascade(label='Tools',menu=fifthMenu)
        fifthMenu.add_command(label='Clip videos',command=shorten_video)
        fifthMenu.add_command(label='Clip video into multiple videos', command=multi_shorten_video)
        fifthMenu.add_command(label='Crop videos',command=crop_video)
        fifthMenu.add_command(label='Multi-crop',command=multicropmenu)
        fifthMenu.add_command(label='Downsample videos',command=video_downsample)
        fifthMenu.add_command(label='Get mm/ppx',command = get_coordinates_from_video)
        fifthMenu.add_command(label='Make line plot', command=makelineplot)
        fifthMenu.add_cascade(label='Change fps...',menu =fpsMenu)
        fifthMenu.add_cascade(label='Visualize pose-estimation in folder...', command=visualize_pose)
        fifthMenu.add_cascade(label='Reorganize Tracking Data', command= reorganizeData)
        fifthMenu.add_cascade(label='Drop body-parts from tracking data', command=droptrackingdata)



        #changefpsmenu organize

        changeformatMenu = Menu(fifthMenu)
        changeformatMenu.add_command(label='Change image file formats',command=change_imageformat)
        changeformatMenu.add_command(label='Change video file formats',command=convert_video)
        changeformatMenu.add_command(label='Change .seq to .mp4', command=lambda:convertseqVideo(askdirectory(title='Please select video folder to convert')))
        fifthMenu.add_cascade(label='Change formats...',menu=changeformatMenu)

        fifthMenu.add_command(label='CLAHE enhance video',command=Red_light_Convertion)
        fifthMenu.add_command(label='Superimpose frame numbers on video',command=lambda:superimposeframe(askopenfilename()))
        fifthMenu.add_command(label='Convert to grayscale',command=lambda:greyscale(askopenfilename()))
        fifthMenu.add_command(label='Merge frames to video',command=mergeframeffmpeg)
        fifthMenu.add_command(label='Generate gifs', command=creategif)

        extractframesMenu = Menu(fifthMenu)
        extractframesMenu.add_command(label='Extract defined frames',command=extract_specificframes)
        extractframesMenu.add_command(label='Extract frames',command=extract_allframes)
        extractframesMenu.add_command(label='Extract frames from seq files', command=extract_seqframe)
        fifthMenu.add_cascade(label='Extract frames...',menu=extractframesMenu)
        convertWftypeMenu = Menu(fifthMenu)
        convertWftypeMenu.add_command(label='Convert CSV to parquet', command=CSV2parquet)
        convertWftypeMenu.add_command(label='Convert parquet o CSV', command=parquet2CSV)
        fifthMenu.add_cascade(label='Convert working file type...', menu=convertWftypeMenu)


        #sixth menu
        sixthMenu = Menu(menu)
        menu.add_cascade(label='Help',menu=sixthMenu)
        #labelling tool
        links = Menu(sixthMenu)
        links.add_command(label='Download weights',command = lambda:webbrowser.open_new(str(r'https://osf.io/sr3ck/')))
        links.add_command(label='Download classifiers', command=lambda: webbrowser.open_new(str(r'https://osf.io/kwge8/')))
        links.add_command(label='Ex. feature list',command=lambda: webbrowser.open_new(str(r'https://github.com/sgoldenlab/simba/blob/master/misc/Feature_description.csv')))
        links.add_command(label='SimBA github', command=lambda: webbrowser.open_new(str(r'https://github.com/sgoldenlab/simba')))
        links.add_command(label='Gitter Chatroom', command=lambda: webbrowser.open_new(str(r'https://gitter.im/SimBA-Resource/community')))
        links.add_command(label='Install FFmpeg',command =lambda: webbrowser.open_new(str(r'https://m.wikihow.com/Install-FFmpeg-on-Windows')))
        links.add_command(label='Install graphviz', command=lambda: webbrowser.open_new(str(r'https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software')))
        sixthMenu.add_cascade(label="Links",menu=links)
        sixthMenu.add_command(label='About', command= aboutgui)

        #Status bar at the bottom
        self.frame = Frame(background, bd=2, relief=SUNKEN, width=300, height=300)
        self.frame.pack(expand=True)
        self.txt = Text(self.frame, bg='white')
        self.txt.config(state=DISABLED)
        self.txt.pack(expand=True, fill='both')
        sys.stdout = StdRedirector(self.txt)

# writes text out in GUI
class StdRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.config(state=NORMAL)
        self.text_space.insert("end", string)
        self.text_space.update()
        self.text_space.see("end")
        self.text_space.config(state=DISABLED)

class SplashScreen:
    def __init__(self, parent):
        self.parent = parent
        self.Splash()
        self.Window()

    def Splash(self):
        scriptdir = os.path.dirname(__file__)
        if currentPlatform == 'Windows':
            self.image = PIL.Image.open(os.path.join(scriptdir,"splash_050122.png"))
        if (currentPlatform == 'Linux') or (currentPlatform == 'Darwin'):
            self.image = PIL.Image.open(os.path.join(scriptdir, "splash_050122.PNG"))
        self.imgSplash = ImageTk.PhotoImage(self.image)


    def Window(self):
        width, height = self.image.size
        halfwidth = (self.parent.winfo_screenwidth()-width)//2
        halfheight = (self.parent.winfo_screenheight()-height)//2
        self.parent.geometry("%ix%i+%i+%i" %(width, height, halfwidth,halfheight))
        Label(self.parent, image=self.imgSplash).pack()

def terminate_children(children):
    for process in children:
        process.terminate()

def main():
    #windows icon
    if currentPlatform == 'Windows':
        import ctypes
        myappid = 'SimBA development wheel'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


    root = Tk()
    root.overrideredirect(True)
    # import tkinter.font as TkFont
    # default_font = TkFont.nametofont("TkDefaultFont")
    # default_font.configure(size=6)
    # root.option_add("*Font", default_font)
    app = SplashScreen(root)
    root.after(2500, root.destroy)
    root.mainloop()

    app = App()
    print('Welcome fellow scientists :)' + '\n')
    print('\n')
    app.root.mainloop()
