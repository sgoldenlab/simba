import os
import time
import subprocess
import deeplabcut
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename,askdirectory
from tkinter_functions import greyscale,downsamplevideo,superimposeframe,shortenvideos1,shortenvideos2,convertavitomp4,convertpowerpoint,extract_allframescommand,mergemovieffmpeg,extractspecificframe,clahe,cropvid,changedlc_config,changeimageformat
from create_project_ini import write_inifile
from tkinter import tix
import subprocess
import platform
import shutil
import datetime
from tkinter_functions_batch import clahe_batch,shortenvideos1_batch,greyscale_batch,superimposeframe_batch,extract_allframescommand_batch,downsamplevideo_batch,cropvid_batch,extract_frames_ini
from dlc_change_yamlfile import select_numfram2pick,updateiteration,update_init_weight,generatetempyaml,generatetempyaml_multi
from correct_devs_loc import dev_loc
from correct_devs_mov import dev_move
from extract_features_wo_targets import extract_features_wotarget
from sklearn_DLC_RF_train_model import RF_trainmodel
from run_RF_model import rfmodel
from merge_frames import merge_frames_config
from plot_sklearn_results import plotsklearnresult
from path_plot import path_plot_config
from gantt import ganntplot_config
from data_plot import data_plot_config
from line_plot import line_plot_config
from merge_movie_ffmpeg import generatevideo_config_ffmpeg
from import_videos_csv_project_ini import copy_singlevideo_ini,copy_multivideo_ini,copy_allcsv_ini,copy_singlecsv_ini
from configparser import ConfigParser
from labelling_aggression import *
from PIL import Image, ImageTk
import tkinter.ttk as ttk
from get_coordinates_tools_v2 import get_coordinates_nilsson
from process_data_log import analyze_process_data_log
from process_severity import analyze_process_severity
from process_movement import analyze_process_movement
import webbrowser

class FolderSelect(Frame):
    def __init__(self,parent=None,folderDescription="",**kw):
        Frame.__init__(self,master=parent,**kw)
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription)
        self.lblName.grid(row=0,column=0)
        self.entPath = Label(self, textvariable=self.folderPath,relief=SUNKEN)
        self.entPath.grid(row=0,column=1)
        self.btnFind = Button(self, text="Browse Folder",command=self.setFolderPath)
        self.btnFind.grid(row=0,column=2)
        self.folderPath.set('No folder selected')
    def setFolderPath(self):
        folder_selected = askdirectory()
        if folder_selected:
            self.folderPath.set(folder_selected)
        else:
            self.folderPath.set('No folder selected')

    @property
    def folder_path(self):
        return self.folderPath.get()

class FileSelect(Frame):
    def __init__(self,parent=None,fileDescription="",**kw):
        Frame.__init__(self,master=parent,**kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription)
        self.lblName.grid(row=0,column=0)
        self.entPath = Label(self, textvariable=self.filePath,relief=SUNKEN)
        self.entPath.grid(row=0,column=1)
        self.btnFind = Button(self, text="Browse File",command=self.setFilePath)
        self.btnFind.grid(row=0,column=2)
        self.filePath.set('No file selected')
    def setFilePath(self):
        file_selected = askopenfilename()
        if file_selected:
            self.filePath.set(file_selected)
        else:
            self.filePath.set('No file selected')
    @property
    def file_path(self):
        return self.filePath.get()

class Entry_Box(Frame):
    def __init__(self,parent=None,fileDescription="",labelwidth='',**kw):
        Frame.__init__(self,master=parent,**kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription,width=labelwidth,anchor=W)
        self.lblName.grid(row=0,column=0)
        self.entPath = Entry(self, textvariable=self.filePath)
        self.entPath.grid(row=0,column=1)

    @property
    def entry_get(self):
        self.entPath.get()
        return self.entPath.get()

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
        a = self.entPath[row].get()
        return a

    def setvariable(self,row,vars):
        b = self.entPathvars[row].set(vars)
        return b

class Button_getcoord(Frame):

    def __init__(self,parent=None,filename=[],knownmm=[],**kw): #set to list and use range i in list to call each elements in list

        Frame.__init__(self, master=parent, **kw)
        self.entPath = []
        self.ppm_list = []
        self.ppmvar = []
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


    def getcoord_forbutton(self,filename,knownmm,count):

        ppm = get_coordinates_nilsson(filename,knownmm)

        if ppm == float('inf'):
            print('Divide by zero error. Please make sure the values in [Distance_in_mm] are updated')
        else:
            self.ppmvar[count].set(ppm)

    def getppm(self,count):
        ppms = self.ppm_list[count].get()
        return ppms

def Exit():
    app.root.destroy()

class video_info_table:

    def __init__(self,configini):

        self.filesFound = [0]

        config = ConfigParser()
        self.configFile = str(configini)
        config.read(self.configFile)
        self.config_videofolders = str(config.get('General settings', 'project_path'))+'\\videos'
        config_fps = config.get('Frame settings', 'fps')
        config_x_reso = config.get('Frame settings', 'resolution_width')
        config_y_reso = config.get('Frame settings', 'resolution_height')
        config_distancemm = config.get('Frame settings', 'distance_mm')

        ########### FIND FILES ###########
        for i in os.listdir(self.config_videofolders):
            if i.__contains__(".mp4"):

                self.filesFound.append(i)

        self.tkintertable = Toplevel()
        self.tkintertable.minsize(1000, 500)
        self.tkintertable.wm_title("Video Info")

        self.xscrollbar = Scrollable(self.tkintertable,width=32)
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
            self.table_col[0].setvariable(count,str(count)+'.')
            self.table_col[1].setvariable(count,i)
            self.table_col[2].setvariable(count, config_fps)
            self.table_col[3].setvariable(count, config_x_reso)
            self.table_col[4].setvariable(count, config_y_reso)
            self.table_col[5].setvariable(count,config_distancemm)
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
            self.data_lists[1].append(str(self.config_videofolders +'\\'+ str(i)))
            self.data_lists[5].append(int(0))

        self.button = Button_getcoord(self.xscrollbar,self.data_lists[1],self.data_lists[5])
        self.button.grid(row=6,column=1)
        #instructions
        label_getdatadesc1 = Label(self.xscrollbar,text='1.Enter the known distance (mm)')
        label_getdatadesc2 = Label(self.xscrollbar,text='2.Click <Update distance_in_mm> button before clicking on Video1,Video2,Video3... to get the pixels/mm')
        label_getdatadesc3 = Label(self.xscrollbar,text='3.Click <Save Data> once all the data are filled.')
        label_getdatadesc1.grid(row=0,sticky=W)
        label_getdatadesc2.grid(row=1, sticky=W)
        label_getdatadesc3.grid(row=2, sticky=W)

        get_data_button = Button(self.xscrollbar, text='Update distance_in_mm', command=self.getdata)
        get_data_button.grid(row=3,sticky=W)

        add_column_button = Button(self.xscrollbar,text='<Add Column>',command=self.addBox,fg='red')
        add_column_button.grid(row=4,sticky=W)

        generate_csv_button = Button(self.xscrollbar,text='Save Data',command=self.generate_video_info_csv,font='bold',fg='red')
        generate_csv_button.grid(row=5)

        self.xscrollbar.update()

    def addBox(self):
        self.new_col_list.append(0)
        self.next_column = len(self.new_col_list)
        print(self.next_column)
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
        self.data_lists[1] = [str(self.config_videofolders)+'\\'+s for s in self.data_lists[1]]

        #update get coord with data
        self.button = Button_getcoord(self.xscrollbar,self.data_lists[1],self.data_lists[5])
        self.button.grid(row=6,column=1)

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
        logfolder=str(os.path.dirname(self.configFile))+'\\logs\\'
        csv_filename = 'video_info.csv'
        output=logfolder+csv_filename
        df.to_csv(str(output),index=False)
        print(os.path.dirname(output),'generated.')

class video_downsample:

    def __init__(self):
        # Popup window
        videosdownsample = Toplevel()
        videosdownsample.minsize(200, 200)
        videosdownsample.wm_title("Downsample Video Resolution")

        # Video Path
        self.videopath1selected = FileSelect(videosdownsample, "Video Path")
        label_choiceq = Label(videosdownsample, text='Choose only one of the following method to downsample videos (Custom/Default)')

        #custom reso
        label_downsamplevidcustom = LabelFrame(videosdownsample,text='Customize Resolution',font='bold',padx=5,pady=5)
        # width
        self.label_width = Entry_Box(label_downsamplevidcustom,'Width','10')
        # height
        self.label_height = Entry_Box(label_downsamplevidcustom,'Height','10')
        # confirm custom resolution
        self.button_downsamplevideo1 = Button(label_downsamplevidcustom, text='Downsample Custom Resolution Video',command=self.downsample_customreso)
        #Default reso
        # Checkbox
        label_downsampleviddefault = LabelFrame(videosdownsample,text='Default Resolution',font ='bold',padx=5,pady=5)
        self.var1 = IntVar()
        self.checkbox1 = Radiobutton(label_downsampleviddefault, text="1980 x 1080", variable=self.var1,value=1)
        self.checkbox2 = Radiobutton(label_downsampleviddefault, text="1280 x 1024", variable=self.var1,value=2)
        # Downsample video
        self.button_downsamplevideo2 = Button(label_downsampleviddefault, text='Downsample Default Resolution Video',command=self.downsample_defaultreso)

        # Organize the window
        self.videopath1selected.grid(row=0,sticky=W)
        label_choiceq.grid(row=1, sticky=W,pady=10)

        label_downsamplevidcustom.grid(row=2,sticky=W,pady=10)
        self.label_height.grid(row=0, column=0,sticky=W)
        self.label_width.grid(row=1, column=0,sticky=W)
        self.button_downsamplevideo1.grid(row=3)

        label_downsampleviddefault.grid(row=3,sticky=W,pady=10)
        self.checkbox1.grid(row=0,stick=W)
        self.checkbox2.grid(row=1,sticky=W)
        self.button_downsamplevideo2.grid(row=3)

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
            self.height2 = str(1024)
            print('The width selected is ' + str(self.width2) + ', the height is ' + str(self.height2))

        ds = downsamplevideo(self.width2, self.height2, self.videopath1selected.file_path)


class Red_light_Convertion:

    def __init__(self):
        # Popup window
        redlightconversion = Toplevel()
        redlightconversion.minsize(200, 200)
        redlightconversion.wm_title("Red Light Conversion Pipeline")

        #GreyScale
        label_greyscale = LabelFrame(redlightconversion,text='Greyscale',font='bold',padx=5,pady=5)
        # Video Path
        self.videopath1selected = FileSelect(label_greyscale, "Video Path")
        button_greyscale = Button(label_greyscale, text='Convert to greyscale',command=lambda:greyscale(self.videopath1selected.file_path))

        #CLAHE
        label_clahe = LabelFrame(redlightconversion,text='Contrast Limited Adaptive Histogram Equalization',font='bold',padx=5,pady=5)
        label_clahe2 = Label(label_clahe,text='You are only able to apply CLAHE using a grayscale video')
        button_clahe = Button(label_clahe,text='Apply CLAHE',command=lambda:clahe(self.videopath1selected.file_path))

        #organize the window
        label_greyscale.grid(row=0,sticky=W,pady=10)
        self.videopath1selected.grid(row=0,sticky=W)
        button_greyscale.grid(row=1,sticky=E,pady=5)

        label_clahe.grid(row=1,sticky=W,pady=10)
        label_clahe2.grid(row=0,sticky=W)
        button_clahe.grid(row=1,sticky=E,pady=5)

class crop_video:

    def __init__(self):
        # Popup window
        cropvideo = Toplevel()
        cropvideo.minsize(200, 200)
        cropvideo.wm_title("Crop Video")

        # Video Path
        label_cropvideo = LabelFrame(cropvideo,text='Crop Video',font='bold',padx=5,pady=5)
        self.videopath1selected = FileSelect(label_cropvideo,"Video Path")
        # CropVideo
        button_cropvid = Button(label_cropvideo, text='Crop Video', command=lambda:cropvid(self.videopath1selected.file_path))

        #organize
        label_cropvideo.grid(row=0,sticky=W)
        self.videopath1selected.grid(row=0,sticky=W)
        button_cropvid.grid(row=1,sticky=W,pady=10)


class create_project_DLC:

    def __init__(self):

        # Popup window
        createproject = Toplevel()
        createproject.minsize(400, 250)
        createproject.wm_title("Create Project")

        label_dlc_createproject = LabelFrame(createproject,text='Create Project',font ='bold')
        #project name
        self.label_projectname = Entry_Box(label_dlc_createproject,'Project Name','15')

        #Experimenter name
        self.label_experimentername = Entry_Box(label_dlc_createproject,'Experimenter Name','15')

        # Video Path
        self.videopath1selected = FileSelect(label_dlc_createproject,'Video Path                ')

        #video folder
        self.folderpath1selected = FolderSelect(label_dlc_createproject,'Working Directory   ')

        # statusbar
        self.projectcreated = IntVar()
        Label(createproject, textvariable=self.projectcreated, bd=1, relief=SUNKEN).grid(row=7,sticky=W)
        self.projectcreated.set('Status: Waiting for input...')

        #checkbox_apply golden aggresion config yaml settings
        self.var_changeyaml = IntVar()
        checkbox2 = Checkbutton(label_dlc_createproject,text='Apply Golden Aggression Config',variable=self.var_changeyaml)

        #run create project
        button_createproject = Button(label_dlc_createproject,text='Create Project',fg='red',command=self.createprojectcommand)

        #organize
        label_dlc_createproject.grid(row=0)
        self.label_projectname.grid(row=0,column=0,sticky=W)
        self.label_experimentername.grid(row=1,column=0,sticky=W)
        self.videopath1selected.grid(row=2,sticky=W)
        self.folderpath1selected.grid(row=3,sticky=W)
        checkbox2.grid(row=5,column=0,sticky=W)
        button_createproject.grid(row=6,column=3,pady=10,padx=5)

    def createprojectcommand(self):
        projectname = self.label_projectname.entry_get
        experimentalname = self.label_experimentername.entry_get

        if self.var_changeyaml.get()==1:
            config_path = deeplabcut.create_new_project(str(projectname), str(experimentalname), [str(self.videopath1selected.file_path)],working_directory=str(self.folderpath1selected.folder_path), copy_videos=True)
            changedlc_config(config_path)
            self.projectcreated.set('Project is created in ' + str(self.folderpath1selected.folder_path))

        else:
            config_path = deeplabcut.create_new_project(str(projectname), str(experimentalname), [str(self.videopath1selected.file_path)],working_directory=str(self.folderpath1selected.folder_path), copy_videos=True)
            self.projectcreated.set('Project is created in ' + str(self.folderpath1selected.folder_path))

class Load_DLC_Model:

    def __init__(self):
        # Popup window
        loadmodel = Toplevel()
        loadmodel.minsize(800, 700)
        loadmodel.wm_title("Load DLC Model")

        scrollbar_lm = Scrollable(loadmodel, width=25)

        #Load Model : configpath
        labelframe_loadmodel = LabelFrame(scrollbar_lm, text='Load Model', font='bold',padx=5,pady=5)
        self.label_set_configpath = FileSelect(labelframe_loadmodel, 'Select config path(.yaml): ')

        # # generate yaml file
        # label_generatetempyaml = LabelFrame(scrollbar_lm,text='Generate Temp yaml', font='bold',padx=5,pady=5)
        # self.label_genyamlsinglevideo = FileSelect(label_generatetempyaml,'Select Single Video:')
        # button_generatetempyaml_single = Button(label_generatetempyaml,text='Add single video',command=lambda:generatetempyaml(self.label_set_configpath.file_path,self.label_genyamlsinglevideo.file_path))
        # self.label_genyamlmultivideo = FolderSelect(label_generatetempyaml,'Select Folder with videos:')
        # button_generatetempyaml_multi = Button(label_generatetempyaml,text='Add multiple videos',command=self.generateyamlmulti)

        #singlevid multivid
        labelframe_singlemultivid = LabelFrame(scrollbar_lm,text='Add Videos into project',font='bold',padx=5,pady=5)
        labelframe_singlevid = LabelFrame(labelframe_singlemultivid,text='Single Video')
        labelframe_multivid = LabelFrame(labelframe_singlemultivid,text='Multiple Videos')
        self.label_set_singlevid = FileSelect(labelframe_singlevid, 'Select Single Video: ')
        self.label_video_folder = FolderSelect(labelframe_multivid, 'Select Folder with videos:')
        button_add_single_video = Button(labelframe_singlevid,text='Add single video',command = lambda :deeplabcut.add_new_videos(self.label_set_configpath.file_path, [str(self.label_set_singlevid.file_path)],copy_videos=True),fg='red')
        button_add_multi_video = Button(labelframe_multivid,text='Add multiple videos',command = self.dlc_addmultivideo_command,fg='red')

        ###########extract frames########
        label_extractframes = LabelFrame(scrollbar_lm, text='Extract Frames DLC', font='bold',padx=15,pady=5)
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
        button_extractframe = Button(label_extractframes, text='Extract Frame', command=self.dlc_extractframes_command)

        ##########label Frames#####
        label_labelframes = LabelFrame(scrollbar_lm, text='Label Frames', font='bold',padx=15,pady=5)
        self.button_label_frames = Button(label_labelframes, text='Label Frames', command=self.dlc_label_frames_command)

        ##########Check Labels#####
        label_checklabels = LabelFrame(scrollbar_lm, text='Check Labels', font='bold',padx=15,pady=5)
        self.button_check_labels = Button(label_checklabels, text='Label Frames', command=self.dlc_check_labels_command)

        ####generate training sets#####
        label_generate_trainingsets = LabelFrame(scrollbar_lm,text='Generate Training Set',font ='bold',padx=15,pady=5)
        self.button_generate_trainingsets = Button(label_generate_trainingsets, text='Generate training Set',command=self.dlc_generate_trainingsets_command)

        #####train network####
        label_train_network = LabelFrame(scrollbar_lm,text= 'Train Network',font ='bold',padx=15,pady=5)
        self.label_iteration = Entry_Box(label_train_network,'iteration','10')
        self.button_update_iteration = Button(label_train_network,text='Update iteration',command =lambda:updateiteration(self.label_set_configpath.file_path,self.label_iteration.entry_get))
        self.init_weight = FileSelect(label_train_network,'init_weight     ')
        self.update_init_weight = Button(label_train_network,text='Update init_weight',command=lambda:update_init_weight(self.label_set_configpath.file_path,self.init_weight.file_path))
        self.button_train_network = Button(label_train_network, text='Train Network',command=self.dlc_train_network_command)

        #######evaluate network####
        label_eva_network = LabelFrame(scrollbar_lm,text='Evaluate Network',font = 'bold',padx=15,pady=5)
        self.button_evaluate_network = Button(label_eva_network, text='Evaluate Network',command=self.dlc_evaluate_network_command)

        #####video analysis####
        label_video_analysis = LabelFrame(scrollbar_lm,text='Video Analysis',font='bold',padx=15,pady=5)
        #singlevideoanalysis
        label_singlevideoanalysis = LabelFrame(label_video_analysis,text='Single Video Analysis',pady=5,padx=5)
        self.videoanalysispath = FileSelect(label_singlevideoanalysis, "Video Path")
        button_vidanalysis = Button(label_singlevideoanalysis, text='Single Video Analysis', command=self.dlc_video_analysis_command1)
        #multi video analysis
        label_multivideoanalysis = LabelFrame(label_video_analysis,text='Multiple Videos Analysis',pady=5,padx=5)
        self.videofolderpath = FolderSelect(label_multivideoanalysis,'Folder Path')
        self.video_type = Entry_Box(label_multivideoanalysis,'Video type(eg:mp4,avi):','18')
        button_multivideoanalysis = Button(label_multivideoanalysis,text='Multi Video Analysis',command=self.dlc_video_analysis_command2)

        #### plot####
        label_plot = LabelFrame(scrollbar_lm,text='Plot Video Graph',font='bold',padx=15,pady=5)
        # videopath
        self.videoplotpath = FileSelect(label_plot, "Video Path")
        # plot button
        button_plot = Button(label_plot, text='Plot Results', command=self.dlc_plot_videoresults_command)

        #####create video####
        label_createvideo = LabelFrame(scrollbar_lm,text='Create Video',font='bold',padx=15,pady=5)
        # videopath
        self.createvidpath = FileSelect(label_createvideo, "Video Path")
        # save frames
        self.var_saveframes = IntVar()
        checkbox_saveframes = Checkbutton(label_createvideo, text='Save Frames', variable=self.var_saveframes)
        # create video button
        button_createvideo = Button(label_createvideo, text='Create Video', command=self.dlc_create_video_command)

        ######Extract Outliers####
        label_extractoutlier = LabelFrame(scrollbar_lm,text='Extract Outliers',font='bold',pady=5,padx=5)
        self.label_extractoutliersvideo = FileSelect(label_extractoutlier,'Videos to correct:')
        button_extractoutliers = Button(label_extractoutlier,text='Extract Outliers',command =lambda:deeplabcut.extract_outlier_frames(self.label_set_configpath.file_path, [str(self.label_extractoutliersvideo.file_path)],automatic=True) )

        ####label outliers###
        label_labeloutliers = LabelFrame(scrollbar_lm,text='Label Outliers',font ='bold',pady=5,padx=5)
        button_refinelabels = Button(label_labeloutliers,text='Refine Outliers',command=lambda:deeplabcut.refine_labels(self.label_set_configpath.file_path))

        ####merge labeled outliers ###
        label_mergeoutliers = LabelFrame(scrollbar_lm,text='Merge Labeled Outliers',font='bold',pady=5,padx=5)
        button_mergelabeledoutlier = Button(label_mergeoutliers,text='Merge Labeled Outliers',command=lambda:deeplabcut.merge_datasets(self.label_set_configpath.file_path))

        #organize
        labelframe_loadmodel.grid(row=0,sticky=W,pady=5)
        self.label_set_configpath.grid(row=0,sticky=W)

        # label_generatetempyaml.grid(row=1,sticky=W)
        # self.label_genyamlsinglevideo.grid(row=0,sticky=W)
        # button_generatetempyaml_single.grid(row=1,sticky=W)
        # self.label_genyamlmultivideo.grid(row=2,sticky=W)
        # button_generatetempyaml_multi.grid(row=3,sticky=W)

        labelframe_singlemultivid.grid(row=2,sticky=W,pady=5)
        labelframe_singlevid.grid(row=0,sticky=W)
        self.label_set_singlevid.grid(row=0,sticky=W)
        button_add_single_video.grid(row=1,sticky=W)
        labelframe_multivid.grid(row=1,sticky=W)
        self.label_video_folder.grid(row=0,sticky=W)
        button_add_multi_video.grid(row=1,sticky=W)

        label_extractframes.grid(row=3, sticky=W,pady=5)
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

        label_labelframes.grid(row=4,sticky=W,pady=5)
        self.button_label_frames.grid(row=0,sticky=W)

        label_checklabels.grid(row=5,sticky=W,pady=15)
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

        scrollbar_lm.update()

    def generateyamlmulti(self):
        config_path = self.label_set_configpath.file_path
        directory = self.label_genyamlmultivideo.folder_path
        filesFound = []

        ########### FIND FILES ###########
        for i in os.listdir(directory):
            if i.__contains__(".mp4"):
                a = os.path.join(directory, i)
                filesFound.append(a)
                print(a)
        print(filesFound)

        generatetempyaml_multi(config_path,filesFound)


    def dlc_addmultivideo_command(self):
        config_path = self.label_set_configpath.file_path
        directory = self.label_video_folder.folder_path
        filesFound = []

        ########### FIND FILES ###########
        for i in os.listdir(directory):
            if i.__contains__(".mp4"):
                a = os.path.join(directory, i)
                deeplabcut.add_new_videos(config_path, [str(a)], copy_videos=True)

        print("Videos added.")

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

        print(config_path,modes,algorithm,clusterstep,clusterresizewidth,clustercolor,useopencv)
        deeplabcut.extract_frames(config_path,mode=modes,algo=algorithm,crop=False,userfeedback=False,cluster_step=clusterstep,cluster_resizewidth=clusterresizewidth,cluster_color=clustercolor,opencv=useopencv)


    def dlc_label_frames_command(self):
        config_path = self.label_set_configpath.file_path
        deeplabcut.label_frames(config_path)

    def dlc_check_labels_command(self):
        config_path = self.label_set_configpath.file_path
        deeplabcut.check_labels(config_path)

    def dlc_generate_trainingsets_command(self):
        config_path = self.label_set_configpath.file_path
        deeplabcut.create_training_dataset(config_path, num_shuffles=1)

    def dlc_train_network_command(self):
        config_path = self.label_set_configpath.file_path
        deeplabcut.train_network(config_path, shuffle=1, gputouse=0)

    def dlc_evaluate_network_command(self):
        config_path = self.label_set_configpath.file_path
        deeplabcut.evaluate_network(config_path, plotting=True)

    def dlc_video_analysis_command1(self):

        config_path = self.label_set_configpath.file_path

        vid_name = os.path.basename(self.videoanalysispath.file_path)
        vid_type = vid_name[-4:]

        deeplabcut.analyze_videos(config_path, [str(self.videoanalysispath.file_path)], shuffle=1,save_as_csv=True, videotype=vid_type)

    def dlc_video_analysis_command2(self):

        config_path = self.label_set_configpath.file_path

        folder_path = self.videofolderpath.folder_path
        vid_type = self.video_type.entry_get

        deeplabcut.analyze_videos(config_path, [str(folder_path)], shuffle=1,save_as_csv=True, videotype=vid_type)

    def dlc_plot_videoresults_command(self):
        config_path = self.label_set_configpath.file_path

        deeplabcut.plot_trajectories(config_path, [str(self.videoplotpath.file_path)])

    def dlc_create_video_command(self):
        config_path = self.label_set_configpath.file_path

        if self.var_saveframes==1:
            saveframes=True

        else:
            saveframes=False

        vid_name = os.path.basename(self.createvidpath.file_path)
        vid_type = vid_name[-4:]

        deeplabcut.create_labeled_video(config_path, [str(self.createvidpath.file_path)],save_frames=saveframes, videotype=vid_type)


class shorten_video:

    def __init__(self):
        # Popup window
        shortenvid = Toplevel()
        shortenvid.minsize(200, 200)
        shortenvid.wm_title("Cut Video")

        # videopath
        self.videopath1selected = FileSelect(shortenvid, "Video Path")

        #timeframe for start and end cut
        label_cutvideomethod1 = LabelFrame(shortenvid,text='Method 1',font='bold',padx=5,pady=5)
        label_timeframe = Label(label_cutvideomethod1, text='Please enter the timeframe in the given format (hh:mm:ss)')
        self.label_starttime = Entry_Box(label_cutvideomethod1,'Start at:','8')
        self.label_endtime = Entry_Box(label_cutvideomethod1, 'End at:',8)
        CreateToolTip(label_cutvideomethod1,
                      'Method 1 will retrieve the specified time input.(eg: input of Start at: 00:00:00, End at: 00:01:00, will create a new video from the chosen video from the very start till it reaches the first minute of the video')

        #express time frame
        label_cutvideomethod2 = LabelFrame(shortenvid,text='Method 2',font='bold',padx=5,pady=5)
        self.var_express = IntVar()
        checkbox_express = Checkbutton(label_cutvideomethod2, text='Check this box to use Method 2', variable=self.var_express)
        self.label_time = Entry_Box(label_cutvideomethod2,'Seconds:','8')
        CreateToolTip(label_cutvideomethod2,'Method 2 will retrieve from the end of the video.(eg: an input of 180 seconds will create a new video with the last 180 second from the chosen video ')

        #button to cut video
        button_cutvideo = Button(shortenvid, text='Cut Video!', command=self.shortenvideocommand)

        # ###
        # label_cutvideomethod3 = LabelFrame(shortenvid, text='Method 1', font='bold', padx=5, pady=5)
        # self.label_fps = Entry_Box(label_cutvideomethod3, 'fps:', '8')
        # self.label_startingframe = Entry_Box(label_cutvideomethod3, 'Starting frame:', 8)
        # button_cutvideobyframe = Button(label_cutvideomethod3,text='Cut Video By Frame',command=lambda:shortenvideosbyframes(self.videopath1selected.file_path,self.label_fps.entry_get,self.label_startingframe.entry_get))

        #organize
        self.videopath1selected.grid(row=0,sticky=W)

        label_cutvideomethod1.grid(row=1,sticky=W,pady=5)
        label_timeframe.grid(row=0,sticky=W)
        self.label_starttime.grid(row=1,sticky=W)
        self.label_endtime.grid(row=2,sticky=W)

        label_cutvideomethod2.grid(row=2,sticky=W,pady=5)
        checkbox_express.grid(row=0,sticky=W)
        self.label_time.grid(row=2,sticky=W)

        button_cutvideo.grid(row=3,pady=5)

        # label_cutvideomethod3.grid(row=4,sticky=W,pady=5)
        # self.label_fps.grid(row=0,sticky=W)
        # self.label_startingframe.grid(row=1,sticky=W)
        # button_cutvideobyframe.grid(row=2,sticky=W)

    def shortenvideocommand(self):

        if self.var_express.get() == 0:
            print('Method 1 was chosen')
            starttime = self.label_starttime.entry_get
            endtime = self.label_endtime.entry_get

            vid = shortenvideos1(self.videopath1selected.file_path,starttime,endtime)


        elif self.var_express.get()==1:
            print('Method 2 was chosen')
            time = self.label_time.entry_get
            vid = shortenvideos2(self.videopath1selected.file_path,time)

class change_imageformat:

    def __init__(self):

        # Popup window
        chgimgformat = Toplevel()
        chgimgformat.minsize(200, 200)
        chgimgformat.wm_title("Change image format")

        #select directory
        self.folderpath1selected = FolderSelect(chgimgformat,"Working Directory")

        #change image format
        label_filetypein = LabelFrame(chgimgformat,text= 'Original image format')
        # Checkbox input
        self.varfiletypein = IntVar()
        checkbox_c1 = Radiobutton(label_filetypein, text=".png", variable=self.varfiletypein, value=1)
        checkbox_c2 = Radiobutton(label_filetypein, text=".jpeg", variable=self.varfiletypein, value=2)
        checkbox_c3 = Radiobutton(label_filetypein, text=".bmp", variable=self.varfiletypein, value=3)

        #ouput image format
        label_filetypeout = LabelFrame(chgimgformat,text='Output image format')
        #checkbox output
        self.varfiletypeout = IntVar()
        checkbox_co1 = Radiobutton(label_filetypeout, text=".png", variable=self.varfiletypeout, value=1)
        checkbox_co2 = Radiobutton(label_filetypeout, text=".jpeg", variable=self.varfiletypeout, value=2)
        checkbox_co3 = Radiobutton(label_filetypeout, text=".bmp", variable=self.varfiletypeout, value=3)

        #button
        button_changeimgformat = Button(chgimgformat, text='Convert Image Format', command=self.changeimgformatcommand)

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

        print('Image converted to '+str(cif))


class convert_video:

    def __init__(self):
        # Popup window
        convertvid = Toplevel()
        convertvid.minsize(200, 200)
        convertvid.wm_title("Convert Video")
        # videopath
        self.videopath1selected = FileSelect(convertvid, "Video Path")


        #checkbox
        label_convert = LabelFrame(convertvid,text='Convert format')
        self.vvformat = IntVar()
        checkbox_v1 = Radiobutton(label_convert, text="Convert .avi to .mp4", variable=self.vvformat, value=1)
        checkbox_v2 = Radiobutton(label_convert, text="Convert mp4 into Powerpoint supported format", variable=self.vvformat, value=2)

        #button
        button_convertvid= Button(label_convert, text='Convert video format', command=self.convertavitomp)

        #organize
        self.videopath1selected.grid(row=0,column=0)
        label_convert.grid(row=1,column=0)
        checkbox_v1.grid(row=2,column=0,sticky=W)
        checkbox_v2.grid(row=3,column=0,sticky=W)
        button_convertvid.grid(row=4,column=0,pady=10)

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
        extractsf.wm_title("Extract Specific Frames")

        # videopath
        self.videopath1selected = FileSelect(extractsf, "Video Path")

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
    extractaf.minsize(200, 200)
    extractaf.wm_title("Extract All Frames")

    # videopath
    videopath = FileSelect(extractaf, "Video Path")

    #button
    button_extractaf = Button(extractaf, text='Extract All Frames', command= lambda:extract_allframescommand(videopath.file_path))

    #organize
    videopath.grid(row=0,column=0)
    button_extractaf.grid(row=1,column=0)

class mergeframeffmpeg:

    def __init__(self):
        # Popup window
        mergeffmpeg = Toplevel()
        mergeffmpeg.minsize(250, 250)
        mergeffmpeg.wm_title("Merge images to movie")

        # select directory
        self.folderpath1selected = FolderSelect(mergeffmpeg,"Working Directory")

        # settings
        label_settings = LabelFrame(mergeffmpeg,text='Settings',padx=5,pady=5)
        self.label_imgformat = Entry_Box(label_settings, 'Image Format','10')
        label_to = Label(label_settings,text=' to ',width=0)
        self.label_vidformat = Entry_Box(label_settings,'Video Format','10')
        self.label_bitrate = Entry_Box(label_settings,'Bitrate','10')
        self.label_fps = Entry_Box(label_settings,'fps','10')


        #button
        button_mergeimg = Button(label_settings, text='Merge Images!', command=self.mergeframeffmpegcommand)

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
        self.vmerge.set('Videos Created!')

class new_window:

    def open_Folder(self):
        print("Current directory is %s" % os.getcwd())
        folder = askdirectory()
        os.chdir(folder)
        print("Working directory is %s" % os.getcwd())
        # self.label = Label(new_window, text="Folder selected: %s" % os.getcwd()).grid(row=3, column=1)

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

class get_coordinates_from_video:

    def __init__(self):
        # Popup window
        getcoord = Toplevel()
        getcoord.minsize(200, 200)
        getcoord.wm_title('Get Coordinates in Video')

        # settings files selected
        self.videopath1selected = FileSelect(getcoord, "Video selected")

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
        toplevel = Toplevel()
        toplevel.minsize(580, 800)
        toplevel.wm_title("Project Configuration")

        projectconfig = Scrollable(toplevel, width=25)

        # General Settings
        label_generalsettings = LabelFrame(projectconfig, text='[General Settings]',fg='blue',font ='bold',padx=5,pady=5)
        self.directory1Select = FolderSelect(label_generalsettings, "Project Path:", )
        self.label_project_name = Entry_Box(label_generalsettings, 'Project Name:', '0')
        label_project_namedescrip = Label(label_generalsettings, text='(project name cannot contain spaces)')

        #SML Settings
        self.label_smlsettings = LabelFrame(label_generalsettings, text='[SML Settings]',padx=5,pady=5)
        self.label_notarget = Entry_Box(self.label_smlsettings,'Number of predictive classifiers (behaviors):','33')
        addboxButton = Button(self.label_smlsettings, text='<Add predictive classifier>', fg="Red", command=lambda:self.addBox(projectconfig))

        #Frame settings
        label_frame_settings = LabelFrame(label_generalsettings, text='[Video Settings]',padx=5,pady=5)
        label_framesettings_des = Label(label_frame_settings,text='(videos should be preprocessed using \'Tools\')')
        label_framesettings_des_2 = Label(label_frame_settings,text='(These settings can be changed later for each individual videos)')
        self.label_fps_settings = Entry_Box(label_frame_settings, 'Frames per second (fps):','19')
        self.label_resolution_width = Entry_Box(label_frame_settings,'Resolution width (x-axis):','19')
        self.label_resolution_height = Entry_Box(label_frame_settings,'Resolution height (y-axis):','19')

        #generate project ini
        button_generateprojectini = Button(label_generalsettings, text='Generate Project Config ', command=self.make_projectini, font=(50),fg='red')

        #####import videos
        label_importvideo = LabelFrame(projectconfig, text='Import Videos into project folder',fg='blue', font='bold', padx=15, pady=5)
        # multi video
        label_multivideoimport = LabelFrame(label_importvideo, text='Import Multiple Videos', pady=5, padx=5)
        self.multivideofolderpath = FolderSelect(label_multivideoimport, 'Folder Path')
        self.video_type = Entry_Box(label_multivideoimport, 'Video type(eg:mp4,avi):', '18')
        button_multivideoimport = Button(label_multivideoimport, text='Import multiple videos',command=lambda: copy_multivideo_ini(self.configinifile,self.multivideofolderpath.folder_path,self.video_type.entry_get),fg='red')
        # singlevideo
        label_singlevideoimport = LabelFrame(label_importvideo, text='Import Single Video', pady=5, padx=5)
        self.singlevideopath = FileSelect(label_singlevideoimport, "Video Path")
        button_importsinglevideo = Button(label_singlevideoimport, text='Import a video',command=lambda: copy_singlevideo_ini(self.configinifile,self.singlevideopath.file_path),fg='red')


        #import all csv file into project folder
        label_import_csv = LabelFrame(projectconfig,text='Import DLC Tracking Data',fg='blue',font='bold',pady=5,padx=5)
        #multicsv
        label_multicsvimport = LabelFrame(label_import_csv, text='Import Multiple csv files', pady=5, padx=5)
        self.folder_csv = FolderSelect(label_multicsvimport,'Folder Select:')
        button_import_csv = Button(label_multicsvimport,text='Import csv to project folder',command=lambda:copy_allcsv_ini(self.configinifile,self.folder_csv.folder_path),fg='red')
        #singlecsv
        label_singlecsvimport = LabelFrame(label_import_csv, text='Import single csv files', pady=5, padx=5)
        self.file_csv = FileSelect(label_singlecsvimport,'File Select')
        button_importsinglecsv = Button(label_singlecsvimport,text='Import single csv to project folder',command=lambda :copy_singlecsv_ini(self.configinifile,self.file_csv.file_path),fg='red')


        #extract videos in projects
        label_extractframes = LabelFrame(projectconfig,text='Extract Frames into project folder',fg='blue',font='bold',pady=5,padx=5)
        button_extractframes = Button(label_extractframes,text='Extract frames',command=self.extract_frames,fg='red')

        #organize
        label_generalsettings.grid(row=0,sticky=W)
        self.directory1Select.grid(row=1,column=0,sticky=W)
        self.label_project_name.grid(row=2,column=0,sticky=W)
        label_project_namedescrip.grid(row=3,sticky=W)

        self.label_smlsettings.grid(row=4,column=0,sticky=W,pady=5)
        self.label_notarget.grid(row=0,column=0,sticky=W,pady=5)
        addboxButton.grid(row=1,column=0,sticky=W,pady=6)

        label_frame_settings.grid(row=5,column=0,sticky=W,pady=5)
        label_framesettings_des.grid(row=0,sticky=W)
        label_framesettings_des_2.grid(row=1,sticky=W)
        self.label_fps_settings.grid(row=2,column=0,sticky=W)
        self.label_resolution_width.grid(row=3,column=0,sticky=W)
        self.label_resolution_height.grid(row=4,column=0,sticky=W)

        button_generateprojectini.grid(row=6, pady=5, ipadx=5, ipady=5)

        label_importvideo.grid(row=4,sticky=W,pady=5)
        label_multivideoimport.grid(row=0, sticky=W)
        self.multivideofolderpath.grid(row=0, sticky=W)
        self.video_type.grid(row=1, sticky=W)
        button_multivideoimport.grid(row=2, sticky=W)
        label_singlevideoimport.grid(row=1,sticky=W)
        self.singlevideopath.grid(row=0,sticky=W)
        button_importsinglevideo.grid(row=1,sticky=W)


        label_import_csv.grid(row=5,sticky=W,pady=5)
        label_multicsvimport.grid(row=0,sticky=W)
        self.folder_csv.grid(row=0,sticky=W)
        button_import_csv.grid(row=1,sticky=W)
        label_singlecsvimport.grid(row=1,sticky=W)
        self.file_csv.grid(row=0,sticky=W)
        button_importsinglecsv.grid(row=1,sticky=W)


        label_extractframes.grid(row=6,sticky=W)
        button_extractframes.grid(row=0,sticky=W)

        projectconfig.update()

    def addBox(self,scroll):

        frame = Frame(self.label_smlsettings)
        frame.grid()

        # I use len(all_entries) to get nuber of next free column
        next_column = len(self.all_entries)

        # add label in first row
        lab = Label(frame, text=str('classifier_') + str(next_column + 1))
        lab.grid(row=0, column=0,sticky=W)

        ent1 = Entry(frame)
        ent1.grid(row=0, column=1,sticky=W)
        self.all_entries.append(ent1)

        self.modelPath = StringVar()
        modelpath = Button(frame,text='Browse',command=self.setFilepath)
        modelpath.grid(row=0,column=2,sticky=W)
        CreateToolTip(modelpath,'Path to existing model. If generating a new model, leave path blank')

        chosenfile = Entry(frame,textvariable=self.modelPath,width=80)
        chosenfile.grid(row=0,column=3,sticky=W)
        self.modelPath.set('No classifier is selected')

        self.allmodels.append(chosenfile)
        scroll.update()

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

        model_list = []
        for numbers, ent2 in enumerate(self.allmodels):
            model_list.append(ent2.get())


        #frame settings
        fpssettings = self.label_fps_settings.entry_get
        resolution_width = self.label_resolution_width.entry_get
        resolution_height = self.label_resolution_height.entry_get

        try:
           self.configinifile = write_inifile(model_list,msconfig,project_path,project_name,no_targets,target_list,fpssettings,resolution_width,resolution_height)
           print(self.configinifile)

        except:
            print('Please fill in the information correctly.')

        dest1 = str(str(project_path)+'\\'+str(project_name) +'\\models\\')
        count=0
        for f in model_list:
            try:
                filetocopy = os.path.basename(f)

                if filetocopy[-4:] != '.sav':
                    print('The classifier selected is not a .sav file')

                elif os.path.exists(dest1+str(target_list[count])+'.sav'):
                    print(f, 'already exist in', dest1)

                elif not os.path.exists(dest1 + '\\' + filetocopy):
                    print('Copying previous models...')
                    shutil.copy(f, dest1)
                    os.rename(dest1+filetocopy,dest1+str(target_list[count])+'.sav')
                    print(f, 'copied to', dest1)
                    count += 1
            except FileNotFoundError:
                print('No path inserted: The user has decided to create their own models')

        print('Finished generating Project Config.')

    def extract_frames(self):
        videopath = str(os.path.dirname(self.configinifile) + '\\videos')

        extract_frames_ini(videopath)

class createprojectinibatch:
    def __init__(self):
        projinibatch = Toplevel()
        projinibatch.minsize(800, 700)
        projinibatch.wm_title("Load project for batch analysis")

        scroll = Scrollable(projinibatch, width=32)

        label_settings = LabelFrame(scroll, text='Settings', font='bold',fg='blue', pady=5, padx=5)
        # load project ini
        label_loadprojectini = LabelFrame(label_settings, text='Load Master Project .ini', pady=5, padx=5)
        self.projectconfigini = FileSelect(label_loadprojectini, 'File Select:')

        # singlecsv
        label_import_csv = LabelFrame(label_settings, text='Import DLC Tracking Data', pady=5, padx=5)
        label_singlecsvimport = LabelFrame(label_import_csv, text='Import single csv files', pady=5, padx=5)
        self.file_csv = FileSelect(label_singlecsvimport, 'File Select')

        # Frame settings
        label_frame_settings = LabelFrame(label_settings, text='[Video Settings]', padx=5, pady=5)
        self.label_fps_settings = Entry_Box(label_frame_settings, 'Frames per second (fps):', '19')
        self.label_resolution_width = Entry_Box(label_frame_settings, 'Resolution width (x-axis):', '19')
        self.label_resolution_height = Entry_Box(label_frame_settings, 'Resolution height (y-axis):', '19')

        # get coordinates
        label_setscale = LabelFrame(label_settings, text='Set Video parameters(distances,resolution,etc.)', pady=5, padx=5)
        self.distanceinmm = Entry_Box(label_setscale,'Distance in mm','15')
        button_setdistanceinmm = Button(label_setscale,text='Auto populate Distance in mm in tables',command=lambda:self.set_distancemm(self.distanceinmm.entry_get))
        button_setscale = Button(label_setscale, text='Set Video Parameters', command=lambda:video_info_table(self.projectconfigini.file_path))

        # outlier correction
        label_outliercorrection = LabelFrame(label_settings, text='Outlier correction', pady=5, padx=5)
        self.movement_criterion = Entry_Box(label_outliercorrection, 'Movement criterion', '15')
        self.location_criterion = Entry_Box(label_outliercorrection, 'Location criterion', '15')

        # plotpathing
        label_plotall = LabelFrame(label_settings, text='Plot Graphs', pady=5, padx=5)
        # path plot
        label_pathplot = LabelFrame(label_plotall, text='Path plot', pady=5, padx=5)
        self.Deque_points = Entry_Box(label_pathplot, 'Max Lines', '15')
        self.severity_brackets = Entry_Box(label_pathplot, 'Severity Scale: (0 - ', '15')
        self.Bodyparts = Entry_Box(label_pathplot, 'Bodyparts', '15')
        self.plotsvvar = IntVar()
        checkboxplotseverity = Checkbutton(label_pathplot, text='plot_severity', variable=self.plotsvvar)
        # distanceplot
        label_distanceplot = LabelFrame(label_plotall, text='Distance plot', pady=5, padx=5)
        self.poi1 = Entry_Box(label_distanceplot, 'POI_1', '15')
        self.poi2 = Entry_Box(label_distanceplot, 'POI_2', '15')

        #generate
        button_generatevideoini = Button(label_settings,text='Generate individual ini',font='bold',fg='red',command =self.generatevideoinicommand)

        #RUN
        label_run = LabelFrame(scroll,text='Run', font='bold',fg='blue', pady=5, padx=5)
        button_correct_outlier = Button(label_run,text='Correct Outlier',command =self.correct_outlier)
        button_extractfeatures = Button(label_run,text='Extract Features',command=lambda:extract_features_wotarget(self.projectconfigini.file_path))
        button_runmachinemodel = Button(label_run,text='Run RF Model',command=lambda:rfmodel(self.projectconfigini.file_path))
        button_plotsklearnr = Button(label_run, text='Plot Sklearn Results',command=lambda: plotsklearnresult(self.projectconfigini.file_path))
        button_ganttplot = Button(label_run, text='Generate Gantt plot',command=lambda: ganntplot_config(self.projectconfigini.file_path))
        button_dataplot = Button(label_run, text='Generate Data plot',command=lambda: data_plot_config(self.projectconfigini.file_path))
        button_pathplot = Button(label_run, text='Generate Path plot',command=lambda: path_plot_config(self.projectconfigini.file_path))
        button_distanceplot = Button(label_run, text='Generate Distance plot',command=lambda: line_plot_config(self.projectconfigini.file_path))
        button_mergeframe = Button(label_run, text='Merge Frames',command=lambda: merge_frames_config(self.projectconfigini.file_path))
        button_createvideo = Button(label_run, text='Create Video',command=lambda: generatevideo_config_ffmpeg(self.projectconfigini.file_path))

        #organize
        label_settings.grid(row=0,sticky=W)
        label_loadprojectini.grid(row=0,sticky=W)
        self.projectconfigini.grid(row=0,sticky=W)

        label_import_csv.grid(row=1,sticky=W)
        label_singlecsvimport.grid(row=0,sticky=W)
        self.file_csv.grid(row=0,sticky=W)

        label_frame_settings.grid(row=2,sticky=W)
        self.label_fps_settings.grid(row=0,sticky=W)
        self.label_resolution_width.grid(row=1,sticky=W)
        self.label_resolution_height.grid(row=2,sticky=W)

        label_setscale.grid(row=3,sticky=W)
        self.distanceinmm.grid(row=0,sticky=W)
        button_setdistanceinmm.grid(row=1,sticky=W)
        button_setscale.grid(row=2,sticky=W)

        label_outliercorrection.grid(row=4,sticky=W)
        self.movement_criterion.grid(row=0,sticky=W)
        self.location_criterion.grid(row=1,sticky=W)

        label_plotall.grid(row=5,sticky=W)
        label_pathplot.grid(row=0,sticky=W)
        self.Deque_points.grid(row=0,sticky=W)
        self.severity_brackets.grid(row=1,sticky=W)
        self.Bodyparts.grid(row=2,sticky=W)
        checkboxplotseverity.grid(row=3,sticky=W)
        label_distanceplot.grid(row=1,sticky=W)
        self.poi1.grid(row=0,sticky=W)
        self.poi2.grid(row=1,sticky=W)

        button_generatevideoini.grid(row=6,sticky=W)

        label_run.grid(row=1,sticky=W)
        button_correct_outlier.grid(row=0,sticky=W,pady=5)
        button_extractfeatures.grid(row=1,sticky=W,pady=5)
        button_runmachinemodel.grid(row=2,sticky=W,pady=5)
        button_plotsklearnr.grid(row=3,sticky=W,pady=5)
        button_ganttplot.grid(row=4,sticky=W,pady=5)
        button_dataplot.grid(row=5,sticky=W,pady=5)
        button_pathplot.grid(row=6,sticky=W,pady=5)
        button_distanceplot.grid(row=7,sticky=W,pady=5)
        button_mergeframe.grid(row=8,sticky=W,pady=5)
        button_createvideo.grid(row=9,sticky=W,pady=5)


        scroll.update()

    def set_distancemm(self,distancemm):
        configini = self.projectconfigini.file_path
        config=ConfigParser()
        config.read(configini)

        config.set('Frame settings', 'distance_mm',distancemm)
        with open(configini, 'w') as configfile:
            config.write(configfile)

    def generatevideoinicommand(self):
        ###first step copy to configs and rename
        masterconfigini = self.projectconfigini.file_path
        csvname = (os.path.basename(self.file_csv.file_path))[:-4]
        config = ConfigParser()
        config.read(masterconfigini)

        config.set('General settings', 'use_master_config', str('no'))
        with open(masterconfigini, 'w') as configfile:
            config.write(configfile)

        dest1 = config.get('General settings','config_folder')
        try:
            filetocopy = os.path.basename(masterconfigini)
            if os.path.exists(dest1 + '\\' + str(csvname) + '.ini'):
                print(filetocopy, 'already exist in', dest1)

            elif not os.path.exists(dest1 + '\\' + filetocopy):
                print('Copying previous models...')
                shutil.copy(masterconfigini, dest1)
                os.rename(dest1 + '\\'+filetocopy, dest1+'\\' + str(csvname) + '.ini')
                print(filetocopy, 'copied to', dest1)

        except FileNotFoundError:
            print('No path inserted: The user has decided to create their own models')

        ##second is to changae fps and shit and what not
        newinifile = str(dest1 + '\\' + str(csvname) + '.ini')
        fps = self.label_fps_settings.entry_get
        width = self.label_resolution_width.entry_get
        height = self.label_resolution_height.entry_get
        ppm = self.ppm
        movementcriterion = self.movement_criterion.entry_get
        locationcriterion = self.location_criterion.entry_get
        poi1 = self.poi1.entry_get
        poi2 = self.poi2.entry_get
        bodyparts = self.Bodyparts.entry_get
        dequepoints = self.Deque_points.entry_get
        severitybracket = self.severity_brackets.entry_get

        if self.plotsvvar.get()==0:
            plotserverity = 'no'
        else:
            plotserverity = 'yes'

        config = ConfigParser()
        config.read(newinifile)

        config.set('Frame settings', 'fps', str(fps))
        config.set('Frame settings', 'resolution_width', str(width))
        config.set('Frame settings', 'resolution_height', str(height))
        config.set('Frame settings', 'mm_per_pixel', str(ppm))

        config.set('Outlier settings', 'movement_criterion', str(movementcriterion))
        config.set('Outlier settings', 'location_criterion', str(locationcriterion))

        config.set('Distance plot', 'poi_1', str(poi1))
        config.set('Distance plot', 'poi_2', str(poi2))

        config.set('Line plot settings', 'bodyparts', str(bodyparts))

        config.set('Path plot settings', 'deque_points', str(dequepoints))
        config.set('Path plot settings', 'plot_severity', str(plotserverity))
        config.set('Path plot settings', 'severity_brackets', str(severitybracket))

        with open(newinifile, 'w') as configfile:
            config.write(configfile)
        print('fps set: ' + str(fps))
        print('resolution_height set: ' + str(height))
        print('resolution_width set: ' + str(width))
        print('mm_per_pixel set:',str(ppm))
        print('movement_criterion set:',str(movementcriterion))
        print('location_criterion set:', str(locationcriterion))
        print('poi_1 set:', str(poi1))
        print('poi_2 set:', str(poi2))
        print('bodyparts set:', str(bodyparts))
        print('deque_points set:', str(dequepoints))
        print('plot_severity set:', str(plotserverity))
        print('severity_brackets set:', str(severitybracket))

        print('Finished generating configs.ini')

    def correct_outlier(self):
        configini = self.projectconfigini.file_path
        dev_move(configini)
        dev_loc(configini)


class loadprojectini:
    def __init__(self):
        simongui = Toplevel()
        simongui.minsize(1000, 700)
        simongui.wm_title("Load project")

        scroll = Scrollable(simongui,width=32)

        #load project ini
        label_loadprojectini = LabelFrame(scroll,text='Load Project .ini',font='bold',pady=5,padx=5,fg='blue')
        self.projectconfigini= FileSelect(label_loadprojectini,'File Select:')

        #label import
        label_import = LabelFrame(scroll)

        #import all csv file into project folder
        label_import_csv = LabelFrame(label_import, text='Import further DLC Tracking Data', font='bold', pady=5, padx=5,fg='blue')
        # multicsv
        label_multicsvimport = LabelFrame(label_import_csv, text='Import Multiple csv files', pady=5, padx=5)
        self.folder_csv = FolderSelect(label_multicsvimport, 'Folder Select:')
        button_import_csv = Button(label_multicsvimport, text='Import csv to project folder',command=lambda: copy_allcsv_ini(self.projectconfigini.file_path, self.folder_csv.folder_path),fg='red')
        # singlecsv
        label_singlecsvimport = LabelFrame(label_import_csv, text='Import single csv files', pady=5, padx=5)
        self.file_csv = FileSelect(label_singlecsvimport, 'File Select')
        button_importsinglecsv = Button(label_singlecsvimport, text='Import Single csv to project folder',command=lambda: copy_singlecsv_ini(self.projectconfigini.file_path, self.file_csv.file_path),fg='red')


        #import videos
        label_importvideo = LabelFrame(label_import, text='Import further Videos into project folder', font='bold', padx=15,pady=5,fg='blue')
        # multi video
        label_multivideoimport = LabelFrame(label_importvideo, text='Import Multiple Videos', pady=5, padx=5)
        self.multivideofolderpath = FolderSelect(label_multivideoimport, 'Folder Path')
        self.video_type = Entry_Box(label_multivideoimport, 'Video type(eg:mp4,avi):', '18')
        button_multivideoimport = Button(label_multivideoimport, text='Import multiple videos',command=lambda: copy_multivideo_ini(self.projectconfigini.file_path,self.multivideofolderpath.folder_path, self.video_type.entry_get), fg='red')
        # singlevideo
        label_singlevideoimport = LabelFrame(label_importvideo, text='Import Single Video', pady=5, padx=5)
        self.singlevideopath = FileSelect(label_singlevideoimport, "Video Path")
        button_importsinglevideo = Button(label_singlevideoimport, text='Import a video',
                                          command=lambda: copy_singlevideo_ini(self.projectconfigini.file_path,self.singlevideopath.file_path),fg='red')


        #extract frames in project folder
        label_extractframes = LabelFrame(label_import, text='Extract Additional Frames into project folder', font='bold', pady=5,padx=5,fg='blue')
        button_extractframes = Button(label_extractframes, text='Extract frames', command=self.extract_frames_loadini)

        #get coordinates
        label_setscale = LabelFrame(scroll,text='Set video parameters (distances,resolution,etc.)', font='bold', pady=5,padx=5,fg='blue')
        self.distanceinmm = Entry_Box(label_setscale, 'Distance in mm', '15')
        button_setdistanceinmm = Button(label_setscale, text='Auto populate Distance in mm in tables',command=lambda: self.set_distancemm(self.distanceinmm.entry_get))
        button_setscale = Button(label_setscale,text='Set Video Parameters',command=lambda:video_info_table(self.projectconfigini.file_path))

        #outlier correction
        label_outliercorrection = LabelFrame(scroll,text='Outlier correction',font='bold',pady=5,padx=5,fg='blue')
        label_link = Label(label_outliercorrection,text='[link to description]',cursor='hand2',font='Verdana 10 underline')
        self.movement_criterion = Entry_Box(label_outliercorrection,'Movement criterion','15')
        self.location_criterion = Entry_Box(label_outliercorrection, 'Location criterion', '15')
        button_outliercorrection = Button(label_outliercorrection,text='Correct outlier',command=self.correct_outlier)

        label_link.bind("<Button-1>",lambda e: self.callback('https://github.com/sgoldenlab/social_tracker/blob/master/Outlier_correction.pdf'))


        #extract features
        label_extractfeatures = LabelFrame(scroll,text='Extract Features',font='bold',pady=5,padx=5,fg='blue')
        button_extractfeatures = Button(label_extractfeatures,text='Extract Features',command=lambda:extract_features_wotarget(self.projectconfigini.file_path))

        #label Behavior
        label_labelaggression = LabelFrame(scroll,text='Label Behavior',font='bold',pady=5,padx=5,fg='blue')
        button_labelaggression = Button(label_labelaggression, text='Select folder with frames',command=chooseFolder )

        #train machine model
        label_trainmachinemodel = LabelFrame(scroll,text='Train Machine Model',font='bold',padx=5,pady=5,fg='blue')
        button_trainmachinemodel = Button(label_trainmachinemodel,text='Train RF Model',command=lambda:RF_trainmodel(self.projectconfigini.file_path))

        #run machine model
        label_runmachinemodel = LabelFrame(scroll,text='Run Machine Model',font='bold',padx=5,pady=5,fg='blue')
        button_runmachinemodel = Button(label_runmachinemodel,text='Run RF Model',command=lambda:rfmodel(self.projectconfigini.file_path))

        # machine results
        label_machineresults = LabelFrame(scroll,text='Analyze Machine Results',font='bold',padx=5,pady=5,fg='blue')
        button_process_datalog = Button(label_machineresults,text='Analyze',command=lambda:analyze_process_data_log(self.projectconfigini.file_path))
        button_process_movement = Button(label_machineresults,text='Analyze distances/velocity',command=lambda:analyze_process_movement(self.projectconfigini.file_path))
        button_process_severity = Button(label_machineresults,text='Analyze severity',command=lambda:analyze_process_severity(self.projectconfigini.file_path))

        #plot sklearn res
        label_plotsklearnr = LabelFrame(scroll,text='Plot Sklearn Results',font='bold',pady=5,padx=5,fg='blue')
        button_plotsklearnr = Button(label_plotsklearnr,text='Plot Sklearn Results',command =lambda:plotsklearnresult(self.projectconfigini.file_path))

        #plotpathing
        label_plotall = LabelFrame(scroll,text='Plot Graphs',font='bold',pady=5,padx=5,fg='blue')
        #ganttplot
        label_ganttplot = LabelFrame(label_plotall,text='Gantt plot',pady=5,padx=5)
        button_ganttplot = Button(label_ganttplot,text='Generate Gantt plot',command=lambda:ganntplot_config(self.projectconfigini.file_path))
        #dataplot
        label_dataplot = LabelFrame(label_plotall,text='Data plot',pady=5,padx=5)
        button_dataplot = Button(label_dataplot,text='Generate Data plot',command=lambda:data_plot_config(self.projectconfigini.file_path))
        #path plot
        label_pathplot = LabelFrame(label_plotall,text='Path plot',pady=5,padx=5)
        self.Deque_points = Entry_Box(label_pathplot,'Max Lines','15')
        self.severity_brackets = Entry_Box(label_pathplot,'Severity Scale: 0 - ','15')
        self.Bodyparts = Entry_Box(label_pathplot, 'Bodyparts', '15')
        self.plotsvvar = IntVar()
        checkboxplotseverity = Checkbutton(label_pathplot,text='plot_severity',variable=self.plotsvvar)
        button_pathplot = Button(label_pathplot,text='Generate Path plot',command=self.pathplotcommand)

        CreateToolTip(self.Deque_points,'Maximum number of path lines in deque list')
        CreateToolTip(self.Bodyparts, 'If golden aggression config: Nose, Left ear, Right ear, Centroid, Left lateral, Right lateral, Tail base, Tail end')
        CreateToolTip(self.severity_brackets,'Set number of brackets to severity measures')

        #distanceplot
        label_distanceplot = LabelFrame(label_plotall,text='Distance plot',pady=5,padx=5)
        self.poi1 = Entry_Box(label_distanceplot, 'Body part 1', '15')
        self.poi2 =Entry_Box(label_distanceplot, 'Body part 2', '15')
        button_distanceplot= Button(label_distanceplot,text='Generate Distance plot',command=self.distanceplotcommand)

        CreateToolTip(self.poi1,'The bodyparts from config yaml. eg: Ear_left_1,Ear_right_1,Nose_1,Center_1,Lateral_left_1,Lateral_right_1,Tail_base_1,Tail_end_1,Ear_left_2,Ear_right_2,Nose_2,Center_2,Lateral_left_2,Lateral_right_2,Tail_base_2,Tail_end_2')

        #Merge frames
        label_mergeframes = LabelFrame(scroll,text='Merge Frames',pady=5,padx=5,font='bold',fg='blue')
        button_mergeframe = Button(label_mergeframes,text='Merge Frames',command=lambda:merge_frames_config(self.projectconfigini.file_path))

        #create video
        label_createvideo = LabelFrame(scroll, text='Create Video', pady=5, padx=5,font='bold',fg='blue')
        self.bitrate = Entry_Box(label_createvideo,'Bitrate',8)
        self.fileformt = Entry_Box(label_createvideo,'File format',8)
        button_createvideo = Button(label_createvideo, text='Create Video',command=self.generate_video)

        #organize
        label_loadprojectini.grid(row=0,sticky=W)
        self.projectconfigini.grid(row=0,sticky=W)

        label_import.grid(row=1,sticky=W,pady=5)
        label_import_csv.grid(row=0, sticky=N, pady=5)
        label_multicsvimport.grid(row=0, sticky=W)
        self.folder_csv.grid(row=0, sticky=W)
        button_import_csv.grid(row=1, sticky=W)
        label_singlecsvimport.grid(row=1, sticky=W)
        self.file_csv.grid(row=0, sticky=W)
        button_importsinglecsv.grid(row=1, sticky=W)


        label_importvideo.grid(row=0,column=1, sticky=N, pady=5,padx=5)
        label_multivideoimport.grid(row=0, sticky=W)
        self.multivideofolderpath.grid(row=0, sticky=W)
        self.video_type.grid(row=1, sticky=W)
        button_multivideoimport.grid(row=2, sticky=W)
        label_singlevideoimport.grid(row=1, sticky=W)
        self.singlevideopath.grid(row=0, sticky=W)
        button_importsinglevideo.grid(row=1, sticky=W)


        label_extractframes.grid(row=0,column=2,sticky=N,pady=5,padx=5)
        button_extractframes.grid(row=0,sticky=W)

        label_setscale.grid(row=2,sticky=W,pady=5,padx=5)
        self.distanceinmm.grid(row=0,column=0,sticky=W)
        button_setdistanceinmm.grid(row=0,column=1)
        button_setscale.grid(row=1,column=0,sticky=W)

        label_outliercorrection.grid(row=3,sticky=W)
        label_link.grid(row=0,sticky=W)
        self.movement_criterion.grid(row=1,sticky=W)
        self.location_criterion.grid(row=2,sticky=W)
        button_outliercorrection.grid(row=3,sticky=W)

        label_extractfeatures.grid(row=4,sticky=W)
        button_extractfeatures.grid(row=0,sticky=W)

        label_labelaggression.grid(row=5,sticky=W)
        button_labelaggression.grid(row=0,sticky=W)

        label_trainmachinemodel.grid(row=6,sticky=W)
        button_trainmachinemodel.grid(row=0,sticky=W)

        label_runmachinemodel.grid(row=7,sticky=W)
        button_runmachinemodel.grid(row=0,sticky=W)

        label_machineresults.grid(row=8,sticky=W)
        button_process_datalog.grid(row=0,column=0,sticky=W,padx=3)
        button_process_movement.grid(row=0,column=1,sticky=W,padx=3)
        button_process_severity.grid(row=0,column=2,sticky=W,padx=3)

        label_plotsklearnr.grid(row=9,sticky=W)
        button_plotsklearnr.grid(row=0,sticky=W)

        label_plotall.grid(row=10,sticky=W)
        #gantt
        label_ganttplot.grid(row=0,sticky=W)
        button_ganttplot.grid(row=0,sticky=W)
        #data
        label_dataplot.grid(row=1,sticky=W)
        button_dataplot.grid(row=0,sticky=W)
        #path
        label_pathplot.grid(row=2,sticky=W)
        self.Deque_points.grid(row=0,sticky=W)
        self.severity_brackets.grid(row=2,sticky=W)
        self.Bodyparts.grid(row=3,sticky=W)
        checkboxplotseverity.grid(row=4,sticky=W)
        button_pathplot.grid(row=5,sticky=W)
        #distance
        label_distanceplot.grid(row=3,sticky=W)
        self.poi1.grid(row=1,sticky=W)
        self.poi2.grid(row=2,sticky=W)
        button_distanceplot.grid(row=3,sticky=W)

        label_mergeframes.grid(row=11,sticky=W)
        button_mergeframe.grid(row=0,sticky=W)

        label_createvideo.grid(row=12,sticky=W)
        self.bitrate.grid(row=0,sticky=W)
        self.fileformt.grid(row=1,sticky=W)
        button_createvideo.grid(row=2,sticky=W)

        scroll.update()

    def set_distancemm(self, distancemm):
        configini = self.projectconfigini.file_path
        config = ConfigParser()
        config.read(configini)

        config.set('Frame settings', 'distance_mm', distancemm)
        with open(configini, 'w') as configfile:
            config.write(configfile)

    def generate_video(self):
        configini = self.projectconfigini.file_path
        bitrate = self.bitrate.entry_get
        fileformat = '.'+ self.fileformt.entry_get

        config = ConfigParser()
        config.read(configini)

        config.set('Create movie settings', 'file_format', str(fileformat))
        config.set('Create movie settings', 'bitrate', str(bitrate))
        with open(configini, 'w') as configfile:
            config.write(configfile)

        generatevideo_config_ffmpeg(configini)

    def extract_frames_loadini(self):
        configini = self.projectconfigini.file_path
        videopath = str(os.path.dirname(configini) + '\\videos')

        extract_frames_ini(videopath)


    def correct_outlier(self):
        configini = self.projectconfigini.file_path
        config = ConfigParser()
        config.read(configini)

        config.set('Outlier settings','movement_criterion',self.movement_criterion.entry_get)
        config.set('Outlier settings','location_criterion',self.location_criterion.entry_get)
        with open(configini,'w') as configfile:
            config.write(configfile)

        dev_move(configini)
        dev_loc(configini)


    def distanceplotcommand(self):
        configini = self.projectconfigini.file_path
        config = ConfigParser()
        config.read(configini)

        config.set('Distance plot', 'POI_1', self.poi1.entry_get)
        config.set('Distance plot', 'POI_2', self.poi2.entry_get)
        with open(configini, 'w') as configfile:
            config.write(configfile)

        line_plot_config(configini)

    def pathplotcommand(self):
        configini = self.projectconfigini.file_path
        config = ConfigParser()
        config.read(configini)

        config.set('Path plot settings', 'Deque_points', self.Deque_points.entry_get)
        config.set('Path plot settings', 'severity_brackets', self.severity_brackets.entry_get)
        config.set('Line plot settings', 'Bodyparts', self.Bodyparts.entry_get)

        if self.plotsvvar.get()==1:
            config.set('Path plot settings', 'plot_severity', 'yes')
        else:
            config.set('Path plot settings', 'plot_severity', 'no')

        with open(configini, 'w') as configfile:
            config.write(configfile)

        path_plot_config(configini)

    def callback(self,url):
        webbrowser.open_new(url)

class batch_processvideo:

    def __init__(self):
        # Popup window
        batchprocess = Toplevel()
        batchprocess.minsize(800, 700)
        batchprocess.wm_title("Batch Process Video")

        scroll= Scrollable(batchprocess,width=32)

        #Video Selection Tab
        label_videoselection = LabelFrame(scroll,text='Folder Selection',font='bold',padx=5,pady=5)
        self.folder1Select = FolderSelect(label_videoselection,'Directory with videos in it:')

        #downsample video
        label_downsample = LabelFrame(scroll,text="Downsample Video",font='bold',padx=5,pady=5)
        self.height = Entry_Box(label_downsample,'Height','5')
        self.width = Entry_Box(label_downsample,'Width','5')
        button_downsample = Button(label_downsample,text='Downsample Video',command=lambda:downsamplevideo_batch(self.width.entry_get,self.height.entry_get,self.folder1Select.folder_path))

        #extract frames
        label_extractallframes = LabelFrame(scroll, text="Extract Frames", font='bold',padx=5,pady=5)
        button_extract_frames = Button(label_extractallframes,text='Extract Frames',command=lambda:extract_allframescommand_batch(self.folder1Select.folder_path))

        #add frame number to videos
        label_addframes = LabelFrame(scroll, text="Superimpose frames number on video", font='bold',padx=5,pady=5)
        button_add_frame_no = Button(label_addframes,text="Add Frame Number to video",command=lambda: superimposeframe_batch(self.folder1Select.folder_path))
        #greyscale
        label_greyscale = LabelFrame(scroll, text="Convert Video to Greyscale", font='bold',padx=5,pady=5)
        button_greyscale = Button(label_greyscale, text="Greyscale",command=lambda: greyscale_batch(self.folder1Select.folder_path))

        #shortenvideos
        label_shotenvid = LabelFrame(scroll, text="Shorten Videos", font='bold',padx=5,pady=5)
        label_shotenvid_des = Label(label_shotenvid, text="Please enter the time frame in the given format: HH:MM:SS")
        self.startat = Entry_Box(label_shotenvid,'Start at:', '5')
        self.endat = Entry_Box(label_shotenvid, 'End at:', '5')
        button_shortenvid = Button(label_shotenvid, text="Shorten Video",command=lambda: shortenvideos1_batch(self.folder1Select.folder_path,self.startat.entry_get,self.endat.entry_get))

        #CLAHE
        label_clahe = LabelFrame(scroll,text='CLAHE',font = 'bold',padx=5,pady=5)
        button_clahe = Button(label_clahe, text='CLAHE',command=lambda: clahe_batch(self.folder1Select.folder_path))

        # crop button
        label_cropvid = LabelFrame(scroll, text='Crop Video', font='bold', padx=5, pady=5)
        self.video1Select = FileSelect(label_cropvid, 'Video Selected:')
        button_cropvid = Button(label_cropvid, text="Compile crop video into queue",command=lambda: cropvid_batch(self.folder1Select.folder_path,self.video1Select.file_path))
        button_do_job = Button(label_cropvid, text="Execute", command=self.execute_command)

        #organize
        label_videoselection.grid(row=0,sticky=W)
        self.folder1Select.grid(row=1,sticky=W)

        #downsample
        label_downsample.grid(row=3,sticky=W)
        self.height.grid(row=1,sticky=W)
        self.width.grid(row=2,sticky=W)
        button_downsample.grid(row=3,sticky=W)

        #extractframes
        label_extractallframes.grid(row=4,sticky=W)
        button_extract_frames.grid(row=0,sticky=W)

        #imposeframes
        label_addframes.grid(row=5,sticky=W)
        button_add_frame_no.grid(row=1,sticky=W)

        #greyscale
        label_greyscale.grid(row=6,sticky=W)
        button_greyscale.grid(row=1,sticky=W)

        #shortenvid
        label_shotenvid.grid(row=7,sticky=W)
        label_shotenvid_des.grid(row=0,sticky=W)
        self.startat.grid(row=1,sticky=W)
        self.endat.grid(row=2,sticky=W)
        button_shortenvid.grid(row=3,sticky=W)

        #clahe
        label_clahe.grid(row=8,sticky=W)
        button_clahe.grid(row=0,sticky=W)

        #crop
        label_cropvid.grid(row=9,sticky=W)
        self.video1Select.grid(row=0,sticky=W)
        button_cropvid.grid(row=1,sticky=W)
        button_do_job.grid(row=2,sticky=W)

        scroll.update()


    def execute_command(self):
        self.filepath = str(self.folder1Select.folder_path) + '\\' + 'process_video_define.txt'

        with open(self.filepath) as fp:
            for cnt,line in enumerate(fp):
                #add probably if ffmpeg then this if not then other subprocess
                subprocess.call(line, shell=True, stdout=subprocess.PIPE)


        os.rename(self.filepath,'Processes_ran.txt')
        file = 'Processes_ran.txt'
        dir = str(str(self.folder1Select.folder_path) + '\\' + 'process_archieve')
        try:
            os.makedirs(dir)
            print("Directory ",dir, " Created ")
        except FileExistsError:
            print("Directory ", dir, " already exists")

        currentDT = datetime.datetime.now()
        currentDT = str(currentDT.month) + '_' + str(currentDT.day) + '_' + str(currentDT.year) + '_' + str(
            currentDT.hour) + 'hour' + '_' + str(currentDT.minute) + 'min' + '_' + str(currentDT.second) + 'sec'
        try:
            shutil.move(file,dir)
        except shutil.Error:
            os.rename(file, file[:-4] + str(currentDT) + '.txt')
            shutil.move(file[:-4] + str(currentDT) + '.txt',dir)

        print('Finished cropping all videos')

class Scrollable(Frame):
    """
       Make a frame scrollable with scrollbar on the right.
       After adding or removing widgets to the scrollable frame,
       call the update() method to refresh the scrollable area.
    """

    def __init__(self, frame, width=16):

        scrollbarX = Scrollbar(frame, width=width, orient='horizontal')
        scrollbarX.pack(side=BOTTOM, fill=X, expand=False)
        scrollbarY = Scrollbar(frame, width=width)
        scrollbarY.pack(side=RIGHT, fill=Y, expand=False)

        self.canvas = Canvas(frame, xscrollcommand=scrollbarX.set, yscrollcommand=scrollbarY.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbarX.config(command=self.canvas.xview)
        scrollbarY.config(command=self.canvas.yview)

        self.canvas.bind('<Configure>', self.__fill_canvas)

        # base class initialization
        Frame.__init__(self, frame)

        # assign this obj (the inner frame) to the windows item of the canvas
        self.windows_item = self.canvas.create_window(0,0, window=self, anchor=NW)

        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    def on_mousewheel(self, event):
        scrollSpeed = event.delta
        if platform.system() == 'Darwin':
            scrollSpeed = event.delta
        elif platform.system() == 'Windows':
            scrollSpeed = int(event.delta/120)
        self.canvas.yview_scroll(-1*(scrollSpeed), "units")

    def __fill_canvas(self, event):
        "Enlarge the windows item to the canvas width"

        canvas_width = event.width
        self.canvas.itemconfig(self.windows_item, width = canvas_width)

    def update(self):
        "Update the canvas and the scrollregion"

        self.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox(self.windows_item))




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

class App(object):
    def __init__(self):
        self.root = Tk()
        self.root.title('The Golden Lab')
        self.root.minsize(300,300)
        self.root.geometry("800x600")
        self.root.iconbitmap('golden_lab.ico')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        ### drop down menu###
        menu = Menu(self.root)
        self.root.config(menu=menu)

        #first menu
        fileMenu = Menu(menu)
        menu.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label='Create a new project', command=project_config)
        fileMenu.add_command(label='Load project', command=loadprojectini)
        fileMenu.add_command(label='Load project for batch analysis', command=createprojectinibatch)
        fileMenu.add_separator()
        fileMenu.add_command(label='Exit', command=Exit)

        # Process video
        pvMenu = Menu(menu)
        menu.add_cascade(label='Process Videos', menu=pvMenu)
        pvMenu.add_command(label='Process Videos in Batch', command=batch_processvideo)

        #third menu
        thirdMenu = Menu(menu)
        menu.add_cascade(label='DeepLabCut',menu=thirdMenu)
        thirdMenu.add_command(label='Create DLC Model',command=create_project_DLC)
        thirdMenu.add_command(label='Load DLC Model',command=Load_DLC_Model)

        #fifth menu
        fifthMenu = Menu(menu)
        menu.add_cascade(label='Tools',menu=fifthMenu)
        fifthMenu.add_command(label='Shortens Videos',command=shorten_video)
        fifthMenu.add_command(label='Crop Videos',command=crop_video)
        fifthMenu.add_command(label='Downsample',command=video_downsample)
        fifthMenu.add_command(label='Get Coordinates',command = get_coordinates_from_video)

        changeformatMenu = Menu(fifthMenu)
        changeformatMenu.add_command(label='Change image formats',command=change_imageformat)
        changeformatMenu.add_command(label='Change video formats',command=convert_video)
        fifthMenu.add_cascade(label='Change formats',menu=changeformatMenu)

        fifthMenu.add_command(label='Contrast(CLAHE)',command=Red_light_Convertion)
        fifthMenu.add_command(label='Add Frame Numbers',command=lambda:superimposeframe(askopenfilename()))
        fifthMenu.add_command(label='Convert to grayscale',command=lambda:greyscale(askopenfilename()))
        fifthMenu.add_command(label='Merge Frames to video',command=mergeframeffmpeg)

        extractframesMenu = Menu(fifthMenu)
        extractframesMenu.add_command(label='Extract Defined Frames',command=extract_specificframes)
        extractframesMenu.add_command(label='Extract Frames',command=extract_allframes)
        fifthMenu.add_cascade(label='Extract Frames',menu=extractframesMenu)


        #Status bar at the bottom
        self.frame = Frame(self.root, bd=2, relief=SUNKEN, width=300, height=300)
        self.frame.grid(row=0, column=0, sticky=E + N + W + S)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.txt = Text(self.frame)
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
        self.image = Image.open(r".\TheGoldenLab.png")
        self.imgSplash = ImageTk.PhotoImage(self.image)

    def Window(self):
        width, height = self.image.size
        halfwidth = (self.parent.winfo_screenwidth()-width)//2
        halfheight = (self.parent.winfo_screenheight()-height)//2
        self.parent.geometry("%ix%i+%i+%i" %(width, height, halfwidth,halfheight))
        Label(self.parent, image=self.imgSplash).pack()


if __name__ == '__main__':
    root = Tk()
    root.overrideredirect(True)
    progressbar = ttk.Progressbar(orient=HORIZONTAL, length=5000, mode='determinate')
    progressbar.pack(side="bottom")
    app = SplashScreen(root)
    progressbar.start()
    root.after(2000, root.destroy)
    root.mainloop()


app = App()
print('Welcome!')
app.root.mainloop()