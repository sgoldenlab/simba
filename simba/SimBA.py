__author__ = "Simon Nilsson", "JJ Choong"

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import seaborn as sns
import csv
from tkinter import *
from tkinter.filedialog import askopenfilename,askdirectory
from tkinter import tix, messagebox
from PIL import ImageTk
import PIL.Image
import tkinter.ttk as ttk
import webbrowser
from simba.plotly_create_h5 import *
from simba.tkinter_functions import (hxtScrollbar,
                                     DropDownMenu,
                                     Entry_Box,
                                     FileSelect,
                                     FolderSelect,
                                     CreateToolTip)
from simba.project_config_creator import ProjectConfigCreator
from simba.import_mars import MarsImporter
from simba.probability_graph_interactive import InteractiveProbabilityGrapher
from simba.Validate_model_one_video_run_clf import ValidateModelRunClf
from simba.cue_light_tools.cue_light_menues import CueLightAnalyzerMenu
from simba.path_plotter import PathPlotter
from simba.gantt_creator_mp import GanttCreatorMultiprocess
from simba.gantt_creator import GanttCreatorSingleProcess
from simba.data_plotter import DataPlotter
from simba.distance_plotter import DistancePlotter
from simba.import_videos_csv_project_ini import *
from simba.get_coordinates_tools_v2 import get_coordinates_nilsson
from simba.create_project_select_datatype import TrackingSelectorMenu
from simba.process_severity import analyze_process_severity
from simba.clf_validator import ClassifierValidationClips
from simba.validate_model_on_single_video import ValidateModelOneVideo
from simba.sleap_csv_importer import SleapCsvImporter
from simba.ROI_reset import *
from simba.ROI_feature_analyzer import ROIFeatureCreator
from simba.ROI_movement_analyzer import ROIMovementAnalyzer
from simba.ROI_feature_visualizer import ROIfeatureVisualizer
from simba.heat_mapper_clf import HeatMapperClf
from simba.outlier_scripts.outlier_corrector_movement import OutlierCorrecterMovement
from simba.outlier_scripts.outlier_corrector_location import OutlierCorrecterLocation
from simba.outlier_scripts.skip_outlier_correction import OutlierCorrectionSkipper
from simba.features_scripts.feature_extractor_16bp import ExtractFeaturesFrom16bps
from simba.features_scripts.feature_extractor_14bp import ExtractFeaturesFrom14bps
from simba.features_scripts.extract_features_14bp_from_16bp import extract_features_wotarget_14_from_16
from simba.features_scripts.extract_features_9bp import extract_features_wotarget_9
from simba.features_scripts.feature_extractor_8bp import ExtractFeaturesFrom8bps
from simba.features_scripts.feature_extractor_7bp import ExtractFeaturesFrom7bps
from simba.features_scripts.feature_extractor_4bp import ExtractFeaturesFrom4bps
from simba.features_scripts.feature_extractor_user_defined import UserDefinedFeatureExtractor
from simba.sklearn_plot_scripts.plot_sklearn_results_all import PlotSklearnResults
from simba.drop_bp_cords import define_bp_drop_down, reverse_dlc_input_files, bodypartConfSchematic
from simba.user_pose_config_creator import PoseConfigCreator
from simba.pose_reset import PoseResetter
from simba.probability_plot_creator import TresholdPlotCreatorSingleProcess
from simba.probability_plot_creator_mp import TresholdPlotCreatorMultiprocess
from simba.appendMars import append_dot_ANNOTT
from simba.dlc_multi_animal_importer import MADLC_Importer
from simba.BORIS_appender import BorisAppender
from simba.Directing_animals_analyzer import DirectingOtherAnimalsAnalyzer
from simba.Directing_animals_visualizer import DirectingOtherAnimalsVisualizer
from simba.solomon_importer import SolomonImporter
from simba.reverse_tracking_order import reverse_tracking_2_animals
from simba.pup_retrieval_1 import pup_retrieval_1
from simba.interpolate_pose import Interpolate
from simba.import_trk import *
from simba.remove_keypoints_in_pose import KeypointRemover
from simba.classifications_per_ROI import *
from simba.read_DANNCE_mat import import_DANNCE_file, import_DANNCE_folder
from simba.ethovision_import import ImportEthovision
from simba.sleap_import_new import ImportSLEAP
from simba.sleap_csv_importer import SleapCsvImporter
from simba.ROI_plot_new import ROIPlot
from simba.train_single_model import TrainSingleModel
from simba.train_mutiple_models_from_meta_new import TrainMultipleModelsFromMeta
from simba.run_model_new import RunModel
import urllib.request
from cefpython3 import cefpython as cef
import threading
import datetime
import atexit
from simba.batch_process_videos.batch_process_menus import BatchProcessFrame
from simba.video_info_table import VideoInfoTable
from simba.read_config_unit_tests import check_int
from simba.roi_tools.ROI_move_shape import *
from simba.roi_tools.ROI_image import *
from simba.roi_tools.ROI_zoom import *
from simba.roi_tools.ROI_define import *
from simba.roi_tools.ROI_menus import *
from simba.roi_tools.ROI_reset import *
from simba.misc_tools import (check_multi_animal_status,
                              smooth_data_gaussian,
                              smooth_data_savitzky_golay,
                              archive_processed_files)
from simba.video_processing import (video_to_greyscale,
                                    superimpose_frame_count)
from simba.setting_menu import SettingsMenu
from simba.pop_up_classes import (HeatmapLocationPopup,
                                  QuickLineplotPopup,
                                  ClfByROIPopUp,
                                  FSTTCPopUp,
                                  KleinbergPopUp,
                                  TimeBinsClfPopUp,
                                  ClfDescriptiveStatsPopUp,
                                  DownsampleVideoPopUp,
                                  CLAHEPopUp,
                                  CropVideoPopUp,
                                  ClipVideoPopUp,
                                  MultiShortenPopUp,
                                  ChangeImageFormatPopUp,
                                  ConertVideoPopUp,
                                  ExtractSpecificFramesPopUp,
                                  ExtractAllFramesPopUp,
                                  Csv2ParquetPopUp,
                                  Parquet2CsvPopUp,
                                  MultiCropPopUp,
                                  ChangeFpsSingleVideoPopUp,
                                  ChangeFpsMultipleVideosPopUp,
                                  ExtractSEQFramesPopUp,
                                  MergeFrames2VideoPopUp,
                                  PrintModelInfoPopUp,
                                  CreateGIFPopUP,
                                  CalculatePixelsPerMMInVideoPopUp,
                                  PoseReorganizerPopUp,
                                  MakePathPlotPopUp,
                                  AboutSimBAPopUp,
                                  ConcatenatingVideosPopUp,
                                  VisualizePoseInFolderPopUp,
                                  DropTrackingDataPopUp,
                                  ConcatenatorPopUp,
                                  SetMachineModelParameters,
                                  OutlierSettingsPopUp,
                                  RemoveAClassifierPopUp,
                                  VisualizeROIFeaturesPopUp,
                                  VisualizeROITrackingPopUp)
from simba.bounding_box_tools.boundary_menus import BoundaryMenus
from simba.labelling_interface import select_labelling_video
from simba.labelling_advanced_interface import select_labelling_video_advanced
from simba.deepethogram_importer import DeepEthogramImporter
import multiprocessing
import sys

sys.setrecursionlimit(10 ** 6)
simBA_version = 1.2
currentPlatform = platform.system()

class BatchPreProcessWindow(object):
    def __init__(self):
        batchprocess = Toplevel()
        batchprocess.minsize(400, 200)
        batchprocess.wm_title("Batch process video")
        label_videoselection = LabelFrame(batchprocess,text='Folder selection',font='bold',padx=5,pady=5)
        self.input_folder = FolderSelect(label_videoselection,'Video directory:',title='Select Folder with Input Videos')

        #output video
        self.output_folder = FolderSelect(label_videoselection,'Output directory:',title='Select Folder for Output videos')

        confirm_btn = Button(label_videoselection,text='Confirm',command=lambda: self.confirm_table())

        #organize
        label_videoselection.grid(row=0,sticky=W)
        self.input_folder.grid(row=0,sticky=W)
        self.output_folder.grid(row=1,sticky=W)
        confirm_btn.grid(row=2,sticky=W)

    def confirm_table(self):
        if (self.output_folder.folder_path=='No folder selected'):
            print('SIMBA ERROR: Please select an "Output directory"')
        elif (self.input_folder.folder_path == 'No folder selected'):
            print('SIMBA ERROR: Please select an "Input video directory"')
        elif (self.output_folder.folder_path == self.input_folder.folder_path):
            print('SIMBA ERROR: The input directory and output directory CANNOT be the same folder')
        else:
            batch_preprocessor = BatchProcessFrame(input_dir=self.input_folder.folder_path, output_dir=self.output_folder.folder_path)
            batch_preprocessor.create_main_window()
            batch_preprocessor.create_video_table_headings()
            batch_preprocessor.create_video_rows()
            batch_preprocessor.create_execute_btn()
            batch_preprocessor.batch_process_main_frame.mainloop()

class LoadProjectPopUp(object):
    def __init__(self):
        main_frm = Toplevel()
        main_frm.minsize(300, 200)
        main_frm.wm_title("Load SimBA project (project_config.ini file)")
        load_project_frm = LabelFrame(main_frm, text='LOAD SIMBA PROJECT_CONFIG.INI', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
        self.selected_file = FileSelect(load_project_frm,'Select file: ', title='Select project_config.ini file')
        load_project_btn = Button(load_project_frm, text='LOAD PROJECT', font=("Helvetica", 12, 'bold'), command=lambda: self.launch_project())

        load_project_frm.grid(row=0)
        self.selected_file.grid(row=0,sticky=NW)
        load_project_btn.grid(row=1,pady=10, sticky=NW)

    def launch_project(self):
        print('Loading {}...'.format(self.selected_file.file_path))
        check_file_exist_and_readable(file_path=self.selected_file.file_path)
        _ = loadprojectini(configini=self.selected_file.file_path)

def Exit():
    app.root.destroy()

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
        self.label_notarget = Entry_Box(self.label_smlsettings,'Number of predictive classifiers (behaviors):','33', validation='numeric')
        addboxButton = Button(self.label_smlsettings, text='<Add predictive classifier>', fg="navy",  command=lambda:self.addBox(self.label_notarget.entry_get))

        ##dropdown for # of mice
        #self.dropdownbox = TrackingSelectorMenu(main_frm=)
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
        self.filetype = DropDownMenu(self.label_import_csv,'File type',['CSV (DLC/DeepPoseKit)','JSON (BENTO)','H5 (multi-animal DLC)','SLP (SLEAP)', 'CSV (SLEAP)', 'TRK (multi-animal APT)', 'MAT (DANNCE 3D)'],'12',com=self.fileselected)
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

        if self.filetype.getChoices()== 'CSV (DLC/DeepPoseKit)':
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
            label_multicsvimport = LabelFrame(self.frame, text='Import multiple DANNCE files', pady=5, padx=5)
            self.folder_csv = FolderSelect(label_multicsvimport, 'Folder selected:', title='Select Folder with MAT files')
            button_import_csv = Button(label_multicsvimport, text='Import DANNCE files to project folder', command = lambda: import_DANNCE_folder(self.configinifile, self.folder_csv.folder_path, self.interpolation.getChoices()), fg='navy')
            label_singlecsvimport = LabelFrame(self.frame, text='Import single DANNCE files', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlecsvimport, 'File selected', title='Select a .csv file')
            button_importsinglecsv = Button(label_singlecsvimport, text='Import single DANNCE to project folder', command = lambda: import_DANNCE_file(self.configinifile, self.file_csv.file_path, self.interpolation.getChoices()), fg='navy')
            self.frame.grid(row=1, sticky=W)

            self.labelmethod.grid(row=0, sticky=W)
            self.smooth_pose_win_lbl.grid(row=1, sticky=W)
            label_multicsvimport.grid(row=2, sticky=W)
            self.folder_csv.grid(row=0, sticky=W)
            button_import_csv.grid(row=1, sticky=W)
            label_singlecsvimport.grid(row=3, sticky=W)
            self.file_csv.grid(row=0, sticky=W)
            button_importsinglecsv.grid(row=1, sticky=W)

        elif self.filetype.getChoices()=='JSON (BENTO)':

            label_multijsonimport = LabelFrame(self.frame, text='Import multiple json files', pady=5, padx=5)
            self.folder_json = FolderSelect(label_multijsonimport, 'Folder Select:',
                                            title='Select Folder with .json(s)')
            button_import_json = Button(label_multijsonimport, text='Import json to project folder', fg='navy', command=lambda: MarsImporter(config_path=self.configinifile,
                                                                                                                                              data_path=self.folder_json.folder_path,
                                                                                                                                              interpolation_method=self.interpolation.getChoices(),
                                                                                                                                              smoothing_method=self.smooth_dropdown.getChoices()))

            label_singlejsonimport = LabelFrame(self.frame, text='Import single json file', pady=5, padx=5)
            self.file_json = FileSelect(label_singlejsonimport, 'File Select', title='Select a .csv file')
            button_importsinglejson = Button(label_singlejsonimport, text='Import single .json to project folder', fg='navy', command=lambda: MarsImporter(config_path=self.configinifile,
                                                                                                                                                            data_path=self.folder_json.folder_path,
                                                                                                                                                            interpolation_method=self.interpolation.getChoices(),
                                                                                                                                                            smoothing_method=self.smooth_dropdown.getChoices()))
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

        elif self.filetype.getChoices() in ('H5 (multi-animal DLC)', 'SLP (SLEAP)', 'CSV (SLEAP)', 'TRK (multi-animal APT)'):
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
                self.dropdowndlc.grid(row=3, sticky=W)

            elif self.filetype.getChoices() == 'SLP (SLEAP)':
                self.h5path = FolderSelect(self.frame, 'Path to .slp files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the multi animal SLEAP tracking data')
                runsettings = Button(self.frame, text='Import .slp', command=self.importh5)

            elif self.filetype.getChoices() == 'TRK (multi-animal APT)':
                self.h5path = FolderSelect(self.frame, 'Path to .trk files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the multi animal trk tracking data')
                runsettings = Button(self.frame, text='Import .trk', command=self.importh5)

            elif self.filetype.getChoices() == 'CSV (SLEAP)':
                self.h5path = FolderSelect(self.frame, 'Path to CSV files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the multi animal trk tracking data')
                runsettings = Button(self.frame, text='Import .csv', command=self.importh5)



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

        config = read_config_file(ini_path=self.configinifile)
        config.set('Multi animal IDs', 'ID_list', ",".join(idlist))
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
            dlc_multi_animal_importer = MADLC_Importer(config_path=self.configinifile,
                                                       data_folder=self.h5path.folder_path,
                                                       file_type=self.dropdowndlc.getChoices(),
                                                       id_lst=idlist,
                                                       interpolation_settings=self.interpolation.getChoices(), smoothing_settings=smooth_settings_dict)
            dlc_multi_animal_importer.import_data()


        if self.filetype.getChoices() == 'SLP (SLEAP)':
            sleap_importer = ImportSLEAP(self.configinifile,self.h5path.folder_path,idlist, self.interpolation.getChoices(), smooth_settings_dict)
            sleap_importer.initate_import_slp()
            if sleap_importer.animals_no > 1:
                sleap_importer.visualize_sleap()
            sleap_importer.save_df()
            sleap_importer.perform_interpolation()
            sleap_importer.perform_smothing()
            print('All SLEAP imports complete.')

        if self.filetype.getChoices() == 'TRK (multi-animal APT)':
            try:
                import_trk(self.configinifile,self.h5path.folder_path,idlist, self.interpolation.getChoices(), smooth_settings_dict)
            except Exception as e:
                messagebox.showerror("Error", str(e))

        if self.filetype.getChoices() == 'CSV (SLEAP)':
            sleap_csv_importer = SleapCsvImporter(config_path=self.configinifile,
                                                  data_folder=self.h5path.folder_path,
                                                  actor_IDs=idlist,
                                                  interpolation_settings=self.interpolation.getChoices(),
                                                  smoothing_settings=smooth_settings_dict)
            sleap_csv_importer.initate_import_slp()
            print('SIMBA COMPLETE: Sleap CSV files imported to project_folder/csv/input_csv directory.')

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
        B1 = Button(popupframe, text='Yes', command=lambda: PoseResetter(master=popup))
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
        self.configname = Entry_Box(self.scroller,'Pose config name','23')

        # no of animals
        self.noOfAnimals = Entry_Box(self.scroller,'# of Animals','23')

        # no of bodyparts
        self.noOfBp = Entry_Box(self.scroller,'# of Bodyparts (per animal)','23')

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
        config_name = self.configname.entry_get
        self.no_animals = self.noOfAnimals.entry_get
        check_int(name='number of animals', value=self.no_animals)
        bp_cnt = self.noOfBp.entry_get
        check_int(name='body part count', value=bp_cnt)
        image_path = self.imgPath.file_path
        check_file_exist_and_readable(image_path)
        bp_lst, animal_id_lst = [], []
        for entry in zip(self.bpnamelist, self.bp_animal_list):
            bp_lst.append(entry[0].entry_get)
            if int(self.no_animals) > 1:
                check_int(name='Animal ID number', value=entry[1].entry_get)
                animal_id_lst.append(entry[1].entry_get)
        pose_config_creator = PoseConfigCreator(pose_name=config_name, no_animals=int(self.no_animals),
                                                img_path=image_path,
                                                bp_list=bp_lst,
                                                animal_id_int_list=animal_id_lst)
        pose_config_creator.launch()
        master.destroy()
        self.toplevel.destroy()
        project_config()
        print('SIMBA COMPLETE: Pose-configuration created.')

    def bpTable(self,master):
        try:
            self.table_frame.destroy()
        except:
            pass

        self.noAnimals = self.noOfAnimals.entry_get
        check_int(name='number of animals', value=self.noAnimals)
        self.noAnimals = int(self.noAnimals)
        noofbp = self.noOfBp.entry_get
        check_int(name='number of body-parts', value=noofbp)
        noofbp = int(noofbp)
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

        copy_singlevideo_ini(self.configinifile, self.singlevideopath.file_path)

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

        project_path = self.directory1Select.folder_path
        project_name = self.label_project_name.entry_get

        #sml settings
        no_targets = self.label_notarget.entry_get
        target_list = []
        for number, ent1 in enumerate(self.all_entries):
            target_list.append(ent1.get())

        ### animal settings
        listindex = self.option_mice.index(str(self.var.get()))
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

        animalNo = int(rows[listindex][0])
        config_creator = ProjectConfigCreator(project_path=project_path,
                                              project_name=project_name,
                                              target_list=target_list,
                                              pose_estimation_bp_cnt=bp,
                                              body_part_config_idx=listindex,
                                              animal_cnt=animalNo,
                                              file_type=self.csvORparquet.getChoices())

        self.configinifile = config_creator.config_path

    def extract_frames(self):
        try:
            videopath = os.path.join(os.path.dirname(self.configinifile), 'videos')
            print(videopath)
            extract_frames_from_all_videos_in_directory(videopath, self.configinifile)
        except Exception as e:
            print(e.args)
            print('Please make sure videos are imported and located in /project_folder/videos')

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
    def __init__(self,
                 configini=str):

        #save project ini as attribute
        self.projectconfigini = configini

        #bodyparts
        bodypartscsv= os.path.join((os.path.dirname(self.projectconfigini)),'logs','measures','pose_configs','bp_names','project_bp_names.csv')
        bp_set = pd.read_csv(bodypartscsv,header=None)[0].to_list()

        # get target
        config = ConfigParser()
        configFile = str(self.projectconfigini)
        config.read(configFile)
        self.config = config
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
        self.label_import_csv = LabelFrame(label_import, text='IMPORT FURTHER TRACKING DATA', font=("Helvetica",12,'bold'), pady=5, padx=5,fg='black')
        filetype = DropDownMenu(self.label_import_csv,'File type',['CSV (DLC/DeepPoseKit)','JSON (BENTO)','H5 (multi-animal DLC)', 'SLP (SLEAP)', 'CSV (SLEAP)', 'TRK (multi-animal APT)', 'MAT (DANNCE 3D)'],'15',com=self.fileselected)
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
        label_importvideo = LabelFrame(label_import, text='IMPORT FURTHER VIDEOS', font=("Helvetica",12,'bold'), padx=15,pady=5,fg='black')
        # multi video
        label_multivideoimport = LabelFrame(label_importvideo, text='Import multiple videos', pady=5, padx=5)
        self.multivideofolderpath = FolderSelect(label_multivideoimport, 'Folder path',title='Select Folder with videos')
        self.video_type = Entry_Box(label_multivideoimport, 'File format (i.e., mp4/avi):', '10')
        button_multivideoimport = Button(label_multivideoimport, text='Import multiple videos',command=self.importvideo_multi, fg='black')

        # singlevideo
        label_singlevideoimport = LabelFrame(label_importvideo, text='Import single video', pady=5, padx=5)
        self.singlevideopath = FileSelect(label_singlevideoimport, "Video Path",title='Select a video file')
        button_importsinglevideo = Button(label_singlevideoimport, text='Import a video',command= self.importvideo_single,fg='black')

        #extract frames in project folder
        label_extractframes = LabelFrame(label_import, text='EXTRACT FURTHER FRAMES', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        button_extractframes = Button(label_extractframes, text='Extract frames', command=self.extract_frames_loadini)

        #import frames
        label_importframefolder = LabelFrame(label_import, text='IMPORT FRAME FOLDERS', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        self.frame_folder = FolderSelect(label_importframefolder,'Main frame directory',title='Select the main directory with frame folders')
        button_importframefolder = Button(label_importframefolder,text='Import frames',command = self.importframefolder )

        #import new classifier
        label_newclassifier = LabelFrame(label_import,text='ADD NEW CLASSIFIER(S)', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        self.classifierentry = Entry_Box(label_newclassifier,'Classifier','8')
        button_addclassifier = Button(label_newclassifier,text='Add classifier',command=lambda:self.add_classifier(new_clf_name=self.classifierentry.entry_get))

        #remove classifier
        label_removeclassifier = LabelFrame(label_import,text='REMOVE EXISTING CLASSIFIER', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        button_removeclassifier = Button(label_removeclassifier,text='Choose a classifier to remove',command=lambda: RemoveAClassifierPopUp(config_path=self.projectconfigini))

        ## archive all csvs
        label_archivecsv =  LabelFrame(label_import,text='ARCHIVE PROCESSED FILES', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        archiveentrybox = Entry_Box(label_archivecsv,'Archive folder name', '16')
        button_archivecsv = Button(label_archivecsv,text='Archive',command = lambda: archive_processed_files(config_path=self.projectconfigini,
                                                                                                             archive_name=archiveentrybox.entry_get))

        #reverse identity
        label_reverseID = LabelFrame(label_import,text='REVERSE TRACKING IDENTITIES',font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        label_reverse_info = Label(label_reverseID,text='Note: This only works for 2 animals tracking')
        label_git_reverse = Label(label_reverseID, text='[Click here to learn more about the reverse identity process]', cursor='hand2', fg='blue')
        label_git_reverse.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/reverse_annotations.md'))
        reverse_button = Button(label_reverseID,text='Reverse ID',command=self.reverseid)

        #get coordinates
        label_setscale = LabelFrame(tab3,text='VIDEO PARAMETERS (fps, resolution, ppx/mm, etc.)', font=("Helvetica",12,'bold'), pady=5,padx=5,fg='black')
        self.distanceinmm = Entry_Box(label_setscale, 'Known distance (mm)', '18', validation='numeric')
        button_setdistanceinmm = Button(label_setscale, text='Auto-populate known distance in table',command=lambda: self.set_distancemm(self.distanceinmm.entry_get))

        button_setscale = Button(label_setscale, text='Set video parameters',command=lambda: self.create_video_info_table())
        #button_setscale = Button(label_setscale,text='Set video parameters',command=lambda:video_info_table(self.projectconfigini))

        self.new_ROI_frm = LabelFrame(tab6, text='SIMBA ROI INTERFACE', font=("Helvetica",12,'bold'))
        self.start_new_ROI = Button(self.new_ROI_frm, text='Define ROIs', command= lambda: ROI_menu(self.projectconfigini))
        self.delete_all_ROIs = Button(self.new_ROI_frm, text='Delete all ROI definitions', command=lambda: delete_all_ROIs(self.projectconfigini))
        self.tutorial_link = Label(self.new_ROI_frm, text='[Link to ROI user-guide]', cursor='hand2',  font='Verdana 8 underline')
        self.tutorial_link.bind("<Button-1>", lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md'))

        self.new_ROI_frm.grid(row=0, sticky=N)
        self.start_new_ROI.grid(row=0, sticky=W, pady=10)
        self.delete_all_ROIs.grid(row=0, column=2, sticky=W, pady=10, padx=10)
        self.tutorial_link.grid(row=1, sticky=W)

        self.roi_draw = LabelFrame(tab6, text='ANALYZE ROI', font=("Helvetica",12,'bold'))
        analyze_roi_btn = Button(self.roi_draw, text='Analyze ROI data',command=lambda: SettingsMenu(config_path=self.projectconfigini, title='ROI ANALYSIS'))
        analyze_roi_time_bins_btn = Button(self.roi_draw, text='Analyze ROI data: time-bins',command=lambda: SettingsMenu(config_path=self.projectconfigini, title='TIME BINS: ANALYZE ROI'))

        self.roi_draw.grid(row=0, column=1, sticky=N)
        analyze_roi_btn.grid(row=0)
        analyze_roi_time_bins_btn.grid(row=1,pady=55)

        ###plot roi
        self.roi_draw1 = LabelFrame(tab6, text='VISUALIZE ROI', font=("Helvetica",12,'bold'))

        # button
        visualizeROI = Button(self.roi_draw1, text='VISUALIZE ROI TRACKING', command= lambda: VisualizeROITrackingPopUp(config_path=self.projectconfigini))
        visualizeROIfeature = Button(self.roi_draw1, text='Visualize ROI features', command= lambda: VisualizeROIFeaturesPopUp(config_path=self.projectconfigini))

        ##organize
        self.roi_draw1.grid(row=0, column=2, sticky=N)
        visualizeROI.grid(row=0)
        visualizeROIfeature.grid(row=1,pady=10)

        processmovementdupLabel = LabelFrame(tab6,text='ANALYZE DISTANCES / VELOCITIES', font=("Helvetica",12,'bold'))
        analyze_distances_velocity_btn = Button(processmovementdupLabel, text='Analyze distances/velocity', command=lambda: SettingsMenu(config_path=self.projectconfigini, title='ANALYZE MOVEMENT'))
        analyze_distances_velocity_timebins_btn = Button(processmovementdupLabel, text='Time bins: Distance/velocity', command=lambda: SettingsMenu(config_path=self.projectconfigini, title='TIME BINS: DISTANCE/VELOCITY'))
        self.hmlvar = IntVar()
        self.hmlvar.set(1)

        heatmaps_location_button = Button(processmovementdupLabel,text='Create heat maps',command=lambda: HeatmapLocationPopup(config_path=self.projectconfigini))

        button_lineplot = Button(processmovementdupLabel, text='Generate path plot', command=lambda: QuickLineplotPopup(config_path=self.projectconfigini))
        button_analyzeDirection = Button(processmovementdupLabel,text='Analyze directionality between animals',command =lambda: self.directing_other_animals_analysis())
        button_visualizeDirection = Button(processmovementdupLabel,text='Visualize directionality between animals',command=lambda:self.directing_other_animals_visualizer())

        #organize
        processmovementdupLabel.grid(row=0,column=3,sticky=N)
        analyze_distances_velocity_btn.grid(row=0)
        heatmaps_location_button.grid(row=1)
        analyze_distances_velocity_timebins_btn.grid(row=2)
        button_lineplot.grid(row=3)
        button_analyzeDirection.grid(row=4)
        button_visualizeDirection.grid(row=5)

        #outlier correction
        label_outliercorrection = LabelFrame(tab4,text='OUTLIER CORRECTION',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        label_link = Label(label_outliercorrection,text='[Link to user guide]',cursor='hand2',font='Verdana 10 underline')
        button_settings_outlier = Button(label_outliercorrection,text='Settings',command = lambda: OutlierSettingsPopUp(config_path=self.projectconfigini))
        button_outliercorrection = Button(label_outliercorrection,text='Run outlier correction',command=self.correct_outlier)
        button_skipOC = Button(label_outliercorrection,text='Skip outlier correction (CAUTION)',fg='red', command=lambda: self.initiate_skip_outlier_correction())

        label_link.bind("<Button-1>",lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf'))

        #extract features
        label_extractfeatures = LabelFrame(tab5,text='EXTRACT FEATURES',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
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
        append_roi_features = Button(label_extractfeatures, text='Append ROI data to features (CAUTION)', fg='red',command=lambda: SettingsMenu(config_path=self.projectconfigini, title='APPEND ROI FEATURES'))
        append_roi_features.grid(row=10,pady=10)

        #label Behavior
        label_behavior_frm = LabelFrame(tab7,text='LABEL BEHAVIOR',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')

        select_video_btn_new = Button(label_behavior_frm, text='Select video (create new video annotation)',command= lambda:select_labelling_video(config_path=self.projectconfigini,
                                                                                                                                                   threshold_dict=None,
                                                                                                                                                   setting='from_scratch',
                                                                                                                                                   continuing=False))
        select_video_btn_continue = Button(label_behavior_frm, text='Select video (continue existing video annotation)', command= lambda:select_labelling_video(config_path=self.projectconfigini,
                                                                                                                                                   threshold_dict=None,
                                                                                                                                                   setting=None,
                                                                                                                                                   continuing=True))
        #select_video_btn_new = Button(label_behavior_frm, text='Select video (create new video annotation)',command= lambda:choose_folder(self.projectconfigini))
        #select_video_btn_continue = Button(label_behavior_frm, text='Select video (continue existing video annotation)',command= lambda:choose_folder(self.projectconfigini))



        #third party annotation
        label_thirdpartyann = LabelFrame(tab7,text='IMPORT THIRD-PARTY BEHAVIOR ANNOTATIONS',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        button_importmars = Button(label_thirdpartyann,text='Import MARS Annotation (select folder with .annot files)',command=self.importMARS)
        button_importboris = Button(label_thirdpartyann,text='Import Boris Annotation (select folder with .csv files)',command = self.importBoris)
        button_importsolomon = Button(label_thirdpartyann,text='Import Solomon Annotation (select folder with .csv files',command = self.importSolomon)
        button_importethovision = Button(label_thirdpartyann, text='Import Ethovision Annotation (select folder with .xls/xlsx files)', command=self.import_ethovision)
        button_importdeepethogram = Button(label_thirdpartyann,text='Import DeepEthogram Annotation (select folder with .csv files)', command=self.import_deepethogram)


        #pseudolabel
        label_pseudo = LabelFrame(tab7,text='PSEUDO-LABELLING',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        pseudo_intructions_lbl_1 = Label(label_pseudo,text='Note that SimBA pseudo-labelling require initial machine predictions.')
        pseudo_intructions_lbl_2 = Label(label_pseudo, text='Click here more information on how to use the SimBA pseudo-labelling interface.', cursor="hand2", fg="blue")
        pseudo_intructions_lbl_2.bind("<Button-1>", lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/docs/label_behavior.md'))
        pLabel_framedir = FileSelect(label_pseudo,'Video Path',lblwidth='10')
        plabelframe_threshold = LabelFrame(label_pseudo,text='Threshold',pady=5,padx=5)
        plabel_threshold =[0]*len(targetlist)
        for count, i in enumerate(list(targetlist)):
            plabel_threshold[count] = Entry_Box(plabelframe_threshold,str(i),'20')
            plabel_threshold[count].grid(row=count+2,sticky=W)

        pseudo_lbl_btn = Button(label_pseudo,text='Correct labels',command = lambda:select_labelling_video(config_path=self.projectconfigini,
                                                                                                           threshold_dict=dict(zip(list(targetlist), plabel_threshold)),
                                                                                                           setting='pseudo',
                                                                                                           continuing=False,
                                                                                                           video_file_path=pLabel_framedir.file_path))
        #pLabel_button = Button(label_pseudo,text='Correct label',command = lambda:semisuperviseLabel(self.projectconfigini,pLabel_framedir.file_path,list(targetlist),plabel_threshold))

        #Advance Label Behavior
        label_adv_label = LabelFrame(tab7,text='ADVANCED LABEL BEHAVIOR',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        label_adv_note_1 = Label(label_adv_label,text='Note that you will have to specify the presence of *both* behavior and non-behavior on your own.')
        label_adv_note_2 = Label(label_adv_label, text='Click here more information on how to use the SimBA labelling interface.', cursor="hand2", fg="blue")
        label_adv_note_2.bind("<Button-1>", lambda e: self.callback('https://github.com/sgoldenlab/simba/blob/master/docs/advanced_labelling.md'))
        adv_label_btn_new = Button(label_adv_label, text='Select video (create new video annotation)',command= lambda: select_labelling_video_advanced(config_path=self.projectconfigini,
                                                                                                                                          continuing=False))
        adv_label_btn_continue = Button(label_adv_label, text='Select video (continue existing video annotation)',command=lambda: select_labelling_video_advanced(config_path=self.projectconfigini, continuing=True))

        #train machine model
        label_trainmachinemodel = LabelFrame(tab8,text='TRAIN MACHINE MODELS',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        button_trainmachinesettings = Button(label_trainmachinemodel,text='Settings',command=self.trainmachinemodelsetting)
        button_trainmachinemodel = Button(label_trainmachinemodel,text='Train single model from global environment',fg='blue',command = lambda: threading.Thread(target=self.train_single_model(config_path=self.projectconfigini)).start())
        button_train_multimodel = Button(label_trainmachinemodel, text='Train multiple models, one for each saved settings',fg='green',command = lambda: threading.Thread(target=self.train_multiple_models_from_meta(config_path=self.projectconfigini)).start())

        ##Single classifier valid
        label_model_validation = LabelFrame(tab9, text='VALIDATE MODEL ON SINGLE VIDEO', pady=5, padx=5,
                                            font=("Helvetica", 12, 'bold'), fg='black')
        self.csvfile = FileSelect(label_model_validation, 'Select features file',
                                  title='Select .csv file in /project_folder/csv/features_extracted')
        self.modelfile = FileSelect(label_model_validation, 'Select model file  ', title='Select the model (.sav) file')
        #button_runvalidmodel = Button(label_model_validation, text='Run Model',command=lambda: validate_model_one_vid_1stStep(self.projectconfigini,self.csvfile.file_path,self.modelfile.file_path))
        button_runvalidmodel = Button(label_model_validation, text='Run Model',command=lambda: self.validate_model_first_step())

        button_generateplot = Button(label_model_validation, text="Generate plot", command=self.updateThreshold)
        self.dis_threshold = Entry_Box(label_model_validation, 'Discrimination threshold', '28')
        self.min_behaviorbout = Entry_Box(label_model_validation, 'Minimum behavior bout length (ms)', '28')
        self.generategantt_dropdown = DropDownMenu(label_model_validation, 'Create Gantt plot', ['None', 'Gantt chart: video', 'Gantt chart: final frame only (slightly faster)'], '15')
        self.generategantt_dropdown.setChoices('None')


        #self.ganttvar = IntVar()
        #self.generategantt = Checkbutton(label_model_validation,text='Generate Gantt plot',variable=self.ganttvar)
        button_validate_model = Button(label_model_validation, text='Validate', command=self.validatemodelsinglevid)

        #run machine model
        label_runmachinemodel = LabelFrame(tab9,text='RUN MACHINE MODEL',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        button_run_rfmodelsettings = Button(label_runmachinemodel,text='Model Settings',command= lambda: SetMachineModelParameters(config_path=self.projectconfigini))
        button_runmachinemodel = Button(label_runmachinemodel,text='Run RF Model',command = self.runrfmodel)

        #kleinberg smoothing
        kleinberg_button = Button(label_runmachinemodel,text='Kleinberg Smoothing',command=lambda: KleinbergPopUp(config_path=self.projectconfigini))
        fsttc_button = Button(label_runmachinemodel,text='FSTTC',command=lambda:FSTTCPopUp(config_path=self.projectconfigini))
        label_machineresults = LabelFrame(tab9,text='ANALYZE MACHINE RESULTS',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        button_process_datalog = Button(label_machineresults, text='Analyze machine predictions',command=lambda: ClfDescriptiveStatsPopUp(config_path=self.projectconfigini))
        button_process_movement = Button(label_machineresults, text='Analyze distances/velocity', command=lambda: SettingsMenu(config_path=self.projectconfigini, title='ANALYZE MOVEMENT'))
        button_movebins = Button(label_machineresults, text='Time bins: Distance/velocity', command=lambda: SettingsMenu(config_path=self.projectconfigini, title='TIME BINS: DISTANCE/VELOCITY'))
        button_classifierbins = Button(label_machineresults,text='Time bins: Machine predictions',command=lambda: TimeBinsClfPopUp(config_path=self.projectconfigini))
        button_classifier_ROI = Button(label_machineresults, text='Classifications by ROI', command=lambda: ClfByROIPopUp(config_path=self.projectconfigini))

        label_severity = LabelFrame(tab9,text='ANALYZE SEVERITY',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        self.severityscale = Entry_Box(label_severity,'Severity scale 0 -',15)
        try:
            self.severityTarget = DropDownMenu(label_severity,'Target',targetlist,'15')
        except TypeError:
            print('ERROR: No classifier names detected in your project_config.ini - please make sure you named your classifier(s) when creating your project')
        self.severityTarget.setChoices(targetlist[(config.get('SML settings', 'target_name_' + str(1)))])
        button_process_severity = Button(label_severity,text='Analyze target severity',command=self.analyzseverity)

        #plot sklearn res
        label_plotsklearnr = LabelFrame(tab10,text='SKLEARN VISUALIZATION',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')

        #lbl prob threshold
        lbl_probthreshold = LabelFrame(label_plotsklearnr,text='Body-part probability threshold',font=("Helvetica",12,'bold'),padx=5,pady=5,fg='black')
        lbl_thresexplain = Label(lbl_probthreshold,text='Bodyparts below the set threshold won\'t be shown in the output.', font=("Helvetica",12,'italic'))
        self.bpthres = Entry_Box(lbl_probthreshold,'Body-part probability threshold','32')
        self.use_default_font_settings_val = BooleanVar(value=True)
        self.auto_compute_font_cb = Checkbutton(lbl_probthreshold, text='Auto-compute font/key-point settings', variable=self.use_default_font_settings_val, command= lambda: self.enable_text_settings())
        self.sklearn_text_size_entry_box = Entry_Box(lbl_probthreshold, 'Text size: ', '12')
        self.sklearn_text_spacing_entry_box = Entry_Box(lbl_probthreshold, 'Text spacing: ', '12')
        self.sklearn_text_thickness_entry_box = Entry_Box(lbl_probthreshold, 'Text thickness: ', '12')
        self.sklearn_circle_size_entry_box = Entry_Box(lbl_probthreshold, 'Circle size: ', '12')
        self.sklearn_text_size_entry_box.set_state('disable')
        self.sklearn_text_spacing_entry_box.set_state('disable')
        self.sklearn_text_thickness_entry_box.set_state('disable')
        self.sklearn_circle_size_entry_box.set_state('disable')


        bpthresbutton = Button(lbl_probthreshold,text='Save threshold',command= self.savethres)

        #set bp threshold
        try:
            thres = config.get('threshold_settings', 'bp_threshold_sklearn')
            self.bpthres.entry_set(str(thres))
        except:
            self.bpthres.entry_set(0.0)

        #all videos
        label_skv_all = LabelFrame(label_plotsklearnr,text='Apply to all videos',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        self.videovar = BooleanVar()
        self.genframevar = BooleanVar()
        self.include_timers_var = BooleanVar()
        self.rotate_image_var = BooleanVar()
        videocheck = Checkbutton(label_skv_all,text='Generate video',variable=self.videovar)
        framecheck = Checkbutton(label_skv_all,text='Generate frames',variable=self.genframevar)
        timerscheck = Checkbutton(label_skv_all, text='Include timers overlay', variable=self.include_timers_var)
        rotate_check = Checkbutton(label_skv_all, text='Rotate video 90 degrees', variable=self.rotate_image_var)
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
        self.videovar2 = BooleanVar()
        self.genframevar2 = BooleanVar()
        self.include_timers_single_var = BooleanVar()
        self.rotate_single_var = BooleanVar()
        videocheck2 = Checkbutton(label_skv_single, text='Generate video', variable=self.videovar2)
        framecheck2 = Checkbutton(label_skv_single, text='Generate frames', variable=self.genframevar2)
        timers_single_check = Checkbutton(label_skv_single, text='Include timers overlay', variable=self.include_timers_single_var)
        rotate_single_check = Checkbutton(label_skv_single, text='Rotate video 90 degrees', variable=self.rotate_single_var)
        button_plotsklearnr2 = Button(label_skv_single, text='Visualize classification results',command=lambda: self.plotsklearnresultsingle(project_config_path=self.projectconfigini,
                                                                                                                                             rotate=self.rotate_single_var.get(),
                                                                                                                                             video_setting=self.videovar2.get(),
                                                                                                                                             frame_setting=self.genframevar2.get(),
                                                                                                                                             video_path=self.video_entry.getChoices(),
                                                                                                                                             timers=self.include_timers_single_var.get()))

        #plotpathing
        label_plotall = LabelFrame(tab10,text='DATA VISUALIZATIONS',font=("Helvetica",12,'bold'),pady=5,padx=5,fg='black')
        #ganttplot
        label_ganttplot = LabelFrame(label_plotall,text='Gantt plot',pady=5,padx=5)
        self.gannt_frames_var, self.gannt_videos_var, self.gantt_multiprocess_var = BooleanVar(), BooleanVar(), BooleanVar()
        gannt_frames_check = Checkbutton(label_ganttplot, text='Create frames', variable=self.gannt_frames_var)
        gannt_videos_check = Checkbutton(label_ganttplot, text='Create videos', variable=self.gannt_videos_var)
        gannt_multiprocess_check = Checkbutton(label_ganttplot, text='Multi-process (faster)', variable=self.gantt_multiprocess_var)
        button_ganttplot = Button(label_ganttplot, text='Generate gantt plot',command=lambda: self.create_gannt_plots())


        #dataplot
        label_dataplot = LabelFrame(label_plotall, text='Data plot', pady=5, padx=5)
        if pose_config_setting == 'user_defined':
            self.SelectedBp = DropDownMenu(label_dataplot, 'Select body part', bp_set, '15')
            self.SelectedBp.setChoices((bp_set)[0])
        button_dataplot = Button(label_dataplot, text='Generate data plot', command=self.plotdataplot)

        #PATH PLOT SETTINGS
        self.bp_set = bp_set
        self.path_plot_frm = LabelFrame(label_plotall,text='PATH PLOTS', pady=5,padx=5, )
        self.deque_points_entry = Entry_Box(self.path_plot_frm,'Max path lines','12', validation='numeric')
        self.path_animal_cnt_dropdown = DropDownMenu(self.path_plot_frm, 'Number of animals',list(range(1, int(animalNumber) + 1, 1)), '15')
        self.path_animal_cnt_dropdown.setChoices(1)
        self.confirm_animal_cnt = Button(self.path_plot_frm,text='Confirm',command=lambda:self.choose_animal_bps())
        path_plot_btn = Button(self.path_plot_frm,text='Create path plot',command=lambda: self.create_path_plots())
        self.path_plot_frames_var, self.path_plot_videos_var = BooleanVar(), BooleanVar()
        self.path_plot_frames_check = Checkbutton(self.path_plot_frm, text='Create frames', variable=self.path_plot_frames_var)
        self.path_plot_videos_check = Checkbutton(self.path_plot_frm, text='Create videos', variable=self.path_plot_videos_var)

        #distanceplot
        label_distanceplot = LabelFrame(label_plotall,text='Distance plot',pady=5,padx=5)
        self.poi1 = DropDownMenu(label_distanceplot,'Body part 1',bp_set,'15')
        self.poi2 = DropDownMenu(label_distanceplot,'Body part 2',bp_set,'15')
        #set choice
        self.poi1.setChoices((bp_set)[0])
        self.poi2.setChoices((bp_set)[len(bp_set)//2])
        self.distance_plot_frames_var = BooleanVar()
        self.distance_plot_videos_var = BooleanVar()
        self.distance_plot_frames_check = Checkbutton(label_distanceplot, text='Create frames', variable=self.distance_plot_frames_var)
        self.distance_plot_videos_check = Checkbutton(label_distanceplot, text='Create videos', variable=self.distance_plot_videos_var)
        button_distanceplot= Button(label_distanceplot,text='Generate Distance plot',command=self.distanceplotcommand)

        CreateToolTip(self.poi1,'The bodyparts from config yaml. eg: Ear_left_1,Ear_right_1,Nose_1,Center_1,Lateral_left_1,Lateral_right_1,Tail_base_1,Tail_end_1,Ear_left_2,Ear_right_2,Nose_2,Center_2,Lateral_left_2,Lateral_right_2,Tail_base_2,Tail_end_2')

        #Heatplot
        label_heatmap = LabelFrame(label_plotall, text='Heatmap', pady=5, padx=5)
        self.BinSize = Entry_Box(label_heatmap, 'Bin size (mm)', '15', validation='numeric')
        self.MaxScale = Entry_Box(label_heatmap, 'Max scale (s)', '15')

        hmchoices = {'viridis','plasma','inferno','magma','jet','gnuplot2'}
        self.hmMenu = DropDownMenu(label_heatmap,'Color Palette',hmchoices,'15')
        self.hmMenu.setChoices('jet')

        #get target called on top
        self.targetMenu = DropDownMenu(label_heatmap,'Classifier', targetlist,'15')
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
        self.heatmap_clf_frames_var = BooleanVar()
        self.heatmap_clf_videos_var = BooleanVar()
        self.heatmap_clf_last_img_var = BooleanVar()
        heatmap_frames_check = Checkbutton(label_heatmap, text='Create frames', variable=self.heatmap_clf_frames_var)
        heatmap_videos_check = Checkbutton(label_heatmap, text='Create videos',variable=self.heatmap_clf_videos_var)
        heatmap_last_img_check = Checkbutton(label_heatmap, text='Last image', variable=self.heatmap_clf_last_img_var)

        button_heatmap = Button(label_heatmap, text='Generate heatmap', command=self.heatmapcommand_clf)

        #plot threshold
        label_plotThreshold = LabelFrame(label_plotall,text='Plot Threshold',pady=5,padx=5)
        self.behaviorMenu = DropDownMenu(label_plotThreshold,'Target',targetlist,'15')
        self.behaviorMenu.setChoices(targetlist[(config.get('SML settings','target_name_'+str(1)))])
        self.threshold_frames_var, self.threshold_videos_var, self.threshold_multiprocess_var = BooleanVar(), BooleanVar(), BooleanVar()
        self.threshold_frames_check = Checkbutton(label_plotThreshold, text='Create frames', variable=self.threshold_frames_var)
        self.threshold_videos_check = Checkbutton(label_plotThreshold, text='Create videos', variable=self.threshold_videos_var)
        self.threshold_multiprocess_check = Checkbutton(label_plotThreshold, text='Multi-process (faster)', variable=self.threshold_multiprocess_var)
        plotThresholdButton = Button(label_plotThreshold, text='Plot classifier probabilities',command=lambda: self.create_threshold_plots())

        #Merge frames
        merge_frm = LabelFrame(tab10, text='MERGE FRAMES', pady=5, padx=5, font=("Helvetica", 12, 'bold'), fg='black')
        merge_frm_btn = Button(merge_frm, text='MERGE FRAMES', fg='black', command=lambda:ConcatenatorPopUp(config_path=self.projectconfigini))


        #Plotly
        plotlyInterface = LabelFrame(tab10, text= 'PLOTLY / DASH', font=("Helvetica", 12, 'bold'), pady=5, padx=5, fg='black')
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
        self.seconds = Entry_Box(label_classifier_validation,'Seconds','8', validation='numeric')
        self.cvTarget = DropDownMenu(label_classifier_validation,'Target',targetlist,'15')
        self.cvTarget.setChoices(targetlist[(config.get('SML settings', 'target_name_' + str(1)))])
        self.one_vid_per_bout_var, self.one_vid_per_video_var = BooleanVar(value=False), BooleanVar(value=False)
        individual_bout_clips_cb = Checkbutton(label_classifier_validation, text='One clip per bout', variable=self.one_vid_per_bout_var)
        individual_clip_per_video_cb = Checkbutton(label_classifier_validation, text='One clip per video', variable=self.one_vid_per_video_var)
        button_validate_classifier = Button(label_classifier_validation,text='Validate',command =self.classifiervalidation)

        #addons
        lbl_addon = LabelFrame(tab12,text='SimBA Expansions',pady=5, padx=5,font=("Helvetica",12,'bold'),fg='black')
        button_bel = Button(lbl_addon,text='Pup retrieval - Analysis Protocol 1',command = self.pupMenu)
        button_unsupervised = Button(lbl_addon,text='Unsupervised',command = lambda :unsupervisedInterface(self.projectconfigini))
        cue_light_analyser_btn = Button(lbl_addon, text='Cue light analysis', command=lambda: CueLightAnalyzerMenu(config_path=self.projectconfigini))
        anchored_roi_analysis_btn = Button(lbl_addon, text='Animal-anchored ROI analysis', command=lambda: BoundaryMenus(config_path=self.projectconfigini))

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

        label_behavior_frm.grid(row=5,sticky=W)
        select_video_btn_new.grid(row=0,sticky=W)
        select_video_btn_continue.grid(row=1,sticky=W)

        label_pseudo.grid(row=6,sticky=W,pady=10)

        pLabel_framedir.grid(row=0,sticky=W)
        pseudo_intructions_lbl_1.grid(row=1,sticky=W)
        pseudo_intructions_lbl_2.grid(row=2, sticky=W)
        plabelframe_threshold.grid(row=4,sticky=W)
        pseudo_lbl_btn.grid(row=5,sticky=W)

        label_adv_label.grid(row=7, sticky=W)
        label_adv_note_1.grid(row=0, pady=10, sticky=W)
        label_adv_note_2.grid(row=1, pady=10, sticky=W)
        adv_label_btn_new.grid(row=3, sticky=W)
        adv_label_btn_continue.grid(row=4, sticky=W)

        label_thirdpartyann.grid(row=8, sticky=W)
        button_importmars.grid(row=0, sticky=W, pady=5)
        button_importboris.grid(row=1,sticky=W,pady=5)
        button_importsolomon.grid(row=2,sticky=W,pady=5)
        button_importethovision.grid(row=3,sticky=W,pady=5)
        button_importdeepethogram.grid(row=4,sticky=W,pady=5)

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
        bpthresbutton.grid(row=2,padx=5)
        self.auto_compute_font_cb.grid(row=3,padx=5, sticky=NW)
        self.sklearn_text_size_entry_box.grid(row=4,padx=5, sticky=NW)
        self.sklearn_text_spacing_entry_box.grid(row=5, padx=5, sticky=NW)
        self.sklearn_text_thickness_entry_box.grid(row=6, padx=5, sticky=NW)
        self.sklearn_circle_size_entry_box.grid(row=7, padx=5, sticky=NW)
        label_skv_all.grid(row=1,sticky=W,padx=5,pady=10)
        videocheck.grid(row=0,sticky=W)
        framecheck.grid(row=1,sticky=W)
        timerscheck.grid(row=2, sticky=W)
        rotate_check.grid(row=3, sticky=W)
        button_plotsklearnr.grid(row=4,sticky=W)
        label_skv_single.grid(row=2,sticky=W,pady=10,padx=5)
        self.video_entry.grid(row=0,sticky=W)
        videocheck2.grid(row=1,sticky=W)
        framecheck2.grid(row=2,sticky=W)
        timers_single_check.grid(row=3,sticky=W)
        rotate_single_check.grid(row=4, sticky=W)
        button_plotsklearnr2.grid(row=5,sticky=W)

        label_plotall.grid(row=11,column=1,sticky=W+N,padx=5)
        #gantt
        label_ganttplot.grid(row=0,sticky=W)
        button_ganttplot.grid(row=0,sticky=W)
        gannt_frames_check.grid(row=0, column=1, sticky=W)
        gannt_videos_check.grid(row=0, column=2, sticky=W)
        gannt_multiprocess_check.grid(row=0, column=3, sticky=W)

        #data
        label_dataplot.grid(row=1,sticky=W)
        if pose_config_setting == 'user_defined':
            self.SelectedBp.grid(row=1, sticky=W)
        button_dataplot.grid(row=2, sticky=W)

        #path
        self.path_plot_frm.grid(row=2,sticky=W)
        self.deque_points_entry.grid(row=0,sticky=W)
        self.path_animal_cnt_dropdown.grid(row=1,sticky=W)
        self.confirm_animal_cnt.grid(row=1,column=1,sticky=W)
        path_plot_btn.grid(row=9,column=0, sticky=W)
        self.path_plot_frames_check.grid(row=9,column=1, sticky=W)
        self.path_plot_videos_check.grid(row=9, column=2, sticky=W)

        #distance
        label_distanceplot.grid(row=3,sticky=W)
        self.poi1.grid(row=1,sticky=W)
        self.poi2.grid(row=2,sticky=W)
        button_distanceplot.grid(row=3, column=0, sticky=W)
        self.distance_plot_frames_check.grid(row=3, column=1, sticky=W)
        self.distance_plot_videos_check.grid(row=3, column=2, sticky=W)

        #heat
        label_heatmap.grid(row=4, sticky=W)
        self.BinSize.grid(row=0, sticky=W)
        self.MaxScale.grid(row=1, sticky=W)
        self.hmMenu.grid(row=3,sticky=W)
        self.targetMenu.grid(row=4,sticky=W)
        self.bp1.grid(row=5,sticky=W)
        button_heatmap.grid(row=6, column=0, sticky=W)
        heatmap_videos_check.grid(row=6, column=1, sticky=W)
        heatmap_frames_check.grid(row=6, column=2, sticky=W)
        heatmap_last_img_check.grid(row=6, column=3, sticky=W)


        #threshold
        label_plotThreshold.grid(row=5, sticky=W)
        self.behaviorMenu.grid(row=0, sticky=W)
        self.behaviorMenu.grid(row=1, sticky=W)
        self.threshold_frames_check.grid(row=2, column=0, sticky=W)
        self.threshold_videos_check.grid(row=2, column=1, sticky=W)
        self.threshold_multiprocess_check.grid(row=2, column=2, sticky=W)
        plotThresholdButton.grid(row=3, sticky=W)

        merge_frm.grid(row=11,column=2,sticky=W+N,padx=5)
        merge_frm_btn.grid(row=0, sticky=NW, padx=5)

        plotlyInterface.grid(row=11, column=3, sticky=W + N, padx=5)
        button_save_plotly_file.grid(row=10, sticky=W)
        self.plotly_file.grid(row=11, sticky=W)
        self.groups_file.grid(row=12, sticky=W)
        button_open_plotly_interface.grid(row=13, sticky=W)

        label_classifier_validation.grid(row=14,sticky=W)
        self.seconds.grid(row=0,sticky=W)
        self.cvTarget.grid(row=1,sticky=W)
        individual_bout_clips_cb.grid(row=2, column=0, sticky=NW)
        individual_clip_per_video_cb.grid(row=3, column=0, sticky=NW)
        button_validate_classifier.grid(row=4,sticky=NW)


        lbl_addon.grid(row=15,sticky=W)
        button_bel.grid(row=0,sticky=W)
        button_unsupervised.grid(row=1,sticky=W)
        cue_light_analyser_btn.grid(row=2, sticky=W)
        anchored_roi_analysis_btn.grid(row=3, sticky=W)

    def enable_text_settings(self):
        if self.use_default_font_settings_val.get():
            self.sklearn_text_size_entry_box.set_state('disable')
            self.sklearn_text_spacing_entry_box.set_state('disable')
            self.sklearn_text_thickness_entry_box.set_state('disable')
            self.sklearn_circle_size_entry_box.set_state('disable')
        else:
            self.sklearn_text_size_entry_box.set_state('normal')
            self.sklearn_text_spacing_entry_box.set_state('normal')
            self.sklearn_text_thickness_entry_box.set_state('normal')
            self.sklearn_circle_size_entry_box.set_state('normal')

    def create_video_info_table(self):
        video_info_tabler = VideoInfoTable(config_path=self.projectconfigini)
        video_info_tabler.create_window()

    def initiate_skip_outlier_correction(self):
        outlier_correction_skipper = OutlierCorrectionSkipper(config_path=self.projectconfigini)
        outlier_correction_skipper.skip_outlier_correction()

    def plotsklearnresultsingle(self,
                                project_config_path: str,
                                rotate: bool,
                                video_setting: bool,
                                frame_setting: bool,
                                video_path: str,
                                timers: bool):

        if not self.use_default_font_settings_val.get():
            print_settings = {'font_size': self.sklearn_text_size_entry_box.entry_get, 'circle_size': self.sklearn_circle_size_entry_box.entry_get, 'space_size': self.sklearn_text_spacing_entry_box.entry_get, 'text_thickness': self.sklearn_text_thickness_entry_box.entry_get}
        else:
            print_settings = False
        simba_plotter = PlotSklearnResults(config_path=project_config_path, video_setting=video_setting, rotate=rotate, frame_setting=frame_setting, video_file_path=video_path, print_timers=timers, text_settings=print_settings)
        simba_plotter.initialize_visualizations()

    def validate_model_first_step(self):
        _ = ValidateModelRunClf(config_path=self.projectconfigini, input_file_path=self.csvfile.file_path, clf_path=self.modelfile.file_path)

    def create_gannt_plots(self):
        if self.gantt_multiprocess_var.get():
            gannt_creator = GanttCreatorMultiprocess(config_path=self.projectconfigini, frame_setting=self.gannt_frames_var.get(), video_setting=self.gannt_videos_var.get())
            gannt_creator.create_gannt()
        else:
            gannt_creator = GanttCreatorSingleProcess(config_path=self.projectconfigini, frame_setting=self.gannt_frames_var.get(), video_setting=self.gannt_videos_var.get())
            gannt_creator.create_gannt()

    def create_threshold_plots(self,):
        if self.threshold_multiprocess_var.get():
            threshold_plot_creator = TresholdPlotCreatorMultiprocess(config_path=self.projectconfigini, clf_name=self.behaviorMenu.getChoices(), frame_setting=self.threshold_frames_var.get(), video_setting=self.threshold_videos_var.get())
            threshold_plot_creator.create_plot()
        else:
            threshold_plot_creator = TresholdPlotCreatorSingleProcess(config_path=self.projectconfigini, clf_name=self.behaviorMenu.getChoices(), frame_setting=self.threshold_frames_var.get(), video_setting=self.threshold_videos_var.get())
            threshold_plot_creator.create_plot()

    def directing_other_animals_analysis(self):
        directing_animals_analyzer = DirectingOtherAnimalsAnalyzer(config_path=self.projectconfigini)
        directing_animals_analyzer.process_directionality()
        directing_animals_analyzer.create_directionality_dfs()
        directing_animals_analyzer.save_directionality_dfs()
        directing_animals_analyzer.summary_statistics()

    def directing_other_animals_visualizer(self):
        directing_animals_visualizer = DirectingOtherAnimalsVisualizer(config_path=self.projectconfigini)
        directing_animals_visualizer.visualize_results()

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

    def train_single_model(self, config_path=None):
        model_trainer = TrainSingleModel(config_path=config_path)
        model_trainer.perform_sampling()
        model_trainer.train_model()
        model_trainer.save_model()

    def train_multiple_models_from_meta(self, config_path=None):
        model_trainer = TrainMultipleModelsFromMeta(config_path=config_path)
        model_trainer.train_models_from_meta()

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

    def importBoris(self):
        ann_folder = askdirectory()
        boris_appender = BorisAppender(config_path=self.projectconfigini, boris_folder=ann_folder)
        boris_appender.create_boris_master_file()
        boris_appender.append_boris()

    def importSolomon(self):
        ann_folder = askdirectory()
        solomon_importer = SolomonImporter(config_path=self.projectconfigini,
                                           solomon_dir=ann_folder)
        solomon_importer.import_solomon()

    def import_ethovision(self):
        ann_folder = askdirectory()
        ImportEthovision(config_path=self.projectconfigini, folder_path=ann_folder)

    def import_deepethogram(self):
        ann_folder = askdirectory()
        deepethogram_importer = DeepEthogramImporter(config_path=self.projectconfigini, deep_ethogram_dir=ann_folder)
        deepethogram_importer.import_deepethogram()

    def importMARS(self):
        ann_folder = askdirectory()
        append_dot_ANNOTT(self.projectconfigini, ann_folder)

    def fileselected(self,val):
        if hasattr(self, 'frame'):
            self.frame.destroy()
        self.frame = Frame(self.label_import_csv)

        config = read_config_file(ini_path=self.projectconfigini)
        animal_cnt = read_config_entry(self.config, 'General settings', 'animal_no', data_type='int')
        if config.has_option('Multi animal IDs', 'id_list'):
            self.animal_ID_list = config.get('Multi animal IDs', 'id_list').split(',')
        else:
            self.animal_ID_list = []
            for a in range(animal_cnt):
                self.animal_ID_list.append('Animal_' + str(a + 1))

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
            button_import_json = Button(label_multijsonimport, text='Import json to project folder', fg='navy', command=lambda: MarsImporter(config_path=self.projectconfigini,
                                                                                                                                              data_path=self.folder_json.folder_path,
                                                                                                                                              interpolation_method=self.interpolation.getChoices(),
                                                                                                                                              smoothing_method=self.smooth_dropdown.getChoices()))

            # singlejson
            label_singlejsonimport = LabelFrame(self.frame, text='Import single json file', pady=5, padx=5)
            self.file_csv = FileSelect(label_singlejsonimport, 'File Select', title='Select a .csv file')
            button_importsinglejson = Button(label_singlejsonimport, text='Import single .json to project folder', fg='navy', command=lambda: MarsImporter(config_path=self.projectconfigini,
                                                                                                                                                 data_path=self.folder_json.folder_path,
                                                                                                                                                 interpolation_method=self.interpolation.getChoices(),
                                                                                                                                                 smoothing_method=self.smooth_dropdown.getChoices()))

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

        elif val in ('SLP (SLEAP)','H5 (multi-animal DLC)', 'TRK (multi-animal APT)', 'CSV (SLEAP)'):
            animalsettings = LabelFrame(self.frame, text='Animal settings', pady=5, padx=5)
            noofanimals = Entry_Box(animalsettings, 'No of animals', '15')
            noofanimals.entry_set(animal_cnt)
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

            elif val == 'CSV (SLEAP)':
                self.h5path = FolderSelect(self.frame, 'Path to CSV files', lblwidth=15)
                labelinstruction = Label(self.frame, text='Please import videos before importing the sleap csv tracking data')
                runsettings = Button(self.frame, text='Import .csv', command=self.importh5)

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

        # try:
        for i in self.animalnamelist:
            idlist.append(i.entry_get)

        self.config.set('Multi animal IDs', 'ID_list', ",".join(idlist))
        with open(self.projectconfigini, 'w') as configfile:
            self.config.write(configfile)

        if self.val =='H5 (multi-animal DLC)':
            #importMultiDLCpose(self.projectconfigini, self.h5path.folder_path, self.dropdowndlc.getChoices(), idlist, self.interpolation.getChoices(), smooth_settings_dict)
            dlc_multi_animal_importer = MADLC_Importer(config_path=self.projectconfigini,
                                                           data_folder=self.h5path.folder_path,
                                                           file_type=self.dropdowndlc.getChoices(),
                                                           id_lst=idlist,
                                                           interpolation_settings=self.interpolation.getChoices(),
                                                           smoothing_settings=smooth_settings_dict)
            dlc_multi_animal_importer.import_data()

        if self.val == 'SLP (SLEAP)':
            sleap_importer = ImportSLEAP(self.projectconfigini, self.h5path.folder_path, idlist,
                                             self.interpolation.getChoices(), smooth_settings_dict)
            sleap_importer.initate_import_slp()
            if sleap_importer.animals_no > 1:
                sleap_importer.visualize_sleap()
            sleap_importer.save_df()
            sleap_importer.perform_interpolation()
            sleap_importer.perform_smothing()
            print('All SLEAP imports complete.')
        if self.val == 'TRK (multi-animal APT)':
            import_trk(self.projectconfigini,self.h5path.folder_path, idlist, self.interpolation.getChoices(), smooth_settings_dict)

        if self.val == 'CSV (SLEAP)':
            sleap_csv_importer = SleapCsvImporter(config_path=self.projectconfigini,
                                                  data_folder=self.h5path.folder_path,
                                                  actor_IDs=idlist,
                                                  interpolation_settings=self.interpolation.getChoices(),
                                                  smoothing_settings=smooth_settings_dict)
            sleap_csv_importer.initate_import_slp()
            print('SIMBA COMPLETE: Sleap CSV files imported to project_folder/csv/input_csv directory.')


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


    def add_classifier(self, new_clf_name):
        config = read_config_file(ini_path=self.projectconfigini)
        target_cnt = read_config_entry(config, 'SML settings', 'no_targets', 'int')
        config.set('SML settings', 'no_targets', str(target_cnt+1))
        config.set('SML settings', 'model_path_{}'.format(str(target_cnt+1)), '')
        config.set('SML settings', 'target_name_{}'.format(str(target_cnt + 1)), str(new_clf_name))
        config.set('threshold_settings', 'threshold_{}'.format(str(target_cnt + 1)), 'None')
        config.set('Minimum_bout_lengths', 'min_bout_{}'.format(str(target_cnt + 1)), 'None')

        with open(self.projectconfigini, 'w') as f:
            config.write(f)

        print('SIMBA COMPLETE: {} classifier added to SimBA project'.format(str(new_clf_name)))


    def choose_animal_bps(self):
        if hasattr(self, 'path_plot_animal_bp_frm'):
            self.path_plot_animal_bp_frm.destroy()
        self.path_plot_animal_bp_frm = LabelFrame(self.path_plot_frm, text='CHOOSE ANIMAL BODY-PARTS',font=('Helvetica',10,'bold'), pady=5, padx=5)
        self.path_plot_bp_dict = {}
        for animal_cnt in range(int(self.path_animal_cnt_dropdown.getChoices())):
            self.path_plot_bp_dict[animal_cnt] = DropDownMenu(self.path_plot_animal_bp_frm, 'Animal {} bodypart'.format(str(animal_cnt + 1)), self.bp_set, '15')
            self.path_plot_bp_dict[animal_cnt].setChoices(self.bp_set[animal_cnt])
            self.path_plot_bp_dict[animal_cnt].grid(row=animal_cnt, sticky=NW)
        self.path_plot_animal_bp_frm.grid(row=2, column=0, sticky=NW)

    def updateThreshold(self):
        interactive_grapher = InteractiveProbabilityGrapher(config_path=self.projectconfigini,
                                                            file_path=self.csvfile.file_path,
                                                            model_path=self.modelfile.file_path)
        interactive_grapher.create_plots()

    def classifiervalidation(self):
        print('Generating videos...')
        clf_validator = ClassifierValidationClips(config_path=self.projectconfigini,
                                                  window=self.seconds.entry_get,
                                                  clf_name=self.cvTarget.getChoices(),
                                                  clips=self.one_vid_per_bout_var.get(),
                                                  concat_video=self.one_vid_per_video_var.get())
        clf_validator.create_clips()
        print('Videos generated')

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
        data_plotter = DataPlotter(config_path=self.projectconfigini)
        data_plotter.process_movement()
        data_plotter_multiprocessor = multiprocessing.Process(data_plotter.create_data_plots())
        data_plotter_multiprocessor.start()

    def plotsklearn_result(self):
        if not self.use_default_font_settings_val.get():
            print_settings = {'font_size': self.sklearn_text_size_entry_box.entry_get, 'circle_size': self.sklearn_circle_size_entry_box.entry_get, 'space_size': self.sklearn_text_spacing_entry_box.entry_get, 'text_thickness': self.sklearn_text_thickness_entry_box.entry_get}
        else:
            print_settings = False

        simba_plotter = PlotSklearnResults(config_path=self.projectconfigini,
                                           rotate=self.rotate_image_var.get(),
                                           video_setting=self.videovar.get(),
                                           frame_setting=self.genframevar.get(),
                                           print_timers=self.include_timers_var.get(),
                                           text_settings=print_settings)
        simba_plotter.initialize_visualizations()

    def analyzseverity(self):
        analyze_process_severity(self.projectconfigini,self.severityscale.entry_get,self.severityTarget.getChoices())

    def runrfmodel(self):
        rf_model_runner = RunModel(config_path=self.projectconfigini)
        rf_model_runner.run_models()

    def validatemodelsinglevid(self):
        model_validator = ValidateModelOneVideo(ini_path=self.projectconfigini, feature_file_path=self.csvfile.file_path, model_path=self.modelfile.file_path, d_threshold=self.dis_threshold.entry_get, shortest_bout=self.min_behaviorbout.entry_get, create_gantt=self.generategantt_dropdown.getChoices())
        model_validator.perform_clf()
        model_validator.plug_small_bouts()
        model_validator.create_video()

    def trainmachinemodelsetting(self):
        trainmachinemodel_settings(self.projectconfigini)

    def extractfeatures(self):
        configini = self.projectconfigini
        config = ConfigParser()
        config.read(configini)
        pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
        print('Pose-estimation body part setting for feature extraction: ' + str(pose_estimation_body_parts))
        userFeatureScriptStatus = self.usVar.get()

        if userFeatureScriptStatus == 1:
            pose_estimation_body_parts == 'user_defined_script'
            import sys
            script = self.scriptfile.file_path
            dir = os.path.dirname(script)
            fscript = os.path.basename(script).split('.')[0]
            sys.path.insert(0,dir)
            import importlib
            mymodule = importlib.import_module(fscript)
            mymodule.extract_features_userdef(self.projectconfigini)

        if userFeatureScriptStatus == 0:
            if pose_estimation_body_parts == '16':
                feature_extractor = ExtractFeaturesFrom16bps(config_path=self.projectconfigini)
                feature_extractor.extract_features()
                #extract_features_wotarget_16(self.projectconfigini)
            if (pose_estimation_body_parts == '14'):
                feature_extractor = ExtractFeaturesFrom14bps(config_path=self.projectconfigini)
                feature_extractor.extract_features()
                #extract_features_wotarget_14(self.projectconfigini)
            if (pose_estimation_body_parts == '987'):
                extract_features_wotarget_14_from_16(self.projectconfigini)
            if pose_estimation_body_parts == '9':
                extract_features_wotarget_9(self.projectconfigini)
            if pose_estimation_body_parts == '8':
                feature_extractor = ExtractFeaturesFrom8bps(config_path=self.projectconfigini)
                feature_extractor.extract_features()
                #extract_features_wotarget_8(self.projectconfigini)
            if pose_estimation_body_parts == '7':
                feature_extractor = ExtractFeaturesFrom7bps(config_path=self.projectconfigini)
                feature_extractor.extract_features()
                #extract_features_wotarget_7(self.projectconfigini)
            if pose_estimation_body_parts == '4':
                feature_extractor = ExtractFeaturesFrom4bps(config_path=self.projectconfigini)
                feature_extractor.extract_features()
                #extract_features_wotarget_4(self.projectconfigini)
            if pose_estimation_body_parts == 'user_defined':
                feature_extractor = UserDefinedFeatureExtractor(config_path=self.projectconfigini)
                feature_extractor.extract_features()
                #extract_features_wotarget_user_defined(self.projectconfigini)


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
                elif (('.csv') and ('DLC_') in filebasename):
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
        check_int(name='DISTANCE IN MILLIMETER', value=distancemm)
        config = read_config_file(ini_path=self.projectconfigini)
        config.set('Frame settings', 'distance_mm', distancemm)
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

    def extract_frames_loadini(self):
        configini = self.projectconfigini
        videopath = os.path.join(os.path.dirname(configini), 'videos')
        extract_frames_from_all_videos_in_directory(videopath, configini)

    def correct_outlier(self):
        outlier_correcter_movement = OutlierCorrecterMovement(config_path=self.projectconfigini)
        outlier_correcter_movement.correct_movement_outliers()
        outlier_correcter_location = OutlierCorrecterLocation(config_path=self.projectconfigini)
        outlier_correcter_location.correct_location_outliers()
        print('SIMBA COMPLETE: Outlier correction complete. Outlier corrected files located in "project_folder/csv/outlier_corrected_movement_location" directory')

    def distanceplotcommand(self):
        config = read_config_file(ini_path=self.projectconfigini)
        config.set('Distance plot', 'POI_1', self.poi1.getChoices())
        config.set('Distance plot', 'POI_2', self.poi2.getChoices())
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

        distance_plot_creator = DistancePlotter(config_path=self.projectconfigini,
                                                frame_setting=self.distance_plot_frames_var.get(),
                                                video_setting=self.distance_plot_videos_var.get())
        distance_plot_creator.create_distance_plot()

    def create_path_plots(self):

        config = read_config_file(ini_path=self.projectconfigini)
        config.set('Path plot settings', 'no_animal_pathplot', self.path_animal_cnt_dropdown.getChoices())
        config.set('Path plot settings', 'deque_points', self.deque_points_entry.entry_get)
        for animal_cnt, animal_val in self.path_plot_bp_dict.items():
            config.set('Path plot settings', 'animal_{}_bp'.format(str(animal_cnt+1)), animal_val.getChoices())
        with open(self.projectconfigini, 'w') as f:
            config.write(f)

        path_plotter = PathPlotter(config_path=self.projectconfigini,
                                   frame_setting=self.path_plot_frames_var.get(),
                                   video_setting=self.path_plot_videos_var.get())
        path_plotter.create_path_plots()

    def heatmapcommand_clf(self):
        config = read_config_file(ini_path=self.projectconfigini)
        config.set('Heatmap settings', 'bin_size_pixels', self.BinSize.entry_get)
        config.set('Heatmap settings', 'Scale_max_seconds', self.MaxScale.entry_get)
        config.set('Heatmap settings', 'Palette', self.hmMenu.getChoices())
        config.set('Heatmap settings', 'Target', self.targetMenu.getChoices())
        config.set('Heatmap settings', 'body_part', self.bp1.getChoices())
        with open(self.projectconfigini, 'w') as configfile:
            config.write(configfile)

        if self.MaxScale.entry_get != 'auto':
            check_int(name='Max scale (s) (int or auto)', value=self.MaxScale.entry_get, min_value=1)
        check_int(name='Bin size (mm)', value=self.BinSize.entry_get)
        heat_mapper = HeatMapperClf(config_path=self.projectconfigini,
                      video_setting=self.heatmap_clf_videos_var.get(),
                      frame_setting=self.heatmap_clf_frames_var.get(),
                      final_img_setting=self.heatmap_clf_last_img_var.get(),
                      bin_size=self.BinSize.entry_get,
                      palette=self.hmMenu.getChoices(),
                      bodypart=self.bp1.getChoices(),
                      clf_name=self.targetMenu.getChoices(),
                      max_scale=self.MaxScale.entry_get)

        heat_mapper_multiprocess = multiprocessing.Process(target=heat_mapper.create_heatmaps())
        heat_mapper_multiprocess.start()

    def callback(self,url):
        webbrowser.open_new(url)

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
        checkbutton6 = Checkbutton(self.label_settings_box, text='Generate Features Importance Bar Graph', variable=self.box6, command = lambda:activate(self.box6, self.label_n_feature_importance_bars))
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

def get_frame(self):
    '''
      Return the "frame" useful to place inner controls.
    '''
    return self.canvas

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

        #img = PhotoImage(file=os.path.join(scriptdir,'golden.png'))
        img = PhotoImage(file=os.path.join(scriptdir, 'bg.png'))
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
        fileMenu.add_command(label='Load project', command=lambda: LoadProjectPopUp())
        fileMenu.add_separator()
        fileMenu.add_command(label='Exit', command=self.root.destroy)

        # Process video
        pvMenu = Menu(menu)
        menu.add_cascade(label='Process Videos', menu=pvMenu)
        pvMenu.add_command(label='Batch pre-process videos', command=BatchPreProcessWindow)

        #third menu
        thirdMenu = Menu(menu)
        menu.add_cascade(label='Tracking',menu=thirdMenu)

        #labelling tool
        labellingtoolmenu = Menu(thirdMenu)
        labellingtoolmenu.add_command(label='labelImg', command=lambda: subprocess.call(["labelImg"]))
        labellingtoolmenu.add_command(label='labelme', command=lambda: subprocess.call(["labelme"]))
        #third menu organize
        thirdMenu.add_cascade(label='Labelling tools', menu=labellingtoolmenu)

        #fifth menu
        fifthMenu = Menu(menu)
        fpsMenu = Menu(fifthMenu)
        fpsMenu.add_command(label='Change fps for single video', command=ChangeFpsSingleVideoPopUp)
        fpsMenu.add_command(label='Change fps for multiple videos',command=ChangeFpsMultipleVideosPopUp)
        menu.add_cascade(label='Tools',menu=fifthMenu)
        fifthMenu.add_command(label='Clip videos',command=ClipVideoPopUp)
        fifthMenu.add_command(label='Clip video into multiple videos', command=MultiShortenPopUp)
        fifthMenu.add_command(label='Crop videos',command=CropVideoPopUp)
        fifthMenu.add_command(label='Multi-crop',command=MultiCropPopUp)
        fifthMenu.add_command(label='Downsample videos',command=DownsampleVideoPopUp)
        fifthMenu.add_command(label='Get mm/ppx',command = CalculatePixelsPerMMInVideoPopUp)
        fifthMenu.add_command(label='Create path plot', command=MakePathPlotPopUp)
        fifthMenu.add_cascade(label='Change fps...',menu =fpsMenu)
        fifthMenu.add_cascade(label='Concatenate two videos', command=ConcatenatingVideosPopUp)
        fifthMenu.add_cascade(label='Concatenate multiple videos', command=lambda:ConcatenatorPopUp(config_path=None))
        fifthMenu.add_cascade(label='Visualize pose-estimation in folder...', command=VisualizePoseInFolderPopUp)
        fifthMenu.add_cascade(label='Reorganize Tracking Data', command= PoseReorganizerPopUp)
        fifthMenu.add_cascade(label='Drop body-parts from tracking data', command=DropTrackingDataPopUp)

        #changefpsmenu organize

        changeformatMenu = Menu(fifthMenu)
        changeformatMenu.add_command(label='Change image file formats',command=ChangeImageFormatPopUp)
        changeformatMenu.add_command(label='Change video file formats',command=ConertVideoPopUp)
        changeformatMenu.add_command(label='Change .seq to .mp4', command=lambda:convertseqVideo(askdirectory(title='Please select video folder to convert')))
        fifthMenu.add_cascade(label='Change formats...',menu=changeformatMenu)

        fifthMenu.add_command(label='CLAHE enhance video',command=CLAHEPopUp)
        fifthMenu.add_command(label='Superimpose frame numbers on video',command=lambda:superimpose_frame_count(file_path=askopenfilename()))
        fifthMenu.add_command(label='Convert to grayscale',command=lambda:video_to_greyscale(file_path=askopenfilename()))
        fifthMenu.add_command(label='Merge frames to video',command=MergeFrames2VideoPopUp)
        fifthMenu.add_command(label='Generate gifs', command=CreateGIFPopUP)
        fifthMenu.add_command(label='Print classifier info...', command=PrintModelInfoPopUp)

        extractframesMenu = Menu(fifthMenu)
        extractframesMenu.add_command(label='Extract defined frames',command=ExtractSpecificFramesPopUp)
        extractframesMenu.add_command(label='Extract frames',command=ExtractAllFramesPopUp)
        extractframesMenu.add_command(label='Extract frames from seq files', command=ExtractSEQFramesPopUp)
        fifthMenu.add_cascade(label='Extract frames...',menu=extractframesMenu)

        convertWftypeMenu = Menu(fifthMenu)
        convertWftypeMenu.add_command(label='Convert CSV to parquet', command=Csv2ParquetPopUp)
        convertWftypeMenu.add_command(label='Convert parquet o CSV', command=Parquet2CsvPopUp)
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
        sixthMenu.add_command(label='About', command= AboutSimBAPopUp)

        #Status bar at the bottom
        self.frame = Frame(background, bd=2, relief=SUNKEN, width=300, height=300)
        self.frame.pack(expand=True)
        self.txt = Text(self.frame, bg='white')
        self.txt.config(state=DISABLED, font=("Rockwell", 11))
        self.txt.pack(expand=True, fill='both')
        #self.txt.configure(font=("Comic Sans MS", 20, "bold"))
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
            try:
                self.image = PIL.Image.open(os.path.join(scriptdir, "splash_050122.PNG"))
            except FileNotFoundError:
                self.image = PIL.Image.open(os.path.join(scriptdir, "splash_050122.png"))
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
    # if platform.system() == "Darwin":
    #     multiprocessing.set_start_method('spawn', force=True)


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
