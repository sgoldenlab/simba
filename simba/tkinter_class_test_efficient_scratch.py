import os
import subprocess
#import deeplabcut
import sys
import tkinter
from tkinter import ttk
from tkinter.filedialog import askopenfilename,askdirectory
# from tkinter_functions import greyscale,downsamplevideo,superimposeframe,colorized,shortenvideos,convertavitomp4,convertpowerpoint,extract_allframescommand,mergemovieffmpeg,extractspecificframe,clahe,cropvid,changedlc_config,changeimageformat
# import correct_large_deviations, Target_insert3, Consolidate_allcsv, DLC_extract_91,run_RF_model
# import plot_sklearn_results, generate_gantt_4, path_plot3, opencv_data_readout_2, Plot_distance, merge_opencv_gantt4, merge_movie_ffmpeg
# from get_coordinates_tools import get_coordinates_mice_tools
# from create_project_ini import write_inifile
from labelling_aggression import *
# from tkinter.tix import *


class App(object):
    def __init__(self):
        self.root = Tk()
        self.root.title('The Golden Lab')
        self.root.minsize(300,300)
        self.root.geometry("800x600")


        ### drop down menu###
        menu = Menu(self.root)
        self.root.config(menu=menu)

        # #first menu
        # fileMenu = Menu(menu)
        # menu.add_cascade(label='File', menu=fileMenu)
        # fileMenu.add_command(label='Create a new project', command=project_config)
        # fileMenu.add_command(label='Load Environment', command=askdirectory)
        # fileMenu.add_separator()
        # fileMenu.add_command(label='Exit', command=Exit)

        #label aggression
        agMenu = Menu(menu)
        menu.add_cascade(label='Label Aggression', menu=agMenu)
        agMenu.add_command(label='Label Aggression',command = lambda: chooseFolder())

        # #machinelearning menu
        # cMenu = Menu(menu)
        # menu.add_cascade(label = "Project Commands", menu = cMenu)
        # cMenu.add_command(label = "1. Correct Large Deviations", command = lambda: createWindow(correct_large_deviations))
        # cMenu.add_command(label = "2. Insert Targets", command = lambda: createWindow(Target_insert3))
        # # cMenu.add_command(label = "3. Consolidate", command = lambda: createWindow(Consolidate_allcsv))
        # cMenu.add_command(label = "4. Extract Features", command = lambda: createWindow(DLC_extract_91))
        # # cMenu.add_command(label = "5. Train RF Models", command = lambda: createWindow(sklearn_DLC_RF_train_model))
        # cMenu.add_command(label = "6. Run RF Models", command = lambda: createWindow(Run_RF_model))
        # cMenu.add_command(label = "7. Plot RF Results", command = lambda: createWindow(plot_sklearn_results))
        # cMenu.add_command(label="8. Create Gantt", command=lambda: createWindow(generate_gantt_4))
        # cMenu.add_command(label="9. Plot Paths", command=lambda: createWindow(path_plot3))
        # cMenu.add_command(label="10. Data Plots", command=lambda: createWindow(opencv_data_readout_2))
        # cMenu.add_command(label="11. Line Plots", command=lambda: createWindow(Plot_distance))
        # cMenu.add_command(label="12. Merge Frames", command=lambda: createWindow(merge_opencv_gantt4))
        # cMenu.add_command(label="13. Merge Into Movie", command=lambda: createWindow(merge_movie_ffmpeg))

        # #second menu
        # processMenu = Menu(menu)
        # menu.add_cascade(label='Process Videos',menu=processMenu)
        # processMenu.add_command(label='Red Light Videos', command= Red_light_Convertion)
        # #processMenu.add_command(label='B&W Videos', command= askopenfilename)
        # #processMenu.add_command(label='Color Videos', command= askopenfilename)
        # processMenu.add_command(label='Downsample', command= video_downsample)
        # processMenu.add_command(label='Crop Video', command=crop_video)
        # processMenu.add_command(label='Length Calibration', command= get_coordinates_from_video)
        #
        # #third menu
        # thirdMenu = Menu(menu)
        # menu.add_cascade(label='DeepLabCut',menu=thirdMenu)
        # createMenu = Menu(thirdMenu)
        # createMenu.add_command(label='Create Project',command=create_project)
        # createMenu.add_command(label='Load yaml file',command=loadvideos)
        # createMenu.add_command(label='Extract Frames',command=extract_frames)
        # createMenu.add_command(label='Label Frames',command=labelframes)
        # createMenu.add_command(label='Check Frames',command=checklabels)
        # createMenu.add_command(label='Generate Training Set',command=generatetrainingset)
        # #createMenu.add_command(label='Extract Outliers',command=Importpath)
        # #createMenu.add_command(label='Correct Outliers',command=Importpath)
        # #createMenu.add_command(label='Merge Corrected Frames',command=Importpath)
        # #createMenu.add_command(label='Train Network',command=trainnetwork)
        # createMenu.add_command(label='Evaluate Trained Network',command=evaluatenetwork)
        # #createMenu.add_command(label='CHECKPOINT',command=Importpath)
        # createMenu.add_command(label='Analyze Video',command=video_analysis)
        # createMenu.add_command(label='Plot Results',command=plot_videoresults)
        # createMenu.add_command(label='Create Overlay Video',command=create_video)
        # thirdMenu.add_cascade(label='Create Project',menu=createMenu)
        #
        # loadprojectMenu=Menu(thirdMenu)
        # loadprojectMenu.add_command(label='Load Training Set (pre-trained network)',command=askopenfilename)
        # loadprojectMenu.add_command(label='Load Training Set with trained network',command=askopenfilename)
        # thirdMenu.add_cascade(label='Load Project',menu=loadprojectMenu)
        #
        #
        # #fourth menu
        # fourthMenu = Menu(menu)
        # menu.add_cascade(label='Analyze Behavior',menu=fourthMenu)
        # fourthMenu.add_command(label='Set machine learning algorithm',command=askopenfilename)
        # fourthMenu.add_command(label='Create machine learning algorithm',command=askopenfilename)
        # fourthMenu.add_command(label='Extend machine learning algorithm',command=askopenfilename)
        # fourthMenu.add_command(label='Run machine learning algorithm',command=askopenfilename)
        # fourthMenu.add_command(label='Train Network',command=askopenfilename)
        #
        # extractmismatchMenu = Menu(fourthMenu)
        # extractmismatchMenu.add_command(label='Human --> Computer',command=askopenfilename)
        # extractmismatchMenu.add_command(label='Computer --> Human',command=askopenfilename)
        # fourthMenu.add_cascade(label='Extract mismatches',menu=extractmismatchMenu)
        #
        # #fifth menu
        # fifthMenu = Menu(menu)
        # menu.add_cascade(label='Tools',menu=fifthMenu)
        # fifthMenu.add_command(label='Shortens Videos',command=shorten_video)
        # fifthMenu.add_command(label='Crop Videos',command=crop_video)
        # fifthMenu.add_command(label='Downsample',command=video_downsample)
        # fifthMenu.add_command(label='Get Coordinates',command = get_coordinates_from_video)
        #
        # changeformatMenu = Menu(fifthMenu)
        # changeformatMenu.add_command(label='Change image formats',command=change_imageformat)
        # changeformatMenu.add_command(label='Change video formats',command=convert_video)
        # fifthMenu.add_cascade(label='Change formats',menu=changeformatMenu)
        #
        # fifthMenu.add_command(label='Contrast(CLAHE)',command=Red_light_Convertion)
        # fifthMenu.add_command(label='Add Color',command=colorized)
        # fifthMenu.add_command(label='Add Frame Numbers',command=lambda :superimposeframe(askopenfilename()))
        # fifthMenu.add_command(label='Convert to grayscale',command=lambda:greyscale(askopenfilename()))
        # fifthMenu.add_command(label='Merge Frames to video',command=mergeframeffmpeg)
        #
        # extractframesMenu = Menu(fifthMenu)
        # extractframesMenu.add_command(label='Extract Defined Frames',command=extract_specificframes)
        # extractframesMenu.add_command(label='Extract Frames',command=extract_allframes)
        # fifthMenu.add_cascade(label='Extract Frames',menu=extractframesMenu)


        #Status bar at the bottom
        self.frame = Frame(self.root, bd=2, relief=SUNKEN, width=300, height=300)
        self.frame.grid(row=0, column=0)
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

all_entries = []
app = App()

app.root.mainloop()
