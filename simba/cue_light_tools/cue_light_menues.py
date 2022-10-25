from tkinter import *
import glob, os
import itertools
import platform
from simba.read_config_unit_tests import read_config_entry, read_config_file, check_float, check_int
from simba.drop_bp_cords import getBpNames
from simba.misc_tools import get_fn_ext, find_video_of_file
import pandas as pd
from simba.cue_light_tools.cue_light_analyzer import CueLightAnalyzer
from simba.cue_light_tools.cue_light_visualizer import CueLightVisualizer
from simba.train_model_functions import get_all_clf_names
from simba.cue_light_tools.cue_light_clf_statistics import CueLightClfAnalyzer
from simba.cue_light_tools.cue_light_movement_statistics import CueLightMovementAnalyzer
import webbrowser

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
    canvas.configure(scrollregion=canvas.bbox("all"))

def hxtScrollbar(master):
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
    acanvas.bind("<Configure>", lambda event, canvas=acanvas: onFrameConfigure(acanvas))
    acanvas.bind('<Enter>', lambda event: bindToMousewheel(event, acanvas))
    acanvas.bind('<Leave>', lambda event: unbindToMousewheel(event,acanvas))
    return frame

class CueLightAnalyzerMenu(object):
    """
    Class for lunching cue light analysis GUI in SimBA.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.


    Examples
    ----------
    >>> cue_light_gui = CueLightAnalyzerMenu(config_path='MySimBAConfigPath')
    >>> cue_light_gui.cue_light_main_frame.mainloop()
    """



    def __init__(self,
                 config_path: str):

        self.config_path = config_path
        self.config = read_config_file(self.config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.logs_path, self.video_dir = os.path.join(self.project_path, 'logs'), os.path.join(self.project_path, 'videos')
        self.data_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.cue_light_data_folder = os.path.join(self.project_path, 'csv', 'cue_lights')
        self.animal_cnt = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.__read_roi_dfs()
        if len(self.shape_names) == 0:
            print('SIMBA ERROR: Cue light analysis require ROI definitions. Please define ROIs before doing cue light analysis')
            raise ValueError('SIMBA ERROR: Cue light analysis require ROI definitions. Please define ROIs before doing cue light analysis')
        self.cue_light_main_frame = Toplevel()
        self.cue_light_main_frame.minsize(750, 300)
        self.cue_light_main_frame.wm_title("SIMBA CUE LIGHT ANALYZER")
        self.cue_light_main_frame.lift()
        self.lights_dict = {}
        self.cue_light_settings_frm = LabelFrame(self.cue_light_main_frame, text='DEFINE CUE LIGHTS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.choose_lights_cnt_lbl = Label(self.cue_light_settings_frm, text='# Cue lights', width=10, anchor=W)
        self.choose_lights_cnt_var = IntVar()
        self.choose_lights_cnt_var.set(1)
        self.choose_lights_cnt_dropdown = OptionMenu(self.cue_light_settings_frm, self.choose_lights_cnt_var, *list(range(1, len(self.shape_names) + 1)), command=self.__create_cue_light_menus)
        self.cue_light_settings_frm.grid(row=0, sticky=NW)
        self.choose_lights_cnt_lbl.grid(row=0, column=0, sticky=W)
        self.choose_lights_cnt_dropdown.grid(row=0, column=1, sticky=W)
        self.analyze_data_frm = LabelFrame(self.cue_light_main_frame, text='ANALYZE', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.analyze_cue_light_data_btn = Button(self.analyze_data_frm, text='Analyze cue light data', command=lambda: self.__analyze_cue_light_data())
        self.visualize_cue_light_data_btn = Button(self.analyze_data_frm, text='Visualize cue light data', command=lambda: self.__visualize_cue_light_data())
        self.video_var = BooleanVar()
        self.frames_var = BooleanVar()
        self.video_check = Checkbutton(self.analyze_data_frm, text='Create videos', variable=self.video_var)
        self.frames_check = Checkbutton(self.analyze_data_frm, text='Create frames', variable=self.frames_var)
        self.analyze_movements_btn = Button(self.analyze_data_frm, text='Analyze cue light movement', command=lambda: self.__inititate_animal_movement_menu())
        self.analyze_clf_btn = Button(self.analyze_data_frm, text='Analyze cue light classifications', command=lambda: self.__inititate_clf_menu())
        self.analyze_data_frm.grid(row=0, column=1, sticky=NW)
        self.analyze_cue_light_data_btn.grid(row=0, column=0, sticky=W)
        self.visualize_cue_light_data_btn.grid(row=1, column=0, sticky=W)
        self.video_check.grid(row=1, column=1, sticky=W)
        self.frames_check.grid(row=1, column=2, sticky=W)
        self.analyze_movements_btn.grid(row=2, column=0, sticky=W)
        self.analyze_clf_btn.grid(row=3, column=0, sticky=W)
        lbl_info_1 = Label(self.cue_light_main_frame, text='NOTE: Cue light analysis require ROI definitions')
        lbl_info_2 = Label(self.cue_light_main_frame, text='[Click here to learn about cue light analysis]', cursor='hand2', fg='blue')
        lbl_info_2.bind('<Button-1>', lambda e: webbrowser.open_new('https://github.com/sgoldenlab/simba/blob/master/docs/FSTTC.md'))
        lbl_info_1.grid(row=5, column=0, sticky=W)
        lbl_info_2.grid(row=6, column=0, sticky=W)

        mainloop()

    def __get_cue_light_names(self):
        self.light_lst = []
        for light_name, light_data in self.lights_dict.items():
            self.light_lst.append(light_data['light_chosen'].get())


    def __create_cue_light_menus(self, no_cue_lights):
        if len(self.lights_dict.keys()) > 0:
            for light_cnt in range(len(self.lights_dict.keys())):
                self.lights_dict[light_cnt]['label'].grid_forget()
                self.lights_dict[light_cnt]['dropdown'].grid_forget()
        for light_cnt in range(no_cue_lights):
            self.lights_dict[light_cnt] = {}
            current_row = 1 + light_cnt
            self.lights_dict[light_cnt]['label'] = Label(self.cue_light_settings_frm, text='Cue light {}'.format(str(light_cnt+1)), width=10, anchor=W)
            self.lights_dict[light_cnt]['light_chosen'] = StringVar()
            self.lights_dict[light_cnt]['light_chosen'].set(self.shape_names[light_cnt])
            self.lights_dict[light_cnt]['dropdown'] = OptionMenu(self.cue_light_settings_frm, self.lights_dict[light_cnt]['light_chosen'], *self.shape_names, command=None)
            self.lights_dict[light_cnt]['label'].grid(row=current_row, column=0, sticky=W)
            self.lights_dict[light_cnt]['dropdown'].grid(row=current_row, column=1, sticky=W)

    def __read_roi_dfs(self):
        if not os.path.isfile(os.path.join(self.logs_path, 'measures', 'ROI_definitions.h5')):
            print('No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data')
            raise FileNotFoundError('No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data')
        else:
            self.roi_h5_path = os.path.join(self.logs_path, 'measures', 'ROI_definitions.h5')
            self.rectangles_df = pd.read_hdf(self.roi_h5_path, key='rectangles')
            self.circles_df = pd.read_hdf(self.roi_h5_path, key='circleDf')
            self.polygon_df = pd.read_hdf(self.roi_h5_path, key='polygons')
            self.shape_names = list(itertools.chain(self.rectangles_df['Name'].unique(), self.circles_df['Name'].unique(), self.polygon_df['Name'].unique()))

    def __analyze_cue_light_data(self):
        self.__get_cue_light_names()
        cue_light_analyzer = CueLightAnalyzer(config_path=self.config_path, in_dir=self.data_dir, cue_light_names=self.light_lst)
        cue_light_analyzer.analyze_files()

    def __visualize_cue_light_data(self):
        self.cue_light_data_files = glob.glob(self.cue_light_data_folder + '/*' + self.file_type)
        if len(self.cue_light_data_files) == 0:
            print('SIMBA ERROR: Zero data files found. Please analyze cue light data prior to visualizing cue light data')
            raise ValueError('SIMBA ERROR: Zero data files found. Please analyze cue light data prior to visualizing cue light data')
        else:
            self.__get_cue_light_names()
            for data_path in self.cue_light_data_files:
                _, file_name, _ = get_fn_ext(data_path)
                video_path = find_video_of_file(self.video_dir, file_name)
                cue_light_visualizer = CueLightVisualizer(config_path=self.config_path,
                                                          cue_light_names=self.light_lst,
                                                          video_path=video_path,
                                                          video_setting=self.video_var.get(),
                                                          frame_setting=self.frames_var.get())
                cue_light_visualizer.visualize_cue_light_data()

    def __inititate_animal_movement_menu(self):
        self.movement_main_frame = Toplevel()
        self.movement_main_frame.minsize(400, 400)
        self.movement_main_frame.wm_title("SIMBA CUE LIGHT ANALYZER: MOVEMENTS")
        self.movement_main_frame.lift()
        self.animal_cnt_frm = LabelFrame(self.movement_main_frame, text='SETTINGS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.choose_animal_cnt_lbl = Label(self.animal_cnt_frm, text='# Animals', width=10, anchor=W)
        self.choose_animal_cnt_var = IntVar()
        self.choose_animal_cnt_var.set(1)
        self.choose_animal_cnt_dropdown = OptionMenu(self.animal_cnt_frm, self.choose_animal_cnt_var,
                                                     *list(range(1, self.animal_cnt+1)),
                                                     command=self.__create_animal_bp_menues)
        self.animal_cnt_frm.grid(row=0, column=0, sticky=W)
        self.choose_animal_cnt_lbl.grid(row=0, column=0, sticky=W)
        self.choose_animal_cnt_dropdown.grid(row=0, column=1)

    def __create_animal_bp_menues(self, no_animals):
        self.animal_dict = {}
        self.bp_names = getBpNames(self.config_path)[0]
        self.bp_names = [x[0:-2] for x in self.bp_names]
        current_row = 0
        for animal_cnt in range(no_animals):
            self.animal_dict[animal_cnt] = {}
            current_row = 1 + animal_cnt
            self.animal_dict[animal_cnt]['label'] = Label(self.animal_cnt_frm, text='Animal {} body-part:'.format(str(animal_cnt+1)), width=17, anchor=W)
            self.animal_dict[animal_cnt]['bp_chosen'] = StringVar()
            self.animal_dict[animal_cnt]['bp_chosen'].set(self.bp_names[animal_cnt])
            self.animal_dict[animal_cnt]['dropdown'] = OptionMenu(self.animal_cnt_frm, self.animal_dict[animal_cnt]['bp_chosen'], *self.bp_names, command=None)
            self.animal_dict[animal_cnt]['label'].grid(row=current_row, column=0, sticky=W)
            self.animal_dict[animal_cnt]['dropdown'].grid(row=current_row, column=1, sticky=W)

        self.pre_window_var = IntVar()
        self.pre_window_var.set(0)
        self.pre_window_lbl = Label(self.animal_cnt_frm, text='Pre-cue window (ms)', width=17, anchor=W)
        self.pre_window_entry = Entry(self.animal_cnt_frm, width=6, textvariable=self.pre_window_var)
        self.post_window_var = IntVar()
        self.post_window_var.set(0)
        self.post_window_lbl = Label(self.animal_cnt_frm, text='Post-cue window (ms)', width=17, anchor=W)
        self.post_window_entry = Entry(self.animal_cnt_frm, width=6, textvariable=self.post_window_var)
        self.pre_window_lbl.grid(row=current_row+1, column=0, sticky=W)
        self.pre_window_entry.grid(row=current_row+1, column=1, sticky=W)
        self.post_window_lbl.grid(row=current_row+2, column=0, sticky=W)
        self.post_window_entry.grid(row=current_row+2, column=1, sticky=W)
        self.threshold_lbl = Label(self.animal_cnt_frm, text='Threshold (0.00 - 1.00)', width=17, anchor=W)
        self.threshold_var = IntVar()
        self.threshold_var.set(0.00)
        self.threshold_entry = Entry(self.animal_cnt_frm, width=6, textvariable=self.threshold_var)
        self.threshold_lbl.grid(row=current_row+3, column=0, sticky=W)
        self.threshold_entry.grid(row=current_row + 3, column=1, sticky=W)
        self.roi_var = BooleanVar()
        self.roi_check = Checkbutton(self.animal_cnt_frm, text='Analyze ROI data', variable=self.roi_var)
        self.roi_check.grid(row=current_row+4, column=0, sticky=W)
        self.analyze_movement_btn = Button(self.animal_cnt_frm, text='Analyze movement data', command=lambda: self.__start_movement_analysis())
        self.analyze_movement_btn.grid(row=current_row+5, column=0, sticky=W)

    def start_movement_analysis(self):
        self.__get_cue_light_names()
        if len(self.light_lst) == 0:
            print('SIMBA ERROR: Zero user-specified cue lights. Please specify cue lights before analyzing movements')
            raise ValueError('SIMBA ERROR: Zero user-specified cue lights. Please specify cue lights before analyzing movements')
        if not self.config.has_section('Cue light analysis'):
            self.config.add_section('Cue light analysis')
        for animal_cnt, animal_data in self.animal_dict.items():
            self.config['Cue light analysis']['animal_{}_bp'.format(str(animal_cnt+1))] = self.animal_dict[animal_cnt]['bp_chosen'].get()
        with open(self.config_path, 'w') as file:
            self.config.write(file)

        check_int('Pre-cue window', value=self.pre_window_var.get())
        check_int('Post-cue window', value=self.post_window_var.get())
        check_float('Threshold', value=self.threshold_entry.get())

        cue_light_movement_analyzer = CueLightMovementAnalyzer(config_path=self.config_path,
                                                               pre_window=self.pre_window_var.get(),
                                                               post_window=self.post_window_var.get(),
                                                               cue_light_names=self.light_lst,
                                                               threshold=float(self.threshold_entry.get()),
                                                               roi_setting=self.roi_var.get())
        cue_light_movement_analyzer.calculate_whole_session_movement()
        cue_light_movement_analyzer.organize_results()
        cue_light_movement_analyzer.save_results()

    def __inititate_clf_menu(self):
        self.clf_main_frame = Toplevel()
        self.clf_main_frame.minsize(400, 400)
        self.clf_main_frame.wm_title("SIMBA CUE LIGHT ANALYZER: CLASSIFICATIONS")
        self.clf_main_frame.lift()
        self.clf_cnt = read_config_entry(self.config, 'SML settings', 'no_targets', 'int')
        self.clf_names = get_all_clf_names(self.config, self.clf_cnt)
        self.clf_frm = LabelFrame(self.clf_main_frame, text='SELECT CLASSIFIERS', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.clf_frm.grid(row=0, column=0, sticky=NW)
        self.__insert_clf_tick_boxes()
        self.window_size_frm = LabelFrame(self.clf_main_frame, text='WINDOW SIZE', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.window_size_frm.grid(row=1, column=0, sticky=NW)
        self.pre_window_var_clf = IntVar()
        self.pre_window_var_clf.set(0)
        self.pre_window_lbl = Label(self.window_size_frm, text='Pre-cue window (ms)', width=17, anchor=W)
        self.pre_window_entry = Entry(self.window_size_frm, width=6, textvariable=self.pre_window_var_clf)
        self.post_window_var_clf = IntVar()
        self.post_window_var_clf.set(0)
        self.post_window_lbl = Label(self.window_size_frm, text='Post-cue window (ms)', width=17, anchor=W)
        self.post_window_entry = Entry(self.window_size_frm, width=6, textvariable=self.post_window_var_clf)
        self.pre_window_lbl.grid(row=0, column=0, sticky=NW)
        self.pre_window_entry.grid(row=0, column=1, sticky=NW)
        self.post_window_lbl.grid(row=1, column=0, sticky=NW)
        self.post_window_entry.grid(row=1, column=1, sticky=NW)
        self.run_frm = LabelFrame(self.clf_main_frame, text='RUN', font=('Helvetica', 15, 'bold'), pady=5, padx=15)
        self.window_size_frm.grid(row=1, column=0, sticky=NW)
        self.run_frm.grid(row=2, column=0, sticky=NW)
        self.analyze_clf_btn = Button(self.run_frm, text='Analyze classifier data', command=lambda: self.__start_clf_analysis())
        self.analyze_clf_btn.grid(row=1, column=0, sticky=NW)


    def __insert_clf_tick_boxes(self):
        self.entry_dict = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.entry_dict[clf_name] = {}
            self.entry_dict[clf_name]['var'] = BooleanVar()
            self.entry_dict[clf_name]['cb'] = Checkbutton(self.clf_frm, text=clf_name, variable=self.entry_dict[clf_name]['var'])
            self.entry_dict[clf_name]['cb'].grid(row=clf_cnt, column=0, sticky=N)

    def __start_clf_analysis(self):
        check_int('Pre-cue window', value=self.pre_window_var_clf.get())
        check_int('Post-cue window', value=self.post_window_var_clf.get())
        clf_lst = []
        self.__get_cue_light_names()
        if len(self.light_lst) == 0:
            print('SIMBA ERROR: Zero data files found. Please analyze cue light data prior to visualizing cue light data')
            raise ValueError('SIMBA ERROR: Zero data files found. Please analyze cue light data prior to visualizing cue light data')
        for clf_name, clf_setting in self.entry_dict.items():
            if clf_setting['var'].get(): clf_lst.append(clf_name)
        if len(clf_lst) == 0:
            print('SIMBA ERROR: Please tick at least one 1 classifier option')
            raise ValueError('SIMBA ERROR: Please tick at least one 1 classifier option')
        else:
            clf_cue_light_analyzer = CueLightClfAnalyzer(config_path=self.config_path,
                                                         pre_window=self.pre_window_var_clf.get(),
                                                         post_window=self.post_window_var_clf.get(),
                                                         clf_list=clf_lst,
                                                         cue_light_names=self.light_lst)
            clf_cue_light_analyzer.analyze_clf()
            clf_cue_light_analyzer.organize_results()
            clf_cue_light_analyzer.save_data()

# test = CueLightAnalyzerMenu(config_path=r'/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini')
# test.cue_light_main_frame.mainloop()




