__author__ = "Simon Nilsson", "JJ Choong"

import matplotlib.pyplot as plt
from simba.read_config_unit_tests import (read_config_file,
                                          check_file_exist_and_readable,
                                          read_project_path_and_file_type)
from simba.misc_tools import get_fn_ext, find_video_of_file
from simba.labelling_aggression import choose_folder2, load_frame2
import cv2
from simba.rw_dfs import read_df
import os
import threading
from simba.utils.errors import NoFilesFoundError
from copy import copy


class InteractiveProbabilityGrapher(object):
    """
    Class for launching and creating interactive GUI for classifier probability inspection.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    clf_name: str
        Name of the classifier to create visualizations for
    frame_setting: bool
       When True, SimBA creates individual frames in png format
    video_setting: bool
       When True, SimBA creates compressed video in mp4 format

    Notes
    ----------
    `Validation tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data>`__.


    Examples
    ----------
    >>> interactive_plotter = InteractiveProbabilityGrapher(config_path=r'MyConfigPath', file_path='MyFeatureFilepath', model_path='MyPickledClassifier.sav')
    >>> interactive_plotter.create_plots()
    """

    def __init__(self,
                 config_path: str=None,
                 file_path: str=None,
                 model_path: str=None):

        if file_path == 'No file selected':
            raise NoFilesFoundError(msg='No feature file path selected. Please select a path to a valid SimBA file containing machine learning features. They are by default located in the `project_folder/csv/features_extracted` directory')
        elif model_path == 'No file selected':
            raise NoFilesFoundError(msg='No model file path selected. Please select a path to a valid machine learning model file.')
        self.click_counter = 0
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.in_dir = os.path.join(self.project_path, 'csv')
        self.videos_path = os.path.join(self.project_path, 'videos')
        self.file_path, self.model_path = file_path, model_path
        self.clf_name = str(os.path.basename(self.model_path)).split(".")[0]
        self.data_path = os.path.join(self.in_dir, "validation", os.path.basename(self.file_path))
        check_file_exist_and_readable(self.data_path)
        self.data_df = read_df(self.data_path, self.file_type)
        self.p_arr = self.data_df[['Probability_{}'.format(self.clf_name)]].to_numpy()
        dir_name, file_name, ext = get_fn_ext(self.data_path)
        current_video_file_path = find_video_of_file(video_dir=self.videos_path, filename=file_name)
        self.cap = cv2.VideoCapture(current_video_file_path)
        self.master_window = choose_folder2(current_video_file_path, config_path)



    @staticmethod
    def __click_event(event):
        global current_x_cord
        if (event.dblclick) and (event.button == 1) and (type(event.xdata) != None):
            current_x_cord = int(event.xdata)

    def create_plots(self):
        """
        Method to launch interactive GUI

        Returns
        -------
        Attribute: matplotlib.plt
            fig
        """

        import matplotlib
        matplotlib.use('TkAgg')
        probability = "Selected frame: {}, {} probability: {}".format(str(0), self.clf_name, str(self.p_arr[0][0]))
        plt_title = 'Click on the points of the graph to display the corresponding video frame. \n {}'.format(str(probability))
        global current_x_cord
        current_x_cord, prior_x_cord = None, None
        fig, ax = plt.subplots()
        ax.plot(self.p_arr)
        plt.xlabel('frame #', fontsize=16)
        plt.ylabel(str(self.clf_name) + ' probability', fontsize=16)
        plt.title(plt_title)
        plt.grid()
        line = None
        fig.canvas.draw()
        fig.canvas.flush_events()

        while True:
            _ = fig.canvas.mpl_connect('button_press_event', lambda event: self.__click_event(event))
            if current_x_cord != prior_x_cord:
                prior_x_cord = copy(current_x_cord)
                probability = "Selected frame: {}, {} probability: {}".format(str(current_x_cord), self.clf_name, str(self.p_arr[current_x_cord][0]))
                plt_title = 'Click on the points of the graph to display the corresponding video frame. \n {}'.format(str(probability))
                load_frame2(current_x_cord, self.master_window.guimaster(), self.master_window.Rfbox())
                if line != None:
                    line.remove()
                plt.title(plt_title)
                line = plt.axvline(x=current_x_cord, color='r')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.ion()
            threading.Thread(plt.show()).start()
            plt.pause(.0001)

# test = InteractiveProbabilityGrapher(config_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', file_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/csv/features_extracted/Together_1.csv', model_path='/Users/simon/Desktop/troubleshooting/train_model_project/models/generated_models/Attack.sav')
# test.create_plots()