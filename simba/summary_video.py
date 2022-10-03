__author__ = "Simon Nilsson", "JJ Choong"

from tkinter.ttk import Progressbar
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_file_exist_and_readable)
from simba.sklearn_plot_scripts.plot_sklearn_results_all import PlotSklearnResults
from simba.gannt_creator import GanntCreator
from simba.misc_tools import get_fn_ext, find_video_of_file
import os


class SummaryVideoCreator(object):
    def __init__(self,
                 config_path: str=None,
                 sklearn_plot: bool=False,
                 gantt_plot: bool=False,
                 path_plot: bool=False,
                 data_plot: bool=False,
                 distance_plot: bool=False,
                 probability_plot: bool=False,
                 resolution: str=None,
                 data_path: str=None):

        self.sklearn_plot, self.gantt_plot = sklearn_plot, gantt_plot
        self.path_plot, self.data_plot = path_plot, data_plot
        self.distance_plot, self.probability_plot = distance_plot, probability_plot
        self.resolution, self.config_path = resolution, data_path
        self.plot_cnt = self.sklearn_plot + self.gantt_plot + self.path_plot + self.distance_plot + self.data_plot + self.distance_plot + self.probability_plot
        self.config_path = config_path
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        _, self.video_name, _ = get_fn_ext(data_path)
        self.video_folder_path = os.path.join(self.project_path, 'videos')
        self.save_folder = os.path.join(self.project_path, 'frames', 'output', 'merged')
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)

    def create_videos(self):
        if self.sklearn_plot:
            video_path = find_video_of_file(self.video_folder_path, self.video_name)
            output_path = os.path.join(self.project_path, 'frames', 'output', 'sklearn_results', self.video_name + '.mp4')
            if not os.path.isfile(output_path):
                sklearn_creator = PlotSklearnResults(config_path=self.config_path,
                                                          video_setting=True,
                                                          frame_setting=False,
                                                          video_file_path=video_path)
                sklearn_creator.initialize_visualizations()
        if self.gantt_plot:
            output_path = os.path.join(self.project_path, 'frames', 'output', 'gantt_plots', self.video_name + '.mp4')
            if not os.path.isfile(output_path):
                gannt_creator = GanntCreator(config_path=self.config_path,
                                               frame_setting=False,
                                               video_setting=True)
                gannt_creator.create_gannt()
        if self.path_plot:
            output_path = os.path.join(self.project_path, 'frames', 'output', 'gantt_plots', self.video_name + '.mp4')
            if not os.path.isfile(output_path):
                gannt_creator = GanntCreator(config_path=self.config_path,
                                               frame_setting=False,
                                               video_setting=True)
                gannt_creator.create_gannt()











# test = SummaryVideoCreator(config_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                  sklearn_plot=True,
#                  gantt_plot=True,
#                  path_plot=True,
#                  data_plot=True,
#                  distance_plot=True,
#                  probability_plot=True,
#                  resolution="1920x1080",
#                  data_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/csv/machine_results/Together_1.csv')
# test.create_videos()


