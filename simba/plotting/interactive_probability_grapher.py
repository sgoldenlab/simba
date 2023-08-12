__author__ = "Simon Nilsson"

import os
import threading
from copy import copy

import matplotlib.pyplot as plt

from simba.mixins.config_reader import ConfigReader
from simba.plotting.tools.tkinter_tools import InteractiveVideoPlotterWindow
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Paths
from simba.utils.errors import ColumnNotFoundError, InvalidInputError
from simba.utils.read_write import get_fn_ext, read_df


class InteractiveProbabilityGrapher(ConfigReader):
    """
    Launch interactive GUI for classifier probability inspection.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str file_path: Data with classification probability field.
    :param str model_path: Path to classifier used to create probability field.

    .. note::
       `Validation tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#critical-validation-step-before-running-machine-model-on-new-data>`__.
        .. image:: _static/img/interactive_probability_plot.png
           :width: 450
           :align: center


    Examples
    ----------
    >>> interactive_plotter = InteractiveProbabilityGrapher(config_path=r'MyConfigPath', file_path='MyFeatureFilepath', model_path='MyPickledClassifier.sav')
    >>> interactive_plotter.run()
    """

    def __init__(self, config_path: str, file_path: str, model_path: str):
        super().__init__(config_path=config_path)
        check_file_exist_and_readable(file_path=file_path)
        check_file_exist_and_readable(file_path=model_path)
        self.file_path, self.model_path = file_path, model_path
        self.click_counter = 0
        _, self.clf_name, _ = get_fn_ext(filepath=self.model_path)
        if self.clf_name not in self.clf_names:
            raise InvalidInputError(
                msg=f"The classifier {self.clf_name} is not a classifier in the SimBA project: {self.clf_names}"
            )
        self.data_path = os.path.join(
            self.project_path,
            Paths.CLF_DATA_VALIDATION_DIR.value,
            os.path.basename(self.file_path),
        )
        check_file_exist_and_readable(self.data_path)
        _, video_name, _ = get_fn_ext(filepath=file_path)
        self.data_df = read_df(self.data_path, self.file_type)
        if f"Probability_{self.clf_name}" not in self.data_df.columns:
            raise ColumnNotFoundError(
                column_name=f"Probability_{self.clf_name}", file_name=self.data_path
            )
        self.p_arr = self.data_df[["Probability_{}".format(self.clf_name)]].to_numpy()
        current_video_file_path = self.find_video_of_file(
            video_dir=self.video_dir, filename=video_name
        )
        self.video_frm = InteractiveVideoPlotterWindow(
            video_path=current_video_file_path, p_arr=self.p_arr
        )

    @staticmethod
    def __click_event(event):
        global current_x_cord
        if (event.dblclick) and (event.button == 1) and (type(event.xdata) != None):
            current_x_cord = int(event.xdata)

    def run(self):
        import matplotlib

        matplotlib.use("TkAgg")
        global current_x_cord
        probability_txt = (
            f"Selected frame: {str(0)}, {self.clf_name} probability: {self.p_arr[0][0]}"
        )
        plt_title = f"Click on the points of the graph to display the corresponding video frame. \n {probability_txt}"
        current_x_cord, prior_x_cord = None, None

        fig, ax = plt.subplots()
        ax.plot(self.p_arr)
        plt.xlabel("frame #", fontsize=16)
        plt.ylabel(f"{self.clf_name} probability", fontsize=16)
        plt.title(plt_title)
        plt.grid()
        line = None
        fig.canvas.draw()
        fig.canvas.flush_events()

        while True:
            _ = fig.canvas.mpl_connect(
                "button_press_event", lambda event: self.__click_event(event)
            )
            if current_x_cord != prior_x_cord:
                prior_x_cord = copy(current_x_cord)
                probability_txt = "Selected frame: {}, {} probability: {}".format(
                    str(current_x_cord),
                    self.clf_name,
                    str(self.p_arr[current_x_cord][0]),
                )
                plt_title = f"Click on the points of the graph to display the corresponding video frame. \n {probability_txt}"
                self.video_frm.load_new_frame(frm_cnt=int(current_x_cord))
                if line != None:
                    line.remove()
                plt.title(plt_title)
                line = plt.axvline(x=current_x_cord, color="r")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.ion()
            threading.Thread(plt.show()).start()
            plt.pause(0.0001)


# test = InteractiveProbabilityGrapher(config_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                                      file_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/SF2.csv',
#                                      model_path='/Users/simon/Desktop/envs/troubleshooting/naresh/models/generated_models/Top.sav')
# test.create_plots()
