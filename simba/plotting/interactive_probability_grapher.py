__author__ = "Simon Nilsson"

import os
import platform
import threading
from copy import copy
from typing import Union

import matplotlib.pyplot as plt

import simba
from simba.mixins.config_reader import ConfigReader
from simba.plotting.tools.tkinter_tools import InteractiveVideoPlotterWindow
from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_valid_dataframe)
from simba.utils.enums import OS, Formats, Paths
from simba.utils.errors import ColumnNotFoundError, InvalidInputError
from simba.utils.read_write import get_fn_ext, get_video_meta_data, read_df
from simba.utils.warnings import FrameRangeWarning

ICON_WINDOWS = os.path.join(os.path.dirname(simba.__file__), Paths.LOGO_ICON_WINDOWS_PATH.value)
ICON_DARWIN = os.path.join(os.path.dirname(simba.__file__), Paths.LOGO_ICON_DARWIN_PATH.value)


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

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 file_path: Union[str, os.PathLike],
                 model_path: Union[str, os.PathLike],
                 lbl_font_size: int = 16):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        check_file_exist_and_readable(file_path=file_path)
        check_file_exist_and_readable(file_path=model_path)
        check_int(name=f'{self.__class__.__name__} lbl_font_size', value=lbl_font_size, min_value=1, raise_error=True)
        self.file_path, self.model_path, self.lbl_font_size = file_path, model_path, lbl_font_size
        self.click_counter = 0
        _, self.clf_name, _ = get_fn_ext(filepath=self.model_path)
        if self.clf_name not in self.clf_names:
            raise InvalidInputError(msg=f"The classifier name {self.clf_name} is not a classifier in the SimBA project. Accepted model names: {self.clf_names}. Try re-naming the classifier name or add the classifier name to the SImBA project", source=self.__class__.__name__)
        self.data_path = os.path.join(self.project_path, Paths.CLF_DATA_VALIDATION_DIR.value, os.path.basename(self.file_path))
        check_file_exist_and_readable(self.data_path)
        _, video_name, _ = get_fn_ext(filepath=file_path)
        self.data_df = read_df(self.data_path, self.file_type)
        p_col = f"Probability_{self.clf_name}"
        check_valid_dataframe(df=self.data_df, source=f'{self.__class__.__name__} {self.data_path}', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=[p_col])
        self.p_arr = self.data_df[["Probability_{}".format(self.clf_name)]].to_numpy()
        current_video_file_path = self.find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
        video_meta_data = get_video_meta_data(video_path=current_video_file_path)
        if video_meta_data['frame_count'] != len(self.data_df):
            FrameRangeWarning(msg=f'The video {current_video_file_path} contains {video_meta_data["frame_count"]} frames, while the data file {self.data_path} contains {len(self.data_df)} frames.', source=self.__class__.__name__)
        self.video_frm = InteractiveVideoPlotterWindow(video_path=current_video_file_path, p_arr=self.p_arr)
        self.video_frm.main_frm.protocol("WM_DELETE_WINDOW", self._close_windows)

    @staticmethod
    def __click_event(event):
        global current_x_cord
        if (event.dblclick) and (event.button == 1) and (type(event.xdata) != None):
            current_x_cord = int(event.xdata)

    def run(self):
        import matplotlib
        matplotlib.use("TkAgg")
        global current_x_cord
        probability_txt = (f"Selected frame: {str(0)}, {self.clf_name} probability: {self.p_arr[0][0]}")
        plt_title = f"Click on the points of the graph to display the corresponding video frame. \n {probability_txt}"
        current_x_cord, prior_x_cord = None, None

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f"SimBA - {self.clf_name} Probability")
        if (platform.system() == OS.WINDOWS.value) and os.path.isfile(ICON_WINDOWS):
            fig.canvas.manager.window.iconbitmap(ICON_WINDOWS)
        if (platform.system() == OS.MAC.value) and os.path.isfile(ICON_DARWIN):
            fig.canvas.manager.window.iconbitmap(ICON_DARWIN)
        ax.plot(self.p_arr)
        plt.xlabel("frame #", fontsize=self.lbl_font_size)
        plt.ylabel(f"{self.clf_name} probability", fontsize=self.lbl_font_size)
        plt.title(plt_title)
        plt.grid()
        line = None
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            _ = fig.canvas.mpl_connect("button_press_event", lambda event: self.__click_event(event))
            if current_x_cord != prior_x_cord:
                prior_x_cord = copy(current_x_cord)
                probability_txt = f"Selected frame: {current_x_cord}, {self.clf_name} probability: {self.p_arr[current_x_cord][0]}"
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

        self.video_frm.main_frm.destroy()

    def _close_windows(self):
        try:
            self.video_frm.main_frm.destroy()
        except:
            pass
        plt.close('all')





#
# test = InteractiveProbabilityGrapher(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                      file_path=r"C:\troubleshooting\mitra\project_folder\csv\features_extracted\501_MA142_Gi_CNO_0521.csv",
#                                      model_path=r"C:\troubleshooting\mitra\models\generated_models\straub_tail.sav")
# test.run()
#

