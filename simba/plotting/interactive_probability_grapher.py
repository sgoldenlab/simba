__author__ = "Simon Nilsson"

import os
import platform
from copy import copy
from typing import Union, Tuple

import matplotlib.pyplot as plt

import simba
from simba.mixins.config_reader import ConfigReader
from simba.plotting.tools.tkinter_tools import InteractiveVideoPlotterWindow
from simba.utils.checks import (check_file_exist_and_readable, check_int, check_valid_dataframe, check_if_valid_rgb_tuple, check_valid_boolean)
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
                 lbl_font_size: int = 16,
                 data_clr: Tuple[int, int, int] = (0, 0, 255),
                 line_clr: Tuple[int, int, int] = (255, 0, 0),
                 show_thresholds: bool = True):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        check_file_exist_and_readable(file_path=file_path)
        check_file_exist_and_readable(file_path=model_path)
        check_int(name=f'{self.__class__.__name__} lbl_font_size', value=lbl_font_size, min_value=1, raise_error=True)
        check_if_valid_rgb_tuple(data=data_clr, raise_error=True, source=f'{self.__class__.__name__} data_clr')
        check_if_valid_rgb_tuple(data=line_clr, raise_error=True, source=f'{self.__class__.__name__} line_clr')
        check_valid_boolean(value=show_thresholds, source=f'{check_valid_boolean.__name__} show_thresholds', raise_error=True)
        self.file_path, self.model_path, self.lbl_font_size = file_path, model_path, lbl_font_size
        self.click_counter, self.is_playing = 0, False
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
        self.video_meta_data = get_video_meta_data(video_path=current_video_file_path)
        self.data_clr, self.line_clr = tuple([x/255 for x in data_clr]), tuple([x/255 for x in line_clr])
        self.show_thresholds = show_thresholds
        self.play_speed = self.video_meta_data['fps'] / 1000
        if self.video_meta_data['frame_count'] != len(self.data_df):
            FrameRangeWarning(msg=f'The video {current_video_file_path} contains {self.video_meta_data["frame_count"]} frames, while the data file {self.data_path} contains {len(self.data_df)} frames.', source=self.__class__.__name__)
        self.video_frm = InteractiveVideoPlotterWindow(video_path=current_video_file_path, p_arr=self.p_arr)
        self.video_frm.main_frm.protocol("WM_DELETE_WINDOW", self._close_windows)

    @staticmethod
    def __click_event(event):
        global current_x_cord
        if (event.dblclick) and (event.button == 1) and (type(event.xdata) != None):
            current_x_cord = int(event.xdata)

    def __key_press_event(self, event):
        global current_x_cord
        if event.key == ' ':
            self.is_playing = not self.is_playing
        if event.key == 'left' and current_x_cord is not None and current_x_cord > 0:
            current_x_cord -= 1
        elif event.key == 'right' and current_x_cord is not None and current_x_cord < len(self.p_arr) - 1:
            current_x_cord += 1

    def run(self):
        import matplotlib
        matplotlib.use("TkAgg")
        global current_x_cord
        prob_val_txt = round(float(self.p_arr[0][0]), 8)
        probability_txt = (f"Selected frame: {str(0)}, {self.clf_name} probability: {prob_val_txt}")
        plt_title = f"Click on the points of the graph to display the corresponding video frame. \n {probability_txt}"
        current_x_cord, prior_x_cord = None, None

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        fig.patch.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=11, colors='#333333', 
                       length=6, width=1.5, direction='out')

        ax.plot(self.p_arr, color='black', linewidth=2, alpha=0.1, zorder=1)  # shadow
        ax.plot(self.p_arr, color=self.data_clr, linewidth=1.5, alpha=0.9, zorder=2, label='Probability')
        if self.show_thresholds:
            ax.axhline(y=0.75, color='#ec4899', linestyle=(0, (3, 1, 1, 1)), linewidth=1.5, alpha=0.9, label='Threshold: 75%')
            ax.axhline(y=0.5, color='#3b82f6', linestyle=(0, (3, 1, 1, 1)), linewidth=1.5, alpha=0.9, label='Threshold: 50%')
            ax.axhline(y=0.25, color='#8b5cf6', linestyle=(0, (3, 1, 1, 1)), linewidth=1.5, alpha=0.9, label='Threshold: 25%')
        ax.legend(loc='upper right', frameon=True, fancybox=True, 
                  framealpha=0.95, edgecolor='#cccccc', fontsize=10)

        fig.canvas.manager.set_window_title(f"SimBA - {self.clf_name} Probability - {get_fn_ext(filepath=self.file_path)[1]}")
        if (platform.system() == OS.WINDOWS.value) and os.path.isfile(ICON_WINDOWS):
            fig.canvas.manager.window.iconbitmap(ICON_WINDOWS)
        if (platform.system() == OS.MAC.value) and os.path.isfile(ICON_DARWIN):
            fig.canvas.manager.window.iconbitmap(ICON_DARWIN)
        
        plt.xlabel("Frame #", fontsize=self.lbl_font_size, fontweight='500')
        plt.ylabel(f"{self.clf_name} Probability", fontsize=self.lbl_font_size, fontweight='500')
        plt.title(plt_title, fontsize=self.lbl_font_size - 2, pad=20)

        ax.text(0.5, 1.20, "Double-click: jump to frame | ← →: navigate | Space: play/pause",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                          edgecolor='#cccccc', alpha=0.9, linewidth=1.5))
        line, marker = None, None
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.canvas.mpl_connect("button_press_event", lambda event: self.__click_event(event))  # ADD THIS - it's missing!
        fig.canvas.mpl_connect("key_press_event", self.__key_press_event)
        plt.show(block=False)

        while plt.fignum_exists(fig.number):
            if current_x_cord != prior_x_cord and current_x_cord <= self.p_arr.shape[0]:
                prior_x_cord = copy(current_x_cord)
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                prob_val_txt = round(float(self.p_arr[current_x_cord][0]), 8)
                probability_txt = f"Selected frame: {current_x_cord}, {self.clf_name} probability: {prob_val_txt}"
                plt_title = f"Click on the points of the graph to display the corresponding video frame. \n {probability_txt}"
                self.video_frm.load_new_frame(frm_cnt=int(current_x_cord))
                if line is not None: line.remove()
                if marker is not None: marker.pop(0).remove()
                plt.title(plt_title)
                line = plt.axvline(x=current_x_cord, color=self.line_clr, alpha=0.8, linewidth=2)
                marker = ax.plot(current_x_cord, self.p_arr[current_x_cord][0], 'o',
                                 markersize=8, color=self.line_clr, markeredgecolor='white',
                                 markeredgewidth=2, zorder=5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                
                fig.canvas.draw()
                fig.canvas.flush_events()

            if self.is_playing and current_x_cord is not None and current_x_cord < len(self.p_arr) - 1:
                current_x_cord += 1

            plt.ion()
            plt.pause(self.play_speed)

        self.video_frm.main_frm.destroy()

    def _close_windows(self):
        try:
            self.video_frm.main_frm.destroy()
        except:
            pass
        plt.close('all')


# test = InteractiveProbabilityGrapher(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                      file_path=r"C:\troubleshooting\mitra\project_folder\csv\features_extracted\501_MA142_Gi_CNO_0521.csv",
#                                      model_path=r"C:\troubleshooting\mitra\models\generated_models\straub_tail.sav")
# test.run()


# test = InteractiveProbabilityGrapher(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/project_config.ini",
#                                      file_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/validation/704_MA115_Gi_CNO_0521.csv",
#                                      model_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/mitra/models/generated_models/grooming.sav")
# test.run()



