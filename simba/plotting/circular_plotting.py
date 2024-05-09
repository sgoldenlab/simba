import os
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm, figure

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.printing import stdout_success
from simba.utils.read_write import read_df


class CircularPlotting(PlottingMixin):
    def __init__(self):
        PlottingMixin.__init__(self)

    def diffusion_plot(self,
                       data: np.ndarray,
                       fps: int,
                       degree_width: Optional[int] = 5,
                       palette: Optional[str] = 'jet',
                       title: Optional[str] = None,
                       save_path: Optional[Union[str, os.PathLike]] = None) -> plt.figure:
        """
        Create polar plot representing the within a video.

        .. image:: _static/img/circular_plotter.png
          :width: 450
          :align: center

        :param np.ndarray data: 1D np.ndarray with angle in degrees with one entry per frame.
        :param int fps: Framerate the video was recorded in.
        :param int degree_width: The width of the bars in the plot.
        :param str palette: The polar plot palette.
        :param str title: Title of the plot
        :param Optional[Union[str, os.PathLike]] save_path: Plot save location on disk. If None, then return plt.figure polar plot.

        :example:
        >>> data = np.random.normal(loc=180, scale=99, size=5000)
        >>> _ = CircularPlotting().diffusion_plot(data=data, title='Mean 180 degree plot', fps=30, degree_width=5, palette='jet', save_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/frames/output/dispersion/20200730_AB_7dpf_850nm_0004.png')
        """

        matplotlib.rcParams["font.size"] = 100
        max_seconds = int(data.shape[0] / fps)
        second_bin = int(max_seconds / 5)
        if second_bin > 1:
            second_bin = 1
        data_rad = [x * 2 * np.pi / 360 for x in data]
        angle_bin_starts = np.arange(0.0, 2 * np.pi, 2 * np.pi * (degree_width / 360))
        n_length_bins = int(max_seconds / second_bin)
        bin_width = 2 * np.pi * (degree_width / 360)
        counts, bin_edges = np.histogram(data_rad, bins=angle_bin_starts)
        colors = self.create_single_color_lst(pallete_name=palette, increments=bin_edges.shape[0], as_rgb_ratio=True)
        norm_counts = counts / (fps * second_bin)
        bin_numbers = [np.round(norm_counts * n_length_bins / max_seconds, 0)]
        bin_lengths = [x / degree_width for x in bin_numbers]
        plt.figure().clear()
        plt.close()
        fig = figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        bars = ax.bar(angle_bin_starts[:-1], bin_lengths[0], width=bin_width, bottom=0.0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.yaxis.grid(False)
        for cnt, (r, bar) in enumerate(zip(bin_lengths[0], bars)):
            bar.set_facecolor(colors[cnt])
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
            stdout_success(msg=f"Diffusion plot {save_path} created!")
        return ax

    def diffusion_time_bin_plot(self,
                                data: np.ndarray,
                                fps: int,
                                time_bin: int,
                                degree_width: int,
                                palette: str,
                                save_path: Union[str, os.PathLike]):
        """
        Create polar plots representing angular diffusion within each N second time-bin of the video.

        .. image:: _static/img/cicular_time_bins.png
          :width: 600
          :align: center

        :param np.ndarray data: 1D np.ndarray with angle in degrees with one entry per frame.
        :param int fps: Framerate the video was recorded in.
        :param int time_bin: The length of each time bin (one plot will be created per time bin).
        :param int degree_width: The width of the bars in the plot.
        :param str palette: The polar plot palette.
        :param Optional[Union[str, os.PathLike]] save_path: Plot save location on disk. If None, then return plt.figure polar plot.

        :example:
        >>> data = np.random.normal(loc=180, scale=99, size=5000)
        >>> _ = CircularPlotting().diffusion_time_bin_plot(data=data, fps=30, degree_width=40, palette='jet', save_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/frames/output/dispertion_time_series/20200730_AB_7dpf_850nm_0004', time_bin=10)
        """
        time_bin_frame_size = time_bin * fps
        split_data = np.array_split(data, data.shape[0] / time_bin_frame_size)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for bin_cnt, time_bin_data in enumerate(split_data):
            time_plot = self.diffusion_plot(
                data=time_bin_data,
                fps=fps,
                degree_width=degree_width,
                palette=palette,
                title=f"Time bin {bin_cnt+1}",
            )
            fig_save_path = os.path.join(save_path, f"Time_bin_{bin_cnt+1}.png")
            time_plot.figure.savefig(fig_save_path)
            plt.figure().clear()
            plt.close()
            plt.clf()
            stdout_success(msg=f"Diffusion plot {save_path} created!")




# data = read_df('/Users/simon/Desktop/envs/simba/troubleshooting/zebrafish/project_folder/csv/features_extracted/test.csv', file_type='csv', usecols=[f'Fish_clockwise_angle_degrees']).values.flatten()
# CircularPlotting().diffusion_plot(data=data, title='Mean 180 degree plot', fps=30, degree_width=15, palette='jet', save_path='/Users/simon/Desktop/envs/simba/troubleshooting/zebrafish/project_folder/csv/features_extracted/test.png')
#
# #
#
#         #data = np.split(data, data.shape[0])
#
#
#
#
#
# data = np.random.normal(loc=90, scale=99, size=5000)
#
#
# _ = CircularPlotting().diffusion_plot(data=data, title='test', fps=30, degree_width=20, palette='jet', save_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/frames/output/dispersion/20200730_AB_7dpf_850nm_0004.png')
#
# #_ = CircularPlotting().diffusion_time_bin_plot(data=data, fps=30, degree_width=40, palette='jet', save_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/frames/output/dispertion_time_series/20200730_AB_7dpf_850nm_0004', time_bin=10)
#
#
# print(np.rad2deg(circstd(np.deg2rad(data))))
# print(np.rad2deg(circmean(np.deg2rad(data))))
#
#
# config_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/project_config.ini'
# save_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/frames/output/20200730_AB_7dpf_850nm_0002.png'
# DW = 20
# MAX_SECONDS = 10
# SECOND_BIN = 1
# LAST_BIN = 2*np.pi
#
# length_raw = np.arange(0,MAX_SECONDS+SECOND_BIN, SECOND_BIN)
# length_dict = {x: x/MAX_SECONDS for x in length_raw}
# data_rad = [x*2*np.pi/360 for x in data]
# angle_bin_starts = np.arange(0.0, 2*np.pi, 2*np.pi * (DW/360))
# n_length_bins = int(MAX_SECONDS/SECOND_BIN)
# bin_width = 2*np.pi * (DW/360)
#
# # for bin_start in angle_bin_starts:
# counts, bin_edges = np.histogram(data_rad, bins=angle_bin_starts)
# colors = PlottingMixin().create_single_color_lst(pallete_name='jet', increments=bin_edges.shape[0], as_rgb_ratio=True)
# norm_counts = counts/(30*SECOND_BIN)
# bin_numbers = [np.round(norm_counts * n_length_bins/MAX_SECONDS, 0)]
# bin_lengths = [x/5 for x in bin_numbers]
# bin_widths = [bin_width for idx in range(0, 72)]
# fig = figure(figsize=(8,8))
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
# #ax.set_yticklabels([list(range(1, MAX_SECONDS))])
# bars = ax.bar(angle_bin_starts[:-1], bin_lengths[0], width=bin_width, bottom=0.0)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# ax.set_yticklabels([])
# #labels = [str(int(x)) + 's' for x in np.linspace(0, MAX_SECONDS, 4)]
# #labels_cnt = np.linspace(0, 1, 4)
#
# #ax.set_rgrids(labels_cnt, labels, fontsize=16)
# #lines, labels = ax.set_rgrids(labels_cnt, labels, fontsize=16)
# for cnt, (r, bar) in enumerate(zip(bin_lengths[0], bars)):
#     bar.set_facecolor(colors[cnt])
#     #bar.set_facecolor([cm.Reds(r)])
# #
# #
# #
#
# # #plt.clf()
# # compass_plot = plt.subplot(1, 1, 1, projection='polar')
# # compass_plot.set_theta_zero_location('N')
# # bars = compass_plot.bar(data, 0.01, width=0.01, bottom=0.0)
# #
# # # N = 20
# # theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
# # radii = 10*np.random.rand(N)
# # width = np.pi/4*np.random.rand(N)
# # bars = ax.bar(theta, radii, width=width, bottom=0.0)
# # for r,bar in zip(radii, bars):
# #     bar.set_facecolor( cm.jet(r/10.))
# #     bar.set_alpha(0.5)
# #
# # show()
