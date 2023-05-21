__author__ = "Simon Nilsson"

import os
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd

from simba.utils.errors import (DataHeaderError, DuplicationError,
                                InvalidFileTypeError)
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    read_config_file, read_df)


class DrawPathPlot(object):
    def __init__(
        self,
        data_path: str,
        video_path: str,
        body_part: str,
        bg_color: str,
        line_color: str,
        line_thinkness: int,
        circle_size: int,
    ):
        self.timer = SimbaTimer(start=True)
        self.named_shape_colors = get_color_dict()
        self.line_clr_bgr = self.named_shape_colors[line_color]
        self.bg_clr_bgr = self.named_shape_colors[bg_color]
        self.line_thinkness, self.circle_size = int(line_thinkness), int(circle_size)
        if self.line_clr_bgr == self.bg_clr_bgr:
            raise DuplicationError(
                msg="The line color and background color are identical"
            )
        directory, file_name, ext = get_fn_ext(filepath=data_path)
        if ext.lower() == ".h5":
            self.data = pd.read_hdf(data_path)
            headers = []
            if len(self.data.columns[0]) == 4:
                for c in self.data.columns:
                    headers.append("{}_{}_{}".format(c[1], c[2], c[3]))
            if len(self.data.columns[0]) == 3:
                for c in self.data.columns:
                    headers.append("{}_{}".format(c[2], c[3]))
            self.data.columns = headers
            if len(self.data.columns[0]) == 4:
                self.data = self.data.loc[3:]
            if len(self.data.columns[0]) == 3:
                self.data = self.data.loc[2:]
        elif ext.lower() == ".csv":
            self.data = pd.read_csv(data_path)
        else:
            raise InvalidFileTypeError(
                msg=f"File type {str(ext)} is not supported (OPTIONS: h5 or csv)"
            )
        body_parts_available = list(set([x[:-2] for x in self.data.columns]))
        self.col_heads = [body_part + "_x", body_part + "_y", body_part + "_likelihood"]
        if self.col_heads[0] not in self.data.columns:
            raise DataHeaderError(
                msg=f"Body-part {body_part} is not present in the data file. The body-parts available are: {body_parts_available}"
            )
        self.data = (
            self.data[self.col_heads]
            .fillna(method="ffill")
            .astype(int)
            .reset_index(drop=True)
        )
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.save_name = os.path.join(
            os.path.dirname(video_path), file_name + "_line_plot.mp4"
        )
        self.bg_image = np.zeros(
            [self.video_meta_data["height"], self.video_meta_data["width"], 3]
        )
        self.bg_image[:] = self.named_shape_colors[bg_color]
        self.bg_image = np.uint8(self.bg_image)
        self.writer = cv2.VideoWriter(
            self.save_name,
            0x7634706D,
            int(self.video_meta_data["fps"]),
            (self.video_meta_data["width"], self.video_meta_data["height"]),
        )
        self.cap = cv2.VideoCapture(video_path)
        self.draw_video()

    def draw_video(self):
        frm_counter, prior_x, prior_y = 0, 0, 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == True:
                current_x, current_y = (
                    self.data.loc[frm_counter, self.col_heads[0]],
                    self.data.loc[frm_counter, self.col_heads[1]],
                )
                if frm_counter > 0:
                    cv2.line(
                        self.bg_image,
                        (prior_x, prior_y),
                        (current_x, current_y),
                        self.line_clr_bgr,
                        self.line_thinkness,
                    )
                prior_x, prior_y = deepcopy(current_x), deepcopy(current_y)
                output_frm = deepcopy(self.bg_image)
                cv2.circle(
                    output_frm,
                    (current_x, current_y),
                    self.circle_size,
                    self.line_clr_bgr,
                    -1,
                )
                self.writer.write(output_frm)
                frm_counter += 1
                print(
                    "Frame {}/{}".format(
                        str(frm_counter), str(self.video_meta_data["frame_count"])
                    )
                )

            else:
                break

        self.cap.release()
        self.timer.stop_timer()
        stdout_success(
            msg=f"Path plot saved at {self.save_name}",
            elapsed_time=self.timer.elapsed_time_str,
        )


def draw_line_plot(configini, video, bodypart):
    configFile = str(configini)
    config = read_config_file(configini)
    configdir = os.path.dirname(configini)
    wfileType = "csv"
    dir_path, vid_name, ext = get_fn_ext(video)
    csvname = vid_name + "." + wfileType
    tracking_csv = os.path.join(
        configdir, "csv", "outlier_corrected_movement_location", csvname
    )
    inputDf = read_df(tracking_csv, wfileType)
    videopath = os.path.join(configdir, "videos", video)
    outputvideopath = os.path.join(configdir, "frames", "output", "simple_path_plots")

    if not os.path.exists(outputvideopath):
        os.mkdir(outputvideopath)

    # datacleaning
    colHeads = [bodypart + "_x", bodypart + "_y", bodypart + "_p"]
    df = inputDf[colHeads].copy()

    widthlist = df[colHeads[0]].astype(float).astype(int)
    heightlist = df[colHeads[1]].astype(float).astype(int)
    circletup = tuple(zip(widthlist, heightlist))

    # get resolution of video
    vcap = cv2.VideoCapture(videopath)
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        totalFrameCount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make white background
    img = np.zeros([height, width, 3])
    img.fill(255)
    img = np.uint8(img)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(
        os.path.join(outputvideopath, video), 0x7634706D, fps, (width, height)
    )
    counter = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if ret == True:
            if counter != 0:
                cv2.line(img, circletup[counter - 1], circletup[counter], 5)

            lineWithCircle = img.copy()
            cv2.circle(lineWithCircle, circletup[counter], 5, [0, 0, 255], -1)

            out.write(lineWithCircle)
            counter += 1
            print("Frame " + str(counter) + "/" + str(totalFrameCount))

        else:
            break

    vcap.release()
    cv2.destroyAllWindows()
    print("Video generated.")


def draw_line_plot_tools(videopath, csvfile, bodypart):
    inputDf = pd.read_csv(csvfile)

    # restructure
    col1 = inputDf.loc[0].to_list()
    col2 = inputDf.loc[1].to_list()
    finalcol = [m + "_" + n for m, n in zip(col1, col2)]
    inputDf.columns = finalcol
    inputDf = inputDf.loc[2:]
    print(inputDf.columns)
    inputDf = inputDf.reset_index(drop=True)
    # datacleaning
    colHeads = [bodypart + "_x", bodypart + "_y", bodypart + "_likelihood"]
    df = inputDf[colHeads].copy()

    widthlist = df[colHeads[0]].astype(float).astype(int)
    heightlist = df[colHeads[1]].astype(float).astype(int)
    circletup = tuple(zip(widthlist, heightlist))

    # get resolution of video
    vcap = cv2.VideoCapture(videopath)
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        totalFrameCount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make white background
    img = np.zeros([height, width, 3])
    img.fill(255)
    img = np.uint8(img)

    outputvideoname = os.path.join(
        os.path.dirname(videopath), "line_plot" + os.path.basename(videopath)
    )
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(outputvideoname, 0x7634706D, fps, (width, height))
    counter = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if ret == True:
            if counter != 0:
                cv2.line(img, circletup[counter - 1], circletup[counter], 5)

            lineWithCircle = img.copy()
            cv2.circle(lineWithCircle, circletup[counter], 5, [0, 0, 255], -1)

            out.write(lineWithCircle)
            counter += 1
            print("Frame " + str(counter) + "/" + str(totalFrameCount))

        else:
            break

    vcap.release()
    cv2.destroyAllWindows()
    stdout_success(msg="Video generated.")
