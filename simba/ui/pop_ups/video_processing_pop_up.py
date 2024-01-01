__author__ = "Simon Nilsson"

import glob
import os
import sys
from tkinter import *

from PIL import Image, ImageTk

import simba
from simba.labelling.extract_labelled_frames import AnnotationFrameExtractor
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.frame_mergerer_ffmpeg import FrameMergererFFmpeg
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        CreateToolTip, DropDownMenu, Entry_Box,
                                        FileSelect, FolderSelect)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty, check_int)
from simba.utils.data import convert_roi_definitions
from simba.utils.enums import Dtypes, Formats, Keys, Links, Options, Paths
from simba.utils.errors import (CountError, FrameRangeError, MixedMosaicError,
                                NoChoosenClassifierError, NoFilesFoundError,
                                NotDirectoryError)
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext,
                                    get_video_meta_data)
from simba.video_processors.extract_seqframes import extract_seq_frames
from simba.video_processors.multi_cropper import MultiCropper
from simba.video_processors.px_to_mm import get_coordinates_nilsson
from simba.video_processors.video_processing import (
    VideoRotator, batch_convert_video_format, batch_create_frames,
    change_fps_of_multiple_videos, change_img_format, change_single_video_fps,
    clahe_enhance_video, clip_video_in_range, convert_to_mp4,
    convert_video_powerpoint_compatible_format, copy_img_folder,
    crop_multiple_videos, crop_single_video, downsample_video,
    extract_frame_range, extract_frames_single_video, frames_to_movie,
    gif_creator, multi_split_video, remove_beginning_of_video,
    superimpose_frame_count, video_concatenator, video_to_greyscale)

sys.setrecursionlimit(10**7)


class CLAHEPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLAHE VIDEO CONVERSION")
        clahe_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Contrast Limited Adaptive Histogram Equalization",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        selected_video = FileSelect(
            clahe_frm,
            "Video path ",
            title="Select a video file",
            file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        button_clahe = Button(
            clahe_frm,
            text="Apply CLAHE",
            command=lambda: clahe_enhance_video(file_path=selected_video.file_path),
        )
        clahe_frm.grid(row=0, sticky=W)
        selected_video.grid(row=0, sticky=W)
        button_clahe.grid(row=1, pady=5)
        # self.main_frm.mainloop()


# _ = CLAHEPopUp()


class CropVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CROP SINGLE VIDEO")
        crop_video_lbl_frm = LabelFrame(
            self.main_frm,
            text="Crop Video",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        selected_video = FileSelect(
            crop_video_lbl_frm,
            "Video path",
            title="Select a video file",
            lblwidth=20,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        use_gpu_var_single = BooleanVar(value=False)
        use_gpu_cb_single = Checkbutton(
            crop_video_lbl_frm,
            text="Use GPU (reduced runtime)",
            variable=use_gpu_var_single,
        )
        button_crop_video_single = Button(
            crop_video_lbl_frm,
            text="Crop Video",
            command=lambda: crop_single_video(
                file_path=selected_video.file_path, gpu=use_gpu_var_single.get()
            ),
        )

        crop_video_lbl_frm_multiple = LabelFrame(
            self.main_frm,
            text="Fixed coordinates crop for multiple videos",
            font="bold",
            padx=5,
            pady=5,
        )
        input_folder = FolderSelect(
            crop_video_lbl_frm_multiple,
            "Video directory:",
            title="Select Folder with videos",
            lblwidth=20,
        )
        output_folder = FolderSelect(
            crop_video_lbl_frm_multiple,
            "Output directory:",
            title="Select a folder for your output videos",
            lblwidth=20,
        )
        use_gpu_var_multiple = BooleanVar(value=False)
        use_gpu_cb_multiple = Checkbutton(
            crop_video_lbl_frm_multiple,
            text="Use GPU (reduced runtime)",
            variable=use_gpu_var_multiple,
        )
        button_crop_video_multiple = Button(
            crop_video_lbl_frm_multiple,
            text="Crop Videos",
            command=lambda: crop_multiple_videos(
                directory_path=input_folder.folder_path,
                output_path=output_folder.folder_path,
                gpu=use_gpu_var_multiple.get(),
            ),
        )
        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        use_gpu_cb_single.grid(row=1, column=0, sticky=NW)
        button_crop_video_single.grid(row=2, sticky=NW)
        crop_video_lbl_frm_multiple.grid(row=1, sticky=NW)
        input_folder.grid(row=0, sticky=NW)
        output_folder.grid(row=1, sticky=NW)
        use_gpu_cb_multiple.grid(row=2, sticky=NW)
        button_crop_video_multiple.grid(row=3, sticky=NW)


#         self.main_frm.mainloop()
#
# _ = CropVideoPopUp()


class ClipVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP VIDEO")
        selected_video_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Video path",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        selected_video = FileSelect(
            selected_video_frm,
            "FILE PATH: ",
            file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        selected_video.grid(row=0, column=0, sticky="NW")
        use_gpu_frm = LabelFrame(self.main_frm, text="GPU", font="bold", padx=5, pady=5)
        self.use_gpu_var = BooleanVar(value=False)
        self.use_gpu_cb = Checkbutton(
            use_gpu_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )
        self.use_gpu_cb.grid(row=0, column=0, sticky=NW)
        method_1_frm = LabelFrame(
            self.main_frm, text="Method 1", font="bold", padx=5, pady=5
        )
        label_set_time_1 = Label(
            method_1_frm, text="Please enter the time frame in HH:MM:SS format"
        )
        start_time = Entry_Box(method_1_frm, "Start at (s):", "8")
        end_time = Entry_Box(method_1_frm, "End at (s):", "8")
        CreateToolTip(
            method_1_frm,
            "Method 1 will retrieve the specified time input. (eg: input of Start at: 00:00:00, End at: 00:01:00, will create a new video from the chosen video from the very start till it reaches the first minute of the video)",
        )
        method_2_frm = LabelFrame(
            self.main_frm, text="Method 2", font="bold", padx=5, pady=5
        )
        method_2_time = Entry_Box(method_2_frm, "Seconds:", "8", validation="numeric")
        label_method_2 = Label(
            method_2_frm,
            text="Method 2 will retrieve from the end of the video (e.g.,: an input of 3 seconds will get rid of the first 3 seconds of the video).",
        )
        button_cutvideo_method_1 = Button(
            method_1_frm,
            text="Cut Video",
            command=lambda: clip_video_in_range(
                file_path=selected_video.file_path,
                start_time=start_time.entry_get,
                end_time=end_time.entry_get,
                gpu=self.use_gpu_var.get(),
            ),
        )
        button_cutvideo_method_2 = Button(
            method_2_frm,
            text="Cut Video",
            command=lambda: remove_beginning_of_video(
                file_path=selected_video.file_path,
                time=method_2_time.entry_get,
                gpu=self.use_gpu_var.get(),
            ),
        )
        selected_video_frm.grid(row=0, sticky=NW)
        use_gpu_frm.grid(row=1, column=0, sticky=NW)
        method_1_frm.grid(row=2, sticky=NW, pady=5)
        label_set_time_1.grid(row=0, sticky=NW)
        start_time.grid(row=1, sticky=NW)
        end_time.grid(row=2, sticky=NW)
        button_cutvideo_method_1.grid(row=3, sticky=NW)
        method_2_frm.grid(row=3, sticky=NW, pady=5)
        label_method_2.grid(row=0, sticky=NW)
        method_2_time.grid(row=2, sticky=NW)
        button_cutvideo_method_2.grid(row=3, sticky=NW)


#       self.main_frm.mainloop()
# _ = ClipVideoPopUp()


class GreyscaleSingleVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="GREYSCALE SINGLE VIDEO")
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="GREYSCALE VIDEO",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.selected_video = FileSelect(
            settings_frm,
            "VIDEO PATH",
            title="Select a video file",
            lblwidth=10,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            settings_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        use_gpu_cb.grid(row=1, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_file_exist_and_readable(file_path=self.selected_video.file_path)
        video_to_greyscale(
            file_path=self.selected_video.file_path, gpu=self.use_gpu_var.get()
        )


class SuperImposeFrameCountPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="SUPERIMPOSE FRAME COUNT ON VIDEO")
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SUPERIMPOSE FRAME COUNT ON VIDEO",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.selected_video = FileSelect(
            settings_frm,
            "VIDEO PATH",
            title="Select a video file",
            lblwidth=10,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            settings_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        use_gpu_cb.grid(row=1, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_file_exist_and_readable(file_path=self.selected_video.file_path)
        superimpose_frame_count(
            file_path=self.selected_video.file_path, gpu=self.use_gpu_var.get()
        )


class MultiShortenPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP VIDEO INTO MULTIPLE VIDEOS", size=(800, 200))
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Split videos into different parts",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.selected_video = FileSelect(
            settings_frm,
            "Video path",
            title="Select a video file",
            lblwidth=10,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.clip_cnt = Entry_Box(
            settings_frm, "# of clips", "10", validation="numeric"
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            settings_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )
        confirm_settings_btn = Button(
            settings_frm, text="Confirm", command=lambda: self.show_start_stop()
        )
        settings_frm.grid(row=0, sticky=NW)
        self.selected_video.grid(row=1, sticky=NW, columnspan=2)
        self.clip_cnt.grid(row=2, sticky=NW)
        confirm_settings_btn.grid(row=2, column=1, sticky=W)
        use_gpu_cb.grid(row=3, column=0, sticky=W)
        instructions = Label(
            settings_frm,
            text="Enter clip start and stop times in HH:MM:SS format",
            fg="navy",
        )
        instructions.grid(row=4, column=0)

        batch_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Batch change time",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.batch_start_entry = Entry_Box(batch_frm, "START:", "10")
        self.batch_start_entry.entry_set("00:00:00")
        self.batch_end_entry = Entry_Box(batch_frm, "END", "10")
        self.batch_end_entry.entry_set("00:00:00")
        batch_start_apply = Button(
            batch_frm, text="APPLY", command=lambda: self.batch_change(value="start")
        )
        batch_end_apply = Button(
            batch_frm, text="APPLY", command=lambda: self.batch_change(value="end")
        )

        batch_frm.grid(row=0, column=1, sticky=NW)
        self.batch_start_entry.grid(row=0, column=0, sticky=NW)
        batch_start_apply.grid(row=0, column=1, sticky=NW)
        self.batch_end_entry.grid(row=1, column=0, sticky=NW)
        batch_end_apply.grid(row=1, column=1, sticky=NW)

        # self.main_frm.mainloop()

    def show_start_stop(self):
        check_int(name="Number of clips", value=self.clip_cnt.entry_get)
        if hasattr(self, "table"):
            self.table.destroy()
        self.table = LabelFrame(self.main_frm)
        self.table.grid(row=2, column=0, sticky=NW)
        Label(self.table, text="Clip #").grid(row=0, column=0)
        Label(self.table, text="Start Time").grid(row=0, column=1, sticky=NW)
        Label(self.table, text="Stop Time").grid(row=0, column=2, sticky=NW)
        self.clip_names, self.start_times, self.end_times = [], [], []
        for i in range(int(self.clip_cnt.entry_get)):
            Label(self.table, text="Clip " + str(i + 1)).grid(row=i + 2, sticky=W)
            self.start_times.append(Entry(self.table))
            self.start_times[i].grid(row=i + 2, column=1, sticky=NW)
            self.end_times.append(Entry(self.table))
            self.end_times[i].grid(row=i + 2, column=2, sticky=NW)
        run_button = Button(
            self.table,
            text="Clip video",
            command=lambda: self.run_clipping(),
            fg="navy",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        run_button.grid(row=int(self.clip_cnt.entry_get) + 2, column=2, sticky=W)

    def batch_change(self, value: str):
        if not hasattr(self, "table"):
            raise CountError(
                msg="Select the number of video clippings first",
                source=self.__class__.__name__,
            )
        for start_time_entry, end_time_entry in zip(self.start_times, self.end_times):
            if value == "start":
                start_time_entry.delete(0, END)
                start_time_entry.insert(0, self.batch_start_entry.entry_get)
            else:
                end_time_entry.delete(0, END)
                end_time_entry.insert(0, self.batch_end_entry.entry_get)

    def run_clipping(self):
        start_times, end_times = [], []
        check_file_exist_and_readable(self.selected_video.file_path)
        for start_time, end_time in zip(self.start_times, self.end_times):
            start_times.append(start_time.get())
            end_times.append(end_time.get())
        multi_split_video(
            file_path=self.selected_video.file_path,
            start_times=start_times,
            end_times=end_times,
            gpu=self.use_gpu_var.get(),
        )


# _ = MultiShortenPopUp()


class ChangeImageFormatPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CHANGE IMAGE FORMAT")

        self.input_folder_selected = FolderSelect(
            self.main_frm, "Image directory", title="Select folder with images:"
        )
        set_input_format_frm = LabelFrame(
            self.main_frm,
            text="Original image format",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            padx=15,
            pady=5,
        )
        set_output_format_frm = LabelFrame(
            self.main_frm,
            text="Output image format",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            padx=15,
            pady=5,
        )

        self.input_file_type, self.out_file_type = StringVar(), StringVar()
        input_png_rb = Radiobutton(
            set_input_format_frm,
            text=".png",
            variable=self.input_file_type,
            value="png",
        )
        input_jpeg_rb = Radiobutton(
            set_input_format_frm,
            text=".jpg",
            variable=self.input_file_type,
            value="jpg",
        )
        input_bmp_rb = Radiobutton(
            set_input_format_frm,
            text=".bmp",
            variable=self.input_file_type,
            value="bmp",
        )
        output_png_rb = Radiobutton(
            set_output_format_frm, text=".png", variable=self.out_file_type, value="png"
        )
        output_jpeg_rb = Radiobutton(
            set_output_format_frm, text=".jpg", variable=self.out_file_type, value="jpg"
        )
        output_bmp_rb = Radiobutton(
            set_output_format_frm, text=".bmp", variable=self.out_file_type, value="bmp"
        )
        run_btn = Button(
            self.main_frm,
            text="Convert image file format",
            command=lambda: self.run_img_conversion(),
        )
        self.input_folder_selected.grid(row=0, column=0)
        set_input_format_frm.grid(row=1, column=0, pady=5)
        set_output_format_frm.grid(row=2, column=0, pady=5)
        input_png_rb.grid(row=0, column=0)
        input_jpeg_rb.grid(row=1, column=0)
        input_bmp_rb.grid(row=2, column=0)
        output_png_rb.grid(row=0, column=0)
        output_jpeg_rb.grid(row=1, column=0)
        output_bmp_rb.grid(row=2, column=0)
        run_btn.grid(row=3, pady=5)

    def run_img_conversion(self):
        if len(os.listdir(self.input_folder_selected.folder_path)) == 0:
            raise NoFilesFoundError(
                msg="SIMBA ERROR: The input folder {} contains ZERO files.".format(
                    self.input_folder_selected.folder_path
                ),
                source=self.__class__.__name__,
            )
        change_img_format(
            directory=self.input_folder_selected.folder_path,
            file_type_in=self.input_file_type.get(),
            file_type_out=self.out_file_type.get(),
        )


class ConvertVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CONVERT VIDEO FORMAT", size=(200, 200))
        convert_multiple_videos_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Convert multiple videos",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        video_dir = FolderSelect(
            convert_multiple_videos_frm,
            "Video directory",
            title="Select folder with videos",
            lblwidth=15,
        )
        video_format_options = ["mp4", "avi", "mov", "flv", "m4v"]
        original_format_dropdown = DropDownMenu(
            convert_multiple_videos_frm,
            "Input format",
            video_format_options,
            labelwidth=15,
        )
        output_format_dropdown = DropDownMenu(
            convert_multiple_videos_frm,
            "Input format",
            video_format_options,
            labelwidth=15,
        )
        original_format_dropdown.setChoices("avi")
        output_format_dropdown.setChoices("mp4")
        gpu_multiple_var = BooleanVar(value=False)
        gpu_multiple_cb = Checkbutton(
            convert_multiple_videos_frm,
            text="Use GPU (reduced runtime)",
            variable=gpu_multiple_var,
        )
        convert_multiple_btn = Button(
            convert_multiple_videos_frm,
            text="Convert multiple videos",
            command=lambda: batch_convert_video_format(
                directory=video_dir.folder_path,
                input_format=original_format_dropdown.getChoices(),
                output_format=output_format_dropdown.getChoices(),
                gpu=gpu_multiple_var.get(),
            ),
        )
        convert_single_video_frm = LabelFrame(
            self.main_frm,
            text="Convert single video",
            font=("Helvetica", 12, "bold"),
            padx=5,
            pady=5,
        )
        self.selected_video = FileSelect(
            convert_single_video_frm,
            "Video path",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.output_format = StringVar()
        checkbox_v1 = Radiobutton(
            convert_single_video_frm,
            text="Convert to .mp4",
            variable=self.output_format,
            value="mp4",
        )
        checkbox_v2 = Radiobutton(
            convert_single_video_frm,
            text="Convert mp4 into PowerPoint supported format",
            variable=self.output_format,
            value="pptx",
        )
        self.gpu_single_var = BooleanVar(value=False)
        gpu_single_cb = Checkbutton(
            convert_single_video_frm,
            text="Use GPU (reduced runtime)",
            variable=self.gpu_single_var,
        )
        convert_single_btn = Button(
            convert_single_video_frm,
            text="Convert video format",
            command=lambda: self.convert_single(),
        )

        convert_multiple_videos_frm.grid(row=0, sticky=NW)
        video_dir.grid(row=0, sticky=NW)
        original_format_dropdown.grid(row=1, sticky=NW)
        output_format_dropdown.grid(row=2, sticky=NW)
        gpu_multiple_cb.grid(row=3, sticky=NW)
        convert_multiple_btn.grid(row=4, pady=10, sticky=NW)
        convert_single_video_frm.grid(row=1, sticky=NW)
        self.selected_video.grid(row=0, sticky=NW)
        checkbox_v1.grid(row=1, column=0, sticky=NW)
        checkbox_v2.grid(row=2, column=0, sticky=NW)
        gpu_single_cb.grid(row=3, column=0, sticky=NW)
        convert_single_btn.grid(row=4, column=0, pady=10, sticky=NW)

    def convert_single(self):
        if self.output_format.get() == "mp4":
            convert_to_mp4(
                file_path=self.selected_video.file_path, gpu=self.gpu_single_var.get()
            )
        if self.output_format.get() == "pptx":
            convert_video_powerpoint_compatible_format(
                file_path=self.selected_video.file_path, gpu=self.gpu_single_var.get()
            )


class ExtractSpecificFramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="EXTRACT DEFINED FRAMES", size=(200, 200))
        self.video_file_selected = FileSelect(
            self.main_frm,
            "Video path",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        select_frames_frm = LabelFrame(
            self.main_frm, text="Frames to be extracted", padx=5, pady=5
        )
        self.start_frm = Entry_Box(select_frames_frm, "Start Frame:", "10")
        self.end_frm = Entry_Box(select_frames_frm, "End Frame:", "10")
        run_btn = Button(
            select_frames_frm,
            text="Extract Frames",
            command=lambda: self.start_frm_extraction(),
        )

        self.video_file_selected.grid(row=0, column=0, sticky=NW, pady=10)
        select_frames_frm.grid(row=1, column=0, sticky=NW)
        self.start_frm.grid(row=2, column=0, sticky=NW)
        self.end_frm.grid(row=3, column=0, sticky=NW)
        run_btn.grid(row=4, pady=5, sticky=NW)

    def start_frm_extraction(self):
        start_frame = self.start_frm.entry_get
        end_frame = self.end_frm.entry_get
        check_int(name="Start frame", value=start_frame)
        check_int(name="End frame", value=end_frame)
        if int(end_frame) < int(start_frame):
            raise FrameRangeError(
                msg="SIMBA ERROR: The end frame ({}) cannot come before the start frame ({})".format(
                    str(end_frame), str(start_frame)
                ),
                source=self.__class__.__name__,
            )
        video_meta_data = get_video_meta_data(
            video_path=self.video_file_selected.file_path
        )
        if int(start_frame) > video_meta_data["frame_count"]:
            raise FrameRangeError(
                msg="SIMBA ERROR: The start frame ({}) is larger than the number of frames in the video ({})".format(
                    str(start_frame), str(video_meta_data["frame_count"])
                ),
                source=self.__class__.__name__,
            )
        if int(end_frame) > video_meta_data["frame_count"]:
            raise FrameRangeError(
                msg="SIMBA ERROR: The end frame ({}) is larger than the number of frames in the video ({})".format(
                    str(end_frame), str(video_meta_data["frame_count"])
                ),
                source=self.__class__.__name__,
            )
        extract_frame_range(
            file_path=self.video_file_selected.file_path,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
        )


class ExtractAllFramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="EXTRACT ALL FRAMES", size=(200, 200))
        single_video_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Single video",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        video_path = FileSelect(
            single_video_frm,
            "Video path",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        single_video_btn = Button(
            single_video_frm,
            text="Extract Frames (Single video)",
            command=lambda: extract_frames_single_video(file_path=video_path.file_path),
        )
        multiple_videos_frm = LabelFrame(
            self.main_frm, text="Multiple videos", padx=5, pady=5, font="bold"
        )
        folder_path = FolderSelect(
            multiple_videos_frm, "Folder path", title=" Select video folder"
        )
        multiple_video_btn = Button(
            multiple_videos_frm,
            text="Extract Frames (Multiple videos)",
            command=lambda: batch_create_frames(directory=folder_path.folder_path),
        )
        single_video_frm.grid(row=0, sticky=NW, pady=10)
        video_path.grid(row=0, sticky=NW)
        single_video_btn.grid(row=1, sticky=W, pady=10)
        multiple_videos_frm.grid(row=1, sticky=W, pady=10)
        folder_path.grid(row=0, sticky=W)
        multiple_video_btn.grid(row=1, sticky=W, pady=10)


class MultiCropPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="MULTI-CROP", size=(500, 300))
        self.input_folder = FolderSelect(
            self.main_frm, "Input Video Folder", lblwidth=15
        )
        self.output_folder = FolderSelect(self.main_frm, "Output Folder", lblwidth=15)
        video_options = ["mp4", "avi", "mov", "flv", "m4v"]
        self.video_type_dropdown = DropDownMenu(
            self.main_frm, "Video type:", video_options, "15"
        )
        self.video_type_dropdown.setChoices("mp4")
        self.crop_cnt_dropdown = DropDownMenu(
            self.main_frm, "Crop count:", list(range(1, 31)), "15"
        )
        self.crop_cnt_dropdown.setChoices(2)
        self.use_gpu_var = BooleanVar(value=False)
        self.use_gpu_cb = Checkbutton(
            self.main_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )
        self.create_run_frm(run_function=self.run)
        self.input_folder.grid(row=0, sticky=NW)
        self.output_folder.grid(row=1, sticky=NW)
        self.video_type_dropdown.grid(row=2, sticky=NW)
        self.crop_cnt_dropdown.grid(row=3, sticky=NW)
        self.use_gpu_cb.grid(row=4, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.input_folder.folder_path)
        check_if_dir_exists(in_dir=self.output_folder.folder_path)
        MultiCropper(
            file_type=self.video_type_dropdown.getChoices(),
            input_folder=self.input_folder.folder_path,
            output_folder=self.output_folder.folder_path,
            crop_cnt=self.crop_cnt_dropdown.getChoices(),
            gpu=self.use_gpu_var.get(),
        ).run()


# MultiCropPopUp()


class ChangeFpsSingleVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(
            self, title="CHANGE FRAME RATE: SINGLE VIDEO", size=(200, 200)
        )
        video_path = FileSelect(
            self.main_frm,
            "Video path",
            title="Select a video file",
            lblwidth=10,
            file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        fps_entry_box = Entry_Box(
            self.main_frm, "Output FPS:", "10", validation="numeric"
        )
        gpu_var = BooleanVar(value=False)
        gpu_cb = Checkbutton(
            self.main_frm, text="Use GPU (reduced runtime)", variable=gpu_var
        )
        run_btn = Button(
            self.main_frm,
            text="Convert",
            command=lambda: change_single_video_fps(
                file_path=video_path.file_path,
                fps=fps_entry_box.entry_get,
                gpu=gpu_var.get(),
            ),
        )
        video_path.grid(row=0, sticky=NW)
        fps_entry_box.grid(row=1, sticky=NW)
        gpu_cb.grid(row=2, sticky=NW)
        run_btn.grid(row=3, sticky=NW)


class ChangeFpsMultipleVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(
            self, title="CHANGE FRAME RATE: MULTIPLE VIDEO", size=(400, 200)
        )
        folder_path = FolderSelect(
            self.main_frm,
            "Folder path",
            title="Select folder with videos: ",
            lblwidth="10",
        )
        fps_entry = Entry_Box(self.main_frm, "Output FPS: ", "10", validation="numeric")
        gpu_var = BooleanVar(value=False)
        gpu_cb = Checkbutton(
            self.main_frm, text="Use GPU (reduced runtime)", variable=gpu_var
        )
        run_btn = Button(
            self.main_frm,
            text="Convert",
            command=lambda: change_fps_of_multiple_videos(
                directory=folder_path.folder_path,
                fps=fps_entry.entry_get,
                gpu=gpu_var.get(),
            ),
        )
        folder_path.grid(row=0, sticky=NW)
        fps_entry.grid(row=1, sticky=NW)
        gpu_cb.grid(row=2, sticky=NW)
        run_btn.grid(row=3, sticky=NW)


class ExtractSEQFramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(
            self, title="EXTRACT ALL FRAMES FROM SEQ FILE", size=(200, 200)
        )
        video_path = FileSelect(
            self.main_frm,
            "Video Path",
            title="Select a video file: ",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        run_btn = Button(
            self.main_frm,
            text="Extract All Frames",
            command=lambda: extract_seq_frames(video_path.file_path),
        )
        video_path.grid(row=0)
        run_btn.grid(row=1)


class MergeFrames2VideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="MERGE IMAGES TO VIDEO", size=(250, 250))
        self.folder_path = FolderSelect(
            self.main_frm, "IMAGE DIRECTORY", title="Select directory with frames: "
        )
        settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            padx=5,
            pady=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.img_format_entry_box = Entry_Box(
            settings_frm, "IMAGE FORMAT (e.g. png): ", "20"
        )
        self.bitrate_entry_box = Entry_Box(
            settings_frm, "BITRATE (e.g. 8000): ", "20", validation="numeric"
        )
        self.fps_entry = Entry_Box(settings_frm, "FPS: ", "20", validation="numeric")
        self.gpu_var = BooleanVar(value=False)
        gpu_cb = Checkbutton(
            settings_frm, text="Use GPU (decreased runtime)", variable=self.gpu_var
        )
        run_btn = Button(settings_frm, text="Create Video", command=lambda: self.run())
        settings_frm.grid(row=1, pady=10)
        self.folder_path.grid(row=0, column=0, pady=10)
        self.img_format_entry_box.grid(row=1, column=0, sticky=W)
        self.fps_entry.grid(row=2, column=0, sticky=W, pady=5)
        self.bitrate_entry_box.grid(row=3, column=0, sticky=W, pady=5)
        gpu_cb.grid(row=4, column=0, sticky=W, pady=5)
        run_btn.grid(row=5, column=1, sticky=E, pady=10)

    def run(self):
        img_format = self.img_format_entry_box.entry_get
        bitrate = self.bitrate_entry_box.entry_get
        fps = self.fps_entry.entry_get
        _ = frames_to_movie(
            directory=self.folder_path.folder_path,
            fps=fps,
            bitrate=bitrate,
            img_format=img_format,
            gpu=self.gpu_var.get(),
        )


class CreateGIFPopUP(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CREATE GIF FROM VIDEO", size=(250, 250))
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        selected_video = FileSelect(
            settings_frm,
            "Video path: ",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        start_time_entry_box = Entry_Box(
            settings_frm, "Start time (s): ", "16", validation="numeric"
        )
        duration_entry_box = Entry_Box(
            settings_frm, "Duration (s): ", "16", validation="numeric"
        )
        width_entry_box = Entry_Box(settings_frm, "Width: ", "16", validation="numeric")
        gpu_var = BooleanVar()
        gpu_cb = Checkbutton(
            settings_frm, text="Use GPU (decreased runtime)", variable=gpu_var
        )
        width_instructions_1 = Label(
            settings_frm,
            text="example Width: 240, 360, 480, 720, 1080",
            font=("Times", 10, "italic"),
        )
        width_instructions_2 = Label(
            settings_frm,
            text="Aspect ratio is kept (i.e., height is automatically computed)",
            font=("Times", 10, "italic"),
        )
        run_btn = Button(
            settings_frm,
            text="CREATE GIF",
            command=lambda: gif_creator(
                file_path=selected_video.file_path,
                start_time=start_time_entry_box.entry_get,
                duration=duration_entry_box.entry_get,
                width=width_entry_box.entry_get,
                gpu=gpu_var.get(),
            ),
        )
        settings_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW, pady=5)
        start_time_entry_box.grid(row=1, sticky=NW)
        duration_entry_box.grid(row=2, sticky=NW)
        width_entry_box.grid(row=3, sticky=NW)
        gpu_cb.grid(row=4, column=0, sticky=NW)
        width_instructions_1.grid(row=4, sticky=NW)
        width_instructions_2.grid(row=5, sticky=NW)
        run_btn.grid(row=6, sticky=NW, pady=10)


class CalculatePixelsPerMMInVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(
            self, title="CALCULATE PIXELS PER MILLIMETER IN VIDEO", size=(200, 200)
        )
        self.video_path = FileSelect(
            self.main_frm,
            "Select a video file: ",
            title="Select a video file",
            file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.known_distance = Entry_Box(
            self.main_frm, "Known length in real life (mm): ", "0", validation="numeric"
        )
        run_btn = Button(
            self.main_frm, text="GET PIXELS PER MILLIMETER", command=lambda: self.run()
        )
        self.video_path.grid(row=0, column=0, pady=10, sticky=W)
        self.known_distance.grid(row=1, column=0, pady=10, sticky=W)
        run_btn.grid(row=2, column=0, pady=10)

    def run(self):
        check_file_exist_and_readable(file_path=self.video_path.file_path)
        check_int(name="Distance", value=self.known_distance.entry_get, min_value=1)
        _ = get_video_meta_data(video_path=self.video_path.file_path)
        mm_cnt = get_coordinates_nilsson(
            self.video_path.file_path, self.known_distance.entry_get
        )
        print(
            f"1 PIXEL REPRESENTS {round(mm_cnt, 4)} MILLIMETERS IN VIDEO {os.path.basename(self.video_path.file_path)}."
        )


class ConcatenatingVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CONCATENATE VIDEOS", size=(300, 300))
        settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        video_path_1 = FileSelect(
            settings_frm,
            "First video path: ",
            title="Select a video file",
            lblwidth=15,
            file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        video_path_2 = FileSelect(
            settings_frm,
            "Second video path: ",
            title="Select a video file",
            lblwidth=15,
            file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        resolutions = ["Video 1", "Video 2", 320, 640, 720, 1280, 1980]
        resolution_dropdown = DropDownMenu(
            settings_frm, "Resolution:", resolutions, "15"
        )
        resolution_dropdown.setChoices(resolutions[0])
        use_gpu_var = BooleanVar(value=False)
        horizontal = BooleanVar(value=False)
        horizontal_radio_btn = Radiobutton(
            settings_frm,
            text="Horizontal concatenation",
            variable=horizontal,
            value=True,
        )
        use_gpu_cb = Checkbutton(
            settings_frm, text="Use GPU (reduced runtime)", variable=use_gpu_var
        )
        vertical_radio_btn = Radiobutton(
            settings_frm,
            text="Vertical concatenation",
            variable=horizontal,
            value=False,
        )
        run_btn = Button(
            self.main_frm,
            text="RUN",
            font=("Helvetica", 12, "bold"),
            command=lambda: video_concatenator(
                video_one_path=video_path_1.file_path,
                video_two_path=video_path_2.file_path,
                resolution=resolution_dropdown.getChoices(),
                horizontal=horizontal.get(),
                gpu=use_gpu_var.get(),
            ),
        )

        settings_frm.grid(row=0, column=0, sticky=NW)
        video_path_1.grid(row=0, column=0, sticky=NW)
        video_path_2.grid(row=1, column=0, sticky=NW)
        resolution_dropdown.grid(row=2, column=0, sticky=NW)
        use_gpu_cb.grid(row=3, column=0, sticky=NW)
        horizontal_radio_btn.grid(row=4, column=0, sticky=NW)
        vertical_radio_btn.grid(row=5, column=0, sticky=NW)
        run_btn.grid(row=1, column=0, sticky=NW)


class ConcatenatorPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str or None):
        PopUpMixin.__init__(self, title="MERGE (CONCATENATE) VIDEOS")
        self.config_path = config_path
        self.select_video_cnt_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="VIDEOS #",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.CONCAT_VIDEOS.value,
        )
        self.select_video_cnt_dropdown = DropDownMenu(
            self.select_video_cnt_frm, "VIDEOS #", list(range(2, 21)), "15"
        )
        self.select_video_cnt_dropdown.setChoices(2)
        self.select_video_cnt_btn = Button(
            self.select_video_cnt_frm,
            text="SELECT",
            command=lambda: self.populate_table(),
        )
        self.select_video_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_btn.grid(row=0, column=1, sticky=NW)

    def populate_table(self):
        if hasattr(self, "video_table_frm"):
            self.video_table_frm.destroy()
            self.join_type_frm.destroy()
        self.video_table_frm = LabelFrame(
            self.main_frm,
            text="VIDEO PATHS",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.video_table_frm.grid(row=1, sticky=NW)
        self.join_type_frm = LabelFrame(
            self.main_frm,
            text="JOIN TYPE",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.join_type_frm.grid(row=2, sticky=NW)
        self.videos_dict = {}
        for cnt in range(int(self.select_video_cnt_dropdown.getChoices())):
            self.videos_dict[cnt] = FileSelect(
                self.video_table_frm,
                "Video {}: ".format(str(cnt + 1)),
                title="Select a video file",
                file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
            )
            self.videos_dict[cnt].grid(row=cnt, column=0, sticky=NW)

        self.join_type_var = StringVar()
        self.icons_dict = {}
        simba_dir = os.path.dirname(simba.__file__)
        icon_assets_dir = os.path.join(simba_dir, Paths.ICON_ASSETS.value)
        concat_icon_dir = os.path.join(icon_assets_dir, "concat_icons")
        for file_cnt, file_path in enumerate(glob.glob(concat_icon_dir + "/*")):
            _, file_name, _ = get_fn_ext(file_path)
            self.icons_dict[file_name] = {}
            self.icons_dict[file_name]["img"] = ImageTk.PhotoImage(
                Image.open(file_path)
            )
            self.icons_dict[file_name]["btn"] = Radiobutton(
                self.join_type_frm,
                text=file_name,
                variable=self.join_type_var,
                value=file_name,
            )
            self.icons_dict[file_name]["btn"].config(
                image=self.icons_dict[file_name]["img"]
            )
            self.icons_dict[file_name]["btn"].image = self.icons_dict[file_name]["img"]
            self.icons_dict[file_name]["btn"].grid(row=0, column=file_cnt, sticky=NW)
        self.join_type_var.set(value="mosaic")
        self.resolution_frm = LabelFrame(
            self.main_frm,
            text="RESOLUTION",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.resolution_width = DropDownMenu(
            self.resolution_frm, "Width", ["480", "640", "1280", "1920", "2560"], "15"
        )
        self.resolution_width.setChoices("640")
        self.resolution_height = DropDownMenu(
            self.resolution_frm, "Height", ["480", "640", "1280", "1920", "2560"], "15"
        )
        self.resolution_height.setChoices("480")
        self.gpu_frm = LabelFrame(
            self.main_frm,
            text="GPU",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            self.gpu_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )
        use_gpu_cb.grid(row=0, column=0, sticky="NW")
        self.resolution_frm.grid(row=3, column=0, sticky=NW)
        self.gpu_frm.grid(row=4, column=0, sticky="NW")
        self.resolution_width.grid(row=0, column=0, sticky=NW)
        self.resolution_height.grid(row=1, column=0, sticky=NW)

        run_btn = Button(self.main_frm, text="RUN", command=lambda: self.run())
        run_btn.grid(row=5, column=0, sticky=NW)

    def run(self):
        videos_info = {}
        for cnt, (video_name, video_data) in enumerate(self.videos_dict.items()):
            _ = get_video_meta_data(video_path=video_data.file_path)
            videos_info["Video {}".format(str(cnt + 1))] = video_data.file_path

        if (len(videos_info.keys()) < 3) & (self.join_type_var.get() == "mixed_mosaic"):
            raise MixedMosaicError(
                msg="Ff using the mixed mosaic join type, please tick check-boxes for at least three video types.",
                source=self.__class__.__name__,
            )
        if (len(videos_info.keys()) < 3) & (self.join_type_var.get() == "mosaic"):
            self.join_type_var.set(value="vertical")

        _ = FrameMergererFFmpeg(
            config_path=self.config_path,
            frame_types=videos_info,
            video_height=int(self.resolution_height.getChoices()),
            video_width=int(self.resolution_width.getChoices()),
            concat_type=self.join_type_var.get(),
            gpu=self.use_gpu_var.get(),
        )


class VideoRotatorPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="ROTATE VIDEOS")
        self.save_dir_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SAVE LOCATION",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.save_dir = FolderSelect(self.save_dir_frm, "Save directory:", lblwidth=20)

        self.setting_frm = LabelFrame(
            self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.use_gpu_var = BooleanVar(value=False)
        self.use_ffmpeg_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            self.setting_frm,
            text="Use GPU (reduced runtime)",
            variable=self.use_gpu_var,
        )
        use_ffmpeg_cb = Checkbutton(
            self.setting_frm,
            text="Use FFMpeg (reduced runtime over default OpenCV)",
            variable=self.use_ffmpeg_var,
        )
        use_gpu_cb.grid(row=0, column=0, sticky=NW)
        use_ffmpeg_cb.grid(row=1, column=0, sticky=NW)

        self.rotate_dir_frm = LabelFrame(
            self.main_frm,
            text="ROTATE VIDEOS IN DIRECTORY",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.input_dir = FolderSelect(
            self.rotate_dir_frm, "Video directory:", lblwidth=20
        )
        self.run_dir = Button(
            self.rotate_dir_frm,
            text="RUN",
            fg="blue",
            command=lambda: self.run(
                input_path=self.input_dir.folder_path,
                output_path=self.save_dir.folder_path,
            ),
        )

        self.rotate_video_frm = LabelFrame(
            self.main_frm,
            text="ROTATE SINGLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.input_file = FileSelect(
            self.rotate_video_frm,
            "Video path:",
            lblwidth=20,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.run_file = Button(
            self.rotate_video_frm,
            text="RUN",
            fg="blue",
            command=lambda: self.run(
                input_path=self.input_file.file_path,
                output_path=self.save_dir.folder_path,
            ),
        )

        self.save_dir_frm.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)

        self.setting_frm.grid(row=1, column=0, sticky=NW)

        self.rotate_dir_frm.grid(row=2, column=0, sticky=NW)
        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.run_dir.grid(row=1, column=0, sticky=NW)

        self.rotate_video_frm.grid(row=3, column=0, sticky=NW)
        self.input_file.grid(row=0, column=0, sticky=NW)
        self.run_file.grid(row=1, column=0, sticky=NW)

    def run(self, input_path: str, output_path: str):
        check_if_dir_exists(in_dir=output_path)
        rotator = VideoRotator(
            input_path=input_path,
            output_dir=output_path,
            ffmpeg=self.use_ffmpeg_var.get(),
            gpu=self.use_gpu_var.get(),
        )
        rotator.run()


class VideoTemporalJoinPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="TEMPORAL JOIN VIDEOS")
        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.input_dir = FolderSelect(
            self.settings_frm, "INPUT DIRECTORY:", lblwidth=20
        )
        self.file_format = DropDownMenu(
            self.settings_frm, "INPUT FORMAT:", Options.VIDEO_FORMAT_OPTIONS.value, "20"
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            self.settings_frm,
            text="Use GPU (reduced runtime)",
            variable=self.use_gpu_var,
        )
        self.file_format.setChoices(Options.VIDEO_FORMAT_OPTIONS.value[0])
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.file_format.grid(row=1, column=0, sticky=NW)
        use_gpu_cb.grid(row=2, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_if_dir_exists(in_dir=self.input_dir.folder_path)
        print(f"Concatenating videos in {self.input_dir.folder_path} directory...")
        save_path = os.path.join(self.input_dir.folder_path, f"concatenated.mp4")
        concatenate_videos_in_folder(
            in_folder=self.input_dir.folder_path,
            save_path=save_path,
            remove_splits=False,
            video_format=self.file_format.getChoices(),
            gpu=self.use_gpu_var.get(),
        )


class ImportFrameDirectoryPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="IMPORT FRAME DIRECTORY")
        ConfigReader.__init__(self, config_path=config_path)
        self.frame_folder = FolderSelect(
            self.main_frm,
            "FRAME DIRECTORY:",
            title="Select the main directory with frame folders",
        )
        import_btn = Button(
            self.main_frm, text="IMPORT FRAMES", fg="blue", command=lambda: self.run()
        )

        self.frame_folder.grid(row=0, column=0, sticky=NW)
        import_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.frame_folder.folder_path):
            raise NotDirectoryError(
                msg=f"SIMBA ERROR: {self.frame_folder.folder_path} is not a valid directory.",
                source=self.__class__.__name__,
            )
        copy_img_folder(
            config_path=self.config_path, source=self.frame_folder.folder_path
        )


class ExtractAnnotationFramesPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(
            self, config_path=config_path, title="EXTRACT ANNOTATED FRAMES"
        )
        ConfigReader.__init__(self, config_path=config_path)
        self.create_clf_checkboxes(main_frm=self.main_frm, clfs=self.clf_names)
        self.settings_frm = LabelFrame(
            self.main_frm,
            text="STYLE SETTINGS",
            font=("Helvetica", 12, "bold"),
            pady=5,
            padx=5,
        )
        down_sample_resolution_options = ["None", "2x", "3x", "4x", "5x"]
        self.resolution_downsample_dropdown = DropDownMenu(
            self.settings_frm,
            "Down-sample images:",
            down_sample_resolution_options,
            "25",
        )
        self.resolution_downsample_dropdown.setChoices(
            down_sample_resolution_options[0]
        )
        self.settings_frm.grid(row=self.children_cnt_main(), column=0, sticky=NW)
        self.resolution_downsample_dropdown.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_filepath_list_is_empty(
            self.target_file_paths,
            error_msg=f"SIMBA ERROR: Zero files found in the {self.targets_folder} directory",
        )
        downsample_setting = self.resolution_downsample_dropdown.getChoices()
        if downsample_setting != Dtypes.NONE.value:
            downsample_setting = int("".join(filter(str.isdigit, downsample_setting)))
        clfs = []
        for clf_name, selection in self.clf_selections.items():
            if selection.get():
                clfs.append(clf_name)
        if len(clfs) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)
        settings = {"downsample": downsample_setting}

        frame_extractor = AnnotationFrameExtractor(
            config_path=self.config_path, clfs=clfs, settings=settings
        )
        frame_extractor.run()


class DownsampleVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="DOWN-SAMPLE VIDEO RESOLUTION")
        instructions = Label(
            self.main_frm,
            text="Choose only one of the following method (Custom or Default)",
        )
        choose_video_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT VIDEO",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DOWNSAMPLE.value,
        )
        self.video_path_selected = FileSelect(
            choose_video_frm,
            "Video path",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        gpu_frm = LabelFrame(
            self.main_frm,
            text="GPU",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
            padx=5,
            pady=5,
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            gpu_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var
        )
        use_gpu_cb.grid(row=0, column=0, sticky="NW")
        custom_frm = LabelFrame(
            self.main_frm,
            text="Custom resolution",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
            padx=5,
            pady=5,
        )
        self.entry_width = Entry_Box(custom_frm, "Width", "10", validation="numeric")
        self.entry_height = Entry_Box(custom_frm, "Height", "10", validation="numeric")

        self.custom_downsample_btn = Button(
            custom_frm,
            text="Downsample to custom resolution",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
            command=lambda: self.custom_downsample(),
        )
        default_frm = LabelFrame(
            self.main_frm, text="Default resolution", font="bold", padx=5, pady=5
        )
        self.radio_btns = {}
        self.var = StringVar()
        for custom_cnt, resolution_radiobtn in enumerate(self.resolutions):
            self.radio_btns[resolution_radiobtn] = Radiobutton(
                default_frm,
                text=resolution_radiobtn,
                variable=self.var,
                value=resolution_radiobtn,
            )
            self.radio_btns[resolution_radiobtn].grid(row=custom_cnt, sticky=NW)

        self.default_downsample_btn = Button(
            default_frm,
            text="Downsample to default resolution",
            command=lambda: self.default_downsample(),
        )
        instructions.grid(row=0, sticky=NW, pady=10)
        choose_video_frm.grid(row=1, column=0, sticky=NW)
        gpu_frm.grid(row=2, column=0, sticky=NW)
        self.video_path_selected.grid(row=0, column=0, sticky=NW)
        custom_frm.grid(row=3, column=0, sticky=NW)
        self.entry_width.grid(row=0, column=0, sticky=NW)
        self.entry_height.grid(row=1, column=0, sticky=NW)
        self.custom_downsample_btn.grid(row=3, column=0, sticky=NW)
        default_frm.grid(row=5, column=0, sticky=NW)
        self.default_downsample_btn.grid(
            row=len(self.resolutions) + 1, column=0, sticky=NW
        )

    def custom_downsample(self):
        width = self.entry_width.entry_get
        height = self.entry_height.entry_get
        check_int(name="Video width", value=width)
        check_int(name="Video height", value=height)
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(
            file_path=self.video_path_selected.file_path,
            video_width=int(width),
            video_height=int(height),
            gpu=self.use_gpu_var.get(),
        )

    def default_downsample(self):
        resolution = self.var.get()
        width, height = (
            resolution.split("×", 2)[0].strip(),
            resolution.split("×", 2)[1].strip(),
        )
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(
            file_path=self.video_path_selected.file_path,
            video_width=int(width),
            video_height=int(height),
            gpu=self.use_gpu_var.get(),
        )


class ConvertROIDefinitionsPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT ROI DEFINITIONS")
        settings_frm = LabelFrame(
            self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.roi_definitions_file_select = FileSelect(
            settings_frm,
            "ROI DEFINITIONS PATH (H5)",
            title="SELECT H5 FILE",
            lblwidth=20,
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.save_dir = FolderSelect(
            settings_frm, "SAVE DIRECTORY", title="SELECT H5 FILE", lblwidth=20
        )
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.roi_definitions_file_select.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_file_exist_and_readable(
            file_path=self.roi_definitions_file_select.file_path
        )
        check_if_dir_exists(in_dir=self.save_dir.folder_path)
        _ = convert_roi_definitions(
            roi_definitions_path=self.roi_definitions_file_select.file_path,
            save_dir=self.save_dir.folder_path,
        )
