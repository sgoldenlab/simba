import copy
import os, glob
from tkinter import *
from configparser import ConfigParser
import cv2
from simba.roi_tools.ROI_image import ROI_image_class
import pandas as pd
from simba.roi_tools.ROI_move_shape import update_all_tags, move_edge
from simba.roi_tools.ROI_multiply import create_emty_df
from simba.roi_tools.ROI_size_calculations import rectangle_size_calc, circle_size_calc, polygon_size_calc
from simba.features_scripts.unit_tests import read_video_info
from simba.drop_bp_cords import get_fn_ext
from simba.features_scripts.unit_tests import read_video_info_csv


class ROI_definitions:
    """
    Class for creating ROI user-interface for drawing user-defined shapes in a video.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    video_path: str
        path to video file for which ROIs should be defined.

    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> _ = ROI_definitions(config_path='MyProjectConfig', video_path='MyVideoPath')

    """

    def __init__(self,
                 config_path: str,
                 video_path: str):

        self.config_path = config_path
        config = ConfigParser()
        config.read(config_path)
        self.video_path = video_path
        _, self.file_name, self.file_ext = get_fn_ext(self.video_path)
        self.project_path = config.get('General settings', 'project_path')
        self.roi_data_folder = os.path.join(self.project_path, 'logs', 'measures')
        if not os.path.exists(self.roi_data_folder):
            os.makedirs(self.roi_data_folder)
        self.store_fn = os.path.join(self.roi_data_folder, 'ROI_definitions.h5')
        self.video_info_path = os.path.join(self.project_path, 'logs', 'video_info.csv')
        self.video_folder_path = os.path.join(self.project_path, 'videos')
        self.other_video_paths = glob.glob(self.video_folder_path + '/*.mp4') + glob.glob(self.video_folder_path + '/*.avi')
        self.other_video_paths.remove(video_path)
        self.other_video_file_names = []
        for video in self.other_video_paths:
            self.other_video_file_names.append(os.path.basename(video))
        self.master_win_h, self.master_win_w = 800, 750

        try:
            #self.video_info_df = pd.read_csv(self.video_info_path)
            self.video_info_df = read_video_info_csv(self.video_info_path)
        except Exception as e:
            print(e.args)
            print('Could not find the video parameters file. Make sure you have defined the video parameters in the [Video parameters] tab')
        self.video_info, self.curr_px_mm, self.curr_fps = read_video_info(self.video_info_df, self.file_name)
        self.master = Tk()
        self.master.minsize(self.master_win_w, self.master_win_h)
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        self.default_top_left_x = self.screen_width - self.master_win_w
        self.master.geometry('%dx%d+%d+%d' % (self.master_win_w, self.master_win_h, self.default_top_left_x, 0))
        self.master.wm_title("Region of Interest Settings")
        self.shape_thickness_list = list(range(1, 26))
        self.ear_tag_size_list = list(range(1, 26))
        self.select_color = 'red'
        self.non_select_color = 'black'
        self.video_ROIs = ['None']
        self.c_shape = None
        self.stored_interact = None
        self.stored_shape = None
        self.img_no = 1
        self.duplicate_jump_size = 20
        self.click_sens = 10
        self.text_size = 5
        self.text_thickness = 3
        self.line_type = -1
        self.named_shape_colors = {'White': (255, 255, 255),
                                   'Grey': (220, 200, 200),
                                   'Red': (0, 0, 255),
                                   'Dark-red': (0, 0, 139),
                                   'Maroon': (0, 0, 128),
                                   'Orange': (0, 165, 255),
                                   'Dark-orange': (0, 140, 255),
                                   'Coral': (80, 127, 255),
                                   'Chocolate': (30, 105, 210),
                                   'Yellow': (0, 255, 255),
                                   'Green': (0, 128, 0),
                                   'Dark-grey': (105, 105, 105),
                                   'Light-grey': (192, 192, 192),
                                   'Pink': (178, 102, 255),
                                   'Lime': (204, 255, 229),
                                   'Purple': (255, 51, 153),
                                   'Cyan': (255, 255, 102)}
        self.window_menus()
        self.show_video_info()
        self.select_img()
        self.apply_from_other_videos_menu()
        self.select_shape()
        self.select_shape_attr()
        self.select_shape_name()
        self.interact_menus()
        self.draw_menu()
        self.save_menu()
        self.image_data = ROI_image_class(self.config_path, self.video_path, self.img_no,
                                          self.named_shape_colors, self.default_top_left_x,
                                          self.duplicate_jump_size, self.line_type, self.click_sens, self.text_size, self.text_thickness,
                                          self.master_win_h, self.master_win_w)
        self.video_frame_count = int(self.image_data.video_frame_count)
        self.get_all_ROI_names()
        if len(self.video_ROIs) > 0:
            self.update_delete_ROI_menu()

        mainloop()

    def show_video_info(self):
        self.video_info_frame = LabelFrame(self.master, text='Video information', font=("Arial", 14, "bold"), padx=5, pady=5)
        self.video_info_frame.grid_configure(ipadx=55)
        self.video_name_lbl_1 = Label(self.video_info_frame, text="Video name: ", font=("Arial", 10)).grid(row=0, column=0)
        self.video_name_lbl_2 = Label(self.video_info_frame, text=str(self.file_name), font=("Arial", 10, "bold"))

        self.video_ext_lbl_1 = Label(self.video_info_frame, text="Video format: ", font=("Arial", 10)).grid(row=0, column=2)
        self.video_ext_lbl_2 = Label(self.video_info_frame, text=str(self.file_ext), font=("Arial", 10, "bold"))

        self.video_fps_lbl_1 = Label(self.video_info_frame, text="FPS: ", font=("Arial", 10)).grid(row=0, column = 4)
        self.video_fps_lbl_2 = Label(self.video_info_frame, text=str(self.curr_fps), font=("Arial", 10, "bold"))

        self.video_frame_lbl_1 = Label(self.video_info_frame, text="Display frame #: ", font=("Arial", 10)).grid(row=0, column=6)
        self.video_frame_lbl_2 = Label(self.video_info_frame, text=str(self.img_no), font=("Arial", 10, "bold"))

        self.video_frame_time_1 = Label(self.video_info_frame,text="Display frame (s): ", font=("Arial", 10)).grid(row=0, column=8)
        self.video_frame_time_2 = Label(self.video_info_frame, text=str(round((self.img_no / self.curr_fps), 2)), font=("Arial", 10, "bold"))

        self.video_info_frame.grid(row=0, sticky=W)
        self.video_name_lbl_2.grid(row=0, column=1)
        self.video_ext_lbl_2.grid(row=0, column=3)
        self.video_fps_lbl_2.grid(row=0, column=5)
        self.video_frame_lbl_2.grid(row=0, column=7)
        self.video_frame_time_2.grid(row=0, column=9)



    def select_img(self):
        self.img_no_frame = LabelFrame(self.master, text='Change image', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.img_no_frame.grid_configure(ipadx=100)
        self.pos_1s = Button(self.img_no_frame, text='+1s', fg=self.non_select_color,command=lambda: self.set_current_image('plus'))
        self.neg_1s = Button(self.img_no_frame, text='-1s', fg=self.non_select_color, command=lambda: self.set_current_image('minus'))
        self.reset_btn = Button(self.img_no_frame, text='Reset first frame', fg=self.non_select_color, command=lambda: self.set_current_image('reset'))
        self.seconds_fw_label = Label(self.img_no_frame, text="Seconds forward: ")
        self.seconds_fw_entry = Entry(self.img_no_frame, width=4)
        self.custom_run_seconds = Button(self.img_no_frame, text='Move', fg=self.non_select_color, command=lambda: self.set_current_image('custom'))
        self.img_no_frame.grid(row=1, sticky=W)
        self.pos_1s.grid(row=1, column=0, sticky=W, pady=10, padx=10)
        self.neg_1s.grid(row=1, column=1, sticky=W, pady=10, padx=10)
        self.seconds_fw_label.grid(row=1, column=2, sticky=W, pady=10)
        self.seconds_fw_entry.grid(row=1, column=3, sticky=W, pady=10)
        self.custom_run_seconds.grid(row=1, column=4, sticky=W, pady=10)
        self.reset_btn.grid(row=1, column=5, sticky=W, pady=10, padx=10)

    def set_current_image(self, stride):
        if stride == 'plus':
            img_no = self.img_no + self.curr_fps
            if (img_no > 0) and (img_no < self.video_frame_count):
                self.img_no = img_no
                self.pos_1s.configure(fg=self.select_color)
                self.neg_1s.configure(fg=self.non_select_color)
                self.custom_run_seconds.configure(fg=self.non_select_color)

        if stride == 'minus':
            img_no = self.img_no - self.curr_fps
            if (img_no > 0) and (img_no < self.video_frame_count):
                self.img_no = img_no
                self.pos_1s.configure(fg=self.non_select_color)
                self.neg_1s.configure(fg=self.select_color)
                self.custom_run_seconds.configure(fg=self.non_select_color)

        if stride == 'reset':
            self.img_no = 1

        if stride == 'custom':
            img_no = self.img_no + int(self.curr_fps * int(self.seconds_fw_entry.get()))
            if (img_no > 0) and (img_no < self.video_frame_count):
                self.img_no = img_no
                self.pos_1s.configure(fg=self.non_select_color)
                self.neg_1s.configure(fg=self.non_select_color)
                self.custom_run_seconds.configure(fg=self.select_color)

        self.video_frame_lbl_2.config(text = str(self.img_no))
        self.video_frame_time_2.config(text = str(round((self.img_no / self.curr_fps), 2)))
        self.image_data.update_frame_no(self.img_no)

    def get_other_videos_w_data(self):
        self.other_videos_w_ROIs = []
        if os.path.isfile(self.store_fn):
            for shape_type in ['rectangles', 'circleDf', 'polygons']:
                c_df = pd.read_hdf(self.store_fn, key=shape_type)
                if len(c_df) > 0:
                    self.other_videos_w_ROIs = list(set(self.other_videos_w_ROIs + list(c_df['Video'].unique())))
        if len(self.other_videos_w_ROIs) == 0:
            self.other_videos_w_ROIs = ['None']

    def get_all_ROI_names(self):
        self.video_ROIs = []
        for shape in [self.image_data.out_rectangles, self.image_data.out_circles, self.image_data.out_polygon]:
            for e in shape:
                shape_type = e['Shape_type']
                shape_name = e['Name']
                self.video_ROIs.append(shape_type + ': ' + shape_name)



    def apply_rois_from_other_video(self):
        target_video = self.selected_other_video.get()
        if target_video != 'None':
            if os.path.isfile(self.store_fn):
                for shape_type in ['rectangles', 'circleDf', 'polygons']:
                    c_df = pd.read_hdf(self.store_fn, key=shape_type)
                    if len(c_df) > 0:
                        c_df = c_df[c_df['Video'] == target_video].reset_index(drop=True)
                        c_df['Video'] = self.file_name
                        c_df = c_df.to_dict('records')
                        if shape_type == 'rectangles':
                            for r in c_df:
                                self.image_data.out_rectangles.append(r)
                        if shape_type == 'circleDf':
                            for c in c_df:
                                self.image_data.out_circles.append(c)
                        if shape_type == 'polygons':
                            for p in c_df:
                                self.image_data.out_polygon.append(p)
                self.get_all_ROI_names()
                self.update_delete_ROI_menu()
                self.image_data.insert_all_ROIs_into_image()

    def apply_from_other_videos_menu(self):
        self.get_other_videos_w_data()
        self.apply_from_other_video = LabelFrame(self.master, text='Apply shapes from another video', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.select_video_label = Label(self.apply_from_other_video, text="Select video: ").grid(row=1, column=0)
        self.selected_other_video = StringVar()
        self.selected_other_video.set(self.other_videos_w_ROIs[0])
        self.video_dropdown = OptionMenu(self.apply_from_other_video, self.selected_other_video, *self.other_videos_w_ROIs)
        self.apply_button = Button(self.apply_from_other_video, text='Apply', fg=self.non_select_color, command=lambda: self.apply_rois_from_other_video())
        self.apply_from_other_video.grid(row=7, sticky=W)
        self.video_dropdown.grid(row=1, column=1, sticky=W, pady=10)
        self.apply_button.grid(row=1, column=3, sticky=W, pady=10)

    def select_shape(self):
        self.new_shape_frame = LabelFrame(self.master, text='New shape', font=("Arial", 16, "bold"),padx=5,pady=5, bd=5)
        self.shape_frame = LabelFrame(self.new_shape_frame, text='Shape type', font=("Arial", 14, "bold"), padx=5, pady=5)
        self.rectangle_button = Button(self.shape_frame, text='Rectangle', fg=self.non_select_color, command=lambda: self.set_current_shape('rectangle'))
        self.circle_button = Button(self.shape_frame, text='Circle', fg=self.non_select_color, command=lambda: self.set_current_shape('circle'))
        self.polygon_button = Button(self.shape_frame, text='Polygon', fg=self.non_select_color, command=lambda: self.set_current_shape('polygon'))

        self.new_shape_frame.grid(row=3, sticky=W)
        self.shape_frame.grid(row=1, sticky=W)
        self.rectangle_button.grid(row=1, sticky=W, pady=10, padx=10)
        self.circle_button.grid(row=1, column=1, sticky=W, pady=10, padx=10)
        self.polygon_button.grid(row=1, column=2, sticky=W, pady=10, padx=10)

    def select_shape_attr(self):
        self.shape_attr_frame = LabelFrame(self.new_shape_frame, text='Shape attributes', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.shape_attr_frame.grid_configure(ipadx=50)
        self.thickness_label = Label(self.shape_attr_frame, text="Shape thickness: ")
        self.color_label = Label(self.shape_attr_frame, text="Shape color: ")
        self.shape_thickness = IntVar()
        self.shape_thickness.set(5)
        self.shape_thickness_dropdown = OptionMenu(self.shape_attr_frame, self.shape_thickness, *self.shape_thickness_list, command=None)
        self.shape_thickness_dropdown.config(width=3)

        self.ear_tag_sizes_lbl = Label(self.shape_attr_frame, text="Ear tag size: ")
        self.ear_tag_size = IntVar()
        self.ear_tag_size.set(10)
        self.ear_tag_size_dropdown = OptionMenu(self.shape_attr_frame, self.ear_tag_size, *list(self.ear_tag_size_list))

        self.color_var = StringVar()
        self.color_var.set('Red')
        self.color_dropdown = OptionMenu(self.shape_attr_frame, self.color_var, *list(self.named_shape_colors.keys()))

        self.shape_attr_frame.grid(row=2, sticky=W, pady=10)
        self.thickness_label.grid(row=1, column=0)
        self.shape_thickness_dropdown.grid(row=1, column=1, sticky=W, pady=10, padx=(0,10))
        self.ear_tag_sizes_lbl.grid(row=1, column=2)
        self.ear_tag_size_dropdown.grid(row=1, column=3, sticky=W, pady=10, padx=(0,10))
        self.color_label.grid(row=1, column=4)
        self.color_dropdown.grid(row=1, column=5, sticky=W, pady=10)

    def select_shape_name(self):
        self.set_shape_name = LabelFrame(self.new_shape_frame, text='Shape name', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.set_shape_name.grid_configure(ipadx=105)
        self.name_label = Label(self.set_shape_name, text="Shape name: ").grid(row=1, column=0)
        self.name_box = Entry(self.set_shape_name, width=55)
        self.set_shape_name.grid(row=3, sticky=W, pady=10)
        self.name_box.grid(row=1, column=2, sticky=W, pady=10)

    def interact_menus(self):
        self.interact_frame = LabelFrame(self.master, text='Shape interaction', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.interact_frame.grid_configure(ipadx=30)
        self.move_shape_button = Button(self.interact_frame, text='Move shape', fg=self.non_select_color, command=lambda: self.set_interact_state('move_shape'))
        self.zoom_in_button = Button(self.interact_frame, text='Zoom IN', fg=self.non_select_color, state=DISABLED, command=lambda: self.set_interact_state('zoom_in'))
        self.zoom_out_button = Button(self.interact_frame, text='Zoom OUT', fg=self.non_select_color, state=DISABLED, command=lambda: self.set_interact_state('zoom_out'))
        self.zoom_home = Button(self.interact_frame, text='Zoom HOME', fg=self.non_select_color, state=DISABLED, command=lambda: self.set_interact_state('zoom_home'))
        self.zoom_pct_label = Label(self.interact_frame, text="Zoom %: ").grid(row=1, column=5, padx=(10,0))
        self.zoom_pct = Entry(self.interact_frame, width=4, state=DISABLED)
        self.zoom_pct.insert(0, 10)
        self.pan = Button(self.interact_frame, text='Pan', fg=self.non_select_color, state=DISABLED,command=lambda: self.set_interact_state('pan'))
        self.shape_info_btn = Button(self.interact_frame, text='Show shape info.', fg=self.non_select_color, command=lambda: self.show_shape_information())

        self.interact_frame.grid(row=6, sticky=W)
        self.move_shape_button.grid(row=1, column=0, sticky=W, pady=10, padx=10)
        self.pan.grid(row=1, column=1, sticky=W, pady=10, padx=10)
        self.zoom_in_button.grid(row=1, column=2, sticky=W, pady=10, padx=10)
        self.zoom_out_button.grid(row=1, column=3, sticky=W, pady=10, padx=10)
        self.zoom_home.grid(row=1, column=4, sticky=W, pady=10, padx=10)
        self.zoom_pct.grid(row=1, column=6, sticky=W, pady=10)
        self.shape_info_btn.grid(row=1, column=7, sticky=W, pady=10)

    def call_remove_ROI(self):
        self.shape_info_btn.configure(text='Show shape info.')
        self.apply_delete_button.configure(fg=self.select_color)
        self.image_data.remove_ROI(self.selected_video.get())
        self.video_ROIs.remove(self.selected_video.get())
        if len(self.video_ROIs) == 0:
            self.video_ROIs = ['None']
        self.selected_video.set(self.video_ROIs[0])
        self.update_delete_ROI_menu()

    def draw_menu(self):
        self.draw_frame = LabelFrame(self.master, text='Draw', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.draw_button = Button(self.draw_frame, text='Draw', fg=self.non_select_color,command=lambda: self.create_draw())
        self.delete_all_rois_btn = Button(self.draw_frame, text='Delete ALL', fg=self.non_select_color, command=lambda: self.call_delete_all_rois())
        self.select_roi_label = Label(self.draw_frame, text="Select ROI: ")
        self.selected_video = StringVar()
        self.selected_video.set(self.video_ROIs[0])
        self.roi_dropdown = OptionMenu(self.draw_frame, self.selected_video, *self.video_ROIs)
        self.apply_delete_button = Button(self.draw_frame, text='Delete ROI', fg=self.non_select_color, command=lambda: self.call_remove_ROI())
        self.duplicate_ROI_btn = Button(self.draw_frame, text='Duplicate ROI', fg=self.non_select_color, command=lambda: self.call_duplicate_ROI())
        self.chg_attr_btn = Button(self.draw_frame, text='Change ROI', fg=self.non_select_color, command=lambda: self.ChangeAttrMenu(self, self.image_data))

        self.draw_frame.grid(row=5, sticky=W)
        self.draw_button.grid(row=1, column=1, sticky=W, pady=2, padx=10)
        self.delete_all_rois_btn.grid(row=1, column=2, sticky=W, pady=2, padx=10)
        self.select_roi_label.grid(row=1, column=3, sticky=W, pady=2, padx=(10, 0))
        self.roi_dropdown.grid(row=1, column=4, sticky=W, pady=2, padx=(0, 10))
        self.apply_delete_button.grid(row=1, column=5, sticky=W, pady=2, padx=10)
        self.duplicate_ROI_btn.grid(row=1, column=6, sticky=W, pady=2, padx=10)
        self.chg_attr_btn.grid(row=1, column=7, sticky=W, pady=2, padx=10)

    def show_shape_information(self):
        if (len(self.image_data.out_rectangles) + len(self.image_data.out_circles) + len(
                self.image_data.out_polygon) == 0):
            print('No shapes to print info for.')

        elif self.shape_info_btn.cget('text') == 'Show shape info.':
            if len(self.image_data.out_rectangles) > 0:
                self.rectangle_size_dict = {}
                self.rectangle_size_dict['Rectangles'] = {}
                for rectangle in self.image_data.out_rectangles:
                    self.rectangle_size_dict['Rectangles'][rectangle['Name']] = rectangle_size_calc(rectangle, self.curr_px_mm)
                self.image_data.rectangle_size_dict = self.rectangle_size_dict

            if len(self.image_data.out_circles) > 0:
                self.circle_size_dict = {}
                self.circle_size_dict['Circles'] = {}
                for circle in self.image_data.out_circles:
                    self.circle_size_dict['Circles'][circle['Name']] = circle_size_calc(circle, self.curr_px_mm)
                self.image_data.circle_size_dict = self.circle_size_dict

            if len(self.image_data.out_polygon) > 0:
                self.polygon_size_dict = {}
                self.polygon_size_dict['Polygons'] = {}
                for polygon in self.image_data.out_polygon:
                    self.polygon_size_dict['Polygons'][polygon['Name']] = polygon_size_calc(polygon, self.curr_px_mm)
                self.image_data.polygon_size_dict = self.polygon_size_dict

            self.image_data.insert_all_ROIs_into_image(show_size_info=True)
            self.shape_info_btn.configure(text='Hide shape info.')

        elif self.shape_info_btn.cget('text') == 'Hide shape info.':
            self.shape_info_btn.configure(text='Show shape info.')
            self.image_data.insert_all_ROIs_into_image()

    def save_menu(self):
        self.save_frame = LabelFrame(self.master, text='Save', font=("Arial", 16, "bold"), padx=5, pady=5)
        self.save_button = Button(self.save_frame, text='Save ROI data', fg=self.non_select_color, command=lambda: self.save_data())
        self.save_frame.grid(row=8, sticky=W)
        self.save_button.grid(row=1, column=3, sticky=W, pady=10)

    def set_current_shape(self, c_shape):
        self.c_shape = c_shape
        self.shape_info_btn.configure(text='Show shape info.')
        if self.c_shape == self.stored_shape:
            self.rectangle_button.configure(fg=self.non_select_color)
            self.circle_button.configure(fg=self.non_select_color)
            self.polygon_button.configure(fg=self.non_select_color)
            self.stored_shape = None
        else:
            if c_shape == 'rectangle':
                self.rectangle_button.configure(fg=self.select_color)
                self.circle_button.configure(fg=self.non_select_color)
                self.polygon_button.configure(fg=self.non_select_color)

            if c_shape == 'circle':
                self.rectangle_button.configure(fg=self.non_select_color)
                self.circle_button.configure(fg=self.select_color)
                self.polygon_button.configure(fg=self.non_select_color)

            if c_shape == 'polygon':
                self.rectangle_button.configure(fg=self.non_select_color)
                self.circle_button.configure(fg=self.non_select_color)
                self.polygon_button.configure(fg=self.select_color)
            self.stored_shape = c_shape

    def reset_selected_buttons(self, category):
        if category == 'interact' or 'all':
            self.move_shape_button.configure(fg=self.non_select_color)
            self.zoom_in_button.configure(fg=self.non_select_color)
            self.zoom_out_button.configure(fg=self.non_select_color)
            self.pan.configure(fg=self.non_select_color)
            self.zoom_home.configure(fg=self.non_select_color)
            self.stored_interact = None

    def set_interact_state(self, c_interact):
        self.shape_info_btn.configure(text = 'Show shape info.')
        if c_interact == self.stored_interact:
            self.move_shape_button.configure(fg=self.non_select_color)
            self.zoom_in_button.configure(fg=self.non_select_color)
            self.zoom_out_button.configure(fg=self.non_select_color)
            self.pan.configure(fg=self.non_select_color)
            self.zoom_home.configure(fg=self.non_select_color)
            self.stored_interact = None

        else:
            if c_interact == 'move_shape':
                if self.image_data.no_shapes > 0:
                    self.move_shape_button.configure(fg=self.select_color)
                    self.zoom_in_button.configure(fg=self.non_select_color)
                    self.zoom_out_button.configure(fg=self.non_select_color)
                    self.zoom_home.configure(fg=self.non_select_color)
                    self.pan.configure(fg=self.non_select_color)
                else:
                    self.reset_selected_buttons('interact')
                    c_interact = None
                    print('You have no shapes that can be moved.')

            if c_interact == 'zoom_in':
                self.move_shape_button.configure(fg=self.non_select_color)
                self.zoom_in_button.configure(fg=self.select_color)
                self.zoom_out_button.configure(fg=self.non_select_color)
                self.zoom_home.configure(fg=self.non_select_color)
                self.pan.configure(fg=self.non_select_color)

            if c_interact == 'zoom_out':
                self.move_shape_button.configure(fg=self.non_select_color)
                self.zoom_in_button.configure(fg=self.non_select_color)
                self.zoom_out_button.configure(fg=self.select_color)
                self.zoom_home.configure(fg=self.non_select_color)
                self.pan.configure(fg=self.non_select_color)

            if c_interact == 'zoom_home':
                self.move_shape_button.configure(fg=self.non_select_color)
                self.zoom_in_button.configure(fg=self.non_select_color)
                self.zoom_out_button.configure(fg=self.non_select_color)
                self.zoom_home.configure(fg=self.select_color)
                self.pan.configure(fg=self.non_select_color)

            if c_interact == 'pan':
                self.move_shape_button.configure(fg=self.non_select_color)
                self.zoom_in_button.configure(fg=self.non_select_color)
                self.zoom_out_button.configure(fg=self.non_select_color)
                self.zoom_home.configure(fg=self.non_select_color)
                self.pan.configure(fg=self.select_color)
            self.stored_interact = c_interact
        self.image_data.interact_functions(self.stored_interact, zoom_val=0)
        self.reset_selected_buttons('interact')

    def call_delete_all_rois(self):
        self.shape_info_btn.configure(text='Show shape info.')
        if len(self.image_data.out_rectangles) + len(self.image_data.out_circles) + len(
                self.image_data.out_polygon) == 0:
            print('SimBA finds no ROIs to delete.')
        else:
            self.image_data.out_rectangles = []
            self.image_data.out_circles = []
            self.image_data.out_polygon = []
            self.video_ROIs = ['None']
            self.selected_video.set(self.video_ROIs[0])
            self.update_delete_ROI_menu()
            self.image_data.insert_all_ROIs_into_image()

    def get_duplicate_shape_name(self):
        c_no = 1
        while True:
            self.new_name = self.current_shape_data['Name'] + '_copy_' + str(c_no)
            if str(self.shape_type)  + ': ' + self.new_name in self.video_ROIs:
                c_no += 1
            else:
                self.new_shape_data['Name'] = str(self.shape_type)  + ': ' + self.new_name
                break

    def get_duplicate_coords(self):
        if self.shape_type == 'Rectangle':
            self.new_shape_x = int(self.current_shape_data['topLeftX'] + self.duplicate_jump_size)
            self.new_shape_y = int(self.current_shape_data['topLeftY'] + self.duplicate_jump_size)
            for shape in self.image_data.out_rectangles:
                if (shape['topLeftX'] == self.new_shape_x) and (shape['topLeftY'] == self.new_shape_y):
                    self.new_shape_x += self.duplicate_jump_size
                    self.new_shape_y += self.duplicate_jump_size
        if self.shape_type == 'Circle':
            self.new_shape_x = int(self.current_shape_data['centerX'] + self.duplicate_jump_size)
            self.new_shape_y = int(self.current_shape_data['centerY'] + self.duplicate_jump_size)
            for shape in self.image_data.out_circles:
                if (shape['centerY'] == self.new_shape_x) and (shape['centerY'] == self.new_shape_y):
                    self.new_shape_x += self.duplicate_jump_size
                    self.new_shape_y += self.duplicate_jump_size
        if self.shape_type == 'Polygon':
            self.new_shape_x = int(self.current_shape_data['centerY'] + self.duplicate_jump_size)
            self.new_shape_y = int(self.current_shape_data['centerY'] + self.duplicate_jump_size)
            for shape in self.image_data.out_polygon:
                if (shape['Center_X'] == self.new_shape_x) and (shape['centerY'] == self.new_shape_y):
                    self.new_shape_x += self.duplicate_jump_size
                    self.new_shape_y += self.duplicate_jump_size

    def call_duplicate_ROI(self):
        shape_name = self.selected_video.get().split(': ')
        self.shape_info_btn.configure(text='Show shape info.')
        if shape_name[0] != 'None':
            all_roi_list = self.image_data.out_rectangles + self.image_data.out_circles + self.image_data.out_polygon
            self.shape_type, shape_name = shape_name[0], shape_name[1]
            self.current_shape_data = [d for d in all_roi_list if d.get('Name') == shape_name][0]
            self.new_shape_data = copy.deepcopy(self.current_shape_data)
            self.get_duplicate_shape_name()
            self.get_duplicate_coords()
            if self.shape_type == 'Rectangle':
                self.new_shape_data['topLeftX'] = self.new_shape_x
                self.new_shape_data['topLeftY'] = self.new_shape_y
                self.new_shape_data['Name'] = self.new_shape_data['Name'].split("Rectangle: ", 1)[-1]
                update_all_tags(self.new_shape_data)
                self.image_data.out_rectangles.append(self.new_shape_data)

            if self.shape_type == 'Circle':
                self.new_shape_data['centerX'] = self.new_shape_x
                self.new_shape_data['centerY'] = self.new_shape_y
                self.new_shape_data['Name'] = self.new_shape_data['Name'].split("Circle: ", 1)[-1]
                update_all_tags(self.new_shape_data)
                self.image_data.out_circles.append(self.new_shape_data)

            if self.shape_type == 'Polygon':
                move_edge(self.new_shape_data, 'Center_tag', (self.new_shape_x, self.new_shape_y))
                self.new_shape_data['Name'] = self.new_shape_data['Name'].split("Polygon: ", 1)[-1]
                self.image_data.out_polygon.append(self.new_shape_data)


            self.video_ROIs.append(self.shape_type + ': ' + self.new_shape_data['Name'])
            self.image_data.insert_all_ROIs_into_image()

            self.update_delete_ROI_menu()
        else:
            print('No ROI selected.')

    def create_draw(self):
        self.shape_info_btn.configure(text='Show shape info.')
        if self.stored_shape is None:
            raise TypeError('No shape type selected.')
        if not self.name_box.get():
            raise TypeError('No shape name selected.')
        if not self.name_box.get().strip():
            raise TypeError('Shape name contains only spaces.')

        c_draw_settings = {'Video_name': self.file_name,
                            'Shape_type': self.stored_shape,
                            'Name': self.name_box.get(),
                            'Shape_thickness': self.shape_thickness.get(),
                            'Shape_ear_tag_size': self.ear_tag_size.get(),
                            'Shape_color_name': self.color_var.get(),
                            'Shape_color_BGR': self.named_shape_colors[self.color_var.get()]}

        self.video_ROIs = self.image_data.initiate_draw(c_draw_settings)
        self.update_delete_ROI_menu()

    def update_delete_ROI_menu(self):
        self.selected_video.set(self.video_ROIs[0])
        self.roi_dropdown = OptionMenu(self.draw_frame, self.selected_video, *self.video_ROIs)
        self.roi_dropdown.grid(row=1, column=4, sticky=W, pady=10)

    def save_data(self):
        if os.path.isfile(self.store_fn):
            rectangles_found = pd.read_hdf(self.store_fn, key='rectangles')
            circles_found = pd.read_hdf(self.store_fn, key='circleDf')
            polygons_found = pd.read_hdf(self.store_fn, key='polygons')
            other_vid_rectangles = rectangles_found[rectangles_found['Video'] != self.file_name]
            other_vid_circles = circles_found[circles_found['Video'] != self.file_name]
            other_vid_polygons = polygons_found[polygons_found['Video'] != self.file_name]

            new_rectangles = pd.DataFrame.from_dict(self.image_data.out_rectangles)
            new_circles = pd.DataFrame.from_dict(self.image_data.out_circles)
            new_polygons = pd.DataFrame.from_dict(self.image_data.out_polygon)

            if len(new_rectangles) > 0:
                out_rectangles = pd.concat([other_vid_rectangles, new_rectangles], axis=0).sort_values(by=['Video']).reset_index(drop=True)
            else:
                out_rectangles = other_vid_rectangles.sort_values(by=['Video']).reset_index(drop=True)

            if len(new_circles) > 0:
                out_circles = pd.concat([other_vid_circles, new_circles], axis=0).sort_values(by=['Video']).reset_index(drop=True)
            else:
                out_circles = other_vid_circles.sort_values(by=['Video']).reset_index(drop=True)

            if len(new_polygons) > 0:
                out_polygons = pd.concat([other_vid_polygons, new_polygons], axis=0).sort_values(by=['Video']).reset_index(drop=True)
            else:
                out_polygons = other_vid_polygons.sort_values(by=['Video']).reset_index(drop=True)

        else:
            out_rectangles = pd.DataFrame.from_dict(self.image_data.out_rectangles)
            out_circles = pd.DataFrame.from_dict(self.image_data.out_circles)
            out_polygons = pd.DataFrame.from_dict(self.image_data.out_polygon)

            if len(out_rectangles) == 0:
                out_rectangles = create_emty_df('rectangles')
            if len(out_circles) == 0:
                out_circles = create_emty_df('circleDf')
            if len(out_polygons) == 0:
                out_polygons = create_emty_df('polygons')


        store = pd.HDFStore(self.store_fn, mode='w')
        store['rectangles'] = out_rectangles
        store['circleDf'] = out_circles
        store['polygons'] = out_polygons
        store.close()
        print('ROI definitions saved for video: ' + str(self.file_name))

    class ChangeAttrMenu:
        def __init__(self, shape_data, image_data):
            shape_name = shape_data.selected_video.get().split(': ')
            if shape_name[0] != 'None':
                self.all_roi_list = shape_data.image_data.out_rectangles + shape_data.image_data.out_circles + shape_data.image_data.out_polygon
                self.shape_type, self.shape_name = shape_name[0], shape_name[1]
                current_shape_data = [d for d in self.all_roi_list if d.get('Name') == self.shape_name][0]
                self.attr_win = Toplevel()
                self.attr_win.minsize(400, 300)
                self.attr_win.wm_title("Selected Shape Attributes")
                attr_lbl_frame = LabelFrame(self.attr_win, text='Attributes', font=("Arial", 16, 'bold'), pady=5, padx=5,fg='black')
                selected_shape_name_lbl = Label(self.attr_win, text="Shape name: ")
                self.selected_shape_name_entry_txt = StringVar()
                self.selected_shape_name_entry_txt.set(current_shape_data['Name'])

                selected_shape_name_entry = Entry(self.attr_win, width=25, textvariable=self.selected_shape_name_entry_txt)
                selected_shape_thickness_lbl = Label(self.attr_win, text="Shape thickness: ")
                self.selected_shape_thickness = IntVar()
                self.selected_shape_thickness.set(current_shape_data['Thickness'])
                selected_shape_thickness_dropdown = OptionMenu(self.attr_win, self.selected_shape_thickness, *list(shape_data.shape_thickness_list))

                selected_shape_eartag_size_lbl = Label(self.attr_win, text="Ear tag size: ")
                self.selected_shape_eartag_size = IntVar()
                self.selected_shape_eartag_size.set(current_shape_data['Ear_tag_size'])
                selected_shape_eartag_size_dropdown = OptionMenu(self.attr_win, self.selected_shape_eartag_size,*list(shape_data.ear_tag_size_list))

                selected_shape_color_lbl = Label(self.attr_win, text="Shape color: ")
                self.selected_shape_color = StringVar()
                self.selected_shape_color.set(current_shape_data['Color name'])
                selected_shape_color_dropdown = OptionMenu(self.attr_win, self.selected_shape_color, *list(shape_data.named_shape_colors.keys()))

                save_button = Button(self.attr_win, text='Save', fg=shape_data.non_select_color, command=lambda: self.save_attr_changes(shape_data, image_data))

                attr_lbl_frame.grid(row=1, sticky=W)
                selected_shape_name_lbl.grid(row=1, column=0, sticky=W, pady=10)
                selected_shape_name_entry.grid(row=1, column=1, sticky=W, pady=10)
                selected_shape_thickness_lbl.grid(row=2, column=0, sticky=W, pady=10)
                selected_shape_thickness_dropdown.grid(row=2, column=1, sticky=W, pady=10)
                selected_shape_eartag_size_lbl.grid(row=3, column=0, sticky=W, pady=10)
                selected_shape_eartag_size_dropdown.grid(row=3, column=1, sticky=W, pady=10)
                selected_shape_color_lbl.grid(row=4, column=0, sticky=W, pady=10)
                selected_shape_color_dropdown.grid(row=4, column=1, sticky=W, pady=10)
                save_button.grid(row=5, column=0, sticky=W, pady=10)

            else:
                raise TypeError('No ROI selected.')

        def save_attr_changes(self, shape_data, image_data):
            new_shape_name = self.selected_shape_name_entry_txt.get()
            new_shape_thickness = self.selected_shape_thickness.get()
            new_shape_ear_tag_size = self.selected_shape_eartag_size.get()
            new_shape_color = self.selected_shape_color.get()
            for shape in [image_data.out_rectangles, image_data.out_circles, image_data.out_polygon]:
                for e in shape:
                    shape_type = e['Shape_type']
                    if e['Name'] == self.shape_name:
                        e['Name'] = new_shape_name
                        e['Thickness'] = new_shape_thickness
                        e['Ear_tag_size'] = new_shape_ear_tag_size
                        e['Color name'] = new_shape_color
                        e['Color BGR'] = shape_data.named_shape_colors[new_shape_color]
                        shape_data.video_ROIs = [w.replace(str(shape_type) + ': ' + self.shape_name, str(shape_type) + ': ' + new_shape_name) for w in shape_data.video_ROIs]
            image_data.insert_all_ROIs_into_image()
            shape_data.update_delete_ROI_menu()
            self.attr_win.destroy()
            self.attr_win.update()


    def window_menus(self):
        menu = Menu(self.master)
        file_menu = Menu(menu)
        menu.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Preferences...', command=lambda:PreferenceMenu(self.image_data))
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.Exit)
        self.master.config(menu=menu)

    def Exit(self):
        cv2.destroyAllWindows()
        self.image_data.destroy_windows()
        self.master.destroy()

class PreferenceMenu:
    def __init__(self, image_data):
        pref_win = Toplevel()
        pref_win.minsize(400, 300)
        pref_win.wm_title("Preference Settings")
        pref_lbl_frame = LabelFrame(pref_win, text='Preferences', font=("Arial", 16, 'bold'), pady=5, padx=5, fg='black')
        line_type_label = Label(pref_lbl_frame, text="Shape line type: ")
        text_size_label = Label(pref_lbl_frame, text="Text size: ")
        text_thickness_label = Label(pref_lbl_frame, text="Text thickness: ")
        line_type_list = [4, 8, 16, -1]
        text_size_list = list(range(1, 20))
        text_thickness_list = list(range(1, 15))
        click_sensitivity_lbl = Label(pref_lbl_frame, text="Mouse click sensitivity: ")
        click_sensitivity_list = list(range(1, 50, 5))
        self.click_sens = IntVar()
        self.line_type = IntVar()
        self.text_size = IntVar()
        self.text_thickness = IntVar()
        self.line_type.set(line_type_list[-1])
        self.text_size.set(text_size_list[0])
        self.click_sens.set(click_sensitivity_list[0])
        line_type_dropdown = OptionMenu(pref_lbl_frame, self.line_type, *line_type_list)
        text_thickness_dropdown = OptionMenu(pref_lbl_frame, self.text_thickness, *text_thickness_list)
        text_size_dropdown = OptionMenu(pref_lbl_frame, self.text_size, *text_size_list)
        click_sens_dropdown = OptionMenu(pref_lbl_frame, self.click_sens, *click_sensitivity_list)
        duplicate_jump_size_lbl = Label(pref_lbl_frame, text="Duplicate shape jump: ")
        duplicate_jump_size_list = list(range(1, 100, 5))
        self.duplicate_jump_size = IntVar()
        self.duplicate_jump_size.set(20)
        duplicate_jump_size_dropdown = OptionMenu(pref_lbl_frame, self.duplicate_jump_size, *duplicate_jump_size_list)
        pref_save_btn = Button(pref_lbl_frame, text='Save', fg='black', command=lambda: self.save_prefs(image_data))

        pref_lbl_frame.grid(row=1, sticky=W)
        line_type_label.grid(row=1, column=0, sticky=W, pady=10)
        line_type_dropdown.grid(row=1, column=1, sticky=W, pady=10)
        click_sensitivity_lbl.grid(row=2, column=0, sticky=W, pady=10)
        click_sens_dropdown.grid(row=2, column=1, sticky=W, pady=10)
        duplicate_jump_size_lbl.grid(row=3, column=0, sticky=W, pady=10)
        duplicate_jump_size_dropdown.grid(row=3, column=1, sticky=W, pady=10)
        text_size_label.grid(row=4, column=0, sticky=W, pady=10)
        text_size_dropdown.grid(row=4, column=1, sticky=W, pady=10)
        text_thickness_label.grid(row=5, column=0, sticky=W, pady=10)
        text_thickness_dropdown.grid(row=5, column=1, sticky=W, pady=10)
        pref_save_btn.grid(row=5, column=2, sticky=W, pady=10)

    def save_prefs(self, image_data):
        image_data.click_sens = self.click_sens.get()
        image_data.text_size = self.text_size.get()
        image_data.text_thickness = self.text_thickness.get()
        image_data.line_type = self.line_type.get()
        image_data.duplicate_jump_size = self.duplicate_jump_size.get()
        print('Saved preference settings.')



#test = ROI_definitions(config_path='/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/project_config.ini', video_path='/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/videos/SI_DAY3_308_CD1_PRESENT_2.mp4')


# class ROI_definitions:
#     def __init__(self, config_path, video_path):