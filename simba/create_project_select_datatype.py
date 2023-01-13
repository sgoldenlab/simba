import tkinter
from tkinter import *
from PIL import Image, ImageTk
from simba.tkinter_functions import (DropDownMenu,
                                     hxtScrollbar,
                                     Entry_Box,
                                     FileSelect)
from simba.drop_bp_cords import bodypartConfSchematic
from simba.read_config_unit_tests import (check_file_exist_and_readable,
                                          check_str)
from simba.user_pose_config_creator import PoseConfigCreator
from simba.pose_reset import PoseResetter

class TrackingSelectorMenu(object):
    def __init__(self,
                 main_frm: tkinter.LabelFrame
                 ):

        tracking_types = ['Classic tracking',
                          'Multi tracking',
                          '3D tracking']

        self.classic_tracking_option_idx = list(range(0, 10))
        self.multi_tracking_option_idx = list(range(10, 13))

        self.main_frm = main_frm
        if hasattr(self, 'import_frm'):
            self.import_frm.destroy()

        self.import_frm = LabelFrame(self.main_frm, text='Animal settings',pady=5,padx=5)
        self.tracking_type_dropdown = DropDownMenu(self.import_frm,'Type of Tracking', tracking_types,'15', com=self.update_tracking_menus)
        self.tracking_type_dropdown.setChoices(tracking_types[0])
        self.bp_option_names, self.bp_option_photos = bodypartConfSchematic()
        self.classic_tracking_dict, self.multi_tracking_dict = {}, {}
        for i in self.classic_tracking_option_idx:
            img = Image.open(self.bp_option_photos[i])
            self.classic_tracking_dict[self.bp_option_names[i]] = ImageTk.PhotoImage(img)
        for i in self.multi_tracking_option_idx:
            self.multi_tracking_dict[self.bp_option_names[i]] = self.bp_option_photos[i]
        self.bp_config_dropdown = DropDownMenu(self.import_frm, '# config', self.classic_tracking_dict.keys(), '15', com=self.update_img)
        self.bp_config_dropdown.setChoices('2 animals, 16bps')
        self.img_frm = Label(self.main_frm, image=self.classic_tracking_dict['2 animals, 16bps'])
        self.reset_btn = Button(self.main_frm,text='Reset user-defined pose configs',command=self.reset_prompt)

        self.import_frm.grid(row=1, sticky=W)
        self.tracking_type_dropdown.grid(row=0, column=0, sticky=NW)
        self.bp_config_dropdown.grid(row=1, column=0, sticky=NW)
        self.reset_btn.grid(row=1, column=2, sticky=W)
        self.img_frm.grid(row=1, sticky=W, columnspan=2)

    def reset_prompt(self):
        reset_prompt_popup = Tk()
        reset_prompt_popup.minsize(300, 100)
        reset_prompt_popup.wm_title("Warning!")
        reset_popup_frm = LabelFrame(reset_prompt_popup)
        reset_popup_txt_lbl = Label(reset_popup_frm, text='D you want to RESET user-defined pose-configs?')
        reset_popup_txt_lbl.grid(row=0, columnspan=2)
        yes_btn = Button(reset_popup_frm, text='YES', font=("Helvetica",12,'bold'), command=lambda: PoseResetter(master=reset_prompt_popup))
        no_btn = Button(reset_popup_frm, text="NO", command=reset_prompt_popup.destroy)
        reset_popup_frm.grid(row=0, columnspan=2)
        yes_btn.grid(row=1, column=0, sticky=W)
        no_btn.grid(row=1, column=1, sticky=W)

    def update_tracking_menus(self):
        if hasattr(self, 'img_frm'):
            self.img_frm.destroy()
        self.img_frm = Label(self.main_frm)
        if self.tracking_type_dropdown.getChoices() == 'Classic tracking':
            self.bp_config_dropdown.choices = list(self.classic_tracking_dict.keys())
            self.img_frm.configure(image=self.classic_tracking_dict['2 animals, 16bps'])
        if self.tracking_type_dropdown.getChoices() == 'Multi tracking':
            self.bp_config_dropdown.choices = list(self.multi_tracking_dict.keys())
            self.img_frm.configure(image=self.classic_tracking_dict['Multi-animals, 4bps'])
        if self.tracking_type_dropdown.getChoices() == '3D tracking':
            self.bp_config_dropdown.choices = ['3D']
            self.img_frm.configure(image=self.bp_option_photos[13])

    def update_img(self):
        if self.bp_config_dropdown.getChoices() == 'Create pose config...':
            self.create_user_defined_pose_configuration()
        elif self.tracking_type_dropdown.getChoices() == 'Classic tracking':
            new_img_path = self.classic_tracking_dict[self.bp_config_dropdown.getChoices()]
            self.img_frm.configure(image=new_img_path)
        elif self.tracking_type_dropdown.getChoices() == 'Multi tracking':
            new_img_path = self.multi_tracking_dict[self.bp_config_dropdown.getChoices()]
            self.img_frm.configure(image=new_img_path)
        elif self.tracking_type_dropdown.getChoices() == '3D tracking':
            self.img_frm.configure(image=self.bp_option_photos[13])

    def create_user_defined_pose_configuration(self):
        self.user_defined_main = Toplevel()
        self.user_defined_main.minsize(400, 400)
        self.user_defined_main.wm_title("NEW POSE CONFIGURATION")
        self.user_defined_main = hxtScrollbar(self.user_defined_main)
        self.user_defined_main.pack(expand=True, fill=BOTH)

        self.config_name_entry = Entry_Box(self.user_defined_main, 'POSE CONFIG NAME','23')
        self.animal_cnt_entry = Entry_Box(self.user_defined_main, '# OF ANIMALS','23', validation='numeric')
        self.bp_cnt_entry = Entry_Box(self.user_defined_main, '# OF BODY-PARTS (per animal)','23', validation='numeric')

        self.img_path_select = FileSelect(self.user_defined_main, 'IMAGE PATH')
        confirm_btn = Button(self.user_defined_main,text='CONFIRM', command=lambda: self.create_bp_table())
        self.save_btn = Button(self.user_defined_main,text='SAVE POSE CONFIG',command=lambda:self.save_pose_config())
        self.save_btn.config(state='disabled')

        self.config_name_entry.grid(row=0,sticky=W)
        self.animal_cnt_entry.grid(row=1,sticky=W)
        self.bp_cnt_entry.grid(row=2,sticky=W)
        self.img_path_select.grid(row=3,sticky=W,pady=2)
        confirm_btn.grid(row=4,pady=5)
        self.save_btn.grid(row=6,pady=5)

    def create_bp_table(self):
        if hasattr(self, 'bp_table'):
            self.bp_table.destroy()
        self.bp_table = Frame(self.user_defined_main)
        self.bp_table = hxtScrollbar(self.bp_table)
        self.body_part_name_lbl = Label(self.bp_table, text='Body-part name', width=8)
        self.animal_id_lbl = Label(self.bp_table, text='Animal ID number', width=8)

        self.bp_names_entry_boxes, self.animal_id_entry_boxes = {}, {}
        for r in range(self.animal_cnt_entry.entry_get * self.bp_cnt_entry.entry_get):
            self.bp_names_entry_boxes[r] = Entry_Box(self.bp_table, 'Body-part {}'.format(str(r+1)), '2')
            self.bp_names_entry_boxes[r].grid(row=r, column=0)
            if self.animal_cnt_entry.entry_get > 1:
                self.animal_id_entry_boxes[r] = Entry_Box(self.bp_table, '', '2', validation='numeric')
                self.animal_id_entry_boxes[r].grid(row=r, column=1)

        self.save_btn.config(state='normal')


    def save_pose_config(self):
        check_file_exist_and_readable(file_path=self.img_path_select.file_path)
        config_name, no_animals = self.config_name_entry.entry_get, self.animal_cnt_entry.entry_get
        check_str(name='Config name', value=config_name.replace(' ', ''))

        bp_lst, animal_id_lst = [], []
        for entry in zip(self.bp_names_entry_boxes, self.animal_id_entry_boxes):
            bp_lst.append(entry[0].entry_get)
            if int(self.animal_cnt_entry.entry_get) > 1:
                animal_id_lst.append(entry[1].entry_get)

        pose_config_creator = PoseConfigCreator(pose_name=config_name,
                                                no_animals=self.animal_cnt_entry.entry_get,
                                                img_path=self.img_path_select.file_path,
                                                bp_list=bp_lst,
                                                animal_id_int_list=animal_id_lst)
        pose_config_creator.launch()
        self.user_defined_main.destroy()
        print('SIMBA COMPLETE: User-defined pose-configuration created.')





