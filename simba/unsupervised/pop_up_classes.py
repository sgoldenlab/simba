import os.path
from tkinter import *
from simba.tkinter_functions import hxtScrollbar, FolderSelect, DropDownMenu
from simba.enums import Formats, ReadConfig, Dtypes, Options
from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.train_model_functions import get_all_clf_names
from simba.unsupervised.visualizers import GridSearchVisualizer

class GridSearchVisualizationPopUp(object):
    def __init__(self,
                 config_path: str):

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.graph_cnt_options = list(range(1, 11))
        self.scatter_sizes_options = list(range(10, 110, 10))
        self.graph_cnt_options.insert(0, 'NONE')
        self.config, self.config_path = read_config_file(config_path), config_path
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.field_types_options = ['FRAME NUMBER', 'VIDEO NAME', 'CLASSIFIER']
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.clf_probability_names = self.clf_names + ['Probability ' + x for x in self.clf_names]
        self.main_frm.wm_title("VISUALIZATION OF GRID SEARCH")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)

        data_frm = LabelFrame(self.main_frm,text='DATA',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        self.embeddings_dir_select = FolderSelect(data_frm, "EMBEDDINGS DIRECTORY:", lblwidth=25)
        self.clusterers_dir_select = FolderSelect(data_frm, "CLUSTERERS DIRECTORY:", lblwidth=25)
        self.save_dir_select = FolderSelect(data_frm, "SAVE DIRECTORY: ", lblwidth=25)

        settings_frm = LabelFrame(self.main_frm,text='SETTINGS',font=Formats.LABELFRAME_HEADER_FORMAT.value,padx=5,pady=5,fg='black')
        self.scatter_size_dropdown = DropDownMenu(settings_frm, 'SCATTER SIZE:', self.scatter_sizes_options, '25')
        self.scatter_size_dropdown.setChoices(50)
        self.categorical_palette_dropdown = DropDownMenu(settings_frm, 'CATEGORICAL PALETTE:', Options.PALETTE_OPTIONS_CATEGORICAL.value, '25')
        self.categorical_palette_dropdown.setChoices('Set1')
        self.continuous_palette_dropdown = DropDownMenu(settings_frm, 'CONTINUOUS PALETTE:', Options.PALETTE_OPTIONS.value, '25')
        self.continuous_palette_dropdown.setChoices('magma')


        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        run_btn = self.create_btn = Button(run_frm, text='RUN', fg='blue', command=lambda: self.run())

        plot_select_cnt_frm = LabelFrame(self.main_frm, text='SELECT PLOTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
        self.plot_cnt_dropdown = DropDownMenu(plot_select_cnt_frm, '# PLOTS:', self.graph_cnt_options, '25', com= lambda x:self.show_plot_table())
        self.plot_cnt_dropdown.setChoices(self.graph_cnt_options[0])

        data_frm.grid(row=0, column=0, sticky=NW)
        self.embeddings_dir_select.grid(row=0, column=0, sticky=NW)
        self.clusterers_dir_select.grid(row=1, column=0, sticky=NW)
        self.save_dir_select.grid(row=2, column=0, sticky=NW)

        settings_frm.grid(row=1, column=0, sticky=NW)
        self.scatter_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.categorical_palette_dropdown.grid(row=1, column=0, sticky=NW)
        self.continuous_palette_dropdown.grid(row=2, column=0, sticky=NW)

        run_frm.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

        plot_select_cnt_frm.grid(row=3, column=0, sticky=NW)
        self.plot_cnt_dropdown.grid(row=0, column=0, sticky=NW)


        self.main_frm.mainloop()

    def show_plot_table(self):
        if hasattr(self, 'plot_table'):
            self.plot_table.destroy()

        if self.plot_cnt_dropdown.getChoices() != 'NONE':
            self.plot_rows = {}
            self.plot_table = LabelFrame(self.main_frm, text='SELECT PLOTS', font=Formats.LABELFRAME_HEADER_FORMAT.value, padx=5, pady=5, fg='black')
            self.plot_table.grid(row=3, column=0, sticky=NW)
            self.scatter_name_header = Label(self.plot_table, text='PLOT NAME').grid(row=0, column=0)
            self.field_type_header = Label(self.plot_table, text='FIELD TYPE').grid(row=0, column=1)
            self.field_name_header = Label(self.plot_table, text='FIELD NAME').grid(row=0, column=2)
            for idx in range(int(self.plot_cnt_dropdown.getChoices())):
                row_name = idx
                self.plot_rows[row_name] = {}
                self.plot_rows[row_name]['label'] = Label(self.plot_table, text=f'Scatter {str(idx+1)}:')
                self.plot_rows[row_name]['field_type_dropdown'] = DropDownMenu(self.plot_table, ' ', self.field_types_options, '10', com=lambda k, x=idx: self.change_field_name_state(k, x))
                self.plot_rows[row_name]['field_type_dropdown'].setChoices(self.field_types_options[0])
                self.plot_rows[row_name]['field_name_dropdown'] = DropDownMenu(self.plot_table, ' ', self.clf_probability_names, '10', com=None)
                self.plot_rows[row_name]['field_name_dropdown'].disable()
                self.plot_rows[idx]['label'].grid(row=idx+1, column=0, sticky=NW)
                self.plot_rows[idx]['field_type_dropdown'].grid(row=idx+1, column=1, sticky=NW)
                self.plot_rows[idx]['field_name_dropdown'].grid(row=idx+1, column=2, sticky=NW)

    def change_field_name_state(self, k, x):
        if k == 'CLASSIFIER PROBABILITY' or k == 'CLASSIFIER':
            self.plot_rows[x]['field_name_dropdown'].enable()
            self.plot_rows[x]['field_name_dropdown'].setChoices(self.clf_probability_names[0])
        else:
            self.plot_rows[x]['field_name_dropdown'].disable()
            self.plot_rows[x]['field_name_dropdown'].setChoices(None)

    def run(self):

        embedders_path = self.embeddings_dir_select.folder_path
        clusterers_path = self.clusterers_dir_select.folder_path
        save_dir = self.save_dir_select.folder_path

        settings = {}
        settings['SCATTER_SIZE'] = int(self.scatter_size_dropdown.getChoices())
        settings['CATEGORICAL_PALETTE'] = self.categorical_palette_dropdown.getChoices()
        settings['CONTINUOUS_PALETTE'] = self.continuous_palette_dropdown.getChoices()
        hue_dict = {}
        if self.plot_cnt_dropdown != 'NONE':
            for cnt, (k, v) in enumerate(self.plot_rows.items()):
                hue_dict[cnt] = {}
                hue_dict[cnt]['FIELD_TYPE'] = v['field_type_dropdown'].getChoices()
                hue_dict[cnt]['FIELD_NAME'] = v['field_name_dropdown'].getChoices()
        settings['HUE'] = hue_dict
        for k, v in settings['HUE'].items():
            if v['FIELD_NAME'].split(' ')[0] == 'Probability':
                v['FIELD_TYPE'] = 'CLASSIFIER_PROBABILITY'

        if not os.path.isdir(embedders_path):
            print('SIMBA ERROR: Embedding directory {} is not a valid directory.')
            raise NotADirectoryError()
        if not os.path.isdir(embedders_path):
            print('SIMBA ERROR: Save path is not a valid directory.')
            raise NotADirectoryError()


        _, GridSearchVisualizer(embedders_path=embedders_path,
                                clusterers_path=clusterers_path,
                                save_dir=save_dir,
                                settings=settings)


# _ = GridSearchVisualizationPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini')