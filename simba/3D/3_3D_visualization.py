import pyvista as pv
import pandas as pd
import numpy as np
from simba.rw_dfs import *
from simba.drop_bp_cords import *
from configparser import ConfigParser, NoOptionError, NoSectionError
from create_environment import add_stls
from simba.drop_bp_cords import get_fn_ext

INI_PATH = '/Volumes/GoogleDrive/My Drive/GitHub/SimBA_DANNCE/project_config.ini'
DATA_PATH = '/Volumes/GoogleDrive/My Drive/GitHub/SimBA_DANNCE/project_folder/csv/features_extracted/DANNCE_import.csv'
# EXPERIMENT_ENVIRONMENT_SHAPE = 'cylinder'
# EXPERIMENT_ENVIRONMENT_RADIUS = 350
# EXPERIMENT_ENVIRONMENT_HEIGHT = 100

EXPERIMENT_ENVIRONMENT_SHAPE = 'rectangle'
EXPERIMENT_ENVIRONMENT_RADIUS = 50
EXPERIMENT_ENVIRONMENT_HEIGHT = 50

class ThreeDViz:
    def __init__(self, CONFIG_PATH, DATA_PATH, EXPERIMENT_ENVIRONMENT_SHAPE, EXPERIMENT_ENVIRONMENT_RADIUS, EXPERIMENT_ENVIRONMENT_HEIGHT):
        self.exp_env_shape = EXPERIMENT_ENVIRONMENT_SHAPE
        self.exp_env_radius = EXPERIMENT_ENVIRONMENT_RADIUS
        self.exp_env_height = EXPERIMENT_ENVIRONMENT_HEIGHT
        self.data_path_fn = DATA_PATH
        _, self.file_name, self.file_ext = get_fn_ext(self.data_path_fn)
        self.out_file_name = self.file_name + '.mp4'
        self.config = ConfigParser()
        self.config.read(str(CONFIG_PATH))
        self.stl_path = r'/Volumes/GoogleDrive/My Drive/GitHub/SimBA_DANNCE_2/stls'
        Xcols, Ycols, Zcols, Pcols = getBpNames(CONFIG_PATH, three_d_data=True)
        self.animal_dict = create_body_part_dictionary(False, ['Animal_1'], 1, Xcols, Ycols, Pcols, [], Zcols, three_d_data=True)
        self.in_df = pd.read_csv(DATA_PATH, index_col=0)
        self.frame_no = len(self.in_df)
        self.bp_df = self.in_df.iloc[:, : len(Xcols) * 4]
        self.bp_df.drop(Pcols, axis=1, inplace=True)
        self.env = pv.Plotter(lighting='light_kit', polygon_smoothing=True, off_screen=True)
        self.env.set_background("black", top="skyblue")
        self.env.enable_eye_dome_lighting()
        add_stls(self)

    def create_movie(self):
        self.env.open_movie(self.out_file_name, framerate=30)
        for index, row in self.bp_df.iterrows():
            animal_data = np.array(list(row)).reshape(-1, 3).astype('float32')
            if index == 0:
                self.env.add_mesh(animal_data, color='salmon', render_points_as_spheres=True, point_size=10, cmap='Accent', ambient=0.5, categories=True, name='Animal_1')
            else:
                self.env.update_coordinates(animal_data, render=False)
            self.env.show(auto_close=False)
            self.env.write_frame()
            print('Frame ' + str(index) + ' / ' + str(self.frame_no))
        self.env.close()
        print(str(self.out_file_name) + ' complete.')

viz = ThreeDViz(INI_PATH, DATA_PATH, EXPERIMENT_ENVIRONMENT_SHAPE, EXPERIMENT_ENVIRONMENT_RADIUS, EXPERIMENT_ENVIRONMENT_HEIGHT)
viz.create_movie()





#
#
#
#
#
#
#
#
#
#
#
#
# floor_act = env.add_mesh(floor_mesh, opacity=0.5, point_size=5)
#
# for index, row in bp_df.iterrows():
#     animal_data = np.array(list(row)).reshape(-1, 3).astype('float32')
#     if index == 0:
#         animal_mesh = env.add_mesh(animal_data, color='salmon', render_points_as_spheres=True, point_size=10, cmap='Accent', ambient=0.5, categories=True, name='Animal_1')
#     else:
#         env.update_coordinates(animal_data, render=False)
#     env.show(auto_close=False)
#     env.write_frame()
#
# env.close()






#
#
# camera_positon = env.camera_position[0][1]
#     if (camera_positon <= pos_max[1]) or (camera_positon < 1):
#         camera_positon += 1
#     else:
#         camera_positon -= 1
#     env.camera_position = [env.camera_position[0], (camera_positon, env.camera_position[1][1], env.camera_position[1][2]), env.camera_position[2]]
#     #env.camera_set = True
#
# camera_position = tuple(path[index])
# env.camera_position = [camera_position, (camera_positon, env.camera_position[1][1], env.camera_position[1][2]),
#                        env.camera_position[2]]
#
# path = env.generate_orbital_path(n_points=len(bp_df), shift=5).points

