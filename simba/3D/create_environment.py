import pyvista as pv
from stl import mesh
import os

def add_stls(self):
    if self.exp_env_shape == 'cylinder':
        data_path = os.path.join(self.stl_path, 'cylinder_1x1cm.stl')
    if self.exp_env_shape == 'rectangle':
        data_path = os.path.join(self.stl_path, 'rectanglar_space_5x5x5cm.stl')

    stl_data = mesh.Mesh.from_file(data_path)
    radius_multiplier = (self.exp_env_radius / 10)
    height_multiplier = (self.exp_env_radius / 10)

    stl_data.x *= radius_multiplier
    stl_data.y *= radius_multiplier
    stl_data.z *= height_multiplier

    self.stl_data_temp_fn = r'/Volumes/GoogleDrive/My Drive/GitHub/SimBA_DANNCE_2/stls/temp.stl'
    stl_data.save(self.stl_data_temp_fn)

    self.polydata_data = pv.PolyData(self.stl_data_temp_fn)
    self.env_obj = self.env.add_mesh(self.polydata_data, opacity=0.1, point_size=5)


