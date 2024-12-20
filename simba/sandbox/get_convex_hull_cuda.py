from simba.data_processors.cuda.geometry import get_convex_hull
from simba.utils.read_write import read_df

video_path = r"C:/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0514.mp4"
data_path = r"C:/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0514 - test.csv"
df = read_df(file_path=data_path, file_type='csv')
frame_data = df.values.reshape(len(df), -1, 2)
x = get_convex_hull(frame_data)