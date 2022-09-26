__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
import cv2
import os, glob
from simba.drop_bp_cords import get_fn_ext
from pathlib import Path
from simba.misc_tools import find_video_of_file
from simba.rw_dfs import read_df
from pylab import cm

def _get_video_meta_data(video_path=None):
    vdata = {}
    cap = cv2.VideoCapture(video_path)
    _, vdata['video_name'], _ = get_fn_ext(video_path)
    vdata['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    vdata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vdata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vdata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vdata['duration_s'] = int(vdata['frame_count'] / vdata['fps'])
    vdata['resolution_str'] = str('{} x {}'.format(vdata['width'], vdata['height']))
    return vdata

def _create_color_list(no_bps):
    currColorMap = cm.get_cmap('Set1', no_bps)
    currColorList = []
    for i in range(currColorMap.N):
        rgb = list((currColorMap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        currColorList.append(rgb)
    return currColorList

def create_video_from_dir(in_directory: str,
                          out_directory: str,
                          circle_size: int):
    """
    Class for creating pose-estimation visualizations from data within a SimBA project folder.

    Parameters
    ----------
    in_directory: str
        Path to SimBA project directory containing pose-estimation data in parquet or CSV format.
    out_directory: str
        Directory to where to save the pose-estimation videos.
    circle_size: int
        Size of the circles denoting the location of the pose-estimated body-parts.

    Notes
    ----------

    Examples
    ----------
    >>> create_video_from_dir(in_directory='InputDirectory', out_directory='OutputDirectory', circle_size=5)

    """

    filesFound = glob.glob(in_directory + '/*.' + 'csv') + glob.glob(in_directory + '/*.' + 'parquet')
    print('Processing ' + str(len(filesFound)) + ' videos ...')

    for video_cnt, file_path in enumerate(filesFound):
        dir_name, file_name, ext = get_fn_ext(file_path)
        video_dir = os.path.join(Path(file_path).parents[2], 'videos')
        video_file_path = find_video_of_file(video_dir, file_name)
        pose_df = read_df(file_path, ext[-3:])
        pose_df = pose_df.apply(pd.to_numeric, errors='coerce')
        pose_df = pose_df.dropna(axis=0, how='all').reset_index(drop=True)
        bp_lst_of_lst = [list(pose_df.columns)[i:i + 3] for i in range(0, len(pose_df.columns), 3)]
        color_list = _create_color_list(len(bp_lst_of_lst))
        video_meta_data = _get_video_meta_data(video_path=video_file_path)
        cap = cv2.VideoCapture(video_file_path)
        save_path = os.path.join(out_directory, file_name + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))

        for index, row in pose_df.iterrows():
            cap.set(1, index)
            _, frame = cap.read()
            for cnt, bp in enumerate(bp_lst_of_lst):
                bp_tuple = (int(pose_df.at[index, bp[0]]), int(pose_df.at[index, bp[1]]))
                cv2.circle(frame, bp_tuple, circle_size, color_list[cnt], -1)
            writer.write(frame)
            print('Video: {} / {} Frame: {} / {}'.format(str(video_cnt + 1), str(len(filesFound)), str(index), str(len(pose_df))))
        print(file_name + ' complete...')
        cap.release()
        writer.release()

    print('All pose videos complete. located in {} directory'.format(str(out_directory)))


#create_video_from_dir(in_directory=r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\csv\input_csv', out_directory=r'Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\videos\test_2', circle_size=5)

