import os
import re
import shutil
from typing import Union

import pandas as pd

from simba.utils.checks import check_if_dir_exists
from simba.utils.errors import InvalidFileTypeError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import read_df, write_df


def convert_dlc_annotation_format(input_dir: Union[str, os.PathLike],
                                  output_dir: Union[str, os.PathLike]):
    """
    Converts DeepLabCut annotation files from format without individuals row to format with individuals row.

    Takes annotation files where bodyparts include individual identifiers as suffixes (e.g., Ear_left_1, Nose_2)
    and converts them to the standard DeepLabCut multi-individual format with a separate individuals header row.
    Recursively searches for CSV files containing 'CollectedData' in the filename, preserves the folder structure,
    and copies all image files from the source folders to the output folders.

    :param Union[str, os.PathLike] input_dir: Path to directory containing source CSV annotation files (searches recursively).
    :param Union[str, os.PathLike] output_dir: Path to directory where converted CSV files and images will be saved (preserves folder structure).
    :return: None

    :example:
    >>> INPUT_DIR = r'E:\maplight_videos'
    >>> OUTPUT_DIR = r'E:\maplight_videos\converted'
    >>> convert_dlc_annotation_format(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    """

    check_if_dir_exists(in_dir=input_dir, source=convert_dlc_annotation_format.__name__, raise_error=True)
    check_if_dir_exists(in_dir=output_dir, source=convert_dlc_annotation_format.__name__, raise_error=True)

    csv_files = {}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv') and 'CollectedData' in file:
                file_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_dir)
                if rel_dir == '.':
                    rel_dir = ''
                csv_files[file_path] = rel_dir

    if len(csv_files) == 0:
        raise InvalidFileTypeError(f'No CSV files with "CollectedData" found in {input_dir} or its subdirectories',
                                   source=convert_dlc_annotation_format.__name__)

    print(f'Found {len(csv_files)} CSV file(s) with "CollectedData"')

    for file_cnt, (file_path, rel_path) in enumerate(csv_files.items()):
        print(f'Processing {file_path}...')
        video_timer = SimbaTimer(start=True)

        source_dir = os.path.dirname(file_path)
        if rel_path:
            output_subdir = os.path.join(output_dir, rel_path)
        else:
            output_subdir = output_dir
        os.makedirs(output_subdir, exist_ok=True)
        print(f'  Output directory: {output_subdir}')

        df = pd.read_csv(file_path, header=None)

        if df.shape[0] < 3:
            raise InvalidFileTypeError(
                f'File {file_path} does not have the expected format (needs at least 3 header rows)',
                source=convert_dlc_annotation_format.__name__)

        scorer_row = df.iloc[0].copy()
        bodyparts_row = df.iloc[1].copy()
        coords_row = df.iloc[2].copy()
        data_rows = df.iloc[3:].copy()

        new_scorer = ['scorer', '', '']
        individuals = ['individuals', '', '']
        new_bodyparts = ['bodyparts', '', '']
        new_coords = ['coords', '', '']

        for i in range(1, len(bodyparts_row)):
            bodypart = bodyparts_row.iloc[i]

            match = re.search(r'_(\d+)$', str(bodypart))
            if match:
                individual_id = match.group(1)
                bodypart_name = bodypart.rsplit('_', 1)[0]
                individuals.append(individual_id)
                new_bodyparts.append(bodypart_name)
            else:
                individuals.append('')
                new_bodyparts.append(bodypart)

            new_scorer.append(scorer_row.iloc[i])
            new_coords.append(coords_row.iloc[i])

        new_data_rows = []
        for idx in range(len(data_rows)):
            row = data_rows.iloc[idx]
            image_path = str(row.iloc[0])

            path_parts = image_path.replace('\\', '/').split('/')
            if len(path_parts) >= 2:
                folder = path_parts[0]
                video_name = path_parts[1]
                img_name = path_parts[-1]
            else:
                folder = 'labeled-data'
                video_name = ''
                img_name = image_path

            new_row = [folder, video_name, img_name] + list(row.iloc[1:])
            new_data_rows.append(new_row)

        all_rows = [new_scorer, individuals, new_bodyparts, new_coords] + new_data_rows
        new_df = pd.DataFrame(all_rows)

        file_name = os.path.basename(file_path)
        save_path = os.path.join(output_subdir, file_name)
        new_df.to_csv(save_path, index=False, header=False)
        print(f'  Saved converted CSV with {new_df.shape[0]} rows and {new_df.shape[1]} columns')

        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        img_count = 0
        for img_file in os.listdir(source_dir):
            if any(img_file.lower().endswith(ext) for ext in img_extensions):
                src_img = os.path.join(source_dir, img_file)
                dst_img = os.path.join(output_subdir, img_file)
                shutil.copy2(src_img, dst_img)
                img_count += 1

        video_timer.stop_timer()
        print(f'Saved converted file {save_path} and copied {img_count} image(s) (elapsed time {video_timer.elapsed_time_str}s)')

    stdout_success(msg=f'{len(csv_files.keys())} annotation file(s) converted and saved in directory {output_dir}',
                   source=convert_dlc_annotation_format.__name__)


#convert_dlc_annotation_format(input_dir=r'E:\maplight_videos\dlc_annotations', output_dir=r'E:\maplight_videos\dlc_annotations\converted')