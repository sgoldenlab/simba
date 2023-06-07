import pandas as pd
import pytest
import json

from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import get_fn_ext, read_df, read_config_file


@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def project_config_args(request):
    return request

@pytest.fixture(params=['tests/data/test_projects/two_c57/project_folder/project_config.ini',
                        'tests/data/test_projects/mouse_open_field/project_folder/project_config.ini',
                        'tests/data/test_projects/zebrafish/project_folder/project_config.ini'])
def all_project_config_args(request):
    return request


@pytest.mark.parametrize("config_path, expected_clf_names", [['tests/data/test_projects/two_c57/project_folder/project_config.ini', ['Attack', 'Sniffing']],
                                                            ['tests/data/test_projects/mouse_open_field/project_folder/project_config.ini', ['Attack']],
                                                            ['tests/data/test_projects/zebrafish/project_folder/project_config.ini', ['Rheotaxis']]])
def test_config_reader(config_path, expected_clf_names):
    config_reader = ConfigReader(config_path=config_path)
    config_reader.read_roi_data()
    assert config_reader.get_all_clf_names() == expected_clf_names


def test_get_number_of_header_columns_in_df(project_config_args):
    config_reader = ConfigReader(config_path=project_config_args.param)
    results = config_reader.get_number_of_header_columns_in_df(df=pd.DataFrame(data=[[1, 2, 3], [1, 2, 3]]))
    assert results == 0
    results = config_reader.get_number_of_header_columns_in_df(df=pd.DataFrame(data=[['sdfsdf', 'sdf', 'asd'], ['5', 'asd', 'asd'], [11, 99, 109], [122, 43, 2091]]))
    assert results == 2

@pytest.mark.parametrize("config_path, expected_path",      [['tests/data/test_projects/two_c57/project_folder/project_config.ini', 'Together_1'],
                                                            ['tests/data/test_projects/mouse_open_field/project_folder/project_config.ini', 'Video1'],
                                                            ['tests/data/test_projects/zebrafish/project_folder/project_config.ini', '20200730_AB_7dpf_850nm_0003']])
def test_find_video_of_file(config_path, expected_path):
    config_reader = ConfigReader(config_path=config_path)
    video_path = config_reader.find_video_of_file(video_dir=config_reader.video_dir, filename=expected_path)
    _, video_name, _ = get_fn_ext(video_path)
    assert video_name == expected_path

def test_add_missing_ROI_cols(project_config_args):
    config_reader = ConfigReader(config_path=project_config_args.param)
    results = config_reader.add_missing_ROI_cols(shape_df=pd.DataFrame(data=[[1, 2, 3], [1, 2, 3]], columns=['Name', 'Center_X', 'Center_Y']))
    s = set()
    s.update(['Color BGR', 'Thickness', 'Color name'])
    assert len(s - set(results.columns)) == 0


@pytest.mark.parametrize("config_path, body_part_name, expected_animal_name", [['tests/data/test_projects/two_c57/project_folder/project_config.ini', 'Nose_1', 'simon'],
                                                                               ['tests/data/test_projects/mouse_open_field/project_folder/project_config.ini', 'Left_ear', 'Animal_1'],
                                                                               ['tests/data/test_projects/zebrafish/project_folder/project_config.ini', 'Zebrafish_LeftEye', 'Animal_1']])
def test_find_animal_name_from_body_part_name(config_path, body_part_name, expected_animal_name):
    config_reader = ConfigReader(config_path=config_path)
    animal_name = config_reader.find_animal_name_from_body_part_name(bp_name=body_part_name, bp_dict=config_reader.animal_bp_dict)
    assert animal_name == expected_animal_name

@pytest.mark.parametrize("config_path, expected", [['tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                    'tests/data/test_projects/two_c57/expected_animal_bp_dict/expected_animal_bp_dict.json']])
def test_create_body_part_dictionary(config_path, expected):
    config_reader = ConfigReader(config_path=config_path)
    bp_dict = config_reader.create_body_part_dictionary(multi_animal_status=config_reader.multi_animal_status,
                                                        animal_id_lst=config_reader.multi_animal_id_list,
                                                        animal_cnt=config_reader.animal_cnt,
                                                        x_cols=config_reader.x_cols,
                                                        y_cols=config_reader.y_cols,
                                                        p_cols=config_reader.p_cols,
                                                        colors=config_reader.clr_lst)
    with open(expected, 'r') as fp:
        assert json.load(fp) == bp_dict

def test_get_body_part_names(project_config_args):
    config_reader = ConfigReader(config_path=project_config_args.param)
    config_reader.get_body_part_names()
    assert len(config_reader.x_cols) == 16
    assert len(config_reader.y_cols) == 16
    assert len(config_reader.y_cols) == 16

def test_drop_bp_cords(project_config_args):
    config_reader = ConfigReader(config_path=project_config_args.param)
    df = read_df(config_reader.machine_results_paths[0], file_type='csv')
    out_df = config_reader.drop_bp_cords(df=df)
    assert (len(df.columns) - len(out_df.columns)) == len(config_reader.bp_headers)


@pytest.mark.parametrize("config_path, expected",      [['tests/data/test_projects/two_c57/project_folder/project_config.ini', 48],
                                                            ['tests/data/test_projects/mouse_open_field/project_folder/project_config.ini', 12],
                                                            ['tests/data/test_projects/zebrafish/project_folder/project_config.ini', 21]])
def test_get_bp_headers(config_path, expected):
    config_reader = ConfigReader(config_path=config_path)
    config_reader.get_bp_headers()
    assert len(config_reader.bp_headers) == expected

@pytest.mark.parametrize("config_path, expected",      [['tests/data/test_projects/two_c57/project_folder/project_config.ini', 2],
                                                            ['tests/data/test_projects/mouse_open_field/project_folder/project_config.ini', 1],
                                                            ['tests/data/test_projects/zebrafish/project_folder/project_config.ini', 1]])
def test_read_config_entry(config_path, expected):
    config = read_config_file(config_path=config_path)
    config_reader = ConfigReader(config_path=config_path)
    result = config_reader.read_config_entry(config=config, section='General settings', option='animal_no', data_type='int')
    assert result == expected

def test_read_video_info_csv(all_project_config_args):
    config_reader = ConfigReader(config_path=all_project_config_args.param)
    df = config_reader.read_video_info_csv(file_path=config_reader.video_info_path)
    print(df)
    assert isinstance(df, pd.DataFrame)

@pytest.mark.parametrize("config_path, video_name, expected_pixels, expected_fps",      [['tests/data/test_projects/two_c57/project_folder/project_config.ini', 'Together_1', 4.0, 30],
                                                                                         ['tests/data/test_projects/mouse_open_field/project_folder/project_config.ini', 'Video1', 2.57425, 30],
                                                                                         ['tests/data/test_projects/zebrafish/project_folder/project_config.ini', '20200730_AB_7dpf_850nm_0003', 1.2273399999999999, 30]])
def test_read_video_info(config_path, video_name, expected_pixels, expected_fps):
    config_reader = ConfigReader(config_path=config_path)
    video_settings, px_per_mm, fps = config_reader.read_video_info(video_name=video_name)
    assert len(video_settings) == 1
    assert px_per_mm == expected_pixels
    assert fps == fps


