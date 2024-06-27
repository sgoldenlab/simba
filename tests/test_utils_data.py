import pytest
import pandas as pd
from simba.utils.read_write import read_df
from simba.utils.data import (detect_bouts,
                              plug_holes_shortest_bout,
                              create_color_palettes,
                              create_color_palette)

@pytest.mark.parametrize("data_path, target_lst, fps", [('tests/data/test_projects/two_c57/project_folder/csv/machine_results/Together_1.csv', ['Attack', 'Sniffing'], 30)])
def test_detect_bouts(data_path, target_lst, fps):
    data_df = read_df(file_path=data_path, file_type='csv')
    results = detect_bouts(data_df=data_df, target_lst=target_lst, fps=fps)

def test_plug_holes_shortest_bout():
    data_df = pd.DataFrame(data=[1, 0, 1, 1, 1], columns=['target'])
    results = plug_holes_shortest_bout(data_df=data_df, clf_name='target', fps=10, shortest_bout=2000)
    pd.testing.assert_frame_equal(results, pd.DataFrame(data=[0, 0, 0, 0, 0], columns=['target']))

def test_create_color_palettes():
    results = create_color_palettes(no_animals=2, map_size=2)
    assert results == [[[255.0, 0.0, 255.0], [0.0, 255.0, 255.0]], [[102.0, 127.5, 0.0], [102.0, 255.0, 255.0]]]

def test_create_color_palette():
    results = create_color_palette(pallete_name='jet', increments=3)
    assert results == [[127.5, 0.0, 0.0], [255.0, 212.5, 0.0], [0.0, 229.81481481481478, 255.0], [0.0, 0.0, 127.5]]
    results = create_color_palette(pallete_name='jet', increments=3, as_rgb_ratio=True)
    assert results == [[0.5, 0.0, 0.0], [1.0, 0.8333333333333334, 0.0], [0.0, 0.9012345679012345, 1.0], [0.0, 0.0, 0.5]]
    results = create_color_palette(pallete_name='jet', increments=3, as_hex=True)
    assert results == ['#800000', '#ffd400', '#00e6ff', '#000080']
