import pytest
import os
from simba.data_processors.agg_clf_calculator import AggregateClfCalculator
from simba.data_processors.fsttc_calculator import FSTTCCalculator
from simba.data_processors.kleinberg_calculator import KleinbergCalculator
from simba.data_processors.movement_calculator import MovementCalculator
from simba.data_processors.timebins_clf_calculator import TimeBinsClfCalculator
from simba.data_processors.timebins_movement_calculator import TimeBinsMovementCalculator




@pytest.mark.parametrize("config_path, data_measures, video_meta_data, classifiers", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', ['Bout count'], [], ['Attack']),
                                                                                      ('tests/data/test_projects/two_c57/project_folder/project_config.ini', ['Total event duration (s)'], [], ['Attack'])])
def test_create_clf_log_use_case(config_path, data_measures, video_meta_data, classifiers):

    clf_log_creator = AggregateClfCalculator(config_path=config_path,
                                             data_measures=data_measures, 
                                             classifiers=classifiers,
                                             video_meta_data=video_meta_data)
    clf_log_creator.run()
    clf_log_creator.save()
    assert len(clf_log_creator.results_df) == len(data_measures)
    assert os.path.isfile(clf_log_creator.save_path)
    os.remove(clf_log_creator.save_path)
    

@pytest.mark.parametrize("config_path, time_window, behavior_lst, create_graph", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', 2000, ['Attack', 'Sniffing'], False)])
def test_fsttc_calculator_use_case(config_path, time_window, behavior_lst, create_graph):
    fsttc_calculator = FSTTCCalculator(config_path=config_path, time_window=time_window, behavior_lst=behavior_lst, create_graphs=create_graph)
    fsttc_calculator.run()
    assert len(fsttc_calculator.out_df) == len(behavior_lst)
    if create_graph:
        assert os.path.isfile(fsttc_calculator.save_plot_path)
    os.remove(fsttc_calculator.file_save_path)

@pytest.mark.parametrize("config_path, classifier_names, sigma, gamma, hierarchy, hierarchical_search", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', ['Attack'], 1.1, 0.3, 5, False)])
def test_kleinberg_calculator_use_case(config_path, classifier_names, gamma, sigma, hierarchy, hierarchical_search):
    calculator = KleinbergCalculator(config_path=config_path,
                                     classifier_names=classifier_names,
                                     sigma=sigma,
                                     gamma=gamma,
                                     hierarchy=hierarchy,
                                     hierarchical_search=hierarchical_search)
    calculator.run()


@pytest.mark.parametrize("config_path, body_parts, threshold", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', ['simon CENTER OF GRAVITY'], 0.0),
                                                                ('tests/data/test_projects/two_c57/project_folder/project_config.ini', ['Nose_1'], 0.0)])
def test_movement_calculator_use_case(config_path, body_parts, threshold):
    calculator = MovementCalculator(config_path=config_path,
                                    body_parts=body_parts,
                                    threshold=threshold)
    calculator.run()
    calculator.save()
    assert os.path.isfile(calculator.save_path)
    os.remove(calculator.save_path)


@pytest.mark.parametrize("config_path, bin_length, measurements, classifiers", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', 2, ['Event count'], ['Attack'])])
def test_time_bins_clf_calculator_use_case(config_path, bin_length, measurements, classifiers):
    calculator = TimeBinsClfCalculator(config_path=config_path,
                                       bin_length=bin_length,
                                       measurements=measurements,
                                       classifiers=classifiers)
    calculator.run()
    assert os.path.isfile(calculator.save_path)
    os.remove(calculator.save_path)


@pytest.mark.parametrize("config_path, bin_length, plots, body_parts", [('tests/data/test_projects/two_c57/project_folder/project_config.ini', 10, True, ['Nose_1', 'Nose_2'])])
def test_time_bins_movement_calculator_use_case(config_path, bin_length, plots, body_parts):
    calculator = TimeBinsMovementCalculator(config_path=config_path,
                                            bin_length=bin_length,
                                            plots=plots,
                                            body_parts=body_parts)
    calculator.run()