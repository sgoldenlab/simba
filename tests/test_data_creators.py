import pytest
from simba.create_clf_log import ClfLogCreator
from simba.Directing_animals_analyzer import DirectingOtherAnimalsAnalyzer
from simba.misc_tools import check_file_exist_and_readable
import os

@pytest.mark.parametrize("config_path, data_measures, classifiers",
                         [('test_data/two_C57_madlc/project_folder/project_config.ini', ['Bout count',
                                                                                         'Total event duration (s)',
                                                                                         'Mean event bout duration (s)',
                                                                                         'Median event bout duration (s)',
                                                                                         'First event occurrence (s)',
                                                                                         'Mean event bout interval duration (s)',
                                                                                         'Median event bout interval duration (s)'
                                                                                         ],
                           ['Attack', 'Sniffing']),
                         ])
def test_create_clf_log(config_path, data_measures, classifiers):
    clf_log_creator = ClfLogCreator(config_path=config_path,
                                    data_measures=data_measures,
                                    classifiers=classifiers)
    clf_log_creator.analyze_data()
    clf_log_creator.save_results()
    assert sorted(list(clf_log_creator.results_df['Classifier'].unique())) == sorted(classifiers)
    assert sorted(list(clf_log_creator.results_df['Measure'].unique())) == sorted(data_measures)
    os.remove(path=clf_log_creator.file_save_name)


@pytest.mark.parametrize("config_path", [('test_data/two_C57_madlc/project_folder/project_config.ini')])
def test_directing_animal_analyzer(config_path):
    directing_animal_analyzer = DirectingOtherAnimalsAnalyzer(config_path=config_path)
    directing_animal_analyzer.process_directionality()
    directing_animal_analyzer.create_directionality_dfs()
    directing_animal_analyzer.save_directionality_dfs()
    directing_animal_analyzer.summary_statistics()
    check_file_exist_and_readable(file_path=directing_animal_analyzer.save_path)
    assert len(directing_animal_analyzer.summary_df) == 4
    assert sorted(list(directing_animal_analyzer.summary_df.index.unique())) == ['Together_1', 'Together_2']
    os.remove(path=directing_animal_analyzer.save_path)





