import os.path, glob

import pytest
from simba.third_party_label_appenders.BORIS_appender import BorisAppender
from simba.third_party_label_appenders.solomon_importer import SolomonImporter
from simba.third_party_label_appenders.ethovision_import import ImportEthovision
from simba.third_party_label_appenders.observer_importer import NoldusObserverImporter
from simba.third_party_label_appenders.BENTO_appender import BentoAppender
from simba.third_party_label_appenders.deepethogram_importer import DeepEthogramImporter


@pytest.mark.parametrize("config_path, boris_path", [('tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                      'tests/data/test_projects/two_c57/boris_annotations')])
def test_boris_import_use_case(config_path, boris_path):
    boris_appender = BorisAppender(config_path=config_path,
                                   data_dir=boris_path)
    boris_appender.create_boris_master_file()
    boris_appender.run()
    #assert len(boris_appender.out_df) == 1738
    #for f in glob.glob(boris_appender.targets_folder + '/*.csv'): os.remove(f)

@pytest.mark.parametrize("config_path, solomon_path", [('tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                        'tests/data/test_projects/two_c57/solomon_annotations')])
def test_solomon_import_use_case(config_path, solomon_path):
    solomon_appender = SolomonImporter(config_path=config_path,
                                       data_dir=solomon_path)
    solomon_appender.run()

@pytest.mark.parametrize("config_path, data_dir", [('tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                    'tests/data/test_projects/two_c57/ethovision_annotations')])
def test_ethovision_import_use_case(config_path, data_dir):
    ethovision_importer = ImportEthovision(config_path=config_path, data_dir=data_dir)
    ethovision_importer.run()
    for f in glob.glob(ethovision_importer.targets_folder + '/*.csv'): os.remove(f)

@pytest.mark.parametrize("config_path, data_dir", [('tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                    'tests/data/test_projects/two_c57/observer_annotations')])
def test_observer_import_use_case(config_path, data_dir):
    observer_importer = NoldusObserverImporter(config_path=config_path, data_dir=data_dir)
    observer_importer.run()
    for f in glob.glob(observer_importer.targets_folder + '/*.csv'): os.remove(f)

@pytest.mark.parametrize("config_path, data_dir", [('tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                    'tests/data/test_projects/two_c57/bento_annotations')])
def test_bento_import_use_case(config_path, data_dir):
    bento_appender = BentoAppender(config_path=config_path, data_dir=data_dir)
    bento_appender.run()
    for f in glob.glob(bento_appender.targets_folder + '/*.csv'): os.remove(f)

@pytest.mark.parametrize("config_path, data_dir", [('tests/data/test_projects/two_c57/project_folder/project_config.ini',
                                                    'tests/data/test_projects/two_c57/deepethogram_annotations')])
def test_deepethogram_import_use_case(config_path, data_dir):
    deepethogram_importer = DeepEthogramImporter(config_path=config_path, data_dir=data_dir)
    deepethogram_importer.run()
    for f in glob.glob(deepethogram_importer.targets_folder + '/*.csv'): os.remove(f)
