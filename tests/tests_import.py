import pytest
import os
from simba.BORIS_appender import BorisAppender
from simba.solomon_importer import SolomonImporter
from simba.ethovision_import import ImportEthovision
from simba.misc_tools import check_file_exist_and_readable, get_fn_ext


@pytest.mark.parametrize("config_path", [('test_data/import_tests/project_folder/project_config.ini')])
class TestImports(object):

    def check_output_files_exist(self, input_files: list, output_dir: str) -> None:
        for file_path in input_files:
            _, video_name, _ = get_fn_ext(filepath=file_path)
            output_file_path = os.path.join(output_dir, video_name + '.csv')
            check_file_exist_and_readable(output_file_path)
            assert os.path.getsize(output_file_path) > 0
            os.remove(output_file_path)

    @pytest.mark.parametrize("data_path", [('test_data/import_tests/boris_data')])
    def test_boris_import(self,
                          config_path,
                          data_path):
        boris_appender = BorisAppender(config_path=config_path,
                                       boris_folder=data_path)
        boris_appender.create_boris_master_file()
        boris_appender.append_boris()
        self.check_output_files_exist(input_files=boris_appender.feature_files_found,
                                      output_dir=boris_appender.out_dir)

    @pytest.mark.parametrize("data_path", [('test_data/import_tests/solomon_data')])
    def test_solomon_import(self, config_path, data_path):
        solomon_appender = SolomonImporter(config_path=config_path,
                                           solomon_dir=data_path)
        solomon_appender.import_solomon()
        self.check_output_files_exist(input_files=solomon_appender.solomon_paths,
                                      output_dir=solomon_appender.out_folder)

    @pytest.mark.parametrize("data_path",[('test_data/import_tests/ethovision_data/error'),
                                          ('test_data/import_tests/ethovision_data/correct')])
    def test_ethovision_import(self, config_path, data_path):
        ethovision_appender = ImportEthovision(config_path=config_path, folder_path=data_path)
        self.check_output_files_exist(input_files=ethovision_appender.processed_videos, output_dir=ethovision_appender.targets_insert_folder_path)
