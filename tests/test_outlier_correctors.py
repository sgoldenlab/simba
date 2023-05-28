import pytest
import os

from simba.outlier_tools.outlier_corrector_location import OutlierCorrecterLocation
from simba.outlier_tools.outlier_corrector_movement import OutlierCorrecterMovement
from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper

@pytest.mark.parametrize("config_path", ['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def test_outlier_corrector_movement_use_case(config_path):
    movement_corrector = OutlierCorrecterMovement(config_path=config_path)
    movement_corrector.run()

@pytest.mark.parametrize("config_path", ['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def test_outlier_corrector_location_use_case(config_path):
    location_corrector = OutlierCorrecterLocation(config_path=config_path)
    location_corrector.run()

@pytest.mark.parametrize("config_path", ['tests/data/test_projects/two_c57/project_folder/project_config.ini'])
def test_outlier_skipper_use_case(config_path):
    skipper = OutlierCorrectionSkipper(config_path=config_path)
    skipper.run()

