"""
Upgrade SimBA while preserving custom pose configurations.

This script automates the process of upgrading SimBA to the latest version while
ensuring that any custom pose configurations in the user's environment are preserved.
It backs up the pose_configurations directory, upgrades the package, and restores
the custom configurations.

Usage:
    Command line:
        python upgrade_simba_keep_configs.py

    Programmatic:
        from scripts.upgrade_simba_keep_configs import upgrade_simba
        upgrade_simba(backup_dir='/path/to/backup')
"""

import sys
import os
from pathlib import Path
import sysconfig
from simba.utils.read_write import remove_a_folder
import shutil
import subprocess
from simba.utils.errors import SimBAPAckageVersionError
from typing import Optional, Union
from simba.utils.checks import check_valid_boolean, check_if_dir_exists
from simba.utils.printing import stdout_success, SimbaTimer

def copytree_compat(src, dst):
    if os.path.exists(dst):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                copytree_compat(s, d)
            else:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
    else:
        shutil.copytree(src, dst)

def get_site_packages_path(raise_error: Optional[bool] = True) -> Union[None, os.PathLike, str]:
    """
    Retrieve the path to the current Python environment's `site-packages` directory.
    """
    check_valid_boolean(value=[raise_error], source=f'{get_site_packages_path.__name__} raise_error')
    try:
        dir_path = sysconfig.get_paths()["purelib"]
        check_if_dir_exists(in_dir=dir_path, source=get_site_packages_path.__name__, create_if_not_exist=False, raise_error=True)
        return dir_path
    except Exception as e:
        if raise_error:
            raise SimBAPAckageVersionError(msg=f'site-package directory could not be found: {e.args}', source=get_site_packages_path.__name__)
        else:
            return None

def get_env_pose_config_dir(raise_error: Optional[bool] = True):
    """
    Locate and validate the `pose_configurations` directory in the active SimBA installation.
    """
    EXPECTED_SUBDIRS = ['bp_names', 'no_animals', 'configuration_names', 'schematics']
    site_packages_dir = get_site_packages_path(raise_error=raise_error)
    pose_config_dir = os.path.join(site_packages_dir, 'simba', 'pose_configurations')
    if os.path.isdir(pose_config_dir):
        subdirs = [d for d in os.listdir(pose_config_dir) if os.path.isdir(os.path.join(pose_config_dir, d))]
        missing_subdirs = [x for x in EXPECTED_SUBDIRS if x not in subdirs]
        if len(missing_subdirs) > 0 and raise_error:
            raise SimBAPAckageVersionError(msg=f'pose_configurations directory did not contain the expected sub-directories: {missing_subdirs}', source=get_env_pose_config_dir.__name__)
        elif len(missing_subdirs) > 0 and not raise_error:
            return None
        else:
            return pose_config_dir
    else:
        if raise_error:
            raise SimBAPAckageVersionError(msg=f'pose_configurations directory could not be found. Expected directory: {pose_config_dir}', source=get_pose_config_dir.__name__)
        return None

def upgrade_simba(backup_dir: Optional[Union[os.PathLike, None]] = None):
    """
    Upgrade SimBA to the latest version while preserving pose configurations.

    This function performs a complete upgrade workflow:
    1. Backs up current pose configurations
    2. Upgrades SimBA via pip
    3. Restores custom pose configurations
    4. Cleans up backup directory

    :param Optional[Union[os.PathLike, None]] backup_dir: Directory to store temporary backup.
        If None, creates 'simba_config_backup' in the current working directory.

    :example:
    >>> upgrade_simba()  # Uses default backup location
    >>> upgrade_simba(backup_dir='/tmp/simba_backup')  # Custom backup location
    """
    print('[STEP 0/4] Starting SimBA upgrade...')
    timer = SimbaTimer(start=True)
    if backup_dir is not None:
        check_if_dir_exists(in_dir=backup_dir, source=get_site_packages_path.__name__, create_if_not_exist=False, raise_error=True)
    else:
        backup_dir = Path.cwd() / "simba_config_backup"
    pose_config_dir = get_env_pose_config_dir(raise_error=True)
    print('[STEP 1/4] Backing-up SimBA project configurations...')
    copytree_compat(pose_config_dir, backup_dir)
    print('[STEP 2/4] Upgrading SimBA...')
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", 'simba-uw-tf-dev'], check=True)
    print('[STEP 3/4] Updating pose-configurations in new SimBA installation...')
    copytree_compat(backup_dir, pose_config_dir)
    print('[STEP 4/4] Removing pose-configurion back-up...')
    remove_a_folder(folder_dir=backup_dir, ignore_errors=True)
    timer.stop_timer()
    stdout_success(msg=f'SIMBA UPDATE COMPLETE. START LATEST SIMBA VERSION BY TYPING: simba.', elapsed_time=timer.elapsed_time_str)

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    upgrade_simba()