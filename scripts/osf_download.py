from osfclient.api import OSF
import os
from typing import Union
from simba.utils.checks import check_if_dir_exists, check_str, check_valid_boolean
from simba.utils.read_write import get_pkg_version

def osf_download(project_id: str, save_dir: Union[str, os.PathLike], storage: str = 'osfstorage', overwrite: bool = False):
    """
    Download all files from an OSF (Open Science Framework) project to a local directory.

    This function connects to the OSF API, accesses the specified project and storage location,
    and downloads all files to the local save directory. Files can be skipped if they already
    exist locally and overwrite is disabled.

    :param str project_id: OSF project identifier (e.g., 'abc123' from osf.io/abc123).
    :param Union[str, os.PathLike] save_dir: Local directory path where files will be downloaded.
    :param str storage: OSF storage location name (default: 'osfstorage').
    :param bool overwrite: If True, overwrite existing files. If False, skip existing files (default: False).

    :example:
    >>> osf_download(project_id="7fgwn", save_dir=r'E:\rgb_white_vs_black_imgs')
    """
    _ = get_pkg_version(pkg='osfclient', raise_error=True)
    osf = OSF()
    storage = osf.project(project_id).storage(storage)
    check_if_dir_exists(in_dir=save_dir, source=f'{osf_download.__name__} save_dir', raise_error=True)
    check_str(name=f'{osf_download.__name__} project_id', value=project_id, allow_blank=False, raise_error=True)
    check_valid_boolean(value=overwrite, source=f'{osf_download.__name__} overwrite', raise_error=True)
    for file_cnt, file in enumerate(storage.files):
        local_path = os.path.join(save_dir, file.path.strip("/")[1:])
        if os.path.isfile(local_path) and not overwrite:
            print(f'Skipping file {file} (exist and overwrite is False...')
        with open(local_path, "wb") as f:
            file.write_to(f)
        print(f"Downloaded {local_path} ({file_cnt+1}/{len(storage.files)})")

