"""
SimBA (Simple Behavioral Analysis)
Toolkit for computer classification and analysis of behaviors in experimental animals
"""
import matplotlib

matplotlib.use("Agg")

import multiprocessing
import os

from simba.utils.checks import is_wsl
from simba.utils.enums import ENV_VARS, OS
from simba.utils.warnings import InvalidValueWarning

mp_method = os.getenv(ENV_VARS.MP_START_METHOD.value)
if mp_method is not None:
    mp_method = mp_method.strip().lower()
    if mp_method not in multiprocessing.get_all_start_methods():
        InvalidValueWarning(msg=f'MP_START_METHOD={mp_method} is not supported on this system (supported: {multiprocessing.get_all_start_methods()}). Using the default start method.', source='simba.__init__.py')
        mp_method = None
if mp_method is not None:
    multiprocessing.set_start_method(mp_method, force=True)
elif is_wsl():
    multiprocessing.set_start_method(OS.SPAWN.value, force=True)

__author__ = "Simon Nilsson"
__author_email__ = "sronilsson@gmail.com"
__maintainer__ = "Simon Nilsson"
__maintainer_email__ = "sronilsson@gmail.com"
__copyright__ = "Copyright 2024, Simon Nilsson"
__license__ = "Modified BSD 3-Clause License"
__url__ = "https://github.com/sgoldenlab/simba"
__description__ = "Toolkit for computer classification and analysis of behaviors in experimental animals"


