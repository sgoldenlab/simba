__author__ = "Simon Nilsson"

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import time
from typing import Optional
import logging

from simba.utils.enums import TagNames, Defaults

def stdout_success(msg: str,
                   source: str = '',
                   elapsed_time: Optional[str] = None) -> None:

    log_event(logger_name=f'{source}.{stdout_success.__name__}', log_type=TagNames.COMPLETE.value, msg=msg)
    if elapsed_time:
        print(f'SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}')
    else:
        print(f'SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}')

def stdout_warning(msg: str, elapsed_time: Optional[str] = None) -> None:
    if elapsed_time:
        print(f'SIMBA WARNING: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}')
    else:
        print(f'SIMBA WARNING: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}')

def stdout_trash(msg: str, source: str = '', elapsed_time: Optional[str] = None) -> None:
    log_event(logger_name=f'{source}.{stdout_trash.__name__}', log_type=TagNames.TRASH.value, msg=msg)
    if elapsed_time:
        print(f'SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.TRASH.value}')
    else:
        print(f'SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.TRASH.value}')

class SimbaTimer(object):
    def __init__(self,
                 start: bool = False):
        if start:
            self.start_timer()

    def start_timer(self):
        self.timer = time.time()

    def stop_timer(self):
        if not hasattr(self, 'timer'):
            self.elapsed_time = -1
            self.elapsed_time_str = '-1'
        else:
            self.elapsed_time = round(time.time() - self.timer, 4)
            self.elapsed_time_str = str(self.elapsed_time)

def log_event(logger_name: str,
              log_type: Literal['CLASS_INIT', 'error', 'warning'],
              msg: str):

    logger = logging.getLogger(str(logger_name))
    if log_type == TagNames.CLASS_INIT.value:
        logger.info(f'{TagNames.CLASS_INIT.value}||{msg}')
    elif log_type == TagNames.ERROR.value:
        logger.error(f'{TagNames.ERROR.value}||{msg}')
    elif log_type == TagNames.WARNING.value:
        logger.warning(f'{TagNames.WARNING.value}||{msg}')
    elif log_type == TagNames.TRASH.value:
        logger.info(f'{TagNames.TRASH.value}||{msg}')
    elif log_type == TagNames.COMPLETE.value:
        logger.info(f'{TagNames.COMPLETE.value}||{msg}')

def perform_timing(func):
    def decorator(*args, **kwargs):
        timer = SimbaTimer(start=True)
        results = func(*args, **kwargs, _timer=timer)
        timer.stop_timer()
        results['timer'] = timer.elapsed_time_str
        return results

    return decorator



