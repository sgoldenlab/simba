__author__ = "Simon Nilsson"

import time
from typing import Optional

from simba.utils.enums import TagNames, Defaults


def stdout_success(msg: str, elapsed_time: Optional[str] = None) -> None:
    if elapsed_time:
        print(f'SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}')
    else:
        print(f'SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}')

def stdout_warning(msg: str, elapsed_time: Optional[str] = None) -> None:
    if elapsed_time:
        print(f'SIMBA WARNING: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}')
    else:
        print(f'SIMBA WARNING: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}')

def stdout_trash(msg: str, elapsed_time: Optional[str] = None) -> None:
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