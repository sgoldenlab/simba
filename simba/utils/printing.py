__author__ = "Simon Nilsson; sronilsson@gmail.com"

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import logging
import time
from datetime import datetime
from typing import Optional

from simba.utils.enums import SIMBA_DOCS_URL, Defaults, TagNames


def tag_docs_url(url: str, medium: str = "docs_icon") -> str:
    """
    Tag a Read the Docs link opened from SimBA so the docs analytics (GA4) can count visits that come from the app.

    Visits from a desktop app arrive without a referrer, so without a tag they are lumped in with "direct" traffic.
    Only Read the Docs URLs are tagged; other URLs, and URLs that already carry a tag, are returned unchanged.

    :param str url: URL to tag.
    :param str medium: Where in SimBA the link was opened from (``utm_medium``), e.g. ``"docs_icon"`` or ``"error"``.
    :return: The URL with ``utm_source=simba_gui&utm_medium=<medium>`` added before any ``#anchor``.
    :rtype: str

    :example:
    >>> tag_docs_url('https://simba-uw-tf-dev.readthedocs.io/en/latest/Scenario1.html#step-5-extract-features')
    'https://simba-uw-tf-dev.readthedocs.io/en/latest/Scenario1.html?utm_source=simba_gui&utm_medium=docs_icon#step-5-extract-features'
    """
    if not isinstance(url, str) or not url.startswith(SIMBA_DOCS_URL) or "utm_source=" in url:
        return url
    page, sep, anchor = url.partition("#")
    return f"{page}{'&' if '?' in page else '?'}utm_source=simba_gui&utm_medium={medium}{sep}{anchor}"


def stdout_success(msg: str, source: Optional[str] = "", elapsed_time: Optional[str] = None) -> None:
    """
    Helper to parse msg of completed operation to SimBA main interface.

    :param str msg: Message to be parsed.
    :param Optional[str] source: Optional string indicating the source method or function of the msg for logging.
    :param Optional[str] elapsed_time: Optional string indicating the runtime of the completed operation.
    :return None:
    """

    log_event(logger_name=f"{source}.{stdout_success.__name__}", log_type=TagNames.COMPLETE.value, msg=msg)
    if elapsed_time:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}")


def stdout_warning(msg: str, elapsed_time: Optional[str] = None) -> None:
    """
    Helper to parse warning msg to SimBA main interface.

    :param str msg: Message to be parsed.
    :param Optional[str] source: Optional string indicating the source method or function of the msg for logging.
    :param elapsed_time: Optional string indicating the runtime.
    :return None:
    """

    if elapsed_time:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA WARNING: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA WARNING: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}")


def stdout_trash(msg: str, source: Optional[str] = "", elapsed_time: Optional[str] = None) -> None:
    """
    Helper to parse msg of delete operation to SimBA main interface.

    :param str msg: Message to be parsed.
    :param Optional[str] source: Optional string indicating the source method or function of the operation for logging.
    :param elapsed_time: Optional string indicating the runtime.
    :return None:
    """

    log_event(logger_name=f"{source}.{stdout_trash.__name__}", log_type=TagNames.TRASH.value, msg=msg)
    if elapsed_time:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.TRASH.value}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.TRASH.value}")


def stdout_information(msg: str, source: Optional[str] = "", elapsed_time: Optional[str] = None) -> None:
    """
    Helper to parse information msg to SimBA main interface. E.g., how many monitors and their resolutions which is available.

    :param str msg: Message to be parsed.
    :param Optional[str] source: Optional string indicating the source method or function of the operation for logging.
    :param elapsed_time: Optional string indicating the runtime.
    :return None:
    """

    log_event(logger_name=f"{source}.{stdout_trash.__name__}", log_type=TagNames.INFORMATION.value, msg=msg)
    if elapsed_time:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.INFORMATION.value}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.INFORMATION.value}")


class SimbaTimer(object):
    """Timer class for keeping track of start and end-times of calls"""

    def __init__(self,
                 start: bool = False,
                 perf_counter: bool = False):

        self.perf_counter = perf_counter

        if start:
            self.start_timer()

    def start_timer(self):
        if not self.perf_counter:
            self.timer = time.time()
        else:
            self.timer = time.perf_counter()

    def stop_timer(self):
        if not hasattr(self, "timer"):
            self.elapsed_time = -1
            self.elapsed_time_str = "-1"
        else:
            if self.perf_counter:
                self.elapsed_time = round(time.perf_counter() - self.timer, 4)
            else:
                self.elapsed_time = round(time.time() - self.timer, 4)
            self.elapsed_time_str = str(self.elapsed_time)


def log_event(logger_name: str, log_type: Literal["CLASS_INIT", "error", "warning"], msg: str):
    logger = logging.getLogger(str(logger_name))
    if log_type == TagNames.CLASS_INIT.value:
        logger.info(f"{TagNames.CLASS_INIT.value}||{msg}")
    elif log_type == TagNames.ERROR.value:
        logger.error(f"{TagNames.ERROR.value}||{msg}")
    elif log_type == TagNames.WARNING.value:
        logger.warning(f"{TagNames.WARNING.value}||{msg}")
    elif log_type == TagNames.TRASH.value:
        logger.info(f"{TagNames.TRASH.value}||{msg}")
    elif log_type == TagNames.COMPLETE.value:
        logger.info(f"{TagNames.COMPLETE.value}||{msg}")


def perform_timing(func):
    def decorator(*args, **kwargs):
        timer = SimbaTimer(start=True)
        results = func(*args, **kwargs, _timer=timer)
        timer.stop_timer()
        results["timer"] = timer.elapsed_time_str
        return results

    return decorator
