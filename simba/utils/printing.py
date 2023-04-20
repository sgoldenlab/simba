from simba.enums import TagNames, Defaults

def stdout_success(msg: str, elapsed_time: str or None=None) -> None:
    if elapsed_time:
        print(f'SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}')
    else:
        print(f'SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.COMPLETE.value}')

def stdout_warning(msg: str, elapsed_time: str or None=None) -> None:
    if elapsed_time:
        print(f'SIMBA WARNING: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}')
    else:
        print(f'SIMBA WARNING: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.WARNING.value}')

def stdout_trash(msg: str, elapsed_time: str or None=None) -> None:
    if elapsed_time:
        print(f'SIMBA COMPLETE: {msg} (elapsed time: {elapsed_time}s) {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.TRASH.value}')
    else:
        print(f'SIMBA COMPLETE: {msg} {Defaults.STR_SPLIT_DELIMITER.value}{TagNames.TRASH.value}')
