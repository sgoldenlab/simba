from simba.utils.data import get_library_version
from simba.utils.enums import ENV_VARS
from simba.utils.read_write import read_sys_env
from simba.utils.warnings import VersionWarning

ENV = read_sys_env()

### IMPORT CURF IF SET BY simba/assets/.env
use_cuml = ENV.get(ENV_VARS.CUML.value, False)  # Avoid KeyError, default to False
#use_cuml = True
if use_cuml:
    cuml_version = get_library_version(library_name='cuml', raise_error=False)
    if cuml_version:
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            print('SimBA CUML enabled.')
        except ImportError:
            VersionWarning(msg="cuML is set but not installed. Falling back to scikit-learn.")
            from sklearn.ensemble import RandomForestClassifier as cuRF
    else:
        print("cuML is set but not installed. Falling back to scikit-learn.")
        from sklearn.ensemble import RandomForestClassifier as cuRF
else:
    from sklearn.ensemble import RandomForestClassifier as cuRF

__all__ = ["cuRF"]