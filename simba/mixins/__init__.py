try:
    from cuml.ensemble import RandomForestClassifier as cuRF
except ImportError:
    cuRF = None

__all__ = ['cuRF']

