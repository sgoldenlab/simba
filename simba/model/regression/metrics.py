from typing import Optional

import numpy as np

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats


def mean_absolute_percentage_error(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   epsilon=1e-10,
                                   weights: Optional[np.ndarray] = None) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE)

    :param np.ndarray y_true: The array containing the true values (dependent variable) of the dataset. Should be a 1D numeric array of shape (n,).
    :param np.ndarray y_pred: The array containing the predicted values for the dataset. Should be a 1D numeric array of shape (n,) and of the same length as `y_true`.
    :param float epsilon: A small pseudovalue to replace zeros in `y_true` to avoid division by zero errors.
    :param Optional[np.ndarray] weights: An optional 1D array of weights to apply to each error. If provided, the weighted mean absolute percentage error is computed.
    :return: The Mean Absolute Percentage Error (MAPE) as a float, in percentage format. A lower value indicates better prediction accuracy.
    :rtype: float

    :example:
    >>> x, y = np.random.random(size=(100000,)), np.random.random(size=(100000,))
    >>> mean_absolute_percentage_error(y_true=x, y_pred=y)
    """

    check_valid_array(data=y_true, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y_pred, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0],accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=mean_absolute_percentage_error.__name__, value=epsilon)
    y_true = np.where(y_true == 0, epsilon, y_true)
    se = np.abs((y_true - y_pred) / y_true)
    if weights is not None:
        check_valid_array(data=weights, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        se = se * weights
        return (np.sum(se) / np.sum(weights)) * 100
    else:
        return np.mean(se * 100)


def mean_squared_error(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       weights: Optional[np.ndarray] = None) -> float:

    """
    Compute the Mean Squared Error (MSE) between the true and predicted values.

    :param np.ndarray y_true: The array containing the true values (dependent variable) of the dataset. Should be a 1D numeric array of shape (n,).
    :param np.ndarray y_pred: The array containing the predicted values for the dataset. Should be a 1D numeric array of shape (n,) and of the same length as `y_true`.
    :param Optional[np.ndarray] weights: An optional 1D array of weights to apply to each squared error. If provided, the weighted mean squared error is computed.
    :return: The Mean Squared Error (MSE) as a float. A lower value indicates better model accuracy.
    :rtype: float
    """

    check_valid_array(data=y_true, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y_pred, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0],accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    se = (y_true - y_pred) ** 2
    if weights is not None:
        check_valid_array(data=weights, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        se = se * weights
        return np.sum(se) / np.sum(weights)
    else:
        return np.mean(se)

def mean_absolute_error(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        weights: Optional[np.ndarray] = None) -> float:
    """
    Compute the Mean Absolute Error (MAE) between the true and predicted values.

    :param np.ndarray y_true: A 1D array of true values (ground truth).
    :param np.ndarray y_pred: A 1D array of predicted values.
    :param np.ndarray weights: An optional 1D array of weights for each observation. If provided, the weighted MAE is computed.
    :return: The Mean Absolute Error (MAE) as a float. A lower value indicates a better fit.
    :rtype: float
    """

    check_valid_array(data=y_true, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y_pred, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    absolute_error = np.abs(y_true - y_pred)
    if weights is not None:
        check_valid_array(data=weights, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        absolute_error = absolute_error * weights
        return np.sum(absolute_error) / np.sum(weights)
    else:
        return np.mean(absolute_error)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Compute the R^2 (coefficient of determination) score.

    :param np.ndarray y_true: 1D array of true values (dependent variable).
    :param np.ndarray y_pred: 1D array of predicted values, same length as `y_true`.
    :param np.ndarray weights: Optional 1D array of weights for each observation.
    :return: The R^2 score as a float. A value closer to 1 indicates better fit.
    :rtype: float

    """

    check_valid_array(data=y_true, source=r2_score.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y_pred, source=r2_score.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0], accepted_dtypes=Formats.NUMERIC_DTYPES.value)

    if weights is not None:
        check_valid_array(data=weights, source=r2_score.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0])

    y_mean = np.average(y_true, weights=weights) if weights is not None else np.mean(y_true)
    residuals, total = (y_true - y_pred) ** 2, (y_true - y_mean) ** 2

    if weights is not None:
        ss_residual = np.sum(residuals * weights)
        ss_total = np.sum(total * weights)
    else:
        ss_residual = np.sum(residuals)
        ss_total = np.sum(total)

    return 1 - (ss_residual / ss_total)


def root_mean_squared_error(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            weights: Optional[np.ndarray] = None) -> float:

    """
    Compute the Root Mean Squared Error (RMSE) between the true and predicted values.

    :param np.ndarray y_true: The array containing the true values (dependent variable) of the dataset. Should be a 1D numeric array of shape (n,).
    :param np.ndarray y_pred: The array containing the predicted values for the dataset. Should be a 1D numeric array of shape (n,) and of the same length as `y_true`.
    :param Optional[np.ndarray] weights: An optional 1D array of weights to apply to each squared error. If provided, the weighted mean squared error is computed.
    :return: The Root Mean Squared Error (MSE) as a float. A lower value indicates better model accuracy.
    :rtype: float
    """

    check_valid_array(data=y_true, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y_pred, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0],accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    se = (y_true - y_pred)  ** 2
    if weights is not None:
        check_valid_array(data=weights, source=mean_absolute_percentage_error.__name__, accepted_ndims=(1,), min_axis_0=y_true.shape[0], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        weighted_mse = np.sum(se * weights) / np.sum(weights)
        return np.sqrt(weighted_mse)
    else:
        return np.sqrt(np.mean(se))