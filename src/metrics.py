import numpy as np
import pandas as pd


def calculate_metrics(actual, predicted):
    actual = pd.Series(actual, dtype="float64")
    predicted = pd.Series(predicted, dtype="float64")
    mask = actual.notna() & predicted.notna()

    if not mask.any():
        return {"rmse": None, "mae": None, "mape": None}

    errors = actual[mask] - predicted[mask]
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    non_zero = actual[mask] != 0
    if non_zero.any():
        mape = float(np.mean(np.abs(errors[non_zero] / actual[mask][non_zero])) * 100)
    else:
        mape = None

    return {"rmse": rmse, "mae": mae, "mape": mape}
