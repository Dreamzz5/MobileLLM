import numpy as np


def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def R2(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)

        y_true_masked = y_true[mask > 0]
        y_pred_masked = y_pred[mask > 0]

        ss_res = np.sum((y_true_masked - y_pred_masked) ** 2)
        ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2


def RMSE_MAE_R2(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        R2(y_true, y_pred),
    )


def MSE_RMSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred),
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )
