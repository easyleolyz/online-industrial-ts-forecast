# src/online_ts_forecast/traditional/metrics.py

from __future__ import annotations
import numpy as np
import pandas as pd

#平均绝对误差
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

#这里是均方根误差
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

#对称平均绝对百分比误差，误差占真实值/预测值规模的比例
def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-12, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def evaluate_backtest(df: pd.DataFrame) -> dict:
    """
    df: 包含列 ['time', 'y_true', 'y_pred'] 的回测结果
    """
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "sMAPE": np.nan, "n": 0}
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "n": int(len(y_true)),
    }
