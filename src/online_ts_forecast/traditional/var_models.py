# src/online_ts_forecast/traditional/var_models.py

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def var_forecast(
    history: pd.DataFrame,
    horizon: int = 1,
    maxlags: int = 12,
) -> pd.DataFrame:
    """
    使用 VAR 对多变量历史序列建模，并预测未来 horizon 步。
    history: DataFrame，index 为时间戳，列为多个变量。
    """
    if history.shape[0] < maxlags + 5:
        # 样本不足，减少 maxlags 或退化
        eff_maxlags = max(1, min(5, history.shape[0] // 2))
    else:
        eff_maxlags = maxlags

    model = VAR(history)
    res = model.fit(maxlags=eff_maxlags, ic="aic")

    # res.k_ar 为实际使用的滞后阶数
    lag_order = res.k_ar
    if history.shape[0] <= lag_order:
        # 防御：历史太短，再降一次阶数
        lag_order = max(1, history.shape[0] - 1)

    fcst = res.forecast(history.values[-lag_order:], steps=horizon)
    df_fcst = pd.DataFrame(fcst, columns=history.columns)
    return df_fcst


def rolling_backtest_var(
    df: pd.DataFrame,
    target_col: str,
    horizon: int = 1,
    min_history: int = 500,
    maxlags: int = 12,
) -> pd.DataFrame:
    """
    对多变量序列做 VAR 滚动回测，仅评估 target_col 这一列。
    返回 DataFrame: [time, y_true, y_pred]
    """
    df = df.dropna()
    n = len(df)
    rows = []

    idx = df.index

    start_idx = max(min_history, 0)
    end_idx = n - horizon

    for t_idx in range(start_idx, end_idx):
        history = df.iloc[: t_idx + 1]
        target_idx = t_idx + horizon

        y_true = float(df[target_col].iloc[target_idx])

        df_fcst = var_forecast(history, horizon=horizon, maxlags=maxlags)
        if len(df_fcst) < horizon:
            continue
        y_pred = float(df_fcst[target_col].iloc[horizon - 1])

        rows.append(
            {
                "time": idx[target_idx],
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)
