# src/online_ts_forecast/traditional/baselines.py

from __future__ import annotations
from typing import Callable, Dict, Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ======= 单步/多步基线预测函数 =======

def naive_forecast(history: pd.Series, horizon: int = 1) -> np.ndarray:
    """
    最简单基线：y_hat_{t+h} = y_t
    history: 截止当前时刻 t 的历史序列（不含未来）
    """
    if len(history) == 0:
        raise ValueError("history is empty")
    last_value = history.iloc[-1]
    return np.repeat(last_value, horizon)


def seasonal_naive_forecast(
    history: pd.Series,
    horizon: int = 1,
    season_length: int = 288,
) -> np.ndarray:
    """
    季节 Naive：y_hat_{t+h} = y_{t+h - k*season_length}
    对 5min 数据，日季节 = 24*60/5 = 288
    """
    if len(history) < season_length:
        # 历史长度不足一个季节，退化为普通 Naive
        return naive_forecast(history, horizon=horizon)

    values = history.values
    n = len(values)
    fcst = []
    for h in range(1, horizon + 1):
        idx = n - season_length + (h - 1)
        if idx < 0:
            # 再次防御性退化
            fcst.append(values[-1])
        else:
            fcst.append(values[idx])
    return np.asarray(fcst, dtype=float)


def moving_average_forecast(
    history: pd.Series,
    horizon: int = 1,
    window: int = 12,
) -> np.ndarray:
    """
    简单移动平均：使用最近 window 个点的均值作为未来所有步的预测。
    window=12 对应最近 1 小时（5min 频率）。
    """
    if len(history) < window:
        # 历史太短时，退化为 Naive
        return naive_forecast(history, horizon=horizon)

    ma = history.iloc[-window:].mean()
    return np.repeat(ma, horizon)


def ets_forecast(
    history: pd.Series,
    horizon: int = 1,
    seasonal: Optional[str] = "add",
    season_length: int = 288,
    trend: Optional[str] = "add",
) -> np.ndarray:
    """
    ETS 指数平滑（Holt-Winters）。
    对 5min 冷机数据，season_length 默认按日季节 288。
    """
    if len(history) < 2 * season_length:
        # 历史不足以支撑季节 ETS，退化为无季节或更简单基线
        model = ExponentialSmoothing(
            history.astype(float),
            trend=trend,
            seasonal=None,
        )
    else:
        model = ExponentialSmoothing(
            history.astype(float),
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=season_length,
        )
    fitted = model.fit(optimized=True, use_brute=False)
    fcst = fitted.forecast(horizon)
    return np.asarray(fcst, dtype=float)


# ======= 通用滚动回测（Rolling Forecast Origin） =======

ForecastFunc = Callable[[pd.Series, int], np.ndarray]


def rolling_backtest(
    series: pd.Series,
    forecast_func: ForecastFunc,
    horizon: int = 1,
    min_history: int = 500,
    forecast_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    对单变量序列做滚动起点回测（单 horizon）：
    - 每次用截至 t 的历史，预测 t+h；
    - 记录 (time, y_true, y_pred)。

    返回 DataFrame: [time, y_true, y_pred]
    """
    if forecast_kwargs is None:
        forecast_kwargs = {}

    series = series.dropna()
    values = series.values
    times = series.index

    n = len(series)
    rows = []

    # t_idx 是“起报位置”（含该点作为历史），预测的是 t_idx + horizon
    start_idx = max(min_history, 0)
    end_idx = n - horizon  # 最后一个可用起报位置

    for t_idx in range(start_idx, end_idx):
        history = series.iloc[: t_idx + 1]  # 含当前点
        target_idx = t_idx + horizon
        y_true = float(values[target_idx])

        fcst = forecast_func(history, horizon=horizon, **forecast_kwargs)
        if len(fcst) < horizon:
            # 容错：若模型没给够，就跳过
            continue
        y_pred = float(fcst[horizon - 1])

        rows.append(
            {
                "time": times[target_idx],
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    df = pd.DataFrame(rows)
    return df
