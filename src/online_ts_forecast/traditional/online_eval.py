# src/online_ts_forecast/traditional/online_eval.py

from __future__ import annotations
from typing import Callable, Dict, Any, Optional

import numpy as np
import pandas as pd


ForecastFuncUnivar = Callable[[pd.Series, int], np.ndarray]


def define_online_window(
    times: pd.DatetimeIndex,
    warmup_days: int = 7,
    eval_days: int = 7,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    根据时间索引自动定义：
    - warmup_end: 预热结束时间（不含）
    - eval_end:   在线评估结束时间（含）

    规则：从最早时间起，前 warmup_days 天做预热，紧接着 eval_days 天做评估。
    """
    t0 = times.min()
    warmup_end = t0 + pd.Timedelta(days=warmup_days)
    eval_end = warmup_end + pd.Timedelta(days=eval_days)
    return warmup_end, eval_end


def online_eval_univar(
    series: pd.Series,
    forecast_func: ForecastFuncUnivar,
    horizon: int = 1,
    warmup_days: int = 7,
    eval_days: int = 7,
    freq_minutes: int = 5,
    min_history: int = 100,
    stride_steps: int = 6,
    forecast_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    对单变量序列做“一周预热 + 一周在线”的滚动评估。

    返回 DataFrame:
        [issue_time, obs_time, y_true, y_pred]
    """
    if forecast_kwargs is None:
        forecast_kwargs = {}

    series = series.dropna()
    times = series.index.sort_values()
    warmup_end, eval_end = define_online_window(times, warmup_days, eval_days)

    freq = pd.Timedelta(minutes=freq_minutes)
    rows = []

    # 从第二周开始，到第三周结束（按时间筛）
    eval_mask = (times > warmup_end) & (times <= eval_end)
    eval_times = times[eval_mask]

    if len(eval_times) == 0:
        print("[WARN] 在线评估窗口内没有数据，请检查时间范围。")
        return pd.DataFrame(columns=["issue_time", "obs_time", "y_true", "y_pred"])

    # 在线步长：每 stride_steps 个 5min 做一次评估（默认 30min）
    for i, obs_time in enumerate(eval_times):
        if (i % stride_steps) != 0:
            continue

        # 视距为 horizon 步：obs_time 对应的起报时刻
        issue_time = obs_time - horizon * freq

        # 仅使用 issue_time 之前的历史
        history = series[series.index <= issue_time]
        if len(history) < min_history:
            continue

        fcst = forecast_func(history, horizon=horizon, **forecast_kwargs)
        if len(fcst) < horizon:
            continue

        y_pred = float(fcst[horizon - 1])
        y_true = float(series.loc[obs_time])

        rows.append(
            {
                "issue_time": issue_time,
                "obs_time": obs_time,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    df = pd.DataFrame(rows)
    return df


def online_eval_multivar_var(
    df: pd.DataFrame,
    target_col: str,
    var_forecast_func: Callable[[pd.DataFrame, int], pd.DataFrame],
    horizon: int = 1,
    warmup_days: int = 7,
    eval_days: int = 7,
    freq_minutes: int = 5,
    min_history: int = 100,
    stride_steps: int = 6,
    maxlags: int = 12,
) -> pd.DataFrame:
    """
    多变量 VAR 在线评估，一周预热 + 一周评估。
    df: index 为时间戳，列为多个变量，包括 target_col。
    var_forecast_func: 接收 (history_df, horizon, maxlags) 返回未来 horizon 步的 DataFrame。
    """
    df = df.dropna()
    times = df.index.sort_values()
    warmup_end, eval_end = define_online_window(times, warmup_days, eval_days)

    freq = pd.Timedelta(minutes=freq_minutes)
    rows = []

    eval_mask = (times > warmup_end) & (times <= eval_end)
    eval_times = times[eval_mask]

    if len(eval_times) == 0:
        print("[WARN] 在线评估窗口内没有数据，请检查时间范围。")
        return pd.DataFrame(columns=["issue_time", "obs_time", "y_true", "y_pred"])

    for i, obs_time in enumerate(eval_times):
        if (i % stride_steps) != 0:
            continue

        issue_time = obs_time - horizon * freq
        history = df[df.index <= issue_time]
        if len(history) < min_history:
            continue

        df_fcst = var_forecast_func(history, horizon=horizon, maxlags=maxlags)
        if len(df_fcst) < horizon:
            continue

        y_pred = float(df_fcst[target_col].iloc[horizon - 1])
        y_true = float(df.loc[obs_time, target_col])

        rows.append(
            {
                "issue_time": issue_time,
                "obs_time": obs_time,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)
