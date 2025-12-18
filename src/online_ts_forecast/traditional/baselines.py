# src/online_ts_forecast/traditional/baselines.py

from __future__ import annotations
from typing import Callable, Dict, Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Any, Dict, Optional
from tqdm import tqdm


'''
我们使用一些简单基线方法，我们的阐述的阶段
来源--->定义--->在线预测的使用
1.Naive基线：
“明天和今天一样”
Naive 预测等价于假设序列是一个随机游走
在线场景：随着数据一条条到来，只保存"最新的版本"---
单步预测时，简单替换，
多步预测时，水平
更新O(1),预测O(1)
2.Seasonal Naive基线：
“周期的同一位置”
Naive 预测等价于假设序列是一个随机游走
在线场景：随着数据一条条到来，只保存"环形缓冲区"---
单步预测时，周期替换
多步预测时，周期替换
更新O(1),预测O(h)
3.Moving Average基线：
"用最近w个点的均值来代表“当前水平”，并预测未来都等于这个水平："
"低通滤波器"---把高频噪声（短时间抖动）平均掉,保留低频成分（慢变化水平）
“窗口内等权，窗口外权重为 0”
单步预测时，窗口等权
多步预测时，窗口等权
更新O(1),预测O(h)
4.ETS指数平滑
维护一组状态变量（水平、趋势、季节），并用递推更新
水平--->趋势--->季节--->
水平：“旧估计”和“新观测”的折中
趋势：“增加涨跌的项”
季节：“维护一个周期记录季节的偏移量”加性为加减绝对数值，乘性为放缩比例”
策略 A：固定参数，递推更新：一段历史离线估计α（水平加权）,β（增减加权）,γ（周期偏离加权） 和初值，参数固定
策略 B：周期性重拟合，参数调整（或许可以做成参数可学习？）
'''
'''预测步骤：接受观测--->更新模型状态（递推）--->输出预测序列--->真实值到来'''

# ======= 单步/多步基线预测函数 =======

def naive_forecast(history: pd.Series, horizon: int = 1) -> np.ndarray:
    """
    最简单基线：y_hat_{t+h} = y_t
    history: 截止当前时刻 t 的历史序列（不含未来）
    """
    if len(history) == 0:
        raise ValueError("history is empty")
    last_value = history.iloc[-1]
    return np.repeat(last_value, horizon)#最后值重复n次后返回


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
    forecast_func,
    horizon: int = 1,
    min_history: int = 100,
    forecast_kwargs: Optional[Dict[str, Any]] = None,
    stride_steps: int = 1,
    progress_desc: Optional[str] = None,
) -> pd.DataFrame:
    """
    通用滚动回测：
      - series: 单变量时间序列（DatetimeIndex）
      - forecast_func(history, horizon, **forecast_kwargs) -> ndarray 长度 >= horizon
      - horizon: 预测步数（这里一般用 1）
      - min_history: 起始历史长度
      - stride_steps: 每隔多少个时间步做一次评估（>=1）
      - progress_desc: 不为 None 时显示 tqdm 进度条文字
    """
    if forecast_kwargs is None:
        forecast_kwargs = {}

    series = series.dropna()
    times = series.index

    n = len(series)
    if n <= min_history + horizon:
        print("[WARN] 序列过短，无法回测。")
        return pd.DataFrame(columns=["time", "y_true", "y_pred"])

    step = max(1, int(stride_steps))
    start = min_history
    end = n - horizon

    idx_iter = range(start, end, step)
    if progress_desc is not None:
        idx_iter = tqdm(idx_iter, desc=progress_desc)

    rows = []

    for i in idx_iter:
        history = series.iloc[:i]
        if len(history) < min_history:
            continue

        fcst = forecast_func(history, horizon=horizon, **forecast_kwargs)
        if fcst is None or len(fcst) < horizon:
            continue

        y_pred = float(fcst[horizon - 1])
        y_true = float(series.iloc[i + horizon])
        t_obs = times[i + horizon]

        rows.append(
            {
                "time": t_obs,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    df = pd.DataFrame(rows)
    return df


'''
一些思考，为什么当时选择这些方法预测呢，总感觉是数值的模拟，是一种自回归吗，
以及预测未来走势是需要许多数据源历史外生特征，
以及在线预测可能还需要截面的特征，相关或者因果还要其他逻辑关系，
否则就算预测接近，你怎么阐释他的原理，请你仔细分析与我讨论

个最常见、也最“经济”的结构假设，这种假设是否起源于我们的直觉？
不加外生特征、不用截面信息、只靠过去 y，真的靠谱么？

当系统满足某种“可观测充分性”：过去的 
y 已经把外界影响的结果都浓缩进来了，使得对未来的预测，额外外生变量的边际收益有限。

预测的本质：利用可用信息做条件均值（或分位数）估计
'''
