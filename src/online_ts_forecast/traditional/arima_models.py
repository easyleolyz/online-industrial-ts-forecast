# src/online_ts_forecast/traditional/arima_models.py

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pmdarima import auto_arima

'''
实现ARIMA和SARIMA模型的预测功能。
auto_arima搜索最好的参数
'''

def auto_arima_forecast(
    history: pd.Series,
    horizon: int = 1,
    seasonal: bool = False,
    season_length: int = 1,
    max_p: int = 3,
    max_q: int = 3,
    max_P: int = 1,
    max_Q: int = 1,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    使用 pmdarima.auto_arima 对 history 进行建模，并预测未来 horizon 步。

    history: 单变量时间序列（pd.Series），index 为时间戳。
    seasonal: 是否使用季节项（SARIMA）。
    season_length: 季节周期长度（对 5min 冷机数据，日季节=288）。
    """
    if len(history) < 50:
        # 样本太少时，直接退化为最后一个值
        last_val = history.iloc[-1]
        return np.repeat(last_val, horizon)

    m = season_length if seasonal else 1

    model = auto_arima(
        history,
        seasonal=seasonal,
        m=m,
        max_p=max_p,
        max_q=max_q,
        max_P=max_P,
        max_Q=max_Q,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        trace=False,
        **kwargs,
    )
    fcst = model.predict(n_periods=horizon)
    return np.asarray(fcst, dtype=float)


def arima_nonseasonal_forecast(history: pd.Series, horizon: int = 1) -> np.ndarray:
    """
    非季节 ARIMA baseline：只捕捉短期自回归/差分/移动平均结构。
    """
    return auto_arima_forecast(
        history,
        horizon=horizon,
        seasonal=False,
        season_length=1,
    )


def sarima_daily_forecast(
    history: pd.Series,
    horizon: int = 1,
    season_length: int = 288,
) -> np.ndarray:
    """
    带日季节的 SARIMA baseline（以日周期建模）。
    对 5min 数据，season_length 默认=288。
    """
    return auto_arima_forecast(
        history,
        horizon=horizon,
        seasonal=True,
        season_length=season_length,
    )


# === 预留：SARIMAX（带外生变量） ===

def sarimax_forecast_placeholder(
    history: pd.Series,
    exog_history: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    season_length: int = 288,
) -> np.ndarray:
    """
    预留接口：未来引入天气/业务指标等外生变量时再补完。
    当前仅占位，暂时直接调用 sarima_daily_forecast。
    """
    # TODO: 使用 statsmodels.tsa.statespace.SARIMAX 实现带 exog 的模型
    return sarima_daily_forecast(history, horizon=horizon, season_length=season_length)
