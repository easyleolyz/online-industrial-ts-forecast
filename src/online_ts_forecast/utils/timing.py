# src/online_ts_forecast/utils/timing.py

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import MutableMapping, Optional


@contextmanager
def time_block(
    label: str,
    stats: Optional[MutableMapping[str, float]] = None,
    verbose: bool = True,
):
    """
    简单计时上下文管理器：

    示例：
        timing = {}
        with time_block("arima_offline_backtest", timing):
            df_bt = rolling_backtest(...)

        # timing["arima_offline_backtest"] 中即为秒级耗时
    """
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        if verbose:
            print(f"[TIMER] {label}: {dt:.3f} s")
        if stats is not None:
            stats[label] = float(dt)


def attach_timing(
    metrics: dict,
    time_sec: float,
    n_samples: Optional[int] = None,
    prefix: str = "",
) -> dict:
    """
    在原有指标 dict 上附加耗时信息：

        - {prefix}time_sec
        - {prefix}time_per_sample_ms （若给了 n_samples）

    返回一个新的 dict（不修改原 metrics）。
    """
    out = dict(metrics)
    key_time = f"{prefix}time_sec" if prefix else "time_sec"
    out[key_time] = float(time_sec)

    if n_samples is not None and n_samples > 0:
        key_ms = f"{prefix}time_per_sample_ms" if prefix else "time_per_sample_ms"
        out[key_ms] = float(time_sec * 1000.0 / n_samples)

    return out
