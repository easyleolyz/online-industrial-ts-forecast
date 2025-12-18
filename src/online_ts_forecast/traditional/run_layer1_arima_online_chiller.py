# src/online_ts_forecast/traditional/run_layer1_arima_online_chiller.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from tqdm.auto import tqdm

from online_ts_forecast.traditional.arima_models import (
    arima_nonseasonal_forecast,
    sarima_daily_forecast,
)
from online_ts_forecast.traditional.metrics import evaluate_backtest
from online_ts_forecast.utils.timing import time_block, attach_timing

# ---- 中文字体 ----
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# ======= 配置 =======

DATA_PATH = Path("data/private/tem_data_07.csv")
TIME_COL_INDEX = 0
TARGET_COL = "冷机3总有功功率"

FREQ_MINUTES = 5
DAILY_SEASON = int(24 * 60 / FREQ_MINUTES)

HORIZON = 1
WARMUP_DAYS = 7
EVAL_DAYS = 7
MIN_HISTORY = DAILY_SEASON * 2
ONLINE_STRIDE_STEPS = 6   # 每 6 个 5min（30min）起报一次

OUTPUT_METRICS_DIR = Path("outputs/metrics")
OUTPUT_FIGURES_DIR = Path("outputs/figures")


def load_chiller_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    time_col = df.columns[TIME_COL_INDEX]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    s = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
    return s


def _define_online_window(
    times: pd.DatetimeIndex,
    warmup_days: int,
    eval_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0 = times.min()
    warmup_end = t0 + pd.Timedelta(days=warmup_days)
    eval_end = warmup_end + pd.Timedelta(days=eval_days)
    return warmup_end, eval_end


def online_eval_univar_with_tqdm(
    series: pd.Series,
    forecast_func,
    method_name: str,
    horizon: int,
    warmup_days: int,
    eval_days: int,
    freq_minutes: int,
    min_history: int,
    stride_steps: int,
    forecast_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    在线评估（本文件局部版本）：
    - 每 stride_steps 个 obs_time 起报一次
    - tqdm 显示在线评估进度
    """
    if forecast_kwargs is None:
        forecast_kwargs = {}

    series = series.dropna()
    series = series.sort_index()
    times = series.index

    warmup_end, eval_end = _define_online_window(times, warmup_days, eval_days)
    freq = pd.Timedelta(minutes=freq_minutes)

    eval_mask = (times > warmup_end) & (times <= eval_end)
    eval_times = times[eval_mask]

    rows = []

    if len(eval_times) == 0:
        print("[WARN] 在线评估窗口为空。")
        return pd.DataFrame(columns=["obs_time", "y_true", "y_pred"])

    iterator = enumerate(eval_times)
    iterator = tqdm(iterator, total=len(eval_times), desc=f"{method_name} online")

    for i, obs_time in iterator:
        if i % stride_steps != 0:
            continue

        issue_time = obs_time - horizon * freq
        history = series[series.index <= issue_time]
        if len(history) < min_history:
            continue

        fcst = forecast_func(history, horizon=horizon, **forecast_kwargs)
        if fcst is None or len(fcst) < horizon:
            continue

        y_pred = float(fcst[horizon - 1])
        try:
            y_true = float(series.loc[obs_time])
        except KeyError:
            continue

        rows.append(
            {
                "obs_time": obs_time,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)


def plot_online(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        print(f"[WARN] 空在线结果，跳过绘图：{title}")
        return

    t = df["obs_time"]
    y = df["y_true"]
    yhat = df["y_pred"]

    plt.figure(figsize=(12, 4))
    plt.plot(t, y, label="真实", linewidth=1.2)
    plt.plot(t, yhat, label="在线预测", linewidth=1.0)

    plt.title(title)
    plt.xlabel("时间")
    plt.ylabel(TARGET_COL)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[图] 已保存：{out_path}")


def run_layer1_arima_online():
    timing_global: Dict[str, float] = {}
    with time_block("load_chiller_series", timing_global):
        series = load_chiller_series()

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    methods = {
        "arima_nonseasonal_online": (arima_nonseasonal_forecast, {}),
        "sarima_daily_online": (sarima_daily_forecast, {"season_length": DAILY_SEASON}),
    }

    metrics_rows = []

    for name, (func, kwargs) in methods.items():
        print(f"\n=== [Layer1 在线] {name} ===")
        timing: Dict[str, float] = {}

        with time_block(f"{name}.online_eval", timing):
            df_online = online_eval_univar_with_tqdm(
                series=series,
                forecast_func=func,
                method_name=name,
                horizon=HORIZON,
                warmup_days=WARMUP_DAYS,
                eval_days=EVAL_DAYS,
                freq_minutes=FREQ_MINUTES,
                min_history=MIN_HISTORY,
                stride_steps=ONLINE_STRIDE_STEPS,
                forecast_kwargs=kwargs,
            )

        out_csv = (
            OUTPUT_METRICS_DIR
            / f"{name}_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
        )
        df_online.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[CSV] 已保存：{out_csv}，样本数={len(df_online)}")

        m = evaluate_backtest(df_online.rename(columns={"obs_time": "time"}))
        m["method"] = name

        spent = timing.get(f"{name}.online_eval", 0.0)
        m = attach_timing(m, spent, n_samples=len(df_online), prefix="online_")
        metrics_rows.append(m)

        out_png = (
            OUTPUT_FIGURES_DIR
            / f"{name}_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.png"
        )
        plot_online(
            df_online,
            title=(
                f"{TARGET_COL} - {name} 在线评估 "
                f"(h={HORIZON}, warmup={WARMUP_DAYS}d, eval={EVAL_DAYS}d)"
            ),
            out_path=out_png,
        )

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)[
            [
                "method",
                "MAE",
                "RMSE",
                "sMAPE",
                "n",
                "online_time_sec",
                "online_time_per_sample_ms",
            ]
        ]
        summary_csv = (
            OUTPUT_METRICS_DIR
            / f"layer1_arima_online_summary_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
        )
        df_metrics.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print("\n=== Layer1 ARIMA/SARIMA 在线评估汇总 ===")
        print(df_metrics)
        print(f"[汇总] 已保存：{summary_csv}")


if __name__ == "__main__":
    run_layer1_arima_online()
