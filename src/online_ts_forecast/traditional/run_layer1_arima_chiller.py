# src/online_ts_forecast/traditional/run_layer1_arima_chiller.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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

# ---- 中文字体（依赖 WSL2 中已安装 SimHei 字体） ----
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# ======= 配置 =======

DATA_PATH = Path("data/private/tem_data_07.csv")
TIME_COL_INDEX = 0
TARGET_COL = "冷机3总有功功率"

FREQ_MINUTES = 5
DAILY_SEASON = int(24 * 60 / FREQ_MINUTES)  # 288

HORIZON = 1                # +5min 单步预测
MIN_HISTORY = DAILY_SEASON * 2
BT_STRIDE_STEPS = 12       # 滚动回测间隔：每 12 个点（1 小时）做一次评估

OUTPUT_METRICS_DIR = Path("outputs/metrics")
OUTPUT_FIGURES_DIR = Path("outputs/figures")


def load_chiller_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    time_col = df.columns[TIME_COL_INDEX]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    s = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
    return s


def offline_backtest_univar(
    series: pd.Series,
    forecast_func,
    method_name: str,
    horizon: int,
    min_history: int,
    forecast_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    本文件内的简易滚动回测：
    - 带 tqdm 进度条
    - 每 BT_STRIDE_STEPS 个点评估一次
    """
    if forecast_kwargs is None:
        forecast_kwargs = {}

    series = series.dropna()
    times = series.index
    n = len(series)

    if n <= min_history + horizon:
        print("[WARN] 序列过短，无法回测。")
        return pd.DataFrame(columns=["time", "y_true", "y_pred"])

    rows = []
    start = min_history
    end = n - horizon

    idx_iter = range(start, end, BT_STRIDE_STEPS)
    idx_iter = tqdm(idx_iter, desc=f"{method_name} rolling BT")

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

    return pd.DataFrame(rows)


def plot_backtest(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        print(f"[WARN] 空回测结果，跳过绘图：{title}")
        return

    t = df["time"]
    y = df["y_true"]
    yhat = df["y_pred"]

    plt.figure(figsize=(12, 4))
    plt.plot(t, y, label="真实", linewidth=1.2)
    plt.plot(t, yhat, label="预测", linewidth=1.0)

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


def run_layer1_arima():
    # 计时：数据加载
    timing_global: Dict[str, float] = {}
    with time_block("load_chiller_series", timing_global):
        series = load_chiller_series()

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    methods = {
        "arima_nonseasonal": (arima_nonseasonal_forecast, {}),
        "sarima_daily": (sarima_daily_forecast, {"season_length": DAILY_SEASON}),
    }

    metrics_rows = []

    for name, (func, kwargs) in methods.items():
        print(f"\n=== Layer1 单变量 {name} 离线回测 ===")
        timing: Dict[str, float] = {}

        with time_block(f"{name}.offline_backtest", timing):
            df_bt = offline_backtest_univar(
                series=series,
                forecast_func=func,
                method_name=name,
                horizon=HORIZON,
                min_history=MIN_HISTORY,
                forecast_kwargs=kwargs,
            )

        out_csv = OUTPUT_METRICS_DIR / f"layer1_{name}_chiller_h{HORIZON}.csv"
        df_bt.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[CSV] 已保存：{out_csv}，样本数={len(df_bt)}")

        m = evaluate_backtest(df_bt)
        m["method"] = name

        spent = timing.get(f"{name}.offline_backtest", 0.0)
        m = attach_timing(m, spent, n_samples=len(df_bt), prefix="offline_")
        metrics_rows.append(m)

        out_png = OUTPUT_FIGURES_DIR / f"layer1_{name}_chiller_h{HORIZON}.png"
        plot_backtest(
            df_bt,
            title=f"{TARGET_COL} - {name} ARIMA 离线回测 (h={HORIZON})",
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
                "offline_time_sec",
                "offline_time_per_sample_ms",
            ]
        ]
        summary_csv = OUTPUT_METRICS_DIR / f"layer1_arima_summary_chiller_h{HORIZON}.csv"
        df_metrics.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print("\n=== Layer1 ARIMA/SARIMA 离线回测汇总 ===")
        print(df_metrics)
        print(f"[汇总] 已保存：{summary_csv}")


if __name__ == "__main__":
    run_layer1_arima()
