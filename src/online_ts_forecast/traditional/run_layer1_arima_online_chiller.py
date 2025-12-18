# src/online_ts_forecast/traditional/run_layer1_arima_online_chiller.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from online_ts_forecast.traditional.arima_models import (
    arima_nonseasonal_forecast,
    sarima_daily_forecast,
)
from online_ts_forecast.traditional.online_eval import online_eval_univar
from online_ts_forecast.traditional.metrics import evaluate_backtest
from online_ts_forecast.utils.timing import time_block, attach_timing


# ======= 配置 =======

DATA_PATH = Path("data/private/tem_data_07.csv")
TIME_COL_INDEX = 0
TARGET_COL = "冷机3总有功功率"

FREQ_MINUTES = 5
DAILY_SEASON = int(24 * 60 / FREQ_MINUTES)

HORIZON = 1              # +5min
WARMUP_DAYS = 7
EVAL_DAYS = 7
MIN_HISTORY = DAILY_SEASON * 2
ONLINE_STRIDE_STEPS = 6  # 在线评估每 30min 起报一次

OUTPUT_METRICS_DIR = Path("outputs/metrics")
OUTPUT_FIGURES_DIR = Path("outputs/figures")


def load_chiller_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH)
    time_col = df.columns[TIME_COL_INDEX]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    s = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
    return s


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
    timing_global = {}
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
        timing = {}

        with time_block(f"{name}.online_eval", timing):
            df_online = online_eval_univar(
                series=series,
                forecast_func=func,
                horizon=HORIZON,
                warmup_days=WARMUP_DAYS,
                eval_days=EVAL_DAYS,
                freq_minutes=FREQ_MINUTES,
                min_history=MIN_HISTORY,
                stride_steps=ONLINE_STRIDE_STEPS,
                forecast_kwargs=kwargs,
            )

        out_csv = OUTPUT_METRICS_DIR / f"{name}_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.csv"
        df_online.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[CSV] 已保存：{out_csv}，样本数={len(df_online)}")

        # 评估指标
        m = evaluate_backtest(df_online.rename(columns={"obs_time": "time"}))
        m["method"] = name

        spent = timing.get(f"{name}.online_eval", 0.0)
        m = attach_timing(m, spent, n_samples=len(df_online), prefix="online_")

        metrics_rows.append(m)

        out_png = OUTPUT_FIGURES_DIR / f"{name}_chiller_h{HORIZON}_warmup{WARMUP_DAYS}d_eval{EVAL_DAYS}d.png"
        plot_online(
            df_online,
            title=f"{TARGET_COL} - {name} 在线评估 (h={HORIZON}, warmup={WARMUP_DAYS}d, eval={EVAL_DAYS}d)",
            out_path=out_png,
        )

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics = df_metrics[
            ["method", "MAE", "RMSE", "sMAPE", "n", "online_time_sec", "online_time_per_sample_ms"]
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
